package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.ArrayList;
import java.util.List;

/**
 * The CPU-side pieces of the shared vision graphs the CPU executor runs (clip.cpp {@code build_inp} and the
 * projector matmuls): the patch embedding as one matmul over the {@link ClipPatches} matrix, biased linears, the
 * qwen vision m-rope and NEOX segment rope hooks, and the image-grid checks. Package-private executor support.
 */
final class ClipVit {

    final ClipVisionModel model;
    final ClipHparams hp;
    final ClipVisionWeights w;

    /** The shared pieces of {@code model} over {@code w}. */
    ClipVit(ClipVisionModel model, ClipVisionWeights w) {
        this.model = model;
        this.hp = model.hparams();
        this.w = w;
    }

    // ------------------------------------------------------------------ patch embedding

    /**
     * Projects every patch of the image to {@code [pw * ph][n_embd]} rows in raster order, with the patch bias
     * when the file has one ({@code build_inp}: {@code ggml_conv_2d} as one matmul over the {@link ClipPatches}
     * matrix).
     */
    float[] patchRows(ClipImage image, int pw, int ph) {
        return patchEmbed(ClipPatches.raster(image, hp.patchSize(), pw, ph), pw * ph);
    }

    /**
     * Projects the {@code nPatches} rows of a {@link ClipPatches} matrix to {@code [nPatches][n_embd]} rows, with
     * the patch bias when the file has one. With the qwen second-frame kernel both convolutions run over the still
     * image and add ({@code build_inp_temporal_merge}). A temporal pair doubles every patch row: the first half
     * runs the first kernel, the second half the second, and the two add &mdash; the two-frame input of
     * {@code build_inp_temporal_merge}.
     *
     * @throws IllegalArgumentException when a patch row is neither one nor two kernel widths, or a pair reaches an
     *                                 encoder with no second kernel
     */
    float[] patchEmbed(float[] patches, int nPatches) {
        int n = Math.multiplyExact(nPatches, Math.toIntExact(w.patchEmbd.rows()));
        float[] rows = new float[n];
        int kernelWidth = (int) w.patchEmbd.cols();
        int rowWidth = patches.length / nPatches;
        if (rowWidth != kernelWidth && rowWidth != 2 * kernelWidth) {
            throw new IllegalArgumentException(patches.length + " patch values are not " + nPatches + " rows of "
                    + kernelWidth);
        }
        if (rowWidth == 2 * kernelWidth) {
            patchEmbedPair(patches, nPatches, kernelWidth, rows);
        } else {
            w.matVec.matmul(w.patchEmbd, patches, 0, rows, 0, nPatches);
            if (w.patchEmbd1 != null) {
                float[] second = new float[n];
                w.matVec.matmul(w.patchEmbd1, patches, 0, second, 0, nPatches);
                CpuOps.add(second, 0, rows, 0, rows, 0, n);
            }
        }
        VisionRows.addBias(rows, w.patchBias);
        return rows;
    }

    /**
     * The two-frame input of {@code build_inp_temporal_merge}: kernel 0 over every row's first half (the first
     * frame's patch), kernel 1 over the second half, the outputs added. Requires the second kernel.
     */
    private void patchEmbedPair(float[] patches, int nPatches, int kernelWidth, float[] rows) {
        if (w.patchEmbd1 == null) {
            throw new IllegalArgumentException("this encoder carries no temporal patch kernel"
                    + " (v.patch_embd.weight.1), so it cannot encode a frame pair");
        }
        float[] first = new float[nPatches * kernelWidth];
        float[] second = new float[nPatches * kernelWidth];
        for (int t = 0; t < nPatches; t++) {
            System.arraycopy(patches, t * 2 * kernelWidth, first, t * kernelWidth, kernelWidth);
            System.arraycopy(patches, t * 2 * kernelWidth + kernelWidth, second, t * kernelWidth, kernelWidth);
        }
        float[] secondRows = new float[rows.length];
        w.matVec.matmul(w.patchEmbd, first, 0, rows, 0, nPatches);
        w.matVec.matmul(w.patchEmbd1, second, 0, secondRows, 0, nPatches);
        CpuOps.add(secondRows, 0, rows, 0, rows, 0, rows.length);
    }

    /** The patch grid of an image whose sides are multiples of the patch size, refused by name otherwise. */
    static int[] grid(ClipHparams hp, ClipImage image) {
        int ps = hp.patchSize();
        if (image.width() % ps != 0 || image.height() % ps != 0) {
            throw new IllegalArgumentException("image size " + image.width() + "x" + image.height() + " is not a"
                    + " multiple of the patch size " + ps);
        }
        return new int[]{image.width() / ps, image.height() / ps};
    }

    /** Refuses a grid other than the square {@code clip.vision.image_size} grid. */
    static void requireImageSizeGrid(ClipVisionModel model, int pw, int ph) {
        int side = model.nPatchesPerSide();
        if (pw != side || ph != side) {
            throw new IllegalArgumentException("this encoder needs a " + side + "x" + side + " patch grid"
                    + " (image_size " + model.hparams().imageSize() + "), got " + pw + "x" + ph);
        }
    }

    // ------------------------------------------------------------------ projector helpers

    /**
     * The qwen vision m-rope over {@code pos} at frequency base {@code base} through {@link CpuRope}:
     * {@link QwenVisionGrid#POSITION_SECTIONS} sections of {@code nTokens} positions each, {@code [y | x | y | x]}
     * ({@link QwenVisionGrid#ropePositions}).
     *
     * @throws IllegalArgumentException when the head width is not a positive multiple of 4
     */
    CpuVitTower.PosHook qwenRope(int[] pos, float base) {
        RopeFreqTable table = RopeFreqTable.of(QwenVisionGrid.rope(model.dHead(), base), null);
        int dh = model.dHead();
        return (x, n, nHeads) -> {
            requirePositions(pos, QwenVisionGrid.POSITION_SECTIONS * n);
            CpuRope.apply(table, x, 0, n, nHeads, dh, nHeads * dh, 0, pos, 0, n);
        };
    }

    /**
     * One NEOX {@link CpuRope} call per segment at frequency base {@code base}, each over its dims of every head
     * ({@code ggml_rope_ext} with the offset of {@code ggml_rope_set_offset}).
     */
    CpuVitTower.PosHook neoxRope(List<VisionOps.NeoxSegment> segments, float base) {
        int dh = model.dHead();
        List<RopeFreqTable> tables = new ArrayList<>(segments.size());
        for (VisionOps.NeoxSegment seg : segments) {
            tables.add(RopeFreqTable.of(RopeSpec.builder(RopeType.NEOX).nDims(seg.nDims).freqBase(base).build(),
                    null));
        }
        return (x, n, nHeads) -> {
            for (int i = 0; i < segments.size(); i++) {
                VisionOps.NeoxSegment seg = segments.get(i);
                requirePositions(seg.positions, n);
                CpuRope.apply(tables.get(i), x, 0, n, nHeads, dh, nHeads * dh, seg.offset, seg.positions, 0, n);
            }
        };
    }

    private static void requirePositions(int[] pos, int expected) {
        if (pos.length != expected) {
            throw new IllegalArgumentException("the rope needs " + expected + " positions, got " + pos.length);
        }
    }

    /**
     * {@code build_mm} plus an optional bias: {@code nTokens} rows of {@code weight}'s output width. The LoRA
     * delta of {@code site}, when given, adds to the product before the bias.
     */
    float[] linear(CpuTensorRef weight, float[] x, int nTokens, float[] bias, CpuProjectionDeltas.Site site) {
        float[] y = new float[(int) (nTokens * weight.rows())];
        w.matVec.matmul(weight, x, 0, y, 0, nTokens);
        if (site != null) {
            site.apply(nTokens, x, 0, (int) weight.cols(), y, 0, (int) weight.rows());
        }
        VisionRows.addBias(y, bias);
        return y;
    }
}
