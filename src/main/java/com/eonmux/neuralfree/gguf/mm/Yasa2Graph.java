package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * The Yasa2 (Reka Edge) ConvNeXt-V2 encoder and its projector (llama.cpp tools/mtmd/models/yasa2.cpp), written
 * once against {@link VisionOps} so the CPU and GPU executors run the same graph: a stride-{@code patch_size}
 * patch convolution and channel layer norm, then the stages in order, each an optional downsample (channel layer
 * norm and a 2x2 stride-2 convolution) followed by its ConvNeXt blocks (7x7 depthwise convolution, channel layer
 * norm, {@code pw1}, GELU-erf, global response normalization, {@code pw2}, residual). The learned
 * {@code v.vision_pos_embed} rows are added to the final grid when their row count equals its token count, as the
 * reference does; the grid is then average-pooled to 8x8 with the reference's kernel
 * ({@code max(1, side / min(8, side))} per axis, no padding) and projected by the {@code mm.0} / {@code mm.2}
 * GELU-erf MLP. The channel layer norm divides by {@code sqrt(max(var, eps))} and the response normalization
 * clamps the channel mean of the norms at {@code 1e-6}, the reference's forms; the executors' row norms divide by
 * {@code sqrt(var + eps)} and agree to float rounding on every grid the norm guards.
 *
 * <p>Tensors: {@code v.patch_embd.*}, {@code v.patch_ln.*}, {@code v.stage.S.down.{ln,conv}.*},
 * {@code v.stage.S.blk.B.{dw,ln,pw1,grn,pw2}.*}, {@code v.vision_pos_embed}, {@code mm.0.*}, {@code mm.2.*}; a
 * block exists when its {@code dw.weight} does, a stage when it has a downsample or a block (clip.cpp
 * {@code load_tensors}). {@code v.backbone_ln.*} is loaded by the reference but never read by its graph.
 * Feature maps are pixel-major rows {@code [y * side + x][channel]}. Package-private executor support.
 */
final class Yasa2Graph {

    /** The side of the pooled token grid: 64 rows per image ({@link ImageTokenFormula#POOLED_8X8}). */
    static final int POOLED_SIDE = 8;
    /** Kernel side of the block depthwise convolution; the reference pads it by 3. */
    static final int DW_KERNEL = 7;
    private static final int DW_PAD = 3;
    /** Kernel side and stride of the stage downsample convolution. */
    private static final int DOWN = 2;
    /** Floor of the channel mean of the response norms ({@code convnext_grn}). */
    static final float GRN_EPS = 1e-6f;
    private static final String POS = "v.vision_pos_embed";

    private Yasa2Graph() {
    }

    // ------------------------------------------------------------------ tensors

    /**
     * Resolves and shape-checks the stage, position and projector tensors of a yasa2 file into {@code c}: the
     * channel chain from the patch kernel through every downsample, every block's widths, and the
     * {@code image_size} grid, which must pool to the 8x8 token grid.
     *
     * @throws MissingTensorException   when a required tensor is absent
     * @throws IllegalArgumentException when a shape differs from the graph's, the file holds no stage, or the
     *                                  {@code image_size} grid does not pool to 8x8 (named)
     */
    static ClipProjectorTensors resolve(ClipProjectorTensors.Collector c, TensorCatalog tc, ClipHparams hp) {
        TensorRef patch = tc.require("v.patch_embd.weight");
        int ch = (int) patch.ne(3);
        c.optionalVector("v.patch_ln.weight", ch).optionalVector("v.patch_ln.bias", ch);
        int side = hp.imageSize() / hp.patchSize();
        int stages = 0;
        for (int s = 0; ; s++) {
            String p = "v.stage." + s + ".";
            boolean down = tc.has(p + "down.conv.weight");
            if (!down && !tc.has(p + "blk.0.dw.weight")) {
                break;
            }
            if (down) {
                c.optionalVector(p + "down.ln.weight", ch).optionalVector(p + "down.ln.bias", ch);
                TensorRef conv = tc.require(p + "down.conv.weight");
                int out = (int) conv.ne(3);
                c.kernel(p + "down.conv.weight", DOWN, DOWN, ch, out).optionalVector(p + "down.conv.bias", out);
                ch = out;
                side /= DOWN;
            }
            for (int b = 0; tc.has(p + "blk." + b + ".dw.weight"); b++) {
                block(c, tc, p + "blk." + b + ".", ch);
            }
            stages++;
        }
        if (stages == 0) {
            throw new MissingTensorException(tc.archName(), -1, "v.stage.0.blk.0.dw.weight");
        }
        requirePooledGrid(side, side, "clip.vision.image_size " + hp.imageSize());
        if (tc.has(POS)) {
            TensorRef pos = tc.require(POS);
            if (pos.ne(0) != ch) {
                throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + POS
                        + " has shape " + pos + ", the last stage has " + ch + " channels");
            }
            c.vector(POS, pos.ne(0), pos.ne(1));
        }
        int hidden = c.outDim("mm.0.weight", ch);
        c.matrix("mm.0.weight", ch, hidden).optionalVector("mm.0.bias", hidden);
        int out = c.outDim("mm.2.weight", hidden);
        c.matrix("mm.2.weight", hidden, out).optionalVector("mm.2.bias", out);
        return c.build(out);
    }

    /** One ConvNeXt block of {@code ch} channels: the depthwise kernel, its norm, {@code pw1}, GRN and {@code pw2}. */
    private static void block(ClipProjectorTensors.Collector c, TensorCatalog tc, String p, int ch) {
        c.depthwise(p + "dw.weight", DW_KERNEL, ch).optionalVector(p + "dw.bias", ch)
                .optionalVector(p + "ln.weight", ch).optionalVector(p + "ln.bias", ch);
        int hidden = c.outDim(p + "pw1.weight", ch);
        c.matrix(p + "pw1.weight", ch, hidden).optionalVector(p + "pw1.bias", hidden)
                .optionalVector(p + "grn.weight", hidden).optionalVector(p + "grn.bias", hidden)
                .matrix(p + "pw2.weight", hidden, ch).optionalVector(p + "pw2.bias", ch);
    }

    /** The pooling kernel of one axis of {@code side} cells: {@code max(1, side / min(8, side))}. */
    static int poolKernel(int side) {
        return Math.max(1, side / Math.min(POOLED_SIDE, side));
    }

    /** Cells one axis of {@code side} cells pools to (no padding, stride = kernel). */
    static int pooledSide(int side) {
        int k = poolKernel(side);
        return (side - k) / k + 1;
    }

    private static void requirePooledGrid(int w, int h, String what) {
        if (w <= 0 || h <= 0 || pooledSide(w) != POOLED_SIDE || pooledSide(h) != POOLED_SIDE) {
            throw new IllegalArgumentException("the yasa2 encoder pools its final " + w + "x" + h + " grid ("
                    + what + ") to " + (w > 0 ? pooledSide(w) : 0) + "x" + (h > 0 ? pooledSide(h) : 0) + ", not the "
                    + POOLED_SIDE + "x" + POOLED_SIDE + " token grid the prompt reserves");
        }
    }

    // ------------------------------------------------------------------ forward

    /**
     * Encodes one image tile to {@code 64} embedding rows, written against {@link VisionOps} so both executors
     * run it: the patch conv and the stage chain as device blocks, the response norms and the 8x8 pool through
     * the executor's convolutional ops ({@link VisionOps#depthwise}, {@link VisionOps#avgPool},
     * {@link VisionOps#grn}).
     *
     * @throws IllegalArgumentException when the image's final grid does not pool to 8x8 (named)
     */
    static float[][] encode(VisionOps ops, ClipImage image) {
        Objects.requireNonNull(image, "image");
        float eps = ops.hp().eps();
        int ps = ops.hp().patchSize();
        int width = image.width() / ps;
        int height = image.height() / ps;
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("image size " + image.width() + "x" + image.height() + " is smaller"
                    + " than the patch size " + ps);
        }
        VisionOps.Block x = ops.patchEmbed(ClipPatches.raster(image, ps, width, height), width * height);
        int ch = x.width();
        normChannels(ops, x, width * height, ch, vectorOrNull(ops, "v.patch_ln.weight"),
                vectorOrNull(ops, "v.patch_ln.bias"), eps);
        for (int s = 0; ; s++) {
            String p = "v.stage." + s + ".";
            TensorRef down = ops.model().projectorMatrices().get(p + "down.conv.weight");
            if (down == null && !ops.hasVector(p + "blk.0.dw.weight")) {
                break;
            }
            if (down != null) {
                normChannels(ops, x, width * height, ch, vectorOrNull(ops, p + "down.ln.weight"),
                        vectorOrNull(ops, p + "down.ln.bias"), eps);
                x = downsample(ops, x, width, height, ch, down, p + "down.conv.bias");
                width /= DOWN;
                height /= DOWN;
                ch = x.width();
            }
            for (int b = 0; ops.hasVector(p + "blk." + b + ".dw.weight"); b++) {
                block(ops, x, width, height, ch, p + "blk." + b + ".", eps);
            }
        }
        requirePooledGrid(width, height, "image " + image.width() + "x" + image.height());
        float[] pos = ops.hostVectorOrNull(POS);
        if (pos != null && pos.length == x.n() * ch) {
            ops.addHost(x, pos);
        }
        x = ops.avgPool(x, width, height, ch, poolKernel(height), poolKernel(width), POOLED_SIDE, POOLED_SIDE);
        int n = POOLED_SIDE * POOLED_SIDE;
        VisionOps.Block hidden = ops.linear("mm.0.weight", x, "mm.0.bias");
        ops.activate(hidden, ActOp.GELU_ERF);
        VisionOps.Block out = ops.linear("mm.2.weight", hidden, "mm.2.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    /** The channel layer norm of {@code rows} pixel rows of {@code ch}, the affine optional. */
    private static void normChannels(VisionOps ops, VisionOps.Block x, int rows, int ch, TensorRef weight,
                                     TensorRef bias, float eps) {
        ops.normByRef(x, NormKind.LAYER, weight, bias, eps, rows, ch);
    }

    /** The 2x2 stride-2 downsample convolution: the im2col and the kernel's matmul. */
    private static VisionOps.Block downsample(VisionOps ops, VisionOps.Block x, int w, int h, int ch, TensorRef kernel,
                                              String bias) {
        int ow = w / DOWN;
        int oh = h / DOWN;
        if (ow == 0 || oh == 0) {
            throw new IllegalArgumentException("the yasa2 downsample needs a grid of at least 2x2, got " + w + "x" + h);
        }
        VisionOps.Block cols = ops.im2colPadded(x, ch, w, h, DOWN, DOWN, 0, 0, ow, oh);
        return ops.linearKernel(kernel.name(), (int) kernel.ne(3), ch * DOWN * DOWN, cols, bias);
    }

    /** One ConvNeXt block on the executor's block: depthwise conv, channel norm, {@code pw1}, GRN, {@code pw2}. */
    private static void block(VisionOps ops, VisionOps.Block x, int w, int h, int ch, String p, float eps) {
        TensorRef dw = ops.model().projectorVectors().get(p + "dw.weight");
        int n = w * h;
        VisionOps.Block y = ops.depthwise(x, w, h, dw, DW_KERNEL, 1, DW_PAD, DW_PAD, w, h,
                vectorOrNull(ops, p + "dw.bias"));
        normChannels(ops, y, n, ch, vectorOrNull(ops, p + "ln.weight"), vectorOrNull(ops, p + "ln.bias"), eps);
        VisionOps.Block hid = ops.linear(p + "pw1.weight", y, p + "pw1.bias");
        ops.activate(hid, ActOp.GELU_ERF);
        ops.grn(hid, n, hid.width(), vectorOrNull(ops, p + "grn.weight"), vectorOrNull(ops, p + "grn.bias"));
        VisionOps.Block out = ops.linear(p + "pw2.weight", hid, p + "pw2.bias");
        ops.addBlock(x, out);
    }

    /** The projector vector {@code name}, or null when the file does not carry it. */
    private static TensorRef vectorOrNull(VisionOps ops, String name) {
        return ops.hasVector(name) ? ops.model().projectorVectors().get(name) : null;
    }
}
