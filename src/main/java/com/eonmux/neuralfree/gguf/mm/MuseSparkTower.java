package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.elements;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The Muse Spark audio tower (llama.cpp tools/mtmd/models/muse-spark-audio.cpp): a causal conv pyramid over the
 * raw 24 kHz waveform, without attention or spectrogram. Each stage folds {@code patch} consecutive rows into the
 * channel dimension, projects them ({@code a.patch.N}) and RMS-normalizes, then runs its residual blocks: a
 * depthwise conv over time padded {@code kernel - 1} frames on the left ({@code a.conv1d.N}), the RMS norm
 * {@code a.conv1d.N.norm}, a GELU-erf MLP ({@code a.blk.N.ffn_up/down}) and the residual. The last stage width
 * goes through the RMS norm {@code a.pre_ln}, the projection {@code mm.a.fc} and the RMS norm {@code a.post_ln}.
 * All norms use {@code clip.audio.attention.layer_norm_epsilon}. Immutable weights; package-private executor
 * support.
 */
final class MuseSparkTower implements FamilyEncoder {

    /** One stage's fold and projection. */
    private static final class Stage {
        final int patch;
        final int width;
        final CpuTensorRef proj;
        final float[] projB;
        final float[] normW;

        Stage(int patch, int width, CpuTensorRef proj, float[] projB, float[] normW) {
            this.patch = patch;
            this.width = width;
            this.proj = proj;
            this.projB = projB;
            this.normW = normW;
        }
    }

    /** One residual block: depthwise causal conv, RMS norm, GELU-erf MLP. */
    private static final class Block {
        final float[] convW;
        final float[] convB;
        final float[] convNormW;
        final CpuTensorRef up;
        final float[] upB;
        final CpuTensorRef down;
        final float[] downB;

        Block(int il, Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            this.convW = floats.apply(conv(il, "weight"));
            this.convB = floats.apply(conv(il, "bias"));
            this.convNormW = floats.apply(conv(il, "norm.weight"));
            this.up = refs.apply(blk(il, "ffn_up.weight"));
            this.upB = floats.apply(blk(il, "ffn_up.bias"));
            this.down = refs.apply(blk(il, "ffn_down.weight"));
            this.downB = floats.apply(blk(il, "ffn_down.bias"));
        }
    }

    private final CpuMatVec matVec;
    private final float eps;
    private final int kernel;
    private final Stage[] stages;
    private final Block[][] blocks;
    private final float[] preLnW;
    private final CpuTensorRef fc;
    private final float[] fcB;
    private final float[] postLnW;
    private final int outWidth;

    private MuseSparkTower(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                           Function<String, CpuTensorRef> refs) {
        ClipHparams hp = model.hparams();
        ClipAudioHparams a = hp.audio();
        this.matVec = matVec;
        this.eps = hp.eps();
        this.kernel = a.convKernel();
        int[] patches = a.patchSizes();
        int[] widths = a.hiddenSizes();
        int[] depths = a.depths();
        this.stages = new Stage[patches.length];
        this.blocks = new Block[patches.length][];
        int il = 0;
        int prev = 1;
        for (int s = 0; s < patches.length; s++) {
            CpuTensorRef proj = CpuVitTower.matrix(refs.apply(patch(s, "weight")), widths[s],
                    (long) prev * patches[s]);
            stages[s] = new Stage(patches[s], widths[s], proj, floats.apply(patch(s, "bias")),
                    floats.apply(patch(s, "norm.weight")));
            blocks[s] = new Block[depths[s]];
            for (int i = 0; i < depths[s]; i++, il++) {
                blocks[s][i] = new Block(il, floats, refs);
            }
            prev = widths[s];
        }
        this.preLnW = floats.apply("a.pre_ln.weight");
        this.fc = refs.apply("mm.a.fc.weight");
        this.fcB = floats.apply("mm.a.fc.bias");
        this.postLnW = floats.apply("a.post_ln.weight");
        this.outWidth = model.outputWidth();
    }

    private static String patch(int stage, String suffix) {
        return "a.patch." + stage + "." + suffix;
    }

    private static String conv(int il, String suffix) {
        return "a.conv1d." + il + "." + suffix;
    }

    private static String blk(int il, String suffix) {
        return "a.blk." + il + "." + suffix;
    }

    /**
     * Shape-checks every stage and block against the stage widths, patch sizes and kernel of the hparams, and the
     * projection against {@code clip.audio.projection_dim}.
     */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        ClipAudioHparams a = hp.audio();
        int[] patches = a.patchSizes();
        int[] widths = a.hiddenSizes();
        int[] depths = a.depths();
        int il = 0;
        int prev = 1;
        for (int s = 0; s < patches.length; s++) {
            int c = widths[s];
            shape(tc, tc.require(patch(s, "weight")), (long) prev * patches[s], c);
            elements(tc, tc.require(patch(s, "bias")), c);
            shape(tc, tc.require(patch(s, "norm.weight")), c);
            for (int i = 0; i < depths[s]; i++, il++) {
                validateBlock(tc, il, c, a.convKernel());
            }
            prev = c;
        }
        shape(tc, tc.require("a.pre_ln.weight"), prev);
        shape(tc, tc.require("mm.a.fc.weight"), prev, hp.projectionDim());
        elements(tc, tc.require("mm.a.fc.bias"), hp.projectionDim());
        shape(tc, tc.require("a.post_ln.weight"), hp.projectionDim());
    }

    private static void validateBlock(TensorCatalog tc, int il, int c, int kernel) {
        TensorRef convW = tc.require(conv(il, "weight"));
        if (convW.rank() == 2) {
            shape(tc, convW, kernel, c);
        } else {
            shape(tc, convW, kernel, 1, c);
        }
        elements(tc, tc.require(conv(il, "bias")), c);
        shape(tc, tc.require(conv(il, "norm.weight")), c);
        TensorRef up = tc.require(blk(il, "ffn_up.weight"));
        long ff = up.ne(1);
        shape(tc, up, c, ff);
        elements(tc, tc.require(blk(il, "ffn_up.bias")), ff);
        shape(tc, tc.require(blk(il, "ffn_down.weight")), ff, c);
        elements(tc, tc.require(blk(il, "ffn_down.bias")), c);
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static MuseSparkTower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                               Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(matVec, "matVec");
        return new MuseSparkTower(model, matVec, floats, refs);
    }

    /**
     * Encodes one waveform row ({@code [1][n]}, {@code n} a multiple of the samples per token): one output row per
     * {@code product(patch_sizes)} samples.
     */
    @Override
    public float[][] encode(MelChunk chunk) {
        int fold = 1;
        for (Stage s : stages) {
            fold *= s.patch;
        }
        if (chunk.nMel() != 1 || chunk.nLen() % fold != 0) {
            throw new IllegalArgumentException("the muse-spark tower reads one waveform row of a multiple of " + fold
                    + " samples, got " + chunk.nMel() + " rows of " + chunk.nLen());
        }
        float[] cur = chunk.data().clone();
        int rows = chunk.nLen();
        for (int s = 0; s < stages.length; s++) {
            Stage st = stages[s];
            // folding consecutive rows into channels re-views the token-major data; nothing moves
            rows /= st.patch;
            cur = AudioOps.linear(matVec, st.proj, st.projB, cur, rows);
            AudioOps.rmsNorm(cur, rows, st.width, st.normW, null, eps);
            for (Block b : blocks[s]) {
                cur = block(b, cur, rows, st.width);
            }
        }
        AudioOps.rmsNorm(cur, rows, stages[stages.length - 1].width, preLnW, null, eps);
        float[] out = AudioOps.linear(matVec, fc, fcB, cur, rows);
        AudioOps.rmsNorm(out, rows, outWidth, postLnW, null, eps);
        return VisionRows.tokenRows(out, outWidth);
    }

    /** One residual block over {@code rows} rows of {@code width}: a new array of the block output. */
    private float[] block(Block b, float[] x, int rows, int width) {
        float[] conv = AudioOps.depthwiseConv1d(x, rows, width, b.convW, kernel, kernel - 1, 0);
        VisionRows.addBias(conv, b.convB);
        AudioOps.rmsNorm(conv, rows, width, b.convNormW, null, eps);
        float[] y = AudioOps.ffn(matVec, conv, rows, b.up, b.upB, ActOp.GELU_ERF, b.down, b.downB);
        VisionRows.addInto(x, y);
        return y;
    }
}
