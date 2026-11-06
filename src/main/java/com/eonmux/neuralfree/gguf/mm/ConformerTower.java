package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The NeMo conformer encoder of lfm2a and d1omni_a and its audio adapter (llama.cpp
 * tools/mtmd/models/conformer.cpp, {@code clip_graph_conformer}): the {@link ConformerOps.Stem} subsampling, the
 * conformer blocks ({@link ConformerOps.Block}, every norm at 1e-5) over the 512-wide sinusoidal relative-position
 * input, then the adapter — a layer norm ({@code mm.a.mlp.0}) and the GELU-erf MLP {@code mm.a.mlp.1} /
 * {@code mm.a.mlp.3} — and for d1omni_a the residual block {@code x + mlp.6(gelu(mlp.5(norm(x, mlp.4))))}.
 * Immutable weights; package-private executor support.
 */
final class ConformerTower implements FamilyEncoder {

    /** Width of the relative-position input conformer.cpp builds ({@code d_model = 512} in clip.cpp). */
    static final int POS_WIDTH = 512;
    /** Taps of the conv module's depthwise conv: the graph pads four frames on each side. */
    static final int CONV_KERNEL = 9;
    /** The tensor whose row count bounds the input frames (conformer.cpp asserts {@code ne[1] >= n_frames / 2}). */
    static final String POSITION_EMBD = "a.position_embd.weight";
    private static final double[] POS_FREQ = posFreq();

    private final int nEmbd;
    private final int outWidth;
    private final CpuMatVec matVec;
    private final ConformerOps.Stem stem;
    private final ConformerOps.Block[] blocks;
    private final Adapter adapter;
    private final Adapter residual;
    private final long positionRows;

    private ConformerTower(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                           Function<String, CpuTensorRef> refs) {
        ClipHparams hp = model.hparams();
        TensorCatalog tc = model.catalog();
        this.nEmbd = hp.nEmbd();
        this.outWidth = model.outputWidth();
        this.matVec = matVec;
        this.stem = ConformerOps.Stem.bind(tc, hp.audio().nMelBins(), floats, refs);
        this.blocks = new ConformerOps.Block[hp.nLayer()];
        for (int il = 0; il < blocks.length; il++) {
            blocks[il] = ConformerOps.Block.bind(tc, ConformerOps.Layout.LFM2A, il, hp.nHead(), CONV_KERNEL,
                    AudioOps.GRAPH_EPS, floats, refs);
        }
        this.adapter = new Adapter(0, floats, refs);
        this.residual = isD1omni(hp) ? new Adapter(4, floats, refs) : null;
        TensorRef pos = tc.get(POSITION_EMBD);
        this.positionRows = pos == null ? Long.MAX_VALUE : pos.ne(1);
    }

    /** {@code inv_freq[i] = exp(-(ln 1e4 / 512) * 2i)} of the clip.cpp LFM2A position fill. */
    private static double[] posFreq() {
        double[] f = new double[POS_WIDTH / 2];
        for (int i = 0; i < f.length; i++) {
            f[i] = Math.exp(-(Math.log(10000.0) / POS_WIDTH) * (2.0 * i));
        }
        return f;
    }

    private static boolean isD1omni(ClipHparams hp) {
        return ProjectorType.D1OMNI_A.equals(hp.projectorType());
    }

    /** Shape-checks the stem, the blocks, the adapter and the d1omni_a residual block against the hparams. */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        int e = hp.nEmbd();
        if (e % hp.nHead() != 0) {
            throw new IllegalArgumentException("the conformer splits clip.audio.embedding_length " + e + " over "
                    + hp.nHead() + " heads unevenly");
        }
        TensorRef pos = tc.get(POSITION_EMBD);
        if (pos != null) {
            // clip.cpp reports the lfm2a output width as this tensor's ne[0]
            shape(tc, pos, hp.projectionDim(), pos.ne(1));
        }
        ConformerOps.Stem.validate(tc, hp.audio().nMelBins(), e);
        for (int il = 0; il < hp.nLayer(); il++) {
            ConformerOps.Block.validate(tc, ConformerOps.Layout.LFM2A, il, e, hp.nHead(), POS_WIDTH, CONV_KERNEL);
        }
        int out = Adapter.validate(tc, 0, e);
        if (out != hp.projectionDim()) {
            throw new IllegalArgumentException("mm.a.mlp.3.weight projects to " + out
                    + " columns, clip.audio.projection_dim is " + hp.projectionDim());
        }
        if (isD1omni(hp)) {
            int back = Adapter.validate(tc, 4, out);
            if (back != out) {
                throw new IllegalArgumentException("mm.a.mlp.6.weight projects to " + back
                        + " columns, the d1omni_a residual block adds it to the " + out + "-wide adapter output");
            }
        }
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static ConformerTower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                               Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(matVec, "matVec");
        return new ConformerTower(model, matVec, floats, refs);
    }

    /**
     * The stem, the blocks and the adapter: {@code ConformerOps.subsampled(nLen)} rows.
     *
     * @throws IllegalArgumentException when the file's {@value #POSITION_EMBD} holds fewer than {@code nLen / 2}
     *                                  rows, the bound conformer.cpp asserts
     */
    @Override
    public float[][] encode(MelChunk chunk) {
        if (chunk.nLen() / 2 > positionRows) {
            throw new IllegalArgumentException("a mel chunk of " + chunk.nLen() + " frames needs " + chunk.nLen() / 2
                    + " rows of " + POSITION_EMBD + ", the file has " + positionRows);
        }
        float[] x = stem.run(matVec, chunk);
        int n = x.length / nEmbd;
        int posRows = 2 * n - 1;
        float[] posEmb = ConformerOps.posEmb(posRows, POS_WIDTH, POS_FREQ);
        for (ConformerOps.Block b : blocks) {
            b.forward(matVec, x, n, posEmb, posRows);
        }
        float[] out = adapter.apply(matVec, x, n);
        if (residual != null) {
            AudioOps.addScaled(out, residual.apply(matVec, out, n), 1f);
        }
        return VisionRows.tokenRows(out, outWidth);
    }

    /**
     * {@code build_ffn(norm(x, mlp.<i>), mlp.<i+1>, GELU-erf, mlp.<i+2 or i+3>)}: the adapter at {@code i = 0}
     * (up {@code mlp.1}, down {@code mlp.3}) and the d1omni_a residual block at {@code i = 4} (up {@code mlp.5},
     * down {@code mlp.6}), every tensor biased, the norm at 1e-5.
     */
    private static final class Adapter {
        private final float[] normW;
        private final float[] normB;
        private final CpuTensorRef up;
        private final float[] upB;
        private final CpuTensorRef down;
        private final float[] downB;

        Adapter(int first, Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            this.normW = floats.apply(name(first, "weight"));
            this.normB = floats.apply(name(first, "bias"));
            this.up = refs.apply(name(first + 1, "weight"));
            this.upB = floats.apply(name(first + 1, "bias"));
            this.down = refs.apply(name(downIndex(first), "weight"));
            this.downB = floats.apply(name(downIndex(first), "bias"));
        }

        private static int downIndex(int first) {
            return first == 0 ? 3 : first + 2;
        }

        private static String name(int i, String suffix) {
            return "mm.a.mlp." + i + "." + suffix;
        }

        /** Shape-checks the block over {@code in} columns; returns its output width. */
        static int validate(TensorCatalog tc, int first, int in) {
            shape(tc, tc.require(name(first, "weight")), in);
            shape(tc, tc.require(name(first, "bias")), in);
            TensorRef up = tc.require(name(first + 1, "weight"));
            long ff = up.ne(1);
            shape(tc, up, in, ff);
            shape(tc, tc.require(name(first + 1, "bias")), ff);
            TensorRef down = tc.require(name(downIndex(first), "weight"));
            long out = down.ne(1);
            shape(tc, down, ff, out);
            shape(tc, tc.require(name(downIndex(first), "bias")), out);
            return (int) out;
        }

        float[] apply(CpuMatVec matVec, float[] x, int n) {
            float[] h = x.clone();
            AudioOps.layerNorm(h, n, x.length / n, normW, normB, AudioOps.GRAPH_EPS);
            return AudioOps.ffn(matVec, h, n, up, upB, ActOp.GELU_ERF, down, downB);
        }
    }
}
