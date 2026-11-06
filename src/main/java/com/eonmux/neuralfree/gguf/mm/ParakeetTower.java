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
 * The Parakeet (FastConformer) encoder and its sound projector (llama.cpp tools/mtmd/models/parakeet.cpp,
 * {@code clip_graph_parakeet}): the {@link ConformerOps.Stem} subsampling, the conformer blocks
 * ({@link ConformerOps.Block} without projection biases, a batch-norm conv module of {@code
 * clip.audio.conv_kernel_size} taps, norms at {@code clip.audio.attention.layer_norm_epsilon}) over the
 * {@code n_embd}-wide sinusoidal relative-position input, then the projector: an RMS norm at 1e-6 times
 * {@code mm.a.norm_pre.weight} and the bias-free ReLU-squared MLP {@code mm.a.mlp.1} / {@code mm.a.mlp.2}.
 * Past {@value #LOCAL_ATTN_THRESHOLD} rows the attention is local, each row seeing the
 * {@value #LOCAL_ATTN_WINDOW} rows on either side. The mel filterbank and window the frontend uses are the file
 * tensors {@value #MEL_FILTERS} and {@value #WINDOW}. Immutable weights; package-private executor support.
 */
final class ParakeetTower implements FamilyEncoder {

    /** The NeMo filterbank the frontend reads from the file ({@code n_mel * (n_fft / 2 + 1)} values). */
    static final String MEL_FILTERS = "a.mel_filters";
    /** The analysis window the frontend reads from the file ({@code window_len} values). */
    static final String WINDOW = "a.window";
    /** Row count above which the attention turns local (parakeet.cpp). */
    static final int LOCAL_ATTN_THRESHOLD = 8192;
    /** Rows on each side a local attention row sees (parakeet.cpp). */
    static final int LOCAL_ATTN_WINDOW = 128;
    private static final float PROJECTOR_EPS = 1e-6f;

    private final int nEmbd;
    private final int outWidth;
    private final CpuMatVec matVec;
    private final ConformerOps.Stem stem;
    private final ConformerOps.Block[] blocks;
    private final double[] posFreq;
    private final float[] normPre;
    private final CpuTensorRef up;
    private final CpuTensorRef down;

    private ParakeetTower(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                          Function<String, CpuTensorRef> refs) {
        ClipHparams hp = model.hparams();
        TensorCatalog tc = model.catalog();
        this.nEmbd = hp.nEmbd();
        this.outWidth = model.outputWidth();
        this.matVec = matVec;
        this.stem = ConformerOps.Stem.bind(tc, hp.audio().nMelBins(), floats, refs);
        this.blocks = new ConformerOps.Block[hp.nLayer()];
        for (int il = 0; il < blocks.length; il++) {
            blocks[il] = ConformerOps.Block.bind(tc, ConformerOps.Layout.PARAKEET, il, hp.nHead(),
                    hp.audio().convKernel(), hp.eps(), floats, refs);
        }
        this.posFreq = posFreq(nEmbd);
        this.normPre = floats.apply("mm.a.norm_pre.weight");
        this.up = refs.apply("mm.a.mlp.1.weight");
        this.down = refs.apply("mm.a.mlp.2.weight");
    }

    /** {@code freqs[k] = expf(-(2k * logf(1e4) / n_state))} in float, as clip.cpp fills {@code pos_freqs}. */
    private static double[] posFreq(int nState) {
        float log10000 = (float) Math.log(10000.0f);
        double[] f = new double[nState / 2];
        for (int k = 0; k < f.length; k++) {
            f[k] = (float) Math.exp(-((float) (k * 2) * log10000 / (float) nState));
        }
        return f;
    }

    /** Shape-checks the frontend tensors, the stem, the blocks and the projector against the hparams. */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        int e = hp.nEmbd();
        if (e % hp.nHead() != 0 || e % 2 != 0) {
            throw new IllegalArgumentException("the parakeet encoder needs an even clip.audio.embedding_length"
                    + " split evenly over the heads, got " + e + " over " + hp.nHead());
        }
        ClipAudioHparams a = hp.audio();
        elements(tc, tc.require(MEL_FILTERS), (long) a.nMelBins() * (a.nFft() / 2 + 1));
        elements(tc, tc.require(WINDOW), a.windowLen());
        ConformerOps.Stem.validate(tc, a.nMelBins(), e);
        for (int il = 0; il < hp.nLayer(); il++) {
            ConformerOps.Block.validate(tc, ConformerOps.Layout.PARAKEET, il, e, hp.nHead(), e, a.convKernel());
        }
        shape(tc, tc.require("mm.a.norm_pre.weight"), e);
        TensorRef up = tc.require("mm.a.mlp.1.weight");
        long ff = up.ne(1);
        shape(tc, up, e, ff);
        shape(tc, tc.require("mm.a.mlp.2.weight"), ff, hp.projectionDim());
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static ParakeetTower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                              Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(matVec, "matVec");
        return new ParakeetTower(model, matVec, floats, refs);
    }

    /** Rows each side of a row its attention sees: all of them up to the local threshold. */
    static int attentionReach(int n) {
        return n > LOCAL_ATTN_THRESHOLD ? LOCAL_ATTN_WINDOW : n - 1;
    }

    /** The stem, the blocks and the projector: {@code ceil(nLen / 8)} rows. */
    @Override
    public float[][] encode(MelChunk chunk) {
        float[] x = stem.run(matVec, chunk);
        int n = x.length / nEmbd;
        int posRows = 2 * attentionReach(n) + 1;
        float[] posEmb = ConformerOps.posEmb(posRows, nEmbd, posFreq);
        for (ConformerOps.Block b : blocks) {
            b.forward(matVec, x, n, posEmb, posRows);
        }
        AudioOps.rmsNorm(x, n, nEmbd, normPre, null, PROJECTOR_EPS);
        float[] out = AudioOps.ffn(matVec, x, n, up, null, ActOp.RELU_SQR, down, null);
        return VisionRows.tokenRows(out, outWidth);
    }
}
