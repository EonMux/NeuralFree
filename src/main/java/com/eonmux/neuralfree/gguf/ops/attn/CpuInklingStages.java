package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.ssm.CpuSsmConv;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * Inkling's query-side stages of {@link CpuAttentionMixer}: the {@code attn_rel} projection of the block input and
 * its {@code [dRel][relExtent]} relative-logit table, the log scaling of a full layer's normed queries, and the
 * short convolutions of the K and V streams with the layer's conv state (vLLM {@code InklingAttention.k_sconv}/
 * {@code v_sconv} over the {@code conv_owner} cache). Not thread-safe; driven by the owning mixer's step thread.
 */
final class CpuInklingStages {

    private static final String T_REL_OUT = "attn.rel_out";
    private static final String T_REL_TABLE = "attn.rel_table";
    private static final String T_CONV = "attn.sconv";

    private final CpuAttentionMixer mixer;

    private CpuTensorRef wr;
    private CpuProjectionDeltas.Site rSite;
    private float[] rBias;
    private int relDim;
    private int relExtent;
    private int logFloor;
    private float logAlpha;
    /** The shared {@code [dRel][relExtent]} relative projection. */
    private float[] relProj;
    private float[] relOut;
    private float[] relTable;
    private float[] convKw;
    private float[] convKvWeight;
    private float[] convTmp;
    private float[] convState;
    private int convRow;
    private int stateSlots;

    /**
     * Inkling's query-side tensors: the {@code attn_rel} projection and its {@code [dRel][relExtent]} table, and the
     * short-convolution kernels of the K and V streams with the layer's conv state (vLLM
     * {@code InklingAttention.k_sconv}/{@code v_sconv} over the {@code conv_owner} cache).
     *
     * @throws IllegalArgumentException when a tensor's shape differs from the layer geometry
     * @throws IllegalStateException    when the layer has no recurrent state for the convolutions
     */
    CpuInklingStages(CpuAttentionMixer mixer, CpuWeights w) {
        this.mixer = mixer;
        CpuForwardContext ctx = mixer.ctx();
        ModelHparams hp = ctx.spec().hparams();
        relDim = hp.inklingRelDim();
        relExtent = mixer.spec().relBiasExtent();
        logFloor = hp.inklingLogScalingFloor();
        logAlpha = hp.inklingLogScalingAlpha();
        long rWidth = (long) mixer.spec().nHead() * relDim;
        wr = w.layer(mixer.layer(), TensorNames.weight(TensorNames.ATTN_REL));
        QkvProjection.checkShape(TensorNames.weight(TensorNames.ATTN_REL), wr.cols(), wr.rows(), mixer.nEmbd(),
                rWidth);
        rSite = mixer.serveDeltas(TensorNames.weight(TensorNames.ATTN_REL), wr);
        rBias = w.layerOrNull(mixer.layer(), TensorNames.bias(TensorNames.ATTN_REL)) != null
                ? mixer.vector(w, TensorNames.bias(TensorNames.ATTN_REL), rWidth) : null;
        relProj = w.layerFloatsOrNull(mixer.layer(), TensorNames.weight(TensorNames.ATTN_REL_PROJ));
        QkvProjection.checkLength(TensorNames.weight(TensorNames.ATTN_REL_PROJ),
                relProj == null ? 0 : relProj.length, (long) relDim * relExtent);
        CpuScratch scratch = ctx.scratch();
        relOut = scratch.temp(T_REL_OUT, (int) ((long) mixer.ubatch() * rWidth));
        relTable = scratch.temp(T_REL_TABLE, (int) ((long) mixer.ubatch() * mixer.spec().nHead() * relExtent));
        int kvDim = mixer.spec().nHeadKv() * mixer.spec().headDimK();
        convKw = convWeights(w, TensorNames.ATTN_SCONV_K, kvDim);
        convKvWeight = convWeights(w, TensorNames.ATTN_SCONV_V, kvDim);
        convTmp = scratch.temp(T_CONV, (int) ((long) mixer.ubatch()
                * Math.max(kvDim, mixer.spec().vDim())));
        RecurrentStateAccessor state = ctx.memory().recurrent(mixer.layer().il());
        if (state == null) {
            throw new IllegalStateException("layer " + mixer.layer().il() + " runs Inkling short convolutions but has"
                    + " no recurrent state");
        }
        convState = state.tensor(LayerMemoryPlan.STATE_CONV);
        convRow = state.slotFloats(LayerMemoryPlan.STATE_CONV);
        stateSlots = state.slots();
    }

    /** The {@code [channels][convWidth]} kernel of one short-convolution stream. */
    private float[] convWeights(CpuWeights w, String base, int channels) {
        float[] v = w.layerFloatsOrNull(mixer.layer(), TensorNames.weight(base));
        int width = mixer.layer().recurrent().dConv();
        QkvProjection.checkLength(TensorNames.weight(base), v == null ? 0 : v.length, (long) channels * width);
        return v;
    }

    /** {@code attn_rel @ x}: the query-side relative output of every token row into {@link #relOut}. */
    void projectRel(int n, float[] in) {
        int width = (int) wr.rows();
        mixer.matVec().matmul(wr, in, 0, relOut, 0, n);
        rSite.applyRows(n, in, 0, relOut, 0);
        if (rBias != null) {
            CpuOps.addRowBias(relOut, 0, n, width, rBias, 0);
        }
    }

    /**
     * The K and V short convolutions of the step (vLLM {@code fused_qkvr_prep}): the raw projections feed each
     * stream's conv state and the convolution adds back into the projection, so {@code kv = conv(kv) + kv}. A paged
     * step runs one segment at a time, each over its own sequence's state row.
     */
    void convKv(int n, float[] k, float[] v) {
        PagedStep pagedStep = mixer.pagedStep();
        if (pagedStep == null) {
            convSegment(0, n, k, 0, v, 0);
            return;
        }
        for (int seg = 0; seg < pagedStep.nSeg(); seg++) {
            int start = pagedStep.segStart(seg);
            int len = pagedStep.segLen(seg);
            convSegment(pagedStep.segSeq(seg), len, k, start * mixer.spec().kDim(), v, start * mixer.spec().vDim());
        }
    }

    private void convSegment(int seq, int len, float[] k, int kOff, float[] v, int vOff) {
        if (len < 1) {
            return;
        }
        if (seq < 0 || seq >= stateSlots) {
            throw new UnsupportedOperationException("layer " + mixer.layer().il() + " runs Inkling sequence " + seq
                    + ", the memory keeps " + stateSlots + " state slots");
        }
        int base = seq * convRow;
        int kvDim = mixer.spec().nHeadKv() * mixer.spec().headDimK();
        convStream(k, kOff, mixer.spec().kDim(), kvDim, convKw, base, RecurrentSpec.INKLING_K, len);
        convStream(v, vOff, mixer.spec().vDim(), mixer.spec().nHeadKv() * mixer.spec().headDimV(), convKvWeight, base,
                RecurrentSpec.INKLING_V, len);
    }

    /** One stream: {@code x = conv(x) + x} over the stream's conv state. */
    private void convStream(float[] x, int off, int stride, int channels, float[] weight, int stateBase, int stream,
                            int len) {
        int width = mixer.layer().recurrent().dConv();
        CpuSsmConv.forward(channels, width, len, 0, x, off, stride, weight, null, convState, stateBase,
                mixer.layer().recurrent().inklingStreamOffset(stream), convTmp, 0, channels);
        for (int t = 0; t < len; t++) {
            int xb = off + t * stride;
            int cb = t * channels;
            for (int c = 0; c < channels; c++) {
                x[xb + c] += convTmp[cb + c];
            }
        }
    }

    /**
     * The log scaling of the step's normed queries (vLLM scales them by the token's log factor); a no-op on an
     * SWA layer.
     */
    void scaleQueries(int n, float[] q, int[] pos) {
        if (!mixer.spec().isSwaLayer()) {
            InklingMath.scaleQueries(n, mixer.spec().nHead(), mixer.spec().headDimK(), mixer.spec().qDim(), q, 0,
                    pos, 0, logFloor, logAlpha);
        }
    }

    /** The relative-logit table of the step's tokens ({@link InklingMath#relLogits}), scaled on a full layer. */
    void relLogits(int n, int[] pos) {
        boolean scaled = !mixer.spec().isSwaLayer();
        InklingMath.relLogits(n, mixer.spec().nHead(), relDim, relExtent, relOut, relProj, pos, 0,
                scaled ? logFloor : 0, logAlpha, relTable);
    }

    /** The step's relative-logit table, read by the attention params. */
    float[] relTable() {
        return relTable;
    }
}
