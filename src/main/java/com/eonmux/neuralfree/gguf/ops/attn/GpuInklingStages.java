package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * Inkling's stages of {@link GpuAttentionMixer}: the {@code attn_rel} projection of the block input and its
 * {@code [dRel][relExtent]} relative-logit table, the short convolutions of the K and V streams over the layer's conv
 * state (vLLM {@code InklingAttention.k_sconv}/{@code v_sconv}, {@code fused_qkvr_prep}), and the post-norm stages —
 * the relative-logit table of the step's tokens and the log scaling of a full layer's normed queries. The convolutions
 * add back into the projection ({@code kv = conv(kv) + kv}), so they run before the K norm normalizes the rows it
 * stores, and the log scaling runs after the norms, on the normed queries. Not thread-safe; driven by the owning
 * mixer's step thread.
 */
final class GpuInklingStages {

    private static final String T_REL_OUT = "attn.rel_out";
    private static final String T_REL_TABLE = "attn.rel_table";
    private static final String T_CONV = "attn.sconv";
    /** Argument indices of {@code LLM_SSM_Conv}'s row offsets (kernels/llm/recurrent/ssm_conv.cl). */
    private static final int ARG_CONV_X_OFF = 5;
    private static final int ARG_CONV_OUT_OFF = 7;
    private static final int ARG_BIAS_SEQ = 1;
    private static final int ARG_BIAS_DATA = 3;

    private final GpuAttentionMixer mixer;

    private GpuWeight wr;
    private GpuProjectionDeltas.Site rSite;
    private GpuBuffer rBuf;
    private GpuBuffer relTableBuf;
    private GpuBuffer relProjBuf;
    private PreparedDispatch rBiasAdd;
    private int relDim;
    private int relExtent;
    private int logFloor;
    private float logAlpha;
    private TokenStep relLogitsStep;
    private TokenStep queryScaleStep;
    private PreparedDispatch convK;
    private PreparedDispatch convV;
    private TokenStep convKStep;
    private TokenStep convVStep;
    private GpuBuffer convTemp;
    private GpuBuffer convStateBuf;
    private GpuBuffer convStateRows;
    private boolean routeConvState;
    private int convStateStride;
    private int convKChannels;
    private int convVChannels;

    /**
     * Binds the query-side tensors and the convolution steps: the {@code attn_rel} projection and its
     * {@code [dRel][relExtent]} table, the K/V short convolutions over the layer's packed conv state, routed per
     * sequence on a paged memory.
     *
     * @throws IllegalArgumentException when a tensor's shape differs from the layer geometry
     * @throws IllegalStateException    when the layer has no conv state for the short convolutions
     */
    GpuInklingStages(GpuAttentionMixer mixer) {
        this.mixer = mixer;
        bindInputs(mixer.ctx().scratch());
    }

    /**
     * The steps that run before the K norm: the two short convolutions of the K and V streams, each over the layer's
     * conv state (vLLM {@code InklingAttention.k_sconv}/{@code v_sconv}, {@code fused_qkvr_prep}). The convolution
     * adds back into the projection ({@code kv = conv(kv) + kv}), so it reads the raw rows into a temporary and adds
     * them, and it must run before the K norm normalizes the rows it stores.
     */
    private void bindInputs(GpuScratch scratch) {
        relDim = mixer.hparams().inklingRelDim();
        relExtent = mixer.spec().relBiasExtent();
        logFloor = mixer.hparams().inklingLogScalingFloor();
        logAlpha = mixer.hparams().inklingLogScalingAlpha();
        int rWidth = mixer.spec().nHead() * relDim;
        wr = mixer.weights().layer(mixer.layer(), TensorNames.weight(TensorNames.ATTN_REL));
        QkvProjection.checkShape(TensorNames.weight(TensorNames.ATTN_REL), wr.cols(), wr.rows(),
                mixer.hparams().nEmbd(), rWidth);
        rBuf = inklingTemp(scratch, T_REL_OUT, (long) mixer.ubatch() * rWidth).buffer();
        rSite = mixer.serveDeltas(TensorNames.weight(TensorNames.ATTN_REL), wr);
        relProjBuf = mixer.vector(mixer.weights().layer(mixer.layer(),
                TensorNames.weight(TensorNames.ATTN_REL_PROJ)), (long) relDim * relExtent);
        relTableBuf = inklingTemp(scratch, T_REL_TABLE, (long) mixer.ubatch() * mixer.spec().nHead() * relExtent)
                .buffer();
        if (mixer.weights().layerOrNull(mixer.layer(), TensorNames.bias(TensorNames.ATTN_REL)) != null) {
            GpuBuffer rBiasValues = mixer.vector(TensorNames.bias(TensorNames.ATTN_REL), rWidth);
            rBiasAdd = mixer.resources().prepare(GpuOps.BATCH_BIAS_ADD,
                    new long[]{mixer.roundUp((long) mixer.ubatch() * rWidth)}, new long[]{mixer.local()},
                    mixer.bindArgs(rWidth, mixer.ubatch(), rBiasValues, rBuf));
        }
        bindConvs(scratch);
    }

    /** The K/V short convolutions and the layer's packed conv state. */
    private void bindConvs(GpuScratch scratch) {
        RecurrentSpec r = mixer.layer().recurrent();
        int width = r.dConv();
        convKChannels = mixer.spec().nHeadKv() * mixer.spec().headDimK();
        convVChannels = mixer.spec().nHeadKv() * mixer.spec().headDimV();
        GpuBuffer kw = mixer.vector(TensorNames.weight(TensorNames.ATTN_SCONV_K), (long) convKChannels * width);
        GpuBuffer vw = mixer.vector(TensorNames.weight(TensorNames.ATTN_SCONV_V), (long) convVChannels * width);
        convTemp = inklingTemp(scratch, T_CONV, (long) mixer.ubatch() * Math.max(convKChannels, convVChannels))
                .buffer();
        GpuRecurrentState state = mixer.ctx().memory().recurrent(mixer.layer().il());
        if (state == null) {
            throw new IllegalStateException("layer " + mixer.layer().il() + " runs Inkling short convolutions but has"
                    + " no recurrent state");
        }
        convStateStride = -1;
        for (LayerMemoryPlan.StateTensor t : state.tensors()) {
            if (t.name().equals(LayerMemoryPlan.STATE_CONV)) {
                convStateStride = (int) t.floats();
            }
        }
        if (convStateStride < 0) {
            throw new IllegalStateException("layer " + mixer.layer().il() + " has no conv state for its short"
                    + " convolutions");
        }
        if (convStateStride != r.convStateFloats()) {
            throw new IllegalStateException("layer " + mixer.layer().il() + ": conv state holds " + convStateStride
                    + " floats, the layer geometry sizes it at " + r.convStateFloats());
        }
        convStateBuf = state.tensor(LayerMemoryPlan.STATE_CONV);
        routeConvState = mixer.ctx().memory() instanceof com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory
                && mixer.ctx().memory().pagedSequences() != null;
        convStateRows = mixer.ctx().memory().stepInputs().stateRows();
        convK = inklingConv(mixer.kBuf(), kw, convKChannels, r.inklingStreamOffset(RecurrentSpec.INKLING_K));
        convV = inklingConv(mixer.vBuf(), vw, convVChannels, r.inklingStreamOffset(RecurrentSpec.INKLING_V));
        convKStep = inklingConvStep(convK);
        convVStep = inklingConvStep(convV);
        mixer.input(convKStep, true);
        mixer.input(convVStep, true);
    }

    /**
     * One stream's {@code LLM_SSM_Conv} over the layer's conv state: {@code x} is the stream's projection buffer
     * (its raw rows), the convolution writes the temporary and the step adds it back into {@code x}. The token
     * count, the row offsets and the state row are re-set per segment.
     */
    private PreparedDispatch inklingConv(GpuBuffer x, GpuBuffer w, int channels, int stateOff) {
        Object[] args = {channels, mixer.layer().recurrent().dConv(), 1, 0,
                x == mixer.kBuf() ? mixer.spec().kDim() : mixer.spec().vDim(), 0, channels, 0, stateOff,
                convStateStride, x, w, mixer.ctx().placeholders().readOnlyFloats(), convStateBuf, convStateRows,
                convTemp};
        if (args.length != SsmKernels.CONV_ARG_COUNT) {
            throw new IllegalStateException(SsmKernels.SSM_CONV + " takes " + SsmKernels.CONV_ARG_COUNT
                    + " arguments, " + args.length + " built");
        }
        return mixer.resources().prepare(SsmKernels.SSM_CONV,
                new long[]{SsmKernels.elementwiseGlobal(channels, SsmKernels.ELEMENTWISE_LOCAL)},
                new long[]{SsmKernels.ELEMENTWISE_LOCAL}, args);
    }

    /** The step of one conv dispatch: one launch per step segment, each over its sequence's state row. */
    private TokenStep inklingConvStep(PreparedDispatch conv) {
        TokenStep step = new TokenStep(conv, 0, 1, 0, 0, -1, mixer.local());
        step.segments = (n, paged) -> runConv(conv, n, paged);
        return step;
    }

    /**
     * Runs a conv dispatch over the step's segments: one launch per segment with its token count and row offsets,
     * routed to its sequence's state row. A step without a paged layout is one segment at row 0.
     */
    private void runConv(PreparedDispatch conv, int n, PagedStep paged) {
        if (paged == null || paged.nSeg() <= 1) {
            convSegment(conv, 0, n, 0);
            return;
        }
        for (int seg = 0; seg < paged.nSeg(); seg++) {
            convSegment(conv, paged.segSeq(seg), paged.segLen(seg), paged.segStart(seg));
        }
    }

    private void convSegment(PreparedDispatch conv, int seq, int len, int start) {
        if (len < 1) {
            return;
        }
        int channels = conv == convK ? convKChannels : convVChannels;
        int stride = conv == convK ? mixer.spec().kDim() : mixer.spec().vDim();
        if (routeConvState) {
            GpuMixer.routeStateRow(seq, convStateRows, mixer.ops().queue());
        }
        conv.setInt(SsmKernels.ARG_CONV_N_TOKENS, len);
        conv.setInt(ARG_CONV_X_OFF, start * stride);
        conv.setInt(ARG_CONV_OUT_OFF, 0);
        conv.dispatch();
        GpuBuffer x = conv == convK ? mixer.kBuf() : mixer.vBuf();
        mixer.ops().add(x, start * stride, convTemp, 0, x, start * stride, len * channels);
    }

    /**
     * The steps that run after the norms: the relative-logit table of the step's tokens and, on a full-attention
     * layer, the log scaling of the normalized queries (vLLM {@code fused_qkvr_prep}).
     */
    void bindPostNorms() {
        queryScaleStep = mixer.step(AttnExtKernels.INKLING_QSCALE,
                AttnExtKernels.inklingScaleArgs(mixer.ubatch(), mixer.spec().nHead(), mixer.spec().headDimK(),
                        mixer.spec().qDim(), 0, mixer.spec().isSwaLayer() ? 0 : logFloor, logAlpha,
                        mixer.posBuffer(), mixer.qBuf()),
                AttnExtKernels.ARG_QSCALE_N_TOKENS, 1, (long) mixer.spec().nHead() * mixer.spec().headDimK(), 0, -1);
        relLogitsStep = mixer.step(AttnExtKernels.INKLING_REL,
                AttnExtKernels.inklingRelArgs(mixer.ubatch(), mixer.spec().nHead(), relDim, relExtent,
                        mixer.spec().isSwaLayer() ? 0 : logFloor, logAlpha, rBuf, relProjBuf, mixer.posBuffer(),
                        relTableBuf),
                AttnExtKernels.ARG_REL_N_TOKENS, 1, (long) mixer.spec().nHead() * relExtent, 0, -1);
        mixer.input(queryScaleStep, false);
        mixer.input(relLogitsStep, false);
    }

    /**
     * The {@code attn_rel @ x} product of the step's rows and its bias: the query-side relative output every
     * attention row reads the table step from.
     */
    void projectRel(int n, GpuBuffer x, int off) {
        mixer.ops().matmul(wr, x, off, rBuf, 0, n);
        mixer.addDeltas(rSite, n, x, off, (int) wr.cols(), rBuf, 0, mixer.spec().nHead() * relDim);
        if (rBiasAdd != null) {
            rBiasAdd.setBuffer(ARG_BIAS_DATA, rBuf);
            rBiasAdd.setInt(ARG_BIAS_SEQ, n);
            rBiasAdd.dispatch();
        }
    }

    /** The step's relative-logit table buffer, the INKLING position bias the attention kernels read. */
    GpuBuffer relTable() {
        return relTableBuf;
    }

    /** A scratch temporary of {@code floats}, which must start at element 0. */
    private static ScratchSlot inklingTemp(GpuScratch scratch, String name, long floats) {
        ScratchSlot slot = scratch.temp(name, floats);
        if (slot.offset() != 0) {
            throw new IllegalStateException("scratch slot " + slot.name() + " starts at element " + slot.offset());
        }
        return slot;
    }
}
