package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;

/**
 * The output stages of {@link GpuAttentionMixer}, between the attention rows and the {@code wo} projection: the
 * muse-spark weightless per-head RMS norm, the output gate ({@code LLM_AttnOutGate}), bitnet's sub-norm, the output
 * bias ({@code LLM_Batch_BiasAdd}) and the value scale ({@code f_attn_value_scale}, models/mimo2.cpp:174-176,
 * models/dflash.cpp:992-993). Not thread-safe; driven by the owning mixer's step thread.
 */
final class GpuOutputStages {

    private static final String T_OUT = "attn.out";
    private static final int ARG_BIAS_SEQ = 1;
    private static final int ARG_BIAS_DATA = 3;
    private static final int ARG_AXPBY_N = 0;
    private static final int ARG_AXPBY_B_OFF = 4;
    private static final int ARG_AXPBY_OUT_OFF = 5;
    private static final int ARG_AXPBY_B = 7;
    private static final int ARG_AXPBY_OUT = 8;

    private final GpuAttentionMixer mixer;

    private PreparedDispatch outBias;
    private PreparedDispatch valueScale;
    private long valueScaleCount = -1;
    private GpuBuffer outTemp;

    GpuOutputStages(GpuAttentionMixer mixer) {
        this.mixer = mixer;
    }

    /**
     * Binds the output steps over the attention rows: the per-head output norm when the projection asks for it, the
     * output gate when the layer has one, the stack's output norm and bitnet's sub-norm.
     */
    void bindSteps() {
        if (mixer.proj().outHeadNorm()) {
            mixer.addOutputStep(mixer.normStep(NormKind.RMS, mixer.rmsEps(), mixer.attnBuf(), mixer.spec().nHead(),
                    mixer.spec().headDimV(), null, null, 1));
        }
        AttnGateKind gate = mixer.spec().gate();
        if (gate != AttnGateKind.NONE) {
            int gw = AttnExtMath.gateWidth(gate, mixer.spec().nHead(), mixer.spec().headDimV());
            GpuBuffer bias = null;
            if (mixer.proj().gateBias() != null) {
                bias = mixer.vector(mixer.proj().gateBias(), mixer.spec().nHead());
            }
            boolean hasBias = bias != null;
            Object[] args = AttnExtKernels.outGateArgs(gate, mixer.ubatch(), mixer.spec().nHead(),
                    mixer.spec().headDimV(), mixer.attnBuf(), 0, mixer.spec().outDim(), mixer.gateTemp(), 0, gw,
                    hasBias ? bias : mixer.ops().placeholders().readOnlyFloats(), hasBias);
            mixer.addOutputStep(mixer.step(AttnExtKernels.OUT_GATE, args, AttnExtKernels.ARG_GATE_N_TOKENS, 1,
                    mixer.spec().outDim(), 0, -1));
        }
        if (mixer.spec().outNormKind() != null) {
            mixer.addOutputStep(mixer.normStep(mixer.spec().outNormKind(), mixer.spec().outNormEps(),
                    mixer.attnBuf(), 1, mixer.spec().outDim(), null, null, 1));
        }
        if (mixer.proj().subNormWeight() != null) {
            NormKind kind = mixer.layer().norm().kind();
            float eps = kind == NormKind.LAYER ? mixer.hparams().normEps() : mixer.rmsEps();
            GpuBuffer w = mixer.vector(mixer.proj().subNormWeight(), mixer.spec().outDim());
            GpuBuffer b = mixer.proj().subNormBias() != null
                    ? mixer.vector(mixer.proj().subNormBias(), mixer.spec().outDim()) : null;
            mixer.addOutputStep(mixer.normStep(kind, eps, mixer.attnBuf(), 1, mixer.spec().outDim(), w, b, 1));
        }
    }

    /** Binds the {@code wo} bias add and the value scale over the projection's {@code outRows} per token. */
    void bindProjection(int outRows) {
        if (mixer.proj().outBias() != null) {
            GpuBuffer bias = mixer.vector(mixer.proj().outBias(), outRows);
            outTemp = mixer.ctx().scratch().temp(T_OUT, (long) mixer.ubatch() * outRows).buffer();
            outBias = mixer.prepare(GpuOps.BATCH_BIAS_ADD, mixer.roundUp(outRows),
                    mixer.bindArgs(outRows, mixer.ubatch(), bias, outTemp));
        }
        if (mixer.spec().valueScale() != 0f) {
            long n = (long) mixer.ubatch() * outRows;
            valueScale = mixer.prepare(GpuOps.AXPBY, mixer.roundUp(n), mixer.bindArgs((int) n, 0.0f,
                    mixer.spec().valueScale(), 0, 0, 0, mixer.ops().placeholders().readOnlyFloats(),
                    mixer.attnBuf(), mixer.attnBuf()));
        }
    }

    /** The {@code wo} projection of the attention rows into {@code out}, with its bias and value scale. */
    void projectOutput(int n, ScratchSlot out) {
        int rows = (int) mixer.wo().rows();
        long count = (long) n * rows;
        out.require(count);
        GpuBuffer dst = out.buffer();
        int dstOff = out.offset();
        int cols = mixer.spec().outDim();
        if (outBias != null) {
            GpuBuffer biased = dstOff == 0 ? dst : outTemp;
            mixer.ops().matmul(mixer.wo(), mixer.attnBuf(), 0, biased, 0, n);
            mixer.addDeltas(mixer.outSite(), n, mixer.attnBuf(), 0, cols, biased, 0, rows);
            outBias.setBuffer(ARG_BIAS_DATA, biased);
            outBias.setInt(ARG_BIAS_SEQ, n);
            outBias.dispatch();
            if (biased != dst) {
                mixer.ops().copy(outTemp, 0, dst, dstOff, (int) count);
            }
        } else {
            mixer.ops().matmul(mixer.wo(), mixer.attnBuf(), 0, dst, dstOff, n);
            mixer.addDeltas(mixer.outSite(), n, mixer.attnBuf(), 0, cols, dst, dstOff, rows);
        }
        if (valueScale != null) {
            valueScale.setInt(ARG_AXPBY_B_OFF, dstOff);
            valueScale.setInt(ARG_AXPBY_OUT_OFF, dstOff);
            valueScale.setBuffer(ARG_AXPBY_B, dst);
            valueScale.setBuffer(ARG_AXPBY_OUT, dst);
            if (count != valueScaleCount) {
                valueScale.setInt(ARG_AXPBY_N, (int) count);
                valueScale.setGlobal(mixer.roundUp(count));
                valueScaleCount = count;
            }
            valueScale.dispatch();
        }
    }
}
