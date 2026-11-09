package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * The Q/K/V projections of {@link GpuAttentionMixer}: the projection weights and their LoRA sites, the fused
 * {@code attn_qkv} product split into the {@code Q}, {@code K} and {@code V} slots with {@code LLM_Copy2D_F32}, the
 * latent-Q path (Step3 text) and its RMS post-norm, the interleaved output gate split off Q, MoVA's routed value
 * experts, and the biases and KQV clamp. {@link #project} runs the whole pass for a step, {@link #projectKv} the
 * cache-write half. Not thread-safe; driven by the owning mixer's bind and step thread.
 */
final class GpuQkvProjections implements AutoCloseable {

    private static final String T_QKV = "attn.qkv";
    private static final String T_QLAT = "attn.qlat";
    private static final String T_QGATE = "attn.qgate";
    private static final String T_GATE = "attn.gate";

    private final GpuAttentionMixer mixer;

    private GpuWeight wq;
    private GpuWeight wqUp;
    private GpuWeight wk;
    private GpuWeight wv;
    private GpuWeight wqkv;
    private GpuWeight wgate;

    private GpuProjectionDeltas.Site qSite;
    private GpuProjectionDeltas.Site kSite;
    private GpuProjectionDeltas.Site vSite;
    private GpuProjectionDeltas.Site qkvSite;
    private GpuProjectionDeltas.Site gateSite;
    /** MoVA's routed value experts; null without them. */
    private GpuValueExperts valueExperts;

    private GpuBuffer qProjBuf;
    private GpuBuffer qLatentBuf;
    /** RMS post-norm of the shared query projection (Step3 text), null for a direct Q projection. */
    private TokenStep qLatentNorm;
    private GpuBuffer qkvTemp;
    private GpuBuffer gateTemp;

    GpuQkvProjections(GpuAttentionMixer mixer) {
        this.mixer = mixer;
    }

    @Override
    public void close() {
        if (valueExperts != null) {
            valueExperts.close();
            valueExperts = null;
        }
    }

    /** The shared query projection weight the Q/K/V adapters chain from. */
    GpuWeight wq() {
        return wq;
    }

    /** The buffer the Q product lands in: the {@code Q} slot, or the interleaved-gate or latent temporary. */
    GpuBuffer qProjBuf() {
        return qProjBuf;
    }

    /** The gate rows: split off Q when interleaved, projected from the input otherwise; null without a gate. */
    GpuBuffer gateTemp() {
        return gateTemp;
    }

    /**
     * Binds the projection weights, their LoRA sites and the step steps of the whole pass.
     *
     * @throws IllegalArgumentException when a projection tensor has the wrong shape
     */
    void bind(GpuScratch scratch) {
        int qw = mixer.proj().qProjWidth();
        qProjBuf = mixer.proj().interleavedGate()
                ? scratch.temp(T_QGATE, (long) mixer.ubatch() * qw).buffer() : mixer.qBuf();
        if (mixer.proj().qLatentDim() > 0) {
            qLatentBuf = scratch.temp(T_QLAT, (long) mixer.ubatch() * qw).buffer();
            qProjBuf = qLatentBuf;
        }
        if (mixer.proj().source() == QkvProjection.Source.FUSED) {
            bindFused(scratch, qw);
        } else {
            bindSeparate(qw);
        }
        bindBiases(qw);
        bindClamp(qw);
        bindGate(scratch);
    }

    /** The fused {@code attn_qkv} product: one matmul whose parts the copy steps place into Q, K and V. */
    private void bindFused(GpuScratch scratch, int qw) {
        wqkv = mixer.weights().layer(mixer.layer(), mixer.proj().qkvWeight());
        QkvProjection.checkShape(wqkv.name(), wqkv.cols(), wqkv.rows(), wqkv.cols(), mixer.proj().fusedWidth());
        qkvSite = mixer.serveDeltas(mixer.proj().qkvWeight(), wqkv);
        qSite = mixer.serveDeltas(TensorNames.weight(TensorNames.ATTN_Q), wqkv);
        kSite = mixer.serveDeltas(TensorNames.weight(TensorNames.ATTN_K), wqkv);
        vSite = mixer.serveDeltas(TensorNames.weight(TensorNames.ATTN_V), wqkv);
        int fw = mixer.proj().fusedWidth();
        qkvTemp = scratch.temp(T_QKV, (long) mixer.ubatch() * fw).buffer();
        if (mixer.proj().qkvBias() != null) {
            mixer.input(mixer.biasStep(mixer.vector(mixer.proj().qkvBias(), fw), fw, qkvTemp), true);
        }
        mixer.input(mixer.copyStep(qkvTemp, mixer.proj().qOffset(), fw, qProjBuf, 0, qw, 1, qw), false);
        if (mixer.proj().projectsKv()) {
            mixer.input(mixer.copyStep(qkvTemp, mixer.proj().kOffset(), fw, mixer.kBuf(), 0, mixer.spec().kDim(), 1,
                    mixer.spec().kDim()), true);
            mixer.input(mixer.copyStep(qkvTemp, mixer.proj().vOffset(), fw, mixer.vBuf(), 0, mixer.spec().vDim(), 1,
                    mixer.spec().vDim()), true);
        }
    }

    /** The separate {@code attn_q}, {@code attn_k} and {@code attn_v} projections and their LoRA sites. */
    private void bindSeparate(int qw) {
        wq = mixer.weights().layer(mixer.layer(), mixer.proj().qWeight());
        QkvProjection.checkShape(wq.name(), wq.cols(), wq.rows(), wq.cols(), qw);
        qSite = mixer.serveDeltas(mixer.proj().qWeight(), wq);
        if (mixer.proj().qLatentDim() > 0) {
            bindLatentQ();
        }
        if (mixer.proj().projectsKv()) {
            bindKv();
        }
    }

    /** The latent-Q path (Step3 text): the shared low-rank projection and its RMS post-norm. */
    private void bindLatentQ() {
        wqUp = mixer.weights().layer(mixer.layer(), mixer.proj().qUpWeight());
        QkvProjection.checkShape(wqUp.name(), wqUp.cols(), wqUp.rows(), mixer.proj().qLatentDim(),
                mixer.spec().qDim());
        GpuBuffer qn = mixer.vector(mixer.proj().qNormWeight(), mixer.proj().qLatentDim());
        qLatentNorm = mixer.normStep(NormKind.RMS, mixer.rmsEps(), qLatentBuf, 1, mixer.proj().qLatentDim(), qn, null,
                1);
    }

    /** The K and V projections: MoVA's value experts, V from the raw K projection, or the layer's own V. */
    private void bindKv() {
        wk = mixer.weights().layer(mixer.layer(), mixer.proj().kWeight());
        QkvProjection.checkShape(wk.name(), wk.cols(), wk.rows(), wq.cols(), mixer.spec().kDim());
        kSite = mixer.serveDeltas(mixer.proj().kWeight(), wk);
        if (mixer.proj().valueExperts()) {
            bindValueExperts();
        } else if (mixer.proj().vFromK()) {
            mixer.input(mixer.copyStep(mixer.kBuf(), 0, mixer.spec().kDim(), mixer.vBuf(), 0, mixer.spec().vDim(), 1,
                    mixer.spec().kDim()), true);
        } else {
            wv = mixer.weights().layer(mixer.layer(), mixer.proj().vWeight());
            QkvProjection.checkShape(wv.name(), wv.cols(), wv.rows(), wq.cols(), mixer.spec().vDim());
            vSite = mixer.serveDeltas(mixer.proj().vWeight(), wv);
        }
    }

    /** The projection biases: the fused product's bias before the split, the separate biases on their rows. */
    private void bindBiases(int qw) {
        if (mixer.proj().qBias() != null) {
            mixer.input(mixer.biasStep(mixer.vector(mixer.proj().qBias(), qw), qw, qProjBuf), false);
        }
        if (mixer.proj().projectsKv()) {
            if (mixer.proj().kBias() != null) {
                mixer.input(mixer.biasStep(mixer.vector(mixer.proj().kBias(), mixer.spec().kDim()),
                        mixer.spec().kDim(), mixer.kBuf()), true);
            }
            if (mixer.proj().vBias() != null) {
                mixer.input(mixer.biasStep(mixer.vector(mixer.proj().vBias(), mixer.spec().vDim()),
                        mixer.spec().vDim(), mixer.vBuf()), true);
            }
        }
    }

    /** The KQV clamp ({@code LLM_Unary CLAMP}) when the layer sets one. */
    private void bindClamp(int qw) {
        if (mixer.spec().clampKqv() <= 0f) {
            return;
        }
        ActParams clamp = ActParams.builder().clamp(-mixer.spec().clampKqv(), mixer.spec().clampKqv()).build();
        mixer.input(mixer.clampStep(clamp, qProjBuf, qw), false);
        if (mixer.proj().projectsKv()) {
            mixer.input(mixer.clampStep(clamp, mixer.kBuf(), mixer.spec().kDim()), true);
            mixer.input(mixer.clampStep(clamp, mixer.vBuf(), mixer.spec().vDim()), true);
        }
    }

    /** The output gate: split off the interleaved Q rows, or projected from the input by {@code attn_gate}. */
    private void bindGate(GpuScratch scratch) {
        int hd = mixer.spec().headDimK();
        if (mixer.proj().interleavedGate()) {
            gateTemp = scratch.temp(T_GATE, (long) mixer.ubatch() * mixer.spec().qDim()).buffer();
            mixer.input(mixer.copyStep(qProjBuf, 0, 2 * hd, mixer.qBuf(), 0, hd, mixer.spec().nHead(), hd), false);
            mixer.input(mixer.copyStep(qProjBuf, hd, 2 * hd, gateTemp, 0, hd, mixer.spec().nHead(), hd), false);
        } else if (mixer.proj().gateWeight() != null) {
            wgate = mixer.weights().layer(mixer.layer(), mixer.proj().gateWeight());
            int gw = AttnExtMath.gateWidth(mixer.spec().gate(), mixer.spec().nHead(), mixer.spec().headDimV());
            QkvProjection.checkShape(wgate.name(), wgate.cols(), wgate.rows(), wgate.cols(), gw);
            gateSite = mixer.serveDeltas(mixer.proj().gateWeight(), wgate);
            gateTemp = scratch.temp(T_GATE, (long) mixer.ubatch() * gw).buffer();
        }
    }

    /** MoVA: V from the routed value experts on the layer input. */
    private void bindValueExperts() {
        GpuBuffer gateBias = mixer.proj().vGateBias() != null
                ? mixer.vector(mixer.proj().vGateBias(), mixer.spec().valueExperts().nExpert()) : null;
        valueExperts = new GpuValueExperts(mixer.ctx(), mixer.layer(), mixer.proj(), mixer.spec().valueExperts(),
                gateBias, mixer.vBuf());
        if (valueExperts.inputWidth() != wq.cols()) {
            throw new IllegalArgumentException("layer " + mixer.layer().il() + ": value router "
                    + mixer.proj().vGateWeight() + " reads " + valueExperts.inputWidth() + " values, the Q projection "
                    + wq.cols());
        }
    }

    /**
     * The base products and their deltas. The deltas of a fused product's parts land in the fused rows, before the
     * split and the biases, matching graniteswitch's {@code Q/K/V += delta} on the sliced views
     * (models/granite-switch.cpp:350-359).
     */
    void project(int n, ScratchSlot in) {
        GpuBuffer x = in.buffer();
        int off = in.offset();
        // The attention projections of a single token all read this row, so its Q8_1 blocks are quantized once.
        int width = (int) (wqkv != null ? wqkv.cols() : wq.cols());
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(off, width)
                ? mixer.ops().shareQ8Activations(x, off, width) : null;
        try {
            if (wqkv != null) {
                projectFused(n, x, off);
            } else {
                projectSeparate(n, x, off);
            }
            if (wqUp != null) {
                qLatentNorm.run(n);
                mixer.ops().matmul(wqUp, qProjBuf, 0, mixer.qBuf(), 0, n);
            }
            if (wgate != null) {
                mixer.ops().matmul(wgate, x, off, gateTemp, 0, n);
                mixer.addDeltas(gateSite, n, x, off, (int) wgate.cols(), gateTemp, 0, (int) wgate.rows());
            }
            if (mixer.inkling() != null) {
                mixer.inkling().projectRel(n, x, off);
            }
            mixer.valueEmbeddings().projectGate(n, in);
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    private void projectFused(int n, GpuBuffer x, int off) {
        int cols = (int) wqkv.cols();
        int fw = mixer.proj().fusedWidth();
        mixer.ops().matmul(wqkv, x, off, qkvTemp, 0, n);
        mixer.addDeltas(qkvSite, n, x, off, cols, qkvTemp, 0, fw);
        mixer.addDeltas(qSite, n, x, off, cols, qkvTemp, mixer.proj().qOffset(), fw);
        mixer.addDeltas(kSite, n, x, off, cols, qkvTemp, mixer.proj().kOffset(), fw);
        mixer.addDeltas(vSite, n, x, off, cols, qkvTemp, mixer.proj().vOffset(), fw);
    }

    private void projectSeparate(int n, GpuBuffer x, int off) {
        int cols = (int) wq.cols();
        mixer.ops().matmul(wq, x, off, qProjBuf, 0, n);
        mixer.addDeltas(qSite, n, x, off, cols, qProjBuf, 0, mixer.proj().qProjWidth());
        if (wk != null) {
            mixer.ops().matmul(wk, x, off, mixer.kBuf(), 0, n);
            mixer.addDeltas(kSite, n, x, off, cols, mixer.kBuf(), 0, mixer.spec().kDim());
        }
        if (wv != null) {
            mixer.ops().matmul(wv, x, off, mixer.vBuf(), 0, n);
            mixer.addDeltas(vSite, n, x, off, cols, mixer.vBuf(), 0, mixer.spec().vDim());
        }
        if (valueExperts != null) {
            valueExperts.run(n, x, off);
        }
    }

    /**
     * The base products the cache writes need: every part of a fused projection (its K and V rows are read from the
     * fused rows, so the product runs whole), otherwise K and V alone. The Q product, the gate and their deltas do
     * not run.
     */
    void projectKv(int n, ScratchSlot in) {
        GpuBuffer x = in.buffer();
        int off = in.offset();
        mixer.valueEmbeddings().projectGate(n, in);
        if (wqkv != null) {
            projectFused(n, x, off);
            return;
        }
        int cols = (int) wq.cols();
        if (wk != null) {
            mixer.ops().matmul(wk, x, off, mixer.kBuf(), 0, n);
            mixer.addDeltas(kSite, n, x, off, cols, mixer.kBuf(), 0, mixer.spec().kDim());
        }
        if (wv != null) {
            mixer.ops().matmul(wv, x, off, mixer.vBuf(), 0, n);
            mixer.addDeltas(vSite, n, x, off, cols, mixer.vBuf(), 0, mixer.spec().vDim());
        }
        if (valueExperts != null) {
            valueExperts.run(n, x, off);
        }
    }
}
