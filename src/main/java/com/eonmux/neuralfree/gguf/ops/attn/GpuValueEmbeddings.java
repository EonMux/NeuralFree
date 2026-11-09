package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * limite's value-side stages of {@link GpuAttentionMixer}: the value embeddings ({@code value_embd} rows added to V,
 * models/limite.cpp:87-89, 149-155) and the exclusive self attention (XSA, models/limite.cpp:177-191) that removes
 * each query head's projection on the token's own value. The lookup reads the step's token ids from the device when
 * the table is one buffer and from the step's host record otherwise; the gate product runs with the other
 * projections. Not thread-safe; driven by the owning mixer's step thread.
 */
final class GpuValueEmbeddings {

    private static final String T_VE = "attn.ve";
    private static final String T_VE_GATE = "attn.ve.gate";

    private final GpuAttentionMixer mixer;

    /** limite value-embedding table and its per-KV-head gate projection; null without value embeddings. */
    private GpuWeight valueEmbd;
    private GpuWeight veGateW;
    private ScratchSlot veValues;
    private ScratchSlot veGate;
    /** Token ids and host-id fallback of the value-embedding lookup; device ids serve a single-shard table. */
    private ScratchSlot stepTokenIds;
    private int[] hostStepIds;
    /** The per-head {@code attn_xsa_alpha} of the exclusive self attention; null without XSA. */
    private GpuBuffer xsaAlpha;

    /**
     * Binds the stage's tensors and steps: the {@code value_embd} table lookup and the per-KV-head
     * {@code attn_ve_gate} projection of the block input as input steps that run after the V projection, and the
     * per-head {@code attn_xsa_alpha} with the output step that removes each head's projection on the token's own
     * value.
     *
     * @throws IllegalArgumentException when a tensor has the wrong shape, or the token-id slot is not at offset 0
     */
    GpuValueEmbeddings(GpuAttentionMixer mixer) {
        this.mixer = mixer;
        if (mixer.spec().valueEmbeddings()) {
            bindValueEmbeddings(mixer.ctx().scratch());
        }
        if (mixer.spec().xsaEps() > 0f) {
            bindXsa();
        }
    }

    /**
     * limite value embeddings (models/limite.cpp:87-89, 149-155): the lookup of each token's {@code value_embd} row
     * and the per-KV-head {@code attn_ve_gate} projection of {@code in} are the inputs of the add step that runs
     * after the V projection.
     *
     * @throws IllegalArgumentException when the table or the gate has the wrong shape, or the token-id slot is not
     *                                  at offset 0
     */
    private void bindValueEmbeddings(GpuScratch scratch) {
        if (!mixer.proj().projectsKv()) {
            throw new IllegalArgumentException("layer " + mixer.layer().il() + ": value embeddings need the layer's"
                    + " own V projection");
        }
        valueEmbd = mixer.weights().get(TensorNames.weight(TensorNames.VALUE_EMBD));
        if (valueEmbd.cols() != mixer.spec().vDim()) {
            throw new IllegalArgumentException("layer " + mixer.layer().il() + ": " + valueEmbd.name()
                    + " holds rows of " + valueEmbd.cols() + " values, V is " + mixer.spec().vDim() + " wide");
        }
        int vDim = mixer.spec().vDim();
        veValues = scratch.temp(T_VE, (long) mixer.ubatch() * vDim);
        GpuBuffer gateBuf;
        if (mixer.spec().valueEmbeddingGate()) {
            veGateW = mixer.weights().layer(mixer.layer(), TensorNames.weight(TensorNames.ATTN_VE_GATE));
            int nEmbd = mixer.hparams().nEmbd();
            if (veGateW.cols() != nEmbd || veGateW.rows() != mixer.spec().nHeadKv()) {
                throw new IllegalArgumentException("layer " + mixer.layer().il() + ": "
                        + TensorNames.weight(TensorNames.ATTN_VE_GATE) + " must be [" + nEmbd + ", "
                        + mixer.spec().nHeadKv() + "], got [" + veGateW.cols() + ", " + veGateW.rows() + "]");
            }
            veGate = scratch.temp(T_VE_GATE, (long) mixer.ubatch() * mixer.spec().nHeadKv());
            gateBuf = veGate.buffer();
        } else {
            gateBuf = mixer.ops().placeholders().readOnlyFloats();
        }
        if (valueEmbd.isSingleShard()) {
            stepTokenIds = scratch.slot(ScratchSlots.TOKEN_IDS);
            if (stepTokenIds.offset() != 0) {
                throw new IllegalStateException("the token-id slot must start at element 0 of its buffer for the "
                        + "value-embedding lookup in " + valueEmbd.name() + ", got offset " + stepTokenIds.offset());
            }
        } else {
            hostStepIds = new int[mixer.ubatch()];
        }
        mixer.input(TokenStep.action(this::lookupValueEmbeddings), true);
        Object[] args = AttnExtKernels.valueEmbeddingArgs(mixer.ubatch(), vDim, mixer.spec().nHeadKv(),
                mixer.spec().headDimV(), mixer.spec().valueEmbeddingGate(), veValues.buffer(), gateBuf,
                mixer.vBuf());
        mixer.input(mixer.step(AttnExtKernels.VALUE_EMBEDDINGS, args, AttnExtKernels.ARG_VE_N_TOKENS, 1, vDim, 0, -1),
                true);
    }

    /** Loads the step's {@code value_embd} rows into the add step's values by the step's token ids. */
    private void lookupValueEmbeddings(int n) {
        if (stepTokenIds != null) {
            mixer.ops().embed(valueEmbd, stepTokenIds.buffer(), n, veValues.buffer(), veValues.offset());
        } else {
            mixer.ctx().scratch().readStepTokens(hostStepIds, 0, n);
            mixer.ops().embed(valueEmbd, hostStepIds, n, veValues.buffer(), veValues.offset());
        }
    }

    /** The {@code attn_ve_gate @ in} product of the value-embedding gate; a no-op without the gate. */
    void projectGate(int n, ScratchSlot in) {
        if (veGateW != null) {
            mixer.ops().matmul(veGateW, in.buffer(), in.offset(), veGate.buffer(), veGate.offset(), n);
        }
    }

    /** Whether the layer carries value embeddings (the spec's {@code valueEmbeddings} flag). */
    boolean hasValues() {
        return valueEmbd != null;
    }

    /**
     * limite exclusive self attention (models/limite.cpp:177-191): the per-head {@code attn_xsa_alpha} and the
     * output step that removes each head's projection on the token's own value, before the output norm.
     *
     * @throws IllegalArgumentException when the alpha tensor does not hold one value per query head
     */
    private void bindXsa() {
        if (!mixer.proj().projectsKv()) {
            throw new IllegalArgumentException("layer " + mixer.layer().il() + ": the exclusive self attention needs"
                    + " the layer's own V projection");
        }
        xsaAlpha = mixer.vector(TensorNames.weight(TensorNames.ATTN_XSA_ALPHA), mixer.spec().nHead());
        Object[] args = AttnExtKernels.xsaArgs(mixer.ubatch(), mixer.spec().nHeadKv(), mixer.spec().gqaRatio(),
                mixer.spec().headDimV(), mixer.spec().vDim(), mixer.spec().xsaEps(), xsaAlpha, mixer.vBuf(),
                mixer.attnBuf());
        mixer.addOutputStep(mixer.step(AttnExtKernels.XSA, args, AttnExtKernels.ARG_XSA_N_TOKENS, 1,
                mixer.spec().nHeadKv(), 0, -1));
    }

    /** Whether the layer runs exclusive self attention (the spec sets {@code xsaEps}). */
    boolean hasXsa() {
        return xsaAlpha != null;
    }
}
