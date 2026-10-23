package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * Immutable description of a dflash draft model: its dimensions, which of the llama.cpp variants it is and the
 * drafting protocol constants of common/speculative.cpp. Variants are detected as llama.cpp detects them:
 * <ul>
 *   <li>DSpark: a {@code markov_w1} tensor (dflash.cpp:138, speculative.cpp:2425-2428);</li>
 *   <li>DFlash2: a {@code selector_hidden} tensor (dflash.cpp:152-177), drafting by lattice walk;</li>
 *   <li>LiLiCorr: {@code lilicorr.num_layers > 0} (dflash.cpp:179-237), drafting by lattice walk;</li>
 *   <li>DFlash2 convolutions: {@code conv_kernel_size > 0 && conv_group_size > 0} (dflash.cpp:153, 324-332);</li>
 *   <li>DeepSeek-V4 DSpark backbone: {@code hyper_connection.count > 0} (dflash.cpp:66-107, 248-296).</li>
 * </ul>
 * Protocol constants (speculative.cpp:1059-1110): {@link #blockSize()} defaults to 16 without
 * {@code dflash.block_size}; {@code dflash.sample_from_anchor} defaults to true; {@code dflash.attention.causal}
 * defaults to false; {@code dflash.has_confidence_head} defaults to true; the mask token is
 * {@code tokenizer.ggml.mask_token_id}.
 */
public final class DflashSpec {

    /** {@code block_size} when the draft has no {@code dflash.block_size} key (speculative.cpp:1059). */
    public static final int DEFAULT_BLOCK_SIZE = 16;
    /** Candidates of the drafter's {@code top_k} sampler (speculative.cpp:1114-1119). */
    public static final int SAMPLER_TOP_K = 10;

    static final String KEY_SAMPLE_FROM_ANCHOR = "dflash.sample_from_anchor";
    static final String KEY_ATTENTION_CAUSAL = "dflash.attention.causal";
    static final String KEY_HAS_CONFIDENCE_HEAD = "dflash.has_confidence_head";

    private final int nEmbd;
    private final int nLayer;
    private final int nLayerAll;
    private final int[] nHead;
    private final int[] nHeadKv;
    private final int headDim;
    private final int nVocab;
    private final int nVocabDraft;
    private final int[] targetLayers;
    private final float rmsEps;
    private final float kqScale;
    private final float embeddingScale;
    private final float attnValueScale;
    private final float logitScale;
    private final float finalSoftcap;
    private final boolean gelu;
    private final int nSwa;
    private final boolean[] swa;

    private final int blockSize;
    private final int graphBlockSize;
    private final boolean sampleFromAnchor;
    private final boolean causal;
    private final boolean confidenceHeadMeta;
    private final int maskTokenId;

    private final boolean dspark;
    private final boolean dflash2;
    private final boolean lilicorr;
    private final boolean conv;
    private final boolean dsv4;
    private final boolean d2t;

    private final int convKernel;
    private final int convGroup;
    private final int selectorRank;
    private final int selectorTopK;

    private final int lilicorrHidden;
    private final int lilicorrTopK;
    private final int lilicorrHeads;
    private final int lilicorrLayers;
    private final float lilicorrVectorEps;
    private final float lilicorrLogitScale;
    private final int lilicorrFactorDim;
    private final int lilicorrMlpHidden;

    private final int hc;
    private final float hcEps;
    private final int hcSinkhornIters;
    private final int nLoraQ;
    private final int oGroups;
    private final int oLoraRank;
    private final int nRot;
    private final float ropeFreqBase;
    private final int nExpert;
    private final int[] nExpertUsed;
    private final int nExpertShared;
    private final boolean expertWeightsNorm;
    private final float expertWeightsScale;
    private final float[] clampExp;
    private final float[] clampShexp;

    private DflashSpec(ModelHparams hp, GgufKv kv, CpuWeights draft) {
        if (hp.arch() != Arch.DFLASH) {
            throw new IllegalArgumentException("dflash drafter needs a dflash model, got " + hp.arch().ggufName());
        }
        nEmbd = hp.nEmbd();
        nLayer = hp.nLayer();
        nLayerAll = hp.nLayerAll();
        nHead = new int[nLayer];
        nHeadKv = new int[nLayer];
        swa = new boolean[nLayer];
        for (int il = 0; il < nLayer; il++) {
            nHead[il] = hp.nHead(il);
            nHeadKv[il] = hp.nHeadKv(il);
            swa[il] = hp.isSwa(il);
            if (hp.headK(il) != hp.headK(0) || hp.headV(il) != hp.headK(0)) {
                throw new IllegalArgumentException("dflash drafter needs one head size for K and V on every layer"
                        + " (dflash.cpp:811), layer " + il + " has " + hp.headK(il) + "/" + hp.headV(il));
            }
        }
        headDim = hp.headK(0);
        nVocab = hp.nVocab();
        targetLayers = hp.targetLayers();
        if (targetLayers.length == 0) {
            throw new IllegalArgumentException("DFlash model requires 'target_layers' in GGUF metadata");
        }
        rmsEps = hp.normRmsEps();
        kqScale = hp.attentionScale() != 0.0f ? hp.attentionScale() : 1.0f / (float) Math.sqrt(headDim);
        embeddingScale = hp.embeddingScale();
        attnValueScale = hp.attnValueScale();
        logitScale = hp.logitScale();
        finalSoftcap = hp.finalSoftcap();
        String act = hp.hiddenActivation();
        gelu = "gelu".equals(act) || "gelu_pytorch_tanh".equals(act);
        nSwa = hp.nSwa();

        graphBlockSize = hp.dflashBlockSize();
        blockSize = kv.has(GgufKeys.DFLASH_BLOCK_SIZE) ? graphBlockSize : DEFAULT_BLOCK_SIZE;
        if (blockSize < 1) {
            throw new IllegalArgumentException("dflash.block_size must be positive, got " + blockSize);
        }
        sampleFromAnchor = metaFlag(kv, KEY_SAMPLE_FROM_ANCHOR, true);
        causal = metaFlag(kv, KEY_ATTENTION_CAUSAL, false);
        confidenceHeadMeta = metaFlag(kv, KEY_HAS_CONFIDENCE_HEAD, true);
        maskTokenId = kv.i32(GgufKeys.TOKENIZER_MASK_ID, -1);
        if (maskTokenId < 0 || maskTokenId >= nVocab) {
            throw new IllegalArgumentException("dflash draft needs '" + GgufKeys.TOKENIZER_MASK_ID
                    + "' inside its vocabulary of " + nVocab + ", got " + maskTokenId);
        }

        dsv4 = hp.dsv4HcMult() > 0;
        dspark = draft.has(TensorNames.weight(TensorNames.DSPARK_MARKOV_W1));
        dflash2 = draft.has(TensorNames.weight(TensorNames.DFLASH_SELECTOR_HIDDEN));
        lilicorr = hp.lilicorrLayers() > 0;
        conv = !dsv4 && hp.dflashConvKernelSize() > 0 && hp.dflashConvGroupSize() > 0;
        d2t = draft.has(TensorNames.D2T);
        nVocabDraft = d2t ? Math.toIntExact(draft.get(TensorNames.D2T).cols() * draft.get(TensorNames.D2T).rows())
                : nVocab;

        convKernel = hp.dflashConvKernelSize();
        convGroup = hp.dflashConvGroupSize();
        selectorRank = hp.dflashSelectorRank();
        selectorTopK = hp.dflashSelectorTopK();

        lilicorrHidden = hp.lilicorrHiddenSize();
        lilicorrTopK = hp.lilicorrCandidateTopK();
        lilicorrHeads = hp.lilicorrNumHeads();
        lilicorrLayers = hp.lilicorrLayers();
        lilicorrVectorEps = hp.lilicorrVectorEps();
        lilicorrLogitScale = hp.lilicorrLogitScale();
        lilicorrFactorDim = hp.lilicorrFactorDim();
        lilicorrMlpHidden = hp.lilicorrMlpHiddenSize();

        hc = hp.dsv4HcMult();
        hcEps = hp.dsv4HcEps();
        hcSinkhornIters = hp.dsv4HcSinkhornIters();
        nLoraQ = hp.nLoraQ();
        oGroups = hp.dsv4OGroupCount();
        oLoraRank = hp.dsv4OLoraRank();
        nRot = dsv4 ? hp.nRot(0) : 0;
        ropeFreqBase = hp.ropeFreqBase();
        nExpert = hp.nExpert();
        nExpertUsed = new int[nLayer];
        clampExp = new float[nLayer];
        clampShexp = new float[nLayer];
        for (int il = 0; il < nLayer; il++) {
            nExpertUsed[il] = hp.nExpertUsed(il);
            clampExp[il] = hp.swigluClampExp(il);
            clampShexp[il] = hp.swigluClampShexp(il);
        }
        nExpertShared = hp.nExpertShared();
        expertWeightsNorm = hp.expertWeightsNorm();
        expertWeightsScale = hp.expertWeightsScale();
        validate(hp);
    }

    /**
     * Spec of a dflash draft model from its hyperparameters, metadata and tensors.
     *
     * @throws IllegalArgumentException when the model is not a supported dflash draft, with the reason
     */
    public static DflashSpec of(ModelHparams hp, GgufKv kv, CpuWeights draft) {
        return new DflashSpec(Objects.requireNonNull(hp, "hp"), Objects.requireNonNull(kv, "kv"),
                Objects.requireNonNull(draft, "draft"));
    }

    private void validate(ModelHparams hp) {
        if (dflash2 && (selectorRank <= 0 || graphBlockSize <= 0 || selectorTopK <= 0 || convKernel <= 0
                || convGroup <= 0)) {
            throw new IllegalArgumentException("DFlash2 model is missing conv/selector metadata");
        }
        if (dflash2 && (long) nEmbd < (long) selectorTopK * (selectorTopK + 1)) {
            throw new IllegalArgumentException("DFlash2 hidden size is too small for the selector lattice");
        }
        if (conv && nEmbd % convGroup != 0) {
            throw new IllegalArgumentException("DFlash hidden size must be divisible by conv_group_size");
        }
        if (dspark && graphBlockSize <= 0) {
            throw new IllegalArgumentException("DSpark draft requires 'dflash.block_size' in GGUF metadata");
        }
        if (lilicorr && graphBlockSize != blockSize) {
            throw new IllegalArgumentException("LiLiCorr draft requires 'dflash.block_size' in GGUF metadata");
        }
        if (dsv4) {
            if (d2t || dflash2 || lilicorr) {
                throw new IllegalArgumentException("DSpark DeepSeek-V4 drafts have no reduced vocabulary, selector or"
                        + " LiLiCorr head in llama.cpp (dflash.cpp:1108-1280)");
            }
            if (hp.gating() != GatingFunc.SQRT_SOFTPLUS) {
                throw new IllegalArgumentException("DSpark DSV4 draft expects sqrtsoftplus MoE scoring");
            }
            if (oGroups <= 0 || nHead[0] % oGroups != 0 || nRot <= 0 || nRot > headDim || nSwa <= 0
                    || nLoraQ <= 0 || oLoraRank <= 0) {
                throw new IllegalArgumentException("DSpark DSV4 draft has inconsistent attention metadata: groups "
                        + oGroups + ", heads " + nHead[0] + ", n_rot " + nRot + ", head " + headDim + ", n_swa "
                        + nSwa + ", q_lora " + nLoraQ + ", o_lora " + oLoraRank);
            }
            for (int il = 0; il < nLayer; il++) {
                if (nHeadKv[il] != 1 || nHead[il] != nHead[0]) {
                    throw new IllegalArgumentException("DSpark DSV4 draft layer " + il + " needs one shared KV head"
                            + " and " + nHead[0] + " query heads");
                }
            }
        }
    }

    /**
     * {@code llama_model_meta_val_str(key) == "true"} (common/speculative.cpp:1053-1078), which formats the value by
     * its stored type ({@code gguf_kv_to_str}, src/llama-impl.cpp:168): true for a bool {@code true} or the string
     * {@code "true"}, false for any other present value (numbers print as digits, arrays as {@code [...]}),
     * {@code dflt} when absent.
     */
    static boolean metaFlag(GgufKv kv, String key, boolean dflt) {
        GGUFValueType type = kv.typeOf(key);
        if (type == null) {
            return dflt;
        }
        switch (type) {
            case STRING:
                return "true".equals(kv.str(key, null));
            case BOOL:
                return kv.bool(key, false);
            default:
                return false;
        }
    }

    public int nEmbd() { return nEmbd; }
    public int nLayer() { return nLayer; }
    /** {@code n_layer_all}; the LiLiCorr blocks follow it (dflash.cpp:187, 212). */
    public int nLayerAll() { return nLayerAll; }
    public int nHead(int il) { return nHead[il]; }
    public int nHeadKv(int il) { return nHeadKv[il]; }
    /** Head size of K, V and Q ({@code n_embd_head_k == n_embd_head_v}). */
    public int headDim() { return headDim; }
    /** Vocabulary of the draft's tokenizer, the target vocabulary of its logits. */
    public int nVocab() { return nVocab; }
    /** Rows of the draft's output head: the {@code d2t} length, or {@link #nVocab()} without one. */
    public int nVocabDraft() { return nVocabDraft; }
    /** Target layers whose inputs feed the drafter ({@code target_layers}). */
    public int[] targetLayers() { return targetLayers.clone(); }
    public int nTargets() { return targetLayers.length; }
    public float rmsEps() { return rmsEps; }
    /** {@code f_attention_scale} when set, else {@code 1/sqrt(head)} (dflash.cpp:826). */
    public float kqScale() { return kqScale; }
    public float embeddingScale() { return embeddingScale; }
    public float attnValueScale() { return attnValueScale; }
    public float logitScale() { return logitScale; }
    public float finalSoftcap() { return finalSoftcap; }
    /** FFN gate activation is GELU (tanh) instead of SiLU (dflash.cpp:13-21). */
    public boolean gelu() { return gelu; }
    public int nSwa() { return nSwa; }
    public boolean isSwa(int il) { return swa[il]; }

    /** Protocol block size: {@code dflash.block_size}, or {@value #DEFAULT_BLOCK_SIZE} when absent. */
    public int blockSize() { return blockSize; }
    /** {@code hparams.dflash_block_size} as the graph reads it (0 when absent). */
    public int graphBlockSize() { return graphBlockSize; }
    public boolean sampleFromAnchor() { return sampleFromAnchor; }
    /** Block-internal attention is causal ({@code dflash.attention.causal}). */
    public boolean causal() { return causal; }
    /** {@code dflash.has_confidence_head} (true when absent). */
    public boolean confidenceHeadMeta() { return confidenceHeadMeta; }
    public int maskTokenId() { return maskTokenId; }

    public boolean dspark() { return dspark; }
    public boolean dflash2() { return dflash2; }
    public boolean lilicorr() { return lilicorr; }
    public boolean conv() { return conv; }
    public boolean dsv4() { return dsv4; }
    public boolean hasD2t() { return d2t; }

    public int convKernel() { return convKernel; }
    public int convGroup() { return convGroup; }
    public int selectorRank() { return selectorRank; }
    public int selectorTopK() { return selectorTopK; }

    public int lilicorrHidden() { return lilicorrHidden; }
    public int lilicorrTopK() { return lilicorrTopK; }
    public int lilicorrHeads() { return lilicorrHeads; }
    public int lilicorrLayers() { return lilicorrLayers; }
    public float lilicorrVectorEps() { return lilicorrVectorEps; }
    public float lilicorrLogitScale() { return lilicorrLogitScale; }
    public int lilicorrFactorDim() { return lilicorrFactorDim; }
    public int lilicorrMlpHidden() { return lilicorrMlpHidden; }

    /** Hyper-connection streams of the DSV4 backbone; 0 otherwise. */
    public int hc() { return hc; }
    public float hcEps() { return hcEps; }
    public int hcSinkhornIters() { return hcSinkhornIters; }
    public int nLoraQ() { return nLoraQ; }
    public int oGroups() { return oGroups; }
    public int oLoraRank() { return oLoraRank; }
    /** Rotated trailing dims of a DSV4 head. */
    public int nRot() { return nRot; }
    public float ropeFreqBase() { return ropeFreqBase; }
    public int nExpert() { return nExpert; }
    public int nExpertUsed(int il) { return nExpertUsed[il]; }
    public int nExpertShared() { return nExpertShared; }
    public boolean expertWeightsNorm() { return expertWeightsNorm; }
    public float expertWeightsScale() { return expertWeightsScale; }
    public float clampExp(int il) { return clampExp[il]; }
    public float clampShexp(int il) { return clampShexp[il]; }

    /**
     * Largest draft: {@code block_size} for an anchor-first DSpark draft, {@code block_size - 1} otherwise
     * (speculative.cpp:1095-1104).
     */
    public int nDraftMax() {
        return dspark && sampleFromAnchor ? blockSize : blockSize - 1;
    }

    /**
     * Tokens of the noise block for a draft of {@code nMax} tokens: {@code block_size} for LiLiCorr, else
     * {@code nMax} plus the anchor slot unless the draft is anchor-first DSpark (speculative.cpp:1252).
     */
    public int blockTokens(int nMax) {
        return lilicorr ? blockSize : nMax + (dspark && sampleFromAnchor ? 0 : 1);
    }

    @Override
    public String toString() {
        return "DflashSpec{embd=" + nEmbd + ", layers=" + nLayer + ", head=" + headDim + ", vocab=" + nVocab + "/"
                + nVocabDraft + ", targets=" + targetLayers.length + ", block=" + blockSize
                + (dsv4 ? ", dsv4 hc=" + hc : "") + (dspark ? ", dspark" : "") + (dflash2 ? ", dflash2" : "")
                + (lilicorr ? ", lilicorr" : "") + (conv ? ", conv" : "") + (d2t ? ", d2t" : "")
                + ", anchor=" + sampleFromAnchor + ", causal=" + causal + '}';
    }
}
