package com.eonmux.neuralfree.gguf.hparams;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchEntry;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.PoolerAct;
import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.arch.UnsupportedArchException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Loads {@link ModelHparams} from GGUF metadata, mirroring llama.cpp {@code llama_model_base::load_hparams}
 * (src/llama-model.cpp:1361-1577):
 * <ol>
 *   <li>standard keys, with {@code context_length} required except for encoder-style models;</li>
 *   <li>NextN/MTP blocks excluded from the trunk ({@code nLayer = block_count - nextn_predict_layers});</li>
 *   <li>per-layer head and FFN arrays ({@code get_key_or_arr});</li>
 *   <li>RoPE scaling and head dimensions;</li>
 *   <li>the architecture hook ({@code load_arch_hparams});</li>
 *   <li>RoPE type resolution and validation.</li>
 * </ol>
 * Malformed or inconsistent metadata throws {@link GgufFormatException}; architectures that cannot run throw
 * {@link UnsupportedArchException}.
 */
public final class HparamsLoader {

    private static final Logger LOGGER = LogManager.getLogger(HparamsLoader.class);

    /** Training context assumed for encoder-style models whose file has no {@code context_length}. */
    public static final int ENCODER_FALLBACK_CTX = 512;

    private HparamsLoader() {
    }

    /**
     * Loads the hparams of the architecture named by {@code general.architecture}.
     *
     * @throws GgufFormatException when {@code general.architecture} is missing
     * @throws UnsupportedArchException when the architecture is unknown or cannot run
     */
    public static ModelHparams load(GgufKv kv) {
        String name = kv.archName();
        if (name == null) {
            throw new GgufFormatException("required metadata key 'general.architecture' is missing");
        }
        Arch arch = Arch.fromGgufName(name);
        if (arch == Arch.UNKNOWN) {
            throw new UnsupportedArchException(arch, "unknown architecture '" + name
                    + "': general.architecture does not name a llama.cpp architecture");
        }
        return load(kv, arch);
    }

    /** Loads the hparams of {@code arch} using its {@link ArchRegistry} entry. */
    public static ModelHparams load(GgufKv kv, Arch arch) {
        return load(kv, ArchRegistry.get(arch));
    }

    /**
     * Loads the hparams with an explicit registry entry.
     *
     * @throws GgufFormatException when the metadata names a different architecture or is malformed
     * @throws UnsupportedArchException when the entry is {@link SupportClass#NA}
     */
    public static ModelHparams load(GgufKv kv, ArchEntry entry) {
        Arch arch = entry.arch();
        ArchTraits traits = entry.traits();
        if (traits.support() == SupportClass.NA) {
            throw new UnsupportedArchException(arch, traits.unsupportedReason());
        }
        if (!arch.ggufName().equals(kv.archName())) {
            throw new GgufFormatException("metadata architecture '" + kv.archName() + "' does not match '"
                    + arch.ggufName() + "'");
        }
        ModelHparamsBuilder b = new ModelHparamsBuilder(traits);
        readGeneral(b, kv);
        readLayerCounts(b, kv);
        readExperts(b, kv);
        readLayerArrays(b, kv);
        readRope(b, kv);
        readDualChunk(b, kv);
        readHeadDims(b, kv);
        readHeads(b, kv);
        entry.hook().apply(b, kv);
        finish(b);
        ModelHparams hp = b.build();
        validate(hp);
        return hp;
    }

    // ---- 1. standard keys (llama-model.cpp:1384-1397) ----

    private static void readGeneral(ModelHparamsBuilder b, GgufKv kv) {
        Arch arch = b.arch();
        b.name(kv.str(GgufKeys.GENERAL_NAME, null));
        b.fileType(kv.u32(GgufKeys.GENERAL_FILE_TYPE, -1));
        b.nCtxTrain(readContextLength(b, kv));
        if (arch == Arch.WAVTOKENIZER_DEC) {
            b.nEmbd(kv.u32Required(GgufKeys.FEATURES_LENGTH));
            b.nEmbdOutImpl(kv.u32Required(GgufKeys.EMBEDDING_LENGTH));
        } else {
            b.nEmbd(kv.u32Required(GgufKeys.EMBEDDING_LENGTH));
            b.nEmbdOutImpl(kv.u32(GgufKeys.EMBEDDING_LENGTH_OUT, 0));
        }
        b.causal(kv.bool(GgufKeys.ATTENTION_CAUSAL, b.traits().causalDefault()));
        PoolingKind pooling = readPooling(kv, GgufKeys.POOLING_TYPE);
        b.pooling(pooling != null ? pooling : PoolingKind.NONE);
        b.poolingCls(readPooling(kv, GgufKeys.CLASSIFIER_POOLING_TYPE));
        String act = kv.str(GgufKeys.CLASSIFIER_ACTIVATION, null);
        if (act != null) {
            b.classifierActivation(classifierActivation(kv, act));
        }
        String poolerAct = kv.str(GgufKeys.CLASSIFIER_POOLER_ACTIVATION, null);
        if (poolerAct != null) {
            try {
                b.poolerActivation(PoolerAct.fromGgufName(poolerAct));
            } catch (IllegalArgumentException e) {
                throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.CLASSIFIER_POOLER_ACTIVATION)
                        + "': " + e.getMessage(), e);
            }
        }
        b.classifierTokenDirect(kv.bool(GgufKeys.CLASSIFIER_TOKEN_DIRECT, false));
        String sparse = kv.str(GgufKeys.CLASSIFIER_SPARSE, null);
        if (sparse != null) {
            if (!"log1p_relu_max".equals(sparse) && !"log1p_relu_sum".equals(sparse)) {
                throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.CLASSIFIER_SPARSE)
                        + "': unsupported sparse pooling '" + sparse + "', expected log1p_relu_max or log1p_relu_sum");
            }
            b.classifierSparse(sparse);
            b.sparseClsToken(kv.u32(GgufKeys.CLASSIFIER_SPARSE_CLS_TOKEN, 101));
            b.sparseSepToken(kv.u32(GgufKeys.CLASSIFIER_SPARSE_SEP_TOKEN, 102));
        }
        b.poolingNormalize(kv.bool(GgufKeys.POOLING_NORMALIZE, false));
        b.classifierBosToken(kv.u32(GgufKeys.CLASSIFIER_BOS_TOKEN, -1));
        b.classifierEosToken(kv.u32(GgufKeys.CLASSIFIER_EOS_TOKEN, -1));
        b.classifierStepTagId(kv.u32(GgufKeys.CLASSIFIER_STEP_TAG_ID, -1));
        String scope = kv.str(GgufKeys.CLASSIFIER_SCOPE, null);
        if (scope != null) {
            try {
                b.classifierScope(ClassifierScope.parse(scope));
            } catch (IllegalArgumentException e) {
                throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.CLASSIFIER_SCOPE) + "': "
                        + e.getMessage(), e);
            }
        }
        b.nVocab(readVocabSize(kv));
        b.logitScaleDivides(HparamsSupport.ScalingDefaults.dividesLogits(arch));
        if (arch.isDiffusion()) {
            b.diffusionShiftLogits(kv.bool(GgufKeys.DIFFUSION_SHIFT_LOGITS, true));
        }
    }

    /**
     * {@code context_length} is required (llama-model.cpp:1384). Embedding, classifier, encoder-decoder and
     * audio models may omit it; they get {@link #ENCODER_FALLBACK_CTX} with a warning.
     */
    private static int readContextLength(ModelHparamsBuilder b, GgufKv kv) {
        if (kv.has(GgufKeys.CONTEXT_LENGTH)) {
            return kv.u32Required(GgufKeys.CONTEXT_LENGTH);
        }
        SupportClass support = b.traits().support();
        boolean encoderStyle = support == SupportClass.EMBED || support == SupportClass.CLS
                || support == SupportClass.ENC_DEC || support == SupportClass.AUDIO;
        if (!encoderStyle) {
            throw new GgufFormatException("required metadata key '" + kv.key(GgufKeys.CONTEXT_LENGTH) + "' is missing");
        }
        LOGGER.warn("{} has no {}; assuming a training context of {} tokens", b.arch().ggufName(),
                kv.key(GgufKeys.CONTEXT_LENGTH), ENCODER_FALLBACK_CTX);
        return ENCODER_FALLBACK_CTX;
    }

    private static PoolingKind readPooling(GgufKv kv, String key) {
        long code = kv.u32Long(key, PoolingKind.GGUF_CODE_UNSPECIFIED);
        try {
            return PoolingKind.fromGgufCode(code);
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("metadata key '" + kv.key(key) + "': " + e.getMessage(), e);
        }
    }

    /**
     * {@code LLM_CLS_ACT_TYPES_FROM_STRING} (llama-model.cpp:1126-1130), plus {@code relu} for the jina_vl
     * reranker's {@code score} head (vLLM's {@code JinaVLScorer}).
     */
    private static ActOp classifierActivation(GgufKv kv, String name) {
        switch (name) {
            case "gelu":
                return ActOp.GELU_ERF;
            case "silu":
                return ActOp.SILU;
            case "tanh":
                return ActOp.TANH;
            case "relu":
                return ActOp.RELU;
            default:
                throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.CLASSIFIER_ACTIVATION)
                        + "': unsupported classifier activation '" + name + "'");
        }
    }

    /** {@code vocab_size}, else the tokenizer token count, else 0 (deepseek2.cpp n_vocab lookup). */
    private static int readVocabSize(GgufKv kv) {
        if (kv.has(GgufKeys.VOCAB_SIZE)) {
            return kv.u32Required(GgufKeys.VOCAB_SIZE);
        }
        return Math.max(0, kv.arrayLength(GgufKeys.TOKENIZER_LIST));
    }

    // ---- 2. layer counts and NextN exclusion (llama-model.cpp:1398-1401) ----

    private static void readLayerCounts(ModelHparamsBuilder b, GgufKv kv) {
        int nLayerAll = kv.u32Required(GgufKeys.BLOCK_COUNT);
        int nextn = kv.u32(GgufKeys.NEXTN_PREDICT_LAYERS, 0);
        b.layerCounts(nLayerAll, nextn);
    }

    // ---- experts (llama-model.cpp:1402-1446) ----

    private static void readExperts(ModelHparamsBuilder b, GgufKv kv) {
        b.moePerGroupTopK(kv.bool(GgufKeys.EXPERT_PER_GROUP_TOP_K, false));
        int n = b.nLayerAll();
        b.nExpert(kv.u32(GgufKeys.EXPERT_COUNT, 0));
        b.nExpertUsedArray(kv.keyOrArr(GgufKeys.EXPERT_USED_COUNT, n, 0));
        b.nExpertGroups(kv.u32(GgufKeys.EXPERT_GROUP_COUNT, 0));
        b.nGroupUsed(kv.u32(GgufKeys.EXPERT_GROUP_USED_COUNT, 0));
        if ((b.arch() == Arch.HUNYUAN_VL || b.arch() == Arch.HUNYUAN_DENSE) && b.nExpert() <= 1) {
            b.nExpert(0);
            b.nExpertUsedArray(new int[n]);
        }
        if (b.arch() == Arch.WAVTOKENIZER_DEC) {
            readWavTokenizer(b, kv);
        }
        if (b.moePerGroupTopK()) {
            // every group contributes nUsed / nGroups experts, so the group count alone defines the selection and the
            // kept-group count is informational: it defaults to all groups
            if (b.nExpertGroups() <= 1) {
                throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.EXPERT_PER_GROUP_TOP_K) + "': a"
                        + " per-group router needs more than one expert group, got " + b.nExpertGroups());
            }
            if (b.nGroupUsed() == 0) {
                b.nGroupUsed(b.nExpertGroups());
            }
        }
        checkExperts(b.arch(), b.nExpert(), max(b.nExpertUsedArray()), b.nExpertGroups(), b.nGroupUsed(),
                b.moePerGroupTopK());
    }

    private static void readWavTokenizer(ModelHparamsBuilder b, GgufKv kv) {
        b.posnetEmbd(kv.u32Required(GgufKeys.POSNET_EMBEDDING_LENGTH));
        b.posnetLayers(kv.u32Required(GgufKeys.POSNET_BLOCK_COUNT));
        b.convnextEmbd(kv.u32Required(GgufKeys.CONVNEXT_EMBEDDING_LENGTH));
        b.convnextLayers(kv.u32Required(GgufKeys.CONVNEXT_BLOCK_COUNT));
        if (b.posnetLayers() > b.nLayerAll() || b.convnextLayers() > b.nLayerAll()) {
            throw new GgufFormatException("wavtokenizer-dec: posnet/convnext block counts " + b.posnetLayers() + "/"
                    + b.convnextLayers() + " exceed the block count " + b.nLayerAll());
        }
    }

    /**
     * Expert count consistency (llama-model.cpp:1430-1446). A per-group router keeps every group, so its kept-group
     * count may equal the group count where the group-limited selection requires it to stay below.
     */
    private static void checkExperts(Arch arch, int nExpert, int usedMax, int groups, int groupUsed, boolean perGroup) {
        String a = arch.ggufName() + ": ";
        if (nExpert > ModelHparams.MAX_EXPERTS) {
            throw new GgufFormatException(a + "expert count " + nExpert + " exceeds " + ModelHparams.MAX_EXPERTS);
        }
        if (usedMax > nExpert) {
            throw new GgufFormatException(a + "experts used " + usedMax + " exceeds expert count " + nExpert);
        }
        if (nExpert > 0) {
            if (usedMax == 0) {
                throw new GgufFormatException(a + "expert_used_count must be positive when experts exist");
            }
            if (groups >= nExpert) {
                throw new GgufFormatException(a + "expert group count " + groups + " must be below expert count " + nExpert);
            }
            if (groups > 1 && (nExpert % groups != 0
                    || (perGroup ? groupUsed > groups : groupUsed <= 0 || groupUsed >= groups))) {
                throw new GgufFormatException(a + "invalid expert groups: " + nExpert + " experts, " + groups
                        + " groups, " + groupUsed + " used");
            }
        } else if (usedMax != 0 || groups != 0) {
            throw new GgufFormatException(a + "expert_used_count / expert_group_count set without experts");
        }
    }

    // ---- 3. per-layer arrays (llama-model.cpp:1448-1475) ----

    private static void readLayerArrays(ModelHparamsBuilder b, GgufKv kv) {
        int n = b.nLayerAll();
        b.nFfArray(kv.keyOrArr(GgufKeys.FEED_FORWARD_LENGTH, n, 0));
        int[] nHead = kv.keyOrArr(GgufKeys.ATTENTION_HEAD_COUNT, n, 0);
        b.nHeadArray(nHead);
        b.nHeadKvArray(kv.keyOrArr(GgufKeys.ATTENTION_HEAD_COUNT_KV, n, nHead));
    }

    // ---- 4. rope scaling (llama-model.cpp:1477-1509) ----

    private static void readRope(ModelHparamsBuilder b, GgufKv kv) {
        b.ropeFinetuned(kv.bool(GgufKeys.ROPE_SCALING_FINETUNED, false));
        b.nCtxOrigYarn(kv.u32(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, b.nCtxTrain()));
        b.ropeFreqBase(kv.f32(GgufKeys.ROPE_FREQ_BASE, 10000.0f));
        String scaling = kv.str(GgufKeys.ROPE_SCALING_TYPE, RopeScalingType.LINEAR.ggufName());
        RopeScalingType scalingType;
        try {
            scalingType = RopeScalingType.fromGgufName(scaling);
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.ROPE_SCALING_TYPE) + "': "
                    + e.getMessage(), e);
        }
        b.ropeScalingType(scalingType);
        String order = kv.str(GgufKeys.ROPE_SECTION_ORDER, "tyxz");
        if ("tyxz".equals(order)) {
            b.ropeSectionOrder(ModelHparams.SECTION_ORDER_TYXZ);
        } else if ("zyxt".equals(order)) {
            b.ropeSectionOrder(ModelHparams.SECTION_ORDER_ZYXT);
        } else {
            throw new GgufFormatException("unknown rope section order: " + order);
        }
        float ropeScale = kv.has(GgufKeys.ROPE_SCALING_FACTOR)
                ? kv.f32(GgufKeys.ROPE_SCALING_FACTOR, 0f)
                : kv.f32(GgufKeys.ROPE_SCALE_LINEAR, 0f);
        b.ropeScalingFactor(ropeScale);
        // dynamic NTK scales the base and fixed/mixed NTK scales the per-pair divisors: freq_scale stays 1
        b.ropeFreqScale(scalingType == RopeScalingType.DYNAMIC || scalingType == RopeScalingType.NTK
                ? 1.0f : ropeScale == 0.0f ? 1.0f : 1.0f / ropeScale);
        b.ropeAttnFactor(kv.f32(GgufKeys.ROPE_SCALING_ATTN_FACTOR, 1.0f));
        b.ropeScalingAlpha(kv.f32(GgufKeys.ROPE_SCALING_ALPHA, 0f));
        b.ropeScalingMixedB(kv.f32(GgufKeys.ROPE_SCALING_MIXED_B, Float.NaN));
        if (scalingType == RopeScalingType.DYNAMIC
                && !(Float.isFinite(b.ropeScalingAlpha()) && b.ropeScalingAlpha() > 0f)
                && !(Float.isFinite(b.ropeScalingFactor()) && b.ropeScalingFactor() > 0f)) {
            throw new GgufFormatException("metadata keys '" + kv.key(GgufKeys.ROPE_SCALING_ALPHA) + "' and '"
                    + kv.key(GgufKeys.ROPE_SCALING_FACTOR) + "': dynamic rope scaling needs a positive alpha or"
                    + " factor");
        }
        if (scalingType == RopeScalingType.NTK
                && !(Float.isFinite(b.ropeScalingFactor()) && b.ropeScalingFactor() > 0f)) {
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.ROPE_SCALING_FACTOR) + "': ntk rope"
                    + " scaling needs a positive factor");
        }
        readFope(b, kv);
    }

    /**
     * The Fourier position embedding keys. The learned coefficients live in the model-level
     * {@code rope_cos_coef.weight}/{@code rope_sin_coef.weight} tensors; the keys describe how the rotation reads
     * them.
     *
     * @throws GgufFormatException for a coefficient dictionary without the use flag, or a count outside the rotary
     *                             width
     */
    private static void readFope(ModelHparamsBuilder b, GgufKv kv) {
        b.ropeFope(kv.bool(GgufKeys.ROPE_FOPE_USE, false));
        b.ropeFopeSepHead(kv.bool(GgufKeys.ROPE_FOPE_SEP_HEAD, false));
        b.ropeFopeNumInvFreq(kv.u32(GgufKeys.ROPE_FOPE_NUM_INV_FREQ, 0));
        b.ropeFopeInitFactor(kv.f32(GgufKeys.ROPE_FOPE_INIT_FACTOR, 0f));
        if (!b.ropeFope()) {
            if (b.ropeFopeSepHead() || b.ropeFopeNumInvFreq() != 0) {
                throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.ROPE_FOPE_USE) + "': the Fourier"
                        + " position embedding tuning keys are set without it");
            }
            return;
        }
        if (b.ropeFopeNumInvFreq() < 0) {
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.ROPE_FOPE_NUM_INV_FREQ)
                    + "' must be >= 0, got " + b.ropeFopeNumInvFreq());
        }
    }

    // ---- dual-chunk attention ----

    /**
     * Dual-chunk attention of Qwen2.5-1M: {@code chunk_size}, {@code local_size}, the trained length of the logit
     * scaling factor and the sparse-attention knobs. The dense path is implemented; a file that enables the
     * vertical-slash sparse attention is refused by name.
     *
     * @throws GgufFormatException for an incomplete or inconsistent configuration, or for enabled sparse attention
     */
    private static void readDualChunk(ModelHparamsBuilder b, GgufKv kv) {
        if (!kv.has(GgufKeys.ATTENTION_DUAL_CHUNK_CHUNK_SIZE)) {
            return;
        }
        int chunk = kv.u32(GgufKeys.ATTENTION_DUAL_CHUNK_CHUNK_SIZE, 0);
        int local = kv.u32(GgufKeys.ATTENTION_DUAL_CHUNK_LOCAL_SIZE, 0);
        int orig = kv.u32(GgufKeys.ATTENTION_DUAL_CHUNK_ORIGINAL_MAX_POSITION_EMBEDDINGS, 0);
        if (chunk <= 0 || local <= 0 || chunk <= local) {
            throw new GgufFormatException("metadata keys '" + kv.key(GgufKeys.ATTENTION_DUAL_CHUNK_CHUNK_SIZE)
                    + "' and '" + kv.key(GgufKeys.ATTENTION_DUAL_CHUNK_LOCAL_SIZE) + "': dual-chunk attention needs"
                    + " 0 < local_size < chunk_size, got " + local + " / " + chunk);
        }
        if (orig < 0) {
            throw new GgufFormatException("metadata key '"
                    + kv.key(GgufKeys.ATTENTION_DUAL_CHUNK_ORIGINAL_MAX_POSITION_EMBEDDINGS)
                    + "': original_max_position_embeddings must be >= 0, got " + orig);
        }
        boolean sparse = kv.bool(GgufKeys.ATTENTION_DUAL_CHUNK_SPARSE_ENABLED, false);
        if (sparse) {
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.ATTENTION_DUAL_CHUNK_SPARSE_ENABLED)
                    + "': vertical-slash sparse dual-chunk attention is not supported; NeuralFree runs the dense"
                    + " dual-chunk attention only");
        }
        int threshold = kv.u32(GgufKeys.ATTENTION_DUAL_CHUNK_SPARSE_THRESHOLD, 32768);
        int lastQ = kv.u32(GgufKeys.ATTENTION_DUAL_CHUNK_SPARSE_LAST_Q, 64);
        if (threshold <= 0 || lastQ <= 0) {
            throw new GgufFormatException("metadata keys '" + kv.key(GgufKeys.ATTENTION_DUAL_CHUNK_SPARSE_THRESHOLD)
                    + "' and '" + kv.key(GgufKeys.ATTENTION_DUAL_CHUNK_SPARSE_LAST_Q)
                    + "': dual-chunk sparse attention needs a positive threshold and last_q, got " + threshold
                    + " / " + lastQ);
        }
        b.dualChunkChunkSize(chunk).dualChunkLocalSize(local)
                .dualChunkOriginalMaxPositionEmbeddings(orig)
                .dualChunkSparseEnabled(false).dualChunkSparseThreshold(threshold).dualChunkSparseLastQ(lastQ);
    }

    // ---- head dimensions (llama-model.cpp:1510-1545) ----

    private static void readHeadDims(ModelHparamsBuilder b, GgufKv kv) {
        int nHead0 = b.nHead(0);
        int headK;
        int headV;
        int nRot;
        if (nHead0 > 0) {
            headK = kv.u32(GgufKeys.ATTENTION_KEY_LENGTH, b.nEmbd() / nHead0);
            headV = kv.u32(GgufKeys.ATTENTION_VALUE_LENGTH, b.nEmbd() / nHead0);
            nRot = kv.u32(GgufKeys.ROPE_DIMENSION_COUNT, headK);
            Arch a = b.arch();
            boolean rotIsHead = a == Arch.LLAMA || a == Arch.DECI || a == Arch.FALCON || a == Arch.LLAMA_EMBED;
            if (rotIsHead && nRot != headK) {
                throw new GgufFormatException(a.ggufName() + ": invalid n_rot " + nRot + ", expected " + headK);
            }
        } else {
            // a mamba-first layer 0 (plamo2) has no attention heads; the metadata still
            // carries the head dims
            headK = kv.u32(GgufKeys.ATTENTION_KEY_LENGTH, 0);
            headV = kv.u32(GgufKeys.ATTENTION_VALUE_LENGTH, 0);
            nRot = kv.u32(GgufKeys.ROPE_DIMENSION_COUNT, headK);
        }
        b.headKFull(headK).headVFull(headV).nRotFull(nRot);
        b.headKSwa(kv.u32(GgufKeys.ATTENTION_KEY_LENGTH_SWA, b.headKFull()));
        b.headVSwa(kv.u32(GgufKeys.ATTENTION_VALUE_LENGTH_SWA, b.headVFull()));
        b.nRotSwa(kv.u32(GgufKeys.ROPE_DIMENSION_COUNT_SWA, b.nRotFull()));
    }

    // ---- classifier and decision labels (llama-model.cpp:1547-1555) ----

    private static void readHeads(ModelHparamsBuilder b, GgufKv kv) {
        String[] labels = kv.strArray(GgufKeys.CLASSIFIER_OUTPUT_LABELS);
        if (labels != null && labels.length > 0) {
            b.classifierLabels(labels);
            b.nClsOut(labels.length);
        }
        String[] decision = kv.strArray(GgufKeys.DECISION_LABELS);
        if (decision != null) {
            b.decisionLabels(decision);
        }
        b.decisionType(kv.str(GgufKeys.DECISION_TYPE, null));
        b.decisionMaxHeadTokens((int) kv.u32(GgufKeys.DECISION_MAX_HEAD_TOKENS, 0));
        b.decisionTemperatures(readDecisionTemperatures(kv));
        b.decisionScoreBias(readDecisionScoreBias(kv));
    }

    /**
     * The per-type decision temperatures under {@code <arch>.decision.temperature.<name>} (optionally bucketed
     * like {@code choice.3_5}), as llama.cpp's server collects them by key enumeration
     * (tools/server/server-decision.cpp init); a temperature of zero or less is refused.
     *
     * @throws GgufFormatException for a non-numeric value or an invalid temperature
     */
    private static Map<String, Float> readDecisionTemperatures(GgufKv kv) {
        String prefix = kv.key(GgufKeys.DECISION_TEMPERATURE_PREFIX);
        Map<String, Float> out = new LinkedHashMap<>();
        for (String k : kv.keysWithPrefix(prefix)) {
            float temp = kv.f32Required(k);
            if (temp <= 0.0f) {
                throw new GgufFormatException("invalid decision temperature: " + k + " = " + temp);
            }
            out.put(k.substring(prefix.length()), temp);
        }
        return out;
    }

    /**
     * The score-question offsets under {@code <arch>.decision.score_bias.<levels>.<level>}, merged into one
     * offset array per level count, as llama.cpp's server collects them (tools/server/server-decision.cpp
     * init); a level count outside {@code [2, 255]} or a level at or past its count is refused.
     *
     * @throws GgufFormatException for a malformed key, an out-of-range index or a non-numeric value
     */
    private static Map<Integer, float[]> readDecisionScoreBias(GgufKv kv) {
        String prefix = kv.key(GgufKeys.DECISION_SCORE_BIAS_PREFIX);
        Map<Integer, float[]> out = new LinkedHashMap<>();
        for (String k : kv.keysWithPrefix(prefix)) {
            String[] parts = k.substring(prefix.length()).split("\\.");
            long n = -1;
            long level = -1;
            if (parts.length == 2) {
                try {
                    n = Long.parseLong(parts[0]);
                    level = Long.parseLong(parts[1]);
                } catch (NumberFormatException e) {
                    n = -1;
                }
            }
            if (n < 2 || n > 255 || level < 0 || level >= n) {
                throw new GgufFormatException("invalid decision score bias: " + k);
            }
            float[] bias = out.computeIfAbsent((int) n, key -> new float[(int) key]);
            bias[(int) level] = kv.f32Required(k);
        }
        return out;
    }

    // ---- 6. post-hook resolution (llama-model.cpp:1563-1576) ----

    private static void finish(ModelHparamsBuilder b) {
        if (b.traits().forcedNonCausal()) {
            b.causal(false);
        }
        if (b.maxAlibiBias() > 0.0f) {
            b.useAlibi(true);
        }
        b.ropeType(resolveRopeType(b));
    }

    /**
     * {@code llama_model_rope_type} conditions (llama-model.cpp:3550-3667): dflash uses MROPE when any section
     * is set, else NORM with hyper-connections and NEOX without; archs with {@code mropeWhenSections} use
     * MROPE when {@code use_mrope()}; every other arch keeps the hook's or the trait's type.
     */
    private static RopeType resolveRopeType(ModelHparamsBuilder b) {
        if (b.arch() == Arch.DFLASH) {
            int[] s = b.ropeSections();
            if (s[0] != 0 || s[1] != 0 || s[2] != 0 || s[3] != 0) {
                return RopeType.MROPE;
            }
            return b.dsv4HcMult() > 0 ? RopeType.NORM : RopeType.NEOX;
        }
        if (b.traits().mropeWhenSections() && b.useMrope()) {
            return RopeType.MROPE;
        }
        return b.ropeType();
    }

    /**
     * Checks the invariants the engine relies on.
     *
     * @throws GgufFormatException naming the first violated invariant
     */
    public static void validate(ModelHparams hp) {
        String a = hp.arch().ggufName() + ": ";
        if (hp.nCtxTrain() <= 0) {
            throw new GgufFormatException(a + "context length must be positive, got " + hp.nCtxTrain());
        }
        if (hp.nEmbd() <= 0) {
            throw new GgufFormatException(a + "embedding length must be positive, got " + hp.nEmbd());
        }
        checkExperts(hp.arch(), hp.nExpert(), hp.nExpertUsedMax(), hp.nExpertGroups(), hp.nGroupUsed(),
                hp.moePerGroupTopK());
        for (int il = 0; il < hp.nLayerAll(); il++) {
            validateLayer(hp, il, a);
        }
        if (hp.swaType() != SwaType.NONE && hp.nSwa() <= 0) {
            throw new GgufFormatException(a + "sliding-window type " + hp.swaType() + " needs a positive window");
        }
        if ((hp.headKMlaImpl() == 0) != (hp.headVMlaImpl() == 0)) {
            throw new GgufFormatException(a + "MLA key and value head sizes must both be set or both be zero");
        }
        if (hp.ropeSectionOrder() != ModelHparams.SECTION_ORDER_TYXZ && hp.nPosPerEmbd() != ModelHparams.MROPE_SECTIONS) {
            throw new GgufFormatException(a + "rope section order zyxt requires M-RoPE");
        }
    }

    private static void validateLayer(ModelHparams hp, int il, String a) {
        int nHead = hp.nHead(il);
        int nHeadKv = hp.nHeadKv(il);
        if (!hp.isRecurrent(il) && nHead > 0 && nHeadKv > 0 && (nHeadKv > nHead || nHead % nHeadKv != 0)) {
            throw new GgufFormatException(a + "layer " + il + " has " + nHead + " query heads and " + nHeadKv
                    + " KV heads; query heads must be a multiple of KV heads");
        }
        boolean rotates = hp.ropeType() != RopeType.NONE && hp.hasRope(il) && nHead > 0;
        if (rotates && hp.nRot(il) % 2 != 0) {
            throw new GgufFormatException(a + "layer " + il + " rotary dimension " + hp.nRot(il) + " must be even");
        }
    }

    private static int max(int[] values) {
        return Arrays.stream(values).max().orElse(0);
    }
}
