package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The hooks of the Qwen-family converter classes of the dense modern table that {@link HfQwenHooks} and
 * {@link HfQwenMoeHooks} do not already cover: {@code QwenModel} (LC/conversion/qwen.py:17-46, the tiktoken
 * vocabulary), {@code Qwen2VLModel} (LC/conversion/qwenvl.py:14-40, the {@code thinker.} strip), the
 * {@code Qwen3VLTextModel} deepstack count (LC/conversion/qwen3vl.py:280-300) with its {@code Qwen3ASRTextModel}
 * metadata and vocabulary overrides (:375-396), and {@code SenseNovaU1Model} (LC/conversion/sensenova_u1.py:11-44)
 * with its language-model filter, its spatial RoPE base and its concatenated q/k norms.
 */
public final class HfDenseHooksB {

    /** The thinker prefix the vision wrappers strip (qwenvl.py:37-38, qwen3vl.py:298). */
    private static final String THINKER_PREFIX = "thinker.";
    /** The language-model prefix SenseNova keeps (sensenova_u1.py:19). */
    private static final String LANGUAGE_MODEL_PREFIX = "language_model.";
    /** The tensor family the branch of SenseNova filters out (sensenova_u1.py:19). */
    private static final String MOT_GEN = "_mot_gen";
    /** The q/k norm weights whose halves SenseNova concatenates (sensenova_u1.py:40-43). */
    private static final String Q_NORM = ".q_norm.weight";
    private static final String K_NORM = ".k_norm.weight";
    private static final String Q_NORM_HW = ".q_norm_hw.weight";
    private static final String K_NORM_HW = ".k_norm_hw.weight";

    /** The chat template of {@code Qwen3ASRTextModel.set_vocab} (qwen3vl.py:387): the correct ChatML format. */
    public static final String CHATML_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role']"
            + " + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}"
            + "{{ '<|im_start|>assistant\\n' }}{% endif %}";
    /** The special token the ChatML template closes with (qwen3vl.py:365, :393). */
    public static final String IM_END = "<|im_end|>";

    private HfDenseHooksB() {
    }

    // ------------------------------------------------------------------ Qwen (v1)

    /**
     * {@code QwenModel.set_vocab} (qwen.py:45-46), whose call is the {@code _set_vocab_qwen} of
     * LC/conversion/base.py:2240-2294 — the tiktoken vocabulary with its rebuilt merges and special tokens,
     * in {@link HfVocabWriters#qwen}.
     */
    public static Map<String, Object> qwenVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.qwen(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ thinker strip

    /**
     * The class filter the vision text wrappers apply (qwenvl.py:33-40, qwen3vl.py:294-300): every occurrence of
     * {@value #THINKER_PREFIX} is removed, then the inherited filters apply. Qwen2-VL and the Qwen3-VL text model
     * share it ({@code qwenvl.py} does the replace the same way).
     */
    public static final HfNameFilter THINKER_FILTER = (name, base) ->
            base.apply(name.contains(THINKER_PREFIX) ? name.replace(THINKER_PREFIX, "") : name);

    /**
     * {@code Qwen2VLModel.set_vocab} (qwenvl.py:27-31): the SentencePiece vocabulary, else GPT-2 — without the
     * HybridDNA branch of {@code Qwen2Model.set_vocab}, which the class does not inherit.
     */
    public static Map<String, Object> qwen2VlVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        return HfVocabWriters.canSentencePiece(in) ? HfVocabWriters.sentencePiece(in, pre)
                : HfVocabWriters.gpt2(in, pre);
    }

    // ------------------------------------------------------------------ Qwen3-VL

    /**
     * {@code Qwen3VLTextModel.set_gguf_parameters} (qwen3vl.py:285-292): the Qwen3 metadata of
     * {@link HfQwenHooks#qwen3Meta}, then {@code n_deepstack_layers} = the length of the vision config's
     * {@code deepstack_visual_indexes}.
     */
    public static void qwen3VlMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenHooks.qwen3Meta(ctx, kv, plan);
        kv.set(GgufKeys.NUM_DEEPSTACK_LAYERS, deepstackCount(ctx.config()));
    }

    /** {@code Qwen3ASRTextModel.set_gguf_parameters} (qwen3vl.py:380-382): the Qwen3 metadata, no deepstack. */
    public static void asrMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenHooks.qwen3Meta(ctx, kv, plan);
        kv.set(GgufKeys.NUM_DEEPSTACK_LAYERS, 0);
    }

    /**
     * The length of {@code deepstack_visual_indexes} of the vision config the class names (qwen3vl.py:287-291):
     * a top-level {@code thinker_config.vision_config} when the root has {@code thinker_config}, else
     * {@code vision_config}; any other shape gives 0, as {@code .get(_, [])} does.
     */
    static int deepstackCount(HfConfig cfg) {
        Map<String, Object> flat = cfg.flat();
        Object thinker = flat.get("thinker_config");
        Object vision = thinker instanceof Map ? ((Map<?, ?>) thinker).get("vision_config") : flat.get("vision_config");
        if (!(vision instanceof Map)) {
            return 0;
        }
        Object indexes = ((Map<?, ?>) vision).get("deepstack_visual_indexes");
        return indexes instanceof List ? ((List<?>) indexes).size() : 0;
    }

    /**
     * {@code Qwen3ASRTextModel.set_vocab} (qwen3vl.py:384-396): the Qwen3 vocabulary (the InternS1 branch, then the
     * Qwen2 vocabulary of {@link HfQwenHooks}), then the correct ChatML template replaces the file's template and
     * bos and eos become the id of {@value #IM_END} in tokenizer_config.json's {@code added_tokens_decoder}.
     *
     * @throws IOException when a tokenizer file cannot be read
     */
    public static Map<String, Object> asrVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfQwenHooks.qwen3Vocab(ctx, pre));
        md.put(GgufKeys.TOKENIZER_CHAT_TEMPLATE, CHATML_TEMPLATE);
        imEnd(ctx.tokenizerFiles(), md);
        return Collections.unmodifiableMap(md);
    }

    /**
     * The bos and eos ids of the ASR/Omni overrides (qwen3vl.py:389-395): the first map key of
     * {@code added_tokens_decoder} whose {@code content} is {@value #IM_END}.
     */
    private static void imEnd(HfTokenizerFiles files, Map<String, Object> md) throws IOException {
        JsonView tc = files.readTokenizerConfig();
        Object decoder = tc.raw("added_tokens_decoder");
        if (decoder == null) {
            return;
        }
        if (!(decoder instanceof Map)) {
            throw new GgufFormatException(tc.where("added_tokens_decoder") + " must be an object");
        }
        for (Map.Entry<?, ?> e : ((Map<?, ?>) decoder).entrySet()) {
            Object content = e.getValue() instanceof Map ? ((Map<?, ?>) e.getValue()).get("content") : null;
            if (IM_END.equals(content)) {
                int id = HfQwenHooks.tokenId(files, IM_END);
                md.put(GgufKeys.TOKENIZER_BOS_ID, id);
                md.put(GgufKeys.TOKENIZER_EOS_ID, id);
                return;
            }
        }
    }

    // ------------------------------------------------------------------ SenseNovaU1

    /**
     * The class filter of {@code SenseNovaU1Model} (sensenova_u1.py:16-21): only the language-model tensors are
     * kept, the {@value #MOT_GEN} branch is dropped, and the inherited filters then apply to the kept name.
     */
    public static final HfNameFilter SENSENOVA_FILTER = (name, base) -> {
        if (!name.startsWith(LANGUAGE_MODEL_PREFIX) || name.contains(MOT_GEN)) {
            return null;
        }
        return base.apply(name);
    };

    /**
     * {@code SenseNovaU1Model.set_gguf_parameters} (sensenova_u1.py:23-33): the dense-understanding restrictions
     * of the class (MoE experts, scaled RoPE, attention bias and sliding attention are refused), the Qwen2
     * metadata and pooling of {@link HfQwenHooks#qwen2Meta}, and the required spatial RoPE base
     * {@code rope_theta_hw}.
     *
     * @throws GgufFormatException with the class's value errors when the checkpoint is not a dense U1.5
     */
    public static void sensenovaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        Object experts = cfg.flat().get("num_experts");
        long nExperts = experts instanceof Number && !(experts instanceof Boolean)
                ? ((Number) experts).longValue() : 0;
        if (nExperts > 1) {
            throw new GgufFormatException(cfg.source() + ": SenseNova U1 conversion currently supports the dense"
                    + " understanding branch");
        }
        Object ropeType = cfg.rope().raw().get("rope_type");
        if (ropeType != null && !"default".equals(String.valueOf(ropeType))) {
            throw new GgufFormatException(cfg.source() + ": SenseNova U1 scaled RoPE is not supported");
        }
        if (truthy(cfg.flat().get("attention_bias"))) {
            throw new GgufFormatException(cfg.source() + ": SenseNova U1 attention bias is not supported");
        }
        if (truthy(cfg.flat().get("use_sliding_window"))) {
            throw new GgufFormatException(cfg.source() + ": SenseNova U1 sliding attention is not supported");
        }
        HfQwenHooks.qwen2Meta(ctx, kv, plan);
        kv.set(GgufKeys.ROPE_FREQ_BASE_SPATIAL, requiredNumber(cfg, "rope_theta_hw"));
    }

    /**
     * The q/k norm tensors of {@code get_tensors} (sensenova_u1.py:35-44), then the Qwen2 tensors: the spatial
     * heads ({@code *_norm_hw.weight}) are never converted; every {@code *_norm.weight} is concatenated with its
     * spatial counterpart as one weight vector whose halves the graph normalizes independently, then mapped.
     */
    public static void sensenovaTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(Q_NORM_HW) || hfName.endsWith(K_NORM_HW)) {
            out.drop(hfName, "the spatial norm half serves the q/k concatenation only");
            return;
        }
        if (hfName.endsWith(Q_NORM) || hfName.endsWith(K_NORM)) {
            String sibling = hfName.replace("_norm.weight", "_norm_hw.weight");
            if (!ctx.index().has(sibling)) {
                throw new GgufFormatException("tensor '" + hfName + "' has no spatial counterpart '" + sibling
                        + "'");
            }
            out.consume(sibling);
            HfTensorHook.DEFAULT.apply(ctx, hfName,
                    RowTransforms.concatRows(Arrays.asList(src, ctx.index().get(sibling))), out);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** The value of {@code key}, required and a number ({@code hparams[key]}). */
    private static double requiredNumber(HfConfig cfg, String key) {
        Object v = cfg.flat().get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing");
        }
        return ((Number) v).doubleValue();
    }

    /** Python truthiness of a JSON value. */
    private static boolean truthy(Object v) {
        if (v == null) {
            return false;
        }
        if (v instanceof Boolean) {
            return (Boolean) v;
        }
        if (v instanceof Number) {
            return ((Number) v).doubleValue() != 0.0;
        }
        if (v instanceof String) {
            return !((String) v).isEmpty();
        }
        if (v instanceof List) {
            return !((List<?>) v).isEmpty();
        }
        return !(v instanceof Map) || !((Map<?, ?>) v).isEmpty();
    }
}
