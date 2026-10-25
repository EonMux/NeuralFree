package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

/**
 * Hooks of the classic MoE converter classes, part D: the InternLM3 SPM fixups (internlm.py:175-234), the Janus and
 * Sarashina VL name filters (januspro.py:18-34, sarashina2.py:19-26) and the Qwen3-VL/Omni MoE rows
 * (qwen3vl.py:303-372).
 */
public final class HfMoeClassicHooksD {

    /** {@code JanusProModel.filter_tensors} (januspro.py:18-34), before the Llama filter. */
    public static final HfNameFilter JANUS_FILTER = HfMoeClassicHooksD::janusFilter;

    /** {@code Sarashina2VLTextModel.filter_tensors} (sarashina2.py:19-26), before the Llama filter. */
    public static final HfNameFilter SARASHINA_FILTER = HfMoeClassicHooksD::sarashinaFilter;

    /** {@code InternLM3Model.filter_tensors} (internlm.py:217-225), before the Llama filter. */
    public static final HfNameFilter INTERN_LM3_FILTER = HfMoeClassicHooksD::internLm3Filter;

    /**
     * The {@code thinker.} strip of {@code Qwen3VLMoeTextModel} and its Omni child (qwen3vl.py:315-319), before the
     * inherited Qwen3-MoE filters.
     */
    public static final HfNameFilter THINKER_FILTER = (name, base) -> base.apply(name.replace("thinker.", ""));

    private static final String WEIGHT = ".weight";
    private static final String OMNI_IM_END = "<|im_end|>";

    private HfMoeClassicHooksD() {
    }

    // ------------------------------------------------------------------ InternLM3Model

    /**
     * {@code InternLM3Model.set_gguf_parameters} (internlm.py:208-215) after the generic metadata: the vocabulary
     * size and the rope width default.
     */
    public static void internLm3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.VOCAB_SIZE, ctx.config().requireInt("vocab_size"));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, HfMoeClassicHooksA.ropeWidth(ctx.config()));
    }

    /**
     * {@code InternLM3Model.set_vocab} (internlm.py:178-206): the SentencePiece conversion writes model {@code
     * llama} and pre {@code default}, {@code add_prefix_space} from tokenizer_config, and rewrites the
     * {@code eos_token_id} from {@code <|im_end|>} when it appears as a special added token and an eos id was
     * resolved.
     */
    public static Map<String, Object> internLm3Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre));
        JsonView tc = files.readTokenizerConfig();
        Object addPrefix = tc.raw("add_prefix_space");
        if (addPrefix != null) {
            if (!(addPrefix instanceof Boolean)) {
                throw new GgufFormatException(tc.where("add_prefix_space") + " must be a boolean");
            }
            md.put(GgufKeys.TOKENIZER_ADD_PREFIX, addPrefix);
        }
        Map<?, ?> decoder = addedDecoderRaw(tc);
        String imEndId = null;
        for (Map.Entry<?, ?> e : decoder.entrySet()) {
            if (e.getValue() instanceof Map) {
                Map<?, ?> d = (Map<?, ?>) e.getValue();
                if (Boolean.TRUE.equals(d.get("special")) && OMNI_IM_END.equals(d.get("content"))) {
                    imEndId = String.valueOf(e.getKey());
                }
            }
        }
        if (imEndId != null && md.containsKey(GgufKeys.TOKENIZER_EOS_ID)) {
            md.put(GgufKeys.TOKENIZER_EOS_ID, Integer.parseInt(imEndId));
        }
        return Collections.unmodifiableMap(md);
    }

    /**
     * {@code InternLM3Model.modify_tensors} (internlm.py:227-234): the Llama q/k permutation (weights and biases),
     * then the name mapping.
     */
    public static void internLm3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource data = HfLlamaHooks.permuteQk(ctx.config(), hfName, src);
        out.add(ctx.names().mapOrThrow(hfName), data);
    }

    /** {@code InternLM3Model.filter_tensors} (internlm.py:217-225): the vision tower prefixes are dropped. */
    public static String internLm3Filter(String name, UnaryOperator<String> base) {
        if (name.startsWith("mlp") || name.startsWith("vision_model")) {
            return null;
        }
        return base.apply(name);
    }

    /** The added_tokens_decoder map of tokenizer_config.json, or an empty map when it is absent. */
    private static Map<?, ?> addedDecoderRaw(JsonView tc) {
        Object raw = tc.raw("added_tokens_decoder");
        if (!(raw instanceof Map)) {
            return Collections.emptyMap();
        }
        return (Map<?, ?>) raw;
    }

    // ------------------------------------------------------------------ Janus / Sarashina

    /** The skip of the vision, aligner and generation prefixes (januspro.py:18-34), then the Llama filter. */
    private static String janusFilter(String name, UnaryOperator<String> base) {
        for (String skip : Arrays.asList("model.vision_model.", "model.aligner.", "model.vqmodel.",
                "model.generation_embeddings.", "model.generation_aligner.", "model.generation_head.")) {
            if (name.startsWith(skip)) {
                return null;
            }
        }
        return HfLlamaHooks.filter(name, base);
    }

    /** The {@code llm.} strip and the top-level {@code norm.} dropout of sarashina2.py:19-26, then the Llama filter. */
    private static String sarashinaFilter(String name, UnaryOperator<String> base) {
        if (name.startsWith("llm.")) {
            return HfLlamaHooks.filter(name.substring("llm.".length()), base);
        }
        if (name.startsWith("norm.")) {
            return null;
        }
        return HfLlamaHooks.filter(name, base);
    }

    // ------------------------------------------------------------------ Qwen3-VL / Omni MoE

    /**
     * {@code Qwen3VLMoeTextModel.set_gguf_parameters} (qwen3vl.py:308-312) after {@code Qwen2MoeModel}:
     * the number of deepstack vision layers.
     */
    public static void qwen3VlMoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfQwenMoeHooks.qwen2MoeMeta(ctx, kv, plan);
        kv.set(GgufKeys.NUM_DEEPSTACK_LAYERS, deepstackLayers(ctx.config()));
    }

    private static int deepstackLayers(HfConfig cfg) {
        Object vision = cfg.flat().get("vision_config");
        Object indexes = vision instanceof Map ? ((Map<?, ?>) vision).get("deepstack_visual_indexes") : null;
        return indexes instanceof List ? ((List<?>) indexes).size() : 0;
    }

    /**
     * {@code Qwen3VLMoeTextModel.modify_tensors} (qwen3vl.py:322-350): the fused experts are TRANSPOSED versus
     * Qwen2MoE — {@code down_proj} {@code [E, F, D]} becomes {@code [E, D, F]}, and the fused
     * {@code gate_up_proj} {@code [E, D, 2F]} is transposed to {@code [E, 2F, D]}, where the gate and up halves
     * sit on separate rows ({@code -2}); these names and the remaining tensors flow into
     * {@code ModelBase.modify_tensors} / {@code Qwen2MoeModel.modify_tensors}.
     */
    public static void qwen3VlMoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith("mlp.experts.down_proj") || hfName.endsWith("mlp.experts.down_proj.weight")) {
            String mapped = hfName.endsWith(WEIGHT) ? hfName : hfName + WEIGHT;
            HfTensorHook.DEFAULT.apply(ctx, mapped, ExpertTransforms.transposeLast2(src), out);
            return;
        }
        if (hfName.endsWith("mlp.experts.gate_up_proj") || hfName.endsWith("mlp.experts.gate_up_proj.weight")) {
            long last = src.shape()[src.shape().length - 1];
            if (src.shape().length < 3 || last % 2 != 0) {
                throw new GgufFormatException("Unexpected gate_up_proj shape for " + hfName + ": "
                        + Arrays.toString(src.shape()));
            }
            RowSource transposed = ExpertTransforms.transposeLast2(src);
            // the transposed [E, 2F, D] rows hold the gate half first, so the split is at dim -2 (llama4-style),
            // which reproduces the converter's per-half permute(0, 2, 1) (qwen3vl.py:338-343)
            RowSource[] gateUp = ExpertTransforms.splitExperts(transposed, -2, 2);
            String base = stripTail(stripTail(hfName, WEIGHT), "gate_up_proj");
            if (base.endsWith(".")) {
                base = base.substring(0, base.length() - 1);
            }
            HfTensorHook.DEFAULT.apply(ctx, base + ".gate_proj" + WEIGHT, gateUp[0], out);
            HfTensorHook.DEFAULT.apply(ctx, base + ".up_proj" + WEIGHT, gateUp[1], out);
            return;
        }
        HfQwenMoeHooks.qwen2MoeTensors(ctx, hfName, src, out);
    }

    private static String stripTail(String s, String tail) {
        return s.endsWith(tail) ? s.substring(0, s.length() - tail.length()) : s;
    }

    /**
     * {@code Qwen3OmniMoeTextModel.set_gguf_parameters} (qwen3vl.py:370-372): zero deepstack layers, after the
     * inherited Qwen2MoE metadata.
     */
    public static void qwen3OmniMoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfQwenMoeHooks.qwen2MoeMeta(ctx, kv, plan);
        kv.set(GgufKeys.NUM_DEEPSTACK_LAYERS, 0);
    }

    /**
     * {@code Qwen3OmniMoeTextModel.set_vocab} (qwen3vl.py:358-368) after the inherited Qwen2MoE (Qwen3) vocabulary:
     * the BOS and EOS token ids are set to the id of {@code <|im_end|>} in tokenizer_config's
     * {@code added_tokens_decoder}, when one exists.
     */
    public static Map<String, Object> qwen3OmniMoeVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfQwenMoeHooks.qwen3MoeVocab(ctx, pre));
        HfTokenizerFiles files = ctx.tokenizerFiles();
        Map<?, ?> decoder = addedDecoderRaw(files.readTokenizerConfig());
        for (Map.Entry<?, ?> e : decoder.entrySet()) {
            if (e.getValue() instanceof Map && OMNI_IM_END.equals(((Map<?, ?>) e.getValue()).get("content"))) {
                int id = Integer.parseInt(String.valueOf(e.getKey()));
                if (id < HfConfig.MAX_VOCAB) {
                    md.put(GgufKeys.TOKENIZER_BOS_ID, id);
                    md.put(GgufKeys.TOKENIZER_EOS_ID, id);
                } else {
                    throw new GgufFormatException("tokenizer_config.json added_tokens_decoder key " + id
                            + " exceeds the vocabulary cap " + HfConfig.MAX_VOCAB);
                }
                return Collections.unmodifiableMap(md);
            }
        }
        return Collections.unmodifiableMap(md);
    }
}
