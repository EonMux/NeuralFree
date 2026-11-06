package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The text model of the jina_vl multimodal reranker (jina-reranker-m0, {@code JinaVLForRanking}), vLLM's {@code
 * JinaVLForSequenceClassification} (vllm/model_executor/models/jina_vl.py): the Qwen2-VL language model with a
 * two-layer score head ({@code score.0}: hidden to hidden, ReLU, {@code score.2}: hidden to the labels) over the
 * last row, the document-first score template {@value #SCORE_TEMPLATE} and the score token {@value #SCORE_TOKEN}
 * appended to every prompt ({@code post_process_tokens}). The vision tower ({@code visual.*}) belongs to the
 * projector file and is not planned here. Every method is a public static hook reference.
 */
public final class HfJinaVlHooks {

    /** The Hugging Face architecture names of the reranker. */
    public static final String JINA_VL_RANKING = "JinaVLForRanking";
    public static final String JINA_VL_SEQUENCE_CLASSIFICATION = "JinaVLForSequenceClassification";
    /** {@code get_score_template}: the document before the query. */
    static final String SCORE_TEMPLATE = "**Document**:\n{document}\n**Query**:\n{query}";
    /** The score target token id {@code post_process_tokens} appends. */
    static final int SCORE_TOKEN = 100;

    private HfJinaVlHooks() {
    }

    /**
     * {@link HfQwenHooks#qwen2Meta}, then RANK pooling over the last row (the causal row pooling), the ReLU
     * classifier activation, the score template as the {@code rerank} chat template and the score token as
     * {@link GgufKeys#NF_RERANK_SUFFIX_IDS}.
     */
    public static void jinaVlMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenHooks.qwen2Meta(ctx, kv, plan);
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.RANK.ggufCode());
        kv.set(GgufKeys.CLASSIFIER_ACTIVATION, "relu");
        kv.set(GgufKeys.NF_RERANK_SUFFIX_IDS, Collections.singletonList(SCORE_TOKEN));
        Map<String, String> template = new LinkedHashMap<>();
        template.put("name", HfQwenHooks.RERANK_TEMPLATE_NAME);
        template.put("template", SCORE_TEMPLATE);
        kv.putAll(HfChatTemplates.toGguf(Collections.singletonList(template)));
    }

    /**
     * vLLM's weight mapper: {@code score.0} becomes {@code cls}, {@code score.2} becomes {@code cls.output}, the
     * {@code visual.} tower stays with the projector, and the language model maps as {@link
     * HfQwenHooks#qwen2Tensors} maps it (the {@code model.language_model.} names of newer checkpoints included).
     */
    public static void jinaVlTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("visual.") || hfName.startsWith("model.visual.")) {
            return;
        }
        if (hfName.startsWith("score.0.")) {
            out.add(TensorNames.CLS + hfName.substring("score.0".length()), src);
        } else if (hfName.startsWith("score.2.")) {
            out.add(TensorNames.CLS_OUT + hfName.substring("score.2".length()), src);
        } else {
            HfQwenHooks.qwen2Tensors(ctx, hfName, src, out);
        }
    }
}
