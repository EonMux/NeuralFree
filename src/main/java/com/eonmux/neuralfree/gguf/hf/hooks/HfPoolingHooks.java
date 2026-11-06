package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;

/**
 * Hugging Face hooks of vLLM's reward, process-reward and sequence-classification heads over the model families
 * NeuralFree already converts (qwen2_rm.py, internlm2.py, gpt2.py and jamba.py). The heads are written as the GGUF
 * {@code cls} / {@code cls.output} projections the pooling head already reads: a token classifier (and vLLM's
 * reward models, whose default token pooling is {@code ALL}) pools per token ({@code NONE}) and applies the chain
 * to every row, while a sequence classifier pools one row (LAST) and applies it once. The BERT-family token and
 * sequence classifiers live in {@link HfBertHooks} and {@link HfBertVariantHooks}.
 */
public final class HfPoolingHooks {

    /** The Qwen2 reward head's first projection ({@code score.0}). */
    private static final String SCORE_FIRST = "score.0.";
    /** The Qwen2 reward head's output projection ({@code score.2}). */
    private static final String SCORE_LAST = "score.2.";
    /** The InternLM2 reward projection. */
    private static final String V_HEAD = "v_head.";
    /** vLLM's process-reward step token (V/model_executor/models/config.py:989). */
    static final int STEP_TAG_ID = 151651;

    private HfPoolingHooks() {
    }

    // ------------------------------------------------------------------ Qwen2 reward

    /**
     * {@code Qwen2ForRewardModel}: the Qwen2 metadata and per-token scores ({@code NONE}, vLLM's default token
     * pooling {@code ALL}), with the ReLU between the two score projections; the reward of a sequence is the score
     * of its last token row.
     */
    public static void qwen2RewardMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        qwen2TokenMeta(ctx, kv, plan);
    }

    /**
     * {@code Qwen2ForProcessRewardModel}: the same per-token score rows as {@link #qwen2RewardMeta}, one row per
     * token holding the step scores, and vLLM's default {@code step_tag_id} applied to a bare {@code token_classify}
     * request (V/model_executor/models/config.py:982-989).
     */
    public static void qwen2ProcessMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        qwen2TokenMeta(ctx, kv, plan);
        kv.set(GgufKeys.CLASSIFIER_STEP_TAG_ID, STEP_TAG_ID);
    }

    private static void qwen2TokenMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfQwenHooks.qwen2Meta(ctx, kv, plan);
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.NONE.ggufCode());
        kv.set(GgufKeys.CLASSIFIER_SCOPE, ClassifierScope.TOKEN.ggufName());
        kv.set(GgufKeys.CLASSIFIER_ACTIVATION, "relu");
    }

    /** Plans {@code score.0.*} as {@code cls.*} and {@code score.2.*} as {@code cls.output.*}; drops the LM head. */
    public static void qwen2RewardTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith(SCORE_FIRST)) {
            out.add(TensorNames.CLS + hfName.substring(SCORE_FIRST.length() - 1), src);
            return;
        }
        if (hfName.startsWith(SCORE_LAST)) {
            out.add(TensorNames.CLS_OUT + hfName.substring(SCORE_LAST.length() - 1), src);
            return;
        }
        if (hfName.startsWith("lm_head.")) {
            out.drop(hfName, "a reward model stores no LM head (vLLM's AutoWeightsLoader ignores lm_head)");
            return;
        }
        HfQwenHooks.qwen2Tensors(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ InternLM2 reward

    /**
     * {@code InternLM2ForRewardModel}: the InternLM2 metadata and per-token scores ({@code NONE}) of the
     * {@code v_head} projection; the reward of a sequence is the score of its last token row.
     */
    public static void internlm2RewardMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDenseClassicHooksE.internlm2Meta(ctx, kv, plan);
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.NONE.ggufCode());
        kv.set(GgufKeys.CLASSIFIER_SCOPE, ClassifierScope.TOKEN.ggufName());
    }

    /** Plans {@code v_head.weight} as {@code cls.output.weight}; drops the LM head. */
    public static void internlm2RewardTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith(V_HEAD)) {
            out.add(TensorNames.CLS_OUT + hfName.substring(V_HEAD.length() - 1), src);
            return;
        }
        if (hfName.startsWith("lm_head.") || hfName.equals("output.weight")) {
            out.drop(hfName, "a reward model stores no LM head (vLLM's AutoWeightsLoader ignores lm_head)");
            return;
        }
        HfDenseClassicHooksE.internlm2Tensors(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ GPT2 sequence classification

    /** {@code GPT2ForSequenceClassification}: the GPT-2 metadata and LAST-token pooling of the sequence. */
    public static void gpt2SeqMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDenseClassicHooksC.gpt2Meta(ctx, kv, plan);
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.LAST.ggufCode());
        kv.set(GgufKeys.CLASSIFIER_SCOPE, ClassifierScope.SEQUENCE.ggufName());
    }

    /** Plans {@code score.weight} / {@code score.bias} as {@code cls.output}; drops the LM head. */
    public static void gpt2SeqTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("score.")) {
            out.add(TensorNames.CLS_OUT + hfName.substring("score".length()), src);
            return;
        }
        if (hfName.startsWith("lm_head.")) {
            out.drop(hfName, "a sequence classifier stores no LM head (vLLM's AutoWeightsLoader ignores lm_head)");
            return;
        }
        HfDenseClassicHooksC.gpt2Tensors(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ Jamba sequence classification

    /** {@code JambaForSequenceClassification}: the Jamba metadata and LAST-token pooling of the sequence. */
    public static void jambaSeqMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfSsmHooksA.jambaMeta(ctx, kv, plan);
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.LAST.ggufCode());
        kv.set(GgufKeys.CLASSIFIER_SCOPE, ClassifierScope.SEQUENCE.ggufName());
    }

    /** Plans {@code score.weight} / {@code score.bias} as {@code cls.output}; drops the LM head. */
    public static void jambaSeqTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("score.")) {
            out.add(TensorNames.CLS_OUT + hfName.substring("score".length()), src);
            return;
        }
        if (hfName.startsWith("lm_head.") || hfName.equals("output.weight")) {
            out.drop(hfName, "a sequence classifier stores no LM head (vLLM's AutoWeightsLoader ignores lm_head)");
            return;
        }
        HfSsmHooksA.jambaTensor(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ Llama sequence classification

    /**
     * {@code LlamaBidirectionalForSequenceClassification} (llama.py:574, vLLM's {@code as_seq_cls_model} over the
     * bidirectional Llama): the LLAMA_EMBED metadata and LAST-token pooling of the sequence.
     */
    public static void llamaBidirectionalSeqMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfLlamaHooks.meta(ctx, kv, plan);
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.LAST.ggufCode());
        kv.set(GgufKeys.CLASSIFIER_SCOPE, ClassifierScope.SEQUENCE.ggufName());
        HfBertHooks.applyPoolerActivation(ctx.config(), kv);
    }

    /** Plans {@code score.weight} / {@code score.bias} as {@code cls.output}; drops the LM head. */
    public static void llamaSeqTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (hfName.startsWith("score.")) {
            out.add(TensorNames.CLS_OUT + hfName.substring("score".length()), src);
            return;
        }
        if (hfName.startsWith("lm_head.") || hfName.equals("output.weight")) {
            out.drop(hfName, "a sequence classifier stores no LM head (vLLM's AutoWeightsLoader ignores lm_head)");
            return;
        }
        HfLlamaHooks.tensors(ctx, hfName, src, out);
    }
}
