package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.PoolerAct;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Hooks of vLLM's {@code Qwen3ASRForcedAlignerForTokenClassification} (V/model_executor/models/
 * qwen3_asr_forced_aligner.py): the Qwen3-ASR text backbone with a per-token classification head that predicts
 * time bins at {@code <timestamp>} tokens. The checkpoint keeps the backbone under {@code thinker.model.*} and the
 * head under {@code thinker.lm_head.weight}, which becomes the GGUF {@code cls.output} projection applied to every
 * row; {@code POOLING_TYPE} is NONE and the scope TOKEN (the forced aligner's default token pooling is ALL). The
 * token pooler resolves vLLM's default classification activation — a softmax over the time bins, since the
 * checkpoint sets no {@code problem_type} — recorded as {@code classifier.pooler_activation}. The {@code talker} and
 * {@code code2wav} branches have no text weights, and the audio tower belongs to the projector file
 * (qwen3vl.py:387-406).
 */
public final class HfAsrForcedAlignerHooks {

    /** The Hugging Face architecture of the forced aligner (qwen3_asr_forced_aligner.py:30). */
    public static final String ARCH = "Qwen3ASRForcedAlignerForTokenClassification";

    /** The checkpoint prefixes that carry no text-model weight (qwen3_asr_forced_aligner.py:54-62). */
    private static final List<String> DROPPED_PREFIXES = Arrays.asList("audio_tower.", "talker.", "code2wav.");

    /** The head's weight name after the thinker prefix is stripped (qwen3_asr_forced_aligner.py:55). */
    private static final String LM_HEAD = "lm_head.weight";

    private HfAsrForcedAlignerHooks() {
    }

    /** The Qwen3-ASR text metadata with per-token classification of the head's time bins. */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfDenseHooksB.asrMeta(ctx, kv, plan);
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.NONE.ggufCode());
        kv.set(GgufKeys.CLASSIFIER_SCOPE, ClassifierScope.TOKEN.ggufName());
        kv.set(GgufKeys.CLASSIFIER_POOLER_ACTIVATION, PoolerAct.SOFTMAX.ggufName());
    }

    /**
     * The classifier head from {@code thinker.lm_head.weight} and the text backbone through the Qwen3 tensor map;
     * the audio tower and the generation branches drop.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (LM_HEAD.equals(hfName)) {
            out.add(TensorNames.weight(TensorNames.CLS_OUT), src);
            return;
        }
        for (String prefix : DROPPED_PREFIXES) {
            if (hfName.startsWith(prefix)) {
                out.drop(hfName, "the forced aligner's " + prefix.substring(0, prefix.length() - 1)
                        + " carries no text weight; the audio tower is planned for the projector file");
                return;
            }
        }
        HfQwenHooks.qwen3Tensors(ctx, hfName, src, out);
    }
}
