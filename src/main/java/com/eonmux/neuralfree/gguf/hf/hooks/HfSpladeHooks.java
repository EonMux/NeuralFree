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
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Map;

/**
 * Hooks of vLLM's {@code BertSpladeSparseEmbeddingModel} (V/model_executor/models/bert.py:684): the BERT backbone
 * with the masked-LM head as the classifier, whose per-token logits the SPLADE pooler reduces to one sparse
 * embedding, {@code log1p(relu(x))} max or sum over the tokens (bert.py:590). The head maps
 * {@code cls.predictions.transform.dense.*} to {@code cls.*} with the GELU, the LayerNorm to {@code cls.norm}, and
 * {@code cls.predictions.decoder.*} to {@code cls.output.*}; {@code classifier.sparse} marks the reduction, with
 * the boundary token ids the pooler drops ({@code remove_cls_sep}, bert.py:16-19). vLLM's pooler also offers the
 * plain {@code token_embed} rows, so the head keeps {@code pooling.normalize} set.
 */
public final class HfSpladeHooks {

    /** The prefix of the masked-LM head tensors the pooler scores through (bert.py:753). */
    static final String MLM_PREFIX = "cls.predictions.";

    /** The BERT name filter keeping the masked-LM head tensors the classifier scores through. */
    public static final com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter FILTER = HfBertHooks::filterKeepingPredictions;

    private HfSpladeHooks() {
    }

    /** {@link HfBertHooks#meta} with the SPLADE classifier head metadata. */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertHooks.meta(ctx, kv, plan);
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.NONE.ggufCode());
        kv.set(GgufKeys.POOLING_NORMALIZE, true);                              // token_embed's PoolerNormalize
        kv.set(GgufKeys.CLASSIFIER_SCOPE, ClassifierScope.TOKEN.ggufName());
        kv.set(GgufKeys.CLASSIFIER_ACTIVATION, "gelu");                       // BertMLMHead's GELU (bert.py:577)
        String pooling = ctx.config().findString("splade_pooling").orElse("max");
        if (!"max".equals(pooling) && !"sum".equals(pooling)) {
            throw new com.eonmux.neuralfree.gguf.hparams.GgufFormatException(ctx.config().source()
                    + ": splade_pooling must be max or sum, got '" + pooling + "'");
        }
        kv.set(GgufKeys.CLASSIFIER_SPARSE, "sum".equals(pooling) ? "log1p_relu_sum" : "log1p_relu_max");
        if (ctx.config().findInt("cls_token_id").isPresent()) {
            kv.set(GgufKeys.CLASSIFIER_SPARSE_CLS_TOKEN, ctx.config().findInt("cls_token_id").getAsInt());
        }
        if (ctx.config().findInt("sep_token_id").isPresent()) {
            kv.set(GgufKeys.CLASSIFIER_SPARSE_SEP_TOKEN, ctx.config().findInt("sep_token_id").getAsInt());
        }
    }

    /**
     * The masked-LM head mapping over the BERT backbone: {@code cls.predictions.transform.dense.*} to {@code
     * cls.*}, {@code cls.predictions.transform.LayerNorm.*} to {@code cls.norm.*}, {@code
     * cls.predictions.decoder.*} to {@code cls.output.*}; the tied {@code cls.predictions.bias} drops beside its
     * {@code decoder.bias} copy (bert.py:772-783), and every other head tensor with it.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (hfName.startsWith(MLM_PREFIX)) {
            String key = hfName.substring(MLM_PREFIX.length());
            String name = MLM_HEAD.get(key);
            if (name == null) {
                out.drop(hfName, "the SPLADE head reads only the transform and decoder tensors; '" + key
                        + "' is unused");
                return;
            }
            out.add(name, src);
            return;
        }
        HfBertHooks.tensors(ctx, hfName, src, out);
    }

    /** The masked-LM head names and their GGUF projections (bert.py:772-783). */
    private static final Map<String, String> MLM_HEAD = Map.of(
            "transform.dense.weight", TensorNames.weight(TensorNames.CLS),
            "transform.dense.bias", TensorNames.bias(TensorNames.CLS),
            "transform.LayerNorm.weight", TensorNames.weight(TensorNames.CLS_NORM),
            "transform.LayerNorm.bias", TensorNames.bias(TensorNames.CLS_NORM),
            "decoder.weight", TensorNames.weight(TensorNames.CLS_OUT),
            "decoder.bias", TensorNames.bias(TensorNames.CLS_OUT));

    /** The BERT WordPiece vocabulary of the backbone. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfBertHooks.vocab(ctx, pre);
    }
}
