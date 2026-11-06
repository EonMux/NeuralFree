package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.util.Locale;

/**
 * Hooks of the vLLM ColBERT late-interaction classes (V/model_executor/models/colbert.py): {@code ColBERTMixin}
 * pools every token ({@code ALL}) behind a projector whose weights the checkpoints carry as {@code linear.*} or
 * {@code colbert_linear.*} (colbert.py:133-134), and the pooler L2-normalizes the projected rows (colbert.py:104-110).
 * The projector becomes the GGUF {@code dense_2} projection applied to every row of NONE pooling, and its outer
 * width the {@code embedding_length_out}; {@code pooling.normalize} marks the rows for the per-token L2 norm.
 *
 * <p>Every method composes the backbone family's hooks: the BERT / XLM-R step, the ModernBERT step or the LFM2
 * ColBERT step, each followed by the late-interaction metadata.</p>
 */
public final class HfColBertHooks {

    /** The first weight name the mixin reads (colbert.py:133). */
    static final String LINEAR = "linear.";
    /** The second weight name the mixin reads (colbert.py:134). */
    static final String COLBERT_LINEAR = "colbert_linear.";
    /** The GGUF projection the mixin's weights become. */
    static final String DENSE_OUT = "dense_2_out";

    private HfColBertHooks() {
    }

    // ------------------------------------------------------------------ metadata

    /** The BERT backbone metadata ({@code ColBERTModel}, colbert.py:195) with the late-interaction pooling. */
    public static void bertMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertHooks.meta(ctx, kv, plan);
        colBertMeta(kv);
    }

    /** The ModernBERT backbone metadata ({@code ColBERTModernBertModel}, colbert.py:238). */
    public static void modernMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertVariantHooks.modernMeta(ctx, kv, plan);
        colBertMeta(kv);
    }

    /** The LFM2 backbone metadata ({@code ColBERTLfm2Model}, colbert.py:378) with its non-causal attention. */
    public static void lfm2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfSsmHybridHooksB.colBertMeta(ctx, kv, plan);
        colBertMeta(kv);
    }

    /** NONE pooling over every token (vLLM's {@code ALL}) and the pooler's per-token L2 norm. */
    private static void colBertMeta(HfMetadataBuilder kv) {
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.NONE.ggufCode());
        kv.set(GgufKeys.POOLING_NORMALIZE, true);
    }

    // ------------------------------------------------------------------ tensors

    /** The projector rename over the BERT backbone ({@code ColBERTModel}). */
    public static void bertTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfBertHooks.tensors(ctx, projection(ctx, hfName, src, out), src, out);
    }

    /** The projector rename over the ModernBERT backbone ({@code ColBERTModernBertModel}). */
    public static void modernTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfBertVariantHooks.modernTensors(ctx, projection(ctx, hfName, src, out), src, out);
    }

    /** The projector rename over the XLM-R backbone ({@code ColBERTJinaRobertaModel}, colbert.py:314). */
    public static void xlmRobertaTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfBertHooks.xlmRobertaTensor(ctx, projection(ctx, hfName, src, out), src, out);
    }

    /** The projector rename over the LFM2 backbone ({@code ColBERTLfm2Model}). */
    public static void lfm2Tensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfSsmHybridHooksB.colBertTensor(ctx, projection(ctx, hfName, src, out), src, out);
    }

    /**
     * The name of {@code hfName} after the projector rename: {@code linear.*} and {@code colbert_linear.*} become
     * {@code dense_2_out.*}, the weight recording {@code embedding_length_out} from its outer width (the
     * {@code nn.Linear(hidden, colbert_dim)} rows, qwen.py:358-360); every other name passes unchanged. A
     * checkpoint without the projector keeps its plain per-token rows, like vLLM's unset {@code colbert_linear}.
     *
     * @throws GgufFormatException when the projector weight is not a matrix
     */
    static String projection(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        String name = hfName;
        if (name.startsWith(COLBERT_LINEAR)) {
            name = LINEAR + name.substring(COLBERT_LINEAR.length());
        }
        if (!name.startsWith(LINEAR)) {
            return hfName;
        }
        String renamed = DENSE_OUT + name.substring(LINEAR.length() - 1);
        if (name.equals(LINEAR + "weight")) {
            long[] shape = src.shape();
            if (shape.length == 0) {
                throw new GgufFormatException("tensor '" + hfName + "' is a scalar, not a ColBERT projection");
            }
            out.addKv(String.format(Locale.ROOT, GgufKeys.EMBEDDING_LENGTH_OUT, ctx.arch().ggufName()),
                    GGUFValueType.UINT32, shape[0]);
        }
        return renamed;
    }
}
