package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Hooks of vLLM's {@code BgeM3EmbeddingModel} (V/model_executor/models/roberta.py:187-269): the XLM-RoBERTa
 * backbone with three outputs. The {@code embed} task is the CLS-pooled, L2-normalized hidden row; {@code
 * token_embed} projects every row through {@code colbert_linear} and normalizes it; {@code token_classify} scores
 * every row through {@code sparse_linear} with a ReLU. The two side projections live in their own {@code
 * colbert_linear.pt} and {@code sparse_linear.pt} files, which this hook reads as a restricted PyTorch checkpoint
 * and writes as the GGUF {@code cls.colbert} and {@code cls.sparse} tensors (roberta.py:203-218,
 * :252-267). The {@code token_classify} pooler's boundary filter drops the leading BOS and trailing EOS rows; the
 * {@code token_embed} filter drops only the leading BOS row (special.py:170-222, roberta.py:236-249).
 */
public final class HfBgeM3Hooks {

    /** The Hugging Face architecture of the multi-output embedder (roberta.py:187). */
    public static final String ARCH = "BgeM3EmbeddingModel";

    /** The sparse projection's side file (roberta.py:204). */
    static final String SPARSE_FILE = "sparse_linear.pt";
    /** The ColBERT projection's side file (roberta.py:204). */
    static final String COLBERT_FILE = "colbert_linear.pt";

    /** The XLM-RoBERTa backbone names; the side files are planned by {@link #sideTensors}. */
    public static final HfNameFilter FILTER = HfBertHooks.FILTER_XLMR;

    private HfBgeM3Hooks() {
    }

    /**
     * {@link HfBertHooks#meta} with the CLS pooling, the pooler's row normalization and the BOS/EOS boundary ids
     * the multi-output poolers drop.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        // BgeM3EmbeddingModel extends RobertaEmbeddingModel, whose default sequence pooling is CLS
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.CLS.ggufCode());
        kv.set(GgufKeys.POOLING_NORMALIZE, true);
        cfg.findInt("bos_token_id").ifPresent(id -> kv.set(GgufKeys.CLASSIFIER_BOS_TOKEN, id));
        cfg.findInt("eos_token_id").ifPresent(id -> kv.set(GgufKeys.CLASSIFIER_EOS_TOKEN, id));
    }

    /**
     * The two side projections ({@code sparse_linear.pt} and {@code colbert_linear.pt}) as the GGUF {@code
     * cls.sparse} and {@code cls.colbert} weights and biases.
     *
     * @throws GgufFormatException when a side file is missing or holds a tensor that is not the expected
     *                             {@code [out, hidden]} linear
     * @throws IOException         when a side file cannot be read
     */
    public static void sideTensors(HfLoadContext ctx, TensorSink out) throws IOException {
        int hidden = ctx.config().findInt(0, "hidden_size");
        readProjection(ctx, SPARSE_FILE, "sparse_linear", TensorNames.CLS_SPARSE, hidden, out);
        readProjection(ctx, COLBERT_FILE, "colbert_linear", TensorNames.CLS_COLBERT, hidden, out);
    }

    /**
     * Plans one side file's tensors: its {@code weight} (required) and {@code bias} (optional), under the GGUF base
     * name {@code ggufName}.
     */
    private static void readProjection(HfLoadContext ctx, String file, String stem, String ggufName, int hidden,
                                       TensorSink out) throws IOException {
        HfConfig cfg = ctx.config();
        Path path = cfg.dir().resolve(file);
        if (!Files.isRegularFile(path)) {
            throw new GgufFormatException(cfg.source() + ": the checkpoint has no '" + file + "'; the "
                    + ARCH + " sparse and ColBERT heads live there");
        }
        boolean weight = false;
        try (CheckpointContainer side = CheckpointContainers.open(path, cfg.dir(), false)) {
            for (StTensor t : side.tensors()) {
                String key = localName(t.name(), stem);
                if ("weight".equals(key)) {
                    weight = true;
                    checkWeight(ctx, file, t, hidden);
                    out.add(TensorNames.weight(ggufName), side.source(t.name()));
                } else if ("bias".equals(key)) {
                    out.add(TensorNames.bias(ggufName), side.source(t.name()));
                } else {
                    throw new GgufFormatException(file + " holds the unexpected tensor '" + t.name() + "'; the "
                            + stem + " state dict has weight and bias only");
                }
            }
        }
        if (!weight) {
            throw new GgufFormatException(file + " holds no '" + stem + ".weight'");
        }
    }

    /** The tensor's key without an optional {@code <stem>.} prefix. */
    private static String localName(String name, String stem) {
        return name.startsWith(stem + ".") ? name.substring(stem.length() + 1) : name;
    }

    /** Checks a projection weight is {@code [out, hidden]}, the torch linear layout. */
    private static void checkWeight(HfLoadContext ctx, String file, StTensor t, int hidden) {
        long[] shape = t.shape();
        if (shape.length != 2 || hidden <= 0 || shape[1] != hidden) {
            throw new GgufFormatException(ctx.config().source() + ": '" + file + "' tensor '" + t.name() + "' has"
                    + " shape " + java.util.Arrays.toString(shape) + ", expected [out, " + hidden + "]");
        }
    }

    /** The XLM-RoBERTa vocabulary of the backbone. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfBertHooks.xlmRobertaVocabEntry(ctx, pre);
    }
}
