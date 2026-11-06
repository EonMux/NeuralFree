package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;

/**
 * The classifier-head fixtures of {@link SyntheticModels}: the bert rank, token-classifier, reward, process-reward,
 * sequence-classifier, ColBERT late-interaction and SPLADE sparse-embedding shapes. Each builder starts from the
 * bert fixture and adds the head keys and tensors vLLM's classifier models carry; {@link SyntheticModels} exposes
 * them.
 */
final class SyntheticClassifiers {

    private SyntheticClassifiers() {
    }

    /**
     * bert with RANK pooling and a {@code cls} classifier head over {@code labels}: the bge-reranker shape with
     * {@code classifier.output_labels}.
     */
    static GgufTestWriter rank(String... labels) {
        if (labels.length == 0) {
            throw new IllegalArgumentException("a rank fixture needs at least one classifier output label");
        }
        SyntheticModels.Config c = SyntheticModels.Config.bert();
        GgufTestWriter w = SyntheticModels.bert(c);
        w.u32(w.archKey("pooling_type"), SyntheticModels.POOLING_RANK);
        w.strArray(w.archKey("classifier.output_labels"), labels);
        w.tensorRandom("cls.weight", GGMLType.F32, c.seed() ^ ("cls.weight".hashCode() * 0x9E3779B97F4A7C15L),
                -0.1f, 0.1f, c.nEmbd(), labels.length);
        w.tensorZeros("cls.bias", GGMLType.F32, labels.length);
        return w;
    }

    /**
     * bert with NONE pooling and a {@code cls.output} per-token classifier over {@code labels}: vLLM's
     * {@code BertForTokenClassification} shape, one score row per token.
     */
    static GgufTestWriter tokenClassifier(String... labels) {
        if (labels.length == 0) {
            throw new IllegalArgumentException("a token classifier needs at least one label");
        }
        SyntheticModels.Config c = SyntheticModels.Config.bert();
        GgufTestWriter w = SyntheticModels.bert(c);
        w.u32(w.archKey("pooling_type"), 0);
        w.str(w.archKey("classifier.scope"), "token");
        w.strArray(w.archKey("classifier.output_labels"), labels);
        w.tensorRandom("cls.output.weight", GGMLType.F32,
                c.seed() ^ ("cls.output.weight".hashCode() * 0x9E3779B97F4A7C15L),
                -0.1f, 0.1f, c.nEmbd(), labels.length);
        w.tensorZeros("cls.output.bias", GGMLType.F32, labels.length);
        return w;
    }

    /**
     * A causal bert with NONE pooling and a two-layer ReLU reward head over every token, vLLM's
     * {@code Qwen2ForRewardModel} shape: one score per token (the last row is the sequence reward).
     */
    static GgufTestWriter reward() {
        return tokenReward(1);
    }

    /**
     * A causal bert with NONE pooling and a two-layer ReLU reward head over every token, vLLM's
     * {@code Qwen2ForProcessRewardModel} shape: one row of step scores per token.
     */
    static GgufTestWriter processReward() {
        return tokenReward(2);
    }

    private static GgufTestWriter tokenReward(int labels) {
        SyntheticModels.Config c = SyntheticModels.Config.bert();
        GgufTestWriter w = SyntheticModels.bert(c);
        w.u32(w.archKey("pooling_type"), 0);
        w.bool(w.archKey("attention.causal"), true);
        w.str(w.archKey("classifier.scope"), "token");
        w.str(w.archKey("classifier.activation"), "relu");
        w.tensorRandom("cls.weight", GGMLType.F32, c.seed() ^ ("cls.weight".hashCode() * 0x9E3779B97F4A7C15L),
                -0.1f, 0.1f, c.nEmbd(), c.nEmbd());
        w.tensorZeros("cls.bias", GGMLType.F32, c.nEmbd());
        w.tensorRandom("cls.output.weight", GGMLType.F32,
                c.seed() ^ ("cls.output.weight".hashCode() * 0x9E3779B97F4A7C15L),
                -0.1f, 0.1f, c.nEmbd(), labels);
        return w;
    }

    /**
     * bert with CLS pooling and a single-linear sequence classifier: {@code cls.output.weight} ({@code n_embd} to
     * {@code labels}), vLLM's cross-encoder sequence classifier.
     */
    static GgufTestWriter sequenceClassify(String... labels) {
        SyntheticModels.Config c = SyntheticModels.Config.bert();
        GgufTestWriter w = SyntheticModels.bert(c);
        w.u32(w.archKey("pooling_type"), 2);
        w.str(w.archKey("classifier.scope"), "sequence");
        w.strArray(w.archKey("classifier.output_labels"), labels);
        w.tensorRandom("cls.output.weight", GGMLType.F32,
                c.seed() ^ ("cls.output.weight".hashCode() * 0x9E3779B97F4A7C15L), -0.1f, 0.1f,
                c.nEmbd(), labels.length);
        w.tensorZeros("cls.output.bias", GGMLType.F32, labels.length);
        return w;
    }

    /**
     * bert with NONE pooling and the per-token L2 norm marked by {@code pooling.normalize}, the vLLM ColBERT
     * late-interaction shape: one L2-normalized projection row per token. The {@code dense_2} projection is the
     * identity, so a token's projected row is its normalized embedding and identical texts reach the maximal
     * MaxSim (every query row meets itself exactly).
     */
    static GgufTestWriter colbert() {
        SyntheticModels.Config c = SyntheticModels.Config.bert();
        GgufTestWriter w = SyntheticModels.bert(c);
        w.u32(w.archKey("pooling_type"), 0);
        w.bool(w.archKey("pooling.normalize"), true);
        w.u32(w.archKey("embedding_length_out"), c.nEmbd());
        float[] identity = new float[c.nEmbd() * c.nEmbd()];
        for (int i = 0; i < c.nEmbd(); i++) {
            identity[i * c.nEmbd() + i] = 1.0f;
        }
        w.tensorF32("dense_2.weight", new long[]{c.nEmbd(), c.nEmbd()}, identity);
        w.tensorZeros("dense_2.bias", GGMLType.F32, c.nEmbd());
        return w;
    }

    /**
     * bert with CLS pooling and the two BGE-M3 side projections, vLLM's {@code BgeM3EmbeddingModel} shape: {@code
     * cls.colbert} projects every row to the ColBERT token embedding (L2-normalized by {@code pooling.normalize}),
     * {@code cls.sparse} to one ReLU score per row; the pooler drops the leading BOS row of the fixture tokenizer
     * (CLS 2), and its token_classify filter also drops the trailing EOS row (SEP 3).
     */
    static GgufTestWriter bgeM3() {
        SyntheticModels.Config c = SyntheticModels.Config.bert();
        GgufTestWriter w = SyntheticModels.bert(c);
        w.u32(w.archKey("pooling_type"), 2);
        w.bool(w.archKey("pooling.normalize"), true);
        w.u32(w.archKey("classifier.bos_token"), 2);
        w.u32(w.archKey("classifier.eos_token"), 3);
        w.tensorRandom("cls.colbert.weight", GGMLType.F32,
                c.seed() ^ ("cls.colbert.weight".hashCode() * 0x9E3779B97F4A7C15L), -0.1f, 0.1f,
                c.nEmbd(), c.nEmbd());
        w.tensorZeros("cls.colbert.bias", GGMLType.F32, c.nEmbd());
        w.tensorRandom("cls.sparse.weight", GGMLType.F32,
                c.seed() ^ ("cls.sparse.weight".hashCode() * 0x9E3779B97F4A7C15L), -0.1f, 0.1f,
                c.nEmbd(), 1);
        w.tensorZeros("cls.sparse.bias", GGMLType.F32, 1);
        return w;
    }

    /**
     * bert with NONE pooling and the masked-LM head as the per-token classifier ({@code cls}, its GELU,
     * {@code cls.norm}, {@code cls.output} over the vocabulary) whose scores reduce into the sparse embedding, the
     * vLLM {@code BertSpladeSparseEmbeddingModel} shape. {@code kind} is {@code log1p_relu_max} or {@code
     * log1p_relu_sum}; the boundary ids follow the fixture tokenizer's CLS 2 and SEP 3.
     */
    static GgufTestWriter splade(String kind) {
        SyntheticModels.Config c = SyntheticModels.Config.bert();
        GgufTestWriter w = SyntheticModels.bert(c);
        w.u32(w.archKey("pooling_type"), 0);
        w.bool(w.archKey("pooling.normalize"), true);
        w.str(w.archKey("classifier.scope"), "token");
        w.str(w.archKey("classifier.activation"), "gelu");
        w.str(w.archKey("classifier.sparse"), kind);
        w.u32(w.archKey("classifier.sparse_cls_token"), 2);
        w.u32(w.archKey("classifier.sparse_sep_token"), 3);
        w.tensorRandom("cls.weight", GGMLType.F32,
                c.seed() ^ ("cls.weight".hashCode() * 0x9E3779B97F4A7C15L), -0.1f, 0.1f, c.nEmbd(), c.nEmbd());
        w.tensorZeros("cls.bias", GGMLType.F32, c.nEmbd());
        w.tensorFilled("cls.norm.weight", GGMLType.F32, 1.0f, c.nEmbd());
        w.tensorFilled("cls.norm.bias", GGMLType.F32, 0.0f, c.nEmbd());
        w.tensorRandom("cls.output.weight", GGMLType.F32,
                c.seed() ^ ("cls.output.weight".hashCode() * 0x9E3779B97F4A7C15L), -0.1f, 0.1f,
                c.nEmbd(), c.nVocab());
        w.tensorZeros("cls.output.bias", GGMLType.F32, c.nVocab());
        return w;
    }
}
