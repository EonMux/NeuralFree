package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfSpecialFixturesBert;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfSsmFixturesHybridA;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * vLLM's pooling heads through the Hugging Face loader: {@code BertForTokenClassification} as a per-token
 * classifier ({@code token_classify}), {@code Qwen2ForRewardModel} and {@code Qwen2ForProcessRewardModel} as
 * per-token reward heads, {@code BertForSequenceClassification} over the CLS row, and {@code
 * GPT2ForSequenceClassification} as a last-row sequence classifier. Each checkpoint builds a spec claiming every
 * tensor, opens in the CPU model and answers the head's request while refusing the other pooling tasks.
 */
public class HfPoolingHeadsSourceTest {

    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int FFN = 16;
    private static final int LAYERS = 2;
    private static final int VOCAB = 8;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ================================================================== registration

    @Test
    public void everyPoolingHeadNameIsRegistered() {
        for (String name : new String[]{"BertForTokenClassification", "RobertaForTokenClassification",
                "XLMRobertaForTokenClassification", "ModernBertForTokenClassification", "Qwen2ForRewardModel",
                "Qwen2ForProcessRewardModel", "InternLM2ForRewardModel", "GPT2ForSequenceClassification",
                "JambaForSequenceClassification", "HF_ColBERT", "ColBERTModernBertModel",
                "ColBERTJinaRobertaModel", "ColBERTLfm2Model", "BertSpladeSparseEmbeddingModel"}) {
            assertTrue(name, HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains(name));
        }
    }

    // ================================================================== BERT token classification

    private Path bertTokenCheckpoint() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(
                "{'architectures': ['BertForTokenClassification'], 'model_type': 'bert', 'hidden_size': 8,"
                        + " 'intermediate_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                        + " 'num_key_value_heads': 2, 'max_position_embeddings': 64, 'vocab_size': 8,"
                        + " 'type_vocab_size': 2, 'pad_token_id': 0, 'num_labels': 3,"
                        + " 'id2label': {'0': 'B', '1': 'I', '2': 'O'}}"))
                .tokenizerJson(HfSpecialFixturesBert.WORD_PIECE_TOKENIZER)
                .tokenizerConfigJson(HfSpecialFixturesBert.WORD_PIECE_CONFIG);
        bertBackbone(b, "bert.");
        b.bf16("classifier.weight", 3, DIM);
        b.bf16("classifier.bias", 3);
        return b.write(tmp.newFolder("bert-token").toPath());
    }

    private static void bertBackbone(SyntheticHf.Builder b, String p) {
        b.bf16(p + "embeddings.word_embeddings.weight", VOCAB, DIM);
        b.bf16(p + "embeddings.position_embeddings.weight", 64, DIM);
        b.bf16(p + "embeddings.token_type_embeddings.weight", 2, DIM);
        b.bf16(p + "embeddings.LayerNorm.weight", DIM);
        b.bf16(p + "embeddings.LayerNorm.bias", DIM);
        for (int l = 0; l < LAYERS; l++) {
            String q = p + "encoder.layer." + l + ".";
            for (String proj : new String[]{"query", "key", "value"}) {
                b.bf16(q + "attention.self." + proj + ".weight", DIM, DIM);
                b.bf16(q + "attention.self." + proj + ".bias", DIM);
            }
            b.bf16(q + "attention.output.dense.weight", DIM, DIM);
            b.bf16(q + "attention.output.dense.bias", DIM);
            b.bf16(q + "attention.output.LayerNorm.weight", DIM);
            b.bf16(q + "attention.output.LayerNorm.bias", DIM);
            b.bf16(q + "intermediate.dense.weight", FFN, DIM);
            b.bf16(q + "intermediate.dense.bias", FFN);
            b.bf16(q + "output.dense.weight", DIM, FFN);
            b.bf16(q + "output.dense.bias", DIM);
            b.bf16(q + "output.LayerNorm.weight", DIM);
            b.bf16(q + "output.LayerNorm.bias", DIM);
        }
    }

    @Test
    public void bertTokenClassificationLoadsAsATokenClassifier() throws IOException {
        Path dir = bertTokenCheckpoint();
        try (ModelSource src = open(dir, "bert-token-cache")) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("bert", kv.archName());
            assertEquals(PoolingKind.NONE.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            TensorCatalog tc = TensorCatalog.of(src);
            assertNotNull(tc.get("cls.output.weight"));
            assertNotNull(tc.get("cls.output.bias"));
            assertNull("the token head is a single linear projection", tc.get("cls.weight"));
            ModelSpec spec = spec(kv, tc);
            assertEquals(PoolingKind.NONE, spec.head().pooling());
            assertTrue(EncodeGate.hasTokenClassifier(spec));
            assertFalse(EncodeGate.hasSequenceClassifier(spec));
            assertEquals(3, spec.head().clsOutput().ne(1));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            int[] tokens = {2, 4, 5, 3};
            assertEquals(3, model.classLabels().length);
            float[][] rows = model.tokenClassify(tokens);
            assertEquals(tokens.length, rows.length);
            for (float[] row : rows) {
                assertEquals(3, row.length);
            }
            assertThrows(UnsupportedOperationException.class, () -> model.encode(tokens));
            assertThrows(UnsupportedOperationException.class, () -> model.encodeTokens(tokens));
            assertThrows(UnsupportedOperationException.class, () -> model.classify(tokens));
        }
    }

    // ================================================================== BERT sequence classification

    @Test
    public void bertSequenceClassificationPoolsTheClsRow() throws IOException {
        Path dir = HfSpecialFixturesBert.bertClassification().write(tmp.newFolder("bert-seq").toPath());
        try (ModelSource src = open(dir, "bert-seq-cache")) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("bert", kv.archName());
            assertEquals(PoolingKind.CLS.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            assertEquals("sequence", kv.str(kv.key(GgufKeys.CLASSIFIER_SCOPE), null));
            TensorCatalog tc = TensorCatalog.of(src);
            assertNotNull(tc.get("cls.output.weight"));
            assertNotNull(tc.get("cls.output.bias"));
            ModelSpec spec = spec(kv, tc);
            assertTrue(EncodeGate.hasSequenceClassifier(spec));
            assertFalse(EncodeGate.hasTokenClassifier(spec));
            assertEquals(3, spec.head().clsOutput().ne(1));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            int[] tokens = {2, 4, 5, 3};
            assertEquals(3, model.classLabels().length);
            assertEquals(3, model.classifyOutput(tokens).size());
            assertEquals(3, model.classify(tokens).length);
            assertThrows(UnsupportedOperationException.class, () -> model.encode(tokens));
            float[][] rows = model.tokenClassify(tokens);
            assertEquals("vLLM's for_seq_cls dispatch also runs the classifier over every token", tokens.length,
                    rows.length);
            for (float[] row : rows) {
                assertEquals(3, row.length);
            }
        }
    }

    // ================================================================== Qwen2 reward models

    private Path qwen2RewardCheckpoint(String architecture, int labels, String name) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(
                "{'architectures': ['" + architecture + "'], 'model_type': 'qwen2', 'hidden_size': 8,"
                        + " 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                        + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06,"
                        + " 'rope_theta': 10000.0, 'vocab_size': 8, 'num_labels': " + labels + "}"));
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.q_proj.bias", DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.k_proj.bias", KV);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.bias", KV);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.up_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, FFN);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", VOCAB, DIM);
        b.f32("score.0.weight", DIM, DIM);
        b.f32("score.2.weight", labels, DIM);
        return b.write(tmp.newFolder(name).toPath());
    }

    @Test
    public void qwen2RewardModelScoresEveryToken() throws IOException {
        Path dir = qwen2RewardCheckpoint("Qwen2ForRewardModel", 1, "qwen2-reward");
        try (ModelSource src = open(dir, "qwen2-reward-cache")) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("qwen2", kv.archName());
            assertEquals(PoolingKind.NONE.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            assertEquals("token", kv.str(kv.key(GgufKeys.CLASSIFIER_SCOPE), null));
            assertEquals("relu", kv.str(kv.key(GgufKeys.CLASSIFIER_ACTIVATION), null));
            TensorCatalog tc = TensorCatalog.of(src);
            assertNotNull(tc.get("cls.weight"));
            assertNotNull(tc.get("cls.output.weight"));
            assertNull("the reward checkpoint's LM head is not converted", tc.get("output.weight"));
            ModelSpec spec = spec(kv, tc);
            assertEquals(PoolingKind.NONE, spec.head().pooling());
            assertEquals(ActOp.RELU, spec.hparams().classifierActivation());
            assertTrue(EncodeGate.hasTokenClassifier(spec));
            assertFalse(EncodeGate.hasSequenceClassifier(spec));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            int[] tokens = {2, 3, 4, 5};
            float[][] rows = model.tokenClassify(tokens);
            assertEquals(tokens.length, rows.length);
            for (float[] row : rows) {
                assertEquals("one reward score per token", 1, row.length);
            }
            assertThrows(UnsupportedOperationException.class, () -> model.classify(tokens));
            assertThrows(UnsupportedOperationException.class, () -> model.classifyOutput(tokens));
        }
    }

    @Test
    public void qwen2ProcessRewardModelScoresEveryToken() throws IOException {
        Path dir = qwen2RewardCheckpoint("Qwen2ForProcessRewardModel", 2, "qwen2-process-reward");
        try (ModelSource src = open(dir, "qwen2-process-cache")) {
            GgufKv kv = GgufKv.of(src);
            assertEquals(PoolingKind.NONE.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            assertEquals("token", kv.str(kv.key(GgufKeys.CLASSIFIER_SCOPE), null));
            TensorCatalog tc = TensorCatalog.of(src);
            ModelSpec spec = spec(kv, tc);
            assertTrue(EncodeGate.hasTokenClassifier(spec));
            assertEquals(2, spec.head().clsOutput().ne(1));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            int[] tokens = {2, 3, 4};
            float[][] rows = model.tokenClassify(tokens);
            assertEquals(tokens.length, rows.length);
            for (float[] row : rows) {
                assertEquals(2, row.length);
            }
            assertThrows(UnsupportedOperationException.class, () -> model.classifyOutput(tokens));
        }
    }

    // ================================================================== GPT-2 sequence classification

    private Path gpt2Checkpoint() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(
                "{'architectures': ['GPT2ForSequenceClassification'], 'model_type': 'gpt2', 'n_embd': 8,"
                        + " 'n_layer': 2, 'n_head': 2, 'n_ctx': 16, 'n_positions': 16, 'n_inner': 32,"
                        + " 'vocab_size': 8, 'layer_norm_epsilon': 1e-05, 'num_labels': 2}"));
        b.bf16("transformer.wte.weight", VOCAB, DIM);
        b.bf16("transformer.wpe.weight", 16, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "transformer.h." + l + ".";
            b.bf16(p + "ln_1.weight", DIM);
            b.bf16(p + "ln_1.bias", DIM);
            b.bf16(p + "attn.c_attn.weight", DIM, 3 * DIM);
            b.bf16(p + "attn.c_attn.bias", 3 * DIM);
            b.bf16(p + "attn.c_proj.weight", DIM, DIM);
            b.bf16(p + "attn.c_proj.bias", DIM);
            b.bf16(p + "ln_2.weight", DIM);
            b.bf16(p + "ln_2.bias", DIM);
            b.bf16(p + "mlp.c_fc.weight", DIM, 4 * DIM);
            b.bf16(p + "mlp.c_fc.bias", 4 * DIM);
            b.bf16(p + "mlp.c_proj.weight", 4 * DIM, DIM);
            b.bf16(p + "mlp.c_proj.bias", DIM);
        }
        b.bf16("transformer.ln_f.weight", DIM);
        b.bf16("transformer.ln_f.bias", DIM);
        b.bf16("score.weight", 2, DIM);
        return b.write(tmp.newFolder("gpt2-seq").toPath());
    }

    @Test
    public void gpt2SequenceClassificationPoolsTheLastRow() throws IOException {
        Path dir = gpt2Checkpoint();
        try (ModelSource src = open(dir, "gpt2-seq-cache")) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("gpt2", kv.archName());
            assertEquals(PoolingKind.LAST.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            TensorCatalog tc = TensorCatalog.of(src);
            assertNotNull(tc.get("cls.output.weight"));
            assertNull(tc.get("output.weight"));
            ModelSpec spec = spec(kv, tc);
            assertTrue(EncodeGate.hasSequenceClassifier(spec));
            assertEquals(2, spec.head().clsOutput().ne(1));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            assertEquals(2, model.classifyOutput(new int[]{2, 3, 4}).size());
            float[][] rows = model.tokenClassify(new int[]{2, 3, 4});
            assertEquals("a sequence classifier also answers token_classify per token", 3, rows.length);
            for (float[] row : rows) {
                assertEquals(2, row.length);
            }
        }
    }

    // ================================================================== Jamba sequence classification

    @Test
    public void jambaSequenceClassificationPoolsTheLastRow() throws IOException {
        SyntheticHf.Builder b = HfSsmFixturesHybridA.jamba();
        b.config(SyntheticHf.json("{'architectures': ['JambaForSequenceClassification'], 'hidden_size': 8,"
                + " 'num_hidden_layers': 2, 'vocab_size': 16, 'mamba_expand': 2, 'mamba_d_state': 8,"
                + " 'mamba_d_conv': 4, 'attn_layer_offset': 0, 'attn_layer_period': 2, 'expert_layer_offset': 0,"
                + " 'expert_layer_period': 2, 'num_key_value_heads': 2, 'num_attention_heads': 2,"
                + " 'intermediate_size': 16, 'max_position_embeddings': 16, 'num_local_experts': 2,"
                + " 'num_experts_per_tok': 1, 'rms_norm_eps': 1e-06, 'num_labels': 2,"
                + " 'rope_parameters': {'rope_theta': 10000.0}}"));
        b.bf16("score.weight", 2, DIM);
        Path dir = b.write(tmp.newFolder("jamba-seq").toPath());
        try (ModelSource src = open(dir, "jamba-seq-cache")) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("jamba", kv.archName());
            assertEquals(PoolingKind.LAST.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            assertEquals("sequence", kv.str(kv.key(GgufKeys.CLASSIFIER_SCOPE), null));
            TensorCatalog tc = TensorCatalog.of(src);
            assertNotNull(tc.get("cls.output.weight"));
            assertNull("the sequence classifier's LM head is not converted", tc.get("output.weight"));
            assertTrue(EncodeGate.hasSequenceClassifier(spec(kv, tc)));
        }
    }

    // ================================================================== ColBERT late interaction

    @Test
    public void hfColBertProjectsEveryTokenRowAndNormalizesIt() throws IOException {
        SyntheticHf.Builder b = HfSpecialFixturesBert.bert().config(SyntheticHf.json(
                "{'architectures': ['HF_ColBERT'], 'model_type': 'bert', 'hidden_size': 8,"
                        + " 'intermediate_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                        + " 'num_key_value_heads': 2, 'max_position_embeddings': 64, 'vocab_size': 8,"
                        + " 'type_vocab_size': 2, 'pad_token_id': 0}"));
        b.f32("linear.weight", 4, DIM);
        b.f32("linear.bias", 4);
        Path dir = b.write(tmp.newFolder("colbert").toPath());
        try (ModelSource src = open(dir, "colbert-cache")) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("bert", kv.archName());
            assertEquals(PoolingKind.NONE.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            assertTrue(kv.bool(kv.key(GgufKeys.POOLING_NORMALIZE), false));
            assertEquals(4, kv.u32(kv.key(GgufKeys.EMBEDDING_LENGTH_OUT), -1));
            TensorCatalog tc = TensorCatalog.of(src);
            assertNotNull(tc.get("dense_2.weight"));
            assertNotNull(tc.get("dense_2.bias"));
            assertEquals(DIM, tc.get("dense_2.weight").ne(0));
            assertEquals(4, tc.get("dense_2.weight").ne(1));
            ModelSpec spec = spec(kv, tc);
            assertTrue(EncodeGate.embedsPerToken(spec));
            assertFalse(EncodeGate.hasTokenClassifier(spec));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            int[] tokens = {2, 4, 5, 3};
            float[][] rows = model.encodeTokens(tokens);
            assertEquals(tokens.length, rows.length);
            for (int t = 0; t < rows.length; t++) {
                assertEquals(4, rows[t].length);
                assertEquals("every projected row is L2-normalized", 1.0f, norm(rows[t]), 1e-4f);
            }
            // the plain encode of a NONE pooling model flattens the same normalized rows
            float[] flat = model.encode(tokens);
            assertEquals(tokens.length * 4, flat.length);
            for (int t = 0; t < tokens.length; t++) {
                for (int i = 0; i < 4; i++) {
                    assertEquals(rows[t][i], flat[t * 4 + i], 1e-5f);
                }
            }
            assertThrows(UnsupportedOperationException.class, () -> model.tokenClassify(tokens));
        }
    }

    // ================================================================== SPLADE sparse embedding

    @Test
    public void bertSpladeReducesTheMlmHeadIntoASparseVector() throws IOException {
        SyntheticHf.Builder b = HfSpecialFixturesBert.bert().config(SyntheticHf.json(
                "{'architectures': ['BertSpladeSparseEmbeddingModel'], 'model_type': 'bert', 'hidden_size': 8,"
                        + " 'intermediate_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 2,"
                        + " 'num_key_value_heads': 2, 'max_position_embeddings': 64, 'vocab_size': 8,"
                        + " 'type_vocab_size': 2, 'pad_token_id': 0, 'cls_token_id': 2, 'sep_token_id': 3}"));
        b.bf16("cls.predictions.transform.dense.weight", DIM, DIM);
        b.bf16("cls.predictions.transform.dense.bias", DIM);
        b.f32("cls.predictions.transform.LayerNorm.weight", DIM);
        b.f32("cls.predictions.transform.LayerNorm.bias", DIM);
        b.bf16("cls.predictions.decoder.weight", VOCAB, DIM);
        b.f32("cls.predictions.decoder.bias", VOCAB);
        Path dir = b.write(tmp.newFolder("splade").toPath());
        try (ModelSource src = open(dir, "splade-cache")) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("bert", kv.archName());
            assertEquals(PoolingKind.NONE.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            assertEquals("token", kv.str(kv.key(GgufKeys.CLASSIFIER_SCOPE), null));
            assertEquals("gelu", kv.str(kv.key(GgufKeys.CLASSIFIER_ACTIVATION), null));
            assertEquals("log1p_relu_max", kv.str(kv.key(GgufKeys.CLASSIFIER_SPARSE), null));
            assertEquals(2, kv.u32(kv.key(GgufKeys.CLASSIFIER_SPARSE_CLS_TOKEN), -1));
            assertEquals(3, kv.u32(kv.key(GgufKeys.CLASSIFIER_SPARSE_SEP_TOKEN), -1));
            assertTrue("token_embed applies the pooler's L2 norm",
                    kv.bool(kv.key(GgufKeys.POOLING_NORMALIZE), false));
            TensorCatalog tc = TensorCatalog.of(src);
            assertNotNull(tc.get("cls.weight"));
            assertNotNull(tc.get("cls.norm.weight"));
            assertNotNull(tc.get("cls.output.weight"));
            assertEquals(DIM, tc.get("cls.output.weight").ne(0));
            assertEquals(VOCAB, tc.get("cls.output.weight").ne(1));
            ModelSpec spec = spec(kv, tc);
            assertEquals("log1p_relu_max", EncodeGate.sparsePooling(spec));
            assertFalse("the classifier reduces into the sparse embed", EncodeGate.hasTokenClassifier(spec));
            assertTrue("the plain rows are the token_embed output", EncodeGate.embedsPerToken(spec));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            int[] tokens = {2, 4, 5, 3};
            float[] vector = model.encode(tokens);
            assertEquals(VOCAB, vector.length);
            for (float v : vector) {
                assertTrue("log1p(relu(x)) keeps the vector non-negative", v >= 0.0f);
            }
            float[][] rows = model.encodeTokens(tokens);
            assertEquals(tokens.length, rows.length);
            for (float[] row : rows) {
                assertEquals("the token rows are the embedding width, not the vocabulary", DIM, row.length);
                assertEquals("vLLM normalizes the token_embed rows", 1.0f, norm(row), 1e-4f);
            }
            assertThrows(UnsupportedOperationException.class, () -> model.tokenClassify(tokens));
        }
    }

    // ================================================================== helpers

    private ModelSource open(Path dir, String cacheName) throws IOException {
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder(cacheName).toPath()).build();
        return ModelSources.open(dir.toString(), options, null);
    }

    private static ModelSpec spec(GgufKv kv, TensorCatalog tc) {
        ModelSpec spec = ModelSpecBuilder.build(kv, tc, MemoryConfig.defaults());
        TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
        return spec;
    }

    /** The Euclidean length of one row. */
    private static float norm(float[] row) {
        double sum = 0;
        for (float v : row) {
            sum += (double) v * v;
        }
        return (float) Math.sqrt(sum);
    }
}
