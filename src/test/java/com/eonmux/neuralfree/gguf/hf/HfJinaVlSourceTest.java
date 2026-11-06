package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
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
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The jina_vl reranker ({@code JinaVLForRanking}) through the Hugging Face loader, against vLLM's {@code
 * JinaVLForSequenceClassification}: the Qwen2-VL text model under the newer {@code model.language_model.} names,
 * the {@code score.0} / {@code score.2} head as {@code cls} / {@code cls.output}, the vision tower left to the
 * projector, RANK pooling with the ReLU head, the document-first score template and the score token; the model
 * opens and scores one pair.
 */
public class HfJinaVlSourceTest {

    private static final int DIM = 8;
    private static final int KV = 4;
    private static final int FFN = 16;
    private static final int VOCAB = 8;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path checkpoint() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures': ['JinaVLForRanking'],"
                + " 'model_type': 'qwen2_vl', 'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'num_hidden_layers': 2, 'max_position_embeddings': 64,"
                + " 'rms_norm_eps': 1e-06, 'rope_theta': 10000.0, 'vocab_size': 8, 'num_labels': 1,"
                + " 'rope_scaling': {'rope_type': 'mrope', 'mrope_section': [1, 1, 0]}}"));
        b.bf16("model.language_model.embed_tokens.weight", VOCAB, DIM);
        for (int l = 0; l < 2; l++) {
            String p = "model.language_model.layers." + l + ".";
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
        b.bf16("model.language_model.norm.weight", DIM);
        b.bf16("lm_head.weight", VOCAB, DIM);
        b.bf16("model.visual.patch_embed.proj.weight", 16, 3, 2, 14, 14);
        b.bf16("model.visual.blocks.0.norm1.weight", 16);
        b.f32("score.0.weight", DIM, DIM);
        b.f32("score.0.bias", DIM);
        b.f32("score.2.weight", 1, DIM);
        b.f32("score.2.bias", 1);
        return b.write(tmp.newFolder("jina-reranker-m0").toPath());
    }

    @Test
    public void theRerankerLoadsAsAQwen2VlRankModelWithItsScoreHeadAndPrompt() throws IOException {
        Path dir = checkpoint();
        ModelSourceOptions options = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("qwen2vl", kv.archName());
            assertEquals(PoolingKind.RANK.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            assertEquals("relu", kv.str(kv.key(GgufKeys.CLASSIFIER_ACTIVATION), null));
            assertEquals("**Document**:\n{document}\n**Query**:\n{query}",
                    src.getString("tokenizer.chat_template.rerank"));
            assertEquals(List.of(100), ((List<?>) src.getMetadata().get(GgufKeys.NF_RERANK_SUFFIX_IDS)).stream()
                    .map(o -> ((Number) o).intValue()).collect(java.util.stream.Collectors.toList()));
            TensorCatalog tc = TensorCatalog.of(src);
            assertNotNull(tc.get("cls.weight"));
            assertNotNull(tc.get("cls.bias"));
            assertNotNull(tc.get("cls.output.weight"));
            assertNotNull(tc.get("cls.output.bias"));
            assertNotNull("the newer language_model names map to the blocks", tc.get("blk.1.attn_q.weight"));
            for (String name : tc.names()) {
                assertTrue(name + " is a vision tensor", !name.startsWith("v.") && !name.contains("visual"));
            }
            ModelSpec spec = ModelSpecBuilder.build(kv, tc, MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(ActOp.RELU, spec.hparams().classifierActivation());
            assertEquals(PoolingKind.RANK, spec.head().pooling());
            assertNull(kv.str(kv.key(GgufKeys.CLASSIFIER_POOLING_TYPE), null));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            assertEquals(1, model.classifyOutput(new int[]{2, 3, 4}).scores().length);
        }
    }
}
