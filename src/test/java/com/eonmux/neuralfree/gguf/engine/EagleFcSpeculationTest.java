package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.testutil.SyntheticEagle;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * Speculative decoding with the four concat-fc drafters: the drafter proposes from the target's post-output-norm
 * hidden state, the target verifies, and the generated tokens are the target's plain greedy tokens (verification is
 * lossless). Each draft family runs on the CPU and on OpenCL, and each family is also converted from a synthetic
 * Hugging Face checkpoint directory, exercising {@code HfEagleFcHooks}' metadata, its tensor mapping and the
 * target-shared vocabulary.
 */
public class EagleFcSpeculationTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26, 3, 64, 8, 51};
    private static final int GREEDY_TOKENS = 12;
    private static final int TARGET_LAYERS = 3;
    private static final int TAP = TARGET_LAYERS;
    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int HEAD_DIM = 16;
    private static final int N_FF = 128;
    private static final int VOCAB = 320;
    private static final int MLA_HEAD_DIM = 32;
    private static final int MLA_ROT = 16;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theCohereDrafterKeepsThePlainGreedyTokens() throws IOException {
        keepsThePlainGreedyTokens(cohereTarget(), cohereDraft(), base().build());
    }

    @Test
    public void theMiniCpmDrafterKeepsThePlainGreedyTokens() throws IOException {
        keepsThePlainGreedyTokens(llamaTarget(), miniCpmDraft(), base().build());
    }

    @Test
    public void theLlama4DrafterKeepsThePlainGreedyTokens() throws IOException {
        keepsThePlainGreedyTokens(llamaTarget(), llama4Draft(), base().build());
    }

    @Test
    public void theMistralLarge3DrafterKeepsThePlainGreedyTokens() throws IOException {
        keepsThePlainGreedyTokens(mistralTarget(), mistralDraft(), base().build());
    }

    @Test
    public void theFourDraftersKeepThePlainGreedyTokensOnOpenCL() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        EngineConfig gpu = base().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0").build();
        keepsThePlainGreedyTokens(cohereTarget(), cohereDraft(), gpu);
        keepsThePlainGreedyTokens(llamaTarget(), miniCpmDraft(), gpu);
        keepsThePlainGreedyTokens(llamaTarget(), llama4Draft(), gpu);
        keepsThePlainGreedyTokens(mistralTarget(), mistralDraft(), gpu);
    }

    /**
     * The Cohere draft as a Hugging Face checkpoint directory: the {@code EagleCohereForCausalLM} routing, the
     * draft's keys, its tensor renames and the target tokenizer all run at load, and the file drafts under
     * {@code --spec-type auto}.
     */
    @Test
    public void aHuggingFaceCohereDraftConvertsAndLoads() throws IOException {
        keepsThePlainGreedyTokens(cohereTarget(), hfCohereDraft(), base().build());
    }

    /** The Mistral Large 3 draft's MLA projections, its fused {@code eagle_linear} and its scaling block. */
    @Test
    public void aHuggingFaceMistralLarge3DraftConvertsAndLoads() throws IOException {
        keepsThePlainGreedyTokens(mistralTarget(), hfMistralDraft(), base().build());
    }

    /** The Llama 4 draft's chunked attention, its {@code no_rope_layers} and its own embeddings and head. */
    @Test
    public void aHuggingFaceLlama4DraftConvertsAndLoads() throws IOException {
        keepsThePlainGreedyTokens(llamaTarget(), hfLlama4Draft(), base().build());
    }

    /** The MiniCPM draft's {@code scale_emb}, {@code scale_depth}, {@code dim_model_base} and input norms. */
    @Test
    public void aHuggingFaceMiniCpmDraftConvertsAndLoads() throws IOException {
        keepsThePlainGreedyTokens(llamaTarget(), hfMiniCpmDraft(), base().build());
    }

    // ------------------------------------------------------------------ fixtures

    private static SyntheticModels.Config targetDims() {
        return SyntheticModels.Config.llama().toBuilder().nLayer(TARGET_LAYERS).build();
    }

    private static SyntheticModels.Config mistralTargetDims() {
        return SyntheticModels.Config.mistralLarge4().toBuilder().nLayer(TARGET_LAYERS).build();
    }

    private static SyntheticModels.Config draftDims() {
        return SyntheticModels.Config.llama().toBuilder().nLayer(1).seed(44L).build();
    }

    private static SyntheticModels.Config mistralDraftDims() {
        return SyntheticModels.Config.mistralLarge4().toBuilder().nLayer(1).seed(45L).build();
    }

    private static EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory);
    }

    private static Path cohereTarget() throws IOException {
        return SyntheticModels.llama(targetDims()).writeTempFile();
    }

    private static Path llamaTarget() throws IOException {
        return SyntheticModels.llama(targetDims()).writeTempFile();
    }

    private static Path mistralTarget() throws IOException {
        return SyntheticModels.deepseek2(mistralTargetDims()).writeTempFile();
    }

    private static Path cohereDraft() throws IOException {
        return SyntheticEagle.eagleFcCohere(draftDims(), TAP, 0.5f).writeTempFile();
    }

    private static Path miniCpmDraft() throws IOException {
        return SyntheticEagle.eagleFcMiniCpm(draftDims(), TAP, 12f, 0.7f, 0.5f).writeTempFile();
    }

    private static Path llama4Draft() throws IOException {
        return SyntheticEagle.eagleFcLlama4(draftDims(), TAP, true, true).writeTempFile();
    }

    private static Path mistralDraft() throws IOException {
        return SyntheticEagle.eagleFcMistralLarge3(mistralDraftDims(), TAP, 32, 32).writeTempFile();
    }

    // ------------------------------------------------------------------ the run

    private void keepsThePlainGreedyTokens(Path target, Path draft, EngineConfig cfg) throws IOException {
        String targetPath = target.toString();
        String draftPath = draft.toString();
        try (EngineModel m = LLMModelFactory.open(targetPath, cfg)) {
            int[] plain = m.generate(PROMPT, greedy(GREEDY_TOKENS), null).tokens();
            m.resetKVCache();

            try (SpeculativeSession s = LLMModelFactory.openSpeculative(targetPath, draftPath, cfg)) {
                GenerationResult r = s.generate(PROMPT, greedy(GREEDY_TOKENS), null);
                assertArrayEquals("the concat-fc drafter keeps the target's greedy tokens", plain, r.tokens());
                SpeculativeDecoder.Stats stats = s.lastStats();
                assertTrue("the drafter made verify rounds: " + stats, stats.rounds() > 0);
                assertTrue("the drafter proposed tokens: " + stats, stats.drafted() > 0);
            }
        }
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    // ------------------------------------------------------------------ the Hugging Face draft

    /** A synthetic {@code EagleCohereForCausalLM} checkpoint: config.json and the draft's safetensors. */
    private Path hfCohereDraft() throws IOException {
        String config = "{\"architectures\": [\"EagleCohereForCausalLM\"], \"hidden_size\": 64,"
                + " \"intermediate_size\": 128, \"num_attention_heads\": 4, \"num_key_value_heads\": 2,"
                + " \"num_hidden_layers\": 1, \"vocab_size\": 320, \"max_position_embeddings\": 64,"
                + " \"layer_norm_eps\": 1e-05, \"head_dim\": 16, \"logit_scale\": 0.5}";
        return hfDraft("cohere-draft", config, w -> {
            add(w, "model.fc.weight", N_EMBD, 2 * N_EMBD);
            add(w, "model.fc.bias", N_EMBD);
            add(w, "model.norm.weight", N_EMBD);
            add(w, "model.layers.0.input_layernorm.weight", N_EMBD);
            add(w, "model.layers.0.self_attn.q_proj.weight", N_HEAD * HEAD_DIM, N_EMBD);
            add(w, "model.layers.0.self_attn.k_proj.weight", N_HEAD_KV * HEAD_DIM, N_EMBD);
            add(w, "model.layers.0.self_attn.v_proj.weight", N_HEAD_KV * HEAD_DIM, N_EMBD);
            add(w, "model.layers.0.self_attn.o_proj.weight", N_EMBD, N_HEAD * HEAD_DIM);
            add(w, "model.layers.0.mlp.gate_proj.weight", N_FF, N_EMBD);
            add(w, "model.layers.0.mlp.up_proj.weight", N_FF, N_EMBD);
            add(w, "model.layers.0.mlp.down_proj.weight", N_EMBD, N_FF);
        });
    }

    /** A synthetic {@code EagleMistralLarge3ForCausalLM} checkpoint in the Mistral Large 3 tensor layout. */
    private Path hfMistralDraft() throws IOException {
        String config = "{\"architectures\": [\"EagleMistralLarge3ForCausalLM\"], \"hidden_size\": 64,"
                + " \"intermediate_size\": 128, \"num_attention_heads\": 4, \"num_hidden_layers\": 1,"
                + " \"vocab_size\": 320, \"max_position_embeddings\": 64, \"rms_norm_eps\": 1e-05,"
                + " \"kv_lora_rank\": 32, \"q_lora_rank\": 32, \"qk_nope_head_dim\": 16,"
                + " \"qk_rope_head_dim\": 16, \"v_head_dim\": 32,"
                + " \"llama_4_scaling\": {\"beta\": 0.1, \"original_max_position_embeddings\": 8192}}";
        int qLora = 32;
        int kvLora = 32;
        int qWidth = N_HEAD * MLA_HEAD_DIM;
        int kvBWidth = (MLA_HEAD_DIM - MLA_ROT + MLA_HEAD_DIM) * N_HEAD;
        return hfDraft("mistral-draft", config, w -> {
            add(w, "eagle_linear.weight", N_EMBD, 2 * N_EMBD);
            add(w, "layers.0.input_layernorm.weight", N_EMBD);
            add(w, "layers.0.self_attn.q_a_proj.weight", qLora, N_EMBD);
            add(w, "layers.0.self_attn.q_a_layernorm.weight", qLora);
            add(w, "layers.0.self_attn.q_b_proj.weight", qWidth, qLora);
            add(w, "layers.0.self_attn.kv_a_proj_with_mqa.weight", kvLora + MLA_ROT, N_EMBD);
            add(w, "layers.0.self_attn.kv_a_layernorm.weight", kvLora);
            add(w, "layers.0.self_attn.kv_b_proj.weight", kvBWidth, kvLora);
            add(w, "layers.0.self_attn.o_proj.weight", N_EMBD, qWidth);
            add(w, "layers.0.post_attention_layernorm.weight", N_EMBD);
            add(w, "layers.0.mlp.gate_proj.weight", N_FF, N_EMBD);
            add(w, "layers.0.mlp.up_proj.weight", N_FF, N_EMBD);
            add(w, "layers.0.mlp.down_proj.weight", N_EMBD, N_FF);
            add(w, "norm.weight", N_EMBD);
        });
    }

    /** A synthetic {@code EagleLlama4ForCausalLM} checkpoint with its own embeddings, fusion and head. */
    private Path hfLlama4Draft() throws IOException {
        String config = "{\"architectures\": [\"EagleLlama4ForCausalLM\"], \"hidden_size\": 64,"
                + " \"intermediate_size\": 128, \"num_attention_heads\": 4, \"num_key_value_heads\": 2,"
                + " \"num_hidden_layers\": 1, \"vocab_size\": 320, \"max_position_embeddings\": 8192,"
                + " \"rms_norm_eps\": 1e-05, \"head_dim\": 16, \"attention_chunk_size\": 8192,"
                + " \"no_rope_layers\": [1], \"use_qk_norm\": true, \"interleave_moe_layer_step\": 4}";
        return hfDraft("llama4-draft", config, w -> {
            add(w, "model.embed_tokens.weight", VOCAB, N_EMBD);
            add(w, "model.fc.weight", N_EMBD, 2 * N_EMBD);
            add(w, "model.norm.weight", N_EMBD);
            add(w, "lm_head.weight", VOCAB, N_EMBD);
            add(w, "model.layers.0.input_layernorm.weight", N_EMBD);
            add(w, "model.layers.0.self_attn.q_proj.weight", N_HEAD * HEAD_DIM, N_EMBD);
            add(w, "model.layers.0.self_attn.k_proj.weight", N_HEAD_KV * HEAD_DIM, N_EMBD);
            add(w, "model.layers.0.self_attn.v_proj.weight", N_HEAD_KV * HEAD_DIM, N_EMBD);
            add(w, "model.layers.0.self_attn.o_proj.weight", N_EMBD, N_HEAD * HEAD_DIM);
            add(w, "model.layers.0.post_attention_layernorm.weight", N_EMBD);
            add(w, "model.layers.0.mlp.gate_proj.weight", N_FF, N_EMBD);
            add(w, "model.layers.0.mlp.up_proj.weight", N_FF, N_EMBD);
            add(w, "model.layers.0.mlp.down_proj.weight", N_EMBD, N_FF);
        });
    }

    /** A synthetic {@code EagleMiniCPMForCausalLM} checkpoint with MiniCPM's scales and input norms. */
    private Path hfMiniCpmDraft() throws IOException {
        String config = "{\"architectures\": [\"EagleMiniCPMForCausalLM\"], \"hidden_size\": 64,"
                + " \"intermediate_size\": 128, \"num_attention_heads\": 4, \"num_key_value_heads\": 2,"
                + " \"num_hidden_layers\": 1, \"vocab_size\": 320, \"max_position_embeddings\": 64,"
                + " \"rms_norm_eps\": 1e-05, \"head_dim\": 16, \"hidden_act\": \"silu\","
                + " \"scale_emb\": 12.0, \"scale_depth\": 1.4, \"mup_denominator\": 4,"
                + " \"dim_model_base\": 32}";
        return hfDraft("minicpm-draft", config, w -> {
            add(w, "model.embed_tokens.weight", VOCAB, N_EMBD);
            add(w, "model.fc.weight", N_EMBD, 2 * N_EMBD);
            add(w, "model.input_norm1.weight", N_EMBD);
            add(w, "model.input_norm2.weight", N_EMBD);
            add(w, "lm_head.weight", VOCAB, N_EMBD);
            add(w, "model.eagle_layers.0.input_layernorm.weight", N_EMBD);
            add(w, "model.eagle_layers.0.self_attn.q_proj.weight", N_HEAD * HEAD_DIM, N_EMBD);
            add(w, "model.eagle_layers.0.self_attn.k_proj.weight", N_HEAD_KV * HEAD_DIM, N_EMBD);
            add(w, "model.eagle_layers.0.self_attn.v_proj.weight", N_HEAD_KV * HEAD_DIM, N_EMBD);
            add(w, "model.eagle_layers.0.self_attn.o_proj.weight", N_EMBD, N_HEAD * HEAD_DIM);
            add(w, "model.eagle_layers.0.post_attention_layernorm.weight", N_EMBD);
            add(w, "model.eagle_layers.0.mlp.gate_proj.weight", N_FF, N_EMBD);
            add(w, "model.eagle_layers.0.mlp.up_proj.weight", N_FF, N_EMBD);
            add(w, "model.eagle_layers.0.mlp.down_proj.weight", N_EMBD, N_FF);
        });
    }

    /** Writes {@code config.json} and one {@code model.safetensors} into a fresh directory. */
    private Path hfDraft(String name, String config, java.util.function.Consumer<SafetensorsWriter.Builder> tensors)
            throws IOException {
        Path dir = tmp.newFolder(name).toPath();
        Files.write(dir.resolve("config.json"), config.getBytes(StandardCharsets.UTF_8));
        SafetensorsWriter.Builder w = SafetensorsWriter.builder();
        tensors.accept(w);
        w.build().write(dir.resolve("model.safetensors"));
        return dir;
    }

    /** Adds one F32 tensor of {@code rows x cols} ({@code cols} only when {@code cols > 0}). */
    private static void add(SafetensorsWriter.Builder w, String name, int rows, int... cols) {
        int width = cols.length == 0 ? 1 : cols[0];
        long count = (long) rows * width;
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(count * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            b.putFloat(0.01f * ((i % 17) - 8));
        }
        long[] shape = cols.length == 0 ? new long[]{rows} : new long[]{rows, width};
        w.add(name, StDType.F32, shape, b.array());
    }
}
