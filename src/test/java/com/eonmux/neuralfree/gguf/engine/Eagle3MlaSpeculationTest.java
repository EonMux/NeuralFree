package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
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
 * Speculative decoding with an EAGLE-3 drafter over a DeepSeek MLA target (vLLM's
 * {@code deepseek_eagle3.Eagle3DeepseekV2ForCausalLM}, the layout of {@link SyntheticEagle#eagle3Mla}): the draft's
 * single decoder layer runs multi-head latent attention over its own latent caches, the DeepSeek MLA metadata resolves
 * its absorbed {@code attn_k_b} / {@code attn_v_b} decompression, and the drafter keeps the target's plain greedy
 * tokens (verification is lossless).
 */
public class Eagle3MlaSpeculationTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26, 3, 64, 8, 51};
    private static final int GREEDY_TOKENS = 12;
    private static final int TARGET_LAYERS = 3;
    private static final int[] TAPS = {0, 1, 2};
    private static final int Q_LORA = 32;
    private static final int KV_LORA = 32;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A three-layer DeepSeek target over the MLA fixture dimensions. */
    private Path target() throws IOException {
        return SyntheticModels.deepseek2(SyntheticModels.Config.mistralLarge4().toBuilder()
                .nLayer(TARGET_LAYERS).build()).writeTempFile();
    }

    private static SyntheticModels.Config drafterDims() {
        return SyntheticModels.Config.mistralLarge4().toBuilder().nLayer(1).seed(44L).build();
    }

    private static EngineConfig.Builder base(EngineConfig.BackendMode mode) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        EngineConfig.Builder b = EngineConfig.builder().backend(mode).memory(memory);
        return mode == EngineConfig.BackendMode.CPU ? b : b.addGpuDevice("0");
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    @Test
    public void theMlaDrafterKeepsThePlainGreedyTokens() throws IOException {
        Path target = target();
        Path draft = SyntheticEagle.eagle3Mla(drafterDims(), TAPS, Q_LORA, KV_LORA).writeTempFile();
        keepsThePlainGreedyTokens(target, draft.toString(), EngineConfig.BackendMode.CPU);
    }

    /** The drafter runs on the OpenCL target's device and keeps the same tokens. */
    @Test
    public void theMlaDrafterKeepsTheGreedyTokensOnOpenCl() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path target = target();
        Path draft = SyntheticEagle.eagle3Mla(drafterDims(), TAPS, Q_LORA, KV_LORA).writeTempFile();
        keepsThePlainGreedyTokens(target, draft.toString(), EngineConfig.BackendMode.OPENCL);
    }

    /** The drafter runs on the CUDA target's device and keeps the same tokens. */
    @Test
    public void theMlaDrafterKeepsTheGreedyTokensOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        Path target = target();
        Path draft = SyntheticEagle.eagle3Mla(drafterDims(), TAPS, Q_LORA, KV_LORA).writeTempFile();
        keepsThePlainGreedyTokens(target, draft.toString(), EngineConfig.BackendMode.CUDA);
    }

    /**
     * The same drafter as a Hugging Face checkpoint directory: the converter's {@code deepseek_eagle3} routing,
     * the EAGLE-3 metadata with the DeepSeek MLA geometry, the {@code kv_b_proj} split and the target tokenizer
     * all run at load, and the file drafts under {@code --spec-type auto}.
     */
    @Test
    public void aHuggingFaceDraftConvertsAndLoadsUnderAuto() throws IOException {
        keepsThePlainGreedyTokens(target(), hfDraft().toString(), EngineConfig.BackendMode.CPU);
    }

    private void keepsThePlainGreedyTokens(Path target, String draft, EngineConfig.BackendMode mode)
            throws IOException {
        try (EngineModel m = LLMModelFactory.open(target.toString(), base(mode).build())) {
            int[] plain = m.generate(PROMPT, greedy(GREEDY_TOKENS), null).tokens();
            m.resetKVCache();

            try (SpeculativeSession s = LLMModelFactory.openSpeculative(target.toString(), draft,
                    base(mode).draftMax(3).build())) {
                GenerationResult r = s.generate(PROMPT, greedy(GREEDY_TOKENS), null);
                assertArrayEquals("MLA eagle3 greedy over the target (" + mode + ")", plain, r.tokens());
                SpeculativeDecoder.Stats stats = s.lastStats();
                assertTrue("the drafter made verify rounds: " + stats, stats.rounds() > 0);
                assertTrue("the drafter proposed tokens: " + stats, stats.drafted() > 0);
            }
        }
    }

    /** A synthetic {@code Eagle3DeepseekV2ForCausalLM} checkpoint: config.json and the draft's safetensors. */
    private Path hfDraft() throws IOException {
        Path dir = tmp.newFolder("draft").toPath();
        String config = "{\"architectures\": [\"Eagle3DeepseekV2ForCausalLM\"], \"hidden_size\": 64,"
                + " \"intermediate_size\": 128, \"num_attention_heads\": 4, \"num_hidden_layers\": 1,"
                + " \"vocab_size\": 320, \"max_position_embeddings\": 64, \"rms_norm_eps\": 1e-06,"
                + " \"draft_vocab_size\": 320, \"q_lora_rank\": 32, \"kv_lora_rank\": 32,"
                + " \"qk_nope_head_dim\": 16, \"qk_rope_head_dim\": 16, \"v_head_dim\": 32,"
                + " \"eagle_aux_hidden_state_layer_ids\": [0, 1, 2], \"target_hidden_size\": 64}";
        Files.write(dir.resolve("config.json"), config.getBytes(StandardCharsets.UTF_8));

        int nEmbd = 64;
        int nHead = 4;
        int headDim = 32;
        int qLora = Q_LORA;
        int kvLora = KV_LORA;
        int nFf = 128;
        int vocab = 320;
        int input = 2 * nEmbd;
        SafetensorsWriter.Builder w = SafetensorsWriter.builder();
        add(w, "embed_tokens.weight", vocab, nEmbd);
        add(w, "fc.weight", nEmbd, 3 * nEmbd);
        add(w, "norm.weight", nEmbd);
        add(w, "lm_head.weight", vocab, nEmbd);
        add(w, "midlayer.input_layernorm.weight", nEmbd);
        add(w, "midlayer.post_attention_layernorm.weight", nEmbd);
        add(w, "midlayer.hidden_norm.weight", nEmbd);
        add(w, "midlayer.self_attn.q_a_proj.weight", qLora, input);
        add(w, "midlayer.self_attn.q_a_layernorm.weight", qLora);
        add(w, "midlayer.self_attn.q_b_proj.weight", nHead * headDim, qLora);
        add(w, "midlayer.self_attn.kv_a_proj_with_mqa.weight", kvLora + 16, input);
        add(w, "midlayer.self_attn.kv_a_layernorm.weight", kvLora);
        add(w, "midlayer.self_attn.kv_b_proj.weight", nHead * (16 + headDim), kvLora);
        add(w, "midlayer.self_attn.o_proj.weight", nEmbd, nHead * headDim);
        add(w, "midlayer.mlp.gate_proj.weight", nFf, nEmbd);
        add(w, "midlayer.mlp.up_proj.weight", nFf, nEmbd);
        add(w, "midlayer.mlp.down_proj.weight", nEmbd, nFf);
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
