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
 * Speculative decoding with a DeepSeek-MTP-shaped EAGLE-1 drafter over a DeepSeek MLA target (vLLM's
 * {@code deepseek_eagle.EagleDeepseekV3ForCausalLM}, the layout of {@link SyntheticEagle#eagleMla}): the fusion
 * normalizes both halves before {@code fc}, the draft's single decoder layer runs multi-head latent attention over
 * its own latent caches, and the drafter keeps the target's plain greedy tokens (verification is lossless). The
 * drafter runs on the CPU and on a GPU target's device.
 */
public class EagleMlaSpeculationTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26, 3, 64, 8, 51};
    private static final int GREEDY_TOKENS = 12;
    private static final int TARGET_LAYER = 0;
    private static final int Q_LORA = 32;
    private static final int KV_LORA = 32;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A one-layer DeepSeek target over the MLA fixture dimensions: the EAGLE-1 drafter fuses its last layer. */
    private Path target() throws IOException {
        return SyntheticModels.deepseek2(SyntheticModels.Config.mistralLarge4().toBuilder().nLayer(1).build())
                .writeTempFile();
    }

    private static SyntheticModels.Config drafterDims() {
        return SyntheticModels.Config.mistralLarge4().toBuilder().nLayer(1).seed(44L).build();
    }

    private static EngineConfig.Builder backend(EngineConfig.BackendMode mode) {
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
        Path draft = SyntheticEagle.eagleMla(drafterDims(), TARGET_LAYER, Q_LORA, KV_LORA).writeTempFile();
        keepsThePlainGreedyTokens(target, draft.toString(), EngineConfig.BackendMode.CPU);
    }

    @Test
    public void aDirectQueryDraftKeepsThePlainGreedyTokens() throws IOException {
        Path target = target();
        Path draft = SyntheticEagle.eagleMla(drafterDims(), TARGET_LAYER, 0, KV_LORA).writeTempFile();
        keepsThePlainGreedyTokens(target, draft.toString(), EngineConfig.BackendMode.CPU);
    }

    @Test
    public void aRoutedExpertDraftKeepsThePlainGreedyTokens() throws IOException {
        Path target = target();
        Path draft = SyntheticEagle.eagleMla(drafterDims(), TARGET_LAYER, Q_LORA, KV_LORA, true).writeTempFile();
        keepsThePlainGreedyTokens(target, draft.toString(), EngineConfig.BackendMode.CPU);
    }

    /**
     * The same drafter as a Hugging Face checkpoint directory: the converter routes the {@code EagleDeepSeekMTPModel}
     * name by its {@code kv_lora_rank} to the MLA mapping, the DeepSeek MLA metadata, the {@code kv_b_proj} split,
     * the fusion input norms and the target tokenizer all run at load, and the file drafts under {@code --spec-type
     * auto}.
     */
    @Test
    public void aHuggingFaceDraftConvertsAndLoadsUnderAuto() throws IOException {
        keepsThePlainGreedyTokens(target(), hfDraft(false).toString(), EngineConfig.BackendMode.CPU);
    }

    /** The expert checkpoint: the routed experts are stacked into the per-layer tensors at conversion. */
    @Test
    public void aHuggingFaceExpertDraftConvertsAndLoadsUnderAuto() throws IOException {
        keepsThePlainGreedyTokens(target(), hfDraft(true).toString(), EngineConfig.BackendMode.CPU);
    }

    /** The drafter runs on the OpenCL target's device and keeps the same tokens. */
    @Test
    public void theMlaDrafterKeepsTheGreedyTokensOnOpenCl() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path target = target();
        Path draft = SyntheticEagle.eagleMla(drafterDims(), TARGET_LAYER, Q_LORA, KV_LORA).writeTempFile();
        keepsThePlainGreedyTokens(target, draft.toString(), EngineConfig.BackendMode.OPENCL);
    }

    /** The drafter runs on the CUDA target's device and keeps the same tokens. */
    @Test
    public void theMlaDrafterKeepsTheGreedyTokensOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        Path target = target();
        Path draft = SyntheticEagle.eagleMla(drafterDims(), TARGET_LAYER, Q_LORA, KV_LORA).writeTempFile();
        keepsThePlainGreedyTokens(target, draft.toString(), EngineConfig.BackendMode.CUDA);
    }

    private void keepsThePlainGreedyTokens(Path target, String draft, EngineConfig.BackendMode mode)
            throws IOException {
        EngineConfig.Builder plainCfg = backend(mode);
        int[] plain;
        try (EngineModel m = LLMModelFactory.open(target.toString(), plainCfg.build())) {
            plain = m.generate(PROMPT, greedy(GREEDY_TOKENS), null).tokens();
        }
        try (SpeculativeSession s = LLMModelFactory.openSpeculative(target.toString(), draft,
                backend(mode).draftMax(3).build())) {
            GenerationResult r = s.generate(PROMPT, greedy(GREEDY_TOKENS), null);
            assertArrayEquals("MLA eagle greedy over the target (" + mode + ")", plain, r.tokens());
            SpeculativeDecoder.Stats stats = s.lastStats();
            assertTrue("the drafter made verify rounds: " + stats, stats.rounds() > 0);
            assertTrue("the drafter proposed tokens: " + stats, stats.drafted() > 0);
        }
    }

    /**
     * A synthetic {@code EagleDeepSeekMTPModel} checkpoint: config.json, the deepseek tokenizer and the draft's
     * safetensors, named like vLLM's checkpoint (no {@code model.} prefix) over one decoder layer.
     */
    private Path hfDraft(boolean moe) throws IOException {
        Path dir = tmp.newFolder(moe ? "draft-moe" : "draft").toPath();
        int nEmbd = 64;
        int nHead = 4;
        int headDim = 32;
        int nFf = 128;
        int vocab = 320;
        int nExpert = 4;
        int nFfExp = 48;
        String config = "{\"architectures\": [\"EagleDeepSeekMTPModel\"], \"hidden_size\": " + nEmbd + ","
                + " \"intermediate_size\": " + nFf + ", \"num_attention_heads\": " + nHead + ","
                + " \"num_key_value_heads\": " + nHead + ", \"num_hidden_layers\": 1, \"vocab_size\": " + vocab + ","
                + " \"max_position_embeddings\": 64, \"rms_norm_eps\": 1e-06, \"kv_lora_rank\": " + KV_LORA + ","
                + " \"qk_nope_head_dim\": 16, \"qk_rope_head_dim\": 16, \"v_head_dim\": 32,"
                + " \"rope_theta\": 10000, \"tie_word_embeddings\": false"
                + (moe ? ", \"n_routed_experts\": " + nExpert + ", \"num_experts_per_tok\": 2,"
                        + " \"moe_intermediate_size\": " + nFfExp + ", \"n_shared_experts\": 1,"
                        + " \"n_group\": 1, \"topk_group\": 1, \"norm_topk_prob\": true,"
                        + " \"routed_scaling_factor\": 2.0, \"scoring_func\": \"sigmoid\","
                        + " \"first_k_dense_replace\": 0" : "") + "}";
        Files.write(dir.resolve("config.json"), config.getBytes(StandardCharsets.UTF_8));

        SafetensorsWriter.Builder w = SafetensorsWriter.builder();
        add(w, "embed_tokens.weight", vocab, nEmbd);
        add(w, "enorm.weight", nEmbd);
        add(w, "hnorm.weight", nEmbd);
        add(w, "fc.weight", nEmbd, 2 * nEmbd);
        add(w, "norm.weight", nEmbd);
        add(w, "lm_head.weight", vocab, nEmbd);
        add(w, "layers.0.input_layernorm.weight", nEmbd);
        add(w, "layers.0.post_attention_layernorm.weight", nEmbd);
        add(w, "layers.0.self_attn.q_proj.weight", nHead * headDim, nEmbd);
        add(w, "layers.0.self_attn.kv_a_proj_with_mqa.weight", KV_LORA + 16, nEmbd);
        add(w, "layers.0.self_attn.kv_a_layernorm.weight", KV_LORA);
        add(w, "layers.0.self_attn.kv_b_proj.weight", nHead * (16 + headDim), KV_LORA);
        add(w, "layers.0.self_attn.o_proj.weight", nEmbd, nHead * headDim);
        if (moe) {
            add(w, "layers.0.mlp.gate.weight", nExpert, nEmbd);
            add(w, "layers.0.mlp.gate.e_score_correction_bias", nExpert);
            for (int x = 0; x < nExpert; x++) {
                add(w, "layers.0.mlp.experts." + x + ".gate_proj.weight", nFfExp, nEmbd);
                add(w, "layers.0.mlp.experts." + x + ".up_proj.weight", nFfExp, nEmbd);
                add(w, "layers.0.mlp.experts." + x + ".down_proj.weight", nEmbd, nFfExp);
            }
            add(w, "layers.0.mlp.shared_experts.gate_proj.weight", nFfExp, nEmbd);
            add(w, "layers.0.mlp.shared_experts.up_proj.weight", nFfExp, nEmbd);
            add(w, "layers.0.mlp.shared_experts.down_proj.weight", nEmbd, nFfExp);
        } else {
            add(w, "layers.0.mlp.gate_proj.weight", nFf, nEmbd);
            add(w, "layers.0.mlp.up_proj.weight", nFf, nEmbd);
            add(w, "layers.0.mlp.down_proj.weight", nEmbd, nFf);
        }
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
