package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;

/**
 * The FP8 KV cache end to end over a real synthetic llama model: generation runs with {@code -ctk/-ctv fp8_e4m3},
 * the checkpoint {@code blk.N.attn_k.kv_scale} / {@code blk.N.attn_v.kv_scale} tensors change the FP8 codes (the
 * store divides by them, attention multiplies them back, as vLLM does), and an F16 cache ignores them (vLLM enforces
 * 1.0 for a non-quantized cache).
 */
public class Fp8KvCacheEngineTest {

    /** A synthetic llama whose every layer carries per-layer KV-cache scales of {@code kvScale}. */
    private static GgufTestWriter model(float kvScale) {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        GgufTestWriter w = SyntheticModels.llama(c);
        for (int il = 0; il < c.nLayer(); il++) {
            w.tensorFilled("blk." + il + ".attn_k.kv_scale", GGMLType.F32, kvScale, 1);
            w.tensorFilled("blk." + il + ".attn_v.kv_scale", GGMLType.F32, kvScale, 1);
        }
        return w;
    }

    private static GenerationResult generate(Path file, KvDType type) throws IOException {
        EngineConfig config = EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).typeK(type).typeV(type).build()).build();
        try (GGUFModel model = new GGUFModel(file.toString(), config)) {
            return model.generate(model.tokenizer().encode("hello"), GenerationRequest.builder().maxNewTokens(4)
                    .sampler(SamplerConfig.builder().greedy(true).seed(7).build()).logprobs(1).build(), null);
        }
    }

    private static void delete(Path file) {
        try {
            Files.deleteIfExists(file);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Test
    public void fp8CacheRunsAndConsumesTheCheckpointScales() throws IOException {
        Path scaled = model(16.0f).writeTempFile();
        Path unscaled = model(1.0f).writeTempFile();
        try {
            GenerationResult withScales = generate(scaled, KvDType.FP8_E4M3);
            GenerationResult without = generate(unscaled, KvDType.FP8_E4M3);
            assertEquals(4, withScales.tokens().length);
            // the store divides by the checkpoint scale and attention multiplies it back: a 16x scale puts the FP8
            // codes on a different grid, which shows in the token log-probabilities even when greedy agrees
            float[][] got = logprobs(withScales);
            float[][] want = logprobs(without);
            assertFalse("the checkpoint KV scales did not reach the FP8 cache", Arrays.deepEquals(got, want));
        } finally {
            delete(scaled);
            delete(unscaled);
        }
    }

    @Test
    public void f16CacheIgnoresTheCheckpointScales() throws IOException {
        Path scaled = model(16.0f).writeTempFile();
        Path unscaled = model(1.0f).writeTempFile();
        try {
            // vLLM enforces k/v_scale 1.0 unless the cache dtype is FP8: an F16 cache stores exact values
            assertArrayEquals(generate(unscaled, KvDType.F16).tokens(), generate(scaled, KvDType.F16).tokens());
        } finally {
            delete(scaled);
            delete(unscaled);
        }
    }

    /** The chosen token's log-probability of every generated token. */
    private static float[][] logprobs(GenerationResult r) {
        float[][] out = new float[r.logprobs().size()][];
        for (int i = 0; i < out.length; i++) {
            TokenScore[] row = r.logprobs().get(i);
            out[i] = new float[row.length];
            for (int j = 0; j < row.length; j++) {
                out[i][j] = row[j].logprob();
            }
        }
        return out;
    }
}
