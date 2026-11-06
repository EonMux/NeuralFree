package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Prefix reuse on the single-sequence loop (llama.cpp's {@code --cache-prompt} prompt-cache
 * path): a generation whose prompt extends the stored sequence keeps the common prefix behind a rollback check
 * and prefills only its tail (its {@code timings().promptTokens()} counts the evaluated tokens), computes the
 * same tokens as the full prefill, and {@code neuralfree.prefix.reuse=false} prefills whole.
 */
public class GenerationLoopPrefixReuseTest {

    private static final int N_TOKENS = 12;
    private static final int N_NEW = 6;

    private static EngineConfig config(boolean reuse) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(reuse).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** A deterministic prompt; {@code suffix} varies the tail so two prompts share a prefix. */
    private static int[] prompt(int seed, int suffix) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 11 + i * 7) % 250;
        }
        t[N_TOKENS - 1] = 4 + suffix;
        return t;
    }

    @Test
    public void aRepeatedPromptPrefillsOnlyItsTailAndKeepsTheResult() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            GenerationLoop loop = GenerationLoop.forModel(m, config(true));
            GenerationResult first = loop.generate(prompt(1, 0), request(N_NEW), null);
            GenerationResult second = loop.generate(prompt(1, 0), request(N_NEW), null);
            assertEquals(N_TOKENS, first.timings().promptTokens());
            assertEquals("only the last prompt token evaluates over the kept prefix", 1,
                    second.timings().promptTokens());
            assertArrayEquals("the reused run computes the same tokens", first.tokens(), second.tokens());
            assertEquals(first.text(), second.text());
        }
    }

    @Test
    public void anUnrelatedPromptPrefillsWhole() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            GenerationLoop loop = GenerationLoop.forModel(m, config(true));
            loop.generate(prompt(1, 0), request(N_NEW), null);
            GenerationResult other = loop.generate(prompt(9, 0), request(N_NEW), null);
            assertEquals("no shared prefix keeps nothing", N_TOKENS, other.timings().promptTokens());
        }
    }

    @Test
    public void theReuseSwitchPrefillsWhole() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(false))) {
            GenerationLoop loop = GenerationLoop.forModel(m, config(false));
            loop.generate(prompt(2, 0), request(N_NEW), null);
            GenerationResult second = loop.generate(prompt(2, 0), request(N_NEW), null);
            assertEquals("neuralfree.prefix.reuse=false prefills the whole prompt", N_TOKENS,
                    second.timings().promptTokens());
        }
    }

    /** A longer prompt reuses only the prefix the stored sequence computed, never past its own length. */
    @Test
    public void anExtensionReusesOnlyTheStoredPrefix() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            GenerationLoop loop = GenerationLoop.forModel(m, config(true));
            GenerationResult first = loop.generate(prompt(3, 0), request(N_NEW), null);
            int[] generated = first.tokens();
            int[] stored = new int[N_TOKENS + generated.length];
            System.arraycopy(prompt(3, 0), 0, stored, 0, N_TOKENS);
            System.arraycopy(generated, 0, stored, N_TOKENS, generated.length);
            int[] extended = new int[N_TOKENS + 4];
            System.arraycopy(prompt(3, 0), 0, extended, 0, N_TOKENS);
            for (int i = N_TOKENS; i < extended.length; i++) {
                extended[i] = 4 + (i * 31) % 250;
            }
            int common = commonPrefix(stored, extended);
            int expectedTail = extended.length - Math.min(common, extended.length - 1);
            GenerationResult reference;
            try (GGUFModel fresh = new GGUFModel(model.toString(), config(true))) {
                reference = GenerationLoop.forModel(fresh, config(true))
                        .generate(extended, request(N_NEW), null);
            }
            GenerationResult second = loop.generate(extended, request(N_NEW), null);
            assertEquals("the stored prefix reuses, the tail prefills", expectedTail,
                    second.timings().promptTokens());
            assertTrue("reuse never passes the stored sequence", expectedTail >= 1);
            assertArrayEquals("same tokens as the same prompt on a fresh model", reference.tokens(),
                    second.tokens());
        }
    }

    /** The longest common prefix length of the two arrays. */
    private static int commonPrefix(int[] a, int[] b) {
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                return i;
            }
        }
        return n;
    }
}
