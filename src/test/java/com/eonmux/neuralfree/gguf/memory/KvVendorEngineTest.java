package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The vendor KV formats end to end over a real synthetic llama model: generation runs with each vLLM cache type, and
 * a session file saved with a vendor cache loads into a fresh model and continues exactly, so the packed rows and
 * their side planes travel through the host state transfer.
 */
public class KvVendorEngineTest {

    private static final int PROMPT = 10;
    private static final int NEW_TOKENS = 4;

    private static final List<KvCacheTypes> FORMATS = Arrays.asList(
            KvCacheTypes.of(KvDType.NVFP4),
            KvCacheTypes.of(KvDType.NVFP4_4OVER6),
            KvCacheTypes.of(KvDType.TQ_K8_KEYS, KvDType.TQ_4BIT_VALUES),
            KvCacheTypes.of(KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES),
            KvCacheTypes.of(KvDType.TQ_3BIT_KEYS, KvDType.TQ_4BIT_VALUES),
            KvCacheTypes.of(KvDType.TQ_3BIT_KEYS, KvDType.TQ_3BIT_VALUES));

    private static EngineConfig config(KvCacheTypes types) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(types.typeK()).typeV(types.typeV())
                .prefixReuse(true).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(NEW_TOKENS).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] prompt() {
        int[] t = new int[PROMPT];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (i * 11 + 5) % 250;
        }
        return t;
    }

    @Test
    public void generationRunsWithEveryVendorFormat() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try {
            for (KvCacheTypes types : FORMATS) {
                try (GGUFModel m = new GGUFModel(model.toString(), config(types))) {
                    GenerationResult r = m.generate(prompt(), request(), null);
                    assertEquals(types + " generated the requested tokens", NEW_TOKENS, r.tokens().length);
                }
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void aVendorSessionRestoresItsRowsAndPlanes() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path session = Files.createTempDirectory("nf-vendor-session-").resolve("vendor.session");
        try {
            for (KvCacheTypes types : FORMATS) {
                int[] expected;
                try (GGUFModel m = new GGUFModel(model.toString(), config(types))) {
                    expected = m.generate(prompt(), request(), null).tokens();
                    m.saveSession(session);
                    assertTrue(types + " wrote a session", Files.size(session) > 0);
                }
                try (GGUFModel restored = new GGUFModel(model.toString(), config(types))) {
                    restored.loadSession(session);
                    GenerationResult second = restored.generate(prompt(), request(), null);
                    assertEquals(types + " restored the whole prompt", 1, second.timings().promptTokens());
                    assertArrayEquals(types + " decodes like the original run", expected, second.tokens());
                }
            }
        } finally {
            Files.deleteIfExists(session);
            Files.deleteIfExists(model);
        }
    }
}
