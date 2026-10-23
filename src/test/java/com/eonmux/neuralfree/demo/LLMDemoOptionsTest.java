package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link LLMDemoOptions}: sampler settings start from the {@code -Dneuralfree.*} properties with the demo defaults
 * filling unset ones, flags override both, the llama.cpp sampling flags reach the generation request, and no short
 * alias contradicts llama.cpp's meaning.
 */
public class LLMDemoOptionsTest {

    private static final String[] SAMPLER_PROPS = {
            SamplerConfig.PROP_TEMPERATURE, SamplerConfig.PROP_TOP_K, SamplerConfig.PROP_TOP_P,
            SamplerConfig.PROP_MIN_P, SamplerConfig.PROP_TYPICAL_P, SamplerConfig.PROP_REPEAT_PENALTY,
            SamplerConfig.PROP_PRESENCE_PENALTY, SamplerConfig.PROP_FREQUENCY_PENALTY,
            SamplerConfig.PROP_PENALTY_LAST_N, SamplerConfig.PROP_SEED, SamplerConfig.PROP_GREEDY
    };

    private final Map<String, String> saved = new HashMap<>();

    @Before
    public void clearSamplerProperties() {
        for (String key : SAMPLER_PROPS) {
            saved.put(key, System.getProperty(key));
            System.clearProperty(key);
        }
    }

    @After
    public void restoreSamplerProperties() {
        for (Map.Entry<String, String> e : saved.entrySet()) {
            if (e.getValue() == null) {
                System.clearProperty(e.getKey());
            } else {
                System.setProperty(e.getKey(), e.getValue());
            }
        }
    }

    private static LLMDemoOptions parse(String... args) throws UsageException {
        return LLMDemoOptions.parse(args);
    }

    @Test
    public void demoDefaultsApplyWithoutProperties() throws UsageException {
        GenerationRequest r = parse().generation;
        SamplerConfig s = r.sampler();
        assertEquals(LLMDemoOptions.DEFAULT_TEMPERATURE, s.temperature(), 0f);
        assertEquals(LLMDemoOptions.DEFAULT_TOP_K, s.topK());
        assertEquals(LLMDemoOptions.DEFAULT_TOP_P, s.topP(), 0f);
        assertEquals(LLMDemoOptions.DEFAULT_MIN_P, s.minP(), 0f);
        assertEquals(LLMDemoOptions.DEFAULT_REPEAT_WINDOW, s.penaltyLastN());
        assertEquals(1.0f, s.typicalP(), 0f);
        assertEquals(LLMDemoOptions.DEFAULT_MAX_TOKENS, r.maxNewTokens());
        assertTrue(r.stopStrings().isEmpty());
        assertFalse(r.ignoreEos());
    }

    @Test
    public void samplerPropertiesReplaceDemoDefaults() throws UsageException {
        System.setProperty(SamplerConfig.PROP_TEMPERATURE, "0.3");
        System.setProperty(SamplerConfig.PROP_TYPICAL_P, "0.9");
        System.setProperty(SamplerConfig.PROP_PRESENCE_PENALTY, "0.4");
        System.setProperty(SamplerConfig.PROP_FREQUENCY_PENALTY, "0.2");
        System.setProperty(SamplerConfig.PROP_PENALTY_LAST_N, "32");
        System.setProperty(SamplerConfig.PROP_GREEDY, "true");
        SamplerConfig s = parse().generation.sampler();
        assertEquals(0.3f, s.temperature(), 0f);
        assertEquals(0.9f, s.typicalP(), 0f);
        assertEquals(0.4f, s.presencePenalty(), 0f);
        assertEquals(0.2f, s.frequencyPenalty(), 0f);
        assertEquals(32, s.penaltyLastN());
        assertTrue(s.greedy());
        assertEquals(LLMDemoOptions.DEFAULT_TOP_K, s.topK());
    }

    @Test
    public void flagsOverrideProperties() throws UsageException {
        System.setProperty(SamplerConfig.PROP_TEMPERATURE, "0.3");
        SamplerConfig s = parse("--temp", "0.9").generation.sampler();
        assertEquals(0.9f, s.temperature(), 0f);
    }

    @Test
    public void llamaCppSamplingFlagsReachTheRequest() throws UsageException {
        GenerationRequest r = parse("--typical", "0.5", "--presence-penalty", "0.25", "--frequency-penalty", "-0.5",
                "--ignore-eos", "-r", "User:", "--reverse-prompt", "###", "-n", "64").generation;
        SamplerConfig s = r.sampler();
        assertEquals(0.5f, s.typicalP(), 0f);
        assertEquals(0.25f, s.presencePenalty(), 0f);
        assertEquals(-0.5f, s.frequencyPenalty(), 0f);
        assertTrue(r.ignoreEos());
        assertEquals(Arrays.asList("User:", "###"), r.stopStrings());
        assertEquals(64, r.maxNewTokens());
        assertEquals(0.75f, parse("--typical-p", "0.75").generation.sampler().typicalP(), 0f);
    }

    @Test
    public void invalidSamplingValuesAreRejected() {
        assertThrows(UsageException.class, () -> parse("--typical", "1.5"));
        assertThrows(UsageException.class, () -> parse("--presence-penalty", "x"));
        assertThrows(UsageException.class, () -> parse("--reverse-prompt", ""));
        assertThrows(UsageException.class, () -> parse("-r"));
    }

    @Test
    public void shortAliasesWithOtherLlamaCppMeaningsAreUnknown() throws UsageException {
        UsageException t = assertThrows(UsageException.class, () -> parse("-t", "8"));
        assertTrue(t.getMessage(), t.getMessage().contains("unknown option -t"));
        UsageException b = assertThrows(UsageException.class, () -> parse("-b", "512"));
        assertTrue(b.getMessage(), b.getMessage().contains("unknown option -b"));
        assertEquals(8, parse("-n", "8").generation.maxNewTokens());
        assertEquals(16, parse("--max-tokens", "16").generation.maxNewTokens());
    }

    @Test
    public void modelPathAndStatsFileAreUnsetUnlessGiven() throws UsageException {
        LLMDemoOptions none = parse();
        assertNull(none.modelPath);
        assertNull(none.statsFile);
        assertEquals("model.gguf", parse("model.gguf").modelPath);
        assertEquals("model.gguf", parse("--model", "model.gguf").modelPath);
        assertEquals("stats.txt", parse("--stats-file", "stats.txt").statsFile);
    }

    @Test
    public void usageListsTheNewFlagsAndNoContradictingAliases() {
        String usage = String.join("\n", LLMDemoOptions.usage());
        assertTrue(usage.contains("--typical"));
        assertTrue(usage.contains("--presence-penalty"));
        assertTrue(usage.contains("--frequency-penalty"));
        assertTrue(usage.contains("--ignore-eos"));
        assertTrue(usage.contains("--reverse-prompt, -r"));
        assertFalse(usage.contains(", -t,"));
        assertFalse(usage.contains("--backend, -b"));
    }
}
