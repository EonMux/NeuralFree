package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.Mode;
import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code --pooling} (llama.cpp's flag) on the demo command line: the override lands in the engine configuration
 * the model opens with, an unknown name refuses naming the flag, and without it the GGUF's pooling runs.
 */
public class LLMDemoPoolingFlagTest {

    @Test
    public void theFlagWritesTheEngineConfigOverride() throws UsageException {
        LLMDemoOptions o = LLMDemoOptions.parse(new String[]{"--pooling", "cls", "-m", "m.gguf"});
        assertEquals(PoolingKind.CLS, o.engine.pooling());
        LLMDemoOptions rank = LLMDemoOptions.parse(new String[]{"--mode", "embed", "--pooling", "RANK"});
        assertEquals(PoolingKind.RANK, rank.engine.pooling());
        assertNull(LLMDemoOptions.parse(new String[]{"-m", "m.gguf"}).engine.pooling());
    }

    @Test
    public void anUnknownNameRefusesNamingTheFlag() {
        UsageException e = assertThrows(UsageException.class,
                () -> LLMDemoOptions.parse(new String[]{"--pooling", "avg"}));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--pooling:"));
        assertTrue(e.getMessage(), e.getMessage().contains("avg"));
    }

    /**
     * AUTO mode reads the head the override resolves: a generative model generates without {@code --pooling} and
     * embeds with it, as llama.cpp's embeddings of any model do; {@code --pooling rank} on it refuses by name.
     */
    @Test
    public void autoModeFollowsTheOverride() throws Exception {
        String model = SyntheticModels.llama().writeTempFile().toString();
        LLMDemoOptions plain = LLMDemoOptions.parse(new String[]{"-m", model, "--cpu"});
        assertEquals(Mode.GENERATE, LLMDemo.autoMode(LLMModelFactory.inspect(model, plain.engine), plain));
        LLMDemoOptions mean = LLMDemoOptions.parse(new String[]{"-m", model, "--cpu", "--pooling", "mean"});
        assertEquals(Mode.EMBED, LLMDemo.autoMode(LLMModelFactory.inspect(model, mean.engine), mean));
        LLMDemoOptions rank = LLMDemoOptions.parse(new String[]{"-m", model, "--cpu", "--pooling", "rank"});
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LLMModelFactory.inspect(model, rank.engine));
        assertTrue(e.getMessage(), e.getMessage().contains("classifier"));
    }

    @Test
    public void theExplicitModeKeepsItsPooledUbatchWithTheOverride() throws UsageException {
        LLMDemoOptions o = LLMDemoOptions.parse(new String[]{"--mode", "embed", "--pooling", "mean",
                "--ctx-size", "128"});
        assertEquals("the override does not change the mode resolution inputs",
                Mode.EMBED, o.mode);
        assertEquals("the pooled modes take the pooled ubatch", Math.max(LLMDemoOptions.POOLED_UBATCH, 128),
                o.engineConfig(Mode.EMBED).memory().ubatch());
        assertEquals(PoolingKind.MEAN, o.engineConfig(Mode.EMBED).pooling());
    }
}
