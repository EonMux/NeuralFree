package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LookaheadOptions;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The lookahead decoding flags ({@code --lookahead}, {@code --no-lookahead} and its window, n-gram size and n-gram
 * count): they reach the engine configuration, size flags turn decoding on, and the configurations the target memory
 * cannot serve are usage errors naming the setting.
 */
public class LLMDemoLookaheadTest {

    private static EngineConfig engine(String... args) throws UsageException {
        return LLMDemoOptions.parse(args).engine;
    }

    @Test
    public void lookaheadSelectsTheDefaultSizes() throws UsageException {
        LookaheadOptions options = engine("-np", "31", "--lookahead").lookaheadOptions();
        assertTrue(options.enabled());
        assertEquals(LookaheadOptions.DEFAULT_WINDOW, options.window());
        assertEquals(LookaheadOptions.DEFAULT_NGRAM_SIZE, options.ngramSize());
        assertEquals(LookaheadOptions.DEFAULT_MAX_NGRAMS, options.maxNgrams());
        assertEquals(31, options.sequenceCount());
    }

    @Test
    public void sizeFlagsSetTheirValueAndTurnDecodingOn() throws UsageException {
        LookaheadOptions options = engine("-np", "10", "--lookahead-window", "6",
                "--lookahead-ngram-size", "4", "--lookahead-max-ngrams", "3").lookaheadOptions();
        assertTrue(options.enabled());
        assertEquals(6, options.window());
        assertEquals(4, options.ngramSize());
        assertEquals(3, options.maxNgrams());
    }

    @Test
    public void noLookaheadTurnsItOff() throws UsageException {
        LookaheadOptions options = engine("-np", "31", "--lookahead", "--no-lookahead").lookaheadOptions();
        assertFalse(options.enabled());
    }

    @Test
    public void lookaheadAndSpeculativeDecodingExcluded() {
        UsageException e = assertThrows(UsageException.class,
                () -> engine("-np", "31", "--lookahead", "--spec-type", "ngram-simple"));
        assertTrue(e.getMessage(), e.getMessage().contains(LookaheadOptions.PROP_ENABLED));
    }

    @Test
    public void lookaheadNeedsAWindowedPagedMemory() {
        UsageException e = assertThrows(UsageException.class, () -> engine("--lookahead"));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.parallel"));
        UsageException tooFew = assertThrows(UsageException.class, () -> engine("-np", "4", "--lookahead"));
        assertTrue(tooFew.getMessage(), tooFew.getMessage().contains("31"));
    }

    @Test
    public void promptCacheRefusedWithLookahead() {
        UsageException e = assertThrows(UsageException.class,
                () -> engine("-np", "31", "--lookahead", "--prompt-cache", "state.bin"));
        assertTrue(e.getMessage(), e.getMessage().contains("--prompt-cache"));
    }

    @Test
    public void outOfRangeSizeIsAUsageError() {
        UsageException e = assertThrows(UsageException.class,
                () -> engine("-np", "31", "--lookahead-window", "0"));
        assertTrue(e.getMessage(), e.getMessage().contains("--lookahead-window"));
        UsageException ngram = assertThrows(UsageException.class,
                () -> engine("-np", "31", "--lookahead-ngram-size", "2"));
        assertTrue(ngram.getMessage(), ngram.getMessage().contains("--lookahead-ngram-size"));
    }
}
