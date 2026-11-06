package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The lookahead decoding settings: the llama.cpp example's defaults, property parsing and their bounds. */
public class LookaheadOptionsTest {

    @Test
    public void defaultsFollowTheLookaheadExample() {
        LookaheadOptions options = LookaheadOptions.defaults();
        assertTrue(options.enabled());
        assertEquals(LookaheadOptions.DEFAULT_WINDOW, options.window());
        assertEquals(LookaheadOptions.DEFAULT_NGRAM_SIZE, options.ngramSize());
        assertEquals(LookaheadOptions.DEFAULT_MAX_NGRAMS, options.maxNgrams());
        assertEquals(15 + 15 + 1, options.sequenceCount());
    }

    @Test
    public void disabledIsOffAndBuildsNoSequence() {
        LookaheadOptions options = LookaheadOptions.disabled();
        assertFalse(options.enabled());
        assertFalse(LookaheadOptions.builder().build().enabled());
    }

    @Test
    public void propertiesSelectTheSizes() {
        Properties p = new Properties();
        p.setProperty(LookaheadOptions.PROP_ENABLED, "true");
        p.setProperty(LookaheadOptions.PROP_WINDOW, "6");
        p.setProperty(LookaheadOptions.PROP_NGRAM_SIZE, "4");
        p.setProperty(LookaheadOptions.PROP_MAX_NGRAMS, "3");
        LookaheadOptions options = LookaheadOptions.fromProperties(p);
        assertTrue(options.enabled());
        assertEquals(6, options.window());
        assertEquals(4, options.ngramSize());
        assertEquals(3, options.maxNgrams());
        assertEquals(10, options.sequenceCount());
    }

    @Test
    public void enabledAloneKeepsTheDefaults() {
        Properties p = new Properties();
        p.setProperty(LookaheadOptions.PROP_ENABLED, "true");
        assertEquals(LookaheadOptions.defaults(), LookaheadOptions.fromProperties(p));
    }

    @Test
    public void absentPropertiesLeaveDecodingOff() {
        assertFalse(LookaheadOptions.fromProperties(new Properties()).enabled());
        Properties off = new Properties();
        off.setProperty(LookaheadOptions.PROP_ENABLED, "false");
        off.setProperty(LookaheadOptions.PROP_WINDOW, "8");
        LookaheadOptions options = LookaheadOptions.fromProperties(off);
        assertFalse(options.enabled());
        assertEquals(8, options.window());
    }

    @Test
    public void positionSpanCoversTheDeepestRow() {
        // window 4, n-gram 3: level 1 row i = 3 sits at n_past + 1 + 3, the last of the step
        assertEquals(4 + 3 - 2, LookaheadOptions.builder().enabled(true).window(4).ngramSize(3).maxNgrams(2)
                .build().positionSpan());
        // a window of one: the n-gram rows run deeper than the single Jacobi level
        assertEquals(5, LookaheadOptions.builder().enabled(true).window(1).ngramSize(5).maxNgrams(1)
                .build().positionSpan());
    }

    @Test
    public void stepRowsCountEveryWrittenRow() {
        LookaheadOptions o = LookaheadOptions.builder().enabled(true).window(4).ngramSize(3).maxNgrams(2).build();
        // 7 input rows + 2 n-grams x 2 tokens + 4*3/2 first-level rows + 4 later-level rows
        assertEquals(7 + 4 + 6 + 4, o.stepRows());
    }

    @Test
    public void outOfRangeSizesRefusedByPropertyName() {
        IllegalArgumentException window = assertThrows(IllegalArgumentException.class,
                () -> LookaheadOptions.builder().window(0).build());
        assertTrue(window.getMessage(), window.getMessage().contains(LookaheadOptions.PROP_WINDOW));
        IllegalArgumentException ngram = assertThrows(IllegalArgumentException.class,
                () -> LookaheadOptions.builder().ngramSize(2).build());
        assertTrue(ngram.getMessage(), ngram.getMessage().contains(LookaheadOptions.PROP_NGRAM_SIZE));
        IllegalArgumentException max = assertThrows(IllegalArgumentException.class,
                () -> LookaheadOptions.builder().maxNgrams(0).build());
        assertTrue(max.getMessage(), max.getMessage().contains(LookaheadOptions.PROP_MAX_NGRAMS));
    }

    @Test
    public void tooManySequencesRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LookaheadOptions.builder().enabled(true).window(LookaheadOptions.MAX_WINDOW)
                        .maxNgrams(LookaheadOptions.MAX_NGRAMS).build());
        assertTrue(e.getMessage(), e.getMessage().contains(LookaheadOptions.PROP_WINDOW));
        assertTrue(e.getMessage(), e.getMessage().contains("sequences"));
    }

    @Test
    public void invalidPropertyValuesNameTheirKey() {
        Properties p = new Properties();
        p.setProperty(LookaheadOptions.PROP_ENABLED, "maybe");
        IllegalArgumentException enabled = assertThrows(IllegalArgumentException.class,
                () -> LookaheadOptions.fromProperties(p));
        assertTrue(enabled.getMessage(), enabled.getMessage().contains(LookaheadOptions.PROP_ENABLED));
        Properties bad = new Properties();
        bad.setProperty(LookaheadOptions.PROP_WINDOW, "wide");
        assertThrows(IllegalArgumentException.class, () -> LookaheadOptions.fromProperties(bad));
    }

    @Test
    public void toBuilderKeepsEverySetting() {
        LookaheadOptions options = LookaheadOptions.builder().enabled(true).window(7).ngramSize(6).maxNgrams(5)
                .build();
        assertEquals(options, options.toBuilder().build());
        assertEquals(options.hashCode(), options.toBuilder().build().hashCode());
        assertNotEquals(options, options.toBuilder().window(8).build());
        assertNotEquals(options, LookaheadOptions.disabled());
    }
}
