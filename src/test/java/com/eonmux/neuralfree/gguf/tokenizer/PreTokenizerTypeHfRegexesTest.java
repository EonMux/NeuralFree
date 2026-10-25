package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Behaviour;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Dialect;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Step;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link PreTokenizerType#hfRegexes()}: the tokenizer.json text llama.cpp records beside the
 * sequences it adapted, which split the same text the same way under the Oniguruma dialect.
 */
public class PreTokenizerTypeHfRegexesTest {

    private static final String TEXT = "Hello world's 12345\n\n  x";
    private static final List<String> SAMPLES = Arrays.asList(TEXT, "I'M here, you'LL see!!  ",
            "日本語 123\r\n\r\n  tab\t", "naïve café — x", "a  b   c\n", "don't won't I've");

    private static PreTokenizerRegistry.PreTokenizer hf(PreTokenizerType type) {
        List<Step> steps = new ArrayList<>();
        for (String r : type.hfRegexes()) steps.add(Step.regex(r, Dialect.ONIGURUMA, Behaviour.ISOLATED, false));
        return PreTokenizerRegistry.fromSpec(new PreTokenizerSpec(steps, type.byteEncode(), false, false, false));
    }

    @Test
    public void llama3HfFormIsTheOriginalRegex() {
        assertEquals(Arrays.asList(UnicodeCustomSplits.LLAMA3_ORIGINAL), PreTokenizerType.LLAMA3.hfRegexes());
        assertEquals(PreTokenizerPatterns.LLAMA3_HF, PreTokenizerType.LLAMA3.hfRegexes());
    }

    @Test
    public void qwen2FamilySharesOneHfForm() {
        for (PreTokenizerType t : Arrays.asList(PreTokenizerType.QWEN2, PreTokenizerType.STABLELM2,
                PreTokenizerType.HUNYUAN, PreTokenizerType.SOLAR_OPEN, PreTokenizerType.GROK_2)) {
            assertEquals(t.name(), PreTokenizerPatterns.QWEN2_HF, t.hfRegexes());
        }
        assertTrue(PreTokenizerType.QWEN2.hfRegexes().get(0).startsWith("(?i:'s|'t|'re|'ve|'m|'ll|'d)"));
    }

    @Test
    public void typesWithoutARecordedFormAreEmpty() {
        assertTrue(PreTokenizerType.DEFAULT.hfRegexes().isEmpty());
        assertTrue(PreTokenizerType.GPT2.hfRegexes().isEmpty());
    }

    @Test
    public void llamaBpeSplitsAlikeInBothDialects() {
        PreTokenizerRegistry.PreTokenizer llama = PreTokenizerRegistry.forName("llama-bpe");
        PreTokenizerRegistry.PreTokenizer onig = hf(PreTokenizerType.LLAMA3);
        assertEquals(Arrays.asList("Hello", " world", "'s", " ", "123", "45", "\n\n", " ", " x"), onig.split(TEXT));
        for (String s : SAMPLES) assertEquals(s, llama.split(s), onig.split(s));
    }

    @Test
    public void qwen2SplitsAlikeInBothDialects() {
        PreTokenizerRegistry.PreTokenizer llama = PreTokenizerRegistry.forName("qwen2");
        PreTokenizerRegistry.PreTokenizer onig = hf(PreTokenizerType.QWEN2);
        assertEquals(Arrays.asList("Hello", " world", "'s", " ", "1", "2", "3", "4", "5", "\n\n", " ", " x"),
                onig.split(TEXT));
        for (String s : SAMPLES) assertEquals(s, llama.split(s), onig.split(s));
    }

    @Test
    public void everyHfFormCompiles() {
        for (PreTokenizerType t : PreTokenizerType.values()) {
            if (t.hfRegexes().isEmpty()) continue;
            assertEquals(t.name(), String.join("", hf(t).split(TEXT)), TEXT);
        }
    }
}
