package com.eonmux.neuralfree.gguf.tokenizer;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Parity tests for {@link PreTokenizerRegistry} against llama.cpp {@code llama_vocab::impl::load}
 * (name mapping and flags) and {@code unicode_regex_split} (split results traced by hand through
 * the custom splitters and the collapsed-regex path).
 */
public class PreTokenizerRegistryTest {

    private static final String TEXT = "Hello world's 12345\n\n  x";

    private static List<String> split(String pre, String text) {
        return PreTokenizerRegistry.forName(pre).split(text);
    }

    @Test
    public void allLlamaCppPreTypesAreDeclaredInOrder() {
        PreTokenizerType[] types = PreTokenizerType.values();
        assertEquals(63, types.length);
        for (int i = 0; i < types.length; i++) assertEquals(i, types[i].code());
        assertEquals(PreTokenizerPatterns.DEFAULT, PreTokenizerType.LLAMA4.regexes());
        assertFalse(PreTokenizerType.GEMMA4.byteEncode());
        assertFalse(PreTokenizerType.WHITESPACE.byteEncode());
        assertTrue(PreTokenizerType.QWEN2.byteEncode());
    }

    @Test
    public void everyNameCompilesAndSplits() {
        assertEquals(94, PreTokenizerRegistry.names().size());
        String sample = "The QUICK brown fox's 1234567 你好 été!\n\n  done\r\n";
        for (String name : PreTokenizerRegistry.names()) {
            List<String> words = PreTokenizerRegistry.forName(name).split(sample);
            assertEquals(name, sample, String.join("", words));
        }
    }

    @Test
    public void unknownNameIsAnError() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> PreTokenizerRegistry.forName("no-such-pre"));
        assertTrue(e.getMessage().contains("unknown pre-tokenizer type: 'no-such-pre'"));
        assertFalse(PreTokenizerRegistry.isKnown("no-such-pre"));
    }

    @Test
    public void missingNameSelectsDefault() {
        PreTokenizerRegistry.PreTokenizer p = PreTokenizerRegistry.forName(null);
        assertEquals(PreTokenizerRegistry.DEFAULT_NAME, p.name());
        assertSame(PreTokenizerType.DEFAULT, p.type());
        assertSame(p, PreTokenizerRegistry.forName(""));
    }

    @Test
    public void flagsFollowTheNameTable() {
        PreTokenizerRegistry.PreTokenizer llama = PreTokenizerRegistry.forName("llama-bpe");
        assertSame(PreTokenizerType.LLAMA3, llama.type());
        assertTrue(llama.ignoreMerges());
        assertEquals(Boolean.TRUE, llama.addBos());
        assertTrue(llama.cleanSpaces());

        PreTokenizerRegistry.PreTokenizer glm = PreTokenizerRegistry.forName("glm5");
        assertSame(PreTokenizerType.CHATGLM4, glm.type());
        assertTrue(glm.suppressBos());
        assertTrue(glm.ignoreMerges());

        PreTokenizerRegistry.PreTokenizer sarvam = PreTokenizerRegistry.forName("sarvam-moe");
        assertTrue(sarvam.escapeWhitespaces());
        assertFalse(sarvam.cleanSpaces());
        assertFalse(sarvam.byteEncode());
        assertFalse(sarvam.wholeNewlineWords());
        assertFalse(sarvam.forceAddBos());

        PreTokenizerRegistry.PreTokenizer mmbert = PreTokenizerRegistry.forName("mmbert");
        assertTrue(mmbert.metaspaceWords());
        assertTrue(mmbert.wholeNewlineWords());

        assertTrue(PreTokenizerRegistry.forName("granite-embed-multi-311m").forceAddBos());
        assertTrue(PreTokenizerRegistry.forName("roberta-bpe").addSep());
        assertEquals(Boolean.FALSE, PreTokenizerRegistry.forName("whitespace").normalizerLowercase());
        assertSame(PreTokenizerType.GPT2, PreTokenizerRegistry.forName("exaone4").type());
        assertSame(PreTokenizerType.GPT4O, PreTokenizerRegistry.forName("llama4").type());
        assertSame(PreTokenizerType.LLAMA3, PreTokenizerRegistry.forName("pixtral").type());
        assertTrue(PreTokenizerRegistry.forName("megrez").cleanSpaces());
        assertFalse(PreTokenizerRegistry.forName("qwen2").cleanSpaces());
        assertNull(PreTokenizerRegistry.forName("qwen2").addBos());
    }

    /** LLAMA3 is one regex served by {@code unicode_regex_split_custom_llama3}: digits in triples. */
    @Test
    public void llama3TraceMatchesLlamaCpp() {
        assertEquals(Arrays.asList("Hello", " world", "'s", " ", "123", "45", "\n\n", " ", " x"),
                split("llama-bpe", TEXT));
        assertEquals(Arrays.asList("AB你好"), split("llama-bpe", "AB你好"));
        assertEquals(Arrays.asList("'S", "x"), split("llama3", "'Sx"));
    }

    /** QWEN2 is served by {@code unicode_regex_split_custom_qwen2}: single digits. */
    @Test
    public void qwen2TraceMatchesLlamaCpp() {
        assertEquals(Arrays.asList("Hello", " world", "'s", " ", "1", "2", "3", "4", "5", "\n\n", " ", " x"),
                split("qwen2", TEXT));
    }

    /**
     * DEEPSEEK3_LLM: {@code \p{N}{1,3}} (collapsed regex), the CJK run regex (code points), then
     * the main word regex (collapsed).
     */
    @Test
    public void deepseekV3TraceMatchesLlamaCpp() {
        assertEquals(Arrays.asList("Hello", " world", "'s", " ", "123", "456", "7", "\n\n", " ", " x"),
                split("deepseek-v3", "Hello world's 1234567\n\n  x"));
        assertEquals(Arrays.asList("AB", "你好"), split("deepseek-v3", "AB你好"));
        assertEquals(Arrays.asList("a", ".b", " !!\n", "x"), split("deepseek-v3", "a.b !!\nx"));
    }

    /** GPT4O splits at lower-to-upper case changes (rewritten lookahead classes); contractions stay attached. */
    @Test
    public void gpt4oSplitsCamelCase() {
        assertEquals(Arrays.asList("Hello", "World's", " ABCdef"), split("gpt-4o", "HelloWorld's ABCdef"));
    }

    /** GPT-2 custom splitter: optional leading space per class run; trailing space run kept whole. */
    @Test
    public void gpt2TraceMatchesLlamaCpp() {
        assertEquals(Arrays.asList("Hello", " world", "'s", " 12", "!!", " ", " x", "  "),
                split("gpt-2", "Hello world's 12!!  x  "));
    }

    @Test
    public void newlineRunsAndDigitGroups() {
        assertEquals(Arrays.asList("a b", "\n\n", "c"), split("gemma4", "a b\n\nc"));
        assertEquals(Arrays.asList("1", "234", "567"), split("tiny_aya", "1234567"));
        assertEquals(Arrays.asList("1", "234"), split("superbpe", "1234"));
    }

    @Test
    public void collapsedRegexRewrite() {
        assertEquals("[\\x{D1}0-9]+", UnicodeRegexSplit.collapseCategories("\\p{N}+"));
        assertEquals("[^\\s\\x{D2}A-Za-z]", UnicodeRegexSplit.collapseCategories("[^\\s\\p{L}]"));
        assertEquals("\\s+\\z", UnicodeRegexSplit.toJavaSyntax("\\s+$"));
        assertEquals("[a\\&b\\[]", UnicodeRegexSplit.toJavaSyntax("[a&b[]"));
        assertEquals("x[[A-Z]&&[^a-z]]*", UnicodeRegexSplit.flattenLookaheadClasses("x((?=[A-Z])([^a-z]))*"));
    }

    @Test
    public void emptyTextHasNoWords() {
        assertTrue(split("qwen2", "").isEmpty());
    }
}
