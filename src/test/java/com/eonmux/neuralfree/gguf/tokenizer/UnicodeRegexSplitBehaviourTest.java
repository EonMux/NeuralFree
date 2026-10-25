package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Behaviour;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Dialect;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.PrependScheme;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Step;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * Spec-driven splitting against Hugging Face tokenizers {@code pre_tokenize_str} results: each
 * split behaviour with and without inversion, the Oniguruma dialect, and the non-regex steps.
 */
public class UnicodeRegexSplitBehaviourTest {

    private static final String DASHES = "a--b-c";
    private static final int METASPACE = 0x2581;

    private static PreTokenizerRegistry.PreTokenizer pre(Step... steps) {
        return PreTokenizerRegistry.fromSpec(new PreTokenizerSpec(Arrays.asList(steps), false, false, false, false));
    }

    private static PreTokenizerRegistry.PreTokenizer byteLevel(boolean addPrefixSpace, boolean useRegex, Step... steps) {
        return PreTokenizerRegistry.fromSpec(new PreTokenizerSpec(Arrays.asList(steps), true, addPrefixSpace, false,
                useRegex));
    }

    private static List<String> dash(Behaviour behaviour, boolean invert) {
        return pre(Step.literal("-", behaviour, invert)).split(DASHES);
    }

    private static List<String> onig(String regex, String text) {
        return pre(Step.regex(regex, Dialect.ONIGURUMA, Behaviour.ISOLATED, false)).split(text);
    }

    private static List<String> list(String... words) {
        return Arrays.asList(words);
    }

    @Test
    public void isolated() {
        assertEquals(list("a", "-", "-", "b", "-", "c"), dash(Behaviour.ISOLATED, false));
    }

    @Test
    public void removed() {
        assertEquals(list("a", "b", "c"), dash(Behaviour.REMOVED, false));
        assertEquals(list("-", "-", "-"), dash(Behaviour.REMOVED, true));
    }

    @Test
    public void mergedWithPrevious() {
        assertEquals(list("a-", "-", "b-", "c"), dash(Behaviour.MERGED_WITH_PREVIOUS, false));
        assertEquals(list("a", "-", "-b", "-c"), dash(Behaviour.MERGED_WITH_PREVIOUS, true));
    }

    @Test
    public void mergedWithNext() {
        assertEquals(list("a", "-", "-b", "-c"), dash(Behaviour.MERGED_WITH_NEXT, false));
        assertEquals(list("a-", "-", "b-", "c"), dash(Behaviour.MERGED_WITH_NEXT, true));
    }

    @Test
    public void contiguous() {
        assertEquals(list("a", "--", "b", "-", "c"), dash(Behaviour.CONTIGUOUS, false));
    }

    @Test
    public void onigurumaWordClassIsUnicode() {
        assertEquals(list("héllo", " ", "wörld"), onig("\\w+", "héllo wörld"));
    }

    @Test
    public void onigurumaDollarIsALineAnchor() {
        assertEquals(list("a", "  ", "\nb", "  "), onig("\\s+$", "a  \nb  "));
    }

    @Test
    public void onigurumaPropertyAndClassSyntax() {
        assertEquals(list("abc", "日本", "def"), onig("\\p{Han}+", "abc日本def"));
        assertEquals(list("ab", "12", "c"), onig("[[:digit:]]+", "ab12c"));
        assertEquals(list("zz", "beef", "zz"), onig("\\h+", "zzbeefzz"));
    }

    @Test
    public void onigurumaRubySyntaxDifferences() {
        assertEquals(list("x", "a\nb", "y"), onig("(?m:a.b)", "xa\nby"));
        assertEquals(list("a", "\u000B", "b"), onig("\\v", "a\u000Bb"));
        assertEquals(list("é", "ab"), onig("\\p{^In_Basic_Latin}+", "éab"));
        assertEquals("(?s:.)(?i-s)\\x{0B}[\\x{0B}]\\p{InBasic_Latin}(?=a)",
                UnicodeRegexSplit.onigurumaToJava("(?m:.)(?i-m)\\v[\\v]\\p{In_Basic_Latin}(?=a)"));
    }

    @Test
    public void emptyMatchesProduceNoWords() {
        assertEquals(list("a", "b"), onig("x*", "ab"));
    }

    @Test
    public void invertedRemovedKeepsOnlyTheMatches() {
        assertEquals(list("hi", "there"),
                pre(Step.regex("\\w+", Dialect.ONIGURUMA, Behaviour.REMOVED, true)).split("hi, there!"));
    }

    @Test
    public void whitespaceAndBert() {
        String text = "Hey friend!     How are you?!?";
        assertEquals(list("Hey", "friend", "!", "How", "are", "you", "?!?"), pre(Step.whitespace()).split(text));
        assertEquals(list("Hey", "friend", "!", "How", "are", "you", "?", "!", "?"), pre(Step.bert()).split(text));
        assertEquals(list("a", "b", "c"), pre(Step.whitespaceSplit()).split("a b c"));
    }

    @Test
    public void punctuation() {
        assertEquals(list("Hey", "!", "!", " you", "?"), pre(Step.punctuation(Behaviour.ISOLATED)).split("Hey!! you?"));
        assertEquals(list("Hey", "!!", " you", "?"), pre(Step.punctuation(Behaviour.CONTIGUOUS)).split("Hey!! you?"));
        assertEquals(list("a", "«", "b", "»", "c", "¿", "d"), pre(Step.punctuation(Behaviour.ISOLATED)).split("a«b»c¿d"));
    }

    @Test
    public void digitsAndCharDelimiter() {
        assertEquals(list("abc", "123", "def", "45"), pre(Step.digits(false)).split("abc123def45"));
        assertEquals(list("x", "1", "2", "3", "y"), pre(Step.digits(true)).split("x123y"));
        assertEquals(list("x", "²", "½", "٣"), pre(Step.digits(true)).split("x²½٣"));
        assertEquals(list("a", "b", "c"), pre(Step.charDelimiter('|')).split("a|b||c|"));
    }

    @Test
    public void stepsRunInSequence() {
        assertEquals(list("ab", "1", "2", "c", "3"), pre(Step.whitespaceSplit(), Step.digits(true)).split("ab12 c3"));
    }

    @Test
    public void metaspacePrependSchemes() {
        String text = "hello world";
        assertEquals(list("▁hello", "▁", "▁world\tx"),
                pre(Step.metaspace(METASPACE, PrependScheme.ALWAYS, true)).split("hello  world\tx"));
        assertEquals(list("▁hello", "▁world"), pre(Step.metaspace(METASPACE, PrependScheme.FIRST, true)).split(text));
        assertEquals(list("hello", "▁world"),
                pre(Step.metaspace(METASPACE, PrependScheme.FIRST, true)).split(text, false));
        assertEquals(list("hello", "▁world"), pre(Step.metaspace(METASPACE, PrependScheme.NEVER, true)).split(text));
        assertEquals(list("▁hello▁world"), pre(Step.metaspace(METASPACE, PrependScheme.ALWAYS, false)).split(text));
    }

    @Test
    public void unicodeScripts() {
        assertEquals(list("abc", "日本語カタカナ ", "abc", ". ", "def"),
                pre(Step.unicodeScripts()).split(" abc日本語カタカナ abc. def"));
    }

    @Test
    public void byteLevelStep() {
        assertEquals(list("Ġhello", "Ġworld"), byteLevel(true, true, Step.byteLevel()).words("hello world"));
        assertEquals(list("hiĠthere"), byteLevel(false, false, Step.byteLevel()).words("hi there"));
    }

    @Test
    public void stepsAfterByteLevelSeeMappedText() {
        PreTokenizerRegistry.PreTokenizer p = byteLevel(false, true, Step.byteLevel(), Step.digits(false));
        assertEquals(list("x", "Ġ", "12", "ĠÂ", "²"), p.words("x 12 ²"));
        assertThrows(IllegalStateException.class, () -> p.split("x 12"));
    }

    @Test
    public void isolatedLlamaCppStepsMatchTheRegistry() {
        String text = "Hello world's 12345\n\n  x";
        for (String name : Arrays.asList("llama-bpe", "qwen2", "gpt-2", "deepseek-coder", "falcon", "tekken")) {
            PreTokenizerRegistry.PreTokenizer named = PreTokenizerRegistry.forName(name);
            PreTokenizerRegistry.PreTokenizer built = PreTokenizerRegistry.fromSpec(PreTokenizerRegistry.specFor(name));
            assertEquals(name, named.split(text), built.split(text));
            assertEquals(name, named.words(text), built.words(text));
        }
        assertEquals(list("Hello", " world", "'s", " ", "123", "45", "\n\n", " ", " x"),
                PreTokenizerRegistry.forName("llama-bpe").split(text));
    }
}
