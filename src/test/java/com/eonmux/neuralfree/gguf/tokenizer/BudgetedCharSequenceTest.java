package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Behaviour;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Dialect;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Step;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * The regex read budget: {@code 1000 x input length + 1,000,000} character reads per match call,
 * exhaustion reported with the pattern and the input length, and shared by sub-sequences. It
 * applies to patterns from model files; the built-in pre-tokenizer patterns run unbudgeted.
 */
public class BudgetedCharSequenceTest {

    private static String repeat(char c, int n) {
        return String.join("", Collections.nCopies(n, String.valueOf(c)));
    }

    private static List<String> splitOniguruma(String regex, String text) {
        PreTokenizerSpec spec = new PreTokenizerSpec(
                Collections.singletonList(Step.regex(regex, Dialect.ONIGURUMA, Behaviour.ISOLATED, false)),
                false, false, false, false);
        return PreTokenizerRegistry.fromSpec(spec).split(text);
    }

    @Test
    public void budgetIsLinearInInputLength() {
        assertEquals(1_000_000L, BudgetedCharSequence.budgetFor(0));
        assertEquals(1_065_000L, BudgetedCharSequence.budgetFor(65));
        assertThrows(IllegalArgumentException.class, () -> BudgetedCharSequence.budgetFor(-1));
    }

    @Test
    public void everyReadIsCharged() {
        BudgetedCharSequence seq = new BudgetedCharSequence("abc", "x");
        long start = seq.remaining();
        seq.charAt(0);
        seq.charAt(2);
        assertEquals(start - 2, seq.remaining());
        assertEquals("abc", seq.toString());
        assertEquals(start - 2, seq.remaining());
        assertEquals("x", seq.pattern());
    }

    @Test
    public void subSequenceSharesTheCounter() {
        BudgetedCharSequence seq = new BudgetedCharSequence("hello", "p");
        CharSequence sub = seq.subSequence(1, 4);
        assertEquals("ell", sub.toString());
        long before = seq.remaining();
        assertEquals('e', sub.charAt(0));
        assertEquals('l', sub.charAt(2));
        assertEquals(before - 2, seq.remaining());
        assertThrows(IndexOutOfBoundsException.class, () -> sub.charAt(3));
    }

    @Test
    public void resetRefillsForANewMatchCall() {
        BudgetedCharSequence seq = new BudgetedCharSequence("abcd", "p");
        for (int i = 0; i < 4; i++) seq.charAt(i);
        seq.reset(10);
        assertEquals(BudgetedCharSequence.budgetFor(10), seq.remaining());
    }

    @Test
    public void exponentialBacktrackingHitsTheBudgetWithANamedError() {
        String input = repeat('a', 64) + "b";
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> splitOniguruma("((a+)+)+$", input));
        assertEquals("tokenizer regex '((a+)+)+$' exceeded its step budget on a 65-char input", e.getMessage());
    }

    @Test
    public void quadraticBacktrackingOnALongInputHitsTheBudget() {
        // Java's matcher memoizes the single nested loop of (a+)+$, which leaves it quadratic in
        // the input; a few thousand characters exceed the linear budget.
        String input = repeat('a', 4096) + "b";
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> splitOniguruma("(a+)+$", input));
        assertEquals("tokenizer regex '(a+)+$' exceeded its step budget on a 4097-char input", e.getMessage());
    }

    @Test
    public void builtInPatternsAreMatchedWithoutABudget() {
        // DEEPSEEK_LLM's "\s+$" step reads to the word end from every space of the run before
        // failing on the CJK character: about 8,000,000 reads, beyond the 5,001,000 budget.
        String spaces = repeat(' ', 4000);
        assertEquals(Arrays.asList(spaces, "中"), PreTokenizerRegistry.forName("deepseek-llm").split(spaces + "中"));
    }

    @Test
    public void theSamePatternFromAModelFileIsBudgeted() {
        String spaces = repeat(' ', 4000);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> splitOniguruma("\\s+$", spaces + "中"));
        assertEquals("tokenizer regex '\\s+$' exceeded its step budget on a 4001-char input", e.getMessage());
    }

    @Test
    public void ordinaryPatternsStayWithinBudget() {
        String text = repeat('a', 5000) + " " + repeat('b', 5000);
        assertEquals(Arrays.asList(repeat('a', 5000), " ", repeat('b', 5000)), splitOniguruma("\\w+", text));
    }
}
