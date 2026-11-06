package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.GbnfParser;
import com.eonmux.neuralfree.gguf.grammar.GbnfPrinter;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;

/**
 * The counted regions of xgrammar's {@code max_tokens} / {@code max_chars} rule attributes and the any-token
 * element, over a multi-character vocabulary: the masks after each prefix are xgrammar 0.2.8's
 * ({@code GrammarMatcher.fill_next_token_bitmask}) for the same structural tags over the same vocabulary — a token
 * that opens a region does not spend its token budget, every code point inside a region spends its character
 * budget, and a spent region lets the output leave it only. Each mask is checked both through {@link
 * GrammarStage#admits} and through the sampling mask {@link GrammarStage#apply} writes.
 */
public class GrammarBudgetTest {

    /** The vocabulary of the xgrammar runs; id 0 is EOG. */
    private static final String[] WORDS = {"<eos>", "x", "xa", "a", "b", "y", "by", "ab", "aby"};

    private static final SamplerVocab VOCAB = new SamplerVocab() {
        @Override public int vocabSize() { return WORDS.length; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) {
            return id == 0 ? new byte[0] : WORDS[id].getBytes(StandardCharsets.UTF_8);
        }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            return new int[]{Arrays.asList(WORDS).indexOf(text)};
        }
    };

    private static Grammar tag(String content) {
        return GbnfBuilder.grammarOf(ConstraintSpec.structuralTag("{\"format\": {\"type\": \"tag\", \"begin\": \"x\","
                + " \"content\": " + content + ", \"end\": \"y\"}}"));
    }

    private static Grammar format(String format) {
        return GbnfBuilder.grammarOf(ConstraintSpec.structuralTag("{\"format\": " + format + "}"));
    }

    /** The tokens the grammar admits after {@code prefix}; the mask and {@code admits} must agree. */
    private static Set<String> allowed(Grammar grammar, String... prefix) {
        GrammarStage stage = new GrammarStage(grammar, VOCAB);
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], List.of(stage), VOCAB);
        float[] uniform = new float[WORDS.length];
        sampler.sample(uniform); // arms the stage
        for (String word : prefix) {
            sampler.accept(Arrays.asList(WORDS).indexOf(word));
        }
        Set<String> admitted = new LinkedHashSet<>();
        Set<String> unmasked = new LinkedHashSet<>();
        for (int id = 0; id < WORDS.length; id++) {
            if (stage.admits(id)) {
                admitted.add(WORDS[id]);
            }
            float[] logits = new float[WORDS.length];
            Arrays.fill(logits, -100f);
            logits[id] = 10f;
            try {
                if (sampler.sample(logits) == id) {
                    unmasked.add(WORDS[id]);
                }
            } catch (RuntimeException e) {
                // every candidate masked: the id is not admitted either
            }
        }
        assertEquals("admits and the sampling mask agree", admitted, unmasked);
        return admitted;
    }

    private static Set<String> words(String... words) {
        return new LinkedHashSet<>(Arrays.asList(words));
    }

    private static final String[] ALL_TEXT = {"x", "xa", "a", "b", "y", "by", "ab", "aby"};

    @Test
    public void aTokenBudgetCountsTheTokensThatStartInsideTheRegion() {
        Grammar grammar = tag("{\"type\": \"any_text\", \"max_tokens\": 2}");
        assertEquals(words("x", "xa"), allowed(grammar));
        assertEquals(words(ALL_TEXT), allowed(grammar, "x"));
        assertEquals("the token that opens the region is free", words(ALL_TEXT), allowed(grammar, "xa"));
        assertEquals(words(ALL_TEXT), allowed(grammar, "x", "a"));
        assertEquals("two tokens spent: only the end follows", words("y"), allowed(grammar, "x", "a", "b"));
        assertEquals(words(ALL_TEXT), allowed(grammar, "xa", "b"));
        assertEquals(words("<eos>"), allowed(grammar, "x", "a", "aby"));
    }

    @Test
    public void aCharacterBudgetCountsEveryCodePointInsideTheRegion() {
        Grammar grammar = tag("{\"type\": \"any_text\", \"max_chars\": 2}");
        assertEquals(words(ALL_TEXT), allowed(grammar, "x"));
        assertEquals(words("x", "a", "b", "y", "by"), allowed(grammar, "xa"));
        assertEquals(words("y"), allowed(grammar, "xa", "b"));
    }

    @Test
    public void aZeroBudgetAdmitsOnlyTheEmptyRegion() {
        assertEquals(words("y"), allowed(tag("{\"type\": \"any_text\", \"max_tokens\": 0}"), "x"));
        assertEquals(words("y"), allowed(tag("{\"type\": \"any_text\", \"max_chars\": 0}"), "x"));
    }

    @Test
    public void anyTokensWithoutExclusionsTakeAnyTokenUpToTheBudget() {
        Grammar grammar = format("{\"type\": \"any_tokens\", \"exclude_tokens\": [], \"max_tokens\": 2}");
        Set<String> every = words(WORDS);
        assertEquals(every, allowed(grammar));
        assertEquals(every, allowed(grammar, "a"));
        assertEquals(words("<eos>"), allowed(grammar, "a", "b"));
        assertEquals(every, allowed(format("{\"type\": \"any_tokens\"}"), "a", "b", "y"));
    }

    @Test
    public void excludeTokenWithoutExclusionsTakesExactlyOneToken() {
        Grammar grammar = format("{\"type\": \"exclude_token\", \"exclude_tokens\": []}");
        assertEquals(words(ALL_TEXT), allowed(grammar));
        assertEquals(words("<eos>"), allowed(grammar, "aby"));
    }

    @Test
    public void budgetsAndTheAnyTokenSurviveThePrintAndReparse() {
        Grammar grammar = GbnfParser.parse("root ::= \"x\" text \"y\" !<>\ntext[max_tokens=2, max_chars=3] ::= [ab]*");
        Grammar reparsed = GbnfParser.parse("root ::= " + GbnfPrinter.ruleName(grammar, grammar.rootRuleIndex(), "p-")
                + "\n" + GbnfPrinter.print(grammar, "p-"));
        for (Grammar g : new Grammar[]{grammar, reparsed}) {
            assertEquals(words("a", "b", "y", "by", "ab", "aby"), allowed(g, "x"));
            assertEquals("three chars spent", words("y"), allowed(g, "x", "ab", "a"));
            assertEquals("two tokens spent", words("y"), allowed(g, "x", "a", "b"));
            assertEquals(words(ALL_TEXT), allowed(g, "x", "a", "y"));
        }
    }
}
