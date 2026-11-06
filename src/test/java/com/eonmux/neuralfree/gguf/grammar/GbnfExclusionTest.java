package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The exclusion grammar (llama.cpp {@code gbnf_excluding_grammar}): every text without an excluded string matches,
 * every text holding one — overlapping prefixes such as {@code <<|end|>} included — does not.
 */
public class GbnfExclusionTest {

    private static Grammar excluding(String... strings) {
        GbnfBuilder b = new GbnfBuilder();
        String start = GbnfExclusion.excluding(b::addRule, "body", Arrays.asList(strings));
        b.addRule(Grammar.ROOT_RULE, start);
        return b.grammar();
    }

    private static boolean matches(Grammar g, String text) {
        return GrammarMatcher.matches(g, text);
    }

    @Test
    public void textWithoutTheDelimiterMatchesAndTextWithItDoesNot() {
        Grammar g = excluding("<|end|>");
        for (String ok : new String[]{"", "plain text", "a < b | c > d", "<|en|>", "<|end|", "|end|>", "<<|end"}) {
            assertTrue(ok, matches(g, ok));
        }
        for (String bad : new String[]{"<|end|>", "x<|end|>", "<|end|>y", "<<|end|>", "<|<|end|>", "a<|end|>b"}) {
            assertFalse(bad, matches(g, bad));
        }
    }

    @Test
    public void severalDelimitersAndSelfOverlappingOnesAreAllExcluded() {
        Grammar g = excluding("abab", "</p>");
        assertTrue(matches(g, "aba bab ab"));
        assertFalse("the overlap aab-abab", matches(g, "aabab"));
        assertFalse(matches(g, "abaabab"));
        assertFalse(matches(g, "x</p>"));
        assertTrue(matches(g, "</ p>"));
    }

    @Test
    public void classMetacharactersAreEscaped() {
        Grammar g = excluding("^]-\\\"");
        assertTrue(matches(g, "^]-\\"));
        assertFalse(matches(g, "a^]-\\\"b"));
    }

    @Test
    public void emptyInputsAndCollidingPrefixesRefuse() {
        GbnfBuilder b = new GbnfBuilder();
        assertThrows(IllegalArgumentException.class,
                () -> GbnfExclusion.excluding(b::addRule, "x", Collections.<String>emptyList()));
        assertThrows(IllegalArgumentException.class,
                () -> GbnfExclusion.excluding(b::addRule, "x", Collections.singletonList("")));
        b.addRule("taken", "\"q\"");
        List<String> end = Collections.singletonList("<|end|>");
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> GbnfExclusion.excluding(b::addRule, "taken", end));
        assertTrue(e.getMessage(), e.getMessage().contains("collides"));
        assertEquals("body", GbnfExclusion.excluding(new GbnfBuilder()::addRule, "body", end));
    }
}
