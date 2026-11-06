package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests of the output-regex dialect of {@link RegexToGbnf#compile}, the {@code guided_regex} /
 * {@code structured_outputs.regex} constraint: ECMAScript syntax over the whole raw output, as
 * vLLM's xgrammar backend reads it ({@code compile_regex}). Each pattern is compiled to a
 * grammar and driven over byte tokens ({@link GrammarMatcher}), so every case asserts the
 * language: the strings the full match accepts and the near misses it must refuse. The
 * refusals assert the named construct. Pure logic; no model file required.
 */
public class RegexToGbnfOutputTest {

    private static Grammar regex(String pattern) {
        return RegexToGbnf.compile(pattern, "root", false);
    }

    private static void accepts(String pattern, String... texts) {
        Grammar grammar = regex(pattern);
        for (String text : texts) {
            assertTrue(pattern + " must match " + text, GrammarMatcher.matchesUtf8(grammar, text));
        }
    }

    private static void rejects(String pattern, String... texts) {
        Grammar grammar = regex(pattern);
        for (String text : texts) {
            assertFalse(pattern + " must not match " + text, GrammarMatcher.matchesUtf8(grammar, text));
        }
    }

    /**
     * Whether some rule of {@code grammar} holds the character {@code codePoint}: NUL cannot be
     * driven through the sampler (a piece starting with byte 0 is refused, as in llama.cpp).
     */
    private static boolean hasChar(Grammar grammar, int codePoint) {
        for (int i = 0; i < grammar.ruleCount(); i++) {
            if (grammar.rule(i).contains(new GrammarElement(GrammarElement.Type.CHAR, codePoint))) {
                return true;
            }
        }
        return false;
    }

    private static void unsupported(String pattern, String construct) {
        try {
            regex(pattern);
            fail(pattern + " must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals(pattern, construct, e.getMessage());
        }
    }

    private static void invalid(String pattern, String problem) {
        try {
            regex(pattern);
            fail(pattern + " must be invalid");
        } catch (RegexToGbnf.InvalidRegexException e) {
            assertEquals(pattern, problem, e.getMessage());
        }
    }

    // -------------------------------------------------- shorthand classes outside a class

    @Test
    public void digitShorthandWithBoundsIsTheTypicalGuidedRegex() {
        accepts("\\d{3}-\\d{4}", "555-1234", "000-0000");
        rejects("\\d{3}-\\d{4}", "55-1234", "555-12345", "abc-defg", "555 1234", "");
    }

    @Test
    public void wordAndSpaceShorthandsAndTheirNegations() {
        accepts("\\w+@\\w+\\.com", "foo_1@Bar9.com");
        rejects("\\w+@\\w+\\.com", "foo@bar-x.com", "foo@barxcom");
        accepts("\\s", " ", "\t", "\n", "\r", "\u000B", "\f", "\u00A0", "\u2028", "\u3000", "\uFEFF");
        rejects("\\s", "a", "_", "\u200B");
        accepts("\\S+", "a-b", "\u00E9");
        rejects("\\S+", "a b", "\t");
        accepts("\\D\\W", "a ", "x-");
        rejects("\\D", "5");
        rejects("\\W", "a", "Z", "_", "7");
    }

    // -------------------------------------------------- character classes

    @Test
    public void shorthandsInsideClassesUniteWithTheOtherMembers() {
        accepts("[\\d\\s]+", "1 2\t3");
        rejects("[\\d\\s]+", "1a");
        accepts("[\\w-]+", "a-b_c9");
        accepts("[^\\d\\s]+", "ab.", "\u00E9");
        rejects("[^\\d\\s]+", "a1", "a b");
    }

    @Test
    public void aNegatedShorthandInsideAPositiveClassIsItsExactComplement() {
        accepts("[a\\D]", "x", "a", " ");
        rejects("[a\\D]", "5");
        accepts("[\\s\\S]*", "", "any\nthing");
        accepts("[^\\D]", "7");
        rejects("[^\\D]", "x");
    }

    @Test
    public void classRangesEscapesAndLiteralMetacharacters() {
        accepts("[a-c\\-\\]\\\\.^]+", "abc-]\\.^");
        rejects("[a-c\\-\\]\\\\.^]+", "d");
        accepts("[\\x41-\\x43\\u0061]+", "ABCa");
        accepts("[a-]", "a", "-");
        accepts("[-a]", "a", "-");
        accepts("[\\b]", "\b");
        accepts("[(){}|*+?$]+", "(){}|*+?$");
        accepts("[\\u{1F600}-\\u{1F64F}]", "\uD83D\uDE00", "\uD83D\uDE4F");
        rejects("[\\u{1F600}-\\u{1F64F}]", "\uD83D\uDE50");
    }

    // -------------------------------------------------- groups

    @Test
    public void capturingAndNamedGroupsOnlyGroup() {
        accepts("(\\d+)-(\\w+)", "12-ab");
        accepts("(a|b)+c", "abbac", "ac");
        rejects("(a|b)+c", "c", "abd");
        accepts("(?<year>\\d{4})-(?P<month>\\d{2})", "2024-05");
        rejects("(?<year>\\d{4})-(?P<month>\\d{2})", "2024-5");
        accepts("(?:ab)*", "", "abab");
        accepts("()a", "a");
    }

    // -------------------------------------------------- anchors and full-match semantics

    @Test
    public void anUnanchoredPatternMatchesTheWholeOutput() {
        accepts("\\d+", "123");
        rejects("\\d+", "a123", "123a", "");
    }

    @Test
    public void anchorsAtTopLevelAlternativeBoundariesAreImplied() {
        accepts("^yes$|^no$", "yes", "no");
        rejects("^yes$|^no$", "yesno", "");
        accepts("^(a|b)$", "a", "b");
        accepts("^abc", "abc");
        accepts("abc$", "abc");
    }

    // -------------------------------------------------- escapes

    @Test
    public void standardCharacterEscapes() {
        accepts("\\t\\n\\r\\f\\v", "\t\n\r\f\u000B");
        assertTrue("\\0 is the NUL character", hasChar(regex("\\0"), 0));
        accepts("\\x41\\u0042\\u{43}\\U00000044", "ABCD");
        accepts("\\cJ\\cm", "\n\r");
        accepts("\\uD83D\\uDE00", "\uD83D\uDE00");
        accepts("\\u00e9+", "\u00E9\u00E9");
    }

    @Test
    public void identityEscapesOfPunctuationAreTheCharacterItself() {
        accepts("\\.\\/\\-\\#\\ \\\"\\\\\\$\\^\\*\\+\\?\\(\\)\\[\\]\\{\\}\\|", "./-# \"\\$^*+?()[]{}|");
        rejects("a\\.b", "axb");
        accepts("a\\$", "a$");
    }

    @Test
    public void quotesAndBackslashesInLiteralsAreWrittenAsGbnfEscapes() {
        accepts("say \"hi\"", "say \"hi\"");
        accepts("a\\\\b", "a\\b");
    }

    // -------------------------------------------------- quantifiers

    @Test
    public void lazyQuantifiersAcceptTheGreedyLanguage() {
        accepts("a+?b", "ab", "aab");
        rejects("a+?b", "b");
        accepts("a{2,3}?", "aa", "aaa");
        rejects("a{2,3}?", "a", "aaaa");
        accepts("a*?", "", "aaa");
        accepts("a??", "", "a");
        rejects("a??", "aa");
    }

    @Test
    public void quantifierBoundsIncludingPythonsOpenLowerBound() {
        accepts("a{2,}", "aa", "aaaaa");
        rejects("a{2,}", "a");
        accepts("a{,2}", "", "a", "aa");
        rejects("a{,2}", "aaa");
        accepts("ab{0}c", "ac");
        accepts("(?:ab){2}", "abab");
        rejects("(?:ab){2}", "ab");
    }

    @Test
    public void aQuantifierBindsToTheLastCharacterOfALiteralRun() {
        accepts("abc+", "abc", "abccc");
        rejects("abc+", "abcabc");
        accepts("a\\.+", "a...");
        rejects("a\\.+", "a.a.");
    }

    @Test
    public void bracesAndBracketsThatStartNoConstructAreLiterals() {
        accepts("x{", "x{");
        accepts("a{1,x}", "a{1,x}");
        accepts("{}", "{}");
        accepts("a]", "a]");
        accepts("a}", "a}");
    }

    // -------------------------------------------------- the dot, empty pieces

    @Test
    public void theDotExcludesLineBreaksUnlessDotAll() {
        accepts("a.c", "abc", "a c");
        rejects("a.c", "a\nc", "a\rc");
        assertTrue(GrammarMatcher.matchesUtf8(RegexToGbnf.compile("a.c", "root", true), "a\nc"));
    }

    @Test
    public void mayWriteFindsTheCharactersAMatchCanHold() {
        java.util.Set<Integer> lt = java.util.Collections.singleton((int) '<');
        assertFalse(RegexToGbnf.mayWrite("[0-9]{4}-[0-9]{2}", lt));
        assertFalse(RegexToGbnf.mayWrite("[^\\s<]+", lt));
        assertFalse(RegexToGbnf.mayWrite("(?:ab|c)*d", lt));
        assertTrue(RegexToGbnf.mayWrite("a<b", lt));
        assertTrue(RegexToGbnf.mayWrite("[a-z<]+", lt));
        assertTrue(RegexToGbnf.mayWrite("\\W", lt));
        assertTrue("a dot holds any character", RegexToGbnf.mayWrite("a.", lt));
        assertTrue("a pattern without a grammar may hold anything", RegexToGbnf.mayWrite("a(?=b)", lt));
        assertFalse(RegexToGbnf.mayWrite("a.", java.util.Collections.<Integer>emptySet()));
    }

    @Test
    public void emptyPatternsAndEmptyAlternatives() {
        accepts("", "");
        rejects("", "a");
        accepts("a|", "a", "");
        accepts("|a", "a", "");
    }

    // -------------------------------------------------- refusals, each naming the construct

    @Test
    public void constructsWithoutAGrammarEquivalentAreRefusedByName() {
        unsupported("\\bfoo", "word boundary assertion: \\b");
        unsupported("a\\B", "word boundary assertion: \\B");
        unsupported("(?=a)a", "lookahead assertion: (?=");
        unsupported("(?!a)b", "lookahead assertion: (?!");
        unsupported("(?<=a)b", "lookbehind assertion: (?<=");
        unsupported("(?<!a)b", "lookbehind assertion: (?<!");
        unsupported("(a)\\1", "backreference: \\1");
        unsupported("(?<n>a)\\k<n>", "backreference: \\k");
        unsupported("(?P<n>a)(?P=n)", "named backreference: (?P=");
        unsupported("(?i)abc", "inline flag group: (?i");
        unsupported("(?>a)", "atomic group: (?>");
        unsupported("(?#note)a", "comment group: (?#");
        unsupported("a++", "possessive quantifier");
        unsupported("\\p{L}", "Unicode property escape: \\p");
        unsupported("\\P{L}", "Unicode property escape: \\P");
        unsupported("\\01", "octal escape: \\01");
        unsupported("\\q", "unsupported escape: \\q");
        unsupported("\\a", "unsupported escape: \\a");
        unsupported("a^b", "anchor inside the pattern");
        unsupported("a$b", "anchor inside the pattern");
        unsupported("(^a)", "anchor inside the pattern");
        unsupported("(a$)", "anchor inside the pattern");
        unsupported("[]a]", "']' first in a character class (an empty class in ECMAScript,"
                + " a literal in Python); write it as \\]");
        unsupported("a{99999999999}", "repetition bound too large: 99999999999");
    }

    @Test
    public void malformedPatternsAreInvalid() {
        invalid("(a", "unbalanced parentheses");
        invalid("a)", "unbalanced parentheses");
        invalid("[a", "unterminated character class");
        invalid("[a-", "unterminated character class");
        invalid("*a", "nothing to repeat");
        invalid("a**", "nothing to repeat");
        invalid("a|*", "nothing to repeat");
        invalid("^*", "nothing to repeat");
        invalid("a{2}{3}", "nothing to repeat");
        invalid("a???", "nothing to repeat");
        invalid("a{3,1}", "numbers out of order in curly brackets: {3,1}");
        invalid("[z-a]", "character range out of order");
        invalid("[\\d-z]", "character range with a class shorthand");
        invalid("[a-\\d]", "character range with a class shorthand");
        invalid("\\x4", "invalid \\x escape");
        invalid("\\u12", "invalid \\u escape");
        invalid("\\u{110000}", "\\u{...} escape beyond U+10FFFF");
        invalid("\\u{}", "invalid \\u{...} escape");
        invalid("\\c1", "\\c must be followed by an ASCII letter");
        invalid("a\\", "trailing backslash");
        invalid("(?<n>a)(?<n>b)", "duplicate group name: n");
        invalid("(?<1n>a)", "invalid group name: 1n");
        invalid("(?<n", "unterminated group name");
        invalid("(?", "incomplete group syntax: (?");
    }

    // -------------------------------------------------- the constraint dispatch names the regex

    @Test
    public void theRegexConstraintRefusalNamesTheConstraintAndTheConstruct() {
        try {
            GbnfBuilder.grammarOf(ConstraintSpec.regex("\\bfoo", false));
            fail("a word boundary must be refused");
        } catch (IllegalArgumentException e) {
            assertEquals("regex constraint \\bfoo uses an unsupported construct:"
                    + " word boundary assertion: \\b", e.getMessage());
        }
        try {
            GbnfBuilder.grammarOf(ConstraintSpec.regex("(a", false));
            fail("an unbalanced group must be refused");
        } catch (IllegalArgumentException e) {
            assertEquals("regex constraint (a is not a valid regex: unbalanced parentheses", e.getMessage());
        }
        assertTrue(GrammarMatcher.matchesUtf8(
                GbnfBuilder.grammarOf(ConstraintSpec.regex("\\d{3}-\\d{4}", false)), "555-1234"));
    }
}
