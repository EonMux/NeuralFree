package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests of the regex to GBNF translator ({@link RegexToGbnf}) for JSON-schema value patterns, object key
 * patterns and output regexes. The value-pattern cases start from the regexp cases of llama.cpp's
 * {@code tests/test-json-schema-to-grammar.cpp} (the C++ test name is cited per test): where llama.cpp and
 * xgrammar agree the rule text is llama.cpp's, and where they differ (unanchored patterns, shorthand classes,
 * lazy quantifiers, the {@code .} and the JSON encoding of matched characters) the expectation is xgrammar
 * 0.2.8's full-match verdict. Pure logic; no model file required.
 */
public class RegexToGbnfTest {

    /** The GBNF text of one GBNF string literal (raw characters {@code "inner"}). */
    private static String q(String inner) {
        return "\"" + inner + "\"";
    }

    /** llama.cpp's wrapped-string form: the whole rule placeholder {@code "\"" ( … ) "\""}. */
    private static String qEscape() {
        return q("\\\"");
    }

    /** {@code valueRule} into a fresh builder; returns the builder for rule-text asserts. */
    private static GbnfBuilder translated(String pattern) {
        GbnfBuilder builder = new GbnfBuilder();
        RegexToGbnf.valueRule(pattern, "root", builder);
        return builder;
    }

    /** The grammar of the JSON strings value {@code pattern} admits. */
    private static Grammar valueGrammar(String pattern) {
        return translated(pattern).grammar();
    }

    // -------------------------------------------------- LC test "simple regexp" (:742-753)

    @Test
    public void literalsAndQuantifiers() {
        assertEquals(qEscape() + " (" + q("ab") + " " + q("c") + "? " + q("d") + "* "
                        + q("ef") + " " + q("g") + "+ " + "(" + q("hij") + ")? " + q("kl")
                        + ") " + qEscape(),
                translated("^abc?d*efg+(hij)?kl$").rule("root"));
    }

    // -------------------------------------------------- LC test "regexp" (:794-807): curly
    // repetitions hoist the class into a helper rule, and the dot becomes the JSON-encoded
    // ECMAScript dot rule

    @Test
    public void hoistsClassesAndDotRules() {
        GbnfBuilder builder = translated("^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$");
        assertEquals(JsonStringCharacters.dot(), builder.rule("json-string-dot"));
        assertEquals("[0-9]", builder.rule("root-1"));
        assertEquals(qEscape() + " (" + "(" + q("(") + " root-1{1,3} " + q(")") + ")?"
                        + " root-1{3,3} " + q("-") + " root-1{4,4} " + q(" ") + " "
                        + q("a") + "{3,5} " + q("nd")
                        + " json-string-dot json-string-dot json-string-dot) " + qEscape(),
                builder.rule("root"));
    }

    // -------------------------------------------------- LC test "regexp escapes" (:755-766)

    @Test
    public void escapesOfOperatorsBecomeOneLiteral() {
        assertEquals(qEscape() + " (" + q("[]{}()|+*?") + ") " + qEscape(),
                translated("^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$").rule("root"));
    }

    // -------------------------------------------------- llama.cpp ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS
    // (common/json-schema-to-grammar.cpp:282) holds '.', so an escaped dot is a literal dot

    @Test
    public void escapedDotBecomesALiteralDot() {
        assertEquals(qEscape() + " (" + q("a.b") + ") " + qEscape(),
                translated("^a\\.b$").rule("root"));
    }

    // -------------------------------------------------- LC test "regexp quote" (:768-779): the
    // matched quote is written JSON-encoded, so the value stays one JSON string (llama.cpp and
    // xgrammar write it raw)

    @Test
    public void quoteInsideThePatternIsWrittenEscaped() {
        assertEquals(qEscape() + " (" + q("\\\\\\\"") + ") " + qEscape(),
                translated("^\"$").rule("root"));
        Grammar grammar = valueGrammar("^\"$");
        assertTrue(GrammarMatcher.matches(grammar, "\"\\\"\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"\"\""));
    }

    // -------------------------------------------------- LC test "regexp with top-level alternation" (:781-792)

    @Test
    public void topLevelAlternation() {
        assertEquals(qEscape() + " (" + q("A") + " | " + q("B") + " | " + q("C") + " | "
                + q("D") + ") " + qEscape(),
                translated("^A|B|C|D$").rule("root"));
    }

    // -------------------------------------------------- LC test "regexp with non-capturing group" (:1424-1433)

    @Test
    public void nonCapturingGroup() {
        assertEquals(qEscape() + " (" + "(" + q("foo") + " | " + q("bar") + ") " + q("baz")
                        + ") " + qEscape(),
                translated("^(?:foo|bar)baz$").rule("root"));
    }

    // -------------------------------------------------- LC test "regexp with nested non-capturing groups" (:1435-1445)

    @Test
    public void nestedNonCapturingGroups() {
        assertEquals(qEscape() + " (" + "(" + "(" + q("ab") + ")+ " + q("c") + ")? "
                        + q("d") + ") " + qEscape(),
                translated("^(?:(?:ab)+c)?d$").rule("root"));
    }

    // -------------------------------------------------- xgrammar reads "+?" as a lazy quantifier,
    // which accepts the same strings as the greedy one (llama.cpp stacks "+" and "?")

    @Test
    public void lazyQuantifierMatchesAsTheGreedyOne() {
        assertEquals(qEscape() + " (" + q("a") + " " + q("b") + "+) " + qEscape(),
                translated("^ab+?$").rule("root"));
    }

    // -------------------------------------------------- LC test "regexp with escaped hyphen in
    // a character class" (:1479-1490): the class is rebuilt as its exact code-point set

    @Test
    public void escapedHyphenInCharacterClass() {
        assertEquals(qEscape() + " (" + "[\\-a-z]" + "+) " + qEscape(),
                translated("^[a-z\\-]+$").rule("root"));
    }

    // -------------------------------------------------- LC test "regexp with escaped hyphen
    // outside a character class" (:1492-1503): the escaped hyphen is a literal hyphen

    @Test
    public void escapedHyphenOutsideCharacterClass() {
        assertEquals(qEscape() + " (" + q("a-b") + ") " + qEscape(),
                translated("^a\\-b$").rule("root"));
    }

    // -------------------------------------------------- the dot is ECMAScript's: any character but a
    // line terminator, written JSON-encoded

    @Test
    public void dotMatchesAnyCharacterButALineTerminatorJsonEncoded() {
        Grammar grammar = valueGrammar("^.$");
        for (String value : new String[]{"\"a\"", "\"\\\"\"", "\"\\\\\"", "\"\\t\"", "\"\u00e9\""}) {
            assertTrue(value, GrammarMatcher.matchesUtf8(grammar, value));
        }
        for (String value : new String[]{"\"\\n\"", "\"\\r\"", "\"\u2028\"", "\"\"\"", "\"ab\""}) {
            assertFalse(value, GrammarMatcher.matchesUtf8(grammar, value));
        }
    }

    /** An XML parameter's value pattern: the same full match, the characters written raw and unquoted. */
    @Test
    public void rawValuePatternMatchesTheWholeRawText() {
        GbnfBuilder builder = new GbnfBuilder();
        RegexToGbnf.rawValueRule("a\"[^x]\\d.", "root", builder);
        Grammar grammar = builder.grammar();
        for (String value : new String[]{"a\"\"1b", "a\"\\2\t", "a\"\n3\u00e9"}) {
            assertTrue(value, GrammarMatcher.matchesUtf8(grammar, value));
        }
        for (String value : new String[]{"\"a\"\"1b\"", "a\\\"\"1b", "a\"x1b", "a\"\"1\n", "a\"\"1\u2028", "xa\"\"1b"}) {
            assertFalse(value, GrammarMatcher.matchesUtf8(grammar, value));
        }
        try {
            RegexToGbnf.rawValueRule("[]", "root", new GbnfBuilder());
            throw new AssertionError("an empty class is refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("character class [] matches no character", e.getMessage());
        }
    }

    // -------------------------------------------------- derived: {@link RegexToGbnf#compile} is
    // the entry point a standalone regex constraint goes through, so its result must parse
    // and validate as a full grammar

    @Test
    public void compileProducesAValidatedGrammar() {
        Grammar grammar = RegexToGbnf.compile("^[0-9]*$", "root", false);
        assertEquals(grammar.symbolId("root"), grammar.rootRuleIndex());
        assertTrue(grammar.ruleCount() >= 2);
        assertTrue(grammar.rule(grammar.rootRuleIndex()).size() >= 2);
    }

    @Test
    public void aStandaloneRegexMatchesTheRawOutputWithOrWithoutAnchors() {
        for (String pattern : new String[]{"^[0-9]+-[a-z]{2}$", "[0-9]+-[a-z]{2}"}) {
            Grammar grammar = RegexToGbnf.compile(pattern, "root", false);
            assertTrue(pattern, GrammarMatcher.matches(grammar, "123-ab"));
            assertFalse(pattern + " writes no JSON quotes", GrammarMatcher.matches(grammar, "\"123-ab\""));
            assertFalse(pattern, GrammarMatcher.matches(grammar, "123-abc"));
        }
        assertTrue("an escaped dollar stays a literal",
                GrammarMatcher.matches(RegexToGbnf.compile("a\\$", "root", false), "a$"));
    }

    // -------------------------------------------------- refusals (anything outside the
    // subset is refused with a message naming the construct)

    // -------------------------------------------------- LC test "unanchored regexp" (:1448-1461): llama.cpp
    // degrades it to any string; xgrammar matches the whole value, anchored or not

    @Test
    public void unanchoredPatternMatchesTheWholeValue() {
        for (String pattern : new String[]{"[0-9]+", "^[0-9]+", "[0-9]+$", "^[0-9]+$"}) {
            Grammar grammar = valueGrammar(pattern);
            assertTrue(pattern, GrammarMatcher.matches(grammar, "\"123\""));
            assertFalse(pattern, GrammarMatcher.matches(grammar, "\"a123\""));
            assertFalse(pattern, GrammarMatcher.matches(grammar, "\"123a\""));
        }
    }

    // -------------------------------------------------- LC test "regexp with unsupported shorthand"
    // (:1464-1477): llama.cpp degrades it to any string; xgrammar reads the shorthand classes

    @Test
    public void shorthandClassesMatchTheirEcmaScriptSets() {
        Grammar grammar = valueGrammar("^[0-9]{3}\\w$");
        assertTrue(GrammarMatcher.matches(grammar, "\"123_\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"123-\""));
        Grammar spaces = valueGrammar("^a\\sb\\S$");
        assertTrue(GrammarMatcher.matches(spaces, "\"a b\\\"\""));
        assertTrue("a matched tab is written as its escape", GrammarMatcher.matches(spaces, "\"a\\tbc\""));
        assertFalse("a raw tab is no JSON string character", GrammarMatcher.matches(spaces, "\"a\tbc\""));
        assertFalse(GrammarMatcher.matches(spaces, "\"a b \""));
    }

    @Test
    public void controlCharactersInAClassAreWrittenAsTheirEscapes() {
        Grammar grammar = valueGrammar("^[\\x00-\\x1F]$");
        assertTrue(GrammarMatcher.matches(grammar, "\"\\n\""));
        assertTrue(GrammarMatcher.matches(grammar, "\"\\u0001\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"\\u0020\""));
        assertFalse(GrammarMatcher.matches(grammar, "\" \""));
    }

    @Test
    public void emptyPatternMatchesTheEmptyString() {
        Grammar grammar = valueGrammar("");
        assertTrue(GrammarMatcher.matches(grammar, "\"\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"a\""));
    }

    @Test
    public void anchorInsideThePatternIsRefused() {
        try {
            translated("a^b");
            fail("a mid-pattern anchor must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("anchor inside the pattern", e.getMessage());
        }
    }

    @Test
    public void lookaheadGroupIsRefused() {
        try {
            translated("^(?=a)a$");
            fail("a lookahead must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("lookahead assertion: (?=", e.getMessage());
        }
    }

    @Test
    public void keyRulesWriteMatchedCharactersJsonEncoded() {
        GbnfBuilder builder = new GbnfBuilder();
        RegexToGbnf.keyRule("^a\"\\\\[\"x]$", "root", builder);
        assertEquals(qEscape() + " (" + q("a\\\\\\\"\\\\\\\\") + " (" + "[x] | " + q("\\\\\\\"") + ")) "
                + qEscape(), builder.rule("root"));
        Grammar grammar = builder.grammar();
        assertTrue(GrammarMatcher.matches(grammar, "\"a\\\"\\\\x\""));
        assertTrue(GrammarMatcher.matches(grammar, "\"a\\\"\\\\\\\"\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"a\"\\x\""));
    }

    @Test
    public void keyRulesSearchOnUnanchoredSides() {
        GbnfBuilder builder = new GbnfBuilder();
        RegexToGbnf.keyRule("id", "root", builder);
        Grammar grammar = builder.grammar();
        assertTrue(GrammarMatcher.matches(grammar, "\"user_id\""));
        assertTrue(GrammarMatcher.matches(grammar, "\"id \\\" more\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"user\""));
    }

    @Test
    public void keyRuleAnchorsBindTheirOwnTopLevelAlternative() {
        Grammar grammar = keyGrammar("^a|b$|^$");
        for (String key : new String[]{"\"a\"", "\"ax\"", "\"b\"", "\"xb\"", "\"\""}) {
            assertTrue(key, GrammarMatcher.matches(grammar, key));
        }
        for (String key : new String[]{"\"xa\"", "\"bx\"", "\"x\""}) {
            assertFalse(key, GrammarMatcher.matches(grammar, key));
        }
        try {
            RegexToGbnf.keyRule("(^a)", "root", new GbnfBuilder());
            fail("an anchor inside a group must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("anchor inside the pattern", e.getMessage());
        }
    }

    @Test
    public void keyRulesReadTheFullDialect() {
        Grammar grammar = keyGrammar("^(?<p>x)-\\d{2,}?\\s?\\w{,2}$");
        assertTrue(GrammarMatcher.matches(grammar, "\"x-12\""));
        assertTrue(GrammarMatcher.matches(grammar, "\"x-123 ab\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"x-1\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"x-12abc\""));
        Grammar notDigit = keyGrammar("^\\D$");
        assertTrue(GrammarMatcher.matches(notDigit, "\"\\\"\""));
        assertTrue(GrammarMatcher.matches(notDigit, "\"\\\\\""));
        assertFalse(GrammarMatcher.matches(notDigit, "\"7\""));
        Grammar any = keyGrammar("^[^]$");
        assertTrue(GrammarMatcher.matches(any, "\"\\\"\""));
        try {
            RegexToGbnf.keyRule("a(?=b)", "root", new GbnfBuilder());
            fail("a lookahead must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("lookahead assertion: (?=", e.getMessage());
        }
        try {
            RegexToGbnf.keyRule("^a[]$", "root", new GbnfBuilder());
            fail("ECMAScript's empty class holds no key character");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("character class [] matches no character", e.getMessage());
        }
    }

    private static Grammar keyGrammar(String pattern) {
        GbnfBuilder builder = new GbnfBuilder();
        RegexToGbnf.keyRule(pattern, "root", builder);
        return builder.grammar();
    }

    @Test
    public void keyRuleClassesOfControlCharactersWriteTheirEscapes() {
        Grammar grammar = keyGrammar("^[\\x00-\\x1F]$");
        assertTrue(GrammarMatcher.matches(grammar, "\"\\t\""));
        assertTrue(GrammarMatcher.matches(grammar, "\"\\u001f\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"a\""));
    }

    @Test
    public void searchesFollowsEcmaScriptClassesAndEndAnchor() {
        assertTrue(RegexToGbnf.searches("[[]", "a[b"));
        assertTrue(RegexToGbnf.searches("^[a&&b]$", "&"));
        assertFalse(RegexToGbnf.searches("^a$", "a\n"));
        assertTrue(RegexToGbnf.searches("^\\U0001F600$", new String(Character.toChars(0x1F600))));
        assertFalse(RegexToGbnf.searches("[]", "a"));
        assertTrue(RegexToGbnf.searches("^[^]$", "\n"));
        try {
            RegexToGbnf.searches("(a", "a");
            fail("an unbalanced pattern is invalid");
        } catch (RegexToGbnf.InvalidRegexException e) {
            assertTrue(e.getMessage(), !e.getMessage().isEmpty());
        }
    }

    @Test
    public void searchesReadsTheFullDialect() {
        assertTrue(RegexToGbnf.searches("^(?<n>[a-z]+)_\\d{2,}?$", "user_42"));
        assertFalse(RegexToGbnf.searches("^(?<n>[a-z]+)_\\d{2,}?$", "user_4"));
        assertTrue(RegexToGbnf.searches("^a|b$", "xb"));
        assertFalse(RegexToGbnf.searches("^a|b$", "ba"));
        assertTrue(RegexToGbnf.searches("a{,2}c\\.", "aac."));
        assertTrue(RegexToGbnf.searches("x{y", "x{y"));
        assertFalse(RegexToGbnf.searches("^a.b$", "a b"));
        assertTrue(RegexToGbnf.searches("\\S\\s", "\"\t"));
        try {
            RegexToGbnf.searches("\\ba", "a");
            fail("a word boundary must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("word boundary assertion: \\b", e.getMessage());
        }
    }

    @Test
    public void wordBoundaryIsRefused() {
        try {
            translated("^\\bfoo$");
            fail("a \\b assertion must be refused, as xgrammar refuses it");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("word boundary assertion: \\b", e.getMessage());
        }
    }

    @Test
    public void nestingBeyondMaxPatternDepthIsRefusedInEveryUse() {
        assertEquals("llama.cpp's COMMON_RECURSIVE_LIMIT", 128, RegexToGbnf.MAX_PATTERN_DEPTH);
        String deepest = nested(RegexToGbnf.MAX_PATTERN_DEPTH);
        assertTrue(GrammarMatcher.matches(valueGrammar(deepest), "\"a\""));
        String tooDeep = nested(RegexToGbnf.MAX_PATTERN_DEPTH + 1);
        Runnable[] uses = {
            () -> translated(tooDeep),
            () -> RegexToGbnf.keyRule(tooDeep, "root", new GbnfBuilder()),
            () -> RegexToGbnf.compile(tooDeep, "root", false),
            () -> RegexToGbnf.searches(tooDeep, "a")};
        for (Runnable use : uses) {
            try {
                use.run();
                fail("patterns deeper than MAX_PATTERN_DEPTH must be refused");
            } catch (RegexToGbnf.UnsupportedRegexException e) {
                assertEquals("pattern nesting too deep", e.getMessage());
            }
        }
    }

    private static String nested(int depth) {
        StringBuilder pattern = new StringBuilder("^");
        for (int i = 0; i < depth; i++) {
            pattern.append("(?:");
        }
        pattern.append('a');
        for (int i = 0; i < depth; i++) {
            pattern.append(')');
        }
        return pattern.append('$').toString();
    }

    // -------------------------------------------------- the invalid (not merely unsupported)
    // patterns: an error, never a graceful fallback

    @Test
    public void unbalancedParenthesesAreInvalid() {
        try {
            translated("^(a");
            fail("unbalanced parentheses must fail");
        } catch (RegexToGbnf.InvalidRegexException e) {
            assertEquals("unbalanced parentheses", e.getMessage());
        }
    }

    @Test
    public void unterminatedCharacterClassIsInvalid() {
        try {
            translated("^[a-z$");
            fail("an unterminated class must fail");
        } catch (RegexToGbnf.InvalidRegexException e) {
            assertEquals("unterminated character class", e.getMessage());
        }
    }

    @Test
    public void aBraceThatStartsNoQuantifierIsALiteral() {
        Grammar grammar = valueGrammar("^a{$");
        assertTrue(GrammarMatcher.matches(grammar, "\"a{\""));
        assertFalse(GrammarMatcher.matches(grammar, "\"a\""));
    }

    @Test
    public void outOfOrderBoundsAreInvalid() {
        try {
            translated("a{2,1}");
            fail("out-of-order bounds must fail, as xgrammar refuses them");
        } catch (RegexToGbnf.InvalidRegexException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("numbers out of order"));
        }
    }
}
