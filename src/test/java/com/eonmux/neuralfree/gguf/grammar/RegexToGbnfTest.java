package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests of the ECMAScript subset to GBNF translator ({@link RegexToGbnf}), the Java port of
 * llama.cpp {@code common_chat_schema_converter::_pattern_to_rule}. The golden cases are the
 * regexp test cases of llama.cpp's {@code tests/test-json-schema-to-grammar.cpp} (the C++ test
 * name is cited per test); each translated rule text is asserted on the {@link GbnfBuilder} the
 * way the C++ test asserts the emitted grammar. The quantifier and refusal cases are derived from
 * the converter's own scan logic (the subset's operators {@code * + ? {m} {m,} {m,n}} and its
 * refusals, each naming the construct). Pure logic; no model file required.
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

    /** {@code patternRule} into a fresh builder; returns the builder for rule-text asserts. */
    private static GbnfBuilder translated(String pattern) {
        return translated(pattern, false);
    }

    private static GbnfBuilder translated(String pattern, boolean dotAll) {
        GbnfBuilder builder = new GbnfBuilder();
        RegexToGbnf.patternRule(pattern, "root", builder, dotAll);
        return builder;
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
    // repetitions hoist the class into a helper rule, and the dot becomes its own rule

    @Test
    public void hoistsClassesAndDotRules() {
        GbnfBuilder builder = translated("^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$");
        assertEquals("[^\\x0A\\x0D]", builder.rule("dot"));
        assertEquals("[0-9]", builder.rule("root-1"));
        assertEquals(qEscape() + " (" + "(" + q("(") + " root-1{1,3} " + q(")") + ")?"
                        + " root-1{3,3} " + q("-") + " root-1{4,4} " + q(" ") + " "
                        + q("a") + "{3,5} " + q("nd") + " dot dot dot) " + qEscape(),
                builder.rule("root"));
    }

    // -------------------------------------------------- LC test "regexp escapes" (:755-766)

    @Test
    public void escapesOfOperatorsBecomeOneLiteral() {
        assertEquals(qEscape() + " (" + q("[]{}()|+*?") + ") " + qEscape(),
                translated("^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$").rule("root"));
    }

    // -------------------------------------------------- LC test "regexp quote" (:768-779)

    @Test
    public void quoteInsideThePattern() {
        assertEquals(qEscape() + " (" + q("\\\"") + ") " + qEscape(),
                translated("^\"$").rule("root"));
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

    // -------------------------------------------------- derived from the converter's own scan: the
    // subset has no lazy operator; llama.cpp's scan treats "+" and "?" as two stacked
    // quantifiers on the same item, exactly as this port does

    @Test
    public void plusFollowedByQuestionMarkStacksTwoQuantifiers() {
        assertEquals(qEscape() + " (" + q("a") + " " + q("b") + "+?) " + qEscape(),
                translated("^ab+?$").rule("root"));
    }

    // -------------------------------------------------- LC test "regexp with escaped hyphen in
    // a character class" (:1479-1490): the class with its \\- escape is copied verbatim

    @Test
    public void escapedHyphenInCharacterClass() {
        assertEquals(qEscape() + " (" + "[a-z\\-]" + "+) " + qEscape(),
                translated("^[a-z\\-]+$").rule("root"));
    }

    // -------------------------------------------------- LC test "regexp with escaped hyphen
    // outside a character class" (:1492-1503): the \\- GBNF escape passes through verbatim

    @Test
    public void escapedHyphenOutsideCharacterClass() {
        assertEquals(qEscape() + " (" + q("a\\-b") + ") " + qEscape(),
                translated("^a\\-b$").rule("root"));
    }

    // -------------------------------------------------- derived: the dotAll flag (llama.cpp's
    // get_dot, the {@code common_grammar_options.dotall} mirror) changes the dot rule only

    @Test
    public void dotAllFlagWidensTheDotRule() {
        GbnfBuilder builder = translated("^.a$", true);
        assertEquals("[\\U00000000-\\U0010FFFF]", builder.rule("dot"));
        assertEquals(qEscape() + " (dot " + q("a") + ") " + qEscape(), builder.rule("root"));
        GbnfBuilder defaultBuilder = translated("^.a$");
        assertEquals("[^\\x0A\\x0D]", defaultBuilder.rule("dot"));
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

    // -------------------------------------------------- refusals (WP clause: anything outside the
    // subset is refused with a message naming the construct)

    @Test
    public void unanchoredPatternIsRefused() {
        try {
            translated("[0-9]+");
            fail("an unanchored pattern must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("not anchored with '^' and '$'", e.getMessage());
        }
    }

    @Test
    public void anchorInsideThePatternIsRefused() {
        try {
            translated("^^a$");
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
            assertEquals("unsupported group syntax", e.getMessage());
        }
    }

    @Test
    public void unsupportedShorthandEscapeIsRefused() {
        try {
            translated("^\\bfoo$");
            fail("a \\b shorthand must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("unsupported escape: \\b", e.getMessage());
        }
    }

    @Test
    public void nestingBeyondMaxPatternDepthIsRefused() {
        StringBuilder pattern = new StringBuilder("^");
        for (int i = 0; i <= RegexToGbnf.MAX_PATTERN_DEPTH; i++) {
            pattern.append("(?:");
        }
        pattern.append('a');
        for (int i = 0; i <= RegexToGbnf.MAX_PATTERN_DEPTH; i++) {
            pattern.append(')');
        }
        pattern.append('$');
        try {
            translated(pattern.toString());
            fail("patterns deeper than MAX_PATTERN_DEPTH must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("pattern nesting too deep", e.getMessage());
        }
    }

    // -------------------------------------------------- the invalid (not merely unsupported)
    // patterns: an error, never a graceful fallback

    @Test
    public void unbalancedParenthesesAreInvalid() {
        try {
            translated("^(a$");
            fail("unbalanced parentheses must fail");
        } catch (RegexToGbnf.InvalidRegexException e) {
            assertEquals("unbalanced parentheses", e.getMessage());
        }
    }

    @Test
    public void unterminatedCharacterClassIsInvalid() {
        try {
            translated("^[a-$");
            fail("an unterminated class must fail");
        } catch (RegexToGbnf.InvalidRegexException e) {
            assertEquals("unterminated character class", e.getMessage());
        }
    }

    @Test
    public void unterminatedCurlyBracketsAreRefused() {
        try {
            translated("^a{$");
            fail("unterminated curly brackets must be refused");
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            assertEquals("unterminated curly brackets", e.getMessage());
        }
    }
}
