package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests of the GBNF parser and grammar model ({@link GbnfParser}, {@link Grammar}), ported from
 * llama.cpp's {@code llama_grammar_parser}. The acceptance grammars are the llama.cpp
 * {@code grammars/*.gbnf} files, copied as test resources; invariants (rule shapes of the
 * repetition expansion, escapes, token references, left recursion, undefined rules, root rule and
 * the untrusted-input limits) are asserted against exact rule layouts. Pure logic; no GPU or
 * model file required.
 */
public class GbnfParserTest {

    private static final Path GRAMMARS = Paths.get("src", "test", "resources", "grammars");
    private static final String[] SAMPLES = {"arithmetic.gbnf", "c.gbnf", "chess.gbnf", "english.gbnf",
            "japanese.gbnf", "json_arr.gbnf", "json.gbnf", "list.gbnf"};

    private static String read(String name) throws IOException {
        return new String(Files.readAllBytes(GRAMMARS.resolve(name)), StandardCharsets.UTF_8);
    }

    private static List<GrammarElement> parse(String source) {
        return GbnfParser.parse(source).rule(0);
    }

    private static GrammarElement element(GrammarElement.Type type, int value) {
        return new GrammarElement(type, value);
    }

    private static GrammarException parseFails(String source) {
        try {
            GbnfParser.parse(source);
        } catch (GrammarException e) {
            return e;
        }
        throw new AssertionError("expected the grammar to fail: " + source);
    }

    // ------------------------------------------------------------------ acceptance resources

    @Test
    public void parsesAllLlamaCppGrammarResources() throws IOException {
        for (String name : SAMPLES) {
            Grammar grammar = GbnfParser.parse(read(name));
            assertTrue(name, grammar.ruleCount() >= 3);
            assertTrue(name, grammar.symbolId("root") >= 0);
            assertEquals(name, grammar.symbolId("root"), grammar.rootRuleIndex());
            List<GrammarElement> root = grammar.rule(grammar.rootRuleIndex());
            assertTrue(name, root.size() >= 2); // at least one element plus END
            for (int i = 0; i < grammar.ruleCount(); i++) {
                assertEquals(name, grammar.symbolId(grammar.symbolName(i)), i);
                for (GrammarElement e : grammar.rule(i)) {
                    if (e.type() == GrammarElement.Type.RULE_REF) {
                        assertTrue(name + " rule " + i + " -> " + e.value(), e.value() < grammar.ruleCount());
                    }
                }
            }
        }
    }

    @Test
    public void jsonGrammarRuleShapes() throws IOException {
        Grammar grammar = GbnfParser.parse(read("json.gbnf"));
        assertEquals(GrammarElement.Type.RULE_REF, grammar.rule(0).get(0).type());
        assertEquals("object", grammar.symbolName(grammar.rule(0).get(0).value()));
        // ws ::= | " " | "\n" [ \t]{0,20}: an empty first alternate, so it may be empty.
        List<GrammarElement> ws = grammar.rule(grammar.symbolId("ws"));
        assertEquals(element(GrammarElement.Type.ALT, 0), ws.get(0));
        // string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (...) )* "\"" ws: the literal opens it.
        List<GrammarElement> string = grammar.rule(grammar.symbolId("string"));
        assertEquals(element(GrammarElement.Type.CHAR, '"'), string.get(0));
        boolean sawNotQuote = false;
        for (int i = 0; i < grammar.ruleCount(); i++) {
            for (GrammarElement e : grammar.rule(i)) {
                sawNotQuote |= e.type() == GrammarElement.Type.CHAR_NOT && e.value() == '"';
            }
        }
        assertTrue(sawNotQuote);
    }

    // ------------------------------------------------------------------ rule shapes

    @Test
    public void literalStringBecomesCharElements() {
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.CHAR, 'b'),
                element(GrammarElement.Type.END, 0)), parse("root ::= \"ab\""));
    }

    @Test
    public void charRangeAndAlternatesShape() {
        // [a-cx-zA0] -> a, a-c, x-z, A, 0; '-' at the very end is an alternate literal.
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.CHAR_RNG_UPPER, 'c'),
                element(GrammarElement.Type.CHAR_ALT, 'x'),
                element(GrammarElement.Type.CHAR_RNG_UPPER, 'z'),
                element(GrammarElement.Type.CHAR_ALT, 'A'),
                element(GrammarElement.Type.CHAR_ALT, '0'),
                element(GrammarElement.Type.END, 0)), parse("root ::= [a-cx-zA0]"));
    }

    @Test
    public void invertedRangeShape() {
        assertEquals(List.of(element(GrammarElement.Type.CHAR_NOT, '\n'),
                element(GrammarElement.Type.CHAR_RNG_UPPER, '\r'),
                element(GrammarElement.Type.END, 0)), parse("root ::= [^\\n-\\r]"));
    }

    @Test
    public void anyCharacterShape() {
        assertEquals(List.of(element(GrammarElement.Type.CHAR_ANY, 0),
                element(GrammarElement.Type.END, 0)), parse("root ::= ."));
    }

    @Test
    public void escapeFamiliesDecodeToCodePoints() {
        List<GrammarElement> expected = new ArrayList<>();
        // a quoted literal is a sequence of CHAR elements (llama-grammar.cpp parse_sequence); CHAR_ALT only
        // continues a [...] class
        expected.add(element(GrammarElement.Type.CHAR, 0x09));
        expected.add(element(GrammarElement.Type.CHAR, 0x0D));
        expected.add(element(GrammarElement.Type.CHAR, 0x0A));
        expected.add(element(GrammarElement.Type.CHAR, 0x41));
        expected.add(element(GrammarElement.Type.CHAR, 0x263A));
        expected.add(element(GrammarElement.Type.CHAR, 0x1F600));
        expected.add(element(GrammarElement.Type.END, 0));
        assertEquals(expected, parse("root ::= \"\\t\\r\\n\\x41\\u263A\\U0001F600\""));
    }

    @Test
    public void dashAndSpecialEscapesInRanges() {
        // [\\\"a\\]] -> literal backslash, quote, 'a', ']' as one class
        assertEquals(List.of(element(GrammarElement.Type.CHAR, '\\'),
                element(GrammarElement.Type.CHAR_ALT, '"'),
                element(GrammarElement.Type.CHAR_ALT, 'a'),
                element(GrammarElement.Type.CHAR_ALT, ']'),
                element(GrammarElement.Type.END, 0)), parse("root ::= [\\\\\\\"a\\]]"));
    }

    @Test
    public void tokenReferenceShapes() {
        assertEquals(List.of(element(GrammarElement.Type.TOKEN, 42),
                element(GrammarElement.Type.TOKEN_NOT, 3),
                element(GrammarElement.Type.END, 0)), parse("root ::= <[42]> !<[3]>"));
    }

    @Test
    public void ruleReferencesAreNamed() {
        Grammar grammar = GbnfParser.parse("root ::= tail\ntail ::= \"t\"");
        assertEquals(2, grammar.ruleCount());
        assertEquals("tail", grammar.symbolName(grammar.rule(0).get(0).value()));
        assertEquals(GrammarElement.Type.CHAR, grammar.rule(1).get(0).type());
    }

    // ------------------------------------------------------------------ repetition expansion

    @Test
    public void boundedRepetitionExpansionShape() {
        // root ::= "a"{2,3} -> rule(0) = "a" "a" root_1; root_1 ::= "a" |
        Grammar grammar = GbnfParser.parse("root ::= \"a\"{2,3}");
        assertEquals(2, grammar.ruleCount());
        assertEquals(element(GrammarElement.Type.RULE_REF, grammar.symbolId("root_1")),
                grammar.rule(0).get(2));
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.ALT, 0),
                element(GrammarElement.Type.END, 0)), grammar.rule(1));
    }

    @Test
    public void starRepetitionExpansionShape() {
        // root ::= "a"* -> rule(0) = root_1; root_1 ::= "a" root_1 |
        Grammar grammar = GbnfParser.parse("root ::= \"a\"*");
        assertEquals(2, grammar.ruleCount());
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.RULE_REF, 1),
                element(GrammarElement.Type.ALT, 0),
                element(GrammarElement.Type.END, 0)), grammar.rule(1));
    }

    @Test
    public void plusRepetitionExpansionShape() {
        // root ::= "a"+ -> rule(0) = "a" root_1 (one char is required, the rest recurses)
        Grammar grammar = GbnfParser.parse("root ::= \"a\"+");
        assertEquals(2, grammar.ruleCount());
        List<GrammarElement> root = grammar.rule(0);
        assertEquals(element(GrammarElement.Type.CHAR, 'a'), root.get(0));
        assertEquals(element(GrammarElement.Type.RULE_REF, 1), root.get(1));
    }

    @Test
    public void optionalRepetitionExpansionShape() {
        // root ::= "a"? -> rule(0) = root_1; root_1 ::= "a" | (optional: zero or one)
        Grammar grammar = GbnfParser.parse("root ::= \"a\"?");
        assertEquals(2, grammar.ruleCount());
        assertEquals(List.of(element(GrammarElement.Type.RULE_REF, 1),
                element(GrammarElement.Type.END, 0)), grammar.rule(0));
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.ALT, 0),
                element(GrammarElement.Type.END, 0)), grammar.rule(1));
    }

    @Test
    public void repetitionOverGroupShape() {
        // root ::= "r" ("x" | "y")? -> the group becomes root_1, the optional becomes root_2, which replaces
        // the group's reference in root (handle_repetitions truncates the sequence when min_times is 0)
        Grammar grammar = GbnfParser.parse("root ::= \"r\" (\"x\" | \"y\")?");
        assertEquals(3, grammar.ruleCount());
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'r'),
                element(GrammarElement.Type.RULE_REF, grammar.symbolId("root_2")),
                element(GrammarElement.Type.END, 0)), grammar.rule(0));
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'x'),
                element(GrammarElement.Type.ALT, 0),
                element(GrammarElement.Type.CHAR, 'y'),
                element(GrammarElement.Type.END, 0)), grammar.rule(grammar.symbolId("root_1")));
        assertEquals(List.of(element(GrammarElement.Type.RULE_REF, grammar.symbolId("root_1")),
                element(GrammarElement.Type.ALT, 0),
                element(GrammarElement.Type.END, 0)), grammar.rule(grammar.symbolId("root_2")));
    }

    @Test
    public void nestedUnboundedRepetitionShape() {
        // root ::= ("a" "b")* -> root ::= root_2; root_1 ::= "a" "b"; root_2 ::= root_1 root_2 |
        Grammar grammar = GbnfParser.parse("root ::= (\"a\" \"b\")*");
        assertEquals(3, grammar.ruleCount());
        assertEquals("root_1", grammar.symbolName(1));
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.CHAR, 'b'),
                element(GrammarElement.Type.END, 0)), grammar.rule(1));
        assertEquals(List.of(element(GrammarElement.Type.RULE_REF, grammar.symbolId("root_1")),
                element(GrammarElement.Type.RULE_REF, grammar.symbolId("root_2")),
                element(GrammarElement.Type.ALT, 0),
                element(GrammarElement.Type.END, 0)), grammar.rule(2));
        assertEquals(element(GrammarElement.Type.RULE_REF, grammar.symbolId("root_2")),
                grammar.rule(0).get(0));
    }

    @Test
    public void largeMaxFallsBackToUnbounded() {
        // {2,10001}: n above the limit snaps to unbounded, so root_1 recurses.
        Grammar grammar = GbnfParser.parse("root ::= \"a\"{2,10001}");
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.RULE_REF, grammar.symbolId("root_1")),
                element(GrammarElement.Type.ALT, 0),
                element(GrammarElement.Type.END, 0)), grammar.rule(grammar.symbolId("root_1")));
    }

    @Test
    public void exactCountRepetitionNeedsNoExpansionRules() {
        // {3}: max == min, so n_opt == 0: exactly three chars, no synthesized rule.
        Grammar grammar = GbnfParser.parse("root ::= \"a\"{3}");
        assertEquals(1, grammar.ruleCount());
        assertEquals(List.of(element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.CHAR, 'a'),
                element(GrammarElement.Type.END, 0)), grammar.rule(0));
    }

    // ------------------------------------------------------------------ errors and limits

    @Test
    public void undefinedRuleIsNamedWithItsReferencePosition() {
        GrammarException e = parseFails("root ::= numeric");
        assertTrue(e.getMessage(), e.getMessage().contains("Undefined rule identifier 'numeric'"));
        assertEquals(1, e.getLine());
        assertEquals(10, e.getColumn());
    }

    @Test
    public void undefinedRuleOnLaterLineCitesThatLine() {
        GrammarException e = parseFails("root ::= \"a\"\nmore ::= b\n");
        assertTrue(e.getMessage(), e.getMessage().contains("Undefined rule identifier 'b'"));
        assertEquals(2, e.getLine());
        assertEquals(10, e.getColumn());
    }

    @Test
    public void repeatedOperatorWithoutPrecedingItemFails() {
        GrammarException e = parseFails("root ::= *");
        assertTrue(e.getMessage(), e.getMessage().contains("expecting preceding item to */+/?/{"));
    }

    @Test
    public void unknownEscapeFails() {
        GrammarException e = parseFails("root ::= \"a\\qb\"");
        assertTrue(e.getMessage(), e.getMessage().contains("unknown escape"));
    }

    @Test
    public void shortHexEscapeFails() {
        GrammarException e = parseFails("root ::= \"\\xZZ\"");
        assertTrue(e.getMessage(), e.getMessage().contains("expecting 2 hex chars"));
    }

    @Test
    public void missingDefinitionOperatorFails() {
        GrammarException e = parseFails("root \"a\"");
        assertTrue(e.getMessage(), e.getMessage().contains("expecting '::='"));
    }

    @Test
    public void twoRulesOnOneLineFail() {
        GrammarException e = parseFails("root ::= \"a\" other ::= \"b\"");
        assertTrue(e.getMessage(), e.getMessage().contains("expecting newline or end"));
        assertEquals(1, e.getLine());
    }

    @Test
    public void unterminatedLiteralAndRangeFail() {
        assertTrue(parseFails("root ::= \"abc").getMessage().contains("unexpected end of input"));
        assertTrue(parseFails("root ::= [a").getMessage().contains("unexpected end of input"));
        assertTrue(parseFails("root ::= [a-").getMessage().contains("unexpected end of input"));
    }

    @Test
    public void malformedRepetitionBoundsFail() {
        assertTrue(parseFails("root ::= (\"a\"){5").getMessage().contains("expecting ','"));
        assertTrue(parseFails("root ::= (\"a\"){5,2").getMessage().contains("expecting '}'"));
        assertTrue(parseFails("root ::= (\"a\"){}").getMessage().contains("expecting an int"));
        assertTrue(parseFails("root ::= (\"a\"){2").getMessage().contains("expecting ','"));
        assertTrue(parseFails("root ::= (\"a\"){").getMessage().contains("expecting an int"));
    }

    @Test
    public void minimumRepetitionAboveLimitFails() {
        GrammarException e = parseFails("root ::= \"a\"{10001}");
        assertTrue(e.getMessage(), e.getMessage().contains("number of repetitions exceeds sane defaults"));
    }

    @Test
    public void repetitionProductAboveLimitFails() {
        // "a"{101} builds 101 copies; a further {100} multiplies 101 * 100 > 10000.
        GrammarException e = parseFails("root ::= \"a\"{101}{100}");
        assertTrue(e.getMessage(),
                e.getMessage().contains("multiplied by the new repetition exceeds sane defaults"));
    }

    @Test
    public void missingRootRuleFails() {
        GrammarException e = parseFails("foo ::= \"a\"");
        assertTrue(e.getMessage(), e.getMessage().contains("grammar does not contain a 'root' symbol"));
    }

    @Test
    public void emptyAndCommentOnlyGrammarsFail() {
        assertTrue(parseFails("").getMessage().contains("failed to parse grammar"));
        assertTrue(parseFails("# nothing here\n# more\n").getMessage().contains("failed to parse grammar"));
    }

    @Test
    public void tokenByNameNeedsTheVocabulary() {
        GrammarException e = parseFails("root ::= <volume>");
        assertTrue(e.getMessage(), e.getMessage().contains("no vocab to parse token"));
        assertEquals(1, e.getLine());
    }

    @Test
    public void tokenByNameResolvesThroughTheVocabularyToExactlyOneToken() {
        GrammarVocab vocab = text -> text.equals("<|channel|>") ? new int[]{7}
                : text.equals("<two>") ? new int[]{1, 2} : new int[0];
        assertEquals(List.of(element(GrammarElement.Type.TOKEN, 7),
                element(GrammarElement.Type.TOKEN_NOT, 7),
                element(GrammarElement.Type.END, 0)),
                GbnfParser.parse("root ::= <|channel|> !<|channel|>", vocab).rule(0));
        try {
            GbnfParser.parse("root ::= <two>", vocab);
            fail("a reference of two tokens is refused");
        } catch (GrammarException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("invalid token '<two>'"));
        }
        try {
            GbnfParser.parse("root ::= <|channel|", vocab);
            fail("an unterminated reference is refused");
        } catch (GrammarException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("expecting '>'"));
        }
    }

    @Test
    public void overflowTokenIdFails() {
        GrammarException e = parseFails("root ::= <[99999999999]>");
        assertTrue(e.getMessage(), e.getMessage().contains("parsed token id is too big"));
    }

    @Test
    public void sourceAboveTheCharacterLimitFails() {
        StringBuilder big = new StringBuilder();
        for (int i = 0; i < GrammarLimits.MAX_SOURCE_CHARS + 1; i++) {
            big.append('a');
        }
        GrammarException e = parseFails(big.toString());
        assertTrue(e.getMessage(),
                e.getMessage().contains("characters, more than " + GrammarLimits.MAX_SOURCE_CHARS));
    }

    @Test
    public void ruleCountAboveLimitFails() {
        StringBuilder many = new StringBuilder("root ::= ");
        for (int i = 0; i < GrammarLimits.MAX_RULES; i++) {
            many.append("(a)");
        }
        GrammarException e = parseFails(many.toString());
        assertTrue(e.getMessage(), e.getMessage().contains("more than " + GrammarLimits.MAX_RULES));
    }

    // ------------------------------------------------------------------ recursion checks

    @Test
    public void directLeftRecursionIsDetected() {
        GrammarException e = parseFails("root ::= root\r");
        assertTrue(e.getMessage(), e.getMessage().contains("left recursion detected for nonterminal 'root'"));
        assertEquals(1, e.getLine());
    }

    @Test
    public void indirectLeftRecursionIsDetected() {
        GrammarException e = parseFails("root ::= a\na ::= root \"x\" | \"b\"");
        assertTrue(e.getMessage(),
                e.getMessage().contains("left recursion detected for nonterminal 'root'"));
        assertEquals(1, e.getLine());
    }

    @Test
    public void leftRecursionThroughMaybeEmptyRuleIsDetected() {
        GrammarException e = parseFails("root ::= e | \"x\"\ne ::= root?");
        assertTrue(e.getMessage(), e.getMessage().contains("left recursion detected for nonterminal 'root'"));
    }

    @Test
    public void horizontalRecursionIsNotLeftRecursion() {
        // llama.cpp's detector allows these: the reference is not leftmost.
        GbnfParser.parse("root ::= \"a\" root");
        GbnfParser.parse("root ::= e \"x\"\ne ::= \"y\"?");
    }

    @Test
    public void nonRecursiveCyclesAreFine() {
        GbnfParser.parse("root ::= tail\ntail ::= (\"a\" tail)?");
        GbnfParser.parse("root ::= a\na ::= b\nb ::= \"b\"");
    }

    // ------------------------------------------------------------------ whitespace and comments

    @Test
    public void commentsAndBlankLinesAreIgnored() {
        Grammar grammar = GbnfParser.parse("# leading\n\n  root  ::=  \"a\"   # trailing\n\n# done\n");
        assertEquals(1, grammar.ruleCount());
        assertEquals(element(GrammarElement.Type.CHAR, 'a'), grammar.rule(0).get(0));
    }

    // ------------------------------------------------------------------ grammar model surface

    @Test
    public void symbolIdsRoundTrip() throws IOException {
        Grammar grammar = GbnfParser.parse(read("json.gbnf"));
        for (int i = 0; i < grammar.ruleCount(); i++) {
            assertEquals(i, grammar.symbolId(grammar.symbolName(i)));
        }
        assertEquals(-1, grammar.symbolId("nosuchrule"));
    }

    @Test
    public void rulesAreImmutable() throws IOException {
        Grammar grammar = GbnfParser.parse(read("list.gbnf"));
        try {
            grammar.rule(0).add(element(GrammarElement.Type.CHAR, 'z'));
            fail("rule() must not return a mutable list");
        } catch (UnsupportedOperationException expected) {
            // immutable model
        }
    }
}
