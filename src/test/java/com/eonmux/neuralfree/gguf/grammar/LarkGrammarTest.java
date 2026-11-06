package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link LarkGrammar}: Lark grammars compile to grammars that accept exactly what xgrammar 0.2.8's
 * {@code compile_lark} accepts (the verdicts of {@code grammars/lark-xgrammar-verdicts.jsonl}, recorded by running
 * xgrammar's matcher over a byte vocabulary), vLLM's Lark detection, the {@code %ignore}, case-folding, token and
 * left-recursion behaviour, and the named refusals with their Lark source positions.
 */
public class LarkGrammarTest {

    private static Grammar lark(String source) {
        return GbnfBuilder.grammarOf(ConstraintSpec.lark(source));
    }

    private static boolean accepts(Grammar grammar, String text) {
        boolean ascii = text.chars().allMatch(c -> c > 0 && c < 0x80);
        return ascii ? GrammarMatcher.matches(grammar, text) : GrammarMatcher.matchesUtf8(grammar, text);
    }

    private static void assertRefused(String source, String messagePart) {
        try {
            lark(source);
            fail("expected a refusal of " + source);
        } catch (IllegalArgumentException e) {
            assertTrue("the refusal names the problem: " + e.getMessage(), e.getMessage().contains(messagePart));
        }
    }

    private static GrammarException refusal(String source) {
        try {
            lark(source);
        } catch (GrammarException e) {
            return e;
        }
        throw new AssertionError("expected a GrammarException for " + source);
    }

    // ------------------------------------------------------------------ parity with xgrammar

    @Test
    public void everyGrammarAcceptsWhatXgrammarAccepts() throws IOException {
        int checked = 0;
        try (InputStream in = LarkGrammarTest.class.getResourceAsStream("/grammars/lark-xgrammar-verdicts.jsonl")) {
            assertNotNull("the verdict fixture is on the test classpath", in);
            BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            for (String line = reader.readLine(); line != null; line = reader.readLine()) {
                List<?> entry = (List<?>) Json.parse(line);
                String source = (String) entry.get(0);
                Grammar grammar = lark(source);
                for (Object verdict : (List<?>) entry.get(1)) {
                    String text = (String) ((List<?>) verdict).get(0);
                    boolean expected = (Boolean) ((List<?>) verdict).get(1);
                    assertEquals("xgrammar's verdict on " + Json.toString(text, Json.Style.CANONICAL)
                            + " under\n" + source, expected, accepts(grammar, text));
                    checked++;
                }
            }
        }
        assertTrue("the fixture holds the recorded verdicts", checked > 150);
    }

    // ------------------------------------------------------------------ detection

    @Test
    public void detectionFollowsVllmGrammarIsLikelyLark() {
        assertTrue(LarkGrammar.isLikelyLark("rule: 'abc'"));
        assertFalse(LarkGrammar.isLikelyLark("rule ::= 'abc'"));
        assertTrue("a ::= inside a comment does not count",
                LarkGrammar.isLikelyLark("# root ::= \"x\"\nstart: \"x\" // a ::= b"));
        assertFalse(LarkGrammar.isLikelyLark("# comment\nroot   ::=   \"x\""));
        assertFalse(LarkGrammar.isLikelyLark(""));
        assertFalse(LarkGrammar.isLikelyLark(null));
        assertTrue("blank text is Lark (and then refused for its missing start rule)",
                LarkGrammar.isLikelyLark("  \n"));
    }

    @Test
    public void theGrammarFactoryPicksTheSyntax() {
        assertEquals(ConstraintSpec.Kind.LARK, ConstraintSpec.grammar("start: \"x\"").kind());
        assertEquals(ConstraintSpec.Kind.GBNF, ConstraintSpec.grammar("root ::= \"x\"").kind());
        assertEquals(ConstraintSpec.lark("start: \"x\""), ConstraintSpec.grammar("start: \"x\""));
    }

    // ------------------------------------------------------------------ llama.cpp's %llguidance header

    @Test
    public void theLlguidanceHeaderCompilesTheGrammarUnchanged() {
        String body = "start: \"id=\" NAME\nNAME: /[0-9]+/\n";
        String plain = LarkGrammar.gbnf(body, null, null);
        assertEquals("the header changes no rule", plain, LarkGrammar.gbnf("%llguidance {}\n" + body, null, null));
        assertEquals(plain, LarkGrammar.gbnf("%llguidance\n{}\n" + body, null, null));
        assertEquals(plain, LarkGrammar.gbnf("%llguidance {\"no_forcing\": true,"
                + " \"allow_invalid_utf8\": false}\n" + body, null, null));
        assertEquals(plain, LarkGrammar.gbnf("%llguidance []\n" + body, null, null));
        assertEquals(plain, LarkGrammar.gbnf("%llguidance [true, false]\n" + body, null, null));
        assertEquals("the header may appear after the rules", plain,
                LarkGrammar.gbnf(body + "%llguidance {}\n", null, null));
        Grammar grammar = lark("%llguidance {}\n" + body);
        assertTrue(GrammarMatcher.matches(grammar, "id=42"));
        assertFalse(GrammarMatcher.matches(grammar, "id=x"));
    }

    @Test
    public void theLlguidanceHeaderSelectsLarkOverVllmDetection() {
        // vLLM's detection alone calls a text with a "::=" line GBNF; llama.cpp's header takes precedence
        String source = "%llguidance {}\nstart: \"a::=b\"";
        assertTrue(LarkGrammar.isLark(source));
        assertEquals(ConstraintSpec.Kind.LARK, ConstraintSpec.grammar(source).kind());
        assertTrue(GrammarMatcher.matches(lark(source), "a::=b"));
        assertFalse("the rest of the text is still told apart as vLLM does",
                LarkGrammar.isLark("root ::= \"x\""));
    }

    @Test
    public void malformedLlguidanceHeadersAreRefusedByName() {
        assertRefused("%llguidance", "expected a JSON value after %llguidance");
        assertRefused("%llguidance\nstart: \"a\"", "failed to parse %llguidance declaration");
        assertRefused("%llguidance {bad}", "failed to parse %llguidance declaration");
        assertRefused("%llguidance {\"a\": }", "failed to parse %llguidance declaration");
        assertRefused("%llguidance \"x\"\nstart: \"a\"",
                "failed to parse %llguidance declaration: expected a JSON object");
        assertRefused("%llguidance 5\nstart: \"a\"",
                "failed to parse %llguidance declaration: expected a JSON object");
        assertRefused("%llguidance true\nstart: \"a\"",
                "failed to parse %llguidance declaration: expected a JSON object");
        assertRefused("%llguidance {\"no_forcing\": 1}\nstart: \"a\"",
                "failed to parse %llguidance declaration: 'no_forcing' must be a boolean");
        assertRefused("%llguidance {\"allow_invalid_utf8\": \"x\"}\nstart: \"a\"",
                "failed to parse %llguidance declaration: 'allow_invalid_utf8' must be a boolean");
        assertRefused("%llguidance [1, 2]\nstart: \"a\"",
                "failed to parse %llguidance declaration: an options array holds booleans");
        assertRefused("%llguidance [true, false, true]\nstart: \"a\"",
                "failed to parse %llguidance declaration: an options array holds at most 2 booleans");
        assertRefused("%llguidance {}\nstart: \"a", "unterminated");
    }

    @Test
    public void nestedLlguidanceFormsAreRefusedByName() {
        assertRefused("start: %lark {\nstart: \"a\"\n}", "nested %lark grammars are not supported");
        assertRefused("start: %regex {\"substring_chars\": \"ab\"}",
                "%regex substring constraints are not supported");
    }

    @Test
    public void braceRepetitionsFollowLlguidance() {
        assertLanguage("start: \"a\"{3}", texts("aaa"), texts("aa", "aaaa"));
        assertLanguage("start: \"a\"{2,4}", texts("aa", "aaa", "aaaa"), texts("a", "aaaaa"));
        assertLanguage("start: \"a\"{2,}", texts("aa", "aaaaa"), texts("a"));
        assertLanguage("start: \"a\"{,2}", texts("", "a", "aa"), texts("aaa"));
        assertLanguage("start: \"a\"{ 2 , 3 }", texts("aa", "aaa"), texts("a", "aaaa"));
        assertLanguage("start: (\"a\" | \"b\"){2}", texts("ab", "ba", "aa"), texts("a"));
        assertRefused("start: \"a\"{}", "expected integer");
        assertRefused("start: \"a\"{2,1}", "repetition end must be greater than or equal to start");
        assertRefused("start: \"a\"{3}{2}", "cannot follow another");
    }

    @Test
    public void aGbnfToLarkConversionCompilesThroughTheHeader() throws IOException {
        StringBuilder source = new StringBuilder();
        try (InputStream in = LarkGrammarTest.class.getResourceAsStream("/grammars/json-lark.lark")) {
            assertNotNull("the llguidance gbnf_to_lark fixture is on the test classpath", in);
            BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            for (String line = reader.readLine(); line != null; line = reader.readLine()) {
                source.append(line).append('\n');
            }
        }
        Grammar grammar = lark(source.toString());
        assertTrue(GrammarMatcher.matches(grammar, "{}"));
        assertTrue(GrammarMatcher.matches(grammar, "{\"a\": [1, 2.5, true, null]}"));
        assertTrue(GrammarMatcher.matches(grammar, "{\"a\": \"x\\\\y\"}"));
        assertFalse(GrammarMatcher.matches(grammar, "{\"a\": }"));
        assertFalse(GrammarMatcher.matches(grammar, "{\"a\": [1,]}"));
    }

    // ------------------------------------------------------------------ compiled shapes

    @Test
    public void startIsTheRootAndRulesKeepTheirNames() {
        String gbnf = LarkGrammar.gbnf("start: greeting \" \" NAME\ngreeting: \"hello\" | \"hi\"\nNAME: /[a-z]+/\n"
                + "root: \"r\"\n_aside: \"x\"", null, null);
        assertTrue(gbnf, gbnf.startsWith("root ::= greeting \" \" NAME\n"));
        assertTrue(gbnf, gbnf.contains("\ngreeting ::= (\"hello\" | \"hi\")\n"));
        assertTrue("a Lark rule named root moves aside: " + gbnf, gbnf.contains("\n0root ::= \"r\"\n"));
        assertTrue("underscores become dashes: " + gbnf, gbnf.contains("\n-aside ::= \"x\"\n"));
    }

    @Test
    public void ignoreFollowsEveryTokenInsideRulesOnly() {
        String gbnf = LarkGrammar.gbnf("start: \"a\" b B\nb: \"x\"\nB: \"y\" \"z\"\n%ignore \" \"", null, null);
        assertTrue(gbnf, gbnf.startsWith("root ::= \"a\" 0ignore b B 0ignore\n"));
        assertTrue(gbnf, gbnf.contains("\nB ::= \"y\" \"z\"\n"));
        assertTrue(gbnf, gbnf.contains("\n0ignore ::= (\" \")*\n"));
        Grammar grammar = lark("start: \"a\" b B\nb: \"x\"\nB: \"y\" \"z\"\n%ignore \" \"");
        assertTrue(GrammarMatcher.matches(grammar, "a  x yz "));
        assertFalse("the ignored text never splits a terminal", GrammarMatcher.matches(grammar, "axy z"));
    }

    @Test
    public void tokenSetsAndNamedSpecialTokensBecomeTokenReferences() {
        GrammarVocab vocab = text -> "<|tool|>".equals(text) ? new int[]{84} : new int[]{1, 2};
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.lark("start: <|tool|> <[^97,98]>"), vocab);
        assertTrue("token 84 is the byte 'T' of the matcher's vocabulary", GrammarMatcher.matches(grammar, "Tc"));
        assertFalse(GrammarMatcher.matches(grammar, "Ta"));
        assertFalse(GrammarMatcher.matches(grammar, "Xc"));
        assertEquals("root ::= <[84]> !<[97],[98]>\n",
                LarkGrammar.gbnf("start: <|tool|> <[^97,98]>", null, vocab));
        try {
            GbnfBuilder.grammarOf(ConstraintSpec.lark("start: <|nope|>"), vocab);
            fail("a name that tokenizes to two tokens is no special token");
        } catch (GrammarException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("special token <|nope|> is not one token"));
        }
    }

    @Test
    public void caseInsensitiveLiteralsAndRegexesFoldAsciiLetters() {
        assertEquals("root ::= [Aa] \"1-\" [Bb]\n", LarkGrammar.gbnf("start: \"a1-b\"i", null, null));
        Grammar regex = lark("start: /[a-c]x[^d]/i");
        assertTrue(GrammarMatcher.matches(regex, "BXe"));
        assertFalse("a negated set refuses both cases", GrammarMatcher.matches(regex, "bxD"));
    }

    @Test
    public void inlineJsonFollowsTheWhitespaceMode() {
        String source = "start: \"v=\" %json {\"type\": \"object\", \"properties\": {\"n\": {\"type\": \"integer\"}},"
                + " \"required\": [\"n\"]}";
        Grammar flexible = LarkGrammar.compile(source, JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE, null);
        Grammar compact = LarkGrammar.compile(source, JsonSchemaToGbnf.WhitespaceMode.COMPACT, null);
        assertTrue(GrammarMatcher.matches(flexible, "v={\"n\": 5}"));
        assertFalse(GrammarMatcher.matches(compact, "v={\"n\": 5}"));
        assertTrue(GrammarMatcher.matches(compact, "v={\"n\":5}"));
    }

    @Test
    public void theSameRegexEmbedsOnce() {
        String gbnf = LarkGrammar.gbnf("start: A A /[0-9]+/\nA: /[0-9]+/", null, null);
        assertTrue(gbnf, gbnf.startsWith("root ::= A A 0re0-root-"));
        assertTrue("the terminal references the same embedded rule: " + gbnf, gbnf.contains("\nA ::= 0re0-root-"));
        assertFalse("the pattern compiles once: " + gbnf, gbnf.contains("0re1-"));
    }

    @Test
    public void leftRecursionIsRewrittenToTheSameLanguage() {
        Grammar grammar = lark("start: e\ne: e \"+\" t | e \"-\" t | t\nt: t \"*\" f | f\nf: \"x\" | \"(\" e \")\"");
        for (String ok : new String[]{"x", "x+x", "x*x-x", "(x+x)*x", "((x))"}) {
            assertTrue(ok, GrammarMatcher.matches(grammar, ok));
        }
        for (String bad : new String[]{"", "+x", "x+", "(x", "x**x"}) {
            assertFalse(bad, GrammarMatcher.matches(grammar, bad));
        }
        String gbnf = LarkGrammar.gbnf("start: e\ne: e \"+\" t | t\nt: \"x\"", null, null);
        assertTrue("the direct recursion becomes a tail rule: " + gbnf,
                gbnf.contains("\ne ::= t 0lr-e\n0lr-e ::= (\"+\" t)*\n"));
    }

    @Test
    public void aLongChainOfRulesCompiles() {
        StringBuilder source = new StringBuilder("start: r0\n");
        int n = 2000;
        for (int i = 0; i < n; i++) {
            source.append('r').append(i).append(": r").append(i + 1).append(" \"a\" | \"b\"\n");
        }
        source.append('r').append(n).append(": \"c\"\n");
        Grammar grammar = lark(source.toString());
        assertTrue(GrammarMatcher.matches(grammar, "b"));
        assertTrue(GrammarMatcher.matches(grammar, "ba"));
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void unsupportedConstructsAreRefusedByName() {
        assertRefused("start.2: \"x\"", "priorities are not supported");
        assertRefused("start: sep{\"a\"}\nsep{x}: x", "templates are not supported");
        assertRefused("start: x\nx[lazy]: /.*/ \"stop\"", "rule attribute 'lazy' is not supported");
        assertRefused("start: x\nX[max_chars=2]: /.*/", "attributes are only supported on rules");
        assertRefused("start: x\nx[max_chars=2, max_chars=3]: \"y\"", "max_chars attribute is specified more than once");
        assertRefused("start: x\nx[foo=1]: \"y\"", "rule attribute 'foo' is not supported");
        assertRefused("start: X\n%declare X", "directive %declare is not supported");
        assertRefused("start: \"a\"\n%override x: \"b\"", "directive %override is not supported");
        assertRefused("start: @foo", "named grammar references (@foo) are not supported");
        assertRefused("start: 'x'", "double quotes");
        assertRefused("start: /a/m", "regular-expression flag 'm' is not supported");
        assertRefused("start: /a/x", "regular-expression flag 'x' is not supported");
        assertRefused("start: \"é\"i", "ASCII characters only");
        assertRefused("start: <|im_end|>", "requires a model vocabulary");
        assertRefused("start: /(?=a)a/", "unsupported construct");
        assertRefused("start: \"a\" & \"b\"", "terminal intersection '&' is not supported");
    }

    @Test
    public void malformedGrammarsAreRefusedAtTheirLarkPosition() {
        GrammarException missing = refusal("rule: \"x\"");
        assertTrue(missing.getMessage(), missing.getMessage().contains("no start rule found"));
        GrammarException unknown = refusal("start: \"a\"\nunused: BAR");
        assertTrue(unknown.getMessage(), unknown.getMessage().contains("unknown name 'BAR'"));
        assertEquals(2, unknown.getLine());
        assertEquals(9, unknown.getColumn());
        GrammarException duplicate = refusal("start: x\nx: \"x\"\nx: \"y\"");
        assertTrue(duplicate.getMessage(), duplicate.getMessage().contains("duplicate rule or terminal 'x'"));
        assertEquals(3, duplicate.getLine());
        assertRefused("start: \"a\"\n  \"b\"", "expected rule or terminal name");
        assertRefused("start: A\nA: a\na: \"x\"", "terminal 'A' cannot reference rule 'a'");
        assertRefused("start: \"a\"\n%ignore x\nx: \"b\"", "%ignore cannot reference rule 'x'");
        assertRefused("start: A\nA: \"x\" | \"y\" A", "circular reference in terminal 'A'");
        assertRefused("start: A\nA: %json {\"type\":\"integer\"}", "%json cannot be used in terminals");
        assertRefused("start: A\nA: <[1]>", "special tokens cannot be used in terminals");
        assertRefused("start: \"a\"~3..1", "repetition end must be greater than or equal to start");
        assertRefused("start: \"a\"??", "cannot follow another");
        assertRefused("start: \"x\"\n%import other.WS", "unknown common import 'other.WS'");
        assertRefused("start: WS\n%import common.WS\n%import common.WS", "duplicate rule or terminal 'WS'");
        assertRefused("start: \"ab\"..\"cd\"", "endpoints must be one character");
        assertRefused("start: <[]>", "empty numeric special-token range");
        assertRefused("start: <[5-3]>", "invalid numeric special-token range '5-3'");
        assertRefused("start: %json [1]", "%json takes a JSON schema object or boolean");
        assertRefused("start: %json {\"type\": 5}", "failed to compile inline JSON schema");
        assertRefused("start: \"a\\x41\"", "invalid string literal");
        assertRefused("start: \"abc", "unterminated string literal");
        assertRefused("start: (\"a\"", "expected ')'");
        assertRefused("not a grammar", "expected ':'");
        assertRefused("start ::= \"a\"", "'::=' is GBNF rule syntax; a Lark rule is written 'start: ...'");
    }

    @Test
    public void leftRecursionWithoutAWayOutMatchesNothing() {
        // a rule whose every alternative starts with itself compiles, as xgrammar accepts it, and derives
        // nothing; the NUL literal no token carries encodes that for the pushdown runtime
        Grammar dead = lark("start: start \"x\"");
        assertFalse(GrammarMatcher.matches(dead, "x"));
        assertFalse(GrammarMatcher.matches(dead, ""));
        Grammar others = lark("start: dead | \"y\"\ndead: dead \"x\" | dead");
        assertTrue(GrammarMatcher.matches(others, "y"));
        assertFalse(GrammarMatcher.matches(others, "yx"));
        Grammar cycle = lark("start: a\na: b \"x\"\nb: a \"y\"");
        assertFalse(GrammarMatcher.matches(cycle, ""));
        assertFalse(GrammarMatcher.matches(cycle, "yx"));
        assertRefused("start: x\nx: \"a\"? x \"b\" | \"c\"", "left recursion");
    }

    @Test
    public void nestingAndRuleCountsAreBounded() {
        StringBuilder deep = new StringBuilder("start: ");
        for (int i = 0; i <= LarkParser.MAX_NESTING; i++) {
            deep.append('(');
        }
        assertRefused(deep.toString(), "groups nest deeper than " + LarkParser.MAX_NESTING);
        assertRefused("start: <[0-70000]>", "more than " + LarkParser.MAX_TOKEN_SET + " tokens");
    }

    @Test
    public void aCompiledLarkConstraintRunsThroughTheSamplerPath() {
        CompiledConstraint compiled = GbnfBuilder.compile(ConstraintSpec.grammar("start: \"ok\" INT\n"
                + "%import common.INT"));
        assertFalse(compiled.gated());
        assertTrue(GrammarMatcher.matches(compiled.grammar(), "ok42"));
        assertFalse(GrammarMatcher.matches(compiled.grammar(), "ok"));
    }

    // ------------------------------------------------------------------ budget attributes

    @Test
    public void budgetAttributesCountTheRulesText() {
        assertLanguage("start: word \".\"\nword[max_chars=3]: /[a-z]+/", texts("abc.", "ab."), texts("abcd."));
        assertLanguage("start: a\na[max_tokens=5, max_chars=2]: /[a-z]+/", texts("aa"), texts("aaa"));
        assertLanguage("start[max_chars=2]: /[a-z]+/", texts("ab"), texts("abc"));
        assertLanguage("start: a\na[ max_chars = 2 ]: \"x\"", texts("x"), texts("xx"));
        assertTrue("the attribute travels as a GBNF rule attribute",
                LarkGrammar.gbnf("start: a\na[max_tokens=5]: \"x\"", null, null).contains("a[max_tokens=5] ::="));
        assertTrue(LarkGrammar.gbnf("start: a\na[max_tokens=5, max_chars=7]: \"x\"", null, null)
                .contains("a[max_tokens=5, max_chars=7] ::="));
    }

    // ------------------------------------------------------------------ stop attributes

    @Test
    public void aStopAttributeEndsTheRulesTextAtTheStop() {
        // the body may write any text but the stop, then the stop itself completes the rule
        assertLanguage("start: t\nt[stop=\"!\"]: /[^!]*/", texts("!", "abc!"), texts("!x", "abc!x", "ab!cd"));
        assertLanguage("start: t\nt[stop=\"}\"]: /[^}]*/", texts("{}"), texts("{}}"));
        // a stop the body could overlap: the body part holds no occurrence of the stop
        assertLanguage("start: t\nt[stop=\"ab\"]: /[^a]*/", texts("ab", "xab", "cdab"), texts("aabb", "aab"));
        // the body may reference terminals and use regexes; the stop stays one string: the rule's text is
        // the body then the stop, concatenated
        assertLanguage("start: t\nt[stop=\"END\"]: W \" \" W\nW: \"w\"+", texts("w wEND", "www wEND"),
                texts("wwEND", "ENDx", "w w END", "w w ENDx"));
        // a budget spans the body and the stop together
        assertLanguage("start: t\nt[max_chars=4, stop=\"!\"]: /[^!]*/", texts("abc!", "ab!", "!"),
                texts("abcde!", "abcd!!"));
        // the compiled rule is the product followed by the stop, not a post-filter
        String gbnf = LarkGrammar.gbnf("start: t\nt[stop=\"END\"]: \"x\"", null, null);
        assertTrue(gbnf, gbnf.contains("t ::= 0stop"));
    }

    @Test
    public void stopAttributesAreRefusedAsXgrammarRefusesThem() {
        assertRefused("start: t\nt[stop=\"\"]: \"x\"", "stop must not be empty");
        assertRefused("start: t\nt[stop=\"a\", stop=\"b\"]: \"x\"", "stop attribute is specified more than once");
        assertRefused("start: t\nt[stop=5]: \"x\"", "expected a string literal after stop=");
        assertRefused("start: t\nt[stop=/a/]: \"x\"", "expected a string literal after stop=");
        assertRefused("start: t\nt[stop=\"a\"i]: \"x\"", "case-insensitive flags are not supported on stop");
        assertRefused("start: t\nt[stop=\"a\"]: r\nr: \"x\"", "a stop body cannot reference rule 'r'");
        assertRefused("start: t\nt[stop=\"a\"]: %json {\"type\": \"integer\"}", "%json cannot be used in terminals");
        assertRefused("start: t\nt[stop=\"a\"]: <[1]>", "special tokens cannot be used in terminals");
        assertRefused("start: t\nt[stop=\"a\"]: /x/i",
                "regular-expression flag 'i' is not supported with suffix or stop attributes");
        assertRefused("start: t\nt[stop=\"x\"]: \"x\"", "which its text cannot avoid");
        assertRefused("start: T\nT[max_tokens=1]: \"x\"", "attributes are only supported on rules");
    }

    // ------------------------------------------------------------------ repetitions above the expansion limit

    @Test
    public void aRepetitionAboveTheLimitIsAcceptedAndStaysExact() {
        Grammar exact = lark("start: \"a\"~20000");
        assertTrue(GrammarMatcher.matches(exact, repeat('a', 20000)));
        assertFalse(GrammarMatcher.matches(exact, repeat('a', 19999)));
        assertFalse(GrammarMatcher.matches(exact, repeat('a', 20001)));
        assertFalse("no generated {n} form: the copies are written out",
                LarkGrammar.gbnf("start: \"a\"~20000", null, null).contains("{20000}"));
    }

    @Test
    public void aRepetitionRangeAboveTheLimitCompilesWithUnboundedUpper() {
        // the upper bound falls back to unbounded, the GBNF runtime's own rule for an over-limit maximum
        assertLanguage("start: \"a\"~5..30000", texts(repeat('a', 5), repeat('a', 6)), texts(repeat('a', 4)));
        assertRefused("start: \"a\"~2147483648", "invalid non-negative repetition count");
        assertRefused("start: \"a\"~99999999999", "invalid non-negative repetition count");
    }

    private static String repeat(char c, int count) {
        StringBuilder out = new StringBuilder(count);
        for (int i = 0; i < count; i++) {
            out.append(c);
        }
        return out.toString();
    }

    // ------------------------------------------------------------------ left recursion

    @Test
    public void indirectLeftRecursionIsRewrittenAndKeepsTheLanguage() {
        assertLanguage("start: a\na: b \"x\" | \"1\"\nb: a \"y\" | \"2\"",
                texts("1", "2x", "1yx", "2xyx", "1yxyx", "2xyxyx"), texts("x", "1x", "2", "1y", "2yxyx", ""));
        assertLanguage("start: a\na: a \"z\" | b \"x\" | \"1\"\nb: a \"y\" | \"2\"",
                texts("1", "1z", "2xz", "1yxz", "2x"), texts("z", "2z", "1x", "2zx", ""));
        assertLanguage("start: expr\nexpr: term \" \" \"+\" \" \" expr | term\nterm: \"(\" expr \")\" | ID\nID: \"n\"",
                texts("n", "n + n", "((n) + n) + n"), texts("n +", "(n"));
    }

    // ------------------------------------------------------------------ language parity with xgrammar

    private static void assertLanguage(String source, String[] accepted, String[] refused) {
        Grammar grammar = lark(source);
        for (String text : accepted) {
            assertTrue("the grammar accepts " + text, accepts(grammar, text));
        }
        for (String text : refused) {
            assertFalse("the grammar refuses " + text, accepts(grammar, text));
        }
    }

    private static String[] texts(String... texts) {
        return texts;
    }

    @Test
    public void operatorsAndContinuationsBuildLarksLanguage() {
        assertLanguage("start: \"a\"? \"b\"* \"c\"+ (\"d\" | \"e\")~2 \"f\"~1..2 [\"g\"]",
                texts("acdef", "bbbcccddff", "ccedffg", "cdef", "abcddfg"),
                texts("cde f", "de f", "cdeffff", "c"));
        assertLanguage("start: \"a\"~0 \"b\"", texts("b"), texts("ab"));
        assertLanguage("start: a\n    | b \"!\"\n    |\na: \"x\" -> ex\n  | \"y\"\n?b: \"z\" // comment\n!c: \"q\"\n",
                texts("", "x", "y", "z!"), texts("z", "q"));
        assertLanguage("start: (\"a\"\n  | \"b\")", texts("a", "b"), texts("ab"));
        assertLanguage("start : \"a\" # comment\n# full line\n", texts("a"), texts(""));
    }

    @Test
    public void literalsRangesAndRegexesReadAsPythonReadsThem() {
        assertLanguage("start: \"\\n\" \"\\t\" \"\\u0041\" \"\\\"\" \"\\\\\" \"\\/\"",
                texts("\n\tA\"\\/"), texts("\n\tA\"\\"));
        assertLanguage("start: \"\" \"a\" \"\"", texts("a"), texts(""));
        assertLanguage("start: \"select\"i \" \" \"x1\"i", texts("SELECT x1", "SeLeCt X1"),
                texts("select  x1", "selec x1"));
        assertLanguage("start: \"a1-\"i", texts("A1-", "a1-"), texts("A1+"));
        assertLanguage("start: \"a\"..\"c\" \"0\"..\"9\"+", texts("a0", "c99"), texts("d0", "b"));
        assertLanguage("start: /[0-9]+(\\.[0-9]+)?/ \" \" /a.b/ \" \" /x\\/y/",
                texts("12.5 acb x/y", "7 a b x/y"), texts("7 a\nb x/y", "1. aab x/y"));
        assertLanguage("start: /\\d{2,3}/ /\\w+/ /\\s/ /[^a-c]/", texts("12ab_9 d", "1234 d"), texts("12a b"));
        assertLanguage("start: /a.b/", texts("a\rb"), texts("a\nb"));
        assertLanguage("start: /ab/i", texts("ab", "AB", "aB"), texts("ac"));
        assertLanguage("start: /[a-c]x/i", texts("Bx", "BX"), texts("dx"));
        assertLanguage("start: /[^a]/i", texts("b"), texts("A", "a"));
        assertLanguage("start: /k\\.Z/i", texts("K.z"), texts("KxZ"));
        assertLanguage("start: /./i", texts("x"), texts("\n"));
        assertLanguage("start: /a.b/s", texts("a\nb", "axb"), texts("ab"));
        assertLanguage("start: /a.b/si", texts("A\nB"), texts("AB"));
        assertLanguage("start: /a/u", texts("a"), texts("A"));
    }

    @Test
    public void terminalsImportsAndIgnoreBehaveAsXgrammarDoes() {
        assertLanguage("start: NAME (\",\" NAME)*\nNAME: LETTER (LETTER | DIGIT)*\n"
                        + "LETTER: \"a\"..\"z\"\nDIGIT: \"0\"..\"9\"",
                texts("ab,c1", "a"), texts("1a", "a,"));
        assertLanguage("start: _inner\n_inner: _X\n_X: \"k\"", texts("k"), texts(""));
        assertLanguage("start: root\nroot: \"r\"", texts("r"), texts(""));
        assertLanguage("%ignore \" \"\nstart: \"a\" \"b\" WORD\nWORD: \"w\"+",
                texts("ab w", "a  b   ww   ", "abw"), texts(" ab w", "a b w w"));
        assertLanguage("%ignore \" \"\n%ignore \"\\n\"\nstart: item+\nitem: \"x\" | \"(\" item \")\"",
                texts("x \n x", "( x )\n", "x", "(  x)"), texts(" x"));
        assertLanguage("%ignore \" \"\nstart: PAIR PAIR\nPAIR: \"a\" \"b\"", texts("abab", "ab ab", "ab ab "),
                texts("a bab"));
        assertLanguage("%ignore /[ \\t]+/\nstart: \"a\" \"b\"", texts("a \t b"), texts(" ab"));
        assertLanguage("%import common.WS\n%ignore WS\nstart: \"a\" \"b\"", texts("a \n b"), texts("a-b"));
        assertLanguage("%ignore \" \"\nstart: %json {\"type\":\"integer\"} \"x\"", texts("1 x", "1x"), texts("x"));
        assertLanguage("%import common.NUMBER\n%import common (WORD, WS)\nstart: WORD WS NUMBER",
                texts("abc 12", "abc\t1.5e3", "abc .5", "abc 1."), texts("abc", "ab1 2"));
        assertLanguage("%import common.SIGNED_INT -> N\n%import common.ESCAPED_STRING\nstart: N \",\" ESCAPED_STRING",
                texts("-12,\"a\\\"b\"", "+0,\"\""), texts("12,\"a", "1.5,\"\""));
        assertLanguage("start: \"call \" %json {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"integer\"}},"
                        + " \"required\": [\"x\"], \"additionalProperties\": false}",
                texts("call {\"x\": 1}", "call {\"x\":-5}"), texts("call {\"x\": \"1\"}", "call {}"));
        assertLanguage("start: %json {\"type\":\"string\",\"pattern\":\"^[{}]+$\"}", texts("\"{}\""),
                texts("\"a\""));
    }
}
