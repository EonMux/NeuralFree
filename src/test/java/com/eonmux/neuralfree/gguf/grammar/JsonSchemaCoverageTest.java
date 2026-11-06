package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.math.BigDecimal;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The JSON-schema keywords beyond llama.cpp's converter that vLLM's xgrammar backend supports
 * ({@code backend_xgrammar.py} {@code STRING_SUPPORTED_FORMATS} and its keyword checks): the string formats, number
 * bounds, {@code minProperties}/{@code maxProperties}, {@code patternProperties}, {@code propertyNames}, and
 * {@code $ref} through {@code $id}/{@code $anchor}. Every case drives the converted grammar over the JSON text
 * ({@link GrammarMatcher}), so a case fails when the grammar admits a value the schema forbids or refuses one it
 * admits. The format vectors agree with xgrammar's acceptance except where xgrammar departs from the format's RFC
 * (noted per case); the refusals name the keyword they cannot honour.
 */
public class JsonSchemaCoverageTest {

    private static Grammar grammar(String schema) {
        return JsonSchemaToGbnf.convert(schema, JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
    }

    /** The JSON text of a string value. */
    private static String str(String value) {
        return Json.toString(value, Json.Style.CANONICAL);
    }

    private static void accepts(Grammar grammar, String... texts) {
        for (String text : texts) {
            assertTrue("should accept " + text, GrammarMatcher.matches(grammar, text));
        }
    }

    private static void rejects(Grammar grammar, String... texts) {
        for (String text : texts) {
            assertFalse("should reject " + text, GrammarMatcher.matches(grammar, text));
        }
    }

    private static void acceptsStrings(Grammar grammar, String... values) {
        for (String value : values) {
            accepts(grammar, str(value));
        }
    }

    private static void rejectsStrings(Grammar grammar, String... values) {
        for (String value : values) {
            rejects(grammar, str(value));
        }
    }

    private static Grammar format(String format) {
        return grammar("{\"type\": \"string\", \"format\": \"" + format + "\"}");
    }

    private static String refusal(String schema) {
        try {
            JsonSchemaToGbnf.gbnf(schema, JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().startsWith("JSON schema conversion failed:"));
            return e.getMessage();
        }
        fail("the schema should be refused: " + schema);
        return null;
    }

    private static void assertMentions(String message, String... parts) {
        for (String part : parts) {
            assertTrue(message, message.contains(part));
        }
    }

    // -------------------------------------------------- string formats

    @Test
    public void emailIsAnRfc5321Mailbox() {
        Grammar email = format("email");
        acceptsStrings(email, "a.b+c@example.com", "\"john doe\"@ex.org", "\"a\\\\b\"@x.io", "a@b",
                // address literals: RFC 5321 section 4.1.3, which xgrammar leaves out
                "x@[192.168.0.1]", "x@[IPv6:::1]");
        rejectsStrings(email, "a..b@x.com", "@x.com", "a@", "a@-x.com", ".a@x.com", "a b@x.com");
        // a raw quote inside the JSON string is never written: the quoted local part is escaped
        rejects(email, "\"\"a\"@x.com\"");
    }

    @Test
    public void durationIsRfc3339AppendixA() {
        Grammar duration = format("duration");
        acceptsStrings(duration, "P1Y2M3DT4H5M6S", "PT1H", "P2W", "P1D", "P1M", "PT36H", "P1DT2M");
        rejectsStrings(duration, "P", "PT", "P1H", "P1Y2W", "1D", "PT0.5S", "P1S");
    }

    @Test
    public void ipv4IsADottedQuadWithoutLeadingZeros() {
        Grammar ipv4 = format("ipv4");
        acceptsStrings(ipv4, "0.0.0.0", "255.255.255.255", "192.168.1.10", "10.0.99.199");
        // leading zeros read as octal elsewhere; JSON schema's ipv4 refuses them (xgrammar admits them)
        rejectsStrings(ipv4, "256.1.1.1", "1.2.3", "1.2.3.4.5", "01.2.3.4", "1.2.3.-4", "a.b.c.d");
    }

    @Test
    public void ipv6IsRfc3986Ipv6Address() {
        Grammar ipv6 = format("ipv6");
        acceptsStrings(ipv6, "::", "::1", "2001:db8::8a2e:370:7334", "fe80::1:2:3:4", "::ffff:192.168.0.1",
                "1:2:3:4:5:6:7:8", "1::", "1:2:3:4:5:6:7::", "ABCD:ef01::",
                // the full form with an embedded IPv4 tail, which xgrammar leaves out
                "1:2:3:4:5:6:1.2.3.4");
        rejectsStrings(ipv6, "1:2:3:4:5:6:7:8:9", "1::2::3", "12345::", ":1", "1:2:3:4:5:6:7", "::g",
                "1:2:3:4:5:6:7:8::");
    }

    @Test
    public void hostnameIsRfc1123() {
        Grammar hostname = format("hostname");
        StringBuilder label63 = new StringBuilder();
        for (int i = 0; i < 63; i++) {
            label63.append('a');
        }
        // host names are case-insensitive (xgrammar admits lower case only)
        acceptsStrings(hostname, "example.com", "A-1.b", "localhost", "x", label63 + ".com", "1.2.3.4");
        rejectsStrings(hostname, "-a.com", "a-.com", "a..b", label63 + "a.com", "a.", ".a", "a_b.com");
    }

    @Test
    public void uriIsRfc3986Uri() {
        Grammar uri = format("uri");
        acceptsStrings(uri, "https://user@example.com:8080/p/a?q=1#frag", "urn:isbn:0451450523",
                "mailto:a@b.c", "file:///etc/hosts", "http:", "ftp://[v1.fe]/x",
                // an IPv6 host literal, which xgrammar leaves out
                "http://[::1]/", "http://x/%2F?a=%20");
        rejectsStrings(uri, "//example.com", "1http://x", "http://exa mple.com", "http://x/%zz", "/relative",
                "a:b\"c");
    }

    @Test
    public void uriReferenceAdmitsRelativeReferences() {
        Grammar reference = format("uri-reference");
        // an absolute URI is a reference too (xgrammar refuses "https://x")
        acceptsStrings(reference, "../a/b?c#d", "//host/p", "", "#frag", "https://x", "a/b:c", "?q");
        rejectsStrings(reference, "a b", "%g0", "a:b c", ":a");
    }

    @Test
    public void uriTemplateIsRfc6570Level4() {
        Grammar template = format("uri-template");
        acceptsStrings(template, "http://example.com/{user}/{+path}{?q,lang}", "{var:3}", "{list*}", "plain",
                "{a.b}", "{#x,y}");
        rejectsStrings(template, "{", "{}", "{a b}", "{var:0}", "}", "{a..b}", "a^b");
    }

    @Test
    public void jsonPointersEscapeTildeAndSlash() {
        Grammar pointer = format("json-pointer");
        acceptsStrings(pointer, "", "/a~1b/0", "/~0", "/", "/a\"b", "/back\\slash");
        rejectsStrings(pointer, "a", "/~2", "/a~");
        // relative pointers admit draft-bhutton index manipulation (xgrammar refuses "0+1/x")
        Grammar relative = format("relative-json-pointer");
        acceptsStrings(relative, "0", "1/a", "2#", "0+1/x", "3-2#");
        rejectsStrings(relative, "-1", "01", "#", "", "/a", "1+#x");
    }

    @Test
    public void unknownFormatsStayAnnotations() {
        // llama.cpp and xgrammar alike: an unknown format leaves the string generic
        String gbnf = JsonSchemaToGbnf.gbnf("{\"type\": \"string\", \"format\": \"int32\"}",
                JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
        assertTrue(gbnf, gbnf.startsWith("char ::= "));
        assertTrue(gbnf, gbnf.contains("root ::= \"\\\"\" char* \"\\\"\""));
    }

    @Test
    public void formatRuleNamesAreReservedAgainstPropertyNames() {
        Grammar grammar = grammar("{\"type\": \"object\", \"properties\": {"
                + "\"ipv4-address\": {\"type\": \"integer\", \"minimum\": 1},"
                + " \"server\": {\"type\": \"string\", \"format\": \"ipv4\"}},"
                + " \"required\": [\"ipv4-address\", \"server\"], \"additionalProperties\": false}");
        accepts(grammar, "{\"ipv4-address\": 4, \"server\": \"10.0.0.1\"}");
        rejects(grammar, "{\"ipv4-address\": \"1.2.3.4\", \"server\": \"10.0.0.1\"}");
    }

    @Test
    public void sanitizedPropertyNamesCannotTakeAFormatRule() {
        Grammar grammar = grammar("{\"type\": \"object\", \"properties\": {"
                + "\"ipv4_address\": {\"type\": \"integer\", \"minimum\": 1},"
                + " \"server\": {\"type\": \"string\", \"format\": \"ipv4\"}},"
                + " \"required\": [\"ipv4_address\", \"server\"], \"additionalProperties\": false}");
        accepts(grammar, "{\"ipv4_address\": 4, \"server\": \"10.0.0.1\"}");
        rejects(grammar, "{\"ipv4_address\": 4, \"server\": \"4\"}");
    }

    // -------------------------------------------------- number bounds

    /** Checks a bounded number grammar against the exact range over a grid of canonical decimals. */
    private static void assertNumberRange(String bounds, BigDecimal min, boolean minExclusive, BigDecimal max,
            boolean maxExclusive) {
        Grammar grammar = grammar("{\"type\": \"number\", " + bounds + "}");
        for (int k = -260; k <= 260; k++) {
            BigDecimal value = BigDecimal.valueOf(k, 2);
            boolean expected = (min == null || (minExclusive ? value.compareTo(min) > 0 : value.compareTo(min) >= 0))
                    && (max == null || (maxExclusive ? value.compareTo(max) < 0 : value.compareTo(max) <= 0));
            String text = value.signum() == 0 ? "0" : value.stripTrailingZeros().toPlainString();
            assertEquals(bounds + " on " + text, expected, GrammarMatcher.matches(grammar, text));
        }
    }

    @Test
    public void numberBoundsAdmitExactlyTheRange() {
        assertNumberRange("\"minimum\": -1.5, \"exclusiveMaximum\": 2.25",
                new BigDecimal("-1.5"), false, new BigDecimal("2.25"), true);
        assertNumberRange("\"exclusiveMinimum\": 0", BigDecimal.ZERO, true, null, false);
        assertNumberRange("\"maximum\": -0.75", null, false, new BigDecimal("-0.75"), false);
        assertNumberRange("\"minimum\": 0.05, \"maximum\": 0.1", new BigDecimal("0.05"), false,
                new BigDecimal("0.1"), false);
        assertNumberRange("\"exclusiveMinimum\": -2, \"maximum\": -1.99", new BigDecimal("-2"), true,
                new BigDecimal("-1.99"), false);
        assertNumberRange("\"minimum\": 1, \"maximum\": 1", BigDecimal.ONE, false, BigDecimal.ONE, false);
        assertNumberRange("\"minimum\": -0.3, \"exclusiveMaximum\": 0", new BigDecimal("-0.3"), false,
                BigDecimal.ZERO, true);
    }

    @Test
    public void boundedNumbersBeyondTheGridKeepTheBounds() {
        Grammar grammar = grammar("{\"type\": \"number\", \"minimum\": -1.5, \"exclusiveMaximum\": 10.25}");
        accepts(grammar, "10.2499999999999", "10", "9.75", "-1.5000", "-1.4999999", "3.14159", "1.50");
        rejects(grammar, "10.25", "10.250", "10.2500001", "11", "-1.5000001", "-2", "1e3", "-0", "007", "1.");
        Grammar positive = grammar("{\"type\": \"number\", \"exclusiveMinimum\": 0}");
        accepts(positive, "0.0000000000000001", "123456789", "5.5");
        rejects(positive, "0", "0.0", "0.0000000000000000", "-1");
        Grammar large = grammar("{\"type\": \"number\", \"minimum\": 100000000000000000000}");
        accepts(large, "100000000000000000000", "100000000000000000000.5", "999999999999999999999");
        rejects(large, "99999999999999999999.9", "1");
    }

    @Test
    public void exclusiveBoundsTakeTheDraft4FlagAndTheTighterForm() {
        Grammar flagged = grammar("{\"type\": \"number\", \"minimum\": 2, \"exclusiveMinimum\": true}");
        accepts(flagged, "2.5", "3");
        rejects(flagged, "2", "1.9");
        Grammar integer = grammar("{\"type\": \"integer\", \"minimum\": 5, \"exclusiveMinimum\": true,"
                + " \"maximum\": 9, \"exclusiveMaximum\": true}");
        accepts(integer, "6", "8");
        rejects(integer, "5", "9");
        Grammar tighter = grammar("{\"type\": \"integer\", \"minimum\": 0, \"exclusiveMinimum\": 3}");
        accepts(tighter, "4", "100");
        rejects(tighter, "3", "0");
        Grammar union = grammar("{\"type\": [\"number\", \"null\"], \"minimum\": 0.5, \"maximum\": 1}");
        accepts(union, "null", "0.5", "0.75", "1");
        rejects(union, "0.4", "1.01");
    }

    @Test
    public void integerBoundsBeyondSixteenDigitsKeepTheirSide() {
        Grammar wide = grammar("{\"type\": \"integer\", \"minimum\": -1e19, \"maximum\": 1e19}");
        accepts(wide, "0", "-5", "9999999999999999", "-9999999999999999");
        rejects(wide, "1.5", "\"1\"");
        Grammar huge = grammar("{\"type\": \"integer\", \"minimum\": 1e19}");
        accepts(huge, "10000000000000000000", "10000000000000000001", "99999999999999999999");
        rejects(huge, "9999999999999999999", "1", "0", "-10000000000000000000");
        Grammar band = grammar("{\"type\": \"integer\", \"minimum\": 100000000000000000000,"
                + " \"maximum\": 100000000000000000005}");
        accepts(band, "100000000000000000000", "100000000000000000005");
        rejects(band, "100000000000000000006", "99999999999999999999");
        Grammar negative = grammar("{\"type\": \"integer\", \"exclusiveMaximum\": -1e17}");
        accepts(negative, "-100000000000000001", "-999999999999999999");
        rejects(negative, "-100000000000000000", "-1", "0");
    }

    @Test
    public void unsatisfiableBoundsAreRefused() {
        assertMentions(refusal("{\"type\": \"number\", \"minimum\": 5, \"maximum\": 1}"),
                "no number lies within the minimum and maximum bounds");
        assertMentions(refusal("{\"type\": \"number\", \"minimum\": 1, \"exclusiveMaximum\": 1}"),
                "no number lies within");
        assertMentions(refusal("{\"type\": \"integer\", \"minimum\": 5.5, \"maximum\": 5.7}"),
                "no integer lies within the minimum and maximum bounds");
        assertMentions(refusal("{\"type\": \"number\", \"exclusiveMinimum\": true}"),
                "exclusiveMinimum true needs minimum");
        assertMentions(refusal("{\"type\": \"number\", \"maximum\": \"ten\"}"), "maximum must be a number");
    }

    // -------------------------------------------------- minProperties / maxProperties

    @Test
    public void propertyCountsBoundDeclaredOptionalProperties() {
        Grammar grammar = grammar("{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"},"
                + " \"b\": {\"type\": \"integer\"}, \"c\": {\"type\": \"integer\"}},"
                + " \"minProperties\": 1, \"maxProperties\": 2}");
        accepts(grammar, "{\"a\": 1}", "{\"c\": 3}", "{\"a\": 1, \"c\": 2}", "{\"b\": 1, \"c\": 2}");
        rejects(grammar, "{}", "{\"a\": 1, \"b\": 2, \"c\": 3}", "{\"c\": 1, \"a\": 2}", "{\"d\": 1}");
    }

    @Test
    public void propertyCountsBoundAdditionalProperties() {
        Grammar grammar = grammar("{\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"integer\"}},"
                + " \"required\": [\"id\"], \"additionalProperties\": {\"type\": \"boolean\"},"
                + " \"minProperties\": 2, \"maxProperties\": 3}");
        accepts(grammar, "{\"id\": 1, \"x\": true}", "{\"id\": 1, \"x\": true, \"y\": false}");
        rejects(grammar, "{\"id\": 1}", "{\"id\": 1, \"x\": true, \"y\": false, \"z\": true}",
                "{\"id\": 1, \"x\": 3}", "{\"x\": true, \"y\": true}");
        Grammar bare = grammar("{\"type\": \"object\", \"minProperties\": 1, \"maxProperties\": 2}");
        accepts(bare, "{\"k\": 1}", "{\"k\": 1, \"j\": [\"x\"]}");
        rejects(bare, "{}", "{\"a\": 1, \"b\": 2, \"c\": 3}");
        Grammar typeless = grammar("{\"maxProperties\": 1}");
        accepts(typeless, "{}", "{\"a\": null}");
        rejects(typeless, "{\"a\": 1, \"b\": 2}", "[]", "1");
    }

    @Test
    public void unsatisfiablePropertyCountsAreRefused() {
        assertMentions(refusal("{\"type\": \"object\", \"properties\": {\"a\": {}, \"b\": {}},"
                + " \"additionalProperties\": false, \"minProperties\": 3}"), "minProperties 3 exceeds the 2");
        assertMentions(refusal("{\"type\": \"object\", \"properties\": {\"a\": {}, \"b\": {}},"
                + " \"required\": [\"a\", \"b\"], \"maxProperties\": 1}"), "maxProperties 1 is less than the 2");
        assertMentions(refusal("{\"type\": \"object\", \"minProperties\": 3, \"maxProperties\": 2}"),
                "minProperties 3 exceeds maxProperties 2");
        assertMentions(refusal("{\"type\": \"object\", \"minProperties\": -1}"),
                "minProperties must be a non-negative integer");
    }

    // -------------------------------------------------- patternProperties

    @Test
    public void patternPropertiesConstrainMatchingKeys() {
        Grammar grammar = grammar("{\"type\": \"object\", \"patternProperties\": {"
                + "\"^x-[a-z]+$\": {\"type\": \"integer\"}}}");
        accepts(grammar, "{}", "{\"x-ab\": 1}", "{\"x-a\": 1, \"x-b\": 2}");
        rejects(grammar, "{\"y\": 1}", "{\"x-a\": \"s\"}", "{\"x-\": 1}", "{\"x-A\": 1}");
    }

    @Test
    public void unanchoredKeyPatternsMatchAnywhere() {
        Grammar grammar = grammar("{\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"}},"
                + " \"patternProperties\": {\"id\": {\"type\": \"string\"}, \"^n[0-9]\": {\"type\": \"string\"}},"
                + " \"required\": [\"name\"]}");
        accepts(grammar, "{\"name\": \"n\", \"user_id\": \"5\"}", "{\"name\": \"n\", \"id\": \"5\", \"n7x\": \"y\"}");
        rejects(grammar, "{\"name\": \"n\", \"user\": \"5\"}", "{\"name\": \"n\", \"xn7\": \"5\"}",
                "{\"name\": \"n\", \"id\": 5}");
    }

    @Test
    public void patternPropertiesCountAndMatchEdgeKeys() {
        Grammar counted = grammar("{\"type\": \"object\", \"patternProperties\": {\"^[a-c]$\": {\"type\": \"null\"}},"
                + " \"minProperties\": 1, \"maxProperties\": 2}");
        accepts(counted, "{\"a\": null}", "{\"a\": null, \"c\": null}");
        rejects(counted, "{}", "{\"a\": null, \"b\": null, \"c\": null}", "{\"d\": null}");
        Grammar empty = grammar("{\"type\": \"object\", \"patternProperties\": {\"^$\": {\"type\": \"integer\"}}}");
        accepts(empty, "{}", "{\"\": 1}");
        rejects(empty, "{\"a\": 1}");
        Grammar any = grammar("{\"type\": \"object\", \"patternProperties\": {\"\": {\"type\": \"integer\"}}}");
        accepts(any, "{\"anything goes\": 1}");
        rejects(any, "{\"x\": \"1\"}");
    }

    @Test
    public void ambiguousPatternPropertiesAreRefused() {
        assertMentions(refusal("{\"type\": \"object\", \"patternProperties\": {\"^a\": {\"type\": \"integer\"},"
                + " \"^b\": {\"type\": \"string\"}}}"), "patternProperties patterns with different schemas");
        assertMentions(refusal("{\"type\": \"object\", \"patternProperties\": {\"^a\": {}},"
                + " \"additionalProperties\": true}"),
                "additionalProperties cannot be combined with patternProperties");
        assertMentions(refusal("{\"type\": \"object\", \"properties\": {\"abc\": {\"type\": \"string\"}},"
                + " \"patternProperties\": {\"^a\": {\"type\": \"integer\"}}}"),
                "property abc also matches patternProperties pattern ^a");
        assertMentions(refusal("{\"type\": \"object\", \"patternProperties\": {\"^(?=a)\": {}}}"),
                "key pattern ^(?=a) is not supported");
        assertMentions(refusal("{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"string\"}},"
                + " \"patternProperties\": {\"(?<=a)b\": {}}}"),
                "key pattern (?<=a)b is not supported: lookbehind assertion");
    }

    @Test
    public void keyPatternsReadTheFullRegexDialect() {
        Grammar alternatives = grammar("{\"type\": \"object\", \"patternProperties\": {\"^a|b$\": {\"type\": \"integer\"}}}");
        accepts(alternatives, "{\"ax\": 1}", "{\"xb\": 1}", "{\"a\": 1, \"b\": 2}");
        rejects(alternatives, "{\"xa\": 1}", "{\"bx\": 1}");
        Grammar shorthand = grammar("{\"type\": \"object\", \"propertyNames\": {\"pattern\": \"^\\\\w+-\\\\d{2}$\"}}");
        accepts(shorthand, "{\"id-42\": 1}");
        rejects(shorthand, "{\"id-4\": 1}", "{\"i d-42\": 1}");
        assertMentions(refusal("{\"type\": \"object\", \"properties\": {\"x_12\": {\"type\": \"string\"}},"
                + " \"patternProperties\": {\"_\\\\d+$\": {\"type\": \"integer\"}}}"),
                "property x_12 also matches patternProperties pattern _\\d+$");
    }

    @Test
    public void keyPatternsWriteTheMatchedKeysJsonEncoded() {
        Grammar dot = grammar("{\"type\": \"object\", \"patternProperties\": {\"^x.$\": {\"type\": \"integer\"}}}");
        accepts(dot, "{\"xa\": 1}", "{\"x\\\"\": 1}", "{\"x\\\\\": 1}", "{\"x~\": 1}");
        rejects(dot, "{\"x\"\": 1}", "{\"x\\n\": 1}", "{\"xab\": 1}");
        Grammar quoted = grammar("{\"type\": \"object\", \"patternProperties\": {\"^a\\\"b$\": {}}}");
        accepts(quoted, "{\"a\\\"b\": 1}");
        rejects(quoted, "{\"ab\": 1}", "{\"a\"b\": 1}");
        Grammar negated = grammar("{\"type\": \"object\", \"patternProperties\": {\"^[^a]$\": {}}}");
        accepts(negated, "{\"b\": 1}", "{\"\\\"\": 1}", "{\"\\\\\": 1}");
        rejects(negated, "{\"a\": 1}", "{\"\"\": 1}");
        Grammar escaped = grammar("{\"type\": \"object\", \"patternProperties\": {\"^v\\\\.[0-9]\\\\.*$\": {}}}");
        accepts(escaped, "{\"v.1\": 1}", "{\"v.1...\": 1}");
        rejects(escaped, "{\"vx1\": 1}", "{\"v.1.1\": 1}");
        Grammar bracket = grammar("{\"type\": \"object\", \"patternProperties\": {\"[[]\": {\"type\": \"integer\"}}}");
        accepts(bracket, "{\"a[b\": 1}");
        rejects(bracket, "{\"ab\": 1}");
    }

    // -------------------------------------------------- value patterns (xgrammar 0.2.8 verdicts: a full match)

    @Test
    public void valuePatternsMatchTheWholeDecodedValue() {
        Grammar phone = grammar("{\"type\": \"string\", \"pattern\": \"\\\\d{3}-\\\\d{4}\"}");
        acceptsStrings(phone, "555-1234");
        rejectsStrings(phone, "x555-1234", "555-12345", "555-123");
        Grammar start = grammar("{\"type\": \"string\", \"pattern\": \"^a\"}");
        acceptsStrings(start, "a");
        rejectsStrings(start, "abc", "ba");
        Grammar words = grammar("{\"type\": \"string\", \"pattern\": \"^\\\\w+@\\\\w+$\"}");
        acceptsStrings(words, "a@b");
        rejectsStrings(words, "a@", "a@b.c");
    }

    @Test
    public void valuePatternsWriteTheMatchedCharactersJsonEncoded() {
        Grammar any = grammar("{\"type\": \"string\", \"pattern\": \"^[\\\\s\\\\S]{2}$\"}");
        acceptsStrings(any, "\"\\", "\n\t", "a\u0001");
        rejects(any, "\"\"\"\"", "\"\\\"", "\"a\nb\"");
        Grammar negated = grammar("{\"type\": \"string\", \"pattern\": \"^[^a]$\"}");
        acceptsStrings(negated, "\"", "\\", "\n", "b");
        rejectsStrings(negated, "a");
        Grammar inObject = grammar("{\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"string\","
                + " \"pattern\": \"[a-f0-9]{4}\"}}, \"required\": [\"id\"], \"additionalProperties\": false}");
        accepts(inObject, "{\"id\": \"00ff\"}");
        rejects(inObject, "{\"id\": \"00fg\"}", "{\"id\": \"x00ff\"}");
    }

    @Test
    public void unsupportedValuePatternsAreRefusedByName() {
        assertMentions(refusal("{\"type\": \"string\", \"pattern\": \"\\\\bfoo\"}"),
                "pattern \\bfoo is not supported: word boundary assertion: \\b");
        assertMentions(refusal("{\"type\": \"string\", \"pattern\": \"(a)\\\\1\"}"),
                "pattern (a)\\1 is not supported: backreference");
        assertMentions(refusal("{\"type\": \"string\", \"pattern\": \"a{2,1}\"}"), "Invalid pattern a{2,1}");
    }

    @Test
    public void declaredNamesAreMatchedWithEcmaScriptClasses() {
        assertMentions(refusal("{\"type\": \"object\", \"properties\": {\"x[\": {\"type\": \"string\"}},"
                + " \"patternProperties\": {\"[[]\": {\"type\": \"integer\"}}}"),
                "property x[ also matches patternProperties pattern [[]");
        Grammar disjoint = grammar("{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"string\"}},"
                + " \"patternProperties\": {\"[&]\": {\"type\": \"integer\"}}}");
        accepts(disjoint, "{\"a\": \"s\", \"&\": 1}");
        rejects(disjoint, "{\"a\": \"s\", \"b\": 1}");
    }

    // -------------------------------------------------- propertyNames

    @Test
    public void propertyNamesConstrainEveryKey() {
        Grammar pattern = grammar("{\"type\": \"object\", \"propertyNames\": {\"pattern\": \"^[a-z]{2}$\"},"
                + " \"additionalProperties\": {\"type\": \"boolean\"}}");
        accepts(pattern, "{}", "{\"ab\": true}", "{\"ab\": true, \"cd\": false}");
        rejects(pattern, "{\"abc\": true}", "{\"ab\": 1}", "{\"A1\": true}");
        Grammar names = grammar("{\"propertyNames\": {\"enum\": [\"x\", \"y\", 3]}}");
        accepts(names, "{\"x\": 1, \"y\": \"a\"}");
        rejects(names, "{\"z\": 1}", "{\"3\": 1}");
        Grammar length = grammar("{\"type\": \"object\", \"propertyNames\": {\"maxLength\": 3},"
                + " \"maxProperties\": 1}");
        accepts(length, "{\"abc\": 1}", "{}");
        rejects(length, "{\"abcd\": 1}", "{\"a\": 1, \"b\": 2}");
        Grammar format = grammar("{\"type\": \"object\", \"propertyNames\": {\"format\": \"ipv4\"}}");
        accepts(format, "{\"10.0.0.1\": \"up\"}");
        rejects(format, "{\"host\": \"up\"}");
        Grammar none = grammar("{\"type\": \"object\", \"propertyNames\": {\"type\": \"integer\"}}");
        accepts(none, "{}");
        rejects(none, "{\"1\": 1}");
    }

    @Test
    public void propertyNamesBesideOtherKeySourcesAreRefused() {
        assertMentions(refusal("{\"type\": \"object\", \"properties\": {\"a\": {}},"
                + " \"propertyNames\": {\"maxLength\": 2}}"), "propertyNames cannot be combined with properties");
        assertMentions(refusal("{\"type\": \"object\", \"patternProperties\": {\"^a\": {}},"
                + " \"propertyNames\": {\"maxLength\": 2}}"),
                "propertyNames cannot be combined with patternProperties");
        assertMentions(refusal("{\"type\": \"object\", \"propertyNames\": {\"allOf\": [{\"maxLength\": 2}]}}"),
                "propertyNames with allOf is not supported");
    }

    // -------------------------------------------------- boolean schemas

    @Test
    public void booleanSchemasAdmitEverythingOrNothing() {
        Grammar root = grammar("true");
        accepts(root, "1", "{}", "[\"a\"]");
        Grammar properties = grammar("{\"type\": \"object\", \"properties\": {\"a\": true, \"b\": false},"
                + " \"additionalProperties\": true}");
        accepts(properties, "{\"a\": [1]}", "{\"c\": 1}", "{\"a\": 1, \"c\": 2}");
        rejects(properties, "{\"b\": 1}", "{\"a\": 1, \"b\": 2}");
        Grammar alternatives = grammar("{\"anyOf\": [false, {\"type\": \"integer\"}]}");
        accepts(alternatives, "1");
        rejects(alternatives, "\"x\"");
        Grammar patterns = grammar("{\"type\": \"object\", \"patternProperties\": {\"^a\": {\"type\": \"integer\"},"
                + " \"^b\": false}}");
        accepts(patterns, "{\"a1\": 1}");
        rejects(patterns, "{\"b\": 1}");
        Grammar noKeys = grammar("{\"type\": \"object\", \"propertyNames\": false}");
        accepts(noKeys, "{}");
        rejects(noKeys, "{\"a\": 1}");
        Grammar anyKeys = grammar("{\"type\": \"object\", \"propertyNames\": true}");
        accepts(anyKeys, "{\"a\": 1}");
        Grammar closedTuple = grammar("{\"type\": \"array\", \"prefixItems\": [{\"type\": \"integer\"}],"
                + " \"items\": false}");
        accepts(closedTuple, "[1]");
        rejects(closedTuple, "[1, 2]", "[\"a\"]");
        Grammar empty = grammar("{\"type\": \"array\", \"items\": false}");
        accepts(empty, "[]");
        rejects(empty, "[1]");
    }

    @Test
    public void falseSchemasWhereAValueIsNeededAreRefused() {
        assertMentions(refusal("false"), "the schema false admits no value");
        assertMentions(refusal("{\"type\": \"object\", \"properties\": {\"b\": false}, \"required\": [\"b\"]}"),
                "required property b has the schema false");
        assertMentions(refusal("{\"anyOf\": [false, false]}"), "every alternative is the schema false");
        assertMentions(refusal("{\"type\": \"array\", \"items\": false, \"minItems\": 1}"),
                "minItems needs items, but items is the schema false");
        assertMentions(refusal("{\"type\": \"object\", \"properties\": {\"b1\": false},"
                + " \"patternProperties\": {\"^b\": {}}}"), "property b1 also matches patternProperties pattern ^b");
    }

    // -------------------------------------------------- $ref through $id and $anchor

    @Test
    public void refsResolveThroughEmbeddedIds() {
        Grammar absolute = grammar("{\"$defs\": {\"pos\": {\"$id\": \"https://example.com/pos.json\","
                + " \"type\": \"integer\", \"minimum\": 1}}, \"type\": \"object\","
                + " \"properties\": {\"n\": {\"$ref\": \"https://example.com/pos.json\"}}, \"required\": [\"n\"]}");
        accepts(absolute, "{\"n\": 5}");
        rejects(absolute, "{\"n\": 0}");
        Grammar relative = grammar("{\"$id\": \"https://example.com/schemas/root.json\","
                + " \"$defs\": {\"a\": {\"$id\": \"item.json\", \"type\": \"string\", \"maxLength\": 2,"
                + " \"$defs\": {\"b\": {\"type\": \"boolean\"}}}},"
                + " \"type\": \"array\", \"prefixItems\": [{\"$ref\": \"item.json\"},"
                + " {\"$ref\": \"https://example.com/schemas/item.json#/$defs/b\"},"
                + " {\"$ref\": \"root.json#/$defs/a\"}]}");
        accepts(relative, "[\"ab\", true, \"c\"]");
        rejects(relative, "[\"abc\", true, \"c\"]", "[\"ab\", 1, \"c\"]");
    }

    @Test
    public void refsResolveAnchorsInTheirResource() {
        Grammar anchor = grammar("{\"$defs\": {\"s\": {\"$anchor\": \"short\", \"type\": \"string\","
                + " \"maxLength\": 2}}, \"$ref\": \"#short\"}");
        acceptsStrings(anchor, "ab");
        rejectsStrings(anchor, "abc");
        Grammar legacy = grammar("{\"definitions\": {\"t\": {\"$id\": \"#flag\", \"type\": \"boolean\"}},"
                + " \"$ref\": \"#flag\"}");
        accepts(legacy, "true");
        rejects(legacy, "1");
        Grammar scoped = grammar("{\"$defs\": {\"inner\": {\"$id\": \"https://e.com/inner\","
                + " \"$defs\": {\"x\": {\"$anchor\": \"x\", \"type\": \"integer\"}}, \"$ref\": \"#x\"}},"
                + " \"$ref\": \"https://e.com/inner\"}");
        accepts(scoped, "7");
        rejects(scoped, "\"7\"");
    }

    @Test
    public void refPointersUnescapeAndPercentDecode() {
        Grammar escaped = grammar("{\"$defs\": {\"a/b\": {\"type\": \"boolean\"}, \"c d\": {\"type\": \"null\"},"
                + " \"e~f\": {\"type\": \"integer\"}}, \"type\": \"array\", \"prefixItems\": ["
                + "{\"$ref\": \"#/$defs/a~1b\"}, {\"$ref\": \"#/$defs/c%20d\"}, {\"$ref\": \"#/$defs/e~0f\"}]}");
        accepts(escaped, "[true, null, 3]");
        rejects(escaped, "[true, null, \"3\"]");
    }

    @Test
    public void recursiveRootRefsTerminate() {
        Grammar tree = grammar("{\"type\": \"object\", \"properties\": {\"v\": {\"type\": \"integer\"},"
                + " \"kids\": {\"type\": \"array\", \"items\": {\"$ref\": \"#\"}}}, \"required\": [\"v\"]}");
        accepts(tree, "{\"v\": 1}", "{\"v\": 1, \"kids\": [{\"v\": 2, \"kids\": [{\"v\": 3}]}]}");
        rejects(tree, "{\"v\": 1, \"kids\": [{}]}");
    }

    @Test
    public void refTargetsWhoseNamesSanitizeAlikeKeepTheirOwnRules() {
        Grammar grammar = grammar("{\"$defs\": {"
                + "\"a_b\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"integer\"},"
                + " \"next\": {\"$ref\": \"#/$defs/a-b\"}}, \"required\": [\"x\"]},"
                + "\"a-b\": {\"type\": \"object\", \"properties\": {\"y\": {\"type\": \"string\"},"
                + " \"back\": {\"$ref\": \"#/$defs/a_b\"}}, \"required\": [\"y\"]}},"
                + " \"$ref\": \"#/$defs/a_b\"}");
        accepts(grammar, "{\"x\": 1}", "{\"x\": 1, \"next\": {\"y\": \"s\", \"back\": {\"x\": 2}}}");
        rejects(grammar, "{\"y\": \"s\"}", "{\"x\": 1, \"next\": {\"x\": 1}}",
                "{\"x\": 1, \"next\": {\"y\": \"s\", \"back\": {\"y\": \"t\"}}}");
    }

    @Test
    public void unresolvableRefsAreRefusedByName() {
        assertMentions(refusal("{\"$ref\": \"other.json\"}"), "unsupported $ref other.json",
                "remote schemas are not fetched");
        assertMentions(refusal("{\"$ref\": \"#nowhere\"}"), "cannot resolve $ref #nowhere, anchor nowhere not found");
        assertMentions(refusal("{\"$ref\": \"#/$defs/missing\", \"$defs\": {}}"),
                "cannot resolve $ref #/$defs/missing, missing not found");
        assertMentions(refusal("{\"$defs\": {\"a\": {\"$id\": \"https://e.com/a\"},"
                + " \"b\": {\"$id\": \"https://e.com/a\"}}}"), "names a second schema resource https://e.com/a");
    }

    @Test
    public void composedSchemasKeepTheirOwnRefTargets() {
        JsonSchemaToGbnf.Composer composer = JsonSchemaToGbnf.composer(JsonSchemaToGbnf.WhitespaceMode.COMPACT);
        // both documents name their target #/$defs/x; each must keep its own rule
        String first = composer.addSchema("first", "{\"$defs\": {\"x\": {\"type\": \"string\", \"maxLength\": 3}},"
                + " \"$ref\": \"#/$defs/x\"}");
        String second = composer.addSchema("second", "{\"$defs\": {\"x\": {\"type\": \"integer\", \"minimum\": 0}},"
                + " \"type\": \"array\", \"items\": {\"$ref\": \"#/$defs/x\"}}");
        composer.addRule("root", first + " \",\" " + second);
        String gbnf = composer.gbnf();
        assertTrue(gbnf, gbnf.contains("ref-defs-x ::= ") && gbnf.contains("ref-defs-x0 ::= "));
        Grammar grammar = GbnfParser.parse(gbnf);
        accepts(grammar, "\"s\",[1,2]");
        rejects(grammar, "\"s\",[\"t\"]", "\"s\",[-1]", "1,[1]");
    }
}
