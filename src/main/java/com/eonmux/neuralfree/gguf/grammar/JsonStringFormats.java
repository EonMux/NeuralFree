package com.eonmux.neuralfree.gguf.grammar;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The JSON-schema string {@code format} values the converter constrains, and their GBNF builtin rules. The set is
 * xgrammar's ({@code STRING_SUPPORTED_FORMATS} of vLLM's xgrammar backend): llama.cpp's {@code date}, {@code time},
 * {@code date-time} and {@code uuid} (rules verbatim from {@code common/json-schema-to-grammar.cpp:251-258}) plus
 * {@code email}, {@code duration}, {@code ipv4}, {@code ipv6}, {@code hostname}, {@code uri},
 * {@code uri-reference}, {@code uri-template}, {@code json-pointer} and {@code relative-json-pointer}.
 *
 * <p>Each rule follows the format's defining grammar: RFC 5321 {@code Mailbox} (dot-string or quoted local part;
 * a hostname, IPv4 or IPv6 address-literal domain), RFC 3339 appendix A {@code duration}, the RFC 2673 dotted quad
 * without leading zeros, RFC 3986 {@code IPv6address}, {@code URI} and {@code URI-reference} (IPvFuture included),
 * RFC 1123 host names (labels of at most 63 letters, digits and inner hyphens), RFC 6570 level-4 templates over
 * ASCII literals, RFC 6901 pointers and the relative JSON pointers of draft-bhutton-relative-json-pointer-00 (index
 * manipulation included). The rules match the JSON text between the quotes, so a character that JSON must escape
 * appears only in its escaped form: a quoted e-mail local part is written {@code \"…\"}, a pointer escapes
 * {@code "} and {@code \}. Each format also has the rule of its raw text, unquoted and unescaped, the value of an
 * XML parameter. Every rule name is reserved against schema-derived rule names.</p>
 *
 * <p>A {@code format} outside the set is an annotation, as in llama.cpp and xgrammar: the string stays generic.</p>
 */
final class JsonStringFormats {

    /** One supported format: its keyword and the builtin rule that matches a JSON string of it. */
    enum Format {
        NONE(null, null),
        UUID("uuid", "uuid-value"),
        DATE("date-string", "date"),
        TIME("time-string", "time"),
        DATE_TIME("date-time-string", "date-time"),
        EMAIL("email-string", "email-raw-address"),
        DURATION("duration-string", "duration-value"),
        IPV4("ipv4-string", "ipv4-address"),
        IPV6("ipv6-string", "ipv6-address"),
        HOSTNAME("hostname-string", "hostname-labels"),
        URI("uri-string", "uri-absolute"),
        URI_REFERENCE("uri-reference-string", "uri-reference-value"),
        URI_TEMPLATE("uri-template-string", "uri-template-value"),
        JSON_POINTER("json-pointer-string", "json-pointer-raw-value"),
        RELATIVE_JSON_POINTER("relative-json-pointer-string", "relative-json-pointer-raw-value");

        /** The builtin rule name of the quoted string; {@code uuid} is one of llama.cpp's primitive rules. */
        final String rule;
        /** The builtin rule name of the raw text, as an XML parameter value is written ({@link XmlParameterRules}). */
        final String raw;

        Format(String rule, String raw) {
            this.rule = rule;
            this.raw = raw;
        }
    }

    /** The builtin rules of every format but {@code uuid}, helpers included, keyed by rule name. */
    static final Map<String, JsonSchemaToGbnf.BuiltinRule> RULES;

    private static final Map<String, Format> KEYWORDS = new LinkedHashMap<>();

    /** A JSON string of the {@code body} rule. */
    private static String quoted(String body) {
        return "\"\\\"\" " + body + " \"\\\"\"";
    }

    private static void put(Map<String, JsonSchemaToGbnf.BuiltinRule> rules, String name, String content,
            String... deps) {
        rules.put(name, new JsonSchemaToGbnf.BuiltinRule(content, deps));
    }

    static {
        KEYWORDS.put("date", Format.DATE);
        KEYWORDS.put("time", Format.TIME);
        KEYWORDS.put("date-time", Format.DATE_TIME);
        KEYWORDS.put("email", Format.EMAIL);
        KEYWORDS.put("duration", Format.DURATION);
        KEYWORDS.put("ipv4", Format.IPV4);
        KEYWORDS.put("ipv6", Format.IPV6);
        KEYWORDS.put("hostname", Format.HOSTNAME);
        KEYWORDS.put("uri", Format.URI);
        KEYWORDS.put("uri-reference", Format.URI_REFERENCE);
        KEYWORDS.put("uri-template", Format.URI_TEMPLATE);
        KEYWORDS.put("json-pointer", Format.JSON_POINTER);
        KEYWORDS.put("relative-json-pointer", Format.RELATIVE_JSON_POINTER);

        Map<String, JsonSchemaToGbnf.BuiltinRule> rules = new LinkedHashMap<>();
        rules.put("uuid-value", new JsonSchemaToGbnf.BuiltinRule("[0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4}"
                + " \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12}", new String[0]));
        dateTimeRules(rules);
        addressRules(rules);
        emailRules(rules);
        durationRules(rules);
        uriRules(rules);
        pointerRules(rules);
        RULES = Collections.unmodifiableMap(rules);
    }

    private JsonStringFormats() {
    }

    /**
     * The format named by a {@code format} keyword value, llama.cpp {@code get_format} extended to the xgrammar
     * set; {@code uuid} and its {@code uuid1}..{@code uuid5} spellings are llama.cpp's. Unknown values are
     * {@link Format#NONE}.
     */
    static Format of(String keyword) {
        if ("uuid".equals(keyword) || (keyword.length() == 5 && keyword.startsWith("uuid")
                && keyword.charAt(4) >= '1' && keyword.charAt(4) <= '5')) {
            return Format.UUID;
        }
        Format format = KEYWORDS.get(keyword);
        return format == null ? Format.NONE : format;
    }

    /** llama.cpp {@code STRING_FORMAT_RULES} ({@code common/json-schema-to-grammar.cpp:251-258}). */
    private static void dateTimeRules(Map<String, JsonSchemaToGbnf.BuiltinRule> rules) {
        put(rules, "date", "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] )"
                + " \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )");
        put(rules, "time", "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9]"
                + " \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" )"
                + " ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )");
        put(rules, "date-time", "date \"T\" time", "date", "time");
        put(rules, "date-string", quoted("date"), "date");
        put(rules, "time-string", quoted("time"), "time");
        put(rules, "date-time-string", quoted("date-time"), "date-time");
    }

    /** The dotted quad, RFC 3986 {@code IPv6address} and RFC 1123 host names. */
    private static void addressRules(Map<String, JsonSchemaToGbnf.BuiltinRule> rules) {
        put(rules, "ipv4-octet", "\"25\" [0-5] | \"2\" [0-4] [0-9] | \"1\" [0-9] [0-9] | [1-9] [0-9] | [0-9]");
        put(rules, "ipv4-address", "ipv4-octet \".\" ipv4-octet \".\" ipv4-octet \".\" ipv4-octet",
                "ipv4-octet");
        put(rules, "ipv4-string", quoted("ipv4-address"), "ipv4-address");

        put(rules, "ipv6-h16", "[0-9a-fA-F]{1,4}");
        put(rules, "ipv6-ls32", "ipv6-h16 \":\" ipv6-h16 | ipv4-address", "ipv6-h16", "ipv4-address");
        String h16c = "(ipv6-h16 \":\")";
        put(rules, "ipv6-address", h16c + "{6} ipv6-ls32"
                + " | \"::\" " + h16c + "{5} ipv6-ls32"
                + " | ipv6-h16? \"::\" " + h16c + "{4} ipv6-ls32"
                + " | (" + h16c + "{0,1} ipv6-h16)? \"::\" " + h16c + "{3} ipv6-ls32"
                + " | (" + h16c + "{0,2} ipv6-h16)? \"::\" " + h16c + "{2} ipv6-ls32"
                + " | (" + h16c + "{0,3} ipv6-h16)? \"::\" ipv6-h16 \":\" ipv6-ls32"
                + " | (" + h16c + "{0,4} ipv6-h16)? \"::\" ipv6-ls32"
                + " | (" + h16c + "{0,5} ipv6-h16)? \"::\" ipv6-h16"
                + " | (" + h16c + "{0,6} ipv6-h16)? \"::\"", "ipv6-h16", "ipv6-ls32");
        put(rules, "ipv6-string", quoted("ipv6-address"), "ipv6-address");

        put(rules, "hostname-label", "[a-zA-Z0-9] ([a-zA-Z0-9-]{0,61} [a-zA-Z0-9])?");
        put(rules, "hostname-labels", "hostname-label (\".\" hostname-label)*", "hostname-label");
        put(rules, "hostname-string", quoted("hostname-labels"), "hostname-labels");
    }

    /** RFC 5321 {@code Mailbox}: the quoted local part in its JSON-escaped form. */
    private static void emailRules(Map<String, JsonSchemaToGbnf.BuiltinRule> rules) {
        put(rules, "email-atext", "[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]");
        put(rules, "email-dot-string", "email-atext+ (\".\" email-atext+)*", "email-atext");
        String qtext = "[\\x20\\x21\\x23-\\x5B\\x5D-\\x7E]";
        String escapedQuote = "\"\\\\\\\"\"";
        String escapedBackslash = "\"\\\\\\\\\"";
        put(rules, "email-quoted", escapedQuote + " ( " + qtext + " | " + escapedBackslash
                + " ( " + qtext + " | " + escapedQuote + " | " + escapedBackslash + " ) )* " + escapedQuote);
        put(rules, "email-domain", "hostname-labels | \"[\" ipv4-address \"]\" | \"[IPv6:\" ipv6-address \"]\"",
                "hostname-labels", "ipv4-address", "ipv6-address");
        put(rules, "email-address", "(email-dot-string | email-quoted) \"@\" email-domain",
                "email-dot-string", "email-quoted", "email-domain");
        put(rules, "email-string", quoted("email-address"), "email-address");
        String quote = "\"\\\"\"";
        String backslash = "\"\\\\\"";
        put(rules, "email-raw-quoted", quote + " ( " + qtext + " | " + backslash + " ( " + qtext + " | " + quote
                + " | " + backslash + " ) )* " + quote);
        put(rules, "email-raw-address", "(email-dot-string | email-raw-quoted) \"@\" email-domain",
                "email-dot-string", "email-raw-quoted", "email-domain");
    }

    /** RFC 3339 appendix A {@code duration}. */
    private static void durationRules(Map<String, JsonSchemaToGbnf.BuiltinRule> rules) {
        put(rules, "duration-second", "[0-9]+ \"S\"");
        put(rules, "duration-minute", "[0-9]+ \"M\" duration-second?", "duration-second");
        put(rules, "duration-hour", "[0-9]+ \"H\" duration-minute?", "duration-minute");
        put(rules, "duration-time", "\"T\" (duration-hour | duration-minute | duration-second)",
                "duration-hour", "duration-minute", "duration-second");
        put(rules, "duration-day", "[0-9]+ \"D\"");
        put(rules, "duration-month", "[0-9]+ \"M\" duration-day?", "duration-day");
        put(rules, "duration-year", "[0-9]+ \"Y\" duration-month?", "duration-month");
        put(rules, "duration-value", "\"P\" ((duration-day | duration-month | duration-year) duration-time?"
                + " | duration-time | [0-9]+ \"W\")",
                "duration-day", "duration-month", "duration-year", "duration-time");
        put(rules, "duration-string", quoted("duration-value"), "duration-value");
    }

    /** RFC 3986 {@code URI} and {@code URI-reference}, RFC 6570 templates. */
    private static void uriRules(Map<String, JsonSchemaToGbnf.BuiltinRule> rules) {
        String unreservedSubDelims = "a-zA-Z0-9._~!$&'()*+,;=";
        put(rules, "uri-pct", "\"%\" [0-9a-fA-F] [0-9a-fA-F]");
        put(rules, "uri-pchar", "[" + unreservedSubDelims + ":@-] | uri-pct", "uri-pct");
        put(rules, "uri-scheme", "[a-zA-Z] [a-zA-Z0-9+.-]*");
        put(rules, "uri-userinfo", "([" + unreservedSubDelims + ":-] | uri-pct)*", "uri-pct");
        put(rules, "uri-reg-name", "([" + unreservedSubDelims + "-] | uri-pct)*", "uri-pct");
        put(rules, "uri-ip-literal", "\"[\" (ipv6-address | \"v\" [0-9a-fA-F]+ \".\" [" + unreservedSubDelims
                + ":-]+) \"]\"", "ipv6-address");
        put(rules, "uri-host", "uri-ip-literal | ipv4-address | uri-reg-name",
                "uri-ip-literal", "ipv4-address", "uri-reg-name");
        put(rules, "uri-authority", "(uri-userinfo \"@\")? uri-host (\":\" [0-9]*)?",
                "uri-userinfo", "uri-host");
        put(rules, "uri-path-abempty", "(\"/\" uri-pchar*)*", "uri-pchar");
        put(rules, "uri-path-absolute", "\"/\" (uri-pchar+ uri-path-abempty)?", "uri-pchar", "uri-path-abempty");
        put(rules, "uri-query", "([" + unreservedSubDelims + ":@/?-] | uri-pct)*", "uri-pct");
        put(rules, "uri-tail", "(\"?\" uri-query)? (\"#\" uri-query)?", "uri-query");
        put(rules, "uri-absolute", "uri-scheme \":\" (\"//\" uri-authority uri-path-abempty"
                + " | uri-path-absolute | uri-pchar+ uri-path-abempty)? uri-tail",
                "uri-scheme", "uri-authority", "uri-path-abempty", "uri-path-absolute", "uri-pchar", "uri-tail");
        put(rules, "uri-string", quoted("uri-absolute"), "uri-absolute");
        put(rules, "uri-relative", "(\"//\" uri-authority uri-path-abempty | uri-path-absolute"
                + " | ([" + unreservedSubDelims + "@-] | uri-pct)+ uri-path-abempty)? uri-tail",
                "uri-authority", "uri-path-abempty", "uri-path-absolute", "uri-pct", "uri-tail");
        put(rules, "uri-reference-value", "uri-absolute | uri-relative", "uri-absolute", "uri-relative");
        put(rules, "uri-reference-string", quoted("uri-reference-value"), "uri-reference-value");

        put(rules, "uri-template-literal",
                "[\\x21\\x23-\\x24\\x26\\x28-\\x3B\\x3D\\x3F-\\x5B\\x5D\\x5F\\x61-\\x7A\\x7E] | uri-pct", "uri-pct");
        put(rules, "uri-template-varchar", "[a-zA-Z0-9_] | uri-pct", "uri-pct");
        put(rules, "uri-template-varspec", "uri-template-varchar (\".\"? uri-template-varchar)*"
                + " (\":\" [1-9] [0-9]{0,3} | \"*\")?", "uri-template-varchar");
        put(rules, "uri-template-expression", "\"{\" [+#./;?&=,!@|]? uri-template-varspec"
                + " (\",\" uri-template-varspec)* \"}\"", "uri-template-varspec");
        put(rules, "uri-template-value", "(uri-template-literal | uri-template-expression)*",
                "uri-template-literal", "uri-template-expression");
        put(rules, "uri-template-string", quoted("uri-template-value"), "uri-template-value");
    }

    /** RFC 6901 pointers and relative JSON pointers; '/' and '~' only in their pointer roles. */
    private static void pointerRules(Map<String, JsonSchemaToGbnf.BuiltinRule> rules) {
        put(rules, "json-pointer-char", "[^\"\\\\\\x7F\\x00-\\x1F/~] | [\\\\] [\"\\\\bfnrt] | \"~\" [01]");
        put(rules, "json-pointer-value", "(\"/\" json-pointer-char*)*", "json-pointer-char");
        put(rules, "json-pointer-string", quoted("json-pointer-value"), "json-pointer-value");
        String nonNegative = "(\"0\" | [1-9] [0-9]*)";
        put(rules, "relative-json-pointer-value", nonNegative + " ([+-] " + nonNegative + ")?"
                + " (\"#\" | json-pointer-value)", "json-pointer-value");
        put(rules, "relative-json-pointer-string", quoted("relative-json-pointer-value"),
                "relative-json-pointer-value");
        put(rules, "json-pointer-raw-char", "[^/~] | \"~\" [01]");
        put(rules, "json-pointer-raw-value", "(\"/\" json-pointer-raw-char*)*", "json-pointer-raw-char");
        put(rules, "relative-json-pointer-raw-value", nonNegative + " ([+-] " + nonNegative + ")?"
                + " (\"#\" | json-pointer-raw-value)", "json-pointer-raw-value");
    }
}
