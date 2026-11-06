package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf;
import com.eonmux.neuralfree.gguf.grammar.RegexToGbnf;
import com.eonmux.neuralfree.utils.json.Json;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammar.ANY_TEXT;
import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammar.SEPARATOR;
import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammar.constString;
import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammar.constStrings;
import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammar.oneOf;
import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammar.optional;
import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammar.regex;
import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammar.sequence;
import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammar.star;

/**
 * The arguments of a call in a tool-call field's format (vLLM {@code response_template_grammar}'s
 * {@code _arguments_format} and its helpers): a {@code json} field's arguments as a JSON schema whose strings hold no
 * closer, and an {@code xml-inline} field's parameters as tags whose values are raw text, JSON, a string choice or a
 * string format/pattern regex. A format that a closer could end early, or that the parser would read back differently,
 * is not derived (null). Stateless.
 */
final class TemplateArguments {

    /** Keywords that describe a schema without constraining its values. */
    private static final Set<String> ANNOTATIONS = new HashSet<>(Arrays.asList("title", "description", "default",
            "examples", "deprecated", "$comment", "$schema", "$id"));
    private static final Set<String> OBJECT_KEYWORDS = new HashSet<>(Arrays.asList("type", "properties", "required",
            "additionalProperties", "$defs", "definitions"));
    /** Characters JSON writes outside strings. */
    private static final String JSON_SYNTAX = " \t\n\r{}[],:\"-+.0123456789eEtrufalsn";
    private static final Piece TEXT_PIECE = new Piece("[\\s\\S]*", "[\\s\\S]+", false);
    /** JSON holds no closer and ends with a character it writes outside strings. */
    private static final Piece JSON_PIECE = new Piece("[\\s\\S]*", "[\\s\\S]*" + characterClass(JSON_SYNTAX), false);
    private static final String HOST_LABEL = "[a-z0-9](?:[a-z0-9-]*[a-z0-9])?";
    private static final String DATE = "[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])";
    private static final String TIME = "(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\\.[0-9]+)?"
            + "(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])";
    private static final String OCTET = "(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])";
    private static final String ATOM = "[A-Za-z0-9_!#$%&'*+/=?^`{|}~-]+";
    private static final String URI_CHARS = "A-Za-z0-9_.~!$&'()*+,;=";
    private static final String ESCAPED = "%[0-9A-Fa-f]{2}";
    private static final String PATH = "(?:[" + URI_CHARS + ":@-]|" + ESCAPED + ")";
    private static final String QUERY = "(?:[" + URI_CHARS + ":@/?-]|" + ESCAPED + ")*";
    /** Raw string forms of the formats xgrammar holds in JSON strings. */
    private static final Map<String, String> STRING_FORMATS = new LinkedHashMap<>();

    static {
        STRING_FORMATS.put("date", DATE);
        STRING_FORMATS.put("time", TIME);
        STRING_FORMATS.put("date-time", DATE + "T" + TIME);
        STRING_FORMATS.put("email", ATOM + "(?:\\." + ATOM + ")*@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
                + "(?:\\.[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9])*");
        STRING_FORMATS.put("hostname", HOST_LABEL + "(?:\\." + HOST_LABEL + ")*");
        STRING_FORMATS.put("ipv4", "(?:" + OCTET + "\\.){3}" + OCTET);
        STRING_FORMATS.put("uuid", "[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}");
        STRING_FORMATS.put("uri", "[A-Za-z][A-Za-z+.-]*:(?://(?:(?:[" + URI_CHARS + ":-]|" + ESCAPED + ")*@)?"
                + "(?:[" + URI_CHARS + "-]|" + ESCAPED + ")*(?::[0-9]*)?(?:/" + PATH + "*)*"
                + "|/?(?:" + PATH + "+(?:/" + PATH + "*)*)?)(?:\\?" + QUERY + ")?(?:#" + QUERY + ")?");
    }

    private TemplateArguments() {
    }

    /** A Java regex class of exactly {@code chars}. */
    private static String characterClass(String chars) {
        StringBuilder out = new StringBuilder("[");
        chars.chars().forEach(c -> out.append(String.format("\\x{%x}", c)));
        return out.append(']').toString();
    }

    /**
     * One piece of a call's text for {@link #spellsCloser}: its text as a Java regex, the regex of the text it may end
     * a closer's spelling with, and whether the piece may hold a whole closer.
     */
    private static final class Piece {
        final String text;
        final String endings;
        final boolean holds;

        Piece(String text, String endings, boolean holds) {
            this.text = text;
            this.endings = endings;
            this.holds = holds;
        }
    }

    /**
     * The arguments format of {@code field} for a function's {@code parameters}, held to the schema when
     * {@code strict} (vLLM {@code _arguments_format}); null when the template cannot hold it.
     *
     * @throws IllegalArgumentException for a strict parameter name the format cannot encode
     */
    static Map<String, Object> format(ResponseTemplate.Field field, Map<String, Object> parameters, boolean strict,
                                      boolean sortedArguments) {
        List<String> closers = TemplateGrammar.closers(field);
        if (closers == null) closers = Collections.emptyList();
        if (field.contentParser().equals("json") && field.content().args().isEmpty()) {
            if (spellsCloser(closers, Collections.singletonList(JSON_PIECE), false)) return null;
            Object schema = strict ? parameters : Collections.singletonMap("type", "object");
            return jsonFormat(schema, closers);
        }
        if (field.contentParser().equals("xml-inline")) {
            return new Xml(field.content().args(), parameters, strict, closers, sortedArguments).format();
        }
        return null;
    }

    // ================================================================= values

    /**
     * JSON unless a closer is ambiguous with its syntax or the schema cannot hold the closers (vLLM
     * {@code _json_format}): strings hold no closer, as the parser ends the call there.
     */
    static Map<String, Object> jsonFormat(Object schema, List<String> closers) {
        if (!closers.isEmpty()) {
            for (String closer : closers) {
                if (closer.chars().allMatch(c -> JSON_SYNTAX.indexOf(c) >= 0)) return null;
            }
            if (!jsonCompiles(schema, closers)) return null;
        }
        Map<String, Object> f = new LinkedHashMap<>();
        f.put("type", "json_schema");
        f.put("json_schema", schema);
        f.put("excludes", new ArrayList<>(closers));
        return f;
    }

    /** Whether the grammar engine converts {@code schema} with none of {@code closers} in its strings. */
    private static boolean jsonCompiles(Object schema, List<String> closers) {
        try {
            JsonSchemaToGbnf.Composer composer = JsonSchemaToGbnf.composer(null);
            String rule = composer.addSchema("arguments", Json.toString(schema, Json.Style.CANONICAL), closers);
            composer.addRule(Grammar.ROOT_RULE, rule);
            composer.gbnf();
            return true;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    /** JSON for decoded values and raw text for string values (vLLM {@code _value_format}). */
    private static Map<String, Object> valueFormat(Object schema, Map<String, Object> parameters, boolean jsonValues,
                                                   List<String> closers) {
        if (Boolean.FALSE.equals(schema)) return null;
        List<String> types = ToolArguments.schemaTypes(schema);
        if (jsonValues || (!types.isEmpty() && !types.contains("string"))) {
            return jsonFormat(jsonValueSchema(schema, parameters), closers);
        }
        return textFormat(schema, closers);
    }

    private static Map<String, Object> jsonValueSchema(Object schema, Map<String, Object> parameters) {
        if (!(schema instanceof Map)) return Collections.emptyMap();
        Object s = schema;
        if (ContentParser.truthy(((Map<?, ?>) schema).get("nullable"))) {
            // xgrammar ignores `nullable`
            s = Collections.singletonMap("anyOf", Arrays.asList(schema, Collections.singletonMap("type", "null")));
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (String key : new String[]{"$defs", "definitions"}) {
            if (parameters.containsKey(key)) out.put(key, parameters.get(key));
        }
        for (Map.Entry<?, ?> e : ((Map<?, ?>) s).entrySet()) out.put(String.valueOf(e.getKey()), e.getValue());
        return out;
    }

    /** The string choices of a {@code const} or {@code enum} schema, or null (vLLM {@code _string_choices}). */
    static List<String> stringChoices(Object schema) {
        if (!(schema instanceof Map)) return null;
        Map<?, ?> s = (Map<?, ?>) schema;
        Object choices = s.containsKey("const") ? Collections.singletonList(s.get("const")) : s.get("enum");
        if (!(choices instanceof List) || ((List<?>) choices).isEmpty()) return null;
        List<String> out = new ArrayList<>();
        for (Object c : (List<?>) choices) {
            if (!(c instanceof String)) return null;
            out.add((String) c);
        }
        return out;
    }

    /** Whether the parser reads schema values with their JSON type (vLLM {@code _typed_by_parser}). */
    private static boolean typedByParser(Object schema) {
        if (Boolean.TRUE.equals(schema) || !ToolArguments.schemaTypes(schema).isEmpty()) return true;
        return schema instanceof Map && (annotationsOnly((Map<?, ?>) schema) || stringChoices(schema) != null);
    }

    private static boolean annotationsOnly(Map<?, ?> schema) {
        for (Object k : schema.keySet()) {
            if (!ANNOTATIONS.contains(String.valueOf(k))) return false;
        }
        return true;
    }

    /** Keys of {@code schema} outside the annotations and {@code allowed}. */
    private static boolean hasKeysBeyond(Map<?, ?> schema, Set<String> allowed) {
        for (Object k : schema.keySet()) {
            String key = String.valueOf(k);
            if (!ANNOTATIONS.contains(key) && !allowed.contains(key)) return true;
        }
        return false;
    }

    /** Raw text the {@code excludes} end, constrained without changing its parsed value (vLLM {@code _text_format}). */
    private static Map<String, Object> textFormat(Object schema, List<String> excludes) {
        if (Boolean.TRUE.equals(schema)) return ANY_TEXT;
        if (schema instanceof Map && (((Map<?, ?>) schema).containsKey("format")
                || ((Map<?, ?>) schema).containsKey("pattern"))) {
            return stringRegex((Map<?, ?>) schema, excludes);
        }
        if (!(schema instanceof Map) || hasKeysBeyond((Map<?, ?>) schema, new HashSet<>(Arrays.asList("type",
                "nullable", "enum", "const", "anyOf", "oneOf")))) {
            return null;
        }
        Map<?, ?> s = (Map<?, ?>) schema;
        List<Object> branches = new ArrayList<>();
        for (String union : new String[]{"anyOf", "oneOf"}) {
            if (s.get(union) instanceof List) branches.addAll((List<?>) s.get(union));
        }
        if (s.containsKey("enum") || s.containsKey("const")) {
            List<String> choices = stringChoices(schema);
            Set<String> types = new HashSet<>(ToolArguments.schemaTypes(schema));
            types.remove("string");
            if (choices == null || !branches.isEmpty() || !types.isEmpty()) return null;
            return constStrings(choices);
        }
        for (Object branch : branches) {
            if (textFormat(branch, excludes) != ANY_TEXT) return null;
        }
        return ANY_TEXT;
    }

    /**
     * The format or pattern of a raw string (vLLM {@code _string_regex}): like xgrammar, a format overrides a pattern,
     * either drops length bounds, and the pattern matches the whole string; null when a match could end the value.
     */
    private static Map<String, Object> stringRegex(Map<?, ?> schema, List<String> excludes) {
        if (hasKeysBeyond(schema, new HashSet<>(Arrays.asList("type", "format", "pattern", "minLength", "maxLength")))
                || !ToolArguments.schemaTypes(schema).equals(Collections.singletonList("string"))) {
            return null;
        }
        Object pattern = schema.containsKey("format") ? STRING_FORMATS.get(schema.get("format")) : schema.get("pattern");
        if (!(pattern instanceof String)) return null;
        Set<Integer> firsts = new HashSet<>();
        for (String x : excludes) {
            if (!x.isEmpty()) firsts.add(x.codePointAt(0));
        }
        if (RegexToGbnf.mayWrite((String) pattern, firsts)) return null;
        return regex((String) pattern);
    }

    // ================================================================= closers inside call text

    /** All suffixes of {@code literals} as a Java alternation (vLLM {@code _endings}). */
    private static String endings(List<String> literals) {
        List<String> out = new ArrayList<>();
        for (String literal : literals) {
            for (int i = 0; i < literal.length(); i++) out.add(Pattern.quote(literal.substring(i)));
        }
        return String.join("|", out);
    }

    private static Piece literalPiece(List<String> literals, String before) {
        List<String> quoted = new ArrayList<>();
        for (String l : literals) quoted.add(Pattern.quote(l));
        String either = before + "(?:" + String.join("|", quoted) + ")";
        return new Piece(either, either + "|" + endings(literals), true);
    }

    /** Whether the call's framing can spell a closer across its pieces (vLLM {@code _spells_closer}). */
    private static boolean spellsCloser(List<String> closers, List<Piece> pieces, boolean repeats) {
        if (closers.isEmpty()) return false;
        StringBuilder all = new StringBuilder();
        for (Piece p : pieces) all.append(p.text);
        String loop = repeats ? "(?:" + all + ")*" : "";
        List<String> quoted = new ArrayList<>();
        for (String c : closers) quoted.add(Pattern.quote(c));
        // another call or an answer may follow the call's own closer
        String closed = "(?:" + String.join("|", quoted) + ")[\\s\\S]*";
        for (int i = 0; i < pieces.size(); i++) {
            Pattern ends = Pattern.compile(pieces.get(i).endings, Pattern.DOTALL);
            StringBuilder after = new StringBuilder();
            for (int k = i + 1; k < pieces.size(); k++) after.append(pieces.get(k).text);
            Pattern rest = Pattern.compile(after + loop + closed, Pattern.DOTALL);
            for (String closer : closers) {
                if (pieces.get(i).holds && partialFullMatch(ends, closer)) return true;
                // a closer that starts in this piece and ends in the text after it
                for (int split = 1; split < closer.length(); split++) {
                    if (ends.matcher(closer.substring(0, split)).matches()
                            && partialFullMatch(rest, closer.substring(split))) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /** Whether {@code text} matches {@code p} whole, or could with more text after it. */
    private static boolean partialFullMatch(Pattern p, String text) {
        Matcher m = p.matcher(text);
        return m.matches() || m.hitEnd();
    }

    // ================================================================= xml-inline parameters

    /** One {@code xml-inline} derivation (vLLM {@code _xml_arguments_format}). */
    private static final class Xml {
        private final Map<String, Object> contentArgs;
        private final Map<String, Object> parameters;
        private final boolean strict;
        private final List<String> closers;
        private final boolean sortedArguments;
        private List<String> begins;
        private List<String> middles;
        private List<String> ends;
        private List<String> follows;
        private String keyPattern;
        private Map<String, Object> text;
        private List<String> starts;
        private ContentParser readBackParser;

        Xml(Map<String, Object> contentArgs, Map<String, Object> parameters, boolean strict, List<String> closers,
            boolean sortedArguments) {
            this.contentArgs = contentArgs;
            this.parameters = parameters;
            this.strict = strict;
            this.closers = closers;
            this.sortedArguments = sortedArguments;
        }

        Map<String, Object> format() {
            for (String k : contentArgs.keySet()) {
                if (!Arrays.asList("tag_pattern", "value_parser", "merge_duplicates").contains(k)) return null;
            }
            Object vp = contentArgs.get("value_parser");
            Map<?, ?> valueParser = vp instanceof Map && !((Map<?, ?>) vp).isEmpty() ? (Map<?, ?>) vp
                    : Collections.singletonMap("name", "text");
            Object vpName = valueParser.get("name");
            Object vpArgs = valueParser.get("args");
            boolean jsonValues = "json".equals(vpName);
            if (!"text".equals(vpName) && !jsonValues) return null;
            if (jsonValues && vpArgs instanceof Map && !((Map<?, ?>) vpArgs).isEmpty()) return null;
            if (!framing()) return null;
            Object properties = parameters.get("properties");
            boolean typed = properties == null || (properties instanceof Map && (jsonValues
                    || allTyped((Map<?, ?>) properties)));
            if (strict && (hasKeysBeyond(parameters, OBJECT_KEYWORDS) || !typed)) return null;
            Object requiredRaw = parameters.get("required");
            List<?> required = requiredRaw instanceof List ? (List<?>) requiredRaw : Collections.emptyList();
            List<String> names = new ArrayList<>();
            if (strict && properties != null) {
                // required entries that are no names leave the tool to the JSON schema
                if (requiredRaw != null && !(requiredRaw instanceof List)) return null;
                for (Object r : required) {
                    if (!(r instanceof String)) return null;
                }
                for (Object k : ((Map<?, ?>) properties).keySet()) names.add(String.valueOf(k));
                for (Object r : required) {
                    if (!names.contains(r)) names.add((String) r);
                }
            }
            List<Piece> pieces = Arrays.asList(literalPiece(begins, "\\n?"),
                    new Piece("(?:" + TemplateRegex.javaPattern(keyPattern) + ")", endings(names), false),
                    literalPiece(middles, ""), TEXT_PIECE, literalPiece(ends, ""));
            if (spellsCloser(closers, pieces, true)) return null;
            // raw text has no escaping for a call closer, nor, where a value may end before the next opener, for any
            // value end
            List<String> excludes = new ArrayList<>(closers);
            if (!follows.isEmpty()) {
                excludes.addAll(follows);
                excludes.addAll(ends);
            }
            text = excludes.isEmpty() ? ANY_TEXT : TemplateGrammar.anyText(excludes);
            // each value but the last then ends by opening the next parameter, whose tag starts after the opener
            starts = follows.isEmpty() ? begins : Collections.singletonList("");
            if (!strict) return anyParameters();
            if (properties == null) {
                Object extra = parameters.containsKey("additionalProperties") ? parameters.get("additionalProperties")
                        : Boolean.TRUE;
                if (Boolean.FALSE.equals(extra)) return constString("");
                return Boolean.TRUE.equals(extra) || (extra instanceof Map && ((Map<?, ?>) extra).isEmpty())
                        ? anyParameters() : null;
            }
            // a declared key repeated as an extra one would bypass its schema
            Object extra = parameters.get("additionalProperties");
            if (extra != null && !Boolean.FALSE.equals(extra)) return null;
            readBackParser = ContentParser.of("tool_calls", "xml-inline", contentArgs);
            return strictParameters(names, (Map<?, ?>) properties, required, jsonValues);
        }

        /** The begins, key pattern, middles, ends and follows of the tag pattern; false when underivable. */
        private boolean framing() {
            Object tagPattern = contentArgs.get("tag_pattern");
            if (!(tagPattern instanceof String)) return false;
            String[] keySplit = TemplateGrammar.splitGroup((String) tagPattern, "key");
            String[] valueSplit = keySplit == null ? null : TemplateGrammar.splitGroup(keySplit[2], "value");
            if (valueSplit == null) return false;
            keyPattern = keySplit[1];
            begins = LiteralPatterns.expand(keySplit[0]);
            middles = LiteralPatterns.expand(valueSplit[0]);
            List<String> allEnds = LiteralPatterns.expand(valueSplit[2], false, true);
            if (begins == null || begins.isEmpty() || middles == null || middles.isEmpty() || allEnds == null
                    || allEnds.isEmpty()) {
                return false;
            }
            ends = new ArrayList<>();
            follows = new ArrayList<>();
            Set<String> aheads = new LinkedHashSet<>();
            for (String end : allEnds) {
                int at = end.indexOf(LiteralPatterns.AHEAD);
                if (at < 0) {
                    ends.add(end);
                } else {
                    aheads.add(end.substring(at + 1));
                    follows.add(end.substring(0, at) + end.substring(at + 1));
                }
            }
            // a value may also end where the next parameter opens, if the template looks ahead for its only opener
            return !ends.isEmpty() && (aheads.isEmpty()
                    || (begins.size() == 1 && aheads.equals(new LinkedHashSet<>(begins))));
        }

        private boolean allTyped(Map<?, ?> properties) {
            for (Object v : properties.values()) {
                if (!typedByParser(v)) return false;
            }
            return true;
        }

        /** A parameter tag; each value but the last also ends by opening the next parameter. */
        private Map<String, Object> tag(String begin, Map<String, Object> value, List<String> fits, boolean last) {
            if (follows.isEmpty() || last) return TemplateGrammar.tag(begin, value, fits);
            List<String> end = new ArrayList<>();
            for (String e : fits) {
                end.add(e + begins.get(0));
                end.add(e + "\n" + begins.get(0));
            }
            end.addAll(follows);
            return TemplateGrammar.tag(begin, value, end);
        }

        private Map<String, Object> separated(List<Map<String, Object>> tags) {
            return follows.isEmpty() ? sequence(SEPARATOR, oneOf(tags)) : oneOf(tags);
        }

        /** {@code body} after the opener of its first parameter. */
        private Map<String, Object> arguments(Map<String, Object> body, boolean optional) {
            if (follows.isEmpty()) return body;
            Map<String, Object> framed = sequence(SEPARATOR, constStrings(begins), body);
            return optional ? optional(framed) : framed;
        }

        private Map<String, Object> anyParameter(boolean last) {
            Map<String, Object> content = sequence(regex(keyPattern), constStrings(middles), text);
            List<Map<String, Object>> tags = new ArrayList<>();
            for (String start : starts) tags.add(tag(start, content, ends, last));
            return separated(tags);
        }

        private Map<String, Object> anyParameters() {
            Map<String, Object> more = star(anyParameter(false));
            return follows.isEmpty() ? more : arguments(sequence(more, anyParameter(true)), true);
        }

        /** The arguments the parser reads from {@code written}, unless a closer ends the call in it. */
        private Map<?, ?> readBack(String written) {
            for (String closer : closers) {
                if (written.contains(closer)) return null;
            }
            try {
                Object v = readBackParser.parse(written);
                return v instanceof Map ? (Map<?, ?>) v : null;
            } catch (IllegalArgumentException e) {
                return null;
            }
        }

        private Map<String, Object> parameter(String key, Map<String, Object> value, List<String> choices,
                                              boolean last) {
            if (!TemplateRegex.compile(keyPattern, "parameter name pattern").fullMatches(key)) {
                throw unencodable(key);
            }
            for (String b : begins) {
                for (String m : middles) {
                    for (String e : ends) {
                        Map<?, ?> read = readBack(b + key + m + "0" + e);
                        if (read == null || !new ArrayList<>(read.keySet()).equals(Collections.singletonList(key))) {
                            throw unencodable(key);
                        }
                    }
                }
            }
            List<Map<String, Object>> tags = new ArrayList<>();
            if (choices == null) {
                for (String s : starts) {
                    for (String m : middles) tags.add(tag(s + key + m, value, ends, last));
                }
                return separated(tags);
            }
            // optional framing whitespace must not change a text choice
            Set<String> framed = new HashSet<>();
            for (int i = 0; i < begins.size() && i < starts.size(); i++) {
                for (String m : middles) {
                    Map<List<String>, List<String>> byEnds = new LinkedHashMap<>();
                    for (String choice : choices) {
                        List<String> fits = new ArrayList<>();
                        for (String e : ends) {
                            Map<?, ?> read = readBack(begins.get(i) + key + m + choice + e);
                            if (read != null && read.equals(Collections.singletonMap(key, choice))) fits.add(e);
                        }
                        byEnds.computeIfAbsent(fits, k -> new ArrayList<>()).add(choice);
                    }
                    for (Map.Entry<List<String>, List<String>> g : byEnds.entrySet()) {
                        if (g.getKey().isEmpty()) continue;
                        tags.add(tag(starts.get(i) + key + m, constStrings(g.getValue()), g.getKey(), last));
                        framed.addAll(g.getValue());
                    }
                }
            }
            return framed.equals(new HashSet<>(choices)) ? separated(tags) : null;
        }

        private IllegalArgumentException unencodable(String key) {
            return new IllegalArgumentException("parameter name '" + key + "' cannot be encoded in tool calls");
        }

        private Map<String, Object> strictParameters(List<String> names, Map<?, ?> properties, List<?> required,
                                                     boolean jsonValues) {
            if (names.isEmpty()) return constString("");
            Map<String, Map<String, Object>> more = new LinkedHashMap<>();
            Map<String, Map<String, Object>> fin = new LinkedHashMap<>();
            List<String> valueEnds = new ArrayList<>(closers);
            valueEnds.addAll(follows);
            valueEnds.addAll(ends);
            for (String key : names) {
                // a required name without a schema takes any value
                Object schema = properties.containsKey(key) ? properties.get(key) : Boolean.TRUE;
                Map<String, Object> value = valueFormat(schema, parameters, jsonValues, valueEnds);
                if (value == null) return null;
                List<String> choices = null;
                if ("const_string".equals(value.get("type")) || "or".equals(value.get("type"))) {
                    choices = stringChoices(schema);
                }
                if (value.equals(ANY_TEXT)) value = text;
                Map<String, Object> joined = parameter(key, value, choices, false);
                Map<String, Object> ending = parameter(key, value, choices, true);
                if (joined == null || ending == null) return null;
                more.put(key, joined);
                fin.put(key, ending);
            }
            List<List<String>> orders = new ArrayList<>();
            orders.add(names);
            List<String> sorted = new ArrayList<>(names);
            Collections.sort(sorted);
            if (sortedArguments && !sorted.equals(names)) orders.add(sorted);
            List<Map<String, Object>> alternatives = new ArrayList<>();
            for (List<String> keys : orders) alternatives.add(ordered(keys, more, fin, required));
            return arguments(oneOf(alternatives), required.isEmpty());
        }

        private Map<String, Object> ordered(List<String> keys, Map<String, Map<String, Object>> more,
                                            Map<String, Map<String, Object>> fin, List<?> required) {
            List<Map<String, Object>> before = new ArrayList<>();
            for (String k : keys) before.add(required.contains(k) ? more.get(k) : optional(more.get(k)));
            if (follows.isEmpty()) return sequence(before);
            // the last parameter follows every required one
            int first = 0;
            for (int i = 0; i < keys.size(); i++) {
                if (required.contains(keys.get(i))) first = i;
            }
            List<Map<String, Object>> out = new ArrayList<>(before.subList(0, first));
            out.add(last(keys, before, fin, first, keys.size()));
            return sequence(out);
        }

        /** Parameters of {@code keys[start, stop)} that end at one of them; halving keeps the tag O(n log n). */
        private Map<String, Object> last(List<String> keys, List<Map<String, Object>> before,
                                         Map<String, Map<String, Object>> fin, int start, int stop) {
            if (stop - start == 1) return fin.get(keys.get(start));
            int middle = (start + stop) / 2;
            List<Map<String, Object>> right = new ArrayList<>(before.subList(start, middle));
            right.add(last(keys, before, fin, middle, stop));
            return oneOf(Arrays.asList(last(keys, before, fin, start, middle), sequence(right)));
        }
    }
}
