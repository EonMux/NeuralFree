package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * One region's content parser (vLLM {@code chat_parsing/content_parsers.py}): the parser name and its
 * {@code content_args}, checked and compiled when the template loads. The parsers:
 * <ul>
 *   <li>{@code text}: the body, stripped unless {@code strip} is false;</li>
 *   <li>{@code int}, {@code float}, {@code bool}: the stripped body as Python's {@code int()}, {@code float()} and
 *       {@code in ("true", "1")} read it;</li>
 *   <li>{@code json}: {@code json.loads} of the body, with the dialect knobs {@code string_delims} (custom string
 *       delimiters), {@code unquoted_keys} and {@code allow_non_json};</li>
 *   <li>{@code xml-inline}: an object of the {@code key}/{@code value} groups of each {@code tag_pattern} match, an
 *       optional {@code value_parser} reading each value, {@code merge_duplicates} collecting repeated keys;</li>
 *   <li>{@code kv-lines}: an object of {@code key<kv_sep>value} lines split at {@code line_sep}.</li>
 * </ul>
 * Values are the {@link Json} value model. A body the parser cannot read throws {@link IllegalArgumentException}.
 * Immutable.
 */
final class ContentParser {

    /** The parser names, as the template writes them. */
    static final List<String> NAMES = Collections.unmodifiableList(Arrays.asList(
            "text", "int", "float", "bool", "json", "xml-inline", "kv-lines"));
    /** The parsers whose value is the body text itself: their chunks stream as written. */
    private static final List<String> STREAMABLE = Arrays.asList("text", "int", "float", "bool");

    private static final String OPEN_SENTINEL = "\u0001";
    private static final String CLOSE_SENTINEL = "\u0002";
    private static final Pattern UNQUOTED_KEY = Pattern.compile("(?<=[{,])(\\w+):", Pattern.UNICODE_CHARACTER_CLASS);
    private static final Pattern FLOAT_SYNTAX = Pattern.compile("[+\\-._eE]");
    private static final Pattern INTEGER = Pattern.compile("[+-]?\\d+(_\\d+)*");
    private static final Pattern FLOAT = Pattern.compile(
            "[+-]?((\\d+(_\\d+)*)(\\.(\\d+(_\\d+)*)?)?|\\.\\d+(_\\d+)*)([eE][+-]?\\d+(_\\d+)*)?");
    private static final Pattern SPECIAL_FLOAT = Pattern.compile("[+-]?(inf|infinity|nan)", Pattern.CASE_INSENSITIVE);

    private final String name;
    private final Map<String, Object> args;
    /** The compiled {@code tag_pattern} of {@code xml-inline}, else null. */
    private final TemplateRegex tagPattern;
    /** The {@code value_parser} of {@code xml-inline} and {@code kv-lines}, or null. */
    private final ContentParser valueParser;

    private ContentParser(String name, Map<String, Object> args, TemplateRegex tagPattern, ContentParser valueParser) {
        this.name = name;
        this.args = args;
        this.tagPattern = tagPattern;
        this.valueParser = valueParser;
    }

    /**
     * The parser {@code name} with {@code args}.
     *
     * @param scope what declares it, for errors
     * @throws IllegalArgumentException for an unknown name, arguments that are not an object, an {@code xml-inline}
     *                                  without a compiling {@code tag_pattern} that has a {@code key} group, or a
     *                                  {@code value_parser} that is not a valid {@code {name, args}} object
     */
    static ContentParser of(String scope, Object name, Object args) {
        if (!(name instanceof String) || !NAMES.contains(name)) {
            throw new IllegalArgumentException(scope + ": unknown content parser '" + name + "'. Available: "
                    + sortedNames());
        }
        if (args != null && !(args instanceof Map)) {
            throw new IllegalArgumentException(scope + ": 'content_args' must be an object");
        }
        Map<String, Object> a = args == null ? Collections.<String, Object>emptyMap() : stringKeys(scope, (Map<?, ?>) args);
        TemplateRegex tag = null;
        if (name.equals("xml-inline")) {
            Object pattern = a.get("tag_pattern");
            if (!(pattern instanceof String)) {
                throw new IllegalArgumentException(scope + ": xml-inline: 'tag_pattern' content_arg is required");
            }
            tag = TemplateRegex.compile((String) pattern, scope + ": tag_pattern");
            if (!tag.groupNames().contains("key")) {
                throw new IllegalArgumentException(scope + ": xml-inline: tag_pattern must have a named group 'key'."
                        + " Pattern: " + pattern);
            }
        }
        ContentParser value = null;
        Object vp = a.get("value_parser");
        if (vp != null && (name.equals("xml-inline") || name.equals("kv-lines"))) {
            if (!(vp instanceof Map)) {
                throw new IllegalArgumentException(scope + ": 'value_parser' must be an object {name, args}");
            }
            Map<?, ?> spec = (Map<?, ?>) vp;
            value = of(scope + ": value_parser", spec.containsKey("name") ? spec.get("name") : "text", spec.get("args"));
        }
        return new ContentParser((String) name, a, tag, value);
    }

    private static List<String> sortedNames() {
        List<String> sorted = new ArrayList<>(NAMES);
        Collections.sort(sorted);
        return sorted;
    }

    private static Map<String, Object> stringKeys(String scope, Map<?, ?> raw) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : raw.entrySet()) {
            out.put(String.valueOf(e.getKey()), e.getValue());
        }
        return Collections.unmodifiableMap(out);
    }

    String name() {
        return name;
    }

    /** The parser's {@code content_args}, empty when none. */
    Map<String, Object> args() {
        return args;
    }

    /** Whether a region's chunks are its value's text ({@code text}, {@code int}, {@code float}, {@code bool}). */
    boolean streamable() {
        return STREAMABLE.contains(name);
    }

    /**
     * The value of {@code body}.
     *
     * @throws IllegalArgumentException when the body does not parse
     */
    Object parse(String body) {
        switch (name) {
            case "text": return text(body);
            case "int": return integer(text(body));
            case "float": return decimal(text(body));
            case "bool": {
                String t = text(body).toLowerCase(java.util.Locale.ROOT);
                return t.equals("true") || t.equals("1");
            }
            case "json": return json(body);
            case "xml-inline": return xmlInline(body);
            default: return kvLines(body);
        }
    }

    private String text(String s) {
        return truthy(args.containsKey("strip") ? args.get("strip") : Boolean.TRUE) ? s.strip() : s;
    }

    /** Python's {@code int(s)} of a stripped string: a {@link Long}, or a {@link BigInteger} past 64 bits. */
    static Object integer(String s) {
        String t = s.strip();
        if (!INTEGER.matcher(t).matches()) throw new IllegalArgumentException("invalid literal for int(): '" + s + "'");
        BigInteger v = new BigInteger(t.replace("_", ""));
        return v.bitLength() < 64 ? (Object) v.longValue() : v;
    }

    /** Python's {@code float(s)}. */
    static double decimal(String s) {
        String t = s.strip();
        if (SPECIAL_FLOAT.matcher(t).matches()) {
            String lower = t.toLowerCase(java.util.Locale.ROOT);
            if (lower.endsWith("nan")) return Double.NaN;
            return lower.startsWith("-") ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        }
        if (!FLOAT.matcher(t).matches() || FLOAT_SYNTAX.matcher(t).replaceAll("").isEmpty()) {
            throw new IllegalArgumentException("could not convert string to float: '" + s + "'");
        }
        return Double.parseDouble(t.replace("_", ""));
    }

    private Object json(String text) {
        List<?> delims = args.get("string_delims") instanceof List ? (List<?>) args.get("string_delims")
                : Collections.emptyList();
        if (!delims.isEmpty() && (text.contains(OPEN_SENTINEL) || text.contains(CLOSE_SENTINEL))) {
            throw new IllegalArgumentException("json: input contains reserved sentinel characters (\\x01/\\x02);"
                    + " cannot parse safely.");
        }
        String working = text;
        List<String> captured = new ArrayList<>();
        for (Object d : delims) {
            if (!(d instanceof List) || ((List<?>) d).size() != 2) {
                throw new IllegalArgumentException("json: each string_delims entry must be [open, close]");
            }
            List<?> pair = (List<?>) d;
            Matcher m = Pattern.compile(Pattern.quote(String.valueOf(pair.get(0))) + "(.*?)"
                    + Pattern.quote(String.valueOf(pair.get(1))), Pattern.DOTALL).matcher(working);
            StringBuilder out = new StringBuilder();
            while (m.find()) {
                captured.add(m.group(1));
                m.appendReplacement(out, Matcher.quoteReplacement(OPEN_SENTINEL + (captured.size() - 1)
                        + CLOSE_SENTINEL));
            }
            m.appendTail(out);
            working = out.toString();
        }
        if (truthy(args.get("unquoted_keys"))) {
            working = UNQUOTED_KEY.matcher(working).replaceAll("\"$1\":");
        }
        for (int i = 0; i < captured.size(); i++) {
            working = working.replace(OPEN_SENTINEL + i + CLOSE_SENTINEL,
                    Json.toString(captured.get(i), Json.Style.PYTHON_DEFAULT));
        }
        try {
            return Json.parse(working, Json.Limits.PYTHON);
        } catch (JsonException e) {
            if (truthy(args.get("allow_non_json"))) return text(text);
            throw new IllegalArgumentException(working.equals(text)
                    ? "json parser could not parse region as JSON. Content: " + text + " Error: " + e.getMessage()
                    : "json: could not parse after dialect transforms. Original: " + text + " Transformed: "
                    + working + " Error: " + e.getMessage(), e);
        }
    }

    private Map<String, Object> xmlInline(String text) {
        boolean merge = truthy(args.get("merge_duplicates"));
        boolean hasValue = tagPattern.groupNames().contains("value");
        Map<String, Object> out = new LinkedHashMap<>();
        for (TemplateRegex.Match m : tagPattern.findAll(text)) {
            String key = m.captures.get("key");
            if (key == null) {
                throw new IllegalArgumentException("xml-inline: tag_pattern must have a named group 'key'. Pattern: "
                        + tagPattern.source());
            }
            String raw = m.captures.get("value");
            Object value = hasValue && raw == null ? null : subParse(raw == null ? "" : raw);
            put(out, key, value, merge);
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static void put(Map<String, Object> out, String key, Object value, boolean merge) {
        if (merge && out.containsKey(key)) {
            Object prior = out.get(key);
            List<Object> list;
            if (prior instanceof List) {
                list = (List<Object>) prior;
            } else {
                list = new ArrayList<>();
                list.add(prior);
                out.put(key, list);
            }
            list.add(value);
        } else {
            out.put(key, value);
        }
    }

    private Map<String, Object> kvLines(String text) {
        String lineSep = args.containsKey("line_sep") ? String.valueOf(args.get("line_sep")) : "\n";
        String kvSep = args.containsKey("kv_sep") ? String.valueOf(args.get("kv_sep")) : ":";
        if (lineSep.isEmpty() || kvSep.isEmpty()) throw new IllegalArgumentException("kv-lines: empty separator");
        Map<String, Object> out = new LinkedHashMap<>();
        for (String line : text.split(Pattern.quote(lineSep), -1)) {
            String l = text(line);
            int at = l.indexOf(kvSep);
            if (l.isEmpty() || at < 0) continue;
            out.put(text(l.substring(0, at)), subParse(text(l.substring(at + kvSep.length()))));
        }
        return out;
    }

    private Object subParse(String raw) {
        return valueParser == null ? raw : valueParser.parse(raw);
    }

    /** Python truthiness of a JSON value. */
    static boolean truthy(Object v) {
        if (v == null) return false;
        if (v instanceof Boolean) return (Boolean) v;
        if (v instanceof Number) return ((Number) v).doubleValue() != 0;
        if (v instanceof String) return !((String) v).isEmpty();
        if (v instanceof List) return !((List<?>) v).isEmpty();
        if (v instanceof Map) return !((Map<?, ?>) v).isEmpty();
        return true;
    }
}
