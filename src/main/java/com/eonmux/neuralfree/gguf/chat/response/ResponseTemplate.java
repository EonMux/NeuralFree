package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * A checkpoint's {@code response_template} (the Hugging Face response-parsing metadata vLLM's {@code hf} and
 * {@code beam} parsers read; vLLM {@code chat_parsing/response_templates.py}): named fields, each a region of the
 * assistant message between an optional opener and an optional closer — literals ({@code open}, {@code close}) or
 * Python regexes ({@code open_pattern}, {@code close_pattern}) — whose body a content parser ({@link ContentParser})
 * and an optional {@code transform} turn into the field's value. A field without an opener is the implicit region,
 * where text outside every explicit region goes; at most one exists. {@code start_anchor} / {@code start_anchor_pattern}
 * marks where the assistant message starts in a prompt ({@link #truncatePastLastAnchor}), and {@code defaults} seed
 * the parsed message.
 *
 * <p>{@link #load} validates the template as vLLM's loader does, by name; {@link #forServing} further holds it to
 * what an OpenAI response carries — the fields {@code thinking}, {@code content} and {@code tool_calls}, plain text
 * {@code thinking} and {@code content}, defaults naming only {@code role} — and returns its streaming form. Immutable.
 */
public final class ResponseTemplate {

    /** The reasoning field. */
    public static final String THINKING = "thinking";
    /** The answer field. */
    public static final String CONTENT = "content";
    /** The tool-call field. */
    public static final String TOOL_CALLS = "tool_calls";

    private static final Set<String> SERVED = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(THINKING, CONTENT, TOOL_CALLS)));
    private static final Set<String> TEMPLATE_KEYS = new LinkedHashSet<>(Arrays.asList(
            "version", "defaults", "fields", "start_anchor", "start_anchor_pattern"));
    private static final Set<String> FIELD_KEYS = new LinkedHashSet<>(Arrays.asList("open", "open_pattern", "close",
            "close_pattern", "content", "content_args", "repeats", "join", "optional", "transform", "transform_each"));

    private final Map<String, Object> spec;
    private final Map<String, Object> defaults;
    private final Map<String, Field> fields;
    private final String implicit;
    private final Delimiter startAnchor;

    private ResponseTemplate(Map<String, Object> spec, Map<String, Object> defaults, Map<String, Field> fields,
                             String implicit, Delimiter startAnchor) {
        this.spec = spec;
        this.defaults = defaults;
        this.fields = fields;
        this.implicit = implicit;
        this.startAnchor = startAnchor;
    }

    /** One field: its region's delimiters and how its body becomes the value. Immutable. */
    public static final class Field {
        private final String name;
        private final Delimiter open;
        private final Delimiter close;
        private final ContentParser content;
        private final boolean repeats;
        private final String join;
        private final boolean optional;
        private final Transform transform;
        private final boolean transformEach;

        private Field(String name, Delimiter open, Delimiter close, ContentParser content, boolean repeats,
                      String join, boolean optional, Transform transform, boolean transformEach) {
            this.name = name;
            this.open = open;
            this.close = close;
            this.content = content;
            this.repeats = repeats;
            this.join = join;
            this.optional = optional;
            this.transform = transform;
            this.transformEach = transformEach;
        }

        public String name() { return name; }

        /** Whether the region has an opener: false for the implicit region. */
        public boolean opens() { return open != null; }

        /** Whether the region has a closer: false for a region that runs to the end of the output. */
        public boolean closes() { return close != null; }

        /** The opener's literals, or null for a regex opener or none. */
        public List<String> openLiterals() { return open == null ? null : open.literals(); }

        /** The closer's literals, or null for a regex closer or none. */
        public List<String> closeLiterals() { return close == null ? null : close.literals(); }

        /** Whether the field collects every region into a list (or a joined string). */
        public boolean repeats() { return repeats; }

        /** The content parser's name ({@code text}, {@code json}, ...). */
        public String contentParser() { return content.name(); }

        Delimiter open() { return open; }

        Delimiter close() { return close; }

        ContentParser content() { return content; }

        String join() { return join; }

        boolean optional() { return optional; }

        Transform transform() { return transform; }

        boolean transformEach() { return transformEach; }
    }

    /**
     * The template a JSON object text spells.
     *
     * @throws IllegalArgumentException when the text is not JSON or the template is invalid
     */
    public static ResponseTemplate parse(String json) {
        Objects.requireNonNull(json, "json");
        Object spec;
        try {
            spec = Json.parse(json, Json.Limits.PYTHON);
        } catch (JsonException e) {
            throw new IllegalArgumentException("response_template is not valid JSON: " + e.getMessage(), e);
        }
        return load(spec);
    }

    /**
     * Loads and validates a template object (the {@link Json} value model).
     *
     * @throws IllegalArgumentException naming what is invalid: a non-object, a version other than 1, unknown keys,
     *                                  non-object defaults, empty fields, an invalid field, two implicit fields, or no
     *                                  start anchor
     */
    public static ResponseTemplate load(Object spec) {
        if (!(spec instanceof Map)) {
            throw new IllegalArgumentException("response_template must be a dict, got " + Transform.typeName(spec));
        }
        Map<?, ?> raw = (Map<?, ?>) spec;
        Object version = raw.containsKey("version") ? raw.get("version") : 1L;
        if (!(version instanceof Number) || version instanceof Double || ((Number) version).longValue() != 1) {
            throw new IllegalArgumentException("Unsupported response_template version: " + version);
        }
        Set<String> unknown = new TreeSet<>();
        for (Object k : raw.keySet()) {
            if (!TEMPLATE_KEYS.contains(String.valueOf(k))) unknown.add(String.valueOf(k));
        }
        if (!unknown.isEmpty()) throw new IllegalArgumentException("Unknown keys in response_template: " + unknown);
        Object defaults = raw.containsKey("defaults") ? raw.get("defaults") : Collections.emptyMap();
        if (!(defaults instanceof Map)) throw new IllegalArgumentException("response_template.defaults must be a dict");
        Object fieldsRaw = raw.get("fields");
        if (!(fieldsRaw instanceof Map) || ((Map<?, ?>) fieldsRaw).isEmpty()) {
            throw new IllegalArgumentException("response_template.fields must be a non-empty dict");
        }
        Map<String, Field> fields = new LinkedHashMap<>();
        List<String> implicit = new ArrayList<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) fieldsRaw).entrySet()) {
            Field f = field(String.valueOf(e.getKey()), e.getValue());
            fields.put(f.name, f);
            if (f.open == null) implicit.add(f.name);
        }
        if (implicit.size() > 1) {
            throw new IllegalArgumentException("At most one field may omit 'open'/'open_pattern' (that field becomes"
                    + " the implicit-open / leftover sink). Found: " + String.join(", ", implicit));
        }
        Delimiter anchor = Delimiter.of("response_template", raw, "start_anchor", "start_anchor_pattern");
        if (anchor == null) {
            throw new IllegalArgumentException("response_template must define 'start_anchor' or 'start_anchor_pattern'.");
        }
        return new ResponseTemplate(immutableCopy(raw), immutableCopy((Map<?, ?>) defaults),
                Collections.unmodifiableMap(fields), implicit.isEmpty() ? null : implicit.get(0), anchor);
    }

    private static Field field(String name, Object raw) {
        String scope = "Field '" + name + "'";
        if (!(raw instanceof Map)) throw new IllegalArgumentException(scope + " must be a dict");
        Map<?, ?> f = (Map<?, ?>) raw;
        Set<String> unknown = new TreeSet<>();
        for (Object k : f.keySet()) {
            if (!FIELD_KEYS.contains(String.valueOf(k))) unknown.add(String.valueOf(k));
        }
        if (!unknown.isEmpty()) throw new IllegalArgumentException(scope + ": unknown keys " + unknown);
        ContentParser content = ContentParser.of(scope, f.containsKey("content") ? f.get("content") : "text",
                f.get("content_args"));
        Delimiter open = Delimiter.of(scope, f, "open", "open_pattern");
        Delimiter close = Delimiter.of(scope, f, "close", "close_pattern");
        boolean repeats = flag(scope, f, "repeats", false);
        Object join = f.get("join");
        if (join != null && !(join instanceof String)) {
            throw new IllegalArgumentException(scope + ": 'join' must be a string, got " + Transform.typeName(join));
        }
        if (join != null && !repeats) throw new IllegalArgumentException(scope + ": 'join' requires 'repeats': true");
        boolean each = flag(scope, f, "transform_each", false);
        Object transform = f.get("transform");
        if (each && transform == null) {
            throw new IllegalArgumentException(scope + ": transform_each is set but no transform was provided");
        }
        if (transform == null) requireNoCaptures(scope, open, close);
        return new Field(name, open, close, content, repeats, (String) join, flag(scope, f, "optional", true),
                transform == null ? null : Transform.of(scope, immutableValue(transform)), each);
    }

    private static boolean flag(String scope, Map<?, ?> f, String key, boolean byDefault) {
        Object v = f.get(key);
        if (v == null) return byDefault;
        if (!(v instanceof Boolean)) {
            throw new IllegalArgumentException(scope + ": " + key + " must be a bool, got " + Transform.typeName(v));
        }
        return (Boolean) v;
    }

    /** Named captures reach the value only through a transform; a field without one must not declare any. */
    private static void requireNoCaptures(String scope, Delimiter open, Delimiter close) {
        Set<String> names = new TreeSet<>();
        if (open != null) names.addAll(open.regex().groupNames());
        if (close != null) names.addAll(close.regex().groupNames());
        if (!names.isEmpty()) {
            throw new IllegalArgumentException(scope + ": open_pattern/close_pattern declares named group(s) " + names
                    + ", but the field has no 'transform'. Named captures are only surfaced through a 'transform'"
                    + " template (where they appear alongside 'content'). Either add a 'transform' that uses the"
                    + " captures, or remove the named groups from the pattern.");
        }
    }

    /**
     * The template as OpenAI serving parses with it (vLLM {@code validate_response_template_for_serving} and
     * {@code _streaming_template}): only the {@code thinking}, {@code content} and {@code tool_calls} fields and a
     * {@code role} default, plain-text {@code thinking} and {@code content}; an implicit {@code content} field is
     * added when there is none and every field opens explicitly, and every field becomes optional.
     *
     * @throws IllegalArgumentException for a field or default serving cannot route, or a thinking or content field
     *                                  with a non-text parser, a transform or a join
     */
    public ResponseTemplate forServing() {
        Set<String> unsupported = new TreeSet<>();
        for (String f : fields.keySet()) {
            if (!SERVED.contains(f)) unsupported.add(f);
        }
        for (String d : defaults.keySet()) {
            if (!d.equals("role")) unsupported.add(d);
        }
        if (!unsupported.isEmpty()) {
            throw new IllegalArgumentException("response_template contains unsupported semantic fields: " + unsupported
                    + ". Supported fields are: " + new TreeSet<>(SERVED));
        }
        for (String name : Arrays.asList(THINKING, CONTENT)) {
            Field f = fields.get(name);
            if (f != null && (!f.content.name().equals("text") || f.transform != null || f.join != null)) {
                throw new IllegalArgumentException("response_template field '" + name + "' uses semantics that cannot"
                        + " be streamed by the OpenAI serving adapter");
            }
        }
        Map<String, Object> streaming = new LinkedHashMap<>(spec);
        Map<String, Object> streamingFields = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) spec.get("fields")).entrySet()) {
            Map<String, Object> f = new LinkedHashMap<>();
            for (Map.Entry<?, ?> k : ((Map<?, ?>) e.getValue()).entrySet()) f.put(String.valueOf(k.getKey()), k.getValue());
            f.put("optional", Boolean.TRUE);
            streamingFields.put(String.valueOf(e.getKey()), f);
        }
        if (!fields.containsKey(CONTENT) && implicit == null) {
            Map<String, Object> content = new LinkedHashMap<>();
            content.put("optional", Boolean.TRUE);
            streamingFields.put(CONTENT, content);
        }
        streaming.put("fields", streamingFields);
        return load(streaming);
    }

    /**
     * The text after the last start anchor in {@code text} (vLLM {@code truncate_past_last_anchor}): the assistant
     * message a prompt has started; all of {@code text} when it has no anchor.
     */
    public String truncatePastLastAnchor(String text) {
        int last = -1;
        for (TemplateRegex.Match m : startAnchor.regex().findAll(text)) last = m.end;
        return last < 0 ? text : text.substring(last);
    }

    /** The field {@code name}, or null. */
    public Field field(String name) {
        return fields.get(name);
    }

    /** The fields in template order. */
    public Map<String, Field> fields() {
        return fields;
    }

    /** The implicit field's name, or null. */
    public String implicitField() {
        return implicit;
    }

    /** The parsed message's starting values. */
    public Map<String, Object> defaults() {
        return defaults;
    }

    /** The start anchor's regex: its pattern, or the alternation of its literals. */
    TemplateRegex startAnchorRegex() {
        return startAnchor.regex();
    }

    /** The literal strings the assistant message starts after: the start anchor's, else those its regex spells; may be null. */
    public List<String> messageStartAnchors() {
        return startAnchor.spelled();
    }

    /** Every literal delimiter of the template — openers, closers and start anchors — for special-token matching. */
    public List<String> literals() {
        Set<String> out = new LinkedHashSet<>();
        if (startAnchor.literals() != null) out.addAll(startAnchor.literals());
        for (Field f : fields.values()) {
            if (f.openLiterals() != null) out.addAll(f.openLiterals());
            if (f.closeLiterals() != null) out.addAll(f.closeLiterals());
        }
        return new ArrayList<>(out);
    }

    @Override public String toString() {
        return "ResponseTemplate" + fields.keySet();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableCopy(Map<?, ?> m) {
        return (Map<String, Object>) immutableValue(m);
    }

    /** A deep unmodifiable copy of a JSON value. */
    static Object immutableValue(Object v) {
        if (v instanceof Map) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
                out.put(String.valueOf(e.getKey()), immutableValue(e.getValue()));
            }
            return Collections.unmodifiableMap(out);
        }
        if (v instanceof List) {
            List<Object> out = new ArrayList<>();
            for (Object o : (List<?>) v) out.add(immutableValue(o));
            return Collections.unmodifiableList(out);
        }
        return v;
    }
}
