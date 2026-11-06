package com.eonmux.neuralfree.gguf.chat.response;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The streaming parser of a {@link ResponseTemplate} (vLLM {@code chat_parsing/response_parser.py}): text fed in pieces
 * opens and closes the template's regions, each event reported to a {@link Listener}. The parser starts in the
 * implicit region (or a sink that drops text when there is none) and returns there after every close; inside an
 * explicit region only its closer is watched, elsewhere every opener and the implicit region's closer. A delimiter
 * that more input could still complete — a partial match at the end of the input, a regex match ending there, or a
 * literal another literal extends — holds back the text from its start until the input decides.
 *
 * <p>The parser first consumes the chat prompt after its last start anchor ({@link ResponseTemplate#truncatePastLastAnchor}),
 * so a region the template pre-writes ({@code <think>} in the generation prompt) is open when the output begins; text
 * held back at the prompt's end belongs to the prompt and never reaches a chunk. A region's value is its body through
 * the field's content parser and transform, tool-call arguments typed by the request's tool schemas; a value that does
 * not parse ends the region as malformed and parsing goes on. Not thread-safe.
 */
public final class ResponseParser {

    /** Receives the parser's events; positions index {@link #inputText()}. */
    public interface Listener {
        /** A region opened; an implicit region opens on its first text with an empty span. */
        default void opened(String field, int start, int end, Map<String, String> captures) {
        }

        /**
         * Text of the open region; {@code raw} is true for a structured parser ({@code json}, {@code xml-inline},
         * {@code kv-lines}) whose chunks are not its value.
         */
        default void chunk(String field, String text, boolean raw) {
        }

        /** A region closed with its value; the span is its closer, empty when the end of the output closed it. */
        default void closed(String field, int start, int end, Object value) {
        }

        /** A region's body did not parse; it ends without a value. */
        default void malformed(String field, int start, int end, String error) {
        }
    }

    private static final Listener IGNORE = new Listener() {
    };

    private final ResponseTemplate template;
    private final ToolArguments tools;
    private final StringBuilder buffer = new StringBuilder();
    private final Map<String, Object> output = new LinkedHashMap<>();
    private final Set<String> malformed = new LinkedHashSet<>();
    private int pos;
    private int prefixLength;
    private String current;
    private Map<String, String> captures = Collections.emptyMap();
    private StringBuilder body = new StringBuilder();
    private boolean opened;
    /** Where the open region's body starts: the end of its opener. */
    private int openEnd;
    private boolean finished;

    /**
     * A parser continuing {@code prefix}, the chat prompt; its events go to {@code initial}.
     *
     * @param template the template
     * @param prefix   the prompt, or "" when the text to parse holds the whole message
     * @param tools    the request's tool schemas that type parsed call arguments
     * @param initial  receives the events the prompt produces
     */
    public ResponseParser(ResponseTemplate template, String prefix, ToolArguments tools, Listener initial) {
        this.template = Objects.requireNonNull(template, "template");
        this.tools = Objects.requireNonNull(tools, "tools");
        Objects.requireNonNull(prefix, "prefix");
        this.current = template.implicitField();
        for (Map.Entry<String, Object> d : template.defaults().entrySet()) {
            output.put(d.getKey(), mutableCopy(d.getValue()));
        }
        String truncated = template.truncatePastLastAnchor(prefix);
        if (!truncated.isEmpty()) {
            buffer.append(truncated);
            process(initial == null ? IGNORE : initial, false);
            prefixLength = truncated.length();
        }
    }

    /** The parser input after the prompt's truncation: the prompt's message part, then every piece fed. */
    public String inputText() {
        return buffer.toString();
    }

    /** The length of {@link #inputText()}. */
    public int inputLength() {
        return buffer.length();
    }

    /** The span {@code [from, to)} of {@link #inputText()}. */
    public String input(int from, int to) {
        return buffer.substring(from, to);
    }

    /** The length of the prompt's message part at the start of {@link #inputText()}. */
    public int prefixLength() {
        return prefixLength;
    }

    /** The region text goes to now: an open explicit region, else the implicit field; null for the dropping sink. */
    public String currentField() {
        return current;
    }

    /** Whether input is held back as a possible delimiter. */
    public boolean holding() {
        return pos < buffer.length();
    }

    /**
     * The rest of a closer the output stopped inside after ending its region (vLLM {@code _closer_rest}): when the open
     * region's text ends with a literal closer cut no earlier than the message start anchor the closer ends with, the
     * closer's missing rest, which feeding closes the region as written; otherwise "".
     */
    public String closerRest() {
        if (current == null || !opened) return "";
        List<String> closers = template.field(current).closeLiterals();
        List<String> anchors = template.messageStartAnchors();
        if (closers == null || anchors == null) return "";
        String text = buffer.substring(Math.min(openEnd, buffer.length()));
        for (String closer : closers) {
            for (String anchor : anchors) {
                if (anchor.isEmpty() || !closer.endsWith(anchor) || closer.equals(anchor)) continue;
                for (int cut = closer.length() - 1; cut > closer.length() - anchor.length() - 1; cut--) {
                    if (text.endsWith(closer.substring(0, cut))) return closer.substring(cut);
                }
            }
        }
        return "";
    }

    /** Feeds the next piece of output. */
    public void feed(String text, Listener listener) {
        if (finished) throw new IllegalStateException("ResponseParser already finalized");
        buffer.append(text);
        process(listener, false);
    }

    /**
     * Ends the output: held text is released, the open region closes, and the parsed message is returned — the
     * defaults and every non-empty field value.
     *
     * @throws IllegalArgumentException when a required field is missing
     */
    public Map<String, Object> finish(Listener listener) {
        if (finished) throw new IllegalStateException("ResponseParser already finalized");
        process(listener, true);
        List<String> missing = new ArrayList<>();
        for (ResponseTemplate.Field f : template.fields().values()) {
            if (!f.optional() && !output.containsKey(f.name()) && !malformed.contains(f.name())) missing.add(f.name());
        }
        if (!missing.isEmpty()) {
            throw new IllegalArgumentException("Required response_template fields missing from parsed output: "
                    + missing);
        }
        Map<String, Object> message = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : output.entrySet()) {
            if (template.defaults().containsKey(e.getKey()) || !isEmpty(e.getValue())) {
                message.put(e.getKey(), e.getValue());
            }
        }
        finished = true;
        return message;
    }

    private static boolean isEmpty(Object v) {
        return v == null || (v instanceof String && ((String) v).isEmpty()) || (v instanceof List && ((List<?>) v).isEmpty())
                || (v instanceof Map && ((Map<?, ?>) v).isEmpty());
    }

    private void process(Listener events, boolean eos) {
        while (true) {
            List<Watch> watch = watchlist();
            int[] hold = {buffer.length()};
            Hit best = scan(watch, eos, hold);
            if (best != null) {
                if (best.match.start > pos) accumulate(events, buffer.substring(pos, best.match.start));
                pos = best.match.end;
                if (best.open) {
                    closeCurrent(events, best.match.start, best.match.start);
                    openExplicit(events, best.field, best.match);
                } else {
                    boolean hadContent = opened;
                    closeCurrent(events, best.match.start, best.match.end);
                    if (!hadContent && best.match.start == best.match.end) break;
                }
                continue;
            }
            if (eos) {
                if (pos < buffer.length()) {
                    accumulate(events, buffer.substring(pos));
                    pos = buffer.length();
                }
                closeCurrent(events, pos, pos);
                break;
            }
            if (hold[0] > pos) {
                accumulate(events, buffer.substring(pos, hold[0]));
                pos = hold[0];
            }
            break;
        }
    }

    /** A watched delimiter: an opener or a closer of a field. */
    private static final class Watch {
        final boolean open;
        final ResponseTemplate.Field field;

        Watch(boolean open, ResponseTemplate.Field field) {
            this.open = open;
            this.field = field;
        }

        Delimiter delimiter() {
            return open ? field.open() : field.close();
        }
    }

    /** A committable delimiter match. */
    private static final class Hit {
        final boolean open;
        final ResponseTemplate.Field field;
        final TemplateRegex.Match match;

        Hit(Watch w, TemplateRegex.Match match) {
            this.open = w.open;
            this.field = w.field;
            this.match = match;
        }
    }

    private List<Watch> watchlist() {
        List<Watch> watch = new ArrayList<>();
        if (current != null && !current.equals(template.implicitField())) {
            ResponseTemplate.Field f = template.field(current);
            if (f.closes()) watch.add(new Watch(false, f));
            return watch;
        }
        for (ResponseTemplate.Field f : template.fields().values()) {
            if (f.opens()) watch.add(new Watch(true, f));
        }
        if (template.implicitField() != null) {
            ResponseTemplate.Field f = template.field(template.implicitField());
            if (f.closes()) watch.add(new Watch(false, f));
        }
        return watch;
    }

    /**
     * The earliest delimiter that can commit now (longest on ties, openers before closers, then by field name), or
     * null; {@code hold[0]} becomes the leftmost start of a still-pending delimiter, the text from which waits.
     */
    private Hit scan(List<Watch> watch, boolean eos, int[] hold) {
        Hit best = null;
        for (Watch w : watch) {
            TemplateRegex.Match m = w.delimiter().regex().search(buffer, pos, !eos);
            if (m == null) continue;
            if (!eos && (m.partial || canGrow(w.delimiter(), m))) {
                hold[0] = Math.min(hold[0], m.start);
                continue;
            }
            if (best == null || before(w, m, best)) best = new Hit(w, m);
        }
        return best != null && best.match.start >= hold[0] ? null : best;
    }

    private static boolean before(Watch w, TemplateRegex.Match m, Hit best) {
        if (m.start != best.match.start) return m.start < best.match.start;
        int len = m.end - m.start;
        int bestLen = best.match.end - best.match.start;
        if (len != bestLen) return len > bestLen;
        if (w.open != best.open) return w.open;
        return w.field.name().compareTo(best.field.name()) < 0;
    }

    /** Whether a complete match ending at the input's end could still change with more input. */
    private boolean canGrow(Delimiter d, TemplateRegex.Match m) {
        if (m.end != buffer.length()) return false;
        if (m.start == m.end) return true;
        return d.literals() == null || d.canExtend();
    }

    private void accumulate(Listener events, String text) {
        if (text.isEmpty() || current == null) return;
        ResponseTemplate.Field f = template.field(current);
        if (!opened) {
            events.opened(current, pos, pos, Collections.<String, String>emptyMap());
            opened = true;
            openEnd = pos;
        }
        body.append(text);
        String fresh = text.substring(Math.min(text.length(), Math.max(0, prefixLength - pos)));
        if (!fresh.isEmpty()) events.chunk(current, fresh, !f.content().streamable());
    }

    private void openExplicit(Listener events, ResponseTemplate.Field f, TemplateRegex.Match m) {
        current = f.name();
        captures = m.captures;
        body = new StringBuilder();
        opened = true;
        openEnd = m.end;
        events.opened(f.name(), m.start, m.end, captures);
    }

    private void closeCurrent(Listener events, int start, int end) {
        if (current == null || !opened) {
            resetToImplicit();
            return;
        }
        ResponseTemplate.Field f = template.field(current);
        Object value;
        try {
            value = tools.coerce(value(f));
            if (f.join() != null && !(value instanceof String)) {
                throw new IllegalArgumentException("Field '" + f.name() + "': 'join' requires each match to parse to"
                        + " a string, got " + Transform.typeName(value) + ".");
            }
        } catch (IllegalArgumentException e) {
            malformed.add(current);
            events.malformed(current, start, end, e.getMessage());
            resetToImplicit();
            return;
        }
        store(f, value);
        events.closed(current, start, end, value);
        resetToImplicit();
    }

    @SuppressWarnings("unchecked")
    private void store(ResponseTemplate.Field f, Object value) {
        if (f.join() != null) {
            Object previous = output.get(f.name());
            output.put(f.name(), previous == null ? value : previous + f.join() + value);
        } else if (f.repeats()) {
            Object list = output.get(f.name());
            if (!(list instanceof List)) {
                list = new ArrayList<>();
                output.put(f.name(), list);
            }
            ((List<Object>) list).add(value);
        } else {
            output.put(f.name(), value);
        }
    }

    /** The body through the content parser, then the transform (once, or per element with transform_each). */
    private Object value(ResponseTemplate.Field f) {
        Object parsed = f.content().parse(body.toString());
        if (f.transform() == null) return parsed;
        if (!f.transformEach()) {
            Map<String, Object> scope = new LinkedHashMap<>(captures);
            scope.put("content", parsed);
            return f.transform().apply(scope);
        }
        if (!(parsed instanceof List)) {
            throw new IllegalArgumentException("Field '" + f.name() + "': transform_each requires the parsed content"
                    + " to be a list, got " + Transform.typeName(parsed) + ".");
        }
        List<Object> out = new ArrayList<>();
        for (Object item : (List<?>) parsed) {
            if (!(item instanceof Map)) {
                throw new IllegalArgumentException("Field '" + f.name() + "': transform_each requires each list element"
                        + " to be a dict, got " + Transform.typeName(item) + ".");
            }
            Map<String, Object> scope = new LinkedHashMap<>(captures);
            for (Map.Entry<?, ?> e : ((Map<?, ?>) item).entrySet()) scope.put(String.valueOf(e.getKey()), e.getValue());
            out.add(f.transform().apply(scope));
        }
        return out;
    }

    private void resetToImplicit() {
        current = template.implicitField();
        captures = Collections.emptyMap();
        body = new StringBuilder();
        opened = false;
    }

    /** A mutable deep copy of a JSON value. */
    static Object mutableCopy(Object v) {
        if (v instanceof Map) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) out.put(String.valueOf(e.getKey()), mutableCopy(e.getValue()));
            return out;
        }
        if (v instanceof List) {
            List<Object> out = new ArrayList<>();
            for (Object o : (List<?>) v) out.add(mutableCopy(o));
            return out;
        }
        return v;
    }
}
