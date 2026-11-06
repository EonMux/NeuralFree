package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.tools.BlockToolParser;
import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolSchemas;
import com.eonmux.neuralfree.utils.json.Json;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads a template's tool calls with the PEG the analysis built ({@link AutoParserPeg}), so an unknown syntax gets
 * the same parsing its derived grammar constrains: the block reader finds the call openers, the PEG matches one
 * call, and the captured spans become the function name and its arguments.
 *
 * <p>The formats map differently: a JSON-native call's fields are read by the analysed paths, a
 * {@code TAG_WITH_JSON} call's captured span is its argument object, and a {@code TAG_WITH_TAGGED} call's parameter
 * tags rebuild the argument object with the request's declared parameter types. While the text grows, an unfinished
 * call is reported with the arguments read so far, so a streamed argument text is always a prefix of the final one.
 */
final class AutoParserToolParser extends BlockToolParser {

    /** The OpenAI function wrapper a JSON-native template's model may emit before the analysed fields. */
    static final String OPENAI_WRAPPER = "{\"type\": \"function\",";

    private final AutoParserAnalysis.Tools spec;
    private final AutoParserAnalysis.Content content;
    private final ToolSchemas schemas;
    private final List<String> names;
    private final Peg callPeg;
    private final Peg sectionCallPeg;

    AutoParserToolParser(AutoParserAnalysis analysis, List<ChatTool> tools) {
        super(openers(analysis.tools()));
        this.spec = analysis.tools();
        this.content = analysis.content();
        this.schemas = ToolSchemas.of(tools);
        List<String> declared = new ArrayList<>();
        for (ChatTool tool : tools) {
            declared.add(tool.name());
        }
        this.names = Collections.unmodifiableList(declared);
        this.callPeg = AutoParserPeg.call(spec, tools, false);
        this.sectionCallPeg = spec.mode() == AutoParserAnalysis.ToolFormat.JSON_NATIVE
                ? callPeg : AutoParserPeg.call(spec, tools, true);
    }

    private static List<String> openers(AutoParserAnalysis.Tools spec) {
        List<String> out = new ArrayList<>();
        if (!spec.sectionStart().isEmpty()) {
            out.add(spec.sectionStart());
        }
        if (!spec.perCallStart().isEmpty() && !spec.perCallStart().equals(spec.sectionStart())) {
            out.add(spec.perCallStart());
        }
        return out;
    }

    /** The openers, plus the OpenAI function wrapper the model may emit before a bare JSON call. */
    @Override
    public List<String> triggers() {
        if (!spec.openaiWrapperTrigger()) {
            return super.triggers();
        }
        List<String> out = new ArrayList<>(super.triggers());
        out.add(OPENAI_WRAPPER);
        return Collections.unmodifiableList(out);
    }

    @Override
    protected int bareStart(String text, boolean partial) {
        if (!spec.sectionStart().isEmpty() || !spec.perCallStart().isEmpty()) {
            return -1;
        }
        if (spec.mode() == AutoParserAnalysis.ToolFormat.JSON_NATIVE && spec.toolsArrayWrapped()) {
            int array = text.indexOf('[');
            int object = text.indexOf('{');
            if (array < 0) {
                return object;
            }
            return object < 0 || array < object ? array : object;
        }
        return text.indexOf('{');
    }

    @Override
    protected Block readBlock(String text, int at, String opener, boolean partial) {
        if (!spec.sectionStart().isEmpty() && opener.equals(spec.sectionStart())
                && !opener.equals(spec.perCallStart())) {
            return readSection(text, at, partial);
        }
        // the OpenAI wrapper is the call's own object opener, so its `{` starts the call the PEG reads
        int start = opener.isEmpty() || opener.equals(OPENAI_WRAPPER) ? at : at + opener.length();
        if (opener.isEmpty() && spec.mode() == AutoParserAnalysis.ToolFormat.JSON_NATIVE
                && spec.toolsArrayWrapped() && at < text.length() && text.charAt(at) == '[') {
            return readArray(text, at, partial);
        }
        return readCall(text, start, partial);
    }

    /** A bare JSON array of calls. */
    private Block readArray(String text, int at, boolean partial) {
        List<ToolCall> calls = new ArrayList<>();
        int i = at + 1;
        while (true) {
            i = skipWhitespace(text, i);
            if (i < text.length() && text.charAt(i) == ']') {
                return Block.complete(calls, i + 1);
            }
            if (i >= text.length()) {
                return partial ? Block.incomplete(calls, at) : Block.complete(calls, at + 1);
            }
            Peg.Result result = callPeg.match(text, i, partial);
            if (result.ok()) {
                ToolCall call = map(text, result, true);
                if (call != null) {
                    calls.add(call);
                }
                i = skipWhitespace(text, result.end());
                if (i < text.length() && text.charAt(i) == ',') {
                    i++;
                }
                continue;
            }
            if (result.needMore()) {
                ToolCall call = map(text, result, false);
                if (call != null) {
                    calls.add(call);
                }
                return Block.incomplete(calls, at);
            }
            return calls.isEmpty() ? Block.malformed(at + 1) : Block.complete(calls, i);
        }
    }

    /** One call through {@link #callPeg}, starting after its opener. */
    private Block readCall(String text, int start, boolean partial) {
        Peg.Result result = callPeg.match(text, start, partial);
        if (result.ok()) {
            ToolCall call = map(text, result, true);
            return Block.complete(call == null ? Collections.<ToolCall>emptyList()
                    : Collections.singletonList(call), result.end());
        }
        if (result.needMore()) {
            ToolCall call = map(text, result, false);
            return Block.incomplete(call == null ? Collections.<ToolCall>emptyList()
                    : Collections.singletonList(call), start);
        }
        return Block.malformed(start + 1);
    }

    /** The section opener, its calls and the section closer. */
    private Block readSection(String text, int at, boolean partial) {
        List<ToolCall> calls = new ArrayList<>();
        int i = at + spec.sectionStart().length();
        boolean array = spec.mode() == AutoParserAnalysis.ToolFormat.JSON_NATIVE && spec.toolsArrayWrapped();
        boolean arrayOpened = false;
        boolean arrayClosed = false;
        while (true) {
            i = skipWhitespace(text, i);
            if (!spec.sectionEnd().isEmpty() && text.startsWith(spec.sectionEnd(), i)) {
                return Block.complete(calls, i + spec.sectionEnd().length());
            }
            if (!spec.sectionEnd().isEmpty() && partial && i < text.length()
                    && spec.sectionEnd().startsWith(text.substring(i))) {
                return Block.incomplete(calls, i);
            }
            if (array) {
                if (!arrayOpened && i < text.length() && text.charAt(i) == '[') {
                    arrayOpened = true;
                    i++;
                    continue;
                }
                if (arrayOpened && !arrayClosed && i < text.length() && text.charAt(i) == ']') {
                    arrayClosed = true;
                    i++;
                    continue;
                }
                if (i >= text.length() && !arrayClosed) {
                    return partial ? Block.incomplete(calls, i) : Block.complete(calls, i);
                }
            }
            if (i >= text.length()) {
                return partial ? Block.incomplete(calls, i) : Block.complete(calls, i);
            }
            Peg.Result result = sectionCallPeg.match(text, i, partial);
            if (result.ok()) {
                ToolCall call = map(text, result, true);
                if (call != null) {
                    calls.add(call);
                }
                i = skipWhitespace(text, result.end());
                if (i < text.length() && text.charAt(i) == ',') {
                    i++;
                }
                continue;
            }
            if (result.needMore()) {
                ToolCall call = map(text, result, false);
                if (call != null) {
                    calls.add(call);
                }
                return Block.incomplete(calls, i);
            }
            return calls.isEmpty() ? Block.malformed(i + 1) : Block.complete(calls, i);
        }
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        ToolParse parsed = super.parse(text, partial);
        if (content.mode() == AutoParserAnalysis.ContentMode.PLAIN) {
            return parsed;
        }
        String stripped = unwrap(parsed.content(), partial);
        return stripped.equals(parsed.content()) ? parsed : new ToolParse(stripped.trim(), parsed.calls());
    }

    /** The content without the template's wrapper markers; a partial marker is held back while it may still grow. */
    private String unwrap(String text, boolean partial) {
        String start = content.start().trim();
        String end = content.end().trim();
        String out = text;
        if (!start.isEmpty()) {
            int at = leadingWhitespace(out);
            if (out.startsWith(start, at)) {
                out = out.substring(0, at) + out.substring(at + start.length());
            } else if (partial && start.startsWith(out.trim())) {
                return "";
            }
        }
        if (!end.isEmpty()) {
            String trimmed = rtrim(out);
            if (trimmed.endsWith(end)) {
                out = trimmed.substring(0, trimmed.length() - end.length());
            } else if (partial) {
                int hold = trailingPrefix(trimmed, end);
                if (hold > 0) {
                    out = trimmed.substring(0, trimmed.length() - hold);
                }
            }
        }
        return out;
    }

    private static int leadingWhitespace(String text) {
        int at = 0;
        while (at < text.length() && Character.isWhitespace(text.charAt(at))) {
            at++;
        }
        return at;
    }

    private static String rtrim(String text) {
        int end = text.length();
        while (end > 0 && Character.isWhitespace(text.charAt(end - 1))) {
            end--;
        }
        return text.substring(0, end);
    }

    /** How many trailing characters of {@code text} are a proper prefix of {@code marker}. */
    private static int trailingPrefix(String text, String marker) {
        for (int k = Math.min(marker.length() - 1, text.length()); k > 0; k--) {
            if (text.regionMatches(text.length() - k, marker, 0, k)) {
                return k;
            }
        }
        return 0;
    }

    /** The call a PEG match captured, or null when it lacks a function name. */
    private ToolCall map(CharSequence text, Peg.Result result, boolean complete) {
        switch (spec.mode()) {
            case JSON_NATIVE:
                return mapJson(result.text(text, "json"), complete);
            case TAG_WITH_JSON:
                return mapTaggedJson(result.text(text, "name"), result.text(text, "args"),
                        result.text(text, "id"), complete);
            case TAG_WITH_TAGGED:
                return mapTaggedArgs(result.text(text, "name"), result.text(text, "id"), result, text, complete);
            default:
                return null;
        }
    }

    private ToolCall mapTaggedJson(String name, String args, String id, boolean complete) {
        if (name == null || !names.contains(name)) {
            return null;
        }
        return new ToolCall(name, args == null ? "" : args, emptyToNull(id), complete);
    }

    private ToolCall mapTaggedArgs(String name, String id, Peg.Result result, CharSequence text,
                                   boolean complete) {
        if (name == null || !names.contains(name)) {
            return null;
        }
        List<Peg.Capture> keys = result.captures("argname");
        List<Peg.Capture> values = result.captures("argvalue");
        String arguments;
        if (complete) {
            Map<String, Object> args = new LinkedHashMap<>();
            for (int i = 0; i < keys.size() && i < values.size(); i++) {
                String key = keys.get(i).text(text);
                args.put(key, schemas.value(name, key, values.get(i).text(text)));
            }
            arguments = ToolSchemas.arguments(args);
        } else {
            arguments = partialTaggedArguments(name, keys, values, text);
        }
        return new ToolCall(name, arguments, emptyToNull(id), complete);
    }

    private static String emptyToNull(String id) {
        return id == null || id.isEmpty() ? null : id;
    }

    /**
     * The argument text of a growing call: a prefix of the final text, so a streamed delta is the text's tail. A
     * value whose closing tag has not arrived is written as a JSON prefix of its final form (an open string).
     */
    private String partialTaggedArguments(String name, List<Peg.Capture> keys, List<Peg.Capture> values,
                                          CharSequence text) {
        StringBuilder out = new StringBuilder("{");
        int pairs = Math.min(keys.size(), values.size());
        int emitted = 0;
        for (int i = 0; i < pairs; i++) {
            Peg.Capture keyCapture = keys.get(i);
            String key = keyCapture.text(text);
            if (key.isEmpty() || keyCapture.end() >= text.length()) {
                break;
            }
            if (emitted++ > 0) {
                out.append(',');
            }
            out.append(quote(key)).append(':');
            Peg.Capture value = values.get(i);
            if (isValueComplete(value, text)) {
                out.append(dump(schemas.value(name, key, value.text(text))));
            } else if (schemas.textValued(name, key)) {
                out.append(openString(withoutPartialSuffix(value.text(text))));
            } else {
                out.append(value.text(text).trim());
            }
        }
        if (keys.size() > pairs) {
            Peg.Capture pending = keys.get(pairs);
            String key = pending.text(text);
            if (!key.isEmpty() && pending.end() < text.length()) {
                if (emitted > 0) {
                    out.append(',');
                }
                out.append(quote(key)).append(':');
            }
        }
        return out.toString();
    }

    private static String quote(String text) {
        return Json.toString(text, Json.Style.API);
    }

    /** A JSON string without its closing quote, a prefix of the string's final text. */
    private static String openString(String text) {
        String quoted = Json.toString(text, Json.Style.API);
        return quoted.substring(0, quoted.length() - 1);
    }

    /** A streamed value with the part of a closing tag that has not fully arrived removed. */
    private String withoutPartialSuffix(String value) {
        String suffix = spec.argValueSuffix();
        if (suffix.isEmpty()) {
            return value;
        }
        for (int k = Math.min(suffix.length() - 1, value.length()); k > 0; k--) {
            if (value.endsWith(suffix.substring(0, k))) {
                return value.substring(0, value.length() - k);
            }
        }
        return value;
    }

    /** Whether a value's closing tag followed it. */
    private boolean isValueComplete(Peg.Capture value, CharSequence text) {
        return value.end() < text.length();
    }

    private ToolCall mapJson(String raw, boolean complete) {
        if (raw == null || raw.isEmpty()) {
            return null;
        }
        Object parsed;
        try {
            parsed = Json.parse(raw);
        } catch (RuntimeException e) {
            return complete ? null : partialJson(raw);
        }
        if (!(parsed instanceof Map)) {
            return null;
        }
        String name;
        String args;
        if (spec.funNameIsKey()) {
            Map.Entry<?, ?> only = ((Map<?, ?>) parsed).entrySet().iterator().next();
            name = String.valueOf(only.getKey());
            args = dump(only.getValue());
        } else {
            name = stringAt(parsed, spec.effectiveNameField());
            Object value = at(parsed, spec.effectiveArgsField());
            args = value instanceof String ? (String) value : value == null ? "" : dump(value);
        }
        if (name == null || name.isEmpty()) {
            return null;
        }
        return new ToolCall(name, args, callId(parsed), complete);
    }

    /** A call whose JSON is still growing: the name once its field is readable and the arguments' text so far. */
    private ToolCall partialJson(String raw) {
        String name;
        String args;
        if (spec.funNameIsKey()) {
            int colon = raw.indexOf(':');
            if (colon < 0) {
                return null;
            }
            name = lastStringBefore(raw, colon);
            args = partialContainer(raw, colon);
        } else {
            name = stringField(raw, lastSegment(spec.effectiveNameField()));
            args = partialContainer(raw, indexOfField(raw, lastSegment(spec.effectiveArgsField())));
        }
        if (name == null || name.isEmpty()) {
            return null;
        }
        return new ToolCall(name, args, null, false);
    }

    private static String lastSegment(String path) {
        int dot = path.lastIndexOf('.');
        return dot < 0 ? path : path.substring(dot + 1);
    }

    /** The index just past {@code "field":} in {@code raw}, or -1. */
    private static int indexOfField(String raw, String field) {
        String needle = "\"" + field + "\"";
        int at = raw.indexOf(needle);
        if (at < 0) {
            return -1;
        }
        int colon = raw.indexOf(':', at + needle.length());
        return colon < 0 ? -1 : colon;
    }

    /** The arguments text from {@code afterField} to the end, when it starts a JSON container. */
    private static String partialContainer(String raw, int afterField) {
        if (afterField < 0) {
            return "";
        }
        int at = afterField + 1;
        while (at < raw.length() && Character.isWhitespace(raw.charAt(at))) {
            at++;
        }
        if (at < raw.length() && (raw.charAt(at) == '{' || raw.charAt(at) == '[')) {
            return raw.substring(at);
        }
        return "";
    }

    /** The last JSON string value before {@code limit}, unescaped as text, or null. */
    private static String lastStringBefore(String raw, int limit) {
        for (int i = limit - 1; i >= 0; i--) {
            if (raw.charAt(i) == '"') {
                int start = i - 1;
                while (start >= 0) {
                    if (raw.charAt(start) == '"' && (start == 0 || raw.charAt(start - 1) != '\\')) {
                        return raw.substring(start + 1, i);
                    }
                    start--;
                }
            }
        }
        return null;
    }

    /** A string field's value found textually in a still-growing JSON text, or null. */
    private static String stringField(String raw, String field) {
        int colon = indexOfField(raw, field);
        if (colon < 0) {
            return null;
        }
        int at = colon + 1;
        while (at < raw.length() && Character.isWhitespace(raw.charAt(at))) {
            at++;
        }
        if (at >= raw.length() || raw.charAt(at) != '"') {
            return null;
        }
        StringBuilder out = new StringBuilder();
        boolean escaped = false;
        for (int i = at + 1; i < raw.length(); i++) {
            char c = raw.charAt(i);
            if (escaped) {
                out.append(c);
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                return out.toString();
            } else {
                out.append(c);
            }
        }
        return null;
    }

    private String callId(Object parsed) {
        String id = stringAt(parsed, spec.idField());
        if (id == null || id.isEmpty()) {
            id = stringAt(parsed, spec.genIdField());
        }
        return id == null || id.isEmpty() ? null : id;
    }

    private static Object at(Object root, String path) {
        if (path == null || path.isEmpty()) {
            return root;
        }
        Object value = root;
        for (String part : path.split("\\.")) {
            if (!(value instanceof Map)) {
                return null;
            }
            value = ((Map<?, ?>) value).get(part);
        }
        return value;
    }

    private static String stringAt(Object root, String path) {
        Object value = at(root, path);
        return value instanceof String ? (String) value : null;
    }

    private static String dump(Object value) {
        return Json.toString(value, Json.Style.API);
    }

    private static int skipWhitespace(CharSequence text, int i) {
        int at = i;
        while (at < text.length() && Character.isWhitespace(text.charAt(at))) {
            at++;
        }
        return at;
    }
}
