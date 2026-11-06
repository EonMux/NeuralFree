package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.utils.json.Json;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Differential template analysis: llama.cpp's {@code autoparser::analyze_template} and the analyzer phases of
 * {@code common/chat-diff-analyzer.cpp}. Each structure is found by rendering two probe conversations that differ in
 * one input and reading the diff between them — a reasoning block that appears only when
 * {@code reasoning_content} is set, a tool section that appears only when {@code tool_calls} is set, a function
 * name that changes with the call's name, an argument name or value that changes inside its tags.
 *
 * <p>The probes are llama.cpp's own ({@code USER_MSG}, {@code FUN_FIRST}, {@code ARG_FIRST}, ...): distinct
 * sentinel strings whose occurrence in a render marks where an input landed. The derived markers are the literal
 * text around them. The analysis is best-effort — a template that fails a probe leaves that structure empty rather
 * than failing the whole analysis. One instance analyzes one template and is not shared.
 */
public final class AutoParserAnalyzer {

    private static final Logger LOGGER = LogManager.getLogger(AutoParserAnalyzer.class);

    static final String FUN_FIRST = "FFF_FIRST_FUN_F";
    static final String FUN_SECOND = "SSS_SECOND_FUN_S";
    static final String ARG_FIRST = "AA_ARG_FST_AA";
    static final String ARG_SECOND = "BB_ARG_SND_BB";
    static final String USER_MSG = "U_USER_MSG Hello END_U";
    static final String USER_MSG_TWO = "V_USER_MSG Hello END_V";
    static final String ASSISTANT_MSG = "A_ASST_MSG I can help END_A";
    static final String THINKING_CONTENT = "REASON_PART I am thinking END_R";
    static final String CALL_ID_001 = "call00001";
    static final String CALL_ID_002 = "call00002";
    static final String CALL_ID_999 = "call99999";
    private static final String TEST_FUNC = "test_func";

    private final TemplateProbe probe;
    // The derived structure. Package-private so AutoParserWorkarounds can patch it, as llama.cpp's workarounds
    // patch the analysis struct after the differential probes.
    AutoParserAnalysis.ReasoningMode reasoningMode = AutoParserAnalysis.ReasoningMode.NONE;
    String reasoningStart = "";
    String reasoningEnd = "";
    AutoParserAnalysis.ContentMode contentMode = AutoParserAnalysis.ContentMode.PLAIN;
    String contentStart = "";
    String contentEnd = "";
    AutoParserAnalysis.ToolFormat toolMode = AutoParserAnalysis.ToolFormat.NONE;
    String sectionStart = "";
    String sectionEnd = "";
    String perCallStart = "";
    String perCallEnd = "";
    boolean funNameIsKey;
    boolean toolsArrayWrapped;
    String functionField = "function";
    String nameField = "name";
    String argsField = "arguments";
    String idField = "";
    String genIdField = "";
    final List<String> parameterOrder = new ArrayList<>();
    String functionNamePrefix = "";
    String functionNameSuffix = "";
    String functionArgsSeparator = "";
    String functionClose = "";
    String argsStart = "";
    String argsEnd = "";
    String argNamePrefix = "";
    String argNameSuffix = "";
    String argValuePrefix = "";
    String argValueSuffix = "";
    String argSeparator = "";
    AutoParserAnalysis.CallIdPosition callIdPosition = AutoParserAnalysis.CallIdPosition.NONE;
    String callIdPrefix = "";
    String callIdSuffix = "";
    String userStart = "";
    String assistantStart = "";
    boolean openaiWrapperTrigger;
    boolean tolerateIntertagWhitespace;
    /** A workaround's replacement for the derived token list; null keeps the derived one. */
    List<String> preservedOverride;
    final List<String> additionalStops = new ArrayList<>();

    private AutoParserAnalyzer(TemplateProbe probe) {
        this.probe = probe;
    }

    /**
     * Analyzes a chat template source.
     *
     * @throws IllegalArgumentException when the template does not parse
     */
    public static AutoParserAnalysis analyze(String templateSource) {
        return analyze(TemplateProbe.of(templateSource));
    }

    /** Analyzes a compiled template. */
    public static AutoParserAnalysis analyze(TemplateProbe probe) {
        return new AutoParserAnalyzer(probe).run();
    }

    private AutoParserAnalysis run() {
        analyzeReasoning(probe.caps().supportsToolCalls());
        analyzeContent();
        if (probe.caps().supportsToolCalls()) {
            analyzeTools();
        }
        assistantStart = detectAssistantStart();
        userStart = detectUserStart();
        AutoParserWorkarounds.apply(probe.source(), this);
        AutoParserAnalysis.Reasoning reasoning = new AutoParserAnalysis.Reasoning(reasoningMode, reasoningStart,
                reasoningEnd);
        AutoParserAnalysis.Content content = new AutoParserAnalysis.Content(contentMode, contentStart,
                contentEnd);
        AutoParserAnalysis.Tools tools = new AutoParserAnalysis.Tools(toolMode, sectionStart, sectionEnd,
                perCallStart, perCallEnd, funNameIsKey, toolsArrayWrapped, functionField,
                nameField, argsField, idField, genIdField, parameterOrder, functionNamePrefix, functionNameSuffix,
                functionArgsSeparator, functionClose, argsStart, argsEnd, argNamePrefix, argNameSuffix,
                argValuePrefix, argValueSuffix, argSeparator, callIdPosition, callIdPrefix, callIdSuffix,
                openaiWrapperTrigger, tolerateIntertagWhitespace);
        List<String> preserved = preservedOverride != null ? preservedOverride
                : preservedTokens(reasoning, content, tools);
        AutoParserAnalysis analysis = new AutoParserAnalysis(userStart, assistantStart, reasoning, content, tools,
                preserved, additionalStops);
        LOGGER.debug("Chat template analysis: {}", analysis);
        return analysis;
    }

    // ================================================================== reasoning

    private void analyzeReasoning(boolean supportsTools) {
        compareReasoningPresence();
        compareThinkingEnabled();
        if (supportsTools) {
            compareReasoningScope();
        }
    }

    /** The reasoning content appears only when the assistant message carries {@code reasoning_content}. */
    private void compareReasoningPresence() {
        List<Map<String, Object>> a = Arrays.asList(userMessage(),
                assistantMessage(ASSISTANT_MSG, null, null));
        List<Map<String, Object>> b = Arrays.asList(userMessage(),
                assistantMessage(ASSISTANT_MSG, THINKING_CONTENT, null));
        TemplateProbe.VariantPair pair = probe.compare(a, b, null, false, true);
        if (pair == null) {
            return;
        }
        if (!pair.diff().right().isEmpty() && pair.outputB().contains(THINKING_CONTENT)) {
            Tagged tagged = wrappedAround(pair.outputB(), THINKING_CONTENT);
            if (tagged != null && tagged.pre != null && tagged.post != null) {
                reasoningMode = AutoParserAnalysis.ReasoningMode.TAG_BASED;
                reasoningStart = tagged.pre;
                reasoningEnd = tagged.post;
            } else if (tagged != null && tagged.post != null) {
                reasoningMode = AutoParserAnalysis.ReasoningMode.TAG_BASED;
                reasoningEnd = tagged.post;
            }
        }
    }

    /** The generation prompt differs with {@code enable_thinking}: read the markers it adds. */
    private void compareThinkingEnabled() {
        List<Map<String, Object>> messages = Arrays.asList(userMessage());
        String a = probe.render(messages, null, true, false);
        String b = probe.render(messages, null, true, true);
        if (a == null || b == null) {
            return;
        }
        MarkerDiff.Diff diff = MarkerDiff.split(a, b);
        String leftTrimmed = diff.left().trim();
        String rightTrimmed = diff.right().trim();
        if (leftTrimmed.isEmpty() && !diff.right().isEmpty()) {
            if (!rightTrimmed.isEmpty() && b.endsWith(rightTrimmed) && reasoningStart.isEmpty()) {
                reasoningStart = diff.right();
                reasoningMode = AutoParserAnalysis.ReasoningMode.TAG_BASED;
            }
        } else if (rightTrimmed.isEmpty() && !diff.left().isEmpty()) {
            if (!leftTrimmed.isEmpty() && a.endsWith(leftTrimmed) && reasoningEnd.isEmpty()) {
                List<MarkerDiff.Segment> segments = MarkerDiff.pruneWhitespace(MarkerDiff.segmentize(a));
                if (segments.size() >= 2) {
                    MarkerDiff.Segment last = segments.get(segments.size() - 1);
                    MarkerDiff.Segment before = segments.get(segments.size() - 2);
                    if (last.value().equals(leftTrimmed) && before.type() == MarkerDiff.Type.MARKER) {
                        reasoningStart = before.value();
                    }
                }
                reasoningEnd = diff.left();
                reasoningMode = AutoParserAnalysis.ReasoningMode.TAG_BASED;
            }
        } else if (!leftTrimmed.isEmpty() && !rightTrimmed.isEmpty()) {
            tagByTailAnchor(a, b);
        }
        if (reasoningMode == AutoParserAnalysis.ReasoningMode.NONE && reasoningStart.isEmpty()
                && !reasoningEnd.isEmpty()) {
            reasoningMode = AutoParserAnalysis.ReasoningMode.TAG_BASED;
        }
    }

    /**
     * One output's generation-prompt tail appears in the other with reasoning markers appended (SmolLM3 changes the
     * system message when thinking flips, so the whole-output diff is noisy).
     */
    private void tagByTailAnchor(String a, String b) {
        String[] base = {b, a};
        String[] extended = {a, b};
        for (int dir = 0; dir < 2; dir++) {
            String anchor = tail(base[dir], 64);
            int at = extended[dir].lastIndexOf(anchor);
            if (at < 0 || at + anchor.length() >= extended[dir].length()) {
                continue;
            }
            String extra = extended[dir].substring(at + anchor.length()).trim();
            if (extra.isEmpty()) {
                continue;
            }
            List<MarkerDiff.Segment> segments = MarkerDiff.pruneWhitespace(MarkerDiff.segmentize(extra));
            if (segments.size() == 2 && segments.get(0).type() == MarkerDiff.Type.MARKER
                    && segments.get(1).type() == MarkerDiff.Type.MARKER) {
                if (reasoningStart.isEmpty()) {
                    reasoningStart = segments.get(0).value();
                }
                if (reasoningEnd.isEmpty()) {
                    reasoningEnd = segments.get(1).value();
                }
                reasoningMode = AutoParserAnalysis.ReasoningMode.TAG_BASED;
                return;
            }
        }
    }

    private static String tail(String text, int max) {
        return text.substring(Math.max(0, text.length() - max));
    }

    /** Reasoning appears with a tool call but not with plain content. */
    private void compareReasoningScope() {
        Map<String, Object> withContent = assistantMessage(ASSISTANT_MSG, THINKING_CONTENT, null);
        Map<String, Object> withTools = assistantMessage(null, THINKING_CONTENT,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "VVVV", ARG_SECOND, "XXXX"), CALL_ID_001)));
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), withContent),
                Arrays.asList(userMessage(), withTools), tools(), false, true);
        if (pair == null) {
            return;
        }
        boolean inA = pair.outputA().contains(THINKING_CONTENT);
        boolean inB = pair.outputB().contains(THINKING_CONTENT);
        if (inA || !inB) {
            return;
        }
        reasoningMode = AutoParserAnalysis.ReasoningMode.TOOLS_ONLY;
        Tagged tagged = wrappedAround(pair.outputB(), THINKING_CONTENT);
        if (tagged != null && tagged.pre != null && tagged.post != null) {
            reasoningStart = tagged.pre;
            reasoningEnd = tagged.post;
        } else if (tagged != null && tagged.post != null) {
            reasoningEnd = tagged.post;
        } else {
            reasoningMode = AutoParserAnalysis.ReasoningMode.NONE;
        }
    }

    // ================================================================== content

    private void analyzeContent() {
        List<Map<String, Object>> contentOnly = Arrays.asList(userMessage(),
                assistantMessage(ASSISTANT_MSG, null, null));
        List<Map<String, Object>> withTools = Arrays.asList(userMessage(),
                assistantMessage("", null, Arrays.asList(toolCall(TEST_FUNC, args("arg1", "value1"), CALL_ID_001))));
        List<Map<String, Object>> withReasoning = Arrays.asList(userMessage(),
                assistantMessage("", THINKING_CONTENT, null));
        TemplateProbe.VariantPair toolsPair = probe.compare(contentOnly, withTools, tools(), false, true);
        TemplateProbe.VariantPair reasoningPair = probe.compare(contentOnly, withReasoning, tools(), false, true);
        if (toolsPair == null || reasoningPair == null) {
            return;
        }
        boolean plain = toolsPair.diff().left().trim().equals(ASSISTANT_MSG);
        if (plain) {
            contentMode = AutoParserAnalysis.ContentMode.PLAIN;
            return;
        }
        String reasoningSide = reasoningPair.diff().left();
        if (!reasoningEnd.isEmpty() && reasoningSide.contains(reasoningEnd)) {
            reasoningSide = reasoningSide.substring(reasoningSide.indexOf(reasoningEnd) + reasoningEnd.length());
        }
        String pure = reasoningSide.length() > toolsPair.diff().left().length()
                ? reasoningSide : toolsPair.diff().left();
        Tagged tagged = wrappedAround(pure, ASSISTANT_MSG);
        if (tagged != null) {
            contentStart = tagged.pre == null ? "" : tagged.pre;
            contentEnd = tagged.post == null ? "" : tagged.post;
        }
        if (!contentStart.isEmpty() || !contentEnd.isEmpty()) {
            contentMode = AutoParserAnalysis.ContentMode.ALWAYS_WRAPPED;
        }
    }

    // ================================================================== tools

    private void analyzeTools() {
        List<Map<String, Object>> a = Arrays.asList(userMessage(), assistantMessage(ASSISTANT_MSG, null, null));
        List<Map<String, Object>> b = Arrays.asList(userMessage(), assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_001))));
        TemplateProbe.VariantPair pair = probe.compare(a, b, tools(), false, true);
        if (pair == null) {
            return;
        }
        String toolSection = pair.diff().right();
        if (toolSection.isEmpty()) {
            return;
        }
        analyzeToolCallFormat(toolSection, probe.caps().supportsParallelToolCalls());
        if (toolMode == AutoParserAnalysis.ToolFormat.NONE
                || toolMode == AutoParserAnalysis.ToolFormat.JSON_NATIVE) {
            return;
        }
        if (probe.caps().supportsParallelToolCalls()) {
            checkPerCallMarkers();
        }
        extractFunctionMarkers();
        if (toolMode == AutoParserAnalysis.ToolFormat.TAG_WITH_TAGGED) {
            extractArgumentNameMarkers();
            extractArgumentValueMarkers();
        }
        extractArgumentSeparator();
        extractArgsMarkers();
        extractCallIdMarkers();
    }

    private void analyzeToolCallFormat(String haystack, boolean parallelSupported) {
        boolean funQuoted = inJsonHaystack(haystack, FUN_FIRST);
        boolean argQuoted = inJsonHaystack(haystack, ARG_FIRST);
        if (funQuoted) {
            toolMode = AutoParserAnalysis.ToolFormat.JSON_NATIVE;
        } else if (argQuoted) {
            toolMode = AutoParserAnalysis.ToolFormat.TAG_WITH_JSON;
        } else {
            toolMode = AutoParserAnalysis.ToolFormat.TAG_WITH_TAGGED;
        }
        String clean = removeMarker(stripMarker(haystack, reasoningStart), reasoningEnd);
        if (toolMode == AutoParserAnalysis.ToolFormat.JSON_NATIVE) {
            analyzeJsonNative(clean);
            if (parallelSupported) {
                analyzeJsonNativeParallelCalls();
            }
        } else {
            analyzeNonJson(clean);
        }
        sectionEnd = sectionEnd.trim();
        perCallEnd = perCallEnd.trim();
    }

    /** Whether {@code needle} is a JSON key in {@code haystack}: {@code {"needle"} or {@code :"needle"} (spaces allowed). */
    private static boolean inJsonHaystack(String haystack, String needle) {
        String quoted = "\"" + needle + "\"";
        for (int i = 0; i < haystack.length(); i++) {
            char c = haystack.charAt(i);
            if (c == '{' || c == ':') {
                int at = skipWhitespace(haystack, i + 1);
                if (haystack.startsWith(quoted, at)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static String stripMarker(String text, String marker) {
        if (marker == null || marker.isEmpty()) {
            return text;
        }
        int at = text.indexOf(marker);
        return at < 0 ? text : text.substring(0, at) + text.substring(at + marker.length());
    }

    private static String removeMarker(String text, String marker) {
        return stripMarker(text, marker);
    }

    private void analyzeJsonNativeParallelCalls() {
        Map<String, Object> one = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_001)));
        Map<String, Object> two = assistantMessage("", null, Arrays.asList(
                toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_001),
                toolCall(FUN_SECOND, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_002)));
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), one),
                Arrays.asList(userMessage(), two), tools(), false, true);
        if (pair == null) {
            return;
        }
        String secondCall = pair.diff().right();
        if (!sectionStart.isEmpty() && secondCall.contains(sectionStart)) {
            perCallStart = sectionStart;
            perCallEnd = sectionEnd;
            sectionStart = "";
            sectionEnd = "";
        }
    }

    private void analyzeJsonNative(String clean) {
        int jsonStart = clean.indexOf('{');
        int jsonEnd = clean.lastIndexOf('}');
        if (jsonStart < 0 || jsonEnd < jsonStart) {
            toolMode = AutoParserAnalysis.ToolFormat.NONE;
            return;
        }
        String cut = clean.substring(jsonStart, jsonEnd + 1);
        Object parsed;
        try {
            parsed = Json.parse(cut);
        } catch (RuntimeException e) {
            LOGGER.debug("Autoparser: tool call probe is not JSON: {}", e.getMessage());
            toolMode = AutoParserAnalysis.ToolFormat.NONE;
            return;
        }
        if (!(parsed instanceof Map)) {
            toolMode = AutoParserAnalysis.ToolFormat.NONE;
            return;
        }
        registerJsonFields((Map<?, ?>) parsed);
        int arrayStart = jsonStart;
        int arrayEnd = jsonEnd + 1;
        int cutAt = clean.indexOf(cut);
        int before = cutAt - 1;
        while (before >= 0 && Character.isWhitespace(clean.charAt(before))) {
            before--;
        }
        int after = skipWhitespace(clean, cutAt + cut.length());
        if (before >= 0 && clean.charAt(before) == '[' && after < clean.length() && clean.charAt(after) == ']') {
            toolsArrayWrapped = true;
            arrayStart = before;
            arrayEnd = after + 1;
        }
        sectionStart = ltrim(clean.substring(0, arrayStart));
        sectionEnd = clean.substring(Math.min(arrayEnd, clean.length())).trim();
        if (toolsArrayWrapped && sectionEnd.equals("]")) {
            sectionEnd = "";
        }
        locateParameterOrder(clean);
    }

    private void registerJsonFields(Map<?, ?> object) {
        for (Map.Entry<?, ?> el : object.entrySet()) {
            String key = String.valueOf(el.getKey());
            Object value = el.getValue();
            if (key.equals(FUN_FIRST)) {
                funNameIsKey = true;
                nameField = "";
                argsField = "";
                continue;
            }
            if (value instanceof Map && !dump(value).contains(ARG_FIRST)) {
                functionField = key;
                for (Map.Entry<?, ?> sub : ((Map<?, ?>) value).entrySet()) {
                    registerField(key, String.valueOf(sub.getKey()), sub.getValue());
                }
            }
            registerField("", key, value);
        }
    }

    private void registerField(String prefix, String key, Object value) {
        String path = prefix.isEmpty() ? key : prefix + "." + key;
        if (value instanceof String && ((String) value).contains("call0000")) {
            idField = path;
        } else if (value instanceof String && value.equals(FUN_FIRST)) {
            nameField = path;
        } else if (dump(value).contains(ARG_FIRST)) {
            argsField = path;
        } else if (key.contains("id")) {
            genIdField = path;
        }
    }

    private static String dump(Object value) {
        return Json.toString(value, Json.Style.API);
    }

    private void locateParameterOrder(String clean) {
        List<int[]> located = new ArrayList<>();
        List<String> names = new ArrayList<>();
        addLocated(clean, nameField, located, names);
        addLocated(clean, argsField, located, names);
        addLocated(clean, idField, located, names);
        addLocated(clean, genIdField, located, names);
        Integer[] order = new Integer[located.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (x, y) -> Integer.compare(located.get(x)[0], located.get(y)[0]));
        for (int index : order) {
            parameterOrder.add(names.get(index));
        }
    }

    private static void addLocated(String clean, String field, List<int[]> located, List<String> names) {
        if (field.isEmpty()) {
            return;
        }
        located.add(new int[]{clean.indexOf(field)});
        names.add(field);
    }

    /** A non-JSON format: tags around the function name (and possibly its arguments). */
    private void analyzeNonJson(String clean) {
        Tagged fun = functionTag(clean, FUN_FIRST);
        String funMarker = fun == null ? FUN_FIRST : fun.whole;

        // Two openers: a section, then a call that carries the function. Otherwise one section opener.
        for (int i = 0; i < clean.length(); i++) {
            int firstEnd = markerAt(clean, i);
            if (firstEnd < 0) {
                continue;
            }
            int second = skipWhitespace(clean, firstEnd);
            int secondEnd = markerAt(clean, second);
            if (secondEnd < 0) {
                continue;
            }
            int funAt = clean.indexOf(funMarker, secondEnd);
            if (funAt < 0) {
                continue;
            }
            sectionStart = clean.substring(i, firstEnd);
            perCallStart = clean.substring(second, secondEnd);
            functionNamePrefix += clean.substring(secondEnd, funAt);
            Tagged end = doubleCloser(clean.substring(funAt + funMarker.length()));
            perCallEnd = end.pre == null ? "" : end.pre;
            sectionEnd = end.post == null ? "" : end.post;
            return;
        }
        for (int i = 0; i < clean.length(); i++) {
            int firstEnd = markerAt(clean, i);
            if (firstEnd < 0) {
                continue;
            }
            int funAt = clean.indexOf(funMarker, firstEnd);
            if (funAt < 0) {
                continue;
            }
            sectionStart = clean.substring(i, skipWhitespace(clean, firstEnd));
            Tagged end = singleCloser(clean.substring(funAt + funMarker.length()));
            sectionEnd = end.pre == null ? "" : end.pre;
            return;
        }
    }

    /** The function tag containing {@code name}: its whole text, the part before the name and after it. */
    private static Tagged functionTag(String haystack, String name) {
        int at = haystack.indexOf(name);
        while (at >= 0) {
            int open = openerBefore(haystack, at);
            if (open >= 0) {
                char closer = haystack.charAt(open) == '<' ? '>' : ']';
                int close = haystack.indexOf(closer, at + name.length());
                int gt = haystack.indexOf('>', open + 1);
                if (close >= 0 && (gt < 0 || gt == close || haystack.charAt(open) == '[')) {
                    return new Tagged(haystack.substring(open, at), haystack.substring(at + name.length(), close),
                            haystack.substring(open, close + 1));
                }
            }
            at = haystack.indexOf(name, at + 1);
        }
        return null;
    }

    /** The opener immediately preceding {@code at} with no closer in between, or -1. */
    private static int openerBefore(String text, int at) {
        for (int i = at - 1; i >= 0; i--) {
            char c = text.charAt(i);
            if (c == '>' || c == ']') {
                return -1;
            }
            if (c == '<' || c == '[') {
                return i;
            }
        }
        return -1;
    }

    /** The rest of a per-call plus section layout: the section and call openers, then the function marker. */
    /** Two consecutive markers closing a call and its section, each with its trailing whitespace. */
    private static Tagged doubleCloser(String rest) {
        int lastStart = lastMarkerStart(rest);
        if (lastStart < 0) {
            return new Tagged(null, null, null);
        }
        int lastEnd = lastStart + markerLengthAt(rest, lastStart);
        if (!isWhitespace(rest, lastEnd, rest.length())) {
            return new Tagged(null, null, null);
        }
        int prevStart = lastMarkerStart(rest.substring(0, lastStart));
        if (prevStart < 0) {
            return new Tagged(null, null, null);
        }
        int prevEnd = prevStart + markerLengthAt(rest, prevStart);
        if (!isWhitespace(rest, prevEnd, lastStart)) {
            return new Tagged(null, null, null);
        }
        return new Tagged(rest.substring(prevStart, lastStart), rest.substring(lastStart), null);
    }

    /** The last marker of the rest, with the whitespace after it. */
    private static Tagged singleCloser(String rest) {
        int lastStart = lastMarkerStart(rest);
        if (lastStart < 0) {
            return new Tagged(null, null, null);
        }
        int lastEnd = lastStart + markerLengthAt(rest, lastStart);
        if (!isWhitespace(rest, lastEnd, rest.length())) {
            return new Tagged(null, null, null);
        }
        return new Tagged(rest.substring(lastStart), null, null);
    }

    // ------------------------------------------------------------------ parallel calls

    private void checkPerCallMarkers() {
        Map<String, Object> one = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_001)));
        Map<String, Object> two = assistantMessage("", null, Arrays.asList(
                toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_001),
                toolCall(FUN_SECOND, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_002)));
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), one),
                Arrays.asList(userMessage(), two), tools(), false, true);
        if (pair == null) {
            return;
        }
        MarkerDiff.Diff split = MarkerDiff.split(pair.diff().suffix(), pair.diff().right());
        String secondTool = ltrim(split.right());
        if (!sectionStart.isEmpty() && secondTool.startsWith(sectionStart)) {
            perCallStart = sectionStart;
            perCallEnd = sectionEnd;
            sectionStart = "";
            sectionEnd = "";
        }
        if (!perCallEnd.isEmpty()) {
            int oneCount = count(pair.outputA(), perCallEnd);
            int twoCount = count(pair.outputB(), perCallEnd);
            if (oneCount > 0 && oneCount == twoCount) {
                sectionEnd = perCallEnd;
                perCallEnd = "";
            }
        }
    }

    private static int count(String haystack, String needle) {
        int n = 0;
        for (int at = haystack.indexOf(needle); at >= 0; at = haystack.indexOf(needle, at + needle.length())) {
            n++;
        }
        return n;
    }

    // ------------------------------------------------------------------ function markers

    private void extractFunctionMarkers() {
        Map<String, Object> a = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_001)));
        Map<String, Object> b = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_SECOND, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_001)));
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), a),
                Arrays.asList(userMessage(), b), tools(), false, true);
        if (pair == null) {
            return;
        }
        MarkerDiff.Diff diff = pair.diff();
        if (!diff.left().contains(FUN_FIRST) || !diff.right().contains(FUN_SECOND)) {
            return;
        }
        String prefixMarker = !perCallStart.isEmpty() ? perCallStart : sectionStart;
        if (!prefixMarker.isEmpty() && diff.prefix().contains(prefixMarker)) {
            functionNamePrefix = diff.prefix().substring(diff.prefix().lastIndexOf(prefixMarker)
                    + prefixMarker.length());
        }
        int nameAt = diff.left().indexOf(FUN_FIRST);
        if (nameAt >= 0) {
            functionNamePrefix += diff.left().substring(0, nameAt);
            String after = diff.left().substring(nameAt + FUN_FIRST.length());
            int marker = firstMarkerStart(after);
            functionNameSuffix = marker < 0 ? after : after.substring(0, marker);
        }
        if (toolMode == AutoParserAnalysis.ToolFormat.TAG_WITH_JSON) {
            String ext = nonJsonPrefix(diff.suffix());
            if (hasMarkerAtOrAfter(diff.suffix(), ext.length())) {
                functionNameSuffix += ext;
            }
        } else {
            int marker = firstMarkerStart(diff.suffix());
            String ext = marker < 0 ? diff.suffix() : diff.suffix().substring(0, marker);
            functionNameSuffix += ext;
            String afterExt = diff.suffix().substring(ext.length());
            int sepEnd = argumentStartIndex(afterExt);
            if (sepEnd >= 0) {
                functionArgsSeparator = afterExt.substring(0, sepEnd).trim();
            }
        }
        functionClose = ltrim(extractCloser(diff, a));
    }

    /**
     * The text before the closer: up to the call/section end marker, or from the no-call diff when there is none.
     */
    private String extractCloser(MarkerDiff.Diff diff, Map<String, Object> oneCallMessage) {
        String suffixMarker = !perCallEnd.isEmpty() ? perCallEnd : sectionEnd;
        String closerSuffix;
        if (suffixMarker.isEmpty()) {
            Map<String, Object> noCall = assistantMessage(ASSISTANT_MSG, null, null);
            TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), oneCallMessage),
                    Arrays.asList(userMessage(), noCall), tools(), false, true);
            if (pair == null) {
                return "";
            }
            String left = pair.diff().left();
            int at = left.indexOf("YYYY");
            closerSuffix = at < 0 ? "" : left.substring(at + 4);
        } else {
            int at = diff.suffix().indexOf(suffixMarker);
            closerSuffix = at < 0 ? diff.suffix() : diff.suffix().substring(0, at);
        }
        if (closerSuffix.isEmpty()) {
            return "";
        }
        if (toolMode == AutoParserAnalysis.ToolFormat.TAG_WITH_TAGGED) {
            int at = closerSuffix.indexOf("YYYY");
            if (at < 0) {
                return "";
            }
            int i = skipWhitespace(closerSuffix, at + 4);
            int end = markerAt(closerSuffix, i);
            if (end < 0) {
                return "";
            }
            i = skipWhitespace(closerSuffix, end);
            return closerSuffix.substring(Math.min(i, closerSuffix.length()));
        }
        int at = closerSuffix.indexOf("YYYY");
        if (at < 0) {
            return "";
        }
        String post = closerSuffix.substring(at + 4);
        int last = lastOf(post, "}]");
        return last >= 0 && last < post.length() - 1 ? post.substring(last + 1) : "";
    }

    /** The prefix of {@code suffix} before the first {@code {} or {@code [} outside a marker; markers count as text. */
    private static String nonJsonPrefix(String suffix) {
        int i = 0;
        while (i < suffix.length()) {
            int marker = markerAt(suffix, i);
            if (marker == i) {
                int end = i + markerLengthAt(suffix, i);
                if (end < 0) {
                    break;
                }
                i = end;
                continue;
            }
            char c = suffix.charAt(i);
            if (c == '{' || c == '[') {
                break;
            }
            i++;
        }
        return suffix.substring(0, i);
    }

    private static boolean hasMarkerAtOrAfter(String text, int from) {
        return firstMarkerStart(text.substring(Math.min(from, text.length()))) >= 0;
    }

    /** Where an argument name begins after a function name: a marker, whitespace, then the first argument name. */
    private static int argumentStartIndex(String text) {
        int i = skipWhitespace(text, 0);
        int marker = markerAt(text, i);
        if (marker < 0) {
            return -1;
        }
        int after = skipWhitespace(text, marker);
        if (text.startsWith(ARG_FIRST, after) || text.startsWith(ARG_SECOND, after)) {
            return i;
        }
        return -1;
    }

    // ------------------------------------------------------------------ argument markers

    private void extractArgumentNameMarkers() {
        Map<String, Object> a = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX"), CALL_ID_001)));
        Map<String, Object> b = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_SECOND, "YYYY"), CALL_ID_001)));
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), a),
                Arrays.asList(userMessage(), b), tools(), false, true);
        if (pair == null) {
            return;
        }
        MarkerDiff.Diff diff = pair.diff();
        if (diff.left().isEmpty() || diff.right().isEmpty()) {
            return;
        }
        Tagged left = nameInStructure(diff.left(), ARG_FIRST);
        Tagged right = nameInStructure(diff.right(), ARG_SECOND);
        if (left != null && right != null && !left.pre.isEmpty() && left.pre.equals(right.pre)
                && left.post != null && left.post.equals(right.post)) {
            argNamePrefix = left.pre;
            argNameSuffix = left.post;
            return;
        }
        if (diff.left().startsWith(ARG_FIRST) && diff.right().startsWith(ARG_SECOND)) {
            int lastMarker = lastMarkerStart(diff.prefix());
            argNamePrefix = lastMarker < 0 ? diff.prefix() : diff.prefix().substring(lastMarker);
            String afterFirst = diff.left().substring(ARG_FIRST.length()) + diff.suffix();
            int marker = firstMarkerStart(afterFirst);
            if (marker >= 0) {
                int end = marker + markerLengthAt(afterFirst, marker);
                argNameSuffix = afterFirst.substring(0, skipWhitespace(afterFirst, end));
            }
        }
    }

    /** The shared structure around a name: the text before it (up to a `"` or the value) and after it. */
    private static Tagged nameInStructure(String text, String name) {
        int at = text.indexOf(name);
        if (at < 0) {
            return null;
        }
        int end = at + name.length();
        int stop = end;
        while (stop < text.length() && text.charAt(stop) != '"' && text.charAt(stop) != 'X'
                && text.charAt(stop) != 'Y') {
            stop++;
        }
        return new Tagged(text.substring(0, at), text.substring(end, stop), null);
    }

    private void extractArgumentValueMarkers() {
        Map<String, Object> a = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX"), CALL_ID_001)));
        Map<String, Object> b = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "YYYY"), CALL_ID_001)));
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), a),
                Arrays.asList(userMessage(), b), tools(), false, true);
        if (pair == null) {
            return;
        }
        MarkerDiff.Diff diff = pair.diff();
        if (!diff.left().equals("XXXX") || !diff.right().equals("YYYY")) {
            return;
        }
        String argNameEnding = ARG_FIRST + argNameSuffix;
        String prefix = diff.prefix();
        int at = prefix.lastIndexOf(argNameEnding);
        if (at >= 0) {
            prefix = prefix.substring(at + argNameEnding.length());
        }
        if (!prefix.isEmpty()) {
            int lastMarker = lastMarkerStart(prefix);
            argValuePrefix = lastMarker < 0 ? prefix : prefix.substring(lastMarker);
        }
        String valueSuffix = diff.suffix();
        if (!functionClose.isEmpty()) {
            int close = valueSuffix.indexOf(functionClose);
            if (close >= 0) {
                valueSuffix = valueSuffix.substring(0, close);
            }
        } else if (!perCallEnd.isEmpty() || !sectionEnd.isEmpty()) {
            String endMarker = !perCallEnd.isEmpty() ? perCallEnd : sectionEnd;
            int close = valueSuffix.indexOf(endMarker);
            if (close >= 0) {
                valueSuffix = valueSuffix.substring(0, close);
            }
        }
        if (!valueSuffix.trim().isEmpty()) {
            argValueSuffix = valueSuffix;
        }
    }

    private void extractArgumentSeparator() {
        Map<String, Object> one = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX"), CALL_ID_001)));
        Map<String, Object> two = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_001)));
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), one),
                Arrays.asList(userMessage(), two), tools(), false, true);
        if (pair == null) {
            return;
        }
        argSeparator = MarkerDiff.untilCommonPrefix(pair.diff().right(), ARG_FIRST, ARG_SECOND);
    }

    private void extractArgsMarkers() {
        if (toolMode != AutoParserAnalysis.ToolFormat.JSON_NATIVE) {
            return;
        }
        Map<String, Object> noArgs = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, emptyArgs(), CALL_ID_001)));
        Map<String, Object> oneArg = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX"), CALL_ID_001)));
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), noArgs),
                Arrays.asList(userMessage(), oneArg), tools(), false, true);
        if (pair == null) {
            return;
        }
        MarkerDiff.Diff diff = pair.diff();
        String prefixMarker = !sectionStart.isEmpty() ? sectionStart : perCallStart;
        String suffixMarker = !sectionEnd.isEmpty() ? sectionEnd : perCallEnd;
        int prefixAt = prefixMarker.isEmpty() ? 0 : diff.prefix().lastIndexOf(prefixMarker);
        if (prefixAt < 0) {
            prefixAt = 0;
        } else {
            prefixAt += prefixMarker.length();
        }
        int suffixAt = suffixMarker.isEmpty() ? diff.suffix().length() : diff.suffix().indexOf(suffixMarker);
        if (suffixAt < 0) {
            suffixAt = diff.suffix().length();
        }
        String prefixCut = diff.prefix().substring(prefixAt);
        String suffixCut = diff.suffix().substring(0, suffixAt);
        String start = MarkerDiff.untilCommonPrefix(prefixCut, "{}", "{\"first\":");
        String end = MarkerDiff.afterCommonSuffix(suffixCut, "{}", "\"XXXX\"}");
        if (!start.isEmpty() || !end.isEmpty()) {
            int fun = start.indexOf(FUN_FIRST);
            if (fun >= 0) {
                start = start.substring(fun + FUN_FIRST.length());
            }
            int id = start.indexOf(CALL_ID_001);
            if (id >= 0) {
                start = start.substring(id + CALL_ID_001.length());
            }
            argsStart = start;
            argsEnd = end;
        }
    }

    // ------------------------------------------------------------------ call id

    private void extractCallIdMarkers() {
        Map<String, Object> first = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_001)));
        Map<String, Object> second = assistantMessage("", null,
                Arrays.asList(toolCall(FUN_FIRST, args(ARG_FIRST, "XXXX", ARG_SECOND, "YYYY"), CALL_ID_999)));
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(userMessage(), first),
                Arrays.asList(userMessage(), second), tools(), false, true);
        if (pair == null) {
            return;
        }
        MarkerDiff.Diff diff = pair.diff();
        if (diff.left().isEmpty() && diff.right().isEmpty()) {
            return;
        }
        int shared = MarkerDiff.commonPrefixLength(CALL_ID_001, CALL_ID_999);
        String commonId = CALL_ID_001.substring(0, shared);
        int funcInPrefix = diff.prefix().lastIndexOf(FUN_FIRST);
        int funcInSuffix = diff.suffix().indexOf(FUN_FIRST);

        if (funcInPrefix >= 0 && funcInSuffix < 0) {
            int argsInPrefix = diff.prefix().indexOf('{', funcInPrefix);
            int argsInSuffix = diff.suffix().indexOf('{');
            if (argsInSuffix >= 0 && (argsInPrefix < 0 || argsInPrefix > diff.prefix().length())) {
                callIdPosition = AutoParserAnalysis.CallIdPosition.BETWEEN_FUNC_AND_ARGS;
                String afterFunc = diff.prefix().substring(funcInPrefix + FUN_FIRST.length());
                int idAt = afterFunc.indexOf(commonId);
                if (idAt >= 0) {
                    int marker = lastMarkerStart(afterFunc.substring(0, idAt));
                    callIdPrefix = marker < 0 ? "" : afterFunc.substring(marker, idAt);
                } else {
                    callIdPrefix = lastMarkerOnly(afterFunc);
                }
                int marker = firstMarkerStart(diff.suffix());
                if (marker >= 0) {
                    callIdSuffix = diff.suffix().substring(marker, marker + markerLengthAt(diff.suffix(), marker));
                }
            } else if (argsInPrefix >= 0) {
                callIdPosition = AutoParserAnalysis.CallIdPosition.POST_ARGS;
                String afterArgs = diff.prefix().substring(argsInPrefix);
                int brace = afterArgs.lastIndexOf('}');
                if (brace >= 0) {
                    callIdPrefix = lastMarkerOnly(afterArgs.substring(brace + 1));
                }
                int marker = firstMarkerStart(diff.suffix());
                if (marker >= 0) {
                    callIdSuffix = diff.suffix().substring(marker, marker + markerLengthAt(diff.suffix(), marker));
                }
            }
        } else if (funcInSuffix >= 0 && funcInPrefix < 0) {
            callIdPosition = AutoParserAnalysis.CallIdPosition.PRE_FUNC_NAME;
            callIdPrefix = lastMarkerOnly(diff.prefix());
            int marker = firstMarkerStart(diff.suffix().substring(0, funcInSuffix));
            if (marker >= 0) {
                callIdSuffix = markerValue(diff.suffix().substring(0, funcInSuffix), marker);
            }
        }
        if (callIdPosition == AutoParserAnalysis.CallIdPosition.BETWEEN_FUNC_AND_ARGS
                && !callIdPrefix.isEmpty()) {
            // The id section sits after the name, so the name suffix ends where the id begins.
            int at = functionNameSuffix.indexOf(callIdPrefix);
            if (at >= 0) {
                functionNameSuffix = functionNameSuffix.substring(0, at);
            }
            at = functionArgsSeparator.indexOf(callIdPrefix);
            if (at >= 0) {
                functionArgsSeparator = functionArgsSeparator.substring(0, at).trim();
            }
        }
        if (callIdPrefix.equals(argsEnd)) {
            callIdPrefix = "";
        }
        if (callIdSuffix.equals(argsStart)) {
            callIdSuffix = "";
        }
        if (callIdPosition != AutoParserAnalysis.CallIdPosition.NONE && !callIdSuffix.isEmpty()
                && perCallEnd.startsWith(callIdSuffix)) {
            perCallEnd = "";
        }
        if (callIdPosition == AutoParserAnalysis.CallIdPosition.PRE_FUNC_NAME) {
            stripCallIdFromFunctionPrefix();
        }
        if (callIdPosition == AutoParserAnalysis.CallIdPosition.POST_ARGS && !functionClose.isEmpty()
                && !callIdPrefix.isEmpty()) {
            removeCallIdFromCloser();
        }
    }

    /**
     * The closer the function-name probe read can run through the id region (the probe keeps the same id and
     * arguments in both renders, so it lies in the common suffix). The id's marker, sample value and closer are
     * cut out, leaving the text the grammar writes after the optional id.
     */
    private void removeCallIdFromCloser() {
        int at = functionClose.indexOf(callIdPrefix);
        if (at < 0) {
            return;
        }
        int end = at + callIdPrefix.length();
        for (String id : new String[]{CALL_ID_001, CALL_ID_999}) {
            if (functionClose.startsWith(id, end)) {
                end += id.length();
                break;
            }
        }
        if (!callIdSuffix.isEmpty() && functionClose.startsWith(callIdSuffix, end)) {
            end += callIdSuffix.length();
        }
        functionClose = functionClose.substring(0, at) + functionClose.substring(end);
    }

    /**
     * A call id before the function name lies inside the function-name prefix the name probe derived (the probe
     * keeps the same id in both renders), so the id region is cut out of it: the grammar then writes the prefix
     * before the id and the rest after it.
     */
    private void stripCallIdFromFunctionPrefix() {
        if (callIdPrefix.isEmpty()) {
            return;
        }
        int at = functionNamePrefix.indexOf(callIdPrefix);
        if (at < 0) {
            return;
        }
        int afterId = at + callIdPrefix.length();
        if (!callIdSuffix.isEmpty()) {
            int end = functionNamePrefix.indexOf(callIdSuffix, afterId);
            if (end < 0) {
                return;
            }
            functionNamePrefix = functionNamePrefix.substring(0, at)
                    + functionNamePrefix.substring(end + callIdSuffix.length());
            return;
        }
        // without an id closer, the probe's own id value bounds the region
        for (String id : new String[]{CALL_ID_001, CALL_ID_999}) {
            int idAt = functionNamePrefix.indexOf(id, afterId);
            if (idAt >= 0) {
                functionNamePrefix = functionNamePrefix.substring(0, at)
                        + functionNamePrefix.substring(idAt + id.length());
                return;
            }
        }
    }

    // ================================================================== role markers

    private String detectAssistantStart() {
        Map<String, Object> user = userMessage();
        Map<String, Object> assistant = assistantMessage(ASSISTANT_MSG, null, null);
        TemplateProbe.VariantPair pair = probe.compare(Arrays.asList(user), Arrays.asList(user, assistant), null,
                false, true);
        if (pair == null) {
            return "";
        }
        String right = pair.diff().right();
        String prefix = right.contains(ASSISTANT_MSG)
                ? right.substring(0, right.indexOf(ASSISTANT_MSG)) : right;
        if (!reasoningStart.isEmpty()) {
            int at = prefix.indexOf(reasoningStart.trim());
            if (at >= 0) {
                prefix = prefix.substring(0, at);
            }
        }
        if (!reasoningEnd.isEmpty()) {
            int at = prefix.indexOf(reasoningEnd.trim());
            if (at >= 0) {
                prefix = prefix.substring(0, at);
            }
        }
        return prefix.trim();
    }

    private String detectUserStart() {
        Map<String, Object> user = userMessage();
        Map<String, Object> assistant = assistantMessage(ASSISTANT_MSG, null, null);
        TemplateProbe.VariantPair pair = probe.compare(new ArrayList<Map<String, Object>>(),
                Arrays.asList(user), null, false, true);
        String right;
        if (pair == null) {
            Map<String, Object> other = message("user", USER_MSG_TWO);
            pair = probe.compare(Arrays.asList(other, assistant), Arrays.asList(other, assistant, user), null,
                    false, true);
            if (pair == null) {
                return "";
            }
            right = pair.diff().right();
        } else {
            right = pair.diff().right();
        }
        if (right.contains(ASSISTANT_MSG)) {
            right = right.substring(right.indexOf(ASSISTANT_MSG) + ASSISTANT_MSG.length());
        }
        int at = right.indexOf(USER_MSG);
        String candidate = at < 0 ? right : right.substring(0, at);
        StringBuilder out = new StringBuilder();
        boolean encounteredMarker = false;
        for (MarkerDiff.Segment segment : MarkerDiff.segmentize(candidate)) {
            String lower = segment.value().toLowerCase(java.util.Locale.ROOT);
            if (segment.type() == MarkerDiff.Type.MARKER && !encounteredMarker
                    && (lower.contains("end") || lower.contains("close"))) {
                continue;
            }
            if (segment.type() == MarkerDiff.Type.TEXT && !encounteredMarker
                    && segment.value().trim().isEmpty()) {
                continue;
            }
            encounteredMarker |= segment.type() == MarkerDiff.Type.MARKER;
            out.append(segment.value());
        }
        return out.toString().trim();
    }

    // ================================================================== preserved tokens

    private static List<String> preservedTokens(AutoParserAnalysis.Reasoning reasoning,
                                                AutoParserAnalysis.Content content,
                                                AutoParserAnalysis.Tools tools) {
        Set<String> out = new LinkedHashSet<>();
        addToken(out, reasoning.start());
        addToken(out, reasoning.end());
        addToken(out, content.start());
        addToken(out, content.end());
        addToken(out, tools.sectionStart());
        addToken(out, tools.sectionEnd());
        addToken(out, tools.perCallStart());
        addToken(out, tools.perCallEnd());
        addToken(out, tools.functionNamePrefix());
        addToken(out, tools.functionNameSuffix());
        addToken(out, tools.functionArgsSeparator());
        addToken(out, tools.functionClose());
        addToken(out, tools.argsStart());
        addToken(out, tools.argsEnd());
        addToken(out, tools.argNamePrefix());
        addToken(out, tools.argNameSuffix());
        addToken(out, tools.argSeparator());
        addToken(out, tools.argValuePrefix());
        addToken(out, tools.argValueSuffix());
        addToken(out, tools.callIdPrefix());
        addToken(out, tools.callIdSuffix());
        return new ArrayList<>(out);
    }

    private static void addToken(Set<String> out, String token) {
        String trimmed = token == null ? "" : token.trim();
        if (!trimmed.isEmpty()) {
            out.add(trimmed);
        }
    }

    // ================================================================== probe data

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content == null ? "" : content);
        return m;
    }

    private static Map<String, Object> userMessage() {
        return message("user", USER_MSG);
    }

    /** An assistant message with optional reasoning and tool calls; null content renders as the empty string. */
    private static Map<String, Object> assistantMessage(String content, String reasoning, List<Object> calls) {
        Map<String, Object> m = message("assistant", content);
        if (reasoning != null) {
            m.put("reasoning_content", reasoning);
        }
        if (calls != null) {
            m.put("tool_calls", calls);
        }
        return m;
    }

    private static Map<String, Object> toolCall(String name, Map<String, Object> arguments, String id) {
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", name);
        function.put("arguments", arguments);
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("id", id);
        call.put("type", "function");
        call.put("name", name);
        call.put("arguments", arguments);
        call.put("function", function);
        return call;
    }

    private static Map<String, Object> args(String k1, String v1) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(k1, v1);
        return out;
    }

    private static Map<String, Object> args(String k1, String v1, String k2, String v2) {
        Map<String, Object> out = args(k1, v1);
        out.put(k2, v2);
        return out;
    }

    private static Map<String, Object> emptyArgs() {
        return new LinkedHashMap<>();
    }

    private static List<Map<String, Object>> tools() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put(ARG_FIRST, property("First argument"));
        properties.put(ARG_SECOND, property("Second argument"));
        schema.put("properties", properties);
        schema.put("required", new ArrayList<String>());
        List<Map<String, Object>> out = new ArrayList<>();
        out.add(tool(FUN_FIRST, "Test function foo", schema));
        out.add(tool(FUN_SECOND, "Test function bar", schema));
        return out;
    }

    private static Map<String, Object> property(String description) {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("type", "string");
        p.put("description", description);
        return p;
    }

    private static Map<String, Object> tool(String name, String description, Object parameters) {
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", name);
        function.put("description", description);
        function.put("parameters", parameters);
        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("type", "function");
        tool.put("function", function);
        return tool;
    }

    // ================================================================== marker helpers

    /** One captured region around a sentinel: the text before it, after it and the whole enclosing marker. */
    private static final class Tagged {
        final String pre;
        final String post;
        final String whole;

        Tagged(String pre, String post, String whole) {
            this.pre = pre;
            this.post = post;
            this.whole = whole;
        }
    }

    /**
     * The marker immediately before {@code needle} (whitespace between) and the marker after it (leading and
     * trailing whitespace captured, as llama.cpp's {@code marker + space} tag does), or null when none is found.
     */
    private static Tagged wrappedAround(String text, String needle) {
        int at = text.indexOf(needle);
        while (at >= 0) {
            String pre = markerBefore(text, at);
            String post = markerAfter(text, at + needle.length());
            if (pre != null || post != null) {
                return new Tagged(pre, post, null);
            }
            at = text.indexOf(needle, at + 1);
        }
        return null;
    }

    /** The marker ending at or before {@code pos} with only whitespace between it and {@code pos}. */
    private static String markerBefore(String text, int pos) {
        int start = -1;
        int end = -1;
        int i = 0;
        while (i < text.length()) {
            char c = text.charAt(i);
            if (c == '<' || c == '[') {
                int close = closeOf(text, i);
                if (close < 0) {
                    break;
                }
                int after = close + 1;
                if (after <= pos && isWhitespace(text, after, pos)) {
                    start = i;
                    end = after;
                }
                i = after;
            } else {
                i++;
            }
        }
        return start < 0 ? null : text.substring(start, pos);
    }

    /** From {@code pos}: optional whitespace, a marker, optional whitespace; null when no marker follows. */
    private static String markerAfter(String text, int pos) {
        int i = skipWhitespace(text, pos);
        int close = closeOf(text, i);
        if (close < 0) {
            return null;
        }
        int end = skipWhitespace(text, close + 1);
        return text.substring(pos, end);
    }

    private static int closeOf(String text, int at) {
        if (at < 0 || at >= text.length()) {
            return -1;
        }
        char c = text.charAt(at);
        char close = c == '<' ? '>' : c == '[' ? ']' : '\0';
        if (close == '\0') {
            return -1;
        }
        int found = text.indexOf(close, at + 1);
        return found < 0 ? -1 : found;
    }

    /** The end index of a marker starting at {@code i}, or -1. */
    private static int markerAt(String text, int i) {
        int close = closeOf(text, i);
        return close < 0 ? -1 : close + 1;
    }

    private static int markerLengthAt(String text, int i) {
        int end = markerAt(text, i);
        return end < 0 ? 0 : end - i;
    }

    private static int skipWhitespace(String text, int i) {
        int at = i;
        while (at < text.length() && Character.isWhitespace(text.charAt(at))) {
            at++;
        }
        return at;
    }

    private static boolean isWhitespace(String text, int from, int to) {
        for (int i = from; i < to; i++) {
            if (!Character.isWhitespace(text.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    /** The start of the first marker in {@code text}, or -1. */
    private static int firstMarkerStart(String text) {
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if ((c == '<' || c == '[') && closeOf(text, i) >= 0) {
                return i;
            }
        }
        return -1;
    }

    /** The start of the last marker in {@code text}, or -1. */
    private static int lastMarkerStart(String text) {
        int last = -1;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '<' || c == '[') {
                int close = closeOf(text, i);
                if (close >= 0) {
                    last = i;
                    i = close;
                }
            }
        }
        return last;
    }

    /** The last marker of {@code text} plus the text after it, or empty when it has none. */
    private static String lastMarker(String text) {
        int at = lastMarkerStart(text);
        return at < 0 ? "" : text.substring(at);
    }

    /** The last marker of {@code text} alone, or empty when it has none. */
    private static String lastMarkerOnly(String text) {
        int at = lastMarkerStart(text);
        return at < 0 ? "" : markerValue(text, at);
    }

    /** The marker starting at {@code at} (text unchanged when it is no marker). */
    private static String markerValue(String text, int at) {
        int end = markerAt(text, at);
        return end < 0 ? "" : text.substring(at, end);
    }

    private static int lastOf(String text, String chars) {
        for (int i = text.length() - 1; i >= 0; i--) {
            if (chars.indexOf(text.charAt(i)) >= 0) {
                return i;
            }
        }
        return -1;
    }

    private static String ltrim(String s) {
        return s.substring(skipWhitespace(s, 0));
    }
}
