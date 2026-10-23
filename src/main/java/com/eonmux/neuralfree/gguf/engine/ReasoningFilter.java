package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.TokenCallback;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.ToIntFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Splits streamed model output into content, reasoning and tool-call text.
 *
 * <p>The filter itself is an immutable configuration; {@link #newState} creates the per-generation
 * streaming {@link State}, which is a {@link TokenCallback} that routes each piece of text to a
 * {@link Sink} together with its {@link Channel}. Three modes exist:
 * <ul>
 *   <li>{@link Mode#THINK_TAGS}: an optional leading {@code start ... end} reasoning block
 *       (llama.cpp common/chat-auto-parser-generator.cpp analyze_reasoning::build_parser:
 *       {@code optional(optspace(start) + reasoning(until(end)) + optspace(end))}); interrupt tags
 *       such as a tool-call opener also end reasoning but stay in the content.</li>
 *   <li>{@link Mode#HARMONY_CHANNELS}: header/body messages ({@code <|start|>header<|message|>body}
 *       closed by a terminator) in two dialects. OpenAI harmony: the {@code analysis} channel is
 *       reasoning, {@code final} and recipient-less {@code commentary} are content, and a {@code to=}
 *       recipient other than {@code assistant} marks a tool call (llama.cpp common/parsers/gpt-oss.cpp:68-127).
 *       Muse recipients: {@code to=self} is reasoning, {@code to=user}, {@code to=commentary} and no
 *       recipient are content, any other recipient is a tool call, and {@code <|eom|>}/{@code <|eot|>}
 *       end a message (llama.cpp common/parsers/muse-spark.cpp:61-76, muse-glimmer.cpp:59-67).</li>
 *   <li>{@link Mode#NONE}: everything is content.</li>
 * </ul>
 * Tags are matched in the decoded text; {@link #withTagTokens} additionally maps special-token ids
 * to their tag text so the filter works even when the detokenizer hides special tokens.
 */
public final class ReasoningFilter {

    /** Parsing mode. */
    public enum Mode { THINK_TAGS, HARMONY_CHANNELS, NONE }

    /** Destination of a piece of output text. */
    public enum Channel { CONTENT, REASONING, TOOL_CALL }

    /** Receives routed text. {@code tokenId} is the token that released the text, or -1 when {@link State#finish()} did. */
    @FunctionalInterface
    public interface Sink {
        void accept(Channel channel, int tokenId, String text);
    }

    static final String H_START = "<|start|>";
    static final String H_CHANNEL = "<|channel|>";
    static final String H_MESSAGE = "<|message|>";
    static final String H_END = "<|end|>";
    static final String H_RETURN = "<|return|>";
    static final String H_CALL = "<|call|>";
    static final String H_CONSTRAIN = "<|constrain|>";
    /** Muse message terminators: more messages follow ({@code <|eom|>}) or the turn ends ({@code <|eot|>}). */
    static final String M_EOM = "<|eom|>";
    static final String M_EOT = "<|eot|>";
    private static final List<String> HARMONY_TAGS = Collections.unmodifiableList(Arrays.asList(
            H_START, H_CHANNEL, H_MESSAGE, H_END, H_RETURN, H_CALL, H_CONSTRAIN, M_EOM, M_EOT));

    /** Apriel 1.6 reasoning: the generation prompt opens it, the final-response marker closes it. */
    private static final String APRIEL_REASONING_START = "Here are my reasoning steps:";
    private static final String APRIEL_FINAL_RESPONSE = "[BEGIN FINAL RESPONSE]";

    /** DeepSeek V3.2 / V4 DSML tool-call block openers; the DSML delimiters are fullwidth vertical bars (U+FF5C). */
    private static final String DSML_FUNCTION_CALLS = "<\uFF5CDSML\uFF5Cfunction_calls>";
    private static final String DSML_TOOL_CALLS = "<\uFF5CDSML\uFF5Ctool_calls>";

    /** Single special-token candidates inside a tag: {@code <...>} and {@code [...]} pieces. */
    private static final Pattern TOKEN_PIECE = Pattern.compile("<[^<>]*>|\\[[^\\[\\]]*\\]");

    private static final ReasoningFilter NONE = new ReasoningFilter(Mode.NONE, Collections.<String>emptyList(),
            Collections.<String>emptyList(), Collections.<String>emptyList(), Collections.<Integer, String>emptyMap());

    private final Mode mode;
    private final List<String> startTags;
    private final List<String> endTags;
    private final List<String> interruptTags;
    private final Map<Integer, String> tagTokens;

    private ReasoningFilter(Mode mode, List<String> startTags, List<String> endTags, List<String> interruptTags,
                            Map<Integer, String> tagTokens) {
        this.mode = mode;
        this.startTags = Collections.unmodifiableList(new ArrayList<>(startTags));
        this.endTags = Collections.unmodifiableList(new ArrayList<>(endTags));
        this.interruptTags = Collections.unmodifiableList(new ArrayList<>(interruptTags));
        this.tagTokens = Collections.unmodifiableMap(new HashMap<>(tagTokens));
    }

    // ================================================================= factories

    /** Passes all text through as content. */
    public static ReasoningFilter none() { return NONE; }

    /** {@code <think>...</think>} reasoning. */
    public static ReasoningFilter thinkTags() {
        return thinkTags(Collections.singletonList("<think>"), Collections.singletonList("</think>"),
                Collections.<String>emptyList());
    }

    /**
     * Tag-delimited reasoning.
     *
     * @param starts     tags that open reasoning (at least one)
     * @param ends       tags that close reasoning and are dropped (at least one)
     * @param interrupts tags that close reasoning and remain part of the content (may be empty)
     * @throws IllegalArgumentException for missing, null or empty tags
     */
    public static ReasoningFilter thinkTags(List<String> starts, List<String> ends, List<String> interrupts) {
        requireTags("start", starts, false);
        requireTags("end", ends, false);
        requireTags("interrupt", interrupts, true);
        return new ReasoningFilter(Mode.THINK_TAGS, starts, ends, interrupts, Collections.<Integer, String>emptyMap());
    }

    /** Header/body messages: OpenAI harmony channels (gpt-oss and the harmony dialects) and Muse recipients. */
    public static ReasoningFilter harmony() {
        return new ReasoningFilter(Mode.HARMONY_CHANNELS, Collections.<String>emptyList(), Collections.<String>emptyList(),
                Collections.<String>emptyList(), Collections.<Integer, String>emptyMap());
    }

    private static void requireTags(String what, List<String> tags, boolean mayBeEmpty) {
        if (tags == null) throw new IllegalArgumentException(what + " tags must not be null");
        if (!mayBeEmpty && tags.isEmpty()) throw new IllegalArgumentException("at least one " + what + " tag is required");
        for (String t : tags) {
            if (t == null || t.isEmpty()) throw new IllegalArgumentException(what + " tags must be non-empty strings");
        }
    }

    /**
     * Picks the filter for a chat template, mirroring llama.cpp's template dispatch
     * (llama.cpp common/chat.cpp common_chat_template_detect_params_init) and the reasoning tags of each
     * dedicated parser. Templates handled by the autoparser use the reasoning delimiters it derives for
     * them: Apriel 1.6's forced-open reasoning ended by its final-response marker, otherwise the think
     * tags the template contains. A null template, or one without reasoning delimiters, yields {@link #none()}.
     */
    public static ReasoningFilter forChatTemplate(String template) {
        if (template == null) return NONE;
        String s = template;
        if (s.contains("[SYSTEM_PROMPT]") && s.contains("[TOOL_CALLS]") && s.contains("[ARGS]") && !s.contains("[CALL_ID]")) {
            return think("[THINK]", "[/THINK]");                              // parsers/ministral3.cpp:52-53
        }
        if (s.contains("chat_format=llm-jp-harmony-v1") || s.contains(H_CHANNEL)) {
            return harmony();                                                 // parsers/gpt-oss.cpp:43-44
        }
        if (s.contains("to=commentary") && s.contains(M_EOM)) {
            return harmony();                                                 // chat.cpp:1121-1124, parsers/muse-spark.cpp:61-76
        }
        if (s.contains("<atem:function_calls>") && s.contains(M_EOM)) {
            return harmony();                                                 // chat.cpp:1127-1130, parsers/muse-glimmer.cpp:59-67
        }
        if (s.contains("<|tool_calls_section_begin|>") && s.contains("<|tool_call_begin|>")) {
            return think("<think>", "</think>");                              // parsers/kimi-k2.cpp:33-34
        }
        if (s.contains("<|open|>") && s.contains("<|close|>") && s.contains("<|end_of_msg|>")) {
            return think("<|open|>think<|sep|>", "<|close|>think<|sep|>");    // parsers/kimi-k3.cpp:19-20
        }
        if (s.contains("<|ifm|im_start|>") && s.contains("<ifm|tool_calls>")) {
            return thinkTags(Arrays.asList("<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"),
                    Arrays.asList("</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"),
                    Collections.<String>emptyList());                          // parsers/k2-horizon.cpp:30-36
        }
        if (s.contains("<role>ASSISTANT</role>") && s.contains("<arg_key>")) {
            return thinkTags(Collections.singletonList("<think>"), Collections.singletonList("</think>"),
                    Collections.singletonList("<tool_call>"));                 // parsers/ling3.cpp:36-39
        }
        if (s.contains("<|START_TEXT|>") && s.contains("<|START_ACTION|>")) {
            return think("<|START_THINKING|>", "<|END_THINKING|>");           // parsers/cohere2moe.cpp:28-29
        }
        if (s.contains("]<]minimax[>[") && s.contains("<tool_call>") && s.contains("<invoke name=")) {
            return thinkTags(Collections.singletonList("<mm:think>"), Collections.singletonList("</mm:think>"),
                    Collections.singletonList("]<]minimax[>[<tool_call>"));     // parsers/minimax-m3.cpp:11-19
        }
        if (s.contains("dsml_token") && s.contains("DSML") && (s.contains("function_calls") || s.contains("tool_calls"))) {
            return thinkTags(Collections.singletonList("<think>"), Collections.singletonList("</think>"),
                    Arrays.asList(DSML_FUNCTION_CALLS, DSML_TOOL_CALLS));         // parsers/deepseek.cpp:93-120
        }
        if (s.contains("'<|tool_call>call:'")) {
            return think("<|channel>thought", "<channel|>");                  // parsers/gemma4.cpp:173-174
        }
        if (s.contains("<tool_call>") && s.contains("<function=") && s.contains("<parameter=")
                && !s.contains("'<tool_call><function=' ~ tool_call.name ~ '>'") && s.contains("<think>")) {
            return thinkTags(Collections.singletonList("<think>"), Arrays.asList("\n</think>", "</think>"),
                    Collections.singletonList("<tool_call>"));                 // chat.cpp:1245-1252, parsers/qwen3-coder.cpp:13-28
        }
        if (s.contains(APRIEL_REASONING_START) && s.contains(APRIEL_FINAL_RESPONSE)) {
            // chat-diff-analyzer.cpp:160-166; the generation prompt opens reasoning and "\n[BEGIN FINAL RESPONSE]\n"
            // closes it (models/templates/Apriel-1.6-15b-Thinker-fixed.jinja:21, :106, :167-168)
            return thinkTags(Collections.singletonList(APRIEL_REASONING_START),
                    Arrays.asList("\n" + APRIEL_FINAL_RESPONSE, APRIEL_FINAL_RESPONSE), Collections.<String>emptyList());
        }
        if (s.contains("[THINK]")) return think("[THINK]", "[/THINK]");
        if (s.contains("<seed:think>")) return think("<seed:think>", "</seed:think>");
        if (s.contains("<|START_THINKING|>")) return think("<|START_THINKING|>", "<|END_THINKING|>");
        if (s.contains("<think>") || s.contains("</think>")) return thinkTags();
        return NONE;
    }

    private static ReasoningFilter think(String start, String end) {
        return thinkTags(Collections.singletonList(start), Collections.singletonList(end), Collections.<String>emptyList());
    }

    /**
     * Returns a copy that also recognizes the special tokens of its tags by id: every tag, and every
     * {@code <...>} or {@code [...]} piece of a tag, for which {@code lookup} returns an id &gt;= 0
     * is matched by id, its text replaced by the tag text. Typical lookup: {@code tokenizer::getTokenId}.
     */
    public ReasoningFilter withTagTokens(ToIntFunction<String> lookup) {
        Objects.requireNonNull(lookup, "lookup");
        if (mode == Mode.NONE) return this;
        Set<String> candidates = new LinkedHashSet<>();
        List<String> tags = new ArrayList<>(mode == Mode.HARMONY_CHANNELS ? HARMONY_TAGS : startTags);
        tags.addAll(endTags);
        tags.addAll(interruptTags);
        for (String tag : tags) {
            candidates.add(tag);
            Matcher m = TOKEN_PIECE.matcher(tag);
            while (m.find()) candidates.add(m.group());
        }
        Map<Integer, String> ids = new HashMap<>(tagTokens);
        for (String c : candidates) {
            int id = lookup.applyAsInt(c);
            if (id >= 0) ids.put(id, c);
        }
        return new ReasoningFilter(mode, startTags, endTags, interruptTags, ids);
    }

    public Mode mode() { return mode; }

    public List<String> startTags() { return startTags; }

    public List<String> endTags() { return endTags; }

    public List<String> interruptTags() { return interruptTags; }

    /** Token id to tag text mapping set by {@link #withTagTokens}. */
    public Map<Integer, String> tagTokens() { return tagTokens; }

    /** A sink forwarding only content to {@code callback}; text released by {@code finish()} carries id -1. */
    public static Sink contentTo(TokenCallback callback) {
        Objects.requireNonNull(callback, "callback");
        return (channel, tokenId, text) -> {
            if (channel == Channel.CONTENT) callback.onToken(tokenId, text);
        };
    }

    /** Starts filtering a generation whose prompt state is unknown (no generation-prompt prefix). */
    public State newState(Sink sink) { return newState(sink, null); }

    /**
     * Starts filtering a generation that continues {@code renderedPrompt}. A prompt whose generation
     * prefix opens a reasoning block (it ends, ignoring trailing whitespace, with a start tag, as the
     * {@code <think>\n} prefill of DeepSeek-R1 and Qwen3.5 templates does) starts in reasoning; a
     * harmony prompt is replayed from its last {@code <|start|>} so a prefilled header or channel
     * carries over.
     */
    public State newState(Sink sink, String renderedPrompt) {
        Objects.requireNonNull(sink, "sink");
        Machine m;
        switch (mode) {
            case THINK_TAGS: m = new ThinkMachine(this, sink, openReasoning(renderedPrompt)); break;
            case HARMONY_CHANNELS: m = new HarmonyMachine(sink, renderedPrompt); break;
            default: m = new PassMachine(sink); break;
        }
        return new State(tagTokens, m);
    }

    /**
     * Whether the prompt's generation prefix leaves reasoning open. Only the prompt tail counts: a
     * start tag quoted earlier in the conversation (a user asking about {@code <think>}, a system
     * prompt describing the format) must not route the whole answer to reasoning.
     */
    private boolean openReasoning(String prompt) {
        if (prompt == null) return false;
        int end = prompt.length();
        while (end > 0 && Character.isWhitespace(prompt.charAt(end - 1))) end--;
        String tail = prompt.substring(0, end);
        for (String t : startTags) {
            if (tail.endsWith(t)) return true;
        }
        return false;
    }

    @Override public String toString() {
        return "ReasoningFilter{" + mode + (mode == Mode.THINK_TAGS ? " " + startTags + " " + endTags + " " + interruptTags : "") + "}";
    }

    // ================================================================= streaming state

    /** Per-generation streaming state. Not thread-safe; feed tokens in order, then call {@link #finish()}. */
    public static final class State implements TokenCallback {
        private final Map<Integer, String> tagTokens;
        private final Machine machine;
        private boolean finished;

        State(Map<Integer, String> tagTokens, Machine machine) {
            this.tagTokens = tagTokens;
            this.machine = machine;
        }

        /** Feeds one generated token. */
        @Override public void onToken(int tokenId, String text) {
            if (finished) throw new IllegalStateException("ReasoningFilter state already finished");
            String tag = tagTokens.get(tokenId);
            String t = tag != null ? tag : text;
            if (t != null && !t.isEmpty()) machine.feed(tokenId, t);
        }

        /** Releases held-back text at the end of generation. Further tokens are rejected. */
        public void finish() {
            if (finished) return;
            finished = true;
            machine.finish();
        }

        /** Whether text currently streams to {@link Channel#REASONING}. */
        public boolean inReasoning() { return machine.inReasoning(); }

        /** Recipient of the current harmony tool call, or null. */
        public String recipient() { return machine.recipient(); }
    }

    /** Mode-specific state machine. */
    private abstract static class Machine {
        private final Sink sink;
        boolean muted;

        Machine(Sink sink) { this.sink = sink; }

        abstract void feed(int tokenId, String text);

        abstract void finish();

        abstract boolean inReasoning();

        String recipient() { return null; }

        final void emit(Channel channel, int tokenId, CharSequence text, int from, int to) {
            if (!muted && to > from) sink.accept(channel, tokenId, text.subSequence(from, to).toString());
        }

        /** Earliest occurrence of any tag at or after {@code from}; ties go to the longer tag. Returns {index, tagIndex} or null. */
        static int[] earliest(CharSequence buf, List<String> tags, int from) {
            String s = buf.toString();
            int bestIdx = -1;
            int bestTag = -1;
            for (int i = 0; i < tags.size(); i++) {
                int idx = s.indexOf(tags.get(i), from);
                if (idx < 0) continue;
                if (bestIdx < 0 || idx < bestIdx || (idx == bestIdx && tags.get(i).length() > tags.get(bestTag).length())) {
                    bestIdx = idx;
                    bestTag = i;
                }
            }
            return bestIdx < 0 ? null : new int[] { bestIdx, bestTag };
        }

        /** Length of the longest suffix of {@code buf} that is a proper prefix of some tag. */
        static int partialSuffix(CharSequence buf, List<String> tags) {
            int best = 0;
            int n = buf.length();
            for (String tag : tags) {
                for (int k = Math.min(tag.length() - 1, n); k > best; k--) {
                    boolean match = true;
                    for (int j = 0; j < k && match; j++) match = buf.charAt(n - k + j) == tag.charAt(j);
                    if (match) {
                        best = k;
                        break;
                    }
                }
            }
            return best;
        }

        static int leadingWhitespace(CharSequence s) {
            int i = 0;
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
            return i;
        }
    }

    private static final class PassMachine extends Machine {
        PassMachine(Sink sink) { super(sink); }

        @Override void feed(int tokenId, String text) { emit(Channel.CONTENT, tokenId, text, 0, text.length()); }

        @Override void finish() {}

        @Override boolean inReasoning() { return false; }
    }

    /** THINK_TAGS: PRELUDE (optional start tag) -> REASONING -> AFTER_END (drop whitespace) -> CONTENT. */
    private static final class ThinkMachine extends Machine {
        private enum Phase { PRELUDE, REASONING, AFTER_END, CONTENT }

        private final List<String> starts;
        private final List<String> closers;
        private final int endCount;
        private final StringBuilder pending = new StringBuilder();
        private Phase phase;

        ThinkMachine(ReasoningFilter f, Sink sink, boolean startInReasoning) {
            super(sink);
            this.starts = f.startTags;
            List<String> c = new ArrayList<>(f.endTags);
            c.addAll(f.interruptTags);
            this.closers = Collections.unmodifiableList(c);
            this.endCount = f.endTags.size();
            this.phase = startInReasoning ? Phase.REASONING : Phase.PRELUDE;
        }

        @Override void feed(int tokenId, String text) {
            if (phase == Phase.CONTENT) {
                emit(Channel.CONTENT, tokenId, text, 0, text.length());
                return;
            }
            pending.append(text);
            process(tokenId);
        }

        private void process(int tokenId) {
            while (true) {
                switch (phase) {
                    case PRELUDE: {
                        int ws = leadingWhitespace(pending);
                        if (ws == pending.length()) return;
                        String rest = pending.substring(ws);
                        String matched = null;
                        boolean partial = false;
                        for (String t : starts) {
                            if (rest.startsWith(t)) matched = t;
                            else if (t.startsWith(rest)) partial = true;
                        }
                        if (matched != null) {
                            pending.delete(0, ws + matched.length());
                            phase = Phase.REASONING;
                        } else if (partial) {
                            return;
                        } else {
                            phase = Phase.CONTENT;
                        }
                        break;
                    }
                    case REASONING: {
                        int[] hit = earliest(pending, closers, 0);
                        if (hit == null) {
                            int keep = partialSuffix(pending, closers);
                            emit(Channel.REASONING, tokenId, pending, 0, pending.length() - keep);
                            pending.delete(0, pending.length() - keep);
                            return;
                        }
                        emit(Channel.REASONING, tokenId, pending, 0, hit[0]);
                        if (hit[1] < endCount) {
                            pending.delete(0, hit[0] + closers.get(hit[1]).length());
                            phase = Phase.AFTER_END;
                        } else {
                            pending.delete(0, hit[0]);
                            phase = Phase.CONTENT;
                        }
                        break;
                    }
                    case AFTER_END: {
                        pending.delete(0, leadingWhitespace(pending));
                        if (pending.length() == 0) return;
                        phase = Phase.CONTENT;
                        break;
                    }
                    default: {
                        emit(Channel.CONTENT, tokenId, pending, 0, pending.length());
                        pending.setLength(0);
                        return;
                    }
                }
            }
        }

        @Override void finish() {
            if (phase == Phase.PRELUDE || phase == Phase.CONTENT) emit(Channel.CONTENT, -1, pending, 0, pending.length());
            else if (phase == Phase.REASONING) emit(Channel.REASONING, -1, pending, 0, pending.length());
            pending.setLength(0);
        }

        @Override boolean inReasoning() { return phase == Phase.REASONING; }
    }

    /**
     * HARMONY_CHANNELS: HEADER (up to {@code <|message|>}) -> BODY (up to {@code <|end|>},
     * {@code <|return|>}, {@code <|call|>}, {@code <|eom|>}, {@code <|eot|>} or a new {@code <|start|>})
     * -> BETWEEN (waits for the next {@code <|start|>} or {@code <|channel|>}).
     */
    private static final class HarmonyMachine extends Machine {
        private enum Phase { HEADER, BODY, BETWEEN }

        /** A header longer than this without {@code <|message|>} means the output is not harmony. */
        private static final int HEADER_LIMIT = 256;
        private static final List<String> BODY_END = Collections.unmodifiableList(
                Arrays.asList(H_END, H_RETURN, H_CALL, M_EOM, M_EOT, H_START));
        /** Muse recipients that are not tool calls (llama.cpp common/parsers/muse-spark.cpp:61-76). */
        private static final String TO_SELF = "self";
        private static final String TO_USER = "user";
        private static final String TO_COMMENTARY = "commentary";
        private static final List<String> NEXT_HEADER = Collections.unmodifiableList(Arrays.asList(H_START, H_CHANNEL));

        private final StringBuilder pending = new StringBuilder();
        private Phase phase = Phase.HEADER;
        private Channel bodyChannel = Channel.CONTENT;
        private String recipient;
        private int primedHeader;

        HarmonyMachine(Sink sink, String prompt) {
            super(sink);
            if (prompt != null) {
                int start = prompt.lastIndexOf(H_START);
                if (start >= 0) {
                    muted = true;
                    pending.append(prompt, start + H_START.length(), prompt.length());
                    process(-1);
                    muted = false;
                    if (phase == Phase.HEADER) primedHeader = pending.length();
                }
            }
        }

        @Override void feed(int tokenId, String text) {
            pending.append(text);
            process(tokenId);
        }

        private void process(int tokenId) {
            while (true) {
                switch (phase) {
                    case HEADER: {
                        int idx = pending.indexOf(H_MESSAGE);
                        if (idx >= 0) {
                            classify(pending.substring(0, idx));
                            pending.delete(0, idx + H_MESSAGE.length());
                            primedHeader = 0;
                            phase = Phase.BODY;
                            break;
                        }
                        if (pending.length() - primedHeader > HEADER_LIMIT) {
                            emit(Channel.CONTENT, tokenId, pending, primedHeader, pending.length());
                            pending.setLength(0);
                            primedHeader = 0;
                            bodyChannel = Channel.CONTENT;
                            recipient = null;
                            phase = Phase.BODY;
                        }
                        return;
                    }
                    case BODY: {
                        int[] hit = earliest(pending, BODY_END, 0);
                        if (hit == null) {
                            int keep = partialSuffix(pending, BODY_END);
                            emit(bodyChannel, tokenId, pending, 0, pending.length() - keep);
                            pending.delete(0, pending.length() - keep);
                            return;
                        }
                        emit(bodyChannel, tokenId, pending, 0, hit[0]);
                        String tag = BODY_END.get(hit[1]);
                        pending.delete(0, hit[0] + tag.length());
                        recipient = null;
                        phase = H_START.equals(tag) ? Phase.HEADER : Phase.BETWEEN;
                        break;
                    }
                    default: {
                        int[] hit = earliest(pending, NEXT_HEADER, 0);
                        if (hit == null) {
                            int keep = partialSuffix(pending, NEXT_HEADER);
                            int upTo = pending.length() - keep;
                            if (leadingWhitespace(pending) < upTo) {
                                emit(Channel.CONTENT, tokenId, pending, 0, upTo);
                                pending.delete(0, upTo);
                            }
                            return;
                        }
                        if (leadingWhitespace(pending) < hit[0]) emit(Channel.CONTENT, tokenId, pending, 0, hit[0]);
                        boolean start = hit[1] == 0;
                        pending.delete(0, start ? hit[0] + H_START.length() : hit[0]);
                        phase = Phase.HEADER;
                        break;
                    }
                }
            }
        }

        /**
         * Classifies a header such as {@code assistant<|channel|>analysis},
         * {@code assistant<|channel|>commentary to=functions.f <|constrain|>json} or the Muse
         * {@code assistant to=self}: the Muse recipient {@code self} is reasoning, {@code user} and
         * {@code commentary} are content, any other recipient except {@code assistant} is a tool call,
         * and otherwise the last channel decides.
         */
        private void classify(String header) {
            String to = word(header, header.lastIndexOf("to="), 3);
            if (TO_SELF.equals(to) || TO_USER.equals(to) || TO_COMMENTARY.equals(to)) {
                recipient = null;
                bodyChannel = TO_SELF.equals(to) ? Channel.REASONING : Channel.CONTENT;
                return;
            }
            recipient = to != null && !to.isEmpty() && !"assistant".equals(to) ? to : null;
            if (recipient != null) {
                bodyChannel = Channel.TOOL_CALL;
                return;
            }
            int ch = header.lastIndexOf(H_CHANNEL);
            String channel = ch < 0 ? "" : word(header, ch, H_CHANNEL.length());
            if ("analysis".equals(channel)) bodyChannel = Channel.REASONING;
            else if (channel.isEmpty() || "final".equals(channel) || "commentary".equals(channel)) bodyChannel = Channel.CONTENT;
            else bodyChannel = Channel.REASONING;
        }

        /** The word starting {@code skip} chars after {@code at}, up to whitespace or '<'; null when {@code at < 0}. */
        private static String word(String s, int at, int skip) {
            if (at < 0) return null;
            int b = at + skip;
            while (b < s.length() && s.charAt(b) == ' ') b++;
            int e = b;
            while (e < s.length() && !Character.isWhitespace(s.charAt(e)) && s.charAt(e) != '<') e++;
            return s.substring(b, e);
        }

        @Override void finish() {
            if (phase == Phase.BODY) {
                emit(bodyChannel, -1, pending, 0, pending.length());
            } else if (phase == Phase.HEADER) {
                String rest = pending.substring(Math.min(primedHeader, pending.length()));
                if (!rest.contains("<|")) emit(Channel.CONTENT, -1, rest, 0, rest.length());
            } else if (leadingWhitespace(pending) < pending.length()) {
                emit(Channel.CONTENT, -1, pending, 0, pending.length());
            }
            pending.setLength(0);
        }

        @Override boolean inReasoning() { return phase == Phase.BODY && bodyChannel == Channel.REASONING; }

        @Override String recipient() { return recipient; }
    }
}
