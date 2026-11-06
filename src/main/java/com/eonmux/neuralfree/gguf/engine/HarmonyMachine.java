package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.ReasoningFilter.Channel;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter.Sink;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.engine.ReasoningFilter.H_CALL;
import static com.eonmux.neuralfree.gguf.engine.ReasoningFilter.H_CHANNEL;
import static com.eonmux.neuralfree.gguf.engine.ReasoningFilter.H_END;
import static com.eonmux.neuralfree.gguf.engine.ReasoningFilter.H_MESSAGE;
import static com.eonmux.neuralfree.gguf.engine.ReasoningFilter.H_RETURN;
import static com.eonmux.neuralfree.gguf.engine.ReasoningFilter.H_START;
import static com.eonmux.neuralfree.gguf.engine.ReasoningFilter.M_EOM;
import static com.eonmux.neuralfree.gguf.engine.ReasoningFilter.M_EOT;

/**
 * HARMONY_CHANNELS: HEADER (up to {@code <|message|>}) -> BODY (up to {@code <|end|>},
 * {@code <|return|>}, {@code <|call|>}, {@code <|eom|>}, {@code <|eot|>} or a new {@code <|start|>})
 * -> BETWEEN (waits for the next {@code <|start|>} or {@code <|channel|>}). With {@code reasoningInline}
 * a reasoning body routes to content; a gpt-oss one carries its header from the first
 * {@code <|channel|>} and its terminator with it.
 */
final class HarmonyMachine extends ReasoningMachine {
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
    private final boolean reasoningInline;
    private Phase phase = Phase.HEADER;
    private Channel bodyChannel = Channel.CONTENT;
    private String recipient;
    private int primedHeader;
    /** The current body is reasoning routed inline to content. */
    private boolean inlineBody;
    /** The current inline body's terminator routes to content too (the gpt-oss form). */
    private boolean inlineTerminator;

    HarmonyMachine(Sink sink, String prompt, boolean reasoningInline) {
        super(sink);
        this.reasoningInline = reasoningInline;
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
                        String header = pending.substring(0, idx);
                        classify(header);
                        if (reasoningInline && bodyChannel == Channel.REASONING) inline(tokenId, header);
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
                        inlineBody = false;
                        inlineTerminator = false;
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
                    if (inlineTerminator && !H_START.equals(tag)) emit(Channel.CONTENT, tokenId, tag, 0, tag.length());
                    pending.delete(0, hit[0] + tag.length());
                    recipient = null;
                    inlineBody = false;
                    inlineTerminator = false;
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
            announceToolCall(recipient);
            return;
        }
        int ch = header.lastIndexOf(H_CHANNEL);
        String channel = ch < 0 ? "" : word(header, ch, H_CHANNEL.length());
        if ("analysis".equals(channel)) bodyChannel = Channel.REASONING;
        else if (channel.isEmpty() || "final".equals(channel) || "commentary".equals(channel)) bodyChannel = Channel.CONTENT;
        else bodyChannel = Channel.REASONING;
    }

    /**
     * Routes the reasoning body of {@code header} to content: a gpt-oss header (one with a channel) first
     * releases its text from the first {@code <|channel|>} through {@code <|message|>}, and its terminator
     * later follows the body; a Muse {@code to=self} body goes alone.
     */
    private void inline(int tokenId, String header) {
        bodyChannel = Channel.CONTENT;
        inlineBody = true;
        int ch = header.indexOf(H_CHANNEL);
        inlineTerminator = ch >= 0;
        if (inlineTerminator) {
            String opener = header.substring(ch) + H_MESSAGE;
            emit(Channel.CONTENT, tokenId, opener, 0, opener.length());
        }
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

    @Override boolean inReasoning() {
        return phase == Phase.BODY && (bodyChannel == Channel.REASONING || inlineBody);
    }

    @Override String recipient() { return recipient; }
}
