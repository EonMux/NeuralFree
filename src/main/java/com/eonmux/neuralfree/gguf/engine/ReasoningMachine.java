package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.ReasoningFilter.Channel;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter.Sink;

import java.util.List;

/**
 * The mode-specific state machine of a {@link ReasoningFilter} state: it routes each token's text to a channel of the
 * filter's {@link Sink}, and holds what may still turn out to be a tag.
 */
abstract class ReasoningMachine {
    private final Sink sink;
    boolean muted;

    ReasoningMachine(Sink sink) { this.sink = sink; }

    abstract void feed(int tokenId, String text);

    abstract void finish();

    abstract boolean inReasoning();

    /** Whether reasoning text is held back as a possible tag; tags of special tokens never hold. */
    boolean holding() { return false; }

    String recipient() { return null; }

    final void emit(Channel channel, int tokenId, CharSequence text, int from, int to) {
        if (!muted && to > from) sink.accept(channel, tokenId, text.subSequence(from, to).toString());
    }

    final void announceToolCall(String recipient) {
        if (!muted) sink.toolCallStarted(recipient);
    }

    final void announceContentFallback() {
        if (!muted) sink.contentFallback();
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
