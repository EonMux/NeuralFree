package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;

import java.io.PrintStream;
import java.util.Locale;

/**
 * Prints routed generation output: content as it streams, and with {@code showAll} the reasoning and tool-call
 * channels under a header at each channel change. Collects the content and the reasoning text of the turn for
 * the chat history and the transcript; a turn whose reasoning came again as its content (a parser's content
 * fallback) keeps it as the content only, as a non-streamed message does.
 */
final class ChannelPrinter implements ReasoningFilter.Sink {

    private final PrintStream out;
    private final boolean showAll;
    private final boolean trace;
    private final StringBuilder content = new StringBuilder();
    private final StringBuilder reasoning = new StringBuilder();
    private ReasoningFilter.Channel current = ReasoningFilter.Channel.CONTENT;
    private int pieces;
    private boolean fallback;

    /** A printer to {@code out}; {@code trace} also logs every piece to standard error. */
    ChannelPrinter(PrintStream out, boolean showAll, boolean trace) {
        if (out == null) {
            throw new IllegalArgumentException("out is required");
        }
        this.out = out;
        this.showAll = showAll;
        this.trace = trace;
    }

    @Override
    public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
        pieces++;
        if (trace) {
            System.err.println("[GEN] piece=" + pieces + " channel=" + channel + " id=" + tokenId + " text='"
                    + text.replace("\n", "\\n") + "'");
        }
        if (channel == ReasoningFilter.Channel.CONTENT) {
            content.append(text);
        } else {
            if (channel == ReasoningFilter.Channel.REASONING) {
                reasoning.append(text);
            }
            if (!showAll) {
                return;
            }
        }
        if (channel != current) {
            out.print("\n[" + channel.name().toLowerCase(Locale.ROOT).replace('_', ' ') + "]\n");
            current = channel;
        }
        out.print(text);
        out.flush();
    }

    @Override
    public void contentFallback() {
        fallback = true;
    }

    /** The content-channel text of the turn. */
    String content() {
        return content.toString();
    }

    /** The reasoning-channel text of the turn, empty when the model gave none or it became the content. */
    String reasoning() {
        return fallback ? "" : reasoning.toString();
    }
}
