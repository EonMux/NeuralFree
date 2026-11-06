package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Parses one chat completion server-sent-event stream, forwarding every non-empty {@code choices[0].delta.content}
 * run to a delta consumer and accumulating the answer text. A chat stream's {@code [DONE]}, usage and empty role
 * chunks pass without a delta; a malformed frame or an error frame fails {@link #finish()}, and a consumer whose
 * write fails fails the write at once so the chat source cancels. The stream writer calls this class as its
 * {@link OutputStream}, so the completions resource streams straight into the speech-to-text mapping.
 */
final class SseChatText extends OutputStream {

    /** Receives one text delta in stream order. */
    @FunctionalInterface
    interface Delta {
        void accept(String text);
    }

    private final ByteArrayOutputStream line = new ByteArrayOutputStream();
    private final Delta sink;
    private final StringBuilder text = new StringBuilder();
    private String finishReason;
    private RuntimeException failure;

    SseChatText(Delta sink) {
        this.sink = sink;
    }

    @Override
    public void write(int b) throws IOException {
        if (b == '\n') {
            flushLine();
        } else {
            line.write(b);
        }
    }

    @Override
    public void write(byte[] bytes, int offset, int length) throws IOException {
        for (int i = 0; i < length; i++) {
            write(bytes[offset + i]);
        }
    }

    /** The answer text the deltas accumulated. */
    String text() {
        return text.toString();
    }

    /** The last finish reason the chat stream reported, or null when it reported none. */
    String finishReason() {
        return finishReason;
    }

    /** Processes the last buffered line and rethrows the first failure, if any. */
    void finish() throws IOException {
        flushLine();
        if (failure != null) {
            throw new IOException(failure.getMessage(), failure);
        }
    }

    private void flushLine() throws IOException {
        String raw = line.toString(StandardCharsets.UTF_8).stripTrailing();
        line.reset();
        if (failure != null || !raw.startsWith("data:")) {
            return;
        }
        String payload = raw.substring("data:".length()).strip();
        if (payload.isEmpty() || payload.equals("[DONE]")) {
            return;
        }
        try {
            JsonView chunk = ServerJson.parse(payload);
            if (chunk.has("error")) {
                failure = new IllegalArgumentException(String.valueOf(
                        chunk.obj("error").optStr("message", "the chat stream failed")));
                return;
            }
            Object rawChoices = chunk.raw("choices");
            if (!(rawChoices instanceof List<?> choices) || choices.isEmpty()) {
                return;
            }
            JsonView choice = JsonView.of(choices.get(0), "chunk.choices[0]");
            String finish = choice.optStr("finish_reason", null);
            if (finish != null) {
                finishReason = finish;
            }
            String delta = choice.optObj("delta").optStr("content", "");
            if (delta.isEmpty()) {
                return;
            }
            sink.accept(delta);
            text.append(delta);
        } catch (UncheckedIOException e) {
            // the consumer's write failed: fail this stream at once so the chat source cancels its batch
            if (e.getCause() instanceof IOException io) {
                throw io;
            }
            throw e;
        } catch (RuntimeException e) {
            failure = e;
        }
    }
}
