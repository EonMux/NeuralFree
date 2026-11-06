package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.TokenScore;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Writes one OpenAI server-sent-event stream over a {@link GenerationBatch}: the chunks the endpoint's handler
 * produces per event, an error event for a choice that failed, the usage (on the last finish chunk, or in its own
 * {@code choices: []} chunk under {@code stream_options.include_usage}) and the terminal {@code [DONE]}. A write
 * that fails — the client went away — cancels the batch, as does any failure while mapping the output (which the
 * client receives as an error event).
 */
final class StreamWriter {

    /** Handles one event, writing its chunks through the writer. */
    interface Handler {
        void on(GenerationBatch.Event event) throws IOException;
    }

    private final OutputStream out;
    private final GenerationBatch batch;
    private final OpenAiBodies bodies;
    private final String id;
    private final long created;
    private final String object;
    private final List<GenerationResult> results = new ArrayList<>();
    private int remaining;
    private long promptTokens;
    private boolean includeUsage;

    StreamWriter(OutputStream out, GenerationBatch batch, OpenAiBodies bodies, String id, long created,
                 String object) {
        this.out = out;
        this.batch = batch;
        this.bodies = bodies;
        this.id = id;
        this.created = created;
        this.object = object;
        this.remaining = batch.size();
    }

    /** Streams every event through {@code handler} until each choice finished, then the usage and [DONE]. */
    void run(Handler handler, long promptTokens, boolean includeUsage) throws IOException {
        this.promptTokens = promptTokens;
        this.includeUsage = includeUsage;
        try {
            int done = 0;
            while (done < batch.size()) {
                GenerationBatch.Event event = batch.next(50);
                if (event == null) {
                    continue;
                }
                handler.on(event);
                if (event instanceof GenerationBatch.Done) {
                    done++;
                }
            }
            if (includeUsage) {
                Map<String, Object> chunk = bodies.head(id, object, created);
                chunk.put("choices", List.of());
                chunk.put("usage", OpenAiBodies.usage(promptTokens, results));
                ServerJson.event(out, null, chunk);
            }
            ServerJson.data(out, "[DONE]");
        } catch (IOException e) {
            batch.cancel();
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            batch.cancel();
            throw new IOException("stream interrupted", e);
        } catch (RuntimeException e) {
            // the request budget ran out or the output could not be mapped: stop the choices, tell the client
            batch.cancel();
            ServerJson.event(out, null, ServerJson.errorBody(String.valueOf(e.getMessage()), "server_error"));
        }
    }

    /**
     * Records a finished choice: its result, or — when it was refused or failed — an error event and null.
     */
    GenerationResult finished(int choice) throws IOException {
        remaining--;
        String failure = batch.failure(choice);
        if (failure != null) {
            ServerJson.event(out, null, ServerJson.errorBody(failure, "invalid_request_error"));
            return null;
        }
        GenerationResult result = batch.result(choice);
        results.add(result);
        return result;
    }

    /**
     * A completion chunk: {@code {"index", "text", "logprobs", "finish_reason"}}; {@code logprobs} is the token's
     * capture at character {@code offset} of the choice text, or null.
     */
    void completionChunk(int index, String text, TokenScore[] logprobs, int offset, String finish)
            throws IOException {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", index);
        choice.put("text", text);
        choice.put("logprobs", logprobs == null ? null : bodies.completionLogprobs(logprobs, offset));
        choice.put("finish_reason", finish);
        write(choice, finish);
    }

    /** A chat chunk: {@code {"index", "delta", "logprobs"?, "finish_reason"}}. */
    void chatChunk(int index, Map<String, Object> delta, TokenScore[] logprobs, String finish) throws IOException {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", index);
        choice.put("delta", delta);
        if (logprobs != null) {
            choice.put("logprobs", bodies.chatLogprobs(logprobs));
        }
        choice.put("finish_reason", finish);
        write(choice, finish);
    }

    private void write(Map<String, Object> choice, String finish) throws IOException {
        Map<String, Object> chunk = bodies.head(id, object, created);
        chunk.put("choices", List.of(choice));
        if (finish != null && remaining == 0 && !includeUsage) {
            chunk.put("usage", OpenAiBodies.usage(promptTokens, results));
        }
        ServerJson.event(out, null, chunk);
    }
}
