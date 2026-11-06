package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.TokenScore;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;

/**
 * Writes one OpenAI server-sent-event stream over a {@link GenerationBatch}: the chunks the endpoint's handler
 * produces per event, an error event for a choice that failed, the usage (on the last finish chunk, in its own
 * {@code choices: []} chunk under {@code stream_options.include_usage}, or on every chunk under
 * {@code stream_options.continuous_usage_stats}) and the terminal {@code [DONE]}. With
 * {@code return_token_ids} the completion chunks carry their token's id and the first chunk of a choice the
 * prompt's ids; chat chunks carry {@code token_ids} on the choice. The finish chunk of a choice names its
 * {@code stop_reason}. A write that fails — the client went away — cancels the batch, as does any failure while
 * mapping the output (which the client receives as an error event).
 */
final class StreamWriter {

    /** Handles one event, writing its chunks through the writer. */
    interface Handler {
        void on(GenerationBatch.Event event) throws IOException;
    }

    /** The response-shaping flags one stream carries. */
    record Flags(boolean includeUsage, boolean continuousUsage, boolean returnTokenIds) { }

    private final OutputStream out;
    private final GenerationBatch batch;
    private final OpenAiBodies bodies;
    private final String id;
    private final long created;
    private final String object;
    private final List<GenerationResult> results = new ArrayList<>();
    private final long promptTokens;
    private final Flags flags;
    private int remaining;

    /**
     * A writer of {@code object} chunks over {@code batch}; {@code promptTokens} is the usage's prompt count and
     * {@code flags} shape every chunk, including those written before {@link #run}.
     */
    StreamWriter(OutputStream out, GenerationBatch batch, OpenAiBodies bodies, String id, long created,
                 String object, long promptTokens, Flags flags) {
        this.out = out;
        this.batch = batch;
        this.bodies = bodies;
        this.id = id;
        this.created = created;
        this.object = object;
        this.promptTokens = promptTokens;
        this.flags = flags;
        this.remaining = batch.size();
    }

    /** Streams every event through {@code handler} until each choice finished, then the usage and [DONE]. */
    void run(Handler handler) throws IOException {
        run(handler, null);
    }

    /**
     * The same as {@link #run(Handler)}, stopping without the usage and [DONE] when {@code stop} returns true —
     * a resumable stream's explicit stop ({@code DELETE /v1/stream}), which cancels the batch as llama.cpp's
     * reader cancels its unfinished tasks, so no slot keeps generating output nobody will read.
     */
    void run(Handler handler, BooleanSupplier stop) throws IOException {
        try {
            int done = 0;
            while (done < batch.size()) {
                if (stop != null && stop.getAsBoolean()) {
                    batch.cancel();
                    return;
                }
                GenerationBatch.Event event = batch.next(50);
                if (event == null) {
                    continue;
                }
                handler.on(event);
                if (event instanceof GenerationBatch.Done) {
                    done++;
                }
            }
            if (flags.includeUsage()) {
                Map<String, Object> chunk = bodies.head(id, object, created);
                chunk.put("choices", List.of());
                chunk.put("usage", OpenAiBodies.usage(promptTokens, cachedTokens(), results));
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

    /** The prompt tokens the prefix cache served ({@code usage.prompt_tokens_details.cached_tokens}). */
    private long cachedTokens() {
        long sum = 0;
        for (int i = 0; i < batch.size(); i++) {
            sum += batch.cachedTokens(i);
        }
        return sum;
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
     * capture at character {@code offset} of the choice text, or null. With {@code return_token_ids} the chunk
     * carries the token's id under {@code token_ids}, and a {@code first} chunk the prompt's ids under
     * {@code prompt_token_ids}.
     */
    void completionChunk(int index, String text, TokenScore[] logprobs, int offset, String finish, Integer tokenId,
                         boolean first) throws IOException {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", index);
        choice.put("text", text);
        choice.put("logprobs", logprobs == null ? null : bodies.completionLogprobs(logprobs, offset));
        choice.put("finish_reason", finish);
        if (flags.returnTokenIds()) {
            if (first) {
                choice.put("prompt_token_ids", toList(batch.promptTokenIds(index)));
            }
            if (tokenId != null) {
                choice.put("token_ids", List.of(tokenId));
            }
        }
        write(choice, finish);
    }

    /**
     * The echo chunk of an {@code echo} stream whose {@code logprobs} body is prebuilt with the merged prompt
     * entries; a {@code first} chunk carries the prompt's ids under {@code return_token_ids}.
     */
    void echoChunk(int index, String text, Map<String, Object> logprobs, boolean first) throws IOException {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", index);
        choice.put("text", text);
        choice.put("logprobs", logprobs);
        choice.put("finish_reason", null);
        if (flags.returnTokenIds() && first) {
            choice.put("prompt_token_ids", toList(batch.promptTokenIds(index)));
        }
        write(choice, null);
    }

    /** A finish completion chunk naming {@code stopReason} (a string, an integer token id, or null). */
    void completionFinish(int index, String finish, Object stopReason, boolean first) throws IOException {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", index);
        choice.put("text", "");
        choice.put("logprobs", null);
        choice.put("finish_reason", finish);
        choice.put("stop_reason", stopReason);
        if (flags.returnTokenIds() && first) {
            choice.put("prompt_token_ids", toList(batch.promptTokenIds(index)));
        }
        write(choice, finish);
    }

    /** A chat chunk: {@code {"index", "delta", "logprobs"?, "finish_reason"}}. */
    void chatChunk(int index, Map<String, Object> delta, TokenScore[] logprobs, String finish) throws IOException {
        chatChunk(index, delta, logprobs, finish, null, null, null);
    }

    /**
     * A chat chunk with {@code return_token_ids}: {@code token_ids} on the choice when the chunk's delta carries
     * one token, {@code prompt_token_ids} at the chunk level when {@code promptTokenIds} is given (the stream's
     * first chunk only).
     */
    void chatChunk(int index, Map<String, Object> delta, TokenScore[] logprobs, String finish, Integer tokenId,
                   int[] promptTokenIds) throws IOException {
        chatChunk(index, delta, logprobs, finish, tokenId, promptTokenIds, null);
    }

    /** A chat chunk whose finish chunk names {@code stopReason} (a string, an integer token id, or null). */
    void chatChunk(int index, Map<String, Object> delta, TokenScore[] logprobs, String finish, Integer tokenId,
                   int[] promptTokenIds, Object stopReason) throws IOException {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", index);
        choice.put("delta", delta);
        if (logprobs != null) {
            choice.put("logprobs", bodies.chatLogprobs(logprobs));
        }
        choice.put("finish_reason", finish);
        if (finish != null) {
            choice.put("stop_reason", stopReason);
        }
        if (flags.returnTokenIds() && tokenId != null) {
            choice.put("token_ids", List.of(tokenId));
        }
        Map<String, Object> top = flags.returnTokenIds() && promptTokenIds != null ? new LinkedHashMap<>() : null;
        if (top != null) {
            top.put("prompt_token_ids", toList(promptTokenIds));
        }
        write(choice, finish, top);
    }

    private static List<Integer> toList(int[] ids) {
        List<Integer> out = new ArrayList<>(ids.length);
        for (int id : ids) {
            out.add(id);
        }
        return out;
    }

    private void write(Map<String, Object> choice, String finish) throws IOException {
        write(choice, finish, null);
    }

    private void write(Map<String, Object> choice, String finish, Map<String, Object> top) throws IOException {
        Map<String, Object> chunk = top == null ? bodies.head(id, object, created) : top;
        chunk.putIfAbsent("id", id);
        chunk.putIfAbsent("object", object);
        chunk.putIfAbsent("created", created);
        chunk.putIfAbsent("model", bodies.model());
        chunk.put("choices", List.of(choice));
        if (flags.continuousUsage() || (finish != null && remaining == 0 && !flags.includeUsage())) {
            chunk.put("usage", OpenAiBodies.usage(promptTokens, cachedTokens(), results));
        }
        if (finish != null) {
            Object index = choice.get("index");
            Map<String, Object> transfer = index instanceof Number ? batch.kvTransferParams(((Number) index).intValue())
                    : null;
            if (transfer != null) {
                chunk.put("kv_transfer_params", transfer);
            }
        }
        ServerJson.event(out, null, chunk);
    }
}
