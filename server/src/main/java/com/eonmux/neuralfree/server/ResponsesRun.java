package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Drives one Responses generation: the batch's events through a {@link ChatChoice} (reasoning, content and the tool
 * parser) into {@link ResponsesEvents}, to an SSE stream, to a stored response's log, or only to the final response.
 * Also builds the conversation a later {@code previous_response_id} continues: the request's messages and the output
 * as chat messages. Immutable; each drive builds its own choice and events.
 */
final class ResponsesRun {

    private static final Logger LOG = Logger.getLogger(ResponsesRun.class.getName());

    private final ChatPlan plan;
    private final Map<String, Object> head;
    private final OpenAiBodies bodies;
    private final boolean logprobs;
    private final List<Map<String, Object>> asked;

    /**
     * @param plan     the chat plan of the converted request
     * @param head     the response's fixed fields
     * @param bodies   the logprobs rendering
     * @param logprobs whether the output text carries logprobs
     * @param asked    the request's chat messages
     */
    ResponsesRun(ChatPlan plan, Map<String, Object> head, OpenAiBodies bodies, boolean logprobs,
                 List<Map<String, Object>> asked) {
        this.plan = plan;
        this.head = head;
        this.bodies = bodies;
        this.logprobs = logprobs;
        // the chat path validated every message; a copy keeps the request's list out of reach
        this.asked = Collections.unmodifiableList(new ArrayList<>(asked));
    }

    /** Streams the response to {@code sink}; a broken client stream cancels the generation. */
    Map<String, Object> stream(GenerationBatch batch, ResponsesEvents.Sink sink) throws IOException {
        try {
            return drive(batch, sink, false);
        } catch (IOException e) {
            batch.cancel();
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("stream interrupted", e);
        }
    }

    /** The final response of a finished batch, its events dropped. */
    Map<String, Object> collect(GenerationBatch batch) {
        try {
            return drive(batch, (type, data) -> { }, false);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "request interrupted", "unavailable_error");
        }
    }

    /** Runs a background response into {@code sink}, which does not fail; any error ends it {@code failed}. */
    Map<String, Object> detached(GenerationBatch batch, ResponsesEvents.Sink sink) {
        try {
            return drive(batch, sink, true);
        } catch (IOException | InterruptedException e) {
            // the store's sink neither fails nor waits, and drive answers an interruption itself when detached
            throw new IllegalStateException("background response " + head.get("id") + " ended abnormally", e);
        }
    }

    /** The request's messages followed by the output of {@code response}, as chat messages. */
    List<Map<String, Object>> conversation(Map<String, Object> response) {
        List<Map<String, Object>> out = new ArrayList<>(asked);
        if (response.get("output") instanceof List<?> output) {
            out.addAll(ResponsesRequests.messagesOf(output));
        }
        return out;
    }

    /**
     * Drives the batch into one {@link ResponsesEvents}: a failure of the engine, the budget or the output mapping
     * cancels the generation and ends that same response {@code failed}, keeping its sequence numbers and the items
     * already produced. An interruption ends a detached response failed and propagates otherwise.
     */
    private Map<String, Object> drive(GenerationBatch batch, ResponsesEvents.Sink sink, boolean detached)
            throws IOException, InterruptedException {
        ChatChoice choice = new ChatChoice(plan.routing().newOutput());
        ResponsesEvents events = new ResponsesEvents(head, sink, bodies, logprobs);
        events.start();
        try {
            while (true) {
                GenerationBatch.Event event = batch.next(50);
                if (event == null) {
                    continue;
                }
                if (event instanceof GenerationBatch.Done) {
                    break;
                }
                choice.accept(event);
                events.accept(choice.drain());
            }
            ToolParse parse = choice.finish();
            events.accept(choice.drain());
            String failure = batch.failure(0);
            if (failure != null) {
                return events.fail(failure);
            }
            return events.finish(batch.result(0), plan.tokens().length, ResponsesResource.cached(batch), parse);
        } catch (InterruptedException e) {
            batch.cancel();
            if (!detached) {
                throw e;
            }
            Thread.currentThread().interrupt();
            return events.fail("the background response was interrupted");
        } catch (RuntimeException e) {
            // the request outlived its budget (IllegalStateException from next), or the output could not be mapped
            batch.cancel();
            if (!(e instanceof IllegalStateException)) {
                LOG.log(Level.WARNING, "response " + head.get("id") + " failed", e);
            }
            return events.fail(String.valueOf(e.getMessage()));
        }
    }
}
