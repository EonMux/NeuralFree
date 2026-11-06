package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.DELETE;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The resumable stream routes (llama.cpp's {@code server_register_stream_routes}), over the sessions a
 * streamed generation records into when its request carries {@code X-Conversation-Id}:
 *
 * <ul>
 *   <li>{@code GET /v1/stream?conv_id=<id>&from=N} replays the recorded server-sent-event bytes from byte
 *       {@code N} (0 without it) and blocks for live bytes until the session finishes — the reattachment of a
 *       client that lost its connection mid-generation.</li>
 *   <li>{@code DELETE /v1/stream?conv_id=<id>} is the explicit stop: it cancels the session's producer, so it
 *       stops pulling output and cancels the generation, forgets the buffer and answers
 *       204, idempotently.</li>
 *   <li>{@code POST /v1/streams/lookup} with {@code {"conversation_ids": [...]}} answers the sessions that
 *       match an id — the exact one and any {@code <id>::<model>} variant — as
 *       {@code {conversation_id, is_done, total_bytes, started_at, completed_at}}. Only ids the caller already
 *       knows are answered, so the route never enumerates live sessions.</li>
 * </ul>
 *
 * <p>The conversation id travels in the {@code conv_id} query string (not the path) because it can embed a
 * model name containing slashes, which the path would split. An unknown id is a 404, a missing one a 400, and
 * a {@code from} below the buffer's dropped prefix a 400 telling the client to restart.</p>
 */
@Path("/")
public class StreamResource {

    @Inject
    StreamSessions sessions;

    /** Test and manual wiring over an explicit session registry. */
    public StreamResource(StreamSessions sessions) {
        this.sessions = sessions;
    }

    public StreamResource() {
    }

    @GET
    @Path("/v1/stream")
    @Produces("text/event-stream")
    public Response stream(@QueryParam("conv_id") String convId, @QueryParam("from") String from) {
        StreamSession session = session(convId);
        long offset = offset(from, session);
        StreamingOutput replay = out -> {
            try {
                // an OFFSET_LOST mid-read ends the stream: the buffer cap overtook a slow reader, as
                // llama.cpp's reader ends there too
                session.readFrom(offset, chunk -> {
                    out.write(chunk);
                    return true;
                });
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("stream replay interrupted", e);
            }
        };
        return Response.ok(replay).type("text/event-stream").build();
    }

    @DELETE
    @Path("/v1/stream")
    public Response stop(@QueryParam("conv_id") String convId) {
        requireConvId(convId);
        sessions.evictAndCancel(convId);
        return Response.noContent().build();
    }

    @POST
    @Path("/v1/streams/lookup")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response lookup(String bodyText) {
        return OpenAiResource.mapped(() -> lookupOf(ServerJson.parse(bodyText)));
    }

    /** The matching sessions of the request's ids, in request order, as a bare JSON array. */
    private Response lookupOf(JsonView body) {
        List<String> requested = body.strList("conversation_ids");
        List<Object> matches = new ArrayList<>();
        for (String id : requested) {
            for (StreamSession session : sessions.all()) {
                if (matches(session.conversationId(), id)) {
                    matches.add(entry(session));
                }
            }
        }
        return Response.ok(ServerJson.write(matches)).build();
    }

    /** The session of {@code convId}, refused when the id is missing or names no session. */
    private StreamSession session(String convId) {
        requireConvId(convId);
        StreamSession session = sessions.get(convId);
        if (session == null) {
            throw ServerJson.error(404, "no stream session for conv_id '" + convId + "': it expired, was"
                    + " stopped, or never existed", "invalid_request_error");
        }
        return session;
    }

    private static void requireConvId(String convId) {
        if (convId == null || convId.isBlank()) {
            throw ServerJson.badRequest("conv_id is required: the conversation id of the stream session");
        }
    }

    /** The replay offset: 0 without {@code from}, refused when it does not parse or sits below the buffer. */
    private static long offset(String from, StreamSession session) {
        if (from == null || from.isBlank()) {
            return 0;
        }
        long offset;
        try {
            offset = Long.parseLong(from.trim());
        } catch (NumberFormatException e) {
            throw ServerJson.badRequest("from must be a byte offset into the stream, got '" + from.trim() + "'");
        }
        if (offset < 0) {
            throw ServerJson.badRequest("from must be 0 or more, got " + offset);
        }
        long dropped = session.droppedPrefix();
        if (offset < dropped) {
            throw ServerJson.badRequest("from=" + offset + " fell below the dropped prefix (" + dropped
                    + "): stream offset lost, please restart from 0");
        }
        return offset;
    }

    /** llama.cpp's lookup matching: the exact id and any {@code <id>::<model>} per-model variant of it. */
    private static boolean matches(String conversationId, String requested) {
        return conversationId.equals(requested) || conversationId.startsWith(requested + "::");
    }

    private static Map<String, Object> entry(StreamSession session) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("conversation_id", session.conversationId());
        out.put("is_done", session.isDone());
        out.put("total_bytes", session.totalBytes());
        out.put("started_at", session.startedAt());
        out.put("completed_at", session.completedAt());
        return out;
    }
}
