package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.parsed;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The resumable stream routes over a live registry: {@code GET /v1/stream} replays the recorded bytes from a
 * byte offset and refuses an offset the buffer dropped, {@code DELETE /v1/stream} stops and forgets
 * idempotently, and {@code POST /v1/streams/lookup} answers only the ids the caller named — exactly or as the
 * {@code <id>::<model>} variant — with the session's state fields.
 */
@Timeout(30)
class StreamResourceTest {

    private static StreamSessions sessions;
    private static StreamResource resource;

    @BeforeAll
    static void open() {
        sessions = new StreamSessions(64, 300, 60);
        resource = new StreamResource(sessions);
    }

    @AfterAll
    static void close() {
        sessions.close();
    }

    /** Records {@code frames} into a fresh session and finishes it, as a streamed generation leaves it. */
    private static StreamSession recorded(String id, String... frames) {
        StreamSession session = sessions.createOrReplace(id);
        for (String frame : frames) {
            session.append(frame.getBytes(StandardCharsets.UTF_8));
        }
        session.finish();
        return session;
    }

    @Test
    void theReplayAnswersTheRecordedBytesFromAnOffset() throws IOException {
        recorded("replay", "data: 1\n\n", "data: 2\n\n", "data: [DONE]\n\n");
        assertEquals("data: 1\n\ndata: 2\n\ndata: [DONE]\n\n",
                streamText(resource.stream("replay", null)));
        assertEquals("2\n\ndata: [DONE]\n\n", streamText(resource.stream("replay", "15")));
        assertEquals("data: 1\n\ndata: 2\n\ndata: [DONE]\n\n",
                streamText(resource.stream("replay", " 0 ")));
    }

    @Test
    void theReplayRefusesUnknownAndMissingIdsByName() {
        assertEquals(404, statusOf(() -> resource.stream("absent", null)));
        assertTrue(errorOf(() -> resource.stream("absent", null)).contains("absent"));
        assertEquals(400, statusOf(() -> resource.stream(" ", null)));
        assertEquals(400, statusOf(() -> resource.stream(null, null)));
        assertTrue(errorOf(() -> resource.stream(null, null)).contains("conv_id"));
    }

    @Test
    void theReplayRefusesOffsetsItCannotServeByName() {
        recorded("offsets", "data: x\n\n");
        assertEquals(400, statusOf(() -> resource.stream("offsets", "tomorrow")));
        assertEquals(400, statusOf(() -> resource.stream("offsets", "-1")));
        assertTrue(errorOf(() -> resource.stream("offsets", "-1")).contains("from"));
    }

    @Test
    void anOffsetTheCapDroppedIsRefusedWithARestart() {
        StreamSession dropped = sessions.createOrReplace("dropped");
        dropped.append(new byte[48]);
        dropped.append(new byte[24]);
        assertEquals(72, dropped.totalBytes());
        assertEquals(8, dropped.droppedPrefix(), "the 64-byte cap dropped the front");
        assertEquals(400, statusOf(() -> resource.stream("dropped", "4")));
        assertTrue(errorOf(() -> resource.stream("dropped", "4")).contains("restart"));
        assertEquals(200, statusOf(() -> resource.stream("dropped", "8")));
    }

    @Test
    void theStopForgetsTheSessionAndIsIdempotent() {
        recorded("stopping", "data: x\n\n");
        assertEquals(204, resource.stop("stopping").getStatus());
        assertEquals(404, statusOf(() -> resource.stream("stopping", null)));
        assertEquals(204, resource.stop("stopping").getStatus(), "stopping again still answers 204");
        assertEquals(400, statusOf(() -> resource.stop(null)));
        assertEquals(400, statusOf(() -> resource.stop("")));
        assertTrue(errorOf(() -> resource.stop(null)).contains("conv_id"));
    }

    @Test
    void theStopCancelsTheProducerOfALiveSession() throws Exception {
        StreamSession live = sessions.createOrReplace("live");
        live.append("data: 1\n\n".getBytes(StandardCharsets.UTF_8));
        assertEquals(204, resource.stop("live").getStatus());
        assertTrue(live.isCancelled(), "the producer sees the stop at its next check");
        assertTrue(live.isDone(), "the readers wake");
        assertEquals(404, statusOf(() -> resource.stream("live", null)));
    }

    @Test
    void theLookupAnswersTheStateOfTheNamedSessionsOnly() {
        recorded("known", "data: x\n\n");
        List<?> found = (List<?>) parsed(resource.lookup(
                "{\"conversation_ids\": [\"known\", \"absent\"]}").getEntity());
        assertEquals(1, found.size(), "only the ids the server holds answer");
        JsonView entry = JsonView.of(found.get(0), "entry");
        assertEquals("known", entry.str("conversation_id"));
        assertEquals(Boolean.TRUE, entry.raw("is_done"));
        assertTrue(entry.i64("total_bytes") > 0);
        assertTrue(entry.i64("started_at") > 0);
        assertTrue(entry.i64("completed_at") >= entry.i64("started_at"));
    }

    @Test
    void theLookupMatchesTheModelVariantsOfAnId() {
        recorded("conv::ggmf/tiny:Q8_0", "data: x\n\n");
        List<?> variants = (List<?>) parsed(resource.lookup("{\"conversation_ids\": [\"conv\"]}").getEntity());
        assertEquals(1, variants.size(), "the <id>::<model> variant answers the bare id");
        assertEquals("conv::ggmf/tiny:Q8_0",
                JsonView.of(variants.get(0), "entry").str("conversation_id"));
        List<?> byExact = (List<?>) parsed(resource.lookup(
                "{\"conversation_ids\": [\"conv::ggmf/tiny:Q8_0\"]}").getEntity());
        assertEquals(1, byExact.size());
        assertEquals(0, ((List<?>) parsed(resource.lookup(
                "{\"conversation_ids\": [\"conv::ggmf/other\"]}").getEntity())).size(),
                "a different variant is no match");
    }

    @Test
    void theLookupAnswersAnEmptyArrayWithoutMatches() {
        assertEquals("[]", String.valueOf(resource.lookup(
                "{\"conversation_ids\": [\"never\", \"here\"]}").getEntity()));
        assertEquals("[]", String.valueOf(resource.lookup("{\"conversation_ids\": []}").getEntity()));
    }

    @Test
    void theLookupRefusesAMalformedBodyByName() {
        assertEquals(400, statusOf(() -> resource.lookup("not json")));
        assertEquals(400, statusOf(() -> resource.lookup("{}")));
        assertTrue(errorOf(() -> resource.lookup("{}")).contains("conversation_ids"));
        assertEquals(400, statusOf(() -> resource.lookup("{\"conversation_ids\": \"known\"}")));
        assertEquals(400, statusOf(() -> resource.lookup("{\"conversation_ids\": [\"known\", 3]}")));
        assertTrue(errorOf(() -> resource.lookup("{\"conversation_ids\": [\"known\", 3]}"))
                .contains("conversation_ids[1]"));
    }
}
