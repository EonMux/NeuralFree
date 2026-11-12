package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The shared drain of the Anthropic, llama {@code /completion}, Responses, Cohere and scale-out stream loops: the
 * events queued behind an awaited one go out under one flush even though each event's writer flushes, the bytes are
 * those of a flush per event, a queue longer than one drain's bound flushes at least every {@link
 * EventDrain#MAX_EVENTS} events, and outside a drain every flush passes through.
 */
@Timeout(60)
class EventDrainTest {

    private static final int TOKENS = 24;

    private static EngineService service;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** An output stream that records its bytes and the size at each flush. */
    private static final class CountingStream extends ByteArrayOutputStream {
        final List<Integer> flushedAt = new ArrayList<>();

        @Override
        public void flush() {
            flushedAt.add(size());
        }
    }

    @Test
    void aFullyQueuedBatchLeavesInOneFlushWithThePerEventBytes() throws Exception {
        GenerationBatch queued = start(1, TOKENS);
        queued.awaitAll();
        CountingStream drained = new CountingStream();
        EventDrain.HeldFlushes held = new EventDrain.HeldFlushes(drained);
        int events = stream(queued, held, 1);
        assertEquals(List.of(drained.size()), drained.flushedAt, "one flush for every queued event");
        assertEquals(events, drained.toString(StandardCharsets.UTF_8).split("\n\n").length);

        GenerationBatch again = start(1, TOKENS);
        again.awaitAll();
        CountingStream perEvent = new CountingStream();
        streamEachFlushed(again, perEvent);
        assertEquals(perEvent.toString(StandardCharsets.UTF_8), drained.toString(StandardCharsets.UTF_8),
                "draining changes when bytes leave, not which");
        assertEquals(events, perEvent.flushedAt.size());
    }

    @Test
    void aQueueLongerThanOneDrainFlushesAtLeastEveryBound() throws Exception {
        GenerationBatch queued = start(4, 48);
        queued.awaitAll();
        CountingStream out = new CountingStream();
        int events = stream(queued, new EventDrain.HeldFlushes(out), 4);
        assertTrue(events > 2 * EventDrain.MAX_EVENTS, events + " events queued");
        byte[] bytes = out.toByteArray();
        int start = 0;
        for (int end : out.flushedAt) {
            int between = eventsIn(bytes, start, end);
            assertTrue(between <= EventDrain.MAX_EVENTS, between + " events between two flushes");
            start = end;
        }
        assertEquals(bytes.length, start, "the last drain flushed its events");
    }

    @Test
    void outsideADrainEveryFlushPassesThrough() throws IOException {
        CountingStream out = new CountingStream();
        EventDrain.HeldFlushes held = new EventDrain.HeldFlushes(out);
        ServerJson.event(held, "message_start", Map.of("type", "message_start"));
        ServerJson.event(held, "message_stop", Map.of("type", "message_stop"));
        assertEquals(2, out.flushedAt.size());
    }

    /** Streams every event through the drain, one flushing event each; returns the events written. */
    private static int stream(GenerationBatch batch, EventDrain.HeldFlushes held, int choices) throws Exception {
        int[] done = {0};
        int[] written = {0};
        while (done[0] < choices) {
            GenerationBatch.Event first = batch.next(50);
            if (first == null) {
                continue;
            }
            EventDrain.drain(batch, first, held, event -> {
                write(held, event);
                written[0]++;
                if (event instanceof GenerationBatch.Done) {
                    done[0]++;
                }
                return done[0] == choices;
            });
        }
        return written[0];
    }

    private static void streamEachFlushed(GenerationBatch batch, CountingStream out) throws Exception {
        while (true) {
            GenerationBatch.Event event = batch.next(50);
            if (event == null) {
                continue;
            }
            write(out, event);
            if (event instanceof GenerationBatch.Done) {
                return;
            }
        }
    }

    private static void write(java.io.OutputStream out, GenerationBatch.Event event) throws IOException {
        String text = event instanceof GenerationBatch.Text t ? t.text() : "";
        ServerJson.event(out, null, Map.of("choice", event.choice(), "text", text,
                "done", event instanceof GenerationBatch.Done));
    }

    /** The server-sent events (each ends in a blank line) in {@code bytes[from, to)}. */
    private static int eventsIn(byte[] bytes, int from, int to) {
        int events = 0;
        for (int i = from + 1; i < to; i++) {
            if (bytes[i] == '\n' && bytes[i - 1] == '\n') {
                events++;
            }
        }
        return events;
    }

    private static GenerationBatch start(int choices, int tokens) {
        int[] prompt = service.encode("hello there", true, true);
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(tokens).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
        List<GenerationBatch.Job> jobs = new ArrayList<>();
        for (int i = 0; i < choices; i++) {
            jobs.add(new GenerationBatch.Job(prompt, request, null, null, null));
        }
        return GenerationBatch.start(service, jobs, null);
    }
}
