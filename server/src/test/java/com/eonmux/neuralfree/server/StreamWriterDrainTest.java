package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
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
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The OpenAI stream writer drains the events queued behind the one it waited for and flushes once per drain: a
 * batch whose events are all queued before the writer runs streams in two flushes (the drained chunks, then the
 * usage and {@code [DONE]}), and its bytes equal those of the same greedy generation streamed live, chunk by chunk;
 * a queue longer than one drain's bound flushes at least every {@link StreamWriter#MAX_DRAIN_EVENTS} events.
 */
@Timeout(60)
class StreamWriterDrainTest {

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

    /** An output stream that records its bytes, counts its flushes and the bytes written at each. */
    private static final class CountingStream extends ByteArrayOutputStream {
        int flushes;
        final List<Integer> flushedAt = new java.util.ArrayList<>();

        @Override
        public void flush() {
            flushes++;
            flushedAt.add(size());
        }
    }

    @Test
    void aFullyQueuedBatchStreamsInOneDrainWithTheLiveBytes() throws Exception {
        GenerationBatch queued = start();
        queued.awaitAll();
        CountingStream drained = new CountingStream();
        stream(queued, drained);
        assertEquals(2, drained.flushes, "one flush for the drained chunks, one for the usage and [DONE]");

        GenerationBatch live = start();
        CountingStream streamed = new CountingStream();
        stream(live, streamed);
        String text = streamed.toString(StandardCharsets.UTF_8);
        // the usage chunk differs by the prompt tokens the second run found in the prefix cache
        assertEquals(beforeUsage(text), beforeUsage(drained.toString(StandardCharsets.UTF_8)),
                "draining changes when bytes leave, not which");
        assertTrue(text.endsWith("data: [DONE]\n\n"), text);
        assertTrue(text.contains("\"finish_reason\":\"length\""), text);
        assertTrue(text.split("\n\n").length > TOKENS / 2, "the chunks of the generated tokens: " + text);
    }

    @Test
    void aQueueLongerThanOneDrainFlushesAtLeastEveryBoundOfEvents() throws Exception {
        int choices = 4;
        int[] prompt = service.encode("hello there", true, true);
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(48).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
        List<GenerationBatch.Job> jobs = new java.util.ArrayList<>();
        for (int i = 0; i < choices; i++) {
            jobs.add(new GenerationBatch.Job(prompt, request, null, null, null));
        }
        GenerationBatch queued = GenerationBatch.start(service, jobs, null);
        queued.awaitAll();
        CountingStream out = new CountingStream();
        stream(queued, out);
        byte[] bytes = out.toByteArray();
        int events = eventsIn(bytes, 0, bytes.length);
        assertTrue(events > 2 * StreamWriter.MAX_DRAIN_EVENTS, events + " events queued");
        int start = 0;
        for (int end : out.flushedAt) {
            int between = eventsIn(bytes, start, end);
            assertTrue(between <= StreamWriter.MAX_DRAIN_EVENTS + 2,
                    between + " events between two flushes, past the drain bound");
            start = end;
        }
        assertTrue(out.flushes >= events / StreamWriter.MAX_DRAIN_EVENTS, out.flushes + " flushes of " + events);
    }

    @Test
    void aStopEndsTheDrainAtTheNextEvent() throws Exception {
        GenerationBatch queued = start();
        queued.awaitAll();
        CountingStream out = new CountingStream();
        StreamWriter writer = writer(queued, out);
        int[] handled = {0};
        writer.run(event -> {
            handled[0]++;
            writer.completionChunk(0, "x", null, 0, null, null, false);
        }, () -> handled[0] >= 3);
        assertEquals(3, handled[0], "the stop is checked after every drained event");
        assertEquals(1, out.flushes);
        assertEquals(3, out.toString(StandardCharsets.UTF_8).split("\n\n").length);
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

    private static String beforeUsage(String stream) {
        int usage = stream.indexOf("\"choices\":[],\"usage\"");
        assertTrue(usage > 0, stream);
        return stream.substring(0, usage);
    }

    private static GenerationBatch start() {
        int[] prompt = service.encode("hello there", true, true);
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(TOKENS).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
        return GenerationBatch.start(service, List.of(new GenerationBatch.Job(prompt, request, null, null, null)),
                null);
    }

    private static StreamWriter writer(GenerationBatch batch, CountingStream out) {
        return new StreamWriter(out, batch, new OpenAiBodies(service, 0, "m"), "cmpl-1", 1700000000L,
                "text_completion", 3, new StreamWriter.Flags(true, false, false));
    }

    private static void stream(GenerationBatch batch, CountingStream out) throws IOException {
        StreamWriter writer = writer(batch, out);
        writer.run(event -> {
            if (event instanceof GenerationBatch.Text t) {
                writer.completionChunk(t.choice(), t.text(), null, 0, null, t.tokenId(), false);
            } else if (event instanceof GenerationBatch.Done d) {
                GenerationResult r = writer.finished(d.choice());
                writer.completionFinish(d.choice(), OpenAiBodies.finishReason(r.stopReason()),
                        OpenAiBodies.stopReason(r), false);
            }
        });
    }
}
