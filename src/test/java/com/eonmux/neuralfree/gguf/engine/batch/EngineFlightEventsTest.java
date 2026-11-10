package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's Flight Recorder events: every step that evaluated rows commits one {@link EngineStepEvent}
 * whose rows and tokens add up to the engine's counters, and every request's first admission one
 * {@link EngineAdmitEvent}.
 */
public class EngineFlightEventsTest {

    private static final int N_TOKENS = 12;
    private static final int MAX_NEW = 5;

    @Test
    public void theStepAndAdmitEventsAccountForTheRun() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(MAX_NEW).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
        Path dump = Files.createTempFile("engine-events", ".jfr");
        BatchStats stats;
        try (GGUFModel loaded = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(loaded, paged);
             Recording recording = new Recording()) {
            recording.enable(EngineStepEvent.NAME).withThreshold(Duration.ZERO);
            recording.enable(EngineAdmitEvent.NAME).withThreshold(Duration.ZERO);
            recording.start();
            for (int r = 0; r < 2; r++) {
                int[] tokens = new int[N_TOKENS];
                for (int i = 0; i < tokens.length; i++) {
                    tokens[i] = 4 + (r * 31 + i * 7) % 200;
                }
                engine.submit(tokens, request, (TokenCallback) null);
            }
            engine.drain();
            recording.stop();
            recording.dump(dump);
            stats = engine.stats();
        }
        List<RecordedEvent> steps = new ArrayList<>();
        List<RecordedEvent> admits = new ArrayList<>();
        try {
            for (RecordedEvent e : RecordingFile.readAllEvents(dump)) {
                String name = e.getEventType().getName();
                if (name.equals(EngineStepEvent.NAME)) {
                    steps.add(e);
                } else if (name.equals(EngineAdmitEvent.NAME)) {
                    admits.add(e);
                }
            }
        } finally {
            Files.deleteIfExists(dump);
        }
        long decodeRows = 0;
        long prefillRows = 0;
        long generated = 0;
        long hostSample = 0;
        for (RecordedEvent e : steps) {
            long rows = e.getLong("decodeRows") + e.getLong("prefillRows");
            assertTrue("a committed step evaluated rows", rows > 0);
            decodeRows += e.getLong("decodeRows");
            prefillRows += e.getLong("prefillRows");
            generated += e.getLong("generatedTokens");
            hostSample += e.getDuration("hostSampleNanos").toNanos();
            assertTrue("the step's duration covers its sampling",
                    e.getDuration().toNanos() >= e.getDuration("hostSampleNanos").toNanos());
        }
        assertEquals("the steps carry every decode row", stats.rowsPerDecode() * stats.decodeSteps(), decodeRows,
                1e-6);
        assertEquals("the steps carry every prefilled row", stats.promptTokens() + stats.recomputedTokens(),
                prefillRows);
        assertEquals("the steps carry every sampled token", 2L * MAX_NEW, generated);
        assertEquals(stats.generatedTokens(), generated);
        assertEquals(Math.round(stats.hostSampleSeconds() * 1e9), hostSample, 1.0);
        assertEquals("one admission per request", 2, admits.size());
        for (RecordedEvent e : admits) {
            assertTrue(e.getInt("seq") >= 0);
            assertTrue(e.getDuration("queueNanos").toNanos() >= 0);
        }
        RecordedEvent last = steps.get(steps.size() - 1);
        assertEquals("nothing runs after the last step", 0, last.getInt("running"));
    }
}
