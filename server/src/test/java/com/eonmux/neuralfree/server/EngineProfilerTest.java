package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The profile window: one at a time, a JFR recording per window, and the report's counter deltas. */
public class EngineProfilerTest {

    @Test
    void oneWindowAtATime(@TempDir Path dir) {
        EngineProfiler profiler = new EngineProfiler(dir);
        assertFalse(profiler.active());
        profiler.start(BatchStats.EMPTY);
        assertTrue(profiler.active());
        assertThrows(IllegalStateException.class, () -> profiler.start(BatchStats.EMPTY));
        profiler.stop(BatchStats.EMPTY);
        assertFalse(profiler.active());
        assertThrows(IllegalStateException.class, () -> profiler.stop(BatchStats.EMPTY));
    }

    @Test
    void theReportCarriesTheCounterDeltasAndTheRecording(@TempDir Path dir) throws Exception {
        EngineProfiler profiler = new EngineProfiler(dir);
        profiler.start(BatchStats.EMPTY);
        Map<String, Object> report = profiler.stop(BatchStats.EMPTY);
        assertEquals(0L, report.get("generated_tokens"));
        assertEquals(0L, report.get("decode_steps"));
        assertEquals(0.0, (Double) report.get("generated_tokens_per_second"), 1e-12);
        assertTrue((Double) report.get("duration_seconds") >= 0.0);
        Path recording = Path.of((String) report.get("recording"));
        assertTrue(recording.startsWith(dir.toAbsolutePath()), recording.toString());
        assertTrue(Files.isRegularFile(recording), "the JFR recording is written");
        assertTrue(Files.size(recording) > 0, "the JFR recording carries the file header");
    }

    @Test
    void aProfilerWithoutADirectoryRefusesByName() {
        String previous = System.getProperty(EngineProfiler.PROFILE_DIR_PROPERTY);
        System.clearProperty(EngineProfiler.PROFILE_DIR_PROPERTY);
        try {
            EngineProfiler profiler = new EngineProfiler();
            IllegalStateException error = assertThrows(IllegalStateException.class,
                    () -> profiler.start(BatchStats.EMPTY));
            assertTrue(error.getMessage().contains(EngineProfiler.PROFILE_DIR_PROPERTY), error.getMessage());
            assertFalse(profiler.active());
        } finally {
            if (previous != null) {
                System.setProperty(EngineProfiler.PROFILE_DIR_PROPERTY, previous);
            }
        }
    }
}
