package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;

import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import jdk.jfr.Recording;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The engine profile window (vLLM's {@code /start_profile} and {@code /stop_profile}): a Java Flight
 * Recorder recording plus the engine's counter deltas between the two calls. {@code /start_profile}
 * opens the window and starts a recording whose file is written when {@code /stop_profile} closes it;
 * the directory is the {@link ProfilerConfig}'s {@code torch_profiler_dir}, else
 * {@code neuralfree.server.profile.dir}, and without one the window refuses by name. One window at a
 * time; every counter reads the snapshots the pump thread published last
 * ({@link EngineService#stats}), so the report covers exactly the engine work of the window.
 */
@Singleton
public class EngineProfiler {

    /** The directory the JFR recordings are written into; unset keeps profiling off. */
    public static final String PROFILE_DIR_PROPERTY = "neuralfree.server.profile.dir";

    private static final DateTimeFormatter STAMP = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS");

    /** The open window: its start clock, the engine snapshot then, and its recording. */
    private record Window(long startNanos, BatchStats start, Recording recording, Path file) { }

    private final AtomicReference<Window> active = new AtomicReference<>();
    private final Path configuredDir;

    /** The profiler configuration, read when a window opens; null in manual wiring. */
    @Inject
    ProfilerConfig profilerConfig;

    /** The profiler over {@link #PROFILE_DIR_PROPERTY}: the system property is read when a window opens. */
    public EngineProfiler() {
        this(null);
    }

    /**
     * The profiler over an explicit directory, for tests and manual wiring.
     *
     * @param directory the JFR recordings' directory, or null to read {@link #PROFILE_DIR_PROPERTY}
     */
    EngineProfiler(Path directory) {
        this.configuredDir = directory;
    }

    /**
     * Opens a window over the engine snapshot {@code start} and starts a JFR recording.
     *
     * @throws IllegalStateException when a window is already open, no profile directory is
     *                               configured, or the JVM cannot start a recording
     */
    public void start(BatchStats start) {
        Path directory = directory();
        try {
            Files.createDirectories(directory);
            if (!Files.isDirectory(directory)) {
                throw new IllegalStateException(PROFILE_DIR_PROPERTY + " is not a directory: " + directory);
            }
            Recording recording = new Recording();
            recording.setName("neuralfree-server");
            recording.start();
            Window window = new Window(System.nanoTime(), start, recording, recordingPath(directory));
            if (!active.compareAndSet(null, window)) {
                recording.stop();
                recording.close();
                throw new IllegalStateException("a profile is already running; stop it first");
            }
        } catch (IOException e) {
            throw new IllegalStateException("cannot create the profile directory " + directory + ": "
                    + e.getMessage(), e);
        } catch (IllegalStateException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IllegalStateException("cannot start the JFR recording: " + e.getMessage(), e);
        }
    }

    /**
     * Closes the window, writes its JFR recording and reports the engine snapshot {@code stop}'s
     * deltas since it opened.
     *
     * @throws IllegalStateException when no window is open, or the recording cannot be written
     */
    public Map<String, Object> stop(BatchStats stop) {
        Window window = active.getAndSet(null);
        if (window == null) {
            throw new IllegalStateException("no profile is running");
        }
        try {
            window.recording().stop();
            window.recording().dump(window.file());
        } catch (IOException e) {
            throw new IllegalStateException("cannot write the JFR recording to " + window.file() + ": "
                    + e.getMessage(), e);
        } finally {
            window.recording().close();
        }
        double seconds = (System.nanoTime() - window.startNanos()) / 1e9;
        long promptTokens = stop.promptTokens() - window.start().promptTokens();
        long generatedTokens = stop.generatedTokens() - window.start().generatedTokens();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("duration_seconds", seconds);
        out.put("prompt_tokens", promptTokens);
        out.put("prompt_tokens_cached", stop.promptTokensCached() - window.start().promptTokensCached());
        out.put("prompt_seconds", stop.promptSeconds() - window.start().promptSeconds());
        out.put("generated_tokens", generatedTokens);
        out.put("generated_tokens_per_second", generatedTokens / seconds);
        out.put("decode_seconds", stop.decodeSeconds() - window.start().decodeSeconds());
        out.put("decode_steps", stop.decodeSteps() - window.start().decodeSteps());
        out.put("preemptions", stop.preemptions() - window.start().preemptions());
        out.put("recomputed_tokens", stop.recomputedTokens() - window.start().recomputedTokens());
        out.put("recording", window.file().toAbsolutePath().toString());
        return out;
    }

    /** Whether a window is open. */
    public boolean active() {
        return active.get() != null;
    }

    /** The recordings' directory: the explicit one, else the profiler configuration's, else the property. */
    private Path directory() {
        if (configuredDir != null) {
            return configuredDir;
        }
        if (profilerConfig != null && profilerConfig.directory() != null) {
            return profilerConfig.directory();
        }
        String property = System.getProperty(PROFILE_DIR_PROPERTY);
        if (property == null || property.trim().isEmpty()) {
            throw new IllegalStateException("profiling is not configured: set " + PROFILE_DIR_PROPERTY
                    + " to a directory for the JFR recordings");
        }
        return Paths.get(property.trim());
    }

    /** A fresh recording path: {@code profile-<timestamp>.jfr}, numbered when the name is taken. */
    private static Path recordingPath(Path directory) {
        String stem = "profile-" + LocalDateTime.now().format(STAMP);
        Path file = directory.resolve(stem + ".jfr");
        int suffix = 1;
        while (Files.exists(file)) {
            file = directory.resolve(stem + "-" + suffix++ + ".jfr");
        }
        return file;
    }
}
