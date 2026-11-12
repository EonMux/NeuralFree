package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordedFrame;
import jdk.jfr.consumer.RecordedStackTrace;
import jdk.jfr.consumer.RecordingFile;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.Assume.assumeTrue;

/**
 * Where the wall time of one paged decode step goes ({@code nftool batched-bench}'s generation step: one output
 * row per sequence through {@code ForwardBackend.forwardBatch} over the paged device memory). For each measured
 * row count the steps run under a Java Flight Recorder capture of the calling thread, sampled every millisecond in
 * Java code ({@code jdk.ExecutionSample}) and in native code ({@code jdk.NativeMethodSample}, the CUDA driver
 * calls through JCuda). Every sample is attributed to the step phase on its stack (the trunk, the head and its
 * reads, the queue finish, the result assembly, the step's paging bookkeeping and uploads) and to its kind (Java,
 * or the native driver call at its top), and the shares are scaled to the measured mean step time. A second pass
 * under {@code neuralfree.gpu.profile=true} reports the profiled device time per step, an upper bound of the
 * device's busy time (each dispatch is timed between its own events). The step replay's state closes each row
 * count's report.
 *
 * <p>Opt-in through {@code -Dneuralfree.test.pagedbreakdown.path=<model.gguf>}; {@code
 * neuralfree.test.pagedbreakdown.rows} (comma-separated, default 1,4,16) sets the row counts, {@code
 * neuralfree.test.pagedbreakdown.depths} (comma-separated, default 1) the keys every sequence holds before the timed
 * steps (its prompt length), and {@code neuralfree.test.pagedbreakdown.backend} the GPU API ({@code cuda} by default,
 * or {@code vulkan}, {@code opencl}, any {@code neuralfree.backend} value; {@code neuralfree.device} picks the
 * device). The profiled pass runs where the backend times dispatches. Nothing is downloaded.</p>
 */
public class PagedStepBreakdownMeasureTest {

    /** System property naming the measured GGUF file; unset, the test skips. */
    public static final String PATH_PROPERTY = "neuralfree.test.pagedbreakdown.path";
    /** System property replacing the measured row counts (comma-separated). */
    public static final String ROWS_PROPERTY = "neuralfree.test.pagedbreakdown.rows";
    /** System property listing the keys per sequence before the timed steps (comma-separated). */
    public static final String DEPTHS_PROPERTY = "neuralfree.test.pagedbreakdown.depths";
    /** System property naming the GPU API ({@code neuralfree.backend} values); unset, CUDA. */
    public static final String BACKEND_PROPERTY = "neuralfree.test.pagedbreakdown.backend";

    private static final int[] DEFAULT_ROWS = {1, 4, 16};
    private static final int[] DEFAULT_DEPTHS = {1};
    private static final int SEQUENCES = 16;
    /** Context cells every sequence has beyond its prompt: the warmup, timed and profiled steps. */
    private static final int STEP_CELLS = 512;
    /** Prompt tokens per prefill call. */
    private static final int PREFILL_CHUNK = 512;
    private static final int WARMUP = 16;
    /** Timed steps per row count; enough 1 ms samples for a stable share at a few ms per step. */
    private static final int TIMED = 160;
    private static final int PROFILE_STEPS = 16;
    private static final Duration SAMPLE_PERIOD = Duration.ofMillis(1);

    /** Stack markers of the step phases, the most specific first: frame substring, then phase name. */
    private static final String[][] PHASES = {
            {"GpuDecodeReplay.trunk", "trunk"},
            {"GpuLayerExecutor.run", "trunk"},
            {"GpuPrefillDriver.readOutputs", "head + reads"},
            {"GpuCallReads.result", "result assembly"},
            {"ComputeQueue.finish", "queue finish"},
            {".beginPagedStep", "step inputs upload"},
            {".appendRows", "row append"},
            {"PagedSequences.finish", "step finish"},
            {".copyPage", "page copies"},
            {"GpuEmbeddingStage.", "embedding"},
            {"GpuPrefillDriver$BatchSteps.ubatch", "other step bookkeeping"},
            {"GpuPrefillDriver.runBatch", "batch setup"},
    };

    @After
    public void clearProfile() {
        System.clearProperty(KernelTiming.PROFILE_PROPERTY);
    }

    @Test
    public void breakdown() throws IOException {
        String path = System.getProperty(PATH_PROPERTY);
        assumeTrue("the paged step breakdown is opt-in (" + PATH_PROPERTY + "=<model.gguf>)",
                path != null && Files.isRegularFile(Path.of(path)));
        EngineConfig.BackendMode mode = EngineConfig.BackendMode.parse(
                System.getProperty(BACKEND_PROPERTY, "cuda"));
        if (mode == EngineConfig.BackendMode.CUDA) {
            GpuTestUtil.assumeCudaAvailable();
        }
        int[] depths = ints(DEPTHS_PROPERTY, DEFAULT_DEPTHS);
        int maxDepth = Arrays.stream(depths).max().orElse(1);
        EngineConfig cfg = pagedConfig(mode, maxDepth);
        System.out.printf(Locale.ROOT, "STEPBREAKDOWN model=%s backend=%s%n", Path.of(path).getFileName(),
                mode.name().toLowerCase(Locale.ROOT));
        try (GGUFModelGPU m = new GGUFModelGPU(path, null, cfg)) {
            ForwardBackend f = m.forwardBackend();
            for (int depth : depths) {
                for (int rows : ints(ROWS_PROPERTY, DEFAULT_ROWS)) {
                    sampledPass(f, rows, depth, m.vocabSize());
                }
            }
        }
        System.setProperty(KernelTiming.PROFILE_PROPERTY, "true");
        try (GGUFModelGPU m = new GGUFModelGPU(path, null, cfg)) {
            ForwardBackend f = m.forwardBackend();
            KernelTiming timing = m.computeQueue(0).kernelTiming();
            for (int depth : depths) {
                for (int rows : ints(ROWS_PROPERTY, DEFAULT_ROWS)) {
                    int pos = warm(f, rows, depth, m.vocabSize());
                    for (int i = 0; i < WARMUP; i++) {
                        f.forwardBatch(decodeBatch(rows, pos++, m.vocabSize()));
                    }
                    if (timing == null) {
                        continue;
                    }
                    timing.reset();
                    for (int i = 0; i < PROFILE_STEPS; i++) {
                        f.forwardBatch(decodeBatch(rows, pos++, m.vocabSize()));
                    }
                    System.out.printf(Locale.ROOT, "STEPBREAKDOWN depth=%d rows=%d profiled-device-ms-per-step=%.3f"
                                    + " dispatches-per-step=%d%n", depth, rows,
                            timing.totalNanos() / 1e6 / PROFILE_STEPS, timing.totalCount() / PROFILE_STEPS);
                }
            }
        }
    }

    /** The sampled steps of one row count: wall times, then the sample attribution scaled to the mean step. */
    private static void sampledPass(ForwardBackend f, int rows, int depth, int vocab) throws IOException {
        int pos = warm(f, rows, depth, vocab);
        for (int i = 0; i < WARMUP; i++) {
            f.forwardBatch(decodeBatch(rows, pos++, vocab));
        }
        Path jfr = Files.createTempFile("paged-step-breakdown", ".jfr");
        long[] wall = new long[TIMED];
        try (Recording recording = new Recording()) {
            recording.enable("jdk.ExecutionSample").withPeriod(SAMPLE_PERIOD).withStackTrace();
            recording.enable("jdk.NativeMethodSample").withPeriod(SAMPLE_PERIOD).withStackTrace();
            recording.start();
            for (int i = 0; i < TIMED; i++) {
                long t0 = System.nanoTime();
                f.forwardBatch(decodeBatch(rows, pos++, vocab));
                wall[i] = System.nanoTime() - t0;
            }
            recording.stop();
            recording.dump(jfr);
        }
        double meanMs = Arrays.stream(wall).average().orElse(0) / 1e6;
        long[] sorted = wall.clone();
        Arrays.sort(sorted);
        System.out.printf(Locale.ROOT, "STEPBREAKDOWN depth=%d rows=%d wall-mean-ms=%.3f wall-median-ms=%.3f%n",
                depth, rows, meanMs, sorted[sorted.length / 2] / 1e6);
        Map<String, Integer> buckets = attribute(jfr, Thread.currentThread().getId());
        Files.deleteIfExists(jfr);
        int total = buckets.values().stream().mapToInt(Integer::intValue).sum();
        for (Map.Entry<String, Integer> e : buckets.entrySet()) {
            System.out.printf(Locale.ROOT,
                    "STEPBREAKDOWN depth=%d rows=%d %-62s %5d samples %7.3f ms/step %5.1f%%%n", depth, rows,
                    e.getKey(), e.getValue(), meanMs * e.getValue() / Math.max(1, total),
                    100.0 * e.getValue() / Math.max(1, total));
        }
        String replay = f instanceof GpuForwardBackend ? ((GpuForwardBackend) f).decodeReplayStats() : "n/a";
        System.out.printf(Locale.ROOT, "STEPBREAKDOWN depth=%d rows=%d samples=%d replay=%s%n", depth, rows, total,
                replay);
    }

    /** Sample counts of the thread {@code tid} by "phase | kind", sorted by key. */
    private static Map<String, Integer> attribute(Path jfr, long tid) throws IOException {
        Map<String, Integer> buckets = new TreeMap<>();
        for (RecordedEvent e : RecordingFile.readAllEvents(jfr)) {
            String type = e.getEventType().getName();
            boolean nativeSample = type.equals("jdk.NativeMethodSample");
            if (!nativeSample && !type.equals("jdk.ExecutionSample")) {
                continue;
            }
            if (e.getThread("sampledThread") == null || e.getThread("sampledThread").getJavaThreadId() != tid) {
                continue;
            }
            RecordedStackTrace stack = e.getStackTrace();
            List<RecordedFrame> frames = stack == null ? List.of() : stack.getFrames();
            String kind = nativeSample && !frames.isEmpty() ? "native " + frameName(frames.get(0)) : "java";
            buckets.merge(phase(frames) + " | " + kind, 1, Integer::sum);
        }
        return buckets;
    }

    private static String phase(List<RecordedFrame> frames) {
        for (String[] marker : PHASES) {
            for (RecordedFrame frame : frames) {
                if (frameName(frame).contains(marker[0])) {
                    return marker[1];
                }
            }
        }
        return "caller / other";
    }

    private static String frameName(RecordedFrame frame) {
        return frame.getMethod().getType().getName().replaceFirst("^.*\\.", "") + "."
                + frame.getMethod().getName();
    }

    /**
     * Resets the memory and prefills {@code depth} tokens per sequence in chunks of at most {@link #PREFILL_CHUNK}
     * rows; returns the first decode position.
     */
    private static int warm(ForwardBackend f, int rows, int depth, int vocab) {
        f.reset();
        for (int seq = 0; seq < rows; seq++) {
            for (int pos = 0; pos < depth; ) {
                int n = Math.min(PREFILL_CHUNK, depth - pos);
                ForwardBatch.Builder prompt = ForwardBatch.builder(n);
                for (int k = 0; k < n; k++) {
                    prompt.add(token(vocab, seq + pos + k), pos + k, seq, false);
                }
                f.forwardBatch(prompt.build());
                pos += n;
            }
        }
        return depth;
    }

    /** One decode step: one output row per sequence at {@code pos}, the bench's step. */
    private static ForwardBatch decodeBatch(int rows, int pos, int vocab) {
        ForwardBatch.Builder step = ForwardBatch.builder(rows);
        for (int j = 0; j < rows; j++) {
            step.add(token(vocab, pos + j), pos, j, true);
        }
        return step.build();
    }

    private static int token(int vocab, int at) {
        return 8 + (at * 131 + 7) % Math.min(100000, vocab - 8);
    }

    private static int[] ints(String property, int[] defaults) {
        String v = System.getProperty(property);
        if (v == null || v.isBlank()) {
            return defaults.clone();
        }
        return Arrays.stream(v.split(",")).map(String::trim).filter(x -> !x.isEmpty())
                .mapToInt(Integer::parseInt).toArray();
    }

    /** Paged memory of {@link #SEQUENCES} sequences, each with room for {@code depth} keys and the steps. */
    private static EngineConfig pagedConfig(EngineConfig.BackendMode mode, int depth) {
        int nCtx = Math.multiplyExact(SEQUENCES, depth + STEP_CELLS);
        MemoryConfig mc = MemoryConfig.builder().nCtx(nCtx).ubatch(PREFILL_CHUNK)
                .parallel(SEQUENCES).typeK(KvDType.F16).typeV(KvDType.F16).build();
        return EngineConfig.builder().backend(mode).memory(mc).build();
    }
}
