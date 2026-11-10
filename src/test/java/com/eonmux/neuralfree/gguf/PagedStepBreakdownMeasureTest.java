package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
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
import java.util.EnumSet;
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
 * neuralfree.test.pagedbreakdown.rows} (comma-separated, default 1,4,16) sets the row counts. Nothing is
 * downloaded.</p>
 */
public class PagedStepBreakdownMeasureTest {

    /** System property naming the measured GGUF file; unset, the test skips. */
    public static final String PATH_PROPERTY = "neuralfree.test.pagedbreakdown.path";
    /** System property replacing the measured row counts (comma-separated). */
    public static final String ROWS_PROPERTY = "neuralfree.test.pagedbreakdown.rows";

    private static final int[] DEFAULT_ROWS = {1, 4, 16};
    private static final int SEQUENCES = 16;
    private static final int N_CTX = 4096;
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

    private HWAcceleration acceleration;
    private ComputeBackend backend;

    @After
    public void closeGpu() {
        System.clearProperty(KernelTiming.PROFILE_PROPERTY);
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (acceleration != null) {
            acceleration.shutdown();
            acceleration = null;
        }
    }

    @Test
    public void breakdownOnCuda() throws IOException {
        String path = System.getProperty(PATH_PROPERTY);
        assumeTrue("the paged step breakdown is opt-in (" + PATH_PROPERTY + "=<model.gguf>)",
                path != null && Files.isRegularFile(Path.of(path)));
        GpuTestUtil.assumeCudaAvailable();
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        System.out.printf(Locale.ROOT, "STEPBREAKDOWN model=%s%n", Path.of(path).getFileName());
        try (GGUFModelGPU m = new GGUFModelGPU(path, backend(), pagedConfig())) {
            ForwardBackend f = m.forwardBackend();
            for (int rows : rows()) {
                sampledPass(f, rows, m.vocabSize());
            }
        }
        reopenBackend();
        System.setProperty(KernelTiming.PROFILE_PROPERTY, "true");
        try (GGUFModelGPU m = new GGUFModelGPU(path, backend(), pagedConfig())) {
            ForwardBackend f = m.forwardBackend();
            KernelTiming timing = m.computeQueue(0).kernelTiming();
            for (int rows : rows()) {
                int pos = warm(f, rows, m.vocabSize());
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
                System.out.printf(Locale.ROOT,
                        "STEPBREAKDOWN rows=%d profiled-device-ms-per-step=%.3f dispatches-per-step=%d%n", rows,
                        timing.totalNanos() / 1e6 / PROFILE_STEPS, timing.totalCount() / PROFILE_STEPS);
            }
        }
    }

    /** The sampled steps of one row count: wall times, then the sample attribution scaled to the mean step. */
    private static void sampledPass(ForwardBackend f, int rows, int vocab) throws IOException {
        int pos = warm(f, rows, vocab);
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
        System.out.printf(Locale.ROOT, "STEPBREAKDOWN rows=%d wall-mean-ms=%.3f wall-median-ms=%.3f%n", rows,
                meanMs, sorted[sorted.length / 2] / 1e6);
        Map<String, Integer> buckets = attribute(jfr, Thread.currentThread().getId());
        Files.deleteIfExists(jfr);
        int total = buckets.values().stream().mapToInt(Integer::intValue).sum();
        for (Map.Entry<String, Integer> e : buckets.entrySet()) {
            System.out.printf(Locale.ROOT, "STEPBREAKDOWN rows=%d %-62s %5d samples %7.3f ms/step %5.1f%%%n",
                    rows, e.getKey(), e.getValue(), meanMs * e.getValue() / Math.max(1, total),
                    100.0 * e.getValue() / Math.max(1, total));
        }
        String replay = f instanceof GpuForwardBackend ? ((GpuForwardBackend) f).decodeReplayStats() : "n/a";
        System.out.printf(Locale.ROOT, "STEPBREAKDOWN rows=%d samples=%d replay=%s%n", rows, total, replay);
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

    /** Resets the memory and prefills one token per sequence; returns the first decode position. */
    private static int warm(ForwardBackend f, int rows, int vocab) {
        f.reset();
        ForwardBatch.Builder prompt = ForwardBatch.builder(rows);
        for (int j = 0; j < rows; j++) {
            prompt.add(token(vocab, j), 0, j, false);
        }
        f.forwardBatch(prompt.build());
        return 1;
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

    private static int[] rows() {
        String v = System.getProperty(ROWS_PROPERTY);
        if (v == null || v.isBlank()) {
            return DEFAULT_ROWS.clone();
        }
        return Arrays.stream(v.split(",")).map(String::trim).filter(x -> !x.isEmpty())
                .mapToInt(Integer::parseInt).toArray();
    }

    private ComputeBackend backend() {
        if (backend == null) {
            backend = new CudaComputeBackend(acceleration.getCUDevices().get(0));
        }
        return backend;
    }

    private void reopenBackend() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    private static EngineConfig pagedConfig() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(512)
                .parallel(SEQUENCES).typeK(KvDType.F16).typeV(KvDType.F16).build();
        return EngineConfig.builder().memory(mc).build();
    }
}
