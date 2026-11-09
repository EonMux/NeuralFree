package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.RealModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;

import static org.junit.Assume.assumeTrue;

/**
 * The paged multi-sequence decode step cost of the machine's real decoders, the step {@code nftool
 * batched-bench} runs in its generation phase: one row per sequence over the paged device memory through
 * {@code ForwardBackend.forwardBatch}, against the contiguous multi-row step of the same width (the speculative
 * verify shape) on a contiguous memory. Reports the median wall step time at 9 and 16 rows, the readback-free
 * step time (no output rows) and the per-kernel device profile of the steps' dispatches
 * ({@code KernelTiming} under {@code neuralfree.gpu.profile=true}), so a step's host share is the wall time
 * against the profiled device total. Opt-in through {@code -Dneuralfree.test.pagedbatch.measure=true}; the
 * shared suite skips in seconds.
 *
 * <p>Models are read from {@code neuralfree.test.pagedbatch.models} (default the user's model directory); a
 * missing file skips its part. Nothing is downloaded.</p>
 */
public class PagedBatchStepMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.pagedbatch.measure";
    /** System property naming this test's model root directory; unset, {@link RealModels#root()} is used. */
    public static final String MODELS_PROPERTY = "neuralfree.test.pagedbatch.models";

    private static final String[] NAMES = {"qwen3-emb-4b-q8", "qwen3-emb-4b-q4k"};
    private static final String[] PATHS = {
            "Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf",
            "Qwen3-Embedding-4B-Q4_K_M.gguf",
    };

    /** Row counts of the measured decode steps, the bench's largest cells. */
    private static final int[] ROWS = {9, 16};
    /** Sequences the paged memory holds (the bench's largest -npl). */
    private static final int SEQUENCES = 16;
    /** The bench's context ({@code -c 4096}). */
    private static final int N_CTX = 4096;
    /** Untimed steps before the timed ones of every pass. */
    private static final int WARMUP = 8;
    /** Timed steps of every pass; the median of their per-step times is reported. */
    private static final int TIMED = 32;
    /** Steps of the device-profile pass. */
    private static final int PROFILE_STEPS = 16;
    /** Tokens the contiguous reference fills its cache with before its steps (the paged steps' live keys). */
    private static final int CONTIG_CONTEXT = 128;

    /** One forward of a profiled or timed pass, at stored position {@code pos}. */
    @FunctionalInterface
    private interface StepAt {
        void run(int pos);
    }

    private HWAcceleration acceleration;
    private ComputeBackend backend;
    private String originalProfile;

    @Before
    public void rememberProfile() {
        originalProfile = System.getProperty(KernelTiming.PROFILE_PROPERTY);
    }

    @After
    public void restoreProfile() {
        setProperty(KernelTiming.PROFILE_PROPERTY, originalProfile);
        closeGpu();
    }

    private void assumeMeasureEnabled() {
        assumeTrue("the paged batch step measurement is opt-in (" + MEASURE_PROPERTY + "=true)",
                Boolean.getBoolean(MEASURE_PROPERTY));
    }

    private static void setProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }

    // ================================================================== tests

    /** The real models' paged decode steps against their contiguous multi-row steps, on the CUDA engine. */
    @Test
    public void measureOnCuda() throws IOException {
        assumeMeasureEnabled();
        GpuTestUtil.assumeCudaAvailable();
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        CUDevice device = acceleration.getCUDevices().get(0);
        for (RealModel model : realModels()) {
            System.out.printf(Locale.ROOT, "PAGEDBATCH-MEASURE backend=CUDA device=\"%s\" model=%s%n",
                    device.getDeviceName(), model.name);
            pagedPass(model);
            contiguousPass(model);
        }
    }

    // ================================================================== passes

    /**
     * The paged decode step of {@link #ROWS} rows: one row per sequence over the paged memory, the median wall
     * time with every row an output row (the bench's logits readback) and with none (the trunk alone), then the
     * per-kernel device profile of {@link #PROFILE_STEPS} steps. The backend reopens for the profile pass: the
     * queue times its dispatches only when the profile property is set at its construction.
     */
    private void pagedPass(RealModel model) throws IOException {
        setProperty(KernelTiming.PROFILE_PROPERTY, "false");
        reopenBackend();
        try (GGUFModelGPU m = new GGUFModelGPU(model.path.toString(), backend(), pagedConfig())) {
            ForwardBackend f = m.forwardBackend();
            int vocab = m.vocabSize();
            for (int rows : ROWS) {
                timedSteps("paged-logits", f, warmPaged(f, rows, vocab), pos -> f.forwardBatch(
                        decodeBatch(rows, pos, vocab, true)), rows);
                timedSteps("paged-noout", f, warmPaged(f, rows, vocab), pos -> f.forwardBatch(
                        decodeBatch(rows, pos, vocab, false)), rows);
            }
        }
        setProperty(KernelTiming.PROFILE_PROPERTY, "true");
        reopenBackend();
        try (GGUFModelGPU m = new GGUFModelGPU(model.path.toString(), backend(), pagedConfig())) {
            ForwardBackend f = m.forwardBackend();
            int vocab = m.vocabSize();
            for (int rows : ROWS) {
                deviceProfile(m, warmPaged(f, rows, vocab), pos -> f.forwardBatch(
                        decodeBatch(rows, pos, vocab, true)), rows, "paged");
            }
        }
    }

    /**
     * The contiguous multi-row step of {@link #ROWS} rows at the same live-key depth (the step-replay path the
     * speculative verify runs), for the wall time and the device profile the paged step is measured against.
     */
    private void contiguousPass(RealModel model) throws IOException {
        setProperty(KernelTiming.PROFILE_PROPERTY, "false");
        reopenBackend();
        try (GGUFModelGPU m = new GGUFModelGPU(model.path.toString(), backend(), contiguousConfig())) {
            ForwardBackend f = m.forwardBackend();
            int vocab = m.vocabSize();
            for (int rows : ROWS) {
                timedSteps("contiguous", f, warmContiguous(f, vocab),
                        pos -> f.prefill(stepTokens(vocab, pos, rows), pos), rows);
            }
        }
        setProperty(KernelTiming.PROFILE_PROPERTY, "true");
        reopenBackend();
        try (GGUFModelGPU m = new GGUFModelGPU(model.path.toString(), backend(), contiguousConfig())) {
            ForwardBackend f = m.forwardBackend();
            int vocab = m.vocabSize();
            for (int rows : ROWS) {
                deviceProfile(m, warmContiguous(f, vocab), pos -> f.prefill(stepTokens(vocab, pos, rows), pos),
                        rows, "contiguous");
            }
        }
    }

    // ================================================================== steps

    /**
     * Resets the paged memory and prefills one prompt token per sequence of the {@code rows}-row step, so the
     * decode steps run at stored position 1 and above; returns that first decode position.
     */
    private static int warmPaged(ForwardBackend f, int rows, int vocab) {
        f.reset();
        ForwardBatch.Builder prompt = ForwardBatch.builder(rows);
        for (int j = 0; j < rows; j++) {
            prompt.add(promptToken(vocab, j), 0, j, false);
        }
        f.forwardBatch(prompt.build());
        return 1;
    }

    /** Resets the contiguous memory and fills it to {@link #CONTIG_CONTEXT} rows; returns the next position. */
    private static int warmContiguous(ForwardBackend f, int vocab) {
        f.reset();
        f.prefill(stepTokens(vocab, 0, CONTIG_CONTEXT), 0);
        return CONTIG_CONTEXT;
    }

    /**
     * The median wall time of {@code WARMUP + TIMED} steps at consecutive positions from {@code pos0}, every
     * step built by {@code step} before its time starts.
     */
    private void timedSteps(String label, ForwardBackend f, int pos0, StepAt step, int rows) {
        int pos = pos0;
        for (int i = 0; i < WARMUP; i++) {
            step.run(pos++);
        }
        long[] samples = new long[TIMED];
        for (int i = 0; i < TIMED; i++) {
            long t0 = System.nanoTime();
            step.run(pos++);
            samples[i] = System.nanoTime() - t0;
        }
        System.out.printf(Locale.ROOT, "PAGEDBATCH-STEP %s rows=%d median-ms=%.2f%n",
                label, rows, median(samples) / 1e6);
    }

    /**
     * The per-kernel device profile of {@code #PROFILE_STEPS} steps built by {@code step}: the timing resets,
     * the steps run (each call finishes the queue, which drains the timing), and the report holds every
     * kernel's dispatch count and device time with the per-step device total.
     */
    private void deviceProfile(GGUFModelGPU model, int pos0, StepAt step, int rows, String label) {
        KernelTiming timing = model.computeQueue(0).kernelTiming();
        if (timing == null) {
            System.out.printf(Locale.ROOT, "PAGEDBATCH-PROFILE %s rows=%d no kernel timing on this queue%n",
                    label, rows);
            return;
        }
        int pos = pos0;
        for (int i = 0; i < WARMUP; i++) {
            step.run(pos++);
        }
        timing.reset();
        for (int i = 0; i < PROFILE_STEPS; i++) {
            step.run(pos++);
        }
        long deviceNanos = timing.totalNanos();
        long dispatches = timing.totalCount();
        System.out.printf(Locale.ROOT,
                "PAGEDBATCH-PROFILE %s rows=%d steps=%d dispatches-per-step=%d device-ms-per-step=%.2f%n",
                label, rows, PROFILE_STEPS, dispatches / PROFILE_STEPS, deviceNanos / 1e6 / PROFILE_STEPS);
        int shown = 0;
        for (KernelTiming.Entry e : timing.snapshot()) {
            if (shown++ >= 18) {
                break;
            }
            System.out.printf(Locale.ROOT, "  %-52s %6d %9.3f ms %8.2f us %5.1f%%%n",
                    e.name(), e.count(), e.totalNanos() / 1e6, e.meanNanos() / 1e3,
                    100.0 * e.totalNanos() / Math.max(1, deviceNanos));
        }
    }

    // ================================================================== helpers

    /** One decode step batch: one output-less or output row per sequence at {@code pos}. */
    private static ForwardBatch decodeBatch(int rows, int pos, int vocab, boolean outputs) {
        ForwardBatch.Builder step = ForwardBatch.builder(rows);
        for (int j = 0; j < rows; j++) {
            step.add(promptToken(vocab, pos + j), pos, j, outputs);
        }
        return step.build();
    }

    /** The contiguous step's filler tokens of a step at {@code pos}. */
    private static int[] stepTokens(int vocab, int pos, int n) {
        int[] tokens = new int[n];
        for (int i = 0; i < n; i++) {
            tokens[i] = promptToken(vocab, pos + i);
        }
        return tokens;
    }

    /** A deterministic token id inside the vocab, the same at the same position across passes. */
    private static int promptToken(int vocab, int at) {
        return 8 + (at * 131 + 7) % Math.min(100000, vocab - 8);
    }

    private static double median(long[] samples) {
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private ComputeBackend backend() {
        if (backend == null) {
            backend = new CudaComputeBackend(acceleration.getCUDevices().get(0));
        }
        return backend;
    }

    /** Closes the open backend so the next one reads the profile property at its construction. */
    private void reopenBackend() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    /** The paged memory of the bench: {@link #SEQUENCES} sequences sharing {@link #N_CTX} rows. */
    private static EngineConfig pagedConfig() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(512)
                .parallel(SEQUENCES).typeK(KvDType.F16).typeV(KvDType.F16).build();
        return EngineConfig.builder().memory(mc).build();
    }

    /** The contiguous memory of the same context, the speculative verify's target memory. */
    private static EngineConfig contiguousConfig() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(512)
                .typeK(KvDType.F16).typeV(KvDType.F16).build();
        return EngineConfig.builder().memory(mc).build();
    }

    /** The model roots: {@link #MODELS_PROPERTY} (a {@code :}-separated list), else {@link RealModels#root()}. */
    private static List<Path> modelRoots() {
        String roots = System.getProperty(MODELS_PROPERTY);
        if (roots == null || roots.isEmpty()) {
            return List.of(RealModels.root());
        }
        List<Path> out = new ArrayList<>();
        for (String root : roots.split(":")) {
            if (!root.isBlank()) {
                out.add(Path.of(root));
            }
        }
        return out;
    }

    /** The relative path's file under the first root that holds it, or null. */
    private static Path findModel(String relative) {
        for (Path root : modelRoots()) {
            Path p = root.resolve(relative);
            if (Files.isRegularFile(p)) {
                return p;
            }
        }
        return null;
    }

    /** The real model files present under the roots, in report order. */
    private static List<RealModel> realModels() {
        List<RealModel> present = new ArrayList<>();
        for (int i = 0; i < PATHS.length; i++) {
            Path p = findModel(PATHS[i]);
            if (p != null) {
                present.add(new RealModel(p, NAMES[i]));
            }
        }
        assumeTrue("no real models under " + modelRoots(), !present.isEmpty());
        return present;
    }

    /** One measured decoder: its file and its report name. */
    private static final class RealModel {
        final Path path;
        final String name;

        RealModel(Path path, String name) {
            this.path = path;
            this.name = name;
        }
    }

    private void closeGpu() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (acceleration != null) {
            acceleration.shutdown();
            acceleration = null;
        }
    }
}
