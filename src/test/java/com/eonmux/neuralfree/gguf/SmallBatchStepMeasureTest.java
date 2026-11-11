package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.testutil.RealModels;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.vulkan.VkInstanceManager;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.diag.GpuProfiler;
import com.eonmux.neuralfree.gguf.gpu.forward.Int8MatvecMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.Assume.assumeTrue;

/**
 * The small-batch step cost of the machine's real decoders: the wall time of one forward over
 * {@code n} rows &mdash; the speculative verify shape, a verify of draft maximum 8 is 9 rows &mdash;
 * against one decode step, with the multi-column int8 matvec lane on (the default dispatch) and off
 * ({@code neuralfree.matvec.int8=off}, the pre-lane dispatch whose few-row projections run the float
 * lanes). Opt-in through {@code -Dneuralfree.test.smallbatch.measure=true}; the shared suite skips in
 * seconds. Every row count primes on its own context fill, then times unobserved steps and reports
 * the median; the one-row figure is a device-greedy decode, the multi-row figures are prefills of
 * {@code n} tokens, exactly the forward a verify runs. One 9-row step per model and backend is
 * additionally profiled per stage under a {@link GpuProfiler}.
 *
 * <p>Models are read from {@code neuralfree.test.smallbatch.models} (default the user's model
 * directory); a missing file or unavailable backend skips its part. Nothing is downloaded.</p>
 */
public class SmallBatchStepMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.smallbatch.measure";
    /** System property naming this test's model root directory; unset, {@link RealModels#root()} is used. */
    public static final String MODELS_PROPERTY = "neuralfree.test.smallbatch.models";
    /**
     * System property that selects the device to measure by index into the backend's device list
     * (CUDA and OpenCL device order, Vulkan physical-device order); unset, zero or negative measures
     * the device the harness picks by default (the first CUDA and OpenCL device, the first Vulkan
     * device reporting cooperative matrices).
     */
    public static final String DEVICE_PROPERTY = "neuralfree.test.smallbatch.measure.device";


    /** Report names of the real decoders on the machine, in report order. */
    private static final String[] REAL_NAMES = {
            "pplx-ctx-0.6b-f16", "qwen3-emb-0.6b-q8", "qwen3-emb-4b-q8", "qwen3-emb-4b-q4k",
    };

    /** Relative paths of the real decoders under the model root, in report order. */
    private static final String[] REAL_PATHS = {
            "pplx-embed-context-0.6b-GGUF/pplx-embed-context-v1-0.6b-f16.gguf",
            "Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
            "Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf",
            "Qwen3-Embedding-4B-Q4_K_M.gguf",
    };

    /**
     * System property naming the row counts to time, comma-separated (each in 1 .. {@link #MAX_ROWS}); unset, the
     * default {@link #DEFAULT_ROW_COUNTS}. One row and {@link #PROFILE_ROWS} are timed in every case.
     */
    public static final String ROWS_PROPERTY = "neuralfree.test.smallbatch.rows";

    /** Default row counts one step is timed at: the draft maximums in between and the verify shape itself. */
    private static final int[] DEFAULT_ROW_COUNTS = {1, 2, 4, 8, 9, 16};
    /** Largest row count {@link #ROWS_PROPERTY} may name: one CM2 GEMM quarter tile. */
    private static final int MAX_ROWS = 64;
    /** Verified rows of one 9-row tree verify (draft maximum 8 plus one), the profiled shape. */
    private static final int PROFILE_ROWS = 9;
    /** The row counts of this run, ascending. */
    private static final int[] ROW_COUNTS = rowCounts(System.getProperty(ROWS_PROPERTY));
    /** Context tokens filled before the timed steps of every row count. */
    private static final int CONTEXT_TOKENS = 256;
    /** Untimed steps of each row count before its timed steps. */
    private static final int WARMUP = 3;
    /** Timed steps of each row count; the median of their per-step times is reported. */
    private static final int TIMED = 7;

    private HWAcceleration acceleration;
    private ComputeBackend backend;
    private String originalLaneMode;

    @Before
    public void rememberLaneMode() {
        originalLaneMode = System.getProperty(Int8MatvecMode.PROPERTY);
    }

    @After
    public void restoreLaneMode() {
        setLaneMode(originalLaneMode);
        closeGpu();
    }

    private void assumeMeasureEnabled() {
        assumeTrue("the small-batch step measurement is opt-in (" + MEASURE_PROPERTY + "=true)",
                Boolean.getBoolean(MEASURE_PROPERTY));
    }

    private void setLaneMode(String value) {
        if (value == null) {
            System.clearProperty(Int8MatvecMode.PROPERTY);
        } else {
            System.setProperty(Int8MatvecMode.PROPERTY, value);
        }
    }

    // ================================================================== backends

    /** The selected device index, or 0 (the default device) when {@link #DEVICE_PROPERTY} is unset or negative. */
    private static int deviceIndex() {
        return Math.max(0, Integer.getInteger(DEVICE_PROPERTY, 0));
    }

    /** The device index to open: the selected one, failed by {@code assumeTrue} when out of range. */
    private static void assumeIndex(int index, int count, String backendName) {
        assumeTrue("no " + backendName + " device " + index + " (found " + count + ")",
                index >= 0 && index < count);
    }

    private ComputeBackend openCLBackend() {
        GpuTestUtil.assumeOpenCLAvailable();
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        int index = deviceIndex();
        assumeIndex(index, acceleration.getCLDevices().size(), "OpenCL");
        backend = new OclComputeBackend(acceleration.getCLDevices().get(index));
        return backend;
    }

    private String openCLDeviceName() {
        CLDevice device = acceleration.getCLDevices().get(deviceIndex());
        Object name = device.getDeviceInfo().get("CL_DEVICE_NAME");
        return name != null ? String.valueOf(name) : String.valueOf(device.getDeviceName());
    }

    private ComputeBackend cudaBackend() {
        GpuTestUtil.assumeCudaAvailable();
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        int index = deviceIndex();
        assumeIndex(index, acceleration.getCUDevices().size(), "CUDA");
        backend = new CudaComputeBackend(acceleration.getCUDevices().get(index));
        return backend;
    }

    private String cudaDeviceName() {
        return acceleration.getCUDevices().get(deviceIndex()).getDeviceName();
    }

    /**
     * The Vulkan compute backend of device {@code index} when {@link #DEVICE_PROPERTY} selects one
     * (plain physical-device order), else of the device the fast lanes enumerate first: one reporting
     * cooperative matrices when any is, else the default device. No {@link HWAcceleration} is kept:
     * the backend owns its instance (its {@code close} releases it).
     */
    private ComputeBackend vulkanBackend() {
        GpuTestUtil.assumeVulkanAvailable();
        acceleration = null;
        int index = deviceIndex();
        VkInstanceManager manager = VkInstanceManager.acquire();
        try {
            if (index > 0) {
                assumeIndex(index, manager.devices().size(), "Vulkan");
                return new VkComputeBackend(manager, manager.devices().get(index));
            }
            for (VkPhysicalDeviceInfo info : manager.devices()) {
                if (info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX)) {
                    return new VkComputeBackend(manager, info);
                }
            }
        } finally {
            manager.release();
        }
        return new VkComputeBackend();
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

    /** {@link #ROW_COUNTS} from the property value, or the defaults when it is null or blank. */
    static int[] rowCounts(String value) {
        if (value == null || value.trim().isEmpty()) {
            return DEFAULT_ROW_COUNTS.clone();
        }
        java.util.TreeSet<Integer> rows = new java.util.TreeSet<>();
        rows.add(1);
        rows.add(PROFILE_ROWS);
        for (String part : value.split(",")) {
            int n = Integer.parseInt(part.trim());
            if (n < 1 || n > MAX_ROWS) {
                throw new IllegalArgumentException(ROWS_PROPERTY + " counts must be in 1.." + MAX_ROWS + ", got " + n);
            }
            rows.add(n);
        }
        int[] out = new int[rows.size()];
        int i = 0;
        for (int n : rows) {
            out[i++] = n;
        }
        return out;
    }

    private static EngineConfig gpuConfig() {
        int maxRows = ROW_COUNTS[ROW_COUNTS.length - 1];
        int ubatch = Math.max(32, maxRows);
        // The context fill and every untimed and timed step of the widest row count, at least 512 positions.
        int nCtx = Math.max(512, CONTEXT_TOKENS + (WARMUP + TIMED) * Math.max(maxRows, PROFILE_ROWS));
        MemoryConfig mc = MemoryConfig.builder().nCtx(nCtx).ubatch(ubatch)
                .typeK(KvDType.F16).typeV(KvDType.F16).build();
        return EngineConfig.builder().memory(mc).build();
    }

    // ================================================================== tests

    /** The real models on the OpenCL engine, each with the lane off and at its default. */
    @Test
    public void measureOnOpenCL() throws IOException {
        assumeMeasureEnabled();
        ComputeBackend device = openCLBackend();
        String deviceName = openCLDeviceName();
        for (RealModel model : realModels()) {
            measure(device, gpuConfig(), model, "OpenCL", deviceName);
        }
    }

    /** The real models on the CUDA engine, each with the lane off and at its default. */
    @Test
    public void measureOnCuda() throws IOException {
        assumeMeasureEnabled();
        ComputeBackend device = cudaBackend();
        String deviceName = cudaDeviceName();
        for (RealModel model : realModels()) {
            measure(device, gpuConfig(), model, "CUDA", deviceName);
        }
    }

    /**
     * The real models on the Vulkan engine, each with the lane off and at its default. The device is
     * the first reporting cooperative matrices, else the fastest one, as the fast GEMM lanes pick.
     */
    @Test
    public void measureOnVulkan() throws IOException {
        assumeMeasureEnabled();
        ComputeBackend device = vulkanBackend();
        for (RealModel model : realModels()) {
            measure(device, gpuConfig(), model, "Vulkan", device.capabilities().getDeviceName());
        }
    }

    // ================================================================== per model

    /**
     * Both measurements of one model on one backend: the step times with the lane off (the pre-lane
     * dispatch) and at the default dispatch, then the lane ratios per row count and the stage profile
     * of one 9-row step at the default dispatch.
     */
    private void measure(ComputeBackend device, EngineConfig config, RealModel file, String backendName,
                         String deviceName) throws IOException {
        System.out.printf(Locale.ROOT, "SMALLBATCH-MEASURE backend=%s device=\"%s\" model=%s%n",
                backendName, deviceName, file.name);

        Map<Integer, Double> before = new LinkedHashMap<>();
        Map<Integer, Double> after = new LinkedHashMap<>();
        setLaneMode("off");
        try (GGUFModelGPU model = new GGUFModelGPU(file.path.toString(), device, config)) {
            ForwardBackend f = model.forwardBackend();
            int vocab = model.vocabSize();
            int[] prompt = prompt(vocab);
            for (int n : ROW_COUNTS) {
                before.put(n, medianStepMillis(f, prompt, vocab, n));
            }
        }

        setLaneMode(originalLaneMode);
        try (GGUFModelGPU model = new GGUFModelGPU(file.path.toString(), device, config)) {
            ForwardBackend f = model.forwardBackend();
            int vocab = model.vocabSize();
            int[] prompt = prompt(vocab);
            for (int n : ROW_COUNTS) {
                after.put(n, medianStepMillis(f, prompt, vocab, n));
            }
            stageProfile(model, f, prompt, PROFILE_ROWS, backendName, file.name);
        }

        for (int n : ROW_COUNTS) {
            double b = before.get(n);
            double a = after.get(n);
            System.out.printf(Locale.ROOT,
                    "SMALLBATCH-RATIO backend=%s model=%s n=%d before-ms=%.2f after-ms=%.2f speedup=%.2fx%n",
                    backendName, file.name, n, b, a, b / Math.max(a, 1e-9));
        }
        double one = after.get(1);
        System.out.printf(Locale.ROOT,
                "SMALLBATCH-VERIFY backend=%s model=%s after-9-vs-1=%.2fx before-9-vs-1=%.2fx%n",
                backendName, file.name, after.get(PROFILE_ROWS) / one, before.get(PROFILE_ROWS) / one);
    }

    /**
     * The median wall time of one forward over {@code n} rows: the context fill primes the cache,
     * {@code n = 1} times device-greedy decodes and {@code n > 1} times prefills of {@code n} tokens
     * (the verify forward), every step at its own position over the filled cache.
     */
    private double medianStepMillis(ForwardBackend f, int[] prompt, int vocab, int n) {
        f.reset();
        if (n == 1) {
            int token = 8;
            f.prefill(prompt, 0);
            int pos = prompt.length;
            for (int i = 0; i < WARMUP; i++) {
                token = f.decodeGreedyOnDevice(token, pos++);
            }
            long[] samples = new long[TIMED];
            for (int i = 0; i < TIMED; i++) {
                long t0 = System.nanoTime();
                token = f.decodeGreedyOnDevice(token, pos++);
                samples[i] = System.nanoTime() - t0;
            }
            return median(samples) / 1e6;
        }
        f.prefill(prompt, 0);
        int pos = prompt.length;
        for (int i = 0; i < WARMUP; i++) {
            f.prefill(step(prompt, vocab, n, pos), pos);
            pos += n;
        }
        long[] samples = new long[TIMED];
        for (int i = 0; i < TIMED; i++) {
            long t0 = System.nanoTime();
            f.prefill(step(prompt, vocab, n, pos), pos);
            samples[i] = System.nanoTime() - t0;
            pos += n;
        }
        return median(samples) / 1e6;
    }

    /**
     * The per-stage times of one {@code n}-row step at the default dispatch: the profiler observes
     * only the profiled step &mdash; the priming fill and a warm-up step run first and are discarded
     * with {@link GpuProfiler#reset()} &mdash; with a queue sync at every boundary, so the shares
     * describe the device work of one step.
     */
    private void stageProfile(GGUFModelGPU model, ForwardBackend f, int[] prompt, int n,
                              String backendName, String name) {
        int vocab = model.vocabSize();
        GpuProfiler profiler = new GpuProfiler(model.computeQueue(0));
        for (int d = 0; d < model.deviceCount(); d++) {
            model.observerSlot(d).use(profiler.device(model.deviceName(d), model.computeQueue(d)));
        }
        try {
            f.reset();
            f.prefill(prompt, 0);
            f.prefill(step(prompt, vocab, n, prompt.length), prompt.length);
            profiler.reset();
            f.prefill(step(prompt, vocab, n, prompt.length + n), prompt.length + n);
        } finally {
            for (int d = 0; d < model.deviceCount(); d++) {
                model.observerSlot(d).restore();
            }
        }
        GpuProfiler.Report r = profiler.report();
        System.out.printf(Locale.ROOT, "SMALLBATCH-PROFILE backend=%s model=%s n=%d step-ms=%.2f%n",
                backendName, name, n, r.nanos(GpuProfiler.STEP) / 1e6);
        System.out.print(r.format(n + "-row step stages", r.nanos(GpuProfiler.STEP)));
    }

    // ================================================================== helpers

    private static double median(long[] samples) {
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static Path modelRoot() {
        String root = System.getProperty(MODELS_PROPERTY);
        return (root == null || root.isEmpty()) ? RealModels.root() : Paths.get(root);
    }

    /** The real model files present under the root, in report order. */
    private static List<RealModel> realModels() {
        List<RealModel> present = new ArrayList<>();
        for (int i = 0; i < REAL_PATHS.length; i++) {
            Path p = modelRoot().resolve(REAL_PATHS[i]);
            if (Files.isRegularFile(p)) {
                present.add(new RealModel(p, REAL_NAMES[i]));
            }
        }
        assumeTrue("no real models under " + modelRoot(), !present.isEmpty());
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

    /** A fixed prompt over the vocabulary. */
    private static int[] prompt(int vocab) {
        int[] tokens = new int[CONTEXT_TOKENS];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = 8 + (i * 131 + 7) % Math.min(100000, vocab - 8);
        }
        return tokens;
    }

    /** {@code n} distinct filler tokens of a step at {@code pos}, the row count the forward runs. */
    private static int[] step(int[] prompt, int vocab, int n, int pos) {
        int[] tokens = new int[n];
        for (int i = 0; i < n; i++) {
            tokens[i] = prompt[(pos + i) % prompt.length];
        }
        return tokens;
    }
}
