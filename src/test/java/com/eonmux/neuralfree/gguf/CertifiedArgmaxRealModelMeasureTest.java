package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.testutil.RealModels;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.head.CpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.cpu.head.CpuLogitsHead;
import com.eonmux.neuralfree.gguf.cpu.quant.CertifiedArgmaxScreen;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.CertifiedArgmaxMode;
import com.eonmux.neuralfree.gguf.engine.CertifiedArgmaxTuning;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

/**
 * The certified exact-argmax head measured on the machine's real decoders. Opt-in through
 * {@code -Dneuralfree.test.certargmax.measure=true}; the shared suite skips in seconds. Per GPU backend
 * it times the head readout both ways on one real activation row — the exact logits row plus the argmax
 * kernel against the certified screen and rescore, each iteration ending in the same four-byte id
 * readback a greedy decode step pays — asserts the two id runs equal, then runs the end-to-end greedy
 * loop ({@code decodeGreedyOnDevice}) with the lane off and on and asserts the committed chains equal.
 * From the model's own hidden rows ({@code hiddenRows} over a prompt) and the production host screen it
 * reports the per-row candidate counts and the full-scan fallback rate at each candidate cap. A
 * synthetic tied BF16 head at the real 0.6B shape (151936 x 1024) covers the BF16 lane, which no local
 * file carries. The CPU engine's head readouts decide the CPU gating.
 *
 * <p>Models are read from {@code neuralfree.test.certargmax.models} (default the user's model
 * directory); a missing file or unavailable backend skips its part. Nothing is downloaded.</p>
 */
public class CertifiedArgmaxRealModelMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.certargmax.measure";
    /** System property naming this test's model root directory; unset, {@link RealModels#root()} is used. */
    public static final String MODELS_PROPERTY = "neuralfree.test.certargmax.models";


    /** Report names of the real decoders on the machine, in report order. */
    private static final String[] REAL_NAMES = {
            "pplx-v1-0.6b-f16", "qwen3-emb-0.6b-q8", "qwen3-emb-4b-q8",
    };

    /** Relative paths of the real decoders under the model root, in report order. */
    private static final String[] REAL_PATHS = {
            "pplx-embed-v1-0.6b-GGUF/pplx-embed-v1-0.6b-f16.gguf",
            "Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
            "Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf",
    };

    /** Timed decode steps of one end-to-end repetition. */
    private static final int E2E_TOKENS = 40;
    /** Untimed decode steps before each timed end-to-end repetition. */
    private static final int E2E_WARMUP = 6;
    /** End-to-end repetitions; the committed chains of all repetitions must equal. */
    private static final int E2E_REPS = 5;
    /** Timed head-readout iterations per path. */
    private static final int HEAD_ITERS = 60;
    /** Untimed head-readout iterations per path (the certified first run also binds the readout). */
    private static final int HEAD_WARMUP = 20;
    /** Prompt length and hidden-row sample count. */
    private static final int PROMPT_TOKENS = 48;
    /** Candidate caps the fallback rate reports at. */
    private static final int[] CAPS = {16, 32, 64, 128, 256};

    private final List<Path> tempModels = new ArrayList<>();
    private HWAcceleration acceleration;
    private ComputeBackend backend;
    private String originalMode;
    private String originalKmax;

    @Before
    public void rememberProperties() {
        originalMode = System.getProperty(CertifiedArgmaxMode.PROPERTY);
        originalKmax = System.getProperty(CertifiedArgmaxTuning.KMAX_PROPERTY);
    }

    @After
    public void restoreProperties() throws IOException {
        if (originalMode == null) {
            System.clearProperty(CertifiedArgmaxMode.PROPERTY);
        } else {
            System.setProperty(CertifiedArgmaxMode.PROPERTY, originalMode);
        }
        if (originalKmax == null) {
            System.clearProperty(CertifiedArgmaxTuning.KMAX_PROPERTY);
        } else {
            System.setProperty(CertifiedArgmaxTuning.KMAX_PROPERTY, originalKmax);
        }
        closeGpu();
        for (Path model : tempModels) {
            Files.deleteIfExists(model);
        }
        tempModels.clear();
    }

    private void assumeMeasureEnabled() {
        assumeTrue("the real-model measurement is opt-in (" + MEASURE_PROPERTY + "=true)",
                Boolean.getBoolean(MEASURE_PROPERTY));
    }

    private void setMode(String value) {
        if (value == null) {
            System.clearProperty(CertifiedArgmaxMode.PROPERTY);
        } else {
            System.setProperty(CertifiedArgmaxMode.PROPERTY, value);
        }
    }

    // ================================================================== backends

    private ComputeBackend openCLBackend() {
        GpuTestUtil.assumeOpenCLAvailable();
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
        return backend;
    }

    private String openCLDeviceName() {
        CLDevice device = acceleration.getCLDevices().get(0);
        Object name = device.getDeviceInfo().get("CL_DEVICE_NAME");
        return name != null ? String.valueOf(name) : String.valueOf(device.getDeviceName());
    }

    private ComputeBackend cudaBackend() {
        GpuTestUtil.assumeCudaAvailable();
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        backend = new CudaComputeBackend(acceleration.getCUDevices().get(0));
        return backend;
    }

    private String cudaDeviceName() {
        return acceleration.getCUDevices().get(0).getDeviceName();
    }

    private EngineConfig vulkanConfig() {
        GpuTestUtil.assumeVulkanAvailable();
        MemoryConfig mc = MemoryConfig.builder().nCtx(512).ubatch(32)
                .typeK(KvDType.F16).typeV(KvDType.F16).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.VULKAN)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(mc).build();
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

    // ================================================================== tests

    /** The real models and the synthetic BF16 head on the OpenCL engine. */
    @Test
    public void measureOnOpenCL() throws IOException {
        assumeMeasureEnabled();
        ComputeBackend device = openCLBackend();
        String deviceName = openCLDeviceName();
        EngineConfig config = gpuConfig();
        for (RealModel model : realModels()) {
            measureGpuModel(device, config, model, "OpenCL", deviceName);
        }
        measureSyntheticBf16Head(device, config, "OpenCL", deviceName);
    }

    /** The real models and the synthetic BF16 head on the CUDA engine. */
    @Test
    public void measureOnCuda() throws IOException {
        assumeMeasureEnabled();
        ComputeBackend device = cudaBackend();
        String deviceName = cudaDeviceName();
        EngineConfig config = gpuConfig();
        for (RealModel model : realModels()) {
            measureGpuModel(device, config, model, "CUDA", deviceName);
        }
        measureSyntheticBf16Head(device, config, "CUDA", deviceName);
    }

    /** The real models and the synthetic BF16 head on the Vulkan engine, the exposed device named. */
    @Test
    public void measureOnVulkan() throws IOException {
        assumeMeasureEnabled();
        String deviceName = GpuTestUtil.getVulkanDeviceName();
        EngineConfig config = vulkanConfig();
        for (RealModel model : realModels()) {
            measureGpuModel(null, config, model, "Vulkan", deviceName);
        }
        measureSyntheticBf16Head(null, config, "Vulkan", deviceName);
    }

    /** The CPU engine's head readout on the real F16 and Q8_0 models. */
    @Test
    public void measureCpu() throws IOException {
        assumeMeasureEnabled();
        List<RealModel> models = realModels();
        assumeTrue("no real models under " + modelRoot(), !models.isEmpty());
        for (RealModel model : models) {
            measureCpuModel(model);
        }
    }

    // ================================================================== per model

    /**
     * The full measurement of one model on one GPU backend: the head readout both ways on a real
     * activation row with the id runs compared, then the end-to-end greedy loop with the lane on and
     * off, chains compared. The candidate distribution runs on the OpenCL engine only (one report per
     * model is enough).
     */
    private void measureGpuModel(ComputeBackend device, EngineConfig config, RealModel file,
                                 String backendName, String deviceName) throws IOException {
        Path path = file.path;
        String name = file.name;
        GGMLType headType = headType(path);
        int vocab = headVocab(path);
        System.out.printf(Locale.ROOT,
                "CERTARGMAX-MEASURE backend=%s device=\"%s\" model=%s head=%s vocab=%d%n",
                backendName, deviceName, name, headType, vocab);

        int[] promptTokens = prompt(vocab);
        setMode("int8");
        int[] certifiedChain;
        try (GGUFModelGPU model = openGpu(path.toString(), device, config)) {
            headReadout(model, promptTokens, backendName, name, headType);
            certifiedChain = e2e(model.forwardBackend(), promptTokens, backendName, name, "int8");
        }

        setMode("off");
        int[] exactChain;
        try (GGUFModelGPU model = openGpu(path.toString(), device, config)) {
            exactChain = e2e(model.forwardBackend(), promptTokens, backendName, name, "off");
        }
        assertArrayEquals("the certified chain equals the exact chain on " + backendName,
                exactChain, certifiedChain);

        if ("OpenCL".equals(backendName)) {
            distribution(path, device, config, name, headType);
        }
    }

    private GGUFModelGPU openGpu(String path, ComputeBackend device, EngineConfig config) throws IOException {
        return device != null
                ? new GGUFModelGPU(path, device, config)
                : new GGUFModelGPU(path, null, config);
    }

    /**
     * The head readout of one model: primes one decode so a real activation row sits in the head's
     * hidden slot, then times both paths over it and asserts their id runs equal.
     */
    private void headReadout(GGUFModelGPU model, int[] promptTokens, String backendName, String name,
                             GGMLType headType) {
        ForwardBackend f = model.forwardBackend();
        f.reset();
        int token = argmaxOf(f.prefill(promptTokens, 0));
        f.decode(token, promptTokens.length);

        try (GpuLogitsHead head = new GpuLogitsHead(model.forwardContext(),
                Collections.<GpuHeadPreprocessor>emptyList())) {
            assertTrue("the " + headType + " head certifies on " + backendName, head.certifiesArgmax());
            if (!head.certifiesRows(1)) {
                System.out.printf(Locale.ROOT,
                        "CERTARGMAX-HEAD-SKIP backend=%s model=%s type=%s reason=the one-row head projection"
                                + " runs on a lane the certified rescore does not repeat%n",
                        backendName, name, headType);
                return;
            }
            ComputeQueue queue = model.computeQueue(0);
            ScratchSlot h = head.hidden(1, new int[]{0}, 1);

            int[] warmIds = new int[HEAD_WARMUP];
            measureGpuPath(head, queue, h, false, HEAD_WARMUP, warmIds);
            measureGpuPath(head, queue, h, true, HEAD_WARMUP, warmIds);
            int[] exactIds = new int[HEAD_ITERS];
            double exactUs = measureGpuPath(head, queue, h, false, HEAD_ITERS, exactIds);
            int[] certIds = new int[HEAD_ITERS];
            double certUs = measureGpuPath(head, queue, h, true, HEAD_ITERS, certIds);
            assertArrayEquals("every certified id equals the exact id", exactIds, certIds);
            System.out.printf(Locale.ROOT,
                    "CERTARGMAX-HEAD backend=%s model=%s type=%s exact-median-us=%.1f cert-median-us=%.1f"
                            + " ratio=%.3f%n",
                    backendName, name, headType, exactUs, certUs, certUs / Math.max(exactUs, 1e-9));
        }
    }

    /**
     * One timed run of one readout path: the exact path runs the logits row plus the argmax kernel, the
     * certified path the screen, threshold, rescore and fallback; every iteration ends in the four-byte
     * id readback of a decode step and records the id. Returns the median per-iteration time in
     * microseconds.
     */
    private double measureGpuPath(GpuLogitsHead head, ComputeQueue queue, ScratchSlot h, boolean certified,
                                  int iterations, int[] idsOut) {
        long[] samples = new long[iterations];
        byte[] idBytes = new byte[4];
        for (int i = 0; i < iterations; i++) {
            long t0 = System.nanoTime();
            ScratchSlot ids;
            if (certified) {
                ids = head.argmaxCertified(1, h);
            } else {
                head.logits(1, h);
                ids = head.argmax(1);
            }
            GpuEvent ev = queue.enqueueReadBuffer(ids.buffer(), (long) ids.offset() * Integer.BYTES,
                    idBytes.length, idBytes);
            if (ev != null) {
                ev.close();
            }
            queue.finish();
            samples[i] = System.nanoTime() - t0;
            idsOut[i] = ByteBuffer.wrap(idBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
        }
        Arrays.sort(samples);
        return samples[iterations / 2] / 1000.0;
    }

    /**
     * The end-to-end greedy loop: warm-up steps, then {@link #E2E_REPS} timed repetitions of
     * {@link #E2E_TOKENS} {@code decodeGreedyOnDevice} steps; all repetitions commit the same chain.
     * Returns the chain.
     */
    private int[] e2e(ForwardBackend f, int[] promptTokens, String backendName, String name, String mode) {
        int[] first = null;
        double[] reps = new double[E2E_REPS];
        for (int rep = 0; rep < E2E_REPS; rep++) {
            f.reset();
            int token = argmaxOf(f.prefill(promptTokens, 0));
            int pos = promptTokens.length;
            for (int i = 0; i < E2E_WARMUP; i++) {
                token = f.decodeGreedyOnDevice(token, pos++);
            }
            int[] chain = new int[E2E_TOKENS];
            long t0 = System.nanoTime();
            for (int i = 0; i < E2E_TOKENS; i++) {
                chain[i] = token;
                token = f.decodeGreedyOnDevice(token, pos++);
            }
            reps[rep] = (System.nanoTime() - t0) / 1e9;
            if (first == null) {
                first = chain;
            } else {
                assertArrayEquals("repetition " + rep + " repeats the chain", first, chain);
            }
        }
        Arrays.sort(reps);
        double median = reps[E2E_REPS / 2];
        System.out.printf(Locale.ROOT,
                "CERTARGMAX-E2E backend=%s model=%s mode=%s tokens=%d median-s=%.3f tok/s=%.2f reps=%s%n",
                backendName, name, mode, E2E_TOKENS, median, E2E_TOKENS / median,
                Arrays.toString(reps));
        return first;
    }

    // ================================================================== distribution

    /**
     * The candidate-count distribution of one model's head over its own hidden rows: the rows come from
     * the GPU prefill ({@code hiddenRows} over a prompt), the screen from the production host build over
     * the file's head tensor, and the counts from the production candidate rule
     * ({@code CpuCertArgmax.argmaxRow}). Reports mean, percentiles, the widest row and the fraction of
     * rows whose candidate count exceeds each cap (the full-scan fallback rate).
     */
    private void distribution(Path path, ComputeBackend device, EngineConfig config, String name,
                              GGMLType headType) throws IOException {
        setMode("off");
        float[][] rows;
        CertifiedArgmaxScreen screen;
        try (GGUFModelGPU model = openGpu(path.toString(), device, config)) {
            ForwardBackend f = model.forwardBackend();
            int vocab = (int) model.forwardContext().spec().head().vocabRows();
            f.reset();
            rows = f.hiddenRows(prompt(vocab), null);
            screen = screenOf(path);
        }
        int count = screen.cols();
        float[] h = new float[count];
        float[] screened = new float[screen.rows()];
        int[] counts = new int[rows.length];
        double normSum = 0.0;
        for (int r = 0; r < rows.length; r++) {
            System.arraycopy(rows[r], 0, h, 0, count);
            counts[r] = candidateCount(screen, h, screened);
            normSum += Math.sqrt(dot(h, h));
        }
        int[] sorted = counts.clone();
        Arrays.sort(sorted);
        StringBuilder caps = new StringBuilder();
        for (int cap : CAPS) {
            caps.append(String.format(Locale.ROOT, " fallback@%d=%.3f", cap, fallbackRate(counts, cap)));
        }
        System.out.printf(Locale.ROOT,
                "CERTARGMAX-DIST model=%s type=%s rows=%d mean=%.1f p50=%d p90=%d p99=%d max=%d"
                        + " hiddenNormMean=%.3f%s%n",
                name, headType, sorted.length, mean(counts),
                sorted[(int) (sorted.length * 0.50)], sorted[(int) (sorted.length * 0.90)],
                sorted[Math.min(sorted.length - 1, (int) (sorted.length * 0.99))],
                sorted[sorted.length - 1], normSum / rows.length, caps);
    }

    /**
     * The production candidate count of one hidden row: the screen maximum under the bounds, the tie
     * window, then every row whose screen plus bound clears the threshold — the rule
     * {@code CpuCertArgmax.argmaxRow} applies.
     */
    private static int candidateCount(CertifiedArgmaxScreen screen, float[] h, float[] screened) {
        float hiddenNorm = (float) Math.sqrt(dot(h, h));
        screen.screenLogits(h, 0, screened, 0);
        float lower = Float.NEGATIVE_INFINITY;
        for (int v = 0; v < screen.rows(); v++) {
            float t = screened[v] - screen.bound(v) * hiddenNorm;
            if (t > lower) {
                lower = t;
            }
        }
        float threshold = lower - screen.tie(lower, hiddenNorm);
        int count = 0;
        for (int v = 0; v < screen.rows(); v++) {
            if (screened[v] + screen.bound(v) * hiddenNorm >= threshold) {
                count++;
            }
        }
        return count;
    }

    private static float dot(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /** The production screen of the model file's LM head ({@code output.weight} or {@code token_embd.weight}). */
    private static CertifiedArgmaxScreen screenOf(Path path) throws IOException {
        try (CpuWeights weights = CpuWeights.open(ModelSources.open(path.toString()))) {
            return CertifiedArgmaxScreen.of(headTensor(weights, path));
        }
    }

    private static GGMLType headType(Path path) throws IOException {
        try (CpuWeights weights = CpuWeights.open(ModelSources.open(path.toString()))) {
            return headTensor(weights, path).type();
        }
    }

    private static int headVocab(Path path) throws IOException {
        try (CpuWeights weights = CpuWeights.open(ModelSources.open(path.toString()))) {
            return Math.toIntExact(headTensor(weights, path).rows());
        }
    }

    private static CpuTensorRef headTensor(CpuWeights weights, Path path) {
        CpuTensorRef head = weights.find("output.weight");
        if (head == null) {
            head = weights.find("token_embd.weight");
        }
        if (head == null) {
            throw new IllegalStateException(path + " holds no LM head tensor");
        }
        return head;
    }

    // ================================================================== synthetic BF16 head

    /**
     * The BF16 lane on one GPU backend over a synthetic tied head at the real 0.6B shape
     * (151936 x 1024, one transformer layer): the same head readout and end-to-end measurement as the
     * real models. No local file carries a BF16 head.
     */
    private void measureSyntheticBf16Head(ComputeBackend device, EngineConfig config, String backendName,
                                          String deviceName) throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder()
                .weightType(GGMLType.BF16).nVocab(151936).nEmbd(1024).nLayer(1)
                .nHead(8).nHeadKv(8).headDim(128).nRot(128).nFf(3072)
                .tiedOutput(true).nCtxTrain(512).build();
        Path path = SyntheticModels.llama(c).writeTempFile();
        tempModels.add(path);
        measureGpuModel(device, config, new RealModel(path, "synthetic-bf16-151936x1024"),
                backendName, deviceName);
    }

    // ================================================================== CPU

    /**
     * The CPU engine's head readout on one real model: primes one decode so the final residual row sits
     * in the X scratch, then times the exact logits-plus-argmax path against the certified screen and
     * rescore over the same prepared hidden row, ids compared.
     */
    private void measureCpuModel(RealModel file) throws IOException {
        Path path = file.path;
        String name = file.name;
        GGMLType headType = headType(path);
        setMode("int8");
        try (GGUFModel model = new GGUFModel(path.toString(), cpuConfig())) {
            CpuForwardContext ctx = model.forwardContext();
            ModelSpec spec = ctx.spec();
            int finalWidth = spec.layer(spec.nLayer() - 1).residual().streamWidth();
            try (CpuLogitsHead head = new CpuLogitsHead(ctx,
                    Collections.<CpuHeadPreprocessor>emptyList())) {
                if (!head.certifiesArgmax()) {
                    System.out.printf(Locale.ROOT,
                            "CERTARGMAX-CPU-SKIP backend=CPU model=%s type=%s"
                                    + " reason=the CPU exact vec-dot of this type does not run the"
                                    + " float-activation arithmetic the weight-only screen bound covers%n",
                            name, headType);
                    return;
                }
                ForwardBackend f = model.forwardBackend();
                int vocab = (int) spec.head().vocabRows();
                int[] promptTokens = prompt(vocab);
                f.reset();
                int token = argmaxOf(f.prefill(promptTokens, 0));
                f.decode(token, promptTokens.length);

                int[] rowIdx = {0};
                float[] h = head.hidden(ctx.scratch().slot(CpuScratch.X), finalWidth, rowIdx, 1);
                System.out.printf(Locale.ROOT,
                        "CERTARGMAX-MEASURE backend=CPU model=%s head=%s vocab=%d hiddenWidth=%d%n",
                        name, headType, vocab, h.length);

                for (int i = 0; i < HEAD_WARMUP; i++) {
                    head.logits(1);
                    head.argmax(0);
                    head.argmaxCertified(1);
                }
                int[] exactIds = new int[HEAD_ITERS];
                double exactUs = measureCpuPath(head, false, HEAD_ITERS, exactIds);
                int[] certIds = new int[HEAD_ITERS];
                double certUs = measureCpuPath(head, true, HEAD_ITERS, certIds);
                assertArrayEquals("every certified id equals the exact id", exactIds, certIds);
                System.out.printf(Locale.ROOT,
                        "CERTARGMAX-HEAD backend=CPU model=%s type=%s exact-median-us=%.1f"
                                + " cert-median-us=%.1f ratio=%.3f%n",
                        name, headType, exactUs, certUs, certUs / Math.max(exactUs, 1e-9));
            }
        }
    }

    /**
     * One timed run of one CPU readout path over the prepared hidden row: the exact path runs the logits
     * row and the host argmax, the certified path the screen, threshold, rescore and fallback. Returns
     * the median per-iteration time in microseconds.
     */
    private double measureCpuPath(CpuLogitsHead head, boolean certified, int iterations, int[] idsOut) {
        long[] samples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long t0 = System.nanoTime();
            if (certified) {
                idsOut[i] = head.argmaxCertified(1)[0];
            } else {
                head.logits(1);
                idsOut[i] = head.argmax(0);
            }
            samples[i] = System.nanoTime() - t0;
        }
        Arrays.sort(samples);
        return samples[iterations / 2] / 1000.0;
    }

    private static EngineConfig gpuConfig() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(512).ubatch(32)
                .typeK(KvDType.F16).typeV(KvDType.F16).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static EngineConfig cpuConfig() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(128).ubatch(8)
                .typeK(KvDType.F16).typeV(KvDType.F16).build();
        return EngineConfig.builder().memory(mc).build();
    }

    // ================================================================== helpers

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

    /** A fixed prompt over the vocabulary, the sample the hidden rows and decode loops share. */
    private static int[] prompt(int vocab) {
        int[] tokens = new int[PROMPT_TOKENS];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = 8 + (i * 131 + 7) % Math.min(100000, vocab - 8);
        }
        return tokens;
    }

    /** Index of the largest value, the first on ties (the greedy sampler order). */
    private static int argmaxOf(float[] logits) {
        if (logits.length == 0) {
            throw new IllegalArgumentException("argmax of an empty logits row");
        }
        int best = 0;
        float bestValue = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > bestValue) {
                bestValue = logits[i];
                best = i;
            }
        }
        return best;
    }

    private static double mean(int[] v) {
        long sum = 0;
        for (int x : v) {
            sum += x;
        }
        return (double) sum / v.length;
    }

    private static double fallbackRate(int[] counts, int cap) {
        int over = 0;
        for (int c : counts) {
            if (c > cap) {
                over++;
            }
        }
        return (double) over / counts.length;
    }
}
