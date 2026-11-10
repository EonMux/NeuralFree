package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * CPU/GPU parity of the limite graph on CUDA, OpenCL and Vulkan: value embeddings (with the per-KV-head gate), the
 * exclusive self attention, the learned residual scales and the MUDD dense cross-layer mixes produce the same logits
 * on the device path as on the host path, for contiguous prefill, decode and a paged multi-sequence batch. The
 * fixture's sigmoid logit cap squeezes every logit into a band about 1e-3 wide, so the tolerance is a fraction of
 * the reference's spread: a fixed tolerance as wide as the band would pass any logits.
 */
@RunWith(Parameterized.class)
public class GpuLimiteParityTest {

    private static final int N_TOKENS = 10;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    /** Allowed difference as a fraction of the reference logits' spread (max - min). */
    private static final double SPREAD_TOL = 0.01;
    /** The same for F16 weights, whose device tiles round their operands. */
    private static final double SPREAD_TOL_F16 = 0.05;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> apis() {
        return Arrays.asList(new Object[]{BackendType.CUDA}, new Object[]{BackendType.OPENCL},
                new Object[]{BackendType.VULKAN});
    }

    private final BackendType api;
    private HWAcceleration acceleration;
    private ComputeBackend backend;

    public GpuLimiteParityTest(BackendType api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        switch (api) {
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            default:
                GpuTestUtil.assumeVulkanAvailable();
                break;
        }
        acceleration = new HWAcceleration(EnumSet.of(api));
        HardwareDevice device = api == BackendType.CUDA
                ? (acceleration.getCUDevices().isEmpty() ? null : acceleration.getCUDevices().get(0))
                : api == BackendType.OPENCL
                ? (acceleration.getCLDevices().isEmpty() ? null : acceleration.getCLDevices().get(0))
                : VkDeviceRanking.fastestDevice(acceleration.getVkDevices(), null);
        Assume.assumeTrue("no " + api + " device", device != null);
        backend = acceleration.createBackend(device);
    }

    /** Element-wise {@code |expected - actual| <= tol * spread(expected)}, with a non-degenerate spread. */
    private void assertLogits(String message, float[] expected, float[] actual, double tol) {
        assertEquals(message + ": logit count", expected.length, actual.length);
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float v : expected) {
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        double spread = (double) max - min;
        assertTrue(message + ": the reference logits spread", spread > 1e-6);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(api + " " + message + " [" + i + "] (spread " + spread + ")", expected[i], actual[i],
                    tol * spread);
        }
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    private static ForwardBatch interleave(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, SEQ_A, i == a.length - 1);
            batch.add(b[i], i, SEQ_B, i == b.length - 1);
        }
        return batch.build();
    }

    @Test
    public void theGpuPrefillMatchesTheCpuPath() throws IOException {
        Path model = SyntheticModels.limite().writeTempFile();
        int[] prompt = tokens(1);
        float[] cpu;
        try (GGUFModel one = new GGUFModel(model.toString(), config(1))) {
            assertTrue("the fixture carries the MUDD extras",
                    one.spec().layers().get(0).hasExtra(LayerSpec.Extras.MUDD));
            assertTrue("the fixture carries value embeddings",
                    one.spec().layers().get(0).attention().valueEmbeddings());
            assertTrue("the fixture carries the exclusive self attention",
                    one.spec().layers().get(0).attention().xsaEps() > 0f);
            cpu = one.forwardBackend().forwardRows(prompt, 0, new int[]{prompt.length - 1})[0];
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(1))) {
            ForwardBackend f = gpu.forwardBackend();
            float[] out = f.forwardRows(prompt, 0, new int[]{prompt.length - 1})[0];
            assertNotNull(out);
            assertLogits("limite prefill logits match the CPU path", cpu, out, SPREAD_TOL);
        }
    }

    @Test
    public void halfPrecisionWeightsKeepTheGpuLogitsCloseToTheCpuPath() throws IOException {
        Path model = SyntheticModels.limite(SyntheticModels.Config.limite().toBuilder()
                .weightType(GGMLType.F16).build()).writeTempFile();
        int[] prompt = tokens(5);
        float[] cpu;
        try (GGUFModel one = new GGUFModel(model.toString(), config(1))) {
            cpu = one.forwardBackend().forwardRows(prompt, 0, new int[]{prompt.length - 1})[0];
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(1))) {
            float[] out = gpu.forwardBackend().forwardRows(prompt, 0, new int[]{prompt.length - 1})[0];
            assertLogits("half-precision limite logits match the CPU path", cpu, out, SPREAD_TOL_F16);
        }
    }

    @Test
    public void aGpuDecodeStepMatchesTheCpuPath() throws IOException {
        Path model = SyntheticModels.limite().writeTempFile();
        int[] prompt = tokens(2);
        float[] cpu;
        try (GGUFModel one = new GGUFModel(model.toString(), config(1))) {
            ForwardBackend f = one.forwardBackend();
            f.forwardRows(prompt, 0, new int[]{prompt.length - 1});
            cpu = f.decode(prompt[0], prompt.length).clone();
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(1))) {
            ForwardBackend f = gpu.forwardBackend();
            f.forwardRows(prompt, 0, new int[]{prompt.length - 1});
            float[] out = f.decode(prompt[0], prompt.length);
            assertLogits("limite decode logits match the CPU path", cpu, out, SPREAD_TOL);
        }
    }

    @Test
    public void aPagedGpuBatchMatchesTheContiguousCpuRuns() throws IOException {
        Path model = SyntheticModels.limite().writeTempFile();
        int[] a = tokens(3);
        int[] b = tokens(4);
        float[][] reference = new float[2][];
        try (GGUFModel one = new GGUFModel(model.toString(), config(1))) {
            reference[0] = one.forwardBackend().forwardRows(a, 0, new int[]{a.length - 1})[0];
            one.forwardBackend().reset();
            reference[1] = one.forwardBackend().forwardRows(b, 0, new int[]{b.length - 1})[0];
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(2))) {
            float[][] out = gpu.forwardBackend().forwardBatch(interleave(a, b));
            assertEquals(2, out.length);
            assertLogits("sequence A logits match the contiguous CPU run", reference[0], out[0], SPREAD_TOL);
            assertLogits("sequence B logits match the contiguous CPU run", reference[1], out[1], SPREAD_TOL);
        }
    }
}
