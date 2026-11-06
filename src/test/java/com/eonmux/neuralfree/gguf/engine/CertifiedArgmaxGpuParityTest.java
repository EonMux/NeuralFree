package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The certified exact-argmax head over the synthetic decoders on the GPU engines: a greedy loop whose
 * committed tokens come from {@code decodeGreedyOnDevice} equals the target-only greedy loop (exact
 * logits rows, host argmax) token for token — through the device screen and rescore and, under a cap of
 * one candidate, through the in-kernel full-scan fallback — and so does the batched verify
 * ({@code forwardRowsGreedy}). An ineligible stored type keeps the exact readout and refuses the
 * certified call by name.
 */
public class CertifiedArgmaxGpuParityTest {

    private static final int[] PROMPT = {4, 17, 30, 43, 56, 69};
    private static final int TOKENS = 8;

    private final List<Path> models = new ArrayList<>();
    private String originalMode;
    private String originalKmax;
    private HWAcceleration acceleration;
    private ComputeBackend backend;

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
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (acceleration != null) {
            acceleration.shutdown();
            acceleration = null;
        }
        for (Path model : models) {
            Files.deleteIfExists(model);
        }
    }

    private void setMode(String value) {
        if (value == null) {
            System.clearProperty(CertifiedArgmaxMode.PROPERTY);
        } else {
            System.setProperty(CertifiedArgmaxMode.PROPERTY, value);
        }
    }

    private void setKmax(String value) {
        if (value == null) {
            System.clearProperty(CertifiedArgmaxTuning.KMAX_PROPERTY);
        } else {
            System.setProperty(CertifiedArgmaxTuning.KMAX_PROPERTY, value);
        }
    }

    private ComputeBackend openCL() {
        GpuTestUtil.assumeOpenCLAvailable();
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
        return backend;
    }

    private ComputeBackend cuda() {
        GpuTestUtil.assumeCudaAvailable();
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        backend = new com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend(
                acceleration.getCUDevices().get(0));
        return backend;
    }

    private GGUFModelGPU open(ComputeBackend device, GGMLType type) throws IOException {
        Path path = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().weightType(type).build())
                .writeTempFile();
        models.add(path);
        return new GGUFModelGPU(path.toString(), device, config());
    }

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().memory(mc).build();
    }

    /** The target-only greedy loop: exact logits rows, host argmax per step. */
    private static int[] targetLoop(GGUFModelGPU model) {
        return loop(model, false);
    }

    /** The device greedy loop: {@code decodeGreedyOnDevice} per step. */
    private static int[] deviceLoop(GGUFModelGPU model) {
        return loop(model, true);
    }

    private static int[] loop(GGUFModelGPU model, boolean onDevice) {
        int[] tokens = new int[TOKENS];
        ForwardBackend f = model.forwardBackend();
        int token = ForwardBackend.argmax(f.prefill(PROMPT, 0));
        for (int i = 0; i < TOKENS; i++) {
            int pos = PROMPT.length + i;
            tokens[i] = token;
            token = onDevice ? f.decodeGreedyOnDevice(token, pos)
                    : ForwardBackend.argmax(f.decode(token, pos));
        }
        return tokens;
    }

    /** Greedy ids of one batched verify call: the certified lane, or the host argmax over the rows. */
    private static int[] verify(GGUFModelGPU model, boolean onDevice) {
        ForwardBackend f = model.forwardBackend();
        return onDevice ? f.forwardRowsGreedy(PROMPT, 0, null) : idsOf(f.forwardRows(PROMPT, 0, null));
    }

    private static int[] idsOf(float[][] rows) {
        int[] ids = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            ids[i] = ForwardBackend.argmax(rows[i]);
        }
        return ids;
    }

    /**
     * The certified F16 loop on the OpenCL engine commits exactly the target loop's tokens — through the
     * screen and rescore and, with candidates capped at one, through the in-kernel full scan — and the
     * batched verify agrees.
     */
    @Test
    public void certifiedF16LoopMatchesTheTargetLoopOnOpenCL() throws IOException {
        setMode("int8");
        ComputeBackend device = openCL();
        int[] target;
        int[] verifyHost;
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            GpuLogitsHead head = new GpuLogitsHead(model.forwardContext(), Collections.<GpuHeadPreprocessor>emptyList());
            assertTrue(head.certifiesArgmax());
            assertTrue("a one-row projection runs on the matvec family", head.certifiesRows(1));
            target = targetLoop(model);
            verifyHost = verify(model, false);
        }
        int[] certified;
        int[] verifyDevice;
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            certified = deviceLoop(model);
            verifyDevice = verify(model, true);
        }
        assertArrayEquals(target, certified);
        assertArrayEquals(verifyHost, verifyDevice);
    }

    /** The certified F16 loop through the in-kernel full-scan fallback agrees with the target loop. */
    @Test
    public void theCapOfOneRescorsesEveryRowOnOpenCL() throws IOException {
        setMode("int8");
        setKmax("1");
        ComputeBackend device = openCL();
        int[] target;
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            target = targetLoop(model);
        }
        int[] certified;
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            certified = deviceLoop(model);
        }
        assertArrayEquals(target, certified);
    }

    /** The certified BF16 loop on the OpenCL engine commits exactly the target loop's tokens. */
    @Test
    public void certifiedBf16LoopMatchesTheTargetLoopOnOpenCL() throws IOException {
        setMode("int8");
        ComputeBackend device = openCL();
        int[] target;
        int[] verifyHost;
        try (GGUFModelGPU model = open(device, GGMLType.BF16)) {
            target = targetLoop(model);
            verifyHost = verify(model, false);
        }
        int[] certified;
        int[] verifyDevice;
        try (GGUFModelGPU model = open(device, GGMLType.BF16)) {
            certified = deviceLoop(model);
            verifyDevice = verify(model, true);
        }
        assertArrayEquals(target, certified);
        assertArrayEquals(verifyHost, verifyDevice);
    }

    /** The certified Q8_0 loop on the OpenCL engine commits exactly the target loop's tokens. */
    @Test
    public void certifiedQ8LoopMatchesTheTargetLoopOnOpenCL() throws IOException {
        setMode("int8");
        ComputeBackend device = openCL();
        int[] target;
        int[] verifyHost;
        try (GGUFModelGPU model = open(device, GGMLType.Q8_0)) {
            target = targetLoop(model);
            verifyHost = verify(model, false);
        }
        int[] certified;
        int[] verifyDevice;
        try (GGUFModelGPU model = open(device, GGMLType.Q8_0)) {
            certified = deviceLoop(model);
            verifyDevice = verify(model, true);
        }
        assertArrayEquals(target, certified);
        assertArrayEquals(verifyHost, verifyDevice);
    }

    /** An F32 head is ineligible by type on the GPU path: the exact readout stays, the certified call refuses. */
    @Test
    public void anF32HeadKeepsTheExactReadoutOnOpenCL() throws IOException {
        setMode("int8");
        ComputeBackend device = openCL();
        int[] target;
        try (GGUFModelGPU model = open(device, GGMLType.F32)) {
            GpuLogitsHead head = new GpuLogitsHead(model.forwardContext(), Collections.<GpuHeadPreprocessor>emptyList());
            assertFalse("an F32 head certifies nothing", head.certifiesArgmax());
            assertFalse(head.certifiesRows(1));
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> head.argmaxCertified(1, head.hidden(1, new int[]{0}, 1)));
            assertTrue(e.getMessage(), e.getMessage().contains("does not certify argmax"));
            target = targetLoop(model);
        }
        int[] deviceLoop;
        try (GGUFModelGPU model = open(device, GGMLType.F32)) {
            deviceLoop = deviceLoop(model);
        }
        assertArrayEquals(target, deviceLoop);
    }

    /** The certified F16 loop on the CUDA engine commits exactly the target loop's tokens. */
    @Test
    public void certifiedF16LoopMatchesTheTargetLoopOnCuda() throws IOException {
        setMode("int8");
        ComputeBackend device = cuda();
        int[] target;
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            target = targetLoop(model);
        }
        int[] certified;
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            certified = deviceLoop(model);
        }
        assertArrayEquals(target, certified);
    }

    /** The certified F16 loop on the Vulkan engine (the GLSL twins) commits the target loop's tokens. */
    @Test
    public void certifiedF16LoopMatchesTheTargetLoopOnVulkan() throws IOException {
        GpuTestUtil.assumeVulkanAvailable();
        setMode("int8");
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig vulkan = EngineConfig.builder().backend(EngineConfig.BackendMode.VULKAN)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(mc).build();
        Path path = SyntheticModels.llama(
                SyntheticModels.Config.llama().toBuilder().weightType(GGMLType.F16).build()).writeTempFile();
        models.add(path);
        int[] target;
        try (GGUFModelGPU model = new GGUFModelGPU(path.toString(), null, vulkan)) {
            target = targetLoop(model);
        }
        int[] certified;
        try (GGUFModelGPU model = new GGUFModelGPU(path.toString(), null, vulkan)) {
            certified = deviceLoop(model);
        }
        assertArrayEquals(target, certified);
    }

    /**
     * The default mode {@code auto} keeps a head below {@link CertifiedArgmaxTuning#HEAD_MIN_ELEMENTS}
     * on the exact readout even on OpenCL, the one backend with a measured win &mdash; the 0.6B-class
     * heads measured the live decode loop neutral-to-slower &mdash; while {@code on} still forces the
     * lane there and stays exact.
     */
    @Test
    public void theLaneStaysOffByDefaultBelowTheSizeGateOnOpenCL() throws IOException {
        setMode(null);
        ComputeBackend device = openCL();
        int[] target;
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            GpuLogitsHead head = new GpuLogitsHead(model.forwardContext(), Collections.<GpuHeadPreprocessor>emptyList());
            assertFalse("the synthetic head sits below the size gate, so auto keeps the exact readout",
                    head.certifiesArgmax());
            target = targetLoop(model);
            assertArrayEquals(target, deviceLoop(model));
        }
        setMode("on");
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            GpuLogitsHead head = new GpuLogitsHead(model.forwardContext(), Collections.<GpuHeadPreprocessor>emptyList());
            assertTrue("on forces the lane below the size gate", head.certifiesArgmax());
            assertArrayEquals(target, deviceLoop(model));
        }
    }

    /**
     * The default mode {@code auto} keeps the exact readout on CUDA, whose F16 head readout measured
     * slower at every head size, while {@code on} still forces the lane there and stays exact.
     */
    @Test
    public void theLaneStaysOffByDefaultOnCuda() throws IOException {
        setMode(null);
        ComputeBackend device = cuda();
        int[] target;
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            GpuLogitsHead head = new GpuLogitsHead(model.forwardContext(), Collections.<GpuHeadPreprocessor>emptyList());
            assertFalse("CUDA F16 measured slower, so auto keeps the exact readout", head.certifiesArgmax());
            target = targetLoop(model);
            assertArrayEquals(target, deviceLoop(model));
        }
        setMode("on");
        try (GGUFModelGPU model = open(device, GGMLType.F16)) {
            GpuLogitsHead head = new GpuLogitsHead(model.forwardContext(), Collections.<GpuHeadPreprocessor>emptyList());
            assertTrue("on forces the lane on the capable CUDA F16 head", head.certifiesArgmax());
            assertArrayEquals(target, deviceLoop(model));
        }
    }
}
