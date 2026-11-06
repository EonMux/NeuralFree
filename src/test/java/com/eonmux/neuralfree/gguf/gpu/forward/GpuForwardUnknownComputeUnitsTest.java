package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * A whole prefill and decode on a GPU backend whose capabilities report
 * {@link DeviceCapabilities#UNKNOWN_COMPUTE_UNITS} (a Vulkan device without the shader-core
 * extensions, such as Intel): the fast GEMM lanes stay on their plain kernels, the forward
 * completes, and its logits match the CPU backend over the same synthetic model. Every real
 * backend is wrapped so that its capabilities hide the count while its kernels run unchanged,
 * which is also what this machine's Intel iGPU reports on Vulkan. Skips without a GPU.
 */
@RunWith(Parameterized.class)
public class GpuForwardUnknownComputeUnitsTest {

    /** F32 weights: the GPU and CPU differ by summation order alone. */
    private static final float DENSE_TOL = 2e-3f;
    /**
     * Q8_0 weights: the GPU's int8 lanes quantize the activations to 8-bit codes while the CPU
     * reference multiplies the dequantized weights by F32 activations, so the two differ by the
     * activation code step rather than by summation order alone.
     */
    private static final float QUANT_TOL = 5e-2f;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend device;
    private ComputeBackend reporting;

    public GpuForwardUnknownComputeUnitsTest(String api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        try {
            device = openBackend();
            Availability availability = LlmKernels.availability(device);
            Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + availability.reason(),
                    availability.available());
            reporting = new UnknownComputeUnitsBackend(device);
            System.out.println(api + " device '" + device.capabilities().getDeviceName() + "' reports "
                    + device.capabilities().getComputeUnits() + " compute units; the wrapped view reports "
                    + reporting.capabilities().getComputeUnits());
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    private ComputeBackend openBackend() {
        if (api.equals("vulkan")) {
            return new VkComputeBackend();
        }
        hwAccel = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hwAccel.getCLDevices().isEmpty());
        return new OclComputeBackend(hwAccel.getCLDevices().get(0));
    }

    @After
    public void tearDown() {
        closeDevice();
    }

    private void closeDevice() {
        if (device != null) {
            try {
                device.close();
            } catch (RuntimeException ignored) {
                // a backend that fails to close must not mask the test's own result
            }
            device = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
    }

    /**
     * The dense model proves the prefill and decode math on the unknown-count device; the Q8_0
     * model reads the fast MMQ lanes, whose split-K heuristic is the code that used to fail for a
     * missing count. Both complete (one slice, plain kernels) and match the CPU logits.
     */
    @Test
    public void prefillAndDecodeOnAnUnknownComputeUnitDeviceMatchTheCpu() throws IOException {
        assertEquals("the wrapped capabilities hide the core count",
                DeviceCapabilities.UNKNOWN_COMPUTE_UNITS, reporting.capabilities().getComputeUnits());
        assertFalse(reporting.capabilities().isComputeUnitsKnown());

        compare("F32", synthetic(GGMLType.F32), DENSE_TOL);
        compare("Q8_0", synthetic(GGMLType.Q8_0), QUANT_TOL);
    }

    private Path synthetic(GGMLType type) throws IOException {
        return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(type).build()).writeTempFile();
    }

    /** Prefills a prompt and decodes one token on the CPU model and on the unknown-count GPU model. */
    private void compare(String label, Path path, float tol) throws IOException {
        /* 12 tokens: two ubatches of 8, the first of which is long enough for the tiled fast GEMM lane. */
        int[] prompt = {5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125, 137};
        int next = 149;
        try {
            float[] cpuPrefill;
            float[] cpuDecode;
            try (GGUFModel cpu = new GGUFModel(path.toString(), config())) {
                ForwardBackend backend = cpu.forwardBackend();
                cpuPrefill = backend.prefill(prompt, 0).clone();
                cpuDecode = backend.decode(next, prompt.length).clone();
            }
            float[] gpuPrefill;
            float[] gpuDecode;
            try (GGUFModelGPU gpu = new GGUFModelGPU(path.toString(), reporting, config())) {
                ForwardBackend backend = gpu.forwardBackend();
                gpuPrefill = backend.prefill(prompt, 0).clone();
                gpuDecode = backend.decode(next, prompt.length).clone();
            }
            System.out.println(api + " " + label + " (" + device.capabilities().getDeviceName() + "): prefill max"
                    + " |gpu-cpu| = " + maxAbsDiff(cpuPrefill, gpuPrefill) + ", decode max |gpu-cpu| = "
                    + maxAbsDiff(cpuDecode, gpuDecode));
            assertArrayEquals(label + " prefill logits", cpuPrefill, gpuPrefill, tol);
            assertArrayEquals(label + " decode logits", cpuDecode, gpuDecode, tol);
        } finally {
            Files.deleteIfExists(path);
        }
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        assertEquals(a.length, b.length);
        float max = 0f;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, Math.abs(a[i] - b[i]));
        }
        return max;
    }

    private static EngineConfig config() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().memory(memory).build();
    }

    /**
     * The backing backend with its capabilities reporting no compute-unit count, everything else
     * delegated: the device runs its real kernels, the engines see the unknown-count capabilities a
     * Vulkan device without the shader-core extensions produces.
     */
    private static final class UnknownComputeUnitsBackend implements ComputeBackend {

        private final ComputeBackend delegate;
        private final DeviceCapabilities capabilities;

        UnknownComputeUnitsBackend(ComputeBackend delegate) {
            this.delegate = delegate;
            this.capabilities = hideComputeUnits(delegate.capabilities());
        }

        private static DeviceCapabilities hideComputeUnits(DeviceCapabilities c) {
            return new DeviceCapabilities.Builder()
                    .deviceName(c.getDeviceName())
                    .totalVramBytes(c.getTotalVramBytes())
                    .maxAllocSizeBytes(c.getMaxAllocSizeBytes())
                    .computeUnits(DeviceCapabilities.UNKNOWN_COMPUTE_UNITS)
                    .maxClockFrequencyMhz(c.getMaxClockFrequencyMhz())
                    .maxWorkGroupSize(c.getMaxWorkGroupSize())
                    .supportsAsyncTransfer(c.supportsAsyncTransfer())
                    .driverVersion(c.getDriverVersion())
                    .supportsFloat64(c.supportsFloat64())
                    .supportsFloat16(c.supportsFloat16())
                    .supportsSubgroups(c.supportsSubgroups())
                    .maxLocalMemoryBytes(c.getMaxLocalMemoryBytes())
                    .supportedActivationFunctions(c.getSupportedActivationFunctions())
                    .supportedCostFunctions(c.getSupportedCostFunctions())
                    .supportedBLASLevels(c.getSupportedBLASLevels())
                    .deviceClass(c.getDeviceClass())
                    .identity(c.getIdentity())
                    .subgroupSize(c.getSubgroupSize())
                    .minSubgroupSize(c.getMinSubgroupSize())
                    .maxSubgroupSize(c.getMaxSubgroupSize())
                    .maxWorkGroupCount(c.getMaxWorkGroupCount())
                    .maxPushConstantBytes(c.getMaxPushConstantBytes())
                    .maxSharedMemoryBytes(c.getMaxSharedMemoryBytes())
                    .maxStorageBufferRange(c.getMaxStorageBufferRange())
                    .computeCapability(c.getComputeCapability())
                    .features(c.getFeatures())
                    .build();
        }

        @Override
        public BackendType backendType() {
            return delegate.backendType();
        }

        @Override
        public DeviceCapabilities capabilities() {
            return capabilities;
        }

        @Override
        public ComputeQueue computeQueue() {
            return delegate.computeQueue();
        }

        @Override
        public ComputeQueue transferQueue() {
            return delegate.transferQueue();
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return delegate.allocateBuffer(sizeInBytes, readOnly);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            return delegate.allocateBuffer(sizeInBytes);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            return delegate.compileKernel(kernelName, kernelSource, buildOptions);
        }

        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            return delegate.loadKernel(artifact);
        }

        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) {
            return delegate.supportsArtifact(kind);
        }

        @Override
        public KernelHandle cloneKernel(KernelHandle template) {
            return delegate.cloneKernel(template);
        }

        @Override
        public ComputeQueue createComputeQueue() {
            return delegate.createComputeQueue();
        }

        @Override
        public long getAllocatedBytes() {
            return delegate.getAllocatedBytes();
        }

        @Override
        public long getFreeVramBytes() {
            return delegate.getFreeVramBytes();
        }

        @Override
        public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
            delegate.prepareKernel(kernel, localWorkSize);
        }

        @Override
        public void close() {
            // the underlying device is owned by the test, which closes it after the model
        }
    }
}
