package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The scope of {@link LLMModelFactory}'s GPU-to-CPU fallback: it happens only when the failure means the device
 * cannot serve the model — no device or driver, the LLM kernel library unavailable, the device memory exhausted, a
 * weight type or architecture the GPU engine does not implement — and the reason the loaded CPU model reports names
 * it. Any other failure is an engine defect and fails the load with the GPU error, like llama.cpp's error on a
 * backend that fails, instead of silently degrading to a fraction of the speed.
 */
public class LLMModelFactoryFallbackScopeTest {

    private final List<Path> files = new ArrayList<>();
    private HWAcceleration acceleration;

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
        files.clear();
        if (acceleration != null) {
            acceleration.shutdown();
            acceleration = null;
        }
    }

    private Path llamaModel() throws IOException {
        Path p = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
        files.add(p);
        return p;
    }

    private static EngineConfig gpuConfig() {
        MemoryConfig memory = MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(128).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(memory).build();
    }

    /** A shared OpenCL backend whose one method fails with {@code failure}; everything else reaches the device. */
    private ComputeBackend openClFailingAt(String method, RuntimeException failure) {
        acceleration = HWAcceleration.forMode(ACCELERATION_MODE.OPENCL);
        ComputeBackend real = acceleration.createBackend(ACCELERATION_MODE.OPENCL);
        return (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                new Class<?>[]{ComputeBackend.class}, (proxy, m, args) -> {
                    if (m.getName().equals(method)) {
                        throw failure;
                    }
                    try {
                        return m.invoke(real, args);
                    } catch (InvocationTargetException e) {
                        throw e.getCause();
                    }
                });
    }

    /**
     * No device of the requested kind is available: the load falls back to the CPU engine and the opened model
     * reports the reason. The test requests a GPU API the machine provably lacks (the AUTO device filter ignores an
     * unmatched name and selects normally, so an explicit API is the real path).
     */
    @Test
    public void aDeviceWithoutDriversFallsBackAndReportsItsReason() throws IOException {
        EngineConfig.BackendMode mode = unavailableGpuMode();
        Assume.assumeTrue("every GPU API candidate is available on this machine", mode != null);
        Path model = llamaModel();
        EngineConfig cfg = EngineConfig.builder().backend(mode)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(128).build()).build();
        try (EngineModel m = LLMModelFactory.open(model.toString(), cfg)) {
            assertTrue(m instanceof GGUFModel);
            assertEquals("no usable GPU device or driver", m.gpuFallbackReason());
        }
    }

    /** The first GPU API of the candidates this machine cannot serve, or null when it serves them all. */
    private static EngineConfig.BackendMode unavailableGpuMode() {
        EngineConfig.BackendMode[] candidates = {EngineConfig.BackendMode.HIP, EngineConfig.BackendMode.SYCL,
                EngineConfig.BackendMode.METAL, EngineConfig.BackendMode.CANN, EngineConfig.BackendMode.MUSA,
                EngineConfig.BackendMode.ZDNN, EngineConfig.BackendMode.ZENDNN};
        for (EngineConfig.BackendMode mode : candidates) {
            HWAcceleration acceleration = HWAcceleration.forMode(mode.toAccelerationMode());
            try {
                ComputeBackend ignored = acceleration.createBackend(mode.toAccelerationMode());
                acceleration.shutdown();
            } catch (NoBackendAvailableException e) {
                return mode;
            }
            acceleration.shutdown();
        }
        return null;
    }

    /** The device is out of memory: the load falls back and the reason names the device memory. */
    @Test
    public void aDeviceOutOfMemoryFallsBackAndReportsItsReason() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = llamaModel();
        ComputeBackend failing = openClFailingAt("allocateBuffer",
                new GpuOutOfMemoryException("simulated: the device is full"));
        try (EngineModel m = LLMModelFactory.open(model.toString(), gpuConfig(), failing)) {
            assertTrue(m instanceof GGUFModel);
            assertEquals("the device memory does not fit the model", m.gpuFallbackReason());
        }
    }

    /** The LLM kernel library is unavailable on the device: the load falls back and the reason names it. */
    @Test
    public void aKernelLibraryRefusalFallsBackAndReportsItsReason() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = llamaModel();
        ComputeBackend failing = openClFailingAt("loadKernel",
                new UnsupportedKernelException("simulated: the LLM kernels are not built for this device"));
        try (EngineModel m = LLMModelFactory.open(model.toString(), gpuConfig(), failing)) {
            assertTrue(m instanceof GGUFModel);
            assertEquals("the LLM kernel library is unavailable on the device", m.gpuFallbackReason());
        }
    }

    /**
     * A weight type the GPU engine does not implement is refused by name: the load falls back and the reason names
     * the refusal, here the MoE expert cache holding experts the device kernels cannot run while the CPU engine
     * computes them on the host.
     */
    @Test
    public void anUnimplementedWeightTypeFallsBackAndReportsItsName() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        SyntheticModels.Config config = SyntheticModels.Config.qwen3moe().toBuilder()
                .weightType(GGMLType.F8_E4M3FN).build();
        Path model = SyntheticModels.qwen3moe(config).writeTempFile();
        files.add(model);
        MemoryConfig memory = MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(128).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(memory).moeCache(MoeCacheConfig.parse("on")).build();
        try (EngineModel m = LLMModelFactory.open(model.toString(), cfg)) {
            assertTrue(m instanceof GGUFModel);
            String reason = m.gpuFallbackReason();
            assertTrue(reason, reason.startsWith("the GPU engine cannot serve the model: "));
            assertTrue(reason, reason.contains("F8_E4M3FN"));
            assertTrue(reason, reason.contains("MoE expert cache"));
        }
    }

    /** The classification of one failure decides the fallback; the reason names the limitation. */
    @Test
    public void theFallbackReasonClassifiesTheDeviceLimitations() {
        assertEquals("no usable GPU device or driver",
                LLMModelFactory.fallbackReason(new NoBackendAvailableException("no OpenCL runtime")));
        assertEquals("the LLM kernel library is unavailable on the device",
                LLMModelFactory.fallbackReason(new UnsupportedKernelException("no LLM kernels")));
        assertEquals("the device memory does not fit the model",
                LLMModelFactory.fallbackReason(new GpuOutOfMemoryException("full")));
        assertTrue(LLMModelFactory.fallbackReason(
                        new GpuModelUnsupportedException("no GPU mixer for KDA")).contains("no GPU mixer for KDA"));
        assertEquals(null, LLMModelFactory.fallbackReason(new IllegalStateException("argument 34 is null")));
        assertEquals(null, LLMModelFactory.fallbackReason(new NullPointerException()));
        assertEquals(null, LLMModelFactory.fallbackReason(new IndexOutOfBoundsException()));
        assertEquals(null, LLMModelFactory.fallbackReason(new LinkageError("no native library")));
    }

    /**
     * No context fits the device memory budget ({@link com.eonmux.neuralfree.gguf.memory.ContextFitter
     * #ContextBudgetException}, as a required hybrid split throws it): the load falls back and the reason names the
     * budget.
     */
    @Test
    public void aContextThatCannotFitItsBudgetFallsBackAndReportsItsReason() throws IOException {
        Path model = llamaModel();
        try (ModelSource source = ModelSources.open(model.toString())) {
            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(source), TensorCatalog.of(source),
                    MemoryConfig.defaults(), 0, PoolingKind.NONE);
            ContextFitter.ContextBudgetException e = assertThrows(ContextFitter.ContextBudgetException.class,
                    () -> ContextFitter.fit(spec, MemoryConfig.defaults().toBuilder().nCtx(64).build(), 1024L,
                            ContextFitter.Companion.NONE));
            assertEquals("no context fits the device memory budget", LLMModelFactory.fallbackReason(e));
        }
    }

    /**
     * An engine defect fails the load: a kernel load that dies the way the unbound MLA arguments did (an
     * {@link IllegalStateException}) propagates instead of falling back, so the GPU error is the load's failure.
     */
    @Test
    public void anEngineDefectFailsTheLoadInsteadOfFallingBack() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = llamaModel();
        ComputeBackend failing = openClFailingAt("loadKernel",
                new IllegalStateException("argument 34 of LLM_FlashDecode_F16 is null"));
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> LLMModelFactory.open(model.toString(), gpuConfig(), failing));
        assertTrue(e.getMessage(), e.getMessage().contains("argument 34 of LLM_FlashDecode_F16 is null"));
    }

    /** A failed allocation that is no device OOM is an engine defect too: it fails the load. */
    @Test
    public void anUnexplainedAllocationFailureFailsTheLoadInsteadOfFallingBack() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = llamaModel();
        ComputeBackend failing = openClFailingAt("allocateBuffer",
                new IllegalStateException("simulated allocation failure: the device is full"));
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> LLMModelFactory.open(model.toString(), gpuConfig(), failing));
        assertTrue(e.getMessage(), e.getMessage().contains("simulated allocation failure: the device is full"));
    }
}
