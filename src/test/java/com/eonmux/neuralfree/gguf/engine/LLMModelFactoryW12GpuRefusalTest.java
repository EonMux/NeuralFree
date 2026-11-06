package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
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
 * The GPU-only request of {@link LLMModelFactory}: a configuration that asks for the W12 streaming
 * ({@code neuralfree.weights.w12}) gets its refusal rethrown instead of a silent fallback to the CPU engine, which
 * would ignore the setting and decode the weights as stored. Failures that mean the device cannot serve the model —
 * the device running out of memory, the way the real backends report it — keep the fallback, with the reason the
 * loaded model reports.
 */
public class LLMModelFactoryW12GpuRefusalTest {

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

    private Path f16Llama() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(2)
                .weightType(GGMLType.F16).build();
        Path p = SyntheticModels.llama(c).writeTempFile();
        files.add(p);
        return p;
    }

    private static EngineConfig w12Config(SpeculativeOptions options) {
        MemoryConfig memory = MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(128).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice("0").memory(memory).speculativeOptions(options).build();
    }

    /** A GPU-only configuration: the W12 streaming with the output head, F16 held to its default grid contract. */
    private static SpeculativeOptions w12() {
        return SpeculativeOptions.defaults().toBuilder().w12(true).w12Head(true).build();
    }

    /** The refusal of an off-grid F16 weight is rethrown by name: no CPU engine silently ignores the request. */
    @Test
    public void aW12RefusalIsRethrownInsteadOfFallingBack() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = f16Llama();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LLMModelFactory.open(model.toString(), w12Config(w12())));
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_W12_F16));
    }

    /** The explicit lossy opt-in loads on the device: the factory hands back the GPU engine. */
    @Test
    public void theRoundOptInLoadsOnTheDevice() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = f16Llama();
        SpeculativeOptions round = w12().toBuilder().w12F16(W12F16Mode.ROUND).build();
        try (EngineModel m = LLMModelFactory.open(model.toString(), w12Config(round))) {
            assertTrue(m instanceof GGUFModelGPU);
        }
    }

    /**
     * A device out of memory keeps the fallback: the allocation fails the way the real backends report it
     * ({@link GpuOutOfMemoryException}), the model loads on the CPU engine even though the configuration asked for
     * the W12 streaming, and the loaded model reports the reason.
     */
    @Test
    public void aDeviceOutOfMemoryStillFallsBackToTheCpuEngine() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = f16Llama();
        acceleration = HWAcceleration.forMode(ACCELERATION_MODE.OPENCL);
        ComputeBackend real = acceleration.createBackend(ACCELERATION_MODE.OPENCL);
        ComputeBackend failing = (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                new Class<?>[]{ComputeBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("allocateBuffer")) {
                        throw new GpuOutOfMemoryException("simulated allocation failure: the device is full");
                    }
                    try {
                        return method.invoke(real, args);
                    } catch (InvocationTargetException e) {
                        throw e.getCause();
                    }
                });
        try (EngineModel m = LLMModelFactory.open(model.toString(), w12Config(w12()), failing)) {
            assertTrue(m instanceof GGUFModel);
            assertEquals("the device memory does not fit the model", m.gpuFallbackReason());
        }
    }
}
