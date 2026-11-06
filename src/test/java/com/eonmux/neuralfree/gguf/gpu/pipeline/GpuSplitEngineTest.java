package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Engine-level loads of a layer split: on one selected device the split falls back to the ordinary one-device
 * forward pass, and over every OpenCL device of the machine the split model's logits match the one-device run.
 */
public class GpuSplitEngineTest {

    @Test
    public void layerModeOnASingleGpuLoadsAsOneDevice() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path file = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .splitMode(EngineConfig.SplitMode.LAYER)
                .addGpuDevice("GeForce")
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(8).build())
                .build();
        try (GGUFModelGPU model = new GGUFModelGPU(file.toString(), null, cfg)) {
            assertEquals(2, model.layerCount());
            ForwardBackend forward = model.forwardBackend();
            assertTrue("the single-device fallback binds the ordinary forward backend, got "
                    + forward.getClass().getSimpleName(), forward instanceof GpuForwardBackend);
            float[] logits = forward.decode(1, 0);
            assertEquals(model.vocabSize(), logits.length);
            forward.reset();
        }
    }

    /** The split reference logits of one device, for the two-device comparison. */
    private static final float TOL = 1e-3f;

    @Test
    public void layerModeOverTwoDevicesGeneratesLikeTheSingleDeviceLoad() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        java.util.List<String> names = openClDeviceNames();
        org.junit.Assume.assumeTrue("the split needs two OpenCL devices, found " + names, names.size() >= 2);
        Path file = SyntheticModels.llama().writeTempFile();

        EngineConfig single = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice("0")
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(8).build())
                .build();
        float[] reference;
        try (GGUFModelGPU model = new GGUFModelGPU(file.toString(), null, single)) {
            reference = model.forwardBackend().forwardRows(new int[]{1, 2, 3, 4}, 0, new int[]{3})[0];
        }

        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .splitMode(EngineConfig.SplitMode.LAYER)
                .addGpuDevice("0")
                .addGpuDevice("1")
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(8).build())
                .build();
        try (GGUFModelGPU model = new GGUFModelGPU(file.toString(), null, cfg)) {
            float[] split = model.forwardBackend().forwardRows(new int[]{1, 2, 3, 4}, 0, new int[]{3})[0];
            org.junit.Assert.assertArrayEquals("split logits match the one-device run", reference, split, TOL);
            model.resetKVCache();
        }
    }

    /** The names of the OpenCL GPU devices of this machine. */
    private static java.util.List<String> openClDeviceNames() {
        com.eonmux.neuralfree.acceleration.HWAcceleration hw =
                new com.eonmux.neuralfree.acceleration.HWAcceleration(java.util.EnumSet.of(
                        com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType.OPENCL));
        java.util.List<String> out = new java.util.ArrayList<>();
        for (Object d : hw.getCLDevices()) {
            out.add(d instanceof com.eonmux.neuralfree.acceleration.ocl.CLDevice
                    ? String.valueOf(((com.eonmux.neuralfree.acceleration.ocl.CLDevice) d).getDeviceName())
                    : String.valueOf(d));
        }
        hw.shutdown();
        return out;
    }
}
