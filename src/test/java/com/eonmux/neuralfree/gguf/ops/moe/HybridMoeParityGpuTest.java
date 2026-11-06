package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * End to end parity of the hybrid CPU+GPU MoE execution on the OpenCL device: a synthetic qwen3moe model runs on
 * the CPU engine, on the GPU with its experts on the device, on the GPU with every expert in RAM
 * ({@code neuralfree.cpu-moe}) and on the GPU with experts in RAM plus a dispatcher expert cache; the logits of
 * the hybrid runs match the reference within the float tolerance of the reordered sums, and the top token agrees.
 */
public class HybridMoeParityGpuTest {

    private static final int[] TOKENS = {7, 13, 29, 41, 5};

    /** The logits of the last input row on the given model. */
    private static float[] lastLogits(com.eonmux.neuralfree.gguf.engine.EngineModel model) {
        return model.forwardBackend().forwardRows(TOKENS, 0, new int[]{TOKENS.length - 1})[0];
    }

    @Test
    public void hybridExpertsMatchTheCpuAndDeviceLogits() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path model = Files.createTempFile("hybrid-parity", ".gguf");
        SyntheticModels.qwen3moe().writeTo(model);
        HWAcceleration acceleration = null;
        ComputeBackend backend = null;
        try {
            acceleration = HWAcceleration.forMode(com.eonmux.neuralfree.Config.ACCELERATION_MODE.OPENCL);
            backend = acceleration.createBackend(com.eonmux.neuralfree.Config.ACCELERATION_MODE.OPENCL,
                    LlmKernels.requirement());
            float[] cpu;
            try (GGUFModel reference = new GGUFModel(model.toString())) {
                cpu = lastLogits(reference);
            }
            float[] device;
            try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, EngineConfig.defaults())) {
                device = lastLogits(gpu);
            }
            float[] hybrid;
            try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend,
                    EngineConfig.builder().cpuMoe(true).build())) {
                hybrid = lastLogits(gpu);
            }
            float[] cached;
            try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend,
                    EngineConfig.builder().cpuMoe(true).moeCacheMb(4).build())) {
                cached = lastLogits(gpu);
            }
            float[] overridden;
            try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend,
                    EngineConfig.builder().addTensorOverride("ffn_(up|down|gate)_exps", "cpu").build())) {
                overridden = lastLogits(gpu);
            }
            for (int i = 0; i < cpu.length; i++) {
                assertTrue("device logits at " + i, Math.abs(device[i] - cpu[i]) < 2e-2f + 2e-2f * Math.abs(cpu[i]));
                assertTrue("hybrid logits at " + i, Math.abs(hybrid[i] - cpu[i]) < 2e-2f + 2e-2f * Math.abs(cpu[i]));
                assertTrue("cached-hybrid logits at " + i,
                        Math.abs(cached[i] - cpu[i]) < 2e-2f + 2e-2f * Math.abs(cpu[i]));
                assertTrue("override-hybrid logits at " + i,
                        Math.abs(overridden[i] - cpu[i]) < 2e-2f + 2e-2f * Math.abs(cpu[i]));
            }
            assertEquals(argmax(cpu), argmax(hybrid));
            assertEquals(argmax(cpu), argmax(cached));
            assertEquals(argmax(cpu), argmax(device));
            assertEquals(argmax(cpu), argmax(overridden));
        } finally {
            if (backend != null) {
                backend.close();
            }
            if (acceleration != null) {
                acceleration.shutdown();
            }
            Files.deleteIfExists(model);
        }
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }
}
