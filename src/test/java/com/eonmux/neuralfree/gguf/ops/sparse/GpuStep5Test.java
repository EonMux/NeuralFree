package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * CPU/GPU agreement of the Step5 sparse region attention kernels: the OpenCL engine's prefill and decode logits
 * against the CPU engine over the same converted graph. The step exercises every kernel of the family: the BF16/E4M3
 * roundings and indexer norm/RoPE, the region-summary ring and update, the weighted-ReLU scorer, the top-k selection
 * layout and the sparse gather attention with the head gate.
 */
public class GpuStep5Test {

    private static final int TOKENS = 32;
    private static final float TOL = 3e-3f;

    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void gpuSingleTokenMatchesTheCpuEngine() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path gguf = export("step5-gpu-short");
        int[] prompt = {1, 2, 3, 4};
        EngineConfig cfg = EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(16).ubatch(8).typeK(KvDType.BF16)
                        .typeV(KvDType.BF16).build()).build();
        HWAcceleration acceleration = new HWAcceleration();
        ComputeBackend backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
        try {
            float[] cpu;
            float[] gpu;
            try (GGUFModel model = new GGUFModel(gguf.toString(), cfg)) {
                cpu = model.forwardBackend().prefill(prompt, 0).clone();
            }
            try (GGUFModelGPU model = new GGUFModelGPU(gguf.toString(), backend, cfg)) {
                gpu = model.forwardBackend().prefill(prompt, 0).clone();
            }
            assertArrayEquals("short prefill logits", cpu, gpu, TOL);
        } finally {
            backend.close();
            acceleration.shutdown();
        }
    }

    @Test
    public void gpuGrowingPromptsMatchTheCpuEngine() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path gguf = export("step5-gpu-grow");
        int[] lengths = {8, 9, 10, 12, 16, 20, 24};
        HWAcceleration acceleration = new HWAcceleration();
        ComputeBackend backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
        try {
            for (int len : lengths) {
                int[] prompt = new int[len];
                for (int i = 0; i < len; i++) {
                    prompt[i] = (i * 3 + i / 5) % 7;
                }
                EngineConfig cfg = EngineConfig.builder()
                        .memory(MemoryConfig.builder().nCtx(32).ubatch(16).typeK(KvDType.BF16)
                                .typeV(KvDType.BF16).build()).build();
                float[] cpu;
                float[] gpu;
                try (GGUFModel model = new GGUFModel(gguf.toString(), cfg)) {
                    cpu = model.forwardBackend().prefill(prompt, 0).clone();
                }
                try (GGUFModelGPU model = new GGUFModelGPU(gguf.toString(), backend, cfg)) {
                    gpu = model.forwardBackend().prefill(prompt, 0).clone();
                }
                assertArrayEquals("prefill logits, " + len + " tokens", cpu, gpu, TOL);
            }
        } finally {
            backend.close();
            acceleration.shutdown();
        }
    }

    @Test
    public void gpuF32CacheMatchesTheCpuEngine() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path gguf = export("step5-gpu-f32");
        int[] prompt = new int[12];
        for (int i = 0; i < prompt.length; i++) {
            prompt[i] = (i * 2 + i / 4) % 7;
        }
        EngineConfig cfg = EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(24).ubatch(6).typeK(KvDType.F32)
                        .typeV(KvDType.F32).build()).build();
        HWAcceleration acceleration = new HWAcceleration();
        ComputeBackend backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
        try {
            float[] cpu;
            float[] gpu;
            try (GGUFModel model = new GGUFModel(gguf.toString(), cfg)) {
                cpu = model.forwardBackend().prefill(prompt, 0).clone();
            }
            try (GGUFModelGPU model = new GGUFModelGPU(gguf.toString(), backend, cfg)) {
                gpu = model.forwardBackend().prefill(prompt, 0).clone();
            }
            assertArrayEquals("F32-cache prefill logits", cpu, gpu, TOL);
        } finally {
            backend.close();
            acceleration.shutdown();
        }
    }

    @Test
    public void gpuSlidingWindowMatchesTheCpuEngine() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path gguf = export("step5-gpu-sliding", CpuStep5ParityTest.slidingBuilder());
        int[] prompt = new int[16];
        for (int i = 0; i < prompt.length; i++) {
            prompt[i] = (i * 5 + i / 3) % 7;
        }
        EngineConfig cfg = EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(32).ubatch(8).typeK(KvDType.BF16)
                        .typeV(KvDType.BF16).build()).build();
        HWAcceleration acceleration = new HWAcceleration();
        ComputeBackend backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
        try {
            float[] cpu;
            float[] gpu;
            try (GGUFModel model = new GGUFModel(gguf.toString(), cfg)) {
                cpu = model.forwardBackend().prefill(prompt, 0).clone();
            }
            try (GGUFModelGPU model = new GGUFModelGPU(gguf.toString(), backend, cfg)) {
                gpu = model.forwardBackend().prefill(prompt, 0).clone();
            }
            assertArrayEquals("sliding-window prefill logits", cpu, gpu, TOL);
        } finally {
            backend.close();
            acceleration.shutdown();
        }
    }

    @Test
    public void gpuPrefillAndDecodeMatchTheCpuEngine() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path gguf = export("step5-gpu");
        int[] prompt = new int[TOKENS];
        for (int i = 0; i < TOKENS; i++) {
            prompt[i] = (i * 3 + i / 5) % 7;
        }
        EngineConfig cfg = EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(TOKENS + 4).ubatch(8).typeK(KvDType.BF16)
                        .typeV(KvDType.BF16).build()).build();

        HWAcceleration acceleration = new HWAcceleration();
        ComputeBackend backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
        try {
            float[] cpuPrefill;
            float[] cpuDecode;
            float[] gpuPrefill;
            float[] gpuDecode;
            try (GGUFModel cpu = new GGUFModel(gguf.toString(), cfg)) {
                cpuPrefill = cpu.forwardBackend().prefill(prompt, 0).clone();
                cpuDecode = cpu.forwardBackend().decode(prompt[0], TOKENS).clone();
            }
            try (GGUFModelGPU gpu = new GGUFModelGPU(gguf.toString(), backend, cfg)) {
                gpuPrefill = gpu.forwardBackend().prefill(prompt, 0).clone();
                gpuDecode = gpu.forwardBackend().decode(prompt[0], TOKENS).clone();
            }
            assertEquals(7, cpuPrefill.length);
            double peak = 0.0;
            for (float v : cpuPrefill) {
                peak = Math.max(peak, Math.abs(v));
            }
            assertTrue("the CPU logits are degenerate", peak > 0.05);
            assertArrayEquals("prefill logits", cpuPrefill, gpuPrefill, TOL);
            assertArrayEquals("decode logits", cpuDecode, gpuDecode, TOL);
        } finally {
            backend.close();
            acceleration.shutdown();
        }
    }

    /** Converts the synthetic checkpoint under {@code name} and exports it as a GGUF. */
    private Path export(String name) throws Exception {
        return export(name, CpuStep5ParityTest.builder());
    }

    /** Converts {@code builder}'s synthetic checkpoint and exports it as a GGUF. */
    private Path export(String name, SyntheticHf.Builder builder) throws Exception {
        Path dir = builder.write(tmp.newFolder(name + "-hf").toPath());
        Path gguf = tmp.getRoot().toPath().resolve(name + ".gguf");
        try (HfModelSource src = HfModelSource.open(dir, ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder(name + "-cache").toPath()).build(), null)) {
            GgufWriter.write(src, gguf, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        return gguf;
    }
}
