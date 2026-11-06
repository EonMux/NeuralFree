package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.batch.SlotResult;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * GPU sequence state export and import over OpenCL ({@link DeviceStateTransfer}): a session file saved on the CPU
 * loads into device memory, saves back byte for byte and decodes the CPU's next logits; a device session continues on
 * the CPU. A paged CPU engine's slot restores into a device engine, saves back byte for byte from device memory, and
 * the restored rows serve the next request; a device slot restores on the CPU. Every file moves between the backends
 * unchanged.
 */
public class GpuStateTransferTest {

    private static final int[] PROMPT = {4, 9, 17, 33, 65, 129, 7, 11, 23, 47};
    private static final int NEXT = 101;
    private static final float TOL = 5e-3f;

    private static EngineConfig contiguous(EngineConfig.BackendMode mode) {
        return EngineConfig.builder().backend(mode).gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F16).typeV(KvDType.F16).build())
                .build();
    }

    private static EngineConfig paged(EngineConfig.BackendMode mode) {
        return EngineConfig.builder().backend(mode).gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2).kvPageSize(4)
                        .typeK(KvDType.F16).typeV(KvDType.F16).build())
                .build();
    }

    private static EngineModel openGpu(Path model, EngineConfig config) throws Exception {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        EngineModel gpu = LLMModelFactory.open(model.toString(), config);
        if (!(gpu instanceof GGUFModelGPU)) {
            gpu.close();
            Assume.assumeTrue("OpenCL did not open the GPU engine", false);
        }
        return gpu;
    }

    @Test
    public void aSessionFileCrossesBetweenTheCpuAndTheDevice() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path dir = Files.createTempDirectory("nf-gpu-session-");
        Path cpuFile = dir.resolve("cpu.nfst");
        Path deviceCopy = dir.resolve("device-copy.nfst");
        Path deviceFile = dir.resolve("device.nfst");
        float[] cpuNext;
        try (EngineModel cpu = LLMModelFactory.open(model.toString(), contiguous(EngineConfig.BackendMode.CPU))) {
            cpu.forwardBackend().prefill(PROMPT, 0);
            cpu.forwardBackend().memory().saveSession(cpuFile);
            cpuNext = cpu.forwardBackend().decode(NEXT, PROMPT.length);
        }
        float[] deviceNext;
        try (EngineModel gpu = openGpu(model, contiguous(EngineConfig.BackendMode.OPENCL))) {
            ModelMemory memory = gpu.forwardBackend().memory();
            assertTrue(memory instanceof DeviceModelMemory);
            gpu.forwardBackend().prefill(new int[] {8, 16, 24, 32}, 0);
            memory.loadSession(cpuFile);
            assertArrayEquals(PROMPT, memory.sequence().tokens());
            memory.saveSession(deviceCopy);
            assertArrayEquals("the device holds the CPU's rows bit for bit", Files.readAllBytes(cpuFile),
                    Files.readAllBytes(deviceCopy));
            float[] logits = gpu.forwardBackend().decode(NEXT, PROMPT.length);
            assertArrayEquals("the device continues the CPU's sequence", cpuNext, logits, TOL);
            memory.reset();
            gpu.forwardBackend().prefill(PROMPT, 0);
            memory.saveSession(deviceFile);
            deviceNext = gpu.forwardBackend().decode(NEXT, PROMPT.length);
        }
        try (EngineModel cpu = LLMModelFactory.open(model.toString(), contiguous(EngineConfig.BackendMode.CPU))) {
            cpu.forwardBackend().memory().loadSession(deviceFile);
            float[] logits = cpu.forwardBackend().decode(NEXT, PROMPT.length);
            assertArrayEquals("the CPU continues the device's sequence", deviceNext, logits, TOL);
        }
    }

    @Test
    public void aSlotCrossesBetweenTheCpuAndTheDevice() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path dir = Files.createTempDirectory("nf-gpu-slot-");
        Path cpuSlot = dir.resolve("cpu.slot");
        Path deviceCopy = dir.resolve("device-copy.slot");
        Path deviceSlot = dir.resolve("device.slot");
        int[] prompt = new int[14];
        for (int i = 0; i < prompt.length; i++) {
            prompt[i] = 4 + (i * 37) % 250;
        }
        SlotResult saved;
        try (EngineModel cpu = LLMModelFactory.open(model.toString(), paged(EngineConfig.BackendMode.CPU));
             BatchEngine engine = BatchEngine.forModel(cpu, paged(EngineConfig.BackendMode.CPU))) {
            run(engine, prompt);
            saved = engine.saveSlot(0, cpuSlot);
        }
        try (EngineModel gpu = openGpu(model, paged(EngineConfig.BackendMode.OPENCL));
             BatchEngine engine = BatchEngine.forModel(gpu, paged(EngineConfig.BackendMode.OPENCL))) {
            assertEquals(saved.tokens(), engine.restoreSlot(0, cpuSlot).tokens());
            engine.saveSlot(0, deviceCopy);
            assertArrayEquals("the device pages hold the CPU's rows bit for bit", Files.readAllBytes(cpuSlot),
                    Files.readAllBytes(deviceCopy));
            BatchSequence adopted = run(engine, prompt);
            assertEquals("the restored device rows serve the prompt", prompt.length - 1, adopted.reusedPrefix());
            engine.eraseSlot(0);
            run(engine, prompt);
            engine.saveSlot(0, deviceSlot);
        }
        try (EngineModel cpu = LLMModelFactory.open(model.toString(), paged(EngineConfig.BackendMode.CPU));
             BatchEngine engine = BatchEngine.forModel(cpu, paged(EngineConfig.BackendMode.CPU))) {
            assertEquals(saved.tokens(), engine.restoreSlot(1, deviceSlot).tokens());
            assertEquals("the device's rows serve the CPU", prompt.length - 1, run(engine, prompt).reusedPrefix());
        }
    }

    private static BatchSequence run(BatchEngine engine, int[] prompt) throws InterruptedException {
        BatchSequence seq = engine.submit(prompt, GenerationRequest.builder().maxNewTokens(5)
                .sampler(SamplerConfig.greedyDecoding()).build(), (TokenCallback) null);
        engine.drain();
        assertEquals(BatchSequence.State.DONE, seq.state());
        return seq;
    }
}
