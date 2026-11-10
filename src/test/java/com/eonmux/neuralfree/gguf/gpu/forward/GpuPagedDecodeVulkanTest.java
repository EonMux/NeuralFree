package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The paged decode of several sequences on a Vulkan device writes each sequence's attention to that sequence's own
 * row: a synthetic llama prefills two prompts in one batch, decodes both sequences in one step twice (one paged flash
 * decode launch over the step's rows, whose few keys fit one split, so the launch writes its output directly), then
 * decodes one sequence beside the other's prefill chunk, and every output row equals the CPU engine's over the same
 * batches.
 */
public class GpuPagedDecodeVulkanTest {

    private static final float TOL = 1e-3f;
    private static final int[] PROMPT_A = {5, 9, 17, 23, 31};
    private static final int[] PROMPT_B = {2, 7, 13};

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build()).build();
    }

    /** The steps: both prompts, two steps that decode both sequences, and a decode beside a prefill chunk. */
    private static List<ForwardBatch> steps() {
        List<ForwardBatch> steps = new ArrayList<>();
        ForwardBatch.Builder prompts = ForwardBatch.builder(PROMPT_A.length + PROMPT_B.length);
        for (int i = 0; i < PROMPT_A.length; i++) {
            prompts.add(PROMPT_A[i], i, 0, i == PROMPT_A.length - 1);
        }
        for (int i = 0; i < PROMPT_B.length; i++) {
            prompts.add(PROMPT_B[i], i, 1, i == PROMPT_B.length - 1);
        }
        steps.add(prompts.build());
        for (int d = 0; d < 2; d++) {
            steps.add(ForwardBatch.builder(2)
                    .add(40 + d, PROMPT_A.length + d, 0, true)
                    .add(50 + d, PROMPT_B.length + d, 1, true)
                    .build());
        }
        int b = PROMPT_B.length + 2;
        steps.add(ForwardBatch.builder(4)
                .add(42, PROMPT_A.length + 2, 0, true)
                .add(60, b, 1, false)
                .add(61, b + 1, 1, false)
                .add(62, b + 2, 1, true)
                .build());
        return steps;
    }

    private static List<float[][]> run(ForwardBackend f, List<ForwardBatch> steps) {
        List<float[][]> out = new ArrayList<>();
        for (ForwardBatch step : steps) {
            out.add(f.forwardBatch(step));
        }
        return out;
    }

    @Test
    public void everySequenceOfAPagedDecodeStepGetsItsOwnRow() throws IOException {
        GpuTestUtil.assumeVulkanAvailable();
        Path model = SyntheticModels.llama().writeTempFile();
        ComputeBackend backend = null;
        try {
            List<ForwardBatch> steps = steps();
            List<float[][]> expected;
            try (GGUFModel cpu = new GGUFModel(model.toString(), config())) {
                expected = run(cpu.forwardBackend(), steps);
            }
            backend = new VkComputeBackend();
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable on Vulkan: " + availability.reason(),
                    availability.available());
            List<float[][]> actual;
            try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config())) {
                actual = run(gpu.forwardBackend(), steps);
            }
            for (int s = 0; s < steps.size(); s++) {
                assertEquals("output rows of step " + s, expected.get(s).length, actual.get(s).length);
                for (int r = 0; r < expected.get(s).length; r++) {
                    assertArrayEquals("step " + s + " output row " + r, expected.get(s)[r], actual.get(s)[r], TOL);
                }
            }
        } finally {
            if (backend != null) {
                backend.close();
            }
            Files.deleteIfExists(model);
        }
    }
}
