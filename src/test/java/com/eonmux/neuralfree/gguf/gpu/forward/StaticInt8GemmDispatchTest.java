package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The int8 MMQ dispatch behind a checkpoint activation scheme: a weight with {@link ActQuantScheme#INT8_STATIC}
 * stages the activations with {@link GemmKernels#QUANT_MMQ_INT8_STATIC} (the checkpoint's own int8 codes, so the
 * int8 GEMM computes W8A8) instead of fake-quantizing the row to float and letting the dynamic staging pass
 * requantize it. When the scheme staging kernel is unavailable, the projection falls back to the fake-quantization
 * path and the dynamic staging pass, with the same GEMM.
 */
public class StaticInt8GemmDispatchTest {

    private static final int TOKENS = 16;
    private static final int COLS = 256;
    private static final int ROWS = 96;

    @Test
    public void staticInt8StagesCodesDirectlyForQuantizedWeights() {
        try (FakeComputeBackend backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
             GpuScratch scratch = scratch(backend);
             GpuPlaceholders placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
             GpuOps ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
             GpuBuffer weight = backend.allocateBuffer((long) ROWS * 144, true);
             GpuBuffer input = backend.allocateBuffer((long) TOKENS * COLS * Float.BYTES);
             GpuBuffer output = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES)) {
            GpuWeight w = staticInt8Weight(weight);
            backend.clearLog();
            ops.matmul(w, input, 0, output, 0, TOKENS);
            List<String> names = kernelNames(backend);
            assertEquals("the checkpoint's int8 codes are staged first", GemmKernels.QUANT_MMQ_INT8_STATIC,
                    names.get(0));
            assertTrue(names.toString(), names.contains("Q4_K_MatMul_MMQ"));
            assertTrue("the fake-quantization pass is skipped: " + names,
                    !names.contains(GpuOps.ACT_FAKE_QUANT));
            assertTrue("the dynamic staging pass is skipped: " + names,
                    !names.contains(GemmKernels.QUANT_MMQ_Q8_1));
        }
    }

    @Test
    public void staticInt8FallsBackToFakeQuantizationWhenTheStagingKernelIsUnavailable() {
        try (FakeComputeBackend backend = FakeComputeBackend.builder().strictLlmArgs(true)
                .unavailableKernels(GemmKernels.QUANT_MMQ_INT8_STATIC).build();
             GpuScratch scratch = scratch(backend);
             GpuPlaceholders placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
             GpuOps ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
             GpuBuffer weight = backend.allocateBuffer((long) ROWS * 144, true);
             GpuBuffer input = backend.allocateBuffer((long) TOKENS * COLS * Float.BYTES);
             GpuBuffer output = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES)) {
            GpuWeight w = staticInt8Weight(weight);
            backend.clearLog();
            ops.matmul(w, input, 0, output, 0, TOKENS);
            List<String> names = kernelNames(backend);
            assertTrue(names.toString(), names.contains(GpuOps.ACT_FAKE_QUANT));
            assertTrue(names.toString(), names.contains(GemmKernels.QUANT_MMQ_Q8_1));
            assertTrue(names.toString(), names.contains("Q4_K_MatMul_MMQ"));
            assertTrue(names.toString(), !names.contains(GemmKernels.QUANT_MMQ_INT8_STATIC));
        }
    }

    private static GpuScratch scratch(FakeComputeBackend backend) {
        return GpuScratch.builder(backend, TOKENS).slot(ScratchSlots.X, (long) TOKENS * COLS).build();
    }

    private static GpuWeight staticInt8Weight(GpuBuffer weight) {
        InputTransform.ActQuant aq = InputTransform.ActQuant.of(ActQuantScheme.INT8_STATIC,
                new float[]{0.125f}, null, Float.POSITIVE_INFINITY);
        return GpuWeight.of("test.weight", GGMLType.Q4_K, new long[]{COLS, ROWS}, weight)
                .withInputTransform(InputTransform.of(null, aq, null));
    }

    private static List<String> kernelNames(FakeComputeBackend backend) {
        return backend.dispatches().stream().map(FakeDispatch::kernelName)
                .collect(Collectors.toList());
    }
}
