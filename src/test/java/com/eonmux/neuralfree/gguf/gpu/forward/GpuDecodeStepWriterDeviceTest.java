package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The kernel {@value GpuDecodeStepWriter#KERNEL} on each device (CUDA, OpenCL and Vulkan on the NVIDIA GPU and the
 * Intel iGPU, skipping what the machine lacks): a sliding-window ring of step inputs written by it holds, after every
 * one-token step and through the ring's wraps, the {@code cell_idx}, {@code kv_pos}, {@code q_pos} and
 * {@code rope_pos} contents that the host uploads leave in a twin ring on the same device.
 */
@RunWith(Parameterized.class)
public class GpuDecodeStepWriterDeviceTest {

    @Parameterized.Parameters(name = "{0} {1}")
    public static List<Object[]> cases() {
        return List.of(new Object[]{EngineConfig.BackendMode.CUDA, ""},
                new Object[]{EngineConfig.BackendMode.OPENCL, "NVIDIA"},
                new Object[]{EngineConfig.BackendMode.OPENCL, "Intel"},
                new Object[]{EngineConfig.BackendMode.VULKAN, "NVIDIA"},
                new Object[]{EngineConfig.BackendMode.VULKAN, "Intel"});
    }

    private final EngineConfig.BackendMode backend;
    private final String device;

    public GpuDecodeStepWriterDeviceTest(EngineConfig.BackendMode backend, String device) {
        this.backend = backend;
        this.device = device;
    }

    private static final int RING_ROWS = 16;
    private static final int N_SWA = 8;
    private static final int UBATCH = 8;
    private static final int STEPS = 3 * RING_ROWS + 5;

    private static Path model;

    @BeforeClass
    public static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void theKernelWritesWhatTheUploadsWriteThroughARingsWraps() throws IOException {
        switch (backend) {
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            default:
                GpuTestUtil.assumeVulkanAvailable();
                break;
        }
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig config = EngineConfig.builder().backend(backend).deviceFilter(device)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(memory).build();
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), config)) {
            Assume.assumeTrue("the model runs on the GPU engine", gpu.forwardBackend() instanceof GpuForwardBackend);
            GpuForwardContext ctx = ((GpuForwardBackend) gpu.forwardBackend()).context();
            ComputeQueue queue = ctx.queue();
            MemoryPlan.KvGroup ring = MemoryPlan.KvGroup.of(0, RING_ROWS, true,
                    KvWindow.of(SwaType.STANDARD, N_SWA, true));
            SequenceState uploadedSeq = new SequenceState(STEPS + 1);
            SequenceState writtenSeq = new SequenceState(STEPS + 1);
            try (DeviceStepInputs uploaded = DeviceStepInputs.allocate(ctx.backend(), queue, List.of(ring), UBATCH,
                    UBATCH, MropeTextPositions.PPP0, uploadedSeq, null);
                 DeviceStepInputs written = DeviceStepInputs.allocate(ctx.backend(), queue, List.of(ring), UBATCH,
                         UBATCH, MropeTextPositions.PPP0, writtenSeq, null);
                 GpuDecodeStepWriter writer = GpuDecodeStepWriter.bind(ctx.ops(), written)) {
                assertNotNull(backend + " " + device + " provides " + GpuDecodeStepWriter.KERNEL, writer);
                for (int pos = 0; pos < STEPS; pos++) {
                    uploaded.upload(pos, 1);
                    assertTrue("position " + pos + " is written on the device", written.uploadDecodeStep(pos, writer));
                    queue.finish();
                    assertArrayEquals("cell_idx at " + pos, ints(queue, uploaded.cellIdx(0), 1),
                            ints(queue, written.cellIdx(0), 1));
                    assertArrayEquals("kv_pos at " + pos, ints(queue, uploaded.kvPos(0), RING_ROWS),
                            ints(queue, written.kvPos(0), RING_ROWS));
                    assertArrayEquals("q_pos at " + pos, ints(queue, uploaded.qPos(), 1), ints(queue, written.qPos(), 1));
                    assertArrayEquals("rope_pos at " + pos, ints(queue, uploaded.ropePos(), 4),
                            ints(queue, written.ropePos(), 4));
                    assertEquals("the wrap mark at " + pos, uploadedSeq.wrapped(), writtenSeq.wrapped());
                }
                assertTrue("the ring wrapped", writtenSeq.wrapped());
            }
        }
    }

    /** The first {@code n} ints of {@code buffer}, read back after the queue drained. */
    private static int[] ints(ComputeQueue queue, GpuBuffer buffer, int n) {
        byte[] bytes = new byte[n * Integer.BYTES];
        GpuEvent read = queue.enqueueReadBuffer(buffer, 0, bytes.length, bytes);
        queue.finish();
        if (read != null) {
            read.close();
        }
        int[] out = new int[n];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out);
        return out;
    }
}
