package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The one-token step's inputs written on the device ({@value GpuDecodeStepWriter#KERNEL}, CUDA) over the real
 * forward backend on the fake queue, whose emulation of the kernel stores its arguments as the kernel does: on a
 * CUDA backend every decode step writes {@code cell_idx}, its {@code kv_pos} row, {@code q_pos} and
 * {@code rope_pos} with one launch and no host upload, and leaves those buffers holding what the uploads of a
 * backend without the kernel leave, step after step; a multi-row step keeps its uploads.
 */
public class GpuDecodeStepWriterTest {

    private static final int STEPS = 6;

    private final List<GpuDecodeReplayTest> rigs = new ArrayList<>();

    @After
    public void tearDown() throws Exception {
        for (GpuDecodeReplayTest rig : rigs) {
            rig.tearDown();
        }
    }

    private GpuDecodeReplayTest rig(ComputeBackend.BackendType type) {
        GpuDecodeReplayTest rig = new GpuDecodeReplayTest();
        rig.backendType = type;
        if (type == ComputeBackend.BackendType.CUDA) {
            rig.capabilities = new DeviceCapabilities.Builder().deviceName("StepInputsDevice").totalVramBytes(8L << 30)
                    .maxAllocSizeBytes(2L << 30).computeUnits(60).maxClockFrequencyMhz(1500).maxWorkGroupSize(1024)
                    .driverVersion("fake-1.0").maxLocalMemoryBytes(48L * 1024)
                    .maxWorkGroupCount(new long[]{65535, 65535, 65535}).computeCapability(120).build();
        }
        rig.emulators.put(GpuDecodeStepWriter.KERNEL, d -> {
            int row = d.intArg(GpuDecodeStepWriter.ARG_ROW);
            int pos = d.intArg(GpuDecodeStepWriter.ARG_POS);
            List<FakeGpuBuffer> buffers = d.bufferArgs();
            buffers.get(0).putInt(0, row);
            buffers.get(1).putInt(row, pos);
            buffers.get(2).putInt(0, pos);
            for (int k = 0; k < 4; k++) {
                buffers.get(3).putInt(k, d.intArg(GpuDecodeStepWriter.ARG_ROPE0 + k));
            }
        });
        rigs.add(rig);
        return rig;
    }

    /** The step-input buffers' contents, in a fixed order. */
    private static int[][] contents(GpuStepInputs inputs) {
        GpuBuffer[] buffers = {inputs.cellIdx(0), inputs.kvPos(0), inputs.qPos(), inputs.ropePos()};
        int[][] out = new int[buffers.length][];
        for (int i = 0; i < buffers.length; i++) {
            FakeGpuBuffer b = (FakeGpuBuffer) buffers[i];
            out[i] = b.getInts(0, (int) (b.sizeInBytes() / Integer.BYTES));
        }
        return out;
    }

    private static long writesInto(GpuDecodeReplayTest rig, GpuStepInputs inputs) {
        List<GpuBuffer> buffers = List.of(inputs.cellIdx(0), inputs.kvPos(0), inputs.qPos(), inputs.ropePos());
        return rig.backend.ops().stream()
                .filter(op -> op.kind() == FakeOpKind.WRITE && buffers.contains(op.buffer())).count();
    }

    @Test
    public void decodeStepsWriteTheUploadsValuesWithOneLaunch() {
        GpuDecodeReplayTest cudaRig = rig(ComputeBackend.BackendType.CUDA);
        GpuDecodeReplayTest uploadRig = rig(ComputeBackend.BackendType.OPENCL);
        GpuForwardBackend cuda = cudaRig.open(true);
        GpuForwardBackend upload = uploadRig.open(true);
        GpuStepInputs cudaInputs = cudaRig.memory().stepInputs();
        GpuStepInputs uploadInputs = uploadRig.memory().stepInputs();

        cuda.prefill(new int[]{1, 2, 3}, 0);
        upload.prefill(new int[]{1, 2, 3}, 0);
        long prefillWrites = writesInto(cudaRig, cudaInputs);
        assertEquals("a multi-row step uploads its inputs", writesInto(uploadRig, uploadInputs), prefillWrites);
        assertEquals("no launch serves a multi-row step", 0,
                cudaRig.backend.dispatchedKernelNames().stream().filter(GpuDecodeStepWriter.KERNEL::equals).count());

        for (int i = 0; i < STEPS; i++) {
            cuda.decode(4 + i, 3 + i);
            upload.decode(4 + i, 3 + i);
            int[][] want = contents(uploadInputs);
            int[][] got = contents(cudaInputs);
            for (int b = 0; b < want.length; b++) {
                assertArrayEquals("step " + i + " buffer " + b, want[b], got[b]);
            }
        }
        assertEquals("the CUDA decode steps upload nothing into the step inputs", prefillWrites,
                writesInto(cudaRig, cudaInputs));
        assertEquals("one launch per decode step", STEPS,
                cudaRig.backend.dispatchedKernelNames().stream().filter(GpuDecodeStepWriter.KERNEL::equals).count());
        assertArrayEquals("both store the same sequence", uploadRig.memory().sequence().tokens(),
                cudaRig.memory().sequence().tokens());
    }

    @Test
    public void aBackendWithoutTheKernelKeepsItsUploads() {
        GpuDecodeReplayTest uploadRig = rig(ComputeBackend.BackendType.OPENCL);
        GpuForwardBackend upload = uploadRig.open(true);
        GpuStepInputs inputs = uploadRig.memory().stepInputs();
        long before = writesInto(uploadRig, inputs);
        upload.decode(4, 0);
        assertEquals("cell_idx, the kv_pos row, q_pos and rope_pos", 4, writesInto(uploadRig, inputs) - before);
    }
}
