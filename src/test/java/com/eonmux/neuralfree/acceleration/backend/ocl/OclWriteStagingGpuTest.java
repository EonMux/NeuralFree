package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * On every OpenCL device here, a run of large uploads that never reads or finishes, 32 writes of 16 MiB, keeps the
 * staging of its writes in flight within {@link OclPendingWrites#MAX_BYTES} instead of holding all 512 MiB until the
 * next synchronization, and every buffer still receives its own data. Covered twice: on the queue directly, and
 * through {@link Tensor#fromArray} with the train context open, whose writes return without finishing the queue.
 */
public class OclWriteStagingGpuTest {

    private static final int WRITES = 32;
    private static final int FLOATS = 4 << 20;
    private static final long BYTES = (long) FLOATS * Float.BYTES;

    private HWAcceleration hw;
    private List<CLDevice> devices;

    @Before
    public void setUp() {
        hw = new HWAcceleration();
        devices = hw.getCLDevices();
        Assume.assumeTrue("no OpenCL device", !devices.isEmpty());
    }

    @After
    public void tearDown() {
        if (hw != null) {
            hw.shutdown();
        }
    }

    @Test
    public void uploadsWithoutAFinishKeepTheirStagingWithinTheBudget() {
        for (CLDevice device : devices) {
            try (OclComputeBackend backend = new OclComputeBackend(device)) {
                OclComputeQueue queue = (OclComputeQueue) backend.transferQueue();
                String name = String.valueOf(backend.capabilities().getDeviceName());
                List<GpuBuffer> buffers = new ArrayList<>();
                float[] host = new float[FLOATS];
                try {
                    long most = 0;
                    for (int i = 0; i < WRITES; i++) {
                        GpuBuffer buffer = backend.allocateBuffer(BYTES);
                        buffers.add(buffer);
                        Arrays.fill(host, i + 0.5f);
                        try (GpuEvent write = queue.enqueueWriteBuffer(buffer, 0, BYTES, host)) {
                            most = Math.max(most, queue.writeStagingBytesInFlight());
                        }
                    }
                    assertTrue(name + ": write staging in flight stays within the budget, at most " + most,
                            most <= OclPendingWrites.MAX_BYTES);
                    assertTrue(name + ": staging buffers in flight, " + queue.stagingBuffersInFlight(),
                            queue.stagingBuffersInFlight() <= OclPendingWrites.MAX_BYTES / BYTES);
                    queue.finish();
                    assertEquals(name + ": a finish drops every pending write", 0, queue.writeStagingBytesInFlight());
                    checkContents(name, queue, buffers);
                } finally {
                    for (GpuBuffer b : buffers) {
                        b.close();
                    }
                }
            }
        }
    }

    @Test
    public void tensorUploadsWithTheTrainContextOpenKeepTheirStagingWithinTheBudget() {
        for (CLDevice device : devices) {
            try (OclComputeBackend backend = new OclComputeBackend(device)) {
                if (TrainGpu.forBackend(backend) == null) {
                    continue;
                }
                OclComputeQueue queue = (OclComputeQueue) backend.transferQueue();
                String name = String.valueOf(backend.capabilities().getDeviceName());
                List<Tensor> tensors = new ArrayList<>();
                float[] host = new float[FLOATS];
                try {
                    int most = 0;
                    for (int i = 0; i < WRITES; i++) {
                        Arrays.fill(host, i + 0.25f);
                        tensors.add(Tensor.fromArray(backend, host));
                        most = Math.max(most, queue.stagingBuffersInFlight());
                    }
                    assertTrue(name + ": staging buffers in flight after asynchronous tensor writes, at most " + most,
                            most <= OclPendingWrites.MAX_BYTES / BYTES);
                    for (int i = 0; i < WRITES; i++) {
                        float[] got = tensors.get(i).toFloatArray();
                        assertEquals(name + ": tensor " + i + " first element", i + 0.25f, got[0], 0f);
                        assertEquals(name + ": tensor " + i + " last element", i + 0.25f, got[FLOATS - 1], 0f);
                    }
                } finally {
                    for (Tensor t : tensors) {
                        t.close();
                    }
                    TrainGpu.release(backend);
                }
            }
        }
    }

    /** Reads the first and last float of every buffer: buffer {@code i} holds {@code i + 0.5}. */
    private static void checkContents(String name, OclComputeQueue queue, List<GpuBuffer> buffers) {
        float[] word = new float[1];
        for (int i = 0; i < buffers.size(); i++) {
            for (long offset : new long[]{0, BYTES - Float.BYTES}) {
                try (GpuEvent read = queue.enqueueReadBuffer(buffers.get(i), offset, Float.BYTES, word)) {
                    queue.finish();
                }
                assertEquals(name + ": buffer " + i + " at byte " + offset, i + 0.5f, word[0], 0f);
            }
        }
    }
}
