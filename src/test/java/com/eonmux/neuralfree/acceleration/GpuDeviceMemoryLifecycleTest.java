package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

/**
 * Device-memory regression for the open/close lifecycle: cycling an enumeration and a backend through creation, the
 * kernel libraries the engine and the tensor dispatch load, a buffer round trip on a caller-owned queue and shutdown
 * must return the device's free memory to its level before the cycle. This guards a test run that shares one JVM: a
 * cycle that leaks its context (an NVIDIA OpenCL context holds about 300 MB of device memory), its queues, programs,
 * modules or buffers fails here, instead of exhausting the device many classes later (CUDA
 * {@code cuDevicePrimaryCtxRetain} out of memory, OpenCL queue creation failing on a context that could not be made).
 *
 * <p>Free memory is the CUDA driver's device-wide reading ({@code cuMemGetInfo} through the backend's memory query),
 * the only free-memory query OpenCL and Vulkan devices share with it; the test skips on a host without CUDA. Other
 * processes on the device move that reading too, but only in some cycles, while a leak loses memory in every cycle,
 * so the assertion is on the median loss of a cycle.</p>
 */
@RunWith(Parameterized.class)
public class GpuDeviceMemoryLifecycleTest {

    /** Measured open/close cycles per test method; odd, so the median is one cycle's loss. */
    private static final int CYCLES = 7;
    /** Cycles run before the baseline reading, so one-time driver warm-up is not counted as loss. */
    private static final int WARMUP_CYCLES = 2;
    /** Median free-memory loss of a cycle that is still driver noise (page tables, allocator granularity). */
    private static final long CYCLE_TOLERANCE_BYTES = 96L << 20;
    /** Size of the buffers each cycle writes, copies and reads. */
    private static final long WORK_BUFFER_BYTES = 8L << 20;

    private final TrainParity.Api api;
    private final List<Object> owned = new ArrayList<>();

    public GpuDeviceMemoryLifecycleTest(TrainParity.Api api) {
        this.api = api;
    }

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> apis() {
        List<Object[]> rows = new ArrayList<>();
        for (Object[] row : TrainParity.apis()) {
            rows.add(row);
        }
        return rows;
    }

    @Before
    public void assumeDeviceAvailable() {
        switch (api) {
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            default:
                GpuTestUtil.assumeVulkanAvailable();
                break;
        }
    }

    @After
    public void closeOwned() {
        for (int i = owned.size() - 1; i >= 0; i--) {
            try {
                ((AutoCloseable) owned.get(i)).close();
            } catch (RuntimeException suppressed) {
                // Best effort; the assertions carry the failure.
            } catch (Exception suppressed) {
                // See above.
            }
        }
        owned.clear();
    }

    @Test
    public void openCloseCyclesReturnDeviceMemory() {
        FreeMemoryProbe probe = FreeMemoryProbe.open();
        owned.add(probe);
        probe.warmUp();
        // One-time driver steps (JIT caches, page-table growth for the first builds of the programs the
        // cycles load) must happen before the baseline reading, so only per-cycle loss is measured.
        for (int i = 0; i < WARMUP_CYCLES; i++) {
            workCycle();
        }

        long[] lost = new long[CYCLES];
        long previous = probe.freeBytes();
        for (int i = 0; i < CYCLES; i++) {
            workCycle();
            long now = probe.freeBytes();
            lost[i] = previous - now;
            previous = now;
        }
        long[] sorted = lost.clone();
        Arrays.sort(sorted);
        long median = sorted[CYCLES / 2];
        assertTrue(api + ": an open/close cycle loses a median " + mb(median) + " MB of device memory (allowance "
                        + mb(CYCLE_TOLERANCE_BYTES) + " MB, per-cycle losses in MB " + mbList(lost)
                        + "); a backend cycle leaks",
                median <= CYCLE_TOLERANCE_BYTES);
    }

    /**
     * One lifecycle of this API's backend, the shape every parity test method runs: enumeration, backend, the train
     * library on the tensor dispatch (kernel modules or the compiled program, placeholder and error-slot buffers),
     * an LLM kernel library load, a buffer round trip on a caller-owned queue, then shutdown of everything.
     */
    private void workCycle() {
        HWAcceleration hw = new HWAcceleration(EnumSet.of(backendTypeOf(api)));
        try {
            HardwareDevice device = firstDevice(hw);
            assumeTrue("no " + api + " device enumerated", device != null);
            ComputeBackend backend = hw.createBackend(device);
            try {
                // The tensor dispatch's train library (compiled program or kernel modules, placeholder and
                // error-slot buffers) and the LLM library the engine load: the heavy loads of a test method.
                // Both are optional per API; the cycle then covers the plain buffer path.
                if (TrainGpu.availability(backend).available()) {
                    assertNotNull(TrainGpu.forBackend(backend));
                }
                if (LlmKernels.availability(backend).available()) {
                    LlmKernels.open(backend);
                }
                roundTrip(backend);
            } finally {
                backend.close();
            }
        } finally {
            hw.shutdown();
        }
    }

    /** The API a {@link TrainParity.Api} row enumerates; its type field is package-private. */
    private static ComputeBackend.BackendType backendTypeOf(TrainParity.Api api) {
        switch (api) {
            case OPENCL:
                return ComputeBackend.BackendType.OPENCL;
            case CUDA:
                return ComputeBackend.BackendType.CUDA;
            default:
                return ComputeBackend.BackendType.VULKAN;
        }
    }

    private HardwareDevice firstDevice(HWAcceleration hw) {
        switch (api) {
            case OPENCL:
                return hw.getCLDevices().isEmpty() ? null : hw.getCLDevices().get(0);
            case CUDA:
                return hw.getCUDevices().isEmpty() ? null : hw.getCUDevices().get(0);
            default:
                return VkDeviceRanking.fastestDevice(hw.getVkDevices(), null);
        }
    }

    /** Allocates two buffers, writes floats to one, copies to the second, reads back, and closes everything. */
    private void roundTrip(ComputeBackend backend) {
        int floats = (int) (WORK_BUFFER_BYTES / Float.BYTES);
        float[] data = new float[floats];
        for (int i = 0; i < floats; i++) {
            data[i] = i;
        }
        float[] readBack = new float[floats];
        try (GpuBuffer a = backend.allocateBuffer(WORK_BUFFER_BYTES);
             GpuBuffer b = backend.allocateBuffer(WORK_BUFFER_BYTES);
             ComputeQueue queue = backend.createComputeQueue()) {
            GpuEvent write = queue.enqueueWriteBuffer(a, 0, WORK_BUFFER_BYTES, data);
            GpuEvent read = null;
            try {
                write.awaitCompletion();
                queue.enqueueCopyBuffer(a, 0, b, 0, WORK_BUFFER_BYTES);
                read = queue.enqueueReadBuffer(b, 0, WORK_BUFFER_BYTES, readBack);
                read.awaitCompletion();
            } finally {
                write.close();
                if (read != null) {
                    read.close();
                }
            }
            queue.finish();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError(api + ": buffer round trip was interrupted", e);
        } catch (Exception e) {
            throw new AssertionError(api + ": buffer round trip of a fresh backend failed", e);
        }
        for (int i = 0; i < floats; i++) {
            if (readBack[i] != (float) i) {
                throw new AssertionError(api + ": read back " + readBack[i] + " at " + i);
            }
        }
    }

    private static String mb(long bytes) {
        return String.valueOf(bytes / (1024L * 1024L));
    }

    private static String mbList(long[] bytes) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < bytes.length; i++) {
            sb.append(i == 0 ? "" : ", ").append(mb(bytes[i]));
        }
        return sb.append(']').toString();
    }

    /**
     * Free device memory as the CUDA driver reports it ({@code cuMemGetInfo}), through one CUDA backend opened for
     * the whole test. The probe allocates nothing, so its primary context is a constant part of every reading;
     * holding it also keeps the context from being destroyed and recreated between cycles of the CUDA row.
     * {@link #warmUp} takes one reading first, so lazy driver initialization is not counted against a cycle.
     */
    private static final class FreeMemoryProbe implements AutoCloseable {
        private HWAcceleration hw;
        private ComputeBackend backend;

        static FreeMemoryProbe open() {
            GpuTestUtil.assumeCudaAvailable();
            FreeMemoryProbe probe = new FreeMemoryProbe();
            probe.hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            if (probe.hw.getCUDevices().isEmpty()) {
                probe.close();
                assumeTrue("no CUDA device although the availability check saw one", false);
            }
            probe.backend = probe.hw.createBackend(probe.hw.getCUDevices().get(0));
            return probe;
        }

        /** One free-memory query, so lazy driver initialization does not skew the first measured reading. */
        void warmUp() {
            backend.getFreeVramBytes();
        }

        /** The free bytes the probe reads. */
        long freeBytes() {
            return backend.getFreeVramBytes();
        }

        @Override
        public void close() {
            if (backend != null) {
                backend.close();
                backend = null;
            }
            if (hw != null) {
                hw.shutdown();
                hw = null;
            }
        }
    }
}
