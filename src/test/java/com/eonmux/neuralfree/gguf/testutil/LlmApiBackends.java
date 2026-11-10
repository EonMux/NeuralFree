package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.Assume;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

/**
 * Every device of one API (CUDA, OpenCL or Vulkan) that runs the LLM kernels, for kernel parity tests that must
 * hold on each driver (OpenCL lists the NVIDIA and the Intel platform, Vulkan every GPU). {@link #open} skips the
 * calling test (JUnit assumption) when the API or such a device is absent. The upload and read helpers track their
 * buffers, which {@link #close} releases with the backends.
 */
public final class LlmApiBackends implements AutoCloseable {

    private final HWAcceleration hw;
    private final List<ComputeBackend> backends;
    private final List<GpuBuffer> buffers = new ArrayList<>();

    private LlmApiBackends(HWAcceleration hw, List<ComputeBackend> backends) {
        this.hw = hw;
        this.backends = Collections.unmodifiableList(backends);
    }

    /** The API's devices with LLM kernels; skips the calling test when there are none. */
    public static LlmApiBackends open(BackendType api) {
        switch (api) {
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            case VULKAN:
                GpuTestUtil.assumeVulkanAvailable();
                break;
            default:
                throw new IllegalArgumentException("no LLM kernel test devices for " + api);
        }
        HWAcceleration hw = new HWAcceleration(EnumSet.of(api));
        List<HardwareDevice> devices = new ArrayList<>(api == BackendType.CUDA ? hw.getCUDevices()
                : api == BackendType.OPENCL ? hw.getCLDevices() : hw.getVkDevices());
        List<ComputeBackend> backends = new ArrayList<>();
        for (HardwareDevice device : devices) {
            ComputeBackend backend = hw.createBackend(device);
            if (LlmKernels.availability(backend).available()) {
                backends.add(backend);
            } else {
                backend.close();
            }
        }
        if (backends.isEmpty()) {
            hw.shutdown();
            Assume.assumeTrue("no " + api + " device runs the LLM kernels", false);
        }
        return new LlmApiBackends(hw, backends);
    }

    /** The opened backends, in enumeration order. */
    public List<ComputeBackend> backends() {
        return backends;
    }

    /** A device buffer holding {@code data}. */
    public GpuBuffer upload(ComputeBackend backend, float[] data) {
        GpuBuffer b = backend.allocateBuffer((long) Math.max(1, data.length) * Float.BYTES, true);
        buffers.add(b);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * Float.BYTES, data));
        return b;
    }

    /** A device buffer holding {@code data} as little-endian 32-bit integers. */
    public GpuBuffer uploadInts(ComputeBackend backend, int[] data) {
        ByteBuffer bytes = ByteBuffer.allocate(Math.max(1, data.length) * Integer.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        bytes.asIntBuffer().put(data);
        GpuBuffer b = backend.allocateBuffer(bytes.capacity(), true);
        buffers.add(b);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, bytes.capacity(), bytes.array()));
        return b;
    }

    /** The first {@code n} floats of {@code b}, after the queue finishes. */
    public static float[] read(ComputeBackend backend, GpuBuffer b, int n) {
        float[] out = new float[n];
        close(backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * Float.BYTES, out));
        backend.computeQueue().finish();
        return out;
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    @Override
    public void close() {
        for (GpuBuffer b : buffers) {
            b.close();
        }
        for (ComputeBackend b : backends) {
            b.close();
        }
        hw.shutdown();
    }
}
