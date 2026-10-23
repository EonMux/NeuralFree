package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import org.junit.Assume;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Parity-test support for the train library: opens each GPU API through {@link GpuTestUtil} (tests are skipped with
 * JUnit {@link Assume} where the API or the train kernels are unavailable), provides the CPU reference backend, and
 * asserts the tolerance classes of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.
 */
public final class TrainParity {

    /** Smallest positive normal float. */
    public static final float FLT_MIN = Float.MIN_NORMAL;

    /** The GPU APIs a parity test runs on. */
    public enum Api {
        OPENCL(ComputeBackend.BackendType.OPENCL),
        CUDA(ComputeBackend.BackendType.CUDA),
        VULKAN(ComputeBackend.BackendType.VULKAN);

        final ComputeBackend.BackendType type;

        Api(ComputeBackend.BackendType type) {
            this.type = type;
        }
    }

    private TrainParity() {
    }

    /** JUnit {@code Parameterized} rows: one per {@link Api}. */
    public static Collection<Object[]> apis() {
        List<Object[]> rows = new ArrayList<>();
        for (Api a : Api.values()) {
            rows.add(new Object[]{a});
        }
        return rows;
    }

    /**
     * An open backend of {@code api} with the train kernels available. Skips the calling test when the API has no
     * device or the train library is unavailable on it.
     */
    public static Session open(Api api) {
        switch (api) {
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            case VULKAN:
                GpuTestUtil.assumeVulkanAvailable();
                break;
            default:
                throw new IllegalArgumentException("unknown API " + api);
        }
        HWAcceleration hw = new HWAcceleration(EnumSet.of(api.type));
        HardwareDevice device = firstDevice(hw, api);
        if (device == null) {
            hw.shutdown();
            Assume.assumeTrue("no " + api + " device enumerated", false);
        }
        ComputeBackend backend = hw.createBackend(device);
        Availability a = TrainGpu.availability(backend);
        if (!a.available()) {
            hw.shutdown();
            Assume.assumeTrue("train kernels unavailable on " + api + ": " + a.reason(), false);
        }
        return new Session(hw, backend);
    }

    /** A CPU reference backend: host memory buffers, every kernel artifact declined, so tensor ops take their CPU path. */
    public static ComputeBackend hostBackend() {
        return new HostBackend();
    }

    private static HardwareDevice firstDevice(HWAcceleration hw, Api api) {
        switch (api) {
            case OPENCL:
                return hw.getCLDevices().isEmpty() ? null : hw.getCLDevices().get(0);
            case CUDA:
                return hw.getCUDevices().isEmpty() ? null : hw.getCUDevices().get(0);
            default:
                return hw.getVkDevices().isEmpty() ? null : hw.getVkDevices().get(0);
        }
    }

    /** One GPU backend opened for a test; {@link #close()} shuts the enumeration down, closing the backend. */
    public static final class Session implements AutoCloseable {
        private final HWAcceleration hw;
        private final ComputeBackend backend;

        Session(HWAcceleration hw, ComputeBackend backend) {
            this.hw = hw;
            this.backend = backend;
        }

        public ComputeBackend backend() {
            return backend;
        }

        public TrainGpu gpu() {
            return TrainGpu.forBackend(backend);
        }

        @Override
        public void close() {
            hw.shutdown();
        }
    }

    // ------------------------------------------------------------------ inputs

    /** {@code n} floats uniform in {@code [lo, hi)} from a seeded stream. */
    public static float[] uniform(long seed, int n, float lo, float hi) {
        Random r = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = lo + (hi - lo) * r.nextFloat();
        }
        return out;
    }

    // ------------------------------------------------------------------ tolerance classes

    /** Bit-identical class: data movement, casts, compare, Java max/min, arg-reductions. */
    public static void assertBitIdentical(String what, byte[] ref, byte[] gpu) {
        assertEquals(what + ": length", ref.length, gpu.length);
        for (int i = 0; i < ref.length; i++) {
            if (ref[i] != gpu[i]) {
                fail(what + ": byte " + i + " differs: ref 0x" + Integer.toHexString(ref[i] & 0xff) + ", gpu 0x"
                        + Integer.toHexString(gpu[i] & 0xff));
            }
        }
    }

    /** Bit-identical class on float bit patterns (NaN payloads and signed zeros included). */
    public static void assertBitIdentical(String what, float[] ref, float[] gpu) {
        assertEquals(what + ": length", ref.length, gpu.length);
        for (int i = 0; i < ref.length; i++) {
            if (Float.floatToRawIntBits(ref[i]) != Float.floatToRawIntBits(gpu[i])) {
                fail(what + ": element " + i + " differs: ref " + ref[i] + ", gpu " + gpu[i]);
            }
        }
    }

    /** Single-rounding class: bit-identical, except that a reference below FLT_MIN may come back as a signed zero. */
    public static void assertSingleRounding(String what, float[] ref, float[] gpu) {
        assertEquals(what + ": length", ref.length, gpu.length);
        for (int i = 0; i < ref.length; i++) {
            boolean same = Float.floatToRawIntBits(ref[i]) == Float.floatToRawIntBits(gpu[i])
                    || (Float.isNaN(ref[i]) && Float.isNaN(gpu[i]))
                    || (Math.abs(ref[i]) < FLT_MIN && gpu[i] == 0.0f);
            if (!same) {
                fail(what + ": element " + i + " differs: ref " + ref[i] + ", gpu " + gpu[i]);
            }
        }
    }

    /** Division and square-root class: within 4 ulp of the reference or within FLT_MIN; NaN matches NaN. */
    public static void assertDivSqrt(String what, float[] ref, float[] gpu) {
        assertEquals(what + ": length", ref.length, gpu.length);
        for (int i = 0; i < ref.length; i++) {
            if (!closeOrSpecial(ref[i], gpu[i], Math.max(4.0 * Math.ulp(ref[i]), FLT_MIN))) {
                fail(what + ": element " + i + ": ref " + ref[i] + ", gpu " + gpu[i] + " beyond 4 ulp");
            }
        }
    }

    /** Transcendental class: {@code |gpu - ref| <= 1e-5 * max(1, |ref|)}; NaN and infinities match exactly. */
    public static void assertTranscendental(String what, float[] ref, float[] gpu) {
        assertEquals(what + ": length", ref.length, gpu.length);
        for (int i = 0; i < ref.length; i++) {
            if (!closeOrSpecial(ref[i], gpu[i], 1e-5 * Math.max(1.0, Math.abs(ref[i])))) {
                fail(what + ": element " + i + ": ref " + ref[i] + ", gpu " + gpu[i]);
            }
        }
    }

    /**
     * Reduction class against a double-precision reference: {@code |gpu - ref64| <= 1e-5 * sqrt(K) * sum|terms| +
     * K * FLT_MIN}, with {@code K} the reduction length and {@code sumAbsTerms} the sum of absolute terms.
     */
    public static void assertReduction(String what, double ref64, float gpu, int k, double sumAbsTerms) {
        double bound = 1e-5 * Math.sqrt(k) * sumAbsTerms + k * (double) FLT_MIN;
        if (!(Math.abs(gpu - ref64) <= bound)) {
            fail(what + ": gpu " + gpu + ", ref64 " + ref64 + ", bound " + bound);
        }
    }

    private static boolean closeOrSpecial(float ref, float gpu, double bound) {
        if (Float.isNaN(ref) || Float.isNaN(gpu)) {
            return Float.isNaN(ref) && Float.isNaN(gpu);
        }
        if (Float.isInfinite(ref) || Float.isInfinite(gpu)) {
            return ref == gpu;
        }
        return Math.abs((double) gpu - ref) <= bound;
    }

    // ------------------------------------------------------------------ CPU reference backend

    /** Host-memory backend that declines every kernel artifact, so tensor ops run their CPU path on it. */
    static final class HostBackend implements ComputeBackend {
        private static final AtomicLong IDS = new AtomicLong();

        private final DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .deviceName("host reference").maxWorkGroupSize(1024).build();
        private final HostQueue compute = new HostQueue(ComputeQueue.QueueType.COMPUTE);
        private final HostQueue transfer = new HostQueue(ComputeQueue.QueueType.TRANSFER);

        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return caps;
        }

        @Override
        public ComputeQueue computeQueue() {
            return compute;
        }

        @Override
        public ComputeQueue transferQueue() {
            return transfer;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            if (sizeInBytes <= 0 || sizeInBytes > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("host buffer size must be in [1, 2^31-1], got " + sizeInBytes);
            }
            return new HostBuffer(IDS.incrementAndGet(), new byte[(int) sizeInBytes]);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            return allocateBuffer(sizeInBytes, false);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            throw new UnsupportedKernelException("the host reference backend runs no kernels");
        }

        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            throw new UnsupportedKernelException("the host reference backend runs no kernels");
        }

        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) {
            return false;
        }

        @Override
        public void close() {
        }
    }

    static final class HostBuffer implements GpuBuffer {
        private final long id;
        final byte[] data;
        boolean closed;

        HostBuffer(long id, byte[] data) {
            this.id = id;
            this.data = data;
        }

        @Override
        public long sizeInBytes() {
            return data.length;
        }

        @Override
        public long bufferId() {
            return id;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static final class DoneEvent implements GpuEvent {
        @Override
        public void awaitCompletion() {
        }

        @Override
        public boolean isComplete() {
            return true;
        }

        @Override
        public void close() {
        }
    }

    /** Executes every command at enqueue time. */
    private static final class HostQueue implements ComputeQueue {
        private final QueueType type;

        HostQueue(QueueType type) {
            this.type = type;
        }

        @Override
        public QueueType getQueueType() {
            return type;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long dstOffset, long size, byte[] src, GpuEvent... waits) {
            System.arraycopy(src, 0, range(dst, dstOffset, size), (int) dstOffset, (int) size);
            return new DoneEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long dstOffset, long size, float[] src, GpuEvent... waits) {
            ByteBuffer view = ByteBuffer.wrap(range(dst, dstOffset, size)).order(ByteOrder.nativeOrder());
            for (int i = 0; i < size / Float.BYTES; i++) {
                view.putFloat((int) dstOffset + i * Float.BYTES, src[i]);
            }
            return new DoneEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long srcOffset, long size, byte[] dst, GpuEvent... waits) {
            System.arraycopy(range(src, srcOffset, size), (int) srcOffset, dst, 0, (int) size);
            return new DoneEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long srcOffset, long size, float[] dst, GpuEvent... waits) {
            ByteBuffer view = ByteBuffer.wrap(range(src, srcOffset, size)).order(ByteOrder.nativeOrder());
            for (int i = 0; i < size / Float.BYTES; i++) {
                dst[i] = view.getFloat((int) srcOffset + i * Float.BYTES);
            }
            return new DoneEvent();
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                      GpuEvent... waits) {
            throw new UnsupportedOperationException("the host reference backend runs no kernels");
        }

        @Override
        public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset, long size,
                                      GpuEvent... waits) {
            System.arraycopy(range(src, srcOffset, size), (int) srcOffset, range(dst, dstOffset, size),
                    (int) dstOffset, (int) size);
        }

        @Override
        public GpuEvent enqueueMarker(GpuEvent... waits) {
            return new DoneEvent();
        }

        @Override
        public void flush() {
        }

        @Override
        public void finish() {
        }

        @Override
        public void close() {
        }

        private static byte[] range(GpuBuffer b, long offset, long size) {
            byte[] data = ((HostBuffer) b).data;
            if (((HostBuffer) b).closed) {
                throw new IllegalStateException("host buffer " + b.bufferId() + " is closed");
            }
            if (offset < 0 || size < 0 || offset + size > data.length) {
                throw new IllegalArgumentException("range [" + offset + ", " + (offset + size)
                        + ") outside host buffer of " + data.length + " bytes: " + Arrays.toString(new long[]{offset, size}));
            }
            return data;
        }
    }
}
