package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * A {@link FakeComputeBackend} whose named kernels have their own work-group limit below the device's, as a
 * register-heavy OpenCL kernel reports a {@code CL_KERNEL_WORK_GROUP_SIZE} below {@code CL_DEVICE_MAX_WORK_GROUP_SIZE}:
 * preparing one of them for a larger work-group fails with {@link UnsupportedKernelException}. Everything else,
 * including the kernel handles, queues and the dispatch log, is the wrapped fake's. Closing this view leaves the fake
 * open.
 */
final class KernelLimitBackend implements ComputeBackend {

    private final FakeComputeBackend fake;
    private final long limit;
    private final Set<String> limited;

    KernelLimitBackend(FakeComputeBackend fake, long limit, String... limited) {
        this.fake = Objects.requireNonNull(fake, "fake");
        if (limit < 1) {
            throw new IllegalArgumentException("work-group limit must be positive, got " + limit);
        }
        this.limit = limit;
        this.limited = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(limited)));
    }

    @Override
    public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
        long items = 1;
        for (long l : localWorkSize) {
            items *= l;
        }
        if (limited.contains(kernel.kernelName()) && items > limit) {
            throw new UnsupportedKernelException("kernel " + kernel.kernelName() + " allows work-groups of at most "
                    + limit + " work-items, requested " + Arrays.toString(localWorkSize));
        }
        fake.prepareKernel(kernel, localWorkSize);
    }

    @Override
    public BackendType backendType() {
        return fake.backendType();
    }

    @Override
    public DeviceCapabilities capabilities() {
        return fake.capabilities();
    }

    @Override
    public ComputeQueue computeQueue() {
        return fake.computeQueue();
    }

    @Override
    public ComputeQueue transferQueue() {
        return fake.transferQueue();
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
        return fake.allocateBuffer(sizeInBytes, readOnly);
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return fake.allocateBuffer(sizeInBytes);
    }

    @Override
    public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
        return fake.compileKernel(kernelName, kernelSource, buildOptions);
    }

    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        return fake.loadKernel(artifact);
    }

    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        return fake.supportsArtifact(kind);
    }

    @Override
    public KernelHandle cloneKernel(KernelHandle template) {
        return fake.cloneKernel(template);
    }

    @Override
    public ComputeQueue createComputeQueue() {
        return fake.createComputeQueue();
    }

    @Override
    public long getAllocatedBytes() {
        return fake.getAllocatedBytes();
    }

    @Override
    public long getFreeVramBytes() {
        return fake.getFreeVramBytes();
    }

    @Override
    public void close() {
    }
}
