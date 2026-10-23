package com.eonmux.neuralfree.acceleration.backend.cuda;

import jcuda.driver.CUdeviceptr;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;

/**
 * {@link CudaMemoryDriver} on the CUDA driver API for one {@link CudaComputeBackend}. A fence is one
 * pooled event recorded on the stream of every open queue of the backend (its compute and transfer
 * queues and the caller-owned queues of {@link CudaComputeBackend#createComputeQueue()}); a queue
 * that has been released drained its stream and needs none.
 */
final class CudaDeviceMemory implements CudaMemoryDriver {

    private final CudaComputeBackend backend;

    /**
     * @throws IllegalArgumentException when {@code backend} is null
     */
    CudaDeviceMemory(CudaComputeBackend backend) {
        if (backend == null) {
            throw new IllegalArgumentException("CUDA backend must not be null");
        }
        this.backend = backend;
    }

    @Override
    public CUdeviceptr allocate(long bytes) {
        CUdeviceptr pointer = new CUdeviceptr();
        CudaErrors.check("cuMemAlloc", bytes, () -> cuMemAlloc(pointer, bytes));
        return pointer;
    }

    @Override
    public void free(CUdeviceptr pointer) {
        CudaErrors.check("cuMemFree", () -> cuMemFree(pointer));
    }

    @Override
    public Fence recordFence() {
        List<CudaGpuEvent> events = new ArrayList<>();
        try {
            for (CudaComputeQueue queue : backend.openQueues()) {
                CudaGpuEvent event = queue.recordFence();
                if (event != null) {
                    events.add(event);
                }
            }
        } catch (RuntimeException e) {
            for (CudaGpuEvent event : events) {
                event.close();
            }
            throw e;
        }
        return new EventFence(events);
    }

    @Override
    public void synchronize() {
        CudaErrors.check("cuCtxSynchronize", () -> cuCtxSynchronize());
    }

    /** The events of one fence; complete when every event is. */
    private static final class EventFence implements Fence {
        private final List<CudaGpuEvent> events;

        EventFence(List<CudaGpuEvent> events) {
            this.events = Collections.unmodifiableList(events);
        }

        @Override
        public boolean isComplete() {
            for (CudaGpuEvent event : events) {
                if (!event.isComplete()) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public void release() {
            for (CudaGpuEvent event : events) {
                event.close();
            }
        }
    }
}
