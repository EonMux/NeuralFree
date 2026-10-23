package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import jcuda.driver.CUdeviceptr;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * A {@link CudaMemoryDriver} over a fake device of fixed capacity, for pool tests that need no
 * device. Fences complete when a test says so or on {@link #synchronize()}; pointers are distinct
 * objects compared by identity.
 */
final class FakeCudaMemoryDriver implements CudaMemoryDriver {

    static final String FENCE_FAILURE = "fence recording failed";

    long capacity = Long.MAX_VALUE;
    boolean failFence;
    int allocCalls;
    int freeCalls;
    int syncCalls;
    private final Map<CUdeviceptr, Long> live = new IdentityHashMap<>();
    private final List<FakeFence> fences = new ArrayList<>();
    private long used;

    @Override
    public CUdeviceptr allocate(long bytes) {
        allocCalls++;
        if (bytes > capacity - used) {
            throw new GpuOutOfMemoryException("fake device out of memory", bytes, "CUDA");
        }
        CUdeviceptr pointer = new CUdeviceptr();
        live.put(pointer, bytes);
        used += bytes;
        return pointer;
    }

    @Override
    public void free(CUdeviceptr pointer) {
        freeCalls++;
        Long bytes = live.remove(pointer);
        if (bytes == null) {
            throw new IllegalStateException("free of an allocation the fake device does not hold");
        }
        used -= bytes;
    }

    @Override
    public Fence recordFence() {
        if (failFence) {
            throw new IllegalStateException(FENCE_FAILURE);
        }
        FakeFence fence = new FakeFence();
        fences.add(fence);
        return fence;
    }

    @Override
    public void synchronize() {
        syncCalls++;
        completeAll();
    }

    /** Marks every fence recorded so far complete. */
    void completeAll() {
        for (FakeFence fence : fences) {
            fence.complete = true;
        }
    }

    /** The fence of the latest release. */
    FakeFence lastFence() {
        return fences.get(fences.size() - 1);
    }

    /** Bytes the fake device holds. */
    long liveBytes() {
        return used;
    }

    /** A fence that completes when a test sets {@link #complete}; releasing it twice fails. */
    static final class FakeFence implements CudaMemoryDriver.Fence {
        boolean complete;
        int releases;

        @Override
        public boolean isComplete() {
            return complete;
        }

        @Override
        public void release() {
            releases++;
            if (releases > 1) {
                throw new IllegalStateException("fence released twice");
            }
        }
    }
}
