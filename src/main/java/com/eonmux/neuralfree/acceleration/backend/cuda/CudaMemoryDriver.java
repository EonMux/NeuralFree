package com.eonmux.neuralfree.acceleration.backend.cuda;

import jcuda.driver.CUdeviceptr;

/**
 * The device-memory operations {@link CudaBufferPool} needs, so the pool's bookkeeping does not
 * depend on a device. {@link CudaDeviceMemory} implements it on the CUDA driver API; tests supply
 * their own. Every call runs with the backend's context current on the calling thread.
 */
interface CudaMemoryDriver {

    /**
     * Allocates {@code bytes} of device memory ({@code cuMemAlloc}).
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when the device is out of memory
     * @throws RuntimeException                                             any other translated driver failure
     */
    CUdeviceptr allocate(long bytes);

    /**
     * Frees an allocation of {@link #allocate} ({@code cuMemFree}).
     *
     * @throws RuntimeException the translated driver failure
     */
    void free(CUdeviceptr pointer);

    /**
     * Records a point on every stream that may still use device memory. Work enqueued on those
     * streams before this call has finished once the fence is complete.
     *
     * @throws RuntimeException the translated driver failure; nothing stays recorded
     */
    Fence recordFence();

    /**
     * Blocks until all work of the context has finished ({@code cuCtxSynchronize}).
     *
     * @throws RuntimeException the translated driver failure
     */
    void synchronize();

    /** Points recorded by {@link #recordFence()}. */
    interface Fence {

        /**
         * True once every recorded point has been reached. Never blocks.
         *
         * @throws RuntimeException the translated driver failure
         */
        boolean isComplete();

        /** Returns the recorded points to their pool. Called exactly once per fence. */
        void release();
    }
}
