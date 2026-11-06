package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;

/**
 * The device-memory operations {@link HipBufferPool} needs, so the pool's bookkeeping does not
 * depend on a device. {@code HipDeviceMemory} implements it on the HIP runtime; tests supply
 * their own. Every call runs with the backend's device current on the calling thread.
 */
interface HipMemoryDriver {

    /**
     * Allocates {@code bytes} of device memory ({@code hipMalloc}).
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when the device is out of memory
     * @throws RuntimeException                                           any other translated driver failure
     */
    long allocate(long bytes);

    /**
     * Frees an allocation of {@link #allocate} ({@code hipFree}).
     *
     * @throws RuntimeException the translated driver failure
     */
    void free(long pointer);

    /**
     * Records a point on every stream that may still use device memory. Work enqueued on those
     * streams before this call has finished once the fence is complete.
     *
     * @throws RuntimeException the translated driver failure; nothing stays recorded
     */
    Fence recordFence();

    /**
     * Blocks until all work of the device has finished ({@code hipDeviceSynchronize}).
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
