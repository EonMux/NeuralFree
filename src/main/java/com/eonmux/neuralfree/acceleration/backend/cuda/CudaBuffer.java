package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import jcuda.driver.CUdeviceptr;

/**
 * A buffer a CUDA kernel binds: device memory of a {@link CudaComputeBackend} or memory of a
 * {@link CudaHostMappedBuffer}. Kernel argument binding and the queue copy operations accept either,
 * so a weight reads the same whether it lives in VRAM or in mapped host memory.
 */
interface CudaBuffer extends GpuBuffer {

    /** The address the device reads, or null once the buffer is closed. */
    CUdeviceptr getCUDevicePtr();

    /** The backend the buffer belongs to. */
    CudaComputeBackend backend();

    /** True once the buffer is closed. */
    boolean isClosed();
}
