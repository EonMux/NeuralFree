package com.eonmux.neuralfree.acceleration.backend;

/**
 * Optional device capabilities that kernels and the engine query through
 * {@link DeviceCapabilities#hasFeature(DeviceFeature)}.
 *
 * <p>A backend reports a feature only when the device supports it <em>and</em>
 * the backend has enabled it, so a reported feature is always usable.</p>
 */
public enum DeviceFeature {
    /** Subgroup (warp, wavefront) arithmetic reductions and broadcasts are usable in kernels. */
    SUBGROUP_ARITHMETIC,
    /** Kernels may read and write 8-bit values in storage buffers. */
    STORAGE_8BIT,
    /** Kernels may read and write 16-bit values in storage buffers. */
    STORAGE_16BIT,
    /** 8-bit integer arithmetic is available in kernels. */
    SHADER_INT8,
    /** 64-bit integer arithmetic is available in kernels. */
    SHADER_INT64,
    /** 16-bit float arithmetic is available in kernels. */
    SHADER_FLOAT16,
    /** Descriptors can be pushed directly into the command stream (Vulkan {@code VK_KHR_push_descriptor}). */
    PUSH_DESCRIPTORS,
    /** The driver reports a per-heap memory budget (Vulkan {@code VK_EXT_memory_budget}). */
    MEMORY_BUDGET,
    /** Timeline semaphores are available for queue synchronisation. */
    TIMELINE_SEMAPHORE,
    /** Dispatches can require full subgroups (Vulkan {@code computeFullSubgroups}). */
    FULL_SUBGROUPS,
    /** Packed integer dot-product instructions are available. */
    INTEGER_DOT_PRODUCT,
    /**
     * Matrix multiply-accumulate on the tensor cores is usable in kernels: CUDA compute capability 8.0+
     * ({@code mma.sync}), Vulkan {@code VK_KHR_cooperative_matrix} with a 16-multiple fp16 x fp16 -> fp32
     * subgroup multiply-add plus fp16 arithmetic, full subgroups and the Vulkan memory model enabled, or OpenCL
     * {@code cl_khr_cooperative_matrix}.
     */
    COOPERATIVE_MATRIX,
    /**
     * Bfloat16 matrix multiply-accumulate on the tensor cores is usable in kernels: CUDA compute capability 8.0+
     * ({@code mma.sync.f32.bf16.bf16.f32}), or Vulkan {@code VK_KHR_shader_bfloat16} with the
     * {@code bfloat16CooperativeMatrix} feature and a 16-multiple bf16 x bf16 -> fp32 subgroup multiply-add (llama.cpp
     * {@code coopmat_bf16_support}) next to the fp16 cooperative-matrix requirements. OpenCL has no bfloat16 tensor
     * path, so it never reports this feature.
     */
    BF16_COOPERATIVE_MATRIX,
    /** Work spread across several compute queues runs concurrently and is worth the synchronisation cost. */
    CONCURRENT_QUEUES_PROFITABLE,
    /**
     * The device records reusable command buffers ({@code cl_khr_command_buffer}, with mutable dispatch when
     * {@code cl_khr_command_buffer_mutable_dispatch} is also present), so a queue can capture its dispatches,
     * copies and fills into a {@link ComputeQueue#supportsGraphCapture() replayable graph}.
     */
    COMMAND_BUFFER
}
