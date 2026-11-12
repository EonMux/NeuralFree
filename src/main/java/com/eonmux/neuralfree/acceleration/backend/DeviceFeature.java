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
    /**
     * Subgroup shuffle (butterfly exchange and broadcast, {@code __shfl_xor_sync}) is usable in kernels: CUDA and
     * HIP, where {@code common/dialect.h} defines {@code NF_HAS_SG_SHUFFLE}. OpenCL 1.2 has no portable subgroup
     * shuffle, so it never reports this feature and kernels keep their local-memory paths.
     */
    SUBGROUP_SHUFFLE,
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
     * The {@code VK_NV_cooperative_matrix2} form of the tensor cores is usable: flexible workgroup-scope
     * dimensions match the CM2 GEMM's fp16 tiles (128 rows, 64 deep, 256/128/64 tokens), with tensor-layout loads
     * (per-element decode callbacks over physical-storage-buffer references), per-element operations and
     * reductions, next to the fp16 cooperative-matrix requirements (llama.cpp {@code coopmat2}). CUDA and OpenCL
     * have no equivalent; they never report it.
     */
    COOPERATIVE_MATRIX_2,
    /**
     * {@link #COOPERATIVE_MATRIX_2} also serves the tensor-core prefill attention ({@code LLM_FlashAttnMma_F16} of
     * shaders/llm/flash_mma_cm2.comp): accumulator-to-operand conversions and flexible workgroup-scope dimensions for
     * fp16 x fp16 -> fp32 tiles of 128 invocations (64 query rows, 64 keys), as llama.cpp's flash_attn_cm2 needs.
     * Vulkan only; CUDA serves its mma attention through {@link #COOPERATIVE_MATRIX}.
     */
    COOPERATIVE_MATRIX_2_ATTENTION,
    /**
     * {@link #COOPERATIVE_MATRIX_2} tensor loads also take four-element decode callbacks
     * ({@code VK_NV_cooperative_matrix_decode_vector}, {@code cooperativeMatrixDecodeVector}), which the quantized CM2
     * GEMM forms {@code <T>_MatMul_CM2V} use (llama.cpp {@code coopmat2_decode_vector}). Vulkan only.
     */
    COOPERATIVE_MATRIX_2_DECODE_VECTOR,
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
