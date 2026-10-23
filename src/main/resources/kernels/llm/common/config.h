/* common/config.h - model-independent compile-time constants shared by every LLM kernel. Requires: none */
/*
 * Java mirrors these values in gguf.kernels.LlmKernelConstants. Only NF_FA_SMEM_FLOATS (a device
 * capability) is set from outside: OpenCL passes -DNF_FA_SMEM_FLOATS=<n>, CUDA builds use 11264.
 */
#ifndef NF_CONFIG_H
#define NF_CONFIG_H

#ifndef LLM_DIMX
#define LLM_DIMX 256              /* max local size of every row-reduce kernel; power of 2 */
#endif
#define NF_FA_WG          128     /* flash kernels' workgroup size */
#define NF_FA_ACC_MAX     32      /* per-thread output accumulators in flash kernels: g_tile*d_v <= NF_FA_WG*NF_FA_ACC_MAX */
#define NF_FD_MAX_G       16      /* max q-heads per flash-decode workgroup (per-thread score registers) */
#ifndef NF_FA_SMEM_FLOATS
#define NF_FA_SMEM_FLOATS 3072    /* floats of static shared memory for flash kernels (12 KiB floor) */
#endif

/* Q8_0: d(f16) + qs[32] (ggml block_q8_0) */
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34
#define Q8_0_THREADS_PER_BLOCK 4
#define Q8_0_ELEMENTS_PER_THREAD (Q8_0_BLOCK_SIZE / Q8_0_THREADS_PER_BLOCK)

/* Q8_K: d(f32) + qs[256] + bsums[16] (ggml block_q8_K) */
#define Q8_K_BLOCK_SIZE 256
#define Q8_K_BLOCK_BYTES 292

#endif /* NF_CONFIG_H */
