/* common/matmul_mmq_config.h - per-architecture tile config of the MMQ GEMM family (mmq-config) and the
 * staged activation record layout. Requires: common/dialect.h */
/*
 * The MMQ datapath and tile per architecture, as llama.cpp picks them per cc in ggml-cuda/mmq.cuh
 * and the mmq-config tables; the host mirrors this table in gguf.ops.gemm.MmqConfig and plans the
 * launch grid with the same row and token tiles:
 * - CUDA sm_80+ (Ampere, Ada, Hopper, Blackwell), MmqConfig.INT8_TENSOR: the int8 tensor cores
 *   (mma.m16n8k32 / m16n8k16 .s8) over 32-deep stages and a 64 x 64 tile. llama.cpp sizes its
 *   token tile (J) up to 128 per architecture by shared memory; with the activations staged once
 *   per projection (LLM_Quant_MMQ_Q8_1), the tile costs only the register accumulators here, and
 *   64 tokens per workgroup doubled the int8 GEMM throughput of the 32-token tile on sm_120 (RTX
 *   5080 Laptop, 4096 x 4096 x 512).
 * - Everything else, MmqConfig.PACKED_DOT: the packed integer dot on __dp4a (CUDA sm_61+) or scalar
 *   int arithmetic (CUDA below sm_61, OpenCL, GLSL) over 16-deep stages and a 64 x 16 tile. The
 *   Vulkan tensor path for quantized weights is the cooperative-matrix matmul_mmq_cm.comp,
 *   dequantizing into fp16 tiles.
 * - HIP RDNA3 (gfx1100-gfx1103), MmqConfig.RDNA3: llama.cpp's mmq-config-rdna3.cuh widest tile,
 *   64 rows x 128 tokens over 16-deep stages (the table's J=128 non-fallback row of every MMQ type
 *   except Q2_K, whose J=128 row is 128 rows).
 * - HIP CDNA (gfx908, gfx90a, gfx940-gfx942, gfx950, gfx1250), MmqConfig.CDNA: llama.cpp's
 *   mmq-config-cdna.cuh tile, 128 rows x 64 tokens over 16-deep stages (I 128, J 64); eight rows
 *   per work-item keep the work-group at 256 work-items, as on the packed-dot tile.
 *
 * NF_MMQ_TM / NF_MMQ_TN / NF_MMQ_TK / NF_MMQ_RM can also be passed as build defines (the OpenCL
 * library passes them for a tile forced with -Dneuralfree.mmq.tile), so the host can run any tile on
 * a runtime-compiled kernel. NF_MMQ_RM is the number of weight rows one work-item covers; the
 * work-group is NF_MMQ_TM / NF_MMQ_RM row lanes times 16 token lanes.
 */
#ifndef NF_MATMUL_MMQ_CONFIG_H
#define NF_MATMUL_MMQ_CONFIG_H

/*
 * NF_MMQ_TENSOR: 1 when this pass compiles the int8 tensor path. Device passes of CUDA
 * sm_80+ build it; the host pass and the NF_CUDA_PREPROCESS inventory pass parse it, so the
 * asm fragments exist wherever the sources are scanned (as NF_MMA_FRAGMENTS does). Every other
 * dialect takes the packed integer dot — HIP too: the s8 mma.sync fragments are PTX-only and
 * MmqConfig maps HIP devices to the packed-dot tile this selects.
 */
#if defined(__CUDACC__) && !defined(__HIP__) \
    && (defined(NF_CUDA_PREPROCESS) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define NF_MMQ_TENSOR 1
#else
#define NF_MMQ_TENSOR 0
#endif

/* Row tile and rows per work-item of every pass: llama.cpp's I is 64 (RDNA3, packed dot) or 128 (CDNA). */
#define NF_MMQ_DEFAULT_TM 64
#define NF_MMQ_DEFAULT_RM 4

#if NF_MMQ_TENSOR
#define NF_MMQ_DEFAULT_TK 32           /* reduction depth of one stage: one m16n8k32 fragment */
#define NF_MMQ_DEFAULT_TN 64           /* tokens per workgroup tile: four n8 blocks per warp */
#else
#define NF_MMQ_DEFAULT_TK 16           /* reduction depth of one stage: one 16-wide chunk */
#define NF_MMQ_DEFAULT_TN 16           /* tokens per workgroup tile: one per work-item */
#endif

/*
 * HIP family tiles, as the host selects them in HipGfxTarget/MmqConfig: the device pass of each
 * --offload-arch target defines its __gfxNNNN__ macro, so one multi-target code object carries the
 * tile of every family it was built for.
 */
#if defined(__HIP__) && !defined(NF_MMQ_TM) && !defined(NF_MMQ_TN) && !defined(NF_MMQ_TK) \
    && !defined(NF_MMQ_RM)
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) \
    || defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)
#define NF_MMQ_TM 128
#define NF_MMQ_TN 64
#define NF_MMQ_TK 16
#define NF_MMQ_RM 8                    /* 256 work-items, the largest work-group the tile fits */
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#define NF_MMQ_TM 64
#define NF_MMQ_TN 128
#define NF_MMQ_TK 16
#endif
#endif

#ifndef NF_MMQ_TM
#define NF_MMQ_TM NF_MMQ_DEFAULT_TM
#endif
#ifndef NF_MMQ_TK
#define NF_MMQ_TK NF_MMQ_DEFAULT_TK
#endif
#ifndef NF_MMQ_TN
#define NF_MMQ_TN NF_MMQ_DEFAULT_TN
#endif
#ifndef NF_MMQ_RM
#define NF_MMQ_RM NF_MMQ_DEFAULT_RM
#endif

/* Largest stage depth the staging kernels instantiate: two 16-element halves. */
#define NF_Q8_MAX_TK 32

/* 32-bit units of one staged activation chunk record of depth tk: tk / 4 code ints, the chunk scale d,
 * the tk / 16 code sums of the 16-element halves and the asymmetric zero point (0 for every symmetric
 * record). The host's gguf.ops.gemm.MmqConfig.recordUnits() is the same count at the stage depth. */
#define NF_Q8_REC(tk) ((tk) / 4 + 2 + (tk) / 16)
#define NF_MMQ_RC NF_Q8_REC(NF_MMQ_TK)

/* The record's zero-point unit of every staging form that carries no asymmetric offset. */
#define NF_Q8_NO_ZERO_POINT 0u

/* Bit-casts of the staged scale unit, per union so every dialect parses them. */
NF_INLINE uint nf_q8_float_rec(float v) {
    union { uint u; float f; } cast;
    cast.f = v;
    return cast.u;
}

#endif /* NF_MATMUL_MMQ_CONFIG_H */
