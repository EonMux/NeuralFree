/* flash_mma.cl - LLM_FlashAttnMma_F16 and LLM_FlashAttnMmaPaged_F16 tensor-core tiled prefill attention.
 * Requires: common/dialect.h, common/config.h, common/math.h, common/attn_common.h */
/*
 * CUDA-only tensor-core twin of the tiled prefill kernel (flash_prefill.cl): the Q K^T and P V
 * products run on the matrix unit through mma.sync.aligned.m16n8k16 fragments, so long prompts,
 * speculative verification and batched decode reach matrix-unit throughput as llama.cpp's
 * fattn-mma-f16 does. Every other semantic matches LLM_FlashAttn_F16: same argument list, same
 * grid, same online softmax, split-K partials, GQA packing, sinks, softcap, ALiBi, T5 bias, SWA
 * and the paged KV addressing of flash_prefill.cl. K and V caches are F16; Q, the score state and
 * the output stay F32, P and the staged tiles are F16 (as in ggml-cuda/fattn-mma-f16.cuh). The
 * OpenCL program and the HIP build define no mma kernels, so the host runs the scalar kernels there;
 * the Vulkan twins are the workgroup-scope cooperative-matrix shaders of
 * shaders/llm/flash_mma_cm2.comp (gguf.ops.attn.FlashAttnTiling.Plan.MMA is selected only where
 * gguf.kernels.LlmKernels.flashMmaPath() holds and the backend has the kernel).
 *
 * Layouts (offsets in elements, 64-bit inside the kernel): as flash_prefill.cl, with
 *   q        q[q_off + t * q_row_stride + h * d_k + d], query position q_pos[t]
 *   K value  k[j * k_row_stride + kvh * k_head_stride + d] (F16), key position kv_pos[j]
 *   out      out[out_off + t * out_row_stride + h * d_v + d]                      (direct_out = 1)
 *   partial  partial[((t * n_head + h) * n_split + s) * (d_v + 2) + d], m at + d_v, l at + d_v + 1
 *
 * Grid: global {ceil(n_tokens / bq) * 128, n_head_kv * ceil(G / g_tile), n_split}, local {128, 1, 1}
 * with R = bq * g_tile == 64. The work-groups of one split take their (token block, head tile)
 * from the linear group index with the last token block first, so the causal blocks that see the
 * most keys start first. Work-group (b, y, s): four warps, warp w owns the 16 query rows
 * r = 16 w .. 16 w + 15 (token t = b * bq + r / g_tile, head h = h0 + r % g_tile,
 * h0 = kvh * G + gt * g_tile) and keys [kv_start + s * split_len, min(+ split_len, kv_end)) in
 * tiles of NF_FM_BN = 64.
 *
 * Shared memory: the K tile and the V tile, 64 key rows of NF_FM_LD halves each in cache order
 * (the K tile first holds the F16 Q tile while the warps load their Q fragments), and the key
 * positions of the K tile.
 *
 * Each warp keeps its Q fragments in registers for the whole key loop. Key tiles invisible to
 * every row are skipped (every warp evaluates the same key positions, so the decision is uniform
 * without a barrier); the visible ones run a two-stage cp.async pipeline as fattn-mma-f16 does:
 * V(t) streams in while S = Q K(t)^T runs, K(t') of the next visible tile while O += P V(t) runs.
 * K fragments come from the K tile through ldmatrix, V fragments through ldmatrix.trans. Score,
 * position bias and -INF masking act on the F32 fragments; a tile every key of which is visible to
 * every row of the warp and that has no bias or softcap only scales. Online softmax per row
 * (fragment max/sum reduced over the row's quad); P converts in the fragments to the F16 A operand
 * of O = O * alpha + P V. Finally rows apply sinks and 1 / l (l == 0 writes 0) or emit (o, m, l)
 * partials, as flash_prefill.cl does.
 *
 * Plan constraints (gguf.ops.attn.FlashAttnTiling.mma guarantees them; a violation writes NaN
 * outputs): d_k and d_v positive multiples of 16 of at most NF_FM_MAX_D, R == 64,
 * g_tile in 1..NF_FD_MAX_G, bq >= 1, bk == NF_FM_BN, split_len > 0, local size == NF_FM_WG,
 * direct_out only with n_split == 1, paged page_shift in 0..20. K and V rows stage through
 * 16-byte cp.async when the cache pointers and strides are 16-byte aligned, otherwise through
 * plain loads.
 */

#if defined(__CUDACC__) && !defined(__HIP__)
/* Kernel bodies need mma.sync.m16n8k16, ldmatrix and cp.async (compute capability 8.0+);
 * NF_CUDA_PREPROCESS is the build's nvcc -E inventory pass, which sees the kernels so the index and
 * inventory list them. */
#if defined(NF_CUDA_PREPROCESS) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

/* Tile shape of the mma kernel. */
#define NF_FM_WG     128   /* work-group size: four warps of 32 lanes */
#define NF_FM_BM     64    /* query rows per work-group (bq * g_tile == NF_FM_BM) */
#define NF_FM_BN     64    /* keys per tile; the K tile also holds the NF_FM_BM-row Q tile */
#define NF_FM_MAX_D  128   /* largest head dim (d_k and d_v) */
/* Halves per staged row: 272 bytes, an odd number of 16-byte chunks, so the eight row addresses of
 * one ldmatrix matrix fall on distinct bank groups. */
#define NF_FM_LD     (NF_FM_MAX_D + 8)
#define NF_FM_KG     (NF_FM_BN / 8)       /* 16 x 8 score fragments of a warp per tile */
#define NF_FM_KS     (NF_FM_MAX_D / 16)   /* 16-wide steps over the head dim */
#define NF_FM_POS_MAX 0x7fffffff         /* above every query position */

/* mma.sync.aligned.m16n8k16.f32.f16.f16.f32: D = A(16 x 16) * B(16 x 8) + D over F16 fragments,
 * F32 accumulate. Fragment layouts (lane 0..31, group = lane / 4, tig = lane % 4):
 *   A a0..a3 (.f16x2): a0 row group,      k 2 tig, 2 tig + 1;  a1 row group + 8, same k;
 *                      a2 row group,      k 2 tig + 8 + {0, 1}; a3 row group + 8, same k;
 *   B b0..b1 (.f16x2): b0 k 2 tig, 2 tig + 1, col group; b1 k 2 tig + 8 + {0, 1}, col group;
 *   C/D c0..c3 (F32):  c0 row group, col 2 tig; c1 row group, col 2 tig + 1;
 *                      c2 row group + 8, same cols.
 * The C fragment pairs map onto the A fragment pairs, so P is converted to the F16 A operand of
 * the P V product in the registers that hold it. */
NF_INLINE void nf_fm_mma(const unsigned a[4], const unsigned b0, const unsigned b1, float d[4]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* Shared-memory address of a staged element. */
NF_INLINE unsigned nf_fm_smem(NF_LOCAL_PTR const NF_HALF* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

/* ldmatrix of four 8 x 8 F16 matrices: lanes 8 i .. 8 i + 7 address the rows of matrix i, and lane l
 * receives row l / 4, columns 2 (l % 4) and 2 (l % 4) + 1 of each. */
NF_INLINE void nf_fm_ldsm4(unsigned r[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

/* ldmatrix.trans of four 8 x 8 F16 matrices: lane l receives rows 2 (l % 4) and 2 (l % 4) + 1 of
 * column l / 4 of each, the B fragment of a row-major k x n tile. */
NF_INLINE void nf_fm_ldsm4_trans(unsigned r[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

/* 16-byte global-to-shared copy that fills the destination with zeros past src_bytes (0 or 16). */
NF_INLINE void nf_fm_cp16(unsigned dst, NF_GLOBAL const NF_HALF* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}

/* Closes the group of cp.async copies issued since the last commit. */
NF_INLINE void nf_fm_cp_commit(void) {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

/* Waits for every committed cp.async group of this thread. */
NF_INLINE void nf_fm_cp_wait(void) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

/* F16 pair (round to nearest) of two F32 values, in the fragment register order. */
NF_INLINE unsigned nf_fm_p2(float a, float b) {
    const __half2 v = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&v);
}

/* Plan constraints of the mma kernels; see the header comment. */
NF_INLINE int nf_fm_plan_ok(int n_head, int n_head_kv, int d_k, int d_v, int bq, int g_tile,
                            int n_split, int split_len, int direct_out) {
    return n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0
        && g_tile >= 1 && g_tile <= NF_FD_MAX_G && bq >= 1 && bq * g_tile == NF_FM_BM
        && d_k > 0 && d_v > 0 && d_k <= NF_FM_MAX_D && d_v <= NF_FM_MAX_D
        && (d_k & 15) == 0 && (d_v & 15) == 0
        && n_split >= 1 && split_len > 0 && (direct_out == 0 || n_split == 1)
        && (int)get_local_size(0) == NF_FM_WG;
}

/* Shared-memory floats the mma kernel stages for d_k and d_v (FaSmem counts 4-byte floats): the K
 * (and Q) tile, the V tile and the key positions. */
NF_INLINE long nf_fm_smem_floats(int d_k, int d_v) {
    const long halves = (long)NF_FM_BN * (d_k + 8) + (long)NF_FM_BN * (d_v + 8);
    return halves / 2 + NF_FM_BN;
}

/* Writes NaN over the outputs of a call whose arguments violate the plan (block (0, 0, 0) only). */
NF_INLINE void nf_fm_poison(int n_tokens, int n_head, int d_v, int n_split, int out_row_stride,
                            int direct_out, int out_off, NF_GLOBAL float* out, NF_GLOBAL float* partial) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0
        || n_tokens <= 0 || n_head <= 0 || d_v <= 0 || n_split <= 0) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    if (direct_out) {
        for (int t = 0; t < n_tokens; t++) {
            for (int i = tid; i < n_head * d_v; i += nt) {
                out[(nf_i64)out_off + (nf_i64)t * out_row_stride + i] = NF_NAN;
            }
        }
    } else {
        const nf_i64 n = (nf_i64)n_tokens * n_head * n_split * (d_v + 2);
        for (nf_i64 i = tid; i < n; i += nt) {
            partial[i] = NF_NAN;
        }
    }
}

/* Key positions of one tile as a warp holds them: keys lane and lane + 32 (-1 for a key past the
 * tile's nk keys or on a missing page). */
NF_INLINE int nf_fm_key_pos(int j, int kt, int nk, NF_GLOBAL const int* NF_RESTRICT kv_pos,
                            int page_shift, NF_GLOBAL const int* NF_RESTRICT pt, int pt_off) {
    return j < nk && nf_kv_page_ok(kt + j, page_shift, pt, pt_off)
        ? kv_pos[nf_kv_row(kt + j, page_shift, pt, pt_off)] : -1;
}

/*
 * Key positions a row with query position in [qmin, qmax] can see under the position rules without
 * fresh tree rows: every visible key lies in [lo, hi] (a necessary bound for the window rules, exact for
 * the causal bound alone).
 */
NF_INLINE int2 nf_fm_key_range(int qmin, int qmax, int causal, int swa_type, int n_swa) {
    int lo = 0;
    int hi = causal ? qmax : NF_FM_POS_MAX;
    if (swa_type == NF_SWA_STANDARD) {
        lo = qmin - n_swa + 1;
    } else if (swa_type == NF_SWA_CHUNKED) {
        lo = (qmin / n_swa) * n_swa;
    } else if (swa_type == NF_SWA_CHUNKED_LOOKBACK) {
        lo = ((qmin / n_swa) - 1) * n_swa;
    } else if (swa_type == NF_SWA_SYMMETRIC) {
        lo = qmin - n_swa / 2;
        hi = min(hi, qmax + n_swa / 2);
    }
    return make_int2(max(lo, 0), hi);
}

/*
 * 1 when some row of the work-group sees a key of the tile whose positions the warp holds (kpa,
 * kpb), the same answer in every warp. Without fresh tree rows a key outside the position range kr
 * (nf_fm_key_range of the block's query extents) is invisible to every row, and with the position
 * rule alone (simple: no SWA either) a key inside it is visible to some row; otherwise each of the
 * n_valid tokens from t0 is tested against the keys as the per-element mask does.
 */
NF_INLINE int nf_fm_tile_visible(int kpa, int kpb, int simple, int2 kr, int t0, int n_valid,
                                 NF_GLOBAL const int* NF_RESTRICT q_pos, int tree,
                                 NF_GLOBAL const ulong* NF_RESTRICT tree_mask, int kv_end, int n_fresh,
                                 int causal, int swa_type, int n_swa) {
    if (n_fresh == 0) {
        const int in = (kpa >= kr.x && kpa <= kr.y) || (kpb >= kr.x && kpb <= kr.y);
        if (simple || !__any_sync(0xffffffffu, in)) {
            return __any_sync(0xffffffffu, in);
        }
    }
    for (int i = 0; i < n_valid; i++) {
        const int pq = q_pos[t0 + i];
        const ulong bits = tree ? tree_mask[t0 + i] : (ulong)0;
        const int vis
            = (kpa >= 0 && nf_attn_tree_visible(kpa, bits, kv_end, n_fresh, pq, causal, swa_type, n_swa))
            || (kpb >= 0 && nf_attn_tree_visible(kpb, bits, kv_end, n_fresh, pq, causal, swa_type, n_swa));
        if (__any_sync(0xffffffffu, vis)) {
            return 1;
        }
    }
    return 0;
}

/*
 * Stages the d columns of the key rows kt .. kt + NF_FM_BN - 1 of one head into the tile dst (row j at
 * dst + j * NF_FM_LD); rows past nk or on a missing page are zeros. Thread tid copies 16-byte chunk
 * tid % 16 of rows tid / 16 + 8 i: cp.async when async_ok (contiguous rows step by a fixed pointer
 * increment, paged rows look their page up), plain loads otherwise.
 */
NF_INLINE void nf_fm_stage(NF_LOCAL_PTR NF_HALF* dst, NF_GLOBAL const NF_HALF* src, const nf_i64 head_off,
                           const int row_stride, const int kt, const int nk, const int d, const int async_ok,
                           const int page_shift, NF_GLOBAL const int* NF_RESTRICT pt, const int pt_off,
                           const int tid) {
    const int c = tid & 15;
    const int j0 = tid >> 4;
    if (8 * c >= d) {
        return;
    }
    const int rows = NF_FM_WG / 16;
    const unsigned s0 = nf_fm_smem(dst + j0 * NF_FM_LD + 8 * c);
    if (async_ok && page_shift < 0) {
        NF_GLOBAL const NF_HALF* p = src + (nf_i64)(kt + j0) * row_stride + head_off + 8 * c;
        const nf_i64 step = (nf_i64)rows * row_stride;
#pragma unroll
        for (int i = 0; i < NF_FM_BN / rows; i++) {
            const int ok = j0 + rows * i < nk;
            nf_fm_cp16(s0 + (unsigned)(i * rows * NF_FM_LD * 2), ok ? p : src, ok ? 16 : 0);
            p += step;
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < NF_FM_BN / rows; i++) {
        const int j = j0 + rows * i;
        const int ok = j < nk && nf_kv_page_ok(kt + j, page_shift, pt, pt_off);
        const nf_i64 row = ok ? (nf_i64)nf_kv_row(kt + j, page_shift, pt, pt_off) : 0;
        NF_GLOBAL const NF_HALF* p = src + row * row_stride + head_off + 8 * c;
        if (async_ok) {
            nf_fm_cp16(s0 + (unsigned)(i * rows * NF_FM_LD * 2), ok ? p : src, ok ? 16 : 0);
        } else {
            NF_LOCAL_PTR NF_HALF* q = dst + j * NF_FM_LD + 8 * c;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                q[e] = ok ? p[e] : (NF_HALF)0;
            }
        }
    }
}

/*
 * Stages the K tile at kt (nk keys) with nf_fm_stage and its key positions (kpa, kpb as every warp holds
 * them; warp 0 writes them), and returns whether the tile only scales for this warp (plain_ok, every key
 * valid and, causally, at most the smallest query position wqmin of the warp's rows).
 */
NF_INLINE int nf_fm_stage_keys(NF_LOCAL_PTR NF_HALF* k_s, NF_LOCAL_PTR int* kp_s, NF_GLOBAL const NF_HALF* k,
                               const nf_i64 k_head, const int k_row_stride, const int kt, const int nk,
                               const int d_k, const int async_ok, const int page_shift,
                               NF_GLOBAL const int* NF_RESTRICT pt, const int pt_off, const int tid,
                               const int kpa, const int kpb, const int plain_ok, const int causal,
                               const int wqmin) {
    nf_fm_stage(k_s, k, k_head, k_row_stride, kt, nk, d_k, async_ok, page_shift, pt, pt_off, tid);
    nf_fm_cp_commit();
    if (tid < 32) {
        kp_s[tid] = kpa;
        kp_s[tid + 32] = kpb;
    }
    return plain_ok && __all_sync(0xffffffffu, kpa >= 0 && kpb >= 0 && (!causal || max(kpa, kpb) <= wqmin));
}

/*
 * The mma tiled attention body over the work-group's shared tiles; DK and DV are the head dims when
 * fixed at compile time (the common 128 and 64, which drop the per-step head-dim guards), 0 to take
 * d_k and d_v at run time.
 */
template <int DK, int DV>
NF_INLINE void nf_fm_body(
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k_arg, const int d_v_arg,
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,
    const int kv_start, const int kv_end,
    const int q_row_stride, const int out_row_stride,
    const int k_row_stride, const int k_head_stride, const int v_row_stride, const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int causal, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int has_sinks, const int direct_out, const int q_off, const int out_off,
    NF_GLOBAL const float* q, NF_GLOBAL const NF_HALF* k, NF_GLOBAL const NF_HALF* v,
    NF_GLOBAL const int* NF_RESTRICT q_pos, NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks, NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out, NF_GLOBAL float* partial,
    const int page_shift, NF_GLOBAL const int* NF_RESTRICT pt, const int pt_off,
    const int tree, NF_GLOBAL const ulong* NF_RESTRICT tree_mask, const int n_fresh,
    NF_LOCAL_PTR NF_HALF* k_s, NF_LOCAL_PTR NF_HALF* v_s, NF_LOCAL_PTR int* kp_s)
{
    const int d_k = DK > 0 ? DK : d_k_arg;
    const int d_v = DV > 0 ? DV : d_v_arg;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int split = (int)blockIdx.z;
    const int G = n_head / n_head_kv;
    const int n_gt = (G + g_tile - 1) / g_tile;
    /* Last token block first: linear group index i takes token block n_tb - 1 - i / n_y. */
    const int n_y = (int)gridDim.y;
    const int lin = (int)blockIdx.x + (int)gridDim.x * (int)blockIdx.y;
    const int tblk = (int)gridDim.x - 1 - lin / n_y;
    const int yblk = lin - (lin / n_y) * n_y;
    const int kvh = yblk / n_gt;
    const int gt = yblk - kvh * n_gt;
    const int h0 = kvh * G + gt * g_tile;
    const int t0 = tblk * bq;
    const int g_valid = min(g_tile, G - gt * g_tile);
    const int k_begin = kv_start + split * split_len;
    const int k_stop = split < n_split ? min(k_begin + split_len, kv_end) : k_begin;

    /* Rows this lane's fragment pairs cover and their (token, head) identity. */
    const int r0 = warp * 16 + group;
    const int r1 = r0 + 8;
    const int i0 = r0 / g_tile, g0 = r0 - i0 * g_tile;
    const int i1 = r1 / g_tile, g1 = r1 - i1 * g_tile;
    const int tq0 = t0 + i0, tq1 = t0 + i1;
    const int hq0 = h0 + g0, hq1 = h0 + g1;
    const int ok0 = tq0 < n_tokens && g0 < g_valid;
    const int ok1 = tq1 < n_tokens && g1 < g_valid;
    const int pq0 = ok0 ? q_pos[tq0] : -1;
    const int pq1 = ok1 ? q_pos[tq1] : -1;
    const float slope0 = (bias_mode == NF_ATTN_BIAS_ALIBI || bias_mode == NF_ATTN_BIAS_ALIBI_SQRT) && ok0
        ? nf_alibi_slope(hq0, n_head, max_bias) : 0.0f;
    const float slope1 = (bias_mode == NF_ATTN_BIAS_ALIBI || bias_mode == NF_ATTN_BIAS_ALIBI_SQRT) && ok1
        ? nf_alibi_slope(hq1, n_head, max_bias) : 0.0f;

    /* Query extents for the tile tests: the key range the block's n_valid tokens can see by position,
     * and the smallest position of this warp's rows. */
    const int n_valid = min(bq, n_tokens - t0);
    const int qa = lane < n_valid ? q_pos[t0 + lane] : -1;
    const int qb = lane + 32 < n_valid ? q_pos[t0 + lane + 32] : -1;
    const int2 krange = nf_fm_key_range(
        __reduce_min_sync(0xffffffffu, min(qa < 0 ? NF_FM_POS_MAX : qa, qb < 0 ? NF_FM_POS_MAX : qb)),
        __reduce_max_sync(0xffffffffu, max(qa, qb)), causal, swa_type, n_swa);
    const int wqmin = __reduce_min_sync(0xffffffffu,
                                        min(ok0 ? pq0 : NF_FM_POS_MAX, ok1 ? pq1 : NF_FM_POS_MAX));
    /* Position rule alone (simple), and tiles that only scale when every key is visible to every row. */
    const int simple = n_fresh == 0 && swa_type == NF_SWA_NONE;
    const int plain_ok = simple && bias_mode == NF_ATTN_BIAS_NONE && !(softcap > 0.0f);

    const nf_i64 k_head = (nf_i64)kvh * k_head_stride;
    const nf_i64 v_head = (nf_i64)kvh * v_head_stride;
    const int async_ok = (((size_t)k | (size_t)v) & 15) == 0 && (k_row_stride & 7) == 0
        && (k_head_stride & 7) == 0 && (v_row_stride & 7) == 0 && (v_head_stride & 7) == 0;

    /* First visible tile, with its key positions. */
    int kt = k_begin;
    int kpa = -1, kpb = -1;
    for (; kt < k_stop; kt += NF_FM_BN) {
        const int nk = min(NF_FM_BN, k_stop - kt);
        kpa = nf_fm_key_pos(lane, kt, nk, kv_pos, page_shift, pt, pt_off);
        kpb = nf_fm_key_pos(lane + 32, kt, nk, kv_pos, page_shift, pt, pt_off);
        if (nf_fm_tile_visible(kpa, kpb, simple, krange, t0, n_valid, q_pos, tree, tree_mask,
                               kv_end, n_fresh, causal, swa_type, n_swa)) {
            break;
        }
    }

    /* 0. Q fragments: the Q tile staged as F16 in the K tile (invalid rows are zeros; their outputs
     *    are never written), then ldmatrix into registers for the whole key loop. */
    for (int idx = tid; idx < NF_FM_BM * (d_k / 4); idx += NF_FM_WG) {
        const int r = idx / (d_k / 4);
        const int c4 = idx - r * (d_k / 4);
        const int i = r / g_tile;
        const int g = r - i * g_tile;
        float4 f = NF_F4S(0.0f);
        if (t0 + i < n_tokens && g < g_valid) {
            /* d_k is a positive multiple of 16, so the 4-wide loads never leave the row. */
            f = vload4(0, q + (nf_i64)q_off + (nf_i64)(t0 + i) * q_row_stride + (nf_i64)(h0 + g) * d_k
                       + 4 * c4);
        }
        uint2 h;
        h.x = nf_fm_p2(f.x, f.y);
        h.y = nf_fm_p2(f.z, f.w);
        *(NF_LOCAL_PTR uint2*)(k_s + r * NF_FM_LD + 4 * c4) = h;
    }
    __syncthreads();
    const int kst = d_k >> 4;
    unsigned qf[NF_FM_KS][4];
#pragma unroll
    for (int ks = 0; ks < NF_FM_KS; ks++) {
        if (ks < kst) {
            nf_fm_ldsm4(qf[ks],
                        nf_fm_smem(k_s + (warp * 16 + (lane & 15)) * NF_FM_LD + ks * 16 + 8 * (lane >> 4)));
        }
    }
    __syncthreads();

    /* Running softmax state: this lane's rows (r0, r1). */
    float m0 = -NF_INF, m1 = -NF_INF, l0 = 0.0f, l1 = 0.0f;
    /* Output accumulators: per 8-column group, this lane holds rows r0 (c0, c1) and r1 (c2, c3). */
    float vkq[NF_FM_MAX_D / 8][4];
#pragma unroll
    for (int dg = 0; dg < NF_FM_MAX_D / 8; dg++) {
        vkq[dg][0] = vkq[dg][1] = vkq[dg][2] = vkq[dg][3] = 0.0f;
    }
    const int vpairs = d_v >> 4;

    /* ldmatrix lane addresses: K matrices (keys + (lane & 7) + 8 (lane >> 4), column 8 ((lane >> 3) & 1));
     * V matrices (keys + (lane & 7) + 8 ((lane >> 3) & 1), column 8 (lane >> 4)). */
    const unsigned k_lane
        = nf_fm_smem(k_s + ((lane & 7) + 8 * (lane >> 4)) * NF_FM_LD + 8 * ((lane >> 3) & 1));
    const unsigned v_lane
        = nf_fm_smem(v_s + ((lane & 7) + 8 * ((lane >> 3) & 1)) * NF_FM_LD + 8 * (lane >> 4));

    int plain = 0;
    if (kt < k_stop) {
        plain = nf_fm_stage_keys(k_s, kp_s, k, k_head, k_row_stride, kt, min(NF_FM_BN, k_stop - kt), d_k,
                                 async_ok, page_shift, pt, pt_off, tid, kpa, kpb, plain_ok, causal, wqmin);
    }
    while (kt < k_stop) {
        const int nk = min(NF_FM_BN, k_stop - kt);
        /* 1. K(kt) and its key positions are in place, and every warp is past the previous P V. */
        nf_fm_cp_wait();
        __syncthreads();
        nf_fm_stage(v_s, v, v_head, v_row_stride, kt, nk, d_v, async_ok, page_shift, pt, pt_off, tid);
        nf_fm_cp_commit();

        /* Key positions of the next tile, loaded ahead of the S product. */
        int kn = kt + NF_FM_BN;
        int nkn = min(NF_FM_BN, k_stop - kn);
        kpa = nf_fm_key_pos(lane, kn, nkn, kv_pos, page_shift, pt, pt_off);
        kpb = nf_fm_key_pos(lane + 32, kn, nkn, kv_pos, page_shift, pt, pt_off);

        /* 2. S = Q K^T on the mma: B fragments of key groups 2 np and 2 np + 1 per ldmatrix. */
        float s[NF_FM_KG][4];
#pragma unroll
        for (int n = 0; n < NF_FM_KG; n++) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < NF_FM_KS; ks++) {
            if (ks < kst) {
#pragma unroll
                for (int np = 0; np < NF_FM_KG / 2; np++) {
                    unsigned b[4];
                    nf_fm_ldsm4(b, k_lane + (unsigned)((np * 16 * NF_FM_LD + ks * 16) * 2));
                    nf_fm_mma(qf[ks], b[0], b[1], s[2 * np]);
                    nf_fm_mma(qf[ks], b[2], b[3], s[2 * np + 1]);
                }
            }
        }

        /* 3. Score, bias and mask on the F32 fragments (key j = 8 n + 2 tig + (e & 1)). */
        if (plain) {
#pragma unroll
            for (int n = 0; n < NF_FM_KG; n++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    s[n][e] *= scale;
                }
            }
        } else {
            const ulong tm0 = tree && ok0 ? tree_mask[tq0] : (ulong)0;
            const ulong tm1 = tree && ok1 ? tree_mask[tq1] : (ulong)0;
#pragma unroll
            for (int n = 0; n < NF_FM_KG; n++) {
                const int2 kp2 = *(NF_LOCAL_PTR const int2*)(kp_s + n * 8 + 2 * tig);
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int kp = (e & 1) ? kp2.y : kp2.x;
                    const int pq = e < 2 ? pq0 : pq1;
                    float sv = -NF_INF;
                    if (pq >= 0 && kp >= 0
                        && nf_attn_tree_visible(kp, e < 2 ? tm0 : tm1, kv_end, n_fresh, pq, causal, swa_type,
                                                n_swa)) {
                        sv = nf_attn_score(s[n][e], scale, softcap)
                           + nf_attn_bias(bias_mode, e < 2 ? hq0 : hq1, n_head, pq, kp,
                                          e < 2 ? slope0 : slope1, rel_bias, n_buckets, max_distance,
                                          e < 2 ? tq0 : tq1);
                    }
                    s[n][e] = sv;
                }
            }
        }

        /* 4. Online softmax per row: the row's fragment values live in the lanes of its quad. */
        float rm0 = -NF_INF, rm1 = -NF_INF, rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
        for (int n = 0; n < NF_FM_KG; n++) {
            rm0 = fmaxf(rm0, fmaxf(s[n][0], s[n][1]));
            rm1 = fmaxf(rm1, fmaxf(s[n][2], s[n][3]));
        }
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, o));
            rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, o));
        }
        const float mn0 = fmaxf(m0, rm0);
        const float mn1 = fmaxf(m1, rm1);
        const float al0 = nf_attn_rescale(m0, mn0);
        const float al1 = nf_attn_rescale(m1, mn1);
#pragma unroll
        for (int n = 0; n < NF_FM_KG; n++) {
            s[n][0] = nf_attn_rescale(s[n][0], mn0);
            s[n][1] = nf_attn_rescale(s[n][1], mn0);
            s[n][2] = nf_attn_rescale(s[n][2], mn1);
            s[n][3] = nf_attn_rescale(s[n][3], mn1);
            rs0 += s[n][0] + s[n][1];
            rs1 += s[n][2] + s[n][3];
        }
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, o);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, o);
        }
        l0 = l0 * al0 + rs0;
        l1 = l1 * al1 + rs1;
        m0 = mn0;
        m1 = mn1;

        /* Next visible tile (the loads above have landed by now unless tiles are skipped). */
        while (kn < k_stop && !nf_fm_tile_visible(kpa, kpb, simple, krange, t0, n_valid, q_pos, tree,
                                                  tree_mask, kv_end, n_fresh, causal, swa_type, n_swa)) {
            kn += NF_FM_BN;
            nkn = min(NF_FM_BN, k_stop - kn);
            kpa = nf_fm_key_pos(lane, kn, nkn, kv_pos, page_shift, pt, pt_off);
            kpb = nf_fm_key_pos(lane + 32, kn, nkn, kv_pos, page_shift, pt, pt_off);
        }

        /* 5. V(kt) is in place and every warp is done with the K tile: K of the next tile streams in
         *    while O = O * alpha + P V runs. */
        nf_fm_cp_wait();
        __syncthreads();
        if (kn < k_stop) {
            plain = nf_fm_stage_keys(k_s, kp_s, k, k_head, k_row_stride, kn, nkn, d_k, async_ok, page_shift, pt,
                                     pt_off, tid, kpa, kpb, plain_ok, causal, wqmin);
        }
#pragma unroll
        for (int dg = 0; dg < NF_FM_MAX_D / 8; dg++) {
            vkq[dg][0] *= al0;
            vkq[dg][1] *= al0;
            vkq[dg][2] *= al1;
            vkq[dg][3] *= al1;
        }
#pragma unroll
        for (int c = 0; c < NF_FM_KG / 2; c++) {
            /* A fragment of P: k chunk c covers key groups 2 c and 2 c + 1. */
            unsigned a[4];
            a[0] = nf_fm_p2(s[2 * c][0], s[2 * c][1]);
            a[1] = nf_fm_p2(s[2 * c][2], s[2 * c][3]);
            a[2] = nf_fm_p2(s[2 * c + 1][0], s[2 * c + 1][1]);
            a[3] = nf_fm_p2(s[2 * c + 1][2], s[2 * c + 1][3]);
#pragma unroll
            for (int dp = 0; dp < NF_FM_MAX_D / 16; dp++) {
                if (dp < vpairs) {
                    /* B fragments of column groups 2 dp and 2 dp + 1 over the chunk's 16 keys. */
                    unsigned b[4];
                    nf_fm_ldsm4_trans(b, v_lane + (unsigned)((c * 16 * NF_FM_LD + dp * 16) * 2));
                    nf_fm_mma(a, b[0], b[1], vkq[2 * dp]);
                    nf_fm_mma(a, b[2], b[3], vkq[2 * dp + 1]);
                }
            }
        }
        kt = kn;
    }

    const int vgroups = d_v >> 3;
    /* 6. Output: normalize with the sink, or emit (o, m, l) partials — as flash_prefill.cl step 5. */
    if (!direct_out) {
#pragma unroll
        for (int dg = 0; dg < NF_FM_MAX_D / 8; dg++) {
            if (dg >= vgroups) {
                continue;
            }
            const int d = dg * 8 + 2 * tig;
            if (ok0 && split < n_split) {
                NF_GLOBAL float* pp = partial
                    + (((nf_i64)tq0 * n_head + hq0) * n_split + split) * (d_v + 2);
                pp[d] = vkq[dg][0];
                pp[d + 1] = vkq[dg][1];
                if (tig == 0 && d == 0) {
                    pp[d_v] = m0;
                    pp[d_v + 1] = l0;
                }
            }
            if (ok1 && split < n_split) {
                NF_GLOBAL float* pp = partial
                    + (((nf_i64)tq1 * n_head + hq1) * n_split + split) * (d_v + 2);
                pp[d] = vkq[dg][2];
                pp[d + 1] = vkq[dg][3];
                if (tig == 0 && d == 0) {
                    pp[d_v] = m1;
                    pp[d_v + 1] = l1;
                }
            }
        }
        return;
    }
#pragma unroll
    for (int dg = 0; dg < NF_FM_MAX_D / 8; dg++) {
        if (dg >= vgroups) {
            continue;
        }
        const int d = dg * 8 + 2 * tig;
        if (ok0) {
            const float2 fl = has_sinks ? nf_attn_sink(m0, l0, sinks[hq0]) : NF_F2(1.0f, l0);
            NF_GLOBAL float* o = out + (nf_i64)out_off + (nf_i64)tq0 * out_row_stride
                + (nf_i64)hq0 * d_v + d;
            o[0] = fl.y > 0.0f ? vkq[dg][0] * fl.x / fl.y : 0.0f;
            o[1] = fl.y > 0.0f ? vkq[dg][1] * fl.x / fl.y : 0.0f;
        }
        if (ok1) {
            const float2 fl = has_sinks ? nf_attn_sink(m1, l1, sinks[hq1]) : NF_F2(1.0f, l1);
            NF_GLOBAL float* o = out + (nf_i64)out_off + (nf_i64)tq1 * out_row_stride
                + (nf_i64)hq1 * d_v + d;
            o[0] = fl.y > 0.0f ? vkq[dg][2] * fl.x / fl.y : 0.0f;
            o[1] = fl.y > 0.0f ? vkq[dg][3] * fl.x / fl.y : 0.0f;
        }
    }
}

/*
 * The mma tiled attention; the arguments are those of nf_fa_run (flash_prefill.cl). k_type and v_type
 * are NF_KV_F16, Q positions come from q_pos, cached keys from kv_pos over the page map.
 */
NF_INLINE void nf_fm_run(
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,
    const int kv_start, const int kv_end,
    const int q_row_stride, const int out_row_stride,
    const int k_row_stride, const int k_head_stride, const int v_row_stride, const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int causal, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int has_sinks, const int direct_out, const int q_off, const int out_off,
    NF_GLOBAL const float* q, NF_GLOBAL const NF_HALF* k, NF_GLOBAL const NF_HALF* v,
    NF_GLOBAL const int* NF_RESTRICT q_pos, NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks, NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out, NF_GLOBAL float* partial,
    const int page_shift, NF_GLOBAL const int* NF_RESTRICT pt, const int pt_off,
    const int tree, NF_GLOBAL const ulong* NF_RESTRICT tree_mask, const int n_fresh)
{
    if (!nf_fm_plan_ok(n_head, n_head_kv, d_k, d_v, bq, g_tile, n_split, split_len, direct_out)
        || bk != NF_FM_BN) {
        nf_fm_poison(n_tokens, n_head, d_v, n_split, out_row_stride, direct_out, out_off, out, partial);
        return;   /* uniform across the work-group: no barrier has been reached */
    }
    __shared__ __align__(16) NF_HALF k_s[NF_FM_BN * NF_FM_LD];
    __shared__ __align__(16) NF_HALF v_s[NF_FM_BN * NF_FM_LD];
    __shared__ __align__(16) int kp_s[NF_FM_BN];
    /* The body's arguments: nf_fm_run's own, then the shared tiles. */
#define NF_FM_BODY_ARGS \
    n_tokens, n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len, kv_start, kv_end, \
    q_row_stride, out_row_stride, k_row_stride, k_head_stride, v_row_stride, v_head_stride, \
    scale, softcap, max_bias, causal, swa_type, n_swa, bias_mode, n_buckets, max_distance, \
    has_sinks, direct_out, q_off, out_off, q, k, v, q_pos, kv_pos, sinks, rel_bias, out, partial, \
    page_shift, pt, pt_off, tree, tree_mask, n_fresh, k_s, v_s, kp_s
    if (d_k == 128 && d_v == 128) {
        nf_fm_body<128, 128>(NF_FM_BODY_ARGS);
    } else if (d_k == 64 && d_v == 64) {
        nf_fm_body<64, 64>(NF_FM_BODY_ARGS);
    } else {
        nf_fm_body<0, 0>(NF_FM_BODY_ARGS);
    }
#undef NF_FM_BODY_ARGS
}

/*
 * LLM_FlashAttnMma_F16(29 scalars, 10 buffers): the LLM_FlashAttn_F16 arguments; K and V caches are
 * F16. Grid and per-step arguments are those of LLM_FlashAttn_F16 with bq * g_tile == 64. Sinks and
 * rel_bias are read-only placeholders when unused.
 */
NF_KERNEL void LLM_FlashAttnMma_F16(
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,
    const int kv_start, const int kv_end,
    const int q_row_stride, const int out_row_stride,
    const int k_row_stride, const int k_head_stride, const int v_row_stride,
    const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int flags, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int q_off, const int out_off,
    NF_GLOBAL const float* q,
    NF_GLOBAL const NF_HALF* k,
    NF_GLOBAL const NF_HALF* v,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out,
    NF_GLOBAL float* partial,
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)
{
    nf_fm_run(n_tokens, n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,
              kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,
              v_row_stride, v_head_stride, scale, softcap, max_bias,
              flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,
              max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,
              out_off, q, k, v, q_pos, kv_pos, sinks, rel_bias, out, partial, -1, kv_pos, 0,
              flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags));
}

/*
 * LLM_FlashAttnMmaPaged_F16(31 scalars, 11 buffers): the LLM_FlashAttnPaged_F16 arguments; K and V
 * caches are F16, K, V and kv_pos are read at the physical row nf_kv_row (common/attn_common.h).
 */
NF_KERNEL void LLM_FlashAttnMmaPaged_F16(
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,
    const int kv_start, const int kv_end,
    const int q_row_stride, const int out_row_stride,
    const int k_row_stride, const int k_head_stride, const int v_row_stride,
    const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int flags, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int q_off, const int out_off,
    const int page_shift, const int pt_off,
    NF_GLOBAL const float* q,
    NF_GLOBAL const NF_HALF* k,
    NF_GLOBAL const NF_HALF* v,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out,
    NF_GLOBAL float* partial,
    NF_GLOBAL const int* NF_RESTRICT page_table,
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)
{
    if (page_shift < 0 || page_shift > 20 || pt_off < 0) {
        nf_fm_poison(n_tokens, n_head, d_v, n_split, out_row_stride,
                     flags & NF_ATTN_FLAG_DIRECT, out_off, out, partial);
        return;   /* uniform across the work-group: no barrier has been reached */
    }
    nf_fm_run(n_tokens, n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,
              kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,
              v_row_stride, v_head_stride, scale, softcap, max_bias,
              flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,
              max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,
              out_off, q, k, v, q_pos, kv_pos, sinks, rel_bias, out, partial, page_shift,
              page_table, pt_off, flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags));
}

#endif /* NF_CUDA_PREPROCESS, or mma.sync.m16n8k16 on compute capability 8.0+ */

#endif /* __CUDACC__ and not __HIP__: the OpenCL program and the HIP build define no mma kernels */
