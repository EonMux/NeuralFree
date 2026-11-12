/* flash_decode_sg.cl - LLM_FlashDecodeSg_<KV> and LLM_FlashDecodePagedSg_<KV> subgroup decode attention (OpenCL only). Requires: common/dialect.h, common/config.h, common/math.h, common/kv_types.h, common/attn_common.h, llm/flash_decode.cl */
/*
 * The decode attention of LLM_FlashDecode_<KV> / LLM_FlashDecodePaged_<KV> (flash_decode.cl) for plain F32, F16,
 * BF16 and Q8_0 caches, with the same argument lists, layouts, masking, biases, sinks and partial outputs, under a work mapping
 * built for GPUs whose work-groups are limited by local memory residency and whose loads are limited by the number of
 * load messages (the Intel Xe integrated GPUs):
 *
 * - A work-group is NF_FDS_SUBGROUPS subgroups of NF_SG_ROW_SIZE lanes (pinned by intel_reqd_sub_group_size) and
 *   serves one query head over one key split. Lane l holds the NF_FDS_DPL consecutive dims
 *   [l * NF_FDS_DPL, +NF_FDS_DPL) of q, of the K and V rows and of the output, so one key's K row (and V row) is one
 *   vector load per lane (a Q8_0 row: the lane's 8 codes as one 8-byte load and its block's scale, shared by the
 *   four lanes of the block). The per-thread state (q, the output accumulator and one key's K and V rows: 64 of the 128
 *   32-byte registers of a SIMD16 thread) leaves no room for a second head or a second key in flight without
 *   spilling; the latency of a key's loads is hidden by the other resident threads instead.
 * - Subgroup s scans keys s, s + NF_FDS_SUBGROUPS, ... of the split with an online softmax (running max m, sum l
 *   and the unnormalized output in registers): a key costs one subgroup reduction and no local memory. Each lane
 *   first resolves the page row, kv_pos and visibility of one of the next NF_SG_ROW_SIZE keys of its subgroup, and
 *   the scan broadcasts them, so the page table -> row -> position chain runs once per 16 keys in parallel instead
 *   of serially per key, and the scan visits only the visible keys.
 * - The subgroups' states merge through NF_FDS_MERGE_FLOATS floats of local memory (about 2 KB, against the
 *   portable kernel's NF_FA_SMEM_FLOATS budget, so every work-group of a launch is resident at once) with one
 *   barrier; subgroup 0 then writes the head's normalized output (direct_out) or its (o, m, l) partial in the
 *   portable layout for LLM_FlashAttn_Combine.
 *
 * Grid: global {n_split * NF_FDS_WG, [n_seq *] n_head}, local {NF_FDS_WG, 1}. Plan constraints (FlashAttnTiling's
 * subgroup decode form; a violation writes NaN outputs): g_tile == 1, d_k == d_v,
 * d_k a positive multiple of NF_FDS_DPL of at most NF_FDS_MAX_D, K and V row and head strides multiples of
 * NF_FDS_DPL (of the 32-value block for Q8_0), n_split >= 1, split_len > 0, direct_out only with
 * n_split == 1, the work-group and subgroup sizes above. The softmax is the portable kernel's in another summation
 * order; results are deterministic (fixed key interleave and merge order).
 */
#if defined(USE_SUBGROUPS) && defined(cl_intel_required_subgroup_size) && defined(NF_SG_ROW_SIZE)

#define NF_FDS_ATTR __attribute__((intel_reqd_sub_group_size(NF_SG_ROW_SIZE)))
/* Subgroups per work-group and dims per lane (the host mirrors both). */
#define NF_FDS_SUBGROUPS 4
#define NF_FDS_DPL 8
#define NF_FDS_WG (NF_FDS_SUBGROUPS * NF_SG_ROW_SIZE)
#define NF_FDS_MAX_D (NF_FDS_DPL * NF_SG_ROW_SIZE)
/* Per subgroup: the output dims, then m and l. */
#define NF_FDS_STATE (NF_FDS_MAX_D + 2)
#define NF_FDS_MERGE_FLOATS (NF_FDS_SUBGROUPS * NF_FDS_STATE)

/* Bytes of the first `elems` values of a cache row (elems a whole number of Q8_0 blocks for Q8_0). */
NF_INLINE nf_i64 nf_fds_bytes(int kv_type, nf_i64 elems) {
    if (kv_type == NF_KV_Q8_0) {
        return (elems >> 5) * Q8_0_BLOCK_BYTES;
    }
    return elems * (kv_type == NF_KV_F32 ? 4 : 2);
}

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
/* Eight halves from one 16-byte word load (vload_half8 promises 2-byte alignment only and splits into 4-byte loads). */
#define NF_FDS_HALF8(w) convert_float8(as_half8(w))
#else
#define NF_FDS_HALF8(w) vload_half8(0, (const NF_HALF*)&(w))
#endif

/*
 * NF_FDS_DPL consecutive values at element dim0 of the cache row starting at row (a uniform pointer: the 64-bit row
 * address is computed once per key in scalar registers, the lane offset is 32-bit). The 2-byte types load as one
 * 16-byte word per lane, which needs 4-byte aligned rows and dims: the plan checks keep every stride and dim0 a
 * multiple of NF_FDS_DPL elements. A Q8_0 row loads the lane's 8 codes of its block (2-byte aligned: the plan keeps
 * every stride a whole number of blocks) and the block's half scale, as nf_kv_load decodes them (scale * code).
 */
NF_INLINE float8 nf_fds_load(NF_GLOBAL const uchar* row, int dim0, int kv_type) {
    if (kv_type == NF_KV_Q8_0) {
        NF_GLOBAL const uchar* bp = row + (dim0 >> 5) * Q8_0_BLOCK_BYTES;
        const ushort4 c = vload4(0, (NF_GLOBAL const ushort*)(bp + 2 + (dim0 & 31)));
        return convert_float8(as_char8(c)) * vload_half(0, (NF_GLOBAL const NF_HALF*)bp);
    }
    if (kv_type == NF_KV_F16) {
        const uint4 w = vload4(0, (NF_GLOBAL const uint*)((NF_GLOBAL const NF_HALF*)row + dim0));
        return NF_FDS_HALF8(w);
    }
    if (kv_type == NF_KV_BF16) {
        const uint4 w = vload4(0, (NF_GLOBAL const uint*)((NF_GLOBAL const ushort*)row + dim0));
        const uint8 u = (uint8)(w.s0 << 16, w.s0 & 0xffff0000u, w.s1 << 16, w.s1 & 0xffff0000u,
                                w.s2 << 16, w.s2 & 0xffff0000u, w.s3 << 16, w.s3 & 0xffff0000u);
        return as_float8(u);
    }
    return vload8(0, (NF_GLOBAL const float*)row + dim0);
}

/* The stride alignment of nf_fds_load: every row and head offset a multiple of NF_FDS_DPL elements (of a whole
 * 32-value block for Q8_0). */
NF_INLINE int nf_fds_strides_ok(int kv_type, int k_row_stride, int k_head_stride, int v_row_stride,
                                int v_head_stride) {
    const int unit = kv_type == NF_KV_Q8_0 ? 32 : NF_FDS_DPL;
    return k_row_stride % unit == 0 && k_head_stride % unit == 0
        && v_row_stride % unit == 0 && v_head_stride % unit == 0;
}

/* Index of the lowest set bit of a non-zero mask (CL1.2 has clz but not ctz). */
NF_INLINE int nf_fds_first(uint mask) {
    return 31 - (int)clz(mask & (0u - mask));
}

NF_INLINE float nf_fds_dot(float8 a, float8 b) {
    return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

NF_INLINE int nf_fds_plan_ok(int n_head, int n_head_kv, int d_k, int d_v, int g_tile, int n_split, int split_len,
                             int direct_out, int kv_type, int k_row_stride, int k_head_stride, int v_row_stride,
                             int v_head_stride) {
    return nf_fds_strides_ok(kv_type, k_row_stride, k_head_stride, v_row_stride, v_head_stride)
        && n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0
        && g_tile == 1
        && d_k == d_v && d_k > 0 && d_k % NF_FDS_DPL == 0 && d_k <= NF_FDS_MAX_D
        && n_split >= 1 && split_len > 0
        && (direct_out == 0 || n_split == 1)
        && (int)get_local_size(0) == NF_FDS_WG
        && (int)get_max_sub_group_size() == NF_SG_ROW_SIZE;
}

/*
 * The shared decode body. page_shift < 0 is the identity row map of the un-paged entry point (pt unused, seq -1);
 * the paged entry point passes its sequence's page table offset and seq >= 0, as nf_fd_run takes them.
 */
NF_INLINE void nf_fds_run(
    const int n_head, const int n_head_kv, const int d,
    const int n_split, const int split_len, const int kv_start, const int kv_end, const int q_pos,
    const int k_row_stride, const int k_head_stride, const int v_row_stride, const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int causal, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int has_sinks, const int direct_out, const int q_off, const int out_off,
    NF_GLOBAL const float* q, NF_GLOBAL const uchar* k, NF_GLOBAL const uchar* v,
    NF_GLOBAL const int* kv_pos, NF_GLOBAL const float* sinks, NF_GLOBAL const float* rel_bias,
    NF_GLOBAL float* out, NF_GLOBAL float* partial, NF_GLOBAL const float* kv_scales, const int kv_type,
    const int page_shift, NF_GLOBAL const int* pt, const int pt_off, const int seq,
    const int tree, NF_GLOBAL const ulong* tree_mask, const int n_fresh,
    NF_LOCAL_PTR float* merge)
{
    const int lane = (int)get_sub_group_local_id();
    const int sg = (int)get_sub_group_id();
    const int split = (int)get_group_id(0);
    int h = (int)get_group_id(1);
    if (seq > 0) {
        h -= seq * n_head;   /* the paged grid folds the sequence in front of the head */
    }
    const int kvh = h / (n_head / n_head_kv);
    const int k_begin = kv_start + split * split_len;
    const int n_keys = max(0, min(k_begin + split_len, kv_end) - k_begin);
    const int dim0 = lane * NF_FDS_DPL;
    const int live = dim0 < d;   /* lanes past d hold zeros (a 64-dim head uses 8 of 16 lanes) */

    const float8 qv = live ? vload8(0, q + (nf_i64)q_off + (nf_i64)h * d + dim0) : (float8)(0.0f);
    const float slope = bias_mode == NF_ATTN_BIAS_ALIBI || bias_mode == NF_ATTN_BIAS_ALIBI_SQRT
                        ? nf_alibi_slope(h, n_head, max_bias) : 1.0f;
    const int q_row = seq > 0 ? seq : 0;
    /* Plain caches carry one scale per KV head and side (1.0 unless a checkpoint scale was set). */
    const float ks = nf_kv_head_scale(kv_type, kv_scales, 0, kvh, n_head_kv, 0);
    const float vs = nf_kv_head_scale(kv_type, kv_scales, 0, kvh, n_head_kv, 1);
    NF_GLOBAL const uchar* k_base = k + nf_fds_bytes(kv_type, (nf_i64)kvh * k_head_stride);
    NF_GLOBAL const uchar* v_base = v + nf_fds_bytes(kv_type, (nf_i64)kvh * v_head_stride);
    const nf_i64 k_row_bytes = nf_fds_bytes(kv_type, k_row_stride);
    const nf_i64 v_row_bytes = nf_fds_bytes(kv_type, v_row_stride);
    const ulong tree_bits = tree ? tree_mask[0] : (ulong)0;

    float m = -NF_INF;
    float l = 0.0f;
    float8 acc = (float8)(0.0f);

    /* Keys of this subgroup: j = sg + NF_FDS_SUBGROUPS * i. A batch is the next NF_SG_ROW_SIZE of them; lane i resolves
     * key i of the batch (row, position, visibility) and the scan broadcasts them. */
    const int batch_keys = NF_FDS_SUBGROUPS * NF_SG_ROW_SIZE;
    for (int base = 0; base < n_keys; base += batch_keys) {
        const int jl = base + sg + NF_FDS_SUBGROUPS * lane;
        int row_l = 0;
        int pos_l = -1;
        int vis_l = 0;
        if (jl < n_keys && nf_kv_page_ok(k_begin + jl, page_shift, pt, pt_off)) {
            row_l = nf_kv_row(k_begin + jl, page_shift, pt, pt_off);
            pos_l = kv_pos[row_l];
            vis_l = nf_attn_tree_visible(pos_l, tree_bits, kv_end, n_fresh, q_pos, causal, swa_type, n_swa);
        }
        /* Visible keys of the batch as a lane mask (uniform), so the scan visits only them. */
        uint vis_mask = 0;
        for (int i = 0; i < NF_SG_ROW_SIZE; i++) {
            vis_mask |= (uint)sub_group_broadcast(vis_l, (uint)i) << i;
        }
        while (vis_mask != 0) {
            const int i = nf_fds_first(vis_mask);
            vis_mask &= vis_mask - 1u;
            const int row = sub_group_broadcast(row_l, (uint)i);
            const int p_k = sub_group_broadcast(pos_l, (uint)i);
            float8 k8 = (float8)(0.0f);
            float8 v8 = (float8)(0.0f);
            if (live) {
                k8 = nf_fds_load(k_base + row * k_row_bytes, dim0, kv_type);
                v8 = nf_fds_load(v_base + row * v_row_bytes, dim0, kv_type);
            }
            const float s = nf_attn_score(sub_group_reduce_add(nf_fds_dot(qv, k8)) * ks, scale, softcap)
                          + nf_attn_bias(bias_mode, h, n_head, q_pos, p_k, slope, rel_bias, n_buckets,
                                         max_distance, q_row);
            const float mn = fmax(m, s);
            const float a = nf_attn_rescale(m, mn);
            const float p = nf_attn_rescale(s, mn);
            l = l * a + p;
            acc = acc * a + (p * vs) * v8;
            m = mn;
        }
    }

    /* Merge the subgroups' states: each writes (acc, m, l), subgroup 0 folds them in subgroup order. */
    NF_LOCAL_PTR float* own = merge + sg * NF_FDS_STATE;
    if (live) {
        vstore8(acc, 0, own + dim0);
    }
    if (lane == 0) {
        own[NF_FDS_MAX_D] = m;
        own[NF_FDS_MAX_D + 1] = l;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0) {
        return;
    }
    m = -NF_INF;
    for (int s = 0; s < NF_FDS_SUBGROUPS; s++) {
        m = fmax(m, merge[s * NF_FDS_STATE + NF_FDS_MAX_D]);
    }
    l = 0.0f;
    float8 o = (float8)(0.0f);
    for (int s = 0; s < NF_FDS_SUBGROUPS; s++) {
        NF_LOCAL_PTR const float* st = merge + s * NF_FDS_STATE;
        const float w = nf_attn_rescale(st[NF_FDS_MAX_D], m);
        l += st[NF_FDS_MAX_D + 1] * w;
        if (live) {
            o += vload8(0, st + dim0) * w;
        }
    }
    if (direct_out) {
        const float2 fl = has_sinks ? nf_attn_sink(m, l, sinks[h]) : NF_F2(1.0f, l);
        if (live) {
            vstore8(fl.y > 0.0f ? o * (fl.x / fl.y) : (float8)(0.0f), 0,
                    out + (nf_i64)out_off + (nf_i64)h * d + dim0);
        }
    } else {
        NF_GLOBAL float* pr = partial + (((nf_i64)max(seq, 0) * n_head + h) * n_split + split) * (d + 2);
        if (live) {
            vstore8(o, 0, pr + dim0);
        }
        if (lane == 0) {
            pr[d] = m;
            pr[d + 1] = l;
        }
    }
}

/* LLM_FlashDecodeSg_<KV>: the arguments of LLM_FlashDecode_<KV> (flash_decode.cl). */
#define NF_FLASH_DECODE_SG_KERNEL(SFX, KV_T, KV_CODE)                                         \
NF_FDS_ATTR NF_KERNEL void LLM_FlashDecodeSg_##SFX(                                           \
    const int n_head, const int n_head_kv, const int d_k, const int d_v, const int g_tile,     \
    const int n_split, const int split_len, const int kv_start, const int kv_end,             \
    const int q_pos,                                                                          \
    const int k_row_stride, const int k_head_stride, const int v_row_stride,                  \
    const int v_head_stride,                                                                  \
    const float scale, const float softcap, const float max_bias,                             \
    const int flags, const int swa_type, const int n_swa,                                     \
    const int bias_mode, const int n_buckets, const int max_distance,                         \
    const int q_off, const int out_off,                                                       \
    NF_GLOBAL const float* q,                                                                 \
    NF_GLOBAL const KV_T* k,                                                                  \
    NF_GLOBAL const KV_T* v,                                                                  \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                  \
    NF_GLOBAL const float* NF_RESTRICT sinks,                                                 \
    NF_GLOBAL const float* NF_RESTRICT rel_bias,                                              \
    NF_GLOBAL float* out,                                                                     \
    NF_GLOBAL float* partial,                                                                 \
    NF_GLOBAL const float* NF_RESTRICT kv_scales,                                             \
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)                                             \
{                                                                                             \
    NF_LOCAL float merge[NF_FDS_MERGE_FLOATS];                                                \
    if (!nf_fds_plan_ok(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len,               \
                        flags & NF_ATTN_FLAG_DIRECT, KV_CODE, k_row_stride, k_head_stride,    \
                        v_row_stride, v_head_stride)) {                                       \
        nf_fd_poison(n_head, d_v, n_split, flags & NF_ATTN_FLAG_DIRECT, out_off, out, partial); \
        return;                                                                               \
    }                                                                                         \
    nf_fds_run(n_head, n_head_kv, d_k, n_split, split_len, kv_start, kv_end, q_pos,   \
               k_row_stride, k_head_stride, v_row_stride, v_head_stride,                      \
               scale, softcap, max_bias, flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa,         \
               bias_mode, n_buckets, max_distance, flags & NF_ATTN_FLAG_SINKS,                 \
               flags & NF_ATTN_FLAG_DIRECT, q_off, out_off,                                   \
               q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v, kv_pos, sinks,          \
               rel_bias, out, partial, kv_scales, KV_CODE, -1, kv_pos, 0, -1,                   \
               flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags), merge);         \
}

/* LLM_FlashDecodePagedSg_<KV>: the arguments of LLM_FlashDecodePaged_<KV> (flash_decode.cl). */
#define NF_FLASH_DECODE_PAGED_SG_KERNEL(SFX, KV_T, KV_CODE)                                   \
NF_FDS_ATTR NF_KERNEL void LLM_FlashDecodePagedSg_##SFX(                                      \
    NF_FLASH_DECODE_PAGED_RUN_ARGS_PAGED(KV_T))                                               \
{                                                                                             \
    NF_LOCAL float merge[NF_FDS_MERGE_FLOATS];                                                \
    const int direct = flags & NF_ATTN_FLAG_DIRECT;                                           \
    if (!(page_shift >= 0 && page_shift <= 20 && max_pages >= 1 && n_seq >= 1                 \
          && q_seq_stride >= 0 && out_seq_stride >= 0                                         \
          && nf_fds_plan_ok(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len, direct,  \
                            KV_CODE, k_row_stride, k_head_stride, v_row_stride,               \
                            v_head_stride))) {                                                \
        nf_fd_paged_poison(n_seq, n_head, d_v, n_split, direct, out_off, out, partial);       \
        return;                                                                               \
    }                                                                                         \
    const int s = (int)get_group_id(1) / n_head;   /* g_tile 1: one work-group row per head */  \
    const nf_i64 kv_cap = (nf_i64)max_pages << page_shift;                                    \
    const int kv_len = seq_kv_len[s] < 0 ? 0                                                  \
                       : ((nf_i64)seq_kv_len[s] > kv_cap ? (int)kv_cap : seq_kv_len[s]);       \
    nf_fds_run(n_head, n_head_kv, d_k, n_split, split_len, 0, kv_len, q_pos_seq[s],           \
               k_row_stride, k_head_stride, v_row_stride, v_head_stride,                      \
               scale, softcap, max_bias, flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa,         \
               bias_mode, n_buckets, max_distance, flags & NF_ATTN_FLAG_SINKS, direct,        \
               q_off + s * q_seq_stride, out_off + s * out_seq_stride,                         \
               q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v, kv_pos, sinks,          \
               rel_bias, out, partial, kv_scales, KV_CODE, page_shift, page_table,            \
               s * max_pages, s, flags & NF_ATTN_FLAG_TREE, tree_mask + s,                     \
               nf_attn_flag_fresh(flags), merge);                                             \
}

NF_FLASH_DECODE_SG_KERNEL(F32, float, NF_KV_F32)
NF_FLASH_DECODE_SG_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_DECODE_SG_KERNEL(BF16, ushort, NF_KV_BF16)
NF_FLASH_DECODE_SG_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_FLASH_DECODE_PAGED_SG_KERNEL(F32, float, NF_KV_F32)
NF_FLASH_DECODE_PAGED_SG_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_DECODE_PAGED_SG_KERNEL(BF16, ushort, NF_KV_BF16)
NF_FLASH_DECODE_PAGED_SG_KERNEL(Q8_0, uchar, NF_KV_Q8_0)

#endif /* USE_SUBGROUPS && cl_intel_required_subgroup_size && NF_SG_ROW_SIZE */
