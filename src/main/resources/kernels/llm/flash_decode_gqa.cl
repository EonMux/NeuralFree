/* flash_decode_gqa.cl - LLM_FlashDecodeGqa_F16 / _BF16 / _Q8_0 (dense) and LLM_FlashDecodePagedGqa_F16 / _BF16 /
 * _Q8_0 (paged):
 * decode attention with every query head of a KV group in one work-group. Requires: common/dialect.h,
 * common/config.h, common/math.h, common/kv_types.h, common/attn_common.h */
/*
 * CUDA twins (a [cuda] manifest entry; shaders/llm/flash_decode_gqa.comp ports every entry to Vulkan) of
 * LLM_FlashDecode_<KV> and LLM_FlashDecodePaged_<KV> (flash_decode.cl) for plain F16, BF16 and Q8_0
 * caches with head dims of 64 or 128 and at most NF_FDG_MAX_G query heads per KV head: the same arguments, the same
 * addressing (the dense key range [kv_start, kv_end) with the q_pos scalar, or the paged per-sequence page tables,
 * seq_kv_len and q_pos_seq), the same masking, score, bias and sink rules and the same outputs (the normalized rows
 * when direct, else the (o, m, l) partials LLM_FlashAttn_Combine merges), computed the way llama.cpp's fattn-vec decode
 * reads a KV cache: one work-group per (sequence, KV head, key split) holds all G query heads of the group, so every K
 * and V row is read once for the G heads instead of once per head, and the keys stream through independent key groups
 * of L = D / 8 lanes (16 groups for D = 64, 8 for D = 128), each with its own online softmax:
 *
 *   - Key loop: key group c (lanes c * L .. c * L + L - 1) takes keys c, c + 128 / L, ... of the split,
 *     NF_FDG_KB of them per iteration with every load issued before the first use. Each lane resolves the page id
 *     -> physical row -> kv_pos chain of its group's keys itself, one iteration ahead (the next iteration's chain
 *     loads are in flight while the current keys compute), so no shared staging pass or barrier sits between the
 *     address chain and the K / V loads; a masked, unmapped or invisible slot loads nothing. Lane i of the group holds
 *     dims [8 i, 8 i + 8) of the G query rows in registers and loads its slice of the K row and of the V row as
 *     one 16-byte vector each (one coalesced 2 * D-byte row per group per key; a Q8_0 row: the slice's 8 codes and
 *     its block's scale); the G dots reduce over the group's L lanes with butterfly shuffles; each group keeps its
 *     own running max, sum and output slice.
 *   - Merge: the groups of a warp merge by shuffles, the four warps through shared memory.
 *
 * Each key group visits its keys in ascending order whatever NF_FDG_KB, so the result does not depend on how many
 * keys are in flight.
 *
 * Arguments: the dense entry takes those of LLM_FlashDecode_<KV>; the paged entry those of
 * LLM_FlashDecodePaged_<KV> (NF_FLASH_DECODE_PAGED_RUN_ARGS_PAGED). g_tile is ignored (the work-group always
 * holds the whole group). Grid: global {n_split * NF_FDG_WG, n_seq * n_head_kv} (dense: n_seq = 1), local
 * {NF_FDG_WG}. Plan constraints (gguf.ops.attn.FlashAttnTiling.decodeGqa and decodePagedGqa guarantee them; a
 * violation writes NaN outputs): d_k == d_v in {64, 128}, 1 <= n_head / n_head_kv <= NF_FDG_MAX_G,
 * split_len >= 1, direct_out only with n_split == 1, the K / V strides and the cache base multiples of 8
 * elements (16 bytes), no V alias (an MLA latent cache never takes this kernel), page_shift in 0..20 (paged). None of
 * these cache types is checkpoint-scaled (KvDType.checkpointScaled: the FP8 types and NVFP4 only), so the host
 * binds the all-ones kv_scales placeholder for each of them; the kernel still reads the bound per-head K and V
 * scales of a Q8_0 cache, as LLM_FlashDecode_<KV> does through nf_kv_head_scale, and skips them for F16 and BF16.
 * Q8_0 strides are whole 32-element blocks.
 */

#if defined(__CUDACC__) && !defined(__HIP__)

#define NF_FDG_WG        128   /* work-group size: four warps */
#define NF_FDG_DS        8     /* dims per lane: one 16-byte vector of a K or V row */
#define NF_FDG_MAX_G     4     /* query heads per KV head the registers hold */
#define NF_FDG_MAX_D     128   /* largest head dim */
#define NF_FDG_KB        2     /* keys per key group in flight */

/*
 * The raw words of a lane's NF_FDG_DS consecutive values of a cache at element offset e (a multiple of 8): one
 * 16-byte load of an F16 / BF16 row; or the 8 codes of a Q8_0 block (e % 32 picks them; four 2-byte loads, the
 * block being 2-byte aligned) in x and y, code i in bits 8 (i % 4) of word i / 4, and the block's half scale
 * bits in z. The words stay packed until nf_fdg_values converts them.
 */
template <int KVT>
NF_INLINE uint4 nf_fdg_raw(NF_GLOBAL const uchar* base, const nf_i64 e) {
    if (KVT == NF_KV_Q8_0) {
        NF_GLOBAL const uchar* bp = base + (e >> 5) * Q8_0_BLOCK_BYTES;
        NF_GLOBAL const ushort* qs = (NF_GLOBAL const ushort*)(bp + 2 + (int)(e & 31));
        return make_uint4((uint)qs[0] | ((uint)qs[1] << 16), (uint)qs[2] | ((uint)qs[3] << 16),
                          (uint)*(NF_GLOBAL const ushort*)bp, 0u);
    }
    return *(NF_GLOBAL const uint4*)((NF_GLOBAL const ushort*)base + e);
}

/* The NF_FDG_DS floats of a slice from its raw words (nf_fdg_raw), as nf_kv_load decodes them. */
template <int KVT>
NF_INLINE void nf_fdg_values(const uint4 r, float v[NF_FDG_DS]) {
    if (KVT == NF_KV_Q8_0) {
        const float d = __half2float(__ushort_as_half((ushort)r.z));
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i] = d * (float)((int)(r.x << (24 - 8 * i)) >> 24);
            v[4 + i] = d * (float)((int)(r.y << (24 - 8 * i)) >> 24);
        }
        return;
    }
    const uint w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
        if (KVT == NF_KV_F16) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[k]));
            v[2 * k] = f.x;
            v[2 * k + 1] = f.y;
        } else {
            v[2 * k] = __uint_as_float(w[k] << 16);
            v[2 * k + 1] = __uint_as_float(w[k] & 0xFFFF0000u);
        }
    }
}

/* The geometry both entries share (the paged entry adds its page and sequence checks). */
NF_INLINE int nf_fdg_shape_ok(int n_head, int n_head_kv, int d_k, int d_v, int n_split, int split_len,
                              int direct_out, int k_row_stride, int k_head_stride, int v_row_stride,
                              int v_head_stride, int v_alias, int kv_type) {
    const int align = kv_type == NF_KV_Q8_0 ? 31 : 7;   /* whole Q8_0 blocks, else whole 16-byte vectors */
    return n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0 && n_head / n_head_kv <= NF_FDG_MAX_G
        && d_k == d_v && (d_k == 64 || d_k == 128)
        && n_split >= 1 && split_len >= 1 && (direct_out == 0 || n_split == 1)
        && (k_row_stride & align) == 0 && (k_head_stride & align) == 0 && (v_row_stride & align) == 0
        && (v_head_stride & align) == 0 && !v_alias
        && (int)get_local_size(0) == NF_FDG_WG;
}

/* NaN over every output row of the launch (work-group (0, 0) only), as nf_fd_paged_poison. */
NF_INLINE void nf_fdg_poison(int n_seq, int n_head, int d_v, int n_split, int direct_out, int out_off,
                             int out_seq_stride, NF_GLOBAL float* out, NF_GLOBAL float* partial) {
    if (get_group_id(0) != 0 || get_group_id(1) != 0 || n_seq <= 0 || n_head <= 0 || d_v <= 0 || n_split <= 0) {
        return;
    }
    const int lid = (int)get_local_id(0);
    if (direct_out) {
        for (int s = 0; s < n_seq; s++) {
            for (int i = lid; i < n_head * d_v; i += NF_FDG_WG) {
                out[(nf_i64)out_off + (nf_i64)s * out_seq_stride + i] = NF_NAN;
            }
        }
    } else {
        const nf_i64 n = (nf_i64)n_seq * n_head * n_split * (d_v + 2);
        for (nf_i64 i = lid; i < n; i += NF_FDG_WG) {
            partial[i] = NF_NAN;
        }
    }
}

/*
 * The physical row and kv_pos of key j of a split starting at slot k_begin: *row = -1 (and *pos = -1) for a key
 * past the split's n_keys or on an unmapped page.
 */
NF_INLINE void nf_fdg_locate(const int j, const int n_keys, const int k_begin, const int page_shift,
                             NF_GLOBAL const int* page_table, const int pt_off, NF_GLOBAL const int* kv_pos,
                             int* row, int* pos) {
    *row = -1;
    *pos = -1;
    const int t = k_begin + j;
    if (j < n_keys && nf_kv_page_ok(t, page_shift, page_table, pt_off)) {
        const int r = nf_kv_row(t, page_shift, page_table, pt_off);
        *row = r;
        *pos = kv_pos[r];
    }
}

/*
 * One (sequence s, KV head kvh, split) work-group over the key slots [kv_lo + split * split_len, + split_len)
 * clamped to kv_hi: slot t reads physical row nf_kv_row(t, page_shift, page_table, pt_off) (page_shift = -1: the
 * dense identity map). q_pos is the query's position, tree_bits its ancestor bitmask; the query rows, outputs and
 * partials are those of sequence s (q_seq_stride, out_seq_stride, partial rows s * n_head + h). GP is the register
 * width of the group's heads: G = n_head / n_head_kv rounded up to a power of two (heads g >= G of a GP = 4
 * instance hold zeros and are skipped).
 */
template <int D, int KVT, int GP>
NF_INLINE void nf_fdg_run(
    const int n_head, const int n_head_kv, const int n_split, const int split_len,
    const int k_row_stride, const int k_head_stride, const int v_row_stride, const int v_head_stride,
    const float scale, const float softcap, const float max_bias, const int flags, const int swa_type,
    const int n_swa, const int bias_mode, const int n_buckets, const int max_distance,
    const int q_off, const int out_off, const int page_shift, const int pt_off,
    const int q_seq_stride, const int out_seq_stride,
    const int s, const int kvh, const int kv_lo, const int kv_hi, const int q_pos, const ulong tree_bits,
    NF_GLOBAL const float* q, NF_GLOBAL const uchar* k, NF_GLOBAL const uchar* v,
    NF_GLOBAL const int* kv_pos, NF_GLOBAL const float* sinks, NF_GLOBAL const float* rel_bias,
    NF_GLOBAL float* out, NF_GLOBAL float* partial, NF_GLOBAL const int* page_table,
    NF_GLOBAL const float* kv_scales, NF_LOCAL_PTR float* mrg)
{
    const int DS = NF_FDG_DS;
    const int L = D / DS;                             /* lanes of a key group: 8 or 16 */
    const int GROUPS = NF_FDG_WG / L;
    const int P = NF_FDG_KB * GP;                     /* partial dots per lane and iteration */
    const int G = n_head / n_head_kv;
    const int lid = (int)get_local_id(0);
    const int warp = lid >> 5;
    const int lane = lid & 31;
    const int grp = lid / L;
    const int d0 = (lid % L) * DS;
    const int split = (int)get_group_id(0);
    const int h0 = kvh * G;
    const int causal = flags & NF_ATTN_FLAG_CAUSAL;
    const int n_fresh = nf_attn_flag_fresh(flags);
    const int k_begin = kv_lo + split * split_len;
    const int k_stop = split < n_split ? min(k_begin + split_len, kv_hi) : k_begin;
    const int n_keys = max(0, k_stop - k_begin);

    /* the lane's query slices and the heads' ALiBi slopes */
    float qr[GP][DS];
    float slope[GP];
    NF_GLOBAL const float* qs = q + (nf_i64)q_off + (nf_i64)s * q_seq_stride + (nf_i64)h0 * D + d0;
#pragma unroll
    for (int g = 0; g < GP; g++) {
#pragma unroll
        for (int i = 0; i < DS; i++) {
            qr[g][i] = g < G ? qs[g * D + i] : 0.0f;
        }
        slope[g] = (bias_mode == NF_ATTN_BIAS_ALIBI || bias_mode == NF_ATTN_BIAS_ALIBI_SQRT) && g < G
                   ? nf_alibi_slope(h0 + g, n_head, max_bias) : 1.0f;
    }

    float m[GP], l[GP], o[GP][DS];
#pragma unroll
    for (int g = 0; g < GP; g++) {
        m[g] = -NF_INF;
        l[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < DS; i++) {
            o[g][i] = 0.0f;
        }
    }
    /* the bound per-head K and V scales of a Q8_0 cache (the host binds all ones; see the header); F16 and BF16
     * skip them */
    const float k_scale = KVT == NF_KV_Q8_0 ? kv_scales[kvh] : 1.0f;
    const float v_scale = KVT == NF_KV_Q8_0 ? kv_scales[n_head_kv + kvh] : 1.0f;
    const nf_i64 k_head = (nf_i64)kvh * k_head_stride + d0;
    const nf_i64 v_head = (nf_i64)kvh * v_head_stride + d0;

    /* the key loop: group grp takes keys grp, grp + GROUPS, ... of the split, NF_FDG_KB at a time; every lane of
     * a warp runs every iteration (the group sums take the whole warp). Each iteration issues the K and V loads of
     * its visible keys before the first use and the row and kv_pos loads of the next iteration's keys, so the
     * visibility test never waits on a load: a key past the split, on an unmapped page or invisible loads
     * nothing and leaves the group's state unchanged */
    const int step = GROUPS * NF_FDG_KB;
    int rown[NF_FDG_KB];
    int posn[NF_FDG_KB];
#pragma unroll
    for (int b = 0; b < NF_FDG_KB; b++) {
        nf_fdg_locate(b * GROUPS + grp, n_keys, k_begin, page_shift, page_table, pt_off, kv_pos, &rown[b],
                      &posn[b]);
    }
    for (int j0 = 0; j0 < n_keys; j0 += step) {
        int rowb[NF_FDG_KB];
        int posb[NF_FDG_KB];
        uint4 kr[NF_FDG_KB];
        uint4 vr[NF_FDG_KB];
#pragma unroll
        for (int b = 0; b < NF_FDG_KB; b++) {
            posb[b] = posn[b];
            rowb[b] = rown[b] >= 0 && nf_attn_tree_visible(posb[b], tree_bits, kv_hi, n_fresh, q_pos, causal,
                                                           swa_type, n_swa) ? rown[b] : -1;
            kr[b] = make_uint4(0u, 0u, 0u, 0u);
            vr[b] = make_uint4(0u, 0u, 0u, 0u);
            if (rowb[b] >= 0) {
                kr[b] = nf_fdg_raw<KVT>(k, (nf_i64)rowb[b] * k_row_stride + k_head);
                vr[b] = nf_fdg_raw<KVT>(v, (nf_i64)rowb[b] * v_row_stride + v_head);
            }
        }
#pragma unroll
        for (int b = 0; b < NF_FDG_KB; b++) {
            nf_fdg_locate(j0 + step + b * GROUPS + grp, n_keys, k_begin, page_shift, page_table, pt_off, kv_pos,
                          &rown[b], &posn[b]);
        }
        float sc[P];
#pragma unroll
        for (int b = 0; b < NF_FDG_KB; b++) {
            float kv[DS];
            nf_fdg_values<KVT>(kr[b], kv);
#pragma unroll
            for (int g = 0; g < GP; g++) {
                float part = 0.0f;
#pragma unroll
                for (int i = 0; i < DS; i++) {
                    part = fmaf(qr[g][i], kv[i], part);
                }
                sc[b * GP + g] = part;
            }
        }
#pragma unroll
        for (int i = 0; i < P; i++) {
#pragma unroll
            for (int off = 1; off < L; off <<= 1) {
                sc[i] += __shfl_xor_sync(0xffffffffu, sc[i], off);
            }
        }
#pragma unroll
        for (int b = 0; b < NF_FDG_KB; b++) {
            if (rowb[b] < 0) {
                continue;
            }
            float vv[DS];
            nf_fdg_values<KVT>(vr[b], vv);
#pragma unroll
            for (int g = 0; g < GP; g++) {
                if (g < G) {
                    const float part = sc[b * GP + g];
                    const float sv = nf_attn_score(KVT == NF_KV_Q8_0 ? part * k_scale : part, scale, softcap)
                                   + nf_attn_bias(bias_mode, h0 + g, n_head, q_pos, posb[b], slope[g], rel_bias,
                                                  n_buckets, max_distance, s);
                    /* one exp per key: the larger of the old maximum and the score rescales by e^0 = 1 exactly */
                    float a = 1.0f;
                    float p;
                    if (sv > m[g]) {
                        a = nf_attn_rescale(m[g], sv);
                        p = 1.0f;
                        m[g] = sv;
                    } else {
                        p = nf_attn_rescale(sv, m[g]);
                    }
                    l[g] = l[g] * a + p;
#pragma unroll
                    for (int i = 0; i < DS; i++) {
                        o[g][i] = fmaf(p, vv[i], o[g][i] * a);
                    }
                }
            }
        }
    }

    /* merge the key groups of each warp (lanes xor L .. xor 16), then the four warps through shared memory:
     * mrg holds per warp NF_FDG_MAX_G maxima, NF_FDG_MAX_G sums and the G x D output rows */
#pragma unroll
    for (int g = 0; g < GP; g++) {
        if (g < G) {
            float mm = m[g];
#pragma unroll
            for (int off = L; off < 32; off <<= 1) {
                mm = fmax(mm, __shfl_xor_sync(0xffffffffu, mm, off));
            }
            const float a = nf_attn_rescale(m[g], mm);
            float ls = l[g] * a;
#pragma unroll
            for (int off = L; off < 32; off <<= 1) {
                ls += __shfl_xor_sync(0xffffffffu, ls, off);
            }
#pragma unroll
            for (int i = 0; i < DS; i++) {
                float x = o[g][i] * a;
#pragma unroll
                for (int off = L; off < 32; off <<= 1) {
                    x += __shfl_xor_sync(0xffffffffu, x, off);
                }
                o[g][i] = x;
            }
            m[g] = mm;
            l[g] = ls;
        }
    }
    const int wstride = 2 * NF_FDG_MAX_G + NF_FDG_MAX_G * D;
    NF_LOCAL_PTR float* mw = mrg + warp * wstride;
    if (lane < L) {
#pragma unroll
        for (int g = 0; g < GP; g++) {
            if (g < G) {
                if (lane == 0) {
                    mw[g] = m[g];
                    mw[NF_FDG_MAX_G + g] = l[g];
                }
#pragma unroll
                for (int i = 0; i < DS; i++) {
                    mw[2 * NF_FDG_MAX_G + g * D + d0 + i] = o[g][i];
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* the work-group's rows: normalized (direct, sinks merged) or the split's (o, m, l) partial */
    const int direct = flags & NF_ATTN_FLAG_DIRECT;
    const int has_sinks = flags & NF_ATTN_FLAG_SINKS;
    const int nw = NF_FDG_WG / 32;
    for (int e = lid; e < G * D; e += NF_FDG_WG) {
        const int g = e / D;
        const int d = e - g * D;
        float mm = -NF_INF;
        for (int w = 0; w < nw; w++) {
            mm = fmax(mm, mrg[w * wstride + g]);
        }
        float acc = 0.0f;
        float ls = 0.0f;
        for (int w = 0; w < nw; w++) {
            const float a = nf_attn_rescale(mrg[w * wstride + g], mm);
            acc += mrg[w * wstride + 2 * NF_FDG_MAX_G + g * D + d] * a;
            ls += mrg[w * wstride + NF_FDG_MAX_G + g] * a;
        }
        if (KVT == NF_KV_Q8_0) {
            acc *= v_scale;
        }
        const int h = h0 + g;
        if (direct) {
            const float2 fl = has_sinks ? nf_attn_sink(mm, ls, sinks[h]) : NF_F2(1.0f, ls);
            out[(nf_i64)out_off + (nf_i64)s * out_seq_stride + (nf_i64)h * D + d]
                = fl.y > 0.0f ? acc * fl.x / fl.y : 0.0f;
        } else {
            NF_GLOBAL float* pr = partial + (((nf_i64)s * n_head + h) * n_split + split) * (D + 2);
            pr[d] = acc;
            if (d == 0) {
                pr[D] = mm;
                pr[D + 1] = ls;
            }
        }
    }
}


/* The work-group's shared array: the warps' merge rows. */
#define NF_FDG_LOCALS                                                                               \
    NF_LOCAL float mrg[(NF_FDG_WG / 32) * (2 * NF_FDG_MAX_G + NF_FDG_MAX_G * NF_FDG_MAX_D)]

/* nf_fdg_run at the launch's head dim D (64 or 128) and head width GP (checked by nf_fdg_shape_ok). */
#define NF_FDG_RUN(D, GP, KV_CODE, PAGE_SHIFT, PT_OFF, Q_SEQ_STRIDE, OUT_SEQ_STRIDE, SEQ, KVH, KV_LO, KV_HI,     \
                   Q_POS, TREE_BITS, PAGE_TABLE)                                                             \
    nf_fdg_run<D, KV_CODE, GP>(n_head, n_head_kv, n_split, split_len, k_row_stride, k_head_stride,               \
        v_row_stride, v_head_stride, scale, softcap, max_bias, flags, swa_type, n_swa, bias_mode,                \
        n_buckets, max_distance, q_off, out_off, PAGE_SHIFT, PT_OFF, Q_SEQ_STRIDE, OUT_SEQ_STRIDE, SEQ,          \
        KVH, KV_LO, KV_HI, Q_POS, TREE_BITS, q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v,            \
        kv_pos, sinks, rel_bias, out, partial, PAGE_TABLE, kv_scales, mrg)

/* nf_fdg_run at the launch's head dim and the group's head count G = n_head / n_head_kv (1 to 4: GP 1, 2, 4). */
#define NF_FDG_DISPATCH(KV_CODE, PAGE_SHIFT, PT_OFF, Q_SEQ_STRIDE, OUT_SEQ_STRIDE, SEQ, KVH, KV_LO, KV_HI,     \
                        Q_POS, TREE_BITS, PAGE_TABLE)                                                        \
    {                                                                                                        \
        const int n_g = n_head / n_head_kv;                                                                  \
        if (d_k == 128) {                                                                                    \
            if (n_g == 1) {                                                                                  \
                NF_FDG_RUN(128, 1, KV_CODE, PAGE_SHIFT, PT_OFF, Q_SEQ_STRIDE, OUT_SEQ_STRIDE, SEQ, KVH, KV_LO, \
                           KV_HI, Q_POS, TREE_BITS, PAGE_TABLE);                                             \
            } else if (n_g == 2) {                                                                           \
                NF_FDG_RUN(128, 2, KV_CODE, PAGE_SHIFT, PT_OFF, Q_SEQ_STRIDE, OUT_SEQ_STRIDE, SEQ, KVH, KV_LO, \
                           KV_HI, Q_POS, TREE_BITS, PAGE_TABLE);                                             \
            } else {                                                                                         \
                NF_FDG_RUN(128, 4, KV_CODE, PAGE_SHIFT, PT_OFF, Q_SEQ_STRIDE, OUT_SEQ_STRIDE, SEQ, KVH, KV_LO, \
                           KV_HI, Q_POS, TREE_BITS, PAGE_TABLE);                                             \
            }                                                                                                \
        } else {                                                                                             \
            if (n_g == 1) {                                                                                  \
                NF_FDG_RUN(64, 1, KV_CODE, PAGE_SHIFT, PT_OFF, Q_SEQ_STRIDE, OUT_SEQ_STRIDE, SEQ, KVH, KV_LO,  \
                           KV_HI, Q_POS, TREE_BITS, PAGE_TABLE);                                             \
            } else if (n_g == 2) {                                                                           \
                NF_FDG_RUN(64, 2, KV_CODE, PAGE_SHIFT, PT_OFF, Q_SEQ_STRIDE, OUT_SEQ_STRIDE, SEQ, KVH, KV_LO,  \
                           KV_HI, Q_POS, TREE_BITS, PAGE_TABLE);                                             \
            } else {                                                                                         \
                NF_FDG_RUN(64, 4, KV_CODE, PAGE_SHIFT, PT_OFF, Q_SEQ_STRIDE, OUT_SEQ_STRIDE, SEQ, KVH, KV_LO,  \
                           KV_HI, Q_POS, TREE_BITS, PAGE_TABLE);                                             \
            }                                                                                                \
        }                                                                                                    \
    }

/*
 * Dense entry: LLM_FlashDecode_<KV>'s arguments, one query row at q_pos over the cache rows [kv_start, kv_end)
 * (identity rows, kv_pos[row] the row's position); work-group (split, kvh) over grid dimension 1 = n_head_kv.
 */
#define NF_FLASH_DECODE_GQA_DENSE_KERNEL(SFX, KV_T, KV_CODE)                                              \
NF_KERNEL void LLM_FlashDecodeGqa_##SFX(                                                                   \
    const int n_head, const int n_head_kv, const int d_k, const int d_v, const int g_tile,                 \
    const int n_split, const int split_len, const int kv_start, const int kv_end,                         \
    const int q_pos,                                                                                      \
    const int k_row_stride, const int k_head_stride, const int v_row_stride,                               \
    const int v_head_stride,                                                                              \
    const float scale, const float softcap, const float max_bias,                                         \
    const int flags, const int swa_type, const int n_swa,                                                 \
    const int bias_mode, const int n_buckets, const int max_distance,                                     \
    const int q_off, const int out_off,                                                                   \
    NF_GLOBAL const float* q,                                                                             \
    NF_GLOBAL const KV_T* k,                                                                              \
    NF_GLOBAL const KV_T* v,                                                                              \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                              \
    NF_GLOBAL const float* NF_RESTRICT sinks,                                                             \
    NF_GLOBAL const float* NF_RESTRICT rel_bias,                                                          \
    NF_GLOBAL float* out,                                                                                 \
    NF_GLOBAL float* partial,                                                                             \
    NF_GLOBAL const float* NF_RESTRICT kv_scales,                                                         \
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)                                                          \
{                                                                                                         \
    NF_FDG_LOCALS;                                                                                        \
    if (!nf_fdg_shape_ok(n_head, n_head_kv, d_k, d_v, n_split, split_len, flags & NF_ATTN_FLAG_DIRECT,     \
                         k_row_stride, k_head_stride, v_row_stride, v_head_stride,                         \
                         (flags & NF_ATTN_FLAG_V_ALIAS) != 0, KV_CODE) || kv_start < 0) {                            \
        nf_fdg_poison(1, n_head, d_v, n_split, flags & NF_ATTN_FLAG_DIRECT, out_off, 0, out, partial);     \
        return;                                                                                           \
    }                                                                                                     \
    const ulong tree_bits = (flags & NF_ATTN_FLAG_TREE) ? tree_mask[0] : (ulong)0;                         \
    NF_FDG_DISPATCH(KV_CODE, -1, 0, 0, 0, 0, (int)get_group_id(1), kv_start, kv_end, q_pos, tree_bits,     \
                    kv_pos)                                                                               \
}

/*
 * Paged entry: LLM_FlashDecodePaged_<KV>'s arguments; work-group (split, s * n_head_kv + kvh) over sequence s's
 * slots [0, seq_kv_len[s]) (clamped to its pages), at its query position q_pos_seq[s].
 */
#define NF_FLASH_DECODE_GQA_PAGED_KERNEL(SFX, KV_T, KV_CODE)                                              \
NF_KERNEL void LLM_FlashDecodePagedGqa_##SFX(NF_FLASH_DECODE_PAGED_RUN_ARGS_PAGED(KV_T))                    \
{                                                                                                         \
    NF_FDG_LOCALS;                                                                                        \
    if (!nf_fdg_shape_ok(n_head, n_head_kv, d_k, d_v, n_split, split_len, flags & NF_ATTN_FLAG_DIRECT,     \
                         k_row_stride, k_head_stride, v_row_stride, v_head_stride,                         \
                         (flags & NF_ATTN_FLAG_V_ALIAS) != 0, KV_CODE)                                     \
            || page_shift < 0 || page_shift > 20 || max_pages < 1 || n_seq < 1                             \
            || q_seq_stride < 0 || out_seq_stride < 0) {                                                   \
        nf_fdg_poison(n_seq, n_head, d_v, n_split, flags & NF_ATTN_FLAG_DIRECT, out_off,                   \
                      out_seq_stride, out, partial);                                                      \
        return;                                                                                           \
    }                                                                                                     \
    const int seq = (int)get_group_id(1) / n_head_kv;                                                     \
    const int kvh = (int)get_group_id(1) - seq * n_head_kv;                                               \
    const nf_i64 kv_cap = (nf_i64)max_pages << page_shift;                                                \
    const int kv_len = seq_kv_len[seq] < 0 ? 0                                                            \
                     : ((nf_i64)seq_kv_len[seq] > kv_cap ? (int)kv_cap : seq_kv_len[seq]);                 \
    const ulong tree_bits = (flags & NF_ATTN_FLAG_TREE) ? tree_mask[seq] : (ulong)0;                       \
    NF_FDG_DISPATCH(KV_CODE, page_shift, seq * max_pages, q_seq_stride, out_seq_stride, seq, kvh, 0,       \
                    kv_len, q_pos_seq[seq], tree_bits, page_table)                                        \
}

NF_FLASH_DECODE_GQA_DENSE_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_DECODE_GQA_DENSE_KERNEL(BF16, ushort, NF_KV_BF16)
NF_FLASH_DECODE_GQA_DENSE_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_FLASH_DECODE_GQA_PAGED_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_DECODE_GQA_PAGED_KERNEL(BF16, ushort, NF_KV_BF16)
NF_FLASH_DECODE_GQA_PAGED_KERNEL(Q8_0, uchar, NF_KV_Q8_0)

#endif /* CUDA */
