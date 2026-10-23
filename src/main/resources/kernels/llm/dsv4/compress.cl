/* dsv4/compress.cl - LLM_DSV4_Compress, LLM_DSV4_LidUpdate_<KV>, LLM_DSV4_StateUpdate, LLM_DSV4_StateSnapshot and LLM_DSV4_CacheStore: DeepSeek-V4 KV compressor and cache commits. Requires: common/dialect.h, common/config.h, common/reduce.h, common/rope_common.h, common/kv_types.h, common/hadamard.h, common/dsv4_common.h */
/*
 * DeepSeek-V4 compressed caches (llama.cpp src/models/deepseek4.cpp, src/llama-kv-cache-dsv4.cpp). Java:
 * gguf.ops.dsv4.Dsv4Kernels (argument lists, geometry, validation) and gguf.ops.dsv4.CpuDsv4Compressor (CPU twin: same
 * formulas, same order).
 *
 * Compressor state of one layer and cache kind: two F32 planes, kv and score, of state_rows x state_width
 * (llama-kv-cache-dsv4.cpp:1319-1335):
 *   CSA and LID (ratio 4, overlapped windows): state_rows = 2*ratio = 8,   state_width = 2*width
 *   HCA (ratio 128):                          state_rows = ratio = 128,  state_width = width
 * Plane row p % state_rows holds the projections of position p: kv = W_kv x_p and score = W_gate x_p + ape[p % ratio]
 * (deepseek4.cpp:953-961, 968-976, 1037-1045; ape row r is ape[r*state_width ..]). The projections of the current
 * ubatch arrive in cur_kv / cur_score, token t at [*_off + t*state_width], the score WITHOUT ape.
 *
 * Compressed row b covers positions [b*ratio, (b+1)*ratio) and is completed by position (b+1)*ratio - 1
 * (llama-kv-cache-dsv4.cpp:523-530). A ubatch of positions [pos0, pos0 + n_tokens) completes the blocks
 * b in [pos0/ratio, (pos0 + n_tokens)/ratio). The window of block b (state_read_idxs) is
 *   overlap: j < ratio      position b*ratio - ratio + j, column c          (previous half, kv/score[0, width))
 *            ratio <= j < 2r position b*ratio + j - ratio, column width + c (current half, [width, 2*width))
 *   plain:   j < ratio      position b*ratio + j, column c
 * (deepseek4.cpp:517-571, 470-498). Position p reads: p < 0 the synthetic row kv 0 / score -INF (dsv4_append_zero_row,
 * deepseek4.cpp:209-215); p >= pos0 the current ubatch row p - pos0 plus ape[p % ratio]; otherwise plane row
 * p % state_rows, which still holds p because every window starts after pos0 - state_rows.
 *
 * Compressed row of block b, channel c < width (deepseek4.cpp:499-511, 572-585):
 *   w_j  = softmax_j(score_j)                           ggml_soft_max over the window (ops.cpp:5897-6000)
 *   y_c  = sum_j w_j * kv_j                             ggml_mul + ggml_sum_rows
 *   y    = y * (1 / sqrt(sum(y^2) / width + eps)) * norm_w   build_norm LLM_NORM_RMS (ops.cpp:4224-4231)
 *   y    = rope(y) on dims [width - n_rot, width) at position b*ratio  ggml_rope_ext + ggml_rope_set_offset
 *   y    = H_rot_block y when rot_block > 0             cache k_rot (deepseek4.cpp:1003-1006, llama-impl.h:57-75)
 * stored as cache row b in the cache element type.
 *
 * Local size of the work-group kernels (Compress, LidUpdate, CacheStore): L, a power of two in [32, LLM_DIMX]
 * (multiple of 32: the subgroup reductions of common/reduce.h need full subgroups); 128 recommended.
 */

/* Window source (kv, score) of position p, state column col; see the file comment. */
NF_INLINE float2 nf_dsv4_src(int p, int col, int pos0, int ratio, int state_rows, int state_width,
                             int cur_kv_off, int cur_score_off,
                             NF_GLOBAL const float* state_kv, NF_GLOBAL const float* state_score,
                             NF_GLOBAL const float* cur_kv, NF_GLOBAL const float* cur_score,
                             NF_GLOBAL const float* ape) {
    if (p < 0) {
        return NF_F2(0.0f, -NF_INF);
    }
    if (p >= pos0) {
        const nf_i64 t = (nf_i64)(p - pos0) * state_width + col;
        return NF_F2(cur_kv[cur_kv_off + t],
                     cur_score[cur_score_off + t] + ape[(p % ratio) * state_width + col]);
    }
    const int r = (p % state_rows) * state_width + col;
    return NF_F2(state_kv[r], state_score[r]);
}

/*
 * Pooled, normalized and rotated row of block b into row[0, width) (see the file comment), up to and excluding the
 * Walsh-Hadamard step. red is the reduction buffer of common/reduce.h. Contains barriers.
 */
NF_INLINE void nf_dsv4_compress_row(
    int b, int pos0, int ratio, int width, int n_rot, int rope_mode, float eps,
    float theta_scale, float freq_scale, float ext_factor, float attn_factor, float corr_lo, float corr_hi,
    int cur_kv_off, int cur_score_off,
    NF_GLOBAL const float* state_kv, NF_GLOBAL const float* state_score,
    NF_GLOBAL const float* cur_kv, NF_GLOBAL const float* cur_score,
    NF_GLOBAL const float* ape, NF_GLOBAL const float* norm_w,
    NF_LOCAL_PTR float* row, NF_LOCAL_PTR float* red) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int overlap = ratio == NF_DSV4_CSA_RATIO;
    const int state_rows = overlap ? 2 * ratio : ratio;
    const int state_width = overlap ? 2 * width : width;
    const int n_win = overlap ? 2 * ratio : ratio;
    const int start = b * ratio;

    float ss = 0.0f;
    for (int c = lid; c < width; c += lsize) {
        float m = -NF_INF;
        for (int j = 0; j < n_win; ++j) {
            const int hi = overlap && j >= ratio;
            const int p = overlap ? (hi ? start + j - ratio : start - ratio + j) : start + j;
            const float2 s = nf_dsv4_src(p, hi ? width + c : c, pos0, ratio, state_rows, state_width,
                                         cur_kv_off, cur_score_off, state_kv, state_score, cur_kv, cur_score, ape);
            m = fmax(m, s.y);
        }
        float l = 0.0f;
        float acc = 0.0f;
        for (int j = 0; j < n_win; ++j) {
            const int hi = overlap && j >= ratio;
            const int p = overlap ? (hi ? start + j - ratio : start - ratio + j) : start + j;
            const float2 s = nf_dsv4_src(p, hi ? width + c : c, pos0, ratio, state_rows, state_width,
                                         cur_kv_off, cur_score_off, state_kv, state_score, cur_kv, cur_score, ape);
            const float e = (s.y == -NF_INF) ? 0.0f : exp(s.y - m);
            l += e;
            acc += e * s.x;
        }
        const float y = acc * (1.0f / l);
        row[c] = y;
        ss += y * y;
    }
    const float sum = nf_wg_sum(red, ss);
    const float scale = 1.0f / sqrt(sum / (float)width + eps);
    for (int c = lid; c < width; c += lsize) {
        row[c] = row[c] * scale * norm_w[c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (n_rot > 0) {
        nf_dsv4_rope_rows(row, 1, width, n_rot, rope_mode, start, 1.0f,
                          theta_scale, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/*
 * Stores local row[0, width) as cache row r (element type kv_type) through nf_kv_store_local_row: F16 / BF16
 * rounded to nearest even, block types by their ggml reference quantizers (common/kv_types.h). No barrier inside;
 * the caller synchronizes before reusing row.
 */
NF_INLINE void nf_dsv4_store_row(NF_GLOBAL uchar* cache, nf_i64 r, NF_LOCAL_PTR const float* row, int width, int kv_type) {
    nf_kv_store_local_row(cache, r * width, row, width, kv_type);
}

/*
 * LLM_DSV4_Compress: compressed rows of the CSA (ratio 4) or HCA (ratio 128) attention cache for every block the
 * ubatch completes, stored as cache rows b. kv_type is the cache element type; rot_block > 0 applies the cache's
 * Walsh-Hadamard rotation (quantized caches, gguf.memory.RotationPlan), 0 none.
 * Launch: 1-D, local L, one work-group per completed block: global = n_blocks * L with
 * n_blocks = (pos0 + n_tokens)/ratio - pos0/ratio (no dispatch when 0).
 */
NF_KERNEL void LLM_DSV4_Compress(
    int pos0, int n_tokens, int ratio, int width, int n_rot, int rope_mode, float eps,
    float theta_scale, float freq_scale, float ext_factor, float attn_factor, float corr_lo, float corr_hi,
    int rot_block, int kv_type, int cur_kv_off, int cur_score_off,
    NF_GLOBAL const float* NF_RESTRICT state_kv,
    NF_GLOBAL const float* NF_RESTRICT state_score,
    NF_GLOBAL const float* NF_RESTRICT cur_kv,
    NF_GLOBAL const float* NF_RESTRICT cur_score,
    NF_GLOBAL const float* NF_RESTRICT ape,
    NF_GLOBAL const float* NF_RESTRICT norm_w,
    NF_GLOBAL uchar* NF_RESTRICT cache) {
    NF_LOCAL float row[NF_DSV4_MAX_WIDTH];
    NF_LOCAL float red[LLM_DIMX];

    const int g = get_group_id(0);
    const int b0 = pos0 / ratio;
    if (g >= (pos0 + n_tokens) / ratio - b0) {
        return;
    }
    const int b = b0 + g;
    nf_dsv4_compress_row(b, pos0, ratio, width, n_rot, rope_mode, eps,
                         theta_scale, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi,
                         cur_kv_off, cur_score_off, state_kv, state_score, cur_kv, cur_score, ape, norm_w, row, red);
    if (rot_block > 0) {
        nf_fwht_local(row, width, rot_block);
    }
    nf_dsv4_store_row(cache, (nf_i64)b, row, width, kv_type);
}

/*
 * LLM_DSV4_LidUpdate_<KV>: lightning-indexer (LID) cache rows for every block the ubatch completes. The
 * LID compressor is the overlapped ratio-4 compressor over the indexer projections (deepseek4.cpp:1037-1080,
 * width = indexer_head_size). Its cache has the context's type_k (llama-kv-cache-dsv4.cpp:1308-1312), the variant's
 * element type, and is always Walsh-Hadamard rotated (llama-kv-cache.cpp:331-336 forces k_rot for the deepseek4
 * indexer), so rot_block > 0 is required. A block type needs width % 32 == 0 (Dsv4Kernels.checkLidCache).
 * Launch: as LLM_DSV4_Compress with ratio 4.
 */
NF_INLINE void nf_dsv4_lid_update(
    int pos0, int n_tokens, int width, int n_rot, int rope_mode, float eps,
    float theta_scale, float freq_scale, float ext_factor, float attn_factor, float corr_lo, float corr_hi,
    int rot_block, int cur_kv_off, int cur_score_off,
    NF_GLOBAL const float* state_kv, NF_GLOBAL const float* state_score,
    NF_GLOBAL const float* cur_kv, NF_GLOBAL const float* cur_score,
    NF_GLOBAL const float* ape, NF_GLOBAL const float* norm_w,
    NF_GLOBAL uchar* cache, int kv_type, NF_LOCAL_PTR float* row, NF_LOCAL_PTR float* red) {
    const int g = get_group_id(0);
    const int b0 = pos0 / NF_DSV4_CSA_RATIO;
    if (g >= (pos0 + n_tokens) / NF_DSV4_CSA_RATIO - b0) {
        return;
    }
    const int b = b0 + g;
    nf_dsv4_compress_row(b, pos0, NF_DSV4_CSA_RATIO, width, n_rot, rope_mode, eps,
                         theta_scale, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi,
                         cur_kv_off, cur_score_off, state_kv, state_score, cur_kv, cur_score, ape, norm_w, row, red);
    nf_fwht_local(row, width, rot_block);
    nf_dsv4_store_row(cache, (nf_i64)b, row, width, kv_type);
}

#define NF_DSV4_LID_KERNEL(SFX, KV_T, KV_CODE)                                                                       \
NF_KERNEL void LLM_DSV4_LidUpdate_##SFX(                                                                             \
    int pos0, int n_tokens, int width, int n_rot, int rope_mode, float eps,                                          \
    float theta_scale, float freq_scale, float ext_factor, float attn_factor, float corr_lo, float corr_hi,          \
    int rot_block, int cur_kv_off, int cur_score_off,                                                                \
    NF_GLOBAL const float* NF_RESTRICT state_kv,                                                                     \
    NF_GLOBAL const float* NF_RESTRICT state_score,                                                                  \
    NF_GLOBAL const float* NF_RESTRICT cur_kv,                                                                       \
    NF_GLOBAL const float* NF_RESTRICT cur_score,                                                                    \
    NF_GLOBAL const float* NF_RESTRICT ape,                                                                          \
    NF_GLOBAL const float* NF_RESTRICT norm_w,                                                                       \
    NF_GLOBAL KV_T* NF_RESTRICT cache) {                                                                             \
    NF_LOCAL float row[NF_DSV4_MAX_WIDTH];                                                                           \
    NF_LOCAL float red[LLM_DIMX];                                                                                    \
    nf_dsv4_lid_update(pos0, n_tokens, width, n_rot, rope_mode, eps, theta_scale, freq_scale, ext_factor,            \
                       attn_factor, corr_lo, corr_hi, rot_block, cur_kv_off, cur_score_off, state_kv, state_score,   \
                       cur_kv, cur_score, ape, norm_w, (NF_GLOBAL uchar*)cache, KV_CODE, row, red);                 \
}

NF_DSV4_LID_KERNEL(F32, float, NF_DSV4_KV_F32)
NF_DSV4_LID_KERNEL(F16, NF_HALF, NF_DSV4_KV_F16)
NF_DSV4_LID_KERNEL(Q8_0, uchar, NF_DSV4_KV_Q8_0)
NF_DSV4_LID_KERNEL(BF16, ushort, NF_DSV4_KV_BF16)
NF_DSV4_LID_KERNEL(Q4_0, uchar, NF_DSV4_KV_Q4_0)
NF_DSV4_LID_KERNEL(Q4_1, uchar, NF_DSV4_KV_Q4_1)
NF_DSV4_LID_KERNEL(IQ4_NL, uchar, NF_DSV4_KV_IQ4_NL)
NF_DSV4_LID_KERNEL(Q5_0, uchar, NF_DSV4_KV_Q5_0)
NF_DSV4_LID_KERNEL(Q5_1, uchar, NF_DSV4_KV_Q5_1)

/*
 * LLM_DSV4_StateUpdate: persists the current ubatch into the compressor planes (state_persist_src/dst_idxs,
 * llama-kv-cache-dsv4.cpp:506-520 and deepseek4.cpp:1025-1031): plane row r receives the latest token t of the ubatch
 * with (pos0 + t) % state_rows == r, kv as is and score plus ape[(pos0 + t) % ratio]; rows no token maps to keep their
 * contents. Runs after every LLM_DSV4_Compress / LLM_DSV4_LidUpdate of the same planes in the step.
 * Launch: 1-D, any local, one work-item per plane element: global >= state_rows * state_width.
 */
NF_KERNEL void LLM_DSV4_StateUpdate(
    int pos0, int n_tokens, int ratio, int width, int cur_kv_off, int cur_score_off,
    NF_GLOBAL const float* NF_RESTRICT cur_kv,
    NF_GLOBAL const float* NF_RESTRICT cur_score,
    NF_GLOBAL const float* NF_RESTRICT ape,
    NF_GLOBAL float* NF_RESTRICT state_kv,
    NF_GLOBAL float* NF_RESTRICT state_score) {
    const int overlap = ratio == NF_DSV4_CSA_RATIO;
    const int state_rows = overlap ? 2 * ratio : ratio;
    const int state_width = overlap ? 2 * width : width;
    const int gid = (int)get_global_id(0);
    if (gid >= state_rows * state_width) {
        return;
    }
    const int r = gid / state_width;
    const int col = gid - r * state_width;
    const int first = (r - pos0 % state_rows + state_rows) % state_rows;
    if (first >= n_tokens) {
        return;
    }
    const int t = first + ((n_tokens - 1 - first) / state_rows) * state_rows;
    const nf_i64 src = (nf_i64)t * state_width + col;
    state_kv[gid] = cur_kv[cur_kv_off + src];
    state_score[gid] = cur_score[cur_score_off + src] + ape[((pos0 + t) % ratio) * state_width + col];
}

/*
 * LLM_DSV4_StateSnapshot: the rollback snapshots of the compressor planes (state_snapshot_src/dst_idxs,
 * llama-kv-cache-dsv4.cpp:684-713 and deepseek4.cpp:245-262, 1013-1023). state_kv / state_score hold 1 + n_snap
 * consecutive planes of state_rows x state_width: the live plane, then the snapshots (n_rs_seq,
 * llama-kv-cache-dsv4.cpp:968-970). Plane d (1..n_snap) receives the live plane as LLM_DSV4_StateUpdate would leave it
 * after only the first n_tokens - d tokens of the ubatch: row r takes the latest such token t with
 * (pos0 + t) % state_rows == r (kv as is, score plus ape[(pos0 + t) % ratio]); rows no such token maps to copy the live
 * row, the state before the ubatch. Dropping the ubatch's last d positions later restores plane d into the live plane.
 * Runs after LLM_DSV4_Compress / LLM_DSV4_LidUpdate and before LLM_DSV4_StateUpdate of the same planes. Reads only the
 * live plane and writes only snapshot planes, so the two buffers are not restrict-qualified.
 * Launch: 1-D, any local, one work-item per snapshot element: global >= n_snap * state_rows * state_width.
 */
NF_KERNEL void LLM_DSV4_StateSnapshot(
    int pos0, int n_tokens, int ratio, int width, int n_snap, int cur_kv_off, int cur_score_off,
    NF_GLOBAL const float* NF_RESTRICT cur_kv,
    NF_GLOBAL const float* NF_RESTRICT cur_score,
    NF_GLOBAL const float* NF_RESTRICT ape,
    NF_GLOBAL float* state_kv,
    NF_GLOBAL float* state_score) {
    const int overlap = ratio == NF_DSV4_CSA_RATIO;
    const int state_rows = overlap ? 2 * ratio : ratio;
    const int state_width = overlap ? 2 * width : width;
    const int plane = state_rows * state_width;
    const int gid = (int)get_global_id(0);
    if (gid >= n_snap * plane) {
        return;
    }
    const int d = 1 + gid / plane;
    const int e = gid - (d - 1) * plane;
    const int r = e / state_width;
    const int col = e - r * state_width;
    const int prefix = n_tokens - d;
    const int first = (r - pos0 % state_rows + state_rows) % state_rows;
    const nf_i64 dst = (nf_i64)d * plane + e;
    if (first < prefix) {
        const int t = first + ((prefix - 1 - first) / state_rows) * state_rows;
        const nf_i64 src = (nf_i64)t * state_width + col;
        state_kv[dst] = cur_kv[cur_kv_off + src];
        state_score[dst] = cur_score[cur_score_off + src] + ape[((pos0 + t) % ratio) * state_width + col];
    } else {
        state_kv[dst] = state_kv[e];
        state_score[dst] = state_score[e];
    }
}

/*
 * LLM_DSV4_CacheStore: stores F32 rows into a K-only cache at rows cell_idx[t] (ggml_set_rows, the DeepSeek-V4 raw
 * window cache cpy_k of deepseek4.cpp:742, 798, 852). Token t reads src[src_off + t*src_stride, + width); rot_block > 0
 * first applies the cache's Walsh-Hadamard rotation (the k_rot that llama rotates kv with before the store,
 * deepseek4.cpp:731-734). Tokens with cell_idx[t] < 0 are skipped.
 * Launch: 1-D, local L, one work-group per token: global = n_tokens * L.
 */
NF_KERNEL void LLM_DSV4_CacheStore(
    int n_tokens, int width, int kv_type, int rot_block, int src_off, int src_stride,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL const int* NF_RESTRICT cell_idx,
    NF_GLOBAL uchar* NF_RESTRICT cache) {
    NF_LOCAL float row[NF_DSV4_MAX_WIDTH];

    const int t = get_group_id(0);
    if (t >= n_tokens) {
        return;
    }
    const int r = cell_idx[t];
    if (r < 0) {
        return;
    }
    NF_GLOBAL const float* x = src + src_off + (nf_i64)t * src_stride;
    for (int c = get_local_id(0); c < width; c += get_local_size(0)) {
        row[c] = x[c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rot_block > 0) {
        nf_fwht_local(row, width, rot_block);
    }
    nf_dsv4_store_row(cache, (nf_i64)r, row, width, kv_type);
}
