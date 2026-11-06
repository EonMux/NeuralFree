/* multitask.cl - LLM_MultiTask, the persistent multi-task kernel of the decode path. Requires: common/dialect.h,
 * common/config.h, common/math.h, common/reduce.h, common/rope_common.h */
/*
 * The decode path launches many independent small kernels back to back (norms, rotations, gate preparation and
 * short F32 projections). Their bodies are recorded as tasks of one dispatch instead: every workgroup of the
 * launch takes a block index of one task and runs that task's body, so the whole set costs one launch, like
 * ggml-cuda/mk.cu. Tasks of one launch must be independent (they read and write disjoint views); the host records
 * only tasks it knows to be independent ({@code gguf.ops.fused.GpuMultiTask}).
 *
 * Descriptor layout: one record of NF_MT_WORDS floats per task in a read-only buffer, NF_MT_MAX_TASKS records.
 *   word 0          task kind
 *   words 2..12     indices into the launch's eleven buffer arguments, per task buffer slot
 *   words 14..33    integer parameters 0..19 (bit patterns of int, read with nf_mt_i)
 *   words 34..41    float parameters 0..7
 *
 * Task kinds and their slots (integer parameters are i0.., float parameters f0..):
 *   ROW_NORM    buffers x, w, b, y;   i0 rows_per_token, i1 row_len, i2 row_stride, i3 mode, i4 flags,
 *                                     i5 w_rows, i6 x_off, i7 y_off; f0 eps
 *   NORM_ROPE   buffers pos, ff, w, b, x, y;
 *                                     i0 n_heads, i1 head_stride, i2 row_stride, i3 row_len, i4 mode, i5 n_dims,
 *                                     i6 n_offs, i7 has_ff, i8 flags, i9 w_rows, i10..13 sections,
 *                                     i14 x_off, i15 y_off; f0 eps, f1 theta_scale, f2 freq_scale,
 *                                     f3 ext_factor, f4 attn_factor, f5 corr_lo, f6 corr_hi
 *   GATE_PREP   buffers a_in, b_in, dt_bias, ssm_a, g_out, beta_out;
 *                                     i0 n_v_heads, i1 n_k_heads, i2 layout, i3 kda_mode, i4 head_dim,
 *                                     i5 a_stride, i6 a_off, i7 b_stride, i8 b_off; f0 lb
 *   MATVEC_F32  buffers w, x, out;    i0 rows, i1 cols, i2 x_stride, i3 x_off, i4 out_stride, i5 out_off
 *
 * Dispatch: 1-D, local = L (the row-reduce local size), blocks_per_task = B, global = n_tasks*B*L. A workgroup's
 * task is (group_id / B) and its block index within the task is (group_id % B); tasks with fewer rows or elements
 * than B*L loop fewer times. The token count is the only argument that changes per step.
 */

#define NF_MT_MAX_TASKS 4
#define NF_MT_WORDS     48
#define NF_MT_BUFS      11

#define NF_MT_ROW_NORM   1
#define NF_MT_NORM_ROPE  2
#define NF_MT_GATE_PREP  3
#define NF_MT_MATVEC_F32 4

/* LLM_RowNorm flags (gguf.ops.norm.NormKernels). */
#define NF_MT_HAS_W 1
#define NF_MT_HAS_B 2

/* Task kind (word 0). */
NF_INLINE int nf_mt_kind(NF_GLOBAL const float* rec) {
#ifdef NF_CUDA
    return __float_as_int(rec[0]);
#else
    return as_int(rec[0]);
#endif
}

/* Integer word k of a task record (words 14..33). */
NF_INLINE int nf_mt_i(NF_GLOBAL const float* rec, int k) {
#ifdef NF_CUDA
    return __float_as_int(rec[14 + k]);
#else
    return as_int(rec[14 + k]);
#endif
}

/* Buffer slot k of a task record. */
NF_INLINE int nf_mt_slot(NF_GLOBAL const float* rec, int k) {
#ifdef NF_CUDA
    return __float_as_int(rec[2 + k]);
#else
    return as_int(rec[2 + k]);
#endif
}

#define NF_MT_ARGS NF_GLOBAL float* b0, NF_GLOBAL float* b1, NF_GLOBAL float* b2, NF_GLOBAL float* b3, \
                   NF_GLOBAL float* b4, NF_GLOBAL float* b5, NF_GLOBAL float* b6, NF_GLOBAL float* b7, \
                   NF_GLOBAL float* b8, NF_GLOBAL float* b9, NF_GLOBAL float* b10
#define NF_MT_PASS b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10

#define NF_MT_RECV NF_GLOBAL const float* rec, NF_LOCAL_PTR float* red

NF_INLINE NF_GLOBAL float* nf_mt_buf(int idx, NF_MT_ARGS) {
    switch (idx) {
        case 0:  return b0;
        case 1:  return b1;
        case 2:  return b2;
        case 3:  return b3;
        case 4:  return b4;
        case 5:  return b5;
        case 6:  return b6;
        case 7:  return b7;
        case 8:  return b8;
        case 9:  return b9;
        default: return b10;
    }
}

/* One RMS/L2_GDN row of a ROW_NORM task. */
NF_INLINE void nf_mt_row_norm(NF_MT_RECV, int row, int lid, int lsize, NF_MT_ARGS) {
    const int row_len = nf_mt_i(rec, 1);
    const int row_stride = nf_mt_i(rec, 2);
    const int mode = nf_mt_i(rec, 3);
    const int flags = nf_mt_i(rec, 4);
    const int w_rows = nf_mt_i(rec, 5);
    const float eps = rec[34];
    NF_GLOBAL float* x = nf_mt_buf(nf_mt_slot(rec, 0), NF_MT_PASS);
    NF_GLOBAL float* w = nf_mt_buf(nf_mt_slot(rec, 1), NF_MT_PASS);
    NF_GLOBAL float* b = nf_mt_buf(nf_mt_slot(rec, 2), NF_MT_PASS);
    NF_GLOBAL float* y = nf_mt_buf(nf_mt_slot(rec, 3), NF_MT_PASS);

    const nf_i64 xb = (nf_i64)nf_mt_i(rec, 6) + (nf_i64)row * row_stride;
    const nf_i64 yb = (nf_i64)nf_mt_i(rec, 7) + (nf_i64)row * row_stride;
    float ss = 0.0f;
    for (int i = lid; i < row_len; i += lsize) {
        const float v = x[xb + i];
        ss += v * v;
    }
    const float sum = nf_wg_sum(red, ss);
    float scale;
    float post = 1.0f;
    if (mode == 3) {
        scale = rsqrt(sum / (float)row_len + eps / (float)row_len);
        post = rsqrt((float)row_len);
    } else {
        scale = rsqrt(sum / (float)row_len + eps);
    }
    const int has_w = (flags & NF_MT_HAS_W) != 0;
    const int has_b = (flags & NF_MT_HAS_B) != 0;
    const nf_i64 wb = (nf_i64)(w_rows > 1 ? row % w_rows : 0) * row_len;
    for (int i = lid; i < row_len; i += lsize) {
        float v = x[xb + i] * scale;
        if (mode == 3) {
            v *= post;
        }
        if (has_w) {
            v *= w[wb + i];
        }
        if (has_b) {
            v += b[wb + i];
        }
        y[yb + i] = v;
    }
}

/* One NORM_ROPE row: RMS-normalize (token t, head h) into y, then rotate it in place. The rotation phase reads
 * values written by the workgroup's other work-items, so the barrier between the phases fences global memory. */
NF_INLINE void nf_mt_norm_rope(NF_MT_RECV, int t, int h, int pos_stride, int lid, int lsize, NF_MT_ARGS) {
    NF_GLOBAL float* pos = nf_mt_buf(nf_mt_slot(rec, 0), NF_MT_PASS);
    NF_GLOBAL float* ff = nf_mt_buf(nf_mt_slot(rec, 1), NF_MT_PASS);
    NF_GLOBAL float* w = nf_mt_buf(nf_mt_slot(rec, 2), NF_MT_PASS);
    NF_GLOBAL float* bb = nf_mt_buf(nf_mt_slot(rec, 3), NF_MT_PASS);
    NF_GLOBAL float* x = nf_mt_buf(nf_mt_slot(rec, 4), NF_MT_PASS);
    NF_GLOBAL float* y = nf_mt_buf(nf_mt_slot(rec, 5), NF_MT_PASS);
    const int head_stride = nf_mt_i(rec, 1);
    const int row_stride = nf_mt_i(rec, 2);
    const int row_len = nf_mt_i(rec, 3);
    const int mode = nf_mt_i(rec, 4);
    const int n_dims = nf_mt_i(rec, 5);
    const int n_offs = nf_mt_i(rec, 6);
    const int has_ff = nf_mt_i(rec, 7);
    const int flags = nf_mt_i(rec, 8);
    const int w_rows = nf_mt_i(rec, 9);
    const int s0 = nf_mt_i(rec, 10);
    const int s1 = nf_mt_i(rec, 11);
    const int s2 = nf_mt_i(rec, 12);
    const int s3 = nf_mt_i(rec, 13);
    const int x_off = nf_mt_i(rec, 14);
    const int y_off = nf_mt_i(rec, 15);
    const float eps = rec[34];
    const float theta_scale = rec[35];
    const float freq_scale = rec[36];
    const float ext_factor = rec[37];
    const float attn_factor = rec[38];
    const float corr_lo = rec[39];
    const float corr_hi = rec[40];

    const nf_i64 xb = (nf_i64)x_off + (nf_i64)t * row_stride + (nf_i64)h * head_stride;
    const nf_i64 yb = (nf_i64)y_off + (nf_i64)t * row_stride + (nf_i64)h * head_stride;
    float ss = 0.0f;
    for (int i = lid; i < row_len; i += lsize) {
        const float v = x[xb + i];
        ss += v * v;
    }
    const float rstd = rsqrt(nf_wg_sum(red, ss) / (float)row_len + eps);
    const int has_w = (flags & NF_MT_HAS_W) != 0;
    const int has_b = (flags & NF_MT_HAS_B) != 0;
    const nf_i64 wrow = (nf_i64)(w_rows > 1 ? h % w_rows : 0) * row_len;
    for (int i = lid; i < row_len; i += lsize) {
        float v = x[xb + i] * rstd;
        if (has_w) {
            v *= w[wrow + i];
        }
        if (has_b) {
            v += bb[wrow + i];
        }
        y[yb + i] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    const int pairs = nf_rope_pairs(mode, n_dims);
    for (int i = lid; i < pairs; i += lsize) {
        const int sec = nf_rope_section(mode, i, s0, s1, s2, s3);
        const int e = nf_rope_theta_exp(mode, i, sec, s0, s1, s2, s3);
#ifdef NF_CUDA
        const int p = __float_as_int(pos[t + sec * pos_stride]);
#else
        const int p = as_int(pos[t + sec * pos_stride]);
#endif
        const float f = has_ff ? ff[i] : 1.0f;
        const float2 cs = nf_rope_cs(p, e, theta_scale, f, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi,
                                     i);
        const nf_i64 a = yb + n_offs + nf_rope_first(mode, i);
        const nf_i64 c = a + nf_rope_span(mode, n_dims);
        const float y0 = y[a];
        const float y1 = y[c];
        y[a] = y0 * cs.x - y1 * cs.y;
        y[c] = y0 * cs.y + y1 * cs.x;
    }
}

/* One GATE_PREP element (global index gid). */
NF_INLINE void nf_mt_gate_prep(NF_MT_RECV, nf_i64 gid, int n_tokens, NF_MT_ARGS) {
    NF_GLOBAL float* a_in = nf_mt_buf(nf_mt_slot(rec, 0), NF_MT_PASS);
    NF_GLOBAL float* b_in = nf_mt_buf(nf_mt_slot(rec, 1), NF_MT_PASS);
    NF_GLOBAL float* dt_bias = nf_mt_buf(nf_mt_slot(rec, 2), NF_MT_PASS);
    NF_GLOBAL float* ssm_a = nf_mt_buf(nf_mt_slot(rec, 3), NF_MT_PASS);
    NF_GLOBAL float* g_out = nf_mt_buf(nf_mt_slot(rec, 4), NF_MT_PASS);
    NF_GLOBAL float* beta_out = nf_mt_buf(nf_mt_slot(rec, 5), NF_MT_PASS);
    const int n_v_heads = nf_mt_i(rec, 0);
    const int n_k_heads = nf_mt_i(rec, 1);
    const int layout = nf_mt_i(rec, 2);
    const int kda_mode = nf_mt_i(rec, 3);
    const int head_dim = nf_mt_i(rec, 4);
    const int a_stride = nf_mt_i(rec, 5);
    const int a_off = nf_mt_i(rec, 6);
    const int b_stride = nf_mt_i(rec, 7);
    const int b_off = nf_mt_i(rec, 8);
    const float lb = rec[34];
    const float beta_scale = rec[35];
    const int per_head = kda_mode < 0 ? 1 : head_dim;
    if (gid >= (nf_i64)n_tokens * n_v_heads * per_head) {
        return;
    }
    const int d = (int)(gid % per_head);
    const nf_i64 th = gid / per_head;
    const int h = (int)(th % n_v_heads);
    const int t = (int)(th / n_v_heads);
    int a_idx;
    int b_idx;
    if (layout == 1) {
        const int R = n_v_heads / n_k_heads;
        const int kh = h / R;
        const int r = h - kh * R;
        b_idx = kh * 2 * R + r;
        a_idx = b_idx + R;
    } else if (layout == 2) {
        b_idx = h;
        a_idx = n_v_heads + h;
    } else {
        a_idx = kda_mode < 0 ? h : h * head_dim + d;
        b_idx = h;
    }
    const int j = kda_mode < 0 ? h : h * head_dim + d;
    const float xv = a_in[(nf_i64)a_off + (nf_i64)t * a_stride + a_idx] + dt_bias[j];
    const float A = ssm_a[h];
    float gv;
    if (kda_mode == 1) {
        gv = lb * nf_sigmoid(-(A * xv));
    } else if (kda_mode == 2) {
        gv = lb * nf_sigmoid(A * xv);
    } else {
        gv = nf_softplus(xv) * A;
    }
    g_out[gid] = gv;
    if (d == 0) {
        beta_out[th] = beta_scale * nf_sigmoid(b_in[(nf_i64)b_off + (nf_i64)t * b_stride + b_idx]);
    }
}

/* One row of a MATVEC_F32 task: out row r of token t is the dot of weight row r with input row t. */
NF_INLINE void nf_mt_matvec_f32(NF_MT_RECV, int row, int t, int lid, int lsize, NF_MT_ARGS) {
    NF_GLOBAL float* w = nf_mt_buf(nf_mt_slot(rec, 0), NF_MT_PASS);
    NF_GLOBAL float* x = nf_mt_buf(nf_mt_slot(rec, 1), NF_MT_PASS);
    NF_GLOBAL float* out = nf_mt_buf(nf_mt_slot(rec, 2), NF_MT_PASS);
    const int cols = nf_mt_i(rec, 1);
    const int x_stride = nf_mt_i(rec, 2);
    const int x_off = nf_mt_i(rec, 3);
    const int out_stride = nf_mt_i(rec, 4);
    const int out_off = nf_mt_i(rec, 5);
    const nf_i64 wb = (nf_i64)row * cols;
    const nf_i64 xb = (nf_i64)x_off + (nf_i64)t * x_stride;
    float acc = 0.0f;
    for (int k = lid; k < cols; k += lsize) {
        acc += w[wb + k] * x[xb + k];
    }
    acc = nf_wg_sum(red, acc);
    if (lid == 0) {
        out[(nf_i64)out_off + (nf_i64)t * out_stride + row] = acc;
    }
}

NF_KERNEL void LLM_MultiTask(
    const int n_tokens,
    const int n_tasks,
    const int blocks_per_task,
    NF_GLOBAL const float* NF_RESTRICT desc,
    NF_MT_ARGS)
{
    const int group = (int)get_group_id(0);
    const int bpt = blocks_per_task;
    const int task = group / bpt;
    if (task >= n_tasks || task >= NF_MT_MAX_TASKS) {
        return;
    }
    const int tb = group - task * bpt;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float red[LLM_DIMX];
    NF_GLOBAL const float* rec = desc + (nf_i64)task * NF_MT_WORDS;
    const int kind = nf_mt_kind(rec);
    if (kind == NF_MT_ROW_NORM) {
        const int rows = n_tokens * nf_mt_i(rec, 0);
        for (int row = tb; row < rows; row += bpt) {
            nf_mt_row_norm(rec, red, row, lid, lsize, NF_MT_PASS);
        }
    } else if (kind == NF_MT_NORM_ROPE) {
        const int n_heads = nf_mt_i(rec, 0);
        const int rows = n_tokens * n_heads;
        for (int cell = tb; cell < rows; cell += bpt) {
            const int h = cell % n_heads;
            const int t = cell / n_heads;
            nf_mt_norm_rope(rec, red, t, h, n_tokens, lid, lsize, NF_MT_PASS);
        }
    } else if (kind == NF_MT_GATE_PREP) {
        const int per_head = nf_mt_i(rec, 3) < 0 ? 1 : nf_mt_i(rec, 4);
        const nf_i64 n = (nf_i64)n_tokens * nf_mt_i(rec, 0) * per_head;
        for (nf_i64 gid = (nf_i64)tb * lsize + lid; gid < n; gid += (nf_i64)bpt * lsize) {
            nf_mt_gate_prep(rec, red, gid, n_tokens, NF_MT_PASS);
        }
    } else if (kind == NF_MT_MATVEC_F32) {
        const int rows = nf_mt_i(rec, 0);
        const int total = n_tokens * rows;
        for (int cell = tb; cell < total; cell += bpt) {
            const int r = cell % rows;
            const int t = cell / rows;
            nf_mt_matvec_f32(rec, red, r, t, lid, lsize, NF_MT_PASS);
        }
    }
}
