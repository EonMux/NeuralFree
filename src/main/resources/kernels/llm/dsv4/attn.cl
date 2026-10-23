/* dsv4/attn.cl - LLM_DSV4_CompressedAttn: DeepSeek-V4 attention over the raw window cache and a compressed cache, with sinks. Requires: common/dialect.h, common/config.h, common/reduce.h, common/rope_common.h, common/kv_types.h, common/attn_common.h, common/hadamard.h, common/dsv4_common.h */
/*
 * DeepSeek-V4 MQA attention (llama.cpp src/models/deepseek4.cpp:714-860, 1185-1215). Java: gguf.ops.dsv4.Dsv4Kernels
 * (arguments, geometry, head tiling) and gguf.ops.dsv4.CpuDsv4Attention (CPU twin: same formulas and order).
 *
 * One shared K row per cached token serves as both K and V for every query head (build_attn_mha(q, k_all, k_all),
 * n_head_kv = 1). Query token t, head h, position p = q_pos[t]; q and out rows are [t][h][head_dim] at
 * q_off / out_off. The keys are the concatenation (ggml_concat of raw_k and the compressed cache) of
 *   raw rows r < n_raw:   visible when 0 <= raw_pos[r] <= p and p - raw_pos[r] < n_swa (causal STANDARD window,
 *                         llama_hparams::is_masked_swa; raw_pos is the layer's kv_pos buffer, -1 for empty rows)
 *   compressed rows       comp_mode 0 (window-only layer): none
 *                         comp_mode 1 (HCA, dense): rows c < n_comp
 *                         comp_mode 2 (CSA, lightning-indexer selection): rows sel[t*sel_stride + j], j < n_comp,
 *                           negative entries skipped; the entries of one token must be distinct (LLM_TopK_Rows)
 *                         a compressed row c is visible when c < (p + 1) / ratio (n_visible of the comp plan,
 *                         llama-graph.cpp:834-870; the top-k mask adds that mask, deepseek4.cpp:686-712)
 * With s_k = (q_rot . k_k) * kq_scale over visible keys and sink = sinks[h] when has_sinks (ggml soft_max with
 * sinks, ops.cpp:5978-5993):
 *   m = max(max_k s_k, sink), l = sum_k e^(s_k - m) + e^(sink - m), o = (1/l) sum_k e^(s_k - m) k_k
 * computed as an online softmax over chunks of L keys. q_rot = H q and out = H o when rot_block > 0 (the k_rot
 * rotation of q, kv and the output for quantized caches, deepseek4.cpp:731-734, 768-769; H is its own inverse). Finally the
 * last n_rot dims of out are rotated back at position p (ggml_rope_ext_back + ggml_rope_set_offset,
 * deepseek4.cpp:1211-1213) with the layer's RoPE parameters; n_rot = 0 skips it.
 *
 * Launch: 1-D, local L (power of two in [32, LLM_DIMX]; 128 recommended), one work-group per (token, tile of g_tile
 * heads): group = t * ceil(n_head / g_tile) + tile, global = n_tokens * ceil(n_head / g_tile) * L. Local budget
 * (Dsv4Kernels.attnHeadTile): g_tile * (2 * head_dim + L) <= NF_FA_SMEM_FLOATS, g_tile <= NF_DSV4_ATTN_MAX_G.
 * head_dim is a multiple of 4 (of 32 for block-type caches) and at most NF_DSV4_MAX_WIDTH; rot_block divides head_dim.
 */

#define NF_DSV4_ATTN_MAX_G 16      /* largest head tile (per-work-item softmax state) */

#define NF_DSV4_COMP_NONE     0
#define NF_DSV4_COMP_DENSE    1
#define NF_DSV4_COMP_SELECTED 2

NF_KERNEL void LLM_DSV4_CompressedAttn(
    int n_tokens, int n_head, int head_dim, int g_tile, int q_off, int out_off,
    int n_raw, int raw_type, int n_swa,
    int comp_mode, int n_comp, int comp_type, int ratio, int sel_stride,
    int has_sinks, float kq_scale, int rot_block,
    int n_rot, int rope_mode, float theta_scale, float freq_scale, float ext_factor, float attn_factor,
    float corr_lo, float corr_hi,
    NF_GLOBAL const float* NF_RESTRICT q,
    NF_GLOBAL const uchar* NF_RESTRICT raw_k,
    NF_GLOBAL const int* NF_RESTRICT raw_pos,
    NF_GLOBAL const uchar* NF_RESTRICT comp_k,
    NF_GLOBAL const int* NF_RESTRICT sel,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL float* NF_RESTRICT out) {
    NF_LOCAL float smem[NF_FA_SMEM_FLOATS];
    NF_LOCAL float red[LLM_DIMX];
    NF_LOCAL int key_code[LLM_DIMX];

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n_tiles = (n_head + g_tile - 1) / g_tile;
    const int grp = get_group_id(0);
    const int t = grp / n_tiles;
    if (t >= n_tokens) {
        return;
    }
    const int h0 = (grp - t * n_tiles) * g_tile;
    const int G = min(g_tile, n_head - h0);
    const int hd = head_dim;

    /* Local layout: q tile [G][hd] | output accumulators [G][hd] | chunk weights [G][L]. */
    NF_LOCAL_PTR float* qs = smem;
    NF_LOCAL_PTR float* acc = smem + G * hd;
    NF_LOCAL_PTR float* pw = smem + 2 * G * hd;

    const int p = q_pos[t];
    const int n_vis_c = (comp_mode != NF_DSV4_COMP_NONE) ? (p + 1) / ratio : 0;
    const int n_keys = n_raw + ((comp_mode != NF_DSV4_COMP_NONE) ? n_comp : 0);

    NF_GLOBAL const float* qrow = q + q_off + ((nf_i64)t * n_head + h0) * hd;
    for (int i = lid; i < G * hd; i += lsize) {
        qs[i] = qrow[i];
        acc[i] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rot_block > 0) {
        nf_fwht_local(qs, G * hd, rot_block);
    }

    float m_run[NF_DSV4_ATTN_MAX_G];
    float l_run[NF_DSV4_ATTN_MAX_G];
    float alpha[NF_DSV4_ATTN_MAX_G];
    for (int g = 0; g < G; ++g) {
        m_run[g] = -NF_INF;
        l_run[g] = 0.0f;
    }

    for (int kb = 0; kb < n_keys; kb += lsize) {
        /* Key of this work-item: code >= 0 raw row, code <= -2 compressed row -(code + 2), -1 masked. */
        const int j = kb + lid;
        int code = -1;
        if (j < n_raw) {
            const int kp = raw_pos[j];
            if (kp >= 0 && kp <= p && p - kp < n_swa) {
                code = j;
            }
        } else if (j < n_keys) {
            const int jc = j - n_raw;
            const int c = (comp_mode == NF_DSV4_COMP_SELECTED) ? sel[(nf_i64)t * sel_stride + jc] : jc;
            if (c >= 0 && c < n_vis_c) {
                code = -(c + 2);
            }
        }
        key_code[lid] = code;

        float dots[NF_DSV4_ATTN_MAX_G];
        for (int g = 0; g < G; ++g) {
            dots[g] = 0.0f;
        }
        if (code != -1) {
            NF_GLOBAL const uchar* base = code >= 0 ? raw_k : comp_k;
            const int ktype = code >= 0 ? raw_type : comp_type;
            const nf_i64 e0 = (nf_i64)(code >= 0 ? code : -code - 2) * hd;
            for (int d = 0; d < hd; d += 4) {
                const float4 k4 = nf_kv_load4(base, e0 + d, ktype);
                for (int g = 0; g < G; ++g) {
                    NF_LOCAL_PTR const float* qg = qs + g * hd + d;
                    dots[g] += k4.x * qg[0] + k4.y * qg[1] + k4.z * qg[2] + k4.w * qg[3];
                }
            }
        }
        for (int g = 0; g < G; ++g) {
            const float s = (code != -1) ? dots[g] * kq_scale : -NF_INF;
            const float cm = nf_wg_max(red, s);
            const float m_new = fmax(m_run[g], cm);
            const float e = (s == -NF_INF) ? 0.0f : exp(s - m_new);
            alpha[g] = (m_run[g] == -NF_INF) ? 0.0f : exp(m_run[g] - m_new);
            pw[g * lsize + lid] = e;
            const float cs = nf_wg_sum(red, e);
            l_run[g] = l_run[g] * alpha[g] + cs;
            m_run[g] = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Accumulate: each work-item owns dims d = lid + k*L of every head of the tile. */
        const int n_chunk = min(lsize, n_keys - kb);
        for (int d = lid; d < hd; d += lsize) {
            float a[NF_DSV4_ATTN_MAX_G];
            for (int g = 0; g < G; ++g) {
                a[g] = acc[g * hd + d] * alpha[g];
            }
            for (int jj = 0; jj < n_chunk; ++jj) {
                const int cj = key_code[jj];
                if (cj == -1) {
                    continue;
                }
                const float v = cj >= 0 ? nf_kv_load1(raw_k, (nf_i64)cj * hd + d, raw_type)
                                        : nf_kv_load1(comp_k, (nf_i64)(-cj - 2) * hd + d, comp_type);
                for (int g = 0; g < G; ++g) {
                    a[g] += pw[g * lsize + jj] * v;
                }
            }
            for (int g = 0; g < G; ++g) {
                acc[g * hd + d] = a[g];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Sinks and normalization (ops.cpp:5978-5993), then un-rotation and the inverse RoPE of the tail dims. */
    float fac[NF_DSV4_ATTN_MAX_G];
    for (int g = 0; g < G; ++g) {
        if (has_sinks) {
            const float sink = sinks[h0 + g];
            const float m2 = fmax(m_run[g], sink);
            const float a = (m_run[g] == -NF_INF) ? 0.0f : exp(m_run[g] - m2);
            fac[g] = a / (l_run[g] * a + exp(sink - m2));
        } else {
            fac[g] = l_run[g] > 0.0f ? 1.0f / l_run[g] : 0.0f;
        }
    }
    for (int i = lid; i < G * hd; i += lsize) {
        acc[i] *= fac[i / hd];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rot_block > 0) {
        nf_fwht_local(acc, G * hd, rot_block);
    }
    if (n_rot > 0) {
        nf_dsv4_rope_rows(acc, G, hd, n_rot, rope_mode, p, -1.0f,
                          theta_scale, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    NF_GLOBAL float* orow = out + out_off + ((nf_i64)t * n_head + h0) * hd;
    for (int i = lid; i < G * hd; i += lsize) {
        orow[i] = acc[i];
    }
}
