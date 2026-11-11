/* topk_select.h - work-group top-k selection over one row: radix select of the k-th largest key, then an ordered
 * compaction. Requires: common/dialect.h, common/config.h */
/*
 * Shared by LLM_TopK_Rows (sparse/topk_rows.cl) and LLM_Row_TopK (sampling_topk.cl); GLSL twin:
 * shaders/llm/common/topk_select.glsl.
 *
 * Eligibility: with_neg_inf = 0 selects only scores greater than -INF (the sparse indexers mask with -INF), 1
 * selects every non-NaN score (-INF included, as the sampling readouts rank them). NaN is never selected.
 *
 * nf_topk_select: three digit passes of 11, 11 and 10 bits of the order-preserving key over a NF_TOPK_BINS-bin
 * local histogram find the key T of the kk-th largest eligible score, kk = min(k, eligible count), and how many of
 * the selected scores equal T (the rest are above it). Every work-item returns the same values.
 *
 * nf_topk_compact: writes to sel[0 .. kk) the columns above T in increasing column order, then the first
 * (lowest-column) n_eq columns equal to T in increasing column order. Each work-item owns one contiguous column
 * range: it counts its above and equal columns, one exclusive scan over the work-items' counts places every range,
 * and a second pass writes the columns, so the pass costs one scan instead of one per block of columns.
 *
 * Both run every barrier in uniform control flow; the local size is a power of two of at most LLM_DIMX (and at most
 * NF_TOPK_BINS). Equal floats are equal keys (+0 and -0 included).
 */

#define NF_TOPK_BINS 2048

/* Order-preserving key: a > b as floats  <=>  key(a) > key(b) as unsigned; -0 maps to the key of +0. */
NF_INLINE uint nf_topk_key(float x) {
    const uint u = (x == 0.0f) ? 0u : as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* 1 when x takes part in the selection (see the eligibility rule above). */
NF_INLINE int nf_topk_eligible(float x, int with_neg_inf) {
    return with_neg_inf ? !isnan(x) : x > -NF_INF;
}

/* In-place inclusive suffix sums over the work-group: part[i] becomes sum of part[i .. L - 1]. */
NF_INLINE void nf_topk_suffix_scan(NF_LOCAL_PTR uint* part, int lid, int lsize) {
    for (int off = 1; off < lsize; off <<= 1) {
        const uint v = part[lid] + (lid + off < lsize ? part[lid + off] : 0u);
        barrier(CLK_LOCAL_MEM_FENCE);
        part[lid] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/*
 * The kk-th largest key T of row[0 .. n) (*key_out), kk = min(k, eligible) (*kk_out) and how many selected columns
 * equal T (*eq_out); kk == 0 leaves T and the count 0. hist (NF_TOPK_BINS), part (local size) and pick (2) are the
 * work-group's scratch.
 */
NF_INLINE void nf_topk_select(NF_GLOBAL const float* row, int n, int k, int with_neg_inf,
                              NF_LOCAL_PTR uint* hist, NF_LOCAL_PTR uint* part, NF_LOCAL_PTR uint* pick,
                              uint* key_out, uint* kk_out, uint* eq_out) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int per = NF_TOPK_BINS / lsize;
    uint prefix = 0u, pmask = 0u, kk = 0u, remaining = 0u;
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
        const uint dmask = pass == 2 ? 0x3FFu : 0x7FFu;
        for (int b = lid; b < NF_TOPK_BINS; b += lsize) {
            hist[b] = 0u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int c = lid; c < n; c += lsize) {
            const float x = row[c];
            if (nf_topk_eligible(x, with_neg_inf)) {
                const uint key = nf_topk_key(x);
                if ((key & pmask) == prefix) {
                    atomic_inc(&hist[(key >> shift) & dmask]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        uint mine = 0u;
        for (int i = 0; i < per; ++i) {
            mine += hist[lid * per + i];
        }
        part[lid] = mine;
        barrier(CLK_LOCAL_MEM_FENCE);
        /* part[i] = candidates in the bins of work-items i .. L-1 */
        nf_topk_suffix_scan(part, lid, lsize);
        if (pass == 0) {
            kk = min((uint)max(k, 0), part[0]);
            remaining = kk;
        }
        if (kk == 0u) {
            break;   /* uniform: nothing eligible, or k == 0 */
        }
        /* The kk-th largest key lies in the digit d with #(digits > d) < remaining <= #(digits >= d). */
        uint above = lid + 1 < lsize ? part[lid + 1] : 0u;
        if (above < remaining && remaining <= above + mine) {
            for (int i = per - 1; i >= 0; --i) {
                const uint h = hist[lid * per + i];
                if (above + h >= remaining) {
                    pick[0] = (uint)(lid * per + i);
                    pick[1] = remaining - above;
                    break;
                }
                above += h;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        prefix |= pick[0] << shift;
        pmask |= dmask << shift;
        remaining = pick[1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    *key_out = prefix;
    *kk_out = kk;
    *eq_out = kk == 0u ? 0u : remaining;
}

/*
 * The ordered compaction (see the header): sel[0 .. n_above) the columns above key T, then sel[n_above ..
 * n_above + n_eq) the first n_eq columns equal to it. part_a and part_e are local-size scratch arrays.
 */
NF_INLINE void nf_topk_compact(NF_GLOBAL const float* row, int n, int with_neg_inf, uint key_k, uint n_above,
                               uint n_eq, NF_LOCAL_PTR uint* part_a, NF_LOCAL_PTR uint* part_e,
                               NF_LOCAL_PTR int* sel) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int span = n / lsize + (n % lsize != 0 ? 1 : 0);
    const int c0 = min(n, lid * span);
    const int c1 = min(n, c0 + span);
    uint na = 0u, ne = 0u;
    for (int c = c0; c < c1; c++) {
        const float x = row[c];
        if (nf_topk_eligible(x, with_neg_inf)) {
            const uint key = nf_topk_key(x);
            na += key > key_k ? 1u : 0u;
            ne += key == key_k ? 1u : 0u;
        }
    }
    part_a[lid] = na;
    part_e[lid] = ne;
    barrier(CLK_LOCAL_MEM_FENCE);
    nf_topk_suffix_scan(part_a, lid, lsize);
    nf_topk_suffix_scan(part_e, lid, lsize);
    /* exclusive prefix of range lid = total - suffix(lid) */
    uint wa = part_a[0] - part_a[lid];
    uint we = part_e[0] - part_e[lid];
    for (int c = c0; c < c1 && (na > 0u || we < n_eq); c++) {
        const float x = row[c];
        if (nf_topk_eligible(x, with_neg_inf)) {
            const uint key = nf_topk_key(x);
            if (key > key_k) {
                sel[wa++] = c;
                na--;
            } else if (key == key_k && we < n_eq) {
                sel[n_above + we++] = c;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
