/* common/topk_select.glsl - workgroup top-k selection over one row: radix select of the k-th largest key, then an
 * ordered compaction. GLSL twin of kernels/llm/common/topk_select.h (the semantic reference).
 * Requires: common/nf_common.glsl.
 *
 * The including shader defines, before the include:
 *   NF_TOPK_SRC      the readonly float buffer array the row is read from (element base + c is column c)
 * and either
 *   NF_TOPK_SEL_MAX  the capacity of the shared nf_topk_sel the compaction fills, the widest selection the shader
 *                    makes,
 * or
 *   NF_TOPK_EMIT(slot, col)  a statement storing the compaction's column col (int) at selection slot (uint)
 *                    itself, which leaves nf_topk_sel undeclared (a shader that writes the selection straight to
 *                    a buffer saves the shared array: the module stays inside the 16 KB of shared memory every
 *                    Vulkan device provides).
 *
 * Eligibility: with_neg_inf false selects only scores greater than -INF, true every non-NaN score (-INF
 * included). NaN is never selected.
 *
 * nf_topk_select: three digit passes of 11, 11 and 10 bits of the order-preserving key over a 2048-bin shared
 * histogram give the key T of the kk-th largest eligible score, kk = min(k, eligible count), and how many of the
 * selected scores equal T. nf_topk_compact: nf_topk_sel[0 .. kk) receives the columns above T in increasing column
 * order, then the first n_eq columns equal to T in increasing column order (through NF_TOPK_EMIT where defined);
 * each invocation owns one contiguous column range, placed by one scan over the invocations' counts.
 *
 * Both are called by every invocation in uniform control flow; the local size is a power of two <= LLM_DIMX.
 * Equal floats are equal keys (+0 and -0 included).
 */
#ifndef NF_TOPK_SELECT_GLSL
#define NF_TOPK_SELECT_GLSL

#ifndef NF_COMMON_GLSL
#error "common/topk_select.glsl requires common/nf_common.glsl"
#endif
#if !defined(NF_TOPK_SRC) || (!defined(NF_TOPK_SEL_MAX) && !defined(NF_TOPK_EMIT))
#error "common/topk_select.glsl needs NF_TOPK_SRC and NF_TOPK_SEL_MAX or NF_TOPK_EMIT"
#endif

#define NF_TOPK_BINS 2048

shared uint nf_topk_hist[NF_TOPK_BINS];
shared uint nf_topk_part_a[gl_WorkGroupSize.x];
shared uint nf_topk_part_e[gl_WorkGroupSize.x];
shared uint nf_topk_pick[2];
#ifndef NF_TOPK_EMIT
shared int nf_topk_sel[NF_TOPK_SEL_MAX];
#define NF_TOPK_EMIT(slot, col) nf_topk_sel[slot] = (col)
#endif

/* Order-preserving key: a > b as floats <=> key(a) > key(b) as unsigned; -0 maps to the key of +0. */
uint nf_topk_key(float x) {
    const uint u = (x == 0.0) ? 0u : floatBitsToUint(x);
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

/* Takes part in the selection; the NaN test reads the bit pattern, independent of the driver's NaN handling. */
bool nf_topk_eligible(float x, bool with_neg_inf) {
    return with_neg_inf ? (floatBitsToUint(x) & 0x7fffffffu) <= 0x7f800000u : x > -NF_INF;
}

/* In-place inclusive suffix sums of nf_topk_part_a (which = false) or nf_topk_part_e (true). */
void nf_topk_suffix_scan(bool which) {
    const uint lid = gl_LocalInvocationID.x;
    const uint lsize = gl_WorkGroupSize.x;
    for (uint off = 1u; off < lsize; off <<= 1) {
        uint v;
        if (which) {
            v = nf_topk_part_e[lid] + (lid + off < lsize ? nf_topk_part_e[lid + off] : 0u);
        } else {
            v = nf_topk_part_a[lid] + (lid + off < lsize ? nf_topk_part_a[lid + off] : 0u);
        }
        barrier();
        if (which) {
            nf_topk_part_e[lid] = v;
        } else {
            nf_topk_part_a[lid] = v;
        }
        barrier();
    }
}

/* The key T of the kk-th largest eligible score of row base (key_out), kk (kk_out) and the selected count equal
 * to T (eq_out); kk == 0 leaves T and the count 0. */
void nf_topk_select(uint base, uint n, uint k, bool with_neg_inf, out uint key_out, out uint kk_out,
                    out uint eq_out) {
    const uint lid = gl_LocalInvocationID.x;
    const uint lsize = gl_WorkGroupSize.x;
    const uint per = uint(NF_TOPK_BINS) / lsize;
    uint prefix = 0u;
    uint pmask = 0u;
    uint kk = 0u;
    uint remaining = 0u;
    for (int pass = 0; pass < 3; ++pass) {
        const uint shift = pass == 0 ? 21u : (pass == 1 ? 10u : 0u);
        const uint dmask = pass == 2 ? 0x3FFu : 0x7FFu;
        for (uint b = lid; b < uint(NF_TOPK_BINS); b += lsize) {
            nf_topk_hist[b] = 0u;
        }
        barrier();
        for (uint c = lid; c < n; c += lsize) {
            const float x = NF_TOPK_SRC[base + c];
            if (nf_topk_eligible(x, with_neg_inf)) {
                const uint key = nf_topk_key(x);
                if ((key & pmask) == prefix) {
                    atomicAdd(nf_topk_hist[(key >> shift) & dmask], 1u);
                }
            }
        }
        barrier();
        uint mine = 0u;
        for (uint i = 0u; i < per; ++i) {
            mine += nf_topk_hist[lid * per + i];
        }
        nf_topk_part_a[lid] = mine;
        barrier();
        /* part[i] = candidates in the bins of invocations i .. L-1 */
        nf_topk_suffix_scan(false);
        if (pass == 0) {
            kk = min(k, nf_topk_part_a[0]);
            remaining = kk;
        }
        if (kk == 0u) {
            break;   /* uniform: nothing eligible, or k == 0 */
        }
        /* The kk-th largest key lies in the digit d with #(digits > d) < remaining <= #(digits >= d). */
        uint above = lid + 1u < lsize ? nf_topk_part_a[lid + 1u] : 0u;
        if (above < remaining && remaining <= above + mine) {
            for (int i = int(per) - 1; i >= 0; --i) {
                const uint h = nf_topk_hist[lid * per + uint(i)];
                if (above + h >= remaining) {
                    nf_topk_pick[0] = lid * per + uint(i);
                    nf_topk_pick[1] = remaining - above;
                    break;
                }
                above += h;
            }
        }
        barrier();
        prefix |= nf_topk_pick[0] << shift;
        pmask |= dmask << shift;
        remaining = nf_topk_pick[1];
        barrier();
    }
    key_out = prefix;
    kk_out = kk;
    eq_out = kk == 0u ? 0u : remaining;
}

/* The ordered compaction into the selection slots (see the header). */
void nf_topk_compact(uint base, uint n, bool with_neg_inf, uint key_k, uint n_above, uint n_eq) {
    const uint lid = gl_LocalInvocationID.x;
    const uint lsize = gl_WorkGroupSize.x;
    const uint span = (n + lsize - 1u) / lsize;
    const uint c0 = min(n, lid * span);
    const uint c1 = min(n, c0 + span);
    uint na = 0u;
    uint ne = 0u;
    for (uint c = c0; c < c1; ++c) {
        const float x = NF_TOPK_SRC[base + c];
        if (nf_topk_eligible(x, with_neg_inf)) {
            const uint key = nf_topk_key(x);
            na += key > key_k ? 1u : 0u;
            ne += key == key_k ? 1u : 0u;
        }
    }
    nf_topk_part_a[lid] = na;
    nf_topk_part_e[lid] = ne;
    barrier();
    nf_topk_suffix_scan(false);
    nf_topk_suffix_scan(true);
    /* exclusive prefix of range lid = total - suffix(lid) */
    uint wa = nf_topk_part_a[0] - nf_topk_part_a[lid];
    uint we = nf_topk_part_e[0] - nf_topk_part_e[lid];
    for (uint c = c0; c < c1 && (na > 0u || we < n_eq); ++c) {
        const float x = NF_TOPK_SRC[base + c];
        if (nf_topk_eligible(x, with_neg_inf)) {
            const uint key = nf_topk_key(x);
            if (key > key_k) {
                NF_TOPK_EMIT(wa, int(c));
                wa++;
                na--;
            } else if (key == key_k && we < n_eq) {
                NF_TOPK_EMIT(n_above + we, int(c));
                we++;
            }
        }
    }
    barrier();
}

#endif
