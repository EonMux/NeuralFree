/* common/reduce.glsl - workgroup sum/max reductions over local dimension 0, through subgroup
 * arithmetic when USE_SUBGROUPS is defined (the __sg variant) and a shared-memory tree otherwise,
 * plus segment reductions. GLSL twin of nf_wg_sum/nf_wg_max/nf_seg_sum/nf_seg_max in
 * kernels/llm/common/reduce.h. Requires: common/nf_common.glsl.
 *
 * Preconditions of every nf_wg_* call:
 *   - all invocations of the workgroup call it, in uniform control flow (no divergent return
 *     before it; inactive invocations pass the identity value instead);
 *   - the workgroup is one-dimensional in practice (gl_WorkGroupSize.y == gl_WorkGroupSize.z == 1);
 *   - with USE_SUBGROUPS, subgroups are full (the backend binds __sg modules only when
 *     local_x % maxSubgroupSize == 0 and requests full subgroups).
 * Every invocation receives the same result. A trailing barrier makes nf_red reusable by the
 * next call, so reductions may be issued back to back.
 */
#ifndef NF_REDUCE_GLSL
#define NF_REDUCE_GLSL

#ifndef NF_COMMON_GLSL
#error "common/reduce.glsl requires common/nf_common.glsl"
#endif

/* One slot per invocation (tree path) or per subgroup (subgroup path). */
shared float nf_red[gl_WorkGroupSize.x];

#ifdef USE_SUBGROUPS

float nf_wg_sum(float v) {
    const float s = subgroupAdd(v);
    if (gl_SubgroupInvocationID == 0u) {
        nf_red[gl_SubgroupID] = s;
    }
    barrier();
    float t = 0.0;
    for (uint i = 0u; i < gl_NumSubgroups; ++i) {
        t += nf_red[i];
    }
    barrier();
    return t;
}

float nf_wg_max(float v) {
    const float s = subgroupMax(v);
    if (gl_SubgroupInvocationID == 0u) {
        nf_red[gl_SubgroupID] = s;
    }
    barrier();
    float t = nf_red[0];
    for (uint i = 1u; i < gl_NumSubgroups; ++i) {
        t = max(t, nf_red[i]);
    }
    barrier();
    return t;
}

#else /* shared-memory tree */

/* Smallest power of two >= the local size; the tree folds the upper half onto the lower half
 * with a bounds check, so any local size is reduced exactly. */
uint nf_red_span() {
    const uint n = gl_WorkGroupSize.x;
    return n <= 1u ? 1u : (1u << uint(findMSB(n - 1u) + 1));
}

float nf_wg_sum(float v) {
    const uint lid = gl_LocalInvocationID.x;
    const uint n = gl_WorkGroupSize.x;
    nf_red[lid] = v;
    barrier();
    for (uint s = nf_red_span() >> 1; s > 0u; s >>= 1) {
        if (lid < s && lid + s < n) {
            nf_red[lid] += nf_red[lid + s];
        }
        barrier();
    }
    const float t = nf_red[0];
    barrier();
    return t;
}

float nf_wg_max(float v) {
    const uint lid = gl_LocalInvocationID.x;
    const uint n = gl_WorkGroupSize.x;
    nf_red[lid] = v;
    barrier();
    for (uint s = nf_red_span() >> 1; s > 0u; s >>= 1) {
        if (lid < s && lid + s < n) {
            nf_red[lid] = max(nf_red[lid], nf_red[lid + s]);
        }
        barrier();
    }
    const float t = nf_red[0];
    barrier();
    return t;
}

#endif /* USE_SUBGROUPS */

/* nf_wg_lockstep(count): whether nf_wg_sum2 / nf_wg_sum4 / nf_wg_max4 of count values take their one-barrier-pair
 * form: subgroup arithmetic, with the leader slots of all count values inside nf_red. Otherwise they run count
 * sequential single reductions, so a caller with fewer live values than count reduces only the live ones itself.
 * Uniform across the workgroup. */
bool nf_wg_lockstep(const uint count) {
#ifdef USE_SUBGROUPS
    return count * gl_NumSubgroups <= gl_WorkGroupSize.x;
#else
    return false;
#endif
}

/* nf_wg_sum2 / nf_wg_sum4: two or four independent workgroup sums in lockstep, each component the exact
 * nf_wg_sum of its value. With subgroups every value takes its subgroupAdd, the leaders store all values'
 * slots behind one barrier and each value's leader slots combine in nf_wg_sum's ascending order, so the
 * reductions share one barrier pair instead of paying one each. The leader slots of value k sit at
 * k * gl_NumSubgroups; a workgroup whose leader slots for all values do not fit nf_red (subgroups narrower
 * than the value count) and the shared-memory tree path take the sequential nf_wg_sum calls (nf_wg_lockstep
 * false). GLSL-only: the OpenCL and CUDA reductions keep their own forms. Same calling preconditions as
 * nf_wg_*. */
vec2 nf_wg_sum2(float a, float b) {
#ifdef USE_SUBGROUPS
    if (nf_wg_lockstep(2u)) {
        const float sa = subgroupAdd(a);
        const float sb = subgroupAdd(b);
        const uint n = gl_NumSubgroups;
        if (gl_SubgroupInvocationID == 0u) {
            nf_red[gl_SubgroupID] = sa;
            nf_red[n + gl_SubgroupID] = sb;
        }
        barrier();
        float ta = 0.0;
        float tb = 0.0;
        for (uint i = 0u; i < n; ++i) {
            ta += nf_red[i];
            tb += nf_red[n + i];
        }
        barrier();
        return vec2(ta, tb);
    }
#endif
    const float ta = nf_wg_sum(a);
    const float tb = nf_wg_sum(b);
    return vec2(ta, tb);
}

vec4 nf_wg_sum4(float a, float b, float c, float d) {
#ifdef USE_SUBGROUPS
    if (nf_wg_lockstep(4u)) {
        const float sa = subgroupAdd(a);
        const float sb = subgroupAdd(b);
        const float sc = subgroupAdd(c);
        const float sd = subgroupAdd(d);
        const uint n = gl_NumSubgroups;
        if (gl_SubgroupInvocationID == 0u) {
            nf_red[gl_SubgroupID] = sa;
            nf_red[n + gl_SubgroupID] = sb;
            nf_red[2u * n + gl_SubgroupID] = sc;
            nf_red[3u * n + gl_SubgroupID] = sd;
        }
        barrier();
        float ta = 0.0;
        float tb = 0.0;
        float tc = 0.0;
        float td = 0.0;
        for (uint i = 0u; i < n; ++i) {
            ta += nf_red[i];
            tb += nf_red[n + i];
            tc += nf_red[2u * n + i];
            td += nf_red[3u * n + i];
        }
        barrier();
        return vec4(ta, tb, tc, td);
    }
#endif
    const vec2 ab = nf_wg_sum2(a, b);
    const vec2 cd = nf_wg_sum2(c, d);
    return vec4(ab, cd);
}

/* nf_wg_max4: four independent workgroup maxima in lockstep, each component the exact nf_wg_max of its value (a
 * maximum does not depend on the order of its operands), with the leader-slot layout and fallbacks of nf_wg_sum4.
 * GLSL-only. Same calling preconditions as nf_wg_*. */
vec4 nf_wg_max4(float a, float b, float c, float d) {
#ifdef USE_SUBGROUPS
    if (nf_wg_lockstep(4u)) {
        const float sa = subgroupMax(a);
        const float sb = subgroupMax(b);
        const float sc = subgroupMax(c);
        const float sd = subgroupMax(d);
        const uint n = gl_NumSubgroups;
        if (gl_SubgroupInvocationID == 0u) {
            nf_red[gl_SubgroupID] = sa;
            nf_red[n + gl_SubgroupID] = sb;
            nf_red[2u * n + gl_SubgroupID] = sc;
            nf_red[3u * n + gl_SubgroupID] = sd;
        }
        barrier();
        float ta = nf_red[0];
        float tb = nf_red[n];
        float tc = nf_red[2u * n];
        float td = nf_red[3u * n];
        for (uint i = 1u; i < n; ++i) {
            ta = max(ta, nf_red[i]);
            tb = max(tb, nf_red[n + i]);
            tc = max(tc, nf_red[2u * n + i]);
            td = max(td, nf_red[3u * n + i]);
        }
        barrier();
        return vec4(ta, tb, tc, td);
    }
#endif
    const float ta = nf_wg_max(a);
    const float tb = nf_wg_max(b);
    const float tc = nf_wg_max(c);
    const float td = nf_wg_max(d);
    return vec4(ta, tb, tc, td);
}

/* nf_seg_sum / nf_seg_max: reduce v over aligned segments of `width` consecutive invocations
 * (width a power of two, 1 <= width <= local size, uniform across the workgroup) and return each
 * segment's result to every invocation of that segment. GLSL twin of nf_seg_sum/nf_seg_max in
 * kernels/llm/common/reduce.h in its local-memory form (the __sg variants require only subgroup arithmetic,
 * so no shuffle form exists here). A trailing segment cut short by the local size reduces the
 * invocations it has. Same calling preconditions as nf_wg_*; nf_red is reusable on return. */
float nf_seg_sum(float v, uint width) {
    const uint lid = gl_LocalInvocationID.x;
    const uint n = gl_WorkGroupSize.x;
    const uint seg_lane = lid & (width - 1u);
    nf_red[lid] = v;
    barrier();
    for (uint s = width >> 1; s > 0u; s >>= 1) {
        if (seg_lane < s && lid + s < n) {
            nf_red[lid] += nf_red[lid + s];
        }
        barrier();
    }
    const float t = nf_red[lid - seg_lane];
    barrier();
    return t;
}

float nf_seg_max(float v, uint width) {
    const uint lid = gl_LocalInvocationID.x;
    const uint n = gl_WorkGroupSize.x;
    const uint seg_lane = lid & (width - 1u);
    nf_red[lid] = v;
    barrier();
    for (uint s = width >> 1; s > 0u; s >>= 1) {
        if (seg_lane < s && lid + s < n) {
            nf_red[lid] = max(nf_red[lid], nf_red[lid + s]);
        }
        barrier();
    }
    const float t = nf_red[lid - seg_lane];
    barrier();
    return t;
}

#endif /* NF_REDUCE_GLSL */
