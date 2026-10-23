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
