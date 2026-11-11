/* common/reduce.h - workgroup and segment reductions (local-memory tree, subgroup and shuffle forms). Requires: common/dialect.h, common/config.h */
/*
 * Preconditions shared by every reduction here:
 *   - every work-item of the workgroup calls it, in uniform control flow (they contain barriers);
 *   - 1-D workgroups along dimension 0 (local size {L, 1, 1}), L a power of two <= LLM_DIMX;
 *   - CUDA: L is a multiple of 32 (full warps).
 * The buffer argument is a kernel-scope NF_LOCAL float array of at least L elements.
 */
#ifndef NF_REDUCE_H
#define NF_REDUCE_H

/*
 * REDUCE_MAX / REDUCE_SUM: in-place reduction of local_buf[0..L) into local_buf[0]. The caller writes
 * local_buf[lid] and issues a barrier before the macro; local_buf[0] holds the result afterwards.
 */
#define REDUCE_MAX(local_buf, lid)                                      \
    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) \
    {                                                                    \
        if (lid < stride)                                                \
            local_buf[lid] = fmax(local_buf[lid],                        \
                                   local_buf[lid + stride]);             \
        barrier(CLK_LOCAL_MEM_FENCE);                                    \
    }

#define REDUCE_SUM(local_buf, lid)                                      \
    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) \
    {                                                                    \
        if (lid < stride)                                                \
            local_buf[lid] += local_buf[lid + stride];                   \
        barrier(CLK_LOCAL_MEM_FENCE);                                    \
    }

#ifdef USE_SUBGROUPS

/* Subgroup form: hardware reduction inside each subgroup, then a tree over the subgroup leaders. */
#undef REDUCE_MAX
#define REDUCE_MAX(local_buf, lid)                                      \
    {                                                                    \
        float _sg_val = sub_group_reduce_max(local_buf[lid]);            \
        local_buf[lid] = _sg_val;                                        \
        barrier(CLK_LOCAL_MEM_FENCE);                                    \
        uint _sg_size = get_sub_group_size();                            \
        for (uint _s = get_local_size(0) / (_sg_size * 2);              \
             _s > 0; _s >>= 1)                                          \
        {                                                                \
            if ((uint)(lid) < _s)                                        \
                local_buf[lid * _sg_size] =                              \
                    fmax(local_buf[lid * _sg_size],                      \
                         local_buf[(lid + _s) * _sg_size]);              \
            barrier(CLK_LOCAL_MEM_FENCE);                                \
        }                                                                \
    }

#undef REDUCE_SUM
#define REDUCE_SUM(local_buf, lid)                                      \
    {                                                                    \
        float _sg_val = sub_group_reduce_add(local_buf[lid]);            \
        local_buf[lid] = _sg_val;                                        \
        barrier(CLK_LOCAL_MEM_FENCE);                                    \
        uint _sg_size = get_sub_group_size();                            \
        for (uint _s = get_local_size(0) / (_sg_size * 2);              \
             _s > 0; _s >>= 1)                                          \
        {                                                                \
            if ((uint)(lid) < _s)                                        \
                local_buf[lid * _sg_size] +=                             \
                    local_buf[(lid + _s) * _sg_size];                    \
            barrier(CLK_LOCAL_MEM_FENCE);                                \
        }                                                                \
    }

#endif /* USE_SUBGROUPS */

/*
 * NF_WG_RSUM(ts, ps, NV, lid, lsize): the workgroup sums of NV register values ps[0 .. NV) (NV a
 * compile-time constant, ps a float array), reduced in lockstep; work-item 0 holds every total in ps
 * afterwards, the other work-items' ps are left partial. ts is a kernel-scope NF_LOCAL float array of
 * at least NV * NF_RSUM_SLOTS elements, free for reuse only after the caller's next barrier.
 *
 * Each value takes exactly REDUCE_SUM's arithmetic over a buffer holding the work-items' values: the
 * same subgroup reduction of the same input, and the same stride tree over the subgroup leaders, whose
 * slots are indexed by subgroup id (REDUCE_SUM's lid * subgroup-size slots in compact order), or over
 * the lanes without subgroups. So the totals are bit-identical to writing each value, a barrier and
 * REDUCE_SUM, while a call stages one leader store per value and subgroup and passes one barrier per
 * tree level for all NV values: the barrier the caller's store needed and the one that retired each
 * value's read-back go, and the last level has none, as only work-item 0 reads what it wrote.
 * Requires the subgroup size to divide lsize (the REDUCE_SUM preconditions).
 */
#if defined(USE_SUBGROUPS) && defined(NF_CUDA)
#define NF_RSUM_SLOTS (LLM_DIMX / 32)
#define NF_RSUM_SG_SIZE 32u
#elif defined(USE_SUBGROUPS)
#define NF_RSUM_SLOTS LLM_DIMX
#define NF_RSUM_SG_SIZE get_sub_group_size()
#endif

#ifdef USE_SUBGROUPS
#define NF_WG_RSUM(ts, ps, NV, lid, lsize) \
    { \
        const uint nf_rs_nsg = (uint)(lsize) / NF_RSUM_SG_SIZE; \
        _Pragma("unroll") \
        for (int nf_rs_j = 0; nf_rs_j < (NV); nf_rs_j++) \
            ps[nf_rs_j] = sub_group_reduce_add(ps[nf_rs_j]); \
        if (nf_rs_nsg > 1u) { \
            if (get_sub_group_local_id() == 0) { \
                _Pragma("unroll") \
                for (int nf_rs_j = 0; nf_rs_j < (NV); nf_rs_j++) \
                    ts[nf_rs_j * NF_RSUM_SLOTS + get_sub_group_id()] = ps[nf_rs_j]; \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
            for (uint nf_rs_s = nf_rs_nsg >> 1; nf_rs_s > 0u; nf_rs_s >>= 1) { \
                if ((uint)(lid) < nf_rs_s) { \
                    _Pragma("unroll") \
                    for (int nf_rs_j = 0; nf_rs_j < (NV); nf_rs_j++) \
                        ts[nf_rs_j * NF_RSUM_SLOTS + (lid)] += ts[nf_rs_j * NF_RSUM_SLOTS + (lid) + nf_rs_s]; \
                } \
                if (nf_rs_s > 1u) \
                    barrier(CLK_LOCAL_MEM_FENCE); \
            } \
            if ((lid) == 0) { \
                _Pragma("unroll") \
                for (int nf_rs_j = 0; nf_rs_j < (NV); nf_rs_j++) \
                    ps[nf_rs_j] = ts[nf_rs_j * NF_RSUM_SLOTS]; \
            } \
        } \
    }
#else
#define NF_RSUM_SLOTS LLM_DIMX
#define NF_WG_RSUM(ts, ps, NV, lid, lsize) \
    { \
        _Pragma("unroll") \
        for (int nf_rs_j = 0; nf_rs_j < (NV); nf_rs_j++) \
            ts[nf_rs_j * NF_RSUM_SLOTS + (lid)] = ps[nf_rs_j]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        for (int nf_rs_s = (lsize) >> 1; nf_rs_s > 0; nf_rs_s >>= 1) { \
            if ((lid) < nf_rs_s) { \
                _Pragma("unroll") \
                for (int nf_rs_j = 0; nf_rs_j < (NV); nf_rs_j++) \
                    ts[nf_rs_j * NF_RSUM_SLOTS + (lid)] += ts[nf_rs_j * NF_RSUM_SLOTS + (lid) + nf_rs_s]; \
            } \
            if (nf_rs_s > 1) \
                barrier(CLK_LOCAL_MEM_FENCE); \
        } \
        if ((lid) == 0) { \
            _Pragma("unroll") \
            for (int nf_rs_j = 0; nf_rs_j < (NV); nf_rs_j++) \
                ps[nf_rs_j] = ts[nf_rs_j * NF_RSUM_SLOTS]; \
        } \
    }
#endif

/*
 * nf_wg_sum / nf_wg_max: reduce v over the whole workgroup and return the result to every work-item.
 * buf is free for reuse when the function returns (it ends with a barrier after the final read).
 */
NF_INLINE float nf_wg_sum(NF_LOCAL_PTR float* buf, float v) {
#ifdef USE_SUBGROUPS
    const float s = sub_group_reduce_add(v);
    if (get_sub_group_local_id() == 0) {
        buf[get_sub_group_id()] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint n_sg = get_num_sub_groups();
    float r = 0.0f;
    for (uint i = 0; i < n_sg; i++) {
        r += buf[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
#else
    const int lid = get_local_id(0);
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            buf[lid] += buf[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
#endif
}

NF_INLINE float nf_wg_max(NF_LOCAL_PTR float* buf, float v) {
#ifdef USE_SUBGROUPS
    const float m = sub_group_reduce_max(v);
    if (get_sub_group_local_id() == 0) {
        buf[get_sub_group_id()] = m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint n_sg = get_num_sub_groups();
    float r = buf[0];
    for (uint i = 1; i < n_sg; i++) {
        r = fmax(r, buf[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
#else
    const int lid = get_local_id(0);
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            buf[lid] = fmax(buf[lid], buf[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
#endif
}

/*
 * nf_seg_sum / nf_seg_max: reduce v over aligned segments of `width` consecutive work-items
 * (width a power of two, 1 <= width <= L, uniform across the workgroup) and return each segment's
 * result to every work-item of that segment. With NF_HAS_SG_SHUFFLE and width <= the subgroup size
 * this is a register butterfly that leaves buf untouched; otherwise it is a local-memory tree that
 * uses buf[0..L) and ends with a barrier.
 */
NF_INLINE float nf_seg_sum(NF_LOCAL_PTR float* buf, float v, int width) {
#ifdef NF_HAS_SG_SHUFFLE
    if (width <= (int)get_sub_group_size()) {
        for (int o = width >> 1; o > 0; o >>= 1) {
            v += nf_sg_shuffle_xor(v, o);
        }
        return v;
    }
#endif
    const int lid = get_local_id(0);
    const int seg_lane = lid & (width - 1);
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        if (seg_lane < stride) {
            buf[lid] += buf[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = buf[lid - seg_lane];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

NF_INLINE float nf_seg_max(NF_LOCAL_PTR float* buf, float v, int width) {
#ifdef NF_HAS_SG_SHUFFLE
    if (width <= (int)get_sub_group_size()) {
        for (int o = width >> 1; o > 0; o >>= 1) {
            v = fmax(v, nf_sg_shuffle_xor(v, o));
        }
        return v;
    }
#endif
    const int lid = get_local_id(0);
    const int seg_lane = lid & (width - 1);
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
        if (seg_lane < stride) {
            buf[lid] = fmax(buf[lid], buf[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = buf[lid - seg_lane];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

#endif /* NF_REDUCE_H */
