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
