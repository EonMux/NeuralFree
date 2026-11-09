/* common/matvec_n.h - the one-pass multi-token work-group reduction of the {T}_MatVecN kernels
 * (float and Q8_1 activation forms). Requires: common/dialect.h, common/config.h, common/reduce.h */
/*
 * A {T}_MatVecN kernel accumulates one register partial per live token while its block loop streams
 * the weight row once, then reduces the whole token group with NF_MVQTN_TREE instead of one
 * nf_wg_sum per token: a per-token nf_wg_sum pays its barrier chain for every token of every weight
 * row, which from a few tokens up costs more than the block loop it follows. Reducing all tokens'
 * slots in lockstep divides the barrier count by the token count.
 *
 * Bit-exactness contract: every reduction step of NF_MVQTN_TREE is the exact nf_wg_sum step the
 * single-row kernels apply to one row (the same subgroup arithmetic and the same slot combine in
 * the same order under USE_SUBGROUPS, the same staged stride tree otherwise), so token j's result
 * is the single-row reduction of token j's partials and does not depend on n_tokens. Under
 * USE_SUBGROUPS the two kernels of a pair must also reduce over the same subgroup size, which the
 * NF_REQD_SG_SIZE pin of the OpenCL host (common/dialect.h) holds on compilers with a required-size
 * extension; CUDA and HIP fix their subgroup size in hardware. The j < n_tokens guards are uniform
 * (n_tokens is a kernel scalar), so every barrier is reached by all work-items of a live workgroup.
 */
#ifndef NF_MATVEC_N_H
#define NF_MATVEC_N_H

/* Staged slots per token: the stride tree uses one slot per work-item and the subgroup form one
 * slot per subgroup, both bounded by the work-group size, itself bounded by LLM_DIMX. */
#if defined(USE_SUBGROUPS) && defined(NF_CUDA)
/* The CUDA and HIP subgroups are 32 lanes (common/dialect.h), so the subgroup form stages at most
 * LLM_DIMX / 32 leader slots per token; the smaller stride keeps the declared array off the device's
 * shared-memory residency limit (at the 64-thread matvec work-groups of CUDA, LLM_DIMX slots per
 * token declare 16 KiB and cap the resident work-groups far below the thread limit). */
#define NF_MVQTN_SLOTS (LLM_DIMX / 32)
#else
#define NF_MVQTN_SLOTS LLM_DIMX
#endif

#ifdef USE_SUBGROUPS

/* Subgroup form: hardware reduction inside each subgroup per token, one leader slot per token,
 * then the linear slot combine of nf_wg_sum applied to every token (the redundant per-work-item
 * sum keeps the single-row order; lane 0 stores). */
#define NF_MVQTN_TREE(ts, ps, n_tokens, out, out_dim, row)              \
    {                                                                   \
        const uint nf_nsg = get_num_sub_groups();                       \
        _Pragma("unroll")                                               \
        for (int j = 0; j < NF_MVQN_MAX; j++)                           \
            if (j < (n_tokens)) ps[j] = sub_group_reduce_add(ps[j]);    \
        if (get_sub_group_local_id() == 0) {                            \
            _Pragma("unroll")                                           \
            for (int j = 0; j < NF_MVQN_MAX; j++)                       \
                if (j < (n_tokens)) ts[j * NF_MVQTN_SLOTS + get_sub_group_id()] = ps[j]; \
        }                                                               \
        barrier(CLK_LOCAL_MEM_FENCE);                                   \
        _Pragma("unroll")                                               \
        for (int j = 0; j < NF_MVQN_MAX; j++) {                         \
            if (j < (n_tokens)) {                                       \
                float nf_r = 0.0f;                                      \
                for (uint nf_i = 0; nf_i < nf_nsg; nf_i++) {            \
                    nf_r += ts[j * NF_MVQTN_SLOTS + nf_i];              \
                }                                                       \
                if (get_local_id(0) == 0)                               \
                    out[(nf_i64)(j) * (out_dim) + (row)] = nf_r;        \
            }                                                           \
        }                                                               \
    }

#else

/* Local-memory form: stage every work-item's partial of every live token and tree over the slots,
 * the exact nf_wg_sum stride sequence applied to every token in lockstep. */
#define NF_MVQTN_TREE(ts, ps, n_tokens, out, out_dim, row)              \
    {                                                                   \
        const int nf_lid = get_local_id(0);                             \
        const int nf_lsize = get_local_size(0);                         \
        _Pragma("unroll")                                               \
        for (int j = 0; j < NF_MVQN_MAX; j++)                           \
            if (j < (n_tokens)) ts[j * NF_MVQTN_SLOTS + nf_lid] = ps[j]; \
        barrier(CLK_LOCAL_MEM_FENCE);                                   \
        for (int nf_stride = nf_lsize >> 1; nf_stride > 0; nf_stride >>= 1) { \
            if (nf_lid < nf_stride) {                                   \
                _Pragma("unroll")                                       \
                for (int j = 0; j < NF_MVQN_MAX; j++)                   \
                    if (j < (n_tokens)) ts[j * NF_MVQTN_SLOTS + nf_lid] += ts[j * NF_MVQTN_SLOTS + nf_lid + nf_stride]; \
            }                                                           \
            barrier(CLK_LOCAL_MEM_FENCE);                               \
        }                                                               \
        if (nf_lid == 0) {                                              \
            _Pragma("unroll")                                           \
            for (int j = 0; j < NF_MVQN_MAX; j++)                       \
                if (j < (n_tokens)) out[(nf_i64)(j) * (out_dim) + (row)] = ts[j * NF_MVQTN_SLOTS]; \
        }                                                               \
    }

#endif /* USE_SUBGROUPS */

#endif /* NF_MATVEC_N_H */
