/* common/flash_sg.glsl - shared pieces of the 16-lane subgroup flash attention (flash_decode_sg.comp,
 * flash_prefill_sg.comp), the GLSL form of the NF_FDS_* helpers of kernels/llm/flash_decode_sg.cl.
 * Requires: common/nf_common.glsl, common/kv_types.glsl, common/attn_common.glsl; KV is NF_KV_F32, NF_KV_F16 or
 * NF_KV_BF16 and the module is built by an sg16 variants row (USE_SUBGROUPS, NF_SG_ROW_SIZE == 16).
 *
 * Lane l of a subgroup holds the NF_FDS_DPL consecutive head dims [l * NF_FDS_DPL, +NF_FDS_DPL), so one K or V row
 * is one 16-byte (2-byte types) or two 16-byte (F32) loads per lane. Every such load is 16-byte aligned: the backend
 * binds each buffer from offset 0 of its own VkBuffer, whose memory starts on a 256-byte boundary (VkGpuBuffer,
 * VkMemoryAllocator.MIN_ALIGNMENT), and every element offset the forms read is a row times a row stride plus a head
 * times a head stride plus a lane's first dim, all multiples of NF_FDS_DPL (nf_fds_strides_ok). The only subgroup
 * operations are basic, ballot (subgroupBroadcast with a dynamically uniform lane, SPIR-V 1.5) and arithmetic ones.
 */
#ifndef NF_FLASH_SG_GLSL
#define NF_FLASH_SG_GLSL

#extension GL_KHR_shader_subgroup_ballot : require

#if !defined(USE_SUBGROUPS) || !defined(NF_SG_ROW_SIZE) || NF_SG_ROW_SIZE != 16
#error "common/flash_sg.glsl is built only by an sg16 variants row"
#endif
#if !defined(KV) || (KV != NF_KV_F32 && KV != NF_KV_F16 && KV != NF_KV_BF16)
#error "common/flash_sg.glsl serves KV=NF_KV_F32, NF_KV_F16 and NF_KV_BF16 only"
#endif

/* Subgroups per workgroup and dims per lane (SubgroupFlashAttention mirrors both). */
#define NF_FDS_SUBGROUPS 4
#define NF_FDS_DPL 8
#define NF_FDS_WG (NF_FDS_SUBGROUPS * NF_SG_ROW_SIZE)
#define NF_FDS_MAX_D (NF_FDS_DPL * NF_SG_ROW_SIZE)

/*
 * Declares the 16-byte view of a K or V binding and name_load8(e, lo, hi): the NF_FDS_DPL values at element e of the
 * binding (e a multiple of NF_FDS_DPL, which the plan's stride checks guarantee).
 */
#if KV == NF_KV_F32
#define NF_FDS_KV_VIEW(binding_index, name) \
    layout(std430, binding = binding_index) readonly buffer NfFdsKv_##name { vec4 name##_v4[]; }; \
    void name##_load8(uint e, out vec4 lo, out vec4 hi) { \
        lo = name##_v4[e >> 2]; \
        hi = name##_v4[(e >> 2) + 1u]; \
    }
#elif KV == NF_KV_F16
#define NF_FDS_KV_VIEW(binding_index, name) \
    layout(std430, binding = binding_index) readonly buffer NfFdsKv_##name { uvec4 name##_u4[]; }; \
    void name##_load8(uint e, out vec4 lo, out vec4 hi) { \
        const uvec4 w = name##_u4[e >> 3]; \
        lo = vec4(unpackHalf2x16(w.x), unpackHalf2x16(w.y)); \
        hi = vec4(unpackHalf2x16(w.z), unpackHalf2x16(w.w)); \
    }
#else
#define NF_FDS_KV_VIEW(binding_index, name) \
    layout(std430, binding = binding_index) readonly buffer NfFdsKv_##name { uvec4 name##_u4[]; }; \
    void name##_load8(uint e, out vec4 lo, out vec4 hi) { \
        const uvec4 w = name##_u4[e >> 3]; \
        lo = uintBitsToFloat(uvec4(w.x << 16, w.x & 0xFFFF0000u, w.y << 16, w.y & 0xFFFF0000u)); \
        hi = uintBitsToFloat(uvec4(w.z << 16, w.z & 0xFFFF0000u, w.w << 16, w.w & 0xFFFF0000u)); \
    }
#endif

/* The stride alignment of the 8-value loads: every row and head offset a multiple of NF_FDS_DPL elements. */
bool nf_fds_strides_ok(int k_row_stride, int k_head_stride, int v_row_stride, int v_head_stride) {
    return k_row_stride % NF_FDS_DPL == 0 && k_head_stride % NF_FDS_DPL == 0
        && v_row_stride % NF_FDS_DPL == 0 && v_head_stride % NF_FDS_DPL == 0;
}

/* The workgroup and subgroup shape every form is written for. */
bool nf_fds_shape_ok() {
    return gl_WorkGroupSize.x == uint(NF_FDS_WG) && gl_WorkGroupSize.y == 1u && gl_SubgroupSize == uint(NF_SG_ROW_SIZE);
}

/* The 8-value dot product, in the order of the OpenCL twin (dot of the low halves plus dot of the high halves). */
float nf_fds_dot(vec4 alo, vec4 ahi, vec4 blo, vec4 bhi) {
    return dot(alo, blo) + dot(ahi, bhi);
}

#endif /* NF_FLASH_SG_GLSL */
