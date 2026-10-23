/* common/nf_common.glsl - shared prelude of every LLM compute shader: the 8/16-bit storage extensions, the
 * library-independent prelude common/nf_base.glsl (extensions, local-size specialization constants, 1-D group
 * linearization, special values, scalar conversions), the shared-memory specialization constant, the
 * kernels/llm/common/config.h sizes, weight-family op ids and the aliased SSBO view macros.
 * Requires: nothing; include it first, right after #version and GL_GOOGLE_include_directive.
 *
 * Specialization constants: ids 0..2 carry the local size set by the backend from
 * the dispatch's local work size (common/nf_base.glsl); id 3 carries NF_FA_SMEM_FLOATS. No other ids are used.
 *
 * Storage: 8/16-bit values are only loaded from or stored to storage buffers, converting to or from
 * 32 bits (StorageBuffer8BitAccess / StorageBuffer16BitAccess). No 8/16-bit arithmetic types are enabled,
 * so the Int8/Int16/Float16 device features are never required.
 */
#ifndef NF_COMMON_GLSL
#define NF_COMMON_GLSL

#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#include "common/nf_base.glsl"

/* Floats of static shared memory available to flash-attention style kernels (FaSmem on the host). */
layout(constant_id = 3) const int NF_FA_SMEM_FLOATS = 3072;

/* Model-independent sizes, equal to kernels/llm/common/config.h (Java: LlmKernelConstants). */
#ifndef LLM_DIMX
#define LLM_DIMX      256   /* max local size of every row-reduce kernel; power of 2 */
#endif
#define NF_FA_WG      128   /* flash kernels' workgroup size */
#define NF_FA_ACC_MAX 32    /* per-invocation output accumulators in flash kernels */
#define NF_FD_MAX_G   16    /* max q-heads per flash-decode workgroup */

/* Op ids of the weight-type family templates (matvec_quant.comp, matvec_dense.comp, embed.comp),
 * selected by the OP define of a variants row. Defined here because a template's binding layout
 * depends on OP and is declared before common/quant_blocks.glsl. */
#define OP_MATVEC       1
#define OP_MATVEC_ADD   2
#define OP_GATEUP_SILU  3
#define OP_GATEUP_GELU  4
#define OP_LOGITS       5
#define OP_BATCH        6
#define OP_BATCH_2D     7
#define OP_EMBED        8
#define OP_BATCH_EMBED  9

/* Declares three read-only aliased views of one storage-buffer binding, for byte-addressed data
 * such as quantized weights: name_u8[] (uint8_t), name_u16[] (uint16_t) and name_u32[] (uint).
 * All views share the binding, so the binding counts once in the kernel signature and is
 * reflected as read-only. Index name_u16 with byteOffset >> 1 (even offsets) and name_u32 with
 * byteOffset >> 2 (offsets divisible by 4). */
#define NF_BUFFER_VIEWS(binding_index, name) \
    layout(std430, binding = binding_index) readonly buffer NfView8_##name  { uint8_t  name##_u8[];  }; \
    layout(std430, binding = binding_index) readonly buffer NfView16_##name { uint16_t name##_u16[]; }; \
    layout(std430, binding = binding_index) readonly buffer NfView32_##name { uint     name##_u32[]; };

/* Writable form of NF_BUFFER_VIEWS for byte-addressed outputs such as quantized KV rows or Q8_0
 * activation blocks; the binding is reflected as read-write. Stores narrow explicitly, e.g.
 * name_u8[off] = uint8_t(v & 0xFFu) and name_u16[off >> 1] = uint16_t(nf_rope_f16_bits(f)) (common/rope_common.glsl,
 * rounded to nearest even; packHalf2x16 rounding is implementation-defined), which needs only the 8/16-bit storage
 * capabilities. */
#define NF_BUFFER_VIEWS_RW(binding_index, name) \
    layout(std430, binding = binding_index) buffer NfView8_##name  { uint8_t  name##_u8[];  }; \
    layout(std430, binding = binding_index) buffer NfView16_##name { uint16_t name##_u16[]; }; \
    layout(std430, binding = binding_index) buffer NfView32_##name { uint     name##_u32[]; };

#endif /* NF_COMMON_GLSL */
