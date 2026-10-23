/* common/nf_base.glsl - library-independent prelude of every kernel-library compute shader: extensions,
 * local-size specialization constants, 1-D group linearization, special values and scalar conversions.
 * Requires: nothing; include it first, right after #version and GL_GOOGLE_include_directive (a library
 * prelude such as common/nf_common.glsl includes it after its own extension lines).
 *
 * Resolution: kernel-library shaders compile with the include directories [shaders/<library>, shaders], so
 * "common/nf_base.glsl" resolves here through shaders/. No library may contain its own common/nf_base.glsl.
 *
 * Specialization constants: ids 0..2 carry the local size set by the backend from the dispatch's local
 * work size.
 *
 * Storage: nothing here needs 8/16-bit storage or 8/16-bit arithmetic types; the helpers below operate on
 * 32-bit values only.
 */
#ifndef NF_BASE_GLSL
#define NF_BASE_GLSL

#extension GL_EXT_control_flow_attributes : enable
#ifdef USE_SUBGROUPS
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif

/* Local size from specialization constants 0..2. Declared here so common code may use
 * gl_WorkGroupSize; a shader may restate the same declaration. */
#define NF_LOCAL_SIZE_LAYOUT layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
NF_LOCAL_SIZE_LAYOUT

/* Total invocations of one workgroup. */
#define NF_WG_INVOCATIONS (gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z)

/* IEEE-754 single-precision special values (no fast-math mode exists in GLSL). */
#define NF_INF uintBitsToFloat(0x7f800000u)
#define NF_NAN uintBitsToFloat(0x7fc00000u)

/* Linear workgroup index of a 1-D dispatch. The backend folds a 1-D group count above
 * maxComputeWorkGroupCount[0] into (x = limit, y = ceil(groups / limit)); every 1-D shader uses
 * nf_wg_id()/nf_global_id() in place of get_group_id(0)/get_global_id(0) and keeps its logical
 * bounds check, so the over-dispatch produced by folding is harmless. 2-D/3-D dispatches are
 * never folded and use gl_WorkGroupID / gl_GlobalInvocationID directly. */
uint nf_wg_id() {
    return gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
}

/* Linear global invocation index of a 1-D dispatch (folding-aware get_global_id(0)). */
uint nf_global_id() {
    return nf_wg_id() * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
}

/* IEEE half bits (low 16 bits of h) to float. */
float nf_half(uint h) {
    return unpackHalf2x16(h & 0xFFFFu).x;
}

/* bfloat16 bits (low 16 bits of b) to float. */
float nf_bf16(uint b) {
    return uintBitsToFloat((b & 0xFFFFu) << 16);
}

/* Sign-extends the low byte of b (int8 two's complement). */
int nf_i8(uint b) {
    return bitfieldExtract(int(b), 0, 8);
}

/* OpenCL isfinite(). */
bool nf_isfinite(float x) {
    return !(isnan(x) || isinf(x));
}

/* C roundf(): nearest integer, halfway cases away from zero. Exact for every float because
 * x - trunc(x) is computed without rounding. */
float nf_roundf(float x) {
    const float t = trunc(x);
    return abs(x - t) >= 0.5 ? t + sign(x) : t;
}

/* ceil(a / b) for b > 0. */
uint nf_ceil_div(uint a, uint b) {
    return (a + b - 1u) / b;
}

#endif /* NF_BASE_GLSL */
