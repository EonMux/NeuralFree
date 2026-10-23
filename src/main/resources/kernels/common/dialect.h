/* common/dialect.h - dual-dialect layer: one kernel text compiles as OpenCL C 1.2 and as CUDA C++. Requires: none */
/*
 * Kernel sources use only the NF_* keywords below plus OpenCL builtin names; the CUDA branch supplies
 * every OpenCL builtin the LLM kernels use.
 *
 * Keywords:
 *   NF_KERNEL      kernel entry point (OpenCL __kernel, CUDA extern "C" __global__)
 *   NF_GLOBAL      global address space qualifier of pointers
 *   NF_LOCAL       kernel-scope workgroup-shared variable or array declaration
 *   NF_LOCAL_PTR   address space of a helper-function parameter that points to NF_LOCAL storage
 *                  (CUDA forbids __shared__ on parameters, so the two need distinct keywords)
 *   NF_INLINE      helper function
 *   NF_RESTRICT    pointer that Java never aliases with another argument
 *   NF_TABLE(t, n, c)  file-scope read-only table; append " = { ... };"
 *   nf_i8 / nf_i64 / nf_u64   signed 8-bit, signed 64-bit, unsigned 64-bit integers on every host ABI
 *   NF_HALF        16-bit float storage type; only vload_half* / vstore_half* touch it
 *   NF_INF / NF_NAN, NF_U64C(x), NF_F4(a,b,c,d), NF_F2(a,b), NF_F4S(s)
 *
 * Capability switches: USE_SUBGROUPS (subgroup reductions; always on in CUDA) and NF_HAS_SG_SHUFFLE
 * (subgroup shuffle; CUDA only). Kernels must provide a local-memory path when NF_HAS_SG_SHUFFLE is
 * undefined.
 */
#ifndef NF_DIALECT_H
#define NF_DIALECT_H

#if defined(__CUDACC__)
/* =============================== CUDA ================================= */
#define NF_CUDA 1
#include <cuda_fp16.h>
#include <stdint.h>

typedef unsigned char      uchar;
typedef unsigned short     ushort;
typedef unsigned int       uint;
typedef signed char        nf_i8;      /* OpenCL char is always signed; CUDA char is not on aarch64 */
typedef long long          nf_i64;     /* OpenCL long is 64-bit; CUDA long is 32-bit on Windows */
typedef unsigned long long nf_u64;
typedef unsigned short     NF_HALF;    /* storage-only half */

#define NF_KERNEL      extern "C" __global__
#define NF_GLOBAL
#define NF_LOCAL       __shared__
#define NF_LOCAL_PTR
/* Each CUDA translation unit holds every common header but uses only part of it, so helpers and
 * tables are marked unused: nvcc then reports unreferenced kernel locals only (warning 177). */
#define NF_INLINE      static __device__ __forceinline__ __attribute__((unused))
#define NF_RESTRICT    __restrict__
/* Global memory with L1 caching, as llama.cpp GGML_TABLE_BEGIN for CUDA (ggml-common.h:493);
 * not __constant__, because divergent indexing would serialize the constant cache. */
#define NF_TABLE(type, name, n) static __device__ const type name[n] __attribute__((unused))
#define NF_INF         __int_as_float(0x7f800000)
#define NF_NAN         __int_as_float(0x7fc00000)
#define NF_U64C(x)     x##ULL

#define CLK_LOCAL_MEM_FENCE  1
#define CLK_GLOBAL_MEM_FENCE 2
#define barrier(flags) __syncthreads()
#define mem_fence(flags)       __threadfence_block()
#define read_mem_fence(flags)  __threadfence_block()
#define write_mem_fence(flags) __threadfence_block()

/* Subgroup == warp (32 lanes). Callers guarantee full warps: local size is a multiple of 32. */
#ifndef USE_SUBGROUPS
#define USE_SUBGROUPS 1
#endif
#define NF_HAS_SG_SHUFFLE 1

/* ---- work-item functions (the engine never uses a global offset) ---- */
NF_INLINE uint get_local_id(uint d)   { return d == 0 ? threadIdx.x : (d == 1 ? threadIdx.y : threadIdx.z); }
NF_INLINE uint get_group_id(uint d)   { return d == 0 ? blockIdx.x  : (d == 1 ? blockIdx.y  : blockIdx.z); }
NF_INLINE uint get_local_size(uint d) { return d == 0 ? blockDim.x  : (d == 1 ? blockDim.y  : blockDim.z); }
NF_INLINE uint get_num_groups(uint d) { return d == 0 ? gridDim.x   : (d == 1 ? gridDim.y   : gridDim.z); }
NF_INLINE size_t get_global_id(uint d)   { return (size_t)get_group_id(d) * get_local_size(d) + get_local_id(d); }
NF_INLINE size_t get_global_size(uint d) { return (size_t)get_num_groups(d) * get_local_size(d); }

/* ---- subgroups ---- */
NF_INLINE uint get_sub_group_size(void)     { return 32u; }
NF_INLINE uint get_sub_group_local_id(void) { return (threadIdx.x + threadIdx.y * blockDim.x) & 31u; }
NF_INLINE uint get_sub_group_id(void)       { return (threadIdx.x + threadIdx.y * blockDim.x) >> 5; }
NF_INLINE uint get_num_sub_groups(void)     { return (blockDim.x * blockDim.y * blockDim.z + 31u) >> 5; }
NF_INLINE float sub_group_reduce_add(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
NF_INLINE float sub_group_reduce_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
NF_INLINE int sub_group_reduce_add(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
NF_INLINE float nf_sg_shuffle_xor(float v, int mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }
NF_INLINE int   nf_sg_shuffle_xor_i(int v, int mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }

/* ---- vector literals ---- */
#define NF_F4(a, b, c, d) make_float4((a), (b), (c), (d))
#define NF_F2(a, b)       make_float2((a), (b))
#define NF_F4S(s)         make_float4((s), (s), (s), (s))

/* ---- float4 / float2 arithmetic (built into OpenCL) ---- */
NF_INLINE float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
NF_INLINE float4 operator-(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
NF_INLINE float4 operator*(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
NF_INLINE float4 operator/(float4 a, float4 b) { return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
NF_INLINE float4 operator+(float4 a, float s)  { return make_float4(a.x + s, a.y + s, a.z + s, a.w + s); }
NF_INLINE float4 operator-(float4 a, float s)  { return make_float4(a.x - s, a.y - s, a.z - s, a.w - s); }
NF_INLINE float4 operator*(float4 a, float s)  { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
NF_INLINE float4 operator/(float4 a, float s)  { return make_float4(a.x / s, a.y / s, a.z / s, a.w / s); }
NF_INLINE float4 operator*(float s, float4 a)  { return a * s; }
NF_INLINE float4 operator+(float s, float4 a)  { return a + s; }
NF_INLINE float4 operator-(float s, float4 a)  { return make_float4(s - a.x, s - a.y, s - a.z, s - a.w); }
NF_INLINE float4 operator-(float4 a)           { return make_float4(-a.x, -a.y, -a.z, -a.w); }
NF_INLINE float4 &operator+=(float4 &a, float4 b) { a = a + b; return a; }
NF_INLINE float4 &operator-=(float4 &a, float4 b) { a = a - b; return a; }
NF_INLINE float4 &operator*=(float4 &a, float4 b) { a = a * b; return a; }
NF_INLINE float4 &operator*=(float4 &a, float s)  { a = a * s; return a; }
NF_INLINE float4 &operator/=(float4 &a, float4 b) { a = a / b; return a; }
NF_INLINE float4 &operator+=(float4 &a, float s)  { a = a + s; return a; }
NF_INLINE float4 &operator-=(float4 &a, float s)  { a = a - s; return a; }
NF_INLINE float4 &operator/=(float4 &a, float s)  { a = a / s; return a; }
NF_INLINE float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
NF_INLINE float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
NF_INLINE float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
NF_INLINE float2 operator/(float2 a, float2 b) { return make_float2(a.x / b.x, a.y / b.y); }
NF_INLINE float2 operator+(float2 a, float s)  { return make_float2(a.x + s, a.y + s); }
NF_INLINE float2 operator-(float2 a, float s)  { return make_float2(a.x - s, a.y - s); }
NF_INLINE float2 operator*(float2 a, float s)  { return make_float2(a.x * s, a.y * s); }
NF_INLINE float2 operator/(float2 a, float s)  { return make_float2(a.x / s, a.y / s); }
NF_INLINE float2 operator*(float s, float2 a)  { return a * s; }
NF_INLINE float2 operator-(float2 a)           { return make_float2(-a.x, -a.y); }
NF_INLINE float2 &operator+=(float2 &a, float2 b) { a = a + b; return a; }
NF_INLINE float2 &operator-=(float2 &a, float2 b) { a = a - b; return a; }
NF_INLINE float2 &operator*=(float2 &a, float2 b) { a = a * b; return a; }
NF_INLINE float2 &operator*=(float2 &a, float s)  { a = a * s; return a; }

/* ---- geometric / common / integer builtins missing from CUDA ---- */
NF_INLINE float  dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
NF_INLINE float  dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
NF_INLINE float4 fabs(float4 a) { return make_float4(fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w)); }
NF_INLINE float4 fmax(float4 a, float4 b) { return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w)); }
NF_INLINE float4 fmin(float4 a, float4 b) { return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w)); }
NF_INLINE float  sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : (x == 0.0f ? copysignf(0.0f, x) : 0.0f)); }
NF_INLINE float  mix(float a, float b, float t) { return a + (b - a) * t; }
NF_INLINE float  select(float a, float b, int c) { return c ? b : a; }
NF_INLINE int    select(int a, int b, int c)     { return c ? b : a; }
NF_INLINE uint   select(uint a, uint b, int c)   { return c ? b : a; }
NF_INLINE float  clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
NF_INLINE int    clamp(int x, int lo, int hi)       { return min(max(x, lo), hi); }
NF_INLINE int4   clamp(int4 v, int lo, int hi)      { return make_int4(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi), clamp(v.w, lo, hi)); }
NF_INLINE float  mad(float a, float b, float c)     { return fmaf(a, b, c); }
NF_INLINE int    mad24(int a, int b, int c)         { return a * b + c; }
NF_INLINE int    mul24(int a, int b)                { return a * b; }
NF_INLINE uint   popcount(uint x)                   { return (uint)__popc(x); }
NF_INLINE uint   clz(uint x)                        { return (uint)__clz((int)x); }
#define rsqrt(x) rsqrtf(x)

/* ---- native_* (fast, reduced precision; used only where accuracy was considered) ---- */
NF_INLINE float  native_exp(float x)  { return __expf(x); }
NF_INLINE float4 native_exp(float4 x) { return make_float4(__expf(x.x), __expf(x.y), __expf(x.z), __expf(x.w)); }
NF_INLINE float  native_exp2(float x) { return exp2f(x); }
NF_INLINE float  native_log(float x)  { return __logf(x); }
NF_INLINE float  native_log2(float x) { return __log2f(x); }
NF_INLINE float  native_sin(float x)  { return __sinf(x); }
NF_INLINE float  native_cos(float x)  { return __cosf(x); }
NF_INLINE float  native_sqrt(float x) { return sqrtf(x); }
NF_INLINE float  native_powr(float x, float y) { return __powf(x, y); }
NF_INLINE float  native_recip(float x) { return __frcp_rn(x); }
NF_INLINE float  native_rsqrt(float x) { return rsqrtf(x); }
NF_INLINE float  native_divide(float a, float b) { return __fdividef(a, b); }

/* ---- reinterpretation and conversion ---- */
NF_INLINE float  as_float(uint u) { return __uint_as_float(u); }
NF_INLINE float  as_float(int i)  { return __int_as_float(i); }
NF_INLINE uint   as_uint(float f) { return __float_as_uint(f); }
NF_INLINE int    as_int(float f)  { return __float_as_int(f); }
NF_INLINE uchar4 as_uchar4(char4 v) { return make_uchar4((uchar)v.x, (uchar)v.y, (uchar)v.z, (uchar)v.w); }
NF_INLINE float4 convert_float4(char4 v)  { return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w); }
NF_INLINE float4 convert_float4(uchar4 v) { return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w); }
NF_INLINE float4 convert_float4(int4 v)   { return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w); }
NF_INLINE int4   convert_int4_rte(float4 v) { return make_int4(__float2int_rn(v.x), __float2int_rn(v.y), __float2int_rn(v.z), __float2int_rn(v.w)); }
NF_INLINE char4  convert_char4(int4 v)    { return make_char4((signed char)v.x, (signed char)v.y, (signed char)v.z, (signed char)v.w); }

/* ---- vloadN / vstoreN: element alignment only (per-element access, as OpenCL requires) ---- */
NF_INLINE float4  vload4(size_t i, const float *p)  { p += 4 * i; return make_float4(p[0], p[1], p[2], p[3]); }
NF_INLINE char4   vload4(size_t i, const nf_i8 *p)  { p += 4 * i; return make_char4(p[0], p[1], p[2], p[3]); }
NF_INLINE uchar4  vload4(size_t i, const uchar *p)  { p += 4 * i; return make_uchar4(p[0], p[1], p[2], p[3]); }
NF_INLINE ushort4 vload4(size_t i, const ushort *p) { p += 4 * i; return make_ushort4(p[0], p[1], p[2], p[3]); }
NF_INLINE float2  vload2(size_t i, const float *p)  { p += 2 * i; return make_float2(p[0], p[1]); }
NF_INLINE void vstore4(float4 v, size_t i, float *p) { p += 4 * i; p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w; }
NF_INLINE void vstore4(uchar4 v, size_t i, uchar *p) { p += 4 * i; p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w; }
NF_INLINE void vstore4(char4 v, size_t i, nf_i8 *p)  { p += 4 * i; p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w; }
NF_INLINE void vstore2(float2 v, size_t i, float *p) { p += 2 * i; p[0] = v.x; p[1] = v.y; }
NF_INLINE float  vload_half(size_t i, const NF_HALF *p)  { return __half2float(__ushort_as_half(p[i])); }
NF_INLINE float2 vload_half2(size_t i, const NF_HALF *p) { p += 2 * i; return make_float2(vload_half(0, p), vload_half(1, p)); }
NF_INLINE float4 vload_half4(size_t i, const NF_HALF *p) { p += 4 * i; return make_float4(vload_half(0, p), vload_half(1, p), vload_half(2, p), vload_half(3, p)); }
NF_INLINE void   vstore_half(float v, size_t i, NF_HALF *p) { p[i] = __half_as_ushort(__float2half_rn(v)); }
NF_INLINE void   vstore_half2(float2 v, size_t i, NF_HALF *p) { p += 2 * i; vstore_half(v.x, 0, p); vstore_half(v.y, 1, p); }
NF_INLINE void   vstore_half4(float4 v, size_t i, NF_HALF *p) { p += 4 * i; vstore_half(v.x, 0, p); vstore_half(v.y, 1, p); vstore_half(v.z, 2, p); vstore_half(v.w, 3, p); }

/* ---- 32-bit atomics on global or shared memory (OpenCL 1.2 core names) ---- */
NF_INLINE int  atomic_add(volatile int *p, int v)    { return atomicAdd((int *)p, v); }
NF_INLINE uint atomic_add(volatile uint *p, uint v)  { return atomicAdd((uint *)p, v); }
NF_INLINE int  atomic_sub(volatile int *p, int v)    { return atomicSub((int *)p, v); }
NF_INLINE uint atomic_sub(volatile uint *p, uint v)  { return atomicSub((uint *)p, v); }
NF_INLINE int  atomic_inc(volatile int *p)           { return atomicAdd((int *)p, 1); }
NF_INLINE uint atomic_inc(volatile uint *p)          { return atomicAdd((uint *)p, 1u); }
NF_INLINE int  atomic_dec(volatile int *p)           { return atomicSub((int *)p, 1); }
NF_INLINE uint atomic_dec(volatile uint *p)          { return atomicSub((uint *)p, 1u); }
NF_INLINE int  atomic_xchg(volatile int *p, int v)   { return atomicExch((int *)p, v); }
NF_INLINE uint atomic_xchg(volatile uint *p, uint v) { return atomicExch((uint *)p, v); }
NF_INLINE float atomic_xchg(volatile float *p, float v) { return atomicExch((float *)p, v); }
NF_INLINE int  atomic_cmpxchg(volatile int *p, int cmp, int v)    { return atomicCAS((int *)p, cmp, v); }
NF_INLINE uint atomic_cmpxchg(volatile uint *p, uint cmp, uint v) { return atomicCAS((uint *)p, cmp, v); }
NF_INLINE int  atomic_min(volatile int *p, int v)    { return atomicMin((int *)p, v); }
NF_INLINE uint atomic_min(volatile uint *p, uint v)  { return atomicMin((uint *)p, v); }
NF_INLINE int  atomic_max(volatile int *p, int v)    { return atomicMax((int *)p, v); }
NF_INLINE uint atomic_max(volatile uint *p, uint v)  { return atomicMax((uint *)p, v); }
NF_INLINE int  atomic_and(volatile int *p, int v)    { return atomicAnd((int *)p, v); }
NF_INLINE uint atomic_and(volatile uint *p, uint v)  { return atomicAnd((uint *)p, v); }
NF_INLINE int  atomic_or(volatile int *p, int v)     { return atomicOr((int *)p, v); }
NF_INLINE uint atomic_or(volatile uint *p, uint v)   { return atomicOr((uint *)p, v); }
NF_INLINE int  atomic_xor(volatile int *p, int v)    { return atomicXor((int *)p, v); }
NF_INLINE uint atomic_xor(volatile uint *p, uint v)  { return atomicXor((uint *)p, v); }

#else
/* ============================ OpenCL C 1.2 ============================ */
#define NF_OPENCL 1
typedef char  nf_i8;
typedef long  nf_i64;
typedef ulong nf_u64;
#define NF_HALF        half
#define NF_KERNEL      __kernel
#define NF_GLOBAL      __global
#define NF_LOCAL       __local
#define NF_LOCAL_PTR   __local
#define NF_INLINE      inline
#define NF_RESTRICT    restrict
#define NF_TABLE(type, name, n) __constant type name[n]
#define NF_INF         INFINITY
#define NF_NAN         NAN
#define NF_U64C(x)     x##UL
#define NF_F4(a, b, c, d) ((float4)((a), (b), (c), (d)))
#define NF_F2(a, b)       ((float2)((a), (b)))
#define NF_F4S(s)         ((float4)(s))
/* The host passes -DUSE_SUBGROUPS=1 only for devices that report a subgroup extension; a compiler
 * that defines neither extension macro gets the local-memory paths. */
#ifdef USE_SUBGROUPS
#  if defined(cl_khr_subgroups)
#    pragma OPENCL EXTENSION cl_khr_subgroups : enable
#  elif defined(cl_intel_subgroups)
#    pragma OPENCL EXTENSION cl_intel_subgroups : enable
#  else
#    undef USE_SUBGROUPS
#  endif
#endif
/* OpenCL 1.2 has no portable subgroup shuffle, so NF_HAS_SG_SHUFFLE stays undefined. */
#endif /* dialect */

#endif /* NF_DIALECT_H */
