/* core.cl - fill, copy, strided copy and dtype casts of the tensor runtime. Requires: common/dialect.h, common/train.h */
/*
 * Offsets are element offsets into the buffer unless named *_bytes or *_halves. Data movement is done on
 * raw 32-bit words, so every kernel here is bit-identical to the CPU path: NaN payloads and denormals pass
 * through unchanged, and FLOAT32 and INT32 tensors share the copy kernels. The element types below name
 * the tensor data; a kernel that only moves or re-encodes bits declares that buffer as uint words.
 *
 * TR_Fill_F32(i n, f value, i out_off, W out)
 *     out[out_off + i] = value, i < n.
 * TR_Copy_F32(i n, i in_off, i out_off, R in, W out)
 *     out[out_off + i] = in[in_off + i], i < n (32-bit words).
 * TR_Copy_Strided_F32(i n, i rank, i d0..d5, i is0..is5, i os0..os5, i in_off, i out_off, R in, W out)
 *     For row-major linear index i < n over dims d[0..rank) (n = their product):
 *     out[out_off + offset(i, os)] = in[in_off + offset(i, is)], offset as tr_coord_offset.
 *     Unused slots carry dim 1 and stride 0. Distinct indices must map to distinct output elements.
 * TR_Cast_I32_F32(i n, i in_off, i out_off, R int in, W float out)          Java (float) of an int.
 * TR_Cast_F32_I32(i n, i in_off, i out_off, R float in, W int out)          Java (int) of a float:
 *     NaN -> 0, saturating at Integer.MIN_VALUE/MAX_VALUE, otherwise toward zero.
 * TR_Cast_BOOL_F32(i n, i in_off_bytes, i out_off, R words in, W float out) byte != 0 -> 1.0f, else 0.0f.
 * TR_Cast_F32_BOOL(i n, i in_off, i out_off_bytes, R float in, W words out) x != 0 -> 1 (NaN -> 1, as Java
 *     x != 0f; a denormal counts as non-zero).
 * TR_Cast_F32_F16(i n, i in_off, i out_off_halves, R float in, W words out)   tensor/HalfFloats RNE.
 * TR_Cast_F16_F32(i n, i in_off_halves, i out_off, R words in, W float out)   exact.
 * TR_Cast_F32_BF16(i n, i in_off, i out_off_halves, R float in, W words out)  tensor/HalfFloats RNE.
 * TR_Cast_BF16_F32(i n, i in_off_halves, i out_off, R words in, W float out)  exact.
 * TR_Cast_I64_I32(i n, i in_off, i out_off, R words in, W int out)
 *     out[out_off + i] = low word of in[in_off + i] (Java (int) of a long; in_off in INT64 elements).
 *
 * Sub-word outputs (BOOL, FLOAT16, BFLOAT16) are written as whole words, one work-item per output word, so
 * out_off_bytes is a multiple of 4 and out_off_halves a multiple of 2 (the host writes them into fresh
 * allocations at offset 0); bytes past element n in the last word are written as 0. Sub-word inputs are
 * read at any element offset.
 *
 * Dispatch: 1-D, local TR_LOCAL. Global = n rounded up, except TR_Cast_F32_BOOL (ceil(n/4) items) and
 * TR_Cast_F32_F16 / TR_Cast_F32_BF16 (ceil(n/2) items). No barriers; in and out never alias.
 */

NF_KERNEL void TR_Fill_F32(
    const int n,
    const float value,
    const int out_off,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    out[out_off + i] = value;
}

NF_KERNEL void TR_Copy_F32(
    const int n,
    const int in_off,
    const int out_off,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    out[out_off + i] = in[in_off + i];
}

NF_KERNEL void TR_Copy_Strided_F32(
    const int n,
    const int rank,
    const int d0, const int d1, const int d2, const int d3, const int d4, const int d5,
    const int is0, const int is1, const int is2, const int is3, const int is4, const int is5,
    const int os0, const int os1, const int os2, const int os3, const int os4, const int os5,
    const int in_off,
    const int out_off,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const int d[TR_MAX_RANK] = { d0, d1, d2, d3, d4, d5 };
    const int is[TR_MAX_RANK] = { is0, is1, is2, is3, is4, is5 };
    const int os[TR_MAX_RANK] = { os0, os1, os2, os3, os4, os5 };
    out[out_off + tr_coord_offset(i, rank, d, os)] = in[in_off + tr_coord_offset(i, rank, d, is)];
}

NF_KERNEL void TR_Cast_I32_F32(
    const int n,
    const int in_off,
    const int out_off,
    NF_GLOBAL const int* NF_RESTRICT in,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    out[out_off + i] = (float)in[in_off + i];
}

NF_KERNEL void TR_Cast_F32_I32(
    const int n,
    const int in_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL int* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float x = in[in_off + i];
    int r;
    if (tr_isnan(x)) {
        r = 0;
    } else if (x >= 2147483648.0f) {
        r = 2147483647;
    } else if (x <= -2147483648.0f) {
        r = (-2147483647 - 1);
    } else {
        r = (int)x;
    }
    out[out_off + i] = r;
}

NF_KERNEL void TR_Cast_BOOL_F32(
    const int n,
    const int in_off_bytes,
    const int out_off,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const int p = in_off_bytes + i;
    out[out_off + i] = tr_word_byte(in[p >> 2], p) != 0u ? 0x3f800000u : 0u;   /* 1.0f : +0.0f */
}

NF_KERNEL void TR_Cast_F32_BOOL(
    const int n,
    const int in_off,
    const int out_off_bytes,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int words = (int)(((uint)n + 3u) >> 2);
    const int w = tr_gid(words);
    if (w < 0) return;
    uint word = 0u;
    for (int b = 0; b < 4; ++b) {
        const int e = 4 * w + b;
        if (e < n && (in[in_off + e] & 0x7fffffffu) != 0u) {
            word |= 1u << (uint)(8 * b);
        }
    }
    out[(out_off_bytes >> 2) + w] = word;
}

NF_KERNEL void TR_Cast_F32_F16(
    const int n,
    const int in_off,
    const int out_off_halves,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int words = (int)(((uint)n + 1u) >> 1);
    const int w = tr_gid(words);
    if (w < 0) return;
    const int e = 2 * w;
    const uint lo = tr_f32_to_f16_bits(in[in_off + e]);
    const uint hi = e + 1 < n ? tr_f32_to_f16_bits(in[in_off + e + 1]) : 0u;
    out[(out_off_halves >> 1) + w] = lo | (hi << 16);
}

NF_KERNEL void TR_Cast_F16_F32(
    const int n,
    const int in_off_halves,
    const int out_off,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const int p = in_off_halves + i;
    out[out_off + i] = tr_f16_bits_to_f32(tr_word_half(in[p >> 1], p));
}

NF_KERNEL void TR_Cast_F32_BF16(
    const int n,
    const int in_off,
    const int out_off_halves,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int words = (int)(((uint)n + 1u) >> 1);
    const int w = tr_gid(words);
    if (w < 0) return;
    const int e = 2 * w;
    const uint lo = tr_f32_to_bf16_bits(in[in_off + e]);
    const uint hi = e + 1 < n ? tr_f32_to_bf16_bits(in[in_off + e + 1]) : 0u;
    out[(out_off_halves >> 1) + w] = lo | (hi << 16);
}

NF_KERNEL void TR_Cast_BF16_F32(
    const int n,
    const int in_off_halves,
    const int out_off,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const int p = in_off_halves + i;
    out[out_off + i] = tr_bf16_bits_to_f32(tr_word_half(in[p >> 1], p));
}

NF_KERNEL void TR_Cast_I64_I32(
    const int n,
    const int in_off,
    const int out_off,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    out[out_off + i] = in[2u * (uint)(in_off + i)];
}
