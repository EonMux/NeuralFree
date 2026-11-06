/* matvec_w12.cl - W12 weight kernels: register decode of the per-row lossless 12-bit records, the 8-bit Q
 * view of the same code bytes, and the whole-tensor BF16 decode for large prefills. Requires:
 * common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/quant_blocks.h */
/*
 * One W12 row record (GGMLType.W12, W12Codec; a device-resident form, never stored in GGUF files) holds, for
 * a row of in_dim elements (in_dim a whole number of 64): the Q region, one byte per element (bits 7..4 the
 * element's 4-bit offset c into the row's 16-binade exponent window [E - 15, E], bit 3 its sign, bits 2..0
 * mantissa bits 6..4), the B region (mantissa bits 3..0: element 2j in the low nibble of byte j, 2j + 1 in
 * the high one), and 12 bytes of metadata: the scale f32 2^(E-127) (bits E << 23), the row's exact escape
 * count rp and its Q escape count rpq as i32 exclusive ends (a row's start is the previous row's end, 0 for
 * row 0). After the row region follow the shard's two escape tables: the exact table at byte
 * out_dim * row_stride, 12-byte entries [col i32][bf16 bits u16][pad u16][val f32], and the Q table after it,
 * 8-byte entries [col i32][val f32], with val the scaled true weight w * 2^(127-E) (exact in fp32, checked at
 * encode). Every plane code decodes with the exponent field pinned at 112 + c: an in-band element's plane
 * value is its true value times 2^(127-E), and an escaped element's all-zero code reads as exactly 2^-15 (the
 * Q pass's midpoint nibble: 1.0625 * 2^-15) whatever the row window — so the per-entry correction
 * acc += x * val - x * 2^-15 cancels the placeholder and adds the scaled true weight, the epilogue scales the
 * row sum by 2^(E-127) once, and every weight's contribution is exact; a row without escapes reproduces the
 * BF16 matvec over the same weights bit for bit (identical products up to the power-of-two scale, identical
 * reduction order). The Q kernels stream only the Q regions (1 B/weight) and correct only their row's Q-table
 * entries; the escapes below the window keep their placeholder, an error far inside the 8-bit view's
 * half-binade bound.
 *
 * Same launch contract as matvec_f16_bf16.cl: row kernels run one workgroup per output row with
 * global = rows * L, local = L (L a power of two <= LLM_DIMX; CUDA: a multiple of 32), the _2D forms
 * global = {out_dim * L, seq_len}, local = {L, 1}; their early returns are uniform per workgroup, and the
 * decode kernel is one workgroup per row. Rows and token rows are addressed in nf_i64. A W12 row holds
 * in_dim/64 whole plane footprints (GpuWeightUploader refuses other rows), so the float4 chunk loop has no
 * tail and every row start is block aligned; all metadata and escape fields are 4-byte aligned and the u16
 * bf16 bits 2-byte aligned. A row spans in_dim * 3 / 2 bytes at most and table bases stay below the shard
 * limit, so element and byte offsets fit nf_i64 and 32-bit locals.
 */

/* Plane bytes of one W12 row footprint (GGMLType.W12: 96 bytes per 64 elements). */
#define W12_PLANE_BYTES 96
/* Elements of one W12 plane footprint. */
#define W12_BLOCK_ELEMENTS 64
/* Metadata bytes of one row record: scale f32, exact escape count, Q escape count. */
#define W12_ROW_META_BYTES 12
/* Bytes of one exact escape table entry. */
#define W12_ESCAPE_ENTRY_BYTES 12
/* Bytes of one Q escape table entry. */
#define W12_Q_ESCAPE_ENTRY_BYTES 8
/* The fixed midpoint nibble the Q view reads the B region as. */
#define W12_QVIEW_NIBBLE 0x8
/* The plane placeholder of an escaped element (the all-zero code, exponent pinned at 112). */
#define W12_PLACEHOLDER 3.0517578125e-05f
/* The Q pass's placeholder over the midpoint nibble. */
#define W12_Q_PLACEHOLDER 3.24249267578125e-05f

/* The bf16 bits of an in-band element: the true exponent field c + top - 15 (the decode kernel's form). */
NF_INLINE ushort nf_w12_true_bits(uchar q, uint nibble, uint top) {
    const uint c = ((uint)q >> 4) & 0xFu;
    const uint s = ((uint)q >> 3) & 0x1u;
    const uint m654 = (uint)q & 0x7u;
    return (ushort)(((s << 15) | ((c + top - 15u) << 7) | (m654 << 4) | nibble) & 0xFFFFu);
}

/* The plane value bits of an element: the exponent field pinned at 112 + c, so the value is the true value
 * times 2^(127 - top) (an escaped element's all-zero bytes read as W12_PLACEHOLDER, the Q nibble as
 * W12_Q_PLACEHOLDER, whatever the row window). */
NF_INLINE ushort nf_w12_plane_bits(uchar q, uint nibble) {
    const uint c = ((uint)q >> 4) & 0xFu;
    const uint s = ((uint)q >> 3) & 0x1u;
    const uint m654 = (uint)q & 0x7u;
    return (ushort)(((s << 15) | ((c + 112u) << 7) | (m654 << 4) | nibble) & 0xFFFFu);
}

NF_INLINE uint nf_w12_u16(NF_GLOBAL const uchar* p) {
    return (uint)p[0] | ((uint)p[1] << 8);
}

NF_INLINE uint nf_w12_u32(NF_GLOBAL const uchar* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

/* The plane region size of one row. */
NF_INLINE uint nf_w12_planes(int in_dim) {
    return (uint)(in_dim / W12_BLOCK_ELEMENTS) * W12_PLANE_BYTES;
}

/* The byte stride of one row record: the planes plus the metadata. */
NF_INLINE nf_i64 nf_w12_row_stride(uint planes) {
    return (nf_i64)planes + W12_ROW_META_BYTES;
}

/* The row record of row. */
NF_INLINE NF_GLOBAL const uchar* nf_w12_row(NF_GLOBAL const uchar* weight, int in_dim, int row) {
    return weight + (nf_i64)row * nf_w12_row_stride(nf_w12_planes(in_dim));
}

/* The escape-table bases of the shard whose row records start at weight: the exact table starts after the
 * row region, the Q table after the exact table (its size is the last row's exact escape count). */
NF_INLINE void nf_w12_tables(NF_GLOBAL const uchar* weight, int out_dim, int in_dim,
                             NF_GLOBAL const uchar** esc, NF_GLOBAL const uchar** qesc) {
    const uint planes = nf_w12_planes(in_dim);
    const nf_i64 stride = nf_w12_row_stride(planes);
    *esc = weight + (nf_i64)out_dim * stride;
    *qesc = *esc + (nf_i64)nf_w12_u32(weight + (nf_i64)(out_dim - 1) * stride + planes + 4u)
            * W12_ESCAPE_ENTRY_BYTES;
}

/* The row's escape range [start, end) of the count at metadata offset metaOff (4: exact, 8: Q) from its and
 * the previous row's metadata counts (0 for row 0). */
NF_INLINE void nf_w12_escapes(NF_GLOBAL const uchar* row, uint planes, uint metaOff, int rowIndex,
                              uint* start, uint* end) {
    *end = nf_w12_u32(row + planes + metaOff);
    *start = (rowIndex == 0) ? 0u : nf_w12_u32(row - W12_ROW_META_BYTES + metaOff);
}

/* The window top of the row: the scale's f32 exponent field. */
NF_INLINE uint nf_w12_top(NF_GLOBAL const uchar* row, uint planes) {
    return (nf_w12_u32(row + planes) >> 23) & 0xFFu;
}

/* The four plane values of the chunk at element offset k (a multiple of 4) of the row whose B region
 * starts at bBase (the row's width, right after the Q bytes); the Q view reads every B nibble as the
 * midpoint. */
NF_INLINE float4 nf_w12_load4(NF_GLOBAL const uchar* row, uint bBase, uint k, uint view) {
    const uchar4 q = vload4(0, row + k);
    const uint b0 = (view != 0u) ? 0u : (uint)row[bBase + (k >> 1)];
    const uint b1 = (view != 0u) ? 0u : (uint)row[bBase + (k >> 1) + 1u];
    const uint n0 = (view != 0u) ? W12_QVIEW_NIBBLE : (b0 & 0xFu);
    const uint n1 = (view != 0u) ? W12_QVIEW_NIBBLE : ((b0 >> 4) & 0xFu);
    const uint n2 = (view != 0u) ? W12_QVIEW_NIBBLE : (b1 & 0xFu);
    const uint n3 = (view != 0u) ? W12_QVIEW_NIBBLE : ((b1 >> 4) & 0xFu);
    return NF_F4(bf16_to_f32(nf_w12_plane_bits(q.x, n0)),
                 bf16_to_f32(nf_w12_plane_bits(q.y, n1)),
                 bf16_to_f32(nf_w12_plane_bits(q.z, n2)),
                 bf16_to_f32(nf_w12_plane_bits(q.w, n3)));
}

/* The plane-product loop over two rows sharing x (float4 strides lid*4, lid*4 + L*4, ...; n a whole number
 * of footprints, so no scalar tail); adds into *gs (row g) and *us (row u). */
NF_INLINE void nf_w12_partial_planes(NF_GLOBAL const uchar* NF_RESTRICT g, NF_GLOBAL const uchar* NF_RESTRICT u,
                                     NF_GLOBAL const float* NF_RESTRICT x, const int n, float* gs, float* us) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    float ga = 0.0f;
    float ua = 0.0f;
    for (int j = lid * 4; j < n; j += lsize * 4) {
        const float4 xv = vload4(0, x + j);
        ga += dot(nf_w12_load4(g, (uint)n, (uint)j, 0u), xv);
        ua += dot(nf_w12_load4(u, (uint)n, (uint)j, 0u), xv);
    }
    *gs += ga;
    *us += ua;
}

/* This work-item's share of a row's exact escape corrections over the 12-byte entries at escBase. */
NF_INLINE float nf_w12_correct(NF_GLOBAL const float* NF_RESTRICT x, NF_GLOBAL const uchar* NF_RESTRICT escBase,
                               const uint escStart, const uint escEnd, float ps) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    for (uint i = escStart + (uint)lid; i < escEnd; i += (uint)lsize) {
        NF_GLOBAL const uchar* e = escBase + (nf_i64)i * W12_ESCAPE_ENTRY_BYTES;
        const float xv = x[nf_w12_u32(e)];
        ps = fma(as_float(nf_w12_u32(e + 8)), xv, fma(-W12_PLACEHOLDER, xv, ps));
    }
    return ps;
}

/* This work-item's share of a row's Q escape corrections over the 8-byte entries at qEscBase. */
NF_INLINE float nf_w12q_correct(NF_GLOBAL const float* NF_RESTRICT x, NF_GLOBAL const uchar* NF_RESTRICT qEscBase,
                                const uint escStart, const uint escEnd, float ps) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    for (uint i = escStart + (uint)lid; i < escEnd; i += (uint)lsize) {
        NF_GLOBAL const uchar* e = qEscBase + (nf_i64)i * W12_Q_ESCAPE_ENTRY_BYTES;
        const float xv = x[nf_w12_u32(e)];
        ps = fma(as_float(nf_w12_u32(e + 4)), xv, fma(-W12_Q_PLACEHOLDER, xv, ps));
    }
    return ps;
}

/* ================================ exact =============================== */

/* output[row] = dot(weight[row, :], input). */
NF_KERNEL void W12_MatVec(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const uchar* esc;
    NF_GLOBAL const uchar* qesc;
    nf_w12_tables(weight, out_dim, in_dim, &esc, &qesc);
    NF_GLOBAL const uchar* wrow = nf_w12_row(weight, in_dim, row);
    const uint planes = nf_w12_planes(in_dim);
    uint escStart;
    uint escEnd;
    nf_w12_escapes(wrow, planes, 4u, row, &escStart, &escEnd);
    float ps = 0.0f;
    {
        const int lid = get_local_id(0);
        const int lsize = get_local_size(0);
        for (int j = lid * 4; j < in_dim; j += lsize * 4) {
            ps += dot(nf_w12_load4(wrow, (uint)in_dim, (uint)j, 0u), vload4(0, input + j));
        }
    }
    ps = nf_w12_correct(input, esc, escStart, escEnd, ps);

    NF_LOCAL float scratch[LLM_DIMX];
    const float sum = nf_wg_sum(scratch, ps);
    if (get_local_id(0) == 0) {
        output[row] = sum * as_float(nf_w12_u32(wrow + planes));
    }
}

/* addend_output[row] += dot(weight[row, :], input). */
NF_KERNEL void W12_MatVecAdd(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT addend_output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const uchar* esc;
    NF_GLOBAL const uchar* qesc;
    nf_w12_tables(weight, out_dim, in_dim, &esc, &qesc);
    NF_GLOBAL const uchar* wrow = nf_w12_row(weight, in_dim, row);
    const uint planes = nf_w12_planes(in_dim);
    uint escStart;
    uint escEnd;
    nf_w12_escapes(wrow, planes, 4u, row, &escStart, &escEnd);
    float ps = 0.0f;
    {
        const int lid = get_local_id(0);
        const int lsize = get_local_size(0);
        for (int j = lid * 4; j < in_dim; j += lsize * 4) {
            ps += dot(nf_w12_load4(wrow, (uint)in_dim, (uint)j, 0u), vload4(0, input + j));
        }
    }
    ps = nf_w12_correct(input, esc, escStart, escEnd, ps);

    NF_LOCAL float scratch[LLM_DIMX];
    const float sum = nf_wg_sum(scratch, ps);
    if (get_local_id(0) == 0) {
        addend_output[row] += sum * as_float(nf_w12_u32(wrow + planes));
    }
}

/* output[row] = silu(dot(gate_weight[row, :], input)) * dot(up_weight[row, :], input). The gate and up
 * weights are separate shard records, each with its own escape tables. */
NF_KERNEL void W12_MatVecFusedGateUp(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight,
    NF_GLOBAL const uchar* NF_RESTRICT up_weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const uchar* gesc;
    NF_GLOBAL const uchar* gqesc;
    nf_w12_tables(gate_weight, out_dim, in_dim, &gesc, &gqesc);
    NF_GLOBAL const uchar* uesc;
    NF_GLOBAL const uchar* uqesc;
    nf_w12_tables(up_weight, out_dim, in_dim, &uesc, &uqesc);
    const uint planes = nf_w12_planes(in_dim);
    NF_GLOBAL const uchar* grow = nf_w12_row(gate_weight, in_dim, row);
    NF_GLOBAL const uchar* urow = nf_w12_row(up_weight, in_dim, row);
    uint gStart;
    uint gEnd;
    nf_w12_escapes(grow, planes, 4u, row, &gStart, &gEnd);
    uint uStart;
    uint uEnd;
    nf_w12_escapes(urow, planes, 4u, row, &uStart, &uEnd);

    float ga = 0.0f;
    float ua = 0.0f;
    nf_w12_partial_planes(grow, urow, input, in_dim, &ga, &ua);
    ga = nf_w12_correct(input, gesc, gStart, gEnd, ga);
    ua = nf_w12_correct(input, uesc, uStart, uEnd, ua);

    NF_LOCAL float scratch[LLM_DIMX];
    const float gateVal = nf_wg_sum(scratch, ga) * as_float(nf_w12_u32(grow + planes));
    const float upVal = nf_wg_sum(scratch, ua) * as_float(nf_w12_u32(urow + planes));
    if (get_local_id(0) == 0) {
        output[row] = nf_silu(gateVal) * upVal;
    }
}

/* output[s * out_dim + row] = dot(weight[row, :], input[s, :]) for s in [0, seq_len). */
NF_KERNEL void W12_BatchMatMul(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const uchar* esc;
    NF_GLOBAL const uchar* qesc;
    nf_w12_tables(weight, out_dim, in_dim, &esc, &qesc);
    NF_GLOBAL const uchar* wrow = nf_w12_row(weight, in_dim, row);
    const uint planes = nf_w12_planes(in_dim);
    uint escStart;
    uint escEnd;
    nf_w12_escapes(wrow, planes, 4u, row, &escStart, &escEnd);
    const float scale = as_float(nf_w12_u32(wrow + planes));

    NF_LOCAL float scratch[LLM_DIMX];
    for (int s = 0; s < seq_len; s++) {
        float ps = 0.0f;
        {
            const int lid = get_local_id(0);
            const int lsize = get_local_size(0);
            for (int j = lid * 4; j < in_dim; j += lsize * 4) {
                ps += dot(nf_w12_load4(wrow, (uint)in_dim, (uint)j, 0u),
                          vload4(0, input + (nf_i64)s * in_dim + j));
            }
        }
        ps = nf_w12_correct(input + (nf_i64)s * in_dim, esc, escStart, escEnd, ps);
        const float sum = nf_wg_sum(scratch, ps);
        if (get_local_id(0) == 0) {
            output[(nf_i64)s * out_dim + row] = sum * scale;
        }
    }
}

/* W12_BatchMatMul with one (row, token) pair per workgroup. */
NF_KERNEL void W12_BatchMatMul_2D(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    const int s = get_global_id(1);
    if (row >= out_dim || s >= seq_len) return;

    NF_GLOBAL const uchar* esc;
    NF_GLOBAL const uchar* qesc;
    nf_w12_tables(weight, out_dim, in_dim, &esc, &qesc);
    NF_GLOBAL const uchar* wrow = nf_w12_row(weight, in_dim, row);
    const uint planes = nf_w12_planes(in_dim);
    uint escStart;
    uint escEnd;
    nf_w12_escapes(wrow, planes, 4u, row, &escStart, &escEnd);
    NF_GLOBAL const float* xin = input + (nf_i64)s * in_dim;
    float ps = 0.0f;
    {
        const int lid = get_local_id(0);
        const int lsize = get_local_size(0);
        for (int j = lid * 4; j < in_dim; j += lsize * 4) {
            ps += dot(nf_w12_load4(wrow, (uint)in_dim, (uint)j, 0u), vload4(0, xin + j));
        }
    }
    ps = nf_w12_correct(xin, esc, escStart, escEnd, ps);

    NF_LOCAL float scratch[LLM_DIMX];
    const float sum = nf_wg_sum(scratch, ps);
    if (get_local_id(0) == 0) {
        output[(nf_i64)s * out_dim + row] = sum * as_float(nf_w12_u32(wrow + planes));
    }
}

/* ================================== Q view ================================== */

/* The same kernels over the nested 8-bit view of the code bytes: every B nibble read as the midpoint, only
 * the Q regions streamed and only the Q-table escapes corrected. */

/* output[row] = dot(qview(weight[row, :]), input). */
NF_KERNEL void W12Q_MatVec(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const uchar* esc;
    NF_GLOBAL const uchar* qesc;
    nf_w12_tables(weight, out_dim, in_dim, &esc, &qesc);
    NF_GLOBAL const uchar* wrow = nf_w12_row(weight, in_dim, row);
    const uint planes = nf_w12_planes(in_dim);
    uint escStart;
    uint escEnd;
    nf_w12_escapes(wrow, planes, 8u, row, &escStart, &escEnd);
    float ps = 0.0f;
    {
        const int lid = get_local_id(0);
        const int lsize = get_local_size(0);
        for (int j = lid * 4; j < in_dim; j += lsize * 4) {
            ps += dot(nf_w12_load4(wrow, (uint)in_dim, (uint)j, 1u), vload4(0, input + j));
        }
    }
    ps = nf_w12q_correct(input, qesc, escStart, escEnd, ps);

    NF_LOCAL float scratch[LLM_DIMX];
    const float sum = nf_wg_sum(scratch, ps);
    if (get_local_id(0) == 0) {
        output[row] = sum * as_float(nf_w12_u32(wrow + planes));
    }
}

/* addend_output[row] += dot(qview(weight[row, :]), input). */
NF_KERNEL void W12Q_MatVecAdd(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT addend_output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const uchar* esc;
    NF_GLOBAL const uchar* qesc;
    nf_w12_tables(weight, out_dim, in_dim, &esc, &qesc);
    NF_GLOBAL const uchar* wrow = nf_w12_row(weight, in_dim, row);
    const uint planes = nf_w12_planes(in_dim);
    uint escStart;
    uint escEnd;
    nf_w12_escapes(wrow, planes, 8u, row, &escStart, &escEnd);
    float ps = 0.0f;
    {
        const int lid = get_local_id(0);
        const int lsize = get_local_size(0);
        for (int j = lid * 4; j < in_dim; j += lsize * 4) {
            ps += dot(nf_w12_load4(wrow, (uint)in_dim, (uint)j, 1u), vload4(0, input + j));
        }
    }
    ps = nf_w12q_correct(input, qesc, escStart, escEnd, ps);

    NF_LOCAL float scratch[LLM_DIMX];
    const float sum = nf_wg_sum(scratch, ps);
    if (get_local_id(0) == 0) {
        addend_output[row] += sum * as_float(nf_w12_u32(wrow + planes));
    }
}

/* output[s * out_dim + row] = dot(qview(weight[row, :]), input[s, :]). */
NF_KERNEL void W12Q_BatchMatMul(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const uchar* esc;
    NF_GLOBAL const uchar* qesc;
    nf_w12_tables(weight, out_dim, in_dim, &esc, &qesc);
    NF_GLOBAL const uchar* wrow = nf_w12_row(weight, in_dim, row);
    const uint planes = nf_w12_planes(in_dim);
    uint escStart;
    uint escEnd;
    nf_w12_escapes(wrow, planes, 8u, row, &escStart, &escEnd);
    const float scale = as_float(nf_w12_u32(wrow + planes));

    NF_LOCAL float scratch[LLM_DIMX];
    for (int s = 0; s < seq_len; s++) {
        float ps = 0.0f;
        {
            const int lid = get_local_id(0);
            const int lsize = get_local_size(0);
            for (int j = lid * 4; j < in_dim; j += lsize * 4) {
                ps += dot(nf_w12_load4(wrow, (uint)in_dim, (uint)j, 1u),
                          vload4(0, input + (nf_i64)s * in_dim + j));
            }
        }
        ps = nf_w12q_correct(input + (nf_i64)s * in_dim, qesc, escStart, escEnd, ps);
        const float sum = nf_wg_sum(scratch, ps);
        if (get_local_id(0) == 0) {
            output[(nf_i64)s * out_dim + row] = sum * scale;
        }
    }
}

/* W12Q_BatchMatMul with one (row, token) pair per workgroup. */
NF_KERNEL void W12Q_BatchMatMul_2D(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    const int s = get_global_id(1);
    if (row >= out_dim || s >= seq_len) return;

    NF_GLOBAL const uchar* esc;
    NF_GLOBAL const uchar* qesc;
    nf_w12_tables(weight, out_dim, in_dim, &esc, &qesc);
    NF_GLOBAL const uchar* wrow = nf_w12_row(weight, in_dim, row);
    const uint planes = nf_w12_planes(in_dim);
    uint escStart;
    uint escEnd;
    nf_w12_escapes(wrow, planes, 8u, row, &escStart, &escEnd);
    NF_GLOBAL const float* xin = input + (nf_i64)s * in_dim;
    float ps = 0.0f;
    {
        const int lid = get_local_id(0);
        const int lsize = get_local_size(0);
        for (int j = lid * 4; j < in_dim; j += lsize * 4) {
            ps += dot(nf_w12_load4(wrow, (uint)in_dim, (uint)j, 1u), vload4(0, xin + j));
        }
    }
    ps = nf_w12q_correct(xin, qesc, escStart, escEnd, ps);

    NF_LOCAL float scratch[LLM_DIMX];
    const float sum = nf_wg_sum(scratch, ps);
    if (get_local_id(0) == 0) {
        output[(nf_i64)s * out_dim + row] = sum * as_float(nf_w12_u32(wrow + planes));
    }
}

/* ================================ decode ================================ */

/* out[(out_row0 + row) * in_dim + i] = the bf16 bits of element i of weight row (one workgroup per row;
 * the BF16 mirror a prefill wider than the matvec row budget decodes into before its GEMM; out_row0 places
 * a row shard of the source tensor in the whole-tensor mirror). In-band elements decode from their code
 * bytes (true exponent field c + top - 15); the row's escaped elements are overwritten from the exact
 * escape table. */
NF_KERNEL void W12_DecodeBF16(
    const int out_dim,
    const int in_dim,
    const int out_row0,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL ushort* NF_RESTRICT out)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const uchar* esc;
    NF_GLOBAL const uchar* qesc;
    nf_w12_tables(weight, out_dim, in_dim, &esc, &qesc);
    NF_GLOBAL const uchar* wrow = nf_w12_row(weight, in_dim, row);
    const uint planes = nf_w12_planes(in_dim);
    const uint top = nf_w12_top(wrow, planes);
    NF_GLOBAL ushort* orow = out + (nf_i64)(out_row0 + row) * in_dim;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    for (int j = lid * 4; j < in_dim; j += lsize * 4) {
        const uint k = (uint)j;
        const uchar4 q = vload4(0, wrow + k);
        const uint b0 = (uint)wrow[in_dim + (k >> 1)];
        const uint b1 = (uint)wrow[in_dim + (k >> 1) + 1u];
        orow[j] = nf_w12_true_bits(q.x, b0 & 0xFu, top);
        orow[j + 1] = nf_w12_true_bits(q.y, (b0 >> 4) & 0xFu, top);
        orow[j + 2] = nf_w12_true_bits(q.z, b1 & 0xFu, top);
        orow[j + 3] = nf_w12_true_bits(q.w, (b1 >> 4) & 0xFu, top);
    }
    /* the escape scatter overwrites plane writes of other work-items: order them first */
    barrier(CLK_GLOBAL_MEM_FENCE);
    uint escStart;
    uint escEnd;
    nf_w12_escapes(wrow, planes, 4u, row, &escStart, &escEnd);
    for (uint i = escStart + (uint)lid; i < escEnd; i += (uint)lsize) {
        NF_GLOBAL const uchar* e = esc + (nf_i64)i * W12_ESCAPE_ENTRY_BYTES;
        orow[nf_w12_u32(e)] = (ushort)nf_w12_u16(e + 4);
    }
}
