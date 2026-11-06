/* vision.cl - LLM_Conv2dDepthwise, LLM_Pool2dAvg and LLM_RelPosAttn, the kernels of the convolutional
 * vision encoders (the yasa2 ConvNeXt, the gemma3nv MobileNetV5 and the DeepSeek-OCR SAM backbone).
 * Requires: common/dialect.h, common/config.h, common/reduce.h */
/*
 * Feature maps are pixel-major F32: element (x, y, c) of a W x H map of C channels is
 * v[v_off + (y*W + x)*C + c], channels innermost (the row blocks of the vision graphs). Offsets are in
 * float elements; every scalar is a 32-bit int; addresses are formed in nf_i64. Java:
 * gguf.ops.vision.VisionKernels; CPU twins: MobileNetV5Maps.depthwise, MobileNetV5Maps.avgPool and
 * SamBackbone.attention.
 *
 * LLM_Conv2dDepthwise(i W, i H, i OW, i OH, i C, i K, i SY, i SX, i PY, i PX, i x_off, i y_off, R x, R w, R b, W y)
 *     ggml_conv_2d_dw with a K x K kernel: x[W*H][C], w[C][K*K] (value (c*K + ky)*K + kx, the F32
 *     decode of the ggml {K, K, 1, C} kernel), b[C], y[OW*OH][C]:
 *     y[(oy*OW + ox)*C + c] = (sum_{ky, kx asc, oy*SY + ky - PY and ox*SX + kx - PX inside} ) + b[c]
 *     Taps outside the map read zero. OW and OH are the caller's output sides ((H + pad - K) / SY + 1
 *     per axis for the graphs' whole-pixel paddings). y must not overlap x.
 *
 * LLM_Pool2dAvg(i W, i H, i OW, i OH, i C, i KY, i KX, i x_off, i y_off, R x, W y)
 *     ggml_pool_2d AVG over whole KY x KX windows at stride KY, KX, no padding:
 *     y[(oy*OW + ox)*C + c] = (sum_{dy < KY, dx < KX} x[((oy*KY + dy)*W + ox*KX + dx)*C + c]) / (KY*KX)
 *     A 1 x n window (KY = H, KX = 1, W = 1) reads the per-channel mean of the whole map, the column
 *     reduction of the yasa2 global response normalization.
 *
 * LLM_RelPosAttn(i groups, i side, i n_head, i dh, f scale, i q_stride, i kv_stride, i out_stride,
 *                i q_off, i kv_off, i out_off, R q, R k, R v, R rw, R rh, W out)
 *     The bidirectional softmax attention of the DeepSeek-OCR SAM backbone with its decomposed
 *     relative-position bias (models/deepseekocr.cpp build_sam): `groups` attention groups (the
 *     window_partition windows, or one group for a global block) of side*side token rows, query head h
 *     reading the h-th dh-wide slice of every q, k and v row. The score of query tq of group g against
 *     key tk, at qx = tq % side, qy = tq / side, kx, ky alike, is
 *         s = q . k * scale + q . rw[qx - kx + side - 1] + q . rh[qy - ky + side - 1]
 *     (rw, rh: [2*side - 1][dh] relative-position tables, row q - k + side - 1 the bias of the pair),
 *     then the softmax over the group's keys and the weighted value sum. One workgroup per
 *     (group, head, query); the probabilities of one query's keys are staged in shared memory
 *     (NF_VISION_T_MAX floats), so T = side*side must not exceed it. Deterministic: keys and output
 *     dims are summed in ascending order.
 *
 * Dispatch: LLM_Conv2dDepthwise and LLM_Pool2dAvg run one work-item per output element, global =
 * roundUp(OW*OH*C, L), local = {L}, no barriers. LLM_RelPosAttn runs one workgroup per
 * (group, head, query), global = groups*n_head*side*side (exact), local = {L}, L a power of two
 * <= LLM_DIMX (a multiple of 32 on CUDA).
 */

/* Tokens per attention group LLM_RelPosAttn stages probabilities for (a 64-pixel SAM grid per side:
 * 4096 patches of the largest DeepSeek-OCR tile). */
#define NF_VISION_T_MAX 4096

NF_KERNEL void LLM_Conv2dDepthwise(
    const int W,
    const int H,
    const int OW,
    const int OH,
    const int C,
    const int K,
    const int SY,
    const int SX,
    const int PY,
    const int PX,
    const int x_off,
    const int y_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)OW * OH * C) return;
    const int c = (int)(idx % C);
    const nf_i64 rest = idx / C;
    const int ox = (int)(rest % OW);
    const int oy = (int)(rest / OW);
    const nf_i64 w_row = (nf_i64)c * K * K;
    float acc = 0.0f;
    for (int ky = 0; ky < K; ky++) {
        const int iy = oy * SY + ky - PY;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < K; kx++) {
            const int ix = ox * SX + kx - PX;
            if (ix < 0 || ix >= W) continue;
            acc += w[w_row + ky * K + kx] * x[x_off + ((nf_i64)iy * W + ix) * C + c];
        }
    }
    y[y_off + idx] = acc + b[c];
}

NF_KERNEL void LLM_Pool2dAvg(
    const int W,
    const int H,
    const int OW,
    const int OH,
    const int C,
    const int KY,
    const int KX,
    const int x_off,
    const int y_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)OW * OH * C) return;
    const int c = (int)(idx % C);
    const nf_i64 rest = idx / C;
    const int ox = (int)(rest % OW);
    const int oy = (int)(rest / OW);
    float acc = 0.0f;
    for (int dy = 0; dy < KY; dy++) {
        for (int dx = 0; dx < KX; dx++) {
            acc += x[x_off + ((nf_i64)(oy * KY + dy) * W + ox * KX + dx) * C + c];
        }
    }
    y[y_off + idx] = acc / (float)(KY * KX);
}

NF_KERNEL void LLM_RelPosAttn(
    const int groups,
    const int side,
    const int n_head,
    const int dh,
    const float scale,
    const int q_stride,
    const int kv_stride,
    const int out_stride,
    const int q_off,
    const int kv_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT q,
    NF_GLOBAL const float* NF_RESTRICT k,
    NF_GLOBAL const float* NF_RESTRICT v,
    NF_GLOBAL const float* NF_RESTRICT rw,
    NF_GLOBAL const float* NF_RESTRICT rh,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int T = side * side;
    /* The early return takes whole workgroups, so every reached barrier stays uniform. */
    if (get_group_id(0) >= (uint)(groups * n_head * T)) return;
    const int lid = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);
    NF_LOCAL float probs[NF_VISION_T_MAX];
    NF_LOCAL float partial[LLM_DIMX];

    const int gid = (int)get_group_id(0);
    const int g = gid / (n_head * T);
    const int h = gid / T % n_head;
    const int tq = gid % T;
    const int qx = tq % side;
    const int qy = tq / side;
    const nf_i64 q_row = (nf_i64)q_off + (nf_i64)(g * T + tq) * q_stride + (nf_i64)h * dh;

    float my_max = -NF_INF;
    for (int tk = lid; tk < T; tk += lsize) {
        const int kx = tk % side;
        const int ky = tk / side;
        const nf_i64 k_row = (nf_i64)kv_off + (nf_i64)(g * T + tk) * kv_stride + (nf_i64)h * dh;
        float dot_kv = 0.0f;
        float dot_w = 0.0f;
        float dot_h = 0.0f;
        for (int d = 0; d < dh; d++) {
            const float qv = q[q_row + d];
            dot_kv += qv * k[k_row + d];
            dot_w += qv * rw[(nf_i64)(qx - kx + side - 1) * dh + d];
            dot_h += qv * rh[(nf_i64)(qy - ky + side - 1) * dh + d];
        }
        const float s = dot_kv * scale + dot_w + dot_h;
        probs[tk] = s;
        my_max = fmax(my_max, s);
    }
    const float row_max = nf_wg_max(partial, my_max);

    float my_sum = 0.0f;
    for (int tk = lid; tk < T; tk += lsize) {
        const float e = exp(probs[tk] - row_max);
        probs[tk] = e;
        my_sum += e;
    }
    const float inv_sum = 1.0f / nf_wg_sum(partial, my_sum);

    for (int tk = lid; tk < T; tk += lsize) {
        probs[tk] *= inv_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d = lid; d < dh; d += lsize) {
        float acc = 0.0f;
        for (int tk = 0; tk < T; tk++) {
            acc += probs[tk] * v[(nf_i64)kv_off + (nf_i64)(g * T + tk) * kv_stride + (nf_i64)h * dh + d];
        }
        out[(nf_i64)out_off + (nf_i64)(g * T + tq) * out_stride + (nf_i64)h * dh + d] = acc;
    }
}
