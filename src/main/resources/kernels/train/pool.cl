/* pool.cl - max, average and adaptive average 2-D pooling and their backward. Requires: common/dialect.h, common/train.h */
/*
 * All tensors are contiguous NCHW FLOAT32; offsets are element offsets into the buffers. Every kernel uses one
 * work-item per element it writes (row-major linear index i, last dim fastest) and fixed sequential loops, so
 * every result is deterministic and matches the CPU loop order. The windows follow PyTorch (max_pool2d,
 * avg_pool2d, adaptive_avg_pool2d). OH and OW are the caller's; for the max and average kernels they come from
 * tensor/ConvOps.poolOutputSize (PyTorch pooling_output_shape: floor or ceil mode, where ceil mode drops a last
 * window that would start inside the right padding, so every window starts inside the input or the left
 * padding).
 *
 * Forward (one work-item per output element out[b, c, oh, ow]; window loops in ascending row, then column order):
 *
 * TR_MaxPool2d_F32(i B, i C, i H, i W, i OH, i OW, i KH, i KW, i SH, i SW, i PH, i PW, i DH, i DW,
 *                  i in_off, i out_off, R in[B,C,H,W], W out[B,C,OH,OW])
 *     m = -inf
 *     for kh < KH, kw < KW: ih = oh*SH + kh*DH - PH, iw = ow*SW + kw*DW - PW
 *         if 0 <= ih < H and 0 <= iw < W: m = tr_jmax(m, in[b, c, ih, iw])
 *     out = m (-inf when the window holds no input element)
 *     nn/layers/MaxPool2d.java forward: maxVal = Math.max(maxVal, x) from NEGATIVE_INFINITY. tr_jmax ranks +0
 *     above -0 and keeps the earlier operand on equal values, so the first maximum in kh, kw order is the one
 *     returned, and the first NaN in the window propagates. Bit-identical to the CPU.
 *
 * TR_AvgPool2d_F32(i B, i C, i H, i W, i OH, i OW, i KH, i KW, i SH, i SW, i PH, i PW, i count_include_pad,
 *                  i in_off, i out_off, R in[B,C,H,W], W out[B,C,OH,OW])
 *     No dilation. The window is clipped to the padded extent, then to the input (PyTorch avg_pool2d):
 *     hs = oh*SH - PH, he = min(hs + KH, H + PH), h0 = max(hs, 0), h1 = min(he, H); columns likewise.
 *     divisor = count_include_pad != 0 ? (he - hs)*(we - ws) : (h1 - h0)*(w1 - w0)   (tr_p_avg_divisor)
 *     sum = 0; for h0 <= ih < h1, w0 <= iw < w1 (row-major): sum = sum + in[b, c, ih, iw]
 *     out = (h0 < h1 and w0 < w1) ? sum / (float)divisor : 0
 *     CPU reference: tensor/CpuConvPool.java avgPool2d, the same sum order and one division.
 *
 * TR_AdaptiveAvgPool2d_F32(i B, i C, i H, i W, i OH, i OW, i in_off, i out_off, R in[B,C,H,W], W out[B,C,OH,OW])
 *     PyTorch windows: h0 = (oh*H) / OH (floor), h1 = ((oh+1)*H + OH - 1) / OH (ceil), w0, w1 likewise. When OH
 *     does not divide H neighbouring windows overlap, and every window holds at least one row when H >= 1.
 *     sum = 0, count = 0; for h0 <= ih < h1, w0 <= iw < w1: sum = sum + in[b, c, ih, iw], count = count + 1
 *     out = count > 0 ? sum / (float)count : 0
 *     CPU reference: tensor/CpuConvPool.java adaptiveAvgPool2d (windows from ConvOps.adaptiveStart/adaptiveEnd).
 *
 * Backward (gather form: one work-item per input-gradient element gx[b, c, ih, iw], so overlapping windows
 * are summed by one owner in a fixed order, without scatter or atomics). For max and average pooling the
 * windows covering input row ih are enumerated by kh = KH-1 down to 0: th = ih + PH - kh*DH, skipped when
 * th < 0 or th % SH != 0, oh = th / SH, skipped when oh >= OH. Distinct kh give distinct oh and descending kh
 * gives ascending oh; columns likewise (kw descending, ow ascending, inner loop). Each covering window
 * (oh, ow) holds (ih, iw) at exactly one tap (kh, kw), and the windows are visited in ascending (oh, ow)
 * order, the order in which the CPU reference (tensor/CpuConvPool.java) loops over the windows and adds into
 * gx. acc starts at 0.
 *
 * TR_MaxPool2dBackward_F32(i B, i C, i H, i W, i OH, i OW, i KH, i KW, i SH, i SW, i PH, i PW, i DH, i DW,
 *                          i x_off, i y_off, i gy_off, i gx_off,
 *                          R x[B,C,H,W], R y[B,C,OH,OW], R gy[B,C,OH,OW], W gx[B,C,H,W])
 *     y is the saved TR_MaxPool2d_F32 output of x. The gradient of window (oh, ow) goes to its first in-bounds
 *     tap in ascending kh, kw order whose value is Java == y[b, c, oh, ow] (tr_p_same; -0 == +0); a NaN max
 *     routes nowhere, and an empty window (y = -inf, no tap) covers no input element.
 *     for each covering window (oh, ow) at tap (kh, kw):
 *         if tr_p_same(x[b, c, ih, iw], y[b, c, oh, ow]) and tr_p_first_max(window) == kh*KW + kw:
 *             acc = acc + gy[b, c, oh, ow]
 *     gx = acc
 *     CPU reference: gx = 0; for b, c, oh, ow ascending: m = y; skip NaN m; for kh, kw ascending in-bounds:
 *     if x == m { gx[b, c, ih, iw] += gy[b, c, oh, ow]; break }. The same adds in the same order from 0:
 *     single-rounding class (bit-identical for normal values).
 *
 * TR_AvgPool2dBackward_F32(i B, i C, i H, i W, i OH, i OW, i KH, i KW, i SH, i SW, i PH, i PW,
 *                          i count_include_pad, i gy_off, i gx_off, R gy[B,C,OH,OW], W gx[B,C,H,W])
 *     DH = DW = 1 (TR_AvgPool2d_F32 has no dilation). The forward's divisor of each covering window (oh, ow) is
 *     recomputed exactly with tr_p_avg_divisor (hs = oh*SH - PH, he = min(hs + KH, H + PH), columns likewise);
 *     a window that covers the in-bounds element (ih, iw) holds input elements, so the forward divided it.
 *     Every tap of the covering window lies below he, because ih < H <= H + PH.
 *     for each covering window (oh, ow): acc = acc + gy[b, c, oh, ow] / (float)divisor(oh, ow)
 *     gx = acc
 *     CPU reference: for b, c, oh, ow ascending: skip a window without input elements; for its in-bounds
 *     elements: gx[b, c, ih, iw] += gy[b, c, oh, ow] / divisor. Reduction class with
 *     K = ceil(KH/SH)*ceil(KW/SW) terms gy/divisor (one division per term; division class when the windows do
 *     not overlap).
 *
 * TR_AdaptiveAvgPool2dBackward_F32(i B, i C, i H, i W, i OH, i OW, i gy_off, i gx_off,
 *                                  R gy[B,C,OH,OW], W gx[B,C,H,W])
 *     The PyTorch windows of TR_AdaptiveAvgPool2d_F32 may overlap. The windows holding input row ih are exactly
 *     oh in [(ih*OH) / H, ((ih+1)*OH - 1) / H]: h0(oh) <= ih  <=>  oh*H <= (ih+1)*OH - 1, and
 *     h1(oh) > ih  <=>  (oh+1)*H > ih*OH  <=>  oh >= (ih*OH) / H. Columns likewise.
 *     acc = 0; for oh ascending, then ow ascending, over those windows:
 *         acc = acc + gy[b, c, oh, ow] / (float)((h1 - h0)*(w1 - w0))   (the forward count, >= 1)
 *     gx = acc
 *     CPU reference: for b, c, oh, ow ascending, gx[ih, iw] += gy / count for each (ih, iw) of the window. The
 *     same adds in the same order from 0. Reduction class with at most K = (floor((OH-1)/H) + 2) *
 *     (floor((OW-1)/W) + 2) terms gy/count; division class when every element lies in one window (OH divides H
 *     and OW divides W).
 *
 * TR_MaxPool2dSelect_F32(i B, i C, i H, i W, i OH, i OW, i KH, i KW, i SH, i SW, i PH, i PW, i DH, i DW,
 *                        i x_off, i y_off, i g_off, i out_off,
 *                        R x[B,C,H,W], R y[B,C,OH,OW], R g[B,C,H,W], W out[B,C,OH,OW])
 *     One work-item per window (oh, ow): out[b, c, oh, ow] = g[b, c, ih, iw] at the tap that
 *     TR_MaxPool2dBackward_F32 routes the window's gradient to (tr_p_first_max), +0 when it routes nowhere.
 *     The adjoint of TR_MaxPool2dBackward_F32 in gy (the create-graph derivative). CPU reference: the same
 *     first-match scan per window. Bit-identical.
 *
 * Preconditions (host): H, W, KH, KW >= 1, OH, OW >= 0 (>= 1 for the adaptive kernels) and a launch over a
 * zero-size result is never enqueued (tr_gid rejects every item before any division when its item count is
 * 0); strides and dilations >= 1; 0 <= PH <= (DH*(KH-1)+1)/2 and its W twin; B*C*H*W and B*C*OH*OW each
 * <= 2^31-1; oh*SH + kh*DH and ih + PH (and their W twins) fit in int; KH*KW fits in int; for the average
 * kernels the window end oh*SH + KH and the padded extent H + PH (and their W twins) fit in int; for the
 * adaptive kernels OH*H + OH - 1 and OW*W + OW - 1 fit in int. Inputs and outputs never alias.
 *
 * Dispatch: 1-D, local TR_LOCAL, no barriers, no local memory. Forward and TR_MaxPool2dSelect_F32:
 * B*C*OH*OW items; the other backward kernels: B*C*H*W items.
 */

/* Java a == b for floats: false when either is NaN, true for -0 == +0, otherwise bit equality, which is
 * exact value equality for non-NaN floats even on devices that flush denormals in float compares. */
NF_INLINE int tr_p_same(const float a, const float b) {
    if (tr_isnan(a) || tr_isnan(b)) return 0;
    if (tr_iszero(a) && tr_iszero(b)) return 1;
    return as_uint(a) == as_uint(b);
}

/* Tap index kh*KW + kw of the first in-bounds tap, in ascending kh, kw order, of the max window whose top-left
 * tap is (ih0, iw0) in the H x W plane of x starting at element base, whose value is Java == m; -1 when no
 * tap matches (a NaN m matches nothing). */
NF_INLINE int tr_p_first_max(NF_GLOBAL const float* x, const int base, const int H, const int W,
                             const int KH, const int KW, const int DH, const int DW,
                             const int ih0, const int iw0, const float m) {
    if (tr_isnan(m)) return -1;
    for (int kh = 0; kh < KH; ++kh) {
        const int ih = ih0 + kh * DH;
        if (ih < 0 || ih >= H) continue;
        const int row = base + ih * W;
        for (int kw = 0; kw < KW; ++kw) {
            const int iw = iw0 + kw * DW;
            if (iw < 0 || iw >= W) continue;
            if (tr_p_same(x[row + iw], m)) return kh * KW + kw;
        }
    }
    return -1;
}

/* Divisor of the average-pool window with padded-extent bounds [hs, he) x [ws, we) and in-bounds bounds
 * [h0, h1) x [w0, w1): the padded-extent size when count_include_pad != 0, otherwise the in-bounds count. */
NF_INLINE int tr_p_avg_divisor(const int count_include_pad, const int hs, const int he, const int ws,
                               const int we, const int h0, const int h1, const int w0, const int w1) {
    return count_include_pad != 0 ? (he - hs) * (we - ws) : (h1 - h0) * (w1 - w0);
}

NF_KERNEL void TR_MaxPool2d_F32(
    const int B, const int C, const int H, const int W,
    const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int DH, const int DW,
    const int in_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = tr_gid(B * C * OH * OW);
    if (i < 0) return;
    const int ow = i % OW;
    const int oh = (i / OW) % OH;
    const int bc = i / (OW * OH);
    const int in_c = in_off + bc * H * W;
    const int ih0 = oh * SH - PH;
    const int iw0 = ow * SW - PW;
    float m = -NF_INF;
    for (int kh = 0; kh < KH; ++kh) {
        const int ih = ih0 + kh * DH;
        if (ih < 0 || ih >= H) continue;
        const int in_row = in_c + ih * W;
        for (int kw = 0; kw < KW; ++kw) {
            const int iw = iw0 + kw * DW;
            if (iw < 0 || iw >= W) continue;
            m = tr_jmax(m, in[in_row + iw]);
        }
    }
    out[out_off + i] = m;
}

NF_KERNEL void TR_AvgPool2d_F32(
    const int B, const int C, const int H, const int W,
    const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int count_include_pad,
    const int in_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = tr_gid(B * C * OH * OW);
    if (i < 0) return;
    const int ow = i % OW;
    const int oh = (i / OW) % OH;
    const int bc = i / (OW * OH);
    const int in_c = in_off + bc * H * W;
    const int hs = oh * SH - PH;
    const int ws = ow * SW - PW;
    const int he = hs + KH < H + PH ? hs + KH : H + PH;
    const int we = ws + KW < W + PW ? ws + KW : W + PW;
    const int h0 = hs > 0 ? hs : 0;
    const int w0 = ws > 0 ? ws : 0;
    const int h1 = he < H ? he : H;
    const int w1 = we < W ? we : W;
    if (h0 >= h1 || w0 >= w1) {
        out[out_off + i] = 0.0f;
        return;
    }
    float sum = 0.0f;
    for (int ih = h0; ih < h1; ++ih) {
        const int in_row = in_c + ih * W;
        for (int iw = w0; iw < w1; ++iw) {
            sum = sum + in[in_row + iw];
        }
    }
    out[out_off + i] = sum / (float)tr_p_avg_divisor(count_include_pad, hs, he, ws, we, h0, h1, w0, w1);
}

NF_KERNEL void TR_AdaptiveAvgPool2d_F32(
    const int B, const int C, const int H, const int W,
    const int OH, const int OW,
    const int in_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = tr_gid(B * C * OH * OW);
    if (i < 0) return;
    const int ow = i % OW;
    const int oh = (i / OW) % OH;
    const int bc = i / (OW * OH);
    const int in_c = in_off + bc * H * W;
    const int h0 = (oh * H) / OH;
    const int h1 = ((oh + 1) * H + OH - 1) / OH;
    const int w0 = (ow * W) / OW;
    const int w1 = ((ow + 1) * W + OW - 1) / OW;
    float sum = 0.0f;
    int count = 0;
    for (int ih = h0; ih < h1; ++ih) {
        const int in_row = in_c + ih * W;
        for (int iw = w0; iw < w1; ++iw) {
            sum = sum + in[in_row + iw];
            count = count + 1;
        }
    }
    out[out_off + i] = count > 0 ? sum / (float)count : 0.0f;
}

NF_KERNEL void TR_MaxPool2dBackward_F32(
    const int B, const int C, const int H, const int W,
    const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int DH, const int DW,
    const int x_off,
    const int y_off,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT y,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT gx)
{
    const int i = tr_gid(B * C * H * W);
    if (i < 0) return;
    const int iw = i % W;
    const int ih = (i / W) % H;
    const int bc = i / (W * H);
    const int x_c = x_off + bc * H * W;
    const int o_c = bc * OH * OW;
    const float v = x[x_off + i];
    float acc = 0.0f;
    for (int kh = KH - 1; kh >= 0; --kh) {
        const int th = ih + PH - kh * DH;
        if (th < 0 || th % SH != 0) continue;
        const int oh = th / SH;
        if (oh >= OH) continue;
        for (int kw = KW - 1; kw >= 0; --kw) {
            const int tw = iw + PW - kw * DW;
            if (tw < 0 || tw % SW != 0) continue;
            const int ow = tw / SW;
            if (ow >= OW) continue;
            const int o = o_c + oh * OW + ow;
            const float m = y[y_off + o];
            if (!tr_p_same(v, m)) continue;
            if (tr_p_first_max(x, x_c, H, W, KH, KW, DH, DW, oh * SH - PH, ow * SW - PW, m) != kh * KW + kw) {
                continue;
            }
            acc = acc + gy[gy_off + o];
        }
    }
    gx[gx_off + i] = acc;
}

NF_KERNEL void TR_AvgPool2dBackward_F32(
    const int B, const int C, const int H, const int W,
    const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int count_include_pad,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT gx)
{
    const int i = tr_gid(B * C * H * W);
    if (i < 0) return;
    const int iw = i % W;
    const int ih = (i / W) % H;
    const int bc = i / (W * H);
    const int o_c = gy_off + bc * OH * OW;
    float acc = 0.0f;
    for (int kh = KH - 1; kh >= 0; --kh) {
        const int th = ih + PH - kh;
        if (th < 0 || th % SH != 0) continue;
        const int oh = th / SH;
        if (oh >= OH) continue;
        const int hs = oh * SH - PH;
        const int he = hs + KH < H + PH ? hs + KH : H + PH;
        const int h0 = hs > 0 ? hs : 0;
        const int h1 = he < H ? he : H;
        for (int kw = KW - 1; kw >= 0; --kw) {
            const int tw = iw + PW - kw;
            if (tw < 0 || tw % SW != 0) continue;
            const int ow = tw / SW;
            if (ow >= OW) continue;
            const int ws = ow * SW - PW;
            const int we = ws + KW < W + PW ? ws + KW : W + PW;
            const int w0 = ws > 0 ? ws : 0;
            const int w1 = we < W ? we : W;
            const int divisor = tr_p_avg_divisor(count_include_pad, hs, he, ws, we, h0, h1, w0, w1);
            acc = acc + gy[o_c + oh * OW + ow] / (float)divisor;
        }
    }
    gx[gx_off + i] = acc;
}

NF_KERNEL void TR_AdaptiveAvgPool2dBackward_F32(
    const int B, const int C, const int H, const int W,
    const int OH, const int OW,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT gx)
{
    const int i = tr_gid(B * C * H * W);
    if (i < 0) return;
    const int iw = i % W;
    const int ih = (i / W) % H;
    const int bc = i / (W * H);
    const int oh_first = (ih * OH) / H;
    const int oh_last = ((ih + 1) * OH - 1) / H;
    const int ow_first = (iw * OW) / W;
    const int ow_last = ((iw + 1) * OW - 1) / W;
    const int o_c = gy_off + bc * OH * OW;
    float acc = 0.0f;
    for (int oh = oh_first; oh <= oh_last; ++oh) {
        const int rows = ((oh + 1) * H + OH - 1) / OH - (oh * H) / OH;
        for (int ow = ow_first; ow <= ow_last; ++ow) {
            const int cols = ((ow + 1) * W + OW - 1) / OW - (ow * W) / OW;
            acc = acc + gy[o_c + oh * OW + ow] / (float)(rows * cols);
        }
    }
    gx[gx_off + i] = acc;
}

NF_KERNEL void TR_MaxPool2dSelect_F32(
    const int B, const int C, const int H, const int W,
    const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int DH, const int DW,
    const int x_off,
    const int y_off,
    const int g_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT y,
    NF_GLOBAL const float* NF_RESTRICT g,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = tr_gid(B * C * OH * OW);
    if (i < 0) return;
    const int ow = i % OW;
    const int oh = (i / OW) % OH;
    const int plane = (i / (OW * OH)) * H * W;
    const int ih0 = oh * SH - PH;
    const int iw0 = ow * SW - PW;
    const int t = tr_p_first_max(x, x_off + plane, H, W, KH, KW, DH, DW, ih0, iw0, y[y_off + i]);
    float r = 0.0f;
    if (t >= 0) {
        const int kh = t / KW;
        const int kw = t - kh * KW;
        r = g[g_off + plane + (ih0 + kh * DH) * W + iw0 + kw * DW];
    }
    out[out_off + i] = r;
}
