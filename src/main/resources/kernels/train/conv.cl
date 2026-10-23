/* conv.cl - direct and transposed 2-D convolution, the convolution weight gradient, and the im2col/col2im
 * column transforms of the GEMM convolution path. Requires: common/dialect.h, common/train.h */
/*
 * All tensors are contiguous NCHW FLOAT32; offsets are element offsets into the buffers. The forward kernels
 * use one work-item per output element out[b, oc, oh, ow] (row-major linear index i, ow fastest); the weight
 * gradient uses one work-group per (weight element, chunk). Every result is deterministic and independent of
 * the launch. Groups split the channels: Cg = C / groups input channels and OCg = OC / groups output channels
 * per group, output channel oc belongs to group g = oc / OCg.
 *
 * TR_Conv2d_F32(i B, i C, i H, i W, i OC, i OH, i OW, i KH, i KW, i SH, i SW, i PH, i PW, i DH, i DW,
 *               i groups, i has_bias, i in_off, i w_off, i b_off, i out_off,
 *               R in[B,C,H,W], R w[OC,Cg,KH,KW], R bias[OC], W out[B,OC,OH,OW])
 *     acc = has_bias ? bias[oc] : 0
 *     for ic < Cg, kh < KH, kw < KW (ascending, ic outermost):
 *         ih = oh*SH + kh*DH - PH, iw = ow*SW + kw*DW - PW
 *         if 0 <= ih < H and 0 <= iw < W: acc = fma(in[b, g*Cg + ic, ih, iw], w[oc, ic, kh, kw], acc)
 *     out = acc
 *     The CPU reference is nn/layers/Conv2d.java forward (loop order b, oc, oh, ow, ic, kh, kw; the sum starts
 *     from the bias and accumulates in the same order). OH and OW are the caller's, computed as the layer does.
 *
 * TR_ConvTranspose2d_F32(the same 21 scalars, R in[B,C,H,W], R w[C,OCg,KH,KW], R bias[OC], W out[B,OC,OH,OW])
 *     The gather form of the transposed convolution (no scatter, no atomics). With ocg = oc % OCg:
 *     acc = has_bias ? bias[oc] : 0
 *     for ic < Cg, kh < KH, kw < KW (ascending, ic outermost), with cin = g*Cg + ic:
 *         th = oh + PH - kh*DH, tw = ow + PW - kw*DW
 *         if th, tw >= 0, th % SH == 0, tw % SW == 0, ih = th / SH < H and iw = tw / SW < W:
 *             acc = fma(in[b, cin, ih, iw], w[cin, ocg, kh, kw], acc)
 *     out = acc
 *     The weight layout is nn/layers/ConvTranspose2d.java's [inChannels, outChannels/groups, KH, KW]. The host
 *     folds output padding into OH and OW: OH = (H-1)*SH - 2*PH + DH*(KH-1) + output_padding_h + 1; output rows
 *     no input reaches hold the bias.
 *     With has_bias = 0, in = the gy of a TR_Conv2d_F32 (C, H, W := its OC, OH, OW), w = its weight and
 *     out = its gx (OC, OH, OW := its C, H, W), this kernel computes that convolution's input gradient: the tap
 *     test is the forward relation (input row) = (output row)*SH + kh*DH - PH, and w[cin, ocg] is the
 *     [OC, Cg, KH, KW] weight row of the forward output channel cin and input channel ocg.
 *
 * TR_Conv2dWeightGrad_F32(i B, i C, i H, i W, i OC, i OH, i OW, i KH, i KW, i SH, i SW, i PH, i PW, i DH, i DW,
 *                         i groups, i chunks, i x_off, i gy_off, i out_off,
 *                         R x[B,C,H,W], R gy[B,OC,OH,OW], W out[nW, chunks])
 *     The weight gradient of TR_Conv2d_F32. Weight element wi < nW = OC*Cg*KH*KW is (oc, ic, kh, kw) in the
 *     [OC,Cg,KH,KW] layout, wi = ((oc*Cg + ic)*KH + kh)*KW + kw, with cin = g*Cg + ic. Over the K = B*OH*OW
 *     output positions k = (b*OH + oh)*OW + ow:
 *         gw[wi] = sum over k of gy[b, oc, oh, ow] * x[b, cin, ih, iw], ih = oh*SH + kh*DH - PH,
 *                  iw = ow*SW + kw*DW - PW, in-bounds terms only (exactly the taps the forward reads)
 *     Group q < nW*chunks reduces chunk ch = q % chunks of weight wi = q / chunks: the positions
 *     [ch*span, min(K, (ch+1)*span)) with span = ceil(K/chunks). Lane l folds k = ch*span + l, + TR_LOCAL,
 *     ... in ascending order with acc = fma(gy, x, acc); tr_wg_sum combines the 256 lanes in its fixed order and
 *     lane 0 writes out[out_off + wi*chunks + ch] (an empty chunk writes 0).
 *     chunks == 1: out is gw[OC,Cg,KH,KW]. chunks > 1: out is a [nW, chunks] scratch that the host sums with
 *     TR_ReduceRowsSmall_F32 (op TR_R_SUM, rows nW, cols chunks, divisor 1) into gw.
 *     Host shape rule: chunks = (nW < 4096 && K > 32768) ? min(64, ceil(K/32768)) : 1. K == 0: the host fills
 *     gw with 0 and launches nothing.
 *     The CPU reference (tensor/CpuConvPool.java, ConvOps.conv2dWeightGrad) sums the in-bounds terms over b, oh,
 *     ow ascending; the lane-and-tree order differs from it, so this is the reduction class with K terms.
 *     The transposed convolution's weight gradient is this kernel with the operands swapped: x = the transposed
 *     convolution's gy (C, H, W = its OC, OH, OW), gy = its input (OC, OH, OW = its C, H, W). Then ih is the
 *     transposed output row reached from input row oh through tap kh (oh*SH + kh*DH - PH, the forward relation),
 *     Cg = its OCg, and out is gw in its [C, OCg, KH, KW] layout.
 *
 * TR_Im2col_F32(i NB, i C, i H, i W, i OH, i OW, i KH, i KW, i SH, i SW, i PH, i PW, i DH, i DW,
 *               i in_off, i col_off, R in[NB,C,H,W], W cols[NB, C*KH*KW, OH*OW])
 *     The column matrix of NB images of a convolution with the TR_Conv2d_F32 geometry. Row r = (c*KH + kh)*KW + kw
 *     (the weight's [Cg,KH,KW] order within each group, so group g owns rows [g*Cg*KH*KW, (g+1)*Cg*KH*KW)) and
 *     column p = oh*OW + ow of image bl hold
 *         cols[col_off + (bl*C*KH*KW + r)*OH*OW + p] = in[in_off + ((bl*C + c)*H + ih)*W + iw]
 *     with ih = oh*SH + kh*DH - PH, iw = ow*SW + kw*DW - PW when 0 <= ih < H and 0 <= iw < W, else +0 (padding).
 *     A TR_Gemm_F32 of the weight rows [OC/groups, Cg*KH*KW] of group g with the group's rows of one image is the
 *     convolution without bias; the products gy[b] x cols^T of the images, summed, are its weight gradient.
 *     Data movement only: bit-exact copies of the input and zeros.
 *
 * TR_Col2im_F32(i NB, i C, i H, i W, i OH, i OW, i KH, i KW, i SH, i SW, i PH, i PW, i DH, i DW,
 *               i has_bias, i col_off, i b_off, i out_off,
 *               R cols[NB, C*KH*KW, OH*OW], R bias[C], W out[NB,C,H,W])
 *     The adjoint of TR_Im2col_F32 in gather form (no scatter, no atomics): image element (bl, c, y, x) sums every
 *     column entry that TR_Im2col_F32 would have read from it. With the same row and column numbering:
 *         acc = has_bias ? bias[c] : 0
 *         for kh < KH, kw < KW (ascending, kh outermost):
 *             th = y + PH - kh*DH, tw = x + PW - kw*DW
 *             if th, tw >= 0, th % SH == 0, tw % SW == 0, oh = th / SH < OH and ow = tw / SW < OW:
 *                 acc = acc + cols[col_off + (bl*C*KH*KW + (c*KH + kh)*KW + kw)*OH*OW + oh*OW + ow]
 *         out[out_off + i] = acc, i = ((bl*C + c)*H + y)*W + x
 *     The tap test is TR_ConvTranspose2d_F32's. For a transposed convolution (input [B,Ct,Ht,Wt], weight
 *     [Ct,OCg,KH,KW], output [B,OC,OH,OW]) the host computes, per image and group g, the TR_Gemm_F32 product
 *     cols[(g*OCg*KH*KW + m), q] = sum over ic < Ct/groups of w[g*Ct/groups + ic, m] * in[b, g*Ct/groups + ic, q]
 *     (m = (ocg*KH + kh)*KW + kw, q = ih*Wt + iw); with C, H, W := OC, OH, OW and OH, OW := Ht, Wt this kernel
 *     then yields the transposed convolution, which is also a convolution's input gradient. The terms are summed
 *     per tap over the input channels by the GEMM, then over the taps here.
 *
 * Preconditions (host): C, H, W, KH, KW, groups >= 1 and a launch over a zero-size output is never enqueued
 * (tr_gid rejects every item before any division when B*OC*OH*OW is 0; the weight gradient returns per group
 * before any use of the extents when nW*chunks is 0); SH, SW, DH, DW >= 1; PH, PW >= 0; C and OC divisible by
 * groups; has_bias is 0 or 1 and bias is the placeholder when 0; chunks >= 1 and nW*chunks <= 2^31-1;
 * B*C*H*W, the weight element count and B*OC*OH*OW each <= 2^31-1, and oh*SH + kh*DH, oh + PH (and the W
 * twins) fit in int. Inputs and outputs never alias. TR_Im2col_F32 and TR_Col2im_F32: NB, OH, OW >= 1,
 * NB*C*KH*KW*OH*OW <= 2^31-1 (the column buffer), y + PH and kh*DH fit in int, has_bias is 0 or 1 and bias is
 * the placeholder when 0.
 *
 * Dispatch: 1-D, local TR_LOCAL. TR_Conv2d_F32, TR_ConvTranspose2d_F32: B*OC*OH*OW items, no barriers, no
 * local memory. TR_Conv2dWeightGrad_F32: nW*chunks groups; a group past the last returns as a whole, so the
 * barriers of tr_wg_sum stay in uniform control flow. TR_Im2col_F32: NB*C*KH*KW*OH*OW items (one per column
 * entry); TR_Col2im_F32: NB*C*H*W items (one per image element); neither uses barriers or local memory.
 */

NF_KERNEL void TR_Conv2d_F32(
    const int B, const int C, const int H, const int W,
    const int OC, const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int DH, const int DW,
    const int groups,
    const int has_bias,
    const int in_off,
    const int w_off,
    const int b_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = tr_gid(B * OC * OH * OW);
    if (i < 0) return;
    const int ow = i % OW;
    const int oh = (i / OW) % OH;
    const int oc = (i / (OW * OH)) % OC;
    const int b = i / (OW * OH * OC);
    const int Cg = C / groups;
    const int g = oc / (OC / groups);
    const int ih0 = oh * SH - PH;
    const int iw0 = ow * SW - PW;
    float acc = has_bias != 0 ? bias[b_off + oc] : 0.0f;
    for (int ic = 0; ic < Cg; ++ic) {
        const int in_c = in_off + (b * C + g * Cg + ic) * H * W;
        const int w_c = w_off + (oc * Cg + ic) * KH * KW;
        for (int kh = 0; kh < KH; ++kh) {
            const int ih = ih0 + kh * DH;
            if (ih < 0 || ih >= H) continue;
            const int in_row = in_c + ih * W;
            const int w_row = w_c + kh * KW;
            for (int kw = 0; kw < KW; ++kw) {
                const int iw = iw0 + kw * DW;
                if (iw < 0 || iw >= W) continue;
                acc = fma(in[in_row + iw], w[w_row + kw], acc);
            }
        }
    }
    out[out_off + i] = acc;
}

NF_KERNEL void TR_ConvTranspose2d_F32(
    const int B, const int C, const int H, const int W,
    const int OC, const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int DH, const int DW,
    const int groups,
    const int has_bias,
    const int in_off,
    const int w_off,
    const int b_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = tr_gid(B * OC * OH * OW);
    if (i < 0) return;
    const int ow = i % OW;
    const int oh = (i / OW) % OH;
    const int oc = (i / (OW * OH)) % OC;
    const int b = i / (OW * OH * OC);
    const int Cg = C / groups;
    const int OCg = OC / groups;
    const int g = oc / OCg;
    const int ocg = oc - g * OCg;
    const int th0 = oh + PH;
    const int tw0 = ow + PW;
    float acc = has_bias != 0 ? bias[b_off + oc] : 0.0f;
    for (int ic = 0; ic < Cg; ++ic) {
        const int cin = g * Cg + ic;
        const int in_c = in_off + (b * C + cin) * H * W;
        const int w_c = w_off + (cin * OCg + ocg) * KH * KW;
        for (int kh = 0; kh < KH; ++kh) {
            const int th = th0 - kh * DH;
            if (th < 0 || th % SH != 0) continue;
            const int ih = th / SH;
            if (ih >= H) continue;
            const int in_row = in_c + ih * W;
            const int w_row = w_c + kh * KW;
            for (int kw = 0; kw < KW; ++kw) {
                const int tw = tw0 - kw * DW;
                if (tw < 0 || tw % SW != 0) continue;
                const int iw = tw / SW;
                if (iw >= W) continue;
                acc = fma(in[in_row + iw], w[w_row + kw], acc);
            }
        }
    }
    out[out_off + i] = acc;
}

NF_KERNEL void TR_Conv2dWeightGrad_F32(
    const int B, const int C, const int H, const int W,
    const int OC, const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int DH, const int DW,
    const int groups,
    const int chunks,
    const int x_off,
    const int gy_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float buf[TR_LOCAL];
    const uint q = get_group_id(0);
    if (chunks <= 0 || groups <= 0) return;
    const int Cg = C / groups;
    const uint nW = (uint)(OC * Cg * KH * KW);
    if (q >= nW * (uint)chunks) return;
    const int wi = (int)(q / (uint)chunks);
    const uint ch = q % (uint)chunks;
    const int kw = wi % KW;
    const int kh = (wi / KW) % KH;
    const int ic = (wi / (KW * KH)) % Cg;
    const int oc = wi / (KW * KH * Cg);
    const int cin = (oc / (OC / groups)) * Cg + ic;
    const int dh = kh * DH - PH;
    const int dw = kw * DW - PW;
    const uint K = (uint)(B * OH * OW);
    const uint span = K / (uint)chunks + (K % (uint)chunks != 0u ? 1u : 0u);
    const uint k0 = ch * span;
    const uint k1 = k0 + span < K ? k0 + span : K;
    const uint lid = get_local_id(0);
    float acc = 0.0f;
    for (uint k = k0 + lid; k < k1; k += TR_LOCAL) {
        const int ki = (int)k;
        const int ow = ki % OW;
        const int oh = (ki / OW) % OH;
        const int b = ki / (OW * OH);
        const int ih = oh * SH + dh;
        const int iw = ow * SW + dw;
        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
        acc = fma(gy[gy_off + ((b * OC + oc) * OH + oh) * OW + ow], x[x_off + ((b * C + cin) * H + ih) * W + iw], acc);
    }
    const float r = tr_wg_sum(buf, acc);
    if (lid == 0u) out[out_off + wi * chunks + (int)ch] = r;
}

NF_KERNEL void TR_Im2col_F32(
    const int NB, const int C, const int H, const int W,
    const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int DH, const int DW,
    const int in_off,
    const int col_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL float* NF_RESTRICT cols)
{
    const int P = OH * OW;
    const int CK = C * KH * KW;
    const int i = tr_gid(NB * CK * P);
    if (i < 0) return;
    const int p = i % P;
    const int r = (i / P) % CK;
    const int bl = i / (P * CK);
    const int ow = p % OW;
    const int oh = p / OW;
    const int kw = r % KW;
    const int kh = (r / KW) % KH;
    const int c = r / (KW * KH);
    const int ih = oh * SH + kh * DH - PH;
    const int iw = ow * SW + kw * DW - PW;
    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        v = in[in_off + ((bl * C + c) * H + ih) * W + iw];
    }
    cols[col_off + i] = v;
}

NF_KERNEL void TR_Col2im_F32(
    const int NB, const int C, const int H, const int W,
    const int OH, const int OW,
    const int KH, const int KW,
    const int SH, const int SW,
    const int PH, const int PW,
    const int DH, const int DW,
    const int has_bias,
    const int col_off,
    const int b_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT cols,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = tr_gid(NB * C * H * W);
    if (i < 0) return;
    const int x = i % W;
    const int y = (i / W) % H;
    const int c = (i / (W * H)) % C;
    const int bl = i / (W * H * C);
    const int P = OH * OW;
    const int th0 = y + PH;
    const int tw0 = x + PW;
    const int col_c = col_off + (bl * C + c) * KH * KW * P;
    float acc = has_bias != 0 ? bias[b_off + c] : 0.0f;
    for (int kh = 0; kh < KH; ++kh) {
        const int th = th0 - kh * DH;
        if (th < 0 || th % SH != 0) continue;
        const int oh = th / SH;
        if (oh >= OH) continue;
        const int row = col_c + kh * KW * P + oh * OW;
        for (int kw = 0; kw < KW; ++kw) {
            const int tw = tw0 - kw * DW;
            if (tw < 0 || tw % SW != 0) continue;
            const int ow = tw / SW;
            if (ow >= OW) continue;
            acc = acc + cols[row + kw * P + ow];
        }
    }
    out[out_off + i] = acc;
}
