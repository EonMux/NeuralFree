/* common/matmul_cm2_tile.glsl - one output tile of matmul_mmq_cm2.comp at a token width of NF_CM2_TW,
 * included once per width (the shader defines NF_CM2_TW and NF_CM2_TILE_FN before each include):
 *     void NF_CM2_TILE_FN(uint m0, uint n0, uint kb, uint ke, uint pos_d, uint stride_d, bool interior)
 * accumulates rows m0 .. m0+NF_CM2_BM of tokens n0 .. n0+NF_CM2_TW over k in kb .. ke and stores the tile.
 * An interior tile (every row below M, k range a whole number of NF_CM2_BK stages) loads through the
 * unclamped layouts in unrolled groups of NF_CM2_UNROLL_K reduction elements; any other tile loads
 * through the ConstantNV clamped layouts one stage at a time. The activation rows are padded with zero
 * rows to a whole number of NF_CM2_BN tokens, so the activation load never needs a clamp; the store
 * clamps to N tokens and M rows. A quantized weight tile decodes through mmq_ref_decode and accumulates in
 * fp16, saturated at +-NF_CM2_ACC_MAX before the fp32 store, so an overflowing sum stores the largest finite
 * fp16 value instead of an infinity. A dense tile (NF_CM2_DENSE) loads its halves directly and accumulates in
 * fp32 (NF_CM2_ACC_T), which stores every finite sum unclipped. Every stage first calls
 * nf_cm2_scales, which publishes the staged scales of a new K block (Q4_K / Q5_K; nothing for the other types);
 * an interior tile's unrolled groups start on block boundaries, as the shader takes the interior path only for
 * a k range that starts on one.
 * Requires: the declarations of matmul_mmq_cm2.comp (bindings, push block, NF_CM2_* sizes,
 * mmq_ref_decode or NF_CM2_DENSE, nf_cm2_scales_begin, nf_cm2_scales).
 */
#ifndef NF_CM2_LOAD_A
#ifdef NF_CM2_DENSE
#define NF_CM2_LOAD_A(ma, slice) coopMatLoadTensorNV(ma, w_f16, 0u, slice)
#define NF_CM2_A_BLOCKS(layout) layout
#define NF_CM2_ACC_T float
#else
#define NF_CM2_LOAD_A(ma, slice) coopMatLoadTensorNV(ma, w_u8, 0u, slice, mmq_ref_decode)
#define NF_CM2_A_BLOCKS(layout) setTensorLayoutBlockSizeNV(layout, 1u, uint(MMQ_BS))
#define NF_CM2_ACC_T float16_t
#endif
#endif
void NF_CM2_TILE_FN(const uint m0, const uint n0, const uint kb, const uint ke, const uint pos_d,
                    const uint stride_d, const bool interior) {
    const uint M = uint(pc.M);
    const uint N = uint(pc.N);
    const uint padded_n = ((N + NF_CM2_BN - 1u) / NF_CM2_BN) * NF_CM2_BN;
    const uint stride_b = uint(pc.x_stride) & ~7u;
    tensorViewNV<2, false, 1, 0> transpose = createTensorViewNV(2, false, 1, 0);
    tensorLayoutNV<2> lb = createTensorLayoutNV(2);
    lb = setTensorLayoutDimensionNV(lb, padded_n, ke);
    lb = setTensorLayoutStrideNV(lb, stride_b, 1u);
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> ld =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    ld = setTensorLayoutDimensionNV(ld, N, M);
    ld = setTensorLayoutStrideNV(ld, stride_d, 1u);

    coopmat<NF_CM2_ACC_T, gl_ScopeWorkgroup, NF_CM2_BM, NF_CM2_TW, gl_MatrixUseAccumulator> sum =
        coopmat<NF_CM2_ACC_T, gl_ScopeWorkgroup, NF_CM2_BM, NF_CM2_TW, gl_MatrixUseAccumulator>(NF_CM2_ACC_T(0.0));
    uint k0 = kb;
    nf_cm2_scales_begin(m0, kb);
    if (interior) {
        tensorLayoutNV<2> la = createTensorLayoutNV(2);
        la = NF_CM2_A_BLOCKS(la);
        la = setTensorLayoutDimensionNV(la, M, ke);
        la = setTensorLayoutStrideNV(la, uint(pc.K) / uint(MMQ_BS), 1u);
        const uint unrolled_end = kb + ((ke - kb) / NF_CM2_UNROLL_K) * NF_CM2_UNROLL_K;
        while (k0 < unrolled_end) {
            nf_cm2_scales(m0, k0, kb, ke);
            [[unroll]] for (uint j = 0u; j < NF_CM2_UNROLL_K / NF_CM2_BK; j++) {
                coopmat<float16_t, gl_ScopeWorkgroup, NF_CM2_BM, NF_CM2_BK, gl_MatrixUseA> ma;
                coopmat<float16_t, gl_ScopeWorkgroup, NF_CM2_BK, NF_CM2_TW, gl_MatrixUseB> mb;
                NF_CM2_LOAD_A(ma, sliceTensorLayoutNV(la, m0, NF_CM2_BM, k0, NF_CM2_BK));
                coopMatLoadTensorNV(mb, x_in, uint(pc.x_off), sliceTensorLayoutNV(lb, n0, NF_CM2_TW, k0, NF_CM2_BK), transpose);
                sum = coopMatMulAdd(ma, mb, sum);
                k0 += NF_CM2_BK;
            }
        }
        while (k0 < ke) {
            nf_cm2_scales(m0, k0, kb, ke);
            coopmat<float16_t, gl_ScopeWorkgroup, NF_CM2_BM, NF_CM2_BK, gl_MatrixUseA> ma;
            coopmat<float16_t, gl_ScopeWorkgroup, NF_CM2_BK, NF_CM2_TW, gl_MatrixUseB> mb;
            NF_CM2_LOAD_A(ma, sliceTensorLayoutNV(la, m0, NF_CM2_BM, k0, NF_CM2_BK));
            coopMatLoadTensorNV(mb, x_in, uint(pc.x_off), sliceTensorLayoutNV(lb, n0, NF_CM2_TW, k0, NF_CM2_BK), transpose);
            sum = coopMatMulAdd(ma, mb, sum);
            k0 += NF_CM2_BK;
        }
    } else {
        tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> la =
            createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
        la = NF_CM2_A_BLOCKS(la);
        la = setTensorLayoutDimensionNV(la, M, ke);
        la = setTensorLayoutStrideNV(la, uint(pc.K) / uint(MMQ_BS), 1u);
        tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> lbc =
            createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
        lbc = setTensorLayoutDimensionNV(lbc, padded_n, ke);
        lbc = setTensorLayoutStrideNV(lbc, stride_b, 1u);
        while (k0 < ke) {
            nf_cm2_scales(m0, k0, kb, ke);
            coopmat<float16_t, gl_ScopeWorkgroup, NF_CM2_BM, NF_CM2_BK, gl_MatrixUseA> ma;
            coopmat<float16_t, gl_ScopeWorkgroup, NF_CM2_BK, NF_CM2_TW, gl_MatrixUseB> mb;
            NF_CM2_LOAD_A(ma, sliceTensorLayoutNV(la, m0, NF_CM2_BM, k0, NF_CM2_BK));
            coopMatLoadTensorNV(mb, x_in, uint(pc.x_off), sliceTensorLayoutNV(lbc, n0, NF_CM2_TW, k0, NF_CM2_BK), transpose);
            sum = coopMatMulAdd(ma, mb, sum);
            k0 += NF_CM2_BK;
        }
    }
#ifdef NF_CM2_DENSE
    coopMatStoreTensorNV(sum, NF_CM2_OUT, pos_d, sliceTensorLayoutNV(ld, n0, NF_CM2_TW, m0, NF_CM2_BM), transpose);
#else
    [[unroll]] for (uint i = 0u; i < sum.length(); ++i) {
        sum[i] = clamp(sum[i], -NF_CM2_ACC_MAX, NF_CM2_ACC_MAX);
    }
    const coopmat<float, gl_ScopeWorkgroup, NF_CM2_BM, NF_CM2_TW, gl_MatrixUseAccumulator> res =
        coopmat<float, gl_ScopeWorkgroup, NF_CM2_BM, NF_CM2_TW, gl_MatrixUseAccumulator>(sum);
    coopMatStoreTensorNV(res, NF_CM2_OUT, pos_d, sliceTensorLayoutNV(ld, n0, NF_CM2_TW, m0, NF_CM2_BM), transpose);
#endif
}
