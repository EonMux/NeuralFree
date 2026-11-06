package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.List;
import java.util.Objects;

/**
 * Names, argument layouts and launch geometry of the RoPE family kernels ({@code kernels/llm/rope.cl},
 * {@code kernels/llm/kv_cache.cl}; GLSL {@code rope.comp}, {@code kv_store.comp}, {@code kv_shift.comp}). The
 * {@code *Args} builders return arguments in kernel order after checking the preconditions the
 * kernels rely on; the {@code ARG_*} constants index the arguments re-set per step.
 *
 * <pre>
 * LLM_RoPE_Ext            (i n_tokens(dyn), i n_heads, i head_stride, i row_stride, i n_dims, i n_offs, i mode,
 *                          i pos_stride(dyn), i has_ff, f theta_scale, f freq_scale, f ext_factor, f attn_factor,
 *                          f corr_lo, f corr_hi, i sect0, i sect1, i sect2, i sect3, i x_off, R&lt;int&gt; pos, R ff, RWn x)
 * LLM_KV_Store_&lt;KV&gt;      (i n_tokens(dyn), i row_elems, i src_row_stride, i src_off, i dst_row_stride,
 *                          R src, R&lt;int&gt; cell_idx, W&lt;KV&gt; dst)
 * LLM_KV_Store_&lt;FP8 type&gt; (the same scalars, then i head_elems; R&lt;float&gt; kv_scale between cell_idx and dst:
 *                          one per-head KV scale, the code of value / scale[e / head_elems])
 * LLM_KV_RopeShift_{F32,F16,BF16,FP8_E4M3,FP8_E5M2}
 *                         (i n_cells, i n_heads, i head_stride, i row_stride, i n_dims, i n_offs, i mode, i has_ff,
 *                          f theta_scale, f freq_scale, f ext_factor, f attn_factor, f corr_lo, f corr_hi,
 *                          R&lt;int&gt; delta, R ff, RWn&lt;KV&gt; k_cache)
 * LLM_KV_RopeShift_&lt;block type&gt;
 *                         (the F32/F16/BF16 scalars, then i rot_k; the same buffers)
 * </pre>
 * {@code <KV>} is a {@link KvDType#kernelSuffix()}: F32, F16, BF16, the FP8 types or one of the block types Q8_0,
 * Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1 ({@link KvDType#quantized()}).
 */
public final class RopeKernels {

    public static final String ROPE_EXT = "LLM_RoPE_Ext";
    /** Fourier position embedding of one layer ({@code kernels/llm/rope.cl}, GLSL {@code fope_rope.comp}). */
    public static final String ROPE_FOPE = "LLM_RoPE_FoPE";
    /** Name prefix of the {@code LLM_KV_Store_<KV>} variants. */
    public static final String KV_STORE_PREFIX = "LLM_KV_Store_";
    /** Name prefix of the {@code LLM_KV_RopeShift_<KV>} variants. */
    public static final String KV_ROPE_SHIFT_PREFIX = "LLM_KV_RopeShift_";
    /**
     * The TurboQuant query rotation ({@code LLM_KvTqRotateQ}, {@code kernels/llm/kv_cache.cl}): the normalized
     * Sylvester Hadamard transform a TurboQuant key cache stores its keys against.
     */
    public static final String KV_TQ_ROTATE_Q = "LLM_KvTqRotateQ";
    /** Every {@code LLM_KV_Store_<KV>} variant, in {@link KvDType} order. */
    public static final List<String> KV_STORE_KERNELS = KvDType.kernelVariants(KV_STORE_PREFIX);
    /** Every {@code LLM_KV_RopeShift_<KV>} variant, in {@link KvDType} order. */
    public static final List<String> KV_ROPE_SHIFT_KERNELS =
            KvDType.kernelVariants(KV_ROPE_SHIFT_PREFIX, KvDType::hasShiftKernel);

    /** {@code n_tokens} of {@link #ROPE_EXT} and of the {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_N_TOKENS = 0;
    /** {@code dst} (the cache buffer) of the F32/F16/BF16/block {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_DST = 7;
    /** {@code cell_idx} of the F32/F16/BF16/block {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_CELL = 6;
    /** {@code head_elems} of the FP8 {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_HEAD_ELEMS = 5;
    /** {@code cell_idx} of the FP8 {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_FP8_CELL = 7;
    /** {@code cell_idx} of the per-token-head {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_PT_CELL = 8;
    /** {@code cell_idx} of the vendor {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_VENDOR_CELL = 9;
    /** {@code dst} (the cache buffer) of the FP8 {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_FP8_DST = 9;
    /** {@code dst} (the cache buffer) of the per-token-head {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_PT_DST = 9;
    /** {@code dst} (the packed code buffer) of the vendor {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_VENDOR_DST = 11;
    /** {@code side} (the vendor cache's side buffer) of the vendor {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_KV_STORE_VENDOR_SIDE = 12;
    /** Arguments of a vendor {@code LLM_KV_Store_*}: 8 scalars and 5 buffers. */
    public static final int KV_STORE_VENDOR_ARGS = 13;
    /** {@code pos_stride} of {@link #ROPE_EXT} (the step's token count for multi-section positions). */
    public static final int ARG_POS_STRIDE = 7;

    /** Arguments of {@link #ROPE_EXT}: 20 scalars and 3 buffers. */
    public static final int ROPE_EXT_ARGS = 23;
    /** Arguments of {@link #ROPE_FOPE}: 9 scalars and 5 buffers. */
    public static final int FOPE_ARGS = 14;
    /** Scalar indices of {@link #ROPE_FOPE} in signature order. */
    public static final int ARG_FOPE_N_TOKENS = 0;
    public static final int ARG_FOPE_N_HEADS = 1;
    public static final int ARG_FOPE_HEAD_STRIDE = 2;
    public static final int ARG_FOPE_ROW_STRIDE = 3;
    public static final int ARG_FOPE_N_DIMS = 4;
    public static final int ARG_FOPE_X_OFF = 5;
    public static final int ARG_FOPE_N_INV = 6;
    public static final int ARG_FOPE_N_COEF_ROWS = 7;
    public static final int ARG_FOPE_SEP_HEAD = 8;
    /** Buffer indices of {@link #ROPE_FOPE} in signature order. */
    public static final int ARG_FOPE_POS = 9;
    public static final int ARG_FOPE_COS = 10;
    public static final int ARG_FOPE_SIN = 11;
    public static final int ARG_FOPE_INV = 12;
    public static final int ARG_FOPE_X = 13;
    /** Arguments of the F32/F16/BF16/block {@code LLM_KV_Store_*}: 5 scalars and 3 buffers. */
    public static final int KV_STORE_ARGS = 8;
    /** Arguments of the FP8 {@code LLM_KV_Store_*}: 6 scalars (head_elems last) and 4 buffers (kv_scale before dst). */
    public static final int KV_STORE_FP8_ARGS = 10;
    /** Arguments of a per-token-head {@code LLM_KV_Store_*}: 7 scalars and 4 buffers (dst then its scales). */
    public static final int KV_STORE_PT_ARGS = 11;
    /** Arguments of {@code LLM_KV_RopeShift_F32/F16/BF16/FP8}: 14 scalars and 3 buffers; block types add {@code rot_k}. */
    public static final int KV_ROPE_SHIFT_ARGS = 17;

    /** Largest work window of one head held by a block-type rope shift; {@code NF_ROPE_SHIFT_MAX_DIMS}. */
    public static final int SHIFT_MAX_DIMS = 1024;

    private RopeKernels() {
    }

    /** {@code LLM_KV_Store_<type>}. */
    public static String kvStore(KvDType type) {
        return Objects.requireNonNull(type, "type").kernel(KV_STORE_PREFIX);
    }

    /** {@code LLM_KV_RopeShift_<type>}. */
    public static String kvRopeShift(KvDType type) {
        return Objects.requireNonNull(type, "type").kernel(KV_ROPE_SHIFT_PREFIX);
    }

    // ------------------------------------------------------------------ LLM_RoPE_Ext

    /**
     * Arguments of {@link #ROPE_EXT} for {@code nTokens} rows of {@code nHeads} heads; the rotation starts at
     * {@code nOffs} within each head (a Q head's {@code n_nope}, a shared K-rope vector's {@code kv_lora}, else
     * {@code spec.nOffs()}).
     *
     * @param posStride section stride of {@code pos} (the step's token count; ignored by single-section types)
     * @param hasFf     true when {@code ff} holds the layer's frequency factors; false binds a placeholder
     * @throws IllegalArgumentException when the geometry does not fit the head or a count is negative
     */
    public static Object[] ropeExtArgs(RopeSpec spec, int nTokens, int nHeads, int headStride, int rowStride,
                                       int nOffs, int posStride, int xOff, boolean hasFf,
                                       GpuBuffer pos, GpuBuffer ff, GpuBuffer x) {
        Objects.requireNonNull(spec, "spec");
        validateRopeExt(spec.type(), spec.nDims(), nOffs, nTokens, nHeads, headStride, rowStride, posStride, xOff);
        final int[] s = spec.sections();
        return new Object[]{
                nTokens, nHeads, headStride, rowStride, spec.nDims(), nOffs, spec.type().ggmlMode(), posStride,
                hasFf ? 1 : 0, spec.thetaScale(), spec.freqScale(), spec.extFactor(), spec.attnFactor(),
                spec.corrLo(), spec.corrHi(), s[0], s[1], s[2], s[3], xOff,
                Objects.requireNonNull(pos, "pos"), Objects.requireNonNull(ff, "ff"), Objects.requireNonNull(x, "x")};
    }

    /**
     * Preconditions of {@link #ROPE_EXT}: a rotary type, even positive {@code nDims}, even non-negative {@code nOffs}
     * with the rotated width inside the head (VISION: {@code nOffs == 0}), non-negative counts and offsets, and a
     * positive section stride for multi-section types.
     *
     * @throws IllegalArgumentException naming the violated condition
     */
    public static void validateRopeExt(RopeType type, int nDims, int nOffs, int nTokens, int nHeads, int headStride,
                                       int rowStride, int posStride, int xOff) {
        Objects.requireNonNull(type, "type");
        if (type == RopeType.NONE) {
            throw new IllegalArgumentException(ROPE_EXT + " needs a rotary type, got NONE");
        }
        if (nDims <= 0 || nDims % 2 != 0 || nOffs < 0 || nOffs % 2 != 0) {
            throw new IllegalArgumentException(ROPE_EXT + " needs even nDims > 0 and even nOffs >= 0, got " + nDims
                    + " / " + nOffs);
        }
        if (type == RopeType.VISION && nOffs != 0) {
            throw new IllegalArgumentException(ROPE_EXT + " VISION rotates whole heads, nOffs must be 0, got " + nOffs);
        }
        if ((long) nOffs + RopeMath.rotatedWidth(type, nDims) > headStride) {
            throw new IllegalArgumentException(ROPE_EXT + ": rotated dims [" + nOffs + ", "
                    + (nOffs + RopeMath.rotatedWidth(type, nDims)) + ") exceed the head stride " + headStride);
        }
        if (nTokens < 0 || nHeads < 0 || rowStride < 0 || xOff < 0) {
            throw new IllegalArgumentException(ROPE_EXT + " counts, strides and offsets must be >= 0: tokens " + nTokens
                    + ", heads " + nHeads + ", row stride " + rowStride + ", x_off " + xOff);
        }
        if ((long) nHeads * headStride > Math.max(rowStride, 0) && nTokens > 1) {
            throw new IllegalArgumentException(ROPE_EXT + ": " + nHeads + " heads of stride " + headStride
                    + " overlap the next row at row stride " + rowStride);
        }
        if (RopeMath.multiSection(type) && posStride < nTokens) {
            throw new IllegalArgumentException(ROPE_EXT + " " + type + " needs a section stride >= n_tokens, got "
                    + posStride + " < " + nTokens);
        }
    }

    /** Work-items of {@link #ROPE_EXT}: one per rotation pair. */
    public static long ropeExtWorkItems(RopeType type, int nTokens, int nHeads, int nDims) {
        return (long) nTokens * nHeads * RopeMath.pairCount(type, nDims);
    }

    // ------------------------------------------------------------------ LLM_RoPE_FoPE

    /**
     * Arguments of {@link #ROPE_FOPE} for {@code nTokens} rows of {@code nHeads} heads rotated by the coefficients
     * of {@code fope}, the head count being the rotated heads (a query projection's heads or the KV heads).
     *
     * @param nInvFreq projected frequencies {@code D}: pairs at or above it rotate by {@code (1, 1)}
     * @param nCoefRows coefficient rows; with {@code sepHead} every head reads its KV group's row, else row 0
     * @throws IllegalArgumentException when the geometry or the coefficient shapes do not fit
     */
    public static Object[] fopeArgs(int nTokens, int nHeads, int headStride, int rowStride, int nDims,
                                    int xOff, int nInvFreq, int nCoefRows, boolean sepHead, GpuBuffer pos,
                                    GpuBuffer cosCoef, GpuBuffer sinCoef, GpuBuffer invFreq, GpuBuffer x) {
        validateFope(nDims, xOff, nTokens, nHeads, headStride, rowStride, nInvFreq, nCoefRows, sepHead);
        return new Object[]{nTokens, nHeads, headStride, rowStride, nDims, xOff, nInvFreq, nCoefRows,
                sepHead ? 1 : 0, Objects.requireNonNull(pos, "pos"), Objects.requireNonNull(cosCoef, "cosCoef"),
                Objects.requireNonNull(sinCoef, "sinCoef"), Objects.requireNonNull(invFreq, "invFreq"),
                Objects.requireNonNull(x, "x")};
    }

    /**
     * Preconditions of {@link #ROPE_FOPE}: even positive {@code nDims}, non-negative counts and offsets, the rotated
     * head inside the row, projected frequencies between 1 and {@code nDims / 2}, and a head count that is a multiple
     * of the coefficient rows when the heads are separate.
     *
     * @throws IllegalArgumentException naming the violated condition
     */
    public static void validateFope(int nDims, int xOff, int nTokens, int nHeads, int headStride, int rowStride,
                                    int nInvFreq, int nCoefRows, boolean sepHead) {
        if (nDims <= 0 || nDims % 2 != 0) {
            throw new IllegalArgumentException(ROPE_FOPE + " needs an even nDims > 0, got " + nDims);
        }
        if (nInvFreq <= 0 || nInvFreq > nDims / 2) {
            throw new IllegalArgumentException(ROPE_FOPE + " projects " + nInvFreq + " frequencies, outside [1, "
                    + (nDims / 2) + "]");
        }
        if (nTokens < 0 || nHeads < 0 || headStride < 0 || rowStride < 0 || xOff < 0) {
            throw new IllegalArgumentException(ROPE_FOPE + " counts, strides and offsets must be >= 0: tokens "
                    + nTokens + ", heads " + nHeads + ", head stride " + headStride + ", row stride " + rowStride
                    + ", x_off " + xOff);
        }
        if (nDims > headStride) {
            throw new IllegalArgumentException(ROPE_FOPE + ": rotation of " + nDims + " dims exceeds the head stride "
                    + headStride);
        }
        if (nCoefRows <= 0) {
            throw new IllegalArgumentException(ROPE_FOPE + " needs positive coefficient rows, got " + nCoefRows);
        }
        if (sepHead && nHeads > 0 && nHeads % nCoefRows != 0) {
            throw new IllegalArgumentException(ROPE_FOPE + " rotates " + nHeads + " heads with " + nCoefRows
                    + " coefficient rows; the head count must be a multiple of it");
        }
        if (nHeads > 0 && (long) nHeads * headStride > rowStride) {
            throw new IllegalArgumentException(ROPE_FOPE + ": " + nHeads + " heads of stride " + headStride
                    + " exceed the row stride " + rowStride);
        }
    }

    /** Work-items of {@link #ROPE_FOPE}: one per (token, head). */
    public static long fopeWorkItems(int nTokens, int nHeads) {
        return (long) nTokens * nHeads;
    }

    // ------------------------------------------------------------------ LLM_KV_Store_*

    /**
     * Arguments of {@code LLM_KV_Store_<type>}: row {@code cellIdx[t]} of {@code dst} receives the {@code rowElems}
     * values of src row {@code t} at {@code src[srcOff + t * srcRowStride]}. The FP8 kernels additionally take
     * {@code headElems} (the head width, one per-head scale of {@code kvScale} per row section) and {@code kvScale}
     * (the cache's per-head checkpoint KV scale, {@code LLM_KV_Store_FP8_*}); for every other type both are unused
     * and {@code kvScale} may be any buffer (a placeholder).
     *
     * @throws IllegalArgumentException when a count or stride is invalid for the type, or the FP8 or per-token-head
     *                                 scale geometry does not tile the row
     */
    public static Object[] kvStoreArgs(KvDType type, int nTokens, int rowElems, int srcRowStride, int srcOff,
                                       int dstRowStride, int headElems, GpuBuffer src, GpuBuffer cellIdx,
                                       GpuBuffer kvScale, GpuBuffer dst) {
        return kvStoreArgs(type, nTokens, rowElems, srcRowStride, srcOff, dstRowStride, headElems, src, cellIdx,
                kvScale, dst, null, 0);
    }

    /**
     * {@link #kvStoreArgs(KvDType, int, int, int, int, int, int, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer)} with
     * {@code rowScales}, the per-token-head scale output: a per-token-head store writes one F32 scale per (token, head)
     * there and ignores {@code kvScale}. The buffer holds the interleaved K and V scale rows of every cache row; a
     * store of {@code vSide} 1 writes the V half. The scale row width is {@code rowElems / headElems} values.
     *
     * @throws IllegalArgumentException for a per-token-head store without a scale buffer, or an INT4 geometry whose
     *                                 head width, row width or row stride is odd
     */
    public static Object[] kvStoreArgs(KvDType type, int nTokens, int rowElems, int srcRowStride, int srcOff,
                                       int dstRowStride, int headElems, GpuBuffer src, GpuBuffer cellIdx,
                                       GpuBuffer kvScale, GpuBuffer dst, GpuBuffer rowScales, int vSide) {
        return kvStoreArgs(type, nTokens, rowElems, srcRowStride, srcOff, dstRowStride, headElems, src, cellIdx,
                kvScale, dst, rowScales, null, 0, 0, vSide);
    }

    /**
     * Arguments of {@code LLM_KV_Store_<type>} with the side buffer of a vendor cache.
     *
     * <p>A vendor store writes the head's share of its row's packed side planes at {@code side +
     * DeviceKvFormats.sidePlaneBase(heads) + cell * planeStride + planeOff}, and reads the Lloyd-Max table of a
     * TurboQuant key stream from the same buffer's header.
     *
     * @param side        the cache's side buffer ({@link GpuKvLayer#vendorSide()})
     * @param planeStride bytes between two consecutive side-buffer rows
     * @param planeOff    byte offset of the side's plane row inside a side-buffer row
     * @throws IllegalArgumentException for a geometry the store kernel does not address
     */
    public static Object[] kvStoreArgs(KvDType type, int nTokens, int rowElems, int srcRowStride, int srcOff,
                                       int dstRowStride, int headElems, GpuBuffer src, GpuBuffer cellIdx,
                                       GpuBuffer kvScale, GpuBuffer dst, GpuBuffer rowScales, GpuBuffer side,
                                       int planeStride, int planeOff, int vSide) {
        validateKvStore(type, nTokens, rowElems, srcRowStride, srcOff, dstRowStride);
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(cellIdx, "cellIdx");
        Objects.requireNonNull(dst, "dst");
        if (type.vendorFormat()) {
            if (headElems <= 0 || rowElems % headElems != 0) {
                throw new IllegalArgumentException(kvStore(type) + " needs a head width that tiles its " + rowElems
                        + " values per row, got " + headElems);
            }
            Objects.requireNonNull(kvScale, "kvScale");
            Objects.requireNonNull(side, "side");
            if (planeStride < 0 || planeOff < 0 || planeOff > planeStride) {
                throw new IllegalArgumentException(kvStore(type) + " needs 0 <= planeOff <= planeStride, got "
                        + planeOff + " and " + planeStride);
            }
            return new Object[]{nTokens, rowElems, srcRowStride, srcOff, dstRowStride, headElems, planeStride,
                    planeOff, src, cellIdx, kvScale, dst, side};
        }
        if (type.perTokenHead()) {
            if (headElems <= 0 || rowElems % headElems != 0) {
                throw new IllegalArgumentException(kvStore(type) + " needs a head width that tiles its " + rowElems
                        + " values per row, got " + headElems);
            }
            if (type == KvDType.INT4_PER_TOKEN_HEAD
                    && (headElems % 2 != 0 || rowElems % 2 != 0 || dstRowStride % 2 != 0)) {
                throw new IllegalArgumentException(kvStore(type) + " packs two values per byte: head width, row width"
                        + " and row stride must be even, got " + headElems + ", " + rowElems + ", " + dstRowStride);
            }
            Objects.requireNonNull(rowScales, "rowScales");
            if (vSide != 0 && vSide != 1) {
                throw new IllegalArgumentException(kvStore(type) + " needs a K (0) or V (1) side, got " + vSide);
            }
            return new Object[]{nTokens, rowElems, srcRowStride, srcOff, dstRowStride, headElems, vSide, src, cellIdx,
                    dst, rowScales};
        }
        if (!type.fp8()) {
            return new Object[]{nTokens, rowElems, srcRowStride, srcOff, dstRowStride, src, cellIdx, dst};
        }
        if (headElems <= 0 || rowElems % headElems != 0) {
            throw new IllegalArgumentException(kvStore(type) + " needs a head width that tiles its " + rowElems
                    + " values per row, got " + headElems);
        }
        Objects.requireNonNull(kvScale, "kvScale");
        return new Object[]{nTokens, rowElems, srcRowStride, srcOff, dstRowStride, headElems, src, cellIdx, kvScale,
                dst};
    }

    /**
     * Index of the {@code dst} cache buffer of {@code kvStore(type)}'s argument list: {@link #ARG_KV_STORE_DST} for
     * every type but the FP8 stores, {@link #ARG_KV_STORE_FP8_DST} for those.
     */
    public static int kvStoreDstArg(KvDType type) {
        Objects.requireNonNull(type, "type");
        if (type.perTokenHead()) {
            return ARG_KV_STORE_PT_DST;
        }
        if (type.vendorFormat()) {
            return ARG_KV_STORE_VENDOR_DST;
        }
        return type.fp8() ? ARG_KV_STORE_FP8_DST : ARG_KV_STORE_DST;
    }

    /**
     * Index of the {@code cell_idx} buffer of {@code kvStore(type)}'s argument list, the companion of
     * {@link #kvStoreDstArg(KvDType)}.
     */
    public static int kvStoreCellArg(KvDType type) {
        Objects.requireNonNull(type, "type");
        if (type.perTokenHead()) {
            return ARG_KV_STORE_PT_CELL;
        }
        if (type.vendorFormat()) {
            return ARG_KV_STORE_VENDOR_CELL;
        }
        return type.fp8() ? ARG_KV_STORE_FP8_CELL : ARG_KV_STORE_CELL;
    }

    /**
     * Preconditions of {@code LLM_KV_Store_*}: non-negative counts and offsets, {@code rowElems} not above either
     * stride, and for a block type {@code rowElems} and {@code dstRowStride} multiples of its 32-value block.
     *
     * @throws IllegalArgumentException naming the violated condition
     */
    public static void validateKvStore(KvDType type, int nTokens, int rowElems, int srcRowStride, int srcOff,
                                       int dstRowStride) {
        Objects.requireNonNull(type, "type");
        final String k = kvStore(type);
        if (nTokens < 0 || rowElems <= 0 || srcOff < 0) {
            throw new IllegalArgumentException(k + " needs nTokens >= 0, rowElems > 0 and srcOff >= 0, got " + nTokens
                    + ", " + rowElems + ", " + srcOff);
        }
        if ((nTokens > 1 && srcRowStride < rowElems) || dstRowStride < rowElems) {
            throw new IllegalArgumentException(k + ": row of " + rowElems + " values exceeds the source stride "
                    + srcRowStride + " or the cache stride " + dstRowStride);
        }
        final int bs = type.blockSize();
        if (rowElems % bs != 0 || dstRowStride % bs != 0) {
            throw new IllegalArgumentException(k + " needs rowElems and dstRowStride that are multiples of " + bs
                    + ", got " + rowElems + " / " + dstRowStride);
        }
    }

    /**
     * Work-items of {@code LLM_KV_Store_<type>}: one per element (F32, F16, BF16, FP8) or per 32-element block (block
     * types).
     */
    public static long kvStoreWorkItems(KvDType type, int nTokens, int rowElems) {
        return (long) nTokens * (rowElems / type.blockSize());
    }

    /**
     * Work-items of a per-token-head {@code LLM_KV_Store_*}: one per (token, KV head). A per-token-head store uses this
     * count; every other type uses {@link #kvStoreWorkItems(KvDType, int, int)}.
     *
     * @throws IllegalArgumentException when the head width does not tile the row
     */
    public static long kvStoreWorkItemsPerTokenHead(KvDType type, int nTokens, int rowElems, int headElems) {
        if (!type.perTokenHead() && !type.vendorFormat()) {
            return kvStoreWorkItems(type, nTokens, rowElems);
        }
        if (headElems <= 0 || rowElems % headElems != 0) {
            throw new IllegalArgumentException(kvStore(type) + " needs a head width that tiles its " + rowElems
                    + " values per row, got " + headElems);
        }
        return (long) nTokens * (rowElems / headElems);
    }

    // ------------------------------------------------------------------ LLM_KV_RopeShift_*

    /**
     * Arguments of {@code LLM_KV_RopeShift_<type>} for a K cache of {@code nCells} rows: dims
     * {@code [nOffs, nOffs + spec.nDims())} of every head rotate by RoPE at {@code delta[c]} using the
     * {@link RopeMath#shiftType shift type} of {@code spec.type()}.
     *
     * @param rotK Hadamard chunk of a rotated block-type cache (0 when the cache is not rotated)
     * @throws IllegalArgumentException when the geometry does not fit the cache type or rotation
     */
    public static Object[] kvRopeShiftArgs(KvDType type, RopeSpec spec, int nCells, int nHeads, int headStride,
                                           int rowStride, int nOffs, int rotK, boolean hasFf,
                                           GpuBuffer delta, GpuBuffer ff, GpuBuffer kCache) {
        Objects.requireNonNull(spec, "spec");
        final RopeType shift = RopeMath.shiftType(spec.type());
        validateKvShift(type, spec.nDims(), nOffs, nCells, nHeads, headStride, rowStride, rotK);
        final Object[] head = {nCells, nHeads, headStride, rowStride, spec.nDims(), nOffs, shift.ggmlMode(),
                hasFf ? 1 : 0, spec.thetaScale(), spec.freqScale(), spec.extFactor(), spec.attnFactor(),
                spec.corrLo(), spec.corrHi()};
        final Object[] bufs = {Objects.requireNonNull(delta, "delta"), Objects.requireNonNull(ff, "ff"),
                Objects.requireNonNull(kCache, "kCache")};
        final boolean blocks = type.quantized();
        final Object[] args = new Object[head.length + (blocks ? 1 : 0) + bufs.length];
        System.arraycopy(head, 0, args, 0, head.length);
        int n = head.length;
        if (blocks) {
            args[n++] = rotK;
        }
        System.arraycopy(bufs, 0, args, n, bufs.length);
        return args;
    }

    /**
     * Preconditions of {@code LLM_KV_RopeShift_*}: even positive {@code nDims} inside the head, non-negative counts;
     * a Hadamard chunk only for a block type. For a block type: {@code headStride} and {@code rowStride} multiples of
     * 32, {@code rotK} zero or a power of two of at least 32 (the cache's rotation chunk, which tiles the head from its
     * start), and a {@link #shiftWindowStart work window} that lies inside the head stride and holds at most
     * {@link #SHIFT_MAX_DIMS} values.
     *
     * @throws IllegalArgumentException naming the violated condition
     */
    public static void validateKvShift(KvDType type, int nDims, int nOffs, int nCells, int nHeads, int headStride,
                                       int rowStride, int rotK) {
        Objects.requireNonNull(type, "type");
        final String k = kvRopeShift(type);
        if (!type.hasShiftKernel()) {
            throw new IllegalArgumentException("the " + type.name().toLowerCase(java.util.Locale.ROOT) + " KV cache"
                    + " has no device context-shift kernel (LLM_KV_RopeShift_" + type.kernelSuffix() + "): the vendor"
                    + " formats re-derive their group scales and centroid codes from a whole row");
        }
        if (nDims <= 0 || nDims % 2 != 0 || nOffs < 0 || (long) nOffs + nDims > headStride) {
            throw new IllegalArgumentException(k + " needs even nDims > 0 with [nOffs, nOffs + nDims) inside the head"
                    + " stride " + headStride + ", got nDims " + nDims + ", nOffs " + nOffs);
        }
        if (nCells < 0 || nHeads < 0 || (long) nHeads * headStride > rowStride) {
            throw new IllegalArgumentException(k + " needs nCells, nHeads >= 0 and nHeads * headStride <= rowStride,"
                    + " got " + nCells + ", " + nHeads + " x " + headStride + " / " + rowStride);
        }
        if (rotK < 0) {
            throw new IllegalArgumentException(k + ": Hadamard chunk must be >= 0, got " + rotK);
        }
        if (type.perTokenHead() && type.blockSize() > 1 && (headStride % 2 != 0 || rowStride % 2 != 0)) {
            throw new IllegalArgumentException(k + " packs two values per byte: headStride and rowStride must be"
                    + " even, got " + headStride + ", " + rowStride);
        }
        if (!type.quantized()) {
            if (rotK != 0) {
                throw new IllegalArgumentException(k + ": only quantized caches are Hadamard-rotated, got rot_k "
                        + rotK);
            }
            return;
        }
        final int bs = type.blockSize();
        if (headStride % bs != 0 || rowStride % bs != 0) {
            throw new IllegalArgumentException(k + " needs headStride and rowStride that are multiples of " + bs
                    + " (every head starts on a block), got " + headStride + ", " + rowStride);
        }
        if (rotK != 0 && ((rotK & (rotK - 1)) != 0 || rotK < bs)) {
            throw new IllegalArgumentException(k + ": Hadamard chunk must be a power of two >= " + bs
                    + ", got " + rotK);
        }
        final int w0 = shiftWindowStart(nOffs, rotK);
        final int w1 = shiftWindowEnd(nDims, nOffs, rotK);
        if (w1 > headStride) {
            throw new IllegalArgumentException(k + ": work window [" + w0 + ", " + w1 + ") of dims [" + nOffs + ", "
                    + (nOffs + nDims) + ") with rot_k " + rotK + " exceeds the head stride " + headStride);
        }
        if (w1 - w0 > SHIFT_MAX_DIMS) {
            throw new IllegalArgumentException(k + " holds at most " + SHIFT_MAX_DIMS + " values per head, the"
                    + " work window [" + w0 + ", " + w1 + ") holds " + (w1 - w0));
        }
    }

    /**
     * Start of the work window of a head in a block-type rope shift ({@code nf_rope_shift_w0}): {@code nOffs} rounded
     * down to the granule, which is {@code rotK} for a Hadamard-rotated cache and the 32-value block otherwise. The
     * kernel dequantizes, rotates and requantizes {@code [start, end)}; blocks outside the window keep their stored
     * bits.
     */
    public static int shiftWindowStart(int nOffs, int rotK) {
        final int g = shiftGranule(rotK);
        return nOffs / g * g;
    }

    /**
     * End of the work window of a block-type rope shift ({@code nf_rope_shift_w1}): {@code nOffs + nDims} rounded up
     * to the granule.
     */
    public static int shiftWindowEnd(int nDims, int nOffs, int rotK) {
        final int g = shiftGranule(rotK);
        return (int) (((long) nOffs + nDims + g - 1) / g * g);
    }

    private static int shiftGranule(int rotK) {
        return rotK > 0 ? rotK : KvBlockCodec.BLOCK;
    }

    /**
     * Work-items of {@code LLM_KV_RopeShift_<type>}: one per pair for F32/F16/BF16, one per (cell, head) for a
     * per-token-head type (the head's pairs rotate sequentially, keeping INT4's two nibbles in one work-item), and one
     * workgroup of {@code local} work-items per (cell, head) for a block type.
     *
     * @throws IllegalArgumentException for a block-type local size that is not a power of two up to {@code LLM_DIMX}
     */
    public static long kvRopeShiftWorkItems(KvDType type, int nCells, int nHeads, int nDims, int local) {
        if (type.perTokenHead()) {
            return (long) nCells * nHeads;
        }
        if (type.quantized()) {
            if (local <= 0 || local > LlmKernelConstants.LLM_DIMX || (local & (local - 1)) != 0) {
                throw new IllegalArgumentException(kvRopeShift(type) + " needs a power-of-two local size <= "
                        + LlmKernelConstants.LLM_DIMX + ", got " + local);
            }
            return (long) nCells * nHeads * local;
        }
        return (long) nCells * nHeads * (nDims / 2);
    }

    /**
     * Local size of a rope-shift launch from the backend's preferred {@code local}: the largest power of two up to
     * {@code LLM_DIMX} for the per-(cell, head) block-type kernels, {@code local} itself otherwise.
     */
    public static int kvRopeShiftLocal(KvDType type, int local) {
        if (!type.quantized()) {
            return local;
        }
        return Integer.highestOneBit(Math.max(1, Math.min(local, LlmKernelConstants.LLM_DIMX)));
    }

    /** {@code n} rounded up to a multiple of {@code local} (the 1-D global size of a launch). */
    public static long roundUp(long n, int local) {
        if (local <= 0) {
            throw new IllegalArgumentException("local size must be positive, got " + local);
        }
        return (n + local - 1) / local * local;
    }
}
