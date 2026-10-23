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
 * LLM_KV_RopeShift_{F32,F16,BF16}
 *                         (i n_cells, i n_heads, i head_stride, i row_stride, i n_dims, i n_offs, i mode, i has_ff,
 *                          f theta_scale, f freq_scale, f ext_factor, f attn_factor, f corr_lo, f corr_hi,
 *                          R&lt;int&gt; delta, R ff, RWn&lt;KV&gt; k_cache)
 * LLM_KV_RopeShift_&lt;block type&gt;
 *                         (the F32/F16/BF16 scalars, then i rot_k; the same buffers)
 * </pre>
 * {@code <KV>} is a {@link KvDType#kernelSuffix()}: F32, F16, BF16 or one of the block types Q8_0, Q4_0, Q4_1,
 * IQ4_NL, Q5_0, Q5_1 ({@link KvDType#quantized()}).
 */
public final class RopeKernels {

    public static final String ROPE_EXT = "LLM_RoPE_Ext";
    /** Name prefix of the {@code LLM_KV_Store_<KV>} variants. */
    public static final String KV_STORE_PREFIX = "LLM_KV_Store_";
    /** Name prefix of the {@code LLM_KV_RopeShift_<KV>} variants. */
    public static final String KV_ROPE_SHIFT_PREFIX = "LLM_KV_RopeShift_";
    /** Every {@code LLM_KV_Store_<KV>} variant, in {@link KvDType} order. */
    public static final List<String> KV_STORE_KERNELS = KvDType.kernelVariants(KV_STORE_PREFIX);
    /** Every {@code LLM_KV_RopeShift_<KV>} variant, in {@link KvDType} order. */
    public static final List<String> KV_ROPE_SHIFT_KERNELS = KvDType.kernelVariants(KV_ROPE_SHIFT_PREFIX);

    /** {@code n_tokens} of {@link #ROPE_EXT} and of the {@code LLM_KV_Store_*} kernels. */
    public static final int ARG_N_TOKENS = 0;
    /** {@code pos_stride} of {@link #ROPE_EXT} (the step's token count for multi-section positions). */
    public static final int ARG_POS_STRIDE = 7;

    /** Arguments of {@link #ROPE_EXT}: 20 scalars and 3 buffers. */
    public static final int ROPE_EXT_ARGS = 23;
    /** Arguments of {@code LLM_KV_Store_*}: 5 scalars and 3 buffers. */
    public static final int KV_STORE_ARGS = 8;
    /** Arguments of {@code LLM_KV_RopeShift_F32/F16/BF16}: 14 scalars and 3 buffers; block types add {@code rot_k}. */
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

    // ------------------------------------------------------------------ LLM_KV_Store_*

    /**
     * Arguments of {@code LLM_KV_Store_<type>}: row {@code cellIdx[t]} of {@code dst} receives the {@code rowElems}
     * values of src row {@code t} at {@code src[srcOff + t * srcRowStride]}.
     *
     * @throws IllegalArgumentException when a count or stride is invalid for the type
     */
    public static Object[] kvStoreArgs(KvDType type, int nTokens, int rowElems, int srcRowStride, int srcOff,
                                       int dstRowStride, GpuBuffer src, GpuBuffer cellIdx, GpuBuffer dst) {
        validateKvStore(type, nTokens, rowElems, srcRowStride, srcOff, dstRowStride);
        return new Object[]{nTokens, rowElems, srcRowStride, srcOff, dstRowStride,
                Objects.requireNonNull(src, "src"), Objects.requireNonNull(cellIdx, "cellIdx"),
                Objects.requireNonNull(dst, "dst")};
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
     * Work-items of {@code LLM_KV_Store_<type>}: one per element (F32, F16, BF16) or per 32-element block (block
     * types).
     */
    public static long kvStoreWorkItems(KvDType type, int nTokens, int rowElems) {
        return (long) nTokens * (rowElems / type.blockSize());
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
     * Work-items of {@code LLM_KV_RopeShift_<type>}: one per pair for F32/F16/BF16; one workgroup of {@code local}
     * work-items per (cell, head) for a block type.
     *
     * @throws IllegalArgumentException for a block-type local size that is not a power of two up to {@code LLM_DIMX}
     */
    public static long kvRopeShiftWorkItems(KvDType type, int nCells, int nHeads, int nDims, int local) {
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
