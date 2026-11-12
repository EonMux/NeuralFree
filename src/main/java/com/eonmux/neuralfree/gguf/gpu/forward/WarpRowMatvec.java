package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * Launch plan of the CUDA one-row integer-dot matvec forms ({@code kernels/llm/matvec_row.cl}):
 * {@code {T}_MatVecRow_Q8}, {@code {T}_MatVecAddRow_Q8} and {@code {T}_MatVecFusedGateUpRow_Q8}, the single-token
 * Q8_1-activation products of {@link #TYPES} with one warp per output row, lane groups that read each weight byte
 * once and a warp-shuffle reduction, in place of the family's 64-lane work-group per row and its barrier tree
 * (llama.cpp's mmvq shape for one column).
 *
 * <p>The forms exist only in the CUDA kernel modules (a {@code [cuda]} manifest source); the dispatcher asks
 * {@link #served} before it looks a twin up, so every other backend keeps the family's forms.</p>
 *
 * <p>{@link #warps} picks the warps per work-group, or the family, per type and shape from the RTX 5080 Laptop's
 * measurements (weights rotated past the L2, medians of five rounds, in microseconds against the family's
 * single-row form):</p>
 * <ul>
 *   <li>Q4_K, two warps: 23.6 against 26.2 at 9728x2560, 21.1 against 23.4 at 2560x9728, the fused gate/up
 *       43.3 against 50.7 at 9728x2560, a 151669x1024 head 124 against 187. One warp per work-group caps the
 *       resident warps at the per-SM work-group limit, which two avoid.</li>
 *   <li>Q5_K, two warps, as Q4_K (two sweeps): the fused gate/up 32.8-34.0 against 49.4-51.2 at 9728x2560,
 *       16.7-19.1 against 24.6-25.3 at 4096x2560 and 11.1-11.6 against 12.9-13.7 at 1024x2560; plain products at or
 *       below the family (38.6-41.1 against 41.1-43.3 at 2560x9728, a 151936x1024 head 162-198 against 198-200).
 *       Below the family lane's threshold, where the float matvec served Q5_K, the form takes 5.7-8.2 against
 *       7.9-9.8 at 1024x1024 and 8.0-9.7 against 13.7-15.6 at 1024x2560. Qwen3-Embedding 0.6B decodes 611-615
 *       tokens/s against 477-500 in Q5_K_M and 599-625 against 467-476 in pure Q5_K with them (tg128, three
 *       alternating runs).</li>
 *   <li>Q6_K, one warp: 31.5 against 34.1 at 2560x9728, 15.0 against 16.1 at 4096x2560, the fused gate/up 65.8
 *       against 74.0 at 9728x2560, a 151669x1024 head 190 against 231. It also serves the small projections
 *       below the family lane's threshold ({@link Int8MatvecTuning#LANE_MIN_ELEMENTS}), where the float matvec beat
 *       the family's integer-dot form: at 1024x2560 the form takes 3.9 against the float matvec's 7.4, and a 4B
 *       Q4_K_M model, whose attn_v projections are such Q6_K rows, decodes 182 tokens/s against 179 with them on the
 *       float matvec (tg128 medians of nine). A head of more than 1024 columns stays on the family, which the form
 *       does not beat there (540 and 555 against 472 and 573 at 151669x2560 in two sweeps).</li>
 *   <li>Q8_0: rows of at most 1024 columns, one warp (5.4 against 5.8 at 3072x1024, the fused gate/up 10.7
 *       against 11.2) and four on a head (240 against 266 at 151669x1024); wider rows measured equal to the
 *       family at the decoder shapes and slower on the head (632 against 613 at 151669x2560), so they keep it.</li>
 * </ul>
 */
final class WarpRowMatvec {

    /** Most warps per work-group ({@code NF_MVR_MAX_WARPS}). */
    static final int MAX_WARPS = 4;
    /** Rows from which a projection counts as an output head for {@link #warps}. */
    static final int HEAD_ROWS = 65536;
    /** Widest Q8_0 row, and widest K-quant head row, the forms take over from the family. */
    static final int NARROW_COLS = 1024;

    /** The weight types with one-row forms. */
    static final Set<GGMLType> TYPES =
            Collections.unmodifiableSet(EnumSet.of(GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K));

    private static final String Q8_SUFFIX = "_Q8";

    private WarpRowMatvec() {
    }

    /** Whether a backend of this type builds the forms: the CUDA backend, whose modules hold the {@code [cuda]} sources. */
    static boolean served(ComputeBackend.BackendType backend) {
        return Objects.requireNonNull(backend, "backend must not be null") == ComputeBackend.BackendType.CUDA;
    }

    /**
     * The one-row twin of an integer-dot single-token kernel of {@code type} ({@code {T}_MatVec_Q8},
     * {@code {T}_MatVecAdd_Q8} or {@code {T}_MatVecFusedGateUp_Q8}: the same name with {@code Row} before
     * {@code _Q8}), or null when the type has no one-row forms or the kernel is not one of those three.
     */
    static String twinName(String int8Kernel, GGMLType type) {
        if (int8Kernel == null || !TYPES.contains(type)) {
            return null;
        }
        String prefix = type.name() + "_";
        String form = int8Kernel.startsWith(prefix) && int8Kernel.endsWith(Q8_SUFFIX)
                ? int8Kernel.substring(prefix.length(), int8Kernel.length() - Q8_SUFFIX.length()) : null;
        if (!"MatVec".equals(form) && !"MatVecAdd".equals(form) && !"MatVecFusedGateUp".equals(form)) {
            return null;
        }
        return prefix + form + "Row" + Q8_SUFFIX;
    }

    /**
     * The stacked single-token Q/K/V form ({@code {T}_MatVecQKVRow_Q8} over three weights of {@code qk}, or
     * {@code {T}_MatVecQKVRowVQ6_K_Q8} for a Q6_K V weight beside Q4_K or Q5_K Q and K weights, as Q4_K_M and Q5_K_M
     * models mix them), or null for any other pair.
     */
    static String qkvTwinName(GGMLType qk, GGMLType v) {
        if (!TYPES.contains(qk) || v == null) {
            return null;
        }
        if (v == qk) {
            return qk.name() + "_MatVecQKVRow" + Q8_SUFFIX;
        }
        return v == GGMLType.Q6_K && (qk == GGMLType.Q4_K || qk == GGMLType.Q5_K)
                ? qk.name() + "_MatVecQKVRowV" + v.name() + Q8_SUFFIX : null;
    }

    /**
     * The warps per work-group of a launch over {@code rows} rows of {@code cols} elements of {@code type}, or 0
     * when the family's forms serve the shape better (or the type has no forms, or the rows are not whole blocks).
     */
    static int warps(GGMLType type, int rows, int cols) {
        if (!TYPES.contains(type) || rows < 1 || cols < 1 || cols % type.getBlockSize() != 0) {
            return 0;
        }
        boolean head = rows >= HEAD_ROWS;
        switch (type) {
            case Q4_K:
            case Q5_K:
                return 2;
            case Q6_K:
                return head && cols > NARROW_COLS ? 0 : 1;
            default:
                return cols > NARROW_COLS ? 0 : head ? 4 : 1;
        }
    }

    /** The one-dimensional global size of a launch over {@code rows} rows at {@code warps} warps per work-group. */
    static long globalSize(int rows, int warps) {
        if (rows < 1) {
            throw new IllegalArgumentException("a one-row matvec needs at least one output row, got " + rows);
        }
        return (long) GpuOps.ceilDiv(rows, checkWarps(warps)) * localSize(warps);
    }

    /** The work-group size of {@code warps} warps. */
    static long localSize(int warps) {
        return 32L * checkWarps(warps);
    }

    private static int checkWarps(int warps) {
        if (warps != 1 && warps != 2 && warps != MAX_WARPS) {
            throw new IllegalArgumentException("a one-row matvec work-group holds 1, 2 or 4 warps, got " + warps);
        }
        return warps;
    }
}
