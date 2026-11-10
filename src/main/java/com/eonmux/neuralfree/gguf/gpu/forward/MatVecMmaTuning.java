package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ops.gemm.MatVecMmaPlan;

/**
 * Measured per-device selection of the tensor-core multi-column matvec ({@code {T}_MatVecMma{8,16}_Q8},
 * {@link MatVecMmaPlan}) inside the integer-dot multi-column lane: for a projection of {@code rows x cols} at
 * {@code n} tokens, the warp count of the tensor-core form when it measured faster than the counted and paired
 * integer-dot forms ({@code {T}_MatVecN<n>_Q8}, {@code {T}_MatVec2N<n>_Q8}) on the device kind, else 0 (the
 * counted forms keep the launch). Device kinds and types without measurements answer 0.
 *
 * <p>The evidence is {@code MatVecMmaMeasureTest} on the RTX 5080 Laptop (CUDA, sm_120), per launch from DRAM over
 * the 4B and 0.6B Qwen3 projection shapes at every token count from 2 to 9 and at 12 and 16 (the small
 * projections, near the per-launch floor, cross-checked by back-to-back launch loops). The tensor-core form's cost is flat in the token
 * count inside a token tile while the counted forms grow per token, so each rule is a smallest token count; the
 * 8-token tile ({@code n <= 8}) and the 16-token tile have their own rules, since the 16-token tile costs more:</p>
 * <ul>
 *   <li>Q4_K from 2 tokens on every shape, at 4 warps (9728x2560 at 2: 26.4 us against the pair's 30.9, at 4:
 *       26.4 against 46.1; at 16: 43.4 against 144.8; the 1024-column head at 2: 139 against 204; the
 *       2560-column head at 16: 510 against 2250). The 1024-row projections tie at 2 and 3 tokens.</li>
 *   <li>Q5_K at 4 warps from 2 tokens (9728x2560 at 2: 28.9 us against the pair's 32.6, at 8: 32.8 against
 *       80.3; 2560x9728 at 4: 30.8 against 47.1; the 1024-column head at 2: 170 against 238), except a projection of
 *       at most {@value #Q5_K_NARROW_ROWS} rows over at most {@value #Q5_K_SHORT_COLS} columns, from 4 tokens
 *       (2048x1024 at 2: 11.2 against 8.9, at 4: 10.6 against 10.8; 1024x1024 at 4: 5.8 against 7.6). Through the
 *       16-token tile a projection of at most {@value #Q5_K_EIGHT_WARP_ROWS} rows takes 8 warps (1024x3072 at 9: 10.7
 *       against 12.5 at 4 warps and 15.8 for the pair; 1024x2560 at 16: 12.0 against 13.4 and 21.5).</li>
 *   <li>Q6_K through the 8-token tile at 4 warps on projections of at least {@value #Q6_K_WIDE_ROWS} rows: an
 *       output head from 3 tokens (151936x1024 at 3: 211 against 318; 151936x2560 at 3: 551 against 609, at 8:
 *       561 against 779 at 8 warps), the others from 4 (9728x2560 at 4: 39.0 against 42.6; 2560x9728 at 4: 42.5
 *       against 43.8, at 8: 42.7 against 48.2 at 8 warps). Narrower projections from 6 tokens at 8 warps
 *       (1024x2560 at 6: 11.9 against 13.2; 1024x3072 at 6: 11.8 against 14.0). Through the 16-token tile at 8
 *       warps (2560x9728 at 9: 70.0 against 84.3; at 16: 56.2 against 95.0), the head at 2 warps (151936x2560 at
 *       16: 801 against 1617).</li>
 *   <li>Q8_0 through the 8-token tile at 8 warps: a head over a reduction of at most
 *       {@value #Q8_0_SHORT_COLS} columns from 4 tokens (151936x1024 at 8: 233 against 530), a wider head from 5
 *       (151936x2560 at 5: 672 against 833), other projections over at most {@value #Q8_0_SHORT_COLS} columns
 *       from 6 (3072x1024 at 6: 13.7 against 16.1; 1024x1024 at 6: 8.1 against 9.3), and the wider ones at 8
 *       (9728x2560: 52.5 against 58.6; 1024x2560: 12.2 against 13.8) except a projection of at least
 *       {@value #Q8_0_WIDE_ROWS} rows over a longer reduction (2560x4096 at 8: 30.4 against 27.2). Through the
 *       16-token tile from 9 tokens on projections of at least {@value #Q8_0_WIDE_ROWS} rows (9728x2560 at 9:
 *       49.2 against 59.2; at 16: 59.5 against 99.8), from 11 where the reduction is more than twice the row
 *       count (2560x9728 at 9: 61.9 against 53.3; at 16: 67.8 against 93.8) and from 10 on narrower projections
 *       (1024x2560 at 9: 13.6 against 12.7; at 16: 12.7 against 20.8).</li>
 * </ul>
 */
public final class MatVecMmaTuning {

    /** Rows from which a projection reads as an output head. */
    static final int HEAD_ROWS = 65536;
    /** Rows from which the Q8_0 16-token form serves from 9 tokens. */
    static final int Q8_0_WIDE_ROWS = 2048;
    /** Widest reduction the Q8_0 8-token form serves from fewer than 8 tokens off a head. */
    static final int Q8_0_SHORT_COLS = 1024;
    /** Most rows of a Q5_K projection over a short reduction that the tensor-core form serves only from 4 tokens. */
    static final int Q5_K_NARROW_ROWS = 2048;
    /** Widest reduction of such a Q5_K projection. */
    static final int Q5_K_SHORT_COLS = 1024;
    /** Most rows of a Q5_K projection whose 16-token form runs at 8 warps. */
    static final int Q5_K_EIGHT_WARP_ROWS = 1024;
    /** Rows from which the Q6_K 8-token form serves at 4 warps from 4 tokens. */
    static final int Q6_K_WIDE_ROWS = 2048;
    /** Largest token count of the 8-token tile. */
    private static final int EIGHT_TILE = 8;

    private MatVecMmaTuning() {
    }

    /**
     * The warp count of the tensor-core form for {@code nTokens} rows through a {@code rows x cols} weight of
     * {@code type} on {@code device}, or 0 when the counted integer-dot forms keep the launch (no measurement,
     * a slower measurement, or a shape the kernel family does not serve).
     */
    public static int warps(DeviceKind device, GGMLType type, int rows, int cols, int nTokens) {
        if (!device.equals(nvidiaDiscreteCudaTensor()) || !MatVecMmaPlan.serves(type, cols)
                || nTokens < 2 || nTokens > MatVecMmaPlan.MAX_TOKENS || rows <= 0) {
            return 0;
        }
        switch (type) {
            case Q4_K:
                return 4;
            case Q5_K:
                return q5kWarps(rows, cols, nTokens);
            case Q6_K:
                return q6kWarps(rows, nTokens);
            case Q8_0:
                return nTokens >= q8_0MinTokens(rows, cols, nTokens) ? 8 : 0;
            default:
                return 0;
        }
    }

    /** The Q5_K tensor-core form's warps for {@code nTokens} rows through a {@code rows x cols} projection, or 0. */
    private static int q5kWarps(int rows, int cols, int nTokens) {
        if (nTokens > EIGHT_TILE) {
            return rows <= Q5_K_EIGHT_WARP_ROWS ? 8 : 4;
        }
        boolean narrow = rows <= Q5_K_NARROW_ROWS && cols <= Q5_K_SHORT_COLS;
        return !narrow || nTokens >= 4 ? 4 : 0;
    }

    /** The Q6_K tensor-core form's warps for {@code nTokens} rows through a projection of {@code rows} rows, or 0. */
    private static int q6kWarps(int rows, int nTokens) {
        boolean head = rows >= HEAD_ROWS;
        if (nTokens > EIGHT_TILE) {
            return head ? 2 : 8;
        }
        if (rows < Q6_K_WIDE_ROWS) {
            return nTokens >= 6 ? 8 : 0;
        }
        return nTokens >= (head ? 3 : 4) ? 4 : 0;
    }

    /**
     * The smallest token count the Q8_0 tensor-core form serves a {@code rows x cols} projection from, inside the
     * token tile of {@code nTokens}; past {@link MatVecMmaPlan#MAX_TOKENS} where the tile never serves the shape.
     */
    private static int q8_0MinTokens(int rows, int cols, int nTokens) {
        boolean head = rows >= HEAD_ROWS;
        boolean wide = rows >= Q8_0_WIDE_ROWS;
        if (nTokens <= EIGHT_TILE) {
            if (head) {
                return cols <= Q8_0_SHORT_COLS ? 4 : 5;
            }
            if (cols <= Q8_0_SHORT_COLS) {
                return 6;
            }
            return wide && cols > rows ? MatVecMmaPlan.MAX_TOKENS + 1 : EIGHT_TILE;
        }
        if (!wide) {
            return 10;
        }
        return (long) cols > 2L * rows ? 11 : 9;
    }

    /** The kind of the NVIDIA discrete CUDA device on the sm_80+ side of the int8 tensor floor. */
    private static DeviceKind nvidiaDiscreteCudaTensor() {
        return DeviceKind.of(BackendType.CUDA, DeviceKind.Vendor.NVIDIA, DeviceClass.DISCRETE,
                DeviceKind.CudaBand.TENSOR);
    }
}
