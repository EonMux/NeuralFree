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
 * the 4B and 0.6B Qwen3 projection shapes at 4, 8, 9 and 16 tokens (2 also for Q6_K). The tensor-core form's
 * cost is flat in the token count inside a token tile while the counted forms grow per token, so each rule is
 * a smallest token count:</p>
 * <ul>
 *   <li>Q4_K from 4 tokens on every shape, at 4 warps (9728x2560 at 4: 26.4 us against the pair's 46.1; at 16:
 *       43.4 against 144.8; the 2560-column head at 16: 510 against 2250).</li>
 *   <li>Q6_K from 8 tokens at 8 warps (2560x9728 at 8: 51.6 against 67.2; at 16: 56.2 against 95.0), the head
 *       from 9 tokens at 2 warps (151936x2560 at 16: 801 against 1617).</li>
 *   <li>Q8_0 through the 16-token form from 9 tokens on projections of at least {@value #Q8_0_WIDE_ROWS} rows
 *       (9728x2560 at 9: 49.2 against 59.2; at 16: 59.5 against 99.8), from 11 where the reduction is more
 *       than twice the row count (2560x9728 at 9: 61.9 against 53.3; at 16: 67.8 against 93.8) and from 10 on
 *       narrower projections (1024x2560 at 9: 13.6 against 12.7; at 16: 12.7 against 20.8); at 8 warps. Through
 *       the 8-token form only a head over a reduction of at most {@value #Q8_0_SHORT_HEAD_COLS} columns, from 4
 *       tokens (151936x1024 at 8: 233 against 530); the 2560-column head keeps the pair there (893 against
 *       771).</li>
 * </ul>
 */
public final class MatVecMmaTuning {

    /** Rows from which a projection reads as an output head. */
    static final int HEAD_ROWS = 65536;
    /** Rows from which the Q8_0 16-token form serves from 9 tokens. */
    static final int Q8_0_WIDE_ROWS = 2048;
    /** Widest reduction of a head the Q8_0 8-token form serves. */
    static final int Q8_0_SHORT_HEAD_COLS = 1024;

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
                return nTokens >= 4 ? 4 : 0;
            case Q6_K:
                if (nTokens < 8) {
                    return 0;
                }
                return rows >= HEAD_ROWS && nTokens >= 9 ? 2 : 8;
            case Q8_0:
                return nTokens >= q8_0MinTokens(rows, cols) ? 8 : 0;
            default:
                return 0;
        }
    }

    /** The smallest token count the Q8_0 tensor-core form serves a {@code rows x cols} projection from. */
    private static int q8_0MinTokens(int rows, int cols) {
        if (rows >= HEAD_ROWS && cols <= Q8_0_SHORT_HEAD_COLS) {
            return 4;
        }
        if (rows < Q8_0_WIDE_ROWS) {
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
