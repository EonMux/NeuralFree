package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;

/**
 * The measured small-batch dispatch crossover of the projection lanes, per backend device kind,
 * weight type and projection size: the smallest row count the fast GEMM lane serves
 * ({@code fastGemmMinTokens}), the largest row count the float-input multi-column matvec
 * ({@code {T}_MatVecN}) serves ({@code floatMatvecNMax}) and the largest row count the integer-dot
 * multi-column matvec ({@code {T}_MatVecN_Q8}) serves ({@code int8MatvecNMax}). It replaces the
 * single global fast-lane threshold the dispatch used before: the lanes' costs, and therefore the
 * crossover row counts, differ per device and weight type (a device without the type's fast lane
 * never reaches one; the multi-column matvecs' cost grows with the row count while a GEMM lane's
 * stays flat), and a shape-independent threshold cannot pick the faster lane at every row count.
 *
 * <p>Every answer is a measured median, keyed by {@link DeviceKind}: an entry applies only to the
 * kind of device it was measured on, so one measurement per backend never governs a different
 * device of the same API. The measured machine carries one device of each kind (an NVIDIA RTX 5080
 * Laptop dGPU over CUDA, OpenCL and Vulkan; an Intel Arrow-Lake integrated GPU over Vulkan and
 * OpenCL). Kernel-level timings of one projection shape per size class ({@code 4096x1024},
 * {@code 19456x2560}) interleave the batched matvec, both multi-column matvecs and the single-row
 * matvec over the same weights at row counts 2, 4, 8, 9 and 16 — for the CUDA and OpenCL NVIDIA
 * kinds every count in [2, 16] and the MMQ lane besides (Q4_0, Q4_K, Q5_K, Q6_K, IQ4_XS) — and
 * every entry's lanes were also validated at forward-step level on the real decoders. Types and
 * device kinds without a measured entry keep the previous dispatch: the fast lane from
 * {@link #DEFAULT_FAST_GEMM_MIN_TOKENS} rows, the batched matvec below it, the integer-dot matvec
 * up to {@link LlmKernelConstants#MATVEC_N_DISPATCH_MAX} rows and no float multi-column matvec.</p>
 */
public final class SmallBatchCrossover {

    /**
     * Separates the small-model projections (every 0.6B decoder projection, at most 151936x1024
     * elements) from the 4B-class ones (smallest 2560x6144): an entry measured on one side decides
     * only that side.
     */
    public static final long SHAPE_MIN_ELEMENTS = 8_000_000L;

    /**
     * The fast GEMM lane's first served row count for projections without a measured entry: the global
     * crossover the dispatch used before (the GEMM lanes' cost is flat in the row count while the matvec
     * pays its dots per row; on an RTX 5080 Laptop over Qwen3-Embedding-4B Q8_0 with the shared-memory
     * stage pipeline, one projection-forward step through the GEMM lanes costs 27.5 ms at 2 rows and
     * stays at 27.0 to 30.0 ms from 4 to 16 rows, against 17.4 ms on the multi-column matvec at 2 rows
     * and 29.6 ms at 4 rows).
     */
    public static final int DEFAULT_FAST_GEMM_MIN_TOKENS = 4;

    /** The largest row count the float multi-column matvec dispatch can ever serve. */
    public static final int FLOAT_MATVEC_N_DISPATCH_MAX = LlmKernelConstants.MATVEC_N_MAX_TOKENS;

    /** The integer-dot multi-column matvec row cap of projections without a measured entry: the previous dispatch. */
    private static final int DEFAULT_INT8_MATVEC_N_MAX = LlmKernelConstants.MATVEC_N_DISPATCH_MAX;

    /**
     * One measured dispatch decision: the fast GEMM lane's first row count, the float multi-column
     * matvec's last and the integer-dot multi-column matvec's last.
     */
    private static final class Decision {
        final int fastMin;
        final int floatNMax;
        final int int8NMax;

        Decision(int fastMin, int floatNMax, int int8NMax) {
            this.fastMin = fastMin;
            this.floatNMax = floatNMax;
            this.int8NMax = int8NMax;
        }
    }

    private static final Map<DeviceKind, Map<GGMLType, Decision>> SMALL = decisions(true);
    private static final Map<DeviceKind, Map<GGMLType, Decision>> LARGE = decisions(false);

    private SmallBatchCrossover() {
    }

    /**
     * The smallest row count the fast GEMM lane serves for a projection of {@code type} with
     * {@code elements} weight elements (rows times columns) on {@code device}: a measured entry's
     * value, or {@link #DEFAULT_FAST_GEMM_MIN_TOKENS} without one. The dispatch only asks
     * when the weight actually has a lane.
     */
    public static int fastGemmMinTokens(DeviceKind device, GGMLType type, long elements) {
        Decision d = decision(device, type, elements);
        return d == null ? DEFAULT_FAST_GEMM_MIN_TOKENS : d.fastMin;
    }

    /**
     * The largest row count the float-input multi-column matvec serves for a projection of {@code type}
     * with {@code elements} weight elements on {@code device}: a measured entry's value, or 0 without
     * one (the dispatch keeps the batched matvec it served before).
     */
    public static int floatMatvecNMax(DeviceKind device, GGMLType type, long elements) {
        Decision d = decision(device, type, elements);
        return d == null ? 0 : d.floatNMax;
    }

    /**
     * The largest row count the integer-dot multi-column matvec serves for a projection of {@code type}
     * with {@code elements} weight elements on {@code device}: a measured entry's value, or the
     * previous dispatch's cap ({@link LlmKernelConstants#MATVEC_N_DISPATCH_MAX} rows for weights
     * without a fast lane, the rows below the fast lane's first for weights with one) without one.
     */
    public static int int8MatvecNMax(DeviceKind device, GGMLType type, long elements, boolean hasFastLane) {
        Decision d = decision(device, type, elements);
        if (d != null) {
            return d.int8NMax;
        }
        return hasFastLane ? Math.min(DEFAULT_FAST_GEMM_MIN_TOKENS - 1, DEFAULT_INT8_MATVEC_N_MAX)
                : DEFAULT_INT8_MATVEC_N_MAX;
    }

    /** The measured decision for the projection's size class and device kind, or null without one. */
    private static Decision decision(DeviceKind device, GGMLType type, long elements) {
        Map<GGMLType, Decision> m = (elements >= SHAPE_MIN_ELEMENTS ? LARGE : SMALL).get(device);
        return m == null ? null : m.get(type);
    }

    /**
     * Whether a measured entry serves the float-input multi-column matvec for {@code type} on
     * {@code device} at any projection size (the load-time preparation asks without knowing a size).
     */
    public static boolean hasFloatMatvecN(DeviceKind device, GGMLType type) {
        Decision small = decision(device, type, 0L);
        Decision large = decision(device, type, Long.MAX_VALUE);
        return (small != null && small.floatNMax > 0) || (large != null && large.floatNMax > 0);
    }

    /**
     * The measured decisions by device kind and type, for the small-model projections ({@code small})
     * and the 4B-class ones. {@code fastMin} is the fast GEMM lane's first served row count (above the
     * dispatch range where the device has no lane or the lane lost at every count), {@code floatNMax}
     * and {@code int8NMax} the multi-column matvecs' last served row counts. Entries exist only where
     * the lanes were measured, on the device kind they were measured on.
     */
    private static Map<DeviceKind, Map<GGMLType, Decision>> decisions(boolean small) {
        Map<DeviceKind, Map<GGMLType, Decision>> m = new HashMap<>();
        if (small) {
            // OpenCL NVIDIA discrete (RTX 5080 Laptop; no packed integer dot, so its MMQ fast lane
            // runs the portable integer arithmetic): the float multi-column matvec reads each weight
            // row once for the whole token group against the batched matvec's one read per row and
            // token, and measured faster or tied at every count on Q8_0 (kernel 4096x1024: 0.098 ms
            // at 2 rows against 0.099, 0.204 against 0.628 at 16) and from 8 rows up on F16 (0.097
            // against 0.138 at 8, 0.116 against 0.260 at 16; at 2 and 4 rows the batched matvec kept
            // a 0.01-0.04 ms kernel edge, noise against the 17-20 ms step). It replaces the tiled
            // GEMM the no-lane dispatch served at 8 rows and up. For Q8_0 the counted integer-dot
            // forms run the portable integer arithmetic without the packed-dot feature and measured
            // faster than the float matvec and the MMQ lane at every count of every measured shape,
            // so they serve the whole range and the float matvec stays the F16 form (1024x1024 at 9
            // tokens: 12.8us against 36.5, 4096x2560: 53.2 against 365.4, 151936x2560: 2220 against
            // 13419).
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q8_0, laneNever(), floatNMax(), int8MaxCounted());
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.F16, laneNever(), floatNMax(), int8Never());
            // The measured K-quant, legacy and i-quant types on the same kind: the counted two-row
            // forms beat the MMQ fast lane at every count of 2..8 on the 4096x1024 projection
            // (kernel medians of nine interleaved rounds: Q4_0 19.8-48.4us counted against the
            // lane's 44.3-50.1, Q4_K 21.5-59.0 against 54.8-60.9, Q5_K 21.1-56.7 against 69.2-72.8,
            // Q6_K 22.9-58.9 paired against 72.5-79.6, IQ4_XS 34.1-51.0 against 93.6-99.1). Above 8
            // rows the counted one-row forms keep winning on the small projection for Q4_K, Q5_K,
            // Q6_K and IQ4_XS (62.3-93.2 against the lane's 94.7-98.5 for Q4_K, 62.8-97.6 against
            // 94.7-99.8 for Q5_K, 57.0-89.2 against 104.8-112.2 for IQ4_XS; Q6_K wins 9..14 there,
            // 59.1-100.1 against 96.9-101.2, and drops 15..16 by 1-4%, 106.1/103.7 against
            // 102.0/102.3, still winning the counts summed 656.0 against 799.5), so they serve to
            // the counted cap; Q4_0's counted form loses from 10 rows up there (64.7-98.8 against
            // the lane's 62.5-68.8) and keeps the old cap.
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q4_0, mmq(), floatNevers(), int8Max());
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q4_K, mmq(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q5_K, mmq(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q6_K, mmq(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.IQ4_XS, mmq(), floatNevers(), int8MaxCounted());
            // Vulkan Intel Arrow-Lake iGPU: both multi-column matvecs lose to the batched matvec at
            // every measured count on Q8_0 (2 rows: batched 1.30 ms against 3.53 float and 3.10 int8;
            // 16 rows: 3.31 against 3.76 and 4.03), so the integer-dot lane stays off (it turned a
            // 520 ms 2-row 4B step into 1061 ms) and the float matvec never serves; the Q8_0
            // cooperative-matrices lane kept its step-validated crossover at 4 rows.
            put(m, intelIntegrated(BackendType.VULKAN), GGMLType.Q8_0, cm(), floatNevers(), int8Never());
            // The F16 MMA lane lost at every count 4..16 at step level (221.8 ms batched against
            // 226.6 MMA at 4 rows, 300.99 tiled against 308.78 MMA at 8) and the float matvec only
            // won at 16 rows (2.57 against 2.71 kernel ms), so the batched matvec and the tiled GEMM
            // keep the whole range.
            put(m, intelIntegrated(BackendType.VULKAN), GGMLType.F16, laneNever(), floatNevers(), int8Never());
            // CUDA NVIDIA discrete, sm_80+ (RTX 5080 Laptop, sm_120): the counted two-row multi-column
            // matvec serves the whole measured dispatch range, the exact-token-count forms beating the
            // MMQ small tile at every count of every measured shape (kernel medians of nine rounds:
            // 1024x1024 4.8-11.0us at 2-16 tokens against the small tile's 21.0-22.9, 3072x1024
            // 4.9-19.7 against 19.6-23.2, 1024x3072 5.3-17.4 against 48.5-59.4).
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q8_0, laneNever(), floatNevers(), int8MaxCounted());
            // The measured K-quant and legacy types on the same kind: the counted two-row forms beat
            // the MMQ small tile at every count of 2..16 on the 4096x1024 projection (kernel medians
            // of nine interleaved rounds, 4096x1024: Q4_0 9.8-15.8us paired against the small tile's
            // 45.8-47.4, Q4_K 8.8-18.9 against 32.6-33.4 and 21.9-35.7 against the plain tile's
            // 37.0-54.4 over 9..16, Q5_K 10.1-20.1 against 53.0-54.8 and 22.1-35.4 against 42.8-60.0,
            // Q6_K 9.3-17.0 against 42.7-43.8 and 19.6-30.8 against 47.9-68.8, IQ4_XS 10.6-18.3
            // against 56.1-56.9 and 19.8-31.0 against 42.4-60.9; Q4_0 over 9..16: paired 17.9-28.5
            // against the tile's 28.6-30.0 through 15 rows and 50.5 at 16), so the matvec serves
            // every row and the tile never takes a small projection.
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q4_K, mmq(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q6_K, mmq(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q4_0, laneNever(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q5_K, laneNever(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.IQ4_XS, laneNever(), floatNevers(), int8MaxCounted());
            // The F16 projections: the batched matvec measured fastest at 2 and 4 rows (0.023 against
            // 0.050 kernel ms at 2) and the step-validated MMA lane serves 4 rows and up (13.17 ms
            // step against 13.16 tiled at 8), the previous dispatch.
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.F16, mmq(), floatNevers(), int8Never());
            // Vulkan NVIDIA discrete (RTX 5080 Laptop): the cooperative-matrices lane keeps the
            // default crossover at four rows, and the integer-dot matvec stays off the rows below
            // it, where it measured slower than the batched matvec (step medians at 2 rows: 17.4 ms
            // against 15.1 on the 0.6B Q8_0, 37.5 against 33.2 on the 4B; kernel 4096x1024: 0.090
            // against 0.086).
            put(m, nvidiaDiscrete(BackendType.VULKAN), GGMLType.Q8_0, cm(), floatNevers(), int8Never());
            // The F16 projections: the fp16-MMA lane keeps the default crossover at four rows (step
            // medians 18.9 to 19.8 ms from 4 to 16 rows against the batched matvec's 15.0 at 2 rows)
            // and the float multi-column matvec stays off, having lost at every kernel count (0.114
            // ms against 0.077 batched at 4 rows, 0.211 against 0.114 at 16).
            put(m, nvidiaDiscrete(BackendType.VULKAN), GGMLType.F16, cm(), floatNevers(), int8Never());
            // OpenCL Intel Arrow-Lake iGPU: the integer-dot matvec keeps the rows below the MMQ
            // lane, where it measured faster than the batched matvec (step medians at 2 rows: 504.8
            // ms against 512.7 on the 0.6B Q8_0, 1249.1 against 1654.9 on the 4B), and the MMQ lane
            // keeps the default crossover at four rows: its steps stay flat (376.1 to 512.5 ms from
            // 4 to 16 rows on the 0.6B, 910.9 to 1552.8 on the 4B) while the multi-column matvec's
            // kernel cost grows with the rows (4.2 to 12.6 ms at 4096x1024, 13.7 to 80.3 at
            // 19456x2560, from 2 to 16 rows).
            put(m, intelIntegrated(BackendType.OPENCL), GGMLType.Q8_0, mmq(), floatNevers(),
                    rowsBelowFastLane());
            // The F16 projections: the float multi-column matvec serves the whole measured dispatch
            // range. The tiled GEMM the no-entry dispatch served from 8 rows collapsed on this
            // device (step medians 5411 to 5511 ms at 8, 9 and 16 rows against 562 at 4), the
            // matvec measured faster than the batched matvec from 4 rows up (3.324 against 3.406
            // kernel ms at 4, 10.700 against 13.746 at 16; at 2 rows the batched matvec kept 1.918
            // against 2.101), and the matvec-served steps validated the entry (692 to 876 ms at 8
            // and 9 rows, 1120 to 1303 at 16).
            put(m, intelIntegrated(BackendType.OPENCL), GGMLType.F16, laneNever(), floatNMax(), int8Never());
        } else {
            // The 4B-class projections on OpenCL NVIDIA discrete (RTX 5080 Laptop): no fast lane exists
            // and the float multi-column matvec reads each weight row once for the whole token group, so
            // it beat the batched matvec at every measured count on Q8_0 (kernel 19456x2560: 2.25 ms at
            // 2 rows against 5.73, 0.73 against 10.0 at 4), the lane the 72 ms 2-row step waited on.
            // The counted integer-dot forms beat it at every count too — the two-row paired form from
            // 8M elements up (9728x2560 at 9 tokens: 117.7us against 139.3, 2560x9728: 69.8 against
            // 120.9, 151936x2560: 2220 against 2579, 4096x2560: 53.2 against 60.4), and behind it the
            // plain MMQ kernel (143.8 to 3024.0us across the same shapes at 9 tokens).
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q8_0, laneNever(), floatNMax(), int8MaxCounted());
            // The 4B-class projections on the same kind (19456x2560): the counted forms beat the MMQ
            // fast lane at every count of 2..8 on every measured type (paired: Q4_0 91.7-283us
            // against the lane's 230.5-313.5, Q4_K 114.5-322.0 against 275.6-361.4, Q5_K 115.5-429.8
            // against 357.1-454.4, Q6_K 109.0-285.3 against 376.0-446.4, IQ4_XS 286.2-331.9 against
            // 552.4-592.0). From 9 to 16 rows the matvec wins the class summed over the measured
            // shapes of every type: Q8_0 and Q4_0 keep the pair everywhere large (Q4_0's 2560x9728
            // launch: paired 93.2-191.2us against the lane's 475.6-520.6), Q6_K keeps the pair
            // (2560x9728: 100.9-176.8 against the lane's 791.8-908.5; head: 2708.2-5599.0 against
            // 3924.9-4681.1 through 15 rows), and Q4_K, Q5_K and IQ4_XS flip to the counted form on
            // the tall launches from nine tokens up (see Int8MatvecTuning's pairing rule), where it
            // beats the pair on 9728x2560 and 19456x2560 (Q4_K 9728x2560 counted 221.2-388.6 against
            // the pair's 255.0-416.3 over 9..16) and holds the lane well behind (Q5_K 19456x2560
            // counted 420.9-777.3 against the lane's 534.8-616.7 at 9..13). The head losses of Q4_K
            // and Q5_K there (counted 3553.1-6092.3 and 3616.1-6361.0 against the lane's 3037.2-3648.1
            // and 3268.6-3978.4) are one launch per step against the per-layer launches the lane loses
            // by 3x on 2560x9728 (Q4_K 129.9-208.1 against 730.0-805.4), so the counted cap rises
            // over the dispatched range on every type — to one row below the counted cap: the
            // 16-row prefill measured the lane faster at step level (4B Q4_K_M medians of two
            // passes per dispatch: the lane's 71.1 against the matvec's 75.1; its kernel medians
            // too take 9728x2560, 19456x2560 and the head at 16: 327.6, 616.2 and 3648.1 against
            // the counted forms' 388.6, 941.3 and 6092.3), so 16 rows stay with the lane and 9..15
            // stay with the matvec, whose 9-row step measured 51.0 against the lane's 65.2.
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q4_0, mmq(), floatNevers(), int8MaxCountedToStep());
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q4_K, mmq(), floatNevers(), int8MaxCountedToStep());
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q5_K, mmq(), floatNevers(), int8MaxCountedToStep());
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.Q6_K, mmq(), floatNevers(), int8MaxCountedToStep());
            put(m, nvidiaDiscrete(BackendType.OPENCL), GGMLType.IQ4_XS, mmq(), floatNevers(), int8MaxCountedToStep());
            // The 4B-class projections on CUDA NVIDIA discrete, sm_80+ (RTX 5080 Laptop, sm_120): the
            // counted two-row multi-column matvec beats the MMQ small tile at every count of every
            // measured shape (kernel medians of nine rounds, 2 to 16 tokens: 4096x2560 9.2-37.6us
            // against the small tile's 57.7-62.5, 9728x2560 18.4-83.6 against 79.8-92.0, 2560x9728
            // 16.7-81.9 against 146.1-180.0, 151936x2560 668.9-1373.7 against 1331.9-1492.9), so the
            // MMQ lane keeps only the rows above the multi-column dispatch cap.
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q8_0, laneNever(), floatNevers(), int8MaxCounted());
            // The 4B-class projections on the same kind (19456x2560): the counted forms beat the MMQ
            // small tile at every count of 2..8 on every measured type (paired: Q4_0 36.5-87.7us
            // against the small tile's 227.6-240.3, Q4_K 52.0-128.7 against 197.4-215.7, Q5_K
            // 53.3-143.1 against 251.8-271.4, Q6_K 51.5-118.8 against 251.4-279.3, IQ4_XS 67.0-122.3
            // against 343.0-381.0), and from 9 to 16 rows the paired forms win the class summed over
            // the measured shapes of every type — Q6_K and IQ4_XS win every shape at every count —
            // with the tile's per-count wins confined to Q4_K's 12..15 rows on 9728x2560 and
            // 19456x2560 (113.6-115.4 and 209.0-227.1 against the matvec's 116.9-136.6 and
            // 220.5-288.6), its 10..15 on the head (1374.3-1417.1 against 1450.8-2194.6) and Q4_0's
            // head at 13..15 (1105.0-1162.6 against 1194.1-1398.2), each a minority of the class's
            // launches that the matvec's medians still win summed (Q4_K 2560x9728 74.3-130.2 against
            // the tile's 248.4-420.3, head Q8_0 858.8-1418.8 against 1347.3-1963.2, Q6_K 985.0-1613.4
            // against 1996.3-2950.0), so the counted cap rises to the full range on every type.
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q4_0, mmq(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q4_K, mmq(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q5_K, mmq(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.Q6_K, mmq(), floatNevers(), int8MaxCounted());
            put(m, nvidiaDiscreteCudaTensor(), GGMLType.IQ4_XS, mmq(), floatNevers(), int8MaxCounted());
            // The 4B-class projections on Vulkan Intel Arrow-Lake iGPU: both multi-column matvecs lose
            // to the batched matvec at every measured count (2 rows: batched 4.69 kernel ms against 8.50
            // float and 9.18 int8), so the integer-dot lane stays off; the cooperative-matrices lane
            // keeps its step-validated crossover at 4 rows.
            put(m, intelIntegrated(BackendType.VULKAN), GGMLType.Q8_0, cm(), floatNevers(), int8Never());
            // The 4B-class projections on Vulkan NVIDIA discrete (RTX 5080 Laptop): the
            // cooperative-matrices lane keeps the default crossover at four rows (49.6 to 54.5 ms
            // step medians at 4 to 16 rows) and the integer-dot matvec stays off the rows below it,
            // where the batched matvec measured faster (33.2 ms step at 2 rows against 37.5); the
            // matvec's kernel cost grows 0.291 to 1.070 ms from 2 to 16 rows against the lane's
            // flat 49.6 at 4.
            put(m, nvidiaDiscrete(BackendType.VULKAN), GGMLType.Q8_0, cm(), floatNevers(), int8Never());
        }
        return Collections.unmodifiableMap(m);
    }

    /** The kind of the NVIDIA discrete GPUs (the RTX 5080 Laptop), outside the CUDA banding. */
    private static DeviceKind nvidiaDiscrete(BackendType backend) {
        return DeviceKind.of(backend, DeviceKind.Vendor.NVIDIA, DeviceClass.DISCRETE, DeviceKind.CudaBand.NONE);
    }

    /** The kind of the Intel integrated GPUs (the Arrow-Lake iGPU). */
    private static DeviceKind intelIntegrated(BackendType backend) {
        return DeviceKind.of(backend, DeviceKind.Vendor.INTEL, DeviceClass.INTEGRATED, DeviceKind.CudaBand.NONE);
    }

    /** The kind of the NVIDIA discrete CUDA device on the sm_80+ side of the int8 tensor floor. */
    private static DeviceKind nvidiaDiscreteCudaTensor() {
        return DeviceKind.of(BackendType.CUDA, DeviceKind.Vendor.NVIDIA, DeviceClass.DISCRETE,
                DeviceKind.CudaBand.TENSOR);
    }

    /** The fast lane never serves inside the dispatch range (the device has no lane, or it lost everywhere). */
    private static int laneNever() {
        return FLOAT_MATVEC_N_DISPATCH_MAX + 1;
    }

    /** The float multi-column matvec serves the whole measured dispatch range. */
    private static int floatNMax() {
        return FLOAT_MATVEC_N_DISPATCH_MAX;
    }

    /** The integer-dot multi-column matvec never serves. */
    private static int int8Never() {
        return 1;
    }

    /** The integer-dot multi-column matvec serves its whole counted range. */
    private static int int8Max() {
        return LlmKernelConstants.MATVEC_N_DISPATCH_MAX;
    }

    /**
     * The integer-dot multi-column matvec serves the whole counted range to
     * {@link LlmKernelConstants#MATVEC_N_COUNTED_MAX} rows: the measured answers where the exact-count
     * forms beat the fast lane at every count above the dispatched range too.
     */
    private static int int8MaxCounted() {
        return LlmKernelConstants.MATVEC_N_COUNTED_MAX;
    }

    /**
     * Highest row count the counted multi-column matvec serves on the OpenCL NVIDIA large side: one
     * below the counted cap, whose 16-row prefill measured the fast lane faster at step level (the
     * 4B Q4_K_M step medians: 71.1 against the matvec dispatch's 75.1ms; see the large-side entries).
     */
    private static int int8MaxCountedToStep() {
        return LlmKernelConstants.MATVEC_N_COUNTED_MAX - 1;
    }

    /** The float multi-column matvec never serves. */
    private static int floatNevers() {
        return 0;
    }

    /** The integer-dot multi-column matvec serves the rows below the fast lane's first. */
    private static int rowsBelowFastLane() {
        return DEFAULT_FAST_GEMM_MIN_TOKENS - 1;
    }

    /** The Vulkan cooperative-matrices lane's measured first row count. */
    private static int cm() {
        return DEFAULT_FAST_GEMM_MIN_TOKENS;
    }

    /** The MMQ lane's measured first row count. */
    private static int mmq() {
        return DEFAULT_FAST_GEMM_MIN_TOKENS;
    }

    private static void put(Map<DeviceKind, Map<GGMLType, Decision>> m, DeviceKind kind, GGMLType type,
                            int fastMin, int floatNMax, int int8NMax) {
        m.computeIfAbsent(kind, k -> new EnumMap<>(GGMLType.class)).put(type, new Decision(fastMin, floatNMax, int8NMax));
    }
}
