package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * The decode evidence behind {@link Int8MatvecMode#AUTO}: per {@link DeviceKind}, the weight types
 * whose integer-dot (Q8_1 activation) matvec measured slower than the F32-activation matvec, so AUTO
 * keeps the exact float path for them even where the device reports
 * {@link com.eonmux.neuralfree.acceleration.backend.DeviceFeature#INTEGER_DOT_PRODUCT}. Every type a
 * device kind does not list keeps the lane there, and a kind without an entry keeps the feature
 * decision (the device's packed dot product support) for every type, exactly as a device whose
 * vendor or class nobody measured does.
 *
 * <p>Two kinds of loss are recorded. A type in {@link #SLOWER} measured slower at every projection
 * shape, so it never takes the lane. A type in {@link #LARGE_ONLY} measured slower on the small
 * projections of a 0.6B-class decoder and faster on 4096-column projections of a 4B/9B-class one, so
 * it takes the lane only from {@link #LANE_MIN_ELEMENTS} elements (weight rows times columns) up: the
 * lane's fixed cost (the quantizer launch and its own launch floor) does not shrink with the
 * projection, while the F32 matvec's activation reads grow with it. {@link Int8MatvecMode#ON} runs the
 * lane regardless of this table.</p>
 *
 * <p>The entries are single-token matvec times of one projection, measured on one device per kind
 * over the shapes of a 0.6B decoder (1024x1024, 3072x1024, 1024x3072), a 4B decoder (9728x2560,
 * 2560x9728) and 7B/8B decoders (4096x4096, 14336x4096, 32768x1024), with and without the Q8_1
 * quantization the lane adds, in interleaved rounds of alternating order so the device's two clock
 * states and an iGPU's throttling bias neither form. The NVIDIA RTX 5080 Laptop GPU contributes three
 * kinds, one per backend: its CUDA device (sm_120, the tensor band), its OpenCL device, which reports
 * no packed dot product so the lane only ran forced there, and its Vulkan device, which reports the
 * feature. The Intel Arrow-Lake iGPU contributes its Vulkan kind (no packed dot product, so only the
 * type whose lane measured slower there is listed) and its OpenCL kind (intel-compute-runtime, which
 * reports the feature). A shape's median of at most 0.97x records a loss and a median of at least
 * 1.02x records a win; the boundary shapes that measured inside that band are named in the entry
 * comments. One entry is recorded against the isolated shapes: Q8_0 on the OpenCL iGPU won up to
 * 1.11x at the 16.8M+ projections in isolation but the whole 4B decoder step measured 0.75-0.85x
 * with the lane there, so the step evidence records it never the lane (the iGPU's quantizer alone
 * costs about 100-150us per launch, against about 14-20us on the RTX 5080).</p>
 */
public final class Int8MatvecTuning {

    /**
     * Smallest projection, in weight elements (rows times columns), at which a {@link #LARGE_ONLY} type
     * measured the lane faster than the F32 matvec: above the 0.6B-class projections where the lane's
     * fixed quantizer cost loses or ties (1024x1024, 3072x1024, 1024x3072) and no larger than the
     * smallest winning shape (4096x4096).
     */
    public static final long LANE_MIN_ELEMENTS = 4_000_000L;

    /**
     * Highest token count the OpenCL NVIDIA kind's large side of {@link #PAIRED_COUNTED_LARGE_ONLY}
     * serves through the paired counted form; from nine tokens up the tall projections of
     * {@link #COUNTED_PAST_EIGHT_TALL} measured the counted one-row form faster (see
     * {@link #pairedCountedIsFaster(DeviceKind, GGMLType, long, int, int)}).
     */
    public static final int PAIRED_THROUGH_TOKENS = 8;

    /**
     * Smallest weight-row count from which the OpenCL NVIDIA kind's {@link #COUNTED_PAST_EIGHT_TALL}
     * types measured the counted one-row form faster than the pair at nine or more tokens: the
     * 4B-class tall projections (9728, 19456 and 151936 rows) win counted, the 2560-row projections
     * (2560x9728) keep the pair.
     */
    public static final int COUNTED_PAST_EIGHT_MIN_ROWS = 9728;

    /**
     * The OpenCL NVIDIA types whose tall 4B-class projections measured the counted one-row form faster
     * than the pair from nine tokens up ({@link #PAIRED_THROUGH_TOKENS}): Q4_K, Q5_K and IQ4_XS. Q8_0,
     * Q4_0 and Q6_K keep the pair on every large shape and count there, and every CUDA shape and count
     * pairs.
     */
    private static final Set<GGMLType> COUNTED_PAST_EIGHT_TALL =
            Collections.unmodifiableSet(EnumSet.of(GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.IQ4_XS));

    private static final Map<DeviceKind, Set<GGMLType>> SLOWER = slower();
    private static final Map<DeviceKind, Set<GGMLType>> LARGE_ONLY = largeOnly();
    private static final Map<DeviceKind, Set<GGMLType>> PAIRED_SLOWER = pairedSlower();
    private static final Map<DeviceKind, Set<GGMLType>> PAIRED_COUNTED_LARGE_ONLY = pairedCountedLargeOnly();

    private Int8MatvecTuning() {
    }

    /**
     * Whether the lane measured faster than the F32 matvec for {@code type} on {@code device} at
     * every projection shape. Types and device kinds without measured evidence answer {@code true}:
     * the device's packed dot product support then decides, as it does for every type.
     */
    public static boolean laneIsFaster(DeviceKind device, GGMLType type) {
        Set<GGMLType> slower = SLOWER.get(device);
        return slower == null || !slower.contains(type);
    }

    /**
     * Whether the lane measured faster than the F32 matvec for a projection of {@code elements} weight
     * elements (rows times columns): false for a type {@link #laneIsFaster} rules out, and false below
     * {@link #LANE_MIN_ELEMENTS} for a type that only measured faster on the larger projections.
     */
    public static boolean laneIsFaster(DeviceKind device, GGMLType type, long elements) {
        if (!laneIsFaster(device, type)) {
            return false;
        }
        Set<GGMLType> largeOnly = LARGE_ONLY.get(device);
        return largeOnly == null || !largeOnly.contains(type) || elements >= LANE_MIN_ELEMENTS;
    }

    /**
     * Whether the counted multi-column lane ({@code {T}_MatVecN_Q8} and its exact-token-count and
     * two-row forms) measured faster than the lanes it replaces for {@code type} on {@code device} at a
     * projection of {@code elements} weight elements. The single-row lane's small-projection losses
     * ({@link #laneIsFaster}) record the single-token matvec's cost against the batched matvec and the
     * float matvec; the counted forms answer a different comparison, and the measured kinds override it
     * where the comparison came out in the lane's favour. On the CUDA kind the Q8_0 small projections
     * measured the counted forms faster than the tiled GEMM and the MMQ small tile they replace
     * (kernel medians of nine rounds, 2 to 16 tokens: 1024x1024 4.8-11.0us against the small tile's
     * 21.0-22.9, 3072x1024 4.9-19.7 against 19.6-23.2, 1024x3072 5.3-17.4 against 48.5-59.4), and the
     * same held for Q4_0 there from {@link #LANE_MIN_ELEMENTS} up (4096x1024 at 2 to 8 tokens: the
     * paired forms 9.8-15.8us against the small tile's 45.8-47.4, 19456x2560: 36.5-87.7 against
     * 227.6-240.3, the runtime-count form's 17.9-222.0 between them). On the OpenCL NVIDIA kind —
     * whose device reports no packed dot product, so the counted forms run the portable integer
     * arithmetic — they measured faster than the float-input multi-column matvec they replace at every
     * count of every measured shape (1024x1024 at 9 tokens: 12.8 against 36.5us, 4096x2560: 53.2
     * against 365.4, 151936x2560: 2220 against 13419), and the same held for Q4_0 and IQ4_XS there,
     * whose single-row lane lost ({@link #SLOWER}), from {@link #LANE_MIN_ELEMENTS} up (4096x1024 at
     * 2 to 8 tokens: Q4_0 counted 19.8-48.4us against the runtime-count form's 74.5-87.3 and the MMQ
     * kernel's 44.3-50.1, IQ4_XS counted 34.1-51.0 against 98.6-244.3 and 93.6-99.1; 19456x2560:
     * Q4_0 paired 91.7-280.7 against 364.7-559.6, IQ4_XS counted 296.7-341.0 against 682.6-2224.8
     * and 552.4-592.0). Every other kind and type inherits the single-row answer.
     */
    public static boolean countedLaneIsFaster(DeviceKind device, GGMLType type, long elements) {
        if (type == GGMLType.Q8_0 && (device.equals(nvidiaDiscreteCudaTensor())
                || device.equals(nvidiaDiscrete(BackendType.OPENCL)))) {
            return true;
        }
        return (elements >= LANE_MIN_ELEMENTS && countedFormsRedeem(device, type))
                || laneIsFaster(device, type, elements);
    }

    /**
     * Whether the counted multi-column forms measured the lane faster on {@code device} for a
     * {@code type} whose single-row lane lost there: Q4_0 on the CUDA kind and on the OpenCL NVIDIA
     * kind, and IQ4_XS on the OpenCL NVIDIA kind (see {@link #countedLaneIsFaster} for the numbers).
     */
    private static boolean countedFormsRedeem(DeviceKind device, GGMLType type) {
        boolean q4_0 = type == GGMLType.Q4_0 && (device.equals(nvidiaDiscreteCudaTensor())
                || device.equals(nvidiaDiscrete(BackendType.OPENCL)));
        boolean iq4_xs = type == GGMLType.IQ4_XS && device.equals(nvidiaDiscrete(BackendType.OPENCL));
        return q4_0 || iq4_xs;
    }

    /**
     * Whether the counted multi-column lane's measured wins on this kind and type rest on the device's
     * packed integer dot product, which AUTO otherwise requires of the lane: true for every kind whose
     * counted lane has no measured evidence of its own, and false on the OpenCL NVIDIA kind — whose
     * device reports no packed dot product — for Q8_0 and for the K-quant, legacy and i-quant types the
     * measured shapes covered (Q4_0, Q4_K, Q5_K, Q6_K, IQ4_XS), whose exact-token-count forms run the
     * portable integer arithmetic ({@code nf_mvq8_slice_idot}'s byte-wide emulation, the arithmetic the
     * kind's forced single-row lane already used) and still measured faster than the multi-column form
     * they replace at every count of every measured shape (4096x1024 at 2 to 8 tokens, medians of nine
     * rounds: Q4_0 19.8-48.4us against the runtime-count form's 74.5-87.3, Q4_K 21.5-55.0 against
     * 67.9-87.3, Q5_K 22.5-56.7 against 84.2-102.1, Q6_K 22.7-56.9 against 79.4-102.4, IQ4_XS
     * 35.1-51.1 against 101.3-244.3; 19456x2560 they won at every count, 1.8-6.5x over the
     * runtime-count form) — the
     * device's missing packed-dot feature does not govern a lane that does not use packed dots.
     */
    public static boolean countedLaneNeedsPackedDots(DeviceKind device, GGMLType type) {
        return !(device.equals(nvidiaDiscrete(BackendType.OPENCL))
                && (type == GGMLType.Q8_0 || type == GGMLType.Q4_0 || type == GGMLType.Q4_K
                || type == GGMLType.Q5_K || type == GGMLType.Q6_K || type == GGMLType.IQ4_XS));
    }

    /**
     * Whether the paired integer-dot matvec ({@code {T}_MatVec2_Q8}) measured faster than the single-row
     * form ({@code {T}_MatVec_Q8}) for {@code type} on {@code device}: the two-row form halves the Q8_1
     * row re-reads but adds a second work-group reduction, which on the RTX 5080's CUDA device costs the
     * K-quant weights more than the shared row saves (Q4_K and Q5_K lost 1-8% paired, Q8_K and Q8_0
     * measured neutral) and on its OpenCL device only Q8_K and Q8_0 failed to gain; on its Vulkan device
     * Q8_K measured neutral (0.95-1.09x, the others gained up to 32%), and on the Arrow-Lake iGPU's
     * OpenCL device Q5_K lost 10-14% and Q6_K 16-22% at every paired shape while Q8_0 measured neutral
     * (0.99-1.05x). Types and device kinds without measured evidence answer {@code true}.
     */
    public static boolean pairedIsFaster(DeviceKind device, GGMLType type) {
        Set<GGMLType> slower = PAIRED_SLOWER.get(device);
        return slower == null || !slower.contains(type);
    }

    /**
     * Whether the counted two-row multi-column matvec ({@code {T}_MatVec2N<n>_Q8}, one work-group per
     * row pair over the exact {@code nTokens} in [2, 16]) measured faster than the counted one-row form
     * ({@code {T}_MatVecN<n>_Q8}) for {@code type} on {@code device} at a launch of {@code elements}
     * weight elements, {@code tokens} rows of the activation and {@code rows} weight rows. The pairing
     * halves the Q8_1 row re-reads, which pays from the 4B-class projections up (medians of nine
     * interleaved rounds of both forms over the same weights, Q8_1 rows of one token count, on the
     * RTX 5080 Laptop: its CUDA device paired every 4B-class shape faster at every count of 2, 4, 8,
     * 9 and 16 (9728x2560 at 9 tokens: 49.4 against 67.0us, 151936x2560: 836 against 1056) and stayed
     * within launch noise at the 0.6B-class ones (plus or minus 9% of a 5us floor)), while on its
     * OpenCL device the 0.6B-class projections measured ties at 3.1M elements and 13-18% losses at
     * 1024x1024 (the second work-group reduction costs the launch-bound small shapes more than the
     * shared activation row saves), so that kind pairs only the large side of the size split. On that
     * kind the tall 4B-class projections of {@link #COUNTED_PAST_EIGHT_TALL} measured the counted
     * one-row form faster than the pair from nine tokens up (Q4_K 9728x2560 at 16 tokens: 388.6
     * against the pair's 416.3us, 19456x2560 at 15: 701.2 against 776.1, IQ4_XS 19456x2560 at 16:
     * 554.8 against 614.9), while the same types' 2560-row projections kept the pair (Q4_K 2560x9728
     * at 9 tokens: 129.9 against the counted form's 168.0) — 2560x9728 and 9728x2560 share one
     * element count with opposite winners, so the launch's row count and token count separate them,
     * and those launches pair only through {@link #PAIRED_THROUGH_TOKENS} tokens or below
     * {@link #COUNTED_PAST_EIGHT_MIN_ROWS} rows. Types and device kinds without measured evidence
     * answer {@code true}.
     */
    public static boolean pairedCountedIsFaster(DeviceKind device, GGMLType type, long elements,
                                                int tokens, int rows) {
        Set<GGMLType> largeOnly = PAIRED_COUNTED_LARGE_ONLY.get(device);
        if (largeOnly == null || !largeOnly.contains(type)) {
            return true;
        }
        if (elements < SmallBatchCrossover.SHAPE_MIN_ELEMENTS) {
            return false;
        }
        if (!device.equals(nvidiaDiscrete(BackendType.OPENCL))
                || !COUNTED_PAST_EIGHT_TALL.contains(type)) {
            return true;
        }
        return tokens <= PAIRED_THROUGH_TOKENS || rows < COUNTED_PAST_EIGHT_MIN_ROWS;
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

    /**
     * The types that measured the lane slower at every projection shape, per device kind.
     */
    private static Map<DeviceKind, Set<GGMLType>> slower() {
        Map<DeviceKind, Set<GGMLType>> slower = new HashMap<>();
        // OpenCL RTX 5080 (lane forced, the device reports no packed dot product).
        slower.put(nvidiaDiscrete(BackendType.OPENCL), EnumSet.of(GGMLType.MXFP4, GGMLType.NVFP4,
                GGMLType.IQ4_NL, GGMLType.IQ4_XS));
        slower.put(nvidiaDiscreteCudaTensor(), EnumSet.of(GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0,
                GGMLType.Q5_1, GGMLType.IQ4_NL));
        // Vulkan Arrow-Lake iGPU (no packed dot product, the lane forced).
        slower.put(intelIntegrated(BackendType.VULKAN), EnumSet.of(GGMLType.Q4_0));
        // Vulkan RTX 5080: no type lost at every shape; its Vulkan {T}_MatVec_Q8 kernels win from
        // 16.8M elements up for all fifteen types, so this kind records only largeOnly losses.
        // OpenCL Arrow-Lake iGPU (intel-compute-runtime; the lane's quantizer alone costs 100-150us
        // per launch there): the Q4_1, Q5_1, Q3_K, MXFP4 and IQ4_XS emulations lost at every shape
        // (0.70-0.99x, Q5_1's 14336x4096 the one 0.99x), Q5_0 and IQ4_NL lost the small shapes and
        // tied the large ones across two runs (0.99-1.06x), and Q8_0's isolated 16.8M+ wins
        // (1.02-1.11x) did not survive the decoder step, which measured 0.75-0.85x with the lane
        // on the Q8_0 4B model.
        slower.put(intelIntegrated(BackendType.OPENCL), EnumSet.of(GGMLType.Q4_1, GGMLType.Q5_0,
                GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q3_K, GGMLType.MXFP4, GGMLType.IQ4_NL,
                GGMLType.IQ4_XS));
        return Collections.unmodifiableMap(slower);
    }

    /**
     * The types that measured the lane slower on the small projections of a 0.6B-class decoder and
     * faster from 4096 columns up, per device kind: the projections of the 0.6B model (1024 columns)
     * lose the lane's fixed quantizer cost, while the 4B/9B-class 4096-column projections gain from
     * the smaller activation row and the packed dot product.
     */
    private static Map<DeviceKind, Set<GGMLType>> largeOnly() {
        Map<DeviceKind, Set<GGMLType>> large = new HashMap<>();
        large.put(nvidiaDiscrete(BackendType.OPENCL), EnumSet.of(GGMLType.Q4_0, GGMLType.Q4_1,
                GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K,
                GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K));
        large.put(nvidiaDiscreteCudaTensor(), EnumSet.of(GGMLType.MXFP4, GGMLType.NVFP4, GGMLType.Q8_0,
                GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K,
                GGMLType.IQ4_XS));
        // Vulkan RTX 5080: every type lost the fixed quantizer cost at 1024x1024 (0.61-0.95x medians)
        // and won from 16.8M up (1.20-3.19x); Q8_K also lost its one widest shape, 14336x4096, at
        // 0.96x, the single large-shape miss in the kind.
        large.put(nvidiaDiscrete(BackendType.VULKAN), EnumSet.of(GGMLType.Q4_0, GGMLType.Q4_1,
                GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.MXFP4, GGMLType.NVFP4,
                GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
                GGMLType.Q8_K, GGMLType.IQ4_NL, GGMLType.IQ4_XS));
        // OpenCL Arrow-Lake iGPU: at or below parity on the small projections (Q6_K and Q8_K only
        // lost at 1024x1024, the others measured 0.77-1.09x across two runs) and 1.01-1.51x from
        // 16.8M up; the boundary shapes inside the noise band are Q2_K's 32768x1024 (0.99x) and
        // Q4_K and Q5_K at 16.8M (1.01x each).
        large.put(intelIntegrated(BackendType.OPENCL), EnumSet.of(GGMLType.Q4_0, GGMLType.Q2_K,
                GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K));
        return Collections.unmodifiableMap(large);
    }

    /**
     * The types whose paired form measured slower than the single-row form, per device kind:
     * interleaved rounds of both forms over the same weights and Q8_1 row at 1024 to 32768 output rows
     * of 4096 and 12288 columns (median of nine rounds). Q4_K and Q5_K lost 1-8% on the RTX 5080's
     * CUDA device (medians 0.95x and 0.96x) and Q8_K and Q8_0 measured neutral there (Q8_0 medians
     * 0.99x, at best 1.00x); on its OpenCL device Q8_K and Q8_0 never gained (Q8_0 median 0.97x, at
     * best 1.02x); the CUDA losers gained 2-13% there.
     */
    private static Map<DeviceKind, Set<GGMLType>> pairedSlower() {
        Map<DeviceKind, Set<GGMLType>> slower = new HashMap<>();
        slower.put(nvidiaDiscrete(BackendType.OPENCL), EnumSet.of(GGMLType.Q8_0, GGMLType.Q8_K));
        slower.put(nvidiaDiscreteCudaTensor(), EnumSet.of(GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q8_0,
                GGMLType.Q8_K));
        // Vulkan RTX 5080: Q8_K measured neutral (0.95-1.09x) while Q8_0 gained 4-8% at the 16.8-50M
        // shapes and the other K-quants gained throughout (up to 1.32x at 134M).
        slower.put(nvidiaDiscrete(BackendType.VULKAN), EnumSet.of(GGMLType.Q8_K));
        // OpenCL Arrow-Lake iGPU: Q5_K lost 10-14% and Q6_K 16-22% at every paired shape; Q8_0
        // measured neutral (0.99-1.05x).
        slower.put(intelIntegrated(BackendType.OPENCL), EnumSet.of(GGMLType.Q8_0, GGMLType.Q5_K,
                GGMLType.Q6_K));
        return Collections.unmodifiableMap(slower);
    }

    /**
     * The types whose counted two-row form measured slower than the counted one-row form on the
     * small-projection side of {@link SmallBatchCrossover#SHAPE_MIN_ELEMENTS} and faster on the large
     * side, per device kind (the CUDA kind of the RTX 5080 paired every measured shape and records
     * nothing; the OpenCL Arrow-Lake iGPU and the Vulkan devices were not measured for this form).
     */
    private static Map<DeviceKind, Set<GGMLType>> pairedCountedLargeOnly() {
        Map<DeviceKind, Set<GGMLType>> largeOnly = new HashMap<>();
        // OpenCL RTX 5080 (4096x1024 and 19456x2560, medians of nine interleaved rounds):
        // Q8_0's 1024x1024 lost 13-18% at 8, 9 and 16 tokens (14.3 against 11.9us at 8), 3072x1024
        // and 1024x3072 measured ties (31.8 against 31.3 and 18.9 against 18.9us at 9); the 4B-class
        // shapes gained (9728x2560 at 9 tokens: 117.7 against 139.3us, 2560x9728: 69.8 against
        // 120.9, 151936x2560: 2220 against 2579). Q4_0's small side tied at 2 tokens (19.5 against
        // 19.8us) and lost from 4 up (55.5 against 48.4 at 8), its large side won at every count
        // (91.7 against 101.1 at 2, 280.7 against 283.0 at 8); Q4_K's small side lost at every count
        // (23.8 against 21.5 at 2, 61.6 against 59.0 at 8) while its large side split the counts
        // (114.5 against 116.4 at 2, 322.0 against 356.7 at 8, 222.2 against 201.7 at 5); Q5_K's
        // small side lost from 5 up (69.5 against 51.3 at 7, 76.4 against 56.7 at 8), its large side
        // gained through 4 tokens and lost the high counts (115.5 against 123.4 at 2, 429.8 against
        // 376.0 at 8); IQ4_XS's small side lost 1-6% from 4 up (52.2 against 51.1 at 8), its large
        // side won at every count (286.2 against 296.7 at 2, 331.9 against 340.6 at 8). Q6_K paired
        // every count of both shapes (58.9 against 61.3 small at 8, 285.3 against 346.6 large at 8)
        // and records nothing.
        //
        // At 9..16 tokens (same protocol over the 4B shapes and the head) the pairing flips toward
        // the counted one-row form on the tall projections of Q4_K, Q5_K and IQ4_XS — the token loop
        // holds nine or more live accumulators per row and the second reduction stops paying (Q4_K
        // 9728x2560: counted 221.2-388.6 against the pair's 255.0-416.3 at 9..16; Q5_K 9728x2560:
        // 237.9-404.7 against 264.7-453.9; IQ4_XS 19456x2560: 554.8 against 614.9 at 16) — while the
        // same types keep the pair on the 2560-row projections (Q4_K 2560x9728 at 9..16: the pair
        // 129.9-202.6 against 168.0-284.0) and on the head's low counts. 2560x9728 and 9728x2560
        // share one element count with opposite winners, so only the launch's row count separates
        // them; Q8_0, Q4_0 and Q6_K keep the pair on every large shape and count (Q6_K 2560x9728 at
        // 9: 64.1 against 69.1; Q8_0 19456x2560 at 9: 178.8 against 192.8). See
        // {@link #pairedCountedIsFaster(DeviceKind, GGMLType, long, int, int)}.
        largeOnly.put(nvidiaDiscrete(BackendType.OPENCL), EnumSet.of(GGMLType.Q8_0, GGMLType.Q4_0,
                GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.IQ4_XS));
        return Collections.unmodifiableMap(largeOnly);
    }
}
