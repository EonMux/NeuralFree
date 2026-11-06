package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/**
 * The measured evidence behind {@link CertifiedArgmaxMode}: the (backend, head type) pairs whose
 * certified exact-argmax readout measured faster than the exact readout ({@link #FASTER}, the pairs
 * {@code auto} serves), the pairs whose kernels exist without a measured win ({@link #CAPABLE}, the
 * pairs {@code on} forces), and the candidate cap one exact rescore covers ({@link #DEFAULT_KMAX}).
 *
 * <p>The numbers are single-token LM-head readouts of the real heads
 * {@code pplx-embed-v1-0.6b-f16.gguf} (F16 151936x1024), {@code Qwen3-Embedding-0.6B-Q8_0.gguf}
 * (Q8_0 151669x1024) and {@code Qwen3-Embedding-4B-Q8_0.gguf} (Q8_0 151665x2560) plus a synthetic tied
 * BF16 head of 151936x1024, as medians of 60 enqueued readouts after 20 warmups, both paths including
 * the four-byte token-id readback; end-to-end greedy decode agreed in direction where its 40-token
 * repetitions were quiet enough to read. Measured on one machine: the NVIDIA RTX 5080 Laptop GPU
 * through its OpenCL (NVIDIA platform) and CUDA backends, and through Vulkan its Intel Arrow-Lake
 * integrated GPU ("Intel(R) Graphics (ARL)"), and the host CPU for the CPU path. A pair
 * {@link #FASTER} measured a certified median below 1.0x there; a pair with no measurement or a
 * measured loss stays with the exact readout under {@code auto} and remains reachable through
 * {@code on} where {@link #CAPABLE} lists it.</p>
 *
 * <p>Readout medians, exact versus certified, in microseconds, over two runs a day-load apart:</p>
 * <ul>
 *   <li>OpenCL on the RTX 5080: F16 151936x1024 945.7 &rarr; 914.8 (0.97) and 1204.2 &rarr; 1158.5
 *       (0.96); Q8_0 151669x1024 934.2 &rarr; 885.2 (0.95) and 1046.4 &rarr; 934.3 (0.89); Q8_0
 *       151665x2560 1472.7 &rarr; 1221.0 (0.83) and 1477.2 &rarr; 1300.9 (0.88); BF16 151936x1024 895.3
 *       &rarr; 971.5 (1.09) and 908.3 &rarr; 908.5 (1.00). The 40-token greedy loops ran the other way
 *       at the 0.6B heads: the certified loop measured 5-33% slower there in both runs (its fixed
 *       per-token cost &mdash; three dispatches instead of two, the threshold pass, the screened-buffer
 *       round trip &mdash; stops amortizing once the exact readout is under a millisecond), and
 *       1-2% faster at the 4B head in both runs. That split, not the isolated readouts, sizes
 *       {@link #HEAD_MIN_ELEMENTS}.</li>
 *   <li>CUDA on the RTX 5080: F16 656.2 &rarr; 755.8 (1.15) and BF16 652.8 &rarr; 758.5 (1.16), both
 *       losses &mdash; the exact head's float matvec family runs 30% faster there than on OpenCL while
 *       the certified screen keeps its cost; Q8_0 was not measured because
 *       {@link com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead#certifiesRows} already answers false
 *       there: the int8 matvec lane ({@link com.eonmux.neuralfree.gguf.gpu.forward.Int8MatvecMode})
 *       serves one-row Q8_0 projections on CUDA, and the certified rescore repeats only the float
 *       matvec family's arithmetic.</li>
 *   <li>Vulkan on the Intel integrated GPU: F16 10355.7 &rarr; 17146.1 (1.66) and BF16 10954.7
 *       &rarr; 15460.4 (1.41), both losses; Q8_0 again lane-refused by the int8 matvec lane.</li>
 *   <li>CPU: F16 8396.4 &rarr; 82134.6 (9.78), an order-of-magnitude loss &mdash; the screen's scalar
 *       code dot cannot match the vectorized exact head; Q8_0 keeps no CPU lane at all because the CPU
 *       exact vec-dot quantizes the activations, whose per-token error no weight-only bound certifies.</li>
 * </ul>
 *
 * <p>{@link #DEFAULT_KMAX} is sized from the candidate distributions of the same heads over their real
 * prefill rows (per-row count of rows whose exact logit can reach the certified threshold): F16 0.6B
 * mean 3.0, p50 2, p90 6, max 13; Q8_0 0.6B mean 3.8, p50 3, p90 7, max 9; Q8_0 4B mean 11.5, p50 10,
 * p90 21, max 32; synthetic BF16 mean 3.6, p50 3, p90 7, max 14. The widest measured row is 32
 * candidates, so 128 covers every measured row four times over and measured a full-scan fallback rate
 * of exactly 0.0 on every model; the cap only bounds the rescore scratch, so the margin costs nothing.</p>
 */
public final class CertifiedArgmaxTuning {

    /** Most candidates one exact rescore covers; a row with more falls back to the full exact head. */
    public static final int DEFAULT_KMAX = 128;

    /**
     * Smallest head, in weight elements (vocab rows times hidden columns), at which {@code auto} serves
     * the lane: larger than every 0.6B-class head whose end-to-end decode measured the lane
     * neutral-to-slower (151669x1024 and 151936x1024, 155M elements) and no larger than the smallest
     * winning one (151665x2560, 388M elements). The lane's fixed per-token cost (three dispatches
     * instead of two, the threshold pass and the screened-buffer round trip) does not shrink with the
     * head, while the exact readout's cost does, so only a large head leaves enough measured saving for
     * the live decode loop to keep.
     */
    public static final long HEAD_MIN_ELEMENTS = 300_000_000L;

    /** System property holding the candidate cap: a positive integer, {@link #DEFAULT_KMAX} by default. */
    public static final String KMAX_PROPERTY = "neuralfree.argmax.kmax";

    /** Draft-lane system property accepted as an alias of {@link #KMAX_PROPERTY}. */
    public static final String KMAX_LEGACY_PROPERTY = "neuralfree.draft.argmax.kmax";

    private static final Map<BackendType, Set<GGMLType>> FASTER = faster();
    private static final Map<BackendType, Set<GGMLType>> CAPABLE = capable();

    private static final Set<GGMLType> CPU_FASTER_TYPES = Collections.unmodifiableSet(EnumSet.noneOf(GGMLType.class));
    private static final Set<GGMLType> CPU_CAPABLE_TYPES =
            Collections.unmodifiableSet(EnumSet.of(GGMLType.F16, GGMLType.BF16));

    private CertifiedArgmaxTuning() {
    }

    /**
     * The candidate cap: {@value KMAX_PROPERTY}, or its draft-lane alias when the new property is unset
     * or blank, or {@link #DEFAULT_KMAX} when neither is set.
     *
     * @throws IllegalArgumentException naming the property and the value when the property that was read
     *                                  holds anything but a positive integer
     */
    public static int kmax() {
        String value = System.getProperty(KMAX_PROPERTY);
        String property = KMAX_PROPERTY;
        if (value == null || value.trim().isEmpty()) {
            value = System.getProperty(KMAX_LEGACY_PROPERTY);
            property = KMAX_LEGACY_PROPERTY;
        }
        if (value == null || value.trim().isEmpty()) {
            return DEFAULT_KMAX;
        }
        int parsed;
        try {
            parsed = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            parsed = -1;
        }
        if (parsed < 1) {
            throw new IllegalArgumentException("-D" + property + " must be a positive integer, got '"
                    + value + "'");
        }
        return parsed;
    }

    /**
     * Whether {@link CertifiedArgmaxMode#AUTO} serves a head of {@code type} on {@code backend} at any
     * measured shape: true exactly for the pairs whose certified readout measured faster than the exact
     * readout. Backends and types without measured evidence answer {@code false} &mdash; the exact
     * readout keeps them until a measurement moves them into {@link #FASTER}.
     */
    public static boolean screenFaster(BackendType backend, GGMLType type) {
        Set<GGMLType> types = FASTER.get(backend);
        return types != null && types.contains(type);
    }

    /**
     * Whether {@link CertifiedArgmaxMode#AUTO} serves a head of {@code type} on {@code backend} of
     * {@code elements} weight elements (vocab rows times hidden columns): false for a type
     * {@link #screenFaster(BackendType, GGMLType)} rules out, and false below {@link #HEAD_MIN_ELEMENTS}
     * for a type that only measured faster on the larger heads.
     */
    public static boolean screenFaster(BackendType backend, GGMLType type, long elements) {
        return screenFaster(backend, type) && elements >= HEAD_MIN_ELEMENTS;
    }

    /** The head types that measured faster on {@code backend} (unmodifiable; empty for an unlisted backend). */
    public static Set<GGMLType> fasterTypes(BackendType backend) {
        Set<GGMLType> types = FASTER.get(backend);
        return types == null ? Collections.<GGMLType>emptySet() : types;
    }

    /**
     * Whether {@link CertifiedArgmaxMode#ON} can force the lane for a head of {@code type} on
     * {@code backend}: true for the backends whose kernels ship with the build and the half-precision
     * and Q8_0 types, measured or not. What {@code on} cannot do is displace a lane that already serves
     * the projection: on CUDA and Vulkan a one-row Q8_0 head projection keeps the int8 matvec lane
     * ({@link com.eonmux.neuralfree.gguf.gpu.forward.Int8MatvecMode}), and the runtime head bind keeps
     * the exact readout wherever the certified kernels fail to warm.
     */
    public static boolean screenCapable(BackendType backend, GGMLType type) {
        Set<GGMLType> types = CAPABLE.get(backend);
        return types != null && types.contains(type);
    }

    /** The head types {@link CertifiedArgmaxMode#ON} can force on {@code backend} (unmodifiable; empty for an unlisted backend). */
    public static Set<GGMLType> capableTypes(BackendType backend) {
        Set<GGMLType> types = CAPABLE.get(backend);
        return types == null ? Collections.<GGMLType>emptySet() : types;
    }

    /** Whether {@link CertifiedArgmaxMode#AUTO} serves a CPU head of {@code type}: no type measured the CPU screen faster. */
    public static boolean cpuFaster(GGMLType type) {
        return CPU_FASTER_TYPES.contains(type);
    }

    /** The CPU head types {@link CertifiedArgmaxMode#AUTO} serves (unmodifiable). */
    public static Set<GGMLType> cpuFasterTypes() {
        return CPU_FASTER_TYPES;
    }

    /** Whether {@link CertifiedArgmaxMode#ON} can force the CPU lane for a head of {@code type}. */
    public static boolean cpuCapable(GGMLType type) {
        return CPU_CAPABLE_TYPES.contains(type);
    }

    /** The CPU head types {@link CertifiedArgmaxMode#ON} can force (unmodifiable). */
    public static Set<GGMLType> cpuCapableTypes() {
        return CPU_CAPABLE_TYPES;
    }

    /**
     * The pairs whose certified readout measured faster than the exact readout at their measured shape:
     * on the machine's RTX 5080 through OpenCL, the Q8_0 4B head (0.83x and 0.88x in the two runs, its
     * 40-token greedy loop 1-2% faster in both). The F16 and Q8_0 0.6B heads also won their isolated
     * readouts (0.95-0.97x) but lost or tied the live decode loop in both runs, so they stay out of this
     * table and {@link #HEAD_MIN_ELEMENTS} keeps them out; every other measured pair lost outright
     * (BF16 1.00-1.09x on the same device; CUDA F16 1.15x, BF16 1.16x; Vulkan F16 1.66x, BF16 1.41x;
     * the CPU F16 head 9.78x) or never runs (Q8_0 on CUDA and Vulkan keeps the int8 matvec lane);
     * nothing unmeasured enters this table.
     */
    private static Map<BackendType, Set<GGMLType>> faster() {
        Map<BackendType, Set<GGMLType>> m = new EnumMap<>(BackendType.class);
        m.put(BackendType.OPENCL, Collections.unmodifiableSet(EnumSet.of(GGMLType.Q8_0)));
        return Collections.unmodifiableMap(m);
    }

    /**
     * The pairs the lane's kernels cover on the GPU backends this build compiles its kernels for
     * (OpenCL and CUDA share the OpenCL C units, Vulkan the SPIR-V twins, HIP the same units as OpenCL,
     * Metal the same SPIR-V as Vulkan); accelerator APIs without kernels here (SYCL, XPU, CANN and the
     * rest) list nothing. The CPU path lists F16 and BF16: its Q8_0 vec-dot quantizes the activations,
     * whose per-token error no weight-only bound certifies.
     */
    private static Map<BackendType, Set<GGMLType>> capable() {
        Map<BackendType, Set<GGMLType>> m = new EnumMap<>(BackendType.class);
        for (BackendType backend : new BackendType[]{BackendType.OPENCL, BackendType.CUDA, BackendType.VULKAN,
                BackendType.HIP, BackendType.METAL}) {
            m.put(backend, Collections.unmodifiableSet(EnumSet.of(GGMLType.F16, GGMLType.BF16, GGMLType.Q8_0)));
        }
        return Collections.unmodifiableMap(m);
    }

    @Override
    public String toString() {
        return "CertifiedArgmaxTuning{kmax=" + kmax() + ", faster=" + FASTER + ", capable=" + CAPABLE
                + ", cpuFaster=" + CPU_FASTER_TYPES + ", cpuCapable=" + CPU_CAPABLE_TYPES + '}';
    }
}
