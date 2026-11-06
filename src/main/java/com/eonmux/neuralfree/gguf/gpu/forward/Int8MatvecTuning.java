package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/**
 * The decode evidence behind {@link Int8MatvecMode#AUTO}: the weight types whose integer-dot (Q8_1
 * activation) matvec measured slower than the F32-activation matvec, so AUTO keeps the exact float
 * path for them even where the device reports
 * {@link com.eonmux.neuralfree.acceleration.backend.DeviceFeature#INTEGER_DOT_PRODUCT}. Every type a
 * backend does not list keeps the lane there, as does every type on a backend without an entry.
 *
 * <p>Two kinds of loss are recorded. A type in {@link #SLOWER} measured slower at every projection
 * shape, so it never takes the lane. A type in {@link #LARGE_ONLY} measured slower on the small
 * projections of a 0.6B-class decoder and faster on 4096-column projections of a 4B/9B-class one, so
 * it takes the lane only from {@link #LANE_MIN_ELEMENTS} elements (weight rows times columns) up: the
 * lane's fixed cost (the quantizer launch and its own launch floor) does not shrink with the
 * projection, while the F32 matvec's activation reads grow with it. {@link Int8MatvecMode#ON} runs the
 * lane regardless of this table.</p>
 *
 * <p>The entries are single-token matvec times of one projection, measured on the CUDA, OpenCL and
 * Vulkan backends of the machine's GPUs (an NVIDIA RTX 5080 Laptop, its Vulkan driver exposing the
 * Intel integrated GPU) over the shapes of a 0.6B decoder (1024x1024, 3072x1024, 1024x3072), a 4B
 * decoder (9728x2560, 2560x9728) and 7B/8B decoders (4096x4096, 14336x4096, 32768x1024), with and
 * without the Q8_1 quantization the lane adds; every type in {@link #SLOWER} measured a median of at
 * most 0.97x across the shapes, and every type in {@link #LARGE_ONLY} measured below 1.0x at every
 * shape under 4M elements and at least 1.02x at every shape from 16.8M elements up. The Vulkan device
 * reports no packed dot product, so only the type whose lane measured slower there is listed; Q8_0
 * measured faster at every Vulkan shape but 4096x4096, where its single-row form lost and its paired
 * form measured 1.7x, so it keeps the feature decision like the other mixed types.</p>
 */
public final class Int8MatvecTuning {

    /**
     * Smallest projection, in weight elements (rows times columns), at which a {@link #LARGE_ONLY} type
     * measured the lane faster than the F32 matvec: larger than every losing shape (1024x1024,
     * 3072x1024, 1024x3072) and no larger than the smallest winning one (4096x4096).
     */
    public static final long LANE_MIN_ELEMENTS = 4_000_000L;

    private static final Map<BackendType, Set<GGMLType>> SLOWER = slower();
    private static final Map<BackendType, Set<GGMLType>> LARGE_ONLY = largeOnly();
    private static final Map<BackendType, Set<GGMLType>> PAIRED_SLOWER = pairedSlower();

    private Int8MatvecTuning() {
    }

    /**
     * Whether the lane measured faster than the F32 matvec for {@code type} on {@code backend} at
     * every projection shape. Types and backends without measured evidence answer {@code true}: the
     * device's packed dot product support then decides, as it does for every type.
     */
    public static boolean laneIsFaster(BackendType backend, GGMLType type) {
        Set<GGMLType> slower = SLOWER.get(backend);
        return slower == null || !slower.contains(type);
    }

    /**
     * Whether the lane measured faster than the F32 matvec for a projection of {@code elements} weight
     * elements (rows times columns): false for a type {@link #laneIsFaster} rules out, and false below
     * {@link #LANE_MIN_ELEMENTS} for a type that only measured faster on the larger projections.
     */
    public static boolean laneIsFaster(BackendType backend, GGMLType type, long elements) {
        if (!laneIsFaster(backend, type)) {
            return false;
        }
        Set<GGMLType> largeOnly = LARGE_ONLY.get(backend);
        return largeOnly == null || !largeOnly.contains(type) || elements >= LANE_MIN_ELEMENTS;
    }

    /**
     * Whether the paired integer-dot matvec ({@code {T}_MatVec2_Q8}) measured faster than the single-row
     * form ({@code {T}_MatVec_Q8}) for {@code type} on {@code backend}: the two-row form halves the Q8_1
     * row re-reads but adds a second work-group reduction, which on CUDA costs the K-quant weights more
     * than the shared row saves (Q4_K and Q5_K lost 1-8% paired, Q8_K and Q8_0 measured neutral) and on
     * OpenCL only Q8_K and Q8_0 failed to gain. Types and backends without measured evidence answer
     * {@code true}.
     */
    public static boolean pairedIsFaster(BackendType backend, GGMLType type) {
        Set<GGMLType> slower = PAIRED_SLOWER.get(backend);
        return slower == null || !slower.contains(type);
    }

    private static Map<BackendType, Set<GGMLType>> slower() {
        Map<BackendType, Set<GGMLType>> slower = new EnumMap<>(BackendType.class);
        slower.put(BackendType.OPENCL, EnumSet.of(GGMLType.MXFP4, GGMLType.NVFP4, GGMLType.IQ4_NL,
                GGMLType.IQ4_XS));
        slower.put(BackendType.CUDA, EnumSet.of(GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0,
                GGMLType.Q5_1, GGMLType.IQ4_NL));
        slower.put(BackendType.VULKAN, EnumSet.of(GGMLType.Q4_0));
        return Collections.unmodifiableMap(slower);
    }

    /**
     * The types that measured the lane slower on the small projections of a 0.6B-class decoder and
     * faster from 4096 columns up, by backend: the projections of the 0.6B model (1024 columns) lose
     * the lane's fixed quantizer cost, while the 4B/9B-class 4096-column projections gain from the
     * smaller activation row and the packed dot product.
     */
    private static Map<BackendType, Set<GGMLType>> largeOnly() {
        Map<BackendType, Set<GGMLType>> large = new EnumMap<>(BackendType.class);
        large.put(BackendType.OPENCL, EnumSet.of(GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1,
                GGMLType.Q8_0, GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
                GGMLType.Q8_K));
        large.put(BackendType.CUDA, EnumSet.of(GGMLType.MXFP4, GGMLType.NVFP4, GGMLType.Q8_0, GGMLType.Q2_K,
                GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K, GGMLType.IQ4_XS));
        return Collections.unmodifiableMap(large);
    }

    /**
     * The types whose paired form measured slower than the single-row form, by backend: interleaved
     * rounds of both forms over the same weights and Q8_1 row at 1024 to 32768 output rows of 4096 and
     * 12288 columns (median of nine rounds). Q4_K and Q5_K lost 1-8% on CUDA (medians 0.95x and 0.96x)
     * and Q8_K and Q8_0 measured neutral there (Q8_0 medians 0.99x, at best 1.00x); on OpenCL Q8_K and
     * Q8_0 never gained (Q8_0 median 0.97x, at best 1.02x); the CUDA losers gained 2-13% there.
     */
    private static Map<BackendType, Set<GGMLType>> pairedSlower() {
        Map<BackendType, Set<GGMLType>> slower = new EnumMap<>(BackendType.class);
        slower.put(BackendType.OPENCL, EnumSet.of(GGMLType.Q8_0, GGMLType.Q8_K));
        slower.put(BackendType.CUDA, EnumSet.of(GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q8_0, GGMLType.Q8_K));
        return Collections.unmodifiableMap(slower);
    }
}
