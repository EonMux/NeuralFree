package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.lang.ModuleLayer;
import java.util.Collections;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * Hand-vectorized CPU vec-dot kernels of this package ({@link SimdKernels}, built on the Vector API of
 * {@code jdk.incubator.vector}), selected over the scalar kernels of {@link CpuMatVec#vecDot} when the JVM
 * offers them: Java 17 or later with the incubator module resolved ({@code --add-modules
 * jdk.incubator.vector}) on hardware whose preferred vector is at least 256 bits wide. {@link #int8()} serves the
 * engine's {@link CpuInt8Kernels} to the fused int8 paths. Every kernel produces the exact values of its scalar
 * counterpart, so selection changes speed only. The scalar kernels remain the fallback on Java 11, whenever the
 * module is absent and on narrower vector hardware.
 * <p>
 * The attention core keeps the scalar {@link CpuAttnKernels} unless {@value #PROP_ATTENTION} is {@code true}
 * ({@link #attention()}). The Vector API attention kernels are faster, but running their {@code FloatVector} code
 * in the same JVM makes C2 compile the int16 {@code Q4_K} and {@code Q6_K} tile kernels without their vector
 * intrinsics in a share of the JVMs, which slows those products 2 to 15 times for the life of the JVM.
 * <p>
 * The kernels use fixed 64-, 128- and 256-bit species. Where the hardware's preferred vector is narrower (128-bit
 * NEON, SSE-only x86, or a JVM capped by {@code -XX:MaxVectorSize=16}), the JIT does not split the 256-bit
 * operations: it runs their Java fallback, about ten times slower than the scalar kernels. {@value #PROP_WIDTH}
 * {@code =auto} (the default) therefore keeps the scalar kernels there; {@code 256} selects the engine whatever
 * the hardware reports.
 * <p>
 * Set {@value #PROP_SIMD} to {@code false} to keep the scalar kernels; any other value of these properties is
 * rejected by name. The module classes ride in the plain jar (class file version 61) and are loaded reflectively
 * only, so a Java 11 runtime never links them.
 */
public final class CpuSimd {

    /** {@code false} keeps the scalar kernels; absent or {@code true} selects the Vector API kernels. */
    public static final String PROP_SIMD = "neuralfree.cpu.simd";

    /**
     * {@code auto} selects the Vector API kernels only when the hardware's preferred vector is at least
     * {@value #ENGINE_BITS} bits wide; {@code 256} selects them whatever the hardware reports.
     */
    public static final String PROP_WIDTH = "neuralfree.cpu.simd.width";

    /**
     * {@code true} serves the engine's Vector API attention kernels to the attention core; absent or {@code false}
     * keeps the scalar ones.
     */
    public static final String PROP_ATTENTION = "neuralfree.cpu.simd.attention";

    /** The widest species the engine's kernels use. */
    static final int ENGINE_BITS = 256;

    /** The kernels the engine exposes, without any Vector API type in its signatures (Java 11 types only). */
    public interface Engine {

        /** The vec-dot of {@code type}, or null when the engine has none for it. */
        CpuVecDot vecDot(GGMLType type);

        /** Every weight type with a kernel (unmodifiable). */
        Set<GGMLType> supportedTypes();

        /** The vec-dot of {@code type}'s repacked layout ({@code com.eonmux.neuralfree.gguf.cpu.repack}), or null. */
        CpuVecDot repacked(GGMLType type);

        /** The vec-dot of the repacked Q4_0x8 layout, or null. */
        CpuVecDot q4_0x8();

        /** The int8 primitives of the fused CPU paths. */
        CpuInt8Kernels int8();

        /** The float primitives of the CPU attention core. */
        CpuAttnKernels attention();

        /** One-line description of the vector hardware level the engine targets. */
        String describe();

        /** The bit size of the JVM's preferred vector shape on this hardware ({@code VectorShape.preferredShape}). */
        int preferredBits();
    }

    /** Anything but an {@link Engine}: the engine is unavailable and this is the reason (null = not yet asked). */
    private static volatile Object resolved;

    /** {@link Boolean} the value of {@value #PROP_ATTENTION}, or its rejection (null = not yet read). */
    private static volatile Object attentionResolved;

    private CpuSimd() {
    }

    /** True when {@link #vecDot}, {@link #repacked} and {@link #q4_0x8} can return kernels on this JVM. */
    public static boolean available() {
        return engine() != null;
    }

    /** True when {@code type} has a Vector API kernel on this JVM. */
    public static boolean supports(GGMLType type) {
        Engine e = engine();
        return e != null && type != null && e.vecDot(type) != null;
    }

    /** The weight types with Vector API kernels on this JVM (unmodifiable; empty without the engine). */
    public static Set<GGMLType> supportedTypes() {
        Engine e = engine();
        return e != null ? e.supportedTypes() : Collections.emptySet();
    }

    /**
     * The Vector API vec-dot of {@code type}, or null when this JVM has none (the caller keeps the scalar
     * kernel). Conflicts with the property are reported the same way for every query.
     */
    public static CpuVecDot vecDot(GGMLType type) {
        Engine e = engine();
        return e != null ? e.vecDot(type) : null;
    }

    /**
     * The Vector API vec-dot of the repacked layout of {@code type}, or null when this JVM has none.
     *
     * @see com.eonmux.neuralfree.gguf.cpu.repack.CpuRepack
     */
    public static CpuVecDot repacked(GGMLType type) {
        Engine e = engine();
        return e != null ? e.repacked(type) : null;
    }

    /**
     * The Vector API vec-dot of the repacked Q4_0x8 layout, or null without the engine.
     *
     * @see com.eonmux.neuralfree.gguf.cpu.repack.Q4_0x8
     */
    public static CpuVecDot q4_0x8() {
        Engine e = engine();
        return e != null ? e.q4_0x8() : null;
    }

    /**
     * The int8 primitives of the fused CPU paths: the Vector API kernels when the engine is available, otherwise
     * their scalar reference ({@link ScalarInt8Kernels#INSTANCE}).
     */
    public static CpuInt8Kernels int8() {
        Engine e = engine();
        CpuInt8Kernels kernels = e != null ? e.int8() : null;
        return kernels != null ? kernels : ScalarInt8Kernels.INSTANCE;
    }

    /**
     * The float primitives of the CPU attention core: the Vector API kernels when {@value #PROP_ATTENTION} is
     * {@code true} and the engine is available, otherwise their scalar reference ({@link ScalarAttnKernels#INSTANCE}).
     *
     * @throws IllegalArgumentException when a property of this class has an invalid value
     */
    public static CpuAttnKernels attention() {
        Engine e = engine();
        if (!attentionEnabled()) {
            return ScalarAttnKernels.INSTANCE;
        }
        CpuAttnKernels kernels = e != null ? e.attention() : null;
        return kernels != null ? kernels : ScalarAttnKernels.INSTANCE;
    }

    /**
     * The engine's Vector API attention kernels whatever {@value #PROP_ATTENTION} says, or null without the engine:
     * for callers that compare or measure both kernel sets.
     */
    public static CpuAttnKernels vectorAttention() {
        Engine e = engine();
        return e != null ? e.attention() : null;
    }

    /** {@value #PROP_ATTENTION} of the system properties, read once (a rejection is rethrown at every call). */
    private static boolean attentionEnabled() {
        Object r = attentionResolved;
        if (r == null) {
            try {
                r = attentionEnabled(System.getProperties());
            } catch (IllegalArgumentException e) {
                r = e;
            }
            attentionResolved = r;
        }
        if (r instanceof Boolean) {
            return (Boolean) r;
        }
        throw (IllegalArgumentException) r;
    }

    /**
     * Whether {@value #PROP_ATTENTION} selects the Vector API attention kernels.
     *
     * @throws IllegalArgumentException when the property is neither {@code true} nor {@code false}
     */
    static boolean attentionEnabled(Properties p) {
        Objects.requireNonNull(p, "properties");
        String v = p.getProperty(PROP_ATTENTION);
        if (v == null || v.trim().isEmpty()) {
            return false;
        }
        String s = v.trim().toLowerCase(java.util.Locale.ROOT);
        if (s.equals("true")) {
            return true;
        }
        if (s.equals("false")) {
            return false;
        }
        throw new IllegalArgumentException(PROP_ATTENTION + "=" + v + " is not a boolean; use true or false");
    }

    /** What the CPU kernels run on: the vector engine's description, or {@code "scalar"} without it. */
    public static String describe() {
        Engine e = engine();
        return e != null ? e.describe() : "scalar";
    }

    /** The engine of this JVM, or null when absent; caches the answer (and a rejection, to rethrow it). */
    static Engine engine() {
        Object r = resolved;
        if (r == null) {
            r = resolve();
            resolved = r;
        }
        if (r instanceof Engine) {
            return (Engine) r;
        }
        if (r == NONE) {
            return null;
        }
        throw (RuntimeException) r;
    }

    /** Property gate plus the reflective load of {@code SimdKernels}; never loads it on an older JVM. */
    private static Object resolve() {
        try {
            if (!enabled(System.getProperties())) {
                return NONE;
            }
            if (Runtime.version().feature() < 17 || !ModuleLayer.boot().findModule("jdk.incubator.vector")
                    .isPresent()) {
                return NONE;
            }
            final boolean forced = forcedWidth(System.getProperties());
            Class<?> cls = Class.forName("com.eonmux.neuralfree.gguf.cpu.quant.SimdKernels", true,
                    CpuSimd.class.getClassLoader());
            Engine engine = (Engine) cls.getDeclaredConstructor().newInstance();
            return usable(forced, engine.preferredBits()) ? engine : NONE;
        } catch (RuntimeException e) {
            return e;
        } catch (LinkageError | ReflectiveOperationException e) {
            // the module is resolved but unusable here: the classes are absent, cannot link or cannot initialize
            return NONE;
        }
    }

    /**
     * Whether the property selects the Vector API kernels.
     *
     * @throws IllegalArgumentException when {@value #PROP_SIMD} is neither {@code true} nor {@code false}
     */
    public static boolean enabled(Properties p) {
        Objects.requireNonNull(p, "properties");
        String v = p.getProperty(PROP_SIMD);
        if (v == null || v.trim().isEmpty()) {
            return true;
        }
        String s = v.trim().toLowerCase(java.util.Locale.ROOT);
        if (s.equals("true")) {
            return true;
        }
        if (s.equals("false")) {
            return false;
        }
        throw new IllegalArgumentException(PROP_SIMD + "=" + v + " is not a boolean; use true or false");
    }

    /**
     * Whether {@value #PROP_WIDTH} forces the engine on whatever vector width the hardware reports.
     *
     * @throws IllegalArgumentException when the property is neither {@code auto} nor {@code 256}
     */
    static boolean forcedWidth(Properties p) {
        Objects.requireNonNull(p, "properties");
        String v = p.getProperty(PROP_WIDTH);
        if (v == null || v.trim().isEmpty() || v.trim().equalsIgnoreCase("auto")) {
            return false;
        }
        if (v.trim().equals(Integer.toString(ENGINE_BITS))) {
            return true;
        }
        throw new IllegalArgumentException(PROP_WIDTH + "=" + v + " is not a vector width; use auto or "
                + ENGINE_BITS);
    }

    /**
     * Whether the engine runs on hardware whose preferred vector is {@code preferredBits} wide: always when
     * {@code forced}, otherwise only when its widest species fits that vector.
     */
    static boolean usable(boolean forced, int preferredBits) {
        return forced || preferredBits >= ENGINE_BITS;
    }

    /** Marker of "no engine on this JVM". */
    private static final Object NONE = new Object();

    @Override
    public String toString() {
        return "CpuSimd{" + describe() + "}";
    }
}
