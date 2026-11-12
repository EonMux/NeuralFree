package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

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
 * The attention core runs the engine's Vector API attention kernels too ({@link #attention()}), which return the
 * scalar kernels' exact values; {@value #PROP_ATTENTION}{@code =false} keeps the scalar {@link CpuAttnKernels}.
 * The int8 and int16 weight kernels compile to the same code whichever Vector API code the JVM ran before them
 * (the rules in {@code SimdI8}), so the float attention code leaves their speed alone.
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
     * Absent or {@code true} serves the engine's Vector API attention kernels to the attention core; {@code false}
     * keeps the scalar ones.
     */
    public static final String PROP_ATTENTION = "neuralfree.cpu.simd.attention";

    /**
     * The summation order of the Vector API single-token dots: absent or {@code exact} keeps every kernel's result
     * bit-identical to its scalar kernel; {@code fast} lets the kernels that have a fast-order form (Q8_0) accumulate
     * in float vector lanes and reduce once per row, which changes results by float rounding.
     */
    public static final String PROP_ORDER = "neuralfree.cpu.simd.order";

    /** The widest species the engine's kernels use. */
    static final int ENGINE_BITS = 256;

    /** The kernels the engine exposes, without any Vector API type in its signatures (Java 11 types only). */
    public interface Engine {

        /** The vec-dot of {@code type}, or null when the engine has none for it. */
        CpuVecDot vecDot(GGMLType type);

        /**
         * The vec-dot of {@code type} whose single-token dots sum in the fast float order ({@value #PROP_ORDER}), or
         * null when the engine has no such form for it.
         */
        CpuVecDot fastOrder(GGMLType type);

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

    /** {@link Boolean} whether {@value #PROP_ORDER} selects the fast order, or its rejection (null = not yet read). */
    private static volatile Object orderResolved;

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
        if (e == null) {
            return null;
        }
        CpuVecDot fast = fastOrderEnabled() ? e.fastOrder(type) : null;
        return fast != null ? fast : e.vecDot(type);
    }

    /**
     * The fast-order form of {@code type}'s Vector API vec-dot whatever {@value #PROP_ORDER} says, or null when this
     * JVM has none: for callers that compare or measure both orders.
     */
    public static CpuVecDot fastOrderVecDot(GGMLType type) {
        Engine e = engine();
        return e != null ? e.fastOrder(type) : null;
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
     * The float primitives of the CPU attention core: the Vector API kernels when the engine is available and
     * {@value #PROP_ATTENTION} is not {@code false}, otherwise their scalar reference
     * ({@link ScalarAttnKernels#INSTANCE}).
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

    /** {@value #PROP_ORDER} of the system properties, read once (a rejection is rethrown at every call). */
    private static boolean fastOrderEnabled() {
        Object r = orderResolved;
        if (r == null) {
            try {
                r = fastOrder(System.getProperties());
            } catch (IllegalArgumentException e) {
                r = e;
            }
            orderResolved = r;
        }
        if (r instanceof Boolean) {
            return (Boolean) r;
        }
        throw (IllegalArgumentException) r;
    }

    /**
     * Whether {@value #PROP_ORDER} selects the fast summation order.
     *
     * @throws IllegalArgumentException when the property is neither {@code exact} nor {@code fast}
     */
    static boolean fastOrder(Properties p) {
        Objects.requireNonNull(p, "properties");
        String v = p.getProperty(PROP_ORDER);
        if (v == null || v.trim().isEmpty() || v.trim().equalsIgnoreCase("exact")) {
            return false;
        }
        if (v.trim().equalsIgnoreCase("fast")) {
            return true;
        }
        throw new IllegalArgumentException(PROP_ORDER + "=" + v + " is not a summation order; use exact or fast");
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
            return true;
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

    /**
     * Why this JVM runs the kernels {@link #describe()} names: the property, Java version, module, vector width or
     * class-loading fact that decided the choice, in words a user can act on.
     *
     * @throws IllegalArgumentException when a property of this class has an invalid value
     */
    public static String reason() {
        engine();
        return resolvedReason;
    }

    /**
     * {@link #describe()} with its {@link #reason()}, naming the fast summation order when {@value #PROP_ORDER}
     * selects it for the engine's kernels: the line the bench header and {@code collect-env} print.
     *
     * @throws IllegalArgumentException when a property of this class has an invalid value
     */
    public static String status() {
        final String order = available() && fastOrderEnabled() ? ", fast summation order" : "";
        return describe() + order + " (" + reason() + ")";
    }

    /** The engine of this JVM, or null when absent; caches the answer (and a rejection, to rethrow it). */
    static Engine engine() {
        Object r = resolved;
        if (r == null) {
            synchronized (CpuSimd.class) {
                r = resolved;
                if (r == null) {
                    r = resolve();
                    resolved = r;
                    if (!(r instanceof RuntimeException)) {
                        log(r instanceof Engine);
                    }
                }
            }
        }
        if (r instanceof Engine) {
            return (Engine) r;
        }
        if (r == NONE) {
            return null;
        }
        throw (RuntimeException) r;
    }

    /**
     * Property gate plus the reflective load of {@code SimdKernels}; never loads it on an older JVM. Sets
     * {@link #resolvedReason} for every answer but a rejection.
     */
    private static Object resolve() {
        try {
            if (!enabled(System.getProperties())) {
                return none(PROP_SIMD + "=false keeps the scalar kernels", false);
            }
            final int feature = Runtime.version().feature();
            if (feature < 17) {
                return none("Java " + feature + " has no jdk.incubator.vector module; the kernels need Java 17+",
                        false);
            }
            if (!ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent()) {
                return none("jdk.incubator.vector is not resolved in this JVM; the engine commands resolve it with"
                        + " --add-modules=jdk.incubator.vector", false);
            }
            final boolean forced = forcedWidth(System.getProperties());
            Class<?> cls = Class.forName("com.eonmux.neuralfree.gguf.cpu.quant.SimdKernels", true,
                    CpuSimd.class.getClassLoader());
            Engine engine = (Engine) cls.getDeclaredConstructor().newInstance();
            final int bits = engine.preferredBits();
            if (!usable(forced, bits)) {
                return none("the preferred vector is " + bits + " bits, narrower than the kernels' " + ENGINE_BITS
                        + "-bit species, which would run the Vector API's Java fallback; " + PROP_WIDTH + "="
                        + ENGINE_BITS + " forces them", true);
            }
            resolvedReason = forced && bits < ENGINE_BITS
                    ? PROP_WIDTH + "=" + ENGINE_BITS + " forces the kernels over a " + bits + "-bit preferred vector"
                    : "jdk.incubator.vector on a " + bits + "-bit preferred vector";
            return engine;
        } catch (RuntimeException e) {
            return e;
        } catch (LinkageError | ReflectiveOperationException e) {
            // the module is resolved but unusable here: the classes are absent, cannot link or cannot initialize
            return none("the Vector API kernels failed to load: " + e, true);
        }
    }

    /** {@link #NONE} with its reason; {@code fallback} marks a JVM that has the module but cannot use it. */
    private static Object none(String reason, boolean fallback) {
        resolvedReason = reason;
        resolvedFallback = fallback;
        return NONE;
    }

    /**
     * Logs the choice once: at info when a JVM that resolved the module still keeps the scalar kernels (the width
     * gate or a load failure, which a user may want to override or report), at debug otherwise.
     */
    private static void log(boolean engine) {
        final Logger log = LogManager.getLogger(CpuSimd.class);
        final String line = "CPU kernels: " + (engine ? resolvedEngine() : "scalar") + " (" + resolvedReason + ")";
        if (!engine && resolvedFallback) {
            log.info(line);
        } else {
            log.debug(line);
        }
    }

    private static String resolvedEngine() {
        return ((Engine) resolved).describe();
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

    /** Why {@link #resolved} holds what it holds; written before it. */
    private static volatile String resolvedReason;

    /** Whether the scalar choice was a fallback in a JVM that resolved the module. */
    private static volatile boolean resolvedFallback;

    @Override
    public String toString() {
        return "CpuSimd{" + describe() + "}";
    }
}
