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
 * jdk.incubator.vector}). {@link #int8()} serves the engine's {@link CpuInt8Kernels} to the fused int8 paths the
 * same way. Every kernel produces the exact values of its scalar counterpart, so selection changes speed only. The
 * scalar kernels remain the fallback on Java 11 and whenever the module is absent.
 * <p>
 * Set {@value #PROP_SIMD} to {@code false} to keep the scalar kernels; any other value is rejected by name.
 * The module classes ride in the plain jar (class file version 61) and are loaded reflectively only, so a
 * Java 11 runtime never links them.
 */
public final class CpuSimd {

    /** {@code false} keeps the scalar kernels; absent or {@code true} selects the Vector API kernels. */
    public static final String PROP_SIMD = "neuralfree.cpu.simd";

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

        /** One-line description of the vector hardware level the engine targets. */
        String describe();
    }

    /** Anything but an {@link Engine}: the engine is unavailable and this is the reason (null = not yet asked). */
    private static volatile Object resolved;

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
            Class<?> cls = Class.forName("com.eonmux.neuralfree.gguf.cpu.quant.SimdKernels", true,
                    CpuSimd.class.getClassLoader());
            return (Engine) cls.getDeclaredConstructor().newInstance();
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

    /** Marker of "no engine on this JVM". */
    private static final Object NONE = new Object();

    @Override
    public String toString() {
        return "CpuSimd{" + describe() + "}";
    }
}
