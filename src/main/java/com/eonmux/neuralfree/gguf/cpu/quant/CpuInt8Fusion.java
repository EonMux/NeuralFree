package com.eonmux.neuralfree.gguf.cpu.quant;

import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * Selection of the fused int8 CPU paths: attention over an int8 per-token-head KV cache
 * ({@code CpuInt8Attention}) and the fused int8 expert pass of a MoE layer ({@code CpuInt8Moe}).
 *
 * <p>Set {@value #PROP_INT8} to {@code false} to keep the generic kernels (attention dequantizes every cached
 * K and V row into float before scoring, experts product by product). Any other value than a boolean is rejected by
 * name. The property is read when a block binds, so a value set before a model is loaded selects the path for that
 * model. Both fused paths apply only where the model qualifies — K and V stored as
 * {@code int8_per_token_head}, or expert weights stored as int8 ({@code Q8_0}, {@code I8_B128}); every other model
 * runs the generic kernels, which the fused paths reproduce within their stated tolerances.</p>
 */
public final class CpuInt8Fusion {

    /** {@code false} keeps the generic kernels; absent or {@code true} selects the fused int8 paths. */
    public static final String PROP_INT8 = "neuralfree.cpu.int8";

    private CpuInt8Fusion() {
    }

    /** Whether the fused int8 paths are selected, from the system properties. */
    public static boolean enabled() {
        return enabled(System.getProperties());
    }

    /**
     * Whether {@value #PROP_INT8} selects the fused int8 paths.
     *
     * @throws IllegalArgumentException when the value is neither {@code true} nor {@code false}
     */
    public static boolean enabled(Properties p) {
        Objects.requireNonNull(p, "properties");
        String v = p.getProperty(PROP_INT8);
        if (v == null || v.trim().isEmpty()) {
            return true;
        }
        String s = v.trim().toLowerCase(Locale.ROOT);
        if (s.equals("true")) {
            return true;
        }
        if (s.equals("false")) {
            return false;
        }
        throw new IllegalArgumentException(PROP_INT8 + "=" + v + " is not a boolean; use true or false");
    }

    /** The int8 kernels of this JVM: the Vector API kernels, or their scalar reference. */
    public static CpuInt8Kernels kernels() {
        return CpuSimd.int8();
    }

    @Override
    public String toString() {
        return "CpuInt8Fusion{" + PROP_INT8 + "=" + enabled() + ", kernels=" + kernels().describe() + '}';
    }
}
