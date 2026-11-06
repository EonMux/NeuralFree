package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * The RoPE and YaRN context-extension overrides ({@code --rope-scaling}, {@code --rope-scale},
 * {@code --rope-freq-base}, {@code --rope-freq-scale}, {@code --yarn-ext-factor}, {@code --yarn-attn-factor},
 * {@code --yarn-beta-fast}, {@code --yarn-beta-slow}, {@code --yarn-orig-ctx}; common/arg.cpp:2338-2369).
 *
 * <p>{@code ModelSources} applies them as metadata overrides of the {@code %s.rope.*} view keys
 * ({@link GgufKeys}) of the opened source's architecture, so hparams resolve the rope parameters as if the model
 * had been stored that way; a value the option does not set is not written and its metadata stays as stored. The
 * resolved values are namespace inputs of a prefix-cache namespace ({@code CacheNamespace}) and value-changing
 * inputs of the weights digest: a model loaded with different rope options is served under a different namespace
 * and digests differently.
 *
 * <p>Which metadata keys the options write, when they are set:
 * <ul>
 *   <li>{@code rope.scaling.type} STRING ({@code none}, {@code linear}, {@code long}, {@code yarn} or
 *       {@code dynamic});</li>
 *   <li>{@code rope.scaling.factor} FLOAT32: the context scaling factor of {@code --rope-scale N}, or {@code 1 / N}
 *       for {@code --rope-freq-scale N}, as llama.cpp reads the two flags;</li>
 *   <li>{@code rope.freq_base} FLOAT32;</li>
 *   <li>{@code rope.scaling.attn_factor}, {@code rope.scaling.yarn_ext_factor},
 *       {@code rope.scaling.yarn_beta_fast} and {@code rope.scaling.yarn_beta_slow} FLOAT32;</li>
 *   <li>{@code rope.scaling.original_context_length} INT32, when {@code --yarn-orig-ctx} is larger than 0 (0
 *       keeps the model's training context, as llama.cpp reads its 0 default).</li>
 * </ul>
 * Immutable; defaults() is unset everywhere.
 */
public final class RopeOverrides {

    public static final String PROP_ROPE_SCALING = "neuralfree.rope.scaling";
    public static final String PROP_ROPE_SCALE = "neuralfree.rope.scale";
    public static final String PROP_ROPE_FREQ_BASE = "neuralfree.rope.freq.base";
    public static final String PROP_ROPE_FREQ_SCALE = "neuralfree.rope.freq.scale";
    public static final String PROP_YARN_EXT_FACTOR = "neuralfree.yarn.ext.factor";
    public static final String PROP_YARN_ATTN_FACTOR = "neuralfree.yarn.attn.factor";
    public static final String PROP_YARN_BETA_FAST = "neuralfree.yarn.beta.fast";
    public static final String PROP_YARN_BETA_SLOW = "neuralfree.yarn.beta.slow";
    public static final String PROP_YARN_ORIG_CTX = "neuralfree.yarn.orig.ctx";

    /** The rope scaling types of llama.cpp's rope extension arithmetic ({@code --rope-scaling}). */
    public enum ScalingType {
        /** RoPE as the model was trained; nothing to scale. */
        NONE,
        /** llama.cpp's linear scaling: the context scales by the factor. */
        LINEAR,
        /** LongLoRA's logarithmic interpolation. */
        LONG,
        /** YaRN (extended yarn interpolation, the llama.cpp of the yarn family). */
        YARN,
        /** NTK-aware scaling by a factor or alpha (vLLM's {@code dynamic} rope type). */
        DYNAMIC,
        /** Fixed and mixed NTK scaling by a factor (vLLM's {@code ntk} rope type). */
        NTK;

        /**
         * The type named by {@code value}, matched case-insensitively with {@code -} read as {@code _} but
         * otherwise exactly, as llama.cpp's {@code --rope-scaling} matches it: surrounding whitespace is
         * part of the value, not trimmed.
         *
         * @throws IllegalArgumentException naming {@value #PROP_ROPE_SCALING} for an unknown name
         */
        public static ScalingType parse(String value) {
            Objects.requireNonNull(value, PROP_ROPE_SCALING);
            String s = value.toUpperCase(Locale.ROOT).replace('-', '_');
            StringBuilder valid = new StringBuilder();
            for (ScalingType t : ScalingType.values()) {
                if (t.name().equals(s)) {
                    return t;
                }
                valid.append(valid.length() == 0 ? "" : ", ").append(t.name().toLowerCase(Locale.ROOT));
            }
            throw new IllegalArgumentException(PROP_ROPE_SCALING + "=" + value + " is invalid; use " + valid);
        }
    }

    private final ScalingType scalingType;
    private final Float scalingFactor;
    private final Float freqBase;
    private final Float yarnExtFactor;
    private final Float yarnAttnFactor;
    private final Float yarnBetaFast;
    private final Float yarnBetaSlow;
    private final Integer yarnOrigCtx;

    private RopeOverrides(Builder b) {
        this.scalingType = b.scalingType;
        this.scalingFactor = b.scalingFactor;
        this.freqBase = b.freqBase;
        this.yarnExtFactor = b.yarnExtFactor;
        this.yarnAttnFactor = b.yarnAttnFactor;
        this.yarnBetaFast = b.yarnBetaFast;
        this.yarnBetaSlow = b.yarnBetaSlow;
        this.yarnOrigCtx = b.yarnOrigCtx;
    }

    /** The overrides that set nothing: the model's stored rope parameters rule. */
    public static RopeOverrides defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** The rope overrides of a property set ({@code neuralfree.rope.*} and {@code neuralfree.yarn.*}); absent or
     *  blank keys are not set. */
    public static RopeOverrides fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        Builder b = builder();
        String v = value(p, PROP_ROPE_SCALING);
        if (v != null) {
            b.scalingType(ScalingType.parse(v));
        }
        v = value(p, PROP_ROPE_SCALE);
        if (v != null) {
            b.scalingFactor(parseFloat(PROP_ROPE_SCALE, v));
        }
        v = value(p, PROP_ROPE_FREQ_BASE);
        if (v != null) {
            b.freqBase(parseFloat(PROP_ROPE_FREQ_BASE, v));
        }
        v = value(p, PROP_ROPE_FREQ_SCALE);
        if (v != null) {
            b.freqScale(parseFloat(PROP_ROPE_FREQ_SCALE, v));
        }
        v = value(p, PROP_YARN_EXT_FACTOR);
        if (v != null) {
            b.yarnExtFactor(parseFloat(PROP_YARN_EXT_FACTOR, v));
        }
        v = value(p, PROP_YARN_ATTN_FACTOR);
        if (v != null) {
            b.yarnAttnFactor(parseFloat(PROP_YARN_ATTN_FACTOR, v));
        }
        v = value(p, PROP_YARN_BETA_FAST);
        if (v != null) {
            b.yarnBetaFast(parseFloat(PROP_YARN_BETA_FAST, v));
        }
        v = value(p, PROP_YARN_BETA_SLOW);
        if (v != null) {
            b.yarnBetaSlow(parseFloat(PROP_YARN_BETA_SLOW, v));
        }
        v = value(p, PROP_YARN_ORIG_CTX);
        if (v != null) {
            b.yarnOrigCtx(parseInt(PROP_YARN_ORIG_CTX, v));
        }
        return b.build();
    }

    /** Whether every override is unset. */
    public boolean isEmpty() {
        return scalingType == null && scalingFactor == null && freqBase == null && yarnExtFactor == null
                && yarnAttnFactor == null && yarnBetaFast == null && yarnBetaSlow == null && yarnOrigCtx == null;
    }

    public ScalingType scalingType() { return scalingType; }
    /** The rope context scaling factor ({@code rope.scaling.factor}); null when unset. */
    public Float scalingFactor() { return scalingFactor; }
    /** The rope frequency base ({@code rope.freq_base}); null when the model's own applies. */
    public Float freqBase() { return freqBase; }
    public Float yarnExtFactor() { return yarnExtFactor; }
    public Float yarnAttnFactor() { return yarnAttnFactor; }
    public Float yarnBetaFast() { return yarnBetaFast; }
    public Float yarnBetaSlow() { return yarnBetaSlow; }
    public Integer yarnOrigCtx() { return yarnOrigCtx; }

    public Builder toBuilder() {
        Builder b = builder();
        b.scalingType = scalingType;
        b.scalingFactor = scalingFactor;
        b.freqBase = freqBase;
        b.yarnExtFactor = yarnExtFactor;
        b.yarnAttnFactor = yarnAttnFactor;
        b.yarnBetaFast = yarnBetaFast;
        b.yarnBetaSlow = yarnBetaSlow;
        b.yarnOrigCtx = yarnOrigCtx;
        return b;
    }

    /**
     * The metadata overrides of {@code arch}'s {@code %s.rope.*} keys, one per set option, in the list order of the
     * options. Only set options appear; with nothing set the list is empty.
     *
     * @param arch the value of {@code general.architecture} the overrides are written for ({@code llama},
     *             {@code qwen3}, ...)
     */
    public List<MetadataOverrides.Override> toOverrides(String arch) {
        Objects.requireNonNull(arch, "arch");
        List<MetadataOverrides.Override> out = new ArrayList<>();
        if (scalingType != null) {
            out.add(new MetadataOverrides.Override(String.format(Locale.ROOT, GgufKeys.ROPE_SCALING_TYPE, arch),
                    GGUFValueType.STRING, scalingType.name().toLowerCase(Locale.ROOT)));
        }
        if (scalingFactor != null) {
            out.add(new MetadataOverrides.Override(
                    String.format(Locale.ROOT, GgufKeys.ROPE_SCALING_FACTOR, arch), GGUFValueType.FLOAT32,
                    scalingFactor));
        }
        if (freqBase != null) {
            out.add(new MetadataOverrides.Override(
                    String.format(Locale.ROOT, GgufKeys.ROPE_FREQ_BASE, arch), GGUFValueType.FLOAT32, freqBase));
        }
        if (yarnAttnFactor != null) {
            out.add(new MetadataOverrides.Override(
                    String.format(Locale.ROOT, GgufKeys.ROPE_SCALING_ATTN_FACTOR, arch), GGUFValueType.FLOAT32,
                    yarnAttnFactor));
        }
        if (yarnExtFactor != null) {
            out.add(new MetadataOverrides.Override(
                    String.format(Locale.ROOT, GgufKeys.ROPE_SCALING_YARN_EXT_FACTOR, arch), GGUFValueType.FLOAT32,
                    yarnExtFactor));
        }
        if (yarnBetaFast != null) {
            out.add(new MetadataOverrides.Override(
                    String.format(Locale.ROOT, GgufKeys.ROPE_SCALING_YARN_BETA_FAST, arch), GGUFValueType.FLOAT32,
                    yarnBetaFast));
        }
        if (yarnBetaSlow != null) {
            out.add(new MetadataOverrides.Override(
                    String.format(Locale.ROOT, GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, arch), GGUFValueType.FLOAT32,
                    yarnBetaSlow));
        }
        if (yarnOrigCtx != null && yarnOrigCtx > 0) {
            out.add(new MetadataOverrides.Override(
                    String.format(Locale.ROOT, GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, arch), GGUFValueType.INT32,
                    yarnOrigCtx));
        }
        return out;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof RopeOverrides)) {
            return false;
        }
        RopeOverrides that = (RopeOverrides) o;
        return scalingType == that.scalingType && Objects.equals(scalingFactor, that.scalingFactor)
                && Objects.equals(freqBase, that.freqBase) && Objects.equals(yarnExtFactor, that.yarnExtFactor)
                && Objects.equals(yarnAttnFactor, that.yarnAttnFactor)
                && Objects.equals(yarnBetaFast, that.yarnBetaFast) && Objects.equals(yarnBetaSlow, that.yarnBetaSlow)
                && Objects.equals(yarnOrigCtx, that.yarnOrigCtx);
    }

    @Override
    public int hashCode() {
        return Objects.hash(scalingType, scalingFactor, freqBase, yarnExtFactor, yarnAttnFactor, yarnBetaFast,
                yarnBetaSlow, yarnOrigCtx);
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder("RopeOverrides{");
        String sep = "";
        if (scalingType != null) {
            s.append("scaling=").append(scalingType.name().toLowerCase(Locale.ROOT));
            sep = ", ";
        }
        if (scalingFactor != null) {
            s.append(sep).append("scalingFactor=").append(scalingFactor);
            sep = ", ";
        }
        if (freqBase != null) {
            s.append(sep).append("freqBase=").append(freqBase);
            sep = ", ";
        }
        if (yarnExtFactor != null) {
            s.append(sep).append("yarnExtFactor=").append(yarnExtFactor);
            sep = ", ";
        }
        if (yarnAttnFactor != null) {
            s.append(sep).append("yarnAttnFactor=").append(yarnAttnFactor);
            sep = ", ";
        }
        if (yarnBetaFast != null) {
            s.append(sep).append("yarnBetaFast=").append(yarnBetaFast);
            sep = ", ";
        }
        if (yarnBetaSlow != null) {
            s.append(sep).append("yarnBetaSlow=").append(yarnBetaSlow);
            sep = ", ";
        }
        if (yarnOrigCtx != null) {
            s.append(sep).append("yarnOrigCtx=").append(yarnOrigCtx);
        }
        return s.append('}').toString();
    }

    private static String value(Properties p, String key) {
        String v = p.getProperty(key);
        if (v == null) {
            return null;
        }
        String s = v.trim();
        return s.isEmpty() ? null : s;
    }

    private static float parseFloat(String property, String v) {
        try {
            float f = Float.parseFloat(v);
            if (Float.isFinite(f)) {
                return f;
            }
        } catch (NumberFormatException e) {
            // reported below
        }
        throw new IllegalArgumentException(property + "=" + v + " is not a finite number");
    }

    private static int parseInt(String property, String v) {
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(property + "=" + v + " is not an integer");
        }
    }

    /** Unset everywhere; set what the CLI flags and properties name. */
    public static final class Builder {
        private ScalingType scalingType;
        private Float scalingFactor;
        private Float freqBase;
        private Float yarnExtFactor;
        private Float yarnAttnFactor;
        private Float yarnBetaFast;
        private Float yarnBetaSlow;
        private Integer yarnOrigCtx;

        private Builder() {
        }

        public Builder scalingType(ScalingType v) {
            this.scalingType = Objects.requireNonNull(v, PROP_ROPE_SCALING);
            return this;
        }

        /**
         * {@code --rope-scale N}: the context scales by {@code N} ({@code rope.scaling.factor = N}).
         *
         * @throws IllegalArgumentException when {@code v} is not finite and positive
         */
        public Builder scalingFactor(float v) {
            this.scalingFactor = positive(PROP_ROPE_SCALE, v);
            return this;
        }

        /**
         * {@code --rope-freq-scale N}: the rotation frequencies scale by {@code N}, so the context scales by
         * {@code 1 / N} ({@code rope.scaling.factor = 1 / N}), as llama.cpp's {@code rope_freq_scale}.
         *
         * @throws IllegalArgumentException when {@code v} is not finite and positive
         */
        public Builder freqScale(float v) {
            this.scalingFactor = 1.0f / positive(PROP_ROPE_FREQ_SCALE, v);
            return this;
        }

        public Builder freqBase(float v) {
            this.freqBase = finite(PROP_ROPE_FREQ_BASE, v);
            return this;
        }

        public Builder yarnExtFactor(float v) {
            this.yarnExtFactor = finite(PROP_YARN_EXT_FACTOR, v);
            return this;
        }

        public Builder yarnAttnFactor(float v) {
            this.yarnAttnFactor = finite(PROP_YARN_ATTN_FACTOR, v);
            return this;
        }

        public Builder yarnBetaFast(float v) {
            this.yarnBetaFast = finite(PROP_YARN_BETA_FAST, v);
            return this;
        }

        public Builder yarnBetaSlow(float v) {
            this.yarnBetaSlow = finite(PROP_YARN_BETA_SLOW, v);
            return this;
        }

        /** {@code --yarn-orig-ctx}: 0 keeps the model's training context (and stores nothing). */
        public Builder yarnOrigCtx(int v) {
            if (v < 0) {
                throw new IllegalArgumentException(PROP_YARN_ORIG_CTX + " must not be negative, got " + v);
            }
            this.yarnOrigCtx = v == 0 ? null : v;
            return this;
        }

        /** @throws IllegalArgumentException when a value is negative or an earlier one is out of range */
        public RopeOverrides build() {
            if (yarnOrigCtx != null && yarnOrigCtx < 0) {
                throw new IllegalArgumentException(PROP_YARN_ORIG_CTX + " must not be negative, got " + yarnOrigCtx);
            }
            return new RopeOverrides(this);
        }

        private static float finite(String what, float v) {
            if (!Float.isFinite(v)) {
                throw new IllegalArgumentException(what + " must be a finite float, got " + v);
            }
            return v;
        }

        private static float positive(String what, float v) {
            if (!(finite(what, v) > 0f)) {
                throw new IllegalArgumentException(what + " must be a positive float, got " + v);
            }
            return v;
        }
    }
}
