package com.eonmux.neuralfree.gguf.quant;

import java.util.Locale;
import java.util.Objects;

/**
 * How projection inputs are treated when a checkpoint declares activation quantization (the
 * {@code neuralfree.act_quant.*} metadata and the {@code <base>.input_scale} / {@code <base>.input_zero_point}
 * siblings). Selected by the engine property {@value #PROPERTY}.
 */
public enum ActivationQuant {

    /** Inputs stay in full precision and the activation scheme is ignored (weight-only, as llama.cpp runs). */
    WEIGHT_ONLY("weight-only"),
    /**
     * Inputs are quantized and dequantized per the checkpoint's scheme before the unchanged weight kernel, so the
     * products equal those of the checkpoint's integer or FP8 kernels up to F32 accumulation.
     */
    CHECKPOINT("checkpoint");

    /** The engine property selecting the mode. */
    public static final String PROPERTY = "neuralfree.act.quant";

    private final String propertyValue;

    ActivationQuant(String propertyValue) {
        this.propertyValue = propertyValue;
    }

    /** The mode's value of {@value #PROPERTY}. */
    public String propertyValue() {
        return propertyValue;
    }

    /**
     * The mode named by a {@value #PROPERTY} value, ignoring case, with {@code -} and {@code _} equivalent.
     *
     * @throws IllegalArgumentException for any other value, naming the property and the valid values
     */
    public static ActivationQuant parse(String value) {
        String s = Objects.requireNonNull(value, "activation quantization").trim().toLowerCase(Locale.ROOT)
                .replace('_', '-');
        StringBuilder valid = new StringBuilder();
        for (ActivationQuant q : values()) {
            if (q.propertyValue.equals(s)) {
                return q;
            }
            valid.append(valid.length() == 0 ? "" : ", ").append(q.propertyValue);
        }
        throw new IllegalArgumentException(PROPERTY + "=" + value + " is invalid; use " + valid);
    }
}
