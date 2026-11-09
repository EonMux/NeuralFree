package com.eonmux.neuralfree.gguf.gpu.forward;

import java.util.Locale;

/**
 * Selection mode of the int8-activation (Q8_1) matvec lane, the single-token decode path that pairs
 * every integer-dot capable weight type with {@code {T}_MatVec_Q8}: the projection input is quantized
 * once to Q8_1 ({@value GpuOps#QUANTIZE_Q8_1}) and the weight codes are dotted with the activation
 * codes by packed int8 dot products, as llama.cpp's MMVQ / vecdotq path does. The lane differs from
 * the float matvec by the 8-bit activation quantization error only.
 *
 * <ul>
 *   <li>{@link #AUTO}: the lane is used where the device reports
 *       {@link com.eonmux.neuralfree.acceleration.backend.DeviceFeature#INTEGER_DOT_PRODUCT}, the
 *       kernel is available, and the weight type measured faster with the lane on that device kind
 *       ({@link Int8MatvecTuning}) (the default).</li>
 *   <li>{@link #ON}: the lane is used whenever the kernels can run, also on a device that does not
 *       report packed dot products (the OpenCL kernels take the portable integer path there).</li>
 *   <li>{@link #OFF}: the lane is never selected; decode runs the exact F32-activation matvec.</li>
 * </ul>
 *
 * The mode is read from the system property {@value #PROPERTY}; an unknown value is refused by name.
 */
public enum Int8MatvecMode {

    AUTO, ON, OFF;

    /** System property holding the mode: {@code auto} (default), {@code on} or {@code off}. */
    public static final String PROPERTY = "neuralfree.matvec.int8";

    /**
     * The configured mode: {@value #PROPERTY}, or {@link #AUTO} when unset or blank.
     *
     * @throws IllegalArgumentException naming the property, the value and the allowed values when the
     *                                  property holds anything else
     */
    public static Int8MatvecMode configured() {
        String value = System.getProperty(PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return AUTO;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        for (Int8MatvecMode mode : values()) {
            if (mode.name().toLowerCase(Locale.ROOT).equals(normalized)) {
                return mode;
            }
        }
        throw new IllegalArgumentException("-D" + PROPERTY + " must be one of auto, on, off, got '" + value + "'");
    }

    /** Whether the mode permits the lane at all. */
    public boolean permitsLane() {
        return this != OFF;
    }

    /** Whether the lane may run on {@code caps} without the device reporting packed integer dot products. */
    public boolean forcesLane() {
        return this == ON;
    }
}
