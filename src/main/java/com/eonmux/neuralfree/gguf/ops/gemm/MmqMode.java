package com.eonmux.neuralfree.gguf.ops.gemm;

import java.util.Locale;

/**
 * Selection mode of the fast quantized-weight GEMM lane, the counterpart of llama.cpp's
 * {@code GGML_CUDA_FORCE_MMQ} / {@code GGML_CUDA_FORCE_CUBLAS} switches. The lane serves the llama.cpp
 * MMQ weight types with the int8 MMQ GEMM ({@code {T}_MatMul_MMQ}; on Vulkan devices with cooperative
 * matrices through the fp16-tile tensor-core forms, {@code {T}_MatMul_CM2} where the device reports
 * {@code VK_NV_cooperative_matrix2} and {@code {T}_MatMul_CM} otherwise) and differs from the exact
 * tiled GEMM by the fast form's rounding.
 *
 * <ul>
 *   <li>{@link #AUTO}: the dispatch picks the fast lane wherever its kernel can run, in the form the
 *       device runs best: the fp16-tile CM2 or CM form on Vulkan devices with cooperative matrices, the
 *       int8 MMQ GEMM everywhere else (the default).</li>
 *   <li>{@link #ON}: the int8 MMQ GEMM for every weight type of the lane on every device, also
 *       where {@link #AUTO} would take the CM form ({@code GGML_CUDA_FORCE_MMQ}).</li>
 *   <li>{@link #OFF}: the lane is never selected; projections of its weight types run the exact
 *       tiled GEMM, for callers that do not accept the fast lane's activation rounding
 *       ({@code GGML_CUDA_FORCE_CUBLAS}).</li>
 * </ul>
 *
 * The mode is read from the system property {@value #PROPERTY}; an unknown value is refused by name.
 * F16 and BF16 weights are not part of the lane: their tensor-core MMA GEMM rounds in 16-bit floats
 * by construction and stays on whenever the device reports tensor cores.
 */
public enum MmqMode {

    AUTO, ON, OFF;

    /** System property holding the mode: {@code auto} (default), {@code on} or {@code off}. */
    public static final String PROPERTY = "neuralfree.gemm.mmq";

    /**
     * The configured mode: {@value #PROPERTY}, or {@link #AUTO} when unset or blank.
     *
     * @throws IllegalArgumentException naming the property, the value and the allowed values when the
     *                               property holds anything else
     */
    public static MmqMode configured() {
        String value = System.getProperty(PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return AUTO;
        }
        String trimmed = value.trim();
        for (MmqMode mode : values()) {
            if (mode.name().toLowerCase(Locale.ROOT).equals(trimmed)) {
                return mode;
            }
        }
        throw new IllegalArgumentException("-D" + PROPERTY + " must be one of auto, on, off, got '" + value + "'");
    }

    /** Whether the fast quantized-weight GEMM lane may be selected at all. */
    public boolean fastQuantGemm() {
        return this != OFF;
    }

    /** Whether the lane may take the fp16-tile CM form where the device has cooperative matrices. */
    public boolean prefersCm() {
        return this == AUTO;
    }
}
