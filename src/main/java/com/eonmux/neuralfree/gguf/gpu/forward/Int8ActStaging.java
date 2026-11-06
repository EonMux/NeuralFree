package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;

import java.util.Objects;

/**
 * The direct int8 staging of one projection's activations: the checkpoint activation scheme is applied by its own
 * staging kernel into the MMQ record layout, so the int8 MMQ GEMM multiplies the checkpoint's int8 codes instead of
 * a fake-quantized float row that the GEMM requantizes. Immutable.
 *
 * <p>Only the int8 schemes are staged this way; the FP8, FP4 and MX schemes keep the fake-quantization path. A
 * weight whose input transform permutes its columns also keeps that path: the permutation runs after the
 * activation quantization, and the staged records are not permutable.</p>
 */
final class Int8ActStaging {

    private final ActQuantScheme scheme;
    private final String kernel;
    private final float scale;
    private final int zeroPoint;

    Int8ActStaging(ActQuantScheme scheme, String kernel, float scale, int zeroPoint) {
        this.scheme = Objects.requireNonNull(scheme, "scheme");
        this.kernel = Objects.requireNonNull(kernel, "kernel");
        this.scale = scale;
        this.zeroPoint = zeroPoint;
    }

    /** The activation scheme the staging kernel reproduces. */
    ActQuantScheme scheme() {
        return scheme;
    }

    /** The staging kernel name. */
    String kernel() {
        return kernel;
    }

    /** The {@code input_scale} argument of the static scheme; 0 for the dynamic schemes. */
    float scale() {
        return scale;
    }

    /** The {@code input_zero_point} argument of the asymmetric scheme; 0 for every other scheme. */
    int zeroPoint() {
        return zeroPoint;
    }

    @Override
    public String toString() {
        return "Int8ActStaging{" + scheme + ", " + kernel + '}';
    }
}
