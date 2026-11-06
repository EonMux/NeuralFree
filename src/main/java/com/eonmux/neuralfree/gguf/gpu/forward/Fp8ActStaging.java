package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;

import java.util.Objects;

/**
 * The direct E4M3 staging of one projection's activations: the checkpoint activation scheme is applied by its own
 * staging kernel into the staged record layout as E4M3 codes with one scale per chunk, so the E4M3 GEMMs
 * ({@code <T>_MatMul_MMQ} for the FP8 weight types and {@code <T>_MatMul_W4A8} for the 4-bit ones) multiply the
 * checkpoint's own activation codes instead of a fake-quantized float row. Immutable.
 *
 * <p>The schemes staged this way are the FP8 forms ({@link ActQuantScheme#FP8_TOKEN},
 * {@link ActQuantScheme#FP8_STATIC}, {@link ActQuantScheme#FP8_GROUP128}, {@link ActQuantScheme#MXFP8}) and the
 * 4-bit forms re-encoded as E4M3 ({@link ActQuantScheme#MXFP4}, {@link ActQuantScheme#NVFP4}). The other schemes
 * keep the fake-quantization path: the int8 schemes stage into the int8 record layout
 * ({@link Int8ActStaging}), and the schemes without a staging kernel are fake-quantized before the GEMM.</p>
 */
final class Fp8ActStaging {

    /** Launch geometry of the staging kernel over the projection input. */
    enum Kind {
        /** One work-item per (token, chunk): the per-chunk schemes with no reduction. */
        CHUNK,
        /** One work-group per token row: {@link ActQuantScheme#FP8_TOKEN}'s row amax. */
        ROW,
        /** One work-group per 128-element run: {@link ActQuantScheme#FP8_GROUP128}. */
        RUN128,
        /** One work-group per (token, chunk): the NVFP4 16-element-run reductions. */
        CHUNK_GROUP
    }

    private final ActQuantScheme scheme;
    private final String kernel;
    private final float argument;
    private final boolean readsArgument;
    private final Kind kind;

    Fp8ActStaging(ActQuantScheme scheme, String kernel, float argument, boolean readsArgument, Kind kind) {
        this.scheme = Objects.requireNonNull(scheme, "scheme");
        this.kernel = Objects.requireNonNull(kernel, "kernel");
        this.argument = argument;
        this.readsArgument = readsArgument;
        this.kind = Objects.requireNonNull(kind, "kind");
    }

    /** The activation scheme the staging kernel reproduces. */
    ActQuantScheme scheme() {
        return scheme;
    }

    /** The staging kernel name. */
    String kernel() {
        return kernel;
    }

    /**
     * The kernel's scalar argument: the {@code input_scale} of the static scheme, the amax bound of
     * {@link ActQuantScheme#FP8_TOKEN}, the global scale of {@link ActQuantScheme#NVFP4}; 0 for the schemes that
     * read none.
     */
    float argument() {
        return argument;
    }

    /** Whether the kernel's argument list carries the scalar ({@link #argument()}) before the buffers. */
    boolean readsArgument() {
        return readsArgument;
    }

    /** The launch geometry of the staging kernel. */
    Kind kind() {
        return kind;
    }

    @Override
    public String toString() {
        return "Fp8ActStaging{" + scheme + ", " + kernel + '}';
    }
}
