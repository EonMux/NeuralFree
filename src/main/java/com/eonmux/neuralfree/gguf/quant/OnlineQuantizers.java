package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * The registry of the online-quantization seam, keyed by {@link OnlineQuant}. It holds the legacy block
 * types Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 and IQ4_NL, each delegating to its block quantizer of {@link KvBlockCodec}
 * (the reference quantizers of llama.cpp ggml-cpu/ggml-cpu.c type traits), and the vLLM online methods through
 * {@link OnlineQuantEncoders}: fp8_per_tensor, fp8_per_block, fp8_per_channel, int8_per_channel_weight_only,
 * experts_int8, nvfp4_per_token, mxfp8 and mxfp4. The tensor-scoped targets (fp8_per_tensor, fp8_per_block and
 * nvfp4_per_token) register prototypes of {@link ScanningQuantizer}: their quantizers need the whole tensor's
 * floats before any row, so {@link #scanning} hands out a fresh scan per tensor and {@link #get} is null for
 * them. {@link #register(OnlineQuant, OnlineQuantizer)} is the extension point the K-quant,
 * IQ and TQ quantizers register through.
 * <p>
 * {@code NONE} is not registered: it carries no quantizer and the planning step that resolves it treats it as a
 * no-op. Neither is {@code ONLINE}, whose quantization config resolves each layer to one of the registered methods.
 */
public final class OnlineQuantizers {

    private static final Map<OnlineQuant, OnlineQuantizer> REGISTRY = new EnumMap<>(OnlineQuant.class);
    private static final Map<OnlineQuant, ScanningQuantizer> SCANNING = new EnumMap<>(OnlineQuant.class);

    static {
        register(OnlineQuant.Q8_0, legacy(KvDType.Q8_0));
        register(OnlineQuant.Q4_0, legacy(KvDType.Q4_0));
        register(OnlineQuant.Q4_1, legacy(KvDType.Q4_1));
        register(OnlineQuant.Q5_0, legacy(KvDType.Q5_0));
        register(OnlineQuant.Q5_1, legacy(KvDType.Q5_1));
        register(OnlineQuant.IQ4_NL, legacy(KvDType.IQ4_NL));
        register(OnlineQuant.FP8_PER_CHANNEL, OnlineQuantEncoders.fp8PerChannel());
        register(OnlineQuant.INT8_PER_CHANNEL_WEIGHT_ONLY, OnlineQuantEncoders.int8PerChannel());
        register(OnlineQuant.EXPERTS_INT8, OnlineQuantEncoders.int8PerChannel());
        register(OnlineQuant.MXFP8, OnlineQuantEncoders.mxfp8());
        register(OnlineQuant.MXFP4, OnlineQuantEncoders.mxfp4());
        SCANNING.put(OnlineQuant.FP8_PER_TENSOR, OnlineQuantEncoders.fp8PerTensor());
        SCANNING.put(OnlineQuant.FP8_PER_BLOCK, OnlineQuantEncoders.fp8PerBlock());
        SCANNING.put(OnlineQuant.NVFP4_PER_TOKEN, OnlineQuantEncoders.nvfp4PerToken());
    }

    private OnlineQuantizers() {
    }

    /**
     * The row quantizer of {@code q}, or null for {@link OnlineQuant#NONE}, {@link OnlineQuant#ONLINE} and the
     * tensor-scoped targets ({@link #scanning} carries their prototype).
     *
     * @throws IllegalArgumentException when {@code q} is null
     */
    public static OnlineQuantizer get(OnlineQuant q) {
        Objects.requireNonNull(q, "q");
        return REGISTRY.get(q);
    }

    /**
     * The scanning prototype of a tensor-scoped target, or null for every other {@code q}: exactly one of
     * {@link #get} and this is non-null, except for {@link OnlineQuant#NONE} and {@link OnlineQuant#ONLINE}.
     *
     * @throws IllegalArgumentException when {@code q} is null
     */
    public static ScanningQuantizer scanning(OnlineQuant q) {
        Objects.requireNonNull(q, "q");
        return q == OnlineQuant.NONE ? null : SCANNING.get(q);
    }

    /**
     * Installs {@code quantizer} as the quantizer of {@code q}, replacing any registered one.
     *
     * @throws IllegalArgumentException when {@code q} is {@link OnlineQuant#NONE}, {@link OnlineQuant#ONLINE}, a
     *                                  tensor-scoped target, or {@code q} or {@code quantizer} is null
     */
    public static void register(OnlineQuant q, OnlineQuantizer quantizer) {
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(quantizer, "quantizer");
        if (q == OnlineQuant.NONE || q == OnlineQuant.ONLINE) {
            throw new IllegalArgumentException(q + " carries no quantizer");
        }
        if (SCANNING.containsKey(q)) {
            throw new IllegalArgumentException(q + " quantizes through a scan; there is no row quantizer to"
                    + " replace");
        }
        REGISTRY.put(q, quantizer);
    }

    /** A legacy block quantizer: every block of the row through {@link KvBlockCodec#quantize(KvDType, float[], int, byte[], int)}. */
    private static OnlineQuantizer legacy(KvDType type) {
        Objects.requireNonNull(type, "type");
        return new OnlineQuantizer() {
            @Override
            public GGMLType type() {
                return type.ggmlType();
            }

            @Override
            public void quantizeRow(float[] x, int off, int n, byte[] dst, int dstOff) {
                int bs = type.blockSize();
                if (n < 0 || n % bs != 0) {
                    throw new IllegalArgumentException("a row of " + n + " values is not a whole number of " + type
                            + " blocks of " + bs);
                }
                for (int b = 0; b < n; b += bs) {
                    KvBlockCodec.quantize(type, x, off + b, dst, dstOff + b / bs * type.blockBytes());
                }
            }
        };
    }
}
