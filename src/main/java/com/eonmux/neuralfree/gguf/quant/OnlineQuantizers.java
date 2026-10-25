package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * The registry of the online-quantization seam (QZ-1), keyed by {@link OnlineQuant}. It holds the legacy block types
 * Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 and IQ4_NL, each delegating to its block quantizer of {@link KvBlockCodec} (the
 * reference quantizers of llama.cpp ggml-cpu/ggml-cpu.c type traits). {@link #register(OnlineQuant, OnlineQuantizer)}
 * is the extension point PAR QN-4 registers the K-quant, IQ, TQ and FP8 quantizers through.
 * <p>
 * {@code NONE} is not registered: it carries no quantizer and the planning step that resolves it treats it as a
 * no-op.
 */
public final class OnlineQuantizers {

    private static final Map<OnlineQuant, OnlineQuantizer> REGISTRY = new EnumMap<>(OnlineQuant.class);

    static {
        register(OnlineQuant.Q8_0, legacy(KvDType.Q8_0));
        register(OnlineQuant.Q4_0, legacy(KvDType.Q4_0));
        register(OnlineQuant.Q4_1, legacy(KvDType.Q4_1));
        register(OnlineQuant.Q5_0, legacy(KvDType.Q5_0));
        register(OnlineQuant.Q5_1, legacy(KvDType.Q5_1));
        register(OnlineQuant.IQ4_NL, legacy(KvDType.IQ4_NL));
    }

    private OnlineQuantizers() {
    }

    /**
     * The quantizer of {@code q}, or null for {@link OnlineQuant#NONE}.
     *
     * @throws IllegalArgumentException when {@code q} is null
     */
    public static OnlineQuantizer get(OnlineQuant q) {
        Objects.requireNonNull(q, "q");
        return REGISTRY.get(q);
    }

    /**
     * Installs {@code quantizer} as the quantizer of {@code q}, replacing any registered one.
     *
     * @throws IllegalArgumentException when {@code q} is {@link OnlineQuant#NONE}, {@code q} or {@code quantizer}
     *                                  is null
     */
    public static void register(OnlineQuant q, OnlineQuantizer quantizer) {
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(quantizer, "quantizer");
        if (q == OnlineQuant.NONE) {
            throw new IllegalArgumentException("NONE carries no quantizer");
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
