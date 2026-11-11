package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Host copies of an F16, BF16 or Q8_0 KV cache for the attention kernel tests: the values a cache stores for a row
 * of floats (rounded to F16 or BF16, or quantized to Q8_0 blocks and dequantized) and the cache's bytes. Q8_0
 * follows llama.cpp {@code quantize_row_q8_0_ref}: per 32 values an F16 scale {@code d = amax / 127}, then
 * {@code round(x / d)} half away from zero. Q8_0 inputs hold a whole number of blocks.
 */
final class KvCacheTestValues {

    private static final int Q8_BLOCK = 32;
    private static final int Q8_BLOCK_BYTES = 34;

    private KvCacheTestValues() {
    }

    /** The values a cache of {@code type} reads back for {@code x}. */
    static float[] stored(KvDType type, float[] x) {
        float[] r = new float[x.length];
        if (type == KvDType.Q8_0) {
            for (int b = 0; b < x.length; b += Q8_BLOCK) {
                float d = FpBlockDecoders.fp16ToFp32(q8Scale(x, b));
                for (int i = 0; i < Q8_BLOCK; i++) {
                    r[b + i] = d * q8Code(x, b, i);
                }
            }
            return r;
        }
        for (int i = 0; i < x.length; i++) {
            r[i] = type == KvDType.F16 ? FpBlockDecoders.fp16ToFp32(FpBlockDecoders.fp32ToFp16(x[i]))
                    : Float.intBitsToFloat(Float.floatToIntBits(x[i]) & 0xFFFF0000);
        }
        return r;
    }

    /** The bytes of a cache of {@code type} holding {@code x} (values {@link #stored} already returned). */
    static byte[] bytes(KvDType type, float[] x) {
        if (type == KvDType.Q8_0) {
            ByteBuffer bb = ByteBuffer.allocate(x.length / Q8_BLOCK * Q8_BLOCK_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            for (int b = 0; b < x.length; b += Q8_BLOCK) {
                bb.putShort((short) q8Scale(x, b));
                for (int i = 0; i < Q8_BLOCK; i++) {
                    bb.put((byte) q8Code(x, b, i));
                }
            }
            return bb.array();
        }
        ByteBuffer bb = ByteBuffer.allocate(x.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : x) {
            bb.putShort((short) (type == KvDType.F16 ? FpBlockDecoders.fp32ToFp16(v)
                    : Float.floatToIntBits(v) >>> 16));   /* v is already representable */
        }
        return bb.array();
    }

    private static float q8Step(float[] x, int b) {
        float amax = 0f;
        for (int i = 0; i < Q8_BLOCK; i++) {
            amax = Math.max(amax, Math.abs(x[b + i]));
        }
        return amax / 127f;
    }

    private static int q8Scale(float[] x, int b) {
        return FpBlockDecoders.fp32ToFp16(q8Step(x, b));
    }

    private static int q8Code(float[] x, int b, int i) {
        float d = q8Step(x, b);
        float v = x[b + i];
        return d == 0f ? 0 : Math.round(Math.abs(v / d)) * (v < 0 ? -1 : 1);
    }
}
