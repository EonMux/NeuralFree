package com.eonmux.neuralfree.gguf.quant;

import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertEquals;

/**
 * The half-to-float widening of {@link FpBlockDecoders}: every one of the 65536 halves, NaN payloads included,
 * widens to the bits of the port of {@code ggml_compute_fp16_to_fp32} through the scalar entry, both readers and
 * the row decoder. The loops repeat so the JIT-compiled forms are checked too.
 */
public class Fp16WideningTest {

    private static void assertSameBits(int h, float expected, float actual) {
        assertEquals("half 0x" + Integer.toHexString(h), Float.floatToRawIntBits(expected),
                Float.floatToRawIntBits(actual));
    }

    @Test
    public void everyHalfWidensToThePortsBits() {
        for (int round = 0; round < 20; round++) {
            for (int h = 0; h < 1 << 16; h++) {
                final float expected = FpBlockDecoders.computeFp16ToFp32(h);
                assertSameBits(h, expected, FpBlockDecoders.fp16ToFp32(h));
                assertSameBits(h, expected, FpBlockDecoders.fp16ToFp32((short) h));
            }
        }
    }

    @Test
    public void theReadersAndTheRowDecoderWidenEveryHalfAlike() {
        ByteBuffer le = ByteBuffer.allocate(2 << 16).order(ByteOrder.LITTLE_ENDIAN);
        for (int h = 0; h < 1 << 16; h++) {
            le.putShort(2 * h, (short) h);
        }
        byte[] bytes = le.array();
        float[] row = new float[1 << 16];
        for (int round = 0; round < 10; round++) {
            FpBlockDecoders.decodeF16(le, 0, row, 0, row.length);
            for (int h = 0; h < 1 << 16; h++) {
                final float expected = FpBlockDecoders.computeFp16ToFp32(h);
                assertSameBits(h, expected, FpBlockDecoders.fp16At(le, 2 * h));
                assertSameBits(h, expected, FpBlockDecoders.fp16At(bytes, 2 * h));
                assertSameBits(h, expected, row[h]);
            }
        }
    }
}
