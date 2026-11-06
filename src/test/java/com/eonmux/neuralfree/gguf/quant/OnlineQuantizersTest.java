package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

/**
 * {@link OnlineQuantizers}: the legacy block types quantize rows exactly as {@link KvBlockCodec} stores the
 * blocks, a registered extension replaces the default it is keyed to and is used until restored, and NONE carries no
 * quantizer.
 */
public class OnlineQuantizersTest {

    /** A registered fake: its whole row is marked with one tag byte. */
    private static final OnlineQuantizer FAKE = new OnlineQuantizer() {
        @Override
        public GGMLType type() {
            return GGMLType.F32;
        }

        @Override
        public void quantizeRow(float[] x, int off, int n, byte[] dst, int dstOff) {
            dst[dstOff] = (byte) 0x5A;
        }
    };

    /** The floats the quantized rows hold: negatives, magnitudes around every block quantizer's range. */
    private static float[] floats(int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) (((i % 32) - 16) * Math.pow(2, (i / 32 % 7) - 3));
        }
        return x;
    }

    @Test
    public void legacyRowsEqualTheKvBlockCodecOutput() {
        int n = 3 * 32;
        float[] x = floats(n);
        for (OnlineQuant q : new OnlineQuant[]{OnlineQuant.Q8_0, OnlineQuant.Q4_0, OnlineQuant.Q4_1,
                OnlineQuant.Q5_0, OnlineQuant.Q5_1, OnlineQuant.IQ4_NL}) {
            OnlineQuantizer quantizer = OnlineQuantizers.get(q);
            GGMLType type = quantizer.type();
            assertEquals(q.name(), GGMLType.valueOf(q.name()), type);
            byte[] mine = new byte[(int) type.getBytesForElements(n)];
            quantizer.quantizeRow(x, 0, n, mine, 0);
            byte[] want = new byte[(int) type.getBytesForElements(n)];
            for (int b = 0; b < n / 32; b++) {
                KvBlockCodec.quantize(KvDType.valueOf(q.name()), x, b * 32, want, b * type.getTypeSize());
            }
            assertArrayEquals(q.name(), want, mine);
        }
    }

    /** A row requested at offsets far from the arrays' starts, still byte-equal to the reference. */
    @Test
    public void offsetRowsEqualTheKvBlockCodecOutputAtTheSameOffsets() {
        float[] src = floats(5 * 32);
        OnlineQuantizer quantizer = OnlineQuantizers.get(OnlineQuant.IQ4_NL);
        byte[] mine = new byte[40];
        quantizer.quantizeRow(src, 96, 32, mine, 20);
        byte[] want = new byte[40];
        KvBlockCodec.quantize(KvDType.IQ4_NL, src, 96, want, 20);
        assertArrayEquals(want, mine);
    }

    @Test
    public void aRegisteredQuantizerReplacesAndIsUsedUntilRestored() {
        OnlineQuantizer replacement = OnlineQuantizers.get(OnlineQuant.Q5_1);
        OnlineQuantizers.register(OnlineQuant.Q5_1, FAKE);
        try {
            assertSame(FAKE, OnlineQuantizers.get(OnlineQuant.Q5_1));
            byte[] dst = new byte[16];
            OnlineQuantizers.get(OnlineQuant.Q5_1).quantizeRow(new float[32], 0, 32, dst, 4);
            assertEquals("the registered fake quantized the row", 0x5A, dst[4]);
            assertEquals("only the fake's slot is touched", OnlineQuantizers.get(OnlineQuant.Q4_0).type(),
                    GGMLType.Q4_0);
        } finally {
            OnlineQuantizers.register(OnlineQuant.Q5_1, replacement);
        }
        assertSame(replacement, OnlineQuantizers.get(OnlineQuant.Q5_1));
    }

    @Test
    public void everyVllmMethodHasExactlyOneQuantizerOfItsTargetType() {
        assertSame(GGMLType.F8_E4M3_B128, OnlineQuantizers.get(OnlineQuant.FP8_PER_CHANNEL).type());
        assertSame(GGMLType.I8_B128, OnlineQuantizers.get(OnlineQuant.INT8_PER_CHANNEL_WEIGHT_ONLY).type());
        assertSame(GGMLType.I8_B128, OnlineQuantizers.get(OnlineQuant.EXPERTS_INT8).type());
        assertSame(GGMLType.MXFP8, OnlineQuantizers.get(OnlineQuant.MXFP8).type());
        assertSame(GGMLType.MXFP4, OnlineQuantizers.get(OnlineQuant.MXFP4).type());
        assertSame(GGMLType.F8_E4M3_B128, OnlineQuantizers.scanning(OnlineQuant.FP8_PER_TENSOR).type());
        assertSame(GGMLType.F8_E4M3_B128, OnlineQuantizers.scanning(OnlineQuant.FP8_PER_BLOCK).type());
        assertSame(GGMLType.NVFP4, OnlineQuantizers.scanning(OnlineQuant.NVFP4_PER_TOKEN).type());
        for (OnlineQuant q : OnlineQuant.values()) {
            boolean row = OnlineQuantizers.get(q) != null;
            boolean scan = OnlineQuantizers.scanning(q) != null;
            if (q == OnlineQuant.NONE || q == OnlineQuant.ONLINE) {
                assertEquals(q.name() + " carries no quantizer", false, row || scan);
            } else {
                assertEquals(q.name() + " has exactly one quantizer", true, row ^ scan);
            }
        }
        assertThrows(IllegalArgumentException.class,
                () -> OnlineQuantizers.register(OnlineQuant.FP8_PER_TENSOR, FAKE));
    }

    @Test
    public void noneCarriesNoQuantizerAndTheRegistryRefusesNullFakes() {
        assertNull(OnlineQuantizers.get(OnlineQuant.NONE));
        assertThrows(IllegalArgumentException.class, () -> OnlineQuantizers.register(OnlineQuant.NONE, FAKE));
        assertThrows(IllegalArgumentException.class, () -> OnlineQuantizers.register(OnlineQuant.ONLINE, FAKE));
        assertThrows(NullPointerException.class, () -> OnlineQuantizers.register(OnlineQuant.Q4_0, null));
        assertThrows(NullPointerException.class, () -> OnlineQuantizers.get(null));
        assertSame(FAKE.type(), GGMLType.F32);
    }
}
