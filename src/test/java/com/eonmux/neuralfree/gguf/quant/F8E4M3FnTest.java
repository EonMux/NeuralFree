package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GgufCompatConverter;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * ggml's per-element F8_E4M3FN type (id 43): the type table matches llama.cpp's traits (ggml.h:435,
 * ggml.c:776-783, ggml-cpu.c:302-307), the decoder is exact over every code, and the CPU matvec follows
 * {@code ggml_vec_dot_f8_e4m3_f32} (F32 activations, any row length) on whole rows and on rows that are not
 * multiples of 32.
 */
public class F8E4M3FnTest {

    /** Independent E4M3FN reference: sign, exponent (bias 7) and mantissa evaluated directly. */
    private static float refE4m3(int code) {
        int c = code & 0xFF;
        int s = c >>> 7;
        int e = (c >>> 3) & 0xF;
        int m = c & 0x7;
        float v;
        if (e == 0) {
            v = Math.scalb((float) m, -9);
        } else if (e == 15) {
            v = m == 7 ? Float.NaN : Math.scalb(1.0f + m / 8.0f, 8);
        } else {
            v = Math.scalb(1.0f + m / 8.0f, e - 7);
        }
        return s == 1 ? -v : v;
    }

    @Test
    public void theTypeTableMatchesGgml() {
        assertEquals(43, GGMLType.F8_E4M3FN.getId());
        assertEquals(1, GGMLType.F8_E4M3FN.getTypeSize());
        assertEquals(1, GGMLType.F8_E4M3FN.getBlockSize());
        assertEquals(GGMLType.F8_E4M3FN, GGMLType.fromId(43));
        assertFalse("ggml marks the type is_quantized = false", GGMLType.F8_E4M3FN.isQuantized());
        assertFalse("id 43 is upstream, not a NeuralFree extension", GGMLType.F8_E4M3FN.isExtension());
        assertTrue(BlockLayout.isDefined(GGMLType.F8_E4M3FN));
        assertEquals(1, BlockLayout.of(GGMLType.F8_E4M3FN).blockElements());
        assertEquals(1, BlockLayout.of(GGMLType.F8_E4M3FN).blockBytes());
        assertEquals(70, BlockLayout.of(GGMLType.F8_E4M3FN).bytesForElements(70));
        assertFalse("the conversion to llama.cpp's own type is exact", GgufCompatConverter.isLossy(
                GGMLType.F8_E4M3, GGMLType.F8_E4M3FN));
    }

    @Test
    public void theDecoderIsExactOverEveryCode() {
        byte[] raw = new byte[256];
        for (int i = 0; i < 256; i++) {
            raw[i] = (byte) i;
        }
        float[] y = CpuBlockDecoders.dequantize(raw, GGMLType.F8_E4M3FN, 256);
        for (int i = 0; i < 256; i++) {
            float ref = refE4m3(i);
            if (Float.isNaN(ref)) {
                assertTrue("code " + i + " decodes to NaN", Float.isNaN(y[i]));
            } else {
                assertEquals("code " + i, ref, y[i], 0f);
            }
        }
        assertEquals(FpBlockDecoders.e4m3ToFp32(0x08), y[0x08], 0f);
    }

    @Test
    public void theDecoderServesAnyLength() {
        byte[] raw = new byte[70];
        new Random(4).nextBytes(raw);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        float[] y = new float[70];
        CpuBlockDecoders.decodeElements(GGMLType.F8_E4M3FN, b, 0, y, 0, 70);
        for (int i = 0; i < 70; i++) {
            assertEquals(FpBlockDecoders.e4m3ToFp32(raw[i] & 0xFF), y[i], 0f);
        }
    }

    @Test
    public void activationsAreF32LikeGgmlVecDotF8E4m3F32() {
        assertEquals(GGMLType.F32, CpuActivationQuantizer.activationType(GGMLType.F8_E4M3FN));
        float[] x = {0.5f, -2f, 4f};
        QuantizedActivations a = CpuActivationQuantizer.quantizeFor(GGMLType.F8_E4M3FN, x, 0, 3);
        assertEquals(GGMLType.F32, a.type());
        assertEquals(3, a.length());
        assertEquals(0.5f, a.value(0), 0f);
        assertTrue(CpuMatVec.supports(GGMLType.F8_E4M3FN));
    }

    @Test
    public void theMatvecMatchesTheDecodedDotProductAtAnyRowLength() {
        Random rnd = new Random(0xF8E43L);
        for (int cols : new int[]{1, 7, 31, 32, 64, 70, 96, 130}) {
            int rows = 2;
            byte[] raw = new byte[rows * cols];
            rnd.nextBytes(raw);
            clearNaNs(raw);
            float[] x = new float[cols];
            for (int j = 0; j < cols; j++) {
                x[j] = (float) rnd.nextGaussian();
            }
            CpuTensorRef w = CpuTensorRef.of(GGMLType.F8_E4M3FN, ByteBuffer.wrap(raw), rows, cols);
            QuantizedActivations a = CpuActivationQuantizer.quantizeFor(GGMLType.F8_E4M3FN, x, 0, cols);
            CpuVecDot vd = CpuMatVec.vecDot(GGMLType.F8_E4M3FN);
            float[] wf = new float[cols];
            float[] y = new float[rows];
            vd.dotRows(w, 0, rows, a, y, 0);
            for (int r = 0; r < rows; r++) {
                w.decodeRow(r, wf, 0);
                // ggml_vec_dot_f8_e4m3_f32: eight float lanes over full groups of 8, then one product at a time
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
                int i = 0;
                for (; i + 8 <= cols; i += 8) {
                    s0 += wf[i] * x[i];
                    s1 += wf[i + 1] * x[i + 1];
                    s2 += wf[i + 2] * x[i + 2];
                    s3 += wf[i + 3] * x[i + 3];
                    s4 += wf[i + 4] * x[i + 4];
                    s5 += wf[i + 5] * x[i + 5];
                    s6 += wf[i + 6] * x[i + 6];
                    s7 += wf[i + 7] * x[i + 7];
                }
                float ref = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
                for (; i < cols; i++) {
                    ref += wf[i] * x[i];
                }
                assertEquals("cols " + cols + " row " + r, ref, vd.dotRow(w, r, a), 1e-4 * (Math.abs(ref) + 1f));
                assertEquals("cols " + cols + " row " + r + " via dotRows", ref, y[r],
                        1e-4 * (Math.abs(ref) + 1f));
            }
        }
    }

    @Test
    public void anE4m3NaNCodePropagatesThroughTheMatvec() {
        byte[] raw = new byte[32];
        raw[3] = (byte) 0x7F;
        CpuTensorRef w = CpuTensorRef.of(GGMLType.F8_E4M3FN, ByteBuffer.wrap(raw), 1, 32);
        float[] x = new float[32];
        QuantizedActivations a = CpuActivationQuantizer.quantizeFor(GGMLType.F8_E4M3FN, x, 0, 32);
        float y = CpuMatVec.vecDot(GGMLType.F8_E4M3FN).dotRow(w, 0, a);
        assertTrue("the NaN weight makes the row NaN", Float.isNaN(y));
    }

    private static void clearNaNs(byte[] raw) {
        for (int i = 0; i < raw.length; i++) {
            if ((raw[i] & 0x7F) == 0x7F) {
                raw[i] = (byte) (raw[i] & 0xFE);
            }
        }
    }
}
