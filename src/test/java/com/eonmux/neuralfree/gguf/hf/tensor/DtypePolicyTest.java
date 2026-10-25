package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.q4Rows;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.seq;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link DtypePolicy}: every step of the converter's type selection in order (kept integers, 1-D and norm tensors over
 * forced types, forced types over forced-F32 names and the weight-suffix rule, forced-F32 and non-weight names, the
 * output types including AUTO and the Q8_0 fallback, integer and no-GGUF-type rejections) and the value codec's
 * eight-bit float decoders.
 */
public class DtypePolicyTest {

    private static final DtypePolicy SOURCE = new DtypePolicy(n -> n.contains("ffn_gate_inp"), OutType.SOURCE);

    private static RowSource as(StDType t, long... shape) {
        int n = 1;
        for (long d : shape) {
            n *= (int) d;
        }
        return RopeTensors.fromBytes(ElementFormat.of(t), shape, new byte[n * t.bits() / 8]);
    }

    @Test
    public void keptIntegersBypassEveryStep() {
        RowSource d2t = ElementTransforms.keepInteger(RopeTensors.fromLongs(new long[]{2}, new long[]{1, 2}));
        assertEquals(GGMLType.I64, SOURCE.typeFor("d2t", d2t, null));
        assertEquals(GGMLType.I64, SOURCE.typeFor("d2t", d2t, GGMLType.F16));
    }

    @Test
    public void stepOneOneDimAndNormsAreF32EvenWhenForced() {
        RowSource bf = as(StDType.BF16, 4, 4);
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.attn_norm.weight", as(StDType.BF16, 4), null));
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.attn_q_norm.weight", bf, null));
        assertEquals(GGMLType.F32, SOURCE.typeFor("a.seanet.0.weight", as(StDType.BF16, 4), GGMLType.F16));
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.attn_q_norm.weight", bf, GGMLType.F16));
    }

    @Test
    public void stepTwoForcedTypeWinsOverTheForcedF32Names() {
        RowSource bf = as(StDType.BF16, 4, 4);
        assertEquals(GGMLType.F16, new DtypePolicy(n -> false, OutType.BF16).typeFor("x.weight", bf, GGMLType.F16));
        assertEquals(GGMLType.F16, SOURCE.typeFor("blk.0.ffn_gate_inp.weight", bf, GGMLType.F16));
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.ffn_gate_inp.weight", bf, null));
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.ffn_gate_inp.weight", q4Rows(2), null));
    }

    @Test
    public void stepTwoForcedTypeOverridesANameWithoutAWeightSuffix() {
        RowSource bf = as(StDType.BF16, 4, 4);
        assertEquals(GGMLType.F16, SOURCE.typeFor("blk.0.attn_q.bias", bf, GGMLType.F16));
        assertEquals(GGMLType.BF16, SOURCE.typeFor("blk.0.exp_probs_b", bf, GGMLType.BF16));
        assertEquals(GGMLType.F16, SOURCE.apply("blk.0.attn_q.bias", bf, GGMLType.F16).format().ggmlType());
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.attn_q.bias", as(StDType.BF16, 4), GGMLType.F16));
    }

    @Test
    public void stepThreeStoresNonWeightNamesAsF32WithoutAnOverride() {
        RowSource bf = as(StDType.BF16, 4, 4);
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.attn_q.bias", bf, null));
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.exp_probs_b", bf, null));
        assertEquals(GGMLType.BF16, SOURCE.typeFor("blk.0.attn_q.lora_a", bf, null));
        assertEquals(GGMLType.BF16, SOURCE.typeFor("blk.0.attn_q.lora_b", bf, null));
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.ssm_a", as(StDType.I32, 4), null));
        assertEquals(GGMLType.F32, SOURCE.typeFor("blk.0.attn_q.input_scale", as(StDType.I32, 2, 2), null));
    }

    @Test
    public void stepFourOutputTypes() {
        RowSource bf = as(StDType.BF16, 4, 32);
        assertEquals(GGMLType.BF16, SOURCE.typeFor("w.weight", bf, null));
        assertEquals(GGMLType.F32, SOURCE.typeFor("w.weight", as(StDType.F64, 4, 4), null));
        assertEquals(GGMLType.F32, new DtypePolicy(n -> false, OutType.F32).typeFor("w.weight", bf, null));
        assertEquals(GGMLType.F16, new DtypePolicy(n -> false, OutType.F16).typeFor("w.weight", bf, null));
        assertEquals(GGMLType.BF16, new DtypePolicy(n -> false, OutType.BF16).typeFor("w.weight",
                as(StDType.F32, 4, 4), null));
        DtypePolicy q8 = new DtypePolicy(n -> false, OutType.Q8_0);
        assertEquals(GGMLType.Q8_0, q8.apply("w.weight", bf, null).format().ggmlType());
        assertEquals(GGMLType.F16, q8.apply("w.weight", as(StDType.BF16, 4, 48), null).format().ggmlType());
        assertEquals(GGMLType.Q4_0, new DtypePolicy(n -> false, OutType.F16).typeFor("w.weight", q4Rows(2), null));
        RowSource f32 = seq(2, 2);
        assertSame(f32, new DtypePolicy(n -> false, OutType.F32).apply("w.weight", f32, null));
    }

    @Test
    public void autoFollowsTheFirstHalfPrecisionMatrix() {
        Map<String, RowSource> m = new LinkedHashMap<>();
        m.put("norm", as(StDType.BF16, 8));
        m.put("a", as(StDType.F32, 4, 4));
        m.put("b", as(StDType.F16, 4, 4));
        m.put("c", as(StDType.BF16, 4, 4));
        assertEquals(GGMLType.F16, DtypePolicy.guessAuto(HfTensorIndex.of(m)));
        m.remove("b");
        assertEquals(GGMLType.BF16, DtypePolicy.guessAuto(HfTensorIndex.of(m)));
        m.remove("c");
        assertEquals(GGMLType.F16, DtypePolicy.guessAuto(HfTensorIndex.of(m)));

        DtypePolicy auto = new DtypePolicy(n -> false, OutType.AUTO);
        assertThrows(IllegalStateException.class, () -> auto.typeFor("w.weight", as(StDType.F32, 4, 4), null));
        m.put("c", as(StDType.BF16, 4, 4));
        DtypePolicy resolved = auto.resolveAuto(HfTensorIndex.of(m));
        assertEquals(GGMLType.BF16, resolved.typeFor("w.weight", as(StDType.F32, 4, 4), null));
        assertSame(SOURCE, SOURCE.resolveAuto(HfTensorIndex.of(m)));
    }

    @Test
    public void stepsFiveAndSixRejectTensorsWithoutAGgufType() {
        GgufFormatException i = assertThrows(GgufFormatException.class,
                () -> SOURCE.typeFor("w.weight", as(StDType.I32, 4, 4), null));
        assertTrue(i.getMessage(), i.getMessage().contains("w.weight") && i.getMessage().contains("I32"));
        for (StDType t : new StDType[]{StDType.F8_E4M3, StDType.U8, StDType.BOOL, StDType.U16}) {
            GgufFormatException e = assertThrows(GgufFormatException.class,
                    () -> SOURCE.typeFor("blk.0.attn_q.weight", as(t, 4, 4), null));
            assertTrue(e.getMessage(), e.getMessage().contains("blk.0.attn_q.weight")
                    && e.getMessage().contains(t.headerName()));
        }
    }

    @Test
    public void decodesEightBitFloatsAndUnsignedIntegers() {
        ByteBuffer b = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
        b.put(0, (byte) 0x40).put(1, (byte) 0x80).put(2, (byte) 0x01);
        float[] out = new float[3];
        DtypePolicy.decode(ElementFormat.of(StDType.F8_E5M2FNUZ), b, 0, out, 0, 3);
        assertEquals(1f, out[0], 0f);
        assertTrue(Float.isNaN(out[1]));
        assertEquals(Math.scalb(1f, -17), out[2], 0f);
        b.putInt(4, 0xFFFF_FFFF);
        DtypePolicy.decode(ElementFormat.of(StDType.U32), b, 4, out, 0, 1);
        assertEquals(4294967295f, out[0], 0f);
        b.put(8, (byte) 0x7F);
        DtypePolicy.decode(ElementFormat.of(StDType.F8_E8M0), b, 8, out, 0, 1);
        assertEquals(1f, out[0], 0f);
        assertThrows(IllegalArgumentException.class,
                () -> DtypePolicy.decode(ElementFormat.of(StDType.F6_E2M3), b, 0, out, 0, 1));
    }
}
