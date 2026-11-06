package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.Fp8SchemeTest.Fixture;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Fp4Scheme}: NVFP4 sub-scales, nibble order and global scales (compressed-tensors inverted, ModelOpt as
 * stored) with their fallback, MXFP4 from compressed-tensors and gpt-oss layouts (identical blocks), DeepSeek-V4
 * experts, E8M0 NaN rejection, and the MXFP6 E2M3/E3M2 repack to native blocks with the exponent limits. Values are
 * checked exactly against {@code code * scale} through the CPU block decoders.
 */
public class Fp4SchemeTest {

    private static final String M = "model.layers.0.mlp.down_proj";
    private static final String P = "model.layers.0.mlp.experts.gate_up_proj";
    /** E4M3 sub-scales: 1, 0.5, 2.25, 2^-6, 0.875, 448, 0, 1.25. */
    private static final int[] SUB_SCALES = {0x38, 0x30, 0x41, 0x08, 0x36, 0x7E, 0x00, 0x3A};

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Fixture fx;

    @After
    public void close() throws IOException {
        if (fx != null) {
            fx.close();
        }
    }

    private Fixture fixture() throws IOException {
        close();
        fx = new Fixture(tmp.getRoot().toPath());
        return fx;
    }

    private HfConfig config(String json) {
        return Fixture.config(tmp.getRoot().toPath(), json);
    }

    private HfConfig plain() {
        return config("{'architectures':['A']}");
    }

    static int[] codes(int n, int salt) {
        int[] c = new int[n];
        for (int i = 0; i < n; i++) {
            c[i] = (i * 5 + i / 7 + salt) & 0x0F;
        }
        return c;
    }

    /** Two codes per byte, element {@code 2j} in the low nibble. */
    static byte[] packLowFirst(int[] codes) {
        byte[] b = new byte[codes.length / 2];
        for (int j = 0; j < b.length; j++) {
            b[j] = (byte) (codes[2 * j] | codes[2 * j + 1] << 4);
        }
        return b;
    }

    private static byte[] bytes(int... v) {
        byte[] b = new byte[v.length];
        for (int i = 0; i < v.length; i++) {
            b[i] = (byte) v[i];
        }
        return b;
    }

    private static byte[] subScales(int n) {
        byte[] b = new byte[n];
        for (int i = 0; i < n; i++) {
            b[i] = (byte) SUB_SCALES[i % SUB_SCALES.length];
        }
        return b;
    }

    @Test
    public void compressedTensorsNvfp4KeepsSubScalesAndNibbleOrderAndInvertsGlobalScales() throws IOException {
        int k = 128;
        int[] c = codes(2 * k, 0);
        byte[] s = subScales(2 * k / 16);
        HfTensorIndex in = fixture().add(M + ".weight_packed", StDType.U8, new long[]{2, k / 2}, packLowFirst(c))
                .add(M + ".weight_scale", StDType.F8_E4M3, new long[]{2, k / 16}, s)
                .f32(M + ".weight_global_scale", new long[]{1}, 4f)
                .f32(M + ".input_global_scale", new long[]{1}, 8f).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.NVFP4_CT).actScheme(ActQuantScheme.NVFP4).build();
        HfTensorIndex out = Fp4Scheme.rewriteModules(args, List.of(M), plain(), in, ModelSourceOptions.defaults());
        RowSource w = out.get(M + ".weight");
        assertEquals(GGMLType.NVFP4, w.format().ggmlType());
        assertArrayEquals(new long[]{2, k}, w.shape());
        assertArrayEquals(new float[]{0.25f}, w.outputScale(), 0f);
        assertArrayEquals(new float[]{0.125f}, w.inputScale(), 0f);
        assertEquals(ActQuantScheme.NVFP4, w.actQuant());
        for (String gone : new String[]{".weight_packed", ".weight_scale", ".weight_global_scale",
            ".input_global_scale"}) {
            assertFalse(gone, out.has(M + gone));
        }
        float[] v = Fixture.values(w);
        for (int i = 0; i < c.length; i++) {
            assertEquals("element " + i, FpBlockDecoders.e2m1ToFp32(c[i])
                    * FpBlockDecoders.e4m3ToFp32(s[i / 16]), v[i], 0f);
        }
        float[] folded = Fixture.values(w.foldOutputScale());
        assertNull(w.foldOutputScale().outputScale());
        for (int i = 0; i < c.length; i++) {
            assertEquals(FpBlockDecoders.e2m1ToFp32(c[i]) * FpBlockDecoders.e4m3ToFp32(s[i / 16]) * 0.25f,
                    folded[i], 0f);
        }
    }

    @Test
    public void negativeOrNanSubScalesAreRejected() throws IOException {
        for (int bad : new int[]{0x7F, 0xFF, 0xB8}) {
            byte[] s = subScales(4);
            s[2] = (byte) bad;
            HfTensorIndex in = fixture().add(M + ".weight_packed", StDType.U8, new long[]{1, 32},
                    packLowFirst(codes(64, 1))).add(M + ".weight_scale", StDType.F8_E4M3, new long[]{1, 4}, s)
                    .f32(M + ".weight_global_scale", new long[]{}, 1f).index();
            RowSource w = Fp4Scheme.rewriteModules(QuantArgs.builder(QuantFormat.NVFP4_CT).build(), List.of(M),
                    plain(), in, ModelSourceOptions.defaults()).get(M + ".weight");
            IOException e = assertThrows(IOException.class, () -> Fixture.bytes(w));
            assertTrue(e.getMessage(), e.getMessage().contains("block 2"));
        }
    }

    @Test
    public void modelOptNvfp4UsesScale2AndInputScaleAsStored() throws IOException {
        int[] c = codes(64, 3);
        byte[] s = subScales(4);
        HfTensorIndex in = fixture().add(M + ".weight", StDType.U8, new long[]{1, 32}, packLowFirst(c))
                .add(M + ".weight_scale", StDType.F8_E4M3, new long[]{1, 4}, s)
                .f32(M + ".weight_scale_2", new long[]{}, 0.5f)
                .f32(M + ".input_scale", new long[]{}, 2f).index();
        QuantArgs w4a4 = QuantArgs.builder(QuantFormat.NVFP4_MODELOPT).actScheme(ActQuantScheme.NVFP4).build();
        RowSource w = Fp4Scheme.rewriteModules(w4a4, List.of(M), plain(), in, ModelSourceOptions.defaults())
                .get(M + ".weight");
        assertArrayEquals(new float[]{0.5f}, w.outputScale(), 0f);
        assertArrayEquals(new float[]{2f}, w.inputScale(), 0f);
        assertFalse(Fp4Scheme.weightOnlyNvfp4(w4a4));
        QuantArgs w4a16 = QuantArgs.builder(QuantFormat.NVFP4_MODELOPT).build();
        assertTrue(Fp4Scheme.weightOnlyNvfp4(w4a16));
        RowSource wo = Fp4Scheme.rewriteModules(w4a16, List.of(M), plain(), in, ModelSourceOptions.defaults())
                .get(M + ".weight");
        assertNull(wo.actQuant());
        assertNull(wo.inputScale());
    }

    @Test
    public void quarkNvfp4UsesScale2AndInputScale2AsStored() throws IOException {
        int[] c = codes(2 * 64, 4);
        byte[] s = subScales(2 * 64 / 16);
        HfTensorIndex in = fixture().add(M + ".weight", StDType.U8, new long[]{2, 32}, packLowFirst(c))
                .add(M + ".weight_scale", StDType.F8_E4M3, new long[]{2, 4}, s)
                .f32(M + ".weight_scale_2", new long[]{}, 0.5f)
                .f32(M + ".input_scale_2", new long[]{}, 2f).index();
        QuantArgs a = QuantArgs.builder(QuantFormat.NVFP4_QUARK).actScheme(ActQuantScheme.NVFP4).build();
        HfTensorIndex out = Fp4Scheme.rewriteModules(a, List.of(M), plain(), in, ModelSourceOptions.defaults());
        RowSource w = out.get(M + ".weight");
        assertEquals(GGMLType.NVFP4, w.format().ggmlType());
        assertArrayEquals(new long[]{2, 64}, w.shape());
        assertArrayEquals(new float[]{0.5f}, w.outputScale(), 0f);
        assertArrayEquals(new float[]{2f}, w.inputScale(), 0f);
        assertEquals(ActQuantScheme.NVFP4, w.actQuant());
        for (String gone : new String[]{".weight_scale", ".weight_scale_2", ".input_scale_2"}) {
            assertFalse(gone, out.has(M + gone));
        }
        assertFalse(Fp4Scheme.weightOnlyNvfp4(a));
        float[] v = Fixture.values(w);
        for (int i = 0; i < c.length; i++) {
            assertEquals("element " + i, FpBlockDecoders.e2m1ToFp32(c[i])
                    * FpBlockDecoders.e4m3ToFp32(s[i / 16]), v[i], 0f);
        }
    }

    @Test
    public void nvfp4WithKNotAMultipleOf64FallsBackWithTheGlobalScaleFolded() throws IOException {
        int[] c = codes(2 * 32, 5);
        byte[] s = subScales(4);
        HfTensorIndex in = fixture().add(M + ".weight_packed", StDType.U8, new long[]{2, 16}, packLowFirst(c))
                .add(M + ".weight_scale", StDType.F8_E4M3, new long[]{2, 2}, s)
                .f32(M + ".weight_global_scale", new long[]{1}, 2f).index();
        RowSource w = Fp4Scheme.rewriteModules(QuantArgs.builder(QuantFormat.NVFP4_CT).build(), List.of(M), plain(),
                in, ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(StDType.BF16, w.format().stDType());
        assertNull(w.outputScale());
        float[] v = Fixture.values(w);
        for (int i = 0; i < c.length; i++) {
            assertEquals(FpBlockDecoders.e2m1ToFp32(c[i]) * FpBlockDecoders.e4m3ToFp32(s[i / 16]) * 0.5f, v[i], 0f);
        }
    }

    @Test
    public void compressedTensorsAndGptOssMxfp4GiveTheSameBlocks() throws IOException {
        int k = 64;
        int[] c = codes(2 * k, 2);
        byte[] packed = packLowFirst(c);
        byte[] e = bytes(120, 127, 130, 1);
        HfTensorIndex in = fixture().add(M + ".weight_packed", StDType.U8, new long[]{2, k / 2}, packed)
                .add(M + ".weight_scale", StDType.U8, new long[]{2, 2}, e)
                .add(P + "_blocks", StDType.U8, new long[]{1, 2, 2, 16}, packed)
                .add(P + "_scales", StDType.U8, new long[]{1, 2, 2}, e)
                .add(P + "_bias", StDType.F32, new long[]{1, 2}, new byte[8]).index();
        QuantArgs ct = QuantArgs.builder(QuantFormat.MXFP4_CT).actScheme(ActQuantScheme.MXFP4).build();
        RowSource a = Fp4Scheme.rewriteModules(ct, List.of(M), plain(), in, ModelSourceOptions.defaults())
                .get(M + ".weight");
        HfTensorIndex gpt = Fp4Scheme.rewriteModules(QuantArgs.builder(QuantFormat.MXFP4_GPTOSS).build(),
                List.of(P), plain(), in, ModelSourceOptions.defaults());
        RowSource b = gpt.get(P);
        assertTrue(gpt.has(P + "_bias"));
        assertFalse(gpt.has(P + "_blocks"));
        assertEquals(GGMLType.MXFP4, a.format().ggmlType());
        assertEquals(ActQuantScheme.MXFP4, a.actQuant());
        assertArrayEquals(new long[]{1, 2, k}, b.shape());
        assertArrayEquals(Fixture.bytes(a), Fixture.bytes(b));
        float[] v = Fixture.values(b);
        for (int i = 0; i < c.length; i++) {
            assertEquals(FpBlockDecoders.e2m1ToFp32(c[i]) * FpBlockDecoders.e8m0ToFp32(e[i / 32] & 0xFF), v[i], 0f);
        }
    }

    @Test
    public void deepseekV4ExpertsAndE8m0NanRejection() throws IOException {
        String x = "layers.0.ffn.experts.3.w1";
        int[] c = codes(64, 7);
        HfTensorIndex in = fixture().add(x + ".weight", StDType.I8, new long[]{1, 32}, packLowFirst(c))
                .add(x + ".scale", StDType.F8_E8M0, new long[]{1, 2}, bytes(126, 0xFF)).index();
        HfTensorIndex out = Fp4Scheme.rewriteModules(QuantArgs.builder(QuantFormat.MXFP4_DSV4).build(), List.of(x),
                plain(), in, ModelSourceOptions.defaults());
        assertFalse(out.has(x + ".scale"));
        RowSource w = out.get(x + ".weight");
        assertEquals(GGMLType.MXFP4, w.format().ggmlType());
        IOException e = assertThrows(IOException.class, () -> Fixture.bytes(w));
        assertTrue(e.getMessage(), e.getMessage().contains("0xFF is NaN"));
    }

    /** Four 6-bit codes per three bytes, little-endian ({@code c0 | c1 << 6 | c2 << 12 | c3 << 18}). */
    static byte[] packFp6(int[] codes) {
        byte[] b = new byte[codes.length / 4 * 3];
        for (int q = 0; q < codes.length / 4; q++) {
            int word = codes[4 * q] | codes[4 * q + 1] << 6 | codes[4 * q + 2] << 12 | codes[4 * q + 3] << 18;
            b[3 * q] = (byte) word;
            b[3 * q + 1] = (byte) (word >>> 8);
            b[3 * q + 2] = (byte) (word >>> 16);
        }
        return b;
    }

    private HfConfig quark(String weightDtype, String extraLayer) {
        String spec = "{'dtype':'%s','qscheme':'per_group','group_size':32,'scale_format':'e8m0'}";
        return config("{'architectures':['A'],'quantization_config':{'quant_method':'quark',"
                + "'global_quant_config':{'weight':" + String.format(spec, weightDtype) + "}"
                + (extraLayer == null ? "" : ",'layer_quant_config':{'*o_proj':{'weight':"
                + String.format(spec, extraLayer) + "}}") + "}}");
    }

    @Test
    public void mxfp6RepacksExactlyToNativeBlocksWithinTheExponentLimits() throws IOException {
        for (boolean e2m3 : new boolean[]{true, false}) {
            int k = 64;
            int[] c = new int[2 * k];
            for (int i = 0; i < c.length; i++) {
                c[i] = (i * 11 + i / 5) & 0x3F;
            }
            int max = e2m3 ? 252 : 250;
            byte[] e = bytes(max, 127, 1, 0);
            HfConfig cfg = quark(e2m3 ? "fp6_e2m3" : "fp6_e3m2", null);
            QuantArgs args = HfQuantConfig.of(cfg).argsFor(M).orElseThrow(AssertionError::new);
            assertEquals(QuantFormat.MXFP6, args.format());
            byte[] packed = packFp6(c);
            HfTensorIndex in = fixture().add(M + ".weight", StDType.U8, new long[]{2, k / 4 * 3}, packed)
                    .add(M + ".weight_scale", StDType.U8, new long[]{2, 2}, e).index();
            RowSource w = Fp4Scheme.rewriteModules(args, List.of(M), cfg, in, ModelSourceOptions.defaults())
                    .get(M + ".weight");
            assertEquals(e2m3 ? GGMLType.MXFP6_E2M3 : GGMLType.MXFP6_E3M2, w.format().ggmlType());
            assertArrayEquals(new long[]{2, k}, w.shape());
            byte[] blocks = Fixture.bytes(w);
            assertEquals(4 * BlockLayout.MXFP6_BYTES, blocks.length);
            for (int b = 0; b < 4; b++) {
                assertEquals("block " + b + " header", e[b], blocks[b * BlockLayout.MXFP6_BYTES + BlockLayout.MXFP6_E]);
                for (int j = 0; j < 24; j++) {
                    assertEquals("block " + b + " code byte " + j, packed[b * 24 + j],
                            blocks[b * BlockLayout.MXFP6_BYTES + BlockLayout.MXFP6_QS + j]);
                }
            }
            float[] v = Fixture.values(w);
            for (int i = 0; i < c.length; i++) {
                float code = e2m3 ? FpBlockDecoders.e2m3ToFp32(c[i]) : FpBlockDecoders.e3m2ToFp32(c[i]);
                assertEquals("element " + i, Math.scalb(code, (e[i / 32] & 0xFF) - 127), v[i], 0f);
            }
            HfTensorIndex over = fixture().add(M + ".weight", StDType.U8, new long[]{1, k / 4 * 3},
                    packFp6(new int[k])).add(M + ".weight_scale", StDType.U8, new long[]{1, 2}, bytes(1, max + 1))
                    .index();
            RowSource o = Fp4Scheme.rewriteModules(args, List.of(M), cfg, over, ModelSourceOptions.defaults())
                    .get(M + ".weight");
            IOException ex = assertThrows(IOException.class, () -> Fixture.bytes(o));
            assertTrue(ex.getMessage(), ex.getMessage().contains("block 1") && ex.getMessage().contains("overflows"));
        }
    }

    @Test
    public void mixedFp6ElementDtypesAreNamed() throws IOException {
        HfConfig cfg = quark("fp6_e2m3", "fp6_e3m2");
        QuantArgs args = HfQuantConfig.of(cfg).argsFor(M).orElseThrow(AssertionError::new);
        HfTensorIndex in = fixture().add(M + ".weight", StDType.U8, new long[]{1, 24}, new byte[24])
                .add(M + ".weight_scale", StDType.U8, new long[]{1, 1}, bytes(127)).index();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> Fp4Scheme.rewriteModules(args,
                List.of(M), cfg, in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("fp6_e2m3, fp6_e3m2"));
    }
}
