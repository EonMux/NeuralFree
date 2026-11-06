package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.st.StRowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Fp8Scheme}: E4M3 codes of every class through each target (zero-copy F8_E4M3FN at any K and its fold,
 * per-channel and
 * block F8_E4M3_B128, E8M0 MXFP8 blocks), block grids that do not fit 128-element runs (fallback), DeepSeek-V4
 * {@code .scale} discovery, E8M0 NaN rejection, the MXFP8 repeat form, E5M2 and FNUZ weights, per-expert scales and
 * static activation scales. Every value is checked exactly against {@code code * scale}.
 */
public class Fp8SchemeTest {

    private static final String M = "model.layers.0.mlp.down_proj";
    /** E4M3 codes of every class: zeros, subnormals, normals, the largest finite and negatives. */
    private static final int[] E4M3_CLASSES = {0x00, 0x80, 0x01, 0x07, 0x08, 0x38, 0x3F, 0x77, 0x7E, 0x81, 0xB8,
        0xFE, 0x40, 0x10, 0x5A, 0xC9};

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

    private HfConfig plainConfig() {
        return Fixture.config(tmp.getRoot().toPath(), "{'architectures':['A']}");
    }

    private static byte[] e4m3Codes(int n) {
        byte[] b = new byte[n];
        for (int i = 0; i < n; i++) {
            b[i] = (byte) E4M3_CLASSES[(i * 7 + i / 16) % E4M3_CLASSES.length];
        }
        return b;
    }

    private static RowSource rewritten(HfTensorIndex out) {
        return out.get(M + ".weight");
    }

    @Test
    public void perTensorE4m3IsZeroCopyWithItsScaleAndFoldsToB128() throws IOException {
        byte[] codes = e4m3Codes(2 * 128);
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{2, 128}, codes)
                .f32(M + ".weight_scale", new long[]{}, 0.375f).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).actScheme(ActQuantScheme.FP8_TOKEN).build();
        HfTensorIndex out = Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                ModelSourceOptions.defaults());
        RowSource w = rewritten(out);
        assertEquals(GGMLType.F8_E4M3FN, w.format().ggmlType());
        assertArrayEquals(codes, Fixture.bytes(w));
        assertArrayEquals(new float[]{0.375f}, w.outputScale(), 0f);
        assertEquals(ActQuantScheme.FP8_TOKEN, w.actQuant());
        assertFalse(out.has(M + ".weight_scale"));
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]), v[i], 0f);
        }
        RowSource folded = w.foldOutputScale();
        assertEquals(GGMLType.F8_E4M3_B128, folded.format().ggmlType());
        assertNull(folded.outputScale());
        assertEquals(ActQuantScheme.FP8_TOKEN, folded.actQuant());
        float[] f = Fixture.values(folded);
        for (int i = 0; i < codes.length; i++) {
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * 0.375f, f[i], 0f);
        }
    }

    @Test
    public void perTensorE4m3IsZeroCopyAtAnyKIncludingOddRowLengths() throws IOException {
        // K = 60: neither a whole 32- nor 128-element run, yet the per-element ggml type stores it zero-copy
        for (int k : new int[]{32, 60, 96}) {
            int n = 3;
            byte[] codes = e4m3Codes(n * k);
            HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{n, k}, codes)
                    .f32(M + ".weight_scale", new long[]{}, 2.0f).index();
            QuantArgs args = QuantArgs.builder(QuantFormat.FP8).build();
            RowSource w = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                    ModelSourceOptions.defaults()));
            assertEquals("K = " + k, GGMLType.F8_E4M3FN, w.format().ggmlType());
            assertArrayEquals("K = " + k, codes, Fixture.bytes(w));
            assertArrayEquals("K = " + k, new float[]{2.0f}, w.outputScale(), 0f);
        }
    }

    @Test
    public void fusedExpertsWithUnderscoreScaleInvGiveB128() throws IOException {
        String p = "model.layers.0.mlp.experts.gate_up_proj";
        byte[] codes = e4m3Codes(2 * 128 * 256);
        float[] s = {0.5f, 2.0f, 0.25f, 4.0f};
        HfTensorIndex in = fixture().add(p, StDType.F8_E4M3, new long[]{2, 128, 256}, codes)
                .f32(p + "_scale_inv", new long[]{2, 1, 2}, s).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).block(128, 128).build();
        HfTensorIndex out = Fp8Scheme.rewriteModules(args, List.of(p), plainConfig(), in,
                ModelSourceOptions.defaults());
        assertFalse(out.has(p + "_scale_inv"));
        RowSource w = out.get(p);
        assertEquals(GGMLType.F8_E4M3_B128, w.format().ggmlType());
        assertArrayEquals(new long[]{2, 128, 256}, w.shape());
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            int expert = i / (128 * 256);
            int colBlock = (i % 256) / 128;
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * s[expert * 2 + colBlock], v[i], 0f);
        }
    }

    @Test
    public void perChannelBf16ScalesGiveB128() throws IOException {
        byte[] codes = e4m3Codes(3 * 256);
        float[] s = {0.5f, 0.0078125f, 3.0f};
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{3, 256}, codes)
                .bf16(M + ".weight_scale", new long[]{3, 1}, s).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).group(QuantArgs.CHANNEL).build();
        RowSource w = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                ModelSourceOptions.defaults()));
        assertEquals(GGMLType.F8_E4M3_B128, w.format().ggmlType());
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * s[i / 256], v[i], 0f);
        }
    }

    @Test
    public void e8m0BlockScalesOverRowsThatAreNotAMultipleOf128() throws IOException {
        int n = 130;
        int k = 256;
        byte[] codes = e4m3Codes(n * k);
        byte[] e = {(byte) 120, (byte) 127, (byte) 130, (byte) 0};
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{n, k}, codes)
                .add(M + ".weight_scale_inv", StDType.F8_E8M0, new long[]{2, 2}, e).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).block(128, 128).actScheme(ActQuantScheme.FP8_GROUP128)
                .build();
        RowSource w = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                ModelSourceOptions.defaults()));
        assertEquals(GGMLType.F8_E4M3_B128, w.format().ggmlType());
        assertTrue(w.describe(), w.describe().startsWith(M + " FP8 ") && w.describe().endsWith(" -> F8_E4M3_B128"));
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            int row = i / k;
            int col = i % k;
            float scale = FpBlockDecoders.e8m0ToFp32(e[(row / 128) * 2 + col / 128] & 0xFF);
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * scale, v[i], 0f);
        }
    }

    @Test
    public void e8m0NanScaleIsRejected() throws IOException {
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{4, 128}, e4m3Codes(512))
                .add(M + ".weight_scale_inv", StDType.F8_E8M0, new long[]{1, 1}, new byte[]{(byte) 0xFF}).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).block(128, 128).build();
        IOException e = assertThrows(IOException.class, () -> Fp8Scheme.rewriteModules(args, List.of(M),
                plainConfig(), in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("NaN"));
    }

    @Test
    public void blockGridThatDoesNotFit128RunsFallsBackToTheScaleDtype() throws IOException {
        int n = 3;
        int k = 160;
        byte[] codes = e4m3Codes(n * k);
        float[] s = {0.25f, 2.0f};
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{n, k}, codes)
                .f32(M + ".weight_scale_inv", new long[]{1, 2}, s).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).block(128, 128).build();
        RowSource w = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                ModelSourceOptions.defaults()));
        assertEquals(StDType.F32, w.format().stDType());
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * s[(i % k) / 128], v[i], 0f);
        }
    }

    @Test
    public void e8m0BlocksOf32ColumnsGiveMxfp8() throws IOException {
        int k = 64;
        byte[] codes = e4m3Codes(2 * k);
        byte[] e = {(byte) 125, (byte) 128, (byte) 1, (byte) 254};
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{2, k}, codes)
                .add(M + ".weight_scale_inv", StDType.U8, new long[]{2, 2}, e).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).block(1, 32).scaleFormat(QuantArgs.ScaleFormat.E8M0)
                .build();
        RowSource w = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                ModelSourceOptions.defaults()));
        assertEquals(GGMLType.MXFP8, w.format().ggmlType());
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            float scale = FpBlockDecoders.e8m0ToFp32(e[(i / k) * 2 + (i % k) / 32] & 0xFF);
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * scale, v[i], 0f);
        }
    }

    @Test
    public void deepseekV4ScaleIsFoundOnlyForDeepseekV4() throws IOException {
        String dsv4 = "layers.0.attn.wq_a";
        int n = 200;
        byte[] codes = e4m3Codes(n * 128);
        HfTensorIndex in = fixture().add(dsv4 + ".weight", StDType.F8_E4M3, new long[]{n, 128}, codes)
                .add(dsv4 + ".scale", StDType.F8_E8M0, new long[]{2, 1}, new byte[]{(byte) 129, (byte) 126})
                .index();
        Path dir = tmp.getRoot().toPath();
        HfConfig cfg = Fixture.config(dir, "{'architectures':['A'],'model_type':'deepseek_v4',"
                + "'quantization_config':{'quant_method':'fp8','activation_scheme':'dynamic',"
                + "'weight_block_size':[128,128]}}");
        QuantArgs args = HfQuantConfig.of(cfg).argsFor(dsv4).orElseThrow(AssertionError::new);
        HfTensorIndex out = Fp8Scheme.rewriteModules(args, List.of(dsv4), cfg, in, ModelSourceOptions.defaults());
        RowSource w = out.get(dsv4 + ".weight");
        assertEquals(GGMLType.F8_E4M3_B128, w.format().ggmlType());
        assertFalse(out.has(dsv4 + ".scale"));
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * (i / 128 < 128 ? 4.0f : 0.5f), v[i], 0f);
        }
        QuantArgs plain = QuantArgs.builder(QuantFormat.FP8).block(128, 128).build();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> Fp8Scheme.rewriteModules(plain,
                List.of(dsv4), plainConfig(), in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("no weight scale"));
    }

    @Test
    public void nanCodesAreRejectedWhenRepacking() throws IOException {
        byte[] codes = e4m3Codes(2 * 128);
        codes[130] = (byte) 0x7F;
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{2, 128}, codes)
                .f32(M + ".weight_scale", new long[]{2}, 1f, 1f).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).group(QuantArgs.CHANNEL).build();
        RowSource w = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                ModelSourceOptions.defaults()));
        IOException e = assertThrows(IOException.class, () -> Fixture.bytes(w));
        assertTrue(e.getMessage(), e.getMessage().contains("row 1 column 2"));
    }

    @Test
    public void mismatchedScaleShapesAreNamed() throws IOException {
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{4, 128}, e4m3Codes(512))
                .f32(M + ".weight_scale_inv", new long[]{2, 1}, 1f, 1f).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).block(128, 128).build();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> Fp8Scheme.rewriteModules(args,
                List.of(M), plainConfig(), in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("[1, 1] blocks of 128x128"));
    }

    @Test
    public void mxfp8RepeatFormSharesScaleRows() throws IOException {
        int k = 64;
        byte[] codes = e4m3Codes(4 * k);
        byte[] e = {(byte) 126, (byte) 127, (byte) 3, (byte) 200};
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{4, k}, codes)
                .add(M + ".weight_scale", StDType.U8, new long[]{2, 2}, e).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.MXFP8).actScheme(ActQuantScheme.MXFP8).build();
        RowSource w = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                ModelSourceOptions.defaults()));
        assertEquals(GGMLType.MXFP8, w.format().ggmlType());
        assertEquals(ActQuantScheme.MXFP8, w.actQuant());
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            float scale = FpBlockDecoders.e8m0ToFp32(e[(i / k / 2) * 2 + (i % k) / 32] & 0xFF);
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * scale, v[i], 0f);
        }
        byte[] bad = {(byte) 126, (byte) 0xFF};
        HfTensorIndex nan = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{1, k}, e4m3Codes(k))
                .add(M + ".weight_scale", StDType.U8, new long[]{1, 2}, bad).index();
        RowSource w2 = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), nan,
                ModelSourceOptions.defaults()));
        assertThrows(IOException.class, () -> Fixture.bytes(w2));
    }

    @Test
    public void staticInputScalePerExpertAndPerExpertWeightScale() throws IOException {
        int e = 2;
        byte[] codes = e4m3Codes(e * 2 * 64);
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{e, 2, 64}, codes)
                .f32(M + ".weight_scale", new long[]{e}, 0.5f, 0.25f)
                .f32(M + ".input_scale", new long[]{e}, 3f, 5f).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).actScheme(ActQuantScheme.FP8_STATIC).build();
        HfTensorIndex out = Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                ModelSourceOptions.defaults());
        RowSource w = rewritten(out);
        assertEquals(GGMLType.F8_E4M3FN, w.format().ggmlType());
        assertArrayEquals(new float[]{0.5f, 0.25f}, w.outputScale(), 0f);
        assertArrayEquals(new float[]{3f, 5f}, w.inputScale(), 0f);
        assertEquals(ActQuantScheme.FP8_STATIC, w.actQuant());
        assertFalse(out.has(M + ".input_scale"));
        RowSource folded = w.foldOutputScale();
        assertEquals(StDType.F32, folded.format().stDType());
        assertArrayEquals(new float[]{3f, 5f}, folded.inputScale(), 0f);
        float[] f = Fixture.values(folded);
        for (int i = 0; i < codes.length; i++) {
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * (i < 128 ? 0.5f : 0.25f), f[i], 0f);
        }
    }

    @Test
    public void fusedExpertParameterKeepsItsBareName() throws IOException {
        String p = "model.layers.0.mlp.experts.down_proj";
        byte[] codes = e4m3Codes(2 * 3 * 128);
        HfTensorIndex in = fixture().add(p, StDType.F8_E4M3, new long[]{2, 3, 128}, codes)
                .f32(p + "_weight_scale", new long[]{2, 3}, 0.5f, 1f, 2f, 4f, 8f, 16f).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).group(QuantArgs.CHANNEL).build();
        HfTensorIndex out = Fp8Scheme.rewriteModules(args, List.of(p), plainConfig(), in,
                ModelSourceOptions.defaults());
        assertEquals(List.of(p), out.names());
        RowSource w = out.get(p);
        assertEquals(GGMLType.F8_E4M3_B128, w.format().ggmlType());
        assertArrayEquals(new long[]{2, 3, 128}, w.shape());
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            assertEquals(FpBlockDecoders.e4m3ToFp32(codes[i]) * (1 << (i / 128)) * 0.5f, v[i], 0f);
        }
    }

    @Test
    public void nonFiniteScalesAreNamed() throws IOException {
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{2, 128}, e4m3Codes(256))
                .f32(M + ".weight_scale", new long[]{2}, 1f, Float.POSITIVE_INFINITY).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).group(QuantArgs.CHANNEL).build();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> Fp8Scheme.rewriteModules(args,
                List.of(M), plainConfig(), in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains(M + ".weight_scale element 1"));
    }

    @Test
    public void staticActivationsWithoutInputScaleAreNamed() throws IOException {
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E4M3, new long[]{2, 64}, e4m3Codes(128))
                .f32(M + ".weight_scale", new long[]{1}, 1f).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).actScheme(ActQuantScheme.FP8_STATIC).build();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> Fp8Scheme.rewriteModules(args,
                List.of(M), plainConfig(), in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("input_scale"));
    }

    @Test
    public void e5m2PerTensorIsF16AndFnuzIsF32() throws IOException {
        byte[] codes = new byte[64];
        for (int i = 0; i < codes.length; i++) {
            codes[i] = (byte) (i * 3 % 0x7B | (i % 2) << 7);
        }
        HfTensorIndex in = fixture().add(M + ".weight", StDType.F8_E5M2, new long[]{1, 64}, codes)
                .f32(M + ".weight_scale", new long[]{}, 2f).index();
        QuantArgs args = QuantArgs.builder(QuantFormat.FP8).build();
        RowSource w = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in,
                ModelSourceOptions.defaults()));
        assertEquals(StDType.F16, w.format().stDType());
        assertArrayEquals(new float[]{2f}, w.outputScale(), 0f);
        float[] v = Fixture.values(w);
        for (int i = 0; i < codes.length; i++) {
            assertEquals(FpBlockDecoders.e5m2ToFp32(codes[i]), v[i], 0f);
        }
        byte[] fnuz = {0x00, 0x01, 0x08, 0x40, 0x7F, (byte) 0x81, (byte) 0xC8, 0x3C};
        HfTensorIndex in2 = fixture().add(M + ".weight", StDType.F8_E4M3FNUZ, new long[]{1, 8}, fnuz)
                .f32(M + ".weight_scale", new long[]{}, 0.5f).index();
        RowSource f = rewritten(Fp8Scheme.rewriteModules(args, List.of(M), plainConfig(), in2,
                ModelSourceOptions.defaults()));
        assertEquals(StDType.F32, f.format().stDType());
        float[] expected = {0f, 0.0009765625f * 0.5f, 0.0078125f * 0.5f, 1f * 0.5f, 240f * 0.5f,
            -0.0009765625f * 0.5f, -2f * 0.5f, 0.75f * 0.5f};
        assertArrayEquals(expected, Fixture.values(f), 0f);
    }

    @Test
    public void unquantizedFloatWeightsLoadAsStored() throws IOException {
        HfTensorIndex in = fixture().bf16(M + ".weight", new long[]{1, 4}, 1f, 2f, 3f, 4f).index();
        HfTensorIndex out = Fp8Scheme.rewriteModules(QuantArgs.builder(QuantFormat.FP8).build(), List.of(M),
                plainConfig(), in, ModelSourceOptions.defaults());
        assertSame(in.get(M + ".weight"), out.get(M + ".weight"));
    }

    // ------------------------------------------------------------------ fixture shared by the scheme tests

    /** A one-file checkpoint written with {@link SafetensorsWriter} and opened as an index. */
    static final class Fixture implements Closeable {
        private final Path dir;
        private final SafetensorsWriter.Builder builder = SafetensorsWriter.builder();
        private SafetensorsFile file;
        private int files;

        Fixture(Path dir) {
            this.dir = dir;
        }

        Fixture add(String name, StDType t, long[] shape, byte[] data) {
            builder.add(name, t, shape, data);
            return this;
        }

        Fixture f32(String name, long[] shape, float... v) {
            ByteBuffer b = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN);
            for (float x : v) {
                b.putFloat(x);
            }
            return add(name, StDType.F32, shape, b.array());
        }

        Fixture bf16(String name, long[] shape, float... v) {
            ByteBuffer b = ByteBuffer.allocate(2 * v.length).order(ByteOrder.LITTLE_ENDIAN);
            for (float x : v) {
                b.putShort((short) HalfFloats.floatToBFloat16(x));
            }
            return add(name, StDType.BF16, shape, b.array());
        }

        /** Writes the tensors added so far to a new file and indexes them; the previous file is closed. */
        HfTensorIndex index() throws IOException {
            close();
            Path p = dir.resolve("model-" + files++ + ".safetensors");
            builder.build().write(p);
            file = SafetensorsFile.open(p);
            Map<String, RowSource> m = new LinkedHashMap<>();
            for (StTensor t : file.tensors()) {
                m.put(t.name(), StRowSource.of(file, t));
            }
            return HfTensorIndex.of(m);
        }

        @Override
        public void close() throws IOException {
            if (file != null) {
                file.close();
                file = null;
            }
        }

        static HfConfig config(Path dir, String singleQuotedJson) {
            return HfConfigLoader.fromJson(dir, singleQuotedJson.replace('\'', '"'));
        }

        static byte[] bytes(RowSource s) throws IOException {
            ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(s.rows() * s.rowBytes()));
            s.readRows(0, Math.toIntExact(s.rows()), b);
            assertEquals(b.capacity(), b.position());
            return b.array();
        }

        static float[] values(RowSource s) throws IOException {
            byte[] raw = bytes(s);
            long n = Fp4Packing.numel(s.shape());
            if (s.format().isBlock()) {
                return CpuBlockDecoders.dequantize(raw, s.format().ggmlType(), n);
            }
            ByteBuffer le = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            float[] out = new float[Math.toIntExact(n)];
            for (int i = 0; i < out.length; i++) {
                switch (s.format().stDType()) {
                    case F32: out[i] = le.getFloat(4 * i); break;
                    case F16: out[i] = FpBlockDecoders.fp16ToFp32(le.getShort(2 * i) & 0xFFFF); break;
                    case BF16: out[i] = FpBlockDecoders.bf16ToFp32(le.getShort(2 * i) & 0xFFFF); break;
                    case F8_E4M3: out[i] = FpBlockDecoders.e4m3ToFp32(le.get(i)); break;
                    default: throw new AssertionError("not a float format: " + s.format());
                }
            }
            return out;
        }
    }
}
