package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.quant.IntSchemeFixtures.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link IntGroupScheme}: GPTQ v1/v2 zero offsets, act-order permutations (GPTQ {@code g_idx} and compressed-tensors
 * {@code weight_g_idx}) with the inverse identity and the folded form, GPTQ 3-bit bit streams, the AWQ nibble order,
 * Quark signed nibbles, compressed-tensors zero points packed along N, stacked experts, the underscore suffix form,
 * fallbacks, activation siblings, transform locations, and the negative cases. Every produced value is decoded
 * through {@link CpuBlockDecoders} and compared exactly with {@code (q - z) * s}.
 */
public class IntGroupSchemeTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private IntFixture fx;

    @After
    public void close() throws IOException {
        if (fx != null) {
            fx.close();
        }
    }

    private IntFixture fixture() throws IOException {
        close();
        fx = new IntFixture(tmp.getRoot().toPath());
        return fx;
    }

    private HfConfig config(String json) {
        return IntFixture.config(tmp.getRoot().toPath(), json);
    }

    private static ModelSourceOptions exact() {
        return ModelSourceOptions.builder().exactQuant(true).build();
    }

    private static QuantArgs gptqArgs(int bits, int group, int version, boolean sym) {
        return QuantArgs.builder(QuantFormat.INT_GPTQ).bits(bits).group(group).gptqVersion(version).symmetric(sym)
                .build();
    }

    private static QuantArgs ctArgs(int bits, int group, boolean sym) {
        return QuantArgs.builder(QuantFormat.INT_PACKED_CT).bits(bits).group(group).symmetric(sym).build();
    }

    private HfTensorIndex rewrite(QuantArgs args, HfTensorIndex in, ModelSourceOptions o, String... modules)
            throws IOException {
        return IntGroupScheme.rewriteModules(args, List.of(modules), config(PLAIN), in, o);
    }

    // ------------------------------------------------------------------ GPTQ

    @Test
    public void gptqV1SymmetricBecomesQ4_0() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 1, null, false);
        IntFixture f = fixture();
        addGptq(f, M, r, 1, false);
        f.f16(M + ".bias", new long[]{4}, 1f, 2f, 3f, 4f);
        HfTensorIndex out = rewrite(gptqArgs(4, 32, 1, true), f.index(), ModelSourceOptions.defaults(), M);
        RowSource s = out.get(M + ".weight");
        assertEquals(GGMLType.Q4_0, s.format().ggmlType());
        assertEquals(M + " INT_GPTQ b4 g32 -> Q4_0", s.describe());
        assertArrayEquals(new long[]{4, 64}, s.shape());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
        assertTrue(out.has(M + ".bias"));
        assertFalse(out.has(M + ".qweight"));
        assertFalse(out.has(M + ".qzeros"));
        assertFalse(out.has(M + ".scales"));
        assertFalse(s.rowLocal());
        assertNull(s.inputPerm());
    }

    @Test
    public void gptqV2AsymmetricZerosBecomeQ4_1OrExactQ8_0() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 2, null, true);
        IntFixture f = fixture();
        addGptq(f, M, r, 0, false);
        HfTensorIndex in = f.index();
        RowSource s = rewrite(gptqArgs(4, 32, 2, false), in, ModelSourceOptions.defaults(), M).get(M + ".weight");
        assertEquals(GGMLType.Q4_1, s.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
        RowSource e = rewrite(gptqArgs(4, 32, 2, false), in, exact(), M).get(M + ".weight");
        assertEquals(GGMLType.Q8_0, e.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(e), 0f);
    }

    @Test
    public void gptqV1ZerosAreOffsetByOne() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 3, null, true);
        for (float[] row : r.zero) {
            for (int g = 0; g < row.length; g++) {
                row[g] = Math.max(1, row[g]);
            }
        }
        IntFixture f = fixture();
        addGptq(f, M, r, 1, false);
        RowSource s = rewrite(gptqArgs(4, 32, 1, false), f.index(), exact(), M).get(M + ".weight");
        assertEquals(GGMLType.Q8_0, s.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
    }

    @Test
    public void gptqActOrderPermutesTheInputAndFolds() throws IOException {
        int[] gidx = alternating(64);
        Ref r = new Ref(4, 64, 4, 32, 4, gidx, false);
        IntFixture f = fixture();
        addGptq(f, M, r, 1, true);
        RowSource s = rewrite(gptqArgs(4, 32, 1, true), f.index(), ModelSourceOptions.defaults(), M)
                .get(M + ".weight");
        assertEquals(GGMLType.Q4_0, s.format().ggmlType());
        int[] perm = s.inputPerm();
        assertArrayEquals(IntPacking.stableArgsort(gidx, 2), perm);
        assertPermuted(r, perm, IntFixture.values(s));
        RowSource folded = s.foldInputPerm();
        assertNull(folded.inputPerm());
        assertEquals(GGMLType.F16, folded.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(folded), 0f);
        assertThrows(UnsupportedOperationException.class, folded::foldInputPerm);
    }

    @Test
    public void gptqThreeBitBitStream() throws IOException {
        Ref r = new Ref(4, 64, 3, 32, 5, null, false);
        IntFixture f = fixture();
        addGptq(f, M, r, 1, false);
        int[] qw = gptqQweight(r, true);
        assertEquals(6 * 4, qw.length);
        f.i32(M + ".qweight", new long[]{6, 4}, qw);
        RowSource s = rewrite(gptqArgs(3, 32, 1, true), f.index(), ModelSourceOptions.defaults(), M)
                .get(M + ".weight");
        assertEquals(GGMLType.Q8_0, s.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
    }

    @Test
    public void groupsThatDoNotSplitInto32TakeTheFloatFallback() throws IOException {
        Ref r = new Ref(4, 48, 4, 48, 6, null, false);
        IntFixture f = fixture();
        addGptq(f, M, r, 1, false);
        RowSource s = rewrite(gptqArgs(4, QuantArgs.CHANNEL, 1, true), f.index(), ModelSourceOptions.defaults(), M)
                .get(M + ".weight");
        assertEquals(GGMLType.F16, s.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
    }

    @Test
    public void exactQuantizationWithScalesOutsideFp16TakesF32() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 7, null, false);
        for (float[] row : r.scale) {
            for (int g = 0; g < row.length; g++) {
                row[g] = FpBlockDecoders.bf16ToFp32((short) HalfFloats.floatToBFloat16(1e-30f * (g + 1)));
            }
        }
        IntFixture f = fixture();
        addGptq(f, M, r, 1, false);
        f.bf16(M + ".scales", new long[]{2, 4}, r.groupMajorScales());
        HfTensorIndex in = f.index();
        RowSource s = rewrite(gptqArgs(4, 32, 1, true), in, exact(), M).get(M + ".weight");
        assertEquals(GGMLType.F32, s.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
        RowSource relaxed = rewrite(gptqArgs(4, 32, 1, true), in, ModelSourceOptions.defaults(), M)
                .get(M + ".weight");
        assertEquals(GGMLType.Q4_0, relaxed.format().ggmlType());
    }

    @Test
    public void gIdxOutOfRangeIsRejected() throws IOException {
        int[] gidx = new int[64];
        for (int k = 0; k < 64; k++) {
            gidx[k] = k / 32;
        }
        Ref r = new Ref(4, 64, 4, 32, 8, null, false);
        IntFixture f = fixture();
        addGptq(f, M, r, 1, false);
        gidx[5] = 2;
        f.i32(M + ".g_idx", new long[]{64}, gidx);
        IOException e = assertThrows(IOException.class,
                () -> rewrite(gptqArgs(4, 32, 1, true), f.index(), ModelSourceOptions.defaults(), M));
        assertTrue(e.getMessage(), e.getMessage().contains(M + ".g_idx[5] = 2 is outside [0, 2)"));
    }

    @Test
    public void mismatchedScaleShapesAreRejected() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 9, null, false);
        IntFixture f = fixture();
        addGptq(f, M, r, 1, false);
        f.f16(M + ".scales", new long[]{3, 4}, new float[12]);
        IOException e = assertThrows(IOException.class,
                () -> rewrite(gptqArgs(4, 32, 1, true), f.index(), ModelSourceOptions.defaults(), M));
        assertTrue(e.getMessage(), e.getMessage().contains(M + ".scales has shape [3, 4]"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid Hugging Face checkpoint '"));
    }

    // ------------------------------------------------------------------ AWQ and Quark

    @Test
    public void awqOrderWithZeroPoints() throws IOException {
        Ref r = new Ref(8, 64, 4, 32, 10, null, true);
        IntFixture f = fixture();
        addAwq(f, M, r, true);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT_AWQ).bits(4).group(32).symmetric(false).build();
        RowSource s = IntGroupScheme.rewriteModules(a, List.of(M), config(PLAIN), f.index(),
                ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.Q4_1, s.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
    }

    @Test
    public void awqWithoutZeroPointsIsSymmetric() throws IOException {
        Ref r = new Ref(8, 64, 4, 32, 11, null, false);
        IntFixture f = fixture();
        addAwq(f, M, r, false);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT_AWQ).bits(4).group(32).symmetric(true).build();
        RowSource s = IntGroupScheme.rewriteModules(a, List.of(M), config(PLAIN), f.index(),
                ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.Q4_0, s.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
    }

    @Test
    public void quarkSymmetricSignedNibblesInSequentialOrder() throws IOException {
        Ref r = new Ref(8, 64, 4, 32, 12, null, false);
        int[] packed = new int[64];
        for (int k = 0; k < 64; k++) {
            for (int c = 0; c < 8; c++) {
                packed[k] |= (r.codes[c][k] ^ 0x8) << (4 * c);
            }
        }
        IntFixture f = fixture();
        f.i32(M + ".weight", new long[]{64, 1}, packed);
        f.f16(M + ".weight_scale", new long[]{2, 8}, r.groupMajorScales());
        QuantArgs a = QuantArgs.builder(QuantFormat.INT_QUARK).bits(4).group(32).symmetric(true).build();
        HfTensorIndex out = IntGroupScheme.rewriteModules(a, List.of(M), config(PLAIN), f.index(),
                ModelSourceOptions.defaults());
        RowSource s = out.get(M + ".weight");
        assertEquals(GGMLType.Q4_0, s.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
        assertFalse(out.has(M + ".weight_scale"));
    }

    // ------------------------------------------------------------------ compressed-tensors

    @Test
    public void ctAsymmetricZeroPointsPackedAlongN() throws IOException {
        Ref r = new Ref(8, 64, 4, 32, 13, null, true);
        IntFixture f = fixture();
        addCt(f, M, r, true);
        HfTensorIndex in = f.index();
        RowSource s = rewrite(ctArgs(4, 32, false), in, ModelSourceOptions.defaults(), M).get(M + ".weight");
        assertEquals(GGMLType.Q4_1, s.format().ggmlType());
        assertTrue(s.rowLocal());
        assertArrayEquals(r.values(), IntFixture.values(s), 0f);
        RowSource e = rewrite(ctArgs(4, 32, false), in, exact(), M).get(M + ".weight");
        assertEquals(GGMLType.Q8_0, e.format().ggmlType());
        assertArrayEquals(r.values(), IntFixture.values(e), 0f);
        assertArrayEquals(slice(r.values(), 3 * 64, 2 * 64), rows(e, 3, 2), 0f);
    }

    @Test
    public void ctWeightGidxPermutesTheInput() throws IOException {
        int[] gidx = alternating(64);
        Ref r = new Ref(4, 64, 4, 32, 14, gidx, false);
        IntFixture f = fixture();
        addCt(f, M, r, false);
        f.i32(M + ".weight_g_idx", new long[]{64}, gidx);
        RowSource s = rewrite(ctArgs(4, 32, true), f.index(), ModelSourceOptions.defaults(), M).get(M + ".weight");
        assertEquals(GGMLType.Q4_0, s.format().ggmlType());
        assertArrayEquals(IntPacking.stableArgsort(gidx, 2), s.inputPerm());
        assertPermuted(r, s.inputPerm(), IntFixture.values(s));
        assertArrayEquals(r.values(), IntFixture.values(s.foldInputPerm()), 0f);
    }

    @Test
    public void ctWeightShapeOverCapacityIsRejected() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 15, null, false);
        IntFixture f = fixture();
        addCt(f, M, r, false);
        f.i64(M + ".weight_shape", new long[]{2}, 4, 1000);
        IOException e = assertThrows(IOException.class,
                () -> rewrite(ctArgs(4, 32, true), f.index(), ModelSourceOptions.defaults(), M));
        assertTrue(e.getMessage(), e.getMessage().contains("is outside the capacity [4, 64]"));
    }

    @Test
    public void scalesOverflowingFp16AreRejectedOnRead() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 22, null, false);
        IntFixture f = fixture();
        addCt(f, M, r, false);
        float[] s = r.rowMajorScales();
        s[5] = 1e5f;
        f.bf16(M + ".weight_scale", new long[]{4, 2}, s);
        RowSource src = rewrite(ctArgs(4, 32, true), f.index(), ModelSourceOptions.defaults(), M).get(M + ".weight");
        assertEquals(GGMLType.Q4_0, src.format().ggmlType());
        IOException e = assertThrows(IOException.class, () -> IntFixture.values(src));
        assertTrue(e.getMessage(), e.getMessage().contains("row 2 column 32: scale"));
        assertTrue(e.getMessage(), e.getMessage().contains("overflows the fp16 range of Q4_0"));
    }

    @Test
    public void stackedExpertsUnderTheUnderscoreSuffixForm() throws IOException {
        String u = "model.layers.0.mlp.experts.gate_up_proj";
        Ref r0 = new Ref(4, 64, 4, 32, 16, null, false);
        Ref r1 = new Ref(4, 64, 4, 32, 17, null, false);
        IntFixture f = fixture();
        int[] p0 = ctPacked(r0);
        int[] p1 = ctPacked(r1);
        f.i32(u + "_weight_packed", new long[]{2, 4, 8}, concat(p0, p1));
        f.f16(u + "_weight_scale", new long[]{2, 4, 2}, concat(r0.rowMajorScales(), r1.rowMajorScales()));
        f.i64(u + "_weight_shape", new long[]{2, 2}, 4, 64, 4, 64);
        HfTensorIndex out = rewrite(ctArgs(4, 32, true), f.index(), ModelSourceOptions.defaults(), u);
        assertFalse(out.has(u + "_weight_packed"));
        assertFalse(out.has(u + "_weight_scale"));
        assertFalse(out.has(u + "_weight_shape"));
        RowSource s = out.get(u);
        assertArrayEquals(new long[]{2, 4, 64}, s.shape());
        assertEquals(8, s.rows());
        assertEquals(GGMLType.Q4_0, s.format().ggmlType());
        assertArrayEquals(concat(r0.values(), r1.values()), IntFixture.values(s), 0f);
        assertArrayEquals(slice(concat(r0.values(), r1.values()), 3 * 64, 3 * 64), rows(s, 3, 3), 0f);
    }

    @Test
    public void staticActivationScalesAreAttachedAndConsumed() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 18, null, false);
        IntFixture f = fixture();
        addCt(f, M, r, false);
        f.f32(M + ".input_scale", new long[]{1}, 0.5f);
        HfTensorIndex in = f.index();
        QuantArgs a = QuantArgs.builder(QuantFormat.INT_PACKED_CT).bits(4).group(32)
                .actScheme(ActQuantScheme.INT8_STATIC).build();
        HfTensorIndex out = rewrite(a, in, ModelSourceOptions.defaults(), M);
        RowSource s = out.get(M + ".weight");
        assertEquals(ActQuantScheme.INT8_STATIC, s.actQuant());
        assertArrayEquals(new float[]{0.5f}, s.inputScale(), 0f);
        assertFalse(out.has(M + ".input_scale"));

        QuantArgs unemulated = QuantArgs.builder(QuantFormat.INT_PACKED_CT).bits(4).group(32)
                .actUnemulated("int4 activations").build();
        HfTensorIndex weightOnly = rewrite(unemulated, in, ModelSourceOptions.defaults(), M);
        assertNull(weightOnly.get(M + ".weight").actQuant());
        assertFalse(weightOnly.has(M + ".input_scale"));
    }

    @Test
    public void staticActivationsWithoutInputScaleAreRejected() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 19, null, false);
        IntFixture f = fixture();
        addCt(f, M, r, false);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT_PACKED_CT).bits(4).group(32)
                .actScheme(ActQuantScheme.INT8_STATIC).build();
        IOException e = assertThrows(IOException.class, () -> rewrite(a, f.index(), ModelSourceOptions.defaults(), M));
        assertTrue(e.getMessage(), e.getMessage().contains(M + ".input_scale"));
    }

    @Test
    public void onlineOutputTransformsAreRejected() throws IOException {
        Ref r = new Ref(4, 64, 4, 32, 20, null, false);
        IntFixture f = fixture();
        addCt(f, M, r, false);
        HfConfig cfg = config(ctConfig("output"));
        QuantArgs a = HfQuantConfig.of(cfg).argsFor(M).get();
        HfTensorIndex in = f.index();
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> IntGroupScheme.rewriteModules(a, List.of(M), cfg, in, ModelSourceOptions.defaults()));
        assertEquals("compressed-tensors transform hadamard@output is not supported", e.getMessage());
    }

    @Test
    public void actOrderBehindAnInputTransformPermutesTheInput() throws IOException {
        int[] gidx = alternating(64);
        Ref r = new Ref(4, 64, 4, 32, 21, gidx, false);
        IntFixture f = fixture();
        addCt(f, M, r, false);
        f.i32(M + ".weight_g_idx", new long[]{64}, gidx);
        HfConfig cfg = config(ctConfig("input"));
        QuantArgs a = HfQuantConfig.of(cfg).argsFor(M).get();
        assertEquals(QuantFormat.INT_PACKED_CT, a.format());
        RowSource s = IntGroupScheme.rewriteModules(a, List.of(M), cfg, f.index(), ModelSourceOptions.defaults())
                .get(M + ".weight");
        assertEquals(GGMLType.Q4_0, s.format().ggmlType());
        assertArrayEquals(IntPacking.stableArgsort(gidx, 2), s.inputPerm());
        assertPermuted(r, s.inputPerm(), IntFixture.values(s));
        assertArrayEquals(r.values(), IntFixture.values(s.foldInputPerm()), 0f);
    }

    @Test
    public void formatsOfOtherSchemesAreRefused() {
        assertTrue(IntGroupScheme.handles(QuantFormat.INT_GPTQ));
        assertFalse(IntGroupScheme.handles(QuantFormat.INT8_CHANNEL));
        QuantArgs a = QuantArgs.builder(QuantFormat.INT8_CHANNEL).build();
        assertThrows(IllegalArgumentException.class, () -> IntGroupScheme.rewriteModules(a, List.of(M),
                config(PLAIN), HfTensorIndex.of(Map.of()), ModelSourceOptions.defaults()));
    }

    private static String ctConfig(String location) {
        return "{'architectures':['A'],'quantization_config':{'quant_method':'compressed-tensors',"
                + "'format':'pack-quantized','config_groups':{'g0':{'targets':['Linear'],'weights':{'num_bits':4,"
                + "'type':'int','strategy':'group','group_size':32,'symmetric':true,'actorder':'group'}}},"
                + "'transform_config':{'config_groups':{'v':{'type':'hadamard','apply':[{'targets':['Linear'],"
                + "'location':'" + location + "'}]}}}}}";
    }
}
