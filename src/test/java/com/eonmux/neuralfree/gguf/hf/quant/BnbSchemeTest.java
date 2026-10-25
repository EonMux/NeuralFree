package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.Fp8SchemeTest.Fixture;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link BnbScheme}: the NF4/FP4 repack (high-nibble-first flattened codes, absmax replicated per 64-run), the
 * double-quant fold, runs that cross rows or blocks (fallback), the {@code quant_map} tolerance, one negative test per
 * {@code quant_state} validation, the uint8 storage rule, and the LLM.int8 targets.
 */
public class BnbSchemeTest {

    private static final String M = "model.layers.0.self_attn.o_proj";
    private static final ModelSourceOptions EXACT = ModelSourceOptions.builder().exactQuant(true).build();

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

    private HfConfig plain() {
        return Fixture.config(tmp.getRoot().toPath(), "{'architectures':['A']}");
    }

    private static int[] codes(int n) {
        int[] c = new int[n];
        for (int i = 0; i < n; i++) {
            c[i] = (i * 7 + i / 9) & 0x0F;
        }
        return c;
    }

    /** The flattened codes, two per byte with the even element in the high nibble, as {@code [ceil(n/2), 1]}. */
    private static byte[] packHighFirst(int[] codes) {
        byte[] b = new byte[(codes.length + 1) / 2];
        for (int i = 0; i < codes.length; i++) {
            b[i / 2] |= (byte) ((i & 1) == 0 ? codes[i] << 4 : codes[i]);
        }
        return b;
    }

    private static float[] codebook(boolean fp4) {
        float[] m = new float[16];
        for (int i = 0; i < 16; i++) {
            m[i] = fp4 ? QuantTables.fp4BnbCode(i) : QuantTables.nf4Code(i);
        }
        return m;
    }

    private static byte[] json(String singleQuoted) {
        return singleQuoted.replace('\'', '"').getBytes(StandardCharsets.UTF_8);
    }

    private static String state(String type, int blocksize, long n, long k, String extra) {
        return "{'quant_type':'" + type + "','blocksize':" + blocksize + ",'dtype':'bfloat16','shape':[" + n + ","
                + k + "]" + (extra == null ? "" : "," + extra) + "}";
    }

    /** A 4-bit module with F32 absmax. */
    private Fixture fourBit(String type, int n, int k, int blocksize, float[] absmax, float[] map, String stateJson)
            throws IOException {
        byte[] st = json(stateJson);
        return fixture().add(M + ".weight", StDType.U8, new long[]{(n * (long) k + 1) / 2, 1},
                packHighFirst(codes(n * k)))
                .f32(M + ".weight.absmax", new long[]{absmax.length}, absmax)
                .f32(M + ".weight.quant_map", new long[]{16}, map)
                .add(M + ".weight.quant_state.bitsandbytes__" + type, StDType.U8, new long[]{st.length}, st);
    }

    private static QuantArgs nf4() {
        return QuantArgs.builder(QuantFormat.NF4_BNB).build();
    }

    private static float[] absmax(int blocks) {
        float[] a = new float[blocks];
        for (int i = 0; i < blocks; i++) {
            a[i] = 0.5f + 1.25f * i;
        }
        return a;
    }

    @Test
    public void nf4RepackReplicatesAbsmaxPer64Run() throws IOException {
        int n = 2;
        int k = 128;
        float[] a = {0.5f, 3f};
        HfTensorIndex in = fourBit("nf4", n, k, 128, a, codebook(false), state("nf4", 128, n, k, null)).index();
        HfTensorIndex out = BnbScheme.rewriteModules(nf4(), List.of(M), plain(), in, ModelSourceOptions.defaults());
        RowSource w = out.get(M + ".weight");
        assertEquals(GGMLType.NF4, w.format().ggmlType());
        for (String gone : new String[]{".weight.absmax", ".weight.quant_map",
            ".weight.quant_state.bitsandbytes__nf4"}) {
            assertFalse(gone, out.has(M + gone));
        }
        int[] c = codes(n * k);
        float[] v = Fixture.values(w);
        for (int i = 0; i < c.length; i++) {
            assertEquals("element " + i, QuantTables.nf4Code(c[i]) * a[i / 128], v[i], 0f);
        }
    }

    @Test
    public void doubleQuantIsFoldedInF32AsBitsandbytes() throws IOException {
        int k = 256;
        byte[] q = {0, (byte) 255, 17, (byte) 128};
        float[] nested = {0.5f, 0.25f};
        float[] nestedMap = new float[256];
        for (int i = 0; i < 256; i++) {
            nestedMap[i] = (i - 127.3f) / 97f;
        }
        float offset = 0.03125f;
        byte[] st = json(state("fp4", 64, 1, k, "'nested_blocksize':2,'nested_dtype':'float32','nested_offset':"
                + offset));
        HfTensorIndex in = fixture().add(M + ".weight", StDType.U8, new long[]{k / 2, 1}, packHighFirst(codes(k)))
                .add(M + ".weight.absmax", StDType.U8, new long[]{4}, q)
                .f32(M + ".weight.nested_absmax", new long[]{2}, nested)
                .f32(M + ".weight.nested_quant_map", new long[]{256}, nestedMap)
                .f32(M + ".weight.quant_map", new long[]{16}, codebook(true))
                .add(M + ".weight.quant_state.bitsandbytes__fp4", StDType.U8, new long[]{st.length}, st).index();
        HfTensorIndex out = BnbScheme.rewriteModules(QuantArgs.builder(QuantFormat.FP4_BNB).build(), List.of(M),
                plain(), in, ModelSourceOptions.defaults());
        assertFalse(out.has(M + ".weight.nested_absmax"));
        assertFalse(out.has(M + ".weight.nested_quant_map"));
        RowSource w = out.get(M + ".weight");
        assertEquals(GGMLType.FP4_BNB, w.format().ggmlType());
        int[] c = codes(k);
        float[] v = Fixture.values(w);
        for (int i = 0; i < k; i++) {
            int b = i / 64;
            float am = nestedMap[q[b] & 0xFF] * nested[b / 2] + offset;
            assertEquals("element " + i, QuantTables.fp4BnbCode(c[i]) * am, v[i], 0f);
        }
    }

    @Test
    public void runsCrossingRowsOrBlocksFallBackThroughTheStoredMap() throws IOException {
        int n = 3;
        int k = 96;
        float[] a = absmax(5);
        float[] map = codebook(false);
        HfTensorIndex in = fourBit("nf4", n, k, 64, a, map, state("nf4", 64, n, k, null)).index();
        RowSource w = BnbScheme.rewriteModules(nf4(), List.of(M), plain(), in, ModelSourceOptions.defaults())
                .get(M + ".weight");
        assertEquals(StDType.BF16, w.format().stDType());
        RowSource exact = BnbScheme.rewriteModules(nf4(), List.of(M), plain(), in, EXACT).get(M + ".weight");
        assertEquals(StDType.F32, exact.format().stDType());
        int[] c = codes(n * k);
        float[] v = Fixture.values(exact);
        for (int i = 0; i < c.length; i++) {
            assertEquals("element " + i, map[c[i]] * a[i / 64], v[i], 0f);
        }
        HfTensorIndex small = fourBit("nf4", 1, 64, 32, absmax(2), map, state("nf4", 32, 1, 64, null)).index();
        RowSource s = BnbScheme.rewriteModules(nf4(), List.of(M), plain(), small, EXACT).get(M + ".weight");
        assertEquals(StDType.F32, s.format().stDType());
        float[] sv = Fixture.values(s);
        for (int i = 0; i < 64; i++) {
            assertEquals(map[codes(64)[i]] * absmax(2)[i / 32], sv[i], 0f);
        }
    }

    @Test
    public void quantMapWithinToleranceRepacksAndBeyondFallsBack() throws IOException {
        float[] near = codebook(false);
        near[3] += 5e-5f;
        float[] a = {2f};
        HfTensorIndex in = fourBit("nf4", 1, 64, 64, a, near, state("nf4", 64, 1, 64, null)).index();
        RowSource w = BnbScheme.rewriteModules(nf4(), List.of(M), plain(), in, ModelSourceOptions.defaults())
                .get(M + ".weight");
        assertEquals(GGMLType.NF4, w.format().ggmlType());
        RowSource exact = BnbScheme.rewriteModules(nf4(), List.of(M), plain(), in, EXACT).get(M + ".weight");
        assertEquals(StDType.F32, exact.format().stDType());
        int[] c = codes(64);
        float[] v = Fixture.values(exact);
        for (int i = 0; i < 64; i++) {
            assertEquals(near[c[i]] * 2f, v[i], 0f);
        }
        float[] far = codebook(false);
        far[3] += 1e-3f;
        HfTensorIndex in2 = fourBit("nf4", 1, 64, 64, a, far, state("nf4", 64, 1, 64, null)).index();
        RowSource f = BnbScheme.rewriteModules(nf4(), List.of(M), plain(), in2, ModelSourceOptions.defaults())
                .get(M + ".weight");
        assertEquals(StDType.BF16, f.format().stDType());
    }

    private void assertRejected(Fixture f, String part) throws IOException {
        HfTensorIndex in = f.index();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> BnbScheme.rewriteModules(nf4(),
                List.of(M), plain(), in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains(part));
    }

    @Test
    public void quantStateValidations() throws IOException {
        float[] map = codebook(false);
        assertRejected(fourBit("nf4", 2, 128, 64, absmax(4), map, state("nf4", 64, 2, 64, null)), "expected [64, 1]");
        assertRejected(fourBit("nf4", 1, 64, 64, absmax(2), map, state("nf4", 48, 1, 64, null)), "blocksize 48");
        assertRejected(fourBit("nf4", 1, 128, 64, absmax(3), map, state("nf4", 64, 1, 128, null)),
                "has 3 values; expected 2");
        assertRejected(fourBit("nf4", 1, 64, 64, absmax(1), map, state("nf4", 64, 1, 64, "'extra':1")),
                "unknown key 'extra'");
        assertRejected(fourBit("nf4", 1, 64, 64, absmax(1), map, state("fp4", 64, 1, 64, null)), "quant_type 'fp4'");
        assertRejected(fourBit("nf4", 1, 64, 64, absmax(1), map, state("nf4", 64, 1, 64, null)
                .replace("'bfloat16'", "'int8'")), "dtype 'int8'");
        byte[] st = json(state("nf4", 64, 1, 256, "'nested_blocksize':2,'nested_dtype':'float32','nested_offset':0"));
        assertRejected(fixture().add(M + ".weight", StDType.U8, new long[]{128, 1}, new byte[128])
                .add(M + ".weight.absmax", StDType.U8, new long[]{4}, new byte[4])
                .f32(M + ".weight.nested_absmax", new long[]{3}, 1f, 1f, 1f)
                .f32(M + ".weight.nested_quant_map", new long[]{256}, new float[256])
                .f32(M + ".weight.quant_map", new long[]{16}, map)
                .add(M + ".weight.quant_state.bitsandbytes__nf4", StDType.U8, new long[]{st.length}, st),
                "nested absmax");
        byte[] big = new byte[BnbScheme.MAX_QUANT_STATE_BYTES + 1];
        assertRejected(fixture().add(M + ".weight", StDType.U8, new long[]{32, 1}, new byte[32])
                .f32(M + ".weight.absmax", new long[]{1}, 1f)
                .f32(M + ".weight.quant_map", new long[]{16}, map)
                .add(M + ".weight.quant_state.bitsandbytes__nf4", StDType.U8, new long[]{big.length}, big),
                "at most " + BnbScheme.MAX_QUANT_STATE_BYTES);
    }

    @Test
    public void quantStorageOtherThanUint8IsRejected() throws IOException {
        HfTensorIndex in = fourBit("nf4", 1, 64, 64, absmax(1), codebook(false), state("nf4", 64, 1, 64, null))
                .index();
        HfConfig cfg = Fixture.config(tmp.getRoot().toPath(), "{'architectures':['A'],'quantization_config':"
                + "{'quant_method':'bitsandbytes','load_in_4bit':true,'bnb_4bit_quant_type':'nf4',"
                + "'bnb_4bit_quant_storage':'bfloat16'}}");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> BnbScheme.rewriteModules(nf4(),
                List.of(M), cfg, in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("bnb_4bit_quant_storage 'bfloat16'"));
    }

    private HfTensorIndex int8Module(int n, int k, float[] scb) throws IOException {
        byte[] q = new byte[n * k];
        for (int i = 0; i < q.length; i++) {
            q[i] = (byte) (i * 37 - 100);
        }
        return fixture().add(M + ".weight", StDType.I8, new long[]{n, k}, q)
                .f32(M + ".SCB", new long[]{n}, scb)
                .add(M + ".weight_format", StDType.U8, new long[]{}, new byte[]{0}).index();
    }

    @Test
    public void int8TargetsByWidth() throws IOException {
        QuantArgs args = QuantArgs.builder(QuantFormat.INT8_BNB).group(QuantArgs.CHANNEL).build();
        float[] scb = {12.5f, 0.3f};
        for (int k : new int[]{128, 32}) {
            for (ModelSourceOptions o : new ModelSourceOptions[]{ModelSourceOptions.defaults(), EXACT}) {
                HfTensorIndex in = int8Module(2, k, scb);
                HfTensorIndex out = BnbScheme.rewriteModules(args, List.of(M), plain(), in, o);
                assertFalse(out.has(M + ".SCB"));
                assertFalse(out.has(M + ".weight_format"));
                RowSource w = out.get(M + ".weight");
                boolean q8 = k == 32 && !o.exactQuant();
                if (k == 128) {
                    assertEquals(GGMLType.I8_B128, w.format().ggmlType());
                } else if (q8) {
                    assertEquals(GGMLType.Q8_0, w.format().ggmlType());
                } else {
                    assertEquals(StDType.F32, w.format().stDType());
                }
                float[] v = Fixture.values(w);
                for (int i = 0; i < v.length; i++) {
                    float d = scb[i / k] / 127f;
                    if (q8) {
                        d = FpBlockDecoders.fp16ToFp32(FpBlockDecoders.fp32ToFp16(d));
                    }
                    assertEquals("k " + k + " element " + i, (byte) (i * 37 - 100) * d, v[i], 0f);
                }
            }
        }
    }

    @Test
    public void floatWeightsSavedBeforeQuantizationLoadAsStored() throws IOException {
        HfTensorIndex in = fixture().bf16(M + ".weight", new long[]{1, 2}, 1f, -2f).index();
        HfTensorIndex out = BnbScheme.rewriteModules(nf4(), List.of(M), plain(), in, ModelSourceOptions.defaults());
        assertSame(in.get(M + ".weight"), out.get(M + ".weight"));
        HfTensorIndex out8 = BnbScheme.rewriteModules(QuantArgs.builder(QuantFormat.INT8_BNB).build(), List.of(M),
                plain(), in, ModelSourceOptions.defaults());
        assertSame(in.get(M + ".weight"), out8.get(M + ".weight"));
    }
}
