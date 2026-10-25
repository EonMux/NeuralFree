package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntSchemeFixtures.IntFixture;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Int8Scheme}: per-channel, per-tensor and block int8 scales into I8_B128 (and Q8_0 or the float fallback when
 * K or the blocks do not split into 128-element runs), asymmetric zero points, stacked experts, and int4 values stored
 * one or two per byte. Every produced value is compared exactly with {@code (q - zp) * s}.
 */
public class Int8SchemeTest {

    private static final String M = IntSchemeFixtures.M;

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

    private HfTensorIndex rewrite(QuantArgs args, HfTensorIndex in, ModelSourceOptions o, String module)
            throws IOException {
        return Int8Scheme.rewriteModules(args, List.of(module),
                IntFixture.config(tmp.getRoot().toPath(), IntSchemeFixtures.PLAIN), in, o);
    }

    private static QuantArgs channel() {
        return QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(QuantArgs.CHANNEL).build();
    }

    /** Signed int8 values covering the whole range. */
    private static int[] int8(int n, int k, int seed) {
        int[] q = new int[n * k];
        for (int i = 0; i < q.length; i++) {
            q[i] = ((i * 37 + seed * 11) % 256) - 128;
        }
        return q;
    }

    private static float[] expected(int[] q, int k, float[] rowScale, int[] rowZero) {
        float[] out = new float[q.length];
        for (int i = 0; i < q.length; i++) {
            int n = i / k;
            out[i] = (q[i] - (rowZero == null ? 0 : rowZero[n])) * rowScale[n];
        }
        return out;
    }

    @Test
    public void perChannelScalesBecomeI8B128() throws IOException {
        int[] q = int8(4, 128, 1);
        float[] s = {0.0123f, 0.5f, 3.3e-3f, 7.77f};
        IntFixture f = fixture();
        f.bytes(M + ".weight", StDType.I8, new long[]{4, 128}, q);
        f.f32(M + ".weight_scale", new long[]{4, 1}, s);
        HfTensorIndex out = rewrite(channel(), f.index(), ModelSourceOptions.defaults(), M);
        RowSource src = out.get(M + ".weight");
        assertEquals(GGMLType.I8_B128, src.format().ggmlType());
        assertTrue(src.rowLocal());
        assertArrayEquals(expected(q, 128, s, null), IntFixture.values(src), 0f);
        assertFalse(out.has(M + ".weight_scale"));
    }

    @Test
    public void shortRowsTakeQ8_0OrExactF32() throws IOException {
        int[] q = int8(4, 64, 2);
        float[] s = {0.0123f, 0.5f, 0.25f, 2f};
        IntFixture f = fixture();
        f.bytes(M + ".weight", StDType.I8, new long[]{4, 64}, q);
        f.f32(M + ".weight_scale", new long[]{4}, s);
        HfTensorIndex in = f.index();
        RowSource relaxed = rewrite(channel(), in, ModelSourceOptions.defaults(), M).get(M + ".weight");
        assertEquals(GGMLType.Q8_0, relaxed.format().ggmlType());
        RowSource exact = rewrite(channel(), in, ModelSourceOptions.builder().exactQuant(true).build(), M)
                .get(M + ".weight");
        assertEquals(GGMLType.F32, exact.format().ggmlType());
        assertArrayEquals(expected(q, 64, s, null), IntFixture.values(exact), 0f);
    }

    @Test
    public void perTensorScaleBroadcasts() throws IOException {
        int[] q = int8(4, 128, 3);
        IntFixture f = fixture();
        f.bytes(M + ".weight", StDType.I8, new long[]{4, 128}, q);
        f.f32(M + ".weight_scale", new long[]{1}, 0.031f);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(QuantArgs.UNGROUPED).build();
        RowSource src = rewrite(a, f.index(), ModelSourceOptions.defaults(), M).get(M + ".weight");
        assertEquals(GGMLType.I8_B128, src.format().ggmlType());
        assertArrayEquals(expected(q, 128, new float[]{0.031f, 0.031f, 0.031f, 0.031f}, null),
                IntFixture.values(src), 0f);
    }

    @Test
    public void nonzeroZeroPointsTakeTheFloatFallback() throws IOException {
        int[] q = int8(4, 128, 4);
        float[] s = {0.5f, 0.25f, 0.125f, 1f};
        int[] zp = {0, 3, -2, 5};
        IntFixture f = fixture();
        f.bytes(M + ".weight", StDType.I8, new long[]{4, 128}, q);
        f.f32(M + ".weight_scale", new long[]{4, 1}, s);
        f.i32(M + ".weight_zero_point", new long[]{4, 1}, zp);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(QuantArgs.CHANNEL).symmetric(false).build();
        HfTensorIndex out = rewrite(a, f.index(), ModelSourceOptions.defaults(), M);
        RowSource src = out.get(M + ".weight");
        assertEquals(GGMLType.F32, src.format().ggmlType());
        assertArrayEquals(expected(q, 128, s, zp), IntFixture.values(src), 0f);
        assertFalse(out.has(M + ".weight_zero_point"));
    }

    @Test
    public void blockScalesFollowTheirBlocks() throws IOException {
        int[] q = int8(4, 256, 5);
        float[] blocks = {0.5f, 0.0625f, 1.5f, 0.0078125f};
        IntFixture f = fixture();
        f.bytes(M + ".weight", StDType.I8, new long[]{4, 256}, q);
        f.f32(M + ".weight_scale_inv", new long[]{2, 2}, blocks);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT8_BLOCK).block(2, 128)
                .actScheme(ActQuantScheme.INT8_GROUP128).build();
        RowSource src = rewrite(a, f.index(), ModelSourceOptions.defaults(), M).get(M + ".weight");
        assertEquals(GGMLType.I8_B128, src.format().ggmlType());
        assertEquals(ActQuantScheme.INT8_GROUP128, src.actQuant());
        float[] v = IntFixture.values(src);
        for (int i = 0; i < q.length; i++) {
            int n = i / 256;
            int k = i % 256;
            assertEquals("element " + i, q[i] * blocks[(n / 2) * 2 + k / 128], v[i], 0f);
        }
    }

    @Test
    public void blocksNarrowerThan128TakeQ8_0() throws IOException {
        int[] q = int8(4, 128, 6);
        float[] blocks = {0.5f, 0.25f, 2f, 0.125f};
        IntFixture f = fixture();
        f.bytes(M + ".weight", StDType.I8, new long[]{4, 128}, q);
        f.f32(M + ".weight_scale_inv", new long[]{1, 2}, 0.5f, 0.25f);
        f.f32(M + ".weight_scale", new long[]{2, 2}, blocks);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT8_BLOCK).block(4, 64).build();
        RowSource src = rewrite(a, f.index(), ModelSourceOptions.defaults(), M).get(M + ".weight");
        assertEquals(GGMLType.Q8_0, src.format().ggmlType());
        float[] v = IntFixture.values(src);
        for (int i = 0; i < q.length; i++) {
            assertEquals("element " + i, q[i] * (i % 128 < 64 ? 0.5f : 0.25f), v[i], 0f);
        }
    }

    @Test
    public void mismatchedBlockScalesAreRejected() throws IOException {
        IntFixture f = fixture();
        f.bytes(M + ".weight", StDType.I8, new long[]{4, 256}, int8(4, 256, 7));
        f.f32(M + ".weight_scale_inv", new long[]{2, 3}, new float[6]);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT8_BLOCK).block(2, 128).build();
        IOException e = assertThrows(IOException.class, () -> rewrite(a, f.index(), ModelSourceOptions.defaults(), M));
        assertTrue(e.getMessage(), e.getMessage().contains("need [2, 2]"));
    }

    @Test
    public void stackedExpertsKeepPerExpertScales() throws IOException {
        String experts = "model.layers.0.mlp.experts.down_proj";
        int[] q = int8(2 * 4, 128, 8);
        float[] s = {0.5f, 0.25f, 0.125f, 1f, 2f, 4f, 0.75f, 0.375f};
        IntFixture f = fixture();
        f.bytes(experts, StDType.I8, new long[]{2, 4, 128}, q);
        f.f32(experts + "_weight_scale", new long[]{2, 4, 1}, s);
        HfTensorIndex out = rewrite(channel(), f.index(), ModelSourceOptions.defaults(), experts);
        RowSource src = out.get(experts);
        assertArrayEquals(new long[]{2, 4, 128}, src.shape());
        assertEquals(GGMLType.I8_B128, src.format().ggmlType());
        assertArrayEquals(expected(q, 128, s, null), IntFixture.values(src), 0f);
        assertFalse(out.has(experts + "_weight_scale"));
    }

    @Test
    public void int4OnePerByteBecomesQ4_0() throws IOException {
        int[] q = new int[4 * 64];
        for (int i = 0; i < q.length; i++) {
            q[i] = (i * 5) % 16 - 8;
        }
        float[] s = new float[4 * 2];
        for (int i = 0; i < s.length; i++) {
            s[i] = 0.125f * (i + 1);
        }
        IntFixture f = fixture();
        f.bytes(M + ".weight_packed", StDType.I8, new long[]{4, 64}, q);
        f.f16(M + ".weight_scale", new long[]{4, 2}, s);
        HfTensorIndex in = f.index();
        QuantArgs a = QuantArgs.builder(QuantFormat.INT4_IN_INT8).bits(4).group(32).build();
        RowSource src = Int8Scheme.rewriteModules(a, List.of(M),
                IntFixture.config(tmp.getRoot().toPath(), IntSchemeFixtures.PLAIN), in, ModelSourceOptions.defaults())
                .get(M + ".weight");
        assertEquals(GGMLType.Q4_0, src.format().ggmlType());
        float[] v = IntFixture.values(src);
        for (int i = 0; i < q.length; i++) {
            assertEquals("element " + i, q[i] * s[(i / 64) * 2 + (i % 64) / 32], v[i], 0f);
        }
    }

    @Test
    public void int4OutsideItsRangeIsRejectedOnRead() throws IOException {
        int[] q = new int[4 * 64];
        q[70] = 9;
        IntFixture f = fixture();
        f.bytes(M + ".weight_packed", StDType.I8, new long[]{4, 64}, q);
        f.f16(M + ".weight_scale", new long[]{4, 1}, 1f, 1f, 1f, 1f);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT4_IN_INT8).bits(4).group(QuantArgs.CHANNEL).build();
        RowSource src = rewrite(a, f.index(), ModelSourceOptions.defaults(), M).get(M + ".weight");
        IOException e = assertThrows(IOException.class, () -> IntFixture.values(src));
        assertTrue(e.getMessage(), e.getMessage().contains("row 1 column 6: code 17 exceeds 4 bits"));
    }

    @Test
    public void w4afp8TwoSignedNibblesPerByte() throws IOException {
        String expert = "model.layers.0.mlp.experts.3.w1";
        int k = 128;
        int[] q = new int[4 * k];
        for (int i = 0; i < q.length; i++) {
            q[i] = (i * 3 + i / 7) % 16 - 8;
        }
        int[] packed = new int[q.length / 2];
        for (int j = 0; j < packed.length; j++) {
            packed[j] = (q[2 * j] & 0xF) | ((q[2 * j + 1] & 0xF) << 4);
        }
        float[] s = {0.25f, 0.5f, 1.25f, 0.0625f};
        IntFixture f = fixture();
        f.bytes(expert + ".weight", StDType.I8, new long[]{4, k / 2}, packed);
        f.f32(expert + ".weight_scale_inv", new long[]{4, 1}, s);
        f.f32(expert + ".input_scale", new long[]{1}, 0.02f);
        QuantArgs a = QuantArgs.builder(QuantFormat.INT4_IN_INT8).group(128).actScheme(ActQuantScheme.FP8_STATIC)
                .build();
        HfTensorIndex out = rewrite(a, f.index(), ModelSourceOptions.defaults(), expert);
        RowSource src = out.get(expert + ".weight");
        assertArrayEquals(new long[]{4, k}, src.shape());
        assertEquals(GGMLType.Q4_0, src.format().ggmlType());
        assertEquals(ActQuantScheme.FP8_STATIC, src.actQuant());
        assertArrayEquals(new float[]{0.02f}, src.inputScale(), 0f);
        assertArrayEquals(expected(q, k, s, null), IntFixture.values(src), 0f);
        assertFalse(out.has(expert + ".weight_scale_inv"));
        assertFalse(out.has(expert + ".input_scale"));
    }
}
