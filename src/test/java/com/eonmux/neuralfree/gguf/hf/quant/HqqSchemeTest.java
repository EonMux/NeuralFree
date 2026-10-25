package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntSchemeFixtures.IntFixture;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
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
 * {@link HqqScheme}: the slab order of {@code W_q} (the high nibble holds the first half of the flattened group rows,
 * as {@code unpack_4bit_u8}; two-bit slabs from the top bits down), float zero points into Q4_1 or, under exact
 * quantization, F32 (Q8_0 for integral zeros), metadata consumption and validation, and the weight shape from
 * {@code bias}. Every value is compared exactly with {@code (q - zero) * scale}.
 */
public class HqqSchemeTest {

    private static final String M = IntSchemeFixtures.M;
    private static final int N = 4;
    private static final int K = 64;
    private static final int G = 32;
    private static final int ROWS = N * K / G;

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

    private static QuantArgs args(int bits) {
        return QuantArgs.builder(QuantFormat.INT_HQQ).bits(bits).group(G).symmetric(false).hqqAxis(1).build();
    }

    private HfTensorIndex rewrite(QuantArgs a, HfTensorIndex in, ModelSourceOptions o) throws IOException {
        return HqqScheme.rewriteModules(a, List.of(M),
                IntFixture.config(tmp.getRoot().toPath(), IntSchemeFixtures.PLAIN), in, o);
    }

    /** Codes of {@code w[n][k]}; flattened row {@code r} of the group view is elements {@code [r*G, r*G + G)}. */
    private static int[] codes(int bits, int seed) {
        int[] c = new int[N * K];
        for (int i = 0; i < c.length; i++) {
            c[i] = (i * 5 + i / 9 + seed) % (1 << bits);
        }
        return c;
    }

    /** HQQ bitpack: packed row {@code p} holds rows {@code p + i*step} at bit {@code (slots - 1 - i) * b}. */
    private static int[] pack(int[] codes, int bits) {
        int slots = 8 / bits;
        int step = ROWS / slots;
        int[] packed = new int[step * G];
        for (int r = 0; r < ROWS; r++) {
            int slab = r / step;
            int p = r % step;
            for (int c = 0; c < G; c++) {
                packed[p * G + c] |= codes[r * G + c] << ((slots - 1 - slab) * bits);
            }
        }
        return packed;
    }

    private static float[] expected(int[] codes, float[] scale, float[] zero) {
        float[] out = new float[codes.length];
        for (int i = 0; i < codes.length; i++) {
            out[i] = (codes[i] - zero[i / G]) * scale[i / G];
        }
        return out;
    }

    private static float[] scales() {
        float[] s = new float[ROWS];
        for (int r = 0; r < ROWS; r++) {
            s[r] = 0.125f * (1 + r % 3);
        }
        return s;
    }

    private void addModule(IntFixture f, int bits, int[] codes, float[] scale, float[] zero) {
        f.bytes(M + ".W_q", StDType.U8, new long[]{ROWS / (8 / bits), G}, pack(codes, bits));
        f.f16(M + ".scale", new long[]{ROWS, 1}, scale);
        f.f16(M + ".zero", new long[]{ROWS, 1}, zero);
        f.i64(M + ".shape", new long[]{2}, N, K);
        f.i64(M + ".nbits", new long[]{1}, bits);
        f.i64(M + ".group_size", new long[]{1}, G);
        f.i64(M + ".axis", new long[]{1}, 1);
        f.bytes(M + ".channel_wise", StDType.BOOL, new long[]{1}, 1);
        f.bytes(M + ".view_as_float", StDType.BOOL, new long[]{1}, 0);
    }

    @Test
    public void fourBitNibbleOrderWithFloatZeros() throws IOException {
        int[] c = codes(4, 1);
        float[] s = scales();
        float[] z = new float[ROWS];
        for (int r = 0; r < ROWS; r++) {
            z[r] = 7.5f - r % 4;
        }
        IntFixture f = fixture();
        addModule(f, 4, c, s, z);
        f.f16(M + ".bias", new long[]{N}, 1f, 2f, 3f, 4f);
        HfTensorIndex in = f.index();
        HfTensorIndex out = rewrite(args(4), in, ModelSourceOptions.defaults());
        RowSource src = out.get(M + ".weight");
        assertEquals(GGMLType.Q4_1, src.format().ggmlType());
        assertArrayEquals(new long[]{N, K}, src.shape());
        assertTrue(src.rowLocal());
        assertArrayEquals(expected(c, s, z), IntFixture.values(src), 0f);
        assertArrayEquals(IntSchemeFixtures.slice(expected(c, s, z), K, 2 * K), IntSchemeFixtures.rows(src, 1, 2),
                0f);
        for (String meta : new String[]{"W_q", "scale", "zero", "shape", "nbits", "group_size", "axis",
                "channel_wise", "view_as_float"}) {
            assertFalse(meta, out.has(M + "." + meta));
        }
        assertTrue(out.has(M + ".bias"));

        RowSource exact = rewrite(args(4), in, ModelSourceOptions.builder().exactQuant(true).build())
                .get(M + ".weight");
        assertEquals(GGMLType.F32, exact.format().ggmlType());
        assertArrayEquals(expected(c, s, z), IntFixture.values(exact), 0f);
    }

    @Test
    public void integralZerosUnderExactQuantizationBecomeQ8_0() throws IOException {
        int[] c = codes(4, 2);
        float[] s = scales();
        float[] z = new float[ROWS];
        for (int r = 0; r < ROWS; r++) {
            z[r] = 3 + r;
        }
        IntFixture f = fixture();
        addModule(f, 4, c, s, z);
        RowSource src = rewrite(args(4), f.index(), ModelSourceOptions.builder().exactQuant(true).build())
                .get(M + ".weight");
        assertEquals(GGMLType.Q8_0, src.format().ggmlType());
        assertArrayEquals(expected(c, s, z), IntFixture.values(src), 0f);
    }

    @Test
    public void twoBitSlabsFromTheTopBitsDown() throws IOException {
        int[] c = codes(2, 3);
        float[] s = scales();
        float[] z = new float[ROWS];
        for (int r = 0; r < ROWS; r++) {
            z[r] = r % 4;
        }
        IntFixture f = fixture();
        addModule(f, 2, c, s, z);
        RowSource src = rewrite(args(2), f.index(), ModelSourceOptions.defaults()).get(M + ".weight");
        assertEquals(GGMLType.Q8_0, src.format().ggmlType());
        assertArrayEquals(expected(c, s, z), IntFixture.values(src), 0f);
    }

    @Test
    public void shapeFromBiasWhenTheMetadataIsAbsent() throws IOException {
        int[] c = codes(4, 4);
        float[] s = scales();
        float[] z = new float[ROWS];
        IntFixture f = fixture();
        f.bytes(M + ".W_q", StDType.U8, new long[]{ROWS / 2, G}, pack(c, 4));
        f.f16(M + ".scale", new long[]{ROWS, 1}, s);
        f.f16(M + ".zero", new long[]{ROWS, 1}, z);
        f.f16(M + ".bias", new long[]{N}, 0f, 0f, 0f, 0f);
        RowSource src = rewrite(args(4), f.index(), ModelSourceOptions.defaults()).get(M + ".weight");
        assertArrayEquals(new long[]{N, K}, src.shape());
        assertArrayEquals(expected(c, s, z), IntFixture.values(src), 0f);
    }

    @Test
    public void axisOtherThanOneIsRejected() throws IOException {
        IntFixture f = fixture();
        addModule(f, 4, codes(4, 5), scales(), new float[ROWS]);
        f.i64(M + ".axis", new long[]{1}, 0);
        HfTensorIndex in = f.index();
        IOException e = assertThrows(IOException.class, () -> rewrite(args(4), in, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains(M + ".axis is 0"));
        QuantArgs axis0 = QuantArgs.builder(QuantFormat.INT_HQQ).bits(4).group(G).symmetric(false).hqqAxis(0).build();
        assertThrows(GgufFormatException.class, () -> rewrite(axis0, in, ModelSourceOptions.defaults()));
    }

    @Test
    public void mismatchedPackedShapeIsRejected() throws IOException {
        IntFixture f = fixture();
        addModule(f, 4, codes(4, 6), scales(), new float[ROWS]);
        f.bytes(M + ".W_q", StDType.U8, new long[]{ROWS, G / 2}, new int[ROWS * G / 2]);
        IOException e = assertThrows(IOException.class, () -> rewrite(args(4), f.index(), ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains(M + ".W_q has shape [8, 16]"));
    }
}
