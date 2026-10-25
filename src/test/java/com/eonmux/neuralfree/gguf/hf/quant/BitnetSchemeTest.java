package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntSchemeFixtures.IntFixture;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.tensor.HalfFloats;
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
 * {@link BitnetScheme}: the packed row order ({@code n = i * N/4 + r} from bits {@code 2i} of row {@code r}, as
 * llama.cpp base.py dequant_bitnet), TQ2_0 with the inverse scale as the output scale, its folded form, and the
 * float fallback for rows that are not a multiple of 256.
 */
public class BitnetSchemeTest {

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

    private HfTensorIndex rewrite(int n, int k, float scale, ModelSourceOptions o, int[][] codes) throws IOException {
        close();
        fx = new IntFixture(tmp.getRoot().toPath());
        int quarter = n / 4;
        int[] packed = new int[quarter * k];
        for (int row = 0; row < n; row++) {
            int i = row / quarter;
            int r = row % quarter;
            for (int c = 0; c < k; c++) {
                packed[r * k + c] |= codes[row][c] << (2 * i);
            }
        }
        fx.bytes(M + ".weight", StDType.U8, new long[]{quarter, k}, packed);
        fx.bf16(M + ".weight_scale", new long[]{1}, scale);
        QuantArgs a = QuantArgs.builder(QuantFormat.BITNET).build();
        return BitnetScheme.rewriteModules(a, List.of(M),
                IntFixture.config(tmp.getRoot().toPath(), IntSchemeFixtures.PLAIN), fx.index(), o);
    }

    private static int[][] ternary(int n, int k) {
        int[][] c = new int[n][k];
        for (int r = 0; r < n; r++) {
            for (int j = 0; j < k; j++) {
                c[r][j] = (r * 7 + j * 3 + j / 11) % 3;
            }
        }
        return c;
    }

    @Test
    public void rowOrderAndOutputScale() throws IOException {
        int[][] codes = ternary(8, 256);
        HfTensorIndex out = rewrite(8, 256, 2.5f, ModelSourceOptions.defaults(), codes);
        assertFalse(out.has(M + ".weight_scale"));
        RowSource s = out.get(M + ".weight");
        assertEquals(GGMLType.TQ2_0, s.format().ggmlType());
        assertArrayEquals(new long[]{8, 256}, s.shape());
        assertArrayEquals(new float[]{1f / 2.5f}, s.outputScale(), 0f);
        float[] v = IntFixture.values(s);
        for (int r = 0; r < 8; r++) {
            for (int j = 0; j < 256; j++) {
                assertEquals("row " + r + " column " + j, codes[r][j] - 1, v[r * 256 + j], 0f);
            }
        }
        float[] tail = IntSchemeFixtures.rows(s, 5, 2);
        for (int j = 0; j < 256; j++) {
            assertEquals(codes[6][j] - 1, tail[256 + j], 0f);
        }
    }

    @Test
    public void foldedOutputScale() throws IOException {
        int[][] codes = ternary(4, 256);
        RowSource s = rewrite(4, 256, 2.5f, ModelSourceOptions.defaults(), codes).get(M + ".weight");
        RowSource folded = s.foldOutputScale();
        assertNull(folded.outputScale());
        assertEquals(GGMLType.TQ2_0, folded.format().ggmlType());
        float d = FpBlockDecoders.roundToFp16(1f / 2.5f);
        float[] v = IntFixture.values(folded);
        for (int i = 0; i < v.length; i++) {
            assertEquals((codes[i / 256][i % 256] - 1) * d, v[i], 0f);
        }
        assertThrows(UnsupportedOperationException.class, folded::foldOutputScale);

        RowSource exact = rewrite(4, 256, 2.5f, ModelSourceOptions.builder().exactQuant(true).build(), codes)
                .get(M + ".weight").foldOutputScale();
        assertEquals(GGMLType.F32, exact.format().ggmlType());
        float[] e = IntFixture.values(exact);
        for (int i = 0; i < e.length; i++) {
            assertEquals((codes[i / 256][i % 256] - 1) / 2.5f, e[i], 0f);
        }
    }

    @Test
    public void shortRowsTakeTheFloatFallback() throws IOException {
        int[][] codes = ternary(4, 64);
        RowSource s = rewrite(4, 64, 0.5f, ModelSourceOptions.defaults(), codes).get(M + ".weight");
        assertEquals(GGMLType.F32, s.format().ggmlType());
        assertEquals(M + " BITNET b2 g64 -> F32", s.describe());
        assertNull(s.outputScale());
        float[] v = IntFixture.values(s);
        for (int i = 0; i < v.length; i++) {
            assertEquals((codes[i / 64][i % 64] - 1) / 0.5f, v[i], 0f);
        }
    }

    @Test
    public void foldingAnInverseScaleOutsideFp16TakesF32() throws IOException {
        int[][] codes = ternary(4, 256);
        float scale = 1e-6f;
        RowSource s = rewrite(4, 256, scale, ModelSourceOptions.defaults(), codes).get(M + ".weight");
        assertEquals(M + " BITNET b2 g256 -> TQ2_0", s.describe());
        RowSource folded = s.foldOutputScale();
        assertEquals(GGMLType.F32, folded.format().ggmlType());
        float stored = FpBlockDecoders.bf16ToFp32((short) HalfFloats.floatToBFloat16(scale));
        float[] v = IntFixture.values(folded);
        for (int i = 0; i < v.length; i++) {
            assertEquals((codes[i / 256][i % 256] - 1) / stored, v[i], 0f);
        }
    }

    @Test
    public void zeroScaleIsRejected() {
        IOException e = assertThrows(IOException.class,
                () -> rewrite(4, 256, 0f, ModelSourceOptions.defaults(), ternary(4, 256)));
        assertTrue(e.getMessage(), e.getMessage().contains(M + ".weight_scale is 0.0"));
    }
}
