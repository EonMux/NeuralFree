package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntSchemeFixtures.IntFixture;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MlxScheme}: four-bit affine weights into Q4_1 ({@code d = scale}, {@code m = bias}), F32 under exact
 * quantization when a bias is not an fp16 value, three-bit bit streams into the float fallback, stacked experts, and
 * mismatched parameter shapes. Every value is compared exactly with {@code q * scale + bias}.
 */
public class MlxSchemeTest {

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

    private HfTensorIndex rewrite(int bits, HfTensorIndex in, ModelSourceOptions o, String module)
            throws IOException {
        QuantArgs a = QuantArgs.builder(QuantFormat.INT_MLX).bits(bits).mlxBits(bits).group(32).symmetric(false)
                .build();
        return MlxScheme.rewriteModules(a, List.of(module),
                IntFixture.config(tmp.getRoot().toPath(), IntSchemeFixtures.PLAIN), in, o);
    }

    /** Affine weight rows: codes, and per-(row, group) scales and biases. */
    private static final class Affine {
        final int n;
        final int k;
        final int[][] codes;
        final float[] scales;
        final float[] biases;

        Affine(int n, int k, int bits, int seed) {
            this.n = n;
            this.k = k;
            codes = new int[n][k];
            int groups = k / 32;
            scales = new float[n * groups];
            biases = new float[n * groups];
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < k; c++) {
                    codes[r][c] = (r * 13 + c * 7 + seed) % (1 << bits);
                }
                for (int g = 0; g < groups; g++) {
                    scales[r * groups + g] = 0.0625f * (1 + (r + g + seed) % 6);
                    biases[r * groups + g] = -0.5f * ((r * 3 + g + seed) % 5);
                }
            }
        }

        float[] values() {
            float[] out = new float[n * k];
            int groups = k / 32;
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < k; c++) {
                    int g = r * groups + c / 32;
                    out[r * k + c] = codes[r][c] * scales[g] + biases[g];
                }
            }
            return out;
        }

        int[] packed(int bits) {
            int[] first = pack(codes[0], bits);
            int[] out = new int[n * first.length];
            for (int r = 0; r < n; r++) {
                System.arraycopy(pack(codes[r], bits), 0, out, r * first.length, first.length);
            }
            return out;
        }

        private static int[] pack(int[] v, int bits) {
            return 32 % bits == 0 ? IntSchemeFixtures.packPerWord(v, bits) : IntSchemeFixtures.packBitStream(v, bits);
        }
    }

    @Test
    public void fourBitAffineBecomesQ4_1() throws IOException {
        Affine w = new Affine(4, 64, 4, 1);
        IntFixture f = fixture();
        f.u32(M + ".weight", new long[]{4, 8}, w.packed(4));
        f.f16(M + ".scales", new long[]{4, 2}, w.scales);
        f.f16(M + ".biases", new long[]{4, 2}, w.biases);
        HfTensorIndex out = rewrite(4, f.index(), ModelSourceOptions.defaults(), M);
        RowSource s = out.get(M + ".weight");
        assertEquals(GGMLType.Q4_1, s.format().ggmlType());
        assertTrue(s.rowLocal());
        assertArrayEquals(w.values(), IntFixture.values(s), 0f);
        assertFalse(out.has(M + ".scales"));
        assertFalse(out.has(M + ".biases"));
    }

    @Test
    public void biasesOutsideFp16TakeF32UnderExactQuantization() throws IOException {
        Affine w = new Affine(4, 64, 4, 2);
        for (int i = 0; i < w.biases.length; i++) {
            w.biases[i] = FpBlockDecoders.bf16ToFp32((short) HalfFloats.floatToBFloat16(1e-20f * (i + 1)));
        }
        IntFixture f = fixture();
        f.u32(M + ".weight", new long[]{4, 8}, w.packed(4));
        f.bf16(M + ".scales", new long[]{4, 2}, w.scales);
        f.bf16(M + ".biases", new long[]{4, 2}, w.biases);
        HfTensorIndex in = f.index();
        RowSource exact = rewrite(4, in, ModelSourceOptions.builder().exactQuant(true).build(), M).get(M + ".weight");
        assertEquals(GGMLType.F32, exact.format().ggmlType());
        assertArrayEquals(w.values(), IntFixture.values(exact), 0f);
        assertEquals(GGMLType.Q4_1, rewrite(4, in, ModelSourceOptions.defaults(), M).get(M + ".weight").format()
                .ggmlType());
    }

    @Test
    public void threeBitBitStreamTakesTheFloatFallback() throws IOException {
        Affine w = new Affine(4, 64, 3, 3);
        IntFixture f = fixture();
        int[] packed = w.packed(3);
        assertEquals(4 * 6, packed.length);
        f.u32(M + ".weight", new long[]{4, 6}, packed);
        f.f16(M + ".scales", new long[]{4, 2}, w.scales);
        f.f16(M + ".biases", new long[]{4, 2}, w.biases);
        RowSource s = rewrite(3, f.index(), ModelSourceOptions.defaults(), M).get(M + ".weight");
        assertEquals(GGMLType.F16, s.format().ggmlType());
        assertArrayEquals(w.values(), IntFixture.values(s), 0f);
    }

    @Test
    public void stackedExperts() throws IOException {
        String experts = "model.layers.0.mlp.switch_mlp.gate_proj";
        Affine e0 = new Affine(4, 64, 4, 4);
        Affine e1 = new Affine(4, 64, 4, 5);
        IntFixture f = fixture();
        f.u32(experts + ".weight", new long[]{2, 4, 8}, IntSchemeFixtures.concat(e0.packed(4), e1.packed(4)));
        f.f16(experts + ".scales", new long[]{2, 4, 2}, IntSchemeFixtures.concat(e0.scales, e1.scales));
        f.f16(experts + ".biases", new long[]{2, 4, 2}, IntSchemeFixtures.concat(e0.biases, e1.biases));
        RowSource s = rewrite(4, f.index(), ModelSourceOptions.defaults(), experts).get(experts + ".weight");
        assertArrayEquals(new long[]{2, 4, 64}, s.shape());
        assertEquals(GGMLType.Q4_1, s.format().ggmlType());
        assertArrayEquals(IntSchemeFixtures.concat(e0.values(), e1.values()), IntFixture.values(s), 0f);
    }

    @Test
    public void mismatchedBiasShapeIsRejected() throws IOException {
        Affine w = new Affine(4, 64, 4, 6);
        IntFixture f = fixture();
        f.u32(M + ".weight", new long[]{4, 8}, w.packed(4));
        f.f16(M + ".scales", new long[]{4, 2}, w.scales);
        f.f16(M + ".biases", new long[]{2, 4}, w.biases);
        IOException e = assertThrows(IOException.class, () -> rewrite(4, f.index(), ModelSourceOptions.defaults(), M));
        assertTrue(e.getMessage(), e.getMessage().contains(M + ".biases has shape [2, 4]; expected [4, 2]"));
    }
}
