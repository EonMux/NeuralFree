package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for {@link HadamardSpec}, {@link CpuHadamard} and {@link PrismHadamardLoader}: orthonormality of the
 * normalized Sylvester matrix, agreement of the fast transform with the dense matrix, the sign orders, the
 * tiled-to-grouped permutation, and the {@code prism.hadamard.*} metadata rules of llama.cpp.
 */
public class HadamardTest {

    private static final float EPS = 1e-5f;

    private static float[] randomVector(int n, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2.0f - 1.0f;
        }
        return v;
    }

    private static float[] randomSigns(int n, long seed) {
        Random rnd = new Random(seed);
        float[] s = new float[n];
        for (int i = 0; i < n; i++) {
            s[i] = rnd.nextBoolean() ? 1.0f : -1.0f;
        }
        return s;
    }

    /** y = M x for a row-major n x n matrix, blockwise over a row of width n * k. */
    private static float[] blockMatVec(float[] m, int n, float[] x) {
        float[] y = new float[x.length];
        for (int base = 0; base < x.length; base += n) {
            for (int r = 0; r < n; r++) {
                double acc = 0.0;
                for (int c = 0; c < n; c++) {
                    acc += (double) m[r * n + c] * x[base + c];
                }
                y[base + r] = (float) acc;
            }
        }
        return y;
    }

    // ---- HadamardSpec ------------------------------------------------------------------------------------

    @Test
    public void matrixSquaredIsIdentityForBlock64() {
        final int n = 64;
        float[] h = CpuHadamard.matrix(n);
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double acc = 0.0;
                for (int j = 0; j < n; j++) {
                    acc += (double) h[i * n + j] * h[j * n + k];
                }
                assertEquals("(H*H)[" + i + "][" + k + "]", i == k ? 1.0 : 0.0, acc, 1e-6);
            }
        }
    }

    @Test
    public void matrixEntriesFollowSylvesterParity() {
        float[] h = CpuHadamard.matrix(4);
        float s = 0.5f;
        float[] expected = {
            s, s, s, s,
            s, -s, s, -s,
            s, s, -s, -s,
            s, -s, -s, s,
        };
        assertArrayEquals(expected, h, 0.0f);
    }

    @Test
    public void fastTransformMatchesDenseMatrix() {
        for (int n : new int[] {1, 2, 4, 8, 64, 128, 1024}) {
            float[] x = randomVector(n, n);
            float[] expected = blockMatVec(CpuHadamard.matrix(n), n, x);
            float[] v = x.clone();
            CpuHadamard.fwht(v, 0, n);
            assertArrayEquals("n=" + n, expected, v, EPS);
        }
    }

    @Test
    public void fastTransformTwiceIsIdentity() {
        float[] x = randomVector(256, 7);
        float[] v = x.clone();
        CpuHadamard.fwht(v, 0, 256);
        CpuHadamard.fwht(v, 0, 256);
        assertArrayEquals(x, v, EPS);
    }

    @Test
    public void plainRotationIsBlockDiagonal() {
        final int block = 64;
        final int width = 256;
        final int rows = 3;
        float[] x = randomVector(rows * width, 11);
        float[] y = new float[rows * width];
        CpuHadamard.apply(HadamardSpec.rotation(block), x, 0, y, 0, rows, width);
        float[] h = CpuHadamard.matrix(block);
        for (int r = 0; r < rows; r++) {
            float[] row = Arrays.copyOfRange(x, r * width, (r + 1) * width);
            assertArrayEquals(blockMatVec(h, block, row), Arrays.copyOfRange(y, r * width, (r + 1) * width), EPS);
        }
    }

    @Test
    public void signsBeforeThenAfterRoundTrip() {
        final int block = 32;
        final int width = 128;
        float[] signs = randomSigns(width, 3);
        HadamardSpec forward = HadamardSpec.of(block, HadamardSpec.SignOrder.BEFORE_ROTATION, signs, 0, 0, 0);
        HadamardSpec inverse = HadamardSpec.of(block, HadamardSpec.SignOrder.AFTER_ROTATION, signs, 0, 0, 0);
        float[] x = randomVector(2 * width, 5);
        float[] y = new float[x.length];
        CpuHadamard.apply(forward, x, 0, y, 0, 2, width);

        float[] dx = new float[width];
        for (int i = 0; i < width; i++) {
            dx[i] = signs[i] * x[i];
        }
        assertArrayEquals(blockMatVec(CpuHadamard.matrix(block), block, dx), Arrays.copyOf(y, width), EPS);

        CpuHadamard.applyInPlace(inverse, y, 0, 2, width);
        assertArrayEquals(x, y, EPS);
    }

    @Test
    public void permutationGathersTiledIntoGroupedOrder() {
        final int hd = 2;
        final int nk = 3;
        final int rep = 2;
        HadamardSpec spec = HadamardSpec.of(4, HadamardSpec.SignOrder.NONE, null, hd, nk, rep);
        // ggml_permute(x, 0, 2, 1, 3) of the [hd, nk, rep] view: dst(h, r, k) = src(h, k, r)
        int[] expected = new int[hd * nk * rep];
        for (int r = 0; r < rep; r++) {
            for (int k = 0; k < nk; k++) {
                for (int h = 0; h < hd; h++) {
                    expected[h + hd * (r + rep * k)] = h + hd * (k + nk * r);
                }
            }
        }
        for (int o = 0; o < expected.length; o++) {
            assertEquals("src(" + o + ")", expected[o], spec.sourceIndex(o));
        }
        assertEquals(12, spec.requiredWidth());
        assertFalse(spec.inPlaceAllowed());
    }

    @Test
    public void permutedRotationInPlaceMatchesOutOfPlace() {
        final int width = 64;
        HadamardSpec spec = HadamardSpec.of(16, HadamardSpec.SignOrder.BEFORE_ROTATION, randomSigns(width, 9), 8, 2, 4);
        float[] x = randomVector(2 * width, 13);
        float[] y = new float[x.length];
        CpuHadamard.apply(spec, x, 0, y, 0, 2, width);
        float[] v = x.clone();
        CpuHadamard.applyInPlace(spec, v, 0, 2, width);
        assertArrayEquals(y, v, 0.0f);

        float[] gathered = new float[width];
        for (int o = 0; o < width; o++) {
            gathered[o] = spec.sign(o) * x[spec.sourceIndex(o)];
        }
        assertArrayEquals(blockMatVec(CpuHadamard.matrix(16), 16, gathered), Arrays.copyOf(y, width), EPS);
    }

    @Test
    public void specRejectsInvalidInput() {
        expectIae(() -> HadamardSpec.rotation(0));
        expectIae(() -> HadamardSpec.rotation(48));
        expectIae(() -> HadamardSpec.of(4, HadamardSpec.SignOrder.NONE, new float[] {1, 1, 1, 1}, 0, 0, 0));
        expectIae(() -> HadamardSpec.of(4, HadamardSpec.SignOrder.BEFORE_ROTATION, null, 0, 0, 0));
        expectIae(() -> HadamardSpec.of(4, HadamardSpec.SignOrder.BEFORE_ROTATION, new float[] {1, 1, 1}, 0, 0, 0));
        expectIae(() -> HadamardSpec.of(4, HadamardSpec.SignOrder.BEFORE_ROTATION, new float[] {1, 0, 1, 1}, 0, 0, 0));
        expectIae(() -> HadamardSpec.of(4, HadamardSpec.SignOrder.NONE, null, 3, 1, 2));
        expectIae(() -> HadamardSpec.rotation(64).checkWidth(96));
        expectIae(() -> HadamardSpec.of(4, HadamardSpec.SignOrder.AFTER_ROTATION, new float[8], 0, 0, 0));
    }

    @Test
    public void kernelContract() {
        HadamardSpec plain = HadamardSpec.rotation(64);
        assertEquals(0, plain.kernelSignMode());
        assertTrue(plain.kernelSupported());
        assertTrue(plain.inPlaceAllowed());
        assertNull(plain.signsName());
        assertFalse(HadamardSpec.rotation(2048).kernelSupported());
        HadamardSpec inv = HadamardSpec.of(2, HadamardSpec.SignOrder.AFTER_ROTATION, new float[] {1, -1, -1, 1}, 0, 0, 0);
        assertEquals(2, inv.kernelSignMode());
        assertEquals("prism.hadamard.signs.4", inv.signsName());
        assertEquals(11, HadamardSpec.ARG_Y);
    }

    // ---- PrismHadamardLoader -----------------------------------------------------------------------------

    private static Map<String, Object> prismMeta(String arch, String signMode) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", arch);
        m.put("prism.hadamard.version", 1);
        m.put("prism.hadamard.block_size", 64);
        m.put("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
        m.put("prism.hadamard.axis", "input-last-dimension");
        m.put("prism.hadamard.sign_mode", signMode);
        m.put("prism.hadamard.weight_names", new String[] {"blk.0.attn_q.weight", "blk.0.ffn_down.weight"});
        return m;
    }

    private static PrismHadamardLoader.TensorShapes shapes(Map<String, long[]> dims) {
        return dims::get;
    }

    private static Map<String, long[]> denseDims() {
        Map<String, long[]> d = new HashMap<>();
        d.put("token_embd.weight", new long[] {128, 1000});
        d.put("output.weight", new long[] {128, 1000});
        d.put("blk.0.attn_q.weight", new long[] {128, 128});
        d.put("blk.0.ffn_down.weight", new long[] {256, 128});
        return d;
    }

    @Test
    public void noPrismMetadataGivesEmptyResult() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "llama");
        PrismHadamardLoader.Result r = PrismHadamardLoader.load(GgufKv.ofMap(m), shapes(denseDims()), 0, 0);
        assertTrue(r.isEmpty());
    }

    @Test
    public void identitySignModeBindsPlainRotations() {
        Map<String, HadamardSpec> f = PrismHadamardLoader.parse(GgufKv.ofMap(prismMeta("qwen3", "identity")),
                shapes(denseDims()), 0, 0);
        assertEquals(2, f.size());
        assertEquals(HadamardSpec.rotation(64), f.get("blk.0.attn_q.weight"));
        assertEquals(HadamardSpec.rotation(64), f.get("blk.0.ffn_down.weight"));
    }

    @Test
    public void explicitSignsAreSelectedByInputWidth() {
        Map<String, Object> m = prismMeta("llama", "explicit");
        float[] s128 = randomSigns(128, 1);
        float[] s256 = randomSigns(256, 2);
        int[] values = new int[384];
        for (int i = 0; i < 128; i++) {
            values[i] = (int) s128[i];
        }
        for (int i = 0; i < 256; i++) {
            values[128 + i] = (int) s256[i];
        }
        m.put("prism.hadamard.sign_widths", new int[] {128, 256});
        m.put("prism.hadamard.sign_values", values);
        Map<String, HadamardSpec> f = PrismHadamardLoader.parse(GgufKv.ofMap(m), shapes(denseDims()), 0, 0);
        HadamardSpec q = f.get("blk.0.attn_q.weight");
        assertEquals(HadamardSpec.SignOrder.BEFORE_ROTATION, q.signOrder());
        assertArrayEquals(s128, q.signs(), 0.0f);
        assertArrayEquals(s256, f.get("blk.0.ffn_down.weight").signs(), 0.0f);
    }

    @Test
    public void gdnValueGroupingSetsSsmOutPermutation() {
        Map<String, Object> m = prismMeta("qwen3next", "identity");
        m.put("prism.hadamard.weight_names", new String[] {"blk.0.ssm_out.weight"});
        m.put("prism.hadamard.gdn_v_grouped", true);
        Map<String, long[]> d = denseDims();
        d.put("blk.0.ssm_out.weight", new long[] {4096, 2048});
        // n_v = ssm_dt_rank = 32 value heads, n_k = ssm_n_group = 16 key heads (llama.cpp src/llama-model.h:682-690)
        HadamardSpec s = PrismHadamardLoader.parse(GgufKv.ofMap(m), shapes(d), 32, 16).get("blk.0.ssm_out.weight");
        assertEquals(128, s.permHd());
        assertEquals(16, s.permNk());
        assertEquals(2, s.permRep());
        assertTrue(s.hasPermutation());
        expectFormat(() -> PrismHadamardLoader.parse(GgufKv.ofMap(m), shapes(d), 30, 16));
    }

    @Test
    public void tiedOutputBindsBothDirectionsOfTheTokenEmbedding() {
        Map<String, Object> m = prismMeta("qwen3", "identity");
        m.put("prism.hadamard.version", 2);
        m.put("prism.hadamard.tied_output", true);
        m.put("prism.hadamard.inverse_weight_names", new String[] {"token_embd.weight"});
        Map<String, long[]> d = denseDims();
        d.remove("output.weight");
        PrismHadamardLoader.Result r = PrismHadamardLoader.load(GgufKv.ofMap(m), shapes(d), 0, 0);
        assertTrue(r.tiedOutput());
        assertEquals(HadamardSpec.rotation(64), r.forward().get("token_embd.weight"));
        assertEquals(HadamardSpec.rotation(64), r.inverse().get("token_embd.weight"));

        expectFormat(() -> PrismHadamardLoader.load(GgufKv.ofMap(m), shapes(denseDims()), 0, 0));
    }

    @Test
    public void malformedMetadataIsRejected() {
        Map<String, long[]> d = denseDims();

        Map<String, Object> badVersion = prismMeta("llama", "identity");
        badVersion.put("prism.hadamard.version", 3);
        expectFormat(() -> PrismHadamardLoader.load(GgufKv.ofMap(badVersion), shapes(d), 0, 0));

        Map<String, Object> badBlock = prismMeta("llama", "identity");
        badBlock.put("prism.hadamard.block_size", 48);
        expectFormat(() -> PrismHadamardLoader.load(GgufKv.ofMap(badBlock), shapes(d), 0, 0));

        Map<String, Object> badTransform = prismMeta("llama", "identity");
        badTransform.put("prism.hadamard.transform", "dct");
        expectFormat(() -> PrismHadamardLoader.load(GgufKv.ofMap(badTransform), shapes(d), 0, 0));

        expectFormat(() -> PrismHadamardLoader.load(GgufKv.ofMap(prismMeta("gemma3", "identity")), shapes(d), 0, 0));

        Map<String, Object> badKind = prismMeta("llama", "identity");
        badKind.put("prism.hadamard.weight_names", new String[] {"blk.0.attn_norm.weight"});
        expectFormat(() -> PrismHadamardLoader.load(GgufKv.ofMap(badKind), shapes(d), 0, 0));

        Map<String, Object> missingWidth = prismMeta("llama", "explicit");
        missingWidth.put("prism.hadamard.sign_widths", new int[] {128});
        int[] ones = new int[128];
        Arrays.fill(ones, 1);
        missingWidth.put("prism.hadamard.sign_values", ones);
        expectFormat(() -> PrismHadamardLoader.load(GgufKv.ofMap(missingWidth), shapes(d), 0, 0));

        Map<String, Object> badSign = prismMeta("llama", "explicit");
        badSign.put("prism.hadamard.sign_widths", new int[] {64});
        int[] twos = new int[64];
        Arrays.fill(twos, 2);
        badSign.put("prism.hadamard.sign_values", twos);
        expectFormat(() -> PrismHadamardLoader.load(GgufKv.ofMap(badSign), shapes(d), 0, 0));

        Map<String, Object> missingTensor = prismMeta("llama", "identity");
        missingTensor.put("prism.hadamard.weight_names", new String[] {"blk.7.attn_q.weight"});
        expectFormat(() -> PrismHadamardLoader.load(GgufKv.ofMap(missingTensor), shapes(d), 0, 0));
    }

    @Test
    public void foldableWeightNames() {
        assertTrue(PrismHadamardLoader.isFoldableWeight("output.weight"));
        assertTrue(PrismHadamardLoader.isFoldableWeight("blk.12.ffn_gate_up_exps.weight"));
        assertFalse(PrismHadamardLoader.isFoldableWeight("blk..attn_q.weight"));
        assertFalse(PrismHadamardLoader.isFoldableWeight("blk.0.attn_q.bias"));
        assertFalse(PrismHadamardLoader.isFoldableWeight("token_embd.weight"));
    }

    private static void expectIae(Runnable r) {
        try {
            r.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    private static void expectFormat(Runnable r) {
        try {
            r.run();
            fail("expected GgufFormatException");
        } catch (GgufFormatException expected) {
            // expected
        }
    }
}
