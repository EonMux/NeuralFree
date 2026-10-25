package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ops.moe.CpuExpertMatmul;
import com.eonmux.neuralfree.gguf.quant.ActFakeQuant;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests of the CPU input-transform application (IT-1): a matvec through a weight's {@link InputTransform} equals the
 * matvec on the permuted weight side (PERMUTE), the ACT_QUANT fake-quantization is applied exactly once per token row
 * on this call's scratch, and the static per-expert schemes fake-quantize each gathered row with its expert's
 * {@code .input_scale}.
 *
 * <p><b>Unchanged path trace.</b> A GGUF without the transform tensors and keys resolves the same
 * {@code TransformSource} lookup as before, the weight's {@link InputTransform} is {@link InputTransform#IDENTITY},
 * {@link CpuMatVec.Transforms#from(InputTransform, float[], int)} reduces it to {@link CpuMatVec.Transforms#NONE},
 * and both {@link CpuMatVec#prepare} and {@link CpuExpertMatmul} run the identity/HADAMARD-only fast branch, the code
 * that served the S3/S4 PRISM-only builds unchanged: no scratch, the same HADAMARD rotation and quantization
 * stanzas.</p>
 *
 * <p><b>Once per row.</b> A fake-quantization is idempotent (re-quantizing a fake-quantized row reproduces its codes,
 * because the run's {@code amax} maps to itself), so "exactly once" is pinned through equality with the
 * single-application reference and the unchanged caller's input; the order test below also fails for an
 * implementation that quantizes runs of the gathered row instead.</p>
 */
public class CpuInputTransformTest {

    /** Builds an F32 {@link CpuTensorRef} of {@code w}'s rows. */
    private static CpuTensorRef f32(float[][] w) {
        ByteBuffer data = ByteBuffer.allocate(w.length * w[0].length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] row : w) {
            for (float v : row) {
                data.putFloat(v);
            }
        }
        return CpuTensorRef.of(GGMLType.F32, data, w.length, w[0].length);
    }

    private static float[][] matrix(int rows, int cols, Random rnd) {
        float[][] w = new float[rows][cols];
        for (float[] row : w) {
            for (int k = 0; k < cols; k++) {
                row[k] = rnd.nextFloat() * 2.0f - 1.0f;
            }
        }
        return w;
    }

    private static int[] permutation(int n, Random rnd) {
        int[] p = new int[n];
        for (int k = 0; k < n; k++) {
            p[k] = k;
        }
        for (int k = n - 1; k > 0; k--) {
            int j = rnd.nextInt(k + 1);
            int t = p[k];
            p[k] = p[j];
            p[j] = t;
        }
        return p;
    }

    /** {@code inverse[perm[k]] == k} for a permutation {@code perm}. */
    private static int[] inversePermutation(int[] perm) {
        int[] inverse = new int[perm.length];
        for (int k = 0; k < perm.length; k++) {
            inverse[perm[k]] = k;
        }
        return inverse;
    }

    /** A row of non-integers of mixed sign, so every scheme's codes are exercised. */
    private static float[] row(int cols, Random rnd) {
        float[] x = new float[cols];
        for (int k = 0; k < cols; k++) {
            x[k] = rnd.nextFloat() * 4.0f - 2.0f;
        }
        return x;
    }

    /** Asserts {@code actual} equals the dot of {@code w} row {@code wOff} with {@code x} at {@code xOff}. */
    private static void assertDot(float actual, float[] w, int wOff, float[] x, int xOff, int cols) {
        float expected = 0.0f;
        float scale = 0.0f;
        for (int k = 0; k < cols; k++) {
            float term = w[wOff + k] * x[xOff + k];
            expected += term;
            scale += Math.abs(term);
        }
        assertEquals(expected, actual, scale * 1e-5f + 1e-5f);
    }

    // ------------------------------------------------------------------ PERMUTE

    /**
     * The transform's gather turns {a weight stored in permuted input order} into a matvec of the same weight with
     * the gathered rows, bit for bit, and equals the matvec of the column-permuted weight matrix with the original
     * rows (the permuted-weight formulation, equal up to the F32 dot's summation order).
     */
    @Test
    public void permutedInputMatchesThePermutedMatrix() {
        Random rnd = new Random(20240517L);
        int rows = 6;
        int cols = 10;
        float[][] w = matrix(rows, cols, rnd);
        CpuTensorRef weight = f32(w);
        int[] perm = permutation(cols, rnd);
        CpuMatVec mv = new CpuMatVec(ref -> CpuMatVec.Transforms.from(
                InputTransform.of(null, null, perm), null, 0));

        float[] x = row(cols, rnd);
        float[] y = new float[rows];
        mv.matvec(weight, x, 0, y, 0);

        // The same weight over the pre-gathered input: the term order matches, so the result is bit-identical.
        float[] xg = new float[2 * cols];
        for (int k = 0; k < cols; k++) {
            xg[k] = x[perm[k]];
        }
        float[] gathered = new float[rows];
        CpuMatVec.PLAIN.matvec(weight, xg, 0, gathered, 0);
        assertArrayEquals(gathered, y, 0.0f);

        // The permuted-weight formulation: y' = sum_j W[r][perm^-1(j)] * x[j].
        int[] inv = inversePermutation(perm);
        float[][] permutedW = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < cols; j++) {
                permutedW[r][j] = w[r][inv[j]];
            }
        }
        float[] pc = new float[rows];
        CpuMatVec.PLAIN.matvec(f32(permutedW), x, 0, pc, 0);
        for (int r = 0; r < rows; r++) {
            assertDot(y[r], w[r], 0, xg, 0, cols);
            assertDot(pc[r], w[r], 0, xg, 0, cols);
        }

        // Two strided token rows: the gather reads the strided rows and packs the gathered rows.
        float[] xs = new float[2 * (cols + 4)];
        for (int t = 0; t < 2; t++) {
            float[] tok = row(cols, rnd);
            System.arraycopy(tok, 0, xs, t * (cols + 4), cols);
        }
        float[] ys = new float[2 * rows];
        mv.matmul(weight, xs, 0, cols + 4, ys, 0, rows, 2);
        for (int t = 0; t < 2; t++) {
            for (int k = 0; k < cols; k++) {
                xg[t * cols + k] = xs[t * (cols + 4) + perm[k]];
            }
        }
        float[] gatheredStrided = new float[2 * rows];
        CpuMatVec.PLAIN.matmul(weight, xg, 0, cols, gatheredStrided, 0, rows, 2);
        assertArrayEquals(gatheredStrided, ys, 0.0f);
    }

    /** A 3-D expert tensor rejects the PERMUTE part: TP-1 folds it into the weights. */
    @Test
    public void permuteNeverReachesTheThreeDimensionalExperts() {
        Random rnd = new Random(3L);
        int experts = 2;
        int rowsPerExpert = 3;
        int cols = 8;
        float[][] w3 = matrix(experts * rowsPerExpert, cols, rnd);
        CpuTensorRef w = f32(w3);
        CpuMatVec mv = new CpuMatVec(ref -> ref == w ? CpuMatVec.Transforms.from(
                InputTransform.of(null, null, permutation(cols, rnd)), null, experts) : CpuMatVec.Transforms.NONE);
        CpuExpertMatmul step = new CpuExpertMatmul(mv);
        CpuExpertMatmul.Groups g = CpuExpertMatmul.Groups.of(new int[]{0, 1}, 2, experts);
        float[] x = new float[g.pairs() * cols];
        float[] y = new float[g.pairs() * rowsPerExpert];
        try {
            step.run(w, experts, g, x, 0, 1, y, 0);
            fail();
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("TP-1"));
        }
    }

    // ------------------------------------------------------------------ ACT_QUANT

    /** Each token row is fake-quantized once, on the call's scratch, before the dots. */
    @Test
    public void actQuantAppliesExactlyOncePerTokenRow() {
        Random rnd = new Random(815L);
        int rows = 3;
        int cols = 128;
        float[][] w = matrix(rows, cols, rnd);
        CpuTensorRef weight = f32(w);
        InputTransform it = InputTransform.of(null, InputTransform.ActQuant.of(
                ActQuantScheme.INT8_GROUP128, null, null, 0.0f), null);
        CpuMatVec mv = new CpuMatVec(ref -> CpuMatVec.Transforms.from(it, null, 0));

        int tokens = 2;
        float[] x = new float[tokens * cols];
        for (int k = 0; k < cols; k++) {
            x[k] = rnd.nextFloat() * 4.0f - 2.0f;
            x[cols + k] = (rnd.nextFloat() * 4.0f - 2.0f) * 1e-11f; // runs under the 1e-10 group floor
        }
        float[] original = x.clone();
        float[] y = new float[tokens * rows];
        mv.matmul(weight, x, 0, y, 0, tokens);
        assertArrayEquals(original, x, 0.0f);

        float[] transformed = new float[cols];
        for (int t = 0; t < tokens; t++) {
            System.arraycopy(x, t * cols, transformed, 0, cols);
            ActFakeQuant.apply(ActQuantScheme.INT8_GROUP128, 0.0f, 0, 0.0f, transformed, 0, 1, cols);
            for (int r = 0; r < rows; r++) {
                assertDot(y[t * rows + r], w[r], 0, transformed, 0, cols);
            }
        }
    }

    // ------------------------------------------------------------------ expert handling

    /** Static per-expert schemes fake-quantize each gathered row with the pair's expert's {@code .input_scale}. */
    @Test
    public void perExpertStaticScalesFakeQuantizeEachGatheredRow() {
        Random rnd = new Random(91L);
        int experts = 2;
        int rows = 3;
        int cols = 8;
        float[][] w3 = matrix(experts * rows, cols, rnd);
        CpuTensorRef w = f32(w3);
        float[] scales = {0.5f, 0.25f};
        InputTransform it = InputTransform.of(null, InputTransform.ActQuant.of(
                ActQuantScheme.INT8_STATIC, scales, null, 0.0f), null);
        CpuMatVec mv = new CpuMatVec(ref -> ref == w ? CpuMatVec.Transforms.from(it, null, experts)
                : CpuMatVec.Transforms.NONE);
        CpuExpertMatmul step = new CpuExpertMatmul(mv);

        int[] ids = {0, 1, 1, 0};
        CpuExpertMatmul.Groups g = CpuExpertMatmul.Groups.of(ids, ids.length, experts);
        float[] x = new float[g.pairs() * cols];
        for (int p = 0; p < g.pairs(); p++) {
            float[] tok = row(cols, rnd);
            System.arraycopy(tok, 0, x, p * cols, cols);
        }
        float[] y = new float[g.pairs() * rows];
        step.run(w, experts, g, x, 0, 1, y, 0);

        float[] transformed = new float[cols];
        for (int p = 0; p < g.pairs(); p++) {
            int e = ids[p];
            System.arraycopy(x, p * cols, transformed, 0, cols);
            ActFakeQuant.apply(ActQuantScheme.INT8_STATIC, scales[e], 0, 0.0f, transformed, 0, 1, cols);
            for (int r = 0; r < rows; r++) {
                assertDot(y[p * rows + r], w3[e * rows + r], 0, transformed, 0, cols);
            }
        }
    }

    /** A single {@link CpuMatVec#matvecExpert} call fake-quantizes with the requested expert's scale only. */
    @Test
    public void matvecExpertAppliesTheRequestedStaticScale() {
        Random rnd = new Random(72L);
        int experts = 2;
        int rows = 3;
        int cols = 8;
        float[][] w3 = matrix(experts * rows, cols, rnd);
        CpuTensorRef w = f32(w3);
        float[] scales = {0.5f, 0.25f};
        InputTransform it = InputTransform.of(null, InputTransform.ActQuant.of(
                ActQuantScheme.INT8_STATIC, scales, null, 0.0f), null);
        CpuMatVec mv = new CpuMatVec(ref -> ref == w ? CpuMatVec.Transforms.from(it, null, experts)
                : CpuMatVec.Transforms.NONE);

        float[] x = row(cols, rnd);
        float[] y = new float[rows];
        mv.matvecExpert(w, 1, experts, x, 0, y, 0);
        float[] transformed = x.clone();
        ActFakeQuant.apply(ActQuantScheme.INT8_STATIC, scales[1], 0, 0.0f, transformed, 0, 1, cols);
        for (int r = 0; r < rows; r++) {
            assertDot(y[r], w3[rows + r], 0, transformed, 0, cols);
        }

        // A row block spanning several experts has no single .input_scale: the call is refused.
        try {
            mv.matvecRows(w, 0, experts * rows, x, 0, new float[experts * rows], 0);
            fail();
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("per expert"));
        }
    }

    // ------------------------------------------------------------------ order

    /**
     * One token, all three parts: the result is the dot of the weight with the row that is first rotated, then
     * fake-quantized, then gathered (quantizing runs of the gathered row instead would split its groups differently).
     */
    @Test
    public void hadamardThenActQuantThenPermute() {
        Random rnd = new Random(5150L);
        int rows = 4;
        int cols = 128;
        HadamardSpec rotation = HadamardSpec.rotation(8);
        float[][] w = matrix(rows, cols, rnd);
        CpuTensorRef weight = f32(w);
        int[] perm = permutation(cols, rnd);
        InputTransform it = InputTransform.of(rotation, InputTransform.ActQuant.of(
                ActQuantScheme.INT8_GROUP128, null, null, 0.0f), perm);
        CpuMatVec mv = new CpuMatVec(ref -> CpuMatVec.Transforms.from(it, null, 0));

        float[] x = row(cols, rnd);
        float[] y = new float[rows];
        mv.matvec(weight, x, 0, y, 0);

        float[] rotated = new float[cols];
        CpuHadamard.apply(rotation, x, 0, rotated, 0, 1, cols);
        ActFakeQuant.apply(ActQuantScheme.INT8_GROUP128, 0.0f, 0, 0.0f, rotated, 0, 1, cols);
        float[] gathered = new float[cols];
        for (int k = 0; k < cols; k++) {
            gathered[k] = rotated[perm[k]];
        }
        for (int r = 0; r < rows; r++) {
            assertDot(y[r], w[r], 0, gathered, 0, cols);
        }
    }

    // ------------------------------------------------------------------ unchanged path

    /** A model without transform tensors and keys binds the identity, which reduces to {@code Transforms.NONE}. */
    @Test
    public void identityTransformsBindAsNone() {
        assertSame(CpuMatVec.Transforms.NONE, CpuMatVec.Transforms.of((HadamardSpec) null, null, 0));
        assertSame(CpuMatVec.Transforms.NONE, CpuMatVec.Transforms.from(InputTransform.IDENTITY, null, 0));
        assertSame(CpuMatVec.Transforms.NONE.transform(), InputTransform.IDENTITY);
        assertTrue(CpuMatVec.Transforms.of((HadamardSpec) null, new float[]{2.0f}, 0).hasScale());
        assertTrue(CpuMatVec.Transforms.of((HadamardSpec) null, null, 2).transform().isIdentity());
    }
}
