package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link RowTransforms}: every dim-0 move against hand-computed tensors, the Q/K permutations against llama.py's
 * reshape/swapaxes formula, whole-row exactness on Q4_0, and the sibling and direct-offset rules. The static helpers
 * are shared by the other transform tests of this package.
 */
public class RowTransformsTest {

    // ------------------------------------------------------------------ shared helpers

    /** An F32 tensor holding {@code 0, 1, 2, ...}. */
    static RowSource seq(long... shape) {
        int n = 1;
        for (long d : shape) {
            n *= (int) d;
        }
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = i;
        }
        return RopeTensors.fromFloats(shape, v);
    }

    /** The values of {@code s} decoded to floats, row-major. */
    static float[] values(RowSource s) throws IOException {
        long[] shape = s.shape();
        int inner = (int) (shape.length == 0 ? 1 : shape[shape.length - 1]);
        ByteBuffer b = RowTransforms.readRows(s, 0, (int) s.rows());
        float[] out = new float[(int) s.rows() * inner];
        for (int r = 0; r < s.rows(); r++) {
            DtypePolicy.decode(s.format(), b, (int) (r * s.rowBytes()), out, r * inner, inner);
        }
        return out;
    }

    /** All bytes of {@code s}. */
    static byte[] bytes(RowSource s) throws IOException {
        ByteBuffer b = RowTransforms.readRows(s, 0, (int) s.rows());
        byte[] out = new byte[b.remaining()];
        b.get(out);
        return out;
    }

    /** Q4_0 rows of 32 elements, one block each: row {@code r} has {@code d = r + 1} and codes {@code 16 r + j}. */
    static RowSource q4Rows(int rows) {
        byte[] data = new byte[rows * 18];
        for (int r = 0; r < rows; r++) {
            int h = FpBlockDecoders.fp32ToFp16(r + 1f);
            data[r * 18] = (byte) h;
            data[r * 18 + 1] = (byte) (h >>> 8);
            for (int j = 0; j < 16; j++) {
                data[r * 18 + 2 + j] = (byte) (16 * r + j);
            }
        }
        return RopeTensors.fromBytes(ElementFormat.of(GGMLType.Q4_0), new long[]{rows, 32}, data);
    }

    /** The bytes of row {@code r} of a source with rows of {@code rowBytes}. */
    static byte[] row(byte[] all, int r, int rowBytes) {
        return Arrays.copyOfRange(all, r * rowBytes, (r + 1) * rowBytes);
    }

    /** {@code s} as if stored unchanged at virtual offset {@code offset}. */
    static RowSource direct(RowSource s, long offset) {
        return new Wrapped(s, offset, null, null, null, null, null, null);
    }

    /** {@code s} carrying the given sibling values. */
    static RowSource withSiblings(RowSource s, float[] outputScale, float[] inputScale, int[] zeroPoint, int[] perm,
                                  ActQuantScheme scheme) {
        return withSiblings(s, outputScale, inputScale, zeroPoint, perm, scheme, null);
    }

    static RowSource withSiblings(RowSource s, float[] outputScale, float[] inputScale, int[] zeroPoint, int[] perm,
                                  ActQuantScheme scheme, float[] inputPreScale) {
        return new Wrapped(s, -1, outputScale, inputScale, zeroPoint, perm, scheme, inputPreScale);
    }

    private static final class Wrapped implements RowSource {
        private final RowSource s;
        private final long direct;
        private final float[] os;
        private final float[] is;
        private final int[] zp;
        private final int[] perm;
        private final ActQuantScheme scheme;
        private final float[] pre;

        Wrapped(RowSource s, long direct, float[] os, float[] is, int[] zp, int[] perm, ActQuantScheme scheme,
                float[] pre) {
            this.s = s;
            this.direct = direct;
            this.os = os;
            this.is = is;
            this.zp = zp;
            this.perm = perm;
            this.scheme = scheme;
            this.pre = pre;
        }

        @Override
        public String describe() {
            return "test " + s.describe();
        }

        @Override
        public ElementFormat format() {
            return s.format();
        }

        @Override
        public long[] shape() {
            return s.shape();
        }

        @Override
        public long rows() {
            return s.rows();
        }

        @Override
        public long rowBytes() {
            return s.rowBytes();
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            s.readRows(first, count, dst);
        }

        @Override
        public long directVirtualOffset() {
            return direct;
        }

        @Override
        public float[] outputScale() {
            return os == null ? null : os.clone();
        }

        @Override
        public float[] inputScale() {
            return is == null ? null : is.clone();
        }

        @Override
        public int[] inputZeroPoint() {
            return zp == null ? null : zp.clone();
        }

        @Override
        public int[] inputPerm() {
            return perm == null ? null : perm.clone();
        }

        @Override
        public ActQuantScheme actQuant() {
            return scheme;
        }

        @Override
        public float[] inputPreScale() {
            return pre == null ? null : pre.clone();
        }
    }

    /** {@code w.reshape(n, 2, d0 / n / 2, cols).swapaxes(1, 2).reshape(d0, cols)} written out with explicit loops. */
    private static float[] llamaPermute(float[] w, int d0, int cols, int n) {
        int half = d0 / n / 2;
        float[] out = new float[w.length];
        for (int h = 0; h < n; h++) {
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < half; b++) {
                    int in = (h * 2 + a) * half + b;
                    int o = (h * half + b) * 2 + a;
                    System.arraycopy(w, in * cols, out, o * cols, cols);
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ Q/K permutations

    @Test
    public void permuteQkMatchesLlamaPyForWeightAndBias() throws IOException {
        RowSource w = seq(8, 3);
        assertArrayEquals(llamaPermute(values(w), 8, 3, 2), values(RowTransforms.permuteQk(w, 2, 2)), 0f);
        assertArrayEquals(llamaPermute(values(w), 8, 3, 2), values(RowTransforms.permuteQk(w, 4, 2)), 0f);
        assertArrayEquals(llamaPermute(values(w), 8, 3, 4), values(RowTransforms.permuteQk(w, 4, 4)), 0f);
        RowSource bias = seq(8);
        RowSource pb = RowTransforms.permuteQk(bias, 2, 2);
        assertArrayEquals(new long[]{8}, pb.shape());
        assertArrayEquals(llamaPermute(values(bias), 8, 1, 2), values(pb), 0f);
        assertArrayEquals(new float[]{0, 2, 1, 3, 4, 6, 5, 7}, values(pb), 0f);
        assertTrue(pb.rowLocal());
    }

    @Test
    public void permuteQkMovesWholeQ4Rows() throws IOException {
        RowSource q = q4Rows(4);
        RowSource p = RowTransforms.permuteQk(q, 1, 1);
        assertEquals(ElementFormat.of(GGMLType.Q4_0), p.format());
        byte[] in = bytes(q);
        byte[] out = bytes(p);
        int[] order = {0, 2, 1, 3};
        for (int r = 0; r < 4; r++) {
            assertArrayEquals(row(in, order[r], 18), row(out, r, 18));
        }
    }

    @Test
    public void reversePermuteUsesHeadRatio() throws IOException {
        RowSource w = seq(8, 2);
        assertArrayEquals(llamaPermute(values(w), 8, 2, 2), values(RowTransforms.reversePermute(w, 4, 2)), 0f);
        assertArrayEquals(llamaPermute(values(w), 8, 2, 4), values(RowTransforms.reversePermute(w, 4, 4)), 0f);
        assertArrayEquals(llamaPermute(values(w), 8, 2, 2), values(RowTransforms.rotateHalfUnpermute(w, 2)), 0f);
    }

    @Test
    public void storedRowsTraceThroughPermutesAndRowGroupReorders() throws IOException {
        RowSource w = seq(8, 2);
        assertSame(w, RowTransforms.storedSource(w));
        assertEquals(5, RowTransforms.storedRow(w, 5));
        RowSource p = RowTransforms.permuteQk(w, 2, 2);
        RowSource g = RowTransforms.permuteRowGroups(p, 2, 2, new int[]{2, 0, 1});
        assertArrayEquals(new float[]{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11},
                values(RowTransforms.permuteRowGroups(w, 2, 2, new int[]{2, 0, 1})), 0f);
        float[] stored = values(w);
        for (RowSource s : new RowSource[]{p, g}) {
            assertSame(s.describe(), w, RowTransforms.storedSource(s));
            float[] got = values(s);
            for (int r = 0; r < 8; r++) {
                long at = RowTransforms.storedRow(s, r);
                assertArrayEquals(s.describe() + " row " + r,
                        new float[]{stored[(int) (2 * at)], stored[(int) (2 * at + 1)]},
                        new float[]{got[2 * r], got[2 * r + 1]}, 0f);
            }
        }
        assertEquals("the pair swap's row 1 is stored row 2", 2, RowTransforms.storedRow(p, 1));
        assertEquals(RowTransforms.storedRow(p, 6), RowTransforms.storedRow(g, 2));
    }

    @Test
    public void untracedReordersAreTheirOwnStoredSource() {
        RowSource w = seq(12);
        RowSource neox = RowTransforms.normalToNeoxPartial(w, 2, 6, 4);
        assertSame(neox, RowTransforms.storedSource(neox));
        assertEquals(1, RowTransforms.storedRow(neox, 1));
        RowSource gathered = RowTransforms.gatherRows(seq(4, 2), 1, new int[]{3, 1});
        assertSame(gathered, RowTransforms.storedSource(gathered));
    }

    @Test
    public void permuteRowGroupsRefusesNonPermutationsAndPartialUnits() {
        RowSource w = seq(8, 2);
        assertThrows(IllegalArgumentException.class,
                () -> RowTransforms.permuteRowGroups(w, 2, 2, new int[]{0, 0, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> RowTransforms.permuteRowGroups(w, 2, 2, new int[]{0, 3, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> RowTransforms.permuteRowGroups(w, 0, 2, new int[]{1, 0, 2}));
        assertThrows(IllegalArgumentException.class,
                () -> RowTransforms.permuteRowGroups(w, 2, 0, new int[]{}));
    }

    @Test
    public void normalToNeoxPartialReordersOnlyTheRotaryPart() throws IOException {
        RowSource w = seq(12);
        float[] got = values(RowTransforms.normalToNeoxPartial(w, 2, 6, 4));
        assertArrayEquals(new float[]{0, 2, 1, 3, 4, 5, 6, 8, 7, 9, 10, 11}, got, 0f);
        assertThrows(IllegalArgumentException.class, () -> RowTransforms.normalToNeoxPartial(w, 2, 6, 3));
    }

    @Test
    public void negateSecondHalfRowsIsExactOnQ8() throws IOException {
        assertArrayEquals(new float[]{0, 1, -2, -3, 4, 5, -6, -7},
                values(RowTransforms.negateSecondHalfRows(seq(4, 2), 2)), 0f);
        RowSource q8 = ElementTransforms.convert(seq(2, 32), GGMLType.Q8_0);
        float[] in = values(q8);
        float[] out = values(RowTransforms.negateSecondHalfRows(q8, 2));
        for (int i = 0; i < in.length; i++) {
            assertEquals(i < 32 ? in[i] : -in[i], out[i], 0f);
        }
    }

    // ------------------------------------------------------------------ slices and splits

    @Test
    public void slicesChunksSplitsAndStridesOf2x4x3() throws IOException {
        RowSource x = seq(2, 4, 3);
        RowSource s = RowTransforms.sliceRows(x, 1, 1);
        assertArrayEquals(new long[]{1, 4, 3}, s.shape());
        assertArrayEquals(Arrays.copyOfRange(values(x), 12, 24), values(s), 0f);
        RowSource mid = RowTransforms.sliceDim(x, 1, 1, 2);
        assertArrayEquals(new long[]{2, 2, 3}, mid.shape());
        assertArrayEquals(new float[]{3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20}, values(mid), 0f);
        RowSource cols = RowTransforms.sliceDim(x, -1, 2, 1);
        assertArrayEquals(new float[]{2, 5, 8, 11, 14, 17, 20, 23}, values(cols), 0f);

        RowSource m = seq(5, 2);
        assertArrayEquals(new float[]{6, 7, 8, 9}, values(RowTransforms.chunkRows(m, 2, 1)), 0f);
        RowSource[] parts = RowTransforms.splitRows(m, 1, 4);
        assertArrayEquals(new float[]{0, 1}, values(parts[0]), 0f);
        assertArrayEquals(new long[]{4, 2}, parts[1].shape());
        assertThrows(IllegalArgumentException.class, () -> RowTransforms.splitRows(m, 1, 3));
        assertArrayEquals(new float[]{2, 3, 6, 7}, values(RowTransforms.interleaveRows(m, 2, 1)), 0f);
        assertArrayEquals(new float[]{0, 1, 4, 5, 8, 9}, values(RowTransforms.interleaveRows(m, 2, 0)), 0f);
        assertArrayEquals(new float[]{4, 5, 6, 7, 0, 1, 2, 3},
                values(RowTransforms.gatherRows(seq(4, 2), 2, new int[]{1, 0})), 0f);
        assertArrayEquals(new float[]{0, 1, 2, 3}, values(RowTransforms.truncateRows(m, 2)), 0f);
    }

    @Test
    public void contiguousSliceOfStoredTensorStaysMapped() {
        RowSource stored = direct(seq(4, 3), 1000);
        assertEquals(1000 + 2 * 12, RowTransforms.sliceRows(stored, 2, 2).directVirtualOffset());
        assertEquals(-1, RowTransforms.interleaveRows(stored, 2, 0).directVirtualOffset());
        assertEquals(1000 + 8, RowTransforms.sliceRows(direct(seq(6), 1000), 2, 3).directVirtualOffset());
    }

    // ------------------------------------------------------------------ concatenation and padding

    @Test
    public void concatRowsJoinsPartsAndChecksFormats() throws IOException {
        RowSource c = RowTransforms.concatRows(Arrays.asList(seq(1, 3), seq(2, 3)));
        assertArrayEquals(new long[]{3, 3}, c.shape());
        assertArrayEquals(new float[]{0, 1, 2, 0, 1, 2, 3, 4, 5}, values(c), 0f);
        assertArrayEquals(new float[]{0, 1, 0, 1, 2},
                values(RowTransforms.concatRows(Arrays.asList(seq(2), seq(3)))), 0f);
        assertThrows(IllegalArgumentException.class,
                () -> RowTransforms.concatRows(Arrays.asList(seq(1, 32), q4Rows(1))));
        assertThrows(IllegalArgumentException.class,
                () -> RowTransforms.concatRows(Arrays.asList(seq(1, 3), seq(1, 4))));
    }

    @Test
    public void concatRowsMergesFusedActivationScalesByMaximum() {
        RowSource a = withSiblings(seq(1, 4), null, new float[]{0.5f}, null, null, ActQuantScheme.FP8_STATIC);
        RowSource b = withSiblings(seq(2, 4), null, new float[]{0.75f}, null, null, ActQuantScheme.FP8_STATIC);
        RowSource c = RowTransforms.concatRows(Arrays.asList(a, b));
        assertArrayEquals(new float[]{0.75f}, c.inputScale(), 0f);
        assertEquals(ActQuantScheme.FP8_STATIC, c.actQuant());
        RowSource d = RowTransforms.concatRows(Arrays.asList(a, seq(1, 4)));
        assertNull(d.actQuant());
        assertNull(d.inputScale());
    }

    @Test
    public void padRowsFillsNewEntries() throws IOException {
        RowSource p = RowTransforms.padRows(seq(2, 2), 3, 1.5f);
        assertArrayEquals(new float[]{0, 1, 2, 3, 1.5f, 1.5f}, values(p), 0f);
        assertArrayEquals(new float[]{0, 1, 2, -1, -1}, values(RowTransforms.padRows(seq(3), 5, -1f)), 0f);
        RowSource q = RowTransforms.padRows(q4Rows(2), 3, 0f);
        assertEquals(ElementFormat.of(GGMLType.Q4_0), q.format());
        byte[] out = bytes(q);
        assertArrayEquals(new byte[18], row(out, 2, 18));
        assertArrayEquals(row(bytes(q4Rows(2)), 1, 18), row(out, 1, 18));
        RowSource qf = RowTransforms.padRows(q4Rows(1), 2, 2f);
        assertEquals(ElementFormat.of(GGMLType.F32), qf.format());
        assertEquals(2f, values(qf)[32], 0f);
        assertThrows(IllegalArgumentException.class, () -> RowTransforms.padRows(seq(3, 1), 2, 0f));
    }

    @Test
    public void rowMovesKeepSiblingsAndReindexExpertScales() {
        RowSource w = withSiblings(seq(4, 2), new float[]{2f}, new float[]{0.5f}, null, null,
                ActQuantScheme.FP8_STATIC);
        RowSource p = RowTransforms.permuteQk(w, 1, 1);
        assertArrayEquals(new float[]{2f}, p.outputScale(), 0f);
        assertArrayEquals(new float[]{0.5f}, p.inputScale(), 0f);
        assertEquals(ActQuantScheme.FP8_STATIC, p.actQuant());
        RowSource experts = withSiblings(seq(3, 2, 2), new float[]{1f, 2f, 3f}, null, null, null, null);
        assertArrayEquals(new float[]{3f, 1f},
                RowTransforms.gatherRows(experts, 1, new int[]{2, 0}).outputScale(), 0f);
    }
}
