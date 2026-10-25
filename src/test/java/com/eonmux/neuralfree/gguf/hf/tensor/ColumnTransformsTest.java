package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.bytes;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.seq;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.values;
import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.withSiblings;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ColumnTransforms}: column slices, concatenations, group permutations and strided de-interleaves against
 * hand-computed tensors, whole-block exactness on Q8_0, the F32 fallback for split blocks, and the act-order fold.
 */
public class ColumnTransformsTest {

    @Test
    public void sliceColsOf2x4x3() throws IOException {
        RowSource s = ColumnTransforms.sliceCols(seq(2, 4, 3), 1, 2);
        assertArrayEquals(new long[]{2, 4, 2}, s.shape());
        assertArrayEquals(new float[]{1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23}, values(s), 0f);
        assertTrue(s.rowLocal());
        assertThrows(IllegalArgumentException.class, () -> ColumnTransforms.sliceCols(seq(2, 3), 2, 2));
    }

    @Test
    public void concatColsJoinsInnerRuns() throws IOException {
        RowSource c = ColumnTransforms.concatCols(Arrays.asList(seq(2, 2), seq(2, 3)));
        assertArrayEquals(new long[]{2, 5}, c.shape());
        assertArrayEquals(new float[]{0, 1, 0, 1, 2, 2, 3, 3, 4, 5}, values(c), 0f);
        assertThrows(IllegalArgumentException.class,
                () -> ColumnTransforms.concatCols(Arrays.asList(seq(2, 2), seq(3, 2))));
    }

    @Test
    public void permuteColGroupsAndInterleave() throws IOException {
        RowSource g = ColumnTransforms.permuteColGroups(seq(2, 6), 2, new int[]{2, 0, 1});
        assertArrayEquals(new float[]{4, 5, 0, 1, 2, 3, 10, 11, 6, 7, 8, 9}, values(g), 0f);
        RowSource odd = ColumnTransforms.interleaveCols(seq(2, 6), 2, 1);
        assertArrayEquals(new float[]{1, 3, 5, 7, 9, 11}, values(odd), 0f);
        RowSource bias = ColumnTransforms.interleaveCols(seq(6), 2, 0);
        assertArrayEquals(new long[]{3}, bias.shape());
        assertArrayEquals(new float[]{0, 2, 4}, values(bias), 0f);
        assertArrayEquals(new float[]{3, 0, 0, 1, 7, 4, 4, 5},
                values(ElementTransforms.gatherInner(seq(2, 4), new int[]{3, 0, 0, 1})), 0f);
    }

    @Test
    public void alignedBlockMovesAreExactAndSplitBlocksFallBackToF32() throws IOException {
        RowSource q8 = ElementTransforms.convert(seq(2, 64), GGMLType.Q8_0);
        byte[] all = bytes(q8);
        RowSource hi = ColumnTransforms.sliceCols(q8, 32, 32);
        assertEquals(ElementFormat.of(GGMLType.Q8_0), hi.format());
        byte[] want = new byte[68];
        System.arraycopy(all, 34, want, 0, 34);
        System.arraycopy(all, 102, want, 34, 34);
        assertArrayEquals(want, bytes(hi));

        RowSource swapped = ColumnTransforms.permuteColGroups(q8, 32, new int[]{1, 0});
        assertEquals(ElementFormat.of(GGMLType.Q8_0), swapped.format());

        RowSource split = ColumnTransforms.sliceCols(q8, 16, 32);
        assertEquals(ElementFormat.of(GGMLType.F32), split.format());
        float[] full = values(q8);
        float[] got = values(split);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 32; c++) {
                assertEquals(full[r * 64 + 16 + c], got[r * 32 + c], 0f);
            }
        }
    }

    @Test
    public void columnMovesFoldTheActOrderPermutationFirst() throws IOException {
        int[] perm = {2, 0, 1};
        RowSource w = withSiblings(seq(2, 3), new float[]{2f}, null, null, perm, ActQuantScheme.FP8_TOKEN);
        RowSource folded = ColumnTransforms.foldPermGeneric(w);
        assertNull(folded.inputPerm());
        assertArrayEquals(new float[]{1, 2, 0, 4, 5, 3}, values(folded), 0f);
        assertArrayEquals(new float[]{2f}, folded.outputScale(), 0f);
        assertEquals(ActQuantScheme.FP8_TOKEN, folded.actQuant());

        RowSource s = ColumnTransforms.sliceCols(w, 0, 2);
        assertNull(s.inputPerm());
        assertArrayEquals(new float[]{1, 2, 4, 5}, values(s), 0f);
        assertThrows(IllegalArgumentException.class, () -> ColumnTransforms.foldPermGeneric(
                withSiblings(seq(1, 3), null, null, null, new int[]{0, 0, 1}, null)));
    }
}
