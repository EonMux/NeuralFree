package com.eonmux.neuralfree.gguf.mm;

import org.junit.Test;

import java.util.List;
import java.util.Random;
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * The host tables the GPU executor reorders by, without a GPU: {@link PixelShuffle#chunkSources} gathers of
 * {@code n_embd} chunks (a {@code -1} entry reads a zero row) and the {@link PixelShuffle#channelMajor} column
 * permutation reproduce every pixel-shuffle transcription at {@code n_embd > 1}, padding included; and
 * {@link VisionRows#keyRuns} groups the query rows of a windowed attention into runs of one key range.
 */
public class GpuReorderTablesTest {

    private static final int N_EMBD = 3;

    private static float[] rows(int nRows, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[nRows * N_EMBD];
        for (int i = 0; i < v.length; i++) {
            v[i] = rnd.nextFloat() - 0.5f;
        }
        return v;
    }

    /** The chunks of {@code rows} gathered as a device executor gathers them by the table. */
    private static float[] gather(float[] rows, int[] src) {
        float[] out = new float[src.length * N_EMBD];
        for (int i = 0; i < src.length; i++) {
            if (src[i] >= 0) {
                System.arraycopy(rows, src[i] * N_EMBD, out, i * N_EMBD, N_EMBD);
            }
        }
        return out;
    }

    private static void assertGatherMatches(int nRows, UnaryOperator<float[]> atOne, UnaryOperator<float[]> wide) {
        float[] in = rows(nRows, 7L * nRows);
        assertArrayEquals(wide.apply(in), gather(in, PixelShuffle.chunkSources(nRows, atOne)), 0f);
    }

    @Test
    public void chunkTablesReproduceTheShufflesAtAWideEmbedding() {
        assertGatherMatches(4 * 6, ids -> PixelShuffle.internvl(ids, 1, 4, 6, 2),
                r -> PixelShuffle.internvl(r, N_EMBD, 4, 6, 2));
        assertGatherMatches(6 * 4, ids -> PixelShuffle.llama4(ids, 1, 6, 4, 2),
                r -> PixelShuffle.llama4(r, N_EMBD, 6, 4, 2));
        assertGatherMatches(6 * 6, ids -> PixelShuffle.mergePermute(ids, 1, 6, 6, 3),
                r -> PixelShuffle.mergePermute(r, N_EMBD, 6, 6, 3));
    }

    @Test
    public void chunkTablesMarkTheZeroPaddingOfAnUnalignedGrid() {
        // a 5x3 grid padded to 6x4 by the merge permute
        int[] src = PixelShuffle.chunkSources(5 * 3, ids -> PixelShuffle.mergePermute(ids, 1, 5, 3, 2));
        assertEquals(6 * 4, src.length);
        int pads = 0;
        for (int i : src) {
            pads += i < 0 ? 1 : 0;
        }
        assertEquals(6 * 4 - 5 * 3, pads);
        assertGatherMatches(5 * 3, ids -> PixelShuffle.mergePermute(ids, 1, 5, 3, 2),
                r -> PixelShuffle.mergePermute(r, N_EMBD, 5, 3, 2));
    }

    @Test
    public void patchMajorGatherAndTheColumnPermutationReproduceTheChannelMajorUnfold() {
        int pw = 7;
        int ph = 4;
        int s = 2;
        int ow = 3;
        int oh = 2;
        float[] in = rows(pw * ph, 11);
        float[] patchMajor = gather(in, PixelShuffle.chunkSources(pw * ph,
                ids -> PixelShuffle.unfoldWindows(ids, 1, pw, ow, oh, s)));
        int[] perm = PixelShuffle.channelMajor(N_EMBD, s);
        int k = perm.length;
        float[] got = new float[patchMajor.length];
        for (int r = 0; r < ow * oh; r++) {
            for (int c = 0; c < k; c++) {
                got[r * k + c] = patchMajor[r * k + perm[c]];
            }
        }
        assertArrayEquals(PixelShuffle.unfoldWindows(in, N_EMBD, pw, ow, oh, s), got, 0f);
    }

    @Test
    public void chunkTablesRefuseAnEmptyReorder() {
        assertThrows(IllegalArgumentException.class, () -> PixelShuffle.chunkSources(0, ids -> ids));
    }

    @Test
    public void keyRunsGroupTheRowsOfEachWindow() {
        List<int[]> all = VisionRows.keyRuns(null, 5);
        assertEquals(1, all.size());
        assertArrayEquals(new int[]{0, 5, 0, 5}, all.get(0));

        final int[] starts = {0, 0, 0, 3, 3, 5};
        final int[] ends = {3, 3, 3, 5, 5, 6};
        List<int[]> runs = VisionRows.keyRuns(ranges(starts, ends), 6);
        assertEquals(3, runs.size());
        assertArrayEquals(new int[]{0, 3, 0, 3}, runs.get(0));
        assertArrayEquals(new int[]{3, 5, 3, 5}, runs.get(1));
        assertArrayEquals(new int[]{5, 6, 5, 6}, runs.get(2));
    }

    @Test
    public void keyRunsRefuseARangeOutsideTheRows() {
        assertThrows(IllegalArgumentException.class,
                () -> VisionRows.keyRuns(ranges(new int[]{0, 0}, new int[]{3, 3}), 2));
        assertThrows(IllegalArgumentException.class,
                () -> VisionRows.keyRuns(ranges(new int[]{1, 1}, new int[]{1, 1}), 2));
    }

    private static VisionRows.KeyRanges ranges(int[] begins, int[] ends) {
        return new VisionRows.KeyRanges() {
            @Override
            public int begin(int row) {
                return begins[row];
            }

            @Override
            public int end(int row) {
                return ends[row];
            }
        };
    }
}
