package com.eonmux.neuralfree.gguf.quant.write;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The statistics llama.cpp's {@code save_imatrix} stores beside the sums, written and read back by
 * {@link ImatrixFile}: the plain activation sums {@code <tensor>.in_sum}, the twelve {@code <tensor>.stats} values in
 * {@code imatrix.stats_schema} order (non-legacy {@code compute_vector_statistics}: mean and spread over the
 * activation averages, sum and entropy over the squared averages) and {@code imatrix.n_layer_nextn}.
 */
public class ImatrixStoredStatisticsTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** One matrix of four columns over two rows: squared averages {1, 2, 3, 4}, activation averages {1, -1, 2, 0}. */
    private static ImatrixFile.Entry dense(String name) {
        return new ImatrixFile.Entry(name, 1, 4, new float[]{2f, 4f, 6f, 8f}, new float[]{2f, -2f, 4f, 0f},
                new long[]{2});
    }

    private static ImatrixFile.Entry experts(String name) {
        return new ImatrixFile.Entry(name, 2, 2, new float[]{1f, 4f, 9f, 16f}, new float[]{1f, 2f, -3f, 4f},
                new long[]{1, 1});
    }

    @Test
    public void sumsStatisticsAndTheNextnCountRoundTrip() throws Exception {
        List<ImatrixFile.Entry> entries = Arrays.asList(dense("blk.0.attn_q.weight"),
                experts("blk.0.ffn_up_exps.weight"), dense("blk.1.attn_q.weight"));
        Path out = tmp.getRoot().toPath().resolve("stats.gguf");
        ImatrixFile.writeGguf(out, entries, Collections.singletonList("corpus.txt"), 3, 64, 1);
        ImatrixFile.Read read = ImatrixFile.read(out);

        assertFalse(read.isLegacy());
        assertEquals(1, read.nLayerNextn());
        assertEquals(ImatrixFile.STATS_SCHEMA, read.statsSchema());
        assertEquals(entries.size(), read.statsTensors().size());
        Map<String, float[]> expected = ImatrixStatistics.stored(entries, 1);
        for (ImatrixFile.Entry e : entries) {
            ImatrixFile.Entry back = entry(read, e.tensor());
            assertNotNull(e.tensor(), back);
            assertTrue(e.tensor() + " keeps its .in_sum", back.hasSums());
            assertArrayEquals(e.sums(), back.sums(), 0f);
            assertArrayEquals(e.sumSq(), back.sumSq(), 0f);
            assertArrayEquals(e.counts(), back.counts());
            assertArrayEquals(e.tensor(), expected.get(e.tensor()), read.stats(e.tensor()), 0f);
        }
    }

    @Test
    public void theStoredRowFollowsTheReferenceFormulas() {
        float[] row = ImatrixStatistics.stored(Collections.singletonList(dense("blk.0.attn_q.weight")), 0)
                .get("blk.0.attn_q.weight");
        assertEquals(ImatrixFile.STATS_SCHEMA.size(), row.length);
        assertEquals("sum_sq: the squared averages", 10f, row[0], 1e-6f);
        assertEquals("mean: the activation averages", 0.5f, row[1], 1e-6f);
        assertEquals("elements", 4f, row[2], 0f);
        assertEquals("std_deviation", (float) Math.sqrt(1.25), row[3], 1e-6f);
        assertEquals("skewness of a symmetric spread", 0f, row[4], 1e-6f);
        assertEquals("kurtosis", 2.5625f / 1.5625f - 3f, row[5], 1e-5f);
        double entropy = 0;
        for (double p : new double[]{0.1, 0.2, 0.3, 0.4}) {
            entropy -= p * Math.log(p) / Math.log(2);
        }
        assertEquals("h_norm: the energy entropy over log2(elements)", (float) (100 * entropy / 2), row[7], 1e-4f);
        for (int i : new int[]{6, 8, 9, 10, 11}) {
            assertTrue(ImatrixFile.STATS_SCHEMA.get(i) + " has no previous layer", Float.isNaN(row[i]));
        }
    }

    @Test
    public void anEntryWithoutPlainSumsWritesNoInSum() throws Exception {
        ImatrixFile.Entry squaresOnly = new ImatrixFile.Entry("blk.0.attn_k.weight", 1, 4,
                new float[]{1f, 1f, 1f, 1f}, new long[]{1});
        Path out = tmp.getRoot().toPath().resolve("legacy-entry.gguf");
        ImatrixFile.writeGguf(out, Arrays.asList(squaresOnly, dense("blk.0.attn_q.weight")),
                Collections.emptyList(), 1, 64);
        ImatrixFile.Read read = ImatrixFile.read(out);
        assertFalse(entry(read, "blk.0.attn_k.weight").hasSums());
        assertNull(entry(read, "blk.0.attn_k.weight").sums());
        assertTrue(entry(read, "blk.0.attn_q.weight").hasSums());
        assertEquals("no NextN count is written for 0", 0, read.nLayerNextn());
        assertNotNull("every entry carries its .stats row", read.stats("blk.0.attn_k.weight"));
    }

    @Test
    public void combiningKeepsPlainSumsOnlyWhenBothSidesHaveThem() {
        ImatrixFile.Entry a = dense("blk.0.attn_q.weight");
        List<ImatrixFile.Entry> both = ImatrixFile.combine(Collections.singletonList(a), Collections.singletonList(a));
        assertArrayEquals(new float[]{4f, -4f, 8f, 0f}, both.get(0).sums(), 0f);
        ImatrixFile.Entry squaresOnly = new ImatrixFile.Entry(a.tensor(), 1, 4, a.sumSq(), a.counts());
        List<ImatrixFile.Entry> mixed = ImatrixFile.combine(Collections.singletonList(a),
                Collections.singletonList(squaresOnly));
        assertFalse("llama.cpp keeps the activations only when every contribution has them",
                mixed.get(0).hasSums());
        assertArrayEquals(new float[]{4f, 8f, 12f, 16f}, mixed.get(0).sumSq(), 0f);
    }

    @Test
    public void invalidInputsAreRefusedByName() {
        Path out = tmp.getRoot().toPath().resolve("bad.gguf");
        IllegalArgumentException nextn = assertThrows(IllegalArgumentException.class, () -> ImatrixFile.writeGguf(out,
                Collections.singletonList(dense("blk.0.attn_q.weight")), Collections.emptyList(), 1, 64, -1));
        assertTrue(nextn.getMessage(), nextn.getMessage().contains("n_layer_nextn"));
        IllegalArgumentException sums = assertThrows(IllegalArgumentException.class, () -> new ImatrixFile.Entry(
                "blk.0.attn_q.weight", 1, 4, new float[4], new float[3], new long[]{1}));
        assertTrue(sums.getMessage(), sums.getMessage().contains("blk.0.attn_q.weight"));
    }

    @Test
    public void theReportLabelsTheNextnBlocks() {
        List<ImatrixFile.Entry> entries = Arrays.asList(dense("blk.0.attn_q.weight"), dense("blk.1.attn_q.weight"),
                dense("blk.2.attn_q.weight"));
        String report = ImatrixStatistics.compute(entries, 1).format();
        assertTrue(report, report.contains("mtp0"));
        assertEquals("block 2 is the first NextN block", 2, ImatrixStatistics.compute(entries, 1).nextnStart());
        assertFalse(ImatrixStatistics.compute(entries, 0).format().contains("mtp"));
        assertTrue("non-legacy entries print the L2 distance column", report.contains("L2 Dist"));
    }

    private static ImatrixFile.Entry entry(ImatrixFile.Read read, String tensor) {
        for (ImatrixFile.Entry e : read.entries()) {
            if (e.tensor().equals(tensor)) {
                return e;
            }
        }
        return null;
    }
}
