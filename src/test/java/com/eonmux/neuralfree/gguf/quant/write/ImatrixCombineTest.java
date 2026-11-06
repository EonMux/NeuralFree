package com.eonmux.neuralfree.gguf.quant.write;

import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ImatrixFile#combine}: the statistics of the same tensor sum, a legacy flat entry broadcasts its count
 * over a multi-matrix GGUF entry, shape mismatches fail naming the tensor, and a combined write reads back.
 */
public class ImatrixCombineTest {

    @Test
    public void sameTensorSumsStatisticsAndUnknownOnesPassThrough() {
        ImatrixFile.Entry a = new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 2, new float[]{1, 2}, new long[]{10});
        ImatrixFile.Entry b = new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 2, new float[]{3, 4}, new long[]{8});
        ImatrixFile.Entry other = new ImatrixFile.Entry("blk.0.ffn_down.weight", 1, 4, new float[4], new long[]{5});
        List<ImatrixFile.Entry> merged = ImatrixFile.combine(List.of(a, other), List.of(b));
        assertEquals(2, merged.size());
        ImatrixFile.Entry attn = merged.get(0);
        assertEquals("blk.0.attn_q.weight", attn.tensor());
        assertEquals(1, attn.nMat());
        assertEquals(2, attn.cols());
        assertEquals(4.0f, attn.sumSq()[0], 1e-6f);
        assertEquals(6.0f, attn.sumSq()[1], 1e-6f);
        assertEquals(18L, attn.count(0));
        assertEquals(5L, merged.get(1).count(0));
    }

    @Test
    public void aLegacyFlatEntryBroadcastsItsCountOverTheMatrices() {
        // a GGUF entry of 2 expert matrices of 4 columns, and a legacy read of the same 8 values as one matrix
        ImatrixFile.Entry gguf = new ImatrixFile.Entry("blk.0.ffn_gate_exps.weight", 2, 4,
                new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new long[]{10, 20});
        ImatrixFile.Entry legacy = new ImatrixFile.Entry("blk.0.ffn_gate_exps.weight", 1, 8,
                new float[]{2, 2, 2, 2, 2, 2, 2, 2}, new long[]{3});
        List<ImatrixFile.Entry> merged = ImatrixFile.combine(List.of(gguf), List.of(legacy));
        assertEquals(1, merged.size());
        ImatrixFile.Entry e = merged.get(0);
        assertEquals(2, e.nMat());
        assertEquals(4, e.cols());
        assertEquals(13L, e.count(0));
        assertEquals(23L, e.count(1));
        assertEquals(3.0f, e.sumSq()[0], 1e-6f);
        assertEquals(10.0f, e.sumSq()[7], 1e-6f);
    }

    @Test
    public void shapeMismatchesFailNamingTheTensor() {
        ImatrixFile.Entry a = new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 2, new float[]{1, 2}, new long[]{1});
        ImatrixFile.Entry wider = new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 4, new float[4], new long[]{1});
        IllegalArgumentException flat = assertThrows(IllegalArgumentException.class,
                () -> ImatrixFile.combine(List.of(a), List.of(wider)));
        assertTrue(flat.getMessage(), flat.getMessage().contains("blk.0.attn_q.weight"));

        ImatrixFile.Entry m2 = new ImatrixFile.Entry("blk.0.attn_q.weight", 2, 3, new float[6], new long[]{1, 1});
        ImatrixFile.Entry m3 = new ImatrixFile.Entry("blk.0.attn_q.weight", 3, 2, new float[6], new long[]{1, 1, 1});
        IllegalArgumentException mats = assertThrows(IllegalArgumentException.class,
                () -> ImatrixFile.combine(List.of(m2), List.of(m3)));
        assertTrue(mats.getMessage(), mats.getMessage().contains("2 matrices against 3"));
    }

    @Test
    public void aCombinedWriteReadsBackWithMergedMetadata() throws IOException {
        Path dir = Files.createTempDirectory("nf-imatrix-combine-");
        Path out = dir.resolve("combined.gguf");
        try {
            ImatrixFile.Entry a = new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 2, new float[]{1, 2}, new long[]{6});
            ImatrixFile.Entry b = new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 2, new float[]{3, 4}, new long[]{10});
            ImatrixFile.writeGguf(dir.resolve("a.gguf"), List.of(a), List.of("cal-a.txt"), 2, 4);
            ImatrixFile.Read ra = ImatrixFile.read(dir.resolve("a.gguf"));
            ImatrixFile.Read rb = ImatrixFile.read(writeLegacyB(dir));
            List<ImatrixFile.Entry> merged = ImatrixFile.combine(ra.entries(), rb.entries());
            ImatrixFile.writeGguf(out, merged, List.of("cal-a.txt", "cal-b.txt"),
                    Math.max(ra.chunkCount(), rb.chunkCount()) + 1, 4);
            ImatrixFile.Read back = ImatrixFile.read(out);
            assertEquals(1, back.entries().size());
            // gguf sums [1, 2] plus the legacy read of the .dat round trip ([0.9, 1.2], its count 3)
            assertEquals(1.9f, back.entries().get(0).sumSq()[0], 1e-4f);
            assertEquals(3.2f, back.entries().get(0).sumSq()[1], 1e-4f);
            assertEquals(9L, back.entries().get(0).count(0));
            assertEquals(3, back.chunkCount());
            assertEquals(4, back.chunkSize());
            assertEquals(List.of("cal-a.txt", "cal-b.txt"), back.datasets());
        } finally {
            cleanup(dir);
        }
    }

    /** A legacy {@code .dat} of the same tensor: sums stored per call, one count. */
    private static Path writeLegacyB(Path dir) throws IOException {
        ImatrixFile.Entry b = new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 2, new float[]{3, 4}, new long[]{10});
        return ImatrixFile.writeLegacy(dir.resolve("b.dat"), List.of(b), 2, 4, List.of("cal-b.txt"), null);
    }

    private static void cleanup(Path dir) {
        List<Path> files = List.of();
        try {
            files = Files.list(dir).collect(java.util.stream.Collectors.toList());
        } catch (IOException e) {
            // the test's temp directory: a failure to clean does not fail the assertion
        }
        for (Path p : files) {
            try {
                Files.deleteIfExists(p);
            } catch (IOException e) {
                // as above
            }
        }
        try {
            Files.deleteIfExists(dir);
        } catch (IOException e) {
            // as above
        }
    }
}
