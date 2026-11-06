package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.quant.write.ImatrixStatistics.Report;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixStatistics.Tensor;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link ImatrixStatistics} against hand-computed values (llama.cpp {@code show_statistics}, the values-only
 * path): the per-column moments and entropy of one tensor, the previous-layer gain and comparison metrics, the
 * per-layer aggregation, the print order, and the layer-gap suppression.
 */
public class ImatrixStatisticsTest {

    /** A dense entry whose per-column means are {@code means} at activation count {@code count}. */
    private static ImatrixFile.Entry entry(String name, float[] means, int count) {
        float[] sumSq = new float[means.length];
        for (int j = 0; j < means.length; j++) {
            sumSq[j] = means[j] * count;
        }
        return new ImatrixFile.Entry(name, 1, means.length, sumSq, new long[]{count});
    }

    @Test
    public void oneTensorCarriesTheMomentsAndEntropyOfItsColumnMeans() {
        // means [1, 2, 3, 4]: mean 2.5, variance 1.25, zero third moment, kurtosis 1.64 - 3,
        // entropy of the energy distribution {1,2,3,4}/10 in bits.
        Report r = ImatrixStatistics.compute(List.of(entry("blk.0.attn_q.weight", new float[]{1, 2, 3, 4}, 8)));
        assertEquals(1, r.tensors().size());
        Tensor t = r.tensors().get(0);
        assertEquals("blk.0.attn_q.weight", t.name());
        // the reference's name split keeps 4-part names whole (blk, index, name, weight)
        assertEquals("blk.0.attn_q.weight", t.shortName());
        assertEquals(0, t.layer());
        assertEquals(10.0, t.sum(), 1e-6);
        assertEquals(2.5f, t.mean(), 1e-6);
        assertEquals(4L, t.elements());
        assertEquals((float) Math.sqrt(1.25), t.stdDeviation(), 1e-5f);
        assertEquals(0.0f, t.skewness(), 1e-5f);
        assertEquals(-1.36f, t.kurtosis(), 1e-4f);
        double bits = -(0.1 * Math.log(0.1) + 0.2 * Math.log(0.2) + 0.3 * Math.log(0.3) + 0.4 * Math.log(0.4))
                / Math.log(2);
        assertEquals((float) (100 * bits / 2), t.hNorm(), 1e-3f);
        // layer 0 has no previous layer: the comparison metrics stay unset
        assertTrue(Float.isNaN(t.gain()));
        assertTrue(Float.isNaN(t.cossim()));
        assertTrue(Float.isNaN(t.pearson()));
        // layer 0 tensors aggregate nowhere: the layer table stays empty
        assertEquals(0, r.layers().size());
    }

    @Test
    public void theNextLayerComparesAgainstThePreviousOne() {
        Report r = ImatrixStatistics.compute(List.of(
                entry("blk.0.attn_q.weight", new float[]{1, 2, 3, 4}, 4),
                entry("blk.1.attn_q.weight", new float[]{2, 4, 6, 8}, 1)));
        Tensor t1 = r.tensors().get(1);
        assertEquals("blk.1.attn_q.weight", t1.name());
        assertEquals(1, t1.layer());
        assertEquals(20.0, t1.sum(), 1e-6);
        assertEquals(5.0f, t1.mean(), 1e-6);
        // RMS gain sqrt(5 / 2.5) between the layers' mean energies
        assertEquals((float) Math.sqrt(2), t1.gain(), 1e-5f);
        // the averages are parallel: cosine, Pearson 1, L2 the 4-difference norm, covariance of the pairs
        assertEquals(1.0f, t1.cossim(), 1e-6f);
        assertEquals(1.0f, t1.pearson(), 1e-6f);
        assertEquals((float) Math.sqrt(30), t1.l2Dist(), 1e-4f);
        assertEquals(2.5f, t1.covariance(), 1e-5f);
        // the layer row: same gain and correlation, the summed energy of the layer's tensors
        assertEquals(1, r.layers().size());
        ImatrixStatistics.Layer l = r.layers().get(0);
        assertEquals(1, l.layer());
        assertEquals(20.0, l.sum(), 1e-6);
        assertEquals((float) Math.sqrt(2), l.gain(), 1e-5f);
        assertEquals(1.0f, l.cossim(), 1e-6f);
        assertEquals(1.0f, l.pearson(), 1e-6f);
        assertEquals(2.5f, l.covariance(), 1e-5f);
    }

    @Test
    public void theOutputHeadSortsLastAndNonBlockLayersAggregateNowhere() {
        Report r = ImatrixStatistics.compute(List.of(
                entry("output.weight", new float[]{1, 2}, 4),
                entry("blk.1.attn_v.weight", new float[]{3, 4}, 4),
                entry("blk.0.attn_v.weight", new float[]{5, 6}, 4)));
        List<Tensor> ts = r.tensors();
        assertEquals(3, ts.size());
        assertEquals("blk.0.attn_v.weight", ts.get(0).name());
        assertEquals("blk.1.attn_v.weight", ts.get(1).name());
        assertEquals("output.weight", ts.get(2).name());
        assertEquals(-1, ts.get(2).layer());
        assertEquals("output", ts.get(2).shortName());
        // the head has no block index, so the layer table holds block 1 alone
        assertEquals(1, r.layers().size());
        assertEquals(1, r.layers().get(0).layer());
    }

    @Test
    public void aGapInTheLayerSequenceSuppressesTheLayerTable() {
        Report r = ImatrixStatistics.compute(List.of(
                entry("blk.0.attn_q.weight", new float[]{1, 2}, 4),
                entry("blk.1.attn_q.weight", new float[]{3, 4}, 4),
                entry("blk.3.attn_q.weight", new float[]{5, 6}, 4)));
        assertTrue(r.layerGap());
        assertEquals(0, r.layers().size());
        // the tensor table still prints
        assertEquals(3, r.tensors().size());
    }

    @Test
    public void zeroCountMatricesHoldNoRow() {
        Report r = ImatrixStatistics.compute(List.of(new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 4,
                new float[4], new long[]{0})));
        assertEquals(0, r.tensors().size());
    }

    @Test
    public void theFormatPrintsTheTwoTables() {
        Report r = ImatrixStatistics.compute(List.of(
                entry("blk.0.attn_q.weight", new float[]{1, 2, 3, 4}, 4),
                entry("blk.1.attn_q.weight", new float[]{2, 4, 6, 8}, 1)));
        String out = r.format();
        assertTrue(out.contains("Layer"));
        assertTrue(out.contains("Tensor"));
        assertTrue(out.contains("blk.0.attn_q.weight"));
        assertTrue(out.contains("∑ E[A²]"));
        assertTrue(out.contains("2.5000")); // the mean of layer 0
        assertTrue(out.contains("CosSim")); // the layer table header
    }
}
