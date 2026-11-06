package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link GpuWeight#slice} and {@link GpuWeight#expertRuns}: a rank's whole-expert rows of a larger tensor keep the
 * tensor's global shape for the kernels that address experts by their global id, start their shards at the held
 * experts' global rows (adjacent for a slice, disjoint for sparse runs), and refuse a range that splits an expert,
 * overlaps another run or leaves the tensor.
 */
public class GpuWeightSliceTest {

    private FakeComputeBackend backend;

    @Before
    public void open() {
        backend = FakeComputeBackend.builder().build();
    }

    @After
    public void close() {
        if (backend != null) {
            backend.close();
        }
    }

    private GpuBuffer buffer(long bytes) {
        return backend.allocateBuffer(bytes, true);
    }

    @Test
    public void aSliceReportsTheGlobalShapeAndItsOwnRowRange() {
        long[] ne = {8, 4, 3};
        GpuBuffer first = buffer(4 * 8 * Float.BYTES);
        GpuBuffer second = buffer(4 * 8 * Float.BYTES);
        GpuWeight w = GpuWeight.slice("ffn_down_exps.weight", GGMLType.F32, ne,
                Arrays.asList(first, second), new long[]{4, 4}, 4);
        assertEquals("the tensor's global row count stays", 12, w.rows());
        assertEquals("the tensor's global shape stays", 3, w.ne()[2]);
        assertEquals(2, w.shardCount());
        assertEquals(4, w.shardRowStart(0));
        assertEquals(4, w.shardRows(0));
        assertEquals(8, w.shardRowStart(1));
        assertEquals(4, w.shardRows(1));
    }

    @Test
    public void aRangeInsideAnExpertIsRefused() {
        long[] ne = {8, 4, 3};
        GpuBuffer b = buffer(2 * 8 * Float.BYTES);
        assertThrows(IllegalArgumentException.class,
                () -> GpuWeight.slice("t", GGMLType.F32, ne, Collections.singletonList(b), new long[]{2}, 2));
    }

    @Test
    public void aRangeThatLeavesTheTensorIsRefused() {
        long[] ne = {8, 4, 3};
        GpuBuffer b = buffer(2 * 8 * Float.BYTES);
        assertThrows(IllegalArgumentException.class,
                () -> GpuWeight.slice("t", GGMLType.F32, ne, Collections.singletonList(b), new long[]{8}, 8));
    }

    @Test
    public void expertRunsHoldDisjointWholeExpertsAtTheirGlobalRows() {
        long[] ne = {8, 4, 3};
        GpuBuffer first = buffer(4 * 8 * Float.BYTES);
        GpuBuffer second = buffer(4 * 8 * Float.BYTES);
        GpuWeight w = GpuWeight.expertRuns("ffn_down_exps.weight", GGMLType.F32, ne,
                Arrays.asList(first, second), new long[]{4, 4}, new long[]{0, 8});
        assertEquals("the tensor's global row count stays", 12, w.rows());
        assertEquals(2, w.shardCount());
        assertEquals("expert 0 starts at row 0", 0, w.shardRowStart(0));
        assertEquals("expert 2 starts at row 8", 8, w.shardRowStart(1));
        assertEquals(4, w.shardRows(1));
    }

    @Test
    public void expertRunsRefuseAnOverlappingOrUnalignedRun() {
        long[] ne = {8, 4, 3};
        GpuBuffer a = buffer(4 * 8 * Float.BYTES);
        GpuBuffer b = buffer(4 * 8 * Float.BYTES);
        assertThrows(IllegalArgumentException.class, () -> GpuWeight.expertRuns("t", GGMLType.F32, ne,
                Arrays.asList(a, b), new long[]{4, 4}, new long[]{0, 0}));
        assertThrows(IllegalArgumentException.class, () -> GpuWeight.expertRuns("t", GGMLType.F32, ne,
                Collections.singletonList(a), new long[]{2}, new long[]{2}));
        assertThrows(IllegalArgumentException.class, () -> GpuWeight.expertRuns("t", GGMLType.F32, ne,
                Collections.singletonList(a), new long[]{8}, new long[]{8}));
        assertThrows(IllegalArgumentException.class, () -> GpuWeight.expertRuns("t", GGMLType.F32,
                new long[]{8, 4}, Collections.singletonList(a), new long[]{4}, new long[]{0}));
    }
}
