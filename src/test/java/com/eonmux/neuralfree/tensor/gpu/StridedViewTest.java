package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/** View metadata of the strided train kernels, and the host readback of the same views. */
public class StridedViewTest {

    private ComputeBackend backend;

    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
    }

    private Tensor iota(int... dims) {
        Shape shape = Shape.of(dims);
        float[] data = new float[(int) shape.numElements()];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }
        return Tensor.fromArray(backend, data).reshape(shape);
    }

    @Test
    public void sliceOfTransposeKeepsStridesAndAddsOffset() {
        Tensor t = iota(4, 6);
        Tensor view = t.transpose().slice(0, 2, 5);
        StridedView v = StridedView.of(view);

        assertEquals(2, v.offset());
        assertEquals(2, v.rank());
        assertEquals(3, v.dim(0));
        assertEquals(4, v.dim(1));
        assertEquals(1, v.stride(0));
        assertEquals(6, v.stride(1));
        assertFalse(v.contiguous());
        assertSame(t.storage().gpuBuffer(), v.buffer());

        float[] expected = new float[12];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                expected[i * 4 + j] = j * 6 + i + 2;
            }
        }
        assertArrayEquals(expected, view.toFloatArray(), 0.0f);
        assertArrayEquals(expected, view.contiguous().toFloatArray(), 0.0f);
    }

    @Test
    public void asWordsScalesTheViewAndAppendsTheWordDim() {
        StridedView v = StridedView.of(iota(4, 6).transpose().slice(0, 2, 5));
        StridedView w = v.asWords(2);

        assertSame(v, v.asWords(1));
        assertSame(v.buffer(), w.buffer());
        assertEquals(4, w.offset());
        assertEquals(3, w.rank());
        assertArrayEquals(new int[]{3, 4, 2}, new int[]{w.dim(0), w.dim(1), w.dim(2)});
        assertArrayEquals(new int[]{2, 12, 1}, new int[]{w.stride(0), w.stride(1), w.stride(2)});
        assertEquals(24, w.numElements());
        assertFalse(w.contiguous());

        StridedView expanded = StridedView.of(iota(3).unsqueeze(0).expand(Shape.of(4, 3))).asWords(2);
        assertArrayEquals(new int[]{0, 2, 1}, new int[]{expanded.stride(0), expanded.stride(1), expanded.stride(2)});
        assertTrue(StridedView.of(iota(2, 3)).asWords(2).contiguous());
    }

    @Test(expected = IllegalArgumentException.class)
    public void asWordsRejectsANonPositiveWordCount() {
        StridedView.of(iota(2)).asWords(0);
    }

    @Test
    public void expandOfUnsqueezeHasZeroStrides() {
        Tensor t = iota(3);
        Tensor view = t.unsqueeze(0).expand(Shape.of(4, 3));
        StridedView v = StridedView.of(view);

        assertEquals(0, v.offset());
        assertEquals(0, v.stride(0));
        assertEquals(1, v.stride(1));
        assertEquals(12, v.numElements());
        assertFalse(v.contiguous());
        assertArrayEquals(new float[]{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}, view.toFloatArray(), 0.0f);
    }

    @Test
    public void expandPrependsZeroStrideDims() {
        Tensor view = iota(3, 1).expand(Shape.of(2, 3, 5));
        StridedView v = StridedView.of(view);

        assertEquals(0, v.stride(0));
        assertEquals(1, v.stride(1));
        assertEquals(0, v.stride(2));
        float[] out = view.toFloatArray();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 5; k++) {
                    assertEquals(j, out[i * 15 + j * 5 + k], 0.0f);
                }
            }
        }
    }

    @Test
    public void sliceOfSliceAccumulatesOffsetAndSharesStorage() {
        Tensor t = iota(10);
        Tensor outer = t.slice(0, 2, 9);
        Tensor inner = outer.slice(0, 3, 5);
        StridedView v = StridedView.of(inner);

        assertEquals(5, v.offset());
        assertEquals(2, v.numElements());
        assertTrue(v.contiguous());
        assertTrue(inner.isDense());
        assertFalse(inner.isContiguous());
        assertSame(t.storage().gpuBuffer(), inner.storage().gpuBuffer());
        assertEquals(t.storage().lengthBytes(), inner.storage().lengthBytes());
        assertArrayEquals(new float[]{5, 6}, inner.toFloatArray(), 0.0f);
    }

    @Test
    public void broadcastViewMatchesExpand() {
        Tensor t = iota(3);
        StridedView v = StridedView.broadcast(t, Shape.of(2, 3));

        assertEquals(2, v.rank());
        assertEquals(0, v.stride(0));
        assertEquals(1, v.stride(1));
    }

    @Test
    public void coalesceMergesDimsContiguousInEveryView() {
        Tensor a = iota(2, 3, 4);
        Tensor b = iota(2, 3, 4);
        StridedView[] merged = StridedView.coalesce(StridedView.of(a), StridedView.of(b));

        assertEquals(1, merged[0].rank());
        assertEquals(24, merged[0].dim(0));
        assertEquals(1, merged[0].stride(0));
        assertEquals(1, merged[1].stride(0));
    }

    @Test
    public void coalesceKeepsDimsAViewIsStridedAcross() {
        Tensor transposed = iota(4, 6).transpose();
        Tensor dense = iota(6, 4);
        StridedView[] merged = StridedView.coalesce(StridedView.of(transposed), StridedView.of(dense));

        assertEquals(2, merged[0].rank());
        assertEquals(1, merged[0].stride(0));
        assertEquals(6, merged[0].stride(1));
        assertEquals(4, merged[1].stride(0));
        assertEquals(1, merged[1].stride(1));
    }

    @Test
    public void coalesceDropsUnitDimsAndHandlesAllOnes() {
        StridedView[] merged = StridedView.coalesce(StridedView.of(iota(1, 1, 1)));
        assertEquals(1, merged[0].rank());
        assertEquals(1, merged[0].dim(0));
        assertEquals(0, merged[0].stride(0));

        StridedView[] squeezed = StridedView.coalesce(StridedView.of(iota(3, 1, 4)));
        assertEquals(1, squeezed[0].rank());
        assertEquals(12, squeezed[0].dim(0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void coalesceRejectsDifferentDims() {
        StridedView.coalesce(StridedView.of(iota(2, 3)), StridedView.of(iota(3, 2)));
    }

    @Test
    public void rankSevenViewSplitsOnLeadingDim() {
        Tensor t = iota(2, 3, 1, 2, 2, 1, 2);
        StridedView v = StridedView.of(t);
        assertEquals(7, v.rank());

        StridedView[] parts = v.splitLeading();
        assertEquals(2, parts.length);
        assertEquals(0, parts[0].offset());
        assertEquals(24, parts[1].offset());
        for (StridedView p : parts) {
            assertEquals(StridedView.MAX_RANK, p.rank());
            assertArrayEquals(new Object[]{3, 1, 2, 2, 1, 2}, p.dimArgs());
            assertEquals(8, p.stride(0));
        }
    }

    @Test
    public void rankSevenPermutedViewSplitsWithLeadingStride() {
        Tensor t = iota(2, 2, 2, 2, 2, 2, 3).transpose(6, 1, 2, 3, 4, 5, 0);
        StridedView[] parts = StridedView.of(t).splitLeading();

        assertEquals(3, parts.length);
        assertEquals(0, parts[0].offset());
        assertEquals(1, parts[1].offset());
        assertEquals(2, parts[2].offset());
        assertEquals(96, parts[0].stride(StridedView.MAX_RANK - 1));
    }

    @Test(expected = IllegalStateException.class)
    public void kernelArgsRequireSplitAboveMaxRank() {
        StridedView.of(iota(2, 2, 2, 2, 2, 2, 2)).dimArgs();
    }

    @Test
    public void kernelArgsArePadded() {
        StridedView v = StridedView.of(iota(4, 6).transpose());
        assertArrayEquals(new Object[]{6, 4, 1, 1, 1, 1}, v.dimArgs());
        assertArrayEquals(new Object[]{1, 6, 0, 0, 0, 0}, v.strideArgs());
    }

    @Test
    public void lowerRankViewIsNotSplit() {
        StridedView v = StridedView.of(iota(2, 3));
        StridedView[] parts = v.splitLeading();
        assertEquals(1, parts.length);
        assertSame(v, parts[0]);
    }
}
