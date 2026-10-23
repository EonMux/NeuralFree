package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Readback of squeezed strided views on {@link TrainParity#hostBackend()}, whose buffers hold data, so the values
 * read through the view's offset and strides are checked, not only its metadata.
 */
public class TensorSqueezeReadbackTest {

    private ComputeBackend backend;

    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        TrainGpu.release(backend);
        backend.close();
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
    public void squeezeOfOffsetStridedViewReadsTheViewedElements() {
        // a [1, 1, 3, 1] view at offset 13 with strides [12, 12, 4, 1]
        Tensor squeezed = iota(2, 1, 3, 4).slice(0, 1, 2).slice(3, 1, 2).squeeze();

        assertEquals(Shape.vector(3), squeezed.shape());
        assertArrayEquals(new long[]{4}, squeezed.strides().strides());
        assertEquals(13L, squeezed.strides().offset());
        assertArrayEquals(new float[]{13, 17, 21}, squeezed.toFloatArray(), 0.0f);
        assertArrayEquals(new float[]{13, 17, 21}, squeezed.contiguous().toFloatArray(), 0.0f);
    }

    @Test
    public void squeezeAxisOfOffsetStridedViewReadsTheViewedElements() {
        // a [3, 1] view at offset 2 with strides [4, 1]; squeezing axis 1 leaves the column at stride 4
        Tensor squeezed = iota(2, 3, 4).slice(0, 0, 1).squeeze(0).slice(1, 2, 3).squeeze(1);

        assertEquals(Shape.vector(3), squeezed.shape());
        assertArrayEquals(new long[]{4}, squeezed.strides().strides());
        assertEquals(2L, squeezed.strides().offset());
        assertArrayEquals(new float[]{2, 6, 10}, squeezed.toFloatArray(), 0.0f);
    }
}
