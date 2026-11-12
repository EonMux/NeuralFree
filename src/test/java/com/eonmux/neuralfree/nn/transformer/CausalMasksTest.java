package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

/**
 * {@link CausalMasks}: the mask of a shape is true exactly above the diagonal, built once per backend and shape and
 * then shared by every handle the cache returns, the least recently used shape is dropped when more than
 * {@link CausalMasks#CAPACITY} are kept, and a handle taken before its shape was evicted stays valid until its holder
 * closes it. Runs on {@link TrainParity#hostBackend()}.
 */
public class CausalMasksTest {

    private ComputeBackend backend;
    private final List<Tensor> handles = new ArrayList<>();

    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        for (Tensor t : handles) {
            t.close();
        }
        backend.close();
    }

    @Test
    public void aMaskIsTrueExactlyAtFutureKeys() {
        Tensor mask = future(3, 5);
        assertEquals(DType.BOOL, mask.dtype());
        assertEquals(Shape.of(3, 5), mask.shape());
        byte[] values = bytes(mask);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                assertEquals("[" + i + "," + j + "]", j > i ? 1 : 0, values[i * 5 + j]);
            }
        }
    }

    @Test
    public void aShapeIsBuiltOnceAndSharedByEveryHandle() {
        Tensor first = future(4, 4);
        Tensor second = future(4, 4);
        assertNotSame("each call returns its own handle", first, second);
        assertSame("over one built mask", first.storage().gpuBuffer(), second.storage().gpuBuffer());
        assertNotSame(first.storage().gpuBuffer(), future(4, 6).storage().gpuBuffer());
        first.close();
        assertEquals("closing one handle leaves the cached mask", 1, bytes(future(4, 4))[1]);
    }

    @Test
    public void theLeastRecentlyUsedShapeIsDroppedBeyondTheCapacity() {
        Object oldest = future(1, 1).storage().gpuBuffer();
        Object recent = future(2, 2).storage().gpuBuffer();
        for (int n = 3; n < 2 + CausalMasks.CAPACITY; n++) {
            future(n, n);
        }
        assertSame("a recent use keeps a shape", recent, future(2, 2).storage().gpuBuffer());
        future(100, 100);
        assertSame(recent, future(2, 2).storage().gpuBuffer());
        Tensor rebuilt = future(1, 1);
        assertNotSame("the oldest shape was dropped and is built again", oldest, rebuilt.storage().gpuBuffer());
        assertEquals(0, bytes(rebuilt)[0]);
    }

    @Test
    public void aHandleStaysValidAfterItsShapeIsEvicted() {
        Tensor held = future(2, 3);
        for (int n = 10; n < 10 + CausalMasks.CAPACITY; n++) {
            future(n, n);
        }
        assertNotSame("the shape was evicted", held.storage().gpuBuffer(), future(2, 3).storage().gpuBuffer());
        assertArrayEquals("the evicted mask is still readable through the held handle",
            new byte[]{0, 1, 1, 0, 0, 1}, bytes(held));
    }

    @Test
    public void emptyShapesAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> CausalMasks.future(backend, 0, 3));
        assertThrows(IllegalArgumentException.class, () -> CausalMasks.future(backend, 3, 0));
    }

    /** A handle of the cached mask, closed by {@link #tearDown}. */
    private Tensor future(int querySeq, int keySeq) {
        Tensor handle = CausalMasks.future(backend, querySeq, keySeq);
        handles.add(handle);
        return handle;
    }

    /** The BOOL elements, one byte each. */
    private static byte[] bytes(Tensor mask) {
        ByteBuffer buffer = mask.toByteBuffer();
        byte[] out = new byte[(int) mask.numElements()];
        buffer.get(out);
        return out;
    }
}
