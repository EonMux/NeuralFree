package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** A wire tensor of one live weight update: shape and byte-count validation naming the first bad field, the F32
 * encoding, and the immutable views over its bytes. */
public class WeightUpdateTensorTest {

    @Test
    public void anF32TensorEncodesItsFloatsLittleEndian() {
        WeightUpdateTensor t = WeightUpdateTensor.f32("blk.0.attn_norm.weight", new long[]{4}, new float[]{
                1f, -2.5f, 3.25e4f, -0f});
        assertEquals("blk.0.attn_norm.weight", t.name());
        assertEquals(GGMLType.F32, t.dtype());
        assertArrayEquals(new long[]{4}, t.ne());
        assertEquals(4, t.numElements());
        assertEquals(16, t.dataBytes());
        float[] round = new float[4];
        t.dataBuffer().asFloatBuffer().get(round);
        assertArrayEquals(new float[]{1f, -2.5f, 3.25e4f, -0f}, round, 0f);
    }

    @Test
    public void aTypedTensorHoldsItsRawBytes() {
        byte[] raw = new byte[340];
        raw[0] = 1;
        WeightUpdateTensor t = WeightUpdateTensor.of("token_embd.weight", GGMLType.Q8_0, new long[]{64, 5}, raw);
        assertEquals(GGMLType.Q8_0, t.dtype());
        assertEquals(340, t.dataBytes());
        assertEquals(320, t.numElements());
        assertEquals(1, t.data()[0]);
        assertEquals("token_embd.weight Q8_0 [64, 5] (340 B)", t.toString());
    }

    @Test
    public void theConstructorCopiesTheCallerArrays() {
        long[] ne = {8, 2};
        byte[] data = new byte[64];
        WeightUpdateTensor t = WeightUpdateTensor.of("blk.0.attn_q.weight", GGMLType.F32, ne, data);
        ne[0] = 99;
        data[0] = 99;
        assertArrayEquals(new long[]{8, 2}, t.ne());
        assertEquals(0, t.data()[0]);
        assertNotEquals("the shape view is a copy too", ne, t.ne());
    }

    @Test
    public void aBlankOrMissingNameIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> WeightUpdateTensor.f32("  ", new long[]{1}, new float[]{0f}));
        assertTrue(e.getMessage(), e.getMessage().contains("no name"));
        e = assertThrows(IllegalArgumentException.class,
                () -> WeightUpdateTensor.of(null, GGMLType.F32, new long[]{1}, new byte[4]));
        assertTrue(e.getMessage(), e.getMessage().contains("no name"));
    }

    @Test
    public void aBadShapeIsRefusedByName() {
        assertThrows(IllegalArgumentException.class,
                () -> WeightUpdateTensor.f32("t.weight", new long[0], new float[0]));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> WeightUpdateTensor.f32("t.weight", new long[]{4, 0}, new float[0]));
        assertTrue(e.getMessage(), e.getMessage().contains("non-positive dimension") && e.getMessage().contains("t.weight"));
        e = assertThrows(IllegalArgumentException.class, () -> WeightUpdateTensor.f32("t.weight", null, new float[0]));
        assertTrue(e.getMessage(), e.getMessage().contains("no dimensions"));
    }

    @Test
    public void aByteCountMismatchNamesTheCount() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> WeightUpdateTensor.of("t.weight", GGMLType.Q8_0, new long[]{64, 2}, new byte[10]));
        assertTrue(e.getMessage(), e.getMessage().contains("needs 136 bytes, got 10"));
    }

    @Test
    public void aFloatCountMismatchNamesTheCount() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> WeightUpdateTensor.f32("t.weight", new long[]{4}, new float[]{0f, 1f}));
        assertTrue(e.getMessage(), e.getMessage().contains("holds 4 elements, got 2 floats"));
    }

    @Test
    public void missingDataAndTypeAreRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> WeightUpdateTensor.of("t.weight", GGMLType.F32, new long[]{1}, null));
        assertTrue(e.getMessage(), e.getMessage().contains("carries no data"));
        assertThrows(NullPointerException.class,
                () -> WeightUpdateTensor.of("t.weight", null, new long[]{1}, new byte[4]));
    }

    @Test
    public void dtypeNamesParseCaseInsensitivelyAndRefuseOtherwise() {
        assertEquals(GGMLType.F32, WeightUpdateTensor.parseDtype("F32", "update_weights"));
        assertEquals(GGMLType.Q8_0, WeightUpdateTensor.parseDtype("q8_0", "update_weights"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> WeightUpdateTensor.parseDtype("fp16", "update_weights"));
        assertTrue(e.getMessage(), e.getMessage().contains("update_weights")
                && e.getMessage().contains("not a ggml type name"));
        e = assertThrows(IllegalArgumentException.class, () -> WeightUpdateTensor.parseDtype(null, "update_weights"));
        assertTrue(e.getMessage(), e.getMessage().contains("not a ggml type name"));
    }

    @Test
    public void theDataBufferViewsTheInternalBytesDirectly() {
        WeightUpdateTensor t = WeightUpdateTensor.f32("t.weight", new long[]{2}, new float[]{1.5f, 2.5f});
        ByteBuffer b = t.dataBuffer();
        assertEquals(ByteOrder.LITTLE_ENDIAN, b.order());
        assertEquals(0, b.position());
        assertEquals(8, b.limit());
        assertEquals(1.5f, b.getFloat(0), 0f);
        assertEquals(2.5f, b.getFloat(4), 0f);
    }
}
