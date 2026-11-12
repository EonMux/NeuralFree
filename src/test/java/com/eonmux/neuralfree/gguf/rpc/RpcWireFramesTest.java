package com.eonmux.neuralfree.gguf.rpc;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * Frames leave straight from the buffer of the payload that built them, batch members copy in by their buffers,
 * and a float read decodes its payload in place: each carries the bytes of the copying forms.
 */
public class RpcWireFramesTest {

    private static final byte[] DATA = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    @Test
    public void aFrameFromThePayloadBufferCarriesThePayloadBytes() throws IOException {
        RpcWire.Writer w = new RpcWire.Writer(RpcWire.OP_WRITE).i64(7).i64(64).rawBytes(DATA).bool(true);
        assertEquals(w.toBytes().length, w.size());
        assertArrayEquals(frame(wire -> wire.writeFrame(w.toBytes(), true)), frame(wire -> wire.writeFrame(w)));
        RpcWire read = new RpcWire(new DataInputStream(new ByteArrayInputStream(frame(wire -> wire.writeFrame(w)))),
                new DataOutputStream(new ByteArrayOutputStream()));
        assertArrayEquals(w.toBytes(), read.readFrame());
    }

    @Test
    public void aBatchFrameCopiesItsMembersVerbatimPastItsPresizedCapacity() {
        RpcWire.Writer a = new RpcWire.Writer(RpcWire.OP_FREE).i64(3);
        RpcWire.Writer b = new RpcWire.Writer(RpcWire.OP_WRITE).i64(4).i64(0).rawBytes(DATA).bool(false);
        RpcWire.Writer batch = new RpcWire.Writer(RpcWire.OP_BATCH, 2).i32(2).body(a).body(b);
        ByteArrayOutputStream expected = new ByteArrayOutputStream();
        expected.write(RpcWire.OP_BATCH);
        expected.writeBytes(new byte[]{0, 0, 0, 2});
        expected.writeBytes(a.toBytes());
        expected.writeBytes(b.toBytes());
        assertArrayEquals(expected.toByteArray(), batch.toBytes());
        assertThrows(IllegalArgumentException.class, () -> new RpcWire.Writer(RpcWire.OP_BATCH, 0));
    }

    @Test
    public void aFloatReadDecodesLittleEndianFloatsAtItsOffset() {
        float[] values = {1.5f, -0f, Float.NaN};
        byte[] le = new byte[values.length * Float.BYTES];
        java.nio.ByteBuffer.wrap(le).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(values);
        RpcWire.Reader r = new RpcWire.Reader(new RpcWire.Writer(RpcWire.STATUS_OK).rawBytes(le).i32(42).toBytes());
        r.u8();
        float[] dst = new float[5];
        r.rawFloatsInto(dst, 1, values.length);
        assertArrayEquals(new float[]{0f, 1.5f, -0f, Float.NaN, 0f}, dst, 0f);
        assertEquals("the reader continues after the floats", 42, r.i32());

        RpcWire.Reader wrong = new RpcWire.Reader(new RpcWire.Writer(RpcWire.STATUS_OK).rawBytes(le).toBytes());
        wrong.u8();
        assertThrows(IllegalArgumentException.class, () -> wrong.rawFloatsInto(new float[4], 0, 4));
    }

    private interface FrameWrite {
        void write(RpcWire wire) throws IOException;
    }

    private static byte[] frame(FrameWrite write) throws IOException {
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        RpcWire wire = new RpcWire(new DataInputStream(new ByteArrayInputStream(new byte[0])),
                new DataOutputStream(sink));
        write.write(wire);
        return sink.toByteArray();
    }
}
