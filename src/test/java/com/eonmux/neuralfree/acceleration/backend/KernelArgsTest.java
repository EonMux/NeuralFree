package com.eonmux.neuralfree.acceleration.backend;

import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class KernelArgsTest {

    private static final class FakeBuffer implements GpuBuffer {
        @Override
        public long sizeInBytes() {
            return 64;
        }

        @Override
        public long bufferId() {
            return 1;
        }

        @Override
        public void close() {
        }
    }

    @Test
    public void kindOfClassifiesSupportedTypes() {
        assertEquals(KernelArgs.ArgKind.BUFFER, KernelArgs.kindOf(new FakeBuffer()));
        assertEquals(KernelArgs.ArgKind.I32, KernelArgs.kindOf(7));
        assertEquals(KernelArgs.ArgKind.I32, KernelArgs.kindOf(Boolean.TRUE));
        assertEquals(KernelArgs.ArgKind.I32, KernelArgs.kindOf(new int[] {3}));
        assertEquals(KernelArgs.ArgKind.F32, KernelArgs.kindOf(1.5f));
        assertEquals(KernelArgs.ArgKind.F32, KernelArgs.kindOf(new float[] {1.5f}));
        assertEquals(KernelArgs.ArgKind.I64, KernelArgs.kindOf(5L));
        assertEquals(KernelArgs.ArgKind.I64, KernelArgs.kindOf(new long[] {5L}));
        assertEquals(KernelArgs.ArgKind.F64, KernelArgs.kindOf(2.0));
        assertEquals(KernelArgs.ArgKind.F64, KernelArgs.kindOf(new double[] {2.0}));
        assertEquals(KernelArgs.ArgKind.BYTES, KernelArgs.kindOf(new int[] {1, 2}));
        assertEquals(KernelArgs.ArgKind.BYTES, KernelArgs.kindOf(new float[] {1, 2, 3}));
        assertEquals(KernelArgs.ArgKind.BYTES, KernelArgs.kindOf(new byte[] {1}));
        assertEquals(KernelArgs.ArgKind.BYTES, KernelArgs.kindOf(new short[] {1, 2}));
    }

    @Test
    public void kindOfRejectsNullEmptyAndUnsupported() {
        expectIllegal(null);
        expectIllegal(new int[0]);
        expectIllegal("text");
        expectIllegal((short) 3);
    }

    @Test
    public void snapshotCopiesArraysAndReusesMatchingStorage() {
        int[] holder = {5};
        Object snap = KernelArgs.snapshot(holder, null);
        assertNotSame(holder, snap);
        holder[0] = 9;
        assertEquals(5, ((int[]) snap)[0]);

        Object reused = KernelArgs.snapshot(holder, snap);
        assertSame(snap, reused);
        assertEquals(9, ((int[]) reused)[0]);

        Object fresh = KernelArgs.snapshot(new int[] {1, 2}, snap);
        assertNotSame(snap, fresh);
        assertArrayEquals(new int[] {1, 2}, (int[]) fresh);

        Object differentType = KernelArgs.snapshot(new float[] {1f}, snap);
        assertTrue(differentType instanceof float[]);
    }

    @Test
    public void snapshotReturnsImmutableValuesAsIs() {
        Integer boxed = 1234;
        assertSame(boxed, KernelArgs.snapshot(boxed, null));
        FakeBuffer buffer = new FakeBuffer();
        assertSame(buffer, KernelArgs.snapshot(buffer, null));
    }

    @Test
    public void sameValueComparesByContentForArraysAndIdentityForBuffers() {
        assertTrue(KernelArgs.sameValue(new int[] {1, 2}, new int[] {1, 2}));
        assertFalse(KernelArgs.sameValue(new int[] {1, 2}, new int[] {1, 3}));
        assertFalse(KernelArgs.sameValue(new int[] {1}, new float[] {1f}));
        assertTrue(KernelArgs.sameValue(3, 3));
        assertFalse(KernelArgs.sameValue(3, 3f));
        assertFalse(KernelArgs.sameValue(new FakeBuffer(), new FakeBuffer()));
    }

    @Test
    public void byteSizeMatchesPackedWidth() {
        assertEquals(4, KernelArgs.byteSize(1));
        assertEquals(4, KernelArgs.byteSize(true));
        assertEquals(4, KernelArgs.byteSize(1f));
        assertEquals(8, KernelArgs.byteSize(1L));
        assertEquals(8, KernelArgs.byteSize(1.0));
        assertEquals(12, KernelArgs.byteSize(new int[] {1, 2, 3}));
        assertEquals(6, KernelArgs.byteSize(new short[] {1, 2, 3}));
        assertEquals(3, KernelArgs.byteSize(new byte[] {1, 2, 3}));
        try {
            KernelArgs.byteSize(new FakeBuffer());
            fail("a buffer has no by-value size");
        } catch (IllegalArgumentException expected) {
            // buffers are bound, not packed
        }
    }

    @Test
    public void writeLEPacksLittleEndianWithoutMovingPosition() {
        ByteBuffer dst = ByteBuffer.allocateDirect(32).order(ByteOrder.BIG_ENDIAN);
        dst.position(3);
        assertEquals(4, KernelArgs.writeLE(dst, 0, 0x01020304));
        assertEquals(4, KernelArgs.writeLE(dst, 4, 1.0f));
        assertEquals(4, KernelArgs.writeLE(dst, 8, Boolean.TRUE));
        assertEquals(8, KernelArgs.writeLE(dst, 16, new int[] {7, 8}));
        assertEquals(3, dst.position());

        ByteBuffer le = dst.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0x04, dst.get(0));
        assertEquals(0x01020304, le.getInt(0));
        assertEquals(1.0f, le.getFloat(4), 0f);
        assertEquals(1, le.getInt(8));
        assertEquals(7, le.getInt(16));
        assertEquals(8, le.getInt(20));

        try {
            KernelArgs.writeLE(dst, 30, 1);
            fail("expected IndexOutOfBoundsException");
        } catch (IndexOutOfBoundsException expected) {
            // value does not fit
        }
    }

    @Test
    public void describeNamesTypes() {
        assertEquals("Float", KernelArgs.describe(1f));
        assertEquals("int[3]", KernelArgs.describe(new int[3]));
        assertEquals("GpuBuffer", KernelArgs.describe(new FakeBuffer()));
        assertEquals("null", KernelArgs.describe(null));
    }

    private static void expectIllegal(Object arg) {
        try {
            KernelArgs.kindOf(arg);
            fail("expected IllegalArgumentException for " + arg);
        } catch (IllegalArgumentException expected) {
            // unsupported argument is rejected
        }
    }
}
