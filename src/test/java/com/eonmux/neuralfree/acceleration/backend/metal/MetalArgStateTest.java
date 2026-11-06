package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * The argument contract of a Metal kernel handle: buffers take the first MSL buffer indices, the
 * scalar arguments are packed into one constant buffer after them, values are little-endian, and
 * the shader index's signature rejects a mismatch by name.
 */
public class MetalArgStateTest {

    private static final KernelSignature SCALE = new KernelSignature(new boolean[]{false}, new char[]{'f', 'u'});

    @Test
    public void buffersTakeTheFirstIndicesAndScalarsArePackedAfterThem() {
        MetalArgState state = new MetalArgState("scale", SCALE);
        FakeBuffer buffer = new FakeBuffer(64);
        state.setAll(new Object[]{buffer, 2.5f, 7});

        MetalArgState.Bindings bindings = state.materialize();
        assertEquals(1, bindings.bufferCount());
        assertEquals(2, bindings.scalarCount());
        assertEquals(buffer, bindings.buffers()[0]);

        ByteBuffer factor = ByteBuffer.wrap(bindings.scalars()[0]).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(2.5f, factor.getFloat(0), 0.0f);
        ByteBuffer count = ByteBuffer.wrap(bindings.scalars()[1]).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(7, count.getInt(0));

        // The MSL push-constant struct: factor at offset 0, n at offset 4.
        ByteBuffer packed = ByteBuffer.wrap(bindings.packedScalars()).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(8, packed.limit());
        assertEquals(2.5f, packed.getFloat(0), 0.0f);
        assertEquals(7, packed.getInt(4));
    }

    @Test
    public void aKernelWithoutScalarsHasAnEmptyConstantBuffer() {
        MetalArgState state = new MetalArgState("embed", new KernelSignature(new boolean[]{true}, new char[0]));
        state.setAll(new Object[]{new FakeBuffer(16)});
        assertEquals(0, state.materialize().packedScalars().length);
    }

    @Test
    public void aScalarOfTheWrongKindIsRejected() {
        MetalArgState state = new MetalArgState("scale", SCALE);
        state.setAll(new Object[]{new FakeBuffer(64), 2, 7});
        try {
            state.materialize();
            fail("an int passed for the float factor must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrueMessage(expected, "scale");
            assertTrueMessage(expected, "scalar 0");
        }
    }

    @Test
    public void aWrongArgumentCountIsRejected() {
        MetalArgState state = new MetalArgState("scale", SCALE);
        state.setAll(new Object[]{new FakeBuffer(64), 2.5f, 7, 9});
        try {
            state.materialize();
            fail("the extra argument must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrueMessage(expected, "exceeds the 2 scalar arguments");
        }
    }

    @Test
    public void aClosedBufferIsRejectedByName() {
        MetalArgState state = new MetalArgState("scale", SCALE);
        FakeBuffer buffer = new FakeBuffer(64);
        buffer.close();
        state.setAll(new Object[]{buffer, 2.5f, 7});
        try {
            state.materialize();
            fail("a closed buffer must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrueMessage(expected, "closed");
        }
    }

    @Test
    public void withoutASignatureArgumentsBindByPosition() {
        MetalArgState state = new MetalArgState("raw", null);
        state.setAll(new Object[]{new FakeBuffer(8), 11, 3.5f});
        MetalArgState.Bindings bindings = state.materialize();
        assertEquals(1, bindings.bufferCount());
        assertEquals(2, bindings.scalarCount());
        assertEquals(11, ByteBuffer.wrap(bindings.scalars()[0]).order(ByteOrder.LITTLE_ENDIAN).getInt(0));
        assertEquals(3.5f, ByteBuffer.wrap(bindings.scalars()[1]).order(ByteOrder.LITTLE_ENDIAN).getFloat(0),
                0.0f);
    }

    @Test
    public void theStoredArgumentsAreSnapshots() {
        MetalArgState state = new MetalArgState("raw", null);
        float[] values = new float[]{1.0f, 2.0f};
        state.set(0, values);
        values[0] = 9.0f;
        MetalArgState.Bindings bindings = state.materialize();
        assertEquals("the two floats are one by-value argument", 1, bindings.scalarCount());
        assertEquals(1.0f, ByteBuffer.wrap(bindings.scalars()[0]).order(ByteOrder.LITTLE_ENDIAN).getFloat(0),
                0.0f);
    }

    private static void assertTrueMessage(IllegalArgumentException e, String text) {
        if (e.getMessage() == null || !e.getMessage().contains(text)) {
            fail("expected the message to name '" + text + "': " + e.getMessage());
        }
    }

    /** A buffer with no backend state; the argument state only needs its size and open flag. */
    private static final class FakeBuffer implements MetalBuffer {
        private final long size;
        private final long id;
        private boolean closed;

        FakeBuffer(long size) {
            this.size = size;
            this.id = size;
        }

        @Override
        public long sizeInBytes() {
            return size;
        }

        @Override
        public long bufferId() {
            return closed ? -1 : id;
        }

        @Override
        public long nativeHandle() {
            if (closed) {
                throw new IllegalStateException("closed");
            }
            return id;
        }

        @Override
        public MetalComputeBackend backend() {
            return null;
        }

        @Override
        public boolean isClosed() {
            return closed;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
