package com.eonmux.neuralfree.acceleration.backend;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Parsing of the shader-index and build-inventory signature forms, and the argument contract
 * ({@link KernelSignature#validateArgs}, {@link KernelSignature#checkScalar},
 * {@link KernelSignature#checkCounts}) the Vulkan and CUDA backends share.
 */
public class KernelSignatureTest {

    @Test
    public void parsesShaderIndexEntries() {
        KernelSignature s = KernelSignature.parse("3", "101", "i,f,u");

        assertEquals(3, s.bufferCount());
        assertArrayEquals(new boolean[]{true, false, true}, s.bufferReadOnly());
        assertArrayEquals(new char[]{'i', 'f', 'u'}, s.pushKinds());
        assertEquals(12, s.pushBytes());
        assertEquals(6, s.argCount());
    }

    @Test
    public void parsesEmptyColumns() {
        KernelSignature s = KernelSignature.parse("0", "", "");

        assertEquals(0, s.bufferCount());
        assertEquals(0, s.scalarCount());
        assertEquals(0, s.argCount());
    }

    @Test
    public void rejectsShaderIndexWithMismatchedReadOnlyFlags() {
        expectIllegal(() -> KernelSignature.parse("2", "1", "i"), "have 1 entries but there are 2 buffers");
    }

    @Test
    public void parsesInventoryColumnsWithEveryBufferWritten() {
        KernelSignature s = KernelSignature.parseInventory("3", "3", "i,i,f");

        assertEquals(3, s.bufferCount());
        assertArrayEquals(new boolean[3], s.bufferReadOnly());
        assertArrayEquals(new char[]{'i', 'i', 'f'}, s.pushKinds());
    }

    @Test
    public void parsesInventoryLineWithoutScalars() {
        KernelSignature s = KernelSignature.parseInventory("2", "0", "");

        assertEquals(2, s.bufferCount());
        assertEquals(0, s.scalarCount());
    }

    @Test
    public void rejectsInventoryWhoseScalarCountDisagreesWithItsKinds() {
        expectIllegal(() -> KernelSignature.parseInventory("1", "3", "i,f"), "scalar count 3 disagrees");
    }

    @Test
    public void rejectsInventoryWithUnknownKindOrBadCount() {
        expectIllegal(() -> KernelSignature.parseInventory("1", "1", "d"), "unknown scalar kind 'd'");
        expectIllegal(() -> KernelSignature.parseInventory("x", "0", ""), "buffer count 'x' is not an integer");
        expectIllegal(() -> KernelSignature.parseInventory("1", "-1", ""), "scalar count must not be negative");
    }

    @Test
    public void validateArgsAcceptsInterleavedBuffersAndScalars() {
        KernelSignature s = KernelSignature.parseInventory("2", "3", "i,f,u");

        s.validateArgs("K", new Object[]{buffer(), 1, 2.0f, buffer(), true});
        s.validateArgs("K", new Object[]{new int[]{1}, buffer(), new float[]{2f}, 3, buffer()});
    }

    @Test
    public void validateArgsNamesTheFirstMismatch() {
        KernelSignature s = KernelSignature.parseInventory("1", "2", "i,f");

        expectIllegal(() -> s.validateArgs("K", new Object[]{buffer(), 1.0f, 2.0f}),
                "K: arg 1 (Float) does not match scalar 0 (int32)");
        expectIllegal(() -> s.validateArgs("K", new Object[]{buffer(), 1, 2.0}),
                "K: arg 2 (Double) does not match scalar 1 (float32)");
        expectIllegal(() -> s.validateArgs("K", new Object[]{buffer(), 1, 2.0f, 3}),
                "K: arg 3 (Integer) exceeds the 2 scalar arguments of the signature");
        expectIllegal(() -> s.validateArgs("K", new Object[]{1, 2.0f}),
                "K: got 0 buffers and 2 scalars, signature expects 1 buffers and 2 scalars");
        expectIllegal(() -> s.validateArgs("K", new Object[]{buffer(), new int[]{1, 2}, 2.0f}),
                "K: arg 1 (int[2]) does not match scalar 0 (int32)");
    }

    @Test
    public void inventoryLayoutChecksEachPosition() {
        KernelSignature s = KernelSignature.parseInventory("2", "3", "i,f,u", "iBfuB");

        assertTrue(s.hasLayout());
        assertFalse(KernelSignature.parseInventory("2", "3", "i,f,u").hasLayout());
        s.validateArgs("K", new Object[]{1, buffer(), 2.0f, 3, buffer()});
        expectIllegal(() -> s.validateArgs("K", new Object[]{buffer(), 1, 2.0f, 3, buffer()}),
                "K: arg 0 (" + KernelArgs.describe(buffer()) + ") is a buffer but parameter 0 is declared as a scalar");
        expectIllegal(() -> s.validateArgs("K", new Object[]{1, 2.0f, buffer(), 3, buffer()}),
                "K: arg 1 (Float) is a scalar but parameter 1 is declared as a buffer");
        assertFalse(s.equals(KernelSignature.parseInventory("2", "3", "i,f,u")));
        assertEquals(s, KernelSignature.parseInventory("2", "3", "i,f,u", "iBfuB"));
        assertTrue(s.toString().contains("layout=iBfuB"));
    }

    @Test
    public void rejectsInventoryLayoutThatDisagreesWithTheCounts() {
        expectIllegal(() -> KernelSignature.parseInventory("1", "1", "i", "BB"), "layout 'BB' has 2 buffers");
        expectIllegal(() -> KernelSignature.parseInventory("1", "2", "i,f", "fBi"), "scalars 'fi'");
        expectIllegal(() -> KernelSignature.parseInventory("1", "1", "i", "Bx"), "has 'x' at parameter 1");
        KernelSignature empty = KernelSignature.parseInventory("0", "0", "", "");
        assertTrue(empty.hasLayout());
        assertEquals(0, empty.argCount());
    }

    @Test
    public void scalarKindsMapToArgumentKinds() {
        KernelSignature s = KernelSignature.parse("0", "", "i,u,f");

        assertEquals(KernelArgs.ArgKind.I32, s.scalarArgKind(0));
        assertEquals(KernelArgs.ArgKind.I32, s.scalarArgKind(1));
        assertEquals(KernelArgs.ArgKind.F32, s.scalarArgKind(2));
        assertEquals("int32", s.scalarTypeName(0));
        assertEquals("uint32", s.scalarTypeName(1));
        assertEquals("float32", s.scalarTypeName(2));
    }

    @Test
    public void equalityCoversAccessAndKinds() {
        assertEquals(KernelSignature.parse("2", "00", "i"), KernelSignature.parseInventory("2", "1", "i"));
        assertFalse(KernelSignature.parse("2", "10", "i").equals(KernelSignature.parseInventory("2", "1", "i")));
        assertFalse(KernelSignature.parse("2", "00", "f").equals(KernelSignature.parse("2", "00", "i")));
        assertTrue(KernelSignature.parse("1", "1", "").toString().contains("ro=1"));
    }

    private static GpuBuffer buffer() {
        return new GpuBuffer() {
            @Override
            public long sizeInBytes() {
                return 4;
            }

            @Override
            public long bufferId() {
                return 1;
            }

            @Override
            public void close() {
            }
        };
    }

    private static void expectIllegal(Runnable r, String fragment) {
        try {
            r.run();
            fail("expected IllegalArgumentException containing '" + fragment + "'");
        } catch (IllegalArgumentException e) {
            assertTrue("message '" + e.getMessage() + "' lacks '" + fragment + "'", e.getMessage().contains(fragment));
        }
    }
}
