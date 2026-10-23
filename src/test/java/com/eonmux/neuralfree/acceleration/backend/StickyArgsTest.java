package com.eonmux.neuralfree.acceleration.backend;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class StickyArgsTest {

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
    public void setSnapshotsAndTracksCompleteness() {
        StickyArgs args = new StickyArgs("k");
        assertTrue(args.isComplete());
        assertEquals(0, args.count());

        int[] holder = {3};
        args.set(1, holder);
        assertEquals(2, args.count());
        assertFalse(args.isComplete());

        holder[0] = 99;
        assertArrayEquals(new int[] {3}, (int[]) args.get(1));
        assertEquals(KernelArgs.ArgKind.I32, args.kind(1));

        args.set(0, new FakeBuffer());
        assertTrue(args.isComplete());
        args.requireComplete();
    }

    @Test
    public void requireCompleteNamesMissingIndex() {
        StickyArgs args = new StickyArgs("LLM_Kernel");
        args.set(2, 1f);
        try {
            args.requireComplete();
            fail("expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("arg 0"));
            assertTrue(e.getMessage().contains("LLM_Kernel"));
        }
    }

    @Test
    public void nullAndUnsupportedArgsAreRejected() {
        StickyArgs args = new StickyArgs("k");
        try {
            args.set(0, null);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertEquals("arg 0 of k is null", e.getMessage());
        }
        try {
            args.set(0, "text");
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // unsupported type
        }
        try {
            args.setAll(new Object[] {1, null});
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // null element
        }
        assertEquals(0, args.count());
    }

    @Test
    public void versionsTrackValueAndLayoutChanges() {
        StickyArgs args = new StickyArgs("k");
        args.setAll(new Object[] {1, 2f});
        long v = args.version();
        long layout = args.layoutVersion();

        args.set(0, 1);
        assertEquals("equal value changes nothing", v, args.version());

        args.set(0, 5);
        assertTrue(args.version() > v);
        assertEquals("same kind keeps the layout", layout, args.layoutVersion());

        args.set(0, 5f);
        assertTrue(args.layoutVersion() > layout);
    }

    @Test
    public void dirtyBitsFollowChanges() {
        StickyArgs args = new StickyArgs("k");
        args.setAll(new Object[] {1, 2, 3});
        assertEquals(3, args.dirty().cardinality());
        args.clearDirty();
        assertTrue(args.dirty().isEmpty());

        args.set(1, 7);
        assertTrue(args.dirty().get(1));
        assertEquals(1, args.dirty().cardinality());
    }

    @Test
    public void setAllShrinksTheArgumentList() {
        StickyArgs args = new StickyArgs("k");
        args.setAll(new Object[] {1, 2, 3, 4});
        long layout = args.layoutVersion();
        args.setAll(new Object[] {1, 2});
        assertEquals(2, args.count());
        assertFalse(args.isSet(3));
        assertTrue(args.layoutVersion() > layout);
        assertTrue(args.isComplete());
        try {
            args.get(2);
            fail("expected IllegalStateException");
        } catch (IllegalStateException expected) {
            // cleared by setAll
        }
    }

    @Test
    public void growsBeyondInitialCapacity() {
        StickyArgs args = new StickyArgs("k");
        args.set(40, 1);
        assertEquals(41, args.count());
        assertEquals(1, args.get(40));
        assertFalse(args.isComplete());
    }
}
