package com.eonmux.neuralfree.acceleration.backend;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class LaunchGeometryTest {

    @Test
    public void matchesTheSizesItWasDerivedFromByValue() {
        long[] global = {256, 4};
        long[] local = {64, 1};
        LaunchGeometry g = new LaunchGeometry(global, local, new long[]{4, 4, 1}, new long[]{64, 1, 1});
        assertTrue(g.matches(new long[]{256, 4}, new long[]{64, 1}));
        global[0] = 512;
        local[0] = 128;
        assertTrue(g.matches(new long[]{256, 4}, new long[]{64, 1}));
        assertFalse(g.matches(global, new long[]{64, 1}));
        assertFalse(g.matches(new long[]{256, 4}, null));
        assertFalse(g.matches(new long[]{256}, new long[]{64}));
        assertArrayEquals(new long[]{4, 4, 1}, g.grid());
        assertArrayEquals(new long[]{64, 1, 1}, g.block());
    }

    @Test
    public void defaultLocalSizeMatchesOnlyANullLocal() {
        LaunchGeometry g = new LaunchGeometry(new long[]{100}, null, new long[]{1, 1, 1}, new long[]{100, 1, 1});
        assertTrue(g.matches(new long[]{100}, null));
        assertFalse(g.matches(new long[]{100}, new long[]{100}));
    }

    @Test
    public void refusesGridsAndBlocksThatAreNotThreeDimensional() {
        try {
            new LaunchGeometry(new long[]{8}, null, new long[]{1}, new long[]{8, 1, 1});
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // padded geometry only
        }
    }
}
