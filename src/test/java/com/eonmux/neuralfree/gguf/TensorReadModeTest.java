package com.eonmux.neuralfree.gguf;

import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link TensorReadMode}: llama.cpp's load-mode names (including the mlock variants), the readahead-free and pinned
 * modes, and the order-independent {@code --no-mmap}/{@code --mlock} transitions.
 */
public class TensorReadModeTest {

    @Test
    public void parsesLlamaCppsLoadModeNames() {
        assertEquals(TensorReadMode.READ, TensorReadMode.parse("none"));
        assertEquals(TensorReadMode.MMAP, TensorReadMode.parse("mmap"));
        assertEquals(TensorReadMode.MLOCK, TensorReadMode.parse("mlock"));
        assertEquals(TensorReadMode.MMAP_MLOCK, TensorReadMode.parse("mmap+mlock"));
        assertEquals(TensorReadMode.DIRECT, TensorReadMode.parse("direct"));
        assertEquals(TensorReadMode.LAZY, TensorReadMode.parse("lazy"));
        assertEquals("llama-bench's none and the engine's read name the same mode",
                TensorReadMode.READ, TensorReadMode.parse("read"));
        assertEquals("the enum spelling is accepted too", TensorReadMode.MMAP_MLOCK,
                TensorReadMode.parse("MMAP_MLOCK"));
        assertEquals(TensorReadMode.MMAP_MLOCK, TensorReadMode.parse("mmap-mlock"));
    }

    @Test
    public void refusesUnknownNamesListingTheValidOnes() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> TensorReadMode.parse("unsafe"));
        assertTrue(e.getMessage(), e.getMessage().startsWith(TensorReadMode.PROP_READ_MODE + "=unsafe is invalid"));
        assertTrue(e.getMessage(), e.getMessage().contains("mmap+mlock"));
    }

    @Test
    public void everyModeRoundTripsThroughItsFlagName() {
        for (TensorReadMode mode : TensorReadMode.values()) {
            assertEquals(mode, TensorReadMode.parse(mode.flagName()));
        }
    }

    @Test
    public void mapsAndPinsAreReportedPerMode() {
        assertTrue(TensorReadMode.MMAP.usesMmap());
        assertTrue(TensorReadMode.MMAP_MLOCK.usesMmap());
        assertFalse(TensorReadMode.MLOCK.usesMmap());
        assertFalse(TensorReadMode.READ.usesMmap());
        assertTrue(TensorReadMode.MLOCK.locksMemory());
        assertTrue(TensorReadMode.MMAP_MLOCK.locksMemory());
        assertFalse(TensorReadMode.MMAP.locksMemory());
        assertFalse(TensorReadMode.READ.locksMemory());
        assertFalse(TensorReadMode.DIRECT.locksMemory());
        assertFalse(TensorReadMode.LAZY.locksMemory());
    }

    @Test
    public void noMmapAndMlockComposeInAnyOrder() {
        assertEquals("--no-mmap alone", TensorReadMode.READ, TensorReadMode.MMAP.withMmap(false));
        assertEquals("mmap back on", TensorReadMode.MMAP, TensorReadMode.READ.withMmap(true));
        assertEquals("--no-mmap of mmap+mlock", TensorReadMode.MLOCK, TensorReadMode.MMAP_MLOCK.withMmap(false));
        assertEquals("--mlock after --no-mmap", TensorReadMode.MLOCK, TensorReadMode.READ.withMlock(true));
        assertEquals("--no-mmap after --mlock", TensorReadMode.MLOCK, TensorReadMode.MMAP_MLOCK.withMmap(false));
        assertEquals("--mlock over mmap", TensorReadMode.MMAP_MLOCK, TensorReadMode.MMAP.withMlock(true));
        assertEquals("turning a pin off", TensorReadMode.MMAP, TensorReadMode.MMAP_MLOCK.withMlock(false));
        assertEquals("turning a read pin off", TensorReadMode.READ, TensorReadMode.MLOCK.withMlock(false));
        assertEquals("direct keeps its reader", TensorReadMode.DIRECT, TensorReadMode.DIRECT.withMmap(false));
        assertEquals("lazy keeps its reader", TensorReadMode.LAZY, TensorReadMode.LAZY.withMmap(true));
    }

    @Test
    public void mlockIsRefusedForModesWithoutResidentTensorBytes() {
        IllegalArgumentException direct = assertThrows(IllegalArgumentException.class,
                () -> TensorReadMode.DIRECT.withMlock(true));
        assertTrue(direct.getMessage(), direct.getMessage().contains("direct I/O bypasses the page cache"));

        IllegalArgumentException lazy = assertThrows(IllegalArgumentException.class,
                () -> TensorReadMode.LAZY.withMlock(true));
        assertTrue(lazy.getMessage(), lazy.getMessage().contains("lazily read tensors"));
    }

    @Test
    public void thePropertyReadsTheFullModeAndWinsOverTheBooleanShorthand() {
        Properties p = new Properties();
        assertEquals(TensorReadMode.MMAP, TensorReadMode.fromProperties(p));
        p.setProperty(TensorReadMode.PROP_DISABLE_MMAP, "true");
        assertEquals(TensorReadMode.READ, TensorReadMode.fromProperties(p));
        p.setProperty(TensorReadMode.PROP_READ_MODE, "mmap+mlock");
        assertEquals("the full form decides", TensorReadMode.MMAP_MLOCK, TensorReadMode.fromProperties(p));
        p.setProperty(TensorReadMode.PROP_DISABLE_MMAP, "not-a-boolean");
        assertEquals("the full form ignores the shorthand", TensorReadMode.MMAP_MLOCK,
                TensorReadMode.fromProperties(p));
        p.remove(TensorReadMode.PROP_READ_MODE);
        assertThrows(IllegalArgumentException.class, () -> TensorReadMode.fromProperties(p));
    }
}
