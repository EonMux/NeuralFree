package com.eonmux.neuralfree.gguf.ops.gemm;

import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@code neuralfree.gemm.mmq} switch: the three documented values, the default and the refusal of
 * unknown values by name. No GPU is used.
 */
public class MmqModeTest {

    @After
    public void clearProperty() {
        System.clearProperty(MmqMode.PROPERTY);
    }

    @Test
    public void unsetPropertyDefaultsToAuto() {
        assertEquals(MmqMode.AUTO, MmqMode.configured());
    }

    @Test
    public void theThreeDocumentedValuesParse() {
        System.setProperty(MmqMode.PROPERTY, "auto");
        assertEquals(MmqMode.AUTO, MmqMode.configured());
        System.setProperty(MmqMode.PROPERTY, "on");
        assertEquals(MmqMode.ON, MmqMode.configured());
        System.setProperty(MmqMode.PROPERTY, "off");
        assertEquals(MmqMode.OFF, MmqMode.configured());
    }

    @Test
    public void blankValuesFallBackToAuto() {
        System.setProperty(MmqMode.PROPERTY, "  ");
        assertEquals(MmqMode.AUTO, MmqMode.configured());
        System.setProperty(MmqMode.PROPERTY, " off ");
        assertEquals(MmqMode.OFF, MmqMode.configured());
    }

    @Test
    public void unknownValuesAreRefusedByName() {
        System.setProperty(MmqMode.PROPERTY, "maybe");
        try {
            MmqMode.configured();
            fail("expected IllegalArgumentException for 'maybe'");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(MmqMode.PROPERTY));
            assertTrue(e.getMessage(), e.getMessage().contains("maybe"));
            assertTrue(e.getMessage(), e.getMessage().contains("auto, on, off"));
        }
        System.setProperty(MmqMode.PROPERTY, "ON");
        try {
            MmqMode.configured();
            fail("expected IllegalArgumentException for 'ON'");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("ON"));
        }
    }

    @Test
    public void onlyOffDisablesTheFastQuantLane() {
        assertTrue(MmqMode.AUTO.fastQuantGemm());
        assertTrue(MmqMode.ON.fastQuantGemm());
        assertFalse(MmqMode.OFF.fastQuantGemm());
    }

    @Test
    public void onlyAutoTakesTheCooperativeMatrixForm() {
        assertTrue(MmqMode.AUTO.prefersCm());
        assertFalse(MmqMode.ON.prefersCm());
        assertFalse(MmqMode.OFF.prefersCm());
    }
}
