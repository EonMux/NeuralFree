package com.eonmux.neuralfree.gguf.cpu.numa;

import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link NumaStrategy}: llama.cpp's three strategy names and the disabled default, parsed strictly by name.
 */
public class NumaStrategyTest {

    @Test
    public void parsesLlamaCppsStrategyNames() {
        assertEquals(NumaStrategy.DISTRIBUTE, NumaStrategy.parse("distribute"));
        assertEquals(NumaStrategy.ISOLATE, NumaStrategy.parse(" ISOLATE "));
        assertEquals(NumaStrategy.NUMACTL, NumaStrategy.parse("numactl"));
        assertEquals(NumaStrategy.DISABLED, NumaStrategy.parse("disabled"));
        assertEquals(NumaStrategy.DISABLED, NumaStrategy.parse("none"));
        assertEquals(NumaStrategy.DISABLED, NumaStrategy.parse("off"));
        for (NumaStrategy strategy : NumaStrategy.values()) {
            assertEquals(strategy, NumaStrategy.parse(strategy.label()));
        }
    }

    @Test
    public void refusesUnknownNamesNamingTheProperty() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> NumaStrategy.parse("mirror"));
        assertTrue(e.getMessage(), e.getMessage().startsWith(NumaStrategy.PROPERTY + "=mirror is invalid"));
        assertTrue(e.getMessage(), e.getMessage().contains("distribute, isolate, numactl"));
    }

    @Test
    public void absentPropertyIsDisabled() {
        Properties p = new Properties();
        assertEquals(NumaStrategy.DISABLED, NumaStrategy.fromProperties(p));
        p.setProperty(NumaStrategy.PROPERTY, "  ");
        assertEquals(NumaStrategy.DISABLED, NumaStrategy.fromProperties(p));
        p.setProperty(NumaStrategy.PROPERTY, "isolate");
        assertEquals(NumaStrategy.ISOLATE, NumaStrategy.fromProperties(p));
        p.setProperty(NumaStrategy.PROPERTY, "sometimes");
        assertThrows(IllegalArgumentException.class, () -> NumaStrategy.fromProperties(p));
    }

    @Test
    public void onlyDisabledIsInactive() {
        assertFalse(NumaStrategy.DISABLED.enabled());
        assertTrue(NumaStrategy.DISTRIBUTE.enabled());
        assertTrue(NumaStrategy.ISOLATE.enabled());
        assertTrue(NumaStrategy.NUMACTL.enabled());
    }
}
