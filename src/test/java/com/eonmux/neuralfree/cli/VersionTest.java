package com.eonmux.neuralfree.cli;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The version chain the {@code --version} flag and {@link CollectEnv} print: the {@code
 * neuralfree.version} property first, then the jar manifest's implementation version, then the
 * development fallback.
 */
public class VersionTest {

    @Test
    public void thePropertyWinsOverTheManifest() {
        assertEquals("1.2.3", Version.resolve("1.2.3", "2.0.0"));
    }

    @Test
    public void theManifestNamesTheVersionWithoutAProperty() {
        assertEquals("2.0.0", Version.resolve(null, "2.0.0"));
    }

    @Test
    public void aBlankPropertyFallsThroughToTheManifest() {
        assertEquals("2.0.0", Version.resolve("   ", "2.0.0"));
        assertEquals("2.0.0", Version.resolve("", "2.0.0"));
    }

    @Test
    public void withoutAnySourceTheRunIsDevelopment() {
        assertEquals("development", Version.resolve(null, null));
        assertEquals("development", Version.resolve("", "  "));
    }

    @Test
    public void theResolvedValuesAreTrimmed() {
        assertEquals("1.0", Version.resolve(" 1.0 ", null));
        assertEquals("2.0", Version.resolve(null, " 2.0 "));
    }

    @Test
    public void theCurrentRunNamesAVersion() {
        String version = Version.current();
        assertTrue(version, version != null && !version.isEmpty());
        assertFalse(version, version.startsWith(" "));
    }

    @Test
    public void thePropertyIsTheOneTheLauncherTasksSet() {
        assertEquals("neuralfree.version", Version.PROPERTY);
    }
}
