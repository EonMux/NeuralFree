package com.eonmux.neuralfree.gpu;

import neuralfree.gpu.GpuToolFingerprint;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The build cache key of every kernel artifact task includes {@link GpuToolFingerprint}: it must identify the
 * toolchain by name and version only, so a second checkout at another path hits the cache. These tests run the
 * real process and prove the fingerprint does not depend on where the executable lives.
 */
public class GpuToolFingerprintTest {

    @Rule
    public final TemporaryFolder temporary = new TemporaryFolder();

    @Test
    public void anAbsentToolHasTheAbsentFingerprint() {
        assertEquals("absent", GpuToolFingerprint.of((File) null));
        assertEquals("absent", GpuToolFingerprint.ABSENT);
    }

    @Test
    public void nameAndVersionOutputFormTheFingerprint() {
        assertEquals("nvcc release 13.4, V13.4.92", GpuToolFingerprint.of("nvcc", " release 13.4, V13.4.92\n"));
        assertEquals("nvcc", GpuToolFingerprint.of("nvcc", "   \n"));
        assertEquals("nvcc", GpuToolFingerprint.of("nvcc", null));
        assertNotEquals(GpuToolFingerprint.of("nvcc", "release 13.4"), GpuToolFingerprint.of("nvcc", "release 12.4"));
        assertNotEquals(GpuToolFingerprint.of("nvcc", "release 13.4"), GpuToolFingerprint.of("clang", "release 13.4"));
    }

    @Test
    public void absolutePathsInVersionOutputArePlaceheld() {
        String clang = GpuToolFingerprint.of("clang", "clang version 23.1.1\nInstalledDir: /usr/bin\n");
        assertEquals("clang clang version 23.1.1\nInstalledDir: <path>", clang);
        assertFalse("the install directory must not leak into the cache key", clang.contains("/usr/bin"));

        String hipcc = GpuToolFingerprint.of("hipcc", "hipcc HIP version: 6.2\nPath: C:\\ROCm\\bin\n");
        assertEquals("hipcc hipcc HIP version: 6.2\nPath: <path>", hipcc);
    }

    @Test
    public void theExecutablePathDoesNotEnterTheFingerprint() throws Exception {
        File echo = new File("/bin/echo");
        Assume.assumeTrue("a POSIX echo is needed to prove the path is ignored", echo.canExecute());

        Path first = Files.createSymbolicLink(temporary.newFolder("first").toPath().resolve("probe"), echo.toPath());
        Path second = Files.createSymbolicLink(temporary.newFolder("second").toPath().resolve("probe"), echo.toPath());

        String fromFirst = GpuToolFingerprint.of(first.toFile());
        String fromSecond = GpuToolFingerprint.of(second.toFile());

        assertTrue("the executable name prefixes the identity: " + fromFirst, fromFirst.startsWith("probe "));
        assertEquals("the same tool reached through another directory fingerprints alike", fromFirst, fromSecond);
        assertFalse("the absolute path must not leak into the cache key", fromFirst.contains(first.toFile().getAbsolutePath()));
        assertFalse("the absolute path must not leak into the cache key", fromSecond.contains(second.toFile().getAbsolutePath()));
    }

    @Test
    public void aDifferentExecutableNameAtTheSameVersionDiffers() throws Exception {
        File echo = new File("/bin/echo");
        Assume.assumeTrue("a POSIX echo is needed to prove the name is part of the identity", echo.canExecute());

        Path probe = Files.createSymbolicLink(temporary.newFolder("names").toPath().resolve("probe"), echo.toPath());
        Path other = Files.createSymbolicLink(temporary.newFolder("names-other").toPath().resolve("other"), echo.toPath());

        assertTrue(GpuToolFingerprint.of(probe.toFile()).startsWith("probe "));
        assertTrue(GpuToolFingerprint.of(other.toFile()).startsWith("other "));
        assertNotEquals(GpuToolFingerprint.of(probe.toFile()), GpuToolFingerprint.of(other.toFile()));
    }

    @Test
    public void aMissingExecutableNameIsRejected() {
        try {
            GpuToolFingerprint.of("", "--version");
            fail("an empty executable name must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("executable name"));
        }
    }
}
