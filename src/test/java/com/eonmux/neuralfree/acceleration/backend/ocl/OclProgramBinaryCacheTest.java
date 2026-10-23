package com.eonmux.neuralfree.acceleration.backend.ocl;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Key derivation, the file format, the write/read round trip and the rejection of corrupt entries of
 * {@link OclProgramBinaryCache}, in a temporary directory. No OpenCL device is used.
 */
public class OclProgramBinaryCacheTest {

    private Path dir;

    @Before
    public void createDirectory() throws IOException {
        dir = Files.createTempDirectory("nf-oclbin-test");
    }

    @After
    public void deleteDirectory() throws IOException {
        if (dir == null || !Files.exists(dir)) {
            return;
        }
        List<Path> files = new ArrayList<>();
        try (DirectoryStream<Path> s = Files.newDirectoryStream(dir)) {
            for (Path p : s) {
                files.add(p);
            }
        }
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
        Files.deleteIfExists(dir);
    }

    private OclProgramBinaryCache cache(String driver) {
        return new OclProgramBinaryCache(dir, "NVIDIA Corporation", "NVIDIA GeForce RTX 5080 Laptop GPU", driver, true);
    }

    @Test
    public void keyIsDeterministicAndCoversSourceOptionsAndDriver() {
        OclProgramBinaryCache a = cache("610.57");
        String key = a.keyFor("kernel void k() {}", "-cl-std=CL1.2");

        assertEquals(key, cache("610.57").keyFor("kernel void k() {}", "-cl-std=CL1.2"));
        assertNotEquals(key, a.keyFor("kernel void k() { }", "-cl-std=CL1.2"));
        assertNotEquals(key, a.keyFor("kernel void k() {}", "-cl-std=CL1.2 -DUSE_SUBGROUPS=1"));
        assertNotEquals(key, cache("610.58").keyFor("kernel void k() {}", "-cl-std=CL1.2"));
        assertEquals(a.keyFor("s", null), a.keyFor("s", ""));
    }

    @Test
    public void keyHasAReadableDevicePrefixAndADigest() {
        String key = cache("610.57").keyFor("src", "");

        assertTrue(key, key.startsWith("nvidia_corporation-nvidia_geforce_rtx_5080_laptop_gpu-610.57-"));
        assertTrue(key, key.endsWith(".bin"));
        String digest = key.substring(key.lastIndexOf('-') + 1, key.length() - ".bin".length());
        assertEquals(64, digest.length());
        assertTrue(digest, digest.matches("[0-9a-f]{64}"));
    }

    @Test
    public void sourceSeparatorPreventsCollisionsBetweenSourceAndOptions() {
        OclProgramBinaryCache c = cache("1");

        assertNotEquals(c.keyFor("ab", "c"), c.keyFor("a", "bc"));
    }

    @Test
    public void labelsAreFileNameSafeAndBounded() {
        assertEquals("unknown", OclProgramBinaryCache.label(null));
        assertEquals("unknown", OclProgramBinaryCache.label("  ///  "));
        assertEquals("intel_r_corporation", OclProgramBinaryCache.label("Intel(R) Corporation"));
        assertEquals(48, OclProgramBinaryCache.label(String.join("", Collections.nCopies(100, "x"))).length());
    }

    @Test
    public void storedBinaryLoadsBack() {
        OclProgramBinaryCache c = cache("610.57");
        String key = c.keyFor("src", "-O");
        byte[] binary = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        assertNull(c.load(key));
        c.store(key, binary);
        assertArrayEquals(binary, c.load(key));
        assertTrue(Files.isRegularFile(dir.resolve(key)));
    }

    @Test
    public void corruptEntryIsDeletedAndReportedAsMiss() throws IOException {
        OclProgramBinaryCache c = cache("610.57");
        String key = c.keyFor("src", "");
        c.store(key, new byte[]{10, 20, 30, 40});
        Path file = dir.resolve(key);
        byte[] bytes = Files.readAllBytes(file);
        bytes[bytes.length - 1] ^= 0x7f;
        Files.write(file, bytes);

        assertNull(c.load(key));
        assertFalse(Files.exists(file));
    }

    @Test
    public void truncatedEntryIsDeletedAndReportedAsMiss() throws IOException {
        OclProgramBinaryCache c = cache("610.57");
        String key = c.keyFor("src", "");
        c.store(key, new byte[]{10, 20, 30, 40});
        Path file = dir.resolve(key);
        byte[] bytes = Files.readAllBytes(file);
        Files.write(file, Arrays.copyOf(bytes, bytes.length - 1));

        assertNull(c.load(key));
        assertFalse(Files.exists(file));
    }

    @Test
    public void formatRejectsWrongMagicLengthAndDigest() {
        byte[] payload = {9, 8, 7};
        byte[] encoded = OclProgramBinaryCache.encode(payload);

        assertArrayEquals(payload, OclProgramBinaryCache.decode(encoded));
        assertNull(OclProgramBinaryCache.decode(null));
        assertNull(OclProgramBinaryCache.decode(Arrays.copyOf(encoded, encoded.length - payload.length)));

        byte[] badMagic = encoded.clone();
        badMagic[0] ^= 1;
        assertNull(OclProgramBinaryCache.decode(badMagic));

        byte[] longer = Arrays.copyOf(encoded, encoded.length + 1);
        assertNull(OclProgramBinaryCache.decode(longer));

        byte[] badDigest = encoded.clone();
        badDigest[badDigest.length - payload.length - 1] ^= 1;
        assertNull(OclProgramBinaryCache.decode(badDigest));
    }

    @Test
    public void invalidateRemovesTheEntry() {
        OclProgramBinaryCache c = cache("610.57");
        String key = c.keyFor("src", "");
        c.store(key, new byte[]{1});

        c.invalidate(key);

        assertNull(c.load(key));
        assertFalse(Files.exists(dir.resolve(key)));
    }

    @Test
    public void emptyBinaryIsNotStored() {
        OclProgramBinaryCache c = cache("610.57");
        String key = c.keyFor("src", "");

        c.store(key, new byte[0]);

        assertFalse(Files.exists(dir.resolve(key)));
    }

    @Test
    public void disabledCacheIsANoOp() {
        OclProgramBinaryCache c = new OclProgramBinaryCache(dir, "v", "d", "1", false);
        String key = c.keyFor("src", "");

        c.store(key, new byte[]{1, 2});

        assertFalse(c.isEnabled());
        assertNull(c.load(key));
        assertFalse(Files.exists(dir.resolve(key)));
    }

    @Test
    public void enabledCacheNeedsADirectory() {
        try {
            new OclProgramBinaryCache(null, "v", "d", "1", true);
            fail("an enabled cache without a directory must be rejected");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("directory"));
        }
    }

    @Test
    public void keysThatEscapeTheDirectoryAreRejected() {
        OclProgramBinaryCache c = cache("610.57");
        for (String bad : new String[]{"../x.bin", "a/b.bin", "a\\b.bin", "x.txt", ""}) {
            try {
                c.load(bad);
                fail("key '" + bad + "' must be rejected");
            } catch (IllegalArgumentException e) {
                assertTrue(e.getMessage(), e.getMessage().contains("invalid OpenCL binary cache key"));
            }
        }
    }

    @Test
    public void propertyDisablesTheDefaultCache() {
        String saved = System.getProperty(OclProgramBinaryCache.ENABLED_PROPERTY);
        try {
            System.setProperty(OclProgramBinaryCache.ENABLED_PROPERTY, "false");
            assertFalse(OclProgramBinaryCache.forDevice("v", "d", "1").isEnabled());
        } finally {
            if (saved == null) {
                System.clearProperty(OclProgramBinaryCache.ENABLED_PROPERTY);
            } else {
                System.setProperty(OclProgramBinaryCache.ENABLED_PROPERTY, saved);
            }
        }
    }
}
