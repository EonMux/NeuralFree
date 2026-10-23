package com.eonmux.neuralfree.acceleration.kernels;

import org.junit.Test;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Parsing, validation, caching and inventory reading of {@link KernelManifest}.
 */
public class KernelManifestTest {

    @Test
    public void parsesEntriesTagsAndComments() {
        KernelManifest m = KernelManifest.parse("lib",
                "# header comment\n"
                        + "common/a.h\n"
                        + "\n"
                        + "  b.cl   # trailing comment\n"
                        + "[opencl] legacy/c.cl\r\n"
                        + "[opencl]d.cl\n", "test");
        assertEquals("lib", m.root());
        assertEquals(Arrays.asList(
                new KernelManifest.Entry("common/a.h", false),
                new KernelManifest.Entry("b.cl", false),
                new KernelManifest.Entry("legacy/c.cl", true),
                new KernelManifest.Entry("d.cl", true)), m.entries());
        assertEquals(Arrays.asList("common/a.h", "b.cl"), m.portablePaths());
        assertEquals(Arrays.asList("legacy/c.cl", "d.cl"), m.openClOnlyPaths());
        assertEquals("lib/legacy/c.cl", m.resourcePath(m.entries().get(2)));
    }

    @Test
    public void sharedEntriesResolveAgainstTheSharedRoot() {
        KernelManifest m = KernelManifest.parse("lib",
                "[shared] common/dialect.h\n"
                        + "[shared]common/extra.h  # comment\n"
                        + "common/a.h\n"
                        + "b.cl\n"
                        + "[opencl] c.cl\n", "test");
        assertEquals(Arrays.asList(
                new KernelManifest.Entry("common/dialect.h", false, true),
                new KernelManifest.Entry("common/extra.h", false, true),
                new KernelManifest.Entry("common/a.h", false),
                new KernelManifest.Entry("b.cl", false),
                new KernelManifest.Entry("c.cl", true)), m.entries());
        KernelManifest.Entry dialect = m.entries().get(0);
        assertTrue(dialect.shared());
        assertFalse(dialect.openClOnly());
        assertEquals("common/dialect.h", dialect.path());
        assertEquals("[shared] common/dialect.h", dialect.toString());
        assertEquals(KernelManifest.SHARED_ROOT + "/common/dialect.h", dialect.resourcePath("lib"));
        assertEquals("kernels/common/dialect.h", m.resourcePath(dialect));
        assertEquals("lib/common/a.h", m.entries().get(2).resourcePath("lib"));
        assertFalse(m.entries().get(2).shared());
        assertEquals(Arrays.asList("common/dialect.h", "common/extra.h", "common/a.h", "b.cl"), m.portablePaths());
        assertEquals(Arrays.asList("c.cl"), m.openClOnlyPaths());
        for (KernelManifest.Entry e : new KernelManifest.Entry[]{dialect, m.entries().get(2)}) {
            try {
                e.resourcePath("lib/");
                fail("a malformed library root must be rejected for " + e);
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("root"));
            }
        }
    }

    @Test
    public void sharedAndLibraryEntriesDiffer() {
        KernelManifest.Entry shared = new KernelManifest.Entry("common/d.h", false, true);
        KernelManifest.Entry library = new KernelManifest.Entry("common/d.h", false);
        assertFalse(shared.equals(library));
        assertEquals(shared, new KernelManifest.Entry("common/d.h", false, true));
        assertEquals(shared.hashCode(), new KernelManifest.Entry("common/d.h", false, true).hashCode());
        try {
            new KernelManifest.Entry("common/d.h", true, true);
            fail("an entry cannot be both OpenCL-only and shared");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("mutually exclusive"));
        }
    }

    @Test
    public void entriesAreUnmodifiable() {
        KernelManifest m = KernelManifest.parse("lib", "a.cl\n", "test");
        try {
            m.entries().add(new KernelManifest.Entry("b.cl", false));
            fail("entries must be unmodifiable");
        } catch (UnsupportedOperationException expected) {
            assertEquals(1, m.entries().size());
        }
    }

    @Test
    public void rejectsMalformedManifests() {
        assertRejected("[cuda] a.cl\n", "unknown tag");
        assertRejected("a.cl\na.cl\n", "duplicate entry");
        assertRejected("/abs/a.cl\n", "relative");
        assertRejected("x/../a.cl\n", "'..'");
        assertRejected("x//a.cl\n", "empty");
        assertRejected("a b.cl\n", "whitespace");
        assertRejected("# only comments\n\n", "no source files");
        assertRejected("[shared] [opencl] a.h\n", "mutually exclusive");
        assertRejected("[opencl] [shared] a.h\n", "mutually exclusive");
        assertRejected("[shared] [shared] a.h\n", "repeated tag");
        assertRejected("a.cl\n[shared] common/d.h\n", "must precede every library entry");
        assertRejected("[shared] ../d.h\n", "'..'");
        assertRejected("[shared] /d.h\n", "relative");
        assertRejected("[shared] common/d.h\ncommon/d.h\n", "duplicate entry");
    }

    @Test
    public void errorsNameTheLine() {
        try {
            KernelManifest.parse("lib", "a.cl\n\n[bogus] b.cl\n", "origin.txt");
            fail("an unknown tag must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().startsWith("origin.txt:3:"));
        }
    }

    @Test
    public void rejectsInvalidRoots() {
        for (String root : new String[]{null, "", " ", "/kernels/llm", "kernels/llm/"}) {
            try {
                KernelManifest.parse(root, "a.cl\n", "test");
                fail("root '" + root + "' must be rejected");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("root"));
            }
        }
    }

    @Test
    public void loadsShippedLlmManifestInDependencyOrder() {
        KernelManifest m = KernelManifest.load(getClass().getClassLoader(), KernelLibraries.LLM_CONFIG.sourceRoot());
        List<KernelManifest.Entry> entries = m.entries();
        assertEquals("common/dialect.h", entries.get(0).path());
        assertEquals("common/config.h", entries.get(1).path());
        boolean seenKernelFile = false;
        for (KernelManifest.Entry e : entries) {
            if (e.path().endsWith(".cl")) {
                seenKernelFile = true;
            } else {
                assertFalse("header " + e.path() + " must precede every .cl file", seenKernelFile);
                assertTrue("headers live in common/: " + e.path(), e.path().startsWith("common/"));
            }
        }
    }

    @Test
    public void shippedLlmManifestTakesTheDialectFromTheSharedRoot() {
        ClassLoader cl = getClass().getClassLoader();
        KernelManifest m = KernelManifest.load(cl, KernelLibraries.LLM_CONFIG.sourceRoot());
        KernelManifest.Entry dialect = m.entries().get(0);
        assertTrue(dialect + " must be a [shared] entry", dialect.shared());
        assertEquals("kernels/common/dialect.h", m.resourcePath(dialect));
        assertTrue("classpath:kernels/common/dialect.h must ship", cl.getResource(m.resourcePath(dialect)) != null);
    }

    @Test
    public void loadIsCachedPerClassLoaderAndRoot() {
        ClassLoader cl = getClass().getClassLoader();
        assertSame(KernelManifest.load(cl, "kernels/llm"), KernelManifest.load(cl, "kernels/llm"));
    }

    @Test
    public void missingManifestIsReported() {
        try {
            KernelManifest.load(getClass().getClassLoader(), "kernels/no_such_library");
            fail("a missing manifest must be reported");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("kernels/no_such_library/manifest.txt"));
        }
    }

    @Test
    public void parsesInventoryFirstColumn() {
        Set<String> names = KernelManifest.parseInventory(
                "# kernel\tscalars\tbuffers\nQ8_0_MatVec\t2\t3\n\nLLM_RMSNorm 5 2\nLLM_Softmax\n", "inv");
        assertEquals(Arrays.asList("Q8_0_MatVec", "LLM_RMSNorm", "LLM_Softmax"), Arrays.asList(names.toArray()));
    }

    @Test
    public void rejectsMalformedInventory() {
        assertInventoryRejected("A\nA\n", "duplicate");
        assertInventoryRejected("9abc\n", "not a kernel name");
        assertInventoryRejected("a-b\t1\t1\n", "not a kernel name");
    }

    @Test
    public void inventoryIsEmptyWhenAbsent() throws IOException {
        Path dir = Files.createTempDirectory("nf-manifest");
        try (URLClassLoader cl = loaderFor(dir)) {
            write(dir.resolve("lib/manifest.txt"), "a.cl\n");
            assertFalse(KernelManifest.inventory(cl, "lib").isPresent());
        }
    }

    @Test
    public void inventoryIsReadFromTheLibraryRoot() throws IOException {
        Path dir = Files.createTempDirectory("nf-manifest");
        try (URLClassLoader cl = loaderFor(dir)) {
            write(dir.resolve("lib/inventory.txt"), "K1\t1\t2\nK2\t0\t1\n");
            Optional<Set<String>> inv = KernelManifest.inventory(cl, "lib");
            assertTrue(inv.isPresent());
            assertEquals(2, inv.get().size());
            assertTrue(inv.get().contains("K1") && inv.get().contains("K2"));
        }
    }

    @Test
    public void inventoryCoverageNamesMissingKernels() throws IOException {
        Path dir = Files.createTempDirectory("nf-manifest");
        try (URLClassLoader cl = loaderFor(dir)) {
            write(dir.resolve("lib/inventory.txt"), "K1\nK2\nK3\n");
            Availability complete = KernelManifest.inventoryCoverage(cl, "llm", "lib",
                    new java.util.HashSet<>(Arrays.asList("K1", "K2", "K3", "Extra")), "the index");
            assertTrue(complete.available());
            Availability partial = KernelManifest.inventoryCoverage(cl, "llm", "lib",
                    new java.util.HashSet<>(Arrays.asList("K2")), "the index");
            assertFalse(partial.available());
            assertTrue(partial.reason(), partial.reason().startsWith("LLM kernel inventory incomplete: 2 of 3"));
            assertTrue(partial.reason(), partial.reason().contains("K1, K3"));
            Availability noInventory = KernelManifest.inventoryCoverage(cl, "llm", "other",
                    new java.util.HashSet<String>(), "the index");
            assertFalse(noInventory.available());
            assertTrue(noInventory.reason(), noInventory.reason().contains("classpath:other/inventory.txt is missing"));
        }
    }

    private static void assertRejected(String text, String messagePart) {
        try {
            KernelManifest.parse("lib", text, "test");
            fail("manifest must be rejected: " + text);
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(messagePart));
        }
    }

    private static void assertInventoryRejected(String text, String messagePart) {
        try {
            KernelManifest.parseInventory(text, "inv");
            fail("inventory must be rejected: " + text);
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(messagePart));
        }
    }

    static URLClassLoader loaderFor(Path dir) throws IOException {
        return new URLClassLoader(new URL[]{dir.toUri().toURL()}, null);
    }

    static void write(Path file, String content) throws IOException {
        Files.createDirectories(file.getParent());
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
    }
}
