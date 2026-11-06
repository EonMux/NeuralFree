package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import org.junit.Assume;
import org.junit.Test;

import java.io.InputStream;
import java.util.Arrays;
import java.util.HashSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The Metal kernel library against the checked-in test shaders: index parsing, kernel coverage,
 * signature resolution, MSL source loading and the wrong-backend and missing-source paths.
 */
public class MetalKernelLibraryTest {

    private static final String METAL_ROOT = "shaders/test/metal";

    private final MetalKernelLibrary library = new MetalKernelLibrary(testConfig());

    private static KernelLibraryConfig testConfig() {
        return new KernelLibraryConfig("metal-test", "shaders/test", "shaders/test/cuda", "shaders/test", false);
    }

    @Test
    public void servesOnlyMetalBackends() {
        Availability a = library.availability(new TypeBackend(ComputeBackend.BackendType.CUDA));
        assertFalse(a.available());
        assertTrue(a.reason().contains("cannot serve a CUDA backend"));

        Availability metal = library.availability(new TypeBackend(ComputeBackend.BackendType.METAL));
        assertTrue(metal.available());
    }

    @Test
    public void indexListsEveryTestKernel() {
        assertEquals(new HashSet<>(Arrays.asList("increment", "scale", "simple_add", "matmul")),
                library.kernelNames());
    }

    @Test
    public void resolveProducesMetalSourceArtifactsWithTheirSignatures() {
        ComputeBackend backend = new TypeBackend(ComputeBackend.BackendType.METAL);
        KernelArtifact scale = library.resolve("scale", backend);
        assertEquals(KernelArtifact.Kind.METAL_SOURCE, scale.kind());
        assertEquals(METAL_ROOT + "/scale.metal", scale.resourcePath());
        assertEquals(METAL_ROOT + "/scale.metal", scale.programKey());
        assertEquals("", scale.buildOptions());
        KernelSignature signature = scale.signature();
        assertEquals(1, signature.bufferCount());
        assertEquals(2, signature.scalarCount());
        assertEquals('f', signature.pushKind(0));
        assertEquals('u', signature.pushKind(1));
        assertTrue("the MSL source is loaded from the classpath", scale.sourceCode().contains("kernel void scale"));

        KernelArtifact add = library.resolve("simple_add", backend);
        assertEquals(3, add.signature().bufferCount());
        assertTrue(add.signature().isBufferReadOnly(0));
        assertTrue(add.signature().isBufferReadOnly(1));
        assertFalse(add.signature().isBufferReadOnly(2));
        assertTrue(add.sourceCode().contains("c[gid] = a[gid] + b[gid]"));
    }

    @Test
    public void signatureLookupMatchesResolve() {
        assertEquals(library.resolve("matmul", new TypeBackend(ComputeBackend.BackendType.METAL)).signature(),
                library.signature("matmul"));
    }

    @Test
    public void anUnknownKernelIsRejectedByName() {
        ComputeBackend backend = new TypeBackend(ComputeBackend.BackendType.METAL);
        try {
            library.resolve("not_a_kernel", backend);
            fail("an unknown kernel must be rejected");
        } catch (UnsupportedKernelException expected) {
            assertTrue(expected.getMessage().contains("not_a_kernel"));
            assertTrue(expected.getMessage().contains("Metal shader index"));
        }
    }

    @Test
    public void aMissingSourceFileIsNamed() {
        MetalKernelLibrary partial = new MetalKernelLibrary(testConfig(), new MissingResourceClassLoader(
                METAL_ROOT + "/scale.metal"));
        ComputeBackend backend = new TypeBackend(ComputeBackend.BackendType.METAL);
        try {
            partial.resolve("scale", backend);
            fail("a kernel whose MSL file is missing must be rejected");
        } catch (UnsupportedKernelException expected) {
            assertTrue(expected.getMessage().contains("scale.metal"));
            assertTrue(expected.getMessage().contains("missing"));
        }
        // The index still lists it, so the library is available; the gap surfaces at resolve.
        assertTrue(partial.availability(backend).available());
    }

    @Test
    public void aMissingIndexMakesTheLibraryUnavailable() {
        MetalKernelLibrary absent = new MetalKernelLibrary(new KernelLibraryConfig("metal-absent",
                "shaders/absent", "shaders/absent/cuda", "shaders/absent", false));
        Availability a = absent.availability(new TypeBackend(ComputeBackend.BackendType.METAL));
        assertFalse(a.available());
        assertTrue(a.reason().contains("Metal shaders were not built"));
        assertTrue(absent.kernelNames().isEmpty());
    }

    @Test
    public void theLlmMetalShadersCoverTheInventoryWhenGenerated() {
        Assume.assumeTrue("the Metal kernel sources were not generated",
                MetalKernelLibraryTest.class.getClassLoader().getResource(
                        "shaders/llm/metal/index.properties") != null);
        Availability a = new MetalKernelLibrary(KernelLibraries.LLM_CONFIG)
                .availability(new TypeBackend(ComputeBackend.BackendType.METAL));
        assertTrue(a.reason(), a.available());
    }

    @Test
    public void aMalformedIndexIsReportedAsUnusable() {
        MetalKernelLibrary broken = new MetalKernelLibrary(testConfig(), new MalformedIndexClassLoader(
                "broken.buffers=1\nbroken.ro=11\nbroken.push=i\n"));
        Availability a = broken.availability(new TypeBackend(ComputeBackend.BackendType.METAL));
        assertFalse(a.available());
        assertTrue("the reason names the index: " + a.reason(), a.reason().contains("is unusable"));
        assertTrue(a.reason().contains("broken"));
    }

    @Test
    public void theMetalRootSitsUnderTheShaderRoot() {
        assertEquals("shaders/llm/metal", new MetalKernelLibrary(KernelLibraries.LLM_CONFIG).metalRoot());
        assertEquals(METAL_ROOT, library.metalRoot());
    }

    /** A backend of a given type with no device state. */
    private static final class TypeBackend implements ComputeBackend {
        private final BackendType type;

        TypeBackend(BackendType type) {
            this.type = type;
        }

        @Override
        public BackendType backendType() {
            return type;
        }

        @Override
        public DeviceCapabilities capabilities() {
            throw new UnsupportedOperationException("the stub has no device");
        }

        @Override
        public com.eonmux.neuralfree.acceleration.backend.ComputeQueue computeQueue() {
            throw new UnsupportedOperationException("the stub has no queues");
        }

        @Override
        public com.eonmux.neuralfree.acceleration.backend.ComputeQueue transferQueue() {
            throw new UnsupportedOperationException("the stub has no queues");
        }

        @Override
        public com.eonmux.neuralfree.acceleration.backend.GpuBuffer allocateBuffer(long sizeInBytes,
                                                                                    boolean readOnly) {
            throw new UnsupportedOperationException("the stub allocates nothing");
        }

        @Override
        public com.eonmux.neuralfree.acceleration.backend.GpuBuffer allocateBuffer(long sizeInBytes) {
            throw new UnsupportedOperationException("the stub allocates nothing");
        }

        @Override
        public com.eonmux.neuralfree.acceleration.backend.KernelHandle compileKernel(String kernelName,
                                                                                     String kernelSource,
                                                                                     String buildOptions) {
            throw new UnsupportedOperationException("the stub compiles nothing");
        }

        @Override
        public void close() {
        }
    }

    /** Delegates to the real class loader except for the index path, which serves {@code text}. */
    private static final class MalformedIndexClassLoader extends ClassLoader {
        private final String text;

        MalformedIndexClassLoader(String text) {
            super(MetalKernelLibraryTest.class.getClassLoader());
            this.text = text;
        }

        @Override
        public InputStream getResourceAsStream(String name) {
            if ((METAL_ROOT + "/index.properties").equals(name)) {
                return new java.io.ByteArrayInputStream(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }
            return super.getResourceAsStream(name);
        }
    }

    /** Delegates to the real class loader except for one resource path, which reports no stream. */
    private static final class MissingResourceClassLoader extends ClassLoader {
        private final String missing;

        MissingResourceClassLoader(String missing) {
            super(MetalKernelLibraryTest.class.getClassLoader());
            this.missing = missing;
        }

        @Override
        public InputStream getResourceAsStream(String name) {
            if (missing.equals(name)) {
                return null;
            }
            return super.getResourceAsStream(name);
        }
    }
}
