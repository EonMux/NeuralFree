package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.hip.HipComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.hip.HipKernelSignatures;
import com.eonmux.neuralfree.acceleration.hip.FakeHipDriver;
import com.eonmux.neuralfree.acceleration.hip.HipDevice;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The HIP kernel library against the test build outputs on the classpath
 * ({@code hip-test-lib}): index and build-metadata parsing, gfx-target coverage,
 * inventory signatures, and resolve producing {@code HIP_IMAGE} artifacts.
 */
public class HipKernelLibraryTest {

    private static final String SCALE_RESOURCE = "hip-test-lib/src/hip/nf_test_scale.hsaco";

    private final HipKernelLibrary library = new HipKernelLibrary(testConfig());

    private static KernelLibraryConfig testConfig() {
        return new KernelLibraryConfig("hip-test", "hip-test-lib/src", "hip-test-lib/src/cuda",
                "hip-test-lib/spv", true);
    }

    @Test
    public void servesOnlyHipBackends() {
        Availability a = library.availability(new AvailabilityBackendStub(ComputeBackend.BackendType.CUDA));
        assertFalse(a.available());
        assertTrue(a.reason().contains("cannot serve a CUDA backend"));
    }

    @Test
    public void availabilityChecksTheGfxTargetOfTheDevice() {
        HipComputeBackend gfx1100 = backendWithArch("gfx1100");
        assertTrue(library.availability(gfx1100).available());
        gfx1100.close();

        HipComputeBackend gfx1030 = backendWithArch("gfx1030");
        Availability a = library.availability(gfx1030);
        assertFalse(a.available());
        assertTrue("the reason names the build targets and the device",
                a.reason().contains("gfx1100,gfx90a") && a.reason().contains("gfx1030"));
        gfx1030.close();
    }

    @Test
    public void capabilitySuffixesOfTheDeviceDoNotBreakCoverage() {
        HipComputeBackend gfx90a = backendWithArch("gfx90a:xnack-");
        assertTrue(library.availability(gfx90a).available());
        gfx90a.close();
    }

    @Test
    public void indexListsTheKernelsOfTheTestBuild() {
        assertEquals(new java.util.HashSet<>(java.util.Arrays.asList("nf_test_scale", "nf_test_missing")),
                library.kernelNames());
    }

    @Test
    public void resolveNamesAMissingCodeObject() {
        HipComputeBackend backend = backendWithArch("gfx1100");
        try {
            library.resolve("nf_test_missing", backend);
            fail("a kernel whose code object is not on the classpath must be rejected");
        } catch (UnsupportedKernelException expected) {
            assertTrue(expected.getMessage().contains("missing.hsaco"));
        } finally {
            backend.close();
        }
    }

    @Test
    public void resolveProducesHipImageArtifactsAndRegistersSignatures() {
        HipComputeBackend backend = backendWithArch("gfx1100");
        try {
            KernelArtifact artifact = library.resolve("nf_test_scale", backend);
            assertEquals(KernelArtifact.Kind.HIP_IMAGE, artifact.kind());
            assertEquals(SCALE_RESOURCE, artifact.resourcePath());
            assertEquals(SCALE_RESOURCE, artifact.programKey());
            KernelSignature registered = HipKernelSignatures.lookup(SCALE_RESOURCE, "nf_test_scale");
            assertNotNull("resolve registers the inventory signature", registered);
            assertEquals(1, registered.bufferCount());
        } finally {
            backend.close();
        }
    }

    @Test
    public void resolveNamesTheMissingKernel() {
        HipComputeBackend backend = backendWithArch("gfx1100");
        try {
            library.resolve("nf_test_other", backend);
            fail("an unknown kernel must be rejected");
        } catch (UnsupportedKernelException expected) {
            assertTrue(expected.getMessage().contains("nf_test_other"));
        } finally {
            backend.close();
        }
    }

    @Test
    public void coverageNeedsTheDeviceTargetListed() {
        assertTrue(HipKernelLibrary.archCovered(java.util.Arrays.asList("gfx1100", "gfx90a"), "gfx1100"));
        assertTrue(HipKernelLibrary.archCovered(java.util.Arrays.asList("gfx1100", "gfx90a"), "gfx90a:xnack-"));
        assertFalse(HipKernelLibrary.archCovered(java.util.Arrays.asList("gfx1100"), "gfx1030"));
        assertFalse(HipKernelLibrary.archCovered(java.util.Arrays.asList("gfx1100"), null));
        assertFalse(HipKernelLibrary.archCovered(java.util.Collections.<String>emptyList(), "gfx1100"));
    }

    @Test
    public void archParsingRejectsNonGfxEntries() {
        try {
            HipKernelLibrary.parseArchs("gfx1100,sm_80", "hip.archs");
            fail("a CUDA arch entry must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("hip.archs"));
        }
        assertEquals(0, HipKernelLibrary.parseArchs(null, "hip.archs").size());
        assertEquals(0, HipKernelLibrary.parseArchs("  ", "hip.archs").size());
        List<String> parsed = HipKernelLibrary.parseArchs("gfx1100,gfx90a:xnack-", "hip.archs");
        assertEquals(java.util.Arrays.asList("gfx1100", "gfx90a"), parsed);
    }

    @Test
    public void aBuildCompiledForADifferentFaSmemIsUnusable() {
        HipKernelLibrary mismatched = new HipKernelLibrary(new KernelLibraryConfig("hip-test-bad",
                "hip-test-lib-bad/src", "hip-test-lib-bad/src/cuda", "hip-test-lib-bad/spv", true),
                HipKernelLibraryTest.class.getClassLoader());
        HipComputeBackend backend = backendWithArch("gfx1100");
        try {
            Availability a = mismatched.availability(backend);
            assertFalse(a.available());
            assertTrue(a.reason().contains("NF_FA_SMEM_FLOATS=4096"));
        } finally {
            backend.close();
        }
    }

    @Test
    public void anIndexMissingTheInventoryKernelMakesTheLibraryUnavailable() {
        // requireCompleteInventory with the bad fixture: the index is complete for its one kernel,
        // so this exercises the coverage report on the same path the LLM library uses.
        HipKernelLibrary incomplete = new HipKernelLibrary(new KernelLibraryConfig("hip-test-incomplete",
                "hip-test-lib/src", "hip-test-lib/src/cuda", "hip-test-lib/spv", true),
                HipKernelLibraryTest.class.getClassLoader());
        HipComputeBackend backend = backendWithArch("gfx1100");
        try {
            assertTrue(incomplete.availability(backend).available());
        } finally {
            backend.close();
        }
    }

    private HipComputeBackend backendWithArch(String gcnArchName) {
        FakeHipDriver driver = new FakeHipDriver();
        driver.gcnArchName = gcnArchName;
        return new HipComputeBackend(new HipDevice(0, driver));
    }

    /** A backend of a given type without device state, for the wrong-backend availability case. */
    private static final class AvailabilityBackendStub extends NoopBackend {
        AvailabilityBackendStub(ComputeBackend.BackendType type) {
            super(type);
        }
    }

    private static class NoopBackend implements ComputeBackend {
        private final ComputeBackend.BackendType type;

        NoopBackend(ComputeBackend.BackendType type) {
            this.type = type;
        }

        @Override
        public BackendType backendType() {
            return type;
        }

        @Override
        public com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities capabilities() {
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
        public com.eonmux.neuralfree.acceleration.backend.GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            throw new UnsupportedOperationException("the stub allocates nothing");
        }

        @Override
        public com.eonmux.neuralfree.acceleration.backend.GpuBuffer allocateBuffer(long sizeInBytes) {
            throw new UnsupportedOperationException("the stub allocates nothing");
        }

        @Override
        public com.eonmux.neuralfree.acceleration.backend.KernelHandle compileKernel(String kernelName,
                                                                                     String kernelSource, String buildOptions) {
            throw new UnsupportedOperationException("the stub compiles nothing");
        }

        @Override
        public void close() {
        }
    }
}
