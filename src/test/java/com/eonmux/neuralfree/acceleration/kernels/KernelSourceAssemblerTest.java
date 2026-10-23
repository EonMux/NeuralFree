package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.Test;

import java.io.IOException;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;

import static com.eonmux.neuralfree.acceleration.kernels.KernelManifestTest.loaderFor;
import static com.eonmux.neuralfree.acceleration.kernels.KernelManifestTest.write;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The OpenCL program inputs: the manifest concatenation of {@link KernelSourceAssembler} (including
 * {@code [shared]} entries), the build options and program keys of {@link OpenClKernelLibrary}, and
 * {@link FaSmem}; and the per-library build profile of {@link KernelLibraryConfig} as the OpenCL, Vulkan and
 * kernel-set layers apply it.
 */
public class KernelSourceAssemblerTest {

    @Test
    public void concatenatesInManifestOrderWithLineMarkers() throws IOException {
        Path dir = Files.createTempDirectory("nf-assembler");
        try (URLClassLoader cl = loaderFor(dir)) {
            write(dir.resolve("lib/manifest.txt"), "common/a.h\n[opencl] legacy/b.cl\nc.cl\n");
            write(dir.resolve("lib/common/a.h"), "#define A 1");
            write(dir.resolve("lib/legacy/b.cl"), "B\n");
            write(dir.resolve("lib/c.cl"), "");
            KernelManifest m = KernelManifest.load(cl, "lib");
            String source = KernelSourceAssembler.assemble(m, cl, "lib");
            assertEquals("#line 1 \"common/a.h\"\n#define A 1\n"
                    + "#line 1 \"legacy/b.cl\"\nB\n"
                    + "#line 1 \"c.cl\"\n", source);
        }
    }

    @Test
    public void includesAreLeftVerbatim() throws IOException {
        Path dir = Files.createTempDirectory("nf-assembler");
        try (URLClassLoader cl = loaderFor(dir)) {
            write(dir.resolve("lib/manifest.txt"), "k.cl\n");
            write(dir.resolve("lib/k.cl"), "#include \"other.h\"\nkernel\n");
            String source = KernelSourceAssembler.assemble(KernelManifest.load(cl, "lib"), cl, "lib");
            assertTrue(source.contains("#include \"other.h\"\n"));
        }
    }

    @Test
    public void missingSourcesAreListed() throws IOException {
        Path dir = Files.createTempDirectory("nf-assembler");
        try (URLClassLoader cl = loaderFor(dir)) {
            write(dir.resolve("lib/manifest.txt"), "a.h\nb.cl\nc.cl\n");
            write(dir.resolve("lib/b.cl"), "B\n");
            KernelManifest m = KernelManifest.load(cl, "lib");
            assertEquals(Arrays.asList("a.h", "c.cl"), KernelSourceAssembler.missingSources(m, cl, "lib"));
            try {
                KernelSourceAssembler.assemble(m, cl, "lib");
                fail("missing sources must be reported");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("[a.h, c.cl]"));
            }
        }
    }

    @Test
    public void assemblesShippedLlmLibraryWithEveryEntryOnceInOrder() {
        ClassLoader cl = getClass().getClassLoader();
        String root = KernelLibraries.LLM_CONFIG.sourceRoot();
        KernelManifest m = KernelManifest.load(cl, root);
        assertEquals(Collections.<String>emptyList(), KernelSourceAssembler.missingSources(m, cl, root));
        String source = KernelSourceAssembler.assemble(m, cl, root);
        int last = -1;
        for (KernelManifest.Entry e : m.entries()) {
            String marker = "#line 1 \"" + e.path() + "\"\n";
            int at = source.indexOf(marker);
            assertTrue("missing marker for " + e.path(), at >= 0);
            assertEquals("marker for " + e.path() + " appears once", at, source.lastIndexOf(marker));
            assertTrue(e.path() + " out of manifest order", at > last);
            last = at;
        }
        assertTrue(source.startsWith("#line 1 \"common/dialect.h\"\n"));
    }

    @Test
    public void sharedEntriesAreReadFromTheSharedRootWithPlainLineMarkers() throws IOException {
        Path dir = Files.createTempDirectory("nf-assembler");
        try (URLClassLoader cl = loaderFor(dir)) {
            write(dir.resolve("lib/manifest.txt"), "[shared] common/d.h\ncommon/a.h\nc.cl\n");
            write(dir.resolve(KernelManifest.SHARED_ROOT + "/common/d.h"), "#define D 1\n");
            write(dir.resolve("lib/common/a.h"), "#define A 1\n");
            write(dir.resolve("lib/c.cl"), "C\n");
            KernelManifest m = KernelManifest.load(cl, "lib");
            assertEquals(Collections.<String>emptyList(), KernelSourceAssembler.missingSources(m, cl, "lib"));
            assertEquals("#line 1 \"common/d.h\"\n#define D 1\n"
                    + "#line 1 \"common/a.h\"\n#define A 1\n"
                    + "#line 1 \"c.cl\"\nC\n", KernelSourceAssembler.assemble(m, cl, "lib"));
        }
    }

    @Test
    public void sharedEntriesMovedIntoTheSharedRootAssembleToTheSameProgram() throws IOException {
        Path before = Files.createTempDirectory("nf-assembler");
        Path after = Files.createTempDirectory("nf-assembler");
        try (URLClassLoader oldLayout = loaderFor(before); URLClassLoader newLayout = loaderFor(after)) {
            write(before.resolve("lib/manifest.txt"), "common/d.h\nk.cl\n");
            write(before.resolve("lib/common/d.h"), "#define D 1");
            write(before.resolve("lib/k.cl"), "K\n");
            write(after.resolve("lib/manifest.txt"), "[shared] common/d.h\nk.cl\n");
            write(after.resolve(KernelManifest.SHARED_ROOT + "/common/d.h"), "#define D 1");
            write(after.resolve("lib/k.cl"), "K\n");
            assertEquals(KernelSourceAssembler.assemble(KernelManifest.load(oldLayout, "lib"), oldLayout, "lib"),
                    KernelSourceAssembler.assemble(KernelManifest.load(newLayout, "lib"), newLayout, "lib"));
        }
    }

    @Test
    public void missingSharedSourcesNameTheSharedRoot() throws IOException {
        Path dir = Files.createTempDirectory("nf-assembler");
        try (URLClassLoader cl = loaderFor(dir)) {
            write(dir.resolve("lib/manifest.txt"), "[shared] common/d.h\nc.cl\n");
            write(dir.resolve("lib/common/d.h"), "#define D 1\n");
            write(dir.resolve("lib/c.cl"), "C\n");
            KernelManifest m = KernelManifest.load(cl, "lib");
            assertEquals(Arrays.asList("[shared] common/d.h"), KernelSourceAssembler.missingSources(m, cl, "lib"));
            try {
                KernelSourceAssembler.assemble(m, cl, "lib");
                fail("a missing shared source must be reported");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("classpath:"
                        + KernelManifest.SHARED_ROOT + "/"));
                assertTrue(expected.getMessage(), expected.getMessage().contains("[[shared] common/d.h]"));
            }
        }
    }

    @Test
    public void openClOptionsHaveNoFastMath() {
        String withSg = OpenClKernelLibrary.buildOptions(OpenClKernelLibrary.BASE_OPTIONS, true, 11264);
        assertEquals("-cl-std=CL1.2 -cl-mad-enable -cl-no-signed-zeros -DNF_FA_SMEM_FLOATS=11264 -DUSE_SUBGROUPS=1",
                withSg);
        String plain = OpenClKernelLibrary.buildOptions(OpenClKernelLibrary.BASE_OPTIONS, false, 3072);
        assertEquals("-cl-std=CL1.2 -cl-mad-enable -cl-no-signed-zeros -DNF_FA_SMEM_FLOATS=3072", plain);
        for (String options : new String[]{withSg, plain}) {
            assertFalse(options.contains("fast-relaxed-math"));
            assertFalse(options.contains("finite-math-only"));
            assertFalse(options.contains("unsafe-math"));
        }
    }

    @Test
    public void openClOptionsStartWithTheLibraryOptions() {
        assertEquals("-cl-std=CL1.2 -DNF_FA_SMEM_FLOATS=3072",
                OpenClKernelLibrary.buildOptions(KernelLibraries.TRAIN_CONFIG.openClOptions(), false, 3072));
        assertEquals("-DNF_FA_SMEM_FLOATS=3072 -DUSE_SUBGROUPS=1", OpenClKernelLibrary.buildOptions("", true, 3072));
    }

    @Test
    public void programKeysSeparateSubgroupAndSharedMemoryVariants() {
        assertEquals("llm__sg#11264", OpenClKernelLibrary.programKey("llm", true, 11264));
        assertEquals("llm#7168", OpenClKernelLibrary.programKey("llm", false, 7168));
    }

    @Test
    public void openClLibraryIsReplaceableOnlyWithTheSameSources() {
        KernelLibraryRegistry registry = new KernelLibraryRegistry();
        KernelLibraryConfig config = new KernelLibraryConfig("lib", "kernels/lib", "kernels/lib/cuda", "shaders/lib", true);
        OpenClKernelLibrary first = new OpenClKernelLibrary(config);
        assertEquals(null, registry.register(first));
        OpenClKernelLibrary sameSources = new OpenClKernelLibrary(config.withRequireCompleteInventory(false));
        assertEquals(first, registry.register(sameSources));
        try {
            registry.register(new OpenClKernelLibrary(
                    new KernelLibraryConfig("lib", "kernels/other", "kernels/other/cuda", "shaders/other", true)));
            fail("an OpenCL library with other sources must not reuse the program key of 'lib'");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("kernels/other"));
        }
        assertEquals(sameSources, registry.get("lib", ComputeBackend.BackendType.OPENCL).get());
    }

    @Test
    public void subgroupsFollowExtensionOrFeature() {
        assertFalse(OpenClKernelLibrary.useSubgroups(new DeviceCapabilities.Builder().deviceName("d").build()));
        assertTrue(OpenClKernelLibrary.useSubgroups(
                new DeviceCapabilities.Builder().deviceName("d").supportsSubgroups(true).build()));
        assertTrue(OpenClKernelLibrary.useSubgroups(new DeviceCapabilities.Builder().deviceName("d")
                .feature(com.eonmux.neuralfree.acceleration.backend.DeviceFeature.SUBGROUP_ARITHMETIC).build()));
    }

    @Test
    public void faSmemFloatsFollowsSharedMemory() {
        assertEquals(FaSmem.MAX_FLOATS, FaSmem.faSmemFloats(ComputeBackend.BackendType.CUDA, sharedBytes(16 * 1024)));
        assertEquals(11264, FaSmem.faSmemFloats(ComputeBackend.BackendType.OPENCL, sharedBytes(64 * 1024)));
        assertEquals(11264, FaSmem.faSmemFloats(ComputeBackend.BackendType.OPENCL, sharedBytes(48 * 1024)));
        assertEquals(7168, FaSmem.faSmemFloats(ComputeBackend.BackendType.VULKAN, sharedBytes(32 * 1024)));
        assertEquals(3072, FaSmem.faSmemFloats(ComputeBackend.BackendType.VULKAN, sharedBytes(16 * 1024)));
        assertEquals(3072, FaSmem.faSmemFloats(ComputeBackend.BackendType.OPENCL, sharedBytes(8 * 1024)));
        assertEquals(3072, FaSmem.faSmemFloats(ComputeBackend.BackendType.OPENCL, sharedBytes(0)));
    }

    private static DeviceCapabilities sharedBytes(long bytes) {
        return new DeviceCapabilities.Builder().deviceName("d").maxSharedMemoryBytes(bytes).build();
    }

    @Test
    public void fiveArgumentConfigKeepsTheLlmProfile() {
        KernelLibraryConfig llm = KernelLibraries.LLM_CONFIG;
        assertEquals(OpenClKernelLibrary.BASE_OPTIONS, llm.openClOptions());
        assertTrue(llm.subgroupVariants());
        assertEquals(EnumSet.of(DeviceFeature.STORAGE_8BIT, DeviceFeature.STORAGE_16BIT,
                DeviceFeature.TIMELINE_SEMAPHORE), llm.vulkanFeatures());
        assertEquals(new KernelLibraryConfig("llm", "kernels/llm", "kernels/llm/cuda", "shaders/llm", true,
                OpenClKernelLibrary.BASE_OPTIONS, true, EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE,
                DeviceFeature.STORAGE_16BIT, DeviceFeature.STORAGE_8BIT)), llm);
    }

    @Test
    public void trainConfigHasItsOwnProfile() {
        KernelLibraryConfig train = KernelLibraries.TRAIN_CONFIG;
        assertEquals(KernelLibraries.TRAIN, train.libraryId());
        assertEquals("kernels/train", train.sourceRoot());
        assertEquals("kernels/train/cuda", train.cudaRoot());
        assertEquals("shaders/train", train.spvRoot());
        assertTrue(train.requireCompleteInventory());
        assertEquals("-cl-std=CL1.2", train.openClOptions());
        assertFalse(train.subgroupVariants());
        assertEquals(EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE), train.vulkanFeatures());
    }

    @Test
    public void configEqualityCoversEveryField() {
        KernelLibraryConfig base = config("-cl-std=CL1.2", false, EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE));
        KernelLibraryConfig same = config("-cl-std=CL1.2", false, EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE));
        assertEquals(base, same);
        assertEquals(base.hashCode(), same.hashCode());
        assertEquals(base.toString(), same.toString());
        assertNotEquals(base, config("-cl-std=CL1.2 -cl-mad-enable", false, EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE)));
        assertNotEquals(base, config("-cl-std=CL1.2", true, EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE)));
        assertNotEquals(base, config("-cl-std=CL1.2", false, EnumSet.noneOf(DeviceFeature.class)));
        assertNotEquals(base, base.withRequireCompleteInventory(false));
        assertTrue(base.toString(), base.toString().contains("openClOptions='-cl-std=CL1.2'"));
        assertTrue(base.toString(), base.toString().contains("subgroupVariants=false"));
        assertTrue(base.toString(), base.toString().contains("TIMELINE_SEMAPHORE"));
    }

    @Test
    public void configCopiesAndProtectsTheFeatureSet() {
        EnumSet<DeviceFeature> features = EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE);
        KernelLibraryConfig c = config("", false, features);
        features.add(DeviceFeature.STORAGE_8BIT);
        assertEquals(EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE), c.vulkanFeatures());
        try {
            c.vulkanFeatures().add(DeviceFeature.STORAGE_16BIT);
            fail("the feature set must be unmodifiable");
        } catch (UnsupportedOperationException expected) {
            assertEquals(1, c.vulkanFeatures().size());
        }
    }

    @Test
    public void configRejectsMalformedProfiles() {
        assertConfigRejected(null, Collections.<DeviceFeature>emptySet(), "openClOptions");
        assertConfigRejected(" -cl-std=CL1.2", Collections.<DeviceFeature>emptySet(), "whitespace");
        assertConfigRejected("-cl-std=CL1.2\n-DX", Collections.<DeviceFeature>emptySet(), "single line");
        assertConfigRejected("", null, "vulkanFeatures");
        assertConfigRejected("", Arrays.asList(DeviceFeature.TIMELINE_SEMAPHORE, null), "null");
    }

    @Test
    public void withRequireCompleteInventoryKeepsEveryOtherField() {
        KernelLibraryConfig train = KernelLibraries.TRAIN_CONFIG;
        assertSame(train, train.withRequireCompleteInventory(true));
        KernelLibraryConfig relaxed = train.withRequireCompleteInventory(false);
        assertFalse(relaxed.requireCompleteInventory());
        assertEquals(train.libraryId(), relaxed.libraryId());
        assertEquals(train.sourceRoot(), relaxed.sourceRoot());
        assertEquals(train.cudaRoot(), relaxed.cudaRoot());
        assertEquals(train.spvRoot(), relaxed.spvRoot());
        assertEquals(train.openClOptions(), relaxed.openClOptions());
        assertEquals(train.subgroupVariants(), relaxed.subgroupVariants());
        assertEquals(train.vulkanFeatures(), relaxed.vulkanFeatures());
        assertEquals(train, relaxed.withRequireCompleteInventory(true));
    }

    @Test
    public void defaultsRegisterBothLibrariesOnEveryBackendType() {
        KernelLibraries.ensureDefaults();
        for (ComputeBackend.BackendType type : ComputeBackend.BackendType.values()) {
            for (String id : new String[]{KernelLibraries.LLM, KernelLibraries.TRAIN}) {
                assertTrue(id + " on " + type, KernelLibraryRegistry.global().get(id, type).isPresent());
            }
        }
        assertTrue(KernelLibraryRegistry.global().libraryIds().containsAll(
                Arrays.asList(KernelLibraries.LLM, KernelLibraries.TRAIN)));
    }

    @Test
    public void openClLibraryAppliesTheConfiguredOptionsAndSubgroupFlag() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder().deviceName("d").supportsSubgroups(true)
                .maxSharedMemoryBytes(48 * 1024).build();
        FakeComputeBackend backend = FakeComputeBackend.builder().capabilities(caps).build();
        int smem = FaSmem.faSmemFloats(ComputeBackend.BackendType.OPENCL, caps);

        KernelArtifact llm = new OpenClKernelLibrary(KernelLibraries.LLM_CONFIG).resolve("K", backend);
        assertEquals(OpenClKernelLibrary.programKey(KernelLibraries.LLM, true, smem), llm.programKey());
        assertEquals(OpenClKernelLibrary.buildOptions(OpenClKernelLibrary.BASE_OPTIONS, true, smem), llm.buildOptions());

        KernelArtifact train = new OpenClKernelLibrary(KernelLibraries.TRAIN_CONFIG).resolve("K", backend);
        assertEquals(OpenClKernelLibrary.programKey(KernelLibraries.TRAIN, false, smem), train.programKey());
        assertEquals("-cl-std=CL1.2 -DNF_FA_SMEM_FLOATS=" + smem, train.buildOptions());
    }

    @Test
    public void kernelSetSubgroupsFollowTheLibrarySubgroupVariants() {
        for (KernelLibraryConfig c : new KernelLibraryConfig[]{KernelLibraries.LLM_CONFIG, KernelLibraries.TRAIN_CONFIG}) {
            assertEquals(c.subgroupVariants(), KernelSet.subgroupVariants(new OpenClKernelLibrary(c)));
            assertEquals(c.subgroupVariants(), KernelSet.subgroupVariants(new CudaKernelLibrary(c)));
            assertEquals(c.subgroupVariants(), KernelSet.subgroupVariants(new VulkanKernelLibrary(c)));
        }
    }

    @Test
    public void vulkanLibraryRequiresTheConfiguredFeaturesAndSubgroupVariants() throws IOException {
        Path dir = Files.createTempDirectory("nf-vulkan");
        try (URLClassLoader cl = loaderFor(dir)) {
            write(dir.resolve("shaders/lib/index.properties"), "K.buffers=1\nK.ro=1\nK.push=i\nK.sg=1\n");
            write(dir.resolve("shaders/lib/K.spv"), "plain");
            write(dir.resolve("shaders/lib/K__sg.spv"), "subgroups");
            DeviceCapabilities caps = new DeviceCapabilities.Builder().deviceName("d")
                    .feature(DeviceFeature.TIMELINE_SEMAPHORE).feature(DeviceFeature.SUBGROUP_ARITHMETIC).build();
            FakeComputeBackend backend = FakeComputeBackend.builder()
                    .backendType(ComputeBackend.BackendType.VULKAN).capabilities(caps).build();

            VulkanKernelLibrary trainLike = new VulkanKernelLibrary(new KernelLibraryConfig("lib", "kernels/lib",
                    "kernels/lib/cuda", "shaders/lib", false, "-cl-std=CL1.2", false,
                    EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE)), cl);
            Availability a = trainLike.availability(backend);
            assertTrue(a.reason(), a.available());
            assertEquals("shaders/lib/K.spv", trainLike.resolve("K", backend).resourcePath());

            VulkanKernelLibrary llmLike = new VulkanKernelLibrary(
                    new KernelLibraryConfig("lib", "kernels/lib", "kernels/lib/cuda", "shaders/lib", false), cl);
            Availability lacking = llmLike.availability(backend);
            assertFalse(lacking.available());
            assertTrue(lacking.reason(), lacking.reason().contains("STORAGE_8BIT, STORAGE_16BIT"));
            assertEquals("shaders/lib/K__sg.spv", llmLike.resolve("K", backend).resourcePath());
        }
    }

    private static KernelLibraryConfig config(String options, boolean subgroupVariants,
                                              java.util.Collection<DeviceFeature> features) {
        return new KernelLibraryConfig("lib", "kernels/lib", "kernels/lib/cuda", "shaders/lib", true, options,
                subgroupVariants, features);
    }

    private static void assertConfigRejected(String options, java.util.Collection<DeviceFeature> features,
                                             String messagePart) {
        try {
            config(options, false, features);
            fail("configuration must be rejected: options='" + options + "', features=" + features);
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(messagePart));
        }
    }
}
