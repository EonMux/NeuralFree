package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.hip.HipComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.hip.HipKernelSignatures;
import com.eonmux.neuralfree.acceleration.hip.FakeHipDriver;
import com.eonmux.neuralfree.acceleration.hip.HipDevice;
import neuralfree.gpu.GpuHipToolkit;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The build-to-runtime contract of the HIP kernels: a layout written by the HIP build's own
 * writers ({@code GpuHipToolkit}, the texts the {@code compileHipKernels} task records under
 * {@code build/generated/gpu/hip}) loads through {@link HipKernelLibrary} over the generated
 * directory as class loader, the way the packaged build serves {@code kernels/&lt;id&gt;/hip}.
 */
public class HipKernelBuildLayoutTest {

    private static final String KERNEL = "nf_test_scale";
    private static final String INVENTORY_LINE =
            KERNEL + "\ttest/eltwise.cl\t1\t2\tf,i\tBfi";

    private final List<Path> generated = new ArrayList<>();

    @After
    public void deleteGeneratedLayouts() throws IOException {
        for (Path root : generated) {
            Files.walk(root).sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    /**
     * Lays out one generated library root: the inventory, the HIP index and build metadata from
     * {@code GpuHipToolkit.indexText}/{@code buildPropertiesText}, and the code object file, as
     * the kernel build writes them; the library loads it over a class loader of {@code root}.
     */
    private LoadedLibrary generateLayout(String archsProperty, boolean recordFaSmem,
                                         Map<String, String> index, boolean writeCodeObject)
            throws IOException {
        Path root = Files.createTempDirectory("hip-build-layout");
        generated.add(root);
        Path src = root.resolve("genlib/src");
        Files.createDirectories(src.resolve("hip"));
        Files.write(src.resolve("inventory.txt"),
                ("# generated build\n" + INVENTORY_LINE + "\n").getBytes(StandardCharsets.UTF_8));

        List<String> targets = GpuHipToolkit.selectArchs(archsProperty, null).selected();
        Files.write(src.resolve("hip/index.tsv"),
                GpuHipToolkit.indexText(index).getBytes(StandardCharsets.UTF_8));
        List<String> properties = new ArrayList<>(Arrays.asList(
                "hipcc.version=6.2.41133",
                "hip.archs=" + String.join(",", targets),
                "kernel.format=" + GpuHipToolkit.CODE_OBJECT_SUFFIX.substring(1)));
        if (recordFaSmem) {
            properties.add("fa.smem.floats=" + FaSmem.MAX_FLOATS);
        }
        Files.write(src.resolve("hip/build.properties"),
                GpuHipToolkit.buildPropertiesText(properties).getBytes(StandardCharsets.UTF_8));
        for (String resource : new java.util.LinkedHashSet<>(index.values())) {
            Path object = src.resolve("hip").resolve(resource);
            if (writeCodeObject) {
                Files.write(object, ("code object of " + resource).getBytes(StandardCharsets.UTF_8));
            }
        }

        URLClassLoader classLoader = new URLClassLoader(new URL[]{root.toUri().toURL()});
        KernelLibraryConfig config = new KernelLibraryConfig("genlib", "genlib/src",
                "genlib/src/cuda", "genlib/spv", true);
        return new LoadedLibrary(new HipKernelLibrary(config, classLoader), classLoader, targets);
    }

    private HipComputeBackend backend(String gcnArchName) {
        FakeHipDriver driver = new FakeHipDriver();
        driver.gcnArchName = gcnArchName;
        return new HipComputeBackend(new HipDevice(0, driver));
    }

    @Test
    public void theGeneratedLayoutServesTheDevice() throws IOException {
        LoadedLibrary lib = generateLayout(GpuHipToolkit.DEFAULT_ARCHS, false,
                Collections.singletonMap(KERNEL, KERNEL + GpuHipToolkit.CODE_OBJECT_SUFFIX), true);
        HipComputeBackend gfx1100 = backend("gfx1100");
        try {
            Availability a = lib.library.availability(gfx1100);
            assertTrue(a.reason(), a.available());
            assertEquals(Collections.singleton(KERNEL), lib.library.kernelNames());
        } finally {
            gfx1100.close();
            lib.classLoader.close();
        }
    }

    @Test
    public void resolveServesTheIndexedCodeObjectAndSignature() throws IOException {
        LoadedLibrary lib = generateLayout(GpuHipToolkit.DEFAULT_ARCHS, true,
                Collections.singletonMap(KERNEL, KERNEL + GpuHipToolkit.CODE_OBJECT_SUFFIX), true);
        HipComputeBackend gfx1100 = backend("gfx1100");
        try {
            assertTrue(lib.library.availability(gfx1100).available());
            KernelArtifact artifact = lib.library.resolve(KERNEL, gfx1100);
            assertEquals(KernelArtifact.Kind.HIP_IMAGE, artifact.kind());
            assertEquals("genlib/src/hip/nf_test_scale.hsaco", artifact.resourcePath());
            assertEquals("genlib/src/hip/nf_test_scale.hsaco", artifact.programKey());
            KernelSignature signature = HipKernelSignatures.lookup(artifact.resourcePath(), KERNEL);
            assertNotNull("resolve registers the inventory signature", signature);
            assertEquals(1, signature.bufferCount());
        } finally {
            gfx1100.close();
            lib.classLoader.close();
        }
    }

    @Test
    public void aDeviceOutsideTheRecordedTargetsIsUnusable() throws IOException {
        LoadedLibrary lib = generateLayout("gfx942,gfx90a", false,
                Collections.singletonMap(KERNEL, KERNEL + GpuHipToolkit.CODE_OBJECT_SUFFIX), true);
        HipComputeBackend gfx1100 = backend("gfx1100");
        try {
            Availability a = lib.library.availability(gfx1100);
            assertFalse(a.available());
            assertTrue("the reason names the recorded targets and the device",
                    a.reason().contains("gfx942,gfx90a") && a.reason().contains("gfx1100"));
        } finally {
            gfx1100.close();
            lib.classLoader.close();
        }
    }

    @Test
    public void anIndexEntryWithoutItsCodeObjectIsNamedAtResolve() throws IOException {
        LoadedLibrary lib = generateLayout(GpuHipToolkit.DEFAULT_ARCHS, false,
                Collections.singletonMap(KERNEL, KERNEL + GpuHipToolkit.CODE_OBJECT_SUFFIX), false);
        HipComputeBackend gfx1100 = backend("gfx1100");
        try {
            lib.library.resolve(KERNEL, gfx1100);
            fail("an index entry whose code object is not on the classpath must be rejected");
        } catch (UnsupportedKernelException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("nf_test_scale.hsaco"));
        } finally {
            gfx1100.close();
            lib.classLoader.close();
        }
    }

    /** One generated layout with the class loader the library reads it through. */
    private static final class LoadedLibrary {
        final HipKernelLibrary library;
        final URLClassLoader classLoader;
        final List<String> targets;

        LoadedLibrary(HipKernelLibrary library, URLClassLoader classLoader, List<String> targets) {
            this.library = library;
            this.classLoader = classLoader;
            this.targets = targets;
        }
    }
}
