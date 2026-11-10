package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import org.junit.Assume;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Properties;
import java.util.TreeMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The pinned-subgroup modules of the Vulkan libraries (variants rows flagged {@code sg<N>}, index key
 * {@code <name>.sgsize}): the index value parser, the library's refusal on devices whose subgroup range or operations
 * cannot run them, the artifact that carries the width to the pipeline, the library's universal kernel set (which
 * leaves pinned modules out, so the {@link KernelSet} probe never loads one), and the built LLM index, whose pinned
 * modules are 16-lane modules without an {@code __sg} twin.
 */
public class VulkanPinnedSubgroupTest {

    private static DeviceCapabilities caps(int minSubgroup, int maxSubgroup, boolean arithmetic) {
        DeviceCapabilities.Builder b = new DeviceCapabilities.Builder()
                .deviceName("test")
                .supportsSubgroups(true)
                .subgroupSize(maxSubgroup)
                .minSubgroupSize(minSubgroup)
                .maxSubgroupSize(maxSubgroup);
        if (arithmetic) {
            b.feature(DeviceFeature.SUBGROUP_ARITHMETIC);
        }
        return b.build();
    }

    @Test
    public void theIndexValueIsAbsentOrAPowerOfTwoFromEightToSixtyFour() {
        assertEquals(0, VulkanKernelLibrary.pinnedSize(null, "K", "test"));
        for (int n : new int[]{8, 16, 32, 64}) {
            assertEquals(n, VulkanKernelLibrary.pinnedSize(" " + n + " ", "K", "test"));
        }
        for (String bad : new String[]{"0", "4", "12", "128", "-16", "sixteen", ""}) {
            try {
                VulkanKernelLibrary.pinnedSize(bad, "K", "test");
                fail("accepted .sgsize '" + bad + "'");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("K.sgsize"));
            }
        }
    }

    @Test
    public void aDeviceWhoseRangeHoldsTheWidthMayLoadTheModule() {
        assertNull(VulkanKernelLibrary.pinnedRefusal(16, caps(8, 32, true)));
        assertNull(VulkanKernelLibrary.pinnedRefusal(16, caps(16, 16, true)));
    }

    @Test
    public void aDeviceOutsideTheWidthOrWithoutSubgroupArithmeticRefusesIt() {
        assertNotNull(VulkanKernelLibrary.pinnedRefusal(16, caps(32, 32, true)));
        assertNotNull(VulkanKernelLibrary.pinnedRefusal(16, caps(32, 64, true)));
        assertNotNull(VulkanKernelLibrary.pinnedRefusal(16, caps(8, 8, true)));
        assertNotNull(VulkanKernelLibrary.pinnedRefusal(16, caps(8, 32, false)));
    }

    @Test
    public void theArtifactCarriesTheWidth() {
        KernelSignature sig = KernelSignature.parse("1", "1", "i");
        KernelArtifact plain = KernelArtifact.spirv("K", "shaders/k.spv", new TreeMap<>(), sig);
        KernelArtifact pinned = KernelArtifact.spirv("K", "shaders/k.spv", new TreeMap<>(), sig, 16);
        assertEquals(0, plain.requiredSubgroupSize());
        assertEquals(16, pinned.requiredSubgroupSize());
        assertNotEquals(plain, pinned);
        assertTrue(pinned.toString(), pinned.toString().contains("subgroup=16"));
        for (int bad : new int[]{-16, 12, 24}) {
            try {
                KernelArtifact.spirv("K", "shaders/k.spv", new TreeMap<>(), sig, bad);
                fail("accepted a required subgroup size of " + bad);
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("power of two"));
            }
        }
    }

    @Test
    public void theBuiltLlmIndexPinsSixteenLaneModulesWithoutSubgroupTwins() throws IOException {
        Properties index = new Properties();
        try (InputStream in = getClass().getClassLoader().getResourceAsStream("shaders/llm/index.properties")) {
            Assume.assumeNotNull("the Vulkan shaders were not built", in);
            index.load(in);
        }
        int pinned = 0;
        for (String key : index.stringPropertyNames()) {
            if (!key.endsWith(".sgsize")) {
                continue;
            }
            String name = key.substring(0, key.length() - ".sgsize".length());
            assertEquals(name, 16, VulkanKernelLibrary.pinnedSize(index.getProperty(key), name, "index"));
            assertEquals(name + " has an __sg twin", "0", index.getProperty(name + ".sg"));
            assertNotNull(name + " lacks .buffers", index.getProperty(name + ".buffers"));
            pinned++;
        }
        assertTrue("no pinned module in the LLM index", pinned > 0);
    }

    /**
     * A pinned module that sorts before every other kernel is not the probe: a device that refuses pinned modules
     * (every non-Intel GPU) would otherwise mark the whole library unusable.
     */
    @Test
    public void theProbeKernelIsNeverAPinnedModule() {
        String index = "AAA_Pinned.buffers=1\nAAA_Pinned.ro=0\nAAA_Pinned.push=i\nAAA_Pinned.sg=0\n"
                + "AAA_Pinned.sgsize=16\n"
                + "B_Plain.buffers=1\nB_Plain.ro=0\nB_Plain.push=i\nB_Plain.sg=1\n"
                + "C_Plain.buffers=2\nC_Plain.ro=10\nC_Plain.push=\nC_Plain.sg=0\n";
        VulkanKernelLibrary library = new VulkanKernelLibrary(
                new KernelLibraryConfig("pinned-test", "kernels/pinned-test", "kernels/pinned-test/cuda",
                        "shaders/pinned-test", false),
                new IndexClassLoader("shaders/pinned-test/" + VulkanKernelLibrary.INDEX_FILE, index));
        assertEquals(new HashSet<>(Arrays.asList("AAA_Pinned", "B_Plain", "C_Plain")), library.kernelNames());
        assertEquals(new HashSet<>(Arrays.asList("B_Plain", "C_Plain")), library.universalKernelNames());
        assertEquals("B_Plain", KernelSet.probeKernel(library));
    }

    @Test
    public void theBuiltLlmLibraryProbesAnUnpinnedKernel() throws IOException {
        Properties index = new Properties();
        try (InputStream in = getClass().getClassLoader().getResourceAsStream("shaders/llm/index.properties")) {
            Assume.assumeNotNull("the Vulkan shaders were not built", in);
            index.load(in);
        }
        String probe = KernelSet.probeKernel(new VulkanKernelLibrary(KernelLibraries.LLM_CONFIG));
        assertNotNull(probe);
        assertNotNull(probe + " is not in the index", index.getProperty(probe + ".buffers"));
        assertNull(probe + " is pinned", index.getProperty(probe + ".sgsize"));
    }

    /** Serves one synthetic resource and delegates the rest. */
    private static final class IndexClassLoader extends ClassLoader {
        private final String path;
        private final byte[] text;

        IndexClassLoader(String path, String text) {
            super(VulkanPinnedSubgroupTest.class.getClassLoader());
            this.path = path;
            this.text = text.getBytes(StandardCharsets.UTF_8);
        }

        @Override
        public InputStream getResourceAsStream(String name) {
            return path.equals(name) ? new ByteArrayInputStream(text) : super.getResourceAsStream(name);
        }
    }
}
