package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryConfig;
import com.eonmux.neuralfree.acceleration.kernels.KernelManifest;
import com.eonmux.neuralfree.acceleration.kernels.MetalKernelLibrary;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The generated Metal kernel libraries: the Metal shader index covers exactly the portable kernel
 * inventory, every indexed kernel has its MSL source and signature, and the MSL binds the buffers
 * at their binding indices with the packed push constants after them, which is how the Metal
 * backend binds arguments.
 *
 * <p>The tests skip when the Metal sources were not generated (no SPIRV-Cross toolchain at build
 * time), so they never require network access.</p>
 */
public class MetalShaderLibraryTest {

    private static final ClassLoader CL = MetalShaderLibraryTest.class.getClassLoader();
    private static final Pattern BUFFER = Pattern.compile("\\[\\[buffer\\((\\d+)\\)\\]\\]");
    private static final Pattern IDENTIFIER = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
    private static final Map<String, KernelLibraryConfig> LIBRARIES = libraries();

    private static Map<String, KernelLibraryConfig> libraries() {
        Map<String, KernelLibraryConfig> libraries = new TreeMap<>();
        libraries.put("llm", KernelLibraries.LLM_CONFIG);
        libraries.put("train", KernelLibraries.TRAIN_CONFIG);
        return libraries;
    }

    @Test
    public void everyMetalIndexCoversThePortableInventory() {
        for (Map.Entry<String, KernelLibraryConfig> e : LIBRARIES.entrySet()) {
            Properties index = index(e.getValue());
            Assume.assumeNotNull(index);
            Optional<Set<String>> inventory = KernelManifest.inventory(CL, e.getValue().sourceRoot());
            Assume.assumeTrue(inventory.isPresent());
            Set<String> built = kernelNames(index);
            Set<String> required = new TreeSet<>(inventory.get());
            required.removeAll(KernelManifest.cudaOnlyInventoryKernels(CL, e.getValue().sourceRoot()));
            for (String name : required) {
                assertTrue(e.getKey() + " Metal index lacks " + name, built.contains(name));
            }
            for (String name : built) {
                assertTrue(e.getKey() + " Metal index has " + name + " outside the inventory",
                        inventory.get().contains(name));
            }
        }
    }

    @Test
    public void everyMetalIndexMatchesTheSpirvIndex() {
        for (Map.Entry<String, KernelLibraryConfig> e : LIBRARIES.entrySet()) {
            Properties metal = index(e.getValue());
            Properties spirv = properties(e.getValue().spvRoot() + "/index.properties");
            Assume.assumeNotNull(metal);
            Assume.assumeNotNull(spirv);
            // a pinned SPIR-V module (.sgsize, written for one subgroup width) and a Vulkan-only one (.vulkan=1)
            // have no Metal translation
            Set<String> portable = new TreeSet<>();
            for (String name : kernelNames(spirv)) {
                if (spirv.getProperty(name + ".sgsize") == null && !"1".equals(spirv.getProperty(name + ".vulkan"))) {
                    portable.add(name);
                }
            }
            assertEquals(e.getKey() + " kernel set", portable, kernelNames(metal));
            for (String name : kernelNames(metal)) {
                assertEquals(e.getKey() + "/" + name + ".buffers",
                        spirv.getProperty(name + ".buffers"), metal.getProperty(name + ".buffers"));
                assertEquals(e.getKey() + "/" + name + ".ro",
                        spirv.getProperty(name + ".ro"), metal.getProperty(name + ".ro"));
                assertEquals(e.getKey() + "/" + name + ".push",
                        spirv.getProperty(name + ".push"), metal.getProperty(name + ".push"));
            }
        }
    }

    @Test
    public void everyKernelHasAnMslSourceWithItsBindingLayout() {
        int checked = 0;
        for (Map.Entry<String, KernelLibraryConfig> e : LIBRARIES.entrySet()) {
            Properties index = index(e.getValue());
            Assume.assumeNotNull(index);
            for (String name : kernelNames(index)) {
                KernelSignature signature = KernelSignature.parse(index.getProperty(name + ".buffers"),
                        index.getProperty(name + ".ro"), index.getProperty(name + ".push"));
                String msl = source(e.getValue(), name);
                assertNotNull(e.getKey() + "/" + name + " has no MSL source", msl);
                assertTrue(e.getKey() + "/" + name + ": the MSL does not define the kernel",
                        msl.contains("kernel void " + name + "("));
                checkBindings(e.getKey(), name, signature, msl);
                SpecConstants.parse(msl, name);
                assertEquals(e.getKey() + "/" + name + ": subgroup kernel detection",
                        "1".equals(index.getProperty(name + ".sg")),
                        MetalKernelHandle.usesSubgroups(msl));
                checked++;
            }
        }
        assertTrue("no Metal kernels were checked", checked > 0);
    }

    private static void checkBindings(String library, String name, KernelSignature signature, String msl) {
        String declaration = null;
        for (String line : msl.split("\n")) {
            if (line.startsWith("kernel void " + name + "(")) {
                declaration = line;
                break;
            }
        }
        assertNotNull(library + "/" + name + ": no kernel declaration", declaration);
        SortedMap<Integer, Integer> counts = new TreeMap<>();
        Matcher m = BUFFER.matcher(declaration);
        while (m.find()) {
            int index = Integer.parseInt(m.group(1));
            Integer previous = counts.get(index);
            counts.put(index, previous == null ? 1 : previous + 1);
        }
        int buffers = signature.bufferCount();
        for (Map.Entry<Integer, Integer> bound : counts.entrySet()) {
            int index = bound.getKey();
            String where = library + "/" + name + " buffer " + index;
            assertEquals(where + " is bound more than once", Integer.valueOf(1), bound.getValue());
            assertTrue(where + " is outside 0.." + buffers, index >= 0 && index <= buffers);
            if (index == buffers) {
                assertTrue(where + " is the push-constant slot but the kernel has no scalars",
                        signature.scalarCount() > 0);
            }
        }
        // Every storage buffer the MSL keeps must be at its binding index; SPIRV-Cross may drop an
        // unused one, which the backend tolerates because it binds each argument at that index.
        for (int index = 0; index < buffers; index++) {
            if (counts.containsKey(index)) {
                assertEquals(library + "/" + name + " buffer " + index + " is not at its binding index",
                        Integer.valueOf(1), counts.get(index));
            }
        }
    }

    private static Properties index(KernelLibraryConfig config) {
        return properties(config.spvRoot() + "/" + MetalKernelLibrary.METAL_DIR + "/"
                + MetalKernelLibrary.INDEX_FILE);
    }

    private static Properties properties(String resource) {
        String text = text(resource);
        if (text == null) {
            return null;
        }
        Properties properties = new Properties();
        try {
            properties.load(new StringReader(text));
        } catch (IOException e) {
            throw new IllegalStateException(resource + " is unreadable: " + e.getMessage(), e);
        }
        return properties;
    }

    private static String source(KernelLibraryConfig config, String kernel) {
        return text(config.spvRoot() + "/" + MetalKernelLibrary.METAL_DIR + "/" + kernel
                + MetalKernelLibrary.SOURCE_SUFFIX);
    }

    private static String text(String resource) {
        try (InputStream in = CL.getResourceAsStream(resource)) {
            if (in == null) {
                return null;
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException(resource + " is unreadable: " + e.getMessage(), e);
        }
    }

    private static Set<String> kernelNames(Properties index) {
        Set<String> names = new TreeSet<>();
        for (String key : index.stringPropertyNames()) {
            if (key.endsWith(".buffers")) {
                String name = key.substring(0, key.length() - ".buffers".length());
                if (!IDENTIFIER.matcher(name).matches()) {
                    fail("the Metal index lists an invalid kernel name '" + name + "'");
                }
                names.add(name);
            }
        }
        return names;
    }
}
