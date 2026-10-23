package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The checked-in LLM kernel inventory as the runtime reads it from the classpath: present, one argument
 * signature (count, kind and layout columns) per kernel, every entry naming a portable manifest source, and
 * accepted by the inventory coverage check that gates the CUDA and Vulkan LLM libraries.
 */
public class LlmInventoryResourceTest {

    private static final String ROOT = KernelLibraries.LLM_CONFIG.sourceRoot();
    private static final String PATH = ROOT + "/" + KernelManifest.INVENTORY_FILE;

    private static ClassLoader loader() {
        return LlmInventoryResourceTest.class.getClassLoader();
    }

    private static String inventoryText() {
        String text = KernelManifest.readText(loader(), PATH);
        assertNotNull("classpath:" + PATH + " must ship with the kernel sources", text);
        return text;
    }

    @Test
    public void inventoryIsOnTheClasspathWithASignaturePerKernel() {
        Optional<Set<String>> names = KernelManifest.inventory(loader(), ROOT);
        assertTrue("classpath:" + PATH + " must ship with the kernel sources", names.isPresent());
        assertFalse("classpath:" + PATH + " lists no kernels", names.get().isEmpty());
        Map<String, KernelSignature> signatures = CudaKernelLibrary.parseSignatures(inventoryText(), "classpath:" + PATH);
        assertEquals("every inventory kernel carries an argument signature", names.get(), signatures.keySet());
    }

    @Test
    public void everyEntryHasSixColumnsAndNamesAPortableKernelSource() {
        Set<String> portableUnits = new HashSet<>();
        for (String path : KernelManifest.load(loader(), ROOT).portablePaths()) {
            if (path.endsWith(".cl")) {
                portableUnits.add(path);
            }
        }
        List<String> violations = new ArrayList<>();
        String[] lines = inventoryText().split("\r?\n", -1);
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            if (line.trim().isEmpty() || line.startsWith("#")) {
                continue;
            }
            String[] cols = line.split("\t", -1);
            if (cols.length != 6) {
                violations.add((i + 1) + ": " + cols.length + " columns, expected 6");
            } else if (!portableUnits.contains(cols[1])) {
                violations.add((i + 1) + ": " + cols[0] + " names source " + cols[1]
                        + ", which is not an untagged .cl manifest entry");
            }
        }
        assertTrue("classpath:" + PATH + ":\n  " + String.join("\n  ", violations), violations.isEmpty());
    }

    @Test
    public void coverageCheckAcceptsABuildThatHoldsEveryInventoryKernel() {
        Set<String> names = KernelManifest.inventory(loader(), ROOT).orElseThrow(AssertionError::new);
        Availability complete = KernelManifest.inventoryCoverage(loader(), KernelLibraries.LLM_CONFIG.libraryId(),
                ROOT, names, "the inventory itself");
        assertTrue(complete.reason(), complete.available());
    }
}
