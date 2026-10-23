package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The checked-in train kernel inventory as the runtime reads it from the classpath: present, one argument
 * signature (count, kind and layout columns) per kernel, every entry naming a portable manifest source,
 * accepted by the inventory coverage check that gates the CUDA and Vulkan train libraries, and holding the
 * core kernels the tensor runtime launches with exactly their documented argument layouts.
 */
public class TrainInventoryResourceTest {

    private static final String LIBRARY_ID = "train";
    private static final String ROOT = "kernels/train";
    private static final String PATH = ROOT + "/" + KernelManifest.INVENTORY_FILE;

    /** Core kernels of kernels/train/core.cl: name -> {buffers, scalars, kinds, layout}. */
    private static final Map<String, String[]> CORE_KERNELS = coreKernels();

    private static Map<String, String[]> coreKernels() {
        Map<String, String[]> m = new LinkedHashMap<>();
        String[] copyLike = {"2", "3", "i,i,i", "iiiBB"};
        m.put("TR_Fill_F32", new String[]{"1", "3", "i,f,i", "ifiB"});
        m.put("TR_Copy_F32", copyLike);
        m.put("TR_Copy_Strided_F32", new String[]{"2", "22", repeat("i", ",", 22), repeat("i", "", 22) + "BB"});
        for (String cast : new String[]{"TR_Cast_I32_F32", "TR_Cast_F32_I32", "TR_Cast_BOOL_F32", "TR_Cast_F32_BOOL",
                "TR_Cast_F32_F16", "TR_Cast_F16_F32", "TR_Cast_F32_BF16", "TR_Cast_BF16_F32", "TR_Cast_I64_I32"}) {
            m.put(cast, copyLike);
        }
        return m;
    }

    private static String repeat(String s, String separator, int times) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < times; i++) {
            sb.append(i == 0 ? "" : separator).append(s);
        }
        return sb.toString();
    }

    private static ClassLoader loader() {
        return TrainInventoryResourceTest.class.getClassLoader();
    }

    private static String inventoryText() {
        String text = KernelManifest.readText(loader(), PATH);
        assertNotNull("classpath:" + PATH + " must ship with the kernel sources", text);
        return text;
    }

    private static Set<String> portableKernelUnits() {
        Set<String> units = new HashSet<>();
        for (String path : KernelManifest.load(loader(), ROOT).portablePaths()) {
            if (path.endsWith(".cl")) {
                units.add(path);
            }
        }
        return units;
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
        Set<String> portableUnits = portableKernelUnits();
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
            } else if (!cols[0].startsWith("TR_") && !cols[0].startsWith("ML_")) {
                violations.add((i + 1) + ": " + cols[0] + " lacks the TR_ or ML_ prefix of train kernels");
            }
        }
        assertTrue("classpath:" + PATH + ":\n  " + String.join("\n  ", violations), violations.isEmpty());
    }

    @Test
    public void everyManifestKernelSourceIsOnTheClasspath() {
        List<String> missing = new ArrayList<>();
        for (String unit : portableKernelUnits()) {
            if (KernelManifest.readText(loader(), ROOT + "/" + unit) == null) {
                missing.add(unit);
            }
        }
        assertTrue("manifest sources missing under classpath:" + ROOT + ": " + missing, missing.isEmpty());
    }

    @Test
    public void coreKernelsCarryTheirDocumentedSignatures() {
        Map<String, KernelSignature> signatures = CudaKernelLibrary.parseSignatures(inventoryText(), "classpath:" + PATH);
        for (Map.Entry<String, String[]> e : CORE_KERNELS.entrySet()) {
            String[] c = e.getValue();
            KernelSignature expected = KernelSignature.parseInventory(c[0], c[1], c[2], c[3]);
            assertEquals("inventory signature of " + e.getKey(), expected, signatures.get(e.getKey()));
        }
    }

    @Test
    public void coverageCheckAcceptsABuildThatHoldsEveryInventoryKernel() {
        Set<String> names = KernelManifest.inventory(loader(), ROOT).orElseThrow(AssertionError::new);
        Availability complete = KernelManifest.inventoryCoverage(loader(), LIBRARY_ID, ROOT, names,
                "the inventory itself");
        assertTrue(complete.reason(), complete.available());
    }

    @Test
    public void coverageCheckRejectsABuildThatLacksACoreKernel() {
        Set<String> names = new HashSet<>(KernelManifest.inventory(loader(), ROOT).orElseThrow(AssertionError::new));
        names.remove("TR_Copy_F32");
        Availability incomplete = KernelManifest.inventoryCoverage(loader(), LIBRARY_ID, ROOT, names,
                "a build without TR_Copy_F32");
        assertFalse("a build missing TR_Copy_F32 must not cover the train inventory", incomplete.available());
        assertTrue(incomplete.reason(), incomplete.reason().contains("TR_Copy_F32"));
    }
}
