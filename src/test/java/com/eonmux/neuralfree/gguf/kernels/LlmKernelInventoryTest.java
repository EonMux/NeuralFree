package com.eonmux.neuralfree.gguf.kernels;

import com.eonmux.neuralfree.acceleration.kernels.KernelInventoryChecks;
import com.eonmux.neuralfree.acceleration.kernels.KernelInventoryChecks.KernelSig;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import static com.eonmux.neuralfree.acceleration.kernels.KernelInventoryChecks.assertNoViolations;
import static com.eonmux.neuralfree.acceleration.kernels.KernelInventoryChecks.duplicates;
import static org.junit.Assert.assertEquals;

/**
 * Static inventory of the LLM kernel sources: every kernel signature of the manifest files (with the
 * kernel-generating family macros expanded) is checked against the argument limits and conventions, kernel
 * names are unique, and every kernel of a portable (untagged) manifest file has exactly one Vulkan variant
 * row in {@code shaders/llm/variants/*.tsv}, every row names such a kernel, and the checked-in
 * {@code kernels/llm/inventory.txt} lists exactly those kernels.
 *
 * <p>Runs from the project directory (the Gradle test working directory) and reads the source tree through
 * {@link KernelInventoryChecks}, because the GLSL variant lists are not shipped as resources.</p>
 */
public class LlmKernelInventoryTest {

    /**
     * Manifest files whose kernels exist only in the OpenCL program; must equal the {@code [opencl]} entries.
     * Empty: every LLM kernel is portable to CUDA and Vulkan.
     */
    private static final Set<String> OPENCL_ONLY = Collections.emptySet();

    private static volatile KernelInventoryChecks cached;

    @Test
    public void openClOnlyAllowlistMatchesManifestTags() throws IOException {
        assertEquals("OPENCL_ONLY must list exactly the [opencl] manifest entries",
                new ArrayList<>(OPENCL_ONLY), inventory().manifest().openClOnlyPaths());
    }

    @Test
    public void kernelsRespectArgumentLimits() throws IOException {
        assertNoViolations("argument limits", inventory().argumentLimitViolations());
    }

    @Test
    public void kernelArgumentsFollowConventions() throws IOException {
        assertNoViolations("argument conventions (no 8-byte scalars, by-value arrays or local-memory args)",
                inventory().conventionViolations());
    }

    @Test
    public void kernelNamesAreUnique() throws IOException {
        assertNoViolations("duplicate kernel names", inventory().duplicateKernelNames());
    }

    @Test
    public void everyPortableKernelHasExactlyOneVariantRow() throws IOException {
        assertNoViolations("variant rows", inventory().variantRowViolations(OPENCL_ONLY));
    }

    @Test
    public void everyVariantRowNamesAPortableKernel() throws IOException {
        assertNoViolations("variant rows without a kernel", inventory().orphanVariantRowViolations(OPENCL_ONLY));
    }

    /**
     * The checked-in {@code kernels/llm/inventory.txt} lists exactly the portable kernels of the sources, each
     * with the manifest file that defines it, so a kernel change that skipped regeneration fails here even
     * where the build has no preprocessor to compare the file with.
     */
    @Test
    public void checkedInInventoryListsEveryPortableKernelWithItsSource() throws IOException {
        inventory().assertInventoryCurrent(OPENCL_ONLY);
    }

    @Test
    public void scannerExpandsFamilyMacrosAndCountsArguments() {
        Map<String, String> files = new LinkedHashMap<>();
        files.put("common/fam.h",
                "#ifndef FAM_H\n#define FAM_H\n"
                        + "#define ONE(N, BS) \\\n"
                        + "NF_KERNEL void N##_MatVec(const int rows, /* c */ NF_GLOBAL const uchar* w, \\\n"
                        + "    NF_GLOBAL float* out) { int x = BS; }\n"
                        + "#define TWO(N) \\\n"
                        + "NF_KERNEL void N##_Embed(const int d, const float s, NF_GLOBAL float* out) { }\n"
                        + "#define ALL(N, BS) ONE(N, BS) TWO(N)\n"
                        + "#endif\n");
        files.put("a.cl",
                "ALL(Q4_K, (256))\n"
                        + "// NF_KERNEL void Commented(int a) {}\n"
                        + "NF_KERNEL void\nLLM_Direct(const int n,\n NF_GLOBAL const float* NF_RESTRICT x) { }\n"
                        + "#ifdef USE_SUBGROUPS\nNF_KERNEL void LLM_Alt(const int n) { }\n"
                        + "#else\nNF_KERNEL void LLM_Alt(const int n) { }\n#endif\n"
                        + "NF_KERNEL void LLM_Wide(const long n, NF_LOCAL float* tmp, const float v[4]) { }\n");
        List<KernelSig> kernels = KernelInventoryChecks.scan(files);
        Map<String, KernelSig> byName = new TreeMap<>();
        for (KernelSig k : kernels) {
            byName.put(k.name, k);
        }
        assertEquals(new TreeSet<>(Arrays.asList("Q4_K_MatVec", "Q4_K_Embed", "LLM_Direct", "LLM_Alt", "LLM_Wide")),
                byName.keySet());
        assertEquals(1, byName.get("Q4_K_MatVec").scalars);
        assertEquals(2, byName.get("Q4_K_MatVec").buffers);
        assertEquals(2, byName.get("Q4_K_Embed").scalars);
        assertEquals(1, byName.get("Q4_K_Embed").buffers);
        assertEquals(1, byName.get("LLM_Direct").scalars);
        assertEquals(1, byName.get("LLM_Direct").buffers);
        assertEquals("exclusive #ifdef/#else declarations are not duplicates", 0, duplicates(kernels).size());
        assertEquals(3, byName.get("LLM_Wide").conventionViolations.size());
        assertEquals(0, byName.get("Q4_K_MatVec").conventionViolations.size());

        files.put("b.cl", "NF_KERNEL void LLM_Direct(const int n) { }\n");
        assertEquals(1, duplicates(KernelInventoryChecks.scan(files)).size());
    }

    private static KernelInventoryChecks inventory() throws IOException {
        KernelInventoryChecks inv = cached;
        if (inv == null) {
            inv = KernelInventoryChecks.load(KernelLibraries.LLM_CONFIG);
            cached = inv;
        }
        return inv;
    }
}
