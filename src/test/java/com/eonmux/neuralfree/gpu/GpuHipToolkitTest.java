package com.eonmux.neuralfree.gpu;

import neuralfree.gpu.GpuHipToolkit;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The hipcc build knowledge the kernel build script runs: the exact command lines
 * ({@code hipcc -E} of the kernel scan, the per-unit {@code --offload-arch} compile), the
 * target selection from {@code -Pneuralfree.hip.archs}, the hipcc release parsing, the
 * supported-target listing, and the {@code index.tsv} / {@code build.properties} texts. The
 * build task calls this very class, so these are the commands the HIP kernels are built with.
 */
public class GpuHipToolkitTest {

    private static final String HIPCC = "/opt/rocm/bin/hipcc";

    @Test
    public void theCompileCommandCarriesStandardOptimizationAndEveryTarget() {
        List<String> cmd = GpuHipToolkit.compileCommand(HIPCC, Arrays.asList("gfx1100", "gfx90a"),
                Collections.<String>emptyList(), Collections.singletonList("-DNF_FA_SMEM_FLOATS=11264"),
                "llm/matmul.cl.cu", "llm/matmul.cl.hsaco");
        assertEquals(Arrays.asList(HIPCC, "-std=c++17", "-O3", "-DNF_FA_SMEM_FLOATS=11264",
                "--offload-arch=gfx1100", "--offload-arch=gfx90a",
                "llm/matmul.cl.cu", "-o", "llm/matmul.cl.hsaco"), cmd);
    }

    @Test
    public void theLibraryHipFlagsFollowTheOptimizationLevel() {
        List<String> cmd = GpuHipToolkit.compileCommand(HIPCC, Collections.singletonList("gfx1100"),
                Collections.singletonList("-ffp-contract=off"), Collections.<String>emptyList(),
                "train/eltwise.cl.cu", "train/eltwise.cl.hsaco");
        assertEquals(Arrays.asList(HIPCC, "-std=c++17", "-O3", "-ffp-contract=off", "--offload-arch=gfx1100",
                "train/eltwise.cl.cu", "-o", "train/eltwise.cl.hsaco"), cmd);
    }

    /**
     * The train library's definition in gradle/gpu-kernels.gradle compiles its HIP code objects with
     * {@code -ffp-contract=off}, and the HIP compile passes each library's flags: training math is not contracted
     * into fused multiply-adds, as its CUDA build's {@code -fmad=false} keeps it.
     */
    @Test
    public void theTrainLibraryDefinitionCompilesWithoutContraction() throws IOException {
        String script = new String(Files.readAllBytes(
                new File(GpuKernelBuildCacheTest.projectRoot(), "gradle/gpu-kernels.gradle").toPath()),
                StandardCharsets.UTF_8);
        Map<String, List<String>> hipFlags = libraryHipFlags(script);
        assertEquals(Collections.singletonList("-ffp-contract=off"), hipFlags.get("train"));
        assertTrue("every library must declare its hipFlags: " + hipFlags, hipFlags.containsKey("llm"));
        assertTrue("compileHipKernels must compile each unit with its library's hipFlags",
                script.contains("GpuHipToolkit.compileCommand(hipcc.path, archs.selected(), lib.hipFlags,"));
    }

    /** The hipFlags of every entry of the script's gpuLibraryDefinitions, by library id. */
    private static Map<String, List<String>> libraryHipFlags(String script) {
        int start = script.indexOf("def gpuLibraryDefinitions = [");
        assertTrue("gradle/gpu-kernels.gradle must define gpuLibraryDefinitions", start >= 0);
        int end = script.indexOf("\n]", start);
        String[] entries = script.substring(start, end).split("\\[id\\s*:");
        Pattern id = Pattern.compile("^\\s*'([a-z0-9]+)'");
        Pattern flags = Pattern.compile("hipFlags\\s*:\\s*\\[([^\\]]*)\\]");
        Map<String, List<String>> byId = new LinkedHashMap<>();
        for (int i = 1; i < entries.length; i++) {
            Matcher idMatch = id.matcher(entries[i]);
            assertTrue("a library entry without an id: " + entries[i], idMatch.find());
            Matcher flagMatch = flags.matcher(entries[i]);
            if (!flagMatch.find()) {
                continue;
            }
            List<String> values = new ArrayList<>();
            for (String value : flagMatch.group(1).split(",")) {
                String v = value.trim();
                if (!v.isEmpty()) {
                    values.add(v.substring(1, v.length() - 1));
                }
            }
            byId.put(idMatch.group(1), values);
        }
        return byId;
    }

    @Test
    public void thePreprocessCommandExpandsOneUnitForTheKernelScan() {
        List<String> cmd = GpuHipToolkit.preprocessCommand(HIPCC, Collections.<String>emptyList(),
                "train/eltwise.cl.cu", "train/eltwise.cl.ii");
        assertEquals(Arrays.asList(HIPCC, "-E", "-std=c++17", "-DNF_HIP_PREPROCESS",
                "train/eltwise.cl.cu", "-o", "train/eltwise.cl.ii"), cmd);
    }

    @Test
    public void aCompileWithoutATargetIsRejected() {
        try {
            GpuHipToolkit.compileCommand(HIPCC, Collections.<String>emptyList(), Collections.<String>emptyList(),
                    Collections.<String>emptyList(), "u.cu", "u.hsaco");
            fail("a HIP unit compiles for at least one gfx target");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("at least one gfx target"));
        }
    }

    @Test
    public void theDefaultArchSetIsTheLlamaCppTargetList() {
        assertEquals(Arrays.asList("gfx906", "gfx908", "gfx90a", "gfx942", "gfx1030", "gfx1100",
                "gfx1101", "gfx1102", "gfx1200", "gfx1201"),
                GpuHipToolkit.selectArchs(GpuHipToolkit.DEFAULT_ARCHS, null).selected());
    }

    @Test
    public void targetParsingDeduplicatesStripsSuffixesAndKeepsOrder() {
        GpuHipToolkit.ArchSelection s = GpuHipToolkit.selectArchs(
                " gfx1100 , gfx90a:xnack- , GFX1100 ,, gfx942 ", null);
        assertEquals(Arrays.asList("gfx1100", "gfx90a", "gfx942"), s.selected());
        assertTrue(s.dropped().isEmpty());
    }

    @Test
    public void anUnsupportedTargetIsDroppedAndReported() {
        GpuHipToolkit.ArchSelection s = GpuHipToolkit.selectArchs("gfx1100,gfx906,gfx90a",
                new HashSet<>(Collections.singletonList("gfx1100")));
        assertEquals(Collections.singletonList("gfx1100"), s.selected());
        assertEquals(Arrays.asList("gfx906", "gfx90a"), s.dropped());
    }

    @Test
    public void anUnknownSupportedSetSelectsEverythingRequested() {
        assertEquals(Collections.singletonList("gfx1201"),
                GpuHipToolkit.selectArchs("gfx1201", null).selected());
        assertEquals(Collections.singletonList("gfx1201"),
                GpuHipToolkit.selectArchs("gfx1201", Collections.<String>emptySet()).selected());
    }

    @Test
    public void nonGfxEntriesAndEmptyListsAreRejected() {
        for (String bad : new String[]{null, "", "   ", "sm_80,gfx1100", "gfx1100,gfx", "gfx1100,gfxzzz"}) {
            try {
                GpuHipToolkit.selectArchs(bad, null);
                fail("'" + bad + "' is not a usable arch list");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("neuralfree.hip.archs"));
            }
        }
    }

    @Test
    public void aSupportedSetMatchingNothingIsRejected() {
        try {
            GpuHipToolkit.selectArchs("gfx1100", new HashSet<>(Collections.singletonList("gfx906")));
            fail("a supported set without any requested target must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("gfx1100"));
            assertTrue(expected.getMessage(), expected.getMessage().contains("gfx906"));
        }
    }

    @Test
    public void theVersionComesFromTheHipReleaseLineOrTheToolchainBanner() {
        assertEquals("6.2.41133-9aa8ec0", GpuHipToolkit.parseVersion(
                "HIP version: 6.2.41133-9aa8ec0\nAMD clang version 17.0.0\n"));
        assertEquals("5.7.1", GpuHipToolkit.parseVersion(
                "/opt/rocm/llvm/bin/clang --version\ntoolchain version 5.7.1 of ROCm\n"));
        assertEquals("17.0.0", GpuHipToolkit.parseVersion(
                "AmD cLanG VersiON 17.0.0 (rocm-langbuild-17.0-1234)\nTarget: x86_64-unknown-linux-gnu\n"));
        try {
            GpuHipToolkit.parseVersion("clang: error: no such file or directory");
            fail("an output without a release must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("HIP release"));
        }
        try {
            GpuHipToolkit.parseVersion(null);
            fail("a missing version output must be rejected");
        } catch (IllegalArgumentException expected) {
            // the message carries the (null) output for the build log
        }
    }

    @Test
    public void theSupportedGpuListingYieldsTheOrderedTargetNames() {
        List<String> gpus = GpuHipToolkit.parseSupportedGpus(
                "AMD GPU list:\n"
                        + "  gfx906\n"
                        + "  gfx908\n"
                        + "  gfx1100\n"
                        + "  gfx90a:xnack-\n"
                        + "  gfx1100 (as published)\n"
                        + "  nvptx64-NVIDIA-CUDA\n");
        assertEquals(Arrays.asList("gfx906", "gfx908", "gfx1100", "gfx90a"), gpus);
        assertTrue(GpuHipToolkit.parseSupportedGpus("").isEmpty());
        assertTrue(GpuHipToolkit.parseSupportedGpus("clang: error: unknown argument").isEmpty());
        assertTrue(GpuHipToolkit.parseSupportedGpus(null).isEmpty());
    }

    @Test
    public void theIndexTextIsSortedTabSeparatedAndNewlineTerminated() {
        Map<String, String> index = new LinkedHashMap<>();
        index.put("nf_zeta", "llm/nf_zeta.cl.hsaco");
        index.put("nf_alpha", "llm/nf_alpha.cl.hsaco");
        assertEquals("nf_alpha\tllm/nf_alpha.cl.hsaco\nnf_zeta\tllm/nf_zeta.cl.hsaco\n",
                GpuHipToolkit.indexText(index));
        assertEquals("", GpuHipToolkit.indexText(new HashMap<String, String>()));
    }

    @Test
    public void theBuildPropertiesTextRecordsTheEntriesInOrder() {
        assertEquals("hipcc.version=6.2.41133\nhip.archs=gfx1100,gfx90a\nkernel.format=hsaco\nfa.smem.floats=11264\n",
                GpuHipToolkit.buildPropertiesText(Arrays.asList(
                        "hipcc.version=6.2.41133", "hip.archs=gfx1100,gfx90a",
                        "kernel.format=hsaco", "fa.smem.floats=11264")));
        assertEquals("", GpuHipToolkit.buildPropertiesText(Collections.<String>emptyList()));
    }

    @Test
    public void theValueObjectsCompareByContent() {
        assertEquals(GpuHipToolkit.selectArchs("gfx1100,gfx90a", null),
                GpuHipToolkit.selectArchs("gfx1100,gfx90a", null));
        GpuHipToolkit.ArchSelection s = GpuHipToolkit.selectArchs("gfx1100,gfx906",
                new HashSet<>(Collections.singletonList("gfx1100")));
        assertEquals(s, GpuHipToolkit.selectArchs("gfx1100,gfx906",
                new HashSet<>(Collections.singletonList("gfx1100"))));
        assertTrue(s.toString().contains("gfx906"));
        assertTrue(s.toString().contains("gfx1100"));
    }
}
