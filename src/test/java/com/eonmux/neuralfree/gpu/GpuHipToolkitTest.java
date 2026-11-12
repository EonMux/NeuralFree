package com.eonmux.neuralfree.gpu;

import neuralfree.gpu.GpuHipToolkit;
import neuralfree.gpu.GpuLibraryDefinitions;
import org.junit.Assume;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

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
     * The train library's definition (the maps {@link GpuLibraryDefinitions} hands the build script) compiles its HIP
     * code objects with {@code -ffp-contract=off} and its CUDA units with {@code -fmad=false}, and the HIP compile
     * passes each library's flags: training math is not contracted into fused multiply-adds on either toolchain.
     */
    @Test
    public void theTrainLibraryDefinitionCompilesWithoutContraction() throws IOException {
        Map<String, Object> train = GpuLibraryDefinitions.definition("train", 11264);
        assertEquals(Collections.singletonList("-ffp-contract=off"), train.get("hipFlags"));
        assertTrue(String.valueOf(train.get("cudaFlags")), ((List<?>) train.get("cudaFlags")).contains("-fmad=false"));
        assertEquals(Collections.emptyList(), GpuLibraryDefinitions.definition("llm", 11264).get("hipFlags"));
        String script = new String(Files.readAllBytes(
                new File(GpuKernelBuildCacheTest.projectRoot(), "gradle/gpu-kernels.gradle").toPath()),
                StandardCharsets.UTF_8);
        assertTrue("the build script must take its libraries from GpuLibraryDefinitions",
                script.contains("GpuLibraryDefinitions.definitions("));
        assertTrue("compileHipKernels must compile each unit with its library's hipFlags",
                script.contains("GpuHipToolkit.compileCommand(hipcc.path, archs.selected(), lib.hipFlags,"));
    }

    /**
     * The train library's HIP flags keep {@code a * b + c} two roundings in the AMDGPU code clang emits: a probe kernel
     * compiled for gfx1100 with those flags holds no fused multiply-add, while the same probe with the llm library's
     * flags (clang's HIP default contracts) holds one, so the probe does detect contraction. Runs where a clang with
     * the amdgcn target is installed ({@code neuralfree.test.clang}, default {@code /usr/bin/clang}).
     */
    @Test
    public void theTrainHipFlagsKeepMultiplyAndAddApartInTheIsa() throws IOException, InterruptedException {
        File clang = new File(System.getProperty("neuralfree.test.clang", "/usr/bin/clang"));
        Assume.assumeTrue("no clang at " + clang, clang.canExecute());
        Assume.assumeTrue("clang lacks the amdgcn target", targets(clang).contains("amdgcn"));
        Path dir = Files.createTempDirectory("nf-hip-contract");
        try {
            Path probe = dir.resolve("probe.hip");
            Files.write(probe, Arrays.asList(
                    "extern \"C\" __attribute__((global)) void probe(float* o, const float* a, const float* b,",
                    "                                                  const float* c) {",
                    "    const int i = __builtin_amdgcn_workitem_id_x();",
                    "    o[i] = a[i] * b[i] + c[i];",
                    "}"), StandardCharsets.UTF_8);
            @SuppressWarnings("unchecked")
            List<String> trainFlags = (List<String>) GpuLibraryDefinitions.definition("train", 11264).get("hipFlags");
            @SuppressWarnings("unchecked")
            List<String> llmFlags = (List<String>) GpuLibraryDefinitions.definition("llm", 11264).get("hipFlags");
            String train = isa(clang, probe, trainFlags, dir.resolve("train.s"));
            String llm = isa(clang, probe, llmFlags, dir.resolve("llm.s"));
            assertTrue("the probe must contract without the train flags:\n" + llm, FMA.matcher(llm).find());
            Matcher fused = FMA.matcher(train);
            if (fused.find()) {
                fail("fused multiply-add " + fused.group() + " with the train flags:\n" + train);
            }
        } finally {
            try (Stream<Path> files = Files.list(dir)) {
                for (Path f : (Iterable<Path>) files::iterator) {
                    Files.deleteIfExists(f);
                }
            }
            Files.deleteIfExists(dir);
        }
    }

    /** An AMDGPU fused or contracted multiply-add instruction. */
    private static final Pattern FMA = Pattern.compile("\\bv_(?:pk_)?(?:fma|fmac|fmamk|fmaak|mad|mac)\\w*_f32\\w*");

    /** The AMDGPU assembly clang emits for {@code source} as a HIP device unit for gfx1100 with {@code flags}. */
    private static String isa(File clang, Path source, List<String> flags, Path out)
            throws IOException, InterruptedException {
        List<String> cmd = new ArrayList<>(Arrays.asList(clang.getPath(), "-x", "hip", "-nogpuinc", "-nogpulib",
                "--cuda-device-only", "--offload-arch=gfx1100", "-std=c++17", "-O3", "-S"));
        cmd.addAll(flags);
        cmd.addAll(Arrays.asList(source.toString(), "-o", out.toString()));
        String log = run(cmd);
        assertTrue("clang did not write " + out + ":\n" + log, Files.isRegularFile(out));
        return new String(Files.readAllBytes(out), StandardCharsets.UTF_8);
    }

    private static String targets(File clang) throws IOException, InterruptedException {
        return run(Arrays.asList(clang.getPath(), "-print-targets"));
    }

    private static String run(List<String> cmd) throws IOException, InterruptedException {
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        byte[] output = p.getInputStream().readAllBytes();
        if (!p.waitFor(120, TimeUnit.SECONDS)) {
            p.destroyForcibly();
            fail("timed out: " + cmd);
        }
        String text = new String(output, StandardCharsets.UTF_8);
        assertEquals("exit status of " + cmd + ":\n" + text, 0, p.exitValue());
        return text;
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
