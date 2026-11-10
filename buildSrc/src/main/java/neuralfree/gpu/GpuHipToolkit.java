package neuralfree.gpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Pure hipcc build knowledge of the generated HIP kernel build (gradle/gpu-kernels.gradle): the
 * {@code --offload-arch} target list, the hipcc release, the per-unit command lines and the
 * {@code index.tsv} / {@code build.properties} texts. The build script finds the tool and runs the
 * commands; this class holds the parts both the script and the module's test suite
 * ({@code GpuHipToolkitTest}) must agree on, so it depends on no Gradle API. The counterpart for
 * nvcc is {@code GpuCudaToolkit}.
 */
public final class GpuHipToolkit {

    /** Default targets of {@code -Pneuralfree.hip.archs}: the llama.cpp HIP build's --offload-arch set. */
    public static final String DEFAULT_ARCHS =
            "gfx906,gfx908,gfx90a,gfx942,gfx1030,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201";

    /** Language standard of every HIP unit: the sources are CUDA C++ dialect, as nvcc's units are. */
    public static final String STD_FLAG = "-std=c++17";

    /** Define of the {@code hipcc -E} pass that expands the units for the kernel scan. */
    public static final String PREPROCESS_DEFINE = "-DNF_HIP_PREPROCESS";

    /** File extension hipcc produces code objects under; kernel.format of build.properties. */
    public static final String CODE_OBJECT_SUFFIX = ".hsaco";

    private static final Pattern HIP_VERSION =
            Pattern.compile("(?i)hip[ -]version:?\\s+(\\d+\\.\\d+\\.\\d+[0-9A-Za-z._-]*)");
    private static final Pattern TOOLCHAIN_VERSION = Pattern.compile("(?i)\\bversion\\s+(\\d+\\.\\d+\\.\\d+)");
    private static final Pattern GFX_TARGET = Pattern.compile("\\bgfx[0-9a-f]+\\b");

    private GpuHipToolkit() {
    }

    /** Result of {@link #selectArchs}: the targets to compile and the requested ones hipcc cannot build. */
    public static final class ArchSelection {
        private final List<String> selected;
        private final List<String> dropped;

        private ArchSelection(List<String> selected, List<String> dropped) {
            this.selected = selected;
            this.dropped = dropped;
        }

        /** The --offload-arch targets of the build, in requested order, capability suffixes stripped. */
        public List<String> selected() {
            return selected;
        }

        /** The requested targets the supported-targets listing does not hold; empty when nothing was dropped. */
        public List<String> dropped() {
            return dropped;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof ArchSelection)) {
                return false;
            }
            ArchSelection that = (ArchSelection) o;
            return selected.equals(that.selected) && dropped.equals(that.dropped);
        }

        @Override
        public int hashCode() {
            return 31 * selected.hashCode() + dropped.hashCode();
        }

        @Override
        public String toString() {
            return "ArchSelection{selected=" + selected + ", dropped=" + dropped + "}";
        }
    }

    /**
     * Selects the gfx targets of the build from the comma-separated {@code archsProperty}
     * ({@code -Pneuralfree.hip.archs}; entries such as {@code gfx1100} or {@code gfx90a:xnack-},
     * recorded without the capability suffix), keeping the requested order and dropping duplicates.
     * When {@code supported} is non-empty (the parsed {@code hipcc --print-supported-gpus} listing)
     * only the supported requested targets are selected and the rest are reported as dropped; a null
     * or empty {@code supported} (no listing available) selects everything requested.
     *
     * @throws IllegalArgumentException when the property is empty or holds a non-gfx entry, or when a
     *                                  known supported list matches none of the requested targets
     */
    public static ArchSelection selectArchs(String archsProperty, Set<String> supported) {
        List<String> requested = new ArrayList<>();
        if (archsProperty != null) {
            for (String part : archsProperty.split(",")) {
                String p = part.trim();
                if (p.isEmpty()) {
                    continue;
                }
                requested.add(requireGfxTarget(p));
            }
        }
        if (requested.isEmpty()) {
            throw new IllegalArgumentException("-Pneuralfree.hip.archs must list gfx targets such as "
                    + DEFAULT_ARCHS + "; got '" + archsProperty + "'");
        }
        Set<String> unique = new LinkedHashSet<>(requested);
        if (supported == null || supported.isEmpty()) {
            return new ArchSelection(Collections.unmodifiableList(new ArrayList<>(unique)),
                    Collections.<String>emptyList());
        }
        List<String> selected = new ArrayList<>();
        List<String> dropped = new ArrayList<>();
        for (String arch : unique) {
            (supported.contains(arch) ? selected : dropped).add(arch);
        }
        if (selected.isEmpty()) {
            throw new IllegalArgumentException("none of the HIP archs " + unique
                    + " is supported by hipcc (supports " + new ArrayList<String>(supported) + ")");
        }
        return new ArchSelection(Collections.unmodifiableList(selected), Collections.unmodifiableList(dropped));
    }

    /** The entry stripped of its {@code :capability} suffix, if it names a gfx target; otherwise throws. */
    private static String requireGfxTarget(String entry) {
        String name = entry.toLowerCase(Locale.ROOT);
        int colon = name.indexOf(':');
        String base = colon >= 0 ? name.substring(0, colon) : name;
        if (!base.startsWith("gfx") || base.length() <= 3) {
            throw new IllegalArgumentException("-Pneuralfree.hip.archs entry '" + entry
                    + "' is not a gfx target (expected gfx plus hex digits)");
        }
        for (int i = 3; i < base.length(); i++) {
            char c = base.charAt(i);
            if ((c < '0' || c > '9') && (c < 'a' || c > 'f')) {
                throw new IllegalArgumentException("-Pneuralfree.hip.archs entry '" + entry
                        + "' is not a gfx target (expected gfx plus hex digits)");
            }
        }
        return base;
    }

    /**
     * The hipcc release from {@code hipcc --version} output: the {@code HIP version} line of modern
     * hipcc, or the clang toolchain banner of the wrapper. Returns the version token, for example
     * {@code 6.2.41133-9aa8ec0} or {@code 17.0.0}.
     *
     * @throws IllegalArgumentException when the output names no release
     */
    public static String parseVersion(String versionOutput) {
        if (versionOutput != null) {
            Matcher hip = HIP_VERSION.matcher(versionOutput);
            if (hip.find()) {
                return hip.group(1);
            }
            Matcher toolchain = TOOLCHAIN_VERSION.matcher(versionOutput);
            if (toolchain.find()) {
                return toolchain.group(1);
            }
        }
        throw new IllegalArgumentException("cannot read the HIP release from hipcc --version:\n" + versionOutput);
    }

    /**
     * The gfx target names in a {@code hipcc --print-supported-gpus} (clang --print-supported-gpus)
     * listing, in order; empty when the listing names none. An empty result tells the build script to
     * treat the supported set as unknown and pass every requested target to hipcc.
     */
    public static List<String> parseSupportedGpus(String listing) {
        if (listing == null) {
            return Collections.emptyList();
        }
        Set<String> found = new LinkedHashSet<>();
        Matcher m = GFX_TARGET.matcher(listing);
        while (m.find()) {
            found.add(m.group());
        }
        return Collections.unmodifiableList(new ArrayList<>(found));
    }

    /**
     * The {@code hipcc -E} command that expands one assembled unit (every manifest header, then the
     * .cl) for {@code GpuKernelSources.scanKernels}: the {@code NF_HIP_PREPROCESS} define of the
     * inventory pass, then the library's defines ({@code -DNF_FA_SMEM_FLOATS} for the LLM library).
     */
    public static List<String> preprocessCommand(String hipcc, List<String> defines, String unitPath,
                                                 String preprocessedPath) {
        List<String> cmd = new ArrayList<>();
        cmd.add(hipcc);
        cmd.add("-E");
        cmd.add(STD_FLAG);
        cmd.add(PREPROCESS_DEFINE);
        cmd.addAll(defines);
        cmd.add(unitPath);
        cmd.add("-o");
        cmd.add(preprocessedPath);
        return cmd;
    }

    /**
     * The hipcc command that compiles one assembled unit to its code object: every selected target is
     * an {@code --offload-arch} flag and the {@code .hsaco} output makes hipcc compile device code
     * only (one multi-target code object per kernel source, as the CUDA build's fatbin holds one
     * module with a gencode entry per SM). {@code flags} are the library's own clang options (the
     * train library's {@code -ffp-contract=off}, the HIP form of its {@code -fmad=false}); nvcc's other
     * math-mode flags have no HIP form that keeps the NaN semantics the kernels rely on.
     */
    public static List<String> compileCommand(String hipcc, List<String> archs, List<String> flags,
                                              List<String> defines, String unitPath, String outputPath) {
        if (archs.isEmpty()) {
            throw new IllegalArgumentException("a HIP unit compiles for at least one gfx target");
        }
        List<String> cmd = new ArrayList<>();
        cmd.add(hipcc);
        cmd.add(STD_FLAG);
        cmd.add("-O3");
        cmd.addAll(flags);
        cmd.addAll(defines);
        for (String arch : archs) {
            cmd.add("--offload-arch=" + arch);
        }
        cmd.add(unitPath);
        cmd.add("-o");
        cmd.add(outputPath);
        return cmd;
    }

    /**
     * The {@code index.tsv} text: {@code kernel_name<TAB>resource} sorted by kernel name, one line per
     * kernel, exactly what {@code HipKernelLibrary.parseIndex} reads back.
     */
    public static String indexText(Map<String, String> kernelToResource) {
        Map<String, String> sorted = new TreeMap<>(kernelToResource);
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : sorted.entrySet()) {
            sb.append(e.getKey()).append('\t').append(e.getValue()).append('\n');
        }
        return sb.toString();
    }

    /**
     * The {@code build.properties} text: the entries in order ({@code hipcc.version}, {@code hip.archs},
     * {@code kernel.format} and the optional {@code fa.smem.floats}), newline-terminated.
     */
    public static String buildPropertiesText(List<String> entries) {
        StringBuilder sb = new StringBuilder();
        for (String entry : entries) {
            sb.append(entry).append('\n');
        }
        return sb.toString();
    }
}
