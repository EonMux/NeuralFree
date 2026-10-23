package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.kernels.KernelInventoryChecks;
import com.eonmux.neuralfree.acceleration.kernels.KernelInventoryChecks.KernelSig;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryConfig;
import com.eonmux.neuralfree.acceleration.kernels.KernelManifest;
import com.eonmux.neuralfree.acceleration.kernels.VulkanKernelLibrary;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.eonmux.neuralfree.acceleration.kernels.KernelInventoryChecks.assertNoViolations;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assume.assumeTrue;

/**
 * Static inventory of the {@code train} kernel library: every kernel signature of the manifest files is
 * checked against the argument limits and conventions, kernel names are unique and carry the library prefix,
 * every kernel has exactly one Vulkan variant row in {@code shaders/train/variants/*.tsv} built from the GLSL
 * twin of its {@code .cl} file, the checked-in {@code kernels/train/inventory.txt} lists exactly those kernels
 * and equals {@link TrainOps.Kernels#ALL}, the GLSL sources keep the library's Vulkan conventions (no 8-,
 * 16- or 64-bit types, no subgroup operations, built-in invocation ids only in the multi-dimensional GEMM),
 * every built Vulkan binding is read-only exactly when its OpenCL buffer argument points to {@code const} data,
 * and the backward kernels keep the argument lists their dispatchers launch them with.
 *
 * <p>Runs from the project directory (the Gradle test working directory) and reads the source tree through
 * {@link KernelInventoryChecks}, because the GLSL sources and variant lists are not shipped as resources.</p>
 */
public class TrainKernelInventoryTest {

    private static final KernelLibraryConfig CONFIG = KernelLibraries.TRAIN_CONFIG;

    /** Manifest files whose kernels exist only in the OpenCL program: none, every train kernel is portable. */
    private static final Set<String> OPENCL_ONLY = Collections.emptySet();

    /** The only shader that reads the built-in ids directly: TR_Gemm_F32 is the one multi-dimensional launch. */
    private static final String BUILTIN_ID_SHADER = "gemm.comp";

    /** Sub-word and 64-bit types or storage extensions, and subgroup operations, none of which the library uses. */
    private static final Pattern FORBIDDEN_GLSL = Pattern.compile(
            "8bit|16bit|int64|int8_t|int16_t|float16_t|float64_t|\\bdouble\\b|\\bdvec[234]\\b|(?i:subgroup)");
    private static final Pattern BUILTIN_IDS = Pattern.compile("\\b(gl_GlobalInvocationID|gl_WorkGroupID)\\b");

    /**
     * Launch ABI of the backward kernels, as their GPU dispatchers pass it: the declaring source, the scalar
     * kinds in argument order, and one read-only flag per buffer in argument order ({@code 1} for an input,
     * {@code 0} for an output or a read-write accumulator).
     */
    private static final Map<String, Abi> BACKWARD_ABI;

    static {
        Map<String, Abi> m = new LinkedHashMap<>();
        put(m, new Abi(TrainOps.Kernels.CONV2D_WEIGHT_GRAD_F32, "conv.cl", ints(20), "110"));
        put(m, new Abi(TrainOps.Kernels.MAX_POOL2D_BACKWARD_F32, "pool.cl", ints(18), "1110"));
        put(m, new Abi(TrainOps.Kernels.AVG_POOL2D_BACKWARD_F32, "pool.cl", ints(15), "10"));
        put(m, new Abi(TrainOps.Kernels.ADAPTIVE_AVG_POOL2D_BACKWARD_F32, "pool.cl", ints(8), "10"));
        put(m, new Abi(TrainOps.Kernels.MAX_POOL2D_SELECT_F32, "pool.cl", ints(18), "1110"));
        put(m, new Abi(TrainOps.Kernels.LAYER_NORM_ROWS_BACKWARD_F32, "norm.cl", ints(2) + "f" + ints(5), "11100"));
        put(m, new Abi(TrainOps.Kernels.RMS_NORM_ROWS_BACKWARD_F32, "norm.cl", ints(2) + "f" + ints(5), "11100"));
        put(m, new Abi(TrainOps.Kernels.NORM_AFFINE_GRAD_COLS_F32, "norm.cl", ints(8), "1110"));
        put(m, new Abi(TrainOps.Kernels.CHANNEL_GRAD_SUMS_F32, "norm.cl", ints(4) + "f" + ints(3), "11110"));
        put(m, new Abi(TrainOps.Kernels.GROUP_GRAD_SUMS_F32, "norm.cl", ints(4) + "f" + ints(3), "111110"));
        put(m, new Abi(TrainOps.Kernels.CHANNEL_NORM_BACKWARD_APPLY_F32, "norm.cl", ints(6) + "f" + ints(4),
                "1111110"));
        put(m, new Abi(TrainOps.Kernels.INDEX_ADD_ROWS_F32, "norm.cl", ints(9), "110"));
        put(m, new Abi(TrainOps.Kernels.SEGMENT_REDUCE_BACKWARD_F32, "segment.cl", ints(9), "111110"));
        BACKWARD_ABI = Collections.unmodifiableMap(m);
    }

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

    /** Classical-ML kernels ({@code ml/}) are named {@code ML_*}, every other train kernel {@code TR_*}. */
    @Test
    public void kernelNamesCarryTheLibraryPrefix() throws IOException {
        List<String> violations = new ArrayList<>();
        for (KernelSig k : inventory().kernels()) {
            String prefix = k.file.startsWith("ml/") ? "ML_" : "TR_";
            if (!k.name.startsWith(prefix)) {
                violations.add(k + " must be named " + prefix + "*");
            }
        }
        assertNoViolations("kernel name prefixes", violations);
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
     * Every variant row builds its kernel from the GLSL twin of the kernel's {@code .cl} file and carries no
     * flags column, because the train library has no subgroup variants.
     */
    @Test
    public void variantRowsUseTheTwinShaderWithoutFlags() throws IOException {
        Map<String, String> fileOf = new HashMap<>();
        for (KernelSig k : inventory().kernels()) {
            fileOf.put(k.name, k.file);
        }
        List<String> violations = new ArrayList<>();
        for (Path tsv : files(shaderRoot().resolve("variants"), ".tsv")) {
            String[] lines = TrainKernelSources.read(tsv).split("\r?\n", -1);
            for (int i = 0; i < lines.length; i++) {
                String line = lines[i];
                if (line.trim().isEmpty() || line.trim().startsWith("#")) {
                    continue;
                }
                String where = tsv.getFileName() + ":" + (i + 1);
                String[] cols = line.split("\t", -1);
                if (cols.length > 3) {
                    violations.add(where + ": train rows take no flags column (no subgroup variants)");
                }
                String file = fileOf.get(cols[0].trim());
                if (file != null && cols.length > 1
                        && !cols[1].trim().equals(TrainKernelSources.twinOf(file))) {
                    violations.add(where + ": " + cols[0].trim() + " is defined in " + file + " but built from "
                            + cols[1].trim() + ", not " + TrainKernelSources.twinOf(file));
                }
            }
        }
        assertNoViolations("variant row sources", violations);
    }

    /**
     * The checked-in {@code kernels/train/inventory.txt} lists exactly the kernels of the sources, each with the
     * manifest file that defines it, so a kernel change that skipped regeneration fails here even where the build
     * has no preprocessor to compare the file with.
     */
    @Test
    public void checkedInInventoryListsEveryPortableKernelWithItsSource() throws IOException {
        inventory().assertInventoryCurrent(OPENCL_ONLY);
    }

    /** The Java kernel-name list that TrainGpu prewarms equals the inventory, so no backend misses a kernel. */
    @Test
    public void inventoryEqualsTrainOpsKernelNames() throws IOException {
        Set<String> expected = new TreeSet<>(TrainOps.Kernels.ALL);
        Path file = KernelInventoryChecks.RESOURCES.resolve(CONFIG.sourceRoot())
                .resolve(KernelManifest.INVENTORY_FILE);
        Set<String> listed = KernelManifest.parseInventory(TrainKernelSources.read(file), file.toString());
        assertEquals(file + " against TrainOps.Kernels.ALL", expected, new TreeSet<>(listed));
        Set<String> declared = inventory().kernels().stream()
                .map(k -> k.name)
                .collect(Collectors.toCollection(TreeSet::new));
        assertEquals("kernel declarations against TrainOps.Kernels.ALL", expected, declared);
    }

    /**
     * Each backward kernel keeps the argument list its dispatcher launches with: the checked-in inventory row
     * carries the expected source, counts, scalar kinds and parameter layout, and the {@code .cl} declaration
     * has the expected source, scalar count and read-only buffer flags. A reordered, added or removed argument
     * fails here even when the kernel sources, the inventory and {@link TrainOps} still agree with each other.
     */
    @Test
    public void backwardKernelsKeepTheirLaunchAbi() throws IOException {
        Map<String, String[]> rows = inventoryRows();
        Map<String, KernelSig> declared = new HashMap<>();
        for (KernelSig k : inventory().kernels()) {
            declared.put(k.name, k);
        }
        List<String> violations = new ArrayList<>();
        for (Abi abi : BACKWARD_ABI.values()) {
            if (!TrainOps.Kernels.ALL.contains(abi.kernel)) {
                violations.add(abi.kernel + " is missing from TrainOps.Kernels.ALL");
            }
            String[] row = rows.get(abi.kernel);
            String expectedRow = String.join("\t", abi.inventoryRow());
            if (row == null) {
                violations.add(abi.kernel + " has no row in kernels/train/inventory.txt");
            } else if (!String.join("\t", row).equals(expectedRow)) {
                violations.add("inventory row " + String.join("\t", row) + " differs from the launch ABI "
                        + expectedRow);
            }
            KernelSig k = declared.get(abi.kernel);
            if (k == null) {
                violations.add(abi.kernel + " is not declared by any manifest source");
            } else if (!k.file.equals(abi.source) || k.scalars != abi.scalarKinds.length()
                    || !k.readOnlyBuffers.equals(abi.readOnly)) {
                violations.add(k + " declares " + k.scalars + " scalars and read-only flags " + k.readOnlyBuffers
                        + "; the launch ABI is " + abi.scalarKinds.length() + " scalars in " + abi.source
                        + " with read-only flags " + abi.readOnly);
            }
        }
        assertNoViolations("backward kernel launch ABI", violations);
    }

    @Test
    public void shadersUseNoSubWordOr64BitTypesAndNoSubgroups() throws IOException {
        List<String> violations = new ArrayList<>();
        for (Path p : glslSources()) {
            Matcher m = FORBIDDEN_GLSL.matcher(TrainKernelSources.stripComments(TrainKernelSources.read(p)));
            while (m.find()) {
                violations.add(shaderRoot().relativize(p) + ": uses '" + m.group() + "'");
            }
        }
        assertNoViolations("GLSL types and extensions outside the train library profile", violations);
    }

    /**
     * 1-D shaders read their position only through {@code nf_global_id()}/{@code nf_wg_id()}, because the Vulkan
     * backend folds a large 1-D group count into two dimensions.
     */
    @Test
    public void onlyGemmReadsBuiltInInvocationIds() throws IOException {
        List<String> violations = new ArrayList<>();
        for (Path p : glslSources()) {
            String rel = shaderRoot().relativize(p).toString().replace('\\', '/');
            if (rel.equals(BUILTIN_ID_SHADER)) {
                continue;
            }
            Matcher m = BUILTIN_IDS.matcher(TrainKernelSources.stripComments(TrainKernelSources.read(p)));
            if (m.find()) {
                violations.add(rel + " reads " + m.group(1) + "; use nf_global_id()/nf_wg_id() or tr_gid()");
            }
        }
        assertNoViolations("built-in invocation ids outside " + BUILTIN_ID_SHADER, violations);
    }

    /**
     * Every Vulkan binding is read-only ({@code NonWritable}, the {@code .ro} flags the build records in
     * {@code shaders/train/index.properties} from the compiled SPIR-V) exactly when the OpenCL buffer argument it
     * stands for points to {@code const} data. The Vulkan backend tracks every binding that is not read-only as
     * written, so an input without {@code readonly} adds barriers between kernels that only read the buffer, and
     * a read-only binding for a non-const argument means the GLSL twin never writes the kernel's output. Needs
     * the SPIR-V the build generates onto the test classpath; skipped when the shaders were not built.
     */
    @Test
    public void vulkanReadOnlyBindingsMatchConstBufferArguments() throws IOException {
        String path = CONFIG.spvRoot() + "/" + VulkanKernelLibrary.INDEX_FILE;
        Properties index = new Properties();
        try (InputStream in = TrainKernelInventoryTest.class.getClassLoader().getResourceAsStream(path)) {
            assumeTrue("classpath:" + path + " is missing: the Vulkan shaders were not built", in != null);
            try (Reader r = new InputStreamReader(in, StandardCharsets.UTF_8)) {
                index.load(r);
            }
        }
        List<String> violations = new ArrayList<>();
        for (KernelSig k : inventory().kernels()) {
            String ro = index.getProperty(k.name + ".ro");
            if (ro == null) {
                violations.add(k + ": classpath:" + path + " has no " + k.name + ".ro");
            } else if (!ro.trim().equals(k.readOnlyBuffers)) {
                violations.add(k + ": SPIR-V read-only flags " + ro.trim() + ", const buffer arguments of the .cl "
                        + k.readOnlyBuffers);
            }
        }
        assertNoViolations("Vulkan read-only bindings against the const buffer arguments", violations);
    }

    /** The shared dialect and GLSL base come from the shared roots; the library carries no copy of its own. */
    @Test
    public void libraryHasNoCopyOfTheSharedSources() {
        Path dialect = KernelInventoryChecks.RESOURCES.resolve(CONFIG.sourceRoot()).resolve("common/dialect.h");
        Path base = shaderRoot().resolve("common/nf_base.glsl");
        assertFalse(dialect + " shadows kernels/common/dialect.h", Files.exists(dialect));
        assertFalse(base + " shadows shaders/common/nf_base.glsl", Files.exists(base));
    }

    private static Path shaderRoot() {
        return KernelInventoryChecks.RESOURCES.resolve(CONFIG.spvRoot());
    }

    private static List<Path> glslSources() throws IOException {
        List<Path> out = new ArrayList<>(files(shaderRoot(), ".comp"));
        out.addAll(files(shaderRoot(), ".glsl"));
        assertFalse("no GLSL sources under " + shaderRoot(), out.isEmpty());
        return out;
    }

    /** Regular files under a directory, recursively, with the given extension, in path order. */
    private static List<Path> files(Path dir, String extension) throws IOException {
        try (Stream<Path> s = Files.walk(dir)) {
            return s.filter(p -> Files.isRegularFile(p) && p.getFileName().toString().endsWith(extension))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    /**
     * Columns after the kernel name of every row of the checked-in {@code kernels/train/inventory.txt}, keyed by
     * kernel name: source, buffers, scalars, scalar kinds and parameter layout.
     */
    private static Map<String, String[]> inventoryRows() throws IOException {
        Path file = KernelInventoryChecks.RESOURCES.resolve(CONFIG.sourceRoot())
                .resolve(KernelManifest.INVENTORY_FILE);
        String text = TrainKernelSources.read(file);
        Set<String> names = KernelManifest.parseInventory(text, file.toString());
        Map<String, String[]> rows = new HashMap<>();
        for (String line : text.split("\r?\n", -1)) {
            String[] cols = line.split("\t", -1);
            if (names.contains(cols[0])) {
                rows.put(cols[0], Arrays.copyOfRange(cols, 1, cols.length));
            }
        }
        return rows;
    }

    private static void put(Map<String, Abi> abis, Abi abi) {
        if (abis.put(abi.kernel, abi) != null) {
            throw new IllegalStateException("duplicate launch ABI for " + abi.kernel);
        }
    }

    /** {@code n} integer scalar kinds. */
    private static String ints(int n) {
        StringBuilder sb = new StringBuilder(n);
        for (int i = 0; i < n; i++) {
            sb.append('i');
        }
        return sb.toString();
    }

    /** Immutable launch ABI of one kernel. */
    private static final class Abi {
        final String kernel;
        final String source;
        /** One of {@code i}, {@code u}, {@code f} per scalar argument, in argument order. */
        final String scalarKinds;
        /** One {@code 0}/{@code 1} per buffer argument, in argument order; {@code 1} for a read-only input. */
        final String readOnly;

        Abi(String kernel, String source, String scalarKinds, String readOnly) {
            if (!scalarKinds.matches("[iuf]*") || !readOnly.matches("[01]+")) {
                throw new IllegalArgumentException(kernel + ": scalar kinds must be [iuf]* and read-only flags [01]+");
            }
            this.kernel = kernel;
            this.source = source;
            this.scalarKinds = scalarKinds;
            this.readOnly = readOnly;
        }

        /** The inventory columns after the name: source, buffers, scalars, comma-separated kinds, layout. */
        String[] inventoryRow() {
            StringBuilder layout = new StringBuilder(scalarKinds);
            for (int i = 0; i < readOnly.length(); i++) {
                layout.append('B');
            }
            String kinds = String.join(",", scalarKinds.split(""));
            return new String[]{source, String.valueOf(readOnly.length()), String.valueOf(scalarKinds.length()),
                    kinds, layout.toString()};
        }
    }

    private static KernelInventoryChecks inventory() throws IOException {
        KernelInventoryChecks inv = cached;
        if (inv == null) {
            inv = KernelInventoryChecks.load(CONFIG);
            cached = inv;
        }
        return inv;
    }
}
