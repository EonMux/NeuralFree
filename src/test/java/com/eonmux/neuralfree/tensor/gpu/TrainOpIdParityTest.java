package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.kernels.KernelInventoryChecks;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryConfig;
import com.eonmux.neuralfree.acceleration.kernels.KernelManifest;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.eonmux.neuralfree.acceleration.kernels.KernelInventoryChecks.assertNoViolations;
import static com.eonmux.neuralfree.tensor.gpu.TrainKernelSources.read;
import static com.eonmux.neuralfree.tensor.gpu.TrainKernelSources.stripComments;
import static com.eonmux.neuralfree.tensor.gpu.TrainKernelSources.twinOf;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Operation-id parity of the {@code train} kernel library: every object-like {@code #define TR_*} and
 * {@code ML_*} of a manifest source under {@code kernels/train} has the same name and value in its GLSL twin
 * under {@code shaders/train} ({@code .cl} to {@code .comp}, {@code .h} to {@code .glsl}), every {@code int}
 * constant {@code TR_*}/{@code ML_*} of {@link TrainOps} equals the define of the same name, every other
 * define is either a launch-geometry constant mirrored by the Java runtime or a kernel-internal layout constant,
 * and the mode selectors of the backward kernels are defined in the sources of the kernels that read them.
 *
 * <p>Resource-only: reads the source tree relative to the project directory (the Gradle test working
 * directory), because the GLSL sources are not shipped as resources. No GPU is needed.</p>
 */
public class TrainOpIdParityTest {

    private static final KernelLibraryConfig CONFIG = KernelLibraries.TRAIN_CONFIG;

    /** {@code #define NAME rest}; group 1 the name, group 2 the rest of the directive line. */
    private static final Pattern DEFINE = Pattern.compile("^\\s*#\\s*define\\s+([A-Za-z_][A-Za-z0-9_]*)(.*)$");
    private static final Pattern INTEGER = Pattern.compile("-?[0-9]+");

    /**
     * Valued defines that are not operation ids but launch geometry the Java runtime mirrors: the 1-D local
     * size and reduction width, the strided-view rank, and the per-dimension local size and output tile of
     * TR_Gemm_F32.
     */
    private static final Map<String, Integer> JAVA_MIRRORED;

    static {
        Map<String, Integer> m = new LinkedHashMap<>();
        m.put("TR_LOCAL", TrainGpu.LINEAR_LOCAL);
        m.put("TR_MAX_RANK", StridedView.MAX_RANK);
        m.put("TR_GM_LX", (int) TrainOps.localSize(TrainOps.Kernels.GEMM_F32)[0]);
        m.put("TR_GM_TILE", TrainOps.GEMM_TILE);
        JAVA_MIRRORED = Collections.unmodifiableMap(m);
    }

    /**
     * Tiling constants of gemm.cl and gemm.comp without a public Java counterpart. The host never passes them;
     * {@link #gemmTilingConstantsAreConsistent} checks them against each other and against the launch geometry
     * of {@code TR_GM_LX} and {@code TR_GM_TILE}.
     */
    private static final Set<String> KERNEL_INTERNAL = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "TR_GM_TK", "TR_GM_RB", "TR_GM_PAD")));

    /**
     * Mode selectors of the backward kernels, each led by the manifest source whose kernels switch on it:
     * TR_NormAffineGradCols_F32 {@code mode}, TR_ChannelNormBackwardApply_F32 {@code mode},
     * TR_IndexAddRows_F32 {@code oob_mode} and TR_SegmentReduceBackward_F32 {@code op}.
     */
    private static final List<List<String>> BACKWARD_SELECTORS = Collections.unmodifiableList(Arrays.asList(
            Arrays.asList("norm.cl", "TR_N_AFF_LN", "TR_N_AFF_RMS"),
            Arrays.asList("norm.cl", "TR_N_CB_BATCH", "TR_N_CB_GROUP", "TR_N_CB_EVAL"),
            Arrays.asList("norm.cl", "TR_IDX_OOB_ERROR", "TR_IDX_OOB_CLAMP"),
            Arrays.asList("segment.cl", "TR_SEG_SUM", "TR_SEG_MEAN", "TR_SEG_MAX")));

    private static volatile Map<String, Map<String, String>> cached;

    /**
     * Every id of a backward-kernel selector is a {@link TrainOps} constant defined with the same value in the
     * source of the kernel that reads it and in that source's GLSL twin, and the ids of one selector are
     * pairwise distinct, so every mode a dispatcher passes reaches its own branch on all three backends.
     */
    @Test
    public void backwardKernelSelectorsAreDefinedInTheirKernelSource() throws IOException, IllegalAccessException {
        Map<String, Integer> java = trainOpsIds();
        Path clRoot = KernelInventoryChecks.RESOURCES.resolve(CONFIG.sourceRoot());
        Path glslRoot = KernelInventoryChecks.RESOURCES.resolve(CONFIG.spvRoot());
        List<String> violations = new ArrayList<>();
        for (List<String> selector : BACKWARD_SELECTORS) {
            String source = selector.get(0);
            Path clFile = clRoot.resolve(source);
            Path glslFile = glslRoot.resolve(twinOf(source));
            Map<String, String> cl = valued(defines(read(clFile), clFile.toString()));
            Map<String, String> glsl = valued(defines(read(glslFile), glslFile.toString()));
            Map<Integer, String> byValue = new TreeMap<>();
            for (String name : selector.subList(1, selector.size())) {
                Integer value = java.get(name);
                if (value == null) {
                    violations.add("TrainOps declares no public static final int " + name);
                    continue;
                }
                String expected = String.valueOf(value);
                if (!expected.equals(cl.get(name))) {
                    violations.add(source + " defines " + name + " as " + cl.get(name) + ", TrainOps as " + value);
                }
                if (!expected.equals(glsl.get(name))) {
                    violations.add(twinOf(source) + " defines " + name + " as " + glsl.get(name) + ", TrainOps as "
                            + value);
                }
                String clash = byValue.putIfAbsent(value, name);
                if (clash != null) {
                    violations.add(name + " and " + clash + " select the same mode (" + value + ")");
                }
            }
        }
        assertNoViolations("backward-kernel selector ids", violations);
    }

    @Test
    public void everySourceDefinesTheSameIdsAsItsGlslTwin() throws IOException {
        List<String> violations = new ArrayList<>();
        for (Map.Entry<String, Map<String, String>> e : clDefines().entrySet()) {
            String twin = twinOf(e.getKey());
            Path twinFile = KernelInventoryChecks.RESOURCES.resolve(CONFIG.spvRoot()).resolve(twin);
            if (!Files.isRegularFile(twinFile)) {
                violations.add(e.getKey() + ": GLSL twin " + twinFile + " is missing");
                continue;
            }
            Map<String, String> cl = valued(e.getValue());
            Map<String, String> glsl = valued(defines(read(twinFile), twinFile.toString()));
            if (!cl.equals(glsl)) {
                violations.add(e.getKey() + " defines " + cl + " but " + twin + " defines " + glsl);
            }
        }
        assertNoViolations("TR_/ML_ defines of the .cl sources and their GLSL twins", violations);
    }

    @Test
    public void everyDefineNameBelongsToOneSource() throws IOException {
        Map<String, String> owner = new TreeMap<>();
        List<String> violations = new ArrayList<>();
        for (Map.Entry<String, Map<String, String>> e : clDefines().entrySet()) {
            for (String name : e.getValue().keySet()) {
                String previous = owner.putIfAbsent(name, e.getKey());
                if (previous != null) {
                    violations.add(name + " is defined in " + previous + " and " + e.getKey()
                            + " (one OpenCL program shares a single macro namespace)");
                }
            }
        }
        assertNoViolations("TR_/ML_ define names defined in more than one source", violations);
    }

    @Test
    public void trainOpsConstantsEqualTheKernelDefines() throws IOException, IllegalAccessException {
        Map<String, String> all = allValuedDefines();
        Map<String, Integer> java = trainOpsIds();
        assertFalse("TrainOps declares no TR_/ML_ op-id constants", java.isEmpty());
        List<String> violations = new ArrayList<>();
        for (Map.Entry<String, Integer> e : java.entrySet()) {
            String value = all.get(e.getKey());
            if (value == null) {
                violations.add("TrainOps." + e.getKey() + " = " + e.getValue() + " has no #define in kernels/train");
            } else if (!INTEGER.matcher(value).matches() || Integer.parseInt(value) != e.getValue()) {
                violations.add("TrainOps." + e.getKey() + " = " + e.getValue() + " but the kernels define " + value);
            }
        }
        assertNoViolations("TrainOps op ids against the kernel defines", violations);
    }

    @Test
    public void everyKernelDefineIsAnOpIdOrADocumentedConstant() throws IOException, IllegalAccessException {
        Map<String, String> all = allValuedDefines();
        Set<String> java = trainOpsIds().keySet();
        List<String> violations = new ArrayList<>();
        for (Map.Entry<String, String> e : all.entrySet()) {
            String name = e.getKey();
            if (java.contains(name) || KERNEL_INTERNAL.contains(name)) {
                continue;
            }
            if (!JAVA_MIRRORED.containsKey(name)) {
                violations.add(name + " = " + e.getValue() + " is neither a TrainOps op id nor a documented"
                        + " launch-geometry or kernel-internal constant");
            }
        }
        for (String name : JAVA_MIRRORED.keySet()) {
            if (!all.containsKey(name)) {
                violations.add("launch-geometry constant " + name + " is no longer defined in kernels/train");
            }
        }
        for (String name : KERNEL_INTERNAL) {
            if (!all.containsKey(name)) {
                violations.add("kernel-internal constant " + name + " is no longer defined in kernels/train");
            }
        }
        assertNoViolations("TR_/ML_ defines without a Java counterpart", violations);
    }

    @Test
    public void launchGeometryDefinesEqualTheJavaRuntime() throws IOException {
        Map<String, String> all = allValuedDefines();
        List<String> violations = new ArrayList<>();
        for (Map.Entry<String, Integer> e : JAVA_MIRRORED.entrySet()) {
            String value = all.get(e.getKey());
            if (value != null && !value.equals(String.valueOf(e.getValue()))) {
                violations.add(e.getKey() + " is " + value + " in the kernels and " + e.getValue() + " in Java");
            }
        }
        assertNoViolations("launch geometry of the kernels against TrainGpu, StridedView and TrainOps", violations);
        assertEquals("TR_LOCAL is also the row-kernel local size", TrainGpu.LINEAR_LOCAL, TrainGpu.ROW_LOCAL);
        long lx = TrainOps.localSize(TrainOps.Kernels.GEMM_F32)[0];
        assertArrayEquals("TR_Gemm_F32 indexes its items as a square {TR_GM_LX, TR_GM_LX, 1} work-group",
                new long[]{lx, lx, 1}, TrainOps.localSize(TrainOps.Kernels.GEMM_F32));
    }

    /**
     * The gemm tiling constants describe one tile: each of the {@code TR_GM_LX x TR_GM_LX} items owns
     * {@code TR_GM_RB x TR_GM_RB} outputs of a {@code TR_GM_TILE}-square tile, the cooperative load of one stage
     * ({@code TR_GM_RB} elements per item) covers exactly {@code TR_GM_TK x TR_GM_TILE} elements of each operand,
     * and the staged slices are padded by one column.
     */
    @Test
    public void gemmTilingConstantsAreConsistent() throws IOException {
        Map<String, String> all = allValuedDefines();
        int tile = intDefine(all, "TR_GM_TILE");
        int tk = intDefine(all, "TR_GM_TK");
        int lx = intDefine(all, "TR_GM_LX");
        int rb = intDefine(all, "TR_GM_RB");
        assertEquals("TR_GM_RB * TR_GM_LX must span TR_GM_TILE", tile, rb * lx);
        assertEquals("one stage's cooperative load (TR_GM_LX^2 items x TR_GM_RB) must cover TR_GM_TK x TR_GM_TILE",
                tk * tile, lx * lx * rb);
        assertEquals("TR_GM_PAD", "(TR_GM_TILE + 1)", all.get("TR_GM_PAD"));
    }

    @Test
    public void defineParserReadsObjectLikeTrainMacrosOnly() {
        String src = "/* #define TR_COMMENTED 1 */\n"
                + "#ifndef TR_GUARD\n#define TR_GUARD\n"
                + "#define TR_A 0 /* first */\n"
                + "  #  define ML_B   12 // second\n"
                + "#define TR_FN(x) ((x) + 1)\n"
                + "#define OTHER_C 3\n"
                + "#define TR_D \\\n    (TR_A + 1)\n"
                + "// #define TR_E 5\n"
                + "#endif\n";
        Map<String, String> d = defines(src, "inline");
        Map<String, String> expected = new LinkedHashMap<>();
        expected.put("TR_GUARD", "");
        expected.put("TR_A", "0");
        expected.put("ML_B", "12");
        expected.put("TR_D", "(TR_A + 1)");
        assertEquals(expected, d);
        assertEquals("include guards carry no value", 3, valued(d).size());
    }

    /** Defines of every non-shared manifest source, keyed by manifest path, in manifest order. */
    private static Map<String, Map<String, String>> clDefines() throws IOException {
        Map<String, Map<String, String>> d = cached;
        if (d == null) {
            Path root = KernelInventoryChecks.RESOURCES.resolve(CONFIG.sourceRoot());
            Path manifestFile = root.resolve(KernelManifest.MANIFEST_FILE);
            KernelManifest manifest = KernelManifest.parse(CONFIG.sourceRoot(), read(manifestFile),
                    manifestFile.toString());
            d = new LinkedHashMap<>();
            for (KernelManifest.Entry entry : manifest.entries()) {
                if (!entry.shared()) {
                    Path file = root.resolve(entry.path());
                    d.put(entry.path(), defines(read(file), file.toString()));
                }
            }
            d = Collections.unmodifiableMap(d);
            cached = d;
        }
        return d;
    }

    /**
     * Valued defines of all non-shared manifest sources by name; {@link #everyDefineNameBelongsToOneSource}
     * keeps the names unique.
     */
    private static Map<String, String> allValuedDefines() throws IOException {
        Map<String, String> all = new TreeMap<>();
        for (Map<String, String> defs : clDefines().values()) {
            all.putAll(valued(defs));
        }
        return all;
    }

    /** The integer value of a define, failing with its name when it is missing or not an integer literal. */
    private static int intDefine(Map<String, String> all, String name) {
        String value = all.get(name);
        if (value == null || !INTEGER.matcher(value).matches()) {
            throw new AssertionError(name + " must be an integer literal define in kernels/train, found " + value);
        }
        return Integer.parseInt(value);
    }

    /** Public static final int constants of {@link TrainOps} named {@code TR_*} or {@code ML_*}. */
    private static Map<String, Integer> trainOpsIds() throws IllegalAccessException {
        Map<String, Integer> ids = new TreeMap<>();
        for (Field f : TrainOps.class.getDeclaredFields()) {
            int mod = f.getModifiers();
            if (f.getType() == int.class && Modifier.isPublic(mod) && Modifier.isStatic(mod) && Modifier.isFinal(mod)
                    && isTrainName(f.getName())) {
                ids.put(f.getName(), f.getInt(null));
            }
        }
        return ids;
    }

    /** The defines with a non-empty value (drops include guards). */
    private static Map<String, String> valued(Map<String, String> defines) {
        Map<String, String> out = new TreeMap<>();
        for (Map.Entry<String, String> e : defines.entrySet()) {
            if (!e.getValue().isEmpty()) {
                out.put(e.getKey(), e.getValue());
            }
        }
        return out;
    }

    /**
     * Object-like {@code #define TR_*}/{@code ML_*} directives of one source, name to value with whitespace
     * collapsed, in source order. Comments are removed and continued lines spliced first; a name defined twice
     * with different values is an error.
     */
    static Map<String, String> defines(String source, String origin) {
        String text = stripComments(source.replace("\r\n", "\n").replace("\\\n", " "));
        Map<String, String> out = new LinkedHashMap<>();
        for (String line : text.split("\n", -1)) {
            Matcher m = DEFINE.matcher(line);
            if (!m.matches() || !isTrainName(m.group(1)) || m.group(2).startsWith("(")) {
                continue;
            }
            String value = m.group(2).trim().replaceAll("\\s+", " ");
            String previous = out.putIfAbsent(m.group(1), value);
            if (previous != null && !previous.equals(value)) {
                throw new IllegalStateException(origin + ": " + m.group(1) + " is defined as both '" + previous
                        + "' and '" + value + "'");
            }
        }
        return out;
    }

    private static boolean isTrainName(String name) {
        return name.startsWith("TR_") || name.startsWith("ML_");
    }
}
