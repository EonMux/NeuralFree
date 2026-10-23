package com.eonmux.neuralfree.acceleration.kernels;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Static inventory of one kernel library for the per-library inventory tests: every kernel signature of the
 * manifest files (family macros expanded), the variant rows of {@code <spvRoot>/variants/*.tsv} and the
 * checked-in {@code <sourceRoot>/inventory.txt}. Entries are read from
 * {@link KernelManifest.Entry#resourcePath(String)}, so {@code [shared]} headers come from
 * {@value KernelManifest#SHARED_ROOT}. Reads the source tree relative to the project directory (the Gradle
 * test working directory), because the GLSL variant lists are not shipped as resources.
 */
public final class KernelInventoryChecks {

    /** Main resources of the source tree, relative to the project directory. */
    public static final Path RESOURCES = Paths.get("src", "main", "resources");
    /** Largest scalar count a kernel may take (128-byte Vulkan push block). */
    public static final int MAX_SCALARS = 32;
    /** Largest buffer count a kernel may take. */
    public static final int MAX_BUFFERS = 12;

    /** Scalar parameter types wider than 4 bytes, which library kernels never take. */
    private static final Set<String> WIDE_SCALAR_TYPES = new HashSet<>(Arrays.asList(
            "long", "ulong", "double", "size_t", "ptrdiff_t", "intptr_t", "uintptr_t", "nf_i64", "nf_u64"));
    /** Address-space keywords of work-group local memory. */
    private static final Set<String> LOCAL_QUALIFIERS = new HashSet<>(Arrays.asList(
            "NF_LOCAL", "NF_LOCAL_PTR", "__local", "local"));

    private final Path kernelRoot;
    private final Path variantsDir;
    private final KernelManifest manifest;
    private final List<KernelSig> kernels;
    private final Map<String, List<String>> variantRows;
    private final List<String> variantFormatErrors;

    private KernelInventoryChecks(Path kernelRoot, Path variantsDir, KernelManifest manifest, List<KernelSig> kernels,
                                  Map<String, List<String>> variantRows, List<String> variantFormatErrors) {
        this.kernelRoot = kernelRoot;
        this.variantsDir = variantsDir;
        this.manifest = manifest;
        this.kernels = Collections.unmodifiableList(kernels);
        this.variantRows = Collections.unmodifiableMap(variantRows);
        this.variantFormatErrors = Collections.unmodifiableList(variantFormatErrors);
    }

    /** Reads the manifest, every manifest source and every variant row of a library from the source tree. */
    public static KernelInventoryChecks load(KernelLibraryConfig config) throws IOException {
        Objects.requireNonNull(config, "config must not be null");
        Path kernelRoot = RESOURCES.resolve(config.sourceRoot());
        Path variantsDir = RESOURCES.resolve(config.spvRoot()).resolve("variants");
        Path manifestFile = kernelRoot.resolve(KernelManifest.MANIFEST_FILE);
        KernelManifest manifest = KernelManifest.parse(config.sourceRoot(), read(manifestFile), manifestFile.toString());
        Map<String, String> files = new LinkedHashMap<>();
        for (KernelManifest.Entry e : manifest.entries()) {
            files.put(e.path(), read(RESOURCES.resolve(e.resourcePath(config.sourceRoot()))));
        }
        Map<String, List<String>> rows = new TreeMap<>();
        List<String> errors = new ArrayList<>();
        if (Files.isDirectory(variantsDir)) {
            List<Path> tsvs = new ArrayList<>();
            try (DirectoryStream<Path> ds = Files.newDirectoryStream(variantsDir, "*.tsv")) {
                ds.forEach(tsvs::add);
            }
            tsvs.sort(null);
            for (Path tsv : tsvs) {
                String[] lines = read(tsv).split("\r?\n", -1);
                for (int i = 0; i < lines.length; i++) {
                    String line = lines[i];
                    if (line.trim().isEmpty() || line.trim().startsWith("#")) {
                        continue;
                    }
                    String[] cols = line.split("\t", -1);
                    String where = tsv.getFileName() + ":" + (i + 1);
                    if (cols.length < 2 || cols.length > 4 || cols[0].trim().isEmpty() || cols[1].trim().isEmpty()) {
                        errors.add(where + ": expected kernel_name<TAB>source.comp[<TAB>defines[<TAB>flags]]");
                        continue;
                    }
                    rows.computeIfAbsent(cols[0].trim(), n -> new ArrayList<>()).add(where);
                }
            }
        }
        return new KernelInventoryChecks(kernelRoot, variantsDir, manifest, new Scanner(files).scan(), rows, errors);
    }

    /** The parsed manifest of the library. */
    public KernelManifest manifest() {
        return manifest;
    }

    /** Every kernel declaration of the manifest files, in manifest and source order. */
    public List<KernelSig> kernels() {
        return kernels;
    }

    /** Kernels above {@link #MAX_SCALARS} scalars or {@link #MAX_BUFFERS} buffers. */
    public List<String> argumentLimitViolations() {
        List<String> violations = new ArrayList<>();
        for (KernelSig k : kernels) {
            if (k.scalars > MAX_SCALARS) {
                violations.add(k + ": " + k.scalars + " scalars > " + MAX_SCALARS);
            }
            if (k.buffers > MAX_BUFFERS) {
                violations.add(k + ": " + k.buffers + " buffers > " + MAX_BUFFERS);
            }
        }
        return violations;
    }

    /** Arguments that break the conventions: 8-byte scalars, by-value arrays and local-memory pointers. */
    public List<String> conventionViolations() {
        List<String> violations = new ArrayList<>();
        for (KernelSig k : kernels) {
            violations.addAll(k.conventionViolations);
        }
        return violations;
    }

    /** Kernel names declared more than once outside mutually exclusive conditional branches. */
    public List<String> duplicateKernelNames() {
        return duplicates(kernels);
    }

    /** Portable kernels (of files not in {@code openClOnly}, the OpenCL-only paths) without exactly one variant row. */
    public List<String> variantRowViolations(Set<String> openClOnly) {
        List<String> violations = new ArrayList<>();
        for (KernelSig k : kernels) {
            if (openClOnly.contains(k.file)) {
                continue;
            }
            List<String> rows = variantRows.get(k.name);
            if (rows == null) {
                violations.add(k + " has no row in " + variantsDir + "/*.tsv");
            } else if (rows.size() != 1) {
                violations.add(k + " has " + rows.size() + " variant rows: " + rows);
            }
        }
        return violations;
    }

    /** Variant rows that name no kernel of a file outside {@code openClOnly}, then the malformed rows. */
    public List<String> orphanVariantRowViolations(Set<String> openClOnly) {
        Set<String> portable = new HashSet<>();
        for (KernelSig k : kernels) {
            if (!openClOnly.contains(k.file)) {
                portable.add(k.name);
            }
        }
        List<String> violations = new ArrayList<>();
        for (Map.Entry<String, List<String>> e : variantRows.entrySet()) {
            if (!portable.contains(e.getKey())) {
                violations.add(e.getKey() + " (" + e.getValue() + ") is not a kernel of a portable manifest file");
            }
        }
        violations.addAll(variantFormatErrors);
        return violations;
    }

    /**
     * Asserts that the checked-in inventory lists exactly the kernels of the files outside
     * {@code openClOnly}, each with the manifest file that defines it.
     */
    public void assertInventoryCurrent(Set<String> openClOnly) throws IOException {
        Map<String, String> expected = new TreeMap<>();
        for (KernelSig k : kernels) {
            if (!openClOnly.contains(k.file)) {
                expected.put(k.name, k.file);
            }
        }
        Path file = kernelRoot.resolve(KernelManifest.INVENTORY_FILE);
        String text = read(file);
        Set<String> names = KernelManifest.parseInventory(text, file.toString());
        Map<String, String> listed = new TreeMap<>();
        for (String line : text.split("\r?\n", -1)) {
            String[] cols = line.split("\t", -1);
            if (cols.length > 1 && names.contains(cols[0])) {
                listed.put(cols[0], cols[1]);
            }
        }
        assertEquals(file + " is stale; regenerate it with the updateKernelInventory Gradle task", expected, listed);
    }

    /** Fails with every violation listed when there is any. */
    public static void assertNoViolations(String what, List<String> violations) {
        if (!violations.isEmpty()) {
            StringBuilder sb = new StringBuilder(what).append(": ").append(violations.size()).append(" violation(s)");
            for (String v : violations) {
                sb.append("\n  ").append(v);
            }
            fail(sb.toString());
        }
    }

    /** Kernel declarations of sources given by manifest path in manifest order (earlier macros apply later). */
    public static List<KernelSig> scan(Map<String, String> files) {
        return new Scanner(new LinkedHashMap<>(files)).scan();
    }

    /** Kernel names declared more than once outside mutually exclusive conditional branches. */
    public static List<String> duplicates(List<KernelSig> kernels) {
        Map<String, List<KernelSig>> byName = new TreeMap<>();
        for (KernelSig k : kernels) {
            byName.computeIfAbsent(k.name, n -> new ArrayList<>()).add(k);
        }
        List<String> out = new ArrayList<>();
        for (Map.Entry<String, List<KernelSig>> e : byName.entrySet()) {
            List<KernelSig> decls = e.getValue();
            for (int i = 0; i < decls.size(); i++) {
                for (int j = i + 1; j < decls.size(); j++) {
                    if (!exclusive(decls.get(i).conditions, decls.get(j).conditions)) {
                        out.add(e.getKey() + " is declared in " + decls.get(i).file + " and " + decls.get(j).file);
                    }
                }
            }
        }
        return out;
    }

    /** Two declarations are exclusive when they sit in different branches of one conditional group. */
    private static boolean exclusive(List<int[]> a, List<int[]> b) {
        for (int[] x : a) {
            for (int[] y : b) {
                if (x[0] == y[0] && x[1] != y[1]) {
                    return true;
                }
            }
        }
        return false;
    }

    private static String read(Path p) throws IOException {
        return new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
    }

    /** One kernel declaration: name, declaring manifest file, parameter counts and convention violations. */
    public static final class KernelSig {
        public final String name;
        public final String file;
        public final int scalars;
        public final int buffers;
        /**
         * One {@code 0}/{@code 1} flag per buffer argument in declaration order, {@code 1} when the argument points
         * to {@code const} data: the encoding of the {@code .ro} key in a SPIR-V {@code index.properties}.
         */
        public final String readOnlyBuffers;
        public final List<String> conventionViolations;
        private final List<int[]> conditions;

        KernelSig(String name, String file, int scalars, String readOnlyBuffers, List<int[]> conditions,
                  List<String> conventionViolations) {
            this.name = name;
            this.file = file;
            this.scalars = scalars;
            this.buffers = readOnlyBuffers.length();
            this.readOnlyBuffers = readOnlyBuffers;
            this.conditions = conditions;
            this.conventionViolations = Collections.unmodifiableList(conventionViolations);
        }

        @Override
        public String toString() {
            return name + " (" + file + ")";
        }
    }

    /**
     * Finds kernel declarations in dual-dialect sources: strips comments, splices continued lines, tracks
     * conditional groups, collects the function-like macros that generate kernels (directly or through other
     * such macros), expands their invocations with parameter substitution and {@code ##} pasting, and parses
     * each {@code NF_KERNEL void name(...)} parameter list.
     */
    static final class Scanner {
        private static final int MAX_DEPTH = 32;

        private final Map<String, String> files;
        private final Map<String, Macro> macros = new LinkedHashMap<>();
        private final Set<String> kernelMacros = new HashSet<>();
        private int nextGroup;

        Scanner(Map<String, String> files) {
            this.files = files;
        }

        List<KernelSig> scan() {
            Map<String, List<Segment>> segments = new LinkedHashMap<>();
            for (Map.Entry<String, String> f : files.entrySet()) {
                segments.put(f.getKey(), split(stripComments(f.getValue().replace("\r\n", "\n").replace("\\\n", " "))));
            }
            findKernelMacros();
            List<KernelSig> out = new ArrayList<>();
            for (Map.Entry<String, List<Segment>> f : segments.entrySet()) {
                for (Segment s : f.getValue()) {
                    findKernels(expand(s.text, new HashSet<String>(), 0), f.getKey(), s.conditions, out);
                }
            }
            return out;
        }

        private List<Segment> split(String text) {
            List<Segment> out = new ArrayList<>();
            List<int[]> stack = new ArrayList<>();
            StringBuilder code = new StringBuilder();
            for (String line : text.split("\n", -1)) {
                String t = line.trim();
                if (!t.startsWith("#")) {
                    code.append(line).append('\n');
                    continue;
                }
                String d = t.substring(1).trim();
                String word = d.split("[\\s(]", 2)[0];
                if (word.equals("if") || word.equals("ifdef") || word.equals("ifndef")) {
                    flush(code, stack, out);
                    stack.add(new int[]{nextGroup++, 0});
                } else if (word.equals("elif") || word.equals("else")) {
                    flush(code, stack, out);
                    if (!stack.isEmpty()) {
                        int[] top = stack.get(stack.size() - 1);
                        stack.set(stack.size() - 1, new int[]{top[0], top[1] + 1});
                    }
                } else if (word.equals("endif")) {
                    flush(code, stack, out);
                    if (!stack.isEmpty()) {
                        stack.remove(stack.size() - 1);
                    }
                } else if (word.equals("define")) {
                    define(d.substring(6).trim());
                }
            }
            flush(code, stack, out);
            return out;
        }

        private static void flush(StringBuilder code, List<int[]> stack, List<Segment> out) {
            if (code.toString().trim().length() > 0) {
                out.add(new Segment(code.toString(), new ArrayList<>(stack)));
            }
            code.setLength(0);
        }

        private void define(String d) {
            int i = 0;
            while (i < d.length() && isIdentChar(d.charAt(i))) {
                i++;
            }
            String name = d.substring(0, i);
            if (name.isEmpty() || macros.containsKey(name)) {
                return;
            }
            if (i < d.length() && d.charAt(i) == '(') {
                int close = d.indexOf(')', i);
                if (close < 0) {
                    return;
                }
                List<String> params = new ArrayList<>();
                for (String p : d.substring(i + 1, close).split(",")) {
                    if (!p.trim().isEmpty()) {
                        params.add(p.trim());
                    }
                }
                macros.put(name, new Macro(params, d.substring(close + 1).trim(), true));
            } else {
                macros.put(name, new Macro(new ArrayList<String>(), d.substring(i).trim(), false));
            }
        }

        private void findKernelMacros() {
            boolean changed = true;
            while (changed) {
                changed = false;
                for (Map.Entry<String, Macro> e : macros.entrySet()) {
                    if (kernelMacros.contains(e.getKey()) || e.getKey().equals("NF_KERNEL")) {
                        continue;
                    }
                    for (String id : identifiers(e.getValue().body)) {
                        if (id.equals("NF_KERNEL") || id.equals("__kernel") || kernelMacros.contains(id)) {
                            kernelMacros.add(e.getKey());
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }

        private String expand(String text, Set<String> active, int depth) {
            if (depth > MAX_DEPTH) {
                throw new IllegalStateException("macro expansion deeper than " + MAX_DEPTH + ": " + active);
            }
            StringBuilder out = new StringBuilder();
            int i = 0;
            while (i < text.length()) {
                char c = text.charAt(i);
                if (!isIdentStart(c) || (i > 0 && isIdentChar(text.charAt(i - 1)))) {
                    out.append(c);
                    i++;
                    continue;
                }
                int j = i;
                while (j < text.length() && isIdentChar(text.charAt(j))) {
                    j++;
                }
                String id = text.substring(i, j);
                Macro m = kernelMacros.contains(id) && !active.contains(id) ? macros.get(id) : null;
                if (m == null) {
                    out.append(id);
                    i = j;
                    continue;
                }
                Set<String> inner = new HashSet<>(active);
                inner.add(id);
                if (!m.functionLike) {
                    out.append(expand(m.body, inner, depth + 1));
                    i = j;
                    continue;
                }
                int k = j;
                while (k < text.length() && Character.isWhitespace(text.charAt(k))) {
                    k++;
                }
                if (k >= text.length() || text.charAt(k) != '(') {
                    out.append(id);
                    i = j;
                    continue;
                }
                int close = matching(text, k);
                List<String> args = splitTopLevel(text.substring(k + 1, close));
                out.append(' ').append(expand(substitute(m, args, id), inner, depth + 1)).append(' ');
                i = close + 1;
            }
            return out.toString();
        }

        private static String substitute(Macro m, List<String> args, String name) {
            boolean variadic = !m.params.isEmpty() && m.params.get(m.params.size() - 1).equals("...");
            int fixed = variadic ? m.params.size() - 1 : m.params.size();
            if (args.size() == 1 && args.get(0).trim().isEmpty() && fixed == 0) {
                args = new ArrayList<>();
            }
            if (args.size() < fixed || (!variadic && args.size() != fixed)) {
                throw new IllegalStateException("macro " + name + " takes " + m.params.size() + " args, got "
                        + args.size() + ": " + args);
            }
            Map<String, String> values = new LinkedHashMap<>();
            for (int p = 0; p < fixed; p++) {
                values.put(m.params.get(p), args.get(p).trim());
            }
            if (variadic) {
                values.put("__VA_ARGS__", String.join(",", args.subList(fixed, args.size())).trim());
            }
            StringBuilder out = new StringBuilder();
            String body = m.body;
            int i = 0;
            while (i < body.length()) {
                char c = body.charAt(i);
                if (isIdentStart(c) && (i == 0 || !isIdentChar(body.charAt(i - 1)))) {
                    int j = i;
                    while (j < body.length() && isIdentChar(body.charAt(j))) {
                        j++;
                    }
                    String id = body.substring(i, j);
                    String v = values.get(id);
                    out.append(v != null ? v : id);
                    i = j;
                } else {
                    out.append(c);
                    i++;
                }
            }
            return out.toString().replaceAll("\\s*##\\s*", "");
        }

        private static void findKernels(String text, String file, List<int[]> conditions, List<KernelSig> out) {
            int from = 0;
            while (true) {
                int at = nextKeyword(text, from);
                if (at < 0) {
                    return;
                }
                int v = indexOfWord(text, "void", at);
                int open = v < 0 ? -1 : text.indexOf('(', v);
                if (open < 0) {
                    throw new IllegalStateException(file + ": kernel declaration without a parameter list near '"
                            + snippet(text, at) + "'");
                }
                String name = text.substring(v + 4, open).trim();
                if (name.isEmpty() || !isIdentStart(name.charAt(0)) || identifiers(name).size() != 1
                        || !identifiers(name).get(0).equals(name)) {
                    throw new IllegalStateException(file + ": cannot read the kernel name in '"
                            + snippet(text, at) + "'");
                }
                int close = matching(text, open);
                out.add(signature(name, file, text.substring(open + 1, close), conditions));
                from = close + 1;
            }
        }

        private static KernelSig signature(String name, String file, String params, List<int[]> conditions) {
            int scalars = 0;
            StringBuilder readOnly = new StringBuilder();
            List<String> violations = new ArrayList<>();
            for (String p : splitTopLevel(params)) {
                String param = p.trim().replaceAll("\\s+", " ");
                if (param.isEmpty() || param.equals("void")) {
                    continue;
                }
                Set<String> words = new HashSet<>(identifiers(param));
                if (param.indexOf('[') >= 0) {
                    violations.add(name + " (" + file + "): by-value array argument '" + param + "'");
                }
                int star = param.indexOf('*');
                if (star >= 0) {
                    readOnly.append(identifiers(param.substring(0, star)).contains("const") ? '1' : '0');
                    for (String q : LOCAL_QUALIFIERS) {
                        if (words.contains(q)) {
                            violations.add(name + " (" + file + "): local-memory argument '" + param + "'");
                            break;
                        }
                    }
                } else {
                    scalars++;
                    for (String w : WIDE_SCALAR_TYPES) {
                        if (words.contains(w)) {
                            violations.add(name + " (" + file + "): 8-byte scalar argument '" + param + "'");
                            break;
                        }
                    }
                }
            }
            return new KernelSig(name, file, scalars, readOnly.toString(), conditions, violations);
        }

        private static int nextKeyword(String text, int from) {
            int a = indexOfWord(text, "NF_KERNEL", from);
            int b = indexOfWord(text, "__kernel", from);
            if (a < 0) {
                return b;
            }
            return b < 0 ? a : Math.min(a, b);
        }

        private static int indexOfWord(String text, String word, int from) {
            int at = text.indexOf(word, from);
            while (at >= 0) {
                boolean startOk = at == 0 || !isIdentChar(text.charAt(at - 1));
                int end = at + word.length();
                boolean endOk = end >= text.length() || !isIdentChar(text.charAt(end));
                if (startOk && endOk) {
                    return at;
                }
                at = text.indexOf(word, at + 1);
            }
            return -1;
        }

        private static int matching(String text, int open) {
            int depth = 0;
            for (int i = open; i < text.length(); i++) {
                char c = text.charAt(i);
                if (c == '(') {
                    depth++;
                } else if (c == ')') {
                    depth--;
                    if (depth == 0) {
                        return i;
                    }
                }
            }
            throw new IllegalStateException("unbalanced parentheses near '" + snippet(text, open) + "'");
        }

        private static List<String> splitTopLevel(String s) {
            List<String> out = new ArrayList<>();
            int depth = 0;
            int start = 0;
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c == '(' || c == '[' || c == '{') {
                    depth++;
                } else if (c == ')' || c == ']' || c == '}') {
                    depth--;
                } else if (c == ',' && depth == 0) {
                    out.add(s.substring(start, i));
                    start = i + 1;
                }
            }
            out.add(s.substring(start));
            return out;
        }

        private static List<String> identifiers(String s) {
            List<String> out = new ArrayList<>();
            int i = 0;
            while (i < s.length()) {
                if (isIdentStart(s.charAt(i))) {
                    int j = i;
                    while (j < s.length() && isIdentChar(s.charAt(j))) {
                        j++;
                    }
                    out.add(s.substring(i, j));
                    i = j;
                } else if (Character.isDigit(s.charAt(i))) {
                    while (i < s.length() && isIdentChar(s.charAt(i))) {
                        i++;
                    }
                } else {
                    i++;
                }
            }
            return out;
        }

        /** Replaces comments with spaces, keeping newlines and string and character literals. */
        private static String stripComments(String s) {
            StringBuilder out = new StringBuilder(s.length());
            int i = 0;
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == '/' && i + 1 < s.length() && s.charAt(i + 1) == '*') {
                    int end = s.indexOf("*/", i + 2);
                    int stop = end < 0 ? s.length() : end + 2;
                    for (int k = i; k < stop; k++) {
                        out.append(s.charAt(k) == '\n' ? '\n' : ' ');
                    }
                    i = stop;
                } else if (c == '/' && i + 1 < s.length() && s.charAt(i + 1) == '/') {
                    while (i < s.length() && s.charAt(i) != '\n') {
                        i++;
                    }
                } else if (c == '"' || c == '\'') {
                    int k = i + 1;
                    while (k < s.length() && s.charAt(k) != c && s.charAt(k) != '\n') {
                        k += s.charAt(k) == '\\' ? 2 : 1;
                    }
                    int stop = Math.min(s.length(), k + 1);
                    out.append(s, i, stop);
                    i = stop;
                } else {
                    out.append(c);
                    i++;
                }
            }
            return out.toString();
        }

        private static String snippet(String text, int at) {
            return text.substring(Math.max(0, at), Math.min(text.length(), at + 80)).replace('\n', ' ');
        }

        private static boolean isIdentStart(char c) {
            return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        }

        private static boolean isIdentChar(char c) {
            return isIdentStart(c) || (c >= '0' && c <= '9');
        }

        private static final class Macro {
            final List<String> params;
            final String body;
            final boolean functionLike;

            Macro(List<String> params, String body, boolean functionLike) {
                this.params = params;
                this.body = body;
                this.functionLike = functionLike;
            }
        }

        private static final class Segment {
            final String text;
            final List<int[]> conditions;

            Segment(String text, List<int[]> conditions) {
                this.text = text;
                this.conditions = conditions;
            }
        }
    }
}
