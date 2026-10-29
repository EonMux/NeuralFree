package com.eonmux.neuralfree.gguf.hf.names;

import com.eonmux.neuralfree.gguf.arch.Arch;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The tensor-naming tables generated from llama.cpp's converter into {@code R/hf} by the {@code updateHfTables}
 * Gradle task, parsed and validated once on first use: {@code tensor_names.tsv} (gguf-py constants.py
 * {@code TENSOR_NAMES}), {@code tensor_patterns.tsv} (tensor_mapping.py {@code mappings_cfg} and
 * {@code block_mappings_cfg}), {@code arch_tensors.tsv} (constants.py {@code MODEL_TENSORS}),
 * {@code arch_block_overrides.tsv} (tensor_mapping.py {@code arch_block_mappings_cfg}) and {@code filters.tsv}
 * (the converter's tensor filters and forced-F32 list). {@link #read} is the one reader of every generated table;
 * {@link HfArchNames}, {@code GgufKeyTable} and {@code HfHparamAliases} use it for theirs.
 *
 * <p>A file that is missing, has a malformed header, a row with the wrong column count, an empty field, an unknown
 * value or a row count that differs from its header throws {@link IllegalStateException}
 * {@code "hf/<file>:<line>: <reason>"}: the tables are checked-in build inputs, so a broken one is a build defect.</p>
 *
 * <p>The filter kinds of {@code filters.tsv} and the converter step each belongs to:</p>
 * <ul>
 *   <li>{@code rename_suffix}, {@code rename_contains}: {@code ModelBase.filter_tensors} (base.py:585-594), applied
 *       to every checkpoint;</li>
 *   <li>{@code strip_prefix}, {@code drop_prefix}, {@code drop_contains}: {@code TextModel.filter_tensors}
 *       (base.py:1556-1573); the stripped prefix only serves the drop tests, the name keeps it;</li>
 *   <li>{@code skip_contains}: {@code MmprojModel.filter_tensors} (base.py:2905-2912);</li>
 *   <li>{@code skip_suffix}: the names {@code prepare_tensors} never converts (base.py:1222-1224);</li>
 *   <li>{@code forced_f32}: the {@code MODEL_TENSOR}s {@code prepare_tensors} always stores as F32
 *       (base.py:1253-1283), see {@link #forcedF32()}.</li>
 * </ul>
 */
public final class HfNameTables {

    /** Classpath directory of the generated tables. */
    public static final String RESOURCE_DIR = "hf/";

    private static final Pattern HEADER = Pattern.compile(
            "# (\\S+) - Generated from llama\\.cpp (\\S+) by the updateHfTables Gradle task; do not edit\\."
                    + " Rows: (\\d+)");
    private static final Pattern PLACEHOLDER = Pattern.compile("\\{([a-z_]+)}");
    private static final String BID = "bid";

    /** The kinds of {@code filters.tsv} rows. */
    public enum FilterKind {
        RENAME_SUFFIX("rename_suffix"),
        RENAME_CONTAINS("rename_contains"),
        STRIP_PREFIX("strip_prefix"),
        DROP_PREFIX("drop_prefix"),
        DROP_CONTAINS("drop_contains"),
        SKIP_CONTAINS("skip_contains"),
        SKIP_SUFFIX("skip_suffix"),
        FORCED_F32("forced_f32");

        private final String tableName;

        FilterKind(String tableName) {
            this.tableName = tableName;
        }

        /** The kind as written in {@code filters.tsv}. */
        public String tableName() {
            return tableName;
        }

        static FilterKind fromTableName(String name) {
            for (FilterKind k : values()) {
                if (k.tableName.equals(name)) {
                    return k;
                }
            }
            return null;
        }
    }

    /** One {@code filters.tsv} row. Immutable. */
    public static final class FilterRule {
        private final FilterKind kind;
        private final String value;
        private final String replacement;
        private final String source;

        FilterRule(FilterKind kind, String value, String replacement, String source) {
            this.kind = kind;
            this.value = value;
            this.replacement = replacement;
            this.source = source;
        }

        public FilterKind kind() { return kind; }

        /** The matched text (a name part, or a {@code MODEL_TENSOR} name for {@link FilterKind#FORCED_F32}). */
        public String value() { return value; }

        /** The replacement of a rename; the empty string when the table writes {@code -}. */
        public String replacement() { return replacement; }

        /** The converter line the rule comes from ({@code base.py:<line>}). */
        public String source() { return source; }

        @Override
        public String toString() {
            return kind.tableName + " '" + value + "'" + (replacement.isEmpty() ? "" : " -> '" + replacement + "'")
                    + " (" + source + ")";
        }
    }

    /** One data row of a generated table: its fields and its 1-based line in the file. Immutable. */
    public static final class TsvRow {
        private final String file;
        private final int line;
        private final String[] fields;

        TsvRow(String file, int line, String[] fields) {
            this.file = file;
            this.line = line;
            this.fields = fields;
        }

        /** Field {@code column} (0-based). */
        public String get(int column) {
            return fields[column];
        }

        /** The 1-based line number in the file. */
        public int line() {
            return line;
        }

        /** An {@link IllegalStateException} {@code "hf/<file>:<line>: <reason>"}. */
        public IllegalStateException defect(String reason) {
            return new IllegalStateException(RESOURCE_DIR + file + ":" + line + ": " + reason);
        }

        @Override
        public String toString() {
            return RESOURCE_DIR + file + ":" + line + " " + Arrays.toString(fields);
        }
    }

    /** One generated table: its llama.cpp commit and its data rows in file order. Immutable. */
    public static final class Tsv {
        private final String file;
        private final String commit;
        private final List<TsvRow> rows;

        Tsv(String file, String commit, List<TsvRow> rows) {
            this.file = file;
            this.commit = commit;
            this.rows = rows;
        }

        public String file() { return file; }

        /** The llama.cpp commit the table was generated from. */
        public String commit() { return commit; }

        /** The data rows (unmodifiable). */
        public List<TsvRow> rows() { return rows; }
    }

    private final Map<String, String> tensorNames;
    private final Map<String, List<String>> globalPatterns;
    private final Map<String, List<String>> blockPatterns;
    private final Map<String, Set<String>> archTensors;
    private final Map<String, Map<String, List<String>>> blockOverrides;
    private final Map<FilterKind, List<FilterRule>> filters;
    private final Predicate<String> forcedF32;

    private HfNameTables() {
        this.tensorNames = loadTensorNames();
        Map<String, List<String>> global = new LinkedHashMap<>();
        Map<String, List<String>> block = new LinkedHashMap<>();
        loadPatterns(tensorNames, global, block);
        this.globalPatterns = freezeLists(global);
        this.blockPatterns = freezeLists(block);
        this.archTensors = loadArchTensors();
        this.blockOverrides = loadBlockOverrides(tensorNames);
        this.filters = loadFilters(tensorNames);
        this.forcedF32 = buildForcedF32(tensorNames, filters.get(FilterKind.FORCED_F32));
    }

    private static final class Holder {
        static final HfNameTables TABLES = new HfNameTables();
    }

    private static HfNameTables tables() {
        return Holder.TABLES;
    }

    // ------------------------------------------------------------------ public views

    /** {@code MODEL_TENSOR} name to GGUF name template ({@code blk.{bid}.attn_q}), in table order (unmodifiable). */
    public static Map<String, String> tensorNames() {
        return tables().tensorNames;
    }

    /**
     * {@code mappings_cfg}: model-level tensor to its Hugging Face names, in dict order (unmodifiable). A tensor with
     * an empty list ({@code -} in the table) is mapped only under its own GGUF name.
     */
    public static Map<String, List<String>> globalPatterns() {
        return tables().globalPatterns;
    }

    /**
     * {@code block_mappings_cfg}: block tensor to its {@code {bid}} templates, in dict order (unmodifiable). A tensor
     * with an empty list ({@code -} in the table) is mapped only under its own GGUF name.
     */
    public static Map<String, List<String>> blockPatterns() {
        return tables().blockPatterns;
    }

    /**
     * {@code MODEL_TENSORS[arch]}: the tensors the converter maps for the GGUF architecture {@code ggufArch}, in
     * table order (unmodifiable); empty for an architecture the table does not list.
     */
    public static Set<String> archTensors(String ggufArch) {
        Set<String> s = tables().archTensors.get(ggufArch);
        return s != null ? s : Collections.<String>emptySet();
    }

    /**
     * {@code arch_block_mappings_cfg[arch]}: the block patterns that replace or extend {@link #blockPatterns()} for
     * {@code ggufArch}, in override order (unmodifiable); empty when the architecture has none.
     */
    public static Map<String, List<String>> blockOverrides(String ggufArch) {
        Map<String, List<String>> m = tables().blockOverrides.get(ggufArch);
        return m != null ? m : Collections.<String, List<String>>emptyMap();
    }

    /** The {@code filters.tsv} rules of one kind, in table order (unmodifiable). */
    public static List<FilterRule> filters(FilterKind kind) {
        return tables().filters.get(Objects.requireNonNull(kind, "kind"));
    }

    /**
     * True for the GGUF names the converter always stores as F32 ({@code prepare_tensors}, base.py:1253-1283): the
     * {@code forced_f32} tensors' names expanded through {@code tensor_names.tsv}, with any block number, followed by
     * {@code .weight} ({@code match_model_tensor_name}, base.py:609-620). Thread-safe.
     */
    public static Predicate<String> forcedF32() {
        return tables().forcedF32;
    }

    /**
     * The GGUF name of {@code tensor} for block {@code bid} ({@code TENSOR_NAMES[tensor].format(bid=bid)}), without a
     * suffix.
     *
     * @param bid the block, or -1 for a model-level tensor
     * @throws IllegalArgumentException when the tensor is unknown, a block tensor gets no block or a model-level
     *                                  tensor gets one
     */
    public static String formatTensorName(String tensor, int bid) {
        String template = tensorNames().get(tensor);
        if (template == null) {
            throw new IllegalArgumentException("unknown converter tensor '" + tensor + "'");
        }
        boolean hasBid = template.contains("{" + BID + "}");
        if (hasBid != (bid >= 0)) {
            throw new IllegalArgumentException("tensor '" + tensor + "' (" + template + ") "
                    + (hasBid ? "needs a block number" : "is model-level and takes no block number, got " + bid));
        }
        String name = hasBid ? template.replace("{" + BID + "}", Integer.toString(bid)) : template;
        if (name.indexOf('{') >= 0) {
            throw new IllegalArgumentException("tensor '" + tensor + "' (" + template
                    + ") has a placeholder besides {bid}");
        }
        return name;
    }

    // ------------------------------------------------------------------ reader

    /**
     * Reads the generated table {@code file} of {@link #RESOURCE_DIR} and checks its header line, its column header
     * line (exactly {@code columns}), the column count and non-empty fields of every row, and the row count recorded
     * in the header.
     *
     * @throws IllegalStateException {@code "hf/<file>:<line>: <reason>"} for any defect, or when the file is missing
     */
    public static Tsv read(String file, String... columns) {
        Objects.requireNonNull(file, "file");
        if (columns.length == 0) {
            throw new IllegalArgumentException("a table has at least one column");
        }
        InputStream in = HfNameTables.class.getClassLoader().getResourceAsStream(RESOURCE_DIR + file);
        if (in == null) {
            throw new IllegalStateException(RESOURCE_DIR + file + ": the resource is missing from the classpath");
        }
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            return parse(file, r, columns);
        } catch (IOException e) {
            throw new UncheckedIOException(RESOURCE_DIR + file + ": cannot read the resource", e);
        }
    }

    private static Tsv parse(String file, BufferedReader r, String[] columns) throws IOException {
        String header = r.readLine();
        Matcher m = header == null ? null : HEADER.matcher(header);
        if (m == null || !m.matches() || !m.group(1).equals(file)) {
            throw new IllegalStateException(RESOURCE_DIR + file + ":1: malformed header line: " + header);
        }
        String columnLine = r.readLine();
        String expected = "# " + String.join("\t", columns);
        if (!expected.equals(columnLine)) {
            throw new IllegalStateException(RESOURCE_DIR + file + ":2: column header '" + columnLine
                    + "' is not '" + expected + "'");
        }
        long declared = Long.parseLong(m.group(3));
        List<TsvRow> rows = new ArrayList<>();
        int lineNo = 2;
        for (String line = r.readLine(); line != null; line = r.readLine()) {
            lineNo++;
            String[] fields = line.split("\t", -1);
            TsvRow row = new TsvRow(file, lineNo, fields);
            if (fields.length != columns.length) {
                throw row.defect("expected " + columns.length + " columns, found " + fields.length);
            }
            for (int i = 0; i < fields.length; i++) {
                if (fields[i].isEmpty()) {
                    throw row.defect("column '" + columns[i] + "' is empty");
                }
            }
            rows.add(row);
        }
        if (rows.size() != declared) {
            throw new IllegalStateException(RESOURCE_DIR + file + ":1: the header records " + declared
                    + " rows but the file has " + rows.size());
        }
        return new Tsv(file, m.group(2), Collections.unmodifiableList(rows));
    }

    /** {@code -} in a generated table means "none". */
    public static boolean isNone(String field) {
        return "-".equals(field);
    }

    /** {@code ggufArch} resolved through {@link Arch#fromGgufName}, or the row's defect when it is not an NF arch. */
    static Arch requireArch(TsvRow row, String ggufArch) {
        Arch a = Arch.fromGgufName(ggufArch);
        if (a == Arch.UNKNOWN) {
            throw row.defect("GGUF architecture '" + ggufArch + "' is not a NeuralFree architecture");
        }
        return a;
    }

    // ------------------------------------------------------------------ loaders

    private static Map<String, String> loadTensorNames() {
        Map<String, String> out = new LinkedHashMap<>();
        for (TsvRow row : read("tensor_names.tsv", "tensor", "gguf_template").rows()) {
            String template = row.get(1);
            Matcher m = PLACEHOLDER.matcher(template);
            while (m.find()) {
                if (!BID.equals(m.group(1)) && !"xid".equals(m.group(1))) {
                    throw row.defect("unknown placeholder {" + m.group(1) + "} in '" + template + "'");
                }
            }
            if (out.put(row.get(0), template) != null) {
                throw row.defect("tensor '" + row.get(0) + "' is listed twice");
            }
        }
        return Collections.unmodifiableMap(out);
    }

    private static void loadPatterns(Map<String, String> names, Map<String, List<String>> global,
                                     Map<String, List<String>> block) {
        long expectedSeq = 0;
        for (TsvRow row : read("tensor_patterns.tsv", "seq", "scope", "tensor", "hf_template").rows()) {
            if (!Long.toString(expectedSeq).equals(row.get(0))) {
                throw row.defect("seq " + row.get(0) + " out of order, expected " + expectedSeq);
            }
            expectedSeq++;
            String tensor = requireTensor(row, names, row.get(2));
            String template = row.get(3);
            Map<String, List<String>> target;
            if ("global".equals(row.get(1))) {
                target = global;
                if (template.indexOf('{') >= 0) {
                    throw row.defect("a model-level pattern has a placeholder: '" + template + "'");
                }
            } else if ("block".equals(row.get(1))) {
                target = block;
                requireOnlyBid(row, template);
            } else {
                throw row.defect("scope '" + row.get(1) + "' is neither global nor block");
            }
            List<String> patterns = target.get(tensor);
            boolean none = isNone(template);
            if (patterns != null && (none || patterns.isEmpty())) {
                throw row.defect("tensor '" + tensor + "' has patterns and an empty-pattern row");
            }
            if (patterns == null) {
                patterns = new ArrayList<>();
                target.put(tensor, patterns);
            }
            if (!none) {
                patterns.add(template);
            }
        }
    }

    /**
     * MODEL_TENSORS membership per arch. llama.cpp's constants.py lists a few tensors that have no TENSOR_NAMES
     * entry (the clip V_PROJ_NORM / V_QF_PROJ_* group); TensorNameMap never maps those, so absence from
     * {@code names} is not a defect here. It also lists clip's V_QF_PROJ_NORM twice; a set membership dedupes.
     */
    private static Map<String, Set<String>> loadArchTensors() {
        Map<String, Set<String>> out = new LinkedHashMap<>();
        for (TsvRow row : read("arch_tensors.tsv", "gguf_arch", "tensor").rows()) {
            requireArch(row, row.get(0));
            out.computeIfAbsent(row.get(0), k -> new LinkedHashSet<>()).add(row.get(1));
        }
        Map<String, Set<String>> frozen = new HashMap<>();
        for (Map.Entry<String, Set<String>> e : out.entrySet()) {
            frozen.put(e.getKey(), Collections.unmodifiableSet(e.getValue()));
        }
        return Collections.unmodifiableMap(frozen);
    }

    private static Map<String, Map<String, List<String>>> loadBlockOverrides(Map<String, String> names) {
        Map<String, Map<String, List<String>>> out = new LinkedHashMap<>();
        long expectedSeq = 0;
        String arch = null;
        for (TsvRow row : read("arch_block_overrides.tsv", "gguf_arch", "tensor", "hf_template", "seq").rows()) {
            requireArch(row, row.get(0));
            if (!row.get(0).equals(arch)) {
                if (out.containsKey(row.get(0))) {
                    throw row.defect("the rows of '" + row.get(0) + "' are not contiguous");
                }
                arch = row.get(0);
                expectedSeq = 0;
            }
            if (!Long.toString(expectedSeq).equals(row.get(3))) {
                throw row.defect("seq " + row.get(3) + " out of order, expected " + expectedSeq);
            }
            expectedSeq++;
            String tensor = requireTensor(row, names, row.get(1));
            requireOnlyBid(row, row.get(2));
            out.computeIfAbsent(arch, k -> new LinkedHashMap<>())
                    .computeIfAbsent(tensor, k -> new ArrayList<>()).add(row.get(2));
        }
        Map<String, Map<String, List<String>>> frozen = new HashMap<>();
        for (Map.Entry<String, Map<String, List<String>>> e : out.entrySet()) {
            frozen.put(e.getKey(), freezeLists(e.getValue()));
        }
        return Collections.unmodifiableMap(frozen);
    }

    private static Map<FilterKind, List<FilterRule>> loadFilters(Map<String, String> names) {
        Map<FilterKind, List<FilterRule>> out = new EnumMap<>(FilterKind.class);
        for (FilterKind k : FilterKind.values()) {
            out.put(k, new ArrayList<>());
        }
        for (TsvRow row : read("filters.tsv", "kind", "value", "replacement", "source").rows()) {
            FilterKind kind = FilterKind.fromTableName(row.get(0));
            if (kind == null) {
                throw row.defect("unknown filter kind '" + row.get(0) + "'");
            }
            boolean rename = kind == FilterKind.RENAME_SUFFIX || kind == FilterKind.RENAME_CONTAINS;
            if (!rename && !isNone(row.get(2))) {
                throw row.defect("a " + kind.tableName() + " rule has no replacement, found '" + row.get(2) + "'");
            }
            if (kind == FilterKind.FORCED_F32) {
                requireTensor(row, names, row.get(1));
            }
            out.get(kind).add(new FilterRule(kind, row.get(1), isNone(row.get(2)) ? "" : row.get(2), row.get(3)));
        }
        for (Map.Entry<FilterKind, List<FilterRule>> e : out.entrySet()) {
            e.setValue(Collections.unmodifiableList(e.getValue()));
        }
        return Collections.unmodifiableMap(out);
    }

    private static Predicate<String> buildForcedF32(Map<String, String> names, List<FilterRule> rules) {
        StringBuilder regex = new StringBuilder();
        for (FilterRule rule : rules) {
            String template = names.get(rule.value());
            if (regex.length() > 0) {
                regex.append('|');
            }
            int at = template.indexOf("{" + BID + "}");
            if (at < 0) {
                regex.append(Pattern.quote(template));
            } else {
                regex.append(Pattern.quote(template.substring(0, at))).append("(?:0|[1-9][0-9]*)")
                        .append(Pattern.quote(template.substring(at + BID.length() + 2)));
            }
        }
        if (regex.length() == 0) {
            return name -> false;
        }
        Pattern p = Pattern.compile("(?:" + regex + ")" + Pattern.quote(".weight"));
        return name -> name != null && p.matcher(name).matches();
    }

    private static String requireTensor(TsvRow row, Map<String, String> names, String tensor) {
        if (!names.containsKey(tensor)) {
            throw row.defect("tensor '" + tensor + "' is not in tensor_names.tsv");
        }
        return tensor;
    }

    private static void requireOnlyBid(TsvRow row, String template) {
        Matcher m = PLACEHOLDER.matcher(template);
        while (m.find()) {
            if (!BID.equals(m.group(1))) {
                throw row.defect("a block pattern may only hold {bid}, found {" + m.group(1) + "} in '"
                        + template + "'");
            }
        }
    }

    private static Map<String, List<String>> freezeLists(Map<String, List<String>> in) {
        Map<String, List<String>> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> e : in.entrySet()) {
            out.put(e.getKey(), Collections.unmodifiableList(new ArrayList<>(e.getValue())));
        }
        return Collections.unmodifiableMap(out);
    }
}
