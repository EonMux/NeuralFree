package com.eonmux.neuralfree.gguf.hf.config;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * vLLM's {@code hf_to_vllm_mapper} of one architecture: the renames vLLM applies to a checkpoint tensor name before
 * it reaches the model's modules ({@code WeightsMapper._map_name_with_shard}, model_executor/models/utils.py), so
 * the vLLM module a checkpoint tensor loads into, which {@code --quantization-config} patterns name. The entries
 * apply in vLLM's order: every matching {@code regex} ({@code re.sub}), the first occurrence of every contained
 * {@code substr}, of every contained {@code stacked} name (a fused module's shard), every matching {@code prefix}
 * and the last occurrence of every matching {@code suffix}; a matched entry without a replacement drops the weight.
 * The table is the generated resource {@value #RESOURCE} ({@code gradle/vllmgen/vllm_mappers.py}). Immutable.
 */
public final class VllmWeightsMapper {

    /** The generated table of every architecture's mapper. */
    public static final String RESOURCE = "vllm/weights_mappers.tsv";

    private static final String DROP = "<drop>";
    private static final Pattern HEADER = Pattern.compile(
            "# weights_mappers\\.tsv - Generated from vLLM (\\S+) by \\S+; do not edit\\. Rows: (\\d+)");
    private static final String COLUMNS = "# architecture\tkind\tmatch\treplacement\tshard";

    /** The mapper of an architecture vLLM gives none. */
    public static final VllmWeightsMapper IDENTITY = new VllmWeightsMapper("", Collections.<Entry>emptyList(), null);

    private final String architecture;
    private final List<Entry> entries;
    private final String unresolved;

    private VllmWeightsMapper(String architecture, List<Entry> entries, String unresolved) {
        this.architecture = architecture;
        this.entries = Collections.unmodifiableList(new ArrayList<>(entries));
        this.unresolved = unresolved;
    }

    /** One entry of the table. */
    private static final class Entry {
        final String kind;
        final String match;
        final String replacement;
        final Pattern regex;
        final String javaReplacement;

        Entry(String kind, String match, String replacement) {
            this.kind = kind;
            this.match = match;
            this.replacement = replacement;
            this.regex = kind.equals("regex") ? Pattern.compile(javaPattern(match)) : null;
            this.javaReplacement = kind.equals("regex") && replacement != null ? javaReplacement(replacement) : null;
        }
    }

    /** The table, read once. */
    private static final class Table {
        static final Map<String, VllmWeightsMapper> BY_ARCH = read();
    }

    /**
     * The mapper of the vLLM architecture {@code architecture} (the {@code architectures[0]} of the checkpoint's
     * {@code config.json}); {@link #IDENTITY} for an architecture vLLM gives no mapper.
     */
    public static VllmWeightsMapper of(String architecture) {
        if (architecture == null) {
            return IDENTITY;
        }
        VllmWeightsMapper m = Table.BY_ARCH.get(architecture);
        return m == null ? IDENTITY : m;
    }

    /**
     * The mapper of {@code entries} ({@code {kind, match, replacement-or-null, shard}} rows), for tests and the
     * table reader.
     *
     * @throws IllegalArgumentException for an unknown kind or a regex Java cannot compile
     */
    static VllmWeightsMapper of(String architecture, List<String[]> entries) {
        List<Entry> out = new ArrayList<>();
        String unresolved = null;
        for (String[] e : entries) {
            if (e[0].equals("unresolved")) {
                unresolved = e[1];
                continue;
            }
            if (!e[0].equals("regex") && !e[0].equals("substr") && !e[0].equals("stacked") && !e[0].equals("prefix")
                    && !e[0].equals("suffix")) {
                throw new IllegalArgumentException("unknown mapper kind '" + e[0] + "'");
            }
            out.add(new Entry(e[0], e[1], e[2]));
        }
        return new VllmWeightsMapper(architecture, out, unresolved);
    }

    /** The architecture, empty for {@link #IDENTITY}. */
    public String architecture() {
        return architecture;
    }

    /**
     * Whether vLLM builds this mapper with code the table does not reduce to renames, so names cannot be mapped;
     * the reason is {@link #unresolvedReason()}.
     */
    public boolean unresolved() {
        return unresolved != null;
    }

    /** The expression the generator could not evaluate, or null. */
    public String unresolvedReason() {
        return unresolved;
    }

    /**
     * The vLLM name of checkpoint tensor {@code name}, or null when vLLM drops it.
     *
     * @throws IllegalStateException when the mapper is {@link #unresolved()}
     */
    public String map(String name) {
        Objects.requireNonNull(name, "name");
        if (unresolved != null) {
            throw new IllegalStateException("vLLM's hf_to_vllm_mapper of " + architecture + " is built by code ("
                    + unresolved + ")");
        }
        String key = name;
        for (String kind : new String[]{"regex", "substr", "stacked", "prefix", "suffix"}) {
            for (Entry e : entries) {
                if (!e.kind.equals(kind)) {
                    continue;
                }
                String next = apply(e, key);
                if (next == null) {
                    return null;
                }
                key = next;
            }
        }
        return key;
    }

    /** {@code key} after entry {@code e}: unchanged when it does not match, null when the entry drops it. */
    private static String apply(Entry e, String key) {
        switch (e.kind) {
            case "regex": {
                Matcher m = e.regex.matcher(key);
                if (!m.find()) {
                    return key;
                }
                return e.replacement == null ? null : m.replaceAll(e.javaReplacement);
            }
            case "substr":
            case "stacked": {
                int at = key.indexOf(e.match);
                if (at < 0) {
                    return key;
                }
                return e.replacement == null ? null
                        : key.substring(0, at) + e.replacement + key.substring(at + e.match.length());
            }
            case "prefix":
                if (!key.startsWith(e.match)) {
                    return key;
                }
                return e.replacement == null ? null : e.replacement + key.substring(e.match.length());
            default: {
                if (!key.endsWith(e.match)) {
                    return key;
                }
                if (e.replacement == null) {
                    return null;
                }
                int at = key.lastIndexOf(e.match);
                return key.substring(0, at) + e.replacement + key.substring(at + e.match.length());
            }
        }
    }

    /**
     * The checkpoint shard names of the fused vLLM module {@code module} that this mapper's {@code stacked}
     * entries load into it, in table order: {@code module} with each entry's replacement put back to its match.
     * Empty when no stacked entry names the module.
     */
    public List<String> stackedShards(String module) {
        Objects.requireNonNull(module, "module");
        List<String> out = new ArrayList<>();
        for (Entry e : entries) {
            if (!e.kind.equals("stacked") || e.replacement == null) {
                continue;
            }
            String fused = trimDot(e.replacement);
            String shard = trimDot(e.match);
            if (fused.isEmpty() || !module.endsWith(fused)) {
                continue;
            }
            int at = module.length() - fused.length();
            if (at > 0 && module.charAt(at - 1) != '.' && e.replacement.startsWith(".")) {
                continue;
            }
            String name = module.substring(0, at) + shard;
            if (!out.contains(name)) {
                out.add(name);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** {@code s} without a leading and a trailing dot, and without a trailing {@code .weight}. */
    private static String trimDot(String s) {
        String t = s.endsWith(".weight") ? s.substring(0, s.length() - ".weight".length()) : s;
        t = t.startsWith(".") ? t.substring(1) : t;
        return t.endsWith(".") ? t.substring(0, t.length() - 1) : t;
    }

    /** A Python {@code re} pattern as a Java one: {@code \Z} is Java's {@code \z}, {@code (?P<n>} is {@code (?<n>}. */
    static String javaPattern(String python) {
        StringBuilder b = new StringBuilder(python.length());
        for (int i = 0; i < python.length(); i++) {
            char c = python.charAt(i);
            if (c == '\\' && i + 1 < python.length()) {
                char n = python.charAt(++i);
                b.append('\\').append(n == 'Z' ? 'z' : n);
            } else if (python.startsWith("(?P<", i)) {
                b.append("(?<");
                i += 3;
            } else if (python.startsWith("(?P=", i)) {
                int end = python.indexOf(')', i);
                b.append("\\k<").append(python, i + 4, end).append('>');
                i = end;
            } else {
                b.append(c);
            }
        }
        return b.toString();
    }

    /**
     * A Python {@code re.sub} template as a Java replacement: {@code \N} (one or two digits) and {@code \g<N>} are
     * numbered group references ({@code $N}), {@code \g<name>} a named one ({@code ${name}}), other escapes are
     * literal characters. Every literal character is escaped, so a digit after a group reference stays literal.
     *
     * @throws IllegalArgumentException for an octal escape or an unterminated {@code \g<}
     */
    static String javaReplacement(String python) {
        StringBuilder b = new StringBuilder(python.length() * 2);
        for (int i = 0; i < python.length(); i++) {
            char c = python.charAt(i);
            if (c != '\\' || i + 1 >= python.length()) {
                b.append('\\').append(c);
                continue;
            }
            char n = python.charAt(i + 1);
            if (n == '0') {
                throw new IllegalArgumentException("octal escape in the replacement '" + python + "'");
            }
            if (Character.isDigit(n)) {
                int end = i + 2 < python.length() && Character.isDigit(python.charAt(i + 2)) ? i + 3 : i + 2;
                b.append('$').append(python, i + 1, end);
                i = end - 1;
            } else if (n == 'g' && i + 2 < python.length() && python.charAt(i + 2) == '<') {
                int end = python.indexOf('>', i + 3);
                if (end < 0) {
                    throw new IllegalArgumentException("unterminated \\g< in the replacement '" + python + "'");
                }
                String group = python.substring(i + 3, end);
                b.append(isNumber(group) ? "$" + group : "${" + group + "}");
                i = end;
            } else {
                b.append('\\').append(unescape(n));
                i++;
            }
        }
        return b.toString();
    }

    private static boolean isNumber(String s) {
        if (s.isEmpty()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            if (!Character.isDigit(s.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    private static char unescape(char n) {
        switch (n) {
            case 'n':
                return '\n';
            case 't':
                return '\t';
            default:
                return n;
        }
    }

    // ------------------------------------------------------------------ the table

    private static Map<String, VllmWeightsMapper> read() {
        InputStream in = VllmWeightsMapper.class.getClassLoader().getResourceAsStream(RESOURCE);
        if (in == null) {
            throw new IllegalStateException(RESOURCE + ": the resource is missing from the classpath");
        }
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            return parse(r);
        } catch (IOException e) {
            throw new UncheckedIOException(RESOURCE + ": cannot read the resource", e);
        }
    }

    /**
     * The table of {@code r}: the header line with its row count, the column line, then 5 tab-separated columns
     * per row.
     *
     * @throws IllegalStateException {@code "vllm/weights_mappers.tsv:<line>: <reason>"} for any defect
     */
    static Map<String, VllmWeightsMapper> parse(BufferedReader r) throws IOException {
        String header = r.readLine();
        Matcher m = header == null ? null : HEADER.matcher(header);
        if (m == null || !m.matches()) {
            throw new IllegalStateException(RESOURCE + ":1: malformed header line: " + header);
        }
        if (!COLUMNS.equals(r.readLine())) {
            throw new IllegalStateException(RESOURCE + ":2: the column line is not '" + COLUMNS + "'");
        }
        Map<String, List<String[]>> rows = new HashMap<>();
        List<String> order = new ArrayList<>();
        int line = 2;
        int count = 0;
        for (String s = r.readLine(); s != null; s = r.readLine()) {
            line++;
            String[] f = s.split("\t", -1);
            if (f.length != 5 || f[0].isEmpty() || f[1].isEmpty()) {
                throw new IllegalStateException(RESOURCE + ":" + line + ": expected 5 columns with an architecture"
                        + " and a kind, got '" + s + "'");
            }
            if (!rows.containsKey(f[0])) {
                order.add(f[0]);
            }
            rows.computeIfAbsent(f[0], k -> new ArrayList<>())
                    .add(new String[]{f[1], f[2], f[3].equals(DROP) ? null : f[3], f[4]});
            count++;
        }
        if (count != Long.parseLong(m.group(2))) {
            throw new IllegalStateException(RESOURCE + ": the header records " + m.group(2) + " rows, found " + count);
        }
        Map<String, VllmWeightsMapper> out = new HashMap<>();
        for (String arch : order) {
            try {
                out.put(arch, of(arch, rows.get(arch)));
            } catch (RuntimeException e) {
                throw new IllegalStateException(RESOURCE + ": the mapper of " + arch + ": " + e.getMessage(), e);
            }
        }
        return Collections.unmodifiableMap(out);
    }

    @Override
    public String toString() {
        return "VllmWeightsMapper[" + (architecture.isEmpty() ? "identity" : architecture) + ", " + entries.size()
                + " entries" + (unresolved == null ? "" : ", unresolved") + "]";
    }
}
