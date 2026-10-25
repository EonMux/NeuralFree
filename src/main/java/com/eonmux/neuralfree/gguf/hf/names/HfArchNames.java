package com.eonmux.neuralfree.gguf.hf.names;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;

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
import java.util.Optional;
import java.util.Set;

/**
 * The Hugging Face architecture names llama.cpp's converter registers, from the generated {@code archs.tsv}
 * ({@code TEXT_MODEL_MAP} and {@code MMPROJ_MODEL_MAP} of LC/conversion/__init__.py with each class's
 * {@code @ModelBase.register} and {@code model_arch}) and {@code dynamic_archs.tsv} (the {@code self.model_arch}
 * assignments that choose the GGUF architecture at runtime). Parsed and validated once on first use.
 *
 * <p>Validation, each failure an {@link IllegalStateException} {@code "hf/<file>:<line>: <reason>"}: kinds are
 * {@code TEXT} or {@code MMPROJ}; a name appears once per kind; every GGUF architecture other than {@code -}
 * resolves through {@link Arch#fromGgufName} to a NeuralFree architecture; and a {@code -} architecture of a row
 * that has a converter class is covered by at least one {@code dynamic_archs.tsv} row of that class. Rows whose class
 * is {@code -} are names the converter lists without registering a class for that kind (the two
 * {@code GraniteSpeech*} TEXT names, which only have projector classes).</p>
 */
public final class HfArchNames {

    /** One {@code archs.tsv} row. Immutable. */
    public static final class Row {
        private final String hfName;
        private final HfArchEntry.Kind kind;
        private final String converterClass;
        private final String source;
        private final Arch arch;
        private final String modify;
        private final List<String> vocabWriters;

        Row(String hfName, HfArchEntry.Kind kind, String converterClass, String source, Arch arch, String modify,
            List<String> vocabWriters) {
            this.hfName = hfName;
            this.kind = kind;
            this.converterClass = converterClass;
            this.source = source;
            this.arch = arch;
            this.modify = modify;
            this.vocabWriters = vocabWriters;
        }

        /** The Hugging Face {@code architectures[0]} name. */
        public String hfName() { return hfName; }

        public HfArchEntry.Kind kind() { return kind; }

        /** The converter class, or null when the converter registers no class for this name and kind. */
        public String converterClass() { return converterClass; }

        /** The class's {@code file:line} in LC/conversion, or null without a class. */
        public String source() { return source; }

        /** The class's static architecture, or null when it is chosen at runtime ({@link #dynamicArchs()}). */
        public Arch arch() { return arch; }

        /** {@code own} when the class defines {@code modify_tensors}, else the class it inherits it from, or null. */
        public String modify() { return modify; }

        /** The {@code set_vocab} writers the class can reach, in dispatch order (unmodifiable). */
        public List<String> vocabWriters() { return vocabWriters; }

        /** True when the converter has a class for this name and kind. */
        public boolean hasClass() { return converterClass != null; }

        /** The runtime architecture choices of this row's class (empty for a static architecture). */
        public List<DynamicArch> dynamicArchs() {
            return converterClass == null ? Collections.<DynamicArch>emptyList()
                    : HfArchNames.dynamicArchs(converterClass);
        }

        /** Every architecture the converter can write for this name: the static one or the runtime choices. */
        public Set<Arch> possibleArchs() {
            Set<Arch> out = new LinkedHashSet<>();
            if (arch != null) {
                out.add(arch);
            }
            for (DynamicArch d : dynamicArchs()) {
                out.add(d.arch());
            }
            return Collections.unmodifiableSet(out);
        }

        @Override
        public String toString() {
            return hfName + " (" + kind + ", " + (converterClass == null ? "no class" : converterClass + " "
                    + source) + ", " + (arch == null ? "runtime arch" : arch.ggufName()) + ")";
        }
    }

    /** One {@code dynamic_archs.tsv} row: an architecture a class chooses at runtime, and when. Immutable. */
    public static final class DynamicArch {
        private final String converterClass;
        private final String source;
        private final String condition;
        private final Arch arch;

        DynamicArch(String converterClass, String source, String condition, Arch arch) {
            this.converterClass = converterClass;
            this.source = source;
            this.condition = condition;
            this.arch = arch;
        }

        public String converterClass() { return converterClass; }

        /** The assignment's {@code file:line} in LC/conversion. */
        public String source() { return source; }

        /** The Python condition under which the class picks {@link #arch()}, or null when unconditional. */
        public String condition() { return condition; }

        public Arch arch() { return arch; }

        @Override
        public String toString() {
            return converterClass + " " + source + ": " + arch.ggufName()
                    + (condition == null ? "" : " when " + condition);
        }
    }

    private final Map<HfArchEntry.Kind, Map<String, Row>> rows;
    private final Map<String, List<DynamicArch>> dynamic;

    private HfArchNames() {
        this.dynamic = loadDynamic();
        this.rows = loadRows(dynamic);
    }

    private static final class Holder {
        static final HfArchNames NAMES = new HfArchNames();
    }

    /** The row of {@code hfName} for {@code kind}, when the converter lists the name for that kind. */
    public static Optional<Row> row(String hfName, HfArchEntry.Kind kind) {
        Objects.requireNonNull(kind, "kind");
        return Optional.ofNullable(Holder.NAMES.rows.get(kind).get(hfName));
    }

    /** Every row of {@code kind}, in table order (unmodifiable). */
    public static List<Row> rows(HfArchEntry.Kind kind) {
        return Collections.unmodifiableList(new ArrayList<>(Holder.NAMES.rows.get(
                Objects.requireNonNull(kind, "kind")).values()));
    }

    /** The runtime architecture choices of {@code converterClass}, in table order (unmodifiable, maybe empty). */
    public static List<DynamicArch> dynamicArchs(String converterClass) {
        List<DynamicArch> d = Holder.NAMES.dynamic.get(converterClass);
        return d != null ? d : Collections.<DynamicArch>emptyList();
    }

    private static Map<String, List<DynamicArch>> loadDynamic() {
        Map<String, List<DynamicArch>> out = new LinkedHashMap<>();
        for (HfNameTables.TsvRow row : HfNameTables.read("dynamic_archs.tsv",
                "class", "source", "condition", "gguf_arch").rows()) {
            Arch arch = HfNameTables.requireArch(row, row.get(3));
            String condition = HfNameTables.isNone(row.get(2)) ? null : row.get(2);
            out.computeIfAbsent(row.get(0), k -> new ArrayList<>())
                    .add(new DynamicArch(row.get(0), row.get(1), condition, arch));
        }
        Map<String, List<DynamicArch>> frozen = new HashMap<>();
        for (Map.Entry<String, List<DynamicArch>> e : out.entrySet()) {
            frozen.put(e.getKey(), Collections.unmodifiableList(e.getValue()));
        }
        return Collections.unmodifiableMap(frozen);
    }

    private static Map<HfArchEntry.Kind, Map<String, Row>> loadRows(Map<String, List<DynamicArch>> dynamic) {
        Map<HfArchEntry.Kind, Map<String, Row>> out = new EnumMap<>(HfArchEntry.Kind.class);
        for (HfArchEntry.Kind k : HfArchEntry.Kind.values()) {
            out.put(k, new LinkedHashMap<>());
        }
        for (HfNameTables.TsvRow row : HfNameTables.read("archs.tsv",
                "hf_name", "kind", "class", "source", "gguf_arch", "modify", "vocab_writers").rows()) {
            HfArchEntry.Kind kind = kindOf(row);
            boolean hasClass = !HfNameTables.isNone(row.get(2));
            String cls = hasClass ? row.get(2) : null;
            Arch arch = HfNameTables.isNone(row.get(4)) ? null : HfNameTables.requireArch(row, row.get(4));
            if (arch == null && hasClass && !dynamic.containsKey(cls)) {
                throw row.defect("class " + cls + " chooses its architecture at runtime but dynamic_archs.tsv"
                        + " has no row for it");
            }
            if (!hasClass && (arch != null || !HfNameTables.isNone(row.get(3)))) {
                throw row.defect("a row without a converter class has a source or an architecture");
            }
            List<String> writers = HfNameTables.isNone(row.get(6)) ? Collections.<String>emptyList()
                    : Collections.unmodifiableList(Arrays.asList(row.get(6).split(",", -1)));
            for (String w : writers) {
                if (w.isEmpty()) {
                    throw row.defect("empty item in vocab_writers");
                }
            }
            Row r = new Row(row.get(0), kind, cls, hasClass ? row.get(3) : null, arch,
                    HfNameTables.isNone(row.get(5)) ? null : row.get(5), writers);
            if (out.get(kind).put(r.hfName(), r) != null) {
                throw row.defect("'" + r.hfName() + "' is listed twice as " + kind);
            }
        }
        for (Map.Entry<HfArchEntry.Kind, Map<String, Row>> e : out.entrySet()) {
            e.setValue(Collections.unmodifiableMap(e.getValue()));
        }
        return Collections.unmodifiableMap(out);
    }

    private static HfArchEntry.Kind kindOf(HfNameTables.TsvRow row) {
        for (HfArchEntry.Kind k : HfArchEntry.Kind.values()) {
            if (k.name().equals(row.get(1))) {
                return k;
            }
        }
        throw row.defect("kind '" + row.get(1) + "' is neither TEXT nor MMPROJ");
    }
}
