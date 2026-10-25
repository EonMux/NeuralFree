package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.hf.names.HfNameTables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The generated {@code hparam_aliases.tsv}: the steps of {@code TextModel.__init__}'s RoPE normalization and of
 * {@code TextModel.set_gguf_parameters} (LC/conversion/base.py:1514-1720), one row per step in source order, each with
 * the GGUF key it writes, the {@code hparams} aliases it reads and its rule. Parsed and validated once on first use;
 * a malformed row throws {@link IllegalStateException} {@code "hf/hparam_aliases.tsv:<line>: <reason>"}.
 *
 * <p>Rule grammar ({@code ; }-separated clauses): a head, then optionally {@code when <python condition>},
 * {@code value <python expression>}, {@code after <python statement>} and {@code default=<python value>}. Heads:</p>
 * <ul>
 *   <li>{@code local <name>}: assigns a local variable from the aliases;</li>
 *   <li>{@code mirror <path>}: copies the first present alias into the RoPE parameters;</li>
 *   <li>{@code required}, {@code optional}: writes the first present alias, failing or skipping when absent;</li>
 *   <li>{@code always}: writes the {@code value} expression;</li>
 *   <li>{@code reject}: raises the {@code value} error.</li>
 * </ul>
 */
public final class HfHparamAliases {

    /** The head of a rule. */
    public enum Head {
        LOCAL, MIRROR, REQUIRED, OPTIONAL, ALWAYS, REJECT
    }

    /** One row. Immutable. */
    public static final class Row {
        private final int line;
        private final String template;
        private final List<String> aliases;
        private final GgufKeyTable.KeyType type;
        private final Head head;
        private final String target;
        private final String condition;
        private final String value;
        private final String after;
        private final String rule;

        Row(int line, String template, List<String> aliases, GgufKeyTable.KeyType type, Head head, String target,
            String condition, String value, String after, String rule) {
            this.line = line;
            this.template = template;
            this.aliases = aliases;
            this.type = type;
            this.head = head;
            this.target = target;
            this.condition = condition;
            this.value = value;
            this.after = after;
            this.rule = rule;
        }

        /** The 1-based line in the file. */
        public int line() { return line; }

        /** The {@link com.eonmux.neuralfree.gguf.hparams.GgufKeys} template written, or null for a non-writing row. */
        public String template() { return template; }

        /** The aliases read, first present wins (unmodifiable, maybe empty). */
        public List<String> aliases() { return aliases; }

        /** The key's type from {@link GgufKeyTable}, or null for a non-writing row. */
        public GgufKeyTable.KeyType type() { return type; }

        public Head head() { return head; }

        /** The variable or path named by a {@code local} or {@code mirror} head, else null. */
        public String target() { return target; }

        /** The {@code when} condition, or null. */
        public String condition() { return condition; }

        /** The {@code value} expression, or null. */
        public String value() { return value; }

        /** The {@code after} statement, or null. */
        public String after() { return after; }

        /** The whole rule as written. */
        public String rule() { return rule; }

        /** An {@link IllegalStateException} {@code "hf/hparam_aliases.tsv:<line>: <reason>"}. */
        public IllegalStateException defect(String reason) {
            return new IllegalStateException(HfNameTables.RESOURCE_DIR + FILE + ":" + line + ": " + reason);
        }

        @Override
        public String toString() {
            return FILE + ":" + line + " " + (template == null ? "-" : template) + " " + aliases + " " + rule;
        }
    }

    static final String FILE = "hparam_aliases.tsv";

    private HfHparamAliases() {
    }

    private static final class Holder {
        static final List<Row> ROWS = load();
    }

    /** Every row in file order (unmodifiable). */
    public static List<Row> rows() {
        return Holder.ROWS;
    }

    private static List<Row> load() {
        List<Row> out = new ArrayList<>();
        for (HfNameTables.TsvRow r : HfNameTables.read(FILE, "gguf_key_template", "hf_aliases", "value_type", "rule")
                .rows()) {
            out.add(parse(r));
        }
        return Collections.unmodifiableList(out);
    }

    private static Row parse(HfNameTables.TsvRow r) {
        String rule = r.get(3);
        String[] clauses = rule.split("; ", -1);
        String[] headWords = clauses[0].split(" ", 2);
        Head head = headOf(r, headWords[0]);
        String target = headWords.length > 1 ? headWords[1] : null;
        if ((head == Head.LOCAL || head == Head.MIRROR) != (target != null)) {
            throw r.defect("rule head '" + clauses[0] + "' does not fit " + head);
        }
        String condition = null;
        String value = null;
        String after = null;
        for (int i = 1; i < clauses.length; i++) {
            String c = clauses[i];
            if (c.startsWith("when ")) {
                condition = c.substring(5);
            } else if (c.startsWith("value ")) {
                value = c.substring(6);
            } else if (c.startsWith("after ")) {
                after = c.substring(6);
            } else if (!c.startsWith("default=")) {
                throw r.defect("unknown rule clause '" + c + "'");
            }
        }
        boolean writes = head == Head.REQUIRED || head == Head.OPTIONAL || head == Head.ALWAYS;
        String template = keyTemplate(r, r.get(0), writes);
        GgufKeyTable.KeyType type = template == null ? null : checkType(r, template, r.get(2));
        if ((head == Head.ALWAYS || head == Head.REJECT) && value == null) {
            throw r.defect(head + " rule without a value");
        }
        List<String> aliases = HfNameTables.isNone(r.get(1)) ? Collections.<String>emptyList()
                : Collections.unmodifiableList(Arrays.asList(r.get(1).split(",", -1)));
        for (String a : aliases) {
            if (a.isEmpty()) {
                throw r.defect("empty alias");
            }
        }
        if ((head == Head.REQUIRED || head == Head.OPTIONAL || head == Head.LOCAL) && aliases.isEmpty()) {
            throw r.defect(head + " rule without aliases");
        }
        return new Row(r.line(), template, aliases, type, head, target, condition, value, after, rule);
    }

    private static Head headOf(HfNameTables.TsvRow r, String word) {
        for (Head h : Head.values()) {
            if (h.name().equalsIgnoreCase(word)) {
                return h;
            }
        }
        throw r.defect("unknown rule head '" + word + "'");
    }

    private static String keyTemplate(HfNameTables.TsvRow r, String field, boolean writes) {
        if (HfNameTables.isNone(field)) {
            if (writes) {
                throw r.defect("a writing rule needs a key");
            }
            return null;
        }
        if (!writes) {
            throw r.defect("a non-writing rule has the key " + field);
        }
        if (!field.startsWith("{arch}.") || field.indexOf('{', 1) >= 0) {
            throw r.defect("key '" + field + "' is not an {arch}-scoped key without other placeholders");
        }
        return "%s." + field.substring("{arch}.".length());
    }

    private static GgufKeyTable.KeyType checkType(HfNameTables.TsvRow r, String template, String declared) {
        GgufKeyTable.KeyType t = GgufKeyTable.type(template);
        if (t == null) {
            throw r.defect("key " + template + " has no type in gguf_keys.tsv");
        }
        if (!t.toString().equals(declared)) {
            throw r.defect("key " + template + " is " + declared + " here but " + t + " in gguf_keys.tsv");
        }
        return t;
    }
}
