package com.eonmux.neuralfree.gguf.jinja;

import com.eonmux.neuralfree.gguf.jinja.JinjaMethods.Params;
import com.eonmux.neuralfree.gguf.jinja.JinjaRuntime.TemplateException;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Jinja2 built-in filters plus the HF-transformers additions used by chat templates
 * ({@code tojson} with HF keyword arguments, {@code from_json}). Undefined inputs yield empty
 * results; None is accepted by the filters for which llama.cpp accepts it (llama.cpp common/jinja/value.cpp)
 * and rejected with a clear error elsewhere.
 */
final class JinjaFilters {

    private static final Set<String> NAMES = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "abs", "attr", "batch", "capitalize", "center", "count", "d", "default", "dictsort", "e", "escape",
            "first", "float", "format", "from_json", "indent", "int", "integer", "items", "join", "last", "length",
            "list", "lower", "lstrip", "map", "max", "min", "reject", "rejectattr", "replace", "reverse", "round",
            "rstrip", "safe", "select", "selectattr", "sort", "split", "string", "sum", "title", "tojson", "trim",
            "strip", "unique", "upper", "wordcount")));

    private static final Pattern TITLE_SPLIT = Pattern.compile("[-\\s({\\[<]+", Pattern.UNICODE_CHARACTER_CLASS);
    private static final Pattern WORD = Pattern.compile("\\w+", Pattern.UNICODE_CHARACTER_CLASS);

    private JinjaFilters() {}

    /** Whether {@code name} is a supported filter. */
    static boolean exists(String name) { return NAMES.contains(name); }

    /** Applies filter {@code name} to {@code v}. */
    static Object apply(String name, Object v, List<Object> args, Map<String, Object> kwargs) {
        Params p = new Params(name, args, kwargs);
        switch (name) {
            case "abs": return abs(v);
            case "attr": return JinjaValue.getAttr(v, p.only("name").string("name", 0));
            case "batch": return batch(v, p.only("linecount", "fill_with"));
            case "capitalize": p.only(); return JinjaMethods.capitalize(JinjaValue.str(v));
            case "center": {
                p.only("width");
                return JinjaMethods.bind(JinjaValue.str(v), "center").call(
                        Collections.<Object>singletonList(p.integer("width", 0, 80)), Collections.<String, Object>emptyMap(), null);
            }
            case "count": case "length": p.only(); return JinjaValue.length(v);
            case "d": case "default": return dflt(v, p.only("default_value", "boolean"));
            case "dictsort": return dictsort(v, p.only("case_sensitive", "by", "reverse"));
            case "e": case "escape": p.only(); return escape(JinjaValue.str(v));
            case "first": p.only(); return firstOrLast(v, true);
            case "last": p.only(); return firstOrLast(v, false);
            case "float": return toFloat(v, p.only("default").get("default", 0, 0.0));
            case "format": return format(v, args, kwargs);
            case "from_json": p.only(); return Json.parse(requireString(v, name));
            case "indent": return indent(JinjaValue.str(v), p.only("width", "first", "blank"));
            case "int": case "integer": return toInt(v, p.only("default", "base"));
            case "items": p.only(); return items(v);
            case "join": return join(v, p.only("d", "attribute"));
            case "list": p.only(); return toList(v);
            case "lower": p.only(); return JinjaValue.str(v).toLowerCase(Locale.ROOT);
            case "upper": p.only(); return JinjaValue.str(v).toUpperCase(Locale.ROOT);
            case "map": return map(v, args, kwargs);
            case "max": case "min": return minMax(v, p.only("case_sensitive", "attribute"), "max".equals(name));
            case "select": case "reject": return select(v, args, false, "select".equals(name));
            case "selectattr": case "rejectattr": return select(v, args, true, "selectattr".equals(name));
            case "replace": {
                p.only("old", "new", "count");
                return JinjaMethods.replace(JinjaValue.str(v), p.string("old", 0), JinjaValue.str(p.require("new", 1)),
                        p.integer("count", 2, -1));
            }
            case "reverse": p.only(); return reverse(v);
            case "round": return round(v, p.only("precision", "method"));
            case "safe": case "string": p.only(); return JinjaValue.str(v);
            case "sort": return sort(v, p.only("reverse", "case_sensitive", "attribute"));
            case "sum": return sum(v, p.only("attribute", "start"));
            case "title": p.only(); return title(JinjaValue.str(v));
            case "tojson": return tojson(v, p.only("ensure_ascii", "indent", "separators", "sort_keys"));
            case "trim": case "strip": return strip(v, p, true, true);
            case "lstrip": return strip(v, p, true, false);
            case "rstrip": return strip(v, p, false, true);
            case "split": {
                p.only("sep", "maxsplit");
                Object sep = p.get("sep", 0, null);
                return JinjaMethods.split(JinjaValue.str(v), sep == null || sep == JinjaValue.UNDEFINED ? null
                        : requireString(sep, "split"), p.integer("maxsplit", 1, -1));
            }
            case "unique": return unique(v, p.only("case_sensitive", "attribute"));
            case "wordcount": {
                p.only();
                Matcher m = WORD.matcher(JinjaValue.str(v));
                long n = 0;
                while (m.find()) n++;
                return n;
            }
            default: throw new TemplateException("unknown filter '" + name + "'");
        }
    }

    // ------------------------------------------------------------------ helpers

    private static String requireString(Object v, String fn) {
        if (!(v instanceof String)) throw new TemplateException(fn + " expects a string, got " + JinjaValue.typeName(v));
        return (String) v;
    }

    /** Items of a sequence filter input; Undefined and None are empty. */
    private static List<Object> seq(Object v) {
        return v == null ? new ArrayList<>() : JinjaValue.iterate(v);
    }

    /**
     * Jinja2 attribute getter: a dotted path whose all-digit parts are integer indices, each step an
     * item lookup with attribute fallback.
     */
    static Object attrPath(Object item, String path) {
        Object v = item;
        for (String part : path.split("\\.", -1)) {
            Object key = !part.isEmpty() && part.chars().allMatch(Character::isDigit) ? (Object) Long.parseLong(part) : part;
            v = JinjaValue.getItem(v, key);
        }
        return v;
    }

    /** Sort/unique key: the attribute when given, lower-cased when case-insensitive. */
    private static Object key(Object item, Object attribute, boolean caseSensitive) {
        Object k = attribute == null || attribute == JinjaValue.UNDEFINED ? item : attrPath(item, JinjaValue.str(attribute));
        return !caseSensitive && k instanceof String ? ((String) k).toLowerCase(Locale.ROOT) : k;
    }

    // ------------------------------------------------------------------ filters

    private static Object abs(Object v) {
        if (v instanceof Long) return Math.abs((Long) v);
        if (v instanceof Double) return Math.abs((Double) v);
        if (v instanceof Boolean) return (Boolean) v ? 1L : 0L;
        throw new TemplateException("abs expects a number, got " + JinjaValue.typeName(v));
    }

    private static Object batch(Object v, Params p) {
        long size = p.integer("linecount", 0, 0);
        if (size <= 0) throw new TemplateException("batch() linecount must be positive");
        Object fill = p.get("fill_with", 1, null);
        List<Object> out = new ArrayList<>();
        List<Object> cur = new ArrayList<>();
        for (Object item : seq(v)) {
            if (cur.size() == size) {
                out.add(cur);
                cur = new ArrayList<>();
            }
            cur.add(item);
        }
        if (!cur.isEmpty()) {
            if (fill != null && fill != JinjaValue.UNDEFINED) while (cur.size() < size) cur.add(fill);
            out.add(cur);
        }
        return out;
    }

    /**
     * {@code default}: replaces Undefined and None (None per llama.cpp, llama.cpp common/jinja/value.cpp
     * default filter); with {@code boolean=true} any falsy value.
     */
    private static Object dflt(Object v, Params p) {
        Object d = p.get("default_value", 0, "");
        boolean bool = JinjaValue.truthy(p.get("boolean", 1, false));
        if (v == JinjaValue.UNDEFINED || v == null || (bool && !JinjaValue.truthy(v))) return d;
        return v;
    }

    private static Object dictsort(Object v, Params p) {
        if (!JinjaValue.isDict(v)) throw new TemplateException("dictsort expects a dict, got " + JinjaValue.typeName(v));
        boolean cs = JinjaValue.truthy(p.get("case_sensitive", 0, false));
        String by = JinjaValue.str(p.get("by", 1, "key"));
        if (!"key".equals(by) && !"value".equals(by)) throw new TemplateException("dictsort 'by' must be 'key' or 'value'");
        int pos = "key".equals(by) ? 0 : 1;
        List<Object> items = JinjaValue.items((Map<?, ?>) v);
        Comparator<Object> cmp = (a, b) -> JinjaValue.compare(key(((List<?>) a).get(pos), null, cs), key(((List<?>) b).get(pos), null, cs));
        items.sort(JinjaValue.truthy(p.get("reverse", 2, false)) ? cmp.reversed() : cmp);
        return items;
    }

    private static String escape(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '&': sb.append("&amp;"); break;
                case '<': sb.append("&lt;"); break;
                case '>': sb.append("&gt;"); break;
                case '"': sb.append("&#34;"); break;
                case '\'': sb.append("&#39;"); break;
                default: sb.append(c);
            }
        }
        return sb.toString();
    }

    private static Object firstOrLast(Object v, boolean first) {
        List<Object> items = seq(v);
        if (items.isEmpty()) return JinjaValue.UNDEFINED;
        return items.get(first ? 0 : items.size() - 1);
    }

    private static Object toFloat(Object v, Object dflt) {
        if (v instanceof Double) return v;
        if (v instanceof Long) return (double) (Long) v;
        if (v instanceof Boolean) return (Boolean) v ? 1.0 : 0.0;
        if (v instanceof String) {
            String s = JinjaMethods.strip((String) v, null, true, true).replace("_", "");
            String lower = s.toLowerCase(Locale.ROOT);
            if (lower.matches("[+-]?(inf|infinity)")) return lower.startsWith("-") ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
            if (lower.matches("[+-]?nan")) return Double.NaN;
            if (s.matches("[+-]?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?")) return Double.parseDouble(s);
        }
        return dflt;
    }

    /** Jinja2 {@code int}: parses strings in {@code base} (0 detects a prefix), falls back to float, else default. */
    private static Object toInt(Object v, Params p) {
        Object dflt = p.get("default", 0, 0L);
        long base = p.integer("base", 1, 10);
        if (v instanceof Long) return v;
        if (v instanceof Boolean) return (Boolean) v ? 1L : 0L;
        if (v instanceof Double) {
            double d = (Double) v;
            return Double.isNaN(d) || Double.isInfinite(d) ? dflt : (long) d;
        }
        if (!(v instanceof String)) return dflt;
        String s = JinjaMethods.strip((String) v, null, true, true).replace("_", "");
        try {
            String digits = s;
            boolean neg = digits.startsWith("-");
            if (neg || digits.startsWith("+")) digits = digits.substring(1);
            if (digits.startsWith("+") || digits.startsWith("-")) throw new NumberFormatException(s);
            int radix = (int) base;
            String lower = digits.toLowerCase(Locale.ROOT);
            if (base == 0 || base == 16 || base == 8 || base == 2) {
                String prefix = lower.length() > 1 ? lower.substring(0, 2) : "";
                int detected = "0x".equals(prefix) ? 16 : "0o".equals(prefix) ? 8 : "0b".equals(prefix) ? 2 : 0;
                if (detected != 0 && (base == 0 || base == detected)) {
                    digits = digits.substring(2);
                    radix = detected;
                } else if (base == 0) {
                    radix = 10;
                }
            }
            long r = Long.parseLong(digits, radix);
            return neg ? -r : r;
        } catch (NumberFormatException e) {
            Object f = toFloat(s, null);
            if (f instanceof Double && !((Double) f).isNaN() && !((Double) f).isInfinite()) return (long) (double) (Double) f;
            return dflt;
        }
    }

    private static Object format(Object v, List<Object> args, Map<String, Object> kwargs) {
        if (!args.isEmpty() && !kwargs.isEmpty()) throw new TemplateException("format filter takes positional or keyword arguments, not both");
        Object values = kwargs.isEmpty() ? new JinjaValue.Tuple(args) : new LinkedHashMap<Object, Object>(kwargs);
        return JinjaMethods.percentFormat(JinjaValue.str(v), values);
    }

    /**
     * Jinja2 {@code indent} (jinja2/filters.py do_indent), including its trailing-newline handling. Translated from
     * Jinja2 (BSD-3-Clause License, Copyright 2007 Pallets); see THIRD-PARTY-NOTICES.md.
     */
    private static Object indent(String s, Params p) {
        Object w = p.get("width", 0, 4L);
        String indention;
        if (w instanceof String) {
            indention = (String) w;
        } else {
            long n = JinjaValue.toLong(w, "indent width");
            StringBuilder sb = new StringBuilder();
            for (long i = 0; i < n; i++) sb.append(' ');
            indention = sb.toString();
        }
        boolean first = JinjaValue.truthy(p.get("first", 1, false));
        boolean blank = JinjaValue.truthy(p.get("blank", 2, false));
        List<Object> lines = JinjaMethods.splitlines(s + "\n", false);
        StringBuilder rv = new StringBuilder();
        if (blank) {
            rv.append(JinjaMethods.join("\n" + indention, lines));
        } else {
            rv.append(lines.isEmpty() ? "" : (String) lines.get(0));
            for (int i = 1; i < lines.size(); i++) {
                String line = (String) lines.get(i);
                rv.append('\n');
                if (!line.isEmpty()) rv.append(indention);
                rv.append(line);
            }
        }
        return first ? indention + rv : rv.toString();
    }

    private static Object items(Object v) {
        if (v == JinjaValue.UNDEFINED || v == null) return new ArrayList<>();
        if (v instanceof Map) return JinjaValue.items((Map<?, ?>) v);
        throw new TemplateException("items expects a mapping, got " + JinjaValue.typeName(v));
    }

    private static Object join(Object v, Params p) {
        String sep = JinjaValue.str(p.get("d", 0, ""));
        Object attribute = p.get("attribute", 1, null);
        List<Object> items = seq(v);
        if (attribute != null && attribute != JinjaValue.UNDEFINED) {
            List<Object> mapped = new ArrayList<>(items.size());
            for (Object item : items) mapped.add(attrPath(item, JinjaValue.str(attribute)));
            items = mapped;
        }
        return JinjaMethods.join(sep, items);
    }

    private static Object toList(Object v) {
        if (v == null) throw new TemplateException("list filter: 'NoneType' object is not iterable");
        return JinjaValue.iterate(v);
    }

    /** {@code map(attribute=..., default=...)} or {@code map('filter', args...)}. */
    private static Object map(Object v, List<Object> args, Map<String, Object> kwargs) {
        List<Object> items = seq(v);
        List<Object> out = new ArrayList<>(items.size());
        if (kwargs.containsKey("attribute")) {
            new Params("map", args, kwargs).only("attribute", "default");
            if (!args.isEmpty()) throw new TemplateException("map(attribute=...) takes no positional arguments");
            String path = JinjaValue.str(kwargs.get("attribute"));
            Object dflt = kwargs.getOrDefault("default", JinjaValue.UNDEFINED);
            for (Object item : items) {
                Object a = attrPath(item, path);
                out.add(a == JinjaValue.UNDEFINED ? dflt : a);
            }
            return out;
        }
        if (args.isEmpty()) throw new TemplateException("map requires a filter name or attribute=");
        String filter = JinjaValue.str(args.get(0));
        List<Object> rest = args.subList(1, args.size());
        for (Object item : items) out.add(apply(filter, item, rest, kwargs));
        return out;
    }

    private static Object minMax(Object v, Params p, boolean max) {
        boolean cs = JinjaValue.truthy(p.get("case_sensitive", 0, false));
        Object attribute = p.get("attribute", 1, null);
        Object best = JinjaValue.UNDEFINED;
        Object bestKey = null;
        for (Object item : seq(v)) {
            Object k = key(item, attribute, cs);
            if (best == JinjaValue.UNDEFINED || (max ? JinjaValue.compare(k, bestKey) > 0 : JinjaValue.compare(k, bestKey) < 0)) {
                best = item;
                bestKey = k;
            }
        }
        return best;
    }

    /** select/reject (test on item) and selectattr/rejectattr (test on an attribute path). */
    private static Object select(Object v, List<Object> args, boolean byAttr, boolean keep) {
        int at = 0;
        String path = null;
        if (byAttr) {
            if (args.isEmpty()) throw new TemplateException("selectattr/rejectattr require an attribute name");
            path = JinjaValue.str(args.get(at++));
        }
        String test = at < args.size() ? JinjaValue.str(args.get(at++)) : null;
        List<Object> testArgs = new ArrayList<>(args.subList(at, args.size()));
        List<Object> out = new ArrayList<>();
        for (Object item : seq(v)) {
            Object subject = path == null ? item : attrPath(item, path);
            boolean ok = test == null ? JinjaValue.truthy(subject) : JinjaRuntime.test(test, subject, testArgs);
            if (ok == keep) out.add(item);
        }
        return out;
    }

    private static Object reverse(Object v) {
        if (v instanceof String) return JinjaValue.slice(v, null, null, -1L);
        List<Object> items = seq(v);
        Collections.reverse(items);
        return items;
    }

    /** Jinja2 {@code round}: 'common' rounds half to even on the exact binary value (Python round()). */
    private static Object round(Object v, Params p) {
        long precision = p.integer("precision", 0, 0);
        String method = JinjaValue.str(p.get("method", 1, "common"));
        if (!"common".equals(method) && !"ceil".equals(method) && !"floor".equals(method)) {
            throw new TemplateException("round method must be 'common', 'ceil' or 'floor'");
        }
        if (v instanceof Long && "common".equals(method) && precision >= 0) return v;
        double d = JinjaValue.toDouble(v, "round");
        if (Double.isNaN(d) || Double.isInfinite(d)) return d;
        RoundingMode mode = "common".equals(method) ? RoundingMode.HALF_EVEN
                : "ceil".equals(method) ? RoundingMode.CEILING : RoundingMode.FLOOR;
        return new BigDecimal(d).setScale((int) precision, mode).doubleValue();
    }

    private static Object sort(Object v, Params p) {
        boolean reverse = JinjaValue.truthy(p.get("reverse", 0, false));
        boolean cs = JinjaValue.truthy(p.get("case_sensitive", 1, false));
        Object attribute = p.get("attribute", 2, null);
        List<Object> items = seq(v);
        Comparator<Object> cmp = (a, b) -> JinjaValue.compare(key(a, attribute, cs), key(b, attribute, cs));
        items.sort(reverse ? cmp.reversed() : cmp);
        return items;
    }

    private static Object sum(Object v, Params p) {
        Object attribute = p.get("attribute", 0, null);
        Object acc = p.get("start", 1, 0L);
        for (Object item : seq(v)) acc = JinjaValue.binary("+", acc, key(item, attribute, true));
        return acc;
    }

    /** Jinja2 {@code title}: words split on {@code [-\s({\[<]+}, first character upper, rest lower. */
    private static String title(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        Matcher m = TITLE_SPLIT.matcher(s);
        int last = 0;
        while (m.find()) {
            sb.append(JinjaMethods.capitalize(s.substring(last, m.start()))).append(s, m.start(), m.end());
            last = m.end();
        }
        return sb.append(JinjaMethods.capitalize(s.substring(last))).toString();
    }

    private static Object strip(Object v, Params p, boolean left, boolean right) {
        p.only("chars");
        Object chars = p.get("chars", 0, null);
        return JinjaMethods.strip(JinjaValue.str(v), chars == null || chars == JinjaValue.UNDEFINED ? null
                : requireString(chars, p.fn), left, right);
    }

    private static Object unique(Object v, Params p) {
        boolean cs = JinjaValue.truthy(p.get("case_sensitive", 0, false));
        Object attribute = p.get("attribute", 1, null);
        List<Object> out = new ArrayList<>();
        List<Object> seen = new ArrayList<>();
        for (Object item : seq(v)) {
            Object k = key(item, attribute, cs);
            boolean dup = false;
            for (Object s : seen) {
                if (JinjaValue.pyEquals(s, k)) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                seen.add(k);
                out.add(item);
            }
        }
        return out;
    }

    /**
     * HF transformers {@code tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)}
     * (transformers/utils/chat_template_utils.py), serialized like Python {@code json.dumps}.
     */
    private static Object tojson(Object v, Params p) {
        boolean ascii = JinjaValue.truthy(p.get("ensure_ascii", 0, false));
        Object indent = p.get("indent", 1, null);
        Object separators = p.get("separators", 2, null);
        boolean sortKeys = JinjaValue.truthy(p.get("sort_keys", 3, false));
        String ind = null;
        if (indent instanceof String) {
            ind = (String) indent;
        } else if (indent != null && indent != JinjaValue.UNDEFINED) {
            long n = JinjaValue.toLong(indent, "tojson indent");
            StringBuilder sb = new StringBuilder();
            for (long i = 0; i < n; i++) sb.append(' ');
            ind = sb.toString();
        }
        String itemSep = ind == null ? ", " : ",";
        String keySep = ": ";
        if (separators != null && separators != JinjaValue.UNDEFINED) {
            if (!(separators instanceof List) || ((List<?>) separators).size() != 2) {
                throw new TemplateException("tojson separators must be an (item_separator, key_separator) pair");
            }
            itemSep = JinjaValue.str(((List<?>) separators).get(0));
            keySep = JinjaValue.str(((List<?>) separators).get(1));
        }
        StringBuilder out = new StringBuilder();
        Json.write(out, v, new Json.Style(ascii, ind, itemSep, keySep, sortKeys), 0);
        return out.toString();
    }

    // ================================================================= JSON

    /** Python-{@code json}-compatible serializer and parser over the template value model. */
    static final class Json {

        /** Immutable {@code json.dumps} options. */
        static final class Style {
            final boolean ensureAscii;
            final String indent;
            final String itemSep;
            final String keySep;
            final boolean sortKeys;

            Style(boolean ensureAscii, String indent, String itemSep, String keySep, boolean sortKeys) {
                this.ensureAscii = ensureAscii;
                this.indent = indent;
                this.itemSep = itemSep;
                this.keySep = keySep;
                this.sortKeys = sortKeys;
            }
        }

        private Json() {}

        static void write(StringBuilder out, Object v, Style st, int depth) {
            if (v == null || v == JinjaValue.UNDEFINED) out.append("null");
            else if (v instanceof Boolean) out.append((Boolean) v ? "true" : "false");
            else if (v instanceof Long) out.append(v);
            else if (v instanceof Double) out.append(number((Double) v));
            else if (v instanceof String) string(out, (String) v, st.ensureAscii);
            else if (v instanceof List) writeList(out, (List<?>) v, st, depth);
            else if (v instanceof Map) writeMap(out, (Map<?, ?>) v, st, depth);
            else throw new TemplateException("Object of type " + JinjaValue.typeName(v) + " is not JSON serializable");
        }

        private static String number(double d) {
            if (Double.isNaN(d)) return "NaN";
            if (Double.isInfinite(d)) return d > 0 ? "Infinity" : "-Infinity";
            return JinjaValue.floatRepr(d);
        }

        private static void newline(StringBuilder out, Style st, int depth) {
            if (st.indent == null) return;
            out.append('\n');
            for (int i = 0; i < depth; i++) out.append(st.indent);
        }

        private static void writeList(StringBuilder out, List<?> list, Style st, int depth) {
            if (list.isEmpty()) {
                out.append("[]");
                return;
            }
            out.append('[');
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) out.append(st.itemSep);
                newline(out, st, depth + 1);
                write(out, list.get(i), st, depth + 1);
            }
            newline(out, st, depth);
            out.append(']');
        }

        private static void writeMap(StringBuilder out, Map<?, ?> map, Style st, int depth) {
            if (map.isEmpty()) {
                out.append("{}");
                return;
            }
            List<Map.Entry<?, ?>> entries = new ArrayList<>(map.entrySet());
            if (st.sortKeys) entries.sort((a, b) -> JinjaValue.compare(a.getKey(), b.getKey()));
            out.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : entries) {
                if (!first) out.append(st.itemSep);
                first = false;
                newline(out, st, depth + 1);
                string(out, key(e.getKey()), st.ensureAscii);
                out.append(st.keySep);
                write(out, e.getValue(), st, depth + 1);
            }
            newline(out, st, depth);
            out.append('}');
        }

        /** Python json key coercion: str as-is, numbers and constants by their JSON text. */
        private static String key(Object k) {
            if (k instanceof String) return (String) k;
            if (k == null) return "null";
            if (k instanceof Boolean) return (Boolean) k ? "true" : "false";
            if (k instanceof Long) return k.toString();
            if (k instanceof Double) return number((Double) k);
            throw new TemplateException("keys must be str, int, float, bool or None, not " + JinjaValue.typeName(k));
        }

        private static void string(StringBuilder out, String s, boolean ascii) {
            out.append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"': out.append("\\\""); break;
                    case '\\': out.append("\\\\"); break;
                    case '\n': out.append("\\n"); break;
                    case '\r': out.append("\\r"); break;
                    case '\t': out.append("\\t"); break;
                    case '\b': out.append("\\b"); break;
                    case '\f': out.append("\\f"); break;
                    default:
                        if (c < 0x20 || (ascii && c > 0x7e)) out.append(String.format("\\u%04x", (int) c));
                        else out.append(c);
                }
            }
            out.append('"');
        }

        /** Parses JSON text (plus Python's NaN/Infinity) into lists, ordered maps, Long, Double, String, Boolean, null. */
        static Object parse(String text) {
            Parser p = new Parser(text);
            p.ws();
            Object v = p.value();
            p.ws();
            if (p.pos != text.length()) throw p.error("extra data");
            return v;
        }

        private static final class Parser {
            final String s;
            int pos;

            Parser(String s) { this.s = s; }

            TemplateException error(String msg) {
                return new TemplateException("from_json: " + msg + " at position " + pos);
            }

            void ws() {
                while (pos < s.length() && " \t\r\n".indexOf(s.charAt(pos)) >= 0) pos++;
            }

            boolean eat(String lit) {
                if (s.startsWith(lit, pos)) {
                    pos += lit.length();
                    return true;
                }
                return false;
            }

            Object value() {
                if (pos >= s.length()) throw error("unexpected end of input");
                char c = s.charAt(pos);
                if (c == '{') return object();
                if (c == '[') return array();
                if (c == '"') return str();
                if (eat("true")) return true;
                if (eat("false")) return false;
                if (eat("null")) return null;
                if (eat("NaN")) return Double.NaN;
                if (eat("Infinity")) return Double.POSITIVE_INFINITY;
                if (eat("-Infinity")) return Double.NEGATIVE_INFINITY;
                if (c == '-' || (c >= '0' && c <= '9')) return number();
                throw error("unexpected character '" + c + "'");
            }

            Object object() {
                Map<Object, Object> m = new LinkedHashMap<>();
                pos++;
                ws();
                if (eat("}")) return m;
                while (true) {
                    ws();
                    if (pos >= s.length() || s.charAt(pos) != '"') throw error("expected property name");
                    String k = str();
                    ws();
                    if (!eat(":")) throw error("expected ':'");
                    ws();
                    m.put(k, value());
                    ws();
                    if (eat("}")) return m;
                    if (!eat(",")) throw error("expected ',' or '}'");
                }
            }

            Object array() {
                List<Object> l = new ArrayList<>();
                pos++;
                ws();
                if (eat("]")) return l;
                while (true) {
                    ws();
                    l.add(value());
                    ws();
                    if (eat("]")) return l;
                    if (!eat(",")) throw error("expected ',' or ']'");
                }
            }

            String str() {
                StringBuilder sb = new StringBuilder();
                pos++;
                while (true) {
                    if (pos >= s.length()) throw error("unterminated string");
                    char c = s.charAt(pos++);
                    if (c == '"') return sb.toString();
                    if (c < 0x20) throw error("invalid control character in string");
                    if (c != '\\') {
                        sb.append(c);
                        continue;
                    }
                    if (pos >= s.length()) throw error("unterminated escape");
                    char e = s.charAt(pos++);
                    switch (e) {
                        case '"': sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        case '/': sb.append('/'); break;
                        case 'b': sb.append('\b'); break;
                        case 'f': sb.append('\f'); break;
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case 'u':
                            if (pos + 4 > s.length()) throw error("truncated \\u escape");
                            try {
                                sb.append((char) Integer.parseInt(s.substring(pos, pos + 4), 16));
                            } catch (NumberFormatException ex) {
                                throw error("invalid \\u escape");
                            }
                            pos += 4;
                            break;
                        default: throw error("invalid escape '\\" + e + "'");
                    }
                }
            }

            Object number() {
                int start = pos;
                if (s.charAt(pos) == '-') pos++;
                while (pos < s.length() && Character.isDigit(s.charAt(pos))) pos++;
                boolean isFloat = false;
                if (pos < s.length() && s.charAt(pos) == '.') {
                    isFloat = true;
                    pos++;
                    while (pos < s.length() && Character.isDigit(s.charAt(pos))) pos++;
                }
                if (pos < s.length() && (s.charAt(pos) == 'e' || s.charAt(pos) == 'E')) {
                    isFloat = true;
                    pos++;
                    if (pos < s.length() && (s.charAt(pos) == '+' || s.charAt(pos) == '-')) pos++;
                    while (pos < s.length() && Character.isDigit(s.charAt(pos))) pos++;
                }
                String t = s.substring(start, pos);
                try {
                    if (!isFloat) {
                        try {
                            return Long.parseLong(t);
                        } catch (NumberFormatException overflow) {
                            return Double.parseDouble(t);
                        }
                    }
                    return Double.parseDouble(t);
                } catch (NumberFormatException ex) {
                    pos = start;
                    throw error("invalid number '" + t + "'");
                }
            }
        }
    }
}
