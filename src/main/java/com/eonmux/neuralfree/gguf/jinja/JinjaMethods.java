package com.eonmux.neuralfree.gguf.jinja;

import com.eonmux.neuralfree.gguf.jinja.JinjaRuntime.TemplateException;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.TextStyle;
import java.time.temporal.ChronoField;
import java.time.temporal.IsoFields;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Python methods callable on template values ({@code s.split()}, {@code l.append(x)},
 * {@code d.items()}), plus the string helpers shared with {@link JinjaFilters}: Python stripping and
 * splitting, {@code str.format}, {@code %}-formatting, the format-spec mini-language and
 * {@code strftime}. String positions are code-point indices, as in Python.
 */
final class JinjaMethods {

    private static final Set<String> STRING_METHODS = names("startswith", "endswith", "split", "rsplit",
            "splitlines", "strip", "lstrip", "rstrip", "trim", "upper", "lower", "title", "capitalize", "swapcase",
            "replace", "count", "find", "rfind", "index", "rindex", "format", "join", "isdigit", "isalpha",
            "isalnum", "isspace", "islower", "isupper", "removeprefix", "removesuffix", "center", "ljust", "rjust",
            "zfill");
    private static final Set<String> TUPLE_METHODS = names("index", "count");
    private static final Set<String> LIST_METHODS = names("append", "extend", "pop", "insert", "index", "count",
            "remove", "reverse", "copy", "clear");
    private static final Set<String> DICT_METHODS = names("get", "keys", "values", "items", "update", "pop",
            "setdefault", "copy", "clear");

    private JinjaMethods() {}

    private static Set<String> names(String... n) { return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(n))); }

    /** The bound method {@code target.name}, or null when the type has no such method. */
    static JinjaValue.Callable bind(Object target, String name) {
        boolean known;
        if (target instanceof String) known = STRING_METHODS.contains(name);
        else if (target instanceof JinjaValue.Tuple) known = TUPLE_METHODS.contains(name);
        else if (target instanceof List) known = LIST_METHODS.contains(name);
        else if (target instanceof JinjaValue.Namespace) known = false;
        else if (target instanceof Map) known = DICT_METHODS.contains(name);
        else known = false;
        return known ? new BoundMethod(target, name) : null;
    }

    private static final class BoundMethod implements JinjaValue.Callable {
        private final Object target;
        private final String name;

        BoundMethod(Object target, String name) {
            this.target = target;
            this.name = name;
        }

        @Override public Object call(List<Object> args, Map<String, Object> kwargs, JinjaRuntime.Scope caller) {
            if (target instanceof String) return callString((String) target, name, new Params(name, args, kwargs));
            if (target instanceof List) return callList((List<?>) target, name, new Params(name, args, kwargs));
            return callDict((Map<?, ?>) target, name, new Params(name, args, kwargs));
        }

        @Override public String toString() { return "method " + JinjaValue.typeName(target) + "." + name; }
    }

    /** Positional-or-keyword argument access for one call. */
    static final class Params {
        final String fn;
        final List<Object> args;
        final Map<String, Object> kwargs;

        Params(String fn, List<Object> args, Map<String, Object> kwargs) {
            this.fn = fn;
            this.args = args;
            this.kwargs = kwargs;
        }

        /** Argument by keyword or position; {@code dflt} when absent. */
        Object get(String key, int pos, Object dflt) {
            if (kwargs.containsKey(key)) return kwargs.get(key);
            return pos >= 0 && pos < args.size() ? args.get(pos) : dflt;
        }

        /**
         * Rejects keywords outside {@code names} and more positionals than {@code names} declares, so
         * misspelled arguments fail loudly instead of being ignored.
         */
        Params only(String... names) {
            if (args.size() > names.length) {
                throw new TemplateException(fn + "() takes at most " + names.length + " argument(s), got " + args.size());
            }
            List<String> allowed = Arrays.asList(names);
            for (String k : kwargs.keySet()) {
                if (!allowed.contains(k)) throw new TemplateException(fn + "() got an unexpected keyword argument '" + k + "'");
            }
            return this;
        }

        Object require(String key, int pos) {
            Object v = get(key, pos, JinjaValue.UNDEFINED);
            if (v == JinjaValue.UNDEFINED) throw new TemplateException(fn + "() missing required argument '" + key + "'");
            return v;
        }

        String string(String key, int pos) {
            Object v = require(key, pos);
            if (!(v instanceof String)) throw new TemplateException(fn + "() argument '" + key + "' must be str, not " + JinjaValue.typeName(v));
            return (String) v;
        }

        long integer(String key, int pos, long dflt) {
            Object v = get(key, pos, null);
            return v == null || v == JinjaValue.UNDEFINED ? dflt : JinjaValue.toLong(v, fn + "()");
        }
    }

    // ================================================================= str

    private static Object callString(String s, String m, Params p) {
        switch (m) {
            case "startswith": case "endswith": return affix(s, p, "startswith".equals(m));
            case "split": return split(s, optString(p.get("sep", 0, null), m), p.integer("maxsplit", 1, -1));
            case "rsplit": return rsplit(s, optString(p.get("sep", 0, null), m), p.integer("maxsplit", 1, -1));
            case "splitlines": return splitlines(s, JinjaValue.truthy(p.get("keepends", 0, false)));
            case "strip": case "trim": return strip(s, optString(p.get("chars", 0, null), m), true, true);
            case "lstrip": return strip(s, optString(p.get("chars", 0, null), m), true, false);
            case "rstrip": return strip(s, optString(p.get("chars", 0, null), m), false, true);
            case "upper": return s.toUpperCase(Locale.ROOT);
            case "lower": return s.toLowerCase(Locale.ROOT);
            case "title": return title(s);
            case "capitalize": return capitalize(s);
            case "swapcase": return swapcase(s);
            case "replace": return replace(s, p.string("old", 0), p.string("new", 1), p.integer("count", 2, -1));
            case "count": return count(s, p);
            case "find": return find(s, p, false, false);
            case "rfind": return find(s, p, true, false);
            case "index": return find(s, p, false, true);
            case "rindex": return find(s, p, true, true);
            case "format": return format(s, p.args, p.kwargs);
            case "join": return join(s, JinjaValue.iterate(p.require("iterable", 0)));
            case "isdigit": return allCodePoints(s, Character::isDigit);
            case "isalpha": return allCodePoints(s, Character::isLetter);
            case "isalnum": return allCodePoints(s, Character::isLetterOrDigit);
            case "isspace": return allCodePoints(s, JinjaValue::isPySpace);
            case "islower": return isCased(s, false);
            case "isupper": return isCased(s, true);
            case "removeprefix": { String a = p.string("prefix", 0); return s.startsWith(a) ? s.substring(a.length()) : s; }
            case "removesuffix": { String a = p.string("suffix", 0); return !a.isEmpty() && s.endsWith(a) ? s.substring(0, s.length() - a.length()) : s; }
            case "center": return pad(s, (int) p.integer("width", 0, 0), fillChar(p), '^');
            case "ljust": return pad(s, (int) p.integer("width", 0, 0), fillChar(p), '<');
            case "rjust": return pad(s, (int) p.integer("width", 0, 0), fillChar(p), '>');
            case "zfill": return zfill(s, (int) p.integer("width", 0, 0));
            default: throw new TemplateException("str has no method '" + m + "'");
        }
    }

    private static String optString(Object v, String fn) {
        if (v == null || v == JinjaValue.UNDEFINED) return null;
        if (!(v instanceof String)) throw new TemplateException(fn + "() argument must be str or None, not " + JinjaValue.typeName(v));
        return (String) v;
    }

    private static int fillChar(Params p) {
        String f = optString(p.get("fillchar", 1, null), p.fn);
        if (f == null) return ' ';
        if (f.codePointCount(0, f.length()) != 1) throw new TemplateException(p.fn + "() fill character must be exactly one character");
        return f.codePointAt(0);
    }

    private interface CodePointTest { boolean test(int cp); }

    private static boolean allCodePoints(String s, CodePointTest t) {
        if (s.isEmpty()) return false;
        return s.codePoints().allMatch(t::test);
    }

    /** Python {@code islower}/{@code isupper}: at least one cased character and none of the other case. */
    static boolean isCased(String s, boolean upper) {
        boolean found = false;
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            boolean lo = Character.isLowerCase(cp), up = Character.isUpperCase(cp) || Character.isTitleCase(cp);
            if (upper ? lo : up) return false;
            if (upper ? up : lo) found = true;
        }
        return found;
    }

    private static boolean affix(String s, Params p, boolean prefix) {
        Object a = p.require(prefix ? "prefix" : "suffix", 0);
        String sub = slice(s, p.get("start", 1, null), p.get("end", 2, null));
        List<?> options = a instanceof List ? (List<?>) a : Collections.singletonList(a);
        for (Object o : options) {
            if (!(o instanceof String)) throw new TemplateException(p.fn + "() argument must be str or a tuple of str");
            if (prefix ? sub.startsWith((String) o) : sub.endsWith((String) o)) return true;
        }
        return false;
    }

    /** Substring by Python start/end code-point bounds. */
    private static String slice(String s, Object start, Object end) {
        if ((start == null || start == JinjaValue.UNDEFINED) && (end == null || end == JinjaValue.UNDEFINED)) return s;
        return (String) JinjaValue.slice(s, start, end, null);
    }

    /** Python {@code str.strip}: whitespace when {@code chars} is null, otherwise any of {@code chars}. */
    static String strip(String s, String chars, boolean left, boolean right) {
        int b = 0, e = s.length();
        if (left) {
            while (b < e) {
                int cp = s.codePointAt(b);
                if (!stripped(cp, chars)) break;
                b += Character.charCount(cp);
            }
        }
        if (right) {
            while (e > b) {
                int cp = s.codePointBefore(e);
                if (!stripped(cp, chars)) break;
                e -= Character.charCount(cp);
            }
        }
        return s.substring(b, e);
    }

    private static boolean stripped(int cp, String chars) {
        return chars == null ? JinjaValue.isPySpace(cp) : chars.indexOf(new String(Character.toChars(cp))) >= 0;
    }

    /** Python {@code str.split}. */
    static List<Object> split(String s, String sep, long maxsplit) {
        List<Object> out = new ArrayList<>();
        if (sep == null) {
            int i = 0, n = s.length();
            while (true) {
                while (i < n && JinjaValue.isPySpace(s.codePointAt(i))) i += Character.charCount(s.codePointAt(i));
                if (i >= n) break;
                if (maxsplit >= 0 && out.size() == maxsplit) {
                    out.add(s.substring(i));
                    break;
                }
                int j = i;
                while (j < n && !JinjaValue.isPySpace(s.codePointAt(j))) j += Character.charCount(s.codePointAt(j));
                out.add(s.substring(i, j));
                i = j;
            }
            return out;
        }
        if (sep.isEmpty()) throw new TemplateException("split(): empty separator");
        int from = 0;
        while (maxsplit < 0 || out.size() < maxsplit) {
            int idx = s.indexOf(sep, from);
            if (idx < 0) break;
            out.add(s.substring(from, idx));
            from = idx + sep.length();
        }
        out.add(s.substring(from));
        return out;
    }

    /** Python {@code str.rsplit}. */
    static List<Object> rsplit(String s, String sep, long maxsplit) {
        if (maxsplit < 0) return split(s, sep, -1);
        List<Object> out = new ArrayList<>();
        if (sep == null) {
            int j = s.length();
            while (true) {
                while (j > 0 && JinjaValue.isPySpace(s.codePointBefore(j))) j -= Character.charCount(s.codePointBefore(j));
                if (j <= 0) break;
                if (out.size() == maxsplit) {
                    out.add(s.substring(0, j));
                    break;
                }
                int i = j;
                while (i > 0 && !JinjaValue.isPySpace(s.codePointBefore(i))) i -= Character.charCount(s.codePointBefore(i));
                out.add(s.substring(i, j));
                j = i;
            }
        } else {
            if (sep.isEmpty()) throw new TemplateException("rsplit(): empty separator");
            int end = s.length();
            while (out.size() < maxsplit) {
                int idx = s.lastIndexOf(sep, end - sep.length());
                if (idx < 0) break;
                out.add(s.substring(idx + sep.length(), end));
                end = idx;
            }
            out.add(s.substring(0, end));
        }
        Collections.reverse(out);
        return out;
    }

    /** Python {@code str.splitlines} with its full set of line boundaries. */
    static List<Object> splitlines(String s, boolean keepends) {
        List<Object> out = new ArrayList<>();
        int start = 0, i = 0, n = s.length();
        while (i < n) {
            char c = s.charAt(i);
            int brk = 0;
            if (c == '\r') brk = i + 1 < n && s.charAt(i + 1) == '\n' ? 2 : 1;
            else if (c == '\n' || c == '\u000b' || c == '\u000c' || (c >= '\u001c' && c <= '\u001e')
                    || c == '\u0085' || c == '\u2028' || c == '\u2029') brk = 1;
            if (brk == 0) {
                i++;
                continue;
            }
            out.add(s.substring(start, keepends ? i + brk : i));
            i += brk;
            start = i;
        }
        if (start < n) out.add(s.substring(start));
        return out;
    }

    /** Python {@code str.title}: upper-case after an uncased character, lower-case otherwise. */
    static String title(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        boolean prevCased = false;
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            boolean cased = Character.isUpperCase(cp) || Character.isLowerCase(cp) || Character.isTitleCase(cp);
            sb.appendCodePoint(prevCased ? Character.toLowerCase(cp) : Character.toTitleCase(cp));
            prevCased = cased;
        }
        return sb.toString();
    }

    /** Python {@code str.capitalize}. */
    static String capitalize(String s) {
        if (s.isEmpty()) return s;
        int cp = s.codePointAt(0);
        return new String(Character.toChars(Character.toTitleCase(cp)))
                + s.substring(Character.charCount(cp)).toLowerCase(Locale.ROOT);
    }

    private static String swapcase(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        s.codePoints().forEach(cp -> sb.appendCodePoint(Character.isUpperCase(cp) ? Character.toLowerCase(cp)
                : Character.isLowerCase(cp) ? Character.toUpperCase(cp) : cp));
        return sb.toString();
    }

    /** Python {@code str.replace}; a negative count replaces all occurrences. */
    static String replace(String s, String old, String rep, long count) {
        if (count == 0) return s;
        StringBuilder sb = new StringBuilder();
        long done = 0;
        if (old.isEmpty()) {
            int i = 0;
            while (i <= s.length()) {
                if (count < 0 || done < count) {
                    sb.append(rep);
                    done++;
                }
                if (i == s.length()) break;
                int cp = s.codePointAt(i);
                sb.appendCodePoint(cp);
                i += Character.charCount(cp);
            }
            return sb.toString();
        }
        int from = 0;
        while (count < 0 || done < count) {
            int idx = s.indexOf(old, from);
            if (idx < 0) break;
            sb.append(s, from, idx).append(rep);
            from = idx + old.length();
            done++;
        }
        return sb.append(s, from, s.length()).toString();
    }

    private static long count(String s, Params p) {
        String sub = p.string("sub", 0);
        String hay = slice(s, p.get("start", 1, null), p.get("end", 2, null));
        if (sub.isEmpty()) return hay.codePointCount(0, hay.length()) + 1L;
        long n = 0;
        for (int idx = hay.indexOf(sub); idx >= 0; idx = hay.indexOf(sub, idx + sub.length())) n++;
        return n;
    }

    /** find/rfind/index/rindex with Python start/end bounds; results are code-point indices. */
    private static long find(String s, Params p, boolean reverse, boolean raise) {
        String sub = p.string("sub", 0);
        long len = s.codePointCount(0, s.length());
        long start = bound(p.get("start", 1, null), len, 0);
        long end = bound(p.get("end", 2, null), len, len);
        long result = -1;
        if (start <= end) {
            String hay = s.substring(s.offsetByCodePoints(0, (int) start), s.offsetByCodePoints(0, (int) end));
            int idx = reverse ? hay.lastIndexOf(sub) : hay.indexOf(sub);
            if (idx >= 0) result = start + hay.codePointCount(0, idx);
        }
        if (result < 0 && raise) throw new TemplateException(p.fn + "(): substring not found");
        return result;
    }

    private static long bound(Object v, long len, long dflt) {
        if (v == null || v == JinjaValue.UNDEFINED) return dflt;
        long b = JinjaValue.toLong(v, "string index");
        if (b < 0) b = Math.max(0, b + len);
        return Math.min(b, len);
    }

    static String join(String sep, List<Object> items) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(sep);
            sb.append(JinjaValue.str(items.get(i)));
        }
        return sb.toString();
    }

    /** {@code str.center/ljust/rjust} padding; centring puts the odd fill character as CPython's {@code str.center} does. */
    private static String pad(String s, int width, int fill, char align) {
        int total = width - s.codePointCount(0, s.length());
        return total <= 0 ? s : fill(s, total, fill, align == '<' ? 0 : align == '>' ? total : total / 2 + (total & width & 1));
    }

    /** Format-spec alignment ({@code <}, {@code >}, {@code ^}); {@code ^} puts the odd fill character on the right. */
    private static String align(String s, int width, int fill, char align) {
        int total = width - s.codePointCount(0, s.length());
        return total <= 0 ? s : fill(s, total, fill, align == '<' ? 0 : align == '>' ? total : total / 2);
    }

    private static String fill(String s, int total, int fill, int left) {
        String f = new String(Character.toChars(fill));
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < left; i++) sb.append(f);
        sb.append(s);
        for (int i = 0; i < total - left; i++) sb.append(f);
        return sb.toString();
    }

    private static String zfill(String s, int width) {
        int len = s.codePointCount(0, s.length());
        if (width <= len) return s;
        boolean signed = !s.isEmpty() && (s.charAt(0) == '+' || s.charAt(0) == '-');
        StringBuilder sb = new StringBuilder();
        if (signed) sb.append(s.charAt(0));
        for (int i = 0; i < width - len; i++) sb.append('0');
        return sb.append(signed ? s.substring(1) : s).toString();
    }

    // ================================================================= list and dict

    @SuppressWarnings("unchecked")
    private static Object callList(List<?> list, String m, Params p) {
        List<Object> l = (List<Object>) list;
        switch (m) {
            case "append": l.add(p.require("object", 0)); return null;
            case "extend": l.addAll(JinjaValue.iterate(p.require("iterable", 0))); return null;
            case "insert": {
                long idx = p.integer("index", 0, 0);
                if (idx < 0) idx = Math.max(0, idx + l.size());
                l.add((int) Math.min(idx, l.size()), p.require("object", 1));
                return null;
            }
            case "pop": {
                if (l.isEmpty()) throw new TemplateException("pop from empty list");
                long idx = p.integer("index", 0, -1);
                if (idx < 0) idx += l.size();
                if (idx < 0 || idx >= l.size()) throw new TemplateException("pop index out of range");
                return l.remove((int) idx);
            }
            case "index": {
                Object x = p.require("value", 0);
                for (int i = 0; i < l.size(); i++) if (JinjaValue.pyEquals(l.get(i), x)) return (long) i;
                throw new TemplateException("list.index(x): x not in list");
            }
            case "count": {
                Object x = p.require("value", 0);
                long n = 0;
                for (Object o : l) if (JinjaValue.pyEquals(o, x)) n++;
                return n;
            }
            case "remove": {
                Object x = p.require("value", 0);
                for (int i = 0; i < l.size(); i++) {
                    if (JinjaValue.pyEquals(l.get(i), x)) {
                        l.remove(i);
                        return null;
                    }
                }
                throw new TemplateException("list.remove(x): x not in list");
            }
            case "reverse": Collections.reverse(l); return null;
            case "copy": return new ArrayList<>(l);
            case "clear": l.clear(); return null;
            default: throw new TemplateException(JinjaValue.typeName(list) + " has no method '" + m + "'");
        }
    }

    @SuppressWarnings("unchecked")
    private static Object callDict(Map<?, ?> map, String m, Params p) {
        Map<Object, Object> d = (Map<Object, Object>) map;
        switch (m) {
            case "get": return JinjaValue.mapGet(d, p.require("key", 0), p.get("default", 1, null));
            case "keys": return new ArrayList<>(d.keySet());
            case "values": return new ArrayList<>(d.values());
            case "items": return JinjaValue.items(d);
            case "copy": return new LinkedHashMap<>(d);
            case "clear": d.clear(); return null;
            case "update": {
                Object other = p.get("other", 0, null);
                if (other instanceof Map) d.putAll((Map<?, ?>) other);
                else if (other instanceof List) {
                    for (Object pair : (List<?>) other) {
                        List<?> kv = (List<?>) pair;
                        d.put(kv.get(0), kv.get(1));
                    }
                } else if (other != null && other != JinjaValue.UNDEFINED) {
                    throw new TemplateException("dict.update() argument must be a mapping or pairs, got " + JinjaValue.typeName(other));
                }
                d.putAll(p.kwargs);
                return null;
            }
            case "pop": {
                Object key = p.require("key", 0);
                if (JinjaValue.mapContainsKey(d, key)) {
                    Object v = JinjaValue.mapGet(d, key, null);
                    d.keySet().removeIf(k -> JinjaValue.pyEquals(k, key));
                    return v;
                }
                Object dflt = p.get("default", 1, JinjaValue.UNDEFINED);
                if (dflt == JinjaValue.UNDEFINED) throw new TemplateException("dict.pop(): key " + JinjaValue.repr(key) + " not found");
                return dflt;
            }
            case "setdefault": {
                Object key = p.require("key", 0);
                if (JinjaValue.mapContainsKey(d, key)) return JinjaValue.mapGet(d, key, null);
                Object dflt = p.get("default", 1, null);
                d.put(key, dflt);
                return dflt;
            }
            default: throw new TemplateException("dict has no method '" + m + "'");
        }
    }

    // ================================================================= str.format and %-formatting

    private static final Pattern FIELD_NAME = Pattern.compile("([^.\\[]*)((?:\\.[^.\\[]+|\\[[^\\]]*\\])*)");
    private static final Pattern FIELD_PART = Pattern.compile("\\.([^.\\[]+)|\\[([^\\]]*)\\]");

    /** Python {@code str.format}: {@code {}}, {@code {0}}, {@code {name.attr[key]!r:spec}}, {@code {{ }}}. */
    static String format(String fmt, List<Object> args, Map<String, Object> kwargs) {
        StringBuilder out = new StringBuilder();
        int auto = 0;
        for (int i = 0; i < fmt.length(); i++) {
            char c = fmt.charAt(i);
            if (c == '}') {
                if (i + 1 < fmt.length() && fmt.charAt(i + 1) == '}') i++;
                else throw new TemplateException("format(): single '}' encountered in format string");
                out.append('}');
                continue;
            }
            if (c != '{') {
                out.append(c);
                continue;
            }
            if (i + 1 < fmt.length() && fmt.charAt(i + 1) == '{') {
                out.append('{');
                i++;
                continue;
            }
            int close = fmt.indexOf('}', i);
            if (close < 0) throw new TemplateException("format(): single '{' encountered in format string");
            String field = fmt.substring(i + 1, close);
            i = close;
            String spec = "";
            int colon = field.indexOf(':');
            if (colon >= 0) {
                spec = field.substring(colon + 1);
                field = field.substring(0, colon);
            }
            char conv = 0;
            int bang = field.indexOf('!');
            if (bang >= 0) {
                if (bang + 2 != field.length()) throw new TemplateException("format(): invalid conversion in '{" + field + "}'");
                conv = field.charAt(bang + 1);
                field = field.substring(0, bang);
            }
            Matcher fm = FIELD_NAME.matcher(field);
            if (!fm.matches()) throw new TemplateException("format(): invalid field '" + field + "'");
            String head = fm.group(1);
            Object v;
            if (head.isEmpty()) {
                if (auto >= args.size()) throw new TemplateException("format(): replacement index " + auto + " out of range");
                v = args.get(auto++);
            } else if (head.chars().allMatch(Character::isDigit)) {
                int idx = Integer.parseInt(head);
                if (idx >= args.size()) throw new TemplateException("format(): replacement index " + idx + " out of range");
                v = args.get(idx);
            } else {
                if (!kwargs.containsKey(head)) throw new TemplateException("format(): missing keyword argument '" + head + "'");
                v = kwargs.get(head);
            }
            Matcher pm = FIELD_PART.matcher(fm.group(2));
            while (pm.find()) {
                if (pm.group(1) != null) v = JinjaValue.getAttr(v, pm.group(1));
                else v = JinjaValue.getItem(v, pm.group(2).chars().allMatch(Character::isDigit) && !pm.group(2).isEmpty()
                        ? (Object) Long.parseLong(pm.group(2)) : pm.group(2));
            }
            if (conv == 'r' || conv == 'a') v = JinjaValue.repr(v);
            else if (conv == 's') v = JinjaValue.str(v);
            else if (conv != 0) throw new TemplateException("format(): unknown conversion '!" + conv + "'");
            out.append(formatSpec(v, spec));
        }
        return out.toString();
    }

    private static final Pattern PERCENT = Pattern.compile("%(?:\\(([^)]*)\\))?([-+ #0]*)(\\*|\\d+)?(?:\\.(\\*|\\d+))?([diouxXeEfFgGcrsa%])");

    /** Python {@code fmt % values}: a tuple supplies positionals, a dict supplies {@code %(name)s} keys. */
    static String percentFormat(String fmt, Object values) {
        List<?> positional = values instanceof JinjaValue.Tuple ? (List<?>) values : Collections.singletonList(values);
        Map<?, ?> mapping = JinjaValue.isDict(values) ? (Map<?, ?>) values : null;
        Matcher m = PERCENT.matcher(fmt);
        StringBuilder out = new StringBuilder();
        int last = 0, next = 0;
        while (m.find()) {
            out.append(fmt, last, m.start());
            last = m.end();
            char type = m.group(5).charAt(0);
            if (type == '%') {
                out.append('%');
                continue;
            }
            Object v;
            if (m.group(1) != null) {
                if (mapping == null) throw new TemplateException("format requires a mapping");
                v = JinjaValue.mapGet(mapping, m.group(1), JinjaValue.UNDEFINED);
                if (v == JinjaValue.UNDEFINED) throw new TemplateException("format key '" + m.group(1) + "' not found");
            } else {
                if (next >= positional.size()) throw new TemplateException("not enough arguments for format string");
                v = positional.get(next++);
            }
            if ("*".equals(m.group(3)) || "*".equals(m.group(4))) throw new TemplateException("'*' width/precision is not supported");
            String flags = m.group(2);
            StringBuilder spec = new StringBuilder();
            boolean left = flags.indexOf('-') >= 0, zero = flags.indexOf('0') >= 0 && !left;
            spec.append(left ? "<" : zero && type != 's' && type != 'r' && type != 'a' && type != 'c' ? "0=" : ">");
            if (flags.indexOf('+') >= 0) spec.append('+');
            else if (flags.indexOf(' ') >= 0) spec.append(' ');
            if (flags.indexOf('#') >= 0) spec.append('#');
            if (m.group(3) != null) spec.append(m.group(3));
            if (m.group(4) != null) spec.append('.').append(m.group(4));
            switch (type) {
                case 'r': case 'a': v = JinjaValue.repr(v); spec.append('s'); break;
                case 's': v = JinjaValue.str(v); spec.append('s'); break;
                case 'i': case 'u': case 'd':
                    v = v instanceof Double ? (long) (double) (Double) v : JinjaValue.toLong(v, "%d format");
                    spec.append('d');
                    break;
                case 'c': spec.append('c'); break;
                default: spec.append(type); break;
            }
            out.append(formatSpec(v, spec.toString()));
        }
        if (mapping == null && next < positional.size() && (values instanceof JinjaValue.Tuple || next == 0)) {
            throw new TemplateException("not all arguments converted during string formatting");
        }
        return out.append(fmt.substring(last)).toString();
    }

    private static final Pattern SPEC = Pattern.compile(
            "(?:(.)?([<>=^]))?([+\\- ])?(#)?(0)?(\\d+)?([,_])?(?:\\.(\\d+))?([bcdeEfFgGnosxX%])?");

    /** Python format-spec mini-language for str, int and float values. */
    static String formatSpec(Object v, String spec) {
        if (spec.isEmpty()) return JinjaValue.str(v);
        Matcher m = SPEC.matcher(spec);
        if (!m.matches()) throw new TemplateException("invalid format specifier '" + spec + "'");
        String fill = m.group(1);
        String align = m.group(2);
        String sign = m.group(3) == null ? "-" : m.group(3);
        boolean alt = m.group(4) != null, zero = m.group(5) != null;
        int width = m.group(6) == null ? 0 : Integer.parseInt(m.group(6));
        String grouping = m.group(7);
        Integer precision = m.group(8) == null ? null : Integer.parseInt(m.group(8));
        String type = m.group(9);
        int fillCp = fill != null ? fill.codePointAt(0) : zero && align == null ? '0' : ' ';
        if (zero && align == null) align = "=";
        boolean numeric = JinjaValue.isNumeric(v) && !"s".equals(type);
        if (!numeric) {
            if (type != null && !"s".equals(type) && !"c".equals(type)) {
                throw new TemplateException("unknown format code '" + type + "' for object of type '" + JinjaValue.typeName(v) + "'");
            }
            String s = JinjaValue.str(v);
            if (precision != null && s.codePointCount(0, s.length()) > precision) s = s.substring(0, s.offsetByCodePoints(0, precision));
            return align(s, width, fillCp, align == null ? '<' : align.charAt(0) == '=' ? '<' : align.charAt(0));
        }
        String body;
        boolean negative;
        String t = type == null ? (v instanceof Double ? "" : "d") : type;
        if ("c".equals(t)) {
            return align(new String(Character.toChars((int) JinjaValue.toLong(v, "format 'c'"))), width, fillCp,
                    align == null ? '<' : align.charAt(0));
        }
        if (t.length() == 1 && "dnbxXo".contains(t)) {
            if (v instanceof Double) throw new TemplateException("unknown format code '" + t + "' for object of type 'float'");
            long x = JinjaValue.toLong(v, "format");
            negative = x < 0;
            String mag = Long.toUnsignedString(negative ? -x : x, "b".equals(t) ? 2 : "o".equals(t) ? 8 : "xX".contains(t) ? 16 : 10);
            if ("X".equals(t)) mag = mag.toUpperCase(Locale.ROOT);
            if (grouping != null && ("d".equals(t) || "n".equals(t))) mag = group(mag, grouping.charAt(0));
            String prefix = alt && !"dn".contains(t) ? "0" + ("X".equals(t) ? "X" : t) : "";
            body = prefix + mag;
        } else {
            double x = JinjaValue.toDouble(v, "format");
            negative = x < 0 || (x == 0 && 1.0 / x < 0);
            body = formatFloat(Math.abs(x), t, precision, alt, grouping);
        }
        String signStr = negative ? "-" : "+".equals(sign) ? "+" : " ".equals(sign) ? " " : "";
        char a = align == null ? '>' : align.charAt(0);
        if (a == '=') {
            int len = signStr.length() + body.codePointCount(0, body.length());
            StringBuilder sb = new StringBuilder(signStr);
            for (int i = len; i < width; i++) sb.appendCodePoint(fillCp);
            return sb.append(body).toString();
        }
        return align(signStr + body, width, fillCp, a);
    }

    private static String formatFloat(double x, String t, Integer precision, boolean alt, String grouping) {
        if (Double.isNaN(x)) return "E".equals(t) || "F".equals(t) || "G".equals(t) ? "NAN" : "nan";
        if (Double.isInfinite(x)) return "E".equals(t) || "F".equals(t) || "G".equals(t) ? "INF" : "inf";
        String out;
        switch (t) {
            case "":
                out = precision == null ? JinjaValue.floatRepr(x) : general(x, precision, true, alt);
                break;
            case "f": case "F":
                out = new BigDecimal(x).setScale(precision == null ? 6 : precision, RoundingMode.HALF_EVEN).toPlainString();
                if (alt && precision != null && precision == 0) out += ".";
                break;
            case "%":
                out = new BigDecimal(x).movePointRight(2).setScale(precision == null ? 6 : precision, RoundingMode.HALF_EVEN).toPlainString() + "%";
                break;
            case "e": case "E":
                out = scientific(x, precision == null ? 6 : precision, alt);
                break;
            case "g": case "G": case "n":
                out = general(x, precision == null ? 6 : precision, false, alt);
                break;
            default:
                throw new TemplateException("unknown format code '" + t + "' for object of type 'float'");
        }
        if ("E".equals(t) || "G".equals(t) || "F".equals(t)) out = out.toUpperCase(Locale.ROOT);
        if (grouping != null) {
            int dot = out.indexOf('.');
            int end = dot >= 0 ? dot : out.indexOf('e') >= 0 ? out.indexOf('e') : out.length();
            out = group(out.substring(0, end), grouping.charAt(0)) + out.substring(end);
        }
        return out;
    }

    private static String scientific(double x, int precision, boolean alt) {
        if (x == 0) {
            StringBuilder z = new StringBuilder("0");
            if (precision > 0 || alt) z.append('.');
            for (int i = 0; i < precision; i++) z.append('0');
            return z.append("e+00").toString();
        }
        BigDecimal bd = new BigDecimal(x).round(new MathContext(precision + 1, RoundingMode.HALF_EVEN));
        int exp = bd.precision() - bd.scale() - 1;
        String digits = bd.unscaledValue().abs().toString();
        StringBuilder sb = new StringBuilder();
        sb.append(digits.charAt(0));
        if (precision > 0 || alt) sb.append('.');
        for (int i = 1; i <= precision; i++) sb.append(i < digits.length() ? digits.charAt(i) : '0');
        sb.append('e').append(exp < 0 ? '-' : '+');
        if (Math.abs(exp) < 10) sb.append('0');
        return sb.append(Math.abs(exp)).toString();
    }

    /** Python {@code g} formatting; {@code reprLike} keeps one fractional digit as the empty type does. */
    private static String general(double x, int precision, boolean reprLike, boolean alt) {
        int p = precision == 0 ? 1 : precision;
        if (x == 0) return reprLike ? "0.0" : alt ? "0." + "0".repeat(p - 1) : "0";
        BigDecimal bd = new BigDecimal(x).round(new MathContext(p, RoundingMode.HALF_EVEN));
        int exp = bd.precision() - bd.scale() - 1;
        String out;
        if (exp >= -4 && exp < p) {
            out = bd.setScale(Math.max(0, p - 1 - exp), RoundingMode.HALF_EVEN).toPlainString();
            if (!alt && out.indexOf('.') >= 0) out = out.replaceAll("0+$", "").replaceAll("\\.$", "");
            if (reprLike && out.indexOf('.') < 0) out += ".0";
        } else {
            out = scientific(x, p - 1, alt);
            if (!alt) out = out.replaceAll("\\.?0+e", "e");
        }
        return out;
    }

    private static String group(String digits, char sep) {
        StringBuilder sb = new StringBuilder();
        int n = digits.length();
        for (int i = 0; i < n; i++) {
            if (i > 0 && (n - i) % 3 == 0) sb.append(sep);
            sb.append(digits.charAt(i));
        }
        return sb.toString();
    }

    // ================================================================= strftime

    /** C/Python {@code strftime} in the C locale; unknown directives are copied verbatim. */
    static String strftime(String fmt, ZonedDateTime t) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < fmt.length(); i++) {
            char c = fmt.charAt(i);
            if (c != '%' || i + 1 >= fmt.length()) {
                sb.append(c);
                continue;
            }
            char d = fmt.charAt(++i);
            switch (d) {
                case 'a': sb.append(t.getDayOfWeek().getDisplayName(TextStyle.SHORT, Locale.ENGLISH)); break;
                case 'A': sb.append(t.getDayOfWeek().getDisplayName(TextStyle.FULL, Locale.ENGLISH)); break;
                case 'b': case 'h': sb.append(t.getMonth().getDisplayName(TextStyle.SHORT, Locale.ENGLISH)); break;
                case 'B': sb.append(t.getMonth().getDisplayName(TextStyle.FULL, Locale.ENGLISH)); break;
                case 'c': sb.append(strftime("%a %b %e %H:%M:%S %Y", t)); break;
                case 'd': sb.append(two(t.getDayOfMonth())); break;
                case 'e': sb.append(t.getDayOfMonth() < 10 ? " " : "").append(t.getDayOfMonth()); break;
                case 'D': case 'x': sb.append(strftime("%m/%d/%y", t)); break;
                case 'F': sb.append(strftime("%Y-%m-%d", t)); break;
                case 'f': sb.append(String.format(Locale.ROOT, "%06d", t.getNano() / 1000)); break;
                case 'G': sb.append(t.get(IsoFields.WEEK_BASED_YEAR)); break;
                case 'H': sb.append(two(t.getHour())); break;
                case 'I': sb.append(two(t.getHour() % 12 == 0 ? 12 : t.getHour() % 12)); break;
                case 'j': sb.append(String.format(Locale.ROOT, "%03d", t.getDayOfYear())); break;
                case 'm': sb.append(two(t.getMonthValue())); break;
                case 'M': sb.append(two(t.getMinute())); break;
                case 'n': sb.append('\n'); break;
                case 'p': sb.append(t.getHour() < 12 ? "AM" : "PM"); break;
                case 'R': sb.append(strftime("%H:%M", t)); break;
                case 'S': sb.append(two(t.getSecond())); break;
                case 't': sb.append('\t'); break;
                case 'T': case 'X': sb.append(strftime("%H:%M:%S", t)); break;
                case 'u': sb.append(t.getDayOfWeek().getValue()); break;
                case 'U': sb.append(two((t.getDayOfYear() + 6 - t.getDayOfWeek().getValue() % 7) / 7)); break;
                case 'V': sb.append(two(t.get(IsoFields.WEEK_OF_WEEK_BASED_YEAR))); break;
                case 'w': sb.append(t.getDayOfWeek().getValue() % 7); break;
                case 'W': sb.append(two((t.getDayOfYear() + 6 - (t.getDayOfWeek().getValue() + 6) % 7) / 7)); break;
                case 'y': sb.append(two(t.getYear() % 100)); break;
                case 'Y': sb.append(t.get(ChronoField.YEAR)); break;
                case 'z': sb.append(t.format(DateTimeFormatter.ofPattern("xx", Locale.ROOT))); break;
                case 'Z': sb.append(t.getZone().getDisplayName(TextStyle.SHORT, Locale.ENGLISH)); break;
                case '%': sb.append('%'); break;
                default: sb.append('%').append(d); break;
            }
        }
        return sb.toString();
    }

    private static String two(int v) { return v < 10 ? "0" + v : Integer.toString(v); }
}
