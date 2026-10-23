package com.eonmux.neuralfree.gguf.jinja;

import java.math.BigDecimal;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.RandomAccess;

/**
 * Value model of the Jinja engine with Python semantics.
 *
 * <p>Runtime values are: {@code null} (None), {@link #UNDEFINED}, {@link Boolean}, {@link Long} (int),
 * {@link Double} (float), {@link String}, {@link List} (list), {@link Tuple}, {@link Map} (dict),
 * {@link Namespace} and {@link Callable}. {@link #fromJava(Object)} converts caller-supplied context
 * data into this model. Truthiness, {@code str()}, {@code repr()}, equality, ordering, membership,
 * arithmetic, attribute/item access and slicing follow Python and Jinja2; where Jinja2 and the
 * llama.cpp engine (common/jinja) differ, the llama.cpp behaviour is used and noted on the method.
 * Strings are measured, indexed and sliced by code point.
 */
public final class JinjaValue {

    /** Jinja2 {@code Undefined}: renders as an empty string and is falsy. */
    public static final Object UNDEFINED = new Object() {
        @Override public String toString() { return ""; }
    };

    /** Callable value: macros, bound methods and global functions. */
    public interface Callable {
        Object call(List<Object> args, Map<String, Object> kwargs, JinjaRuntime.Scope caller);
    }

    /** Result of {@code namespace(...)}: attribute-only mutable object without dict methods. */
    public static final class Namespace extends LinkedHashMap<String, Object> {
        private static final long serialVersionUID = 1L;
    }

    /** Immutable Python tuple. */
    public static final class Tuple extends AbstractList<Object> implements RandomAccess {
        private final Object[] items;

        public Tuple(Object... items) { this.items = items.clone(); }

        public Tuple(List<?> items) { this.items = items.toArray(); }

        @Override public Object get(int index) { return items[index]; }

        @Override public int size() { return items.length; }
    }

    private JinjaValue() {}

    // ------------------------------------------------------------------ conversion

    /**
     * Converts caller data into the value model: integral boxes become {@link Long}, floating boxes
     * {@link Double}, characters and char sequences {@link String}, collections and arrays fresh
     * mutable lists, maps fresh insertion-ordered maps and enums their name. Containers are copied so
     * template-side mutation never reaches caller data.
     *
     * @throws IllegalArgumentException for types that have no Jinja equivalent
     */
    public static Object fromJava(Object v) {
        if (v == null || v == UNDEFINED || v instanceof String || v instanceof Boolean
                || v instanceof Long || v instanceof Double || v instanceof Callable) {
            return v;
        }
        if (v instanceof Integer || v instanceof Short || v instanceof Byte) return ((Number) v).longValue();
        if (v instanceof Float) return ((Float) v).doubleValue();
        if (v instanceof java.math.BigInteger) {
            java.math.BigInteger big = (java.math.BigInteger) v;
            if (big.bitLength() > 63) throw new IllegalArgumentException("integer " + big + " exceeds the 64-bit range");
            return big.longValue();
        }
        if (v instanceof BigDecimal) return ((BigDecimal) v).doubleValue();
        if (v instanceof Character || v instanceof CharSequence) return v.toString();
        if (v instanceof Enum) return ((Enum<?>) v).name();
        if (v instanceof Tuple) {
            List<Object> items = new ArrayList<>();
            for (Object o : (Tuple) v) items.add(fromJava(o));
            return new Tuple(items);
        }
        if (v instanceof Namespace) {
            Namespace ns = new Namespace();
            for (Map.Entry<String, Object> e : ((Namespace) v).entrySet()) ns.put(e.getKey(), fromJava(e.getValue()));
            return ns;
        }
        if (v instanceof Map) {
            Map<Object, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) out.put(fromJava(e.getKey()), fromJava(e.getValue()));
            return out;
        }
        if (v instanceof Collection) {
            List<Object> out = new ArrayList<>(((Collection<?>) v).size());
            for (Object o : (Collection<?>) v) out.add(fromJava(o));
            return out;
        }
        if (v instanceof Object[]) return fromJava(Arrays.asList((Object[]) v));
        if (v.getClass().isArray()) {
            int n = java.lang.reflect.Array.getLength(v);
            List<Object> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) out.add(fromJava(java.lang.reflect.Array.get(v, i)));
            return out;
        }
        throw new IllegalArgumentException("unsupported template context value of type " + v.getClass().getName());
    }

    // ------------------------------------------------------------------ type predicates

    public static boolean isInt(Object v) { return v instanceof Long; }

    /** Python numbers: int, float and bool. */
    public static boolean isNumeric(Object v) { return v instanceof Long || v instanceof Double || v instanceof Boolean; }

    public static boolean isSequence(Object v) { return v instanceof List; }

    /** Dict values; namespaces are attribute objects, not mappings. */
    public static boolean isDict(Object v) { return v instanceof Map && !(v instanceof Namespace); }

    public static String typeName(Object v) {
        if (v == null) return "NoneType";
        if (v == UNDEFINED) return "Undefined";
        if (v instanceof Boolean) return "bool";
        if (v instanceof Long) return "int";
        if (v instanceof Double) return "float";
        if (v instanceof String) return "str";
        if (v instanceof Tuple) return "tuple";
        if (v instanceof List) return "list";
        if (v instanceof Namespace) return "Namespace";
        if (v instanceof Map) return "dict";
        if (v instanceof Callable) return "function";
        return v.getClass().getSimpleName();
    }

    // ------------------------------------------------------------------ truthiness and strings

    public static boolean truthy(Object v) {
        if (v == null || v == UNDEFINED) return false;
        if (v instanceof Boolean) return (Boolean) v;
        if (v instanceof Long) return (Long) v != 0L;
        if (v instanceof Double) return (Double) v != 0.0;
        if (v instanceof String) return !((String) v).isEmpty();
        if (v instanceof List) return !((List<?>) v).isEmpty();
        if (v instanceof Namespace) return true;
        if (v instanceof Map) return !((Map<?, ?>) v).isEmpty();
        return true;
    }

    /** Python {@code str()}; {@link #UNDEFINED} renders as the empty string. */
    public static String str(Object v) {
        if (v instanceof String) return (String) v;
        if (v == UNDEFINED) return "";
        if (v == null) return "None";
        if (v instanceof Boolean) return (Boolean) v ? "True" : "False";
        if (v instanceof Long) return v.toString();
        if (v instanceof Double) return floatRepr((Double) v);
        return repr(v);
    }

    /** Python {@code repr()}. */
    public static String repr(Object v) {
        if (v instanceof String) return strRepr((String) v);
        if (v == UNDEFINED) return "Undefined";
        if (v instanceof Tuple) {
            Tuple t = (Tuple) v;
            if (t.size() == 1) return "(" + repr(t.get(0)) + ",)";
            return joinRepr(t, "(", ")");
        }
        if (v instanceof List) return joinRepr((List<?>) v, "[", "]");
        if (v instanceof Map) {
            StringBuilder sb = new StringBuilder(v instanceof Namespace ? "<Namespace {" : "{");
            boolean first = true;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
                if (!first) sb.append(", ");
                first = false;
                sb.append(repr(e.getKey())).append(": ").append(repr(e.getValue()));
            }
            return sb.append(v instanceof Namespace ? "}>" : "}").toString();
        }
        if (v instanceof Callable) return "<function " + v + ">";
        return str(v);
    }

    private static String joinRepr(List<?> items, String open, String close) {
        StringBuilder sb = new StringBuilder(open);
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(repr(items.get(i)));
        }
        return sb.append(close).toString();
    }

    private static String strRepr(String s) {
        char quote = s.indexOf('\'') >= 0 && s.indexOf('"') < 0 ? '"' : '\'';
        StringBuilder sb = new StringBuilder(s.length() + 2).append(quote);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == quote || c == '\\') sb.append('\\').append(c);
            else if (c == '\n') sb.append("\\n");
            else if (c == '\r') sb.append("\\r");
            else if (c == '\t') sb.append("\\t");
            else if (c < 0x20 || c == 0x7f) sb.append(String.format("\\x%02x", (int) c));
            else sb.append(c);
        }
        return sb.append(quote).toString();
    }

    /**
     * Python {@code repr(float)}: shortest round-tripping digits, fixed notation for decimal exponents
     * in [-4, 16), otherwise {@code d.ddde±XX} (matches llama.cpp common/jinja/value.cpp {@code float_repr}).
     */
    public static String floatRepr(double d) {
        if (Double.isNaN(d)) return "nan";
        if (Double.isInfinite(d)) return d > 0 ? "inf" : "-inf";
        if (d == 0.0) return (1.0 / d) < 0 ? "-0.0" : "0.0";
        BigDecimal bd = new BigDecimal(Double.toString(Math.abs(d))).stripTrailingZeros();
        String digits = bd.unscaledValue().toString();
        int n = digits.length();
        int exp10 = n - 1 - bd.scale();
        StringBuilder out = new StringBuilder(d < 0 ? "-" : "");
        if (exp10 >= -4 && exp10 < 16) {
            int point = exp10 + 1;
            if (point <= 0) {
                out.append("0.");
                for (int i = 0; i < -point; i++) out.append('0');
                out.append(digits);
            } else if (point >= n) {
                out.append(digits);
                for (int i = 0; i < point - n; i++) out.append('0');
                out.append(".0");
            } else {
                out.append(digits, 0, point).append('.').append(digits, point, n);
            }
        } else {
            out.append(digits.charAt(0));
            if (n > 1) out.append('.').append(digits, 1, n);
            out.append('e').append(exp10 < 0 ? '-' : '+');
            int ae = Math.abs(exp10);
            if (ae < 10) out.append('0');
            out.append(ae);
        }
        return out.toString();
    }

    // ------------------------------------------------------------------ numbers

    static long toLong(Object v, String what) {
        if (v instanceof Long) return (Long) v;
        if (v instanceof Boolean) return (Boolean) v ? 1L : 0L;
        throw new JinjaRuntime.TemplateException(what + " expects an integer, got " + typeName(v));
    }

    static double toDouble(Object v, String what) {
        if (v instanceof Double) return (Double) v;
        if (v instanceof Long) return (Long) v;
        if (v instanceof Boolean) return (Boolean) v ? 1.0 : 0.0;
        throw new JinjaRuntime.TemplateException(what + " expects a number, got " + typeName(v));
    }

    /** Python string-to-whitespace test ({@code str.isspace} per character). */
    static boolean isPySpace(int cp) {
        return Character.isWhitespace(cp) || Character.isSpaceChar(cp) || cp == 0x85;
    }

    // ------------------------------------------------------------------ equality, ordering, membership

    /** Python {@code ==}: numeric across int/float/bool, element-wise for lists, tuples and dicts. */
    public static boolean pyEquals(Object a, Object b) {
        if (a == b) return true;
        if (a == null || b == null || a == UNDEFINED || b == UNDEFINED) return false;
        if (isNumeric(a) && isNumeric(b)) {
            if ((a instanceof Double) || (b instanceof Double)) return toDouble(a, "==") == toDouble(b, "==");
            return toLong(a, "==") == toLong(b, "==");
        }
        if (a instanceof String || b instanceof String) return a.equals(b);
        if (a instanceof List && b instanceof List) {
            if ((a instanceof Tuple) != (b instanceof Tuple)) return false;
            List<?> la = (List<?>) a, lb = (List<?>) b;
            if (la.size() != lb.size()) return false;
            for (int i = 0; i < la.size(); i++) if (!pyEquals(la.get(i), lb.get(i))) return false;
            return true;
        }
        if (a instanceof Map && b instanceof Map) {
            Map<?, ?> ma = (Map<?, ?>) a, mb = (Map<?, ?>) b;
            if (ma.size() != mb.size()) return false;
            for (Map.Entry<?, ?> e : ma.entrySet()) {
                if (!mb.containsKey(e.getKey()) || !pyEquals(e.getValue(), mb.get(e.getKey()))) return false;
            }
            return true;
        }
        return a.equals(b);
    }

    /** Python ordering for numbers, strings (by code point) and sequences. */
    public static int compare(Object a, Object b) {
        if (isNumeric(a) && isNumeric(b)) {
            if ((a instanceof Double) || (b instanceof Double)) return Double.compare(toDouble(a, "<"), toDouble(b, "<"));
            return Long.compare(toLong(a, "<"), toLong(b, "<"));
        }
        if (a instanceof String && b instanceof String) return compareCodePoints((String) a, (String) b);
        if (a instanceof List && b instanceof List) {
            List<?> la = (List<?>) a, lb = (List<?>) b;
            int n = Math.min(la.size(), lb.size());
            for (int i = 0; i < n; i++) {
                if (!pyEquals(la.get(i), lb.get(i))) return compare(la.get(i), lb.get(i));
            }
            return Integer.compare(la.size(), lb.size());
        }
        throw new JinjaRuntime.TemplateException(
                "ordering is not supported between " + typeName(a) + " and " + typeName(b));
    }

    private static int compareCodePoints(String a, String b) {
        int i = 0, j = 0;
        while (i < a.length() && j < b.length()) {
            int ca = a.codePointAt(i), cb = b.codePointAt(j);
            if (ca != cb) return Integer.compare(ca, cb);
            i += Character.charCount(ca);
            j += Character.charCount(cb);
        }
        return Integer.compare(a.length() - i, b.length() - j);
    }

    /**
     * Python {@code item in container}. {@code x in undefined} is false (llama.cpp runtime.cpp special case);
     * a None container is an error.
     */
    public static boolean contains(Object container, Object item) {
        if (container == UNDEFINED) return false;
        if (container instanceof String) {
            if (!(item instanceof String)) {
                throw new JinjaRuntime.TemplateException("'in <string>' requires a string operand, got " + typeName(item));
            }
            return ((String) container).contains((String) item);
        }
        if (container instanceof List) {
            for (Object o : (List<?>) container) if (pyEquals(o, item)) return true;
            return false;
        }
        if (container instanceof Map) return mapContainsKey((Map<?, ?>) container, item);
        throw new JinjaRuntime.TemplateException("argument of type '" + typeName(container) + "' is not iterable");
    }

    static boolean mapContainsKey(Map<?, ?> map, Object key) {
        if (map.containsKey(key)) return true;
        if (key instanceof String || key == null) return false;
        for (Object k : map.keySet()) if (pyEquals(k, key)) return true;
        return false;
    }

    static Object mapGet(Map<?, ?> map, Object key, Object missing) {
        if (map.containsKey(key)) return map.get(key);
        if (key instanceof String || key == null) return missing;
        for (Map.Entry<?, ?> e : map.entrySet()) if (pyEquals(e.getKey(), key)) return e.getValue();
        return missing;
    }

    // ------------------------------------------------------------------ attribute, item, slice, iteration

    /**
     * Jinja2 {@code obj.name}: methods of strings, lists and dicts take precedence over dict keys
     * (so {@code d.items} is the method and {@code d['items']} the key); namespaces expose only keys.
     * Missing attributes, and attributes of None or Undefined, are Undefined.
     */
    public static Object getAttr(Object obj, String name) {
        if (obj instanceof Namespace) return mapGet((Map<?, ?>) obj, name, UNDEFINED);
        Object method = JinjaMethods.bind(obj, name);
        if (method != null) return method;
        if (obj instanceof Map) return mapGet((Map<?, ?>) obj, name, UNDEFINED);
        return UNDEFINED;
    }

    /** Jinja2 {@code obj[key]}: item lookup first, attribute lookup as fallback for string keys. */
    public static Object getItem(Object obj, Object key) {
        if (obj instanceof Map) {
            Object v = mapGet((Map<?, ?>) obj, key, UNDEFINED);
            if (v != UNDEFINED || !(key instanceof String) || obj instanceof Namespace) return v;
            return getAttr(obj, (String) key);
        }
        if ((obj instanceof List || obj instanceof String) && (key instanceof Long || key instanceof Boolean)) {
            long idx = toLong(key, "index");
            if (obj instanceof List) {
                List<?> list = (List<?>) obj;
                if (idx < 0) idx += list.size();
                return idx >= 0 && idx < list.size() ? list.get((int) idx) : UNDEFINED;
            }
            int[] cps = ((String) obj).codePoints().toArray();
            if (idx < 0) idx += cps.length;
            return idx >= 0 && idx < cps.length ? new String(cps, (int) idx, 1) : UNDEFINED;
        }
        if (key instanceof String) return getAttr(obj, (String) key);
        return UNDEFINED;
    }

    /** Python {@code len()} for strings (code points), sequences and dicts. */
    public static long length(Object v) {
        if (v instanceof String) return ((String) v).codePointCount(0, ((String) v).length());
        if (v instanceof List) return ((List<?>) v).size();
        if (v instanceof Map) return ((Map<?, ?>) v).size();
        if (v == UNDEFINED) return 0L;
        throw new JinjaRuntime.TemplateException("object of type '" + typeName(v) + "' has no len()");
    }

    /** Python slicing ({@code PySlice_AdjustIndices}) of lists, tuples and strings; null bounds use defaults. */
    public static Object slice(Object obj, Object start, Object stop, Object step) {
        long st = step == null || step == UNDEFINED ? 1L : toLong(step, "slice step");
        if (st == 0) throw new JinjaRuntime.TemplateException("slice step cannot be zero");
        if (obj == UNDEFINED) return new ArrayList<>();
        int[] cps = obj instanceof String ? ((String) obj).codePoints().toArray() : null;
        List<?> list = cps == null ? asListForSlice(obj) : null;
        long len = cps != null ? cps.length : list.size();
        long lo = adjust(start, len, st, st < 0 ? len - 1 : 0);
        long hi = adjust(stop, len, st, st < 0 ? -1 : len);
        if (cps != null) {
            StringBuilder sb = new StringBuilder();
            for (long i = lo; st > 0 ? i < hi : i > hi; i += st) sb.appendCodePoint(cps[(int) i]);
            return sb.toString();
        }
        List<Object> out = new ArrayList<>();
        for (long i = lo; st > 0 ? i < hi : i > hi; i += st) out.add(list.get((int) i));
        return obj instanceof Tuple ? new Tuple(out) : out;
    }

    private static List<?> asListForSlice(Object obj) {
        if (obj instanceof List) return (List<?>) obj;
        throw new JinjaRuntime.TemplateException("'" + typeName(obj) + "' object is not subscriptable");
    }

    private static long adjust(Object bound, long len, long step, long dflt) {
        if (bound == null || bound == UNDEFINED) return dflt;
        long v = toLong(bound, "slice index");
        if (v < 0) {
            v += len;
            if (v < 0) v = step < 0 ? -1 : 0;
        } else if (v >= len) {
            v = step < 0 ? len - 1 : len;
        }
        return v;
    }

    /**
     * Items of a for-loop or iterating filter: lists and tuples as-is, dict keys, string code points;
     * Undefined iterates as empty.
     */
    public static List<Object> iterate(Object v) {
        if (v == UNDEFINED) return new ArrayList<>();
        if (v instanceof List) return new ArrayList<>((List<?>) v);
        if (v instanceof Map) return new ArrayList<>(((Map<?, ?>) v).keySet());
        if (v instanceof String) {
            List<Object> out = new ArrayList<>();
            ((String) v).codePoints().forEach(cp -> out.add(new String(Character.toChars(cp))));
            return out;
        }
        throw new JinjaRuntime.TemplateException("'" + typeName(v) + "' object is not iterable");
    }

    /** Dict items as a list of (key, value) tuples. */
    public static List<Object> items(Map<?, ?> map) {
        List<Object> out = new ArrayList<>(map.size());
        for (Map.Entry<?, ?> e : map.entrySet()) out.add(new Tuple(e.getKey(), e.getValue()));
        return out;
    }

    // ------------------------------------------------------------------ arithmetic

    /**
     * Binary arithmetic and concatenation. {@code ~} concatenates {@code str()} of both operands and
     * {@code +} does the same when either operand is a string; in both, None and Undefined count as
     * the empty string (llama.cpp common/jinja/runtime.cpp binary_expression).
     */
    public static Object binary(String op, Object l, Object r) {
        if ("~".equals(op)) return concatStr(l) + concatStr(r);
        if ("+".equals(op) && (l instanceof String || r instanceof String)) return concatStr(l) + concatStr(r);
        if (l == UNDEFINED || r == UNDEFINED) {
            throw new JinjaRuntime.TemplateException("cannot apply '" + op + "' to an undefined value");
        }
        if (isNumeric(l) && isNumeric(r)) return numeric(op, l, r);
        switch (op) {
            case "+":
                if (l instanceof List && r instanceof List) {
                    List<Object> out = new ArrayList<>((List<?>) l);
                    out.addAll((List<?>) r);
                    return l instanceof Tuple && r instanceof Tuple ? new Tuple(out) : out;
                }
                break;
            case "*":
                if ((l instanceof String || l instanceof List) && r instanceof Long) return repeat(l, (Long) r);
                if ((r instanceof String || r instanceof List) && l instanceof Long) return repeat(r, (Long) l);
                break;
            case "%":
                if (l instanceof String) return JinjaMethods.percentFormat((String) l, r);
                break;
            default:
                break;
        }
        throw new JinjaRuntime.TemplateException(
                "unsupported operand types for " + op + ": '" + typeName(l) + "' and '" + typeName(r) + "'");
    }

    private static String concatStr(Object v) {
        return v == null ? "" : str(v);
    }

    private static Object repeat(Object seq, long times) {
        if (seq instanceof String) {
            StringBuilder sb = new StringBuilder();
            for (long i = 0; i < times; i++) sb.append((String) seq);
            return sb.toString();
        }
        List<Object> out = new ArrayList<>();
        for (long i = 0; i < times; i++) out.addAll((List<?>) seq);
        return seq instanceof Tuple ? new Tuple(out) : out;
    }

    private static Object numeric(String op, Object l, Object r) {
        boolean ints = !(l instanceof Double) && !(r instanceof Double);
        if (ints) {
            long a = toLong(l, op), b = toLong(r, op);
            switch (op) {
                case "+": return a + b;
                case "-": return a - b;
                case "*": return a * b;
                case "/":
                    if (b == 0) throw new JinjaRuntime.TemplateException("division by zero");
                    return (double) a / (double) b;
                case "//":
                    if (b == 0) throw new JinjaRuntime.TemplateException("integer division by zero");
                    return Math.floorDiv(a, b);
                case "%":
                    if (b == 0) throw new JinjaRuntime.TemplateException("integer modulo by zero");
                    return Math.floorMod(a, b);
                case "**":
                    return b >= 0 ? intPow(a, b) : Math.pow(a, b);
                default:
                    break;
            }
        } else {
            double a = toDouble(l, op), b = toDouble(r, op);
            switch (op) {
                case "+": return a + b;
                case "-": return a - b;
                case "*": return a * b;
                case "/":
                    if (b == 0) throw new JinjaRuntime.TemplateException("float division by zero");
                    return a / b;
                case "//":
                    if (b == 0) throw new JinjaRuntime.TemplateException("float floor division by zero");
                    return Math.floor(a / b);
                case "%":
                    if (b == 0) throw new JinjaRuntime.TemplateException("float modulo by zero");
                    return a - b * Math.floor(a / b);
                case "**": return Math.pow(a, b);
                default:
                    break;
            }
        }
        throw new JinjaRuntime.TemplateException("unknown arithmetic operator '" + op + "'");
    }

    /** Integer power by repeated squaring; results outside the 64-bit range are an error. */
    private static long intPow(long base, long exp) {
        long result = 1;
        long b = base;
        long e = exp;
        try {
            while (e > 0) {
                if ((e & 1L) != 0) result = Math.multiplyExact(result, b);
                e >>= 1;
                if (e > 0) b = Math.multiplyExact(b, b);
            }
        } catch (ArithmeticException overflow) {
            throw new JinjaRuntime.TemplateException(base + " ** " + exp + " exceeds the 64-bit integer range");
        }
        return result;
    }

    /** Unary minus and plus. */
    public static Object unary(String op, Object v) {
        if (v instanceof Double) return "-".equals(op) ? -(Double) v : v;
        if (v instanceof Long || v instanceof Boolean) {
            long x = toLong(v, op);
            return "-".equals(op) ? -x : x;
        }
        throw new JinjaRuntime.TemplateException("bad operand type for unary " + op + ": '" + typeName(v) + "'");
    }
}
