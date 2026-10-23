package com.eonmux.neuralfree.gguf.jinja;

import com.eonmux.neuralfree.gguf.jinja.JinjaAst.Node;
import com.eonmux.neuralfree.gguf.jinja.JinjaAst.Target;

import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Execution support: variable scopes, statement execution, macros, global functions and tests.
 *
 * <p>Scoping follows Jinja2: {@code if} blocks share the enclosing scope, every loop iteration and
 * every macro call gets a child scope, so {@code set} inside a loop or macro does not leak out
 * ({@code namespace()} objects carry state across). Macro bodies resolve free names through the
 * caller's scope, as the llama.cpp engine does (llama.cpp common/jinja/runtime.cpp macro_statement).
 */
public final class JinjaRuntime {

    /** Template syntax or evaluation error, carrying the 1-based template line when known. */
    public static final class TemplateException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        private final int line;

        public TemplateException(String message) { this(-1, message, null); }

        public TemplateException(int line, String message) { this(line, message, null); }

        public TemplateException(int line, String message, Throwable cause) {
            super(line > 0 ? "line " + line + ": " + message : message, cause);
            this.line = line;
        }

        /** Template line of the failing construct, or -1. */
        public int line() { return line; }

        TemplateException atLine(int l) {
            return line > 0 || l <= 0 ? this : new TemplateException(l, getMessage(), getCause() == null ? this : getCause());
        }
    }

    /** Non-local exit used by {@code break} and {@code continue}. */
    static final class LoopSignal extends RuntimeException {
        private static final long serialVersionUID = 1L;

        LoopSignal() { super(null, null, false, false); }
    }

    static final LoopSignal BREAK = new LoopSignal();
    static final LoopSignal CONTINUE = new LoopSignal();

    /** Variable scope chain. Not thread-safe; one chain exists per render. */
    public static final class Scope {
        private final Scope parent;
        private final Map<String, Object> vars = new HashMap<>();

        Scope(Scope parent) { this.parent = parent; }

        /** Variable value, then global function, otherwise Undefined. */
        Object resolve(String name) {
            for (Scope s = this; s != null; s = s.parent) {
                if (s.vars.containsKey(name)) return s.vars.get(name);
            }
            Object g = JinjaGlobals.lookup(name);
            return g != null ? g : JinjaValue.UNDEFINED;
        }

        void set(String name, Object value) { vars.put(name, value); }
    }

    private JinjaRuntime() {}

    static String render(List<Node> body, Map<String, ?> context) {
        Scope root = new Scope(null);
        if (context != null) {
            for (Map.Entry<String, ?> e : context.entrySet()) {
                try {
                    root.set(e.getKey(), JinjaValue.fromJava(e.getValue()));
                } catch (IllegalArgumentException ex) {
                    throw new TemplateException(-1, "context variable '" + e.getKey() + "': " + ex.getMessage(), ex);
                }
            }
        }
        StringBuilder out = new StringBuilder(1024);
        try {
            execBody(body, root, out);
        } catch (LoopSignal s) {
            throw new TemplateException("'break' or 'continue' outside of a loop");
        }
        return out.toString();
    }

    /** Executes statements, attaching the statement line to errors that lack one. */
    static void execBody(List<Node> body, Scope scope, StringBuilder out) {
        for (Node n : body) {
            try {
                n.exec(scope, out);
            } catch (LoopSignal s) {
                throw s;
            } catch (TemplateException e) {
                throw e.atLine(n.line);
            } catch (RuntimeException e) {
                String msg = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
                throw new TemplateException(n.line, msg, e);
            }
        }
    }

    // ------------------------------------------------------------------ for

    /**
     * Runs a for loop. The optional {@code if} filter is applied before the loop object is built, so
     * {@code loop.length}, {@code loop.last} and friends count only kept items (Jinja2 semantics).
     */
    static void execFor(JinjaAst.For f, Scope scope, StringBuilder out) {
        Object iterable = f.iterable.eval(scope);
        boolean dictPairs = JinjaValue.isDict(iterable) && f.target.isTuple() && f.target.elements.size() == 2;
        List<Object> items = dictPairs ? JinjaValue.items((Map<?, ?>) iterable) : JinjaValue.iterate(iterable);
        if (f.filter != null) {
            List<Object> kept = new ArrayList<>(items.size());
            for (Object item : items) {
                Scope s = new Scope(scope);
                assign(f.target, item, s);
                if (JinjaValue.truthy(f.filter.eval(s))) kept.add(item);
            }
            items = kept;
        }
        if (items.isEmpty()) {
            execBody(f.elseBody, scope, out);
            return;
        }
        Object[] lastChanged = { JinjaValue.UNDEFINED };
        int n = items.size();
        for (int i = 0; i < n; i++) {
            Scope s = new Scope(scope);
            s.set("loop", loopObject(items, i, lastChanged));
            assign(f.target, items.get(i), s);
            try {
                execBody(f.body, s, out);
            } catch (LoopSignal sig) {
                if (sig == BREAK) break;
            }
        }
    }

    private static JinjaValue.Namespace loopObject(List<Object> items, int i, Object[] lastChanged) {
        int n = items.size();
        JinjaValue.Namespace loop = new JinjaValue.Namespace();
        loop.put("index", (long) (i + 1));
        loop.put("index0", (long) i);
        loop.put("revindex", (long) (n - i));
        loop.put("revindex0", (long) (n - i - 1));
        loop.put("first", i == 0);
        loop.put("last", i == n - 1);
        loop.put("length", (long) n);
        loop.put("depth", 1L);
        loop.put("depth0", 0L);
        loop.put("previtem", i > 0 ? items.get(i - 1) : JinjaValue.UNDEFINED);
        loop.put("nextitem", i + 1 < n ? items.get(i + 1) : JinjaValue.UNDEFINED);
        loop.put("cycle", (JinjaValue.Callable) (args, kwargs, caller) -> {
            if (args.isEmpty()) throw new TemplateException("loop.cycle() requires at least one argument");
            return args.get(i % args.size());
        });
        loop.put("changed", (JinjaValue.Callable) (args, kwargs, caller) -> {
            Object key = new JinjaValue.Tuple(args);
            boolean changed = !JinjaValue.pyEquals(lastChanged[0], key);
            lastChanged[0] = key;
            return changed;
        });
        return loop;
    }

    // ------------------------------------------------------------------ assignment

    /** Binds a value to a {@code set}/{@code for} target in the given scope. */
    static void assign(Target target, Object value, Scope scope) {
        if (target.isTuple()) {
            List<Object> parts = unpack(value, target.elements.size());
            for (int i = 0; i < parts.size(); i++) assign(target.elements.get(i), parts.get(i), scope);
            return;
        }
        if (target.attribute == null) {
            scope.set(target.name, value);
            return;
        }
        Object obj = scope.resolve(target.name);
        if (!(obj instanceof Map)) {
            throw new TemplateException("cannot assign attribute '" + target.attribute + "' on '" + target.name
                    + "' of type " + JinjaValue.typeName(obj) + "; use namespace()");
        }
        @SuppressWarnings("unchecked")
        Map<Object, Object> map = (Map<Object, Object>) obj;
        map.put(target.attribute, value);
    }

    private static List<Object> unpack(Object value, int count) {
        List<Object> parts = value instanceof String ? JinjaValue.iterate(value)
                : value instanceof List ? new ArrayList<>((List<?>) value) : null;
        if (parts == null) throw new TemplateException("cannot unpack non-sequence " + JinjaValue.typeName(value));
        if (parts.size() != count) {
            throw new TemplateException((parts.size() > count ? "too many" : "not enough")
                    + " values to unpack (expected " + count + ", got " + parts.size() + ")");
        }
        return parts;
    }

    // ------------------------------------------------------------------ macros

    /**
     * A macro bound as a callable. Parameters bind from keywords, then positions, then defaults
     * (evaluated in the new macro scope so they may refer to earlier parameters); missing ones are
     * Undefined. Extra positionals and keywords are exposed as {@code varargs} and {@code kwargs}.
     */
    static final class MacroFunction implements JinjaValue.Callable {
        private final JinjaAst.Macro macro;

        MacroFunction(JinjaAst.Macro macro) { this.macro = macro; }

        @Override public Object call(List<Object> args, Map<String, Object> kwargs, Scope caller) {
            Scope s = new Scope(caller);
            Map<String, Object> extraKw = new LinkedHashMap<>(kwargs);
            List<String> params = macro.params;
            for (int i = 0; i < params.size(); i++) {
                String p = params.get(i);
                Object v;
                if (extraKw.containsKey(p)) v = extraKw.remove(p);
                else if (i < args.size()) v = args.get(i);
                else if (macro.defaults.get(i) != null) v = macro.defaults.get(i).eval(s);
                else v = JinjaValue.UNDEFINED;
                s.set(p, v);
            }
            s.set("varargs", args.size() > params.size()
                    ? new ArrayList<>(args.subList(params.size(), args.size())) : new ArrayList<>());
            s.set("kwargs", extraKw);
            StringBuilder out = new StringBuilder();
            try {
                execBody(macro.body, s, out);
            } catch (LoopSignal sig) {
                throw new TemplateException(macro.line, "'break' or 'continue' outside of a loop in macro '" + macro.name + "'");
            }
            return out.toString();
        }

        @Override public String toString() { return "macro " + macro.name; }
    }

    // ------------------------------------------------------------------ tests

    /**
     * Applies a Jinja2 test ({@code value is name(args)}). Type tests follow Python: strings, lists,
     * dicts and Undefined are iterable and sequences; bool counts as a number but not an integer.
     */
    static boolean test(String name, Object v, List<Object> args) {
        switch (name) {
            case "defined": return v != JinjaValue.UNDEFINED;
            case "undefined": return v == JinjaValue.UNDEFINED;
            case "none": return v == null;
            case "boolean": return v instanceof Boolean;
            case "true": return Boolean.TRUE.equals(v);
            case "false": return Boolean.FALSE.equals(v);
            case "integer": return v instanceof Long;
            case "float": return v instanceof Double;
            case "number": return JinjaValue.isNumeric(v);
            case "string": return v instanceof String;
            case "mapping": return JinjaValue.isDict(v);
            case "iterable":
            case "sequence":
                return v instanceof String || v instanceof List || v instanceof Map || v == JinjaValue.UNDEFINED;
            case "callable": return v instanceof JinjaValue.Callable;
            case "odd": return Math.floorMod(JinjaValue.toLong(v, "odd"), 2L) == 1L;
            case "even": return Math.floorMod(JinjaValue.toLong(v, "even"), 2L) == 0L;
            case "divisibleby": {
                long d = JinjaValue.toLong(arg(name, args, 0), "divisibleby");
                if (d == 0) throw new TemplateException("divisibleby(0)");
                return Math.floorMod(JinjaValue.toLong(v, "divisibleby"), d) == 0L;
            }
            case "eq": case "equalto": case "==": return JinjaValue.pyEquals(v, arg(name, args, 0));
            case "ne": case "!=": return !JinjaValue.pyEquals(v, arg(name, args, 0));
            case "lt": case "lessthan": case "<": return JinjaValue.compare(v, arg(name, args, 0)) < 0;
            case "le": case "<=": return JinjaValue.compare(v, arg(name, args, 0)) <= 0;
            case "gt": case "greaterthan": case ">": return JinjaValue.compare(v, arg(name, args, 0)) > 0;
            case "ge": case ">=": return JinjaValue.compare(v, arg(name, args, 0)) >= 0;
            case "in": return JinjaValue.contains(arg(name, args, 0), v);
            case "sameas": return sameAs(v, arg(name, args, 0));
            case "lower": return v instanceof String && JinjaMethods.isCased((String) v, false);
            case "upper": return v instanceof String && JinjaMethods.isCased((String) v, true);
            case "escaped": return false;
            case "filter": return JinjaFilters.exists(JinjaValue.str(arg(name, args, 0)));
            case "test": return isTest(JinjaValue.str(arg(name, args, 0)));
            default: throw new TemplateException("unknown test '" + name + "'");
        }
    }

    /** Whether {@code name} is a supported test. */
    static boolean isTest(String name) {
        return TEST_NAMES.contains(name);
    }

    private static final java.util.Set<String> TEST_NAMES = Collections.unmodifiableSet(new java.util.HashSet<>(
            java.util.Arrays.asList("defined", "undefined", "none", "boolean", "true", "false", "integer", "float",
                    "number", "string", "mapping", "iterable", "sequence", "callable", "odd", "even", "divisibleby",
                    "eq", "equalto", "==", "ne", "!=", "lt", "lessthan", "<", "le", "<=", "gt", "greaterthan", ">",
                    "ge", ">=", "in", "sameas", "lower", "upper", "escaped", "filter", "test")));

    private static Object arg(String test, List<Object> args, int i) {
        if (i >= args.size()) throw new TemplateException("test '" + test + "' requires an argument");
        return args.get(i);
    }

    /** Python identity for the singletons and small values templates compare with {@code sameas}. */
    private static boolean sameAs(Object a, Object b) {
        if (a == null || b == null) return a == b;
        if (a instanceof Boolean || b instanceof Boolean) return a.equals(b);
        if (a instanceof Long && b instanceof Long) return a.equals(b);
        return a == b;
    }

    // ------------------------------------------------------------------ globals

    /** Global functions available to every template. */
    static final class JinjaGlobals {
        private static final Map<String, JinjaValue.Callable> GLOBALS;

        static {
            Map<String, JinjaValue.Callable> g = new HashMap<>();
            g.put("range", (args, kw, s) -> range(args));
            g.put("dict", (args, kw, s) -> dict(args, kw));
            g.put("namespace", (args, kw, s) -> {
                JinjaValue.Namespace ns = new JinjaValue.Namespace();
                for (Map.Entry<Object, Object> e : dict(args, kw).entrySet()) ns.put(JinjaValue.str(e.getKey()), e.getValue());
                return ns;
            });
            g.put("raise_exception", (args, kw, s) -> {
                throw new TemplateException("Jinja template error: " + (args.isEmpty() ? "" : JinjaValue.str(args.get(0))));
            });
            g.put("strftime_now", (args, kw, s) -> {
                if (args.isEmpty()) throw new TemplateException("strftime_now() requires a format string");
                return JinjaMethods.strftime(JinjaValue.str(args.get(0)), ZonedDateTime.now());
            });
            g.put("cycler", (args, kw, s) -> cycler(args));
            g.put("joiner", (args, kw, s) -> joiner(args.isEmpty() ? ", " : JinjaValue.str(args.get(0))));
            GLOBALS = Collections.unmodifiableMap(g);
        }

        private JinjaGlobals() {}

        static Object lookup(String name) { return GLOBALS.get(name); }

        private static Object range(List<Object> args) {
            if (args.isEmpty() || args.size() > 3) throw new TemplateException("range() takes 1 to 3 arguments");
            long start = 0, stop, step = 1;
            if (args.size() == 1) {
                stop = JinjaValue.toLong(args.get(0), "range");
            } else {
                start = JinjaValue.toLong(args.get(0), "range");
                stop = JinjaValue.toLong(args.get(1), "range");
                if (args.size() == 3) step = JinjaValue.toLong(args.get(2), "range");
            }
            if (step == 0) throw new TemplateException("range() step must not be zero");
            List<Object> out = new ArrayList<>();
            for (long i = start; step > 0 ? i < stop : i > stop; i += step) out.add(i);
            return out;
        }

        /** {@code dict()}/{@code namespace()}: an optional mapping or pair list, then keyword entries. */
        private static Map<Object, Object> dict(List<Object> args, Map<String, Object> kwargs) {
            if (args.size() > 1) throw new TemplateException("dict()/namespace() takes at most one positional argument");
            Map<Object, Object> out = new LinkedHashMap<>();
            if (!args.isEmpty()) {
                Object src = args.get(0);
                if (src instanceof Map) {
                    out.putAll((Map<?, ?>) src);
                } else if (src instanceof List) {
                    for (Object pair : (List<?>) src) {
                        if (!(pair instanceof List) || ((List<?>) pair).size() != 2) {
                            throw new TemplateException("dict() sequence elements must be (key, value) pairs");
                        }
                        out.put(((List<?>) pair).get(0), ((List<?>) pair).get(1));
                    }
                } else {
                    throw new TemplateException("dict() argument must be a mapping or a sequence of pairs, got "
                            + JinjaValue.typeName(src));
                }
            }
            out.putAll(kwargs);
            return out;
        }

        private static Object cycler(List<Object> items) {
            if (items.isEmpty()) throw new TemplateException("cycler() requires at least one item");
            JinjaValue.Namespace c = new JinjaValue.Namespace();
            int[] pos = { 0 };
            c.put("current", items.get(0));
            c.put("next", (JinjaValue.Callable) (a, k, s) -> {
                Object v = items.get(pos[0]);
                pos[0] = (pos[0] + 1) % items.size();
                c.put("current", items.get(pos[0]));
                return v;
            });
            c.put("reset", (JinjaValue.Callable) (a, k, s) -> {
                pos[0] = 0;
                c.put("current", items.get(0));
                return null;
            });
            return c;
        }

        private static Object joiner(String sep) {
            boolean[] used = { false };
            return (JinjaValue.Callable) (a, k, s) -> {
                if (!used[0]) {
                    used[0] = true;
                    return "";
                }
                return sep;
            };
        }
    }
}
