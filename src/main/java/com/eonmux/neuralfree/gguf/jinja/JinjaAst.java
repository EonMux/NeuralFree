package com.eonmux.neuralfree.gguf.jinja;

import com.eonmux.neuralfree.gguf.jinja.JinjaRuntime.Scope;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable syntax tree of a parsed template. Expressions evaluate against a {@link Scope};
 * statements write their output to a {@link StringBuilder}. Nodes are built only by
 * {@link JinjaParser} and are safe to share between threads.
 */
public final class JinjaAst {

    private JinjaAst() {}

    /** A parsed template; rendering is thread-safe. */
    public static final class Program {
        private final List<Node> body;

        Program(List<Node> body) { this.body = Collections.unmodifiableList(new ArrayList<>(body)); }

        /** Renders this template with the given variables (converted by {@link JinjaValue#fromJava}). */
        public String render(Map<String, ?> context) {
            return JinjaRuntime.render(body, context);
        }
    }

    // ================================================================= expressions

    /** Expression node. */
    abstract static class Expr {
        final int line;

        Expr(int line) { this.line = line; }

        abstract Object eval(Scope scope);
    }

    static final class Literal extends Expr {
        final Object value;

        Literal(int line, Object value) {
            super(line);
            this.value = value;
        }

        @Override Object eval(Scope scope) { return value; }
    }

    static final class Name extends Expr {
        final String name;

        Name(int line, String name) {
            super(line);
            this.name = name;
        }

        @Override Object eval(Scope scope) { return scope.resolve(name); }
    }

    /** List, tuple or dict display; every evaluation yields a fresh container. */
    static final class Display extends Expr {
        enum Kind { LIST, TUPLE, DICT }

        final Kind kind;
        final List<Expr> keys;
        final List<Expr> values;

        Display(int line, Kind kind, List<Expr> keys, List<Expr> values) {
            super(line);
            this.kind = kind;
            this.keys = keys == null ? null : Collections.unmodifiableList(new ArrayList<>(keys));
            this.values = Collections.unmodifiableList(new ArrayList<>(values));
        }

        @Override Object eval(Scope scope) {
            if (kind == Kind.DICT) {
                Map<Object, Object> map = new LinkedHashMap<>();
                for (int i = 0; i < keys.size(); i++) map.put(keys.get(i).eval(scope), values.get(i).eval(scope));
                return map;
            }
            List<Object> items = new ArrayList<>(values.size());
            for (Expr e : values) items.add(e.eval(scope));
            return kind == Kind.TUPLE ? new JinjaValue.Tuple(items) : items;
        }
    }

    static final class Unary extends Expr {
        final String op;
        final Expr operand;

        Unary(int line, String op, Expr operand) {
            super(line);
            this.op = op;
            this.operand = operand;
        }

        @Override Object eval(Scope scope) {
            Object v = operand.eval(scope);
            return "not".equals(op) ? !JinjaValue.truthy(v) : JinjaValue.unary(op, v);
        }
    }

    /** Arithmetic, concatenation and the short-circuit {@code and}/{@code or}. */
    static final class Binary extends Expr {
        final String op;
        final Expr left;
        final Expr right;

        Binary(int line, String op, Expr left, Expr right) {
            super(line);
            this.op = op;
            this.left = left;
            this.right = right;
        }

        @Override Object eval(Scope scope) {
            Object l = left.eval(scope);
            if ("and".equals(op)) return JinjaValue.truthy(l) ? right.eval(scope) : l;
            if ("or".equals(op)) return JinjaValue.truthy(l) ? l : right.eval(scope);
            return JinjaValue.binary(op, l, right.eval(scope));
        }
    }

    /** Python comparison chain: {@code a < b <= c} means {@code a < b and b <= c}. */
    static final class Compare extends Expr {
        final Expr first;
        final List<String> ops;
        final List<Expr> operands;

        Compare(int line, Expr first, List<String> ops, List<Expr> operands) {
            super(line);
            this.first = first;
            this.ops = Collections.unmodifiableList(new ArrayList<>(ops));
            this.operands = Collections.unmodifiableList(new ArrayList<>(operands));
        }

        @Override Object eval(Scope scope) {
            Object left = first.eval(scope);
            for (int i = 0; i < ops.size(); i++) {
                Object right = operands.get(i).eval(scope);
                if (!compare(ops.get(i), left, right)) return false;
                left = right;
            }
            return true;
        }

        private static boolean compare(String op, Object l, Object r) {
            switch (op) {
                case "==": return JinjaValue.pyEquals(l, r);
                case "!=": return !JinjaValue.pyEquals(l, r);
                case "<": return JinjaValue.compare(l, r) < 0;
                case "<=": return JinjaValue.compare(l, r) <= 0;
                case ">": return JinjaValue.compare(l, r) > 0;
                case ">=": return JinjaValue.compare(l, r) >= 0;
                case "in": return JinjaValue.contains(r, l);
                case "not in": return !JinjaValue.contains(r, l);
                default: throw new JinjaRuntime.TemplateException("unknown comparison operator '" + op + "'");
            }
        }
    }

    /** {@code a if cond else b}; a missing else branch yields Undefined. */
    static final class Conditional extends Expr {
        final Expr whenTrue;
        final Expr condition;
        final Expr whenFalse;

        Conditional(int line, Expr whenTrue, Expr condition, Expr whenFalse) {
            super(line);
            this.whenTrue = whenTrue;
            this.condition = condition;
            this.whenFalse = whenFalse;
        }

        @Override Object eval(Scope scope) {
            if (JinjaValue.truthy(condition.eval(scope))) return whenTrue.eval(scope);
            return whenFalse == null ? JinjaValue.UNDEFINED : whenFalse.eval(scope);
        }
    }

    static final class Attribute extends Expr {
        final Expr target;
        final String name;

        Attribute(int line, Expr target, String name) {
            super(line);
            this.target = target;
            this.name = name;
        }

        @Override Object eval(Scope scope) { return JinjaValue.getAttr(target.eval(scope), name); }
    }

    static final class Item extends Expr {
        final Expr target;
        final Expr key;

        Item(int line, Expr target, Expr key) {
            super(line);
            this.target = target;
            this.key = key;
        }

        @Override Object eval(Scope scope) { return JinjaValue.getItem(target.eval(scope), key.eval(scope)); }
    }

    /** {@code target[start:stop:step]}; absent bounds are null. */
    static final class Slice extends Expr {
        final Expr target;
        final Expr start;
        final Expr stop;
        final Expr step;

        Slice(int line, Expr target, Expr start, Expr stop, Expr step) {
            super(line);
            this.target = target;
            this.start = start;
            this.stop = stop;
            this.step = step;
        }

        @Override Object eval(Scope scope) {
            Object t = target.eval(scope);
            return JinjaValue.slice(t, start == null ? null : start.eval(scope),
                    stop == null ? null : stop.eval(scope), step == null ? null : step.eval(scope));
        }
    }

    /** Positional and keyword argument expressions of a call, filter or test. */
    static final class Args {
        static final Args NONE = new Args(Collections.<Expr>emptyList(), Collections.<String, Expr>emptyMap());

        final List<Expr> positional;
        final Map<String, Expr> keywords;

        Args(List<Expr> positional, Map<String, Expr> keywords) {
            this.positional = Collections.unmodifiableList(new ArrayList<>(positional));
            this.keywords = Collections.unmodifiableMap(new LinkedHashMap<>(keywords));
        }

        List<Object> evalPositional(Scope scope) {
            List<Object> out = new ArrayList<>(positional.size());
            for (Expr e : positional) out.add(e.eval(scope));
            return out;
        }

        Map<String, Object> evalKeywords(Scope scope) {
            if (keywords.isEmpty()) return new LinkedHashMap<>();
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<String, Expr> e : keywords.entrySet()) out.put(e.getKey(), e.getValue().eval(scope));
            return out;
        }
    }

    static final class Call extends Expr {
        final Expr callee;
        final Args args;

        Call(int line, Expr callee, Args args) {
            super(line);
            this.callee = callee;
            this.args = args;
        }

        @Override Object eval(Scope scope) {
            Object fn = callee.eval(scope);
            if (!(fn instanceof JinjaValue.Callable)) {
                throw new JinjaRuntime.TemplateException(describe(callee) + " is not callable (it is "
                        + (fn == JinjaValue.UNDEFINED ? "undefined" : "of type " + JinjaValue.typeName(fn)) + ")");
            }
            return ((JinjaValue.Callable) fn).call(args.evalPositional(scope), args.evalKeywords(scope), scope);
        }

        private static String describe(Expr e) {
            if (e instanceof Name) return "'" + ((Name) e).name + "'";
            if (e instanceof Attribute) return "'" + ((Attribute) e).name + "'";
            return "expression";
        }
    }

    /** One filter application: name plus arguments. */
    static final class FilterCall {
        final String name;
        final Args args;
        final int line;

        FilterCall(int line, String name, Args args) {
            this.line = line;
            this.name = name;
            this.args = args;
        }

        Object apply(Object input, Scope scope) {
            return JinjaFilters.apply(name, input, args.evalPositional(scope), args.evalKeywords(scope));
        }
    }

    static final class Filter extends Expr {
        final Expr input;
        final FilterCall filter;

        Filter(int line, Expr input, FilterCall filter) {
            super(line);
            this.input = input;
            this.filter = filter;
        }

        @Override Object eval(Scope scope) { return filter.apply(input.eval(scope), scope); }
    }

    static final class Test extends Expr {
        final Expr input;
        final String name;
        final Args args;
        final boolean negated;

        Test(int line, Expr input, String name, Args args, boolean negated) {
            super(line);
            this.input = input;
            this.name = name;
            this.args = args;
            this.negated = negated;
        }

        @Override Object eval(Scope scope) {
            boolean r = JinjaRuntime.test(name, input.eval(scope), args.evalPositional(scope));
            return negated != r;
        }
    }

    // ================================================================= assignment targets

    /** Left-hand side of {@code set} and {@code for}: a name, {@code ns.attr} or a tuple of names. */
    static final class Target {
        final String name;
        final String attribute;
        final List<Target> elements;

        private Target(String name, String attribute, List<Target> elements) {
            this.name = name;
            this.attribute = attribute;
            this.elements = elements == null ? null : Collections.unmodifiableList(new ArrayList<>(elements));
        }

        static Target name(String name) { return new Target(name, null, null); }

        static Target attribute(String object, String attribute) { return new Target(object, attribute, null); }

        static Target tuple(List<Target> elements) { return new Target(null, null, elements); }

        boolean isTuple() { return elements != null; }
    }

    // ================================================================= statements

    /** Statement node. */
    abstract static class Node {
        final int line;

        Node(int line) { this.line = line; }

        abstract void exec(Scope scope, StringBuilder out);
    }

    static final class Text extends Node {
        final String text;

        Text(int line, String text) {
            super(line);
            this.text = text;
        }

        @Override void exec(Scope scope, StringBuilder out) { out.append(text); }
    }

    static final class Output extends Node {
        final Expr expr;

        Output(int line, Expr expr) {
            super(line);
            this.expr = expr;
        }

        @Override void exec(Scope scope, StringBuilder out) { out.append(JinjaValue.str(expr.eval(scope))); }
    }

    /** {@code if}/{@code elif} branches; {@code elseBody} is empty when absent. */
    static final class If extends Node {
        final List<Expr> conditions;
        final List<List<Node>> bodies;
        final List<Node> elseBody;

        If(int line, List<Expr> conditions, List<List<Node>> bodies, List<Node> elseBody) {
            super(line);
            this.conditions = Collections.unmodifiableList(new ArrayList<>(conditions));
            List<List<Node>> copies = new ArrayList<>();
            for (List<Node> b : bodies) copies.add(Collections.unmodifiableList(new ArrayList<>(b)));
            this.bodies = Collections.unmodifiableList(copies);
            this.elseBody = Collections.unmodifiableList(new ArrayList<>(elseBody));
        }

        @Override void exec(Scope scope, StringBuilder out) {
            for (int i = 0; i < conditions.size(); i++) {
                if (JinjaValue.truthy(conditions.get(i).eval(scope))) {
                    JinjaRuntime.execBody(bodies.get(i), scope, out);
                    return;
                }
            }
            JinjaRuntime.execBody(elseBody, scope, out);
        }
    }

    static final class For extends Node {
        final Target target;
        final Expr iterable;
        final Expr filter;
        final List<Node> body;
        final List<Node> elseBody;

        For(int line, Target target, Expr iterable, Expr filter, List<Node> body, List<Node> elseBody) {
            super(line);
            this.target = target;
            this.iterable = iterable;
            this.filter = filter;
            this.body = Collections.unmodifiableList(new ArrayList<>(body));
            this.elseBody = Collections.unmodifiableList(new ArrayList<>(elseBody));
        }

        @Override void exec(Scope scope, StringBuilder out) { JinjaRuntime.execFor(this, scope, out); }
    }

    /** {@code set target = value}. */
    static final class Set extends Node {
        final Target target;
        final Expr value;

        Set(int line, Target target, Expr value) {
            super(line);
            this.target = target;
            this.value = value;
        }

        @Override void exec(Scope scope, StringBuilder out) { JinjaRuntime.assign(target, value.eval(scope), scope); }
    }

    /** {@code set target | filters %}body{% endset}. */
    static final class SetBlock extends Node {
        final Target target;
        final List<FilterCall> filters;
        final List<Node> body;

        SetBlock(int line, Target target, List<FilterCall> filters, List<Node> body) {
            super(line);
            this.target = target;
            this.filters = Collections.unmodifiableList(new ArrayList<>(filters));
            this.body = Collections.unmodifiableList(new ArrayList<>(body));
        }

        @Override void exec(Scope scope, StringBuilder out) {
            StringBuilder captured = new StringBuilder();
            JinjaRuntime.execBody(body, scope, captured);
            Object v = captured.toString();
            for (FilterCall f : filters) v = f.apply(v, scope);
            JinjaRuntime.assign(target, v, scope);
        }
    }

    static final class Macro extends Node {
        final String name;
        final List<String> params;
        final List<Expr> defaults;
        final List<Node> body;

        Macro(int line, String name, List<String> params, List<Expr> defaults, List<Node> body) {
            super(line);
            this.name = name;
            this.params = Collections.unmodifiableList(new ArrayList<>(params));
            this.defaults = Collections.unmodifiableList(new ArrayList<>(defaults));
            this.body = Collections.unmodifiableList(new ArrayList<>(body));
        }

        @Override void exec(Scope scope, StringBuilder out) { scope.set(name, new JinjaRuntime.MacroFunction(this)); }
    }

    /** {@code filter name(args) | ...} block: the rendered body passes through the filters. */
    static final class FilterBlock extends Node {
        final List<FilterCall> filters;
        final List<Node> body;

        FilterBlock(int line, List<FilterCall> filters, List<Node> body) {
            super(line);
            this.filters = Collections.unmodifiableList(new ArrayList<>(filters));
            this.body = Collections.unmodifiableList(new ArrayList<>(body));
        }

        @Override void exec(Scope scope, StringBuilder out) {
            StringBuilder captured = new StringBuilder();
            JinjaRuntime.execBody(body, scope, captured);
            Object v = captured.toString();
            for (FilterCall f : filters) v = f.apply(v, scope);
            out.append(JinjaValue.str(v));
        }
    }

    /** {@code generation} block (HF assistant-mask marker): renders its body unchanged. */
    static final class Generation extends Node {
        final List<Node> body;

        Generation(int line, List<Node> body) {
            super(line);
            this.body = Collections.unmodifiableList(new ArrayList<>(body));
        }

        @Override void exec(Scope scope, StringBuilder out) { JinjaRuntime.execBody(body, scope, out); }
    }

    /** {@code break} or {@code continue}. */
    static final class LoopControl extends Node {
        final boolean isBreak;

        LoopControl(int line, boolean isBreak) {
            super(line);
            this.isBreak = isBreak;
        }

        @Override void exec(Scope scope, StringBuilder out) {
            throw isBreak ? JinjaRuntime.BREAK : JinjaRuntime.CONTINUE;
        }
    }
}
