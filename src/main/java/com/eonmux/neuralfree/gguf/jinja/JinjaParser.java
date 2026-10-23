package com.eonmux.neuralfree.gguf.jinja;

import com.eonmux.neuralfree.gguf.jinja.JinjaAst.Args;
import com.eonmux.neuralfree.gguf.jinja.JinjaAst.Expr;
import com.eonmux.neuralfree.gguf.jinja.JinjaAst.FilterCall;
import com.eonmux.neuralfree.gguf.jinja.JinjaAst.Node;
import com.eonmux.neuralfree.gguf.jinja.JinjaAst.Target;
import com.eonmux.neuralfree.gguf.jinja.JinjaLexer.Kind;
import com.eonmux.neuralfree.gguf.jinja.JinjaLexer.Token;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Recursive-descent parser producing a {@link JinjaAst.Program}. Expression precedence follows
 * Jinja2 (jinja2/parser.py): conditional, {@code or}, {@code and}, {@code not}, comparisons and
 * {@code in}, {@code + -}, {@code ~}, {@code * / // %}, {@code **}, unary sign, then primaries with
 * attribute/item/slice/call postfixes and {@code |filter} / {@code is test} chains.
 *
 * <p>Statements: {@code if/elif/else}, {@code for} (tuple targets, {@code if} filter, {@code else}),
 * {@code set} (name, {@code ns.attr}, tuple, block form with filters), {@code macro},
 * {@code filter} blocks, {@code generation} blocks, {@code break}, {@code continue}. Template
 * inheritance, includes, imports and {@code call} blocks are rejected with a clear error.
 */
public final class JinjaParser {

    private static final Set<String> UNSUPPORTED = new HashSet<>(Arrays.asList(
            "call", "include", "import", "from", "extends", "block", "with", "do", "autoescape"));
    private static final Set<String> LITERAL_NAMES = new HashSet<>(Arrays.asList(
            "true", "True", "false", "False", "none", "None"));
    /** Names that end a single-argument test ({@code x is divisibleby 3}) instead of starting its argument. */
    private static final Set<String> TEST_ARG_STOP = new HashSet<>(Arrays.asList(
            "else", "or", "and", "if", "in", "is", "not", "recursive"));

    private final List<Token> tokens;
    private int pos;
    private int loopDepth;

    private JinjaParser(List<Token> tokens) { this.tokens = tokens; }

    /** Parses template source. */
    public static JinjaAst.Program parse(String source) {
        JinjaParser p = new JinjaParser(JinjaLexer.tokenize(source));
        List<Node> body = p.body(Collections.<String>emptySet());
        return new JinjaAst.Program(body);
    }

    // ================================================================= token helpers

    private Token peek() { return tokens.get(pos); }

    private Token peekAt(int offset) {
        int i = Math.min(pos + offset, tokens.size() - 1);
        return tokens.get(i);
    }

    private Token next() {
        Token t = tokens.get(pos);
        if (t.kind != Kind.EOF) pos++;
        return t;
    }

    private boolean acceptOp(String op) {
        if (peek().isOp(op)) {
            pos++;
            return true;
        }
        return false;
    }

    private boolean acceptName(String name) {
        if (peek().isName(name)) {
            pos++;
            return true;
        }
        return false;
    }

    private void expectOp(String op) {
        if (!acceptOp(op)) throw error("expected '" + op + "', got " + peek());
    }

    private void expectName(String name) {
        if (!acceptName(name)) throw error("expected '" + name + "', got " + peek());
    }

    private String expectIdentifier(String what) {
        Token t = peek();
        if (t.kind != Kind.NAME) throw error("expected " + what + ", got " + t);
        pos++;
        return t.text;
    }

    private void expectStmtEnd() {
        if (peek().kind != Kind.STMT_CLOSE) throw error("expected end of statement block, got " + peek());
        pos++;
    }

    private JinjaRuntime.TemplateException error(String message) {
        return new JinjaRuntime.TemplateException(peek().line, "syntax error: " + message);
    }

    // ================================================================= statements

    /**
     * Parses nodes until a statement whose keyword is in {@code endNames}; on return the position is
     * at that keyword (its {@code {%} already consumed). With no end names, parses to the end.
     */
    private List<Node> body(Set<String> endNames) {
        List<Node> nodes = new ArrayList<>();
        while (true) {
            Token t = peek();
            switch (t.kind) {
                case EOF:
                    if (!endNames.isEmpty()) throw error("unexpected end of template, expected one of " + endNames);
                    return nodes;
                case TEXT:
                    pos++;
                    nodes.add(new JinjaAst.Text(t.line, t.text));
                    break;
                case EXPR_OPEN: {
                    pos++;
                    Expr e = tuple(true);
                    if (peek().kind != Kind.EXPR_CLOSE) throw error("expected '}}', got " + peek());
                    pos++;
                    nodes.add(new JinjaAst.Output(t.line, e));
                    break;
                }
                case STMT_OPEN: {
                    Token kw = peekAt(1);
                    if (kw.kind == Kind.NAME && endNames.contains(kw.text)) {
                        pos += 2;
                        return nodes;
                    }
                    pos++;
                    nodes.add(statement());
                    break;
                }
                default:
                    throw error("unexpected " + t);
            }
        }
    }

    private Node statement() {
        Token kw = peek();
        if (kw.kind != Kind.NAME) throw error("expected statement name, got " + kw);
        pos++;
        switch (kw.text) {
            case "if": return ifStatement(kw.line);
            case "for": return forStatement(kw.line);
            case "set": return setStatement(kw.line);
            case "macro": return macroStatement(kw.line);
            case "filter": {
                List<FilterCall> filters = filterChain(false);
                expectStmtEnd();
                List<Node> b = body(set("endfilter"));
                expectStmtEnd();
                return new JinjaAst.FilterBlock(kw.line, filters, b);
            }
            case "generation": {
                expectStmtEnd();
                List<Node> b = body(set("endgeneration"));
                expectStmtEnd();
                return new JinjaAst.Generation(kw.line, b);
            }
            case "break":
            case "continue":
                if (loopDepth == 0) throw new JinjaRuntime.TemplateException(kw.line, "syntax error: '" + kw.text + "' outside of a loop");
                expectStmtEnd();
                return new JinjaAst.LoopControl(kw.line, "break".equals(kw.text));
            default:
                if (UNSUPPORTED.contains(kw.text)) {
                    throw new JinjaRuntime.TemplateException(kw.line, "statement '" + kw.text + "' is not supported in chat templates");
                }
                if (kw.text.startsWith("end") || "else".equals(kw.text) || "elif".equals(kw.text)) {
                    throw new JinjaRuntime.TemplateException(kw.line, "syntax error: unexpected '" + kw.text + "'");
                }
                throw new JinjaRuntime.TemplateException(kw.line, "syntax error: unknown statement '" + kw.text + "'");
        }
    }

    private static Set<String> set(String... names) { return new HashSet<>(Arrays.asList(names)); }

    private Node ifStatement(int line) {
        List<Expr> conditions = new ArrayList<>();
        List<List<Node>> bodies = new ArrayList<>();
        List<Node> elseBody = Collections.emptyList();
        conditions.add(tuple(false));
        expectStmtEnd();
        while (true) {
            bodies.add(body(set("elif", "else", "endif")));
            String kw = tokens.get(pos - 1).text;
            if ("elif".equals(kw)) {
                conditions.add(tuple(false));
                expectStmtEnd();
                continue;
            }
            expectStmtEnd();
            if ("else".equals(kw)) {
                elseBody = body(set("endif"));
                expectStmtEnd();
            }
            return new JinjaAst.If(line, conditions, bodies, elseBody);
        }
    }

    private Node forStatement(int line) {
        Target target = target(false, "in");
        expectName("in");
        Expr iterable = tuple(false);
        Expr filter = null;
        if (acceptName("if")) filter = expression(true);
        if (peek().isName("recursive")) throw error("recursive for loops are not supported");
        expectStmtEnd();
        loopDepth++;
        List<Node> b;
        try {
            b = body(set("endfor", "else"));
        } finally {
            loopDepth--;
        }
        List<Node> elseBody = Collections.emptyList();
        if ("else".equals(tokens.get(pos - 1).text)) {
            expectStmtEnd();
            elseBody = body(set("endfor"));
        }
        expectStmtEnd();
        return new JinjaAst.For(line, target, iterable, filter, b, elseBody);
    }

    private Node setStatement(int line) {
        Target target = target(true, null);
        if (acceptOp("=")) {
            Expr value = tuple(true);
            expectStmtEnd();
            return new JinjaAst.Set(line, target, value);
        }
        if (target.isTuple() || target.attribute != null) throw error("block set requires a single variable name");
        List<FilterCall> filters = peek().isOp("|") ? filterChain(true) : Collections.<FilterCall>emptyList();
        expectStmtEnd();
        List<Node> b = body(set("endset"));
        expectStmtEnd();
        return new JinjaAst.SetBlock(line, target, filters, b);
    }

    private Node macroStatement(int line) {
        String name = expectIdentifier("macro name");
        List<String> params = new ArrayList<>();
        List<Expr> defaults = new ArrayList<>();
        expectOp("(");
        while (!acceptOp(")")) {
            if (!params.isEmpty()) expectOp(",");
            if (acceptOp(")")) break;
            String param = expectIdentifier("parameter name");
            if (params.contains(param)) throw error("duplicate macro parameter '" + param + "'");
            params.add(param);
            defaults.add(acceptOp("=") ? expression(true) : null);
        }
        expectStmtEnd();
        int savedDepth = loopDepth;
        loopDepth = 0;
        List<Node> b;
        try {
            b = body(set("endmacro"));
        } finally {
            loopDepth = savedDepth;
        }
        if (peek().kind == Kind.NAME) pos++;
        expectStmtEnd();
        return new JinjaAst.Macro(line, name, params, defaults, b);
    }

    /**
     * Assignment target: a name, {@code ns.attr} (set only) or a comma-separated / parenthesized
     * tuple of names. {@code stopName} ends a bare tuple (the {@code in} of a for loop).
     */
    private Target target(boolean allowAttribute, String stopName) {
        List<Target> parts = new ArrayList<>();
        boolean isTuple = false;
        while (true) {
            parts.add(targetElement(allowAttribute));
            if (!acceptOp(",")) break;
            isTuple = true;
            if (peek().isOp("=") || (stopName != null && peek().isName(stopName))) break;
        }
        return isTuple ? Target.tuple(parts) : parts.get(0);
    }

    private Target targetElement(boolean allowAttribute) {
        if (acceptOp("(")) {
            Target inner = target(false, null);
            expectOp(")");
            return inner;
        }
        String name = expectIdentifier("variable name");
        if (LITERAL_NAMES.contains(name)) throw error("cannot assign to '" + name + "'");
        if (allowAttribute && acceptOp(".")) return Target.attribute(name, expectIdentifier("attribute name"));
        return Target.name(name);
    }

    /** {@code name(args) | name ...}; {@code leadingPipe} when the chain starts with {@code |}. */
    private List<FilterCall> filterChain(boolean leadingPipe) {
        List<FilterCall> out = new ArrayList<>();
        if (!leadingPipe) out.add(filterCall());
        while (acceptOp("|")) out.add(filterCall());
        return out;
    }

    private FilterCall filterCall() {
        int line = peek().line;
        String name = expectIdentifier("filter name");
        if (!JinjaFilters.exists(name)) throw new JinjaRuntime.TemplateException(line, "unknown filter '" + name + "'");
        Args args = peek().isOp("(") ? callArgs() : Args.NONE;
        return new FilterCall(line, name, args);
    }

    // ================================================================= expressions

    /**
     * Comma-separated expressions; more than one (or a trailing comma) builds a tuple.
     * {@code withCond} allows the inline {@code if/else}.
     */
    private Expr tuple(boolean withCond) {
        int line = peek().line;
        List<Expr> items = new ArrayList<>();
        boolean isTuple = false;
        while (!isTupleEnd()) {
            items.add(expression(withCond));
            if (!acceptOp(",")) break;
            isTuple = true;
        }
        if (!isTuple) {
            if (items.isEmpty()) throw error("expected an expression, got " + peek());
            return items.get(0);
        }
        return new JinjaAst.Display(line, JinjaAst.Display.Kind.TUPLE, null, items);
    }

    private boolean isTupleEnd() {
        Token t = peek();
        return t.kind == Kind.EXPR_CLOSE || t.kind == Kind.STMT_CLOSE || t.kind == Kind.EOF || t.isOp(")")
                || t.isName("if") || t.isName("recursive");
    }

    private Expr expression(boolean withCond) {
        Expr e = or();
        if (!withCond) return e;
        while (peek().isName("if")) {
            int line = next().line;
            Expr cond = or();
            Expr otherwise = acceptName("else") ? expression(true) : null;
            e = new JinjaAst.Conditional(line, e, cond, otherwise);
        }
        return e;
    }

    private Expr or() {
        Expr e = and();
        while (peek().isName("or")) {
            int line = next().line;
            e = new JinjaAst.Binary(line, "or", e, and());
        }
        return e;
    }

    private Expr and() {
        Expr e = not();
        while (peek().isName("and")) {
            int line = next().line;
            e = new JinjaAst.Binary(line, "and", e, not());
        }
        return e;
    }

    private Expr not() {
        if (peek().isName("not")) {
            int line = next().line;
            return new JinjaAst.Unary(line, "not", not());
        }
        return compare();
    }

    private Expr compare() {
        int line = peek().line;
        Expr first = math1();
        List<String> ops = new ArrayList<>();
        List<Expr> operands = new ArrayList<>();
        while (true) {
            Token t = peek();
            String op = null;
            if (t.kind == Kind.OP && ("==".equals(t.text) || "!=".equals(t.text) || "<".equals(t.text)
                    || ">".equals(t.text) || "<=".equals(t.text) || ">=".equals(t.text))) {
                op = t.text;
                pos++;
            } else if (t.isName("in")) {
                op = "in";
                pos++;
            } else if (t.isName("not") && peekAt(1).isName("in")) {
                op = "not in";
                pos += 2;
            }
            if (op == null) break;
            ops.add(op);
            operands.add(math1());
        }
        return ops.isEmpty() ? first : new JinjaAst.Compare(line, first, ops, operands);
    }

    private Expr math1() {
        Expr e = concat();
        while (peek().isOp("+") || peek().isOp("-")) {
            Token op = next();
            e = new JinjaAst.Binary(op.line, op.text, e, concat());
        }
        return e;
    }

    private Expr concat() {
        Expr e = math2();
        while (peek().isOp("~")) {
            int line = next().line;
            e = new JinjaAst.Binary(line, "~", e, math2());
        }
        return e;
    }

    private Expr math2() {
        Expr e = pow();
        while (peek().isOp("*") || peek().isOp("/") || peek().isOp("//") || peek().isOp("%")) {
            Token op = next();
            e = new JinjaAst.Binary(op.line, op.text, e, pow());
        }
        return e;
    }

    private Expr pow() {
        Expr e = unary(true);
        while (peek().isOp("**")) {
            int line = next().line;
            e = new JinjaAst.Binary(line, "**", e, unary(true));
        }
        return e;
    }

    /** Unary sign binds looser than filters on its operand only through the outer chain (jinja2 parse_unary). */
    private Expr unary(boolean withFilter) {
        Token t = peek();
        Expr e;
        if (t.isOp("-") || t.isOp("+")) {
            pos++;
            e = new JinjaAst.Unary(t.line, t.text, unary(false));
        } else {
            e = primary();
        }
        e = postfix(e);
        return withFilter ? filterAndTest(e) : e;
    }

    private Expr primary() {
        Token t = next();
        switch (t.kind) {
            case NAME:
                switch (t.text) {
                    case "true": case "True": return new JinjaAst.Literal(t.line, Boolean.TRUE);
                    case "false": case "False": return new JinjaAst.Literal(t.line, Boolean.FALSE);
                    case "none": case "None": return new JinjaAst.Literal(t.line, null);
                    default: return new JinjaAst.Name(t.line, t.text);
                }
            case STRING: {
                StringBuilder sb = new StringBuilder(t.text);
                while (peek().kind == Kind.STRING) sb.append(next().text);
                return new JinjaAst.Literal(t.line, sb.toString());
            }
            case INT:
                try {
                    return new JinjaAst.Literal(t.line, Long.parseLong(t.text));
                } catch (NumberFormatException e) {
                    throw new JinjaRuntime.TemplateException(t.line, "syntax error: integer literal out of range: " + t.text);
                }
            case FLOAT:
                return new JinjaAst.Literal(t.line, Double.parseDouble(t.text));
            case OP:
                if ("(".equals(t.text)) return parenthesized(t.line);
                if ("[".equals(t.text)) return listDisplay(t.line);
                if ("{".equals(t.text)) return dictDisplay(t.line);
                break;
            default:
                break;
        }
        throw new JinjaRuntime.TemplateException(t.line, "syntax error: unexpected " + t);
    }

    private Expr parenthesized(int line) {
        if (acceptOp(")")) return new JinjaAst.Display(line, JinjaAst.Display.Kind.TUPLE, null, Collections.<Expr>emptyList());
        List<Expr> items = new ArrayList<>();
        boolean isTuple = false;
        while (true) {
            items.add(expression(true));
            if (!acceptOp(",")) break;
            isTuple = true;
            if (peek().isOp(")")) break;
        }
        expectOp(")");
        return isTuple ? new JinjaAst.Display(line, JinjaAst.Display.Kind.TUPLE, null, items) : items.get(0);
    }

    private Expr listDisplay(int line) {
        List<Expr> items = new ArrayList<>();
        while (!acceptOp("]")) {
            if (!items.isEmpty()) {
                expectOp(",");
                if (acceptOp("]")) break;
            }
            items.add(expression(true));
        }
        return new JinjaAst.Display(line, JinjaAst.Display.Kind.LIST, null, items);
    }

    private Expr dictDisplay(int line) {
        List<Expr> keys = new ArrayList<>();
        List<Expr> values = new ArrayList<>();
        while (!acceptOp("}")) {
            if (!keys.isEmpty()) {
                expectOp(",");
                if (acceptOp("}")) break;
            }
            keys.add(expression(true));
            expectOp(":");
            values.add(expression(true));
        }
        return new JinjaAst.Display(line, JinjaAst.Display.Kind.DICT, keys, values);
    }

    private Expr postfix(Expr target) {
        Expr e = target;
        while (true) {
            Token t = peek();
            if (t.isOp(".")) {
                pos++;
                Token attr = next();
                if (attr.kind == Kind.NAME) e = new JinjaAst.Attribute(t.line, e, attr.text);
                else if (attr.kind == Kind.INT) e = new JinjaAst.Item(t.line, e, new JinjaAst.Literal(attr.line, Long.parseLong(attr.text)));
                else throw error("expected attribute name after '.', got " + attr);
            } else if (t.isOp("[")) {
                pos++;
                e = subscript(t.line, e);
            } else if (t.isOp("(")) {
                e = new JinjaAst.Call(t.line, e, callArgs());
            } else {
                return e;
            }
        }
    }

    /** {@code [index]} or {@code [start:stop:step]} after the opening bracket. */
    private Expr subscript(int line, Expr target) {
        Expr start = null;
        if (!peek().isOp(":")) {
            start = tuple(true);
            if (acceptOp("]")) return new JinjaAst.Item(line, target, start);
        }
        expectOp(":");
        Expr stop = null;
        Expr step = null;
        if (!peek().isOp("]") && !peek().isOp(":")) stop = expression(true);
        if (acceptOp(":") && !peek().isOp("]")) step = expression(true);
        expectOp("]");
        return new JinjaAst.Slice(line, target, start, stop, step);
    }

    /** Applies trailing {@code |filter}, {@code is test} and call postfixes. */
    private Expr filterAndTest(Expr input) {
        Expr e = input;
        while (true) {
            Token t = peek();
            if (t.isOp("|")) {
                pos++;
                e = new JinjaAst.Filter(t.line, e, filterCall());
            } else if (t.isName("is")) {
                pos++;
                boolean negated = acceptName("not");
                Token nameTok = peek();
                String name = expectIdentifier("test name");
                if (!JinjaRuntime.isTest(name)) throw new JinjaRuntime.TemplateException(nameTok.line, "unknown test '" + name + "'");
                Args args = Args.NONE;
                if (peek().isOp("(")) {
                    args = callArgs();
                } else if (startsTestArgument(peek())) {
                    args = new Args(Collections.singletonList(postfix(primary())), Collections.<String, Expr>emptyMap());
                }
                e = new JinjaAst.Test(t.line, e, name, args, negated);
            } else if (t.isOp("(")) {
                e = new JinjaAst.Call(t.line, e, callArgs());
            } else {
                return e;
            }
        }
    }

    private static boolean startsTestArgument(Token t) {
        switch (t.kind) {
            case STRING: case INT: case FLOAT: return true;
            case NAME: return !TEST_ARG_STOP.contains(t.text);
            case OP: return "[".equals(t.text) || "{".equals(t.text);
            default: return false;
        }
    }

    /** {@code (a, b, key=value)}; a trailing comma is allowed. */
    private Args callArgs() {
        expectOp("(");
        List<Expr> positional = new ArrayList<>();
        Map<String, Expr> keywords = new LinkedHashMap<>();
        boolean first = true;
        while (!acceptOp(")")) {
            if (!first) {
                expectOp(",");
                if (acceptOp(")")) break;
            }
            first = false;
            if (peek().isOp("*") || peek().isOp("**")) throw error("argument unpacking is not supported");
            if (peek().kind == Kind.NAME && peekAt(1).isOp("=")) {
                String key = next().text;
                pos++;
                if (keywords.containsKey(key)) throw error("keyword argument repeated: " + key);
                keywords.put(key, expression(true));
            } else {
                if (!keywords.isEmpty()) throw error("positional argument follows keyword argument");
                positional.add(expression(true));
            }
        }
        return new Args(positional, keywords);
    }
}
