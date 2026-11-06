package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.grammar.GbnfExclusion;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A small parsing expression grammar whose expressions both parse text and emit a GBNF grammar for it: llama.cpp's
 * {@code common_peg_parser} ({@code common/peg-parser.cpp}) with the subset the chat autoparser builds.
 *
 * <p>{@link #match} runs an expression at one position and reports success, failure, or that the input ended while a
 * match was still possible ({@code NEED_MORE_INPUT}, the partial-stream state). Captures produced by {@link Tag} nodes
 * name the spans a caller reads — the function name, the argument text — the way llama.cpp's tagged AST nodes do.
 * {@link #gbnf} writes the same expression as GBNF rules through a {@link GbnfEmitter} registry; lookaheads emit nothing,
 * as in llama.cpp, and {@code until} emits an exclusion automaton ({@link GbnfExclusion}). Immutable.
 */
public abstract class Peg {

    /** A match outcome. */
    public enum Status { OK, NEED_MORE, FAIL }

    /** A named span of the input produced by a {@link Tag}. */
    public static final class Capture {
        private final String tag;
        private final int start;
        private final int end;

        Capture(String tag, int start, int end) {
            this.tag = tag;
            this.start = start;
            this.end = end;
        }

        public String tag() { return tag; }
        public int start() { return start; }
        public int end() { return end; }

        /** The captured text. */
        public String text(CharSequence in) {
            return in.subSequence(start, end).toString();
        }

        @Override public String toString() {
            return tag + "[" + start + "," + end + ")";
        }
    }

    /** The result of a match: its outcome, where it ended and the captures it produced. */
    public static final class Result {
        private final Status status;
        private final int end;
        private final List<Capture> captures;

        private Result(Status status, int end, List<Capture> captures) {
            this.status = status;
            this.end = end;
            this.captures = captures;
        }

        static Result ok(int end, List<Capture> captures) {
            return new Result(Status.OK, end, captures);
        }

        static Result needMore(int end, List<Capture> captures) {
            return new Result(Status.NEED_MORE, end, captures);
        }

        static Result fail() {
            return new Result(Status.FAIL, 0, Collections.<Capture>emptyList());
        }

        public Status status() { return status; }
        public boolean ok() { return status == Status.OK; }
        public boolean needMore() { return status == Status.NEED_MORE; }
        public int end() { return end; }
        public List<Capture> captures() { return captures; }

        /** The capture tagged {@code tag}, or null. */
        public Capture capture(String tag) {
            for (Capture c : captures) {
                if (c.tag().equals(tag)) {
                    return c;
                }
            }
            return null;
        }

        /** The text of the capture tagged {@code tag}, or null. */
        public String text(CharSequence input, String tag) {
            Capture c = capture(tag);
            return c == null ? null : c.text(input);
        }

        /** Every capture tagged {@code tag}, in order. */
        public List<Capture> captures(String tag) {
            List<Capture> out = new ArrayList<>();
            for (Capture c : captures) {
                if (c.tag().equals(tag)) {
                    out.add(c);
                }
            }
            return out;
        }
    }

    /** Whether the expression can match anything at all. */
    public final boolean empty() {
        return this instanceof Eps;
    }

    /**
     * Matches at {@code pos}.
     *
     * @param partial whether the input may still grow: a literal the input ends inside yields {@link Status#NEED_MORE}
     */
    public abstract Result match(CharSequence input, int pos, boolean partial);

    /** The expression's GBNF right-hand text, adding any rules it needs to {@code grammar}. */
    public abstract String gbnf(GbnfEmitter grammar);

    // ================================================================== construction

    /** The empty expression. */
    public static Peg eps() {
        return Eps.INSTANCE;
    }

    /** The empty expression ({@code epsilon}). */
    public static Peg epsilon() {
        return eps();
    }

    /** A literal. */
    public static Peg lit(String text) {
        return text.isEmpty() ? eps() : new Lit(text);
    }

    /** A literal ({@code literal}). */
    public static Peg literal(String text) {
        return lit(text);
    }

    /** Flexible whitespace, the grammar's {@code space} rule. */
    public static Peg space() {
        return Space.INSTANCE;
    }

    /** Any one code point. */
    public static Peg any() {
        return Any.INSTANCE;
    }

    /** The end of the input. */
    public static Peg end() {
        return End.INSTANCE;
    }

    /** Text up to the first of {@code delimiters}, or to the end when none occurs. */
    public static Peg until(String... delimiters) {
        return new Until(Arrays.asList(delimiters));
    }

    /** A choice; the first alternative that matches wins. */
    public static Peg choice(Peg... alternatives) {
        List<Peg> list = new ArrayList<>();
        for (Peg p : alternatives) {
            if (p instanceof Choice) {
                list.addAll(((Choice) p).alternatives);
            } else {
                list.add(p);
            }
        }
        return list.size() == 1 ? list.get(0) : new Choice(list);
    }

    /** An ordered sequence ({@code sequence}). */
    public static Peg sequence(Peg... parts) {
        return seq(parts);
    }

    /** An ordered sequence. */
    public static Peg seq(Peg... parts) {
        List<Peg> list = new ArrayList<>();
        for (Peg p : parts) {
            if (p == null || p.empty()) {
                continue;
            }
            if (p instanceof Seq) {
                list.addAll(((Seq) p).parts);
            } else {
                list.add(p);
            }
        }
        if (list.isEmpty()) {
            return eps();
        }
        return list.size() == 1 ? list.get(0) : new Seq(list);
    }

    /** Zero or one. */
    public static Peg opt(Peg child) {
        return child.empty() ? child : new Opt(child);
    }

    /** Zero or more. */
    public static Peg star(Peg child) {
        return child.empty() ? eps() : new Star(child);
    }

    /** One or more. */
    public static Peg plus(Peg child) {
        return new Plus(child);
    }

    /** Captures what {@code child} matches under {@code tag}. */
    public static Peg tag(String tag, Peg child) {
        return new Tag(tag, child);
    }

    /** Positive lookahead: matches without consuming. */
    public static Peg peek(Peg child) {
        return new Peek(child);
    }

    /** Negative lookahead: matches without consuming when the child does not. */
    public static Peg not(Peg child) {
        return new Not(child);
    }

    /** One JSON value, scanned by its delimiters. */
    public static Peg json() {
        return JsonValue.INSTANCE;
    }

    /** One JSON value ({@code jsonValue}). */
    public static Peg jsonValue() {
        return json();
    }

    /** One JSON value held to {@code schemaJson}: the argument text of one tool parameter. */
    public static Peg schema(String name, String schemaJson) {
        return new SchemaValue(name, schemaJson);
    }

    /** A named grammar rule; parsing delegates to the child. */
    public static Peg rule(String name, Peg child) {
        return new Rule(name, child);
    }

    /**
     * A reference to a rule another node emitted: the grammar writes {@code gbnf} verbatim and parsing matches
     * nothing. For grammar-only structures whose parse form is data-driven.
     */
    public static Peg raw(String gbnf) {
        return new Raw(gbnf);
    }

    // ================================================================== nodes

    private static final class Eps extends Peg {
        static final Eps INSTANCE = new Eps();

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            return Result.ok(pos, Collections.<Capture>emptyList());
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return "";
        }
    }

    private static final class Raw extends Peg {
        private final String gbnf;

        Raw(String gbnf) {
            this.gbnf = gbnf;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            return Result.ok(pos, Collections.<Capture>emptyList());
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return gbnf;
        }
    }

    private static final class Lit extends Peg {
        private final String text;

        Lit(String text) {
            this.text = text;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            if (regionMatches(input, pos, text)) {
                return Result.ok(pos + text.length(), Collections.<Capture>emptyList());
            }
            if (partial && pos + text.length() > input.length() && regionMatches(input, pos,
                    text.substring(0, Math.min(text.length(), input.length() - pos)))) {
                return Result.needMore(input.length(), Collections.<Capture>emptyList());
            }
            return Result.fail();
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return "\"" + MarkerDiff.literal(text) + "\"";
        }
    }

    private static final class Space extends Peg {
        static final Space INSTANCE = new Space();

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            int at = pos;
            while (at < input.length() && Character.isWhitespace(input.charAt(at))) {
                at++;
            }
            return Result.ok(at, Collections.<Capture>emptyList());
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return "space";
        }
    }

    private static final class Any extends Peg {
        static final Any INSTANCE = new Any();

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            if (pos < input.length()) {
                return Result.ok(pos + Character.charCount(Character.codePointAt(input, pos)),
                        Collections.<Capture>emptyList());
            }
            return partial ? Result.needMore(pos, Collections.<Capture>emptyList()) : Result.fail();
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return ".";
        }
    }

    private static final class End extends Peg {
        static final End INSTANCE = new End();

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            return pos == input.length() ? Result.ok(pos, Collections.<Capture>emptyList()) : Result.fail();
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return "";
        }
    }

    private static final class Until extends Peg {
        private final List<String> delimiters;

        Until(List<String> delimiters) {
            this.delimiters = Collections.unmodifiableList(new ArrayList<>(delimiters));
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            int earliest = -1;
            for (String d : delimiters) {
                if (d.isEmpty()) {
                    continue;
                }
                int at = indexOf(input, d, pos);
                if (at >= 0 && (earliest < 0 || at < earliest)) {
                    earliest = at;
                }
            }
            if (earliest >= 0) {
                return Result.ok(earliest, Collections.<Capture>emptyList());
            }
            if (partial && endsInsideDelimiter(input, pos)) {
                return Result.needMore(input.length(), Collections.<Capture>emptyList());
            }
            return Result.ok(input.length(), Collections.<Capture>emptyList());
        }

        private boolean endsInsideDelimiter(CharSequence input, int pos) {
            for (String d : delimiters) {
                for (int k = Math.min(d.length() - 1, input.length() - pos); k > 0; k--) {
                    if (regionMatches(input, input.length() - k, d.substring(0, k))) {
                        return true;
                    }
                }
            }
            return false;
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return delimiters.isEmpty() ? ".*" : grammar.excluding(delimiters);
        }
    }

    private static final class Choice extends Peg {
        final List<Peg> alternatives;

        Choice(List<Peg> alternatives) {
            this.alternatives = Collections.unmodifiableList(new ArrayList<>(alternatives));
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            Result needMore = null;
            for (Peg alternative : alternatives) {
                Result m = alternative.match(input, pos, partial);
                if (m.ok()) {
                    return m;
                }
                if (m.needMore() && needMore == null) {
                    needMore = m;
                }
            }
            return needMore != null ? needMore : Result.fail();
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            List<String> parts = new ArrayList<>();
            for (Peg alternative : alternatives) {
                parts.add(alternative.gbnf(grammar));
            }
            return "( " + String.join(" | ", parts) + " )";
        }
    }

    private static final class Seq extends Peg {
        final List<Peg> parts;

        Seq(List<Peg> parts) {
            this.parts = Collections.unmodifiableList(new ArrayList<>(parts));
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            int at = pos;
            List<Capture> captures = new ArrayList<>();
            for (Peg part : parts) {
                Result m = part.match(input, at, partial);
                if (m.status() == Status.FAIL) {
                    return Result.fail();
                }
                captures.addAll(m.captures());
                at = m.end();
                if (m.needMore()) {
                    return Result.needMore(at, captures);
                }
            }
            return Result.ok(at, captures);
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            List<String> parts = new ArrayList<>();
            for (Peg part : this.parts) {
                String text = part.gbnf(grammar);
                if (text.isEmpty()) {
                    continue;
                }
                parts.add(needsGroup(part) ? "( " + text + " )" : text);
            }
            return String.join(" ", parts);
        }
    }

    private static final class Opt extends Peg {
        private final Peg child;

        Opt(Peg child) {
            this.child = child;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            Result m = child.match(input, pos, partial);
            return m.status() == Status.FAIL ? Result.ok(pos, Collections.<Capture>emptyList()) : m;
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            String text = child.gbnf(grammar);
            if (text.isEmpty()) {
                return "";
            }
            return (needsGroup(child) ? "( " + text + " )" : text) + "?";
        }
    }

    private static final class Star extends Peg {
        private final Peg child;

        Star(Peg child) {
            this.child = child;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            int at = pos;
            List<Capture> captures = new ArrayList<>();
            while (true) {
                Result m = child.match(input, at, partial);
                if (m.status() == Status.FAIL) {
                    break;
                }
                if (m.needMore()) {
                    captures.addAll(m.captures());
                    return Result.needMore(m.end(), captures);
                }
                if (m.end() == at) {
                    break;
                }
                captures.addAll(m.captures());
                at = m.end();
            }
            return Result.ok(at, captures);
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            String text = child.gbnf(grammar);
            if (text.isEmpty()) {
                return "";
            }
            return (needsGroup(child) ? "( " + text + " )" : text) + "*";
        }
    }

    private static final class Plus extends Peg {
        private final Peg child;

        Plus(Peg child) {
            this.child = child;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            Result first = child.match(input, pos, partial);
            if (first.status() == Status.FAIL) {
                return Result.fail();
            }
            if (first.needMore()) {
                return first;
            }
            List<Capture> captures = new ArrayList<>(first.captures());
            int at = first.end();
            while (true) {
                Result m = child.match(input, at, partial);
                if (m.status() == Status.FAIL) {
                    break;
                }
                if (m.needMore()) {
                    captures.addAll(m.captures());
                    return Result.needMore(m.end(), captures);
                }
                if (m.end() == at) {
                    break;
                }
                captures.addAll(m.captures());
                at = m.end();
            }
            return Result.ok(at, captures);
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            String text = child.gbnf(grammar);
            if (text.isEmpty()) {
                return "";
            }
            return (needsGroup(child) ? "( " + text + " )" : text) + "+";
        }
    }

    private static final class Tag extends Peg {
        private final String tag;
        private final Peg child;

        Tag(String tag, Peg child) {
            this.tag = tag;
            this.child = child;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            Result m = child.match(input, pos, partial);
            if (m.status() == Status.FAIL) {
                return m;
            }
            List<Capture> captures = new ArrayList<>(m.captures());
            captures.add(new Capture(tag, pos, m.end()));
            return m.needMore() ? Result.needMore(m.end(), captures) : Result.ok(m.end(), captures);
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return child.gbnf(grammar);
        }
    }

    private static final class Peek extends Peg {
        private final Peg child;

        Peek(Peg child) {
            this.child = child;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            Result m = child.match(input, pos, partial);
            if (m.status() == Status.FAIL) {
                return m;
            }
            return m.needMore() ? Result.needMore(pos, Collections.<Capture>emptyList())
                    : Result.ok(pos, Collections.<Capture>emptyList());
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return "";
        }
    }

    private static final class Not extends Peg {
        private final Peg child;

        Not(Peg child) {
            this.child = child;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            Result m = child.match(input, pos, partial);
            if (m.status() == Status.NEED_MORE) {
                return m;
            }
            return m.ok() ? Result.fail() : Result.ok(pos, Collections.<Capture>emptyList());
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return "";
        }
    }

    /** A balanced JSON value: object, array, string or scalar, with escapes and nesting respected. */
    private static final class JsonValue extends Peg {
        static final JsonValue INSTANCE = new JsonValue();

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            int at = pos;
            while (at < input.length() && Character.isWhitespace(input.charAt(at))) {
                at++;
            }
            int end = JsonBalance.end(input, at);
            if (end >= 0) {
                return Result.ok(end, Collections.<Capture>emptyList());
            }
            return partial ? Result.needMore(input.length(), Collections.<Capture>emptyList()) : Result.fail();
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return grammar.schema("json", "{}");
        }
    }

    private static final class SchemaValue extends Peg {
        private final String name;
        private final String schemaJson;

        SchemaValue(String name, String schemaJson) {
            this.name = name;
            this.schemaJson = schemaJson;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            return JsonValue.INSTANCE.match(input, pos, partial);
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return grammar.schema(name, schemaJson);
        }
    }

    private static final class Rule extends Peg {
        private final String name;
        private final Peg child;

        Rule(String name, Peg child) {
            this.name = name;
            this.child = child;
        }

        @Override public Result match(CharSequence input, int pos, boolean partial) {
            return child.match(input, pos, partial);
        }

        @Override public String gbnf(GbnfEmitter grammar) {
            return grammar.define(name, child.gbnf(grammar));
        }
    }

    // ================================================================== helpers

    private static boolean needsGroup(Peg p) {
        return p instanceof Choice || p instanceof Seq;
    }

    private static boolean regionMatches(CharSequence input, int pos, String text) {
        if (pos < 0 || pos + text.length() > input.length()) {
            return false;
        }
        for (int i = 0; i < text.length(); i++) {
            if (input.charAt(pos + i) != text.charAt(i)) {
                return false;
            }
        }
        return true;
    }

    private static int indexOf(CharSequence input, String text, int from) {
        int last = input.length() - text.length();
        for (int i = Math.max(0, from); i <= last; i++) {
            if (regionMatches(input, i, text)) {
                return i;
            }
        }
        return -1;
    }

    // ================================================================== grammar registry

    /**
     * The rule registry a {@link #gbnf} pass writes into: a {@link JsonSchemaToGbnf.Composer} with a name prefix, so
     * rules of different expressions never collide. Not thread-safe; one instance per grammar build.
     */
    public static final class GbnfEmitter {
        private final JsonSchemaToGbnf.Composer composer = JsonSchemaToGbnf.composer(null);
        private final String prefix;
        private int counter;

        /** An emitter whose rule names carry the {@code peg} prefix. */
        public GbnfEmitter() {
            this("peg");
        }

        public GbnfEmitter(String prefix) {
            this.prefix = prefix;
        }

        /** Adds a schema rule and returns its name. */
        public String schema(String base, String schemaJson) {
            return composer.addSchema(nextName(base), schemaJson);
        }

        /** Adds a literal rule and returns its name. */
        public String define(String base, String definition) {
            return composer.addRule(nextName(base), definition);
        }

        /** Adds the exclusion automaton of {@code strings} and returns its start rule name. */
        public String excluding(List<String> strings) {
            return GbnfExclusion.excluding(composer::addRule, nextName("until"), strings);
        }

        /** Adds the root rule and returns the GBNF text. */
        public String complete(String root) {
            return finish(root);
        }

        /** Adds the root rule and returns the GBNF text. */
        public String finish(String root) {
            composer.addRule("root", root);
            return composer.gbnf();
        }

        private String nextName(String base) {
            return prefix + "-" + base + "-" + (counter++);
        }
    }
}
