package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The product of a grammar with the Aho-Corasick automaton of a rule's stop strings (xgrammar's {@code stop}
 * rule attribute): every rule is duplicated once per automaton state it can be entered in, and every character
 * set element is split by the state its code points lead to, so no derivation of the product ever writes a stop
 * string. The caller appends the stop itself after the product's root, which makes the rule's text end at the
 * stop — the product of the body with "no stop inside", followed by the stop, enforced by the grammar runtime
 * itself rather than a post-filter.
 *
 * <p>Code points outside the automaton's alphabet always return to the root state, so a character set splits
 * into one finite class per reachable state plus, for a negated set, one negated class for the rest. A
 * character set whose every matched code point completes a stop kills its alternative; an alternative holding a
 * dead rule dies with it, repeatedly, and a body whose root dies is refused by name. The product keeps the
 * body's rule structure, so a body without left recursion stays without it. Not thread-safe; confined to the
 * thread compiling the grammar.</p>
 */
final class GrammarStop {

    private final GbnfExclusion.Automaton automaton;
    private final Grammar body;
    private final String ruleName;
    /** The product rules in creation order; an entry is filled right after its body is walked. */
    private final List<List<GrammarElement>> rules = new ArrayList<>();
    /** The body rule each product rule came from, for the diagnostic positions of the built grammar. */
    private final List<Integer> sources = new ArrayList<>();
    private final List<String> names = new ArrayList<>();
    private final List<Boolean> dead = new ArrayList<>();
    private final Map<Long, Integer> indices = new HashMap<>();
    private long elements;

    private GrammarStop(Grammar body, List<String> stops, String ruleName) {
        this.body = body;
        this.ruleName = ruleName;
        this.automaton = new GbnfExclusion.Automaton(stops);
    }

    /**
     * The body grammar with no stop string in any of its derivations; its root rule is the product of the
     * body's root from the automaton's start state.
     *
     * @param body     the grammar of a stop rule's body, its root rule the body itself
     * @param stops    the stop strings, none empty
     * @param ruleName the Lark rule's name, for the refusals
     * @throws GrammarException when no derivation of the body avoids the stop strings, or the product exceeds a
     *                          {@link GrammarLimits} bound
     */
    static Grammar apply(Grammar body, List<String> stops, String ruleName) {
        GrammarStop product = new GrammarStop(body, stops, ruleName);
        int root = product.enter(body.rootRuleIndex(), 0);
        product.propagateDead();
        if (product.dead.get(root)) {
            throw new GrammarException("invalid Lark grammar: rule '" + ruleName + "' declares stop \""
                    + stops.get(0) + "\", which its text cannot avoid: every derivation of the body writes the"
                    + " stop", body.ruleLine(body.rootRuleIndex()), body.ruleColumn(body.rootRuleIndex()));
        }
        return product.grammar(root);
    }

    /** The index of the product rule of body rule {@code rule} entered at automaton state {@code state}. */
    private int enter(int rule, int state) {
        long key = ((long) rule << 32) | (state & 0xffffffffL);
        Integer known = indices.get(key);
        if (known != null) {
            return known;
        }
        int index = reserve(body.symbolName(rule), rule);
        indices.put(key, index);
        fill(index, walk(rule, state));
        return index;
    }

    /** The product body of body rule {@code rule} entered at {@code state}: one alternate per surviving walk. */
    private List<GrammarElement> walk(int rule, int state) {
        List<GrammarElement> source = body.rule(rule);
        List<GrammarElement> out = new ArrayList<>();
        boolean any = false;
        int start = 0;
        while (start < source.size()) {
            int end = sequenceEnd(source, start);
            List<GrammarElement> alternate = walkSequence(source, start, end, state, rule);
            if (alternate != null) {
                if (any) {
                    out.add(new GrammarElement(GrammarElement.Type.ALT, 0));
                }
                out.addAll(alternate);
                any = true;
            }
            start = end + 1;
        }
        return out;
    }

    /**
     * The product of one alternate — the elements of {@code source} from {@code from} before {@code end} —
     * entered at {@code state}; null when every path of the alternate completes a stop. A character set split
     * over several target states continues all but the last target through a fork rule over the rest.
     */
    private List<GrammarElement> walkSequence(List<GrammarElement> source, int from, int end, int state,
            int sourceRule) {
        List<GrammarElement> alternate = new ArrayList<>();
        int at = state;
        int i = from;
        while (i < end) {
            GrammarElement e = source.get(i);
            GrammarElement.Type type = e.type();
            if (GrammarElement.isBudget(type)) {
                alternate.add(e);
                i++;
                continue;
            }
            if (type == GrammarElement.Type.RULE_REF) {
                alternate.add(new GrammarElement(GrammarElement.Type.RULE_REF, enter(e.value(), at)));
                i++;
                continue;
            }
            if (type == GrammarElement.Type.CHAR || type == GrammarElement.Type.CHAR_NOT) {
                SetRun set = setRun(source, i, end);
                i = set.next;
                List<Bucket> buckets = buckets(set, at);
                for (int b = 0; b < buckets.size(); b++) {
                    Bucket bucket = buckets.get(b);
                    alternate.addAll(bucket.classElements());
                    if (b == buckets.size() - 1) {
                        at = bucket.state;
                    } else {
                        alternate.add(new GrammarElement(GrammarElement.Type.RULE_REF,
                                fork(source, i, end, bucket.state, sourceRule)));
                    }
                }
                if (buckets.isEmpty()) {
                    return null;
                }
                continue;
            }
            throw new IllegalStateException("a stop body holds no " + type + " element");
        }
        return alternate;
    }

    /** A fork rule over the rest of an alternate, entered at {@code state}; dead when the rest has no path. */
    private int fork(List<GrammarElement> source, int from, int end, int state, int sourceRule) {
        int index = reserve(ruleName + "-rest-" + index(), sourceRule);
        List<GrammarElement> alternate = walkSequence(source, from, end, state, sourceRule);
        fill(index, alternate == null ? new ArrayList<>() : alternate);
        return index;
    }

    /** The next product rule index, for a fork's unique name. */
    private int index() {
        return rules.size();
    }

    // ------------------------------------------------------------------ character sets

    /** One character set element run: its set and the position after it. */
    private static final class SetRun {
        final boolean negated;
        final List<int[]> members;
        final int next;

        SetRun(boolean negated, List<int[]> members, int next) {
            this.negated = negated;
            this.members = members;
            this.next = next;
        }

        /** Whether the set matches {@code c}. */
        boolean matches(int c) {
            for (int[] range : members) {
                if (range[0] <= c && c <= range[1]) {
                    return !negated;
                }
            }
            return negated;
        }
    }

    /** The set run at {@code from}: a {@code CHAR}/{@code CHAR_NOT} head with its {@code CHAR_ALT} members. */
    private static SetRun setRun(List<GrammarElement> source, int from, int end) {
        boolean negated = source.get(from).type() == GrammarElement.Type.CHAR_NOT;
        List<int[]> members = new ArrayList<>();
        int lo = source.get(from).value();
        int hi = lo;
        int i = from + 1;
        while (i < end) {
            GrammarElement.Type type = source.get(i).type();
            if (type == GrammarElement.Type.CHAR_RNG_UPPER) {
                hi = source.get(i).value();
                i++;
            } else if (type == GrammarElement.Type.CHAR_ALT) {
                members.add(new int[]{lo, hi});
                lo = source.get(i).value();
                hi = lo;
                i++;
            } else {
                break;
            }
        }
        members.add(new int[]{lo, hi});
        return new SetRun(negated, members, i);
    }

    /** One split of a character set: the ranges that reach {@code state}, as one character class. */
    private static final class Bucket {
        final int state;
        final boolean negated;
        final List<int[]> ranges = new ArrayList<>();

        Bucket(int state, boolean negated) {
            this.state = state;
            this.negated = negated;
        }

        void add(int c) {
            addRange(c, c);
        }

        void addRange(int lo, int hi) {
            int at = 0;
            while (at < ranges.size() && ranges.get(at)[1] + 1 < lo) {
                at++;
            }
            if (at == ranges.size()) {
                ranges.add(new int[]{lo, hi});
                return;
            }
            int[] merged = {Math.min(lo, ranges.get(at)[0]), Math.max(hi, ranges.get(at)[1])};
            while (ranges.size() > at + 1 && ranges.get(at + 1)[0] <= merged[1] + 1) {
                merged[1] = Math.max(merged[1], ranges.get(at + 1)[1]);
                ranges.remove(at + 1);
            }
            ranges.set(at, merged);
        }

        /** The class as grammar elements: {@code CHAR}/{@code CHAR_NOT}, its ranges, one {@code CHAR_ALT} each. */
        List<GrammarElement> classElements() {
            List<GrammarElement> out = new ArrayList<>(ranges.size() * 2);
            for (int k = 0; k < ranges.size(); k++) {
                int[] range = ranges.get(k);
                if (k == 0) {
                    out.add(new GrammarElement(negated ? GrammarElement.Type.CHAR_NOT : GrammarElement.Type.CHAR,
                            range[0]));
                } else {
                    out.add(new GrammarElement(GrammarElement.Type.CHAR_ALT, range[0]));
                }
                if (range[1] != range[0]) {
                    out.add(new GrammarElement(GrammarElement.Type.CHAR_RNG_UPPER, range[1]));
                }
            }
            return out;
        }
    }

    /**
     * The character set {@code set} entered at {@code state}, split per target state. A matched code point
     * whose transition completes a stop matches nothing; outside the alphabet every code point returns to the
     * root, which a positive set lists as ranges and a negated set leaves to its negated rest class. Empty
     * buckets are dropped: a bucket without a class has no text and no state to continue from.
     */
    private List<Bucket> buckets(SetRun set, int state) {
        Map<Integer, Bucket> finite = new TreeMap<>();
        for (int c : automaton.alphabet()) {
            if (!set.matches(c)) {
                continue;
            }
            int target = automaton.next(state, c);
            if (automaton.terminal(target)) {
                continue;
            }
            if (set.negated && target == 0) {
                continue;
            }
            finite.computeIfAbsent(target, t -> new Bucket(t, false)).add(c);
        }
        if (!set.negated) {
            // a positive set's members outside the alphabet all return to the root
            for (int[] range : set.members) {
                addNonAlphabetGaps(range, finite.computeIfAbsent(0, t -> new Bucket(t, false)));
            }
        }
        List<Bucket> out = new ArrayList<>();
        for (Bucket bucket : finite.values()) {
            if (!bucket.ranges.isEmpty()) {
                out.add(bucket);
            }
        }
        if (set.negated) {
            // the rest class matches every code point but the ones listed here: the set's own members, and
            // the alphabet chars that match nothing, complete a stop, or head to another state's bucket
            Bucket rest = new Bucket(0, true);
            for (int c : automaton.alphabet()) {
                boolean matched = set.matches(c);
                int target = matched ? automaton.next(state, c) : -1;
                if (!matched || automaton.terminal(target) || target != 0) {
                    rest.add(c);
                }
            }
            for (int[] range : set.members) {
                rest.addRange(range[0], range[1]);
            }
            if (!rest.ranges.isEmpty()) {
                out.add(rest);
            }
        }
        return out;
    }

    /** The sub-ranges of {@code range} outside the automaton's alphabet, into {@code bucket}. */
    private void addNonAlphabetGaps(int[] range, Bucket bucket) {
        int at = range[0];
        for (int c : alphabetWithin(range[0], range[1])) {
            if (at <= c - 1) {
                bucket.addRange(at, c - 1);
            }
            at = c + 1;
        }
        if (at <= range[1]) {
            bucket.addRange(at, range[1]);
        }
    }

    /** The automaton's alphabet code points within {@code lo..hi}, in order. */
    private Iterable<Integer> alphabetWithin(int lo, int hi) {
        List<Integer> out = new ArrayList<>();
        for (int c : automaton.alphabet()) {
            if (c > hi) {
                break;
            }
            if (c >= lo) {
                out.add(c);
            }
        }
        return out;
    }

    /** The index of the first {@code ALT} or {@code END} at or after {@code from}. */
    private static int sequenceEnd(List<GrammarElement> source, int from) {
        for (int i = from; i < source.size(); i++) {
            GrammarElement.Type type = source.get(i).type();
            if (type == GrammarElement.Type.ALT || type == GrammarElement.Type.END) {
                return i;
            }
        }
        return source.size();
    }

    // ------------------------------------------------------------------ assembly

    private int reserve(String name, int source) {
        int index = rules.size();
        rules.add(null);
        sources.add(source);
        names.add("0stop" + index + "-" + name);
        dead.add(Boolean.FALSE);
        return index;
    }

    private void fill(int index, List<GrammarElement> body) {
        List<GrammarElement> full = new ArrayList<>(body.size() + 1);
        full.addAll(body);
        full.add(new GrammarElement(GrammarElement.Type.END, 0));
        elements += full.size();
        dead.set(index, body.isEmpty());
        checkLimits(index);
        rules.set(index, full);
    }

    /** An alternative holding a dead rule dies with it; one pass per rule until nothing changes. */
    private void propagateDead() {
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int r = 0; r < rules.size(); r++) {
                if (dead.get(r) || alive(rules.get(r))) {
                    continue;
                }
                dead.set(r, Boolean.TRUE);
                changed = true;
            }
        }
    }

    private boolean alive(List<GrammarElement> alternate) {
        for (GrammarElement e : alternate) {
            if (e.type() == GrammarElement.Type.RULE_REF && dead.get(e.value())) {
                return false;
            }
        }
        return true;
    }

    private void checkLimits(int at) {
        if (rules.size() > GrammarLimits.MAX_RULES) {
            throw new GrammarException("invalid Lark grammar: rule '" + ruleName + "' declares a stop, and its"
                    + " product holds more than " + GrammarLimits.MAX_RULES + " rules", line(at), 1);
        }
        if (elements > GrammarLimits.MAX_ELEMENTS) {
            throw new GrammarException("invalid Lark grammar: rule '" + ruleName + "' declares a stop, and its"
                    + " product holds more than " + GrammarLimits.MAX_ELEMENTS + " elements", line(at), 1);
        }
    }

    private int line(int productRule) {
        int source = sources.get(productRule);
        return body.ruleLine(source);
    }

    /** The finished grammar over the product rules, its root {@code root}. */
    private Grammar grammar(int root) {
        int[] lines = new int[rules.size()];
        int[] columns = new int[rules.size()];
        Map<String, Integer> symbolIds = new TreeMap<>();
        for (int r = 0; r < rules.size(); r++) {
            symbolIds.put(names.get(r), r);
            lines[r] = body.ruleLine(sources.get(r));
            columns[r] = body.ruleColumn(sources.get(r));
        }
        return new Grammar(rules, symbolIds, names, root, lines, columns);
    }
}
