package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The member selections of an object: in declaration order, every required member and any subset of the optional
 * ones, followed by any number of the repeated (additional or pattern) member, the count of picked members between
 * two bounds. {@link #first} writes the first picked member bare, {@link #rest} prefixes every picked member with the
 * separator, as an object rule joins its members (a JSON object's comma, an XML object's whitespace). Each state
 * (next member, bounds left) is one shared rule named {@code <prefix><f|c><i>-<low>-<high>} in the
 * {@link GbnfBuilder}; the repeated member becomes a bounded repetition. Callers pass bounds the members can meet
 * ({@link #capacity}, {@link #requiredCount}).
 */
final class CountedMembers {

    /** An upper bound that does not bound. */
    static final int UNBOUNDED = Integer.MAX_VALUE;

    private final GbnfBuilder rules;
    private final List<String> members;
    /** {@code requiredFrom[i]}: how many required members sit at index {@code i} or later. */
    private final int[] requiredFrom;
    private final String repeated;
    private final String separator;
    private final String prefix;
    private final Map<String, String> memo = new HashMap<>();

    /**
     * @param rules     the registry the selection rules go into
     * @param members   the member rule names, in order
     * @param required  which members every selection holds, one flag per member
     * @param repeated  the member rule name of the repeatable member, or null
     * @param separator the GBNF text written before every member but the first
     * @param prefix    the rule name prefix of the selection rules
     */
    CountedMembers(GbnfBuilder rules, List<String> members, boolean[] required, String repeated, String separator,
            String prefix) {
        if (required.length != members.size()) {
            throw new IllegalArgumentException(required.length + " required flags for " + members.size()
                    + " members");
        }
        this.rules = rules;
        this.members = new ArrayList<>(members);
        this.requiredFrom = new int[members.size() + 1];
        for (int i = members.size() - 1; i >= 0; i--) {
            requiredFrom[i] = requiredFrom[i + 1] + (required[i] ? 1 : 0);
        }
        this.repeated = repeated;
        this.separator = separator;
        this.prefix = prefix;
    }

    /** How many members the object can hold. */
    int capacity() {
        return capacity(0);
    }

    /** How many members every selection holds. */
    int requiredCount() {
        return requiredFrom[0];
    }

    /** The selections of {@code low..high} members, the first one bare; empty when only none qualifies. */
    String first(int low, int high) {
        return select(0, low, Math.min(high, capacity(0)), true);
    }

    /** The selections of {@code low..high} members, each separator-prefixed; empty when only none qualifies. */
    String rest(int low, int high) {
        return select(0, low, Math.min(high, capacity(0)), false);
    }

    private int capacity(int i) {
        return repeated != null ? UNBOUNDED : members.size() - i;
    }

    private boolean feasible(int i, int low, int high) {
        return low <= high && low <= capacity(i) && requiredFrom[i] <= high;
    }

    private static int less(int high) {
        return high == UNBOUNDED ? UNBOUNDED : high - 1;
    }

    private boolean required(int i) {
        return requiredFrom[i] > requiredFrom[i + 1];
    }

    private String select(int i, int low, int high, boolean first) {
        if (high == 0) {
            return "";
        }
        if (i == members.size()) {
            return repeatedSelection(low, high, first);
        }
        String key = (first ? "f" : "c") + i + "-" + low + "-" + (high == UNBOUNDED ? "n" : String.valueOf(high));
        String cached = memo.get(key);
        if (cached != null) {
            return cached;
        }
        List<String> alternatives = new ArrayList<>();
        boolean empty = false;
        if (!required(i) && feasible(i + 1, low, high)) {
            String skip = select(i + 1, low, Math.min(high, capacity(i + 1)), first);
            if (skip.isEmpty()) {
                empty = true;
            } else {
                alternatives.add(skip);
            }
        }
        int nextLow = Math.max(0, low - 1);
        if (feasible(i + 1, nextLow, less(high))) {
            String after = select(i + 1, nextLow, Math.min(less(high), capacity(i + 1)), false);
            alternatives.add((first ? "" : separator + " ") + members.get(i) + (after.isEmpty() ? "" : " " + after));
        }
        String body = String.join(" | ", alternatives);
        String rule = alternatives.isEmpty() ? "" : rules.addRule(prefix + key, empty ? "( " + body + " )?" : body);
        memo.put(key, rule);
        return rule;
    }

    private String repeatedSelection(int low, int high, boolean first) {
        if (repeated == null) {
            return "";
        }
        String member = "( " + separator + " " + repeated + " )";
        if (!first) {
            return GbnfBuilder.repetition(member, low, high, "");
        }
        String after = GbnfBuilder.repetition(member, Math.max(0, low - 1), less(high), "");
        String one = repeated + (after.isEmpty() ? "" : " " + after);
        return low == 0 ? "( " + one + " )?" : one;
    }
}
