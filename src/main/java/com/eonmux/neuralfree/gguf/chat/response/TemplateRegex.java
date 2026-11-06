package com.eonmux.neuralfree.gguf.chat.response;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A {@code response_template} regular expression: a Python {@code regex} pattern compiled to {@link Pattern} with
 * Python's semantics — dot-all (the template loader's {@code re.DOTALL}), Unicode {@code \w}/{@code \d}/{@code \s},
 * named groups {@code (?P<name>...)} and {@code (?<name>...)} (identifier names, underscores included) and back
 * references {@code (?P=name)} and {@code \g<name>} under their Python names, {@code \Z} as the absolute end, a
 * literal {@code [} or {@code &&} inside a character class, a {@code {} that starts no quantifier as a literal and the
 * {@code {,n}} and {@code {,}} quantifiers. Any construct Java's engine lacks fails the compile, naming the pattern.
 *
 * <p>{@link #search} with {@code partial} emulates the {@code regex} module's partial matching: the leftmost position
 * where the pattern either matches or runs out of input while still matching. Immutable.
 */
final class TemplateRegex {

    private static final int FLAGS = Pattern.DOTALL | Pattern.UNICODE_CHARACTER_CLASS;
    private static final Pattern QUANTIFIER = Pattern.compile("\\{(\\d*)(,(\\d*))?\\}");
    /** A Python group name: an identifier, Unicode letters and underscores included. */
    private static final Pattern GROUP_NAME = Pattern.compile("[^\\W\\d]\\w*", Pattern.UNICODE_CHARACTER_CLASS);

    private final Pattern pattern;
    /** Python group name to the Java group name it compiled to, in pattern order. */
    private final Map<String, String> groups;
    private final String source;

    private TemplateRegex(Pattern pattern, Map<String, String> groups, String source) {
        this.pattern = pattern;
        this.groups = Collections.unmodifiableMap(groups);
        this.source = source;
    }

    /**
     * Compiles a Python pattern.
     *
     * @param python the pattern
     * @param what   what the pattern is, for the error ({@code Field 'tool_calls': open_pattern})
     * @throws IllegalArgumentException when the pattern does not compile
     */
    static TemplateRegex compile(String python, String what) {
        Objects.requireNonNull(python, "python");
        Map<String, String> groups = new LinkedHashMap<>();
        try {
            return new TemplateRegex(Pattern.compile(translate(python, groups), FLAGS), groups, python);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(what + ": invalid regex " + python + ": " + e.getMessage(), e);
        }
    }

    /** Any of {@code literals}, the longest preferred where several match at one position. */
    static TemplateRegex anyOf(List<String> literals) {
        List<String> ordered = new ArrayList<>(literals);
        ordered.sort(Comparator.comparingInt(String::length).reversed());
        StringBuilder b = new StringBuilder();
        for (String s : ordered) {
            if (b.length() > 0) b.append('|');
            b.append(Pattern.quote(s));
        }
        return new TemplateRegex(Pattern.compile(b.toString(), FLAGS), new LinkedHashMap<String, String>(),
                String.join("|", ordered));
    }

    /**
     * The Java form of a Python pattern, for composing it into a larger Java pattern; its named groups get fresh
     * names.
     *
     * @throws IllegalArgumentException when the pattern does not translate
     */
    static String javaPattern(String python) {
        return translate(python, new LinkedHashMap<String, String>());
    }

    /** Whether the pattern matches all of {@code text}. */
    boolean fullMatches(CharSequence text) {
        return pattern.matcher(text).matches();
    }

    /**
     * Whether the pattern matches {@code text} from {@code from} to its end (Python {@code fullmatch(text, pos)}): a
     * {@code ^} still matches only at the text's real start.
     */
    boolean fullMatchesFrom(CharSequence text, int from) {
        Matcher m = pattern.matcher(text);
        m.useTransparentBounds(true).useAnchoringBounds(false);
        m.region(from, text.length());
        return m.matches();
    }

    /** The Python names of the pattern's named groups. */
    List<String> groupNames() {
        return new ArrayList<>(groups.keySet());
    }

    /** The pattern as written. */
    String source() {
        return source;
    }

    /**
     * The leftmost match at or after {@code from}, or null. With {@code partial} a position where the input ends inside
     * a match that could still complete counts too, as a partial match ending at the input's end.
     */
    Match search(CharSequence text, int from, boolean partial) {
        Matcher m = pattern.matcher(text);
        if (!partial) {
            return m.find(from) ? match(m) : null;
        }
        m.useTransparentBounds(true).useAnchoringBounds(false);
        for (int start = from; start <= text.length(); start++) {
            m.region(start, text.length());
            if (m.lookingAt()) {
                return match(m);
            }
            if (m.hitEnd()) {
                return new Match(start, text.length(), true, Collections.<String, String>emptyMap());
            }
        }
        return null;
    }

    /** Every non-overlapping match, left to right. */
    List<Match> findAll(CharSequence text) {
        List<Match> out = new ArrayList<>();
        Matcher m = pattern.matcher(text);
        while (m.find()) {
            out.add(match(m));
        }
        return out;
    }

    /** One match: its span, whether it is partial and its named groups that took part (Python names). */
    static final class Match {
        final int start;
        final int end;
        final boolean partial;
        final Map<String, String> captures;

        Match(int start, int end, boolean partial, Map<String, String> captures) {
            this.start = start;
            this.end = end;
            this.partial = partial;
            this.captures = captures;
        }

        /** Whether the group named {@code name} took part in the match. */
        boolean has(String name) {
            return captures.containsKey(name);
        }
    }

    /** The complete match {@code m} holds. */
    private Match match(Matcher m) {
        return new Match(m.start(), m.end(), false, captures(m));
    }

    /** {@code m}'s named groups that took part, under their Python names. */
    private Map<String, String> captures(Matcher m) {
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<String, String> g : groups.entrySet()) {
            String v = m.group(g.getValue());
            if (v != null) out.put(g.getKey(), v);
        }
        return Collections.unmodifiableMap(out);
    }

    // ================================================================= translation

    /** The Java form of a Python pattern; records each named group's Java name in {@code groups}. */
    private static String translate(String p, Map<String, String> groups) {
        StringBuilder out = new StringBuilder(p.length() + 16);
        boolean inClass = false;
        int i = 0;
        while (i < p.length()) {
            char c = p.charAt(i);
            if (c == '\\') {
                i = escape(p, i, inClass, out, groups);
            } else if (inClass && c == ']') {
                out.append(']');
                inClass = false;
                i++;
            } else if (inClass) {
                i = classChar(p, i, out);
            } else if (c == '[') {
                out.append('[');
                i++;
                if (i < p.length() && p.charAt(i) == '^') {
                    out.append('^');
                    i++;
                }
                if (i < p.length() && p.charAt(i) == ']') {
                    out.append("\\]");
                    i++;
                }
                inClass = true;
            } else if (p.startsWith("(?P<", i)) {
                i = namedGroup(p, i + 4, out, groups);
            } else if (p.startsWith("(?<", i) && !p.startsWith("(?<=", i) && !p.startsWith("(?<!", i)) {
                i = namedGroup(p, i + 3, out, groups);
            } else if (p.startsWith("(?P=", i)) {
                int close = p.indexOf(')', i);
                if (close < 0) throw new IllegalArgumentException("an unterminated back reference");
                out.append(backReference(p.substring(i + 4, close), groups));
                i = close + 1;
            } else if (c == '{') {
                i = brace(p, i, out);
            } else {
                out.append(c);
                i++;
            }
        }
        if (inClass) throw new IllegalArgumentException("an unterminated character set");
        return out.toString();
    }

    /**
     * The escape at {@code i}: {@code \Z} outside a class as Java's absolute end, {@code \g<name>} outside a class as
     * a back reference, any other escape as written; answers the index after it.
     */
    private static int escape(String p, int i, boolean inClass, StringBuilder out, Map<String, String> groups) {
        if (i + 1 >= p.length()) throw new IllegalArgumentException("a trailing backslash");
        char e = p.charAt(i + 1);
        if (!inClass && e == 'g' && i + 2 < p.length() && p.charAt(i + 2) == '<') {
            int close = p.indexOf('>', i + 3);
            if (close < 0) throw new IllegalArgumentException("an unterminated group reference");
            out.append(backReference(p.substring(i + 3, close), groups));
            return close + 1;
        }
        out.append('\\').append(!inClass && e == 'Z' ? 'z' : e);
        return i + 2;
    }

    /** A named group's opener whose name starts at {@code nameStart}; answers the index after its {@code >}. */
    private static int namedGroup(String p, int nameStart, StringBuilder out, Map<String, String> groups) {
        int close = p.indexOf('>', nameStart);
        if (close < 0) throw new IllegalArgumentException("an unterminated group name");
        String name = p.substring(nameStart, close);
        if (!GROUP_NAME.matcher(name).matches()) {
            throw new IllegalArgumentException("bad character in group name '" + name + "'");
        }
        if (groups.containsKey(name)) throw new IllegalArgumentException("group '" + name + "' is redefined");
        String javaName = "g" + groups.size() + "x";
        groups.put(name, javaName);
        out.append("(?<").append(javaName).append('>');
        return close + 1;
    }

    /** The Java back reference to the group a Python reference names: a group name or a group number. */
    private static String backReference(String ref, Map<String, String> groups) {
        if (!ref.isEmpty() && ref.chars().allMatch(ch -> ch >= '0' && ch <= '9')) {
            return "(?:\\" + Integer.parseInt(ref) + ")";
        }
        String javaName = groups.get(ref);
        if (javaName == null) throw new IllegalArgumentException("unknown group name '" + ref + "'");
        return "\\k<" + javaName + ">";
    }

    /** One character inside a class: Java reads {@code [} and {@code &&} as set operators, Python literally. */
    private static int classChar(String p, int i, StringBuilder out) {
        char c = p.charAt(i);
        if (c == '[' || (c == '&' && i + 1 < p.length() && p.charAt(i + 1) == '&')) {
            out.append('\\').append(c);
        } else {
            out.append(c);
        }
        return i + 1;
    }

    /** A {@code {}: a quantifier as Java writes it, else a literal brace. */
    private static int brace(String p, int i, StringBuilder out) {
        Matcher q = QUANTIFIER.matcher(p).region(i, p.length());
        if (q.lookingAt() && !(q.group(1).isEmpty() && q.group(2) == null)) {
            String min = q.group(1).isEmpty() ? "0" : q.group(1);
            out.append('{').append(min);
            if (q.group(2) != null) out.append(',').append(q.group(3));
            out.append('}');
            return q.end();
        }
        out.append("\\{");
        return i + 1;
    }
}
