package com.eonmux.neuralfree.gguf.chat.autoparser;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The differential engine behind {@link AutoParserAnalysis}: llama.cpp's marker segmentation and diff split
 * ({@code common/chat-auto-parser-helpers.cpp}), which read one template's structure by rendering two
 * conversations that differ in a single input and comparing the outputs.
 *
 * <p>{@link #segmentize} splits text into runs and the {@code <...>} / {@code [...]} markers between them, so a
 * protocol tag compares equal to itself but never to the text around it. {@link #split} returns the shared prefix
 * and suffix of two strings and what differs between them — the text one variant writes that the other does not —
 * preferring marker boundaries over a character-level split. Immutable.
 */
public final class MarkerDiff {

    /** A run of rendered text. */
    public enum Type { TEXT, MARKER }

    /** One segment of rendered text: its kind and its text. Immutable. */
    public static final class Segment {
        private final Type type;
        private final String value;

        public Segment(Type type, String value) {
            this.type = Objects.requireNonNull(type, "type");
            this.value = Objects.requireNonNull(value, "value");
        }

        public Type type() {
            return type;
        }

        public String value() {
            return value;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Segment)) {
                return false;
            }
            Segment that = (Segment) o;
            return type == that.type && value.equals(that.value);
        }

        @Override
        public int hashCode() {
            return Objects.hash(type, value);
        }

        @Override
        public String toString() {
            return type + "(" + value + ")";
        }
    }

    /** The two variants' shared prefix and suffix and the text unique to each. Immutable. */
    public static final class Diff {
        private final String prefix;
        private final String suffix;
        private final String left;
        private final String right;

        public Diff(String prefix, String suffix, String left, String right) {
            this.prefix = Objects.requireNonNull(prefix, "prefix");
            this.suffix = Objects.requireNonNull(suffix, "suffix");
            this.left = Objects.requireNonNull(left, "left");
            this.right = Objects.requireNonNull(right, "right");
        }

        /** Text both variants write before the difference. */
        public String prefix() {
            return prefix;
        }

        /** Text both variants write after the difference. */
        public String suffix() {
            return suffix;
        }

        /** Text only the first variant writes between the shared parts. */
        public String left() {
            return left;
        }

        /** Text only the second variant writes between the shared parts. */
        public String right() {
            return right;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Diff)) {
                return false;
            }
            Diff that = (Diff) o;
            return prefix.equals(that.prefix) && suffix.equals(that.suffix) && left.equals(that.left)
                    && right.equals(that.right);
        }

        @Override
        public int hashCode() {
            return Objects.hash(prefix, suffix, left, right);
        }

        @Override
        public String toString() {
            return "Diff(prefix=" + prefix + ", left=" + left + ", right=" + right + ", suffix=" + suffix + ")";
        }
    }

    private MarkerDiff() {
    }

    /**
     * Splits {@code text} into text runs and markers: a marker runs from a {@code <} or {@code [} to the next
     * {@code >} or {@code ]} of the same pair, as {@code segmentize_markers} does. An unfinished marker (no closer)
     * stays text.
     */
    public static List<Segment> segmentize(String text) {
        Objects.requireNonNull(text, "text");
        List<Segment> out = new ArrayList<>();
        boolean inMarker = false;
        char opener = '\0';
        int lastBorder = 0;
        for (int pos = 0; pos < text.length(); pos++) {
            char c = text.charAt(pos);
            if (!inMarker && (c == '<' || c == '[')) {
                if (lastBorder < pos) {
                    out.add(new Segment(Type.TEXT, text.substring(lastBorder, pos)));
                }
                lastBorder = pos;
                inMarker = true;
                opener = c;
            } else if (inMarker && ((opener == '<' && c == '>') || (opener == '[' && c == ']'))) {
                out.add(new Segment(Type.MARKER, text.substring(lastBorder, pos + 1)));
                lastBorder = pos + 1;
                inMarker = false;
                opener = '\0';
            }
        }
        if (lastBorder < text.length()) {
            out.add(new Segment(Type.TEXT, text.substring(lastBorder)));
        }
        return out;
    }

    /** The segments whose text is not blank. */
    public static List<Segment> pruneWhitespace(List<Segment> segments) {
        List<Segment> out = new ArrayList<>();
        for (Segment s : segments) {
            if (!s.value().trim().isEmpty()) {
                out.add(s);
            }
        }
        return out;
    }

    /** The longest prefix the two strings share. */
    public static String commonPrefix(String left, String right) {
        return left.substring(0, commonPrefixLength(left, right));
    }

    /** The longest suffix the two strings share. */
    public static String commonSuffix(String left, String right) {
        return left.substring(left.length() - commonSuffixLength(left, right));
    }

    /** The length of the longest prefix {@code left} and {@code right} share. */
    static int commonPrefixLength(String left, String right) {
        int n = 0;
        int max = Math.min(left.length(), right.length());
        while (n < max && left.charAt(n) == right.charAt(n)) {
            n++;
        }
        return n;
    }

    /** The length of the longest suffix {@code left} and {@code right} share. */
    static int commonSuffixLength(String left, String right) {
        int n = 0;
        int max = Math.min(left.length(), right.length());
        while (n < max && left.charAt(left.length() - 1 - n) == right.charAt(right.length() - 1 - n)) {
            n++;
        }
        return n;
    }

    /**
     * The prefix of {@code full} before the first occurrence of the common prefix of {@code left} and
     * {@code right}; empty when they share none or {@code full} does not hold it.
     */
    public static String untilCommonPrefix(String full, String left, String right) {
        String shared = commonPrefix(left, right);
        if (shared.isEmpty()) {
            return "";
        }
        int at = full.indexOf(shared);
        return at < 0 ? "" : full.substring(0, at);
    }

    /**
     * The suffix of {@code full} after the last occurrence of the common suffix of {@code left} and {@code right};
     * empty when they share none or {@code full} does not hold it.
     */
    public static String afterCommonSuffix(String full, String left, String right) {
        String shared = commonSuffix(left, right);
        if (shared.isEmpty()) {
            return "";
        }
        int at = full.lastIndexOf(shared);
        return at < 0 ? "" : full.substring(at + shared.length());
    }

    /** The shared prefix and suffix of the two texts and the text unique to each. */
    public static Diff split(String left, String right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        List<Segment> leftSeg = segmentize(left);
        List<Segment> rightSeg = segmentize(right);
        if (leftSeg.isEmpty()) {
            return new Diff("", "", "", right);
        }
        if (rightSeg.isEmpty()) {
            return new Diff("", "", left, "");
        }

        StringBuilder prefix = new StringBuilder();
        String suffix = "";
        int ls = 0;
        int le = leftSeg.size() - 1;
        int rs = 0;
        int re = rightSeg.size() - 1;
        boolean leftConsumed = false;
        boolean rightConsumed = false;

        while (ls != le && rs != re) {
            boolean advanced = false;
            if (leftSeg.get(ls).equals(rightSeg.get(rs))) {
                prefix.append(leftSeg.get(ls).value());
                ls++;
                rs++;
                advanced = true;
            }
            if (leftSeg.get(le).equals(rightSeg.get(re))) {
                suffix = leftSeg.get(le).value() + suffix;
                if (ls != le) {
                    le--;
                } else {
                    leftConsumed = true;
                }
                if (rs != re) {
                    re--;
                } else {
                    rightConsumed = true;
                }
                advanced = true;
            }
            if (!advanced) {
                break;
            }
        }

        if (ls == le && rs != re) {
            if (leftSeg.get(le).equals(rightSeg.get(re))) {
                suffix = rightSeg.get(re).value() + suffix;
                re--;
                leftConsumed = true;
            } else if (leftSeg.get(ls).equals(rightSeg.get(rs))) {
                prefix.append(rightSeg.get(rs).value());
                rs++;
                leftConsumed = true;
            }
        } else if (rs == re && ls != le) {
            if (leftSeg.get(le).equals(rightSeg.get(rs))) {
                suffix = leftSeg.get(le).value() + suffix;
                le--;
                rightConsumed = true;
            } else if (leftSeg.get(ls).equals(rightSeg.get(rs))) {
                prefix.append(leftSeg.get(ls).value());
                ls++;
                rightConsumed = true;
            }
        } else if (ls == le && rs == re && leftSeg.get(ls).equals(rightSeg.get(rs))
                && leftSeg.get(ls).type() == Type.MARKER) {
            prefix.append(rightSeg.get(rs).value());
            leftConsumed = true;
            rightConsumed = true;
        }

        String remainderLeft = join(leftSeg, ls, leftConsumed ? le : le + 1);
        String remainderRight = join(rightSeg, rs, rightConsumed ? re : re + 1);

        boolean textSuffix = leftSeg.get(le).type() == Type.TEXT && rightSeg.get(re).type() == Type.TEXT;
        boolean textPrefix = rightSeg.get(rs).type() == Type.TEXT && leftSeg.get(ls).type() == Type.TEXT;

        int suffixLen = textSuffix ? commonSuffixLength(remainderLeft, remainderRight) : 0;
        int prefixLen = textPrefix
                ? commonPrefixLength(remainderLeft.substring(0, remainderLeft.length() - suffixLen),
                        remainderRight.substring(0, remainderRight.length() - suffixLen))
                : 0;

        prefix.append(remainderLeft, 0, prefixLen);
        suffix = remainderLeft.substring(remainderLeft.length() - suffixLen) + suffix;
        String leftOnly = remainderLeft.substring(prefixLen, remainderLeft.length() - suffixLen);
        String rightOnly = remainderRight.substring(prefixLen, remainderRight.length() - suffixLen);

        if (leftOnly.isEmpty() && rightOnly.isEmpty()) {
            // No difference: the whole first text is the shared representation.
            return new Diff(left, "", "", "");
        }
        if (leftOnly.isEmpty() && !rightOnly.isEmpty() && left.length() <= right.length()
                && right.startsWith(left)) {
            // The first text is a prefix of the second: the second's tail is its unique part, not a suffix.
            return new Diff(left, "", "", right.substring(left.length()));
        }
        return new Diff(prefix.toString(), suffix, leftOnly, rightOnly);
    }

    private static String join(List<Segment> segments, int from, int toExclusive) {
        StringBuilder out = new StringBuilder();
        for (int i = from; i < toExclusive && i < segments.size(); i++) {
            out.append(segments.get(i).value());
        }
        return out.toString();
    }

    /** The segments' text joined. */
    static String join(List<Segment> segments) {
        return join(segments, 0, segments.size());
    }

    /** The value of the last marker in {@code segments}, or null when there is none. */
    static String lastMarker(List<Segment> segments) {
        for (int i = segments.size() - 1; i >= 0; i--) {
            if (segments.get(i).type() == Type.MARKER) {
                return segments.get(i).value();
            }
        }
        return null;
    }

    /** The value of the first marker in {@code segments}, or null when there is none. */
    static String firstMarker(List<Segment> segments) {
        for (Segment s : segments) {
            if (s.type() == Type.MARKER) {
                return s.value();
            }
        }
        return null;
    }

    /** The characters of {@code text} with JSON-safe escapes, for a rule literal. */
    static String literal(String text) {
        StringBuilder out = new StringBuilder(text.length() + 8);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '\\': out.append("\\\\"); break;
                case '"': out.append("\\\""); break;
                case '\n': out.append("\\n"); break;
                case '\r': out.append("\\r"); break;
                case '\t': out.append("\\t"); break;
                default: out.append(c);
            }
        }
        return out.toString();
    }

    /** An unmodifiable copy of {@code segments}. */
    static List<Segment> unmodifiable(List<Segment> segments) {
        return Collections.unmodifiableList(new ArrayList<>(segments));
    }
}
