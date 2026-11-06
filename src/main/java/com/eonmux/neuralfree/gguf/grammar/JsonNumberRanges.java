package com.eonmux.neuralfree.gguf.grammar;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

/**
 * GBNF for bounded JSON numbers: llama.cpp's {@code _build_min_max_int} digit ranges for integers
 * ({@code common/json-schema-to-grammar.cpp:45-227}, ported verbatim) and the decimal ranges of bounded
 * {@code number} schemas, xgrammar's number-range support.
 *
 * <p>A decimal range matches the numbers {@code -?I(.F)?} whose value lies within the bounds, each bound
 * inclusive or exclusive: an integral part {@code I} without leading zeros and an optional fraction {@code F} of at
 * most 16 digits (llama.cpp's {@code decimal-part}), never an exponent and never {@code -0}. The range splits by sign
 * and by integral part: the integral parts strictly inside the bounds take any fraction, and the bound's own integral
 * part takes the fractions on the admitted side of the bound's fraction digits, compared digit by digit. An upper
 * magnitude at or beyond the 16-digit integral parts the generic {@code number} rule writes counts as unbounded.</p>
 */
final class JsonNumberRanges {

    /** Longest fraction a bounded number writes, llama.cpp's {@code decimal-part} {@code [0-9]{1,16}}. */
    static final int MAX_FRACTION_DIGITS = 16;

    /** Longest integral part an unbounded magnitude writes, llama.cpp's {@code integral-part}. */
    static final int MAX_INTEGRAL_DIGITS = 16;

    private static final String ANY_FRACTION = "(\".\" [0-9]{1," + MAX_FRACTION_DIGITS + "})?";

    private JsonNumberRanges() {
    }

    // ------------------------------------------------------------------ decimal ranges

    /**
     * The GBNF alternatives matching the JSON numbers between {@code min} and {@code max}.
     *
     * @param min          the lower bound, or null when unbounded below
     * @param minExclusive whether {@code min} itself is excluded
     * @param max          the upper bound, or null when unbounded above
     * @param maxExclusive whether {@code max} itself is excluded
     * @return the rule text, or null when no number lies within the bounds
     */
    static String decimalRange(BigDecimal min, boolean minExclusive, BigDecimal max, boolean maxExclusive) {
        List<String> alternatives = new ArrayList<>();
        if (min == null || min.signum() < 0) {
            // x < 0 written as "-" m: m > 0, m <= -min, m >= -max
            BigDecimal low = max == null || max.signum() >= 0 ? BigDecimal.ZERO : max.negate();
            boolean lowExclusive = max == null || max.signum() >= 0 || maxExclusive;
            String negative = magnitude(low, lowExclusive, min == null ? null : min.negate(), minExclusive);
            if (negative != null) {
                alternatives.add("\"-\" (" + negative + ")");
            }
        }
        if (max == null || max.signum() >= 0) {
            boolean fromZero = min == null || min.signum() < 0;
            String positive = magnitude(fromZero ? BigDecimal.ZERO : min, !fromZero && minExclusive,
                    max, maxExclusive);
            if (positive != null) {
                alternatives.add(positive);
            }
        }
        return alternatives.isEmpty() ? null : String.join(" | ", alternatives);
    }

    /** The unsigned numbers {@code m} with {@code low <(=) m <(=) high}; {@code low >= 0}, {@code high} nullable. */
    private static String magnitude(BigDecimal low, boolean lowExclusive, BigDecimal high, boolean highExclusive) {
        BigInteger lowIntegral = low.setScale(0, RoundingMode.FLOOR).toBigIntegerExact();
        String lowFraction = fractionDigits(low);
        int unboundedDigits = Math.max(MAX_INTEGRAL_DIGITS, lowIntegral.add(BigInteger.ONE).toString().length());
        if (high != null && high.compareTo(BigDecimal.TEN.pow(unboundedDigits)) >= 0) {
            high = null; // every magnitude the unbounded form writes stays below it
        }
        List<String> alternatives = new ArrayList<>();
        if (high == null) {
            addAlternative(alternatives, lowIntegral.toString(),
                    fraction(lowFraction, lowExclusive, null, false));
            alternatives.add("(" + naturalRange(lowIntegral.add(BigInteger.ONE), null) + ") " + ANY_FRACTION);
            return String.join(" | ", alternatives);
        }
        BigInteger highIntegral = high.setScale(0, RoundingMode.FLOOR).toBigIntegerExact();
        String highFraction = fractionDigits(high);
        int order = lowIntegral.compareTo(highIntegral);
        if (order > 0) {
            return null;
        }
        if (order == 0) {
            addAlternative(alternatives, lowIntegral.toString(),
                    fraction(lowFraction, lowExclusive, highFraction, highExclusive));
        } else {
            addAlternative(alternatives, lowIntegral.toString(), fraction(lowFraction, lowExclusive, null, false));
            BigInteger inner = lowIntegral.add(BigInteger.ONE);
            BigInteger innerHigh = highIntegral.subtract(BigInteger.ONE);
            if (inner.compareTo(innerHigh) <= 0) {
                alternatives.add("(" + naturalRange(inner, innerHigh) + ") " + ANY_FRACTION);
            }
            addAlternative(alternatives, highIntegral.toString(), fraction(null, false, highFraction, highExclusive));
        }
        return alternatives.isEmpty() ? null : String.join(" | ", alternatives);
    }

    private static void addAlternative(List<String> alternatives, String integral, String fraction) {
        if (fraction != null) {
            alternatives.add(fraction.isEmpty() ? "\"" + integral + "\"" : "\"" + integral + "\" " + fraction);
        }
    }

    /** The fraction digits of {@code value} without trailing zeros, empty for an integral value. */
    private static String fractionDigits(BigDecimal value) {
        BigDecimal fraction = value.subtract(value.setScale(0, RoundingMode.FLOOR));
        if (fraction.signum() == 0) {
            return "";
        }
        String plain = fraction.stripTrailingZeros().toPlainString();
        return plain.substring(plain.indexOf('.') + 1);
    }

    /**
     * The optional {@code "." F} part whose value {@code 0.F} lies between {@code 0.low} and {@code 0.high} (null:
     * unbounded on that side); an empty text when only the absent fraction qualifies, null when nothing does.
     */
    private static String fraction(String low, boolean lowExclusive, String high, boolean highExclusive) {
        FractionSet set = fractionSet(0, low, lowExclusive, high, highExclusive);
        if (set == null) {
            return null;
        }
        if (set.digits == null) {
            return "";
        }
        return "(\".\" (" + set.digits + "))" + (set.empty ? "?" : "");
    }

    /** A set of fraction digit strings: whether it holds the empty string, and the rule of its non-empty ones. */
    private static final class FractionSet {
        final boolean empty;
        final String digits;

        FractionSet(boolean empty, String digits) {
            this.empty = empty;
            this.digits = digits;
        }
    }

    /**
     * The digit strings {@code s} of at most {@code MAX_FRACTION_DIGITS - depth} digits with {@code 0.low <(=) 0.s
     * <(=) 0.high}, the bounds being the digits still to compare (trailing zeros stripped; null when that side no
     * longer constrains). Null when the set is empty.
     */
    private static FractionSet fractionSet(int depth, String low, boolean lowExclusive, String high,
            boolean highExclusive) {
        int remaining = MAX_FRACTION_DIGITS - depth;
        if (low != null && low.isEmpty() && !lowExclusive) {
            low = null; // >= 0 always holds
        }
        if (low == null && high == null) {
            return new FractionSet(true, remaining == 0 ? null
                    : remaining == 1 ? "[0-9]" : "[0-9]{1," + remaining + "}");
        }
        boolean emptyAdmitted = low == null && (high == null || !high.isEmpty() || !highExclusive);
        List<String> alternatives = new ArrayList<>();
        int digit = 0;
        while (remaining > 0 && digit <= 9) {
            String nextLow = nextLow(low, lowExclusive, digit);
            String nextHigh = nextHigh(high, highExclusive, digit);
            int last = digit;
            while (last < 9 && sameState(low, lowExclusive, high, highExclusive, digit, last + 1)) {
                last++;
            }
            if (!REJECT.equals(nextLow) && !REJECT.equals(nextHigh)) {
                FractionSet rest = fractionSet(depth + 1, nextLow, lowExclusive, nextHigh, highExclusive);
                if (rest != null) {
                    alternatives.add(digitClass(digit, last) + suffix(rest));
                }
            }
            digit = last + 1;
        }
        if (alternatives.isEmpty() && !emptyAdmitted) {
            return null;
        }
        return new FractionSet(emptyAdmitted, alternatives.isEmpty() ? null : String.join(" | ", alternatives));
    }

    /** Marks a digit the bound rules out. */
    private static final String REJECT = "\u0000reject";

    /** The lower bound left after {@code digit}; null once satisfied, {@link #REJECT} when violated. */
    private static String nextLow(String low, boolean exclusive, int digit) {
        if (low == null) {
            return null;
        }
        if (low.isEmpty()) {
            // only reached exclusive: the digits so far equal the bound, so a non-zero digit exceeds it
            return digit > 0 ? null : "";
        }
        int bound = low.charAt(0) - '0';
        return digit > bound ? null : digit == bound ? low.substring(1) : REJECT;
    }

    /** The upper bound left after {@code digit}; null once satisfied, {@link #REJECT} when violated. */
    private static String nextHigh(String high, boolean exclusive, int digit) {
        if (high == null) {
            return null;
        }
        if (high.isEmpty()) {
            // the digits so far equal the bound: only zeros keep 0.s <= 0.high, none keeps it strictly below
            return exclusive || digit > 0 ? REJECT : "";
        }
        int bound = high.charAt(0) - '0';
        return digit < bound ? null : digit == bound ? high.substring(1) : REJECT;
    }

    private static boolean sameState(String low, boolean lowExclusive, String high, boolean highExclusive,
            int a, int b) {
        return equal(nextLow(low, lowExclusive, a), nextLow(low, lowExclusive, b))
                && equal(nextHigh(high, highExclusive, a), nextHigh(high, highExclusive, b));
    }

    private static boolean equal(String a, String b) {
        return a == null ? b == null : a.equals(b);
    }

    private static String digitClass(int from, int to) {
        return from == to ? "[" + from + "]" : "[" + from + "-" + to + "]";
    }

    /** The text following a digit whose remaining digits form {@code rest}. */
    private static String suffix(FractionSet rest) {
        if (rest.digits == null) {
            return "";
        }
        return " (" + rest.digits + ")" + (rest.empty ? "?" : "");
    }

    /**
     * The canonical natural numbers from {@code from} to {@code to} (null: up to {@value #MAX_INTEGRAL_DIGITS}
     * digits, or {@code from}'s own length when longer), as digit-length bands of llama.cpp's uniform ranges.
     */
    static String naturalRange(BigInteger from, BigInteger to) {
        String fromDigits = from.toString();
        StringBuilder out = new StringBuilder();
        if (to == null) {
            int length = fromDigits.length();
            uniformRange(fromDigits, "9".repeat(length), out);
            if (length < MAX_INTEGRAL_DIGITS) {
                out.append(" | [1-9] ");
                moreDigits(length, MAX_INTEGRAL_DIGITS - 1, out);
            }
            return out.toString();
        }
        String toDigits = to.toString();
        for (int digits = fromDigits.length(); digits < toDigits.length(); digits++) {
            uniformRange(fromDigits, "9".repeat(digits), out);
            fromDigits = "1" + "0".repeat(digits);
            out.append(" | ");
        }
        uniformRange(fromDigits, toDigits, out);
        return out.toString();
    }

    // ------------------------------------------------------------------ integer ranges

    /** The least magnitude of 17 digits, beyond the integral parts llama.cpp's {@code integer} rule writes. */
    private static final BigInteger SEVENTEEN_DIGITS = BigInteger.TEN.pow(MAX_INTEGRAL_DIGITS);

    /**
     * The GBNF matching the JSON integers from {@code min} to {@code max}, each null when unbounded; null when no
     * bound applies. Bounds below 17 digits go through {@link #buildMinMaxInt}, a bound of 17 or more digits on
     * the far side of zero is left out (the unbounded side writes 16 digits at most, all within it), and a range
     * lying wholly beyond 16 digits is written as a natural range of its own.
     */
    static String integerRange(BigInteger min, BigInteger max) {
        if (min != null && min.compareTo(SEVENTEEN_DIGITS) >= 0) {
            return naturalRange(min, max);
        }
        if (max != null && max.compareTo(SEVENTEEN_DIGITS.negate()) <= 0) {
            return "\"-\" (" + naturalRange(max.negate(), min == null ? null : min.negate()) + ")";
        }
        boolean hasMin = min != null && min.compareTo(SEVENTEEN_DIGITS.negate()) > 0;
        boolean hasMax = max != null && max.compareTo(SEVENTEEN_DIGITS) < 0;
        if (!hasMin && !hasMax) {
            return null;
        }
        StringBuilder out = new StringBuilder();
        buildMinMaxInt(hasMin ? min.longValueExact() : Long.MIN_VALUE, hasMax ? max.longValueExact() : Long.MAX_VALUE,
                out, MAX_INTEGRAL_DIGITS, true);
        return out.toString();
    }

    /**
     * {@code _build_min_max_int} ({@code :45-227}) ported verbatim; the exact output of every
     * bounded-integer case in llama.cpp's tests is the golden reference.
     */
    static void buildMinMaxInt(long minValue, long maxValue, StringBuilder out,
            int decimalsLeft, boolean topLevel) {
        boolean hasMin = minValue != Long.MIN_VALUE;
        boolean hasMax = maxValue != Long.MAX_VALUE;

        if (hasMin && hasMax) {
            if (minValue < 0 && maxValue < 0) {
                out.append("\"-\" (");
                buildMinMaxInt(-maxValue, -minValue, out, decimalsLeft, true);
                out.append(')');
                return;
            }
            if (minValue < 0) {
                out.append("\"-\" (");
                buildMinMaxInt(0, -minValue, out, decimalsLeft, true);
                out.append(") | ");
                minValue = 0;
            }

            String minDigits = Long.toString(minValue);
            String maxDigits = Long.toString(maxValue);
            for (int digits = minDigits.length(); digits < maxDigits.length(); digits++) {
                uniformRange(minDigits, "9".repeat(digits), out);
                minDigits = "1" + "0".repeat(digits);
                out.append(" | ");
            }
            uniformRange(minDigits, maxDigits, out);
            return;
        }

        int lessDecimals = Math.max(decimalsLeft - 1, 1);

        if (hasMin) {
            if (minValue < 0) {
                out.append("\"-\" (");
                buildMinMaxInt(Long.MIN_VALUE, -minValue, out, decimalsLeft, false);
                out.append(") | [0] | [1-9] ");
                moreDigits(0, decimalsLeft - 1, out);
            } else if (minValue == 0) {
                if (topLevel) {
                    out.append("[0] | [1-9] ");
                    moreDigits(0, lessDecimals, out);
                } else {
                    moreDigits(1, decimalsLeft, out);
                }
            } else if (minValue <= 9) {
                char c = (char) ('0' + minValue);
                char rangeStart = topLevel ? '1' : '0';
                if (c > rangeStart) {
                    digitRange(rangeStart, (char) (c - 1), out);
                    out.append(' ');
                    moreDigits(1, lessDecimals, out);
                    out.append(" | ");
                }
                digitRange(c, '9', out);
                out.append(' ');
                moreDigits(0, lessDecimals, out);
            } else {
                String digits = Long.toString(minValue);
                int length = digits.length();
                char c = digits.charAt(0);

                if (c > '1') {
                    digitRange(topLevel ? '1' : '0', (char) (c - 1), out);
                    out.append(' ');
                    moreDigits(length, lessDecimals, out);
                    out.append(" | ");
                }
                digitRange(c, c, out);
                out.append(" (");
                buildMinMaxInt(Long.parseLong(digits.substring(1)), Long.MAX_VALUE, out,
                        lessDecimals, false);
                out.append(')');
                if (c < '9') {
                    out.append(" | ");
                    digitRange((char) (c + 1), '9', out);
                    out.append(' ');
                    moreDigits(length - 1, lessDecimals, out);
                }
            }
            return;
        }

        if (hasMax) {
            if (maxValue >= 0) {
                if (topLevel) {
                    out.append("\"-\" [1-9] ");
                    moreDigits(0, lessDecimals, out);
                    out.append(" | ");
                }
                buildMinMaxInt(0, maxValue, out, decimalsLeft, true);
            } else {
                out.append("\"-\" (");
                buildMinMaxInt(-maxValue, Long.MAX_VALUE, out, decimalsLeft, false);
                out.append(')');
            }
            return;
        }
        throw new JsonSchemaToGbnf.SchemaErrorException("", "At least one of min_value or max_value must be set");
    }

    private static void digitRange(char from, char to, StringBuilder out) {
        out.append('[');
        if (from == to) {
            out.append(from);
        } else {
            out.append(from).append('-').append(to);
        }
        out.append(']');
    }

    private static void moreDigits(int minDigits, int maxDigits, StringBuilder out) {
        out.append("[0-9]");
        if (minDigits == maxDigits && minDigits == 1) {
            return;
        }
        out.append('{').append(minDigits);
        if (maxDigits != minDigits) {
            out.append(',');
            if (maxDigits != Integer.MAX_VALUE) {
                out.append(maxDigits);
            }
        }
        out.append('}');
    }

    private static void uniformRange(String from, String to, StringBuilder out) {
        int i = 0;
        while (i < from.length() && i < to.length() && from.charAt(i) == to.charAt(i)) {
            i++;
        }
        if (i > 0) {
            out.append('"').append(from, 0, i).append('"');
        }
        if (i < from.length() && i < to.length()) {
            if (i > 0) {
                out.append(' ');
            }
            int subLength = from.length() - i - 1;
            if (subLength > 0) {
                String fromSub = from.substring(i + 1);
                String toSub = to.substring(i + 1);
                String subZeros = "0".repeat(subLength);
                String subNines = "9".repeat(subLength);

                boolean toReached = false;
                out.append('(');
                if (fromSub.equals(subZeros)) {
                    digitRange(from.charAt(i), (char) (to.charAt(i) - 1), out);
                    out.append(' ');
                    moreDigits(subLength, subLength, out);
                } else {
                    out.append('[').append(from.charAt(i)).append("] (");
                    uniformRange(fromSub, subNines, out);
                    out.append(')');
                    if (from.charAt(i) < to.charAt(i) - 1) {
                        out.append(" | ");
                        if (toSub.equals(subNines)) {
                            digitRange((char) (from.charAt(i) + 1), to.charAt(i), out);
                            toReached = true;
                        } else {
                            digitRange((char) (from.charAt(i) + 1),
                                    (char) (to.charAt(i) - 1), out);
                        }
                        out.append(' ');
                        moreDigits(subLength, subLength, out);
                    }
                }
                if (!toReached) {
                    out.append(" | ");
                    digitRange(to.charAt(i), to.charAt(i), out);
                    out.append(' ');
                    uniformRange(subZeros, toSub, out);
                }
                out.append(')');
            } else {
                out.append('[').append(from.charAt(i)).append('-').append(to.charAt(i))
                        .append(']');
            }
        }
    }
}
