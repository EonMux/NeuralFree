package com.eonmux.neuralfree.gguf.grammar;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Map;

/**
 * One numeric bound of a JSON schema: its value and whether the value itself is excluded. {@link #lower} and
 * {@link #upper} read a schema's {@code minimum}/{@code exclusiveMinimum} and {@code maximum}/{@code
 * exclusiveMaximum}: an inclusive and a numeric exclusive bound both apply, so the tighter one wins; the draft 4
 * boolean form makes the inclusive bound exclusive. Immutable.
 */
final class SchemaBound {

    final BigDecimal value;
    final boolean exclusive;

    SchemaBound(BigDecimal value, boolean exclusive) {
        this.value = value;
        this.exclusive = exclusive;
    }

    /**
     * The lower bound of {@code schema}, null when unbounded below.
     *
     * @throws JsonSchemaToGbnf.SchemaErrorException naming the keyword when a bound is not a number, or the draft
     *                                              4 flag is set without its inclusive bound
     */
    static SchemaBound lower(Map<?, ?> schema, String path) {
        return read(schema, path, "minimum", "exclusiveMinimum", true);
    }

    /** The upper bound of {@code schema}, null when unbounded above; failures as {@link #lower}. */
    static SchemaBound upper(Map<?, ?> schema, String path) {
        return read(schema, path, "maximum", "exclusiveMaximum", false);
    }

    /** The least integer this bound admits, read as a lower bound. */
    BigInteger lowestInteger() {
        return exclusive ? value.setScale(0, RoundingMode.FLOOR).toBigIntegerExact().add(BigInteger.ONE)
                : value.setScale(0, RoundingMode.CEILING).toBigIntegerExact();
    }

    /** The greatest integer this bound admits, read as an upper bound. */
    BigInteger highestInteger() {
        return exclusive ? value.setScale(0, RoundingMode.CEILING).toBigIntegerExact().subtract(BigInteger.ONE)
                : value.setScale(0, RoundingMode.FLOOR).toBigIntegerExact();
    }

    /** Whether some value lies between {@code lower} and {@code upper}, either of them null. */
    static boolean admitsValue(SchemaBound lower, SchemaBound upper) {
        if (lower == null || upper == null) {
            return true;
        }
        int order = lower.value.compareTo(upper.value);
        return order < 0 || (order == 0 && !lower.exclusive && !upper.exclusive);
    }

    private static SchemaBound read(Map<?, ?> schema, String path, String inclusiveKey, String exclusiveKey,
            boolean lower) {
        SchemaBound inclusive = schema.containsKey(inclusiveKey)
                ? new SchemaBound(decimal(schema.get(inclusiveKey), inclusiveKey, path), false) : null;
        if (!schema.containsKey(exclusiveKey)) {
            return inclusive;
        }
        Object exclusive = schema.get(exclusiveKey);
        if (exclusive instanceof Boolean) {
            if (!(Boolean) exclusive) {
                return inclusive;
            }
            if (inclusive == null) {
                throw new JsonSchemaToGbnf.SchemaErrorException(path, exclusiveKey + " true needs " + inclusiveKey);
            }
            return new SchemaBound(inclusive.value, true);
        }
        SchemaBound strict = new SchemaBound(decimal(exclusive, exclusiveKey, path), true);
        if (inclusive == null) {
            return strict;
        }
        int order = strict.value.compareTo(inclusive.value);
        return (lower ? order >= 0 : order <= 0) ? strict : inclusive;
    }

    private static BigDecimal decimal(Object value, String key, String path) {
        if (value instanceof Long || value instanceof Integer) {
            return BigDecimal.valueOf(((Number) value).longValue());
        }
        if (value instanceof BigInteger) {
            return new BigDecimal((BigInteger) value);
        }
        if (value instanceof Double && Double.isFinite((Double) value)) {
            return new BigDecimal(Double.toString((Double) value));
        }
        throw new JsonSchemaToGbnf.SchemaErrorException(path, key + " must be a number");
    }
}
