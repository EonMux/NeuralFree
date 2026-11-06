package com.eonmux.neuralfree.gguf.engine;

/**
 * The n-gram repetition stop of a request (vLLM's {@code RepetitionDetectionParams}): generation ends with
 * {@link StopReason#REPETITION} once the output ends in one pattern of {@link #minPatternSize()} to
 * {@link #maxPatternSize()} tokens repeated {@link #minCount()} times in a row, as vLLM's
 * {@code check_sequence_repetition} (vllm/v1/core/sched/utils.py) decides it. Immutable.
 *
 * <p>A {@link #maxPatternSize()} of 0 turns detection off ({@link #disabled()}); a {@link #minPatternSize()} of 0
 * means patterns from one token up.
 */
public final class RepetitionDetection {

    private static final RepetitionDetection DISABLED = new RepetitionDetection(0, 0, 0);

    private final int maxPatternSize;
    private final int minPatternSize;
    private final int minCount;

    private RepetitionDetection(int maxPatternSize, int minPatternSize, int minCount) {
        this.maxPatternSize = maxPatternSize;
        this.minPatternSize = minPatternSize;
        this.minCount = minCount;
    }

    /** No repetition detection, the default of every request. */
    public static RepetitionDetection disabled() {
        return DISABLED;
    }

    /**
     * Detection of patterns of {@code minPatternSize} to {@code maxPatternSize} tokens repeated {@code minCount}
     * times; {@code maxPatternSize} 0 returns {@link #disabled()}. Validated as vLLM's
     * {@code RepetitionDetectionParams.__post_init__}.
     *
     * @throws IllegalArgumentException when a size is negative, {@code minPatternSize} exceeds
     *                                  {@code maxPatternSize}, or detection is on with {@code minCount} below 2
     */
    public static RepetitionDetection of(int maxPatternSize, int minPatternSize, int minCount) {
        if (maxPatternSize < 0 || minPatternSize < 0 || minPatternSize > maxPatternSize) {
            throw new IllegalArgumentException("repetition_detection: max_pattern_size and min_pattern_size must be"
                    + " >= 0 with min_pattern_size <= max_pattern_size (got " + maxPatternSize + " and "
                    + minPatternSize + "); set both to 0 to disable repetition detection");
        }
        if (maxPatternSize == 0) {
            return DISABLED;
        }
        if (minCount < 2) {
            throw new IllegalArgumentException("repetition_detection: min_count must be >= 2 to detect repeated"
                    + " patterns, got " + minCount + "; set max_pattern_size to 0 to disable repetition detection");
        }
        return new RepetitionDetection(maxPatternSize, minPatternSize, minCount);
    }

    /** Whether detection is on. */
    public boolean enabled() {
        return maxPatternSize > 0;
    }

    /** The longest pattern checked, in tokens; 0 when detection is off. */
    public int maxPatternSize() {
        return maxPatternSize;
    }

    /** The shortest pattern checked, in tokens; 0 means 1. */
    public int minPatternSize() {
        return minPatternSize;
    }

    /** The back-to-back repeats of a pattern that end generation; 0 when detection is off. */
    public int minCount() {
        return minCount;
    }

    /**
     * Whether the first {@code count} entries of {@code tokens} end in a pattern of a checked size repeated
     * {@link #minCount()} times. Pattern sizes are tried from the shortest; the search ends at the first size whose
     * repeats no longer fit in {@code count} tokens.
     *
     * @throws IllegalArgumentException when {@code count} is negative or exceeds {@code tokens.length}
     */
    public boolean detects(int[] tokens, int count) {
        if (count < 0 || count > tokens.length) {
            throw new IllegalArgumentException("count " + count + " outside [0, " + tokens.length + "]");
        }
        if (!enabled()) {
            return false;
        }
        for (int size = Math.max(1, minPatternSize); size <= maxPatternSize; size++) {
            if ((long) size * minCount > count) {
                return false;
            }
            if (endsRepeating(tokens, count, size)) {
                return true;
            }
        }
        return false;
    }

    /** Whether the last {@code size} tokens equal each of the {@code minCount - 1} blocks before them. */
    private boolean endsRepeating(int[] tokens, int count, int size) {
        for (int n = 1; n <= size; n++) {
            int target = tokens[count - n];
            for (int m = 1; m < minCount; m++) {
                if (tokens[count - (size * m + n)] != target) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof RepetitionDetection)) {
            return false;
        }
        RepetitionDetection other = (RepetitionDetection) o;
        return maxPatternSize == other.maxPatternSize && minPatternSize == other.minPatternSize
                && minCount == other.minCount;
    }

    @Override
    public int hashCode() {
        return (maxPatternSize * 31 + minPatternSize) * 31 + minCount;
    }

    @Override
    public String toString() {
        return enabled() ? "RepetitionDetection{patterns=" + Math.max(1, minPatternSize) + ".." + maxPatternSize
                + ", minCount=" + minCount + '}' : "RepetitionDetection{disabled}";
    }
}
