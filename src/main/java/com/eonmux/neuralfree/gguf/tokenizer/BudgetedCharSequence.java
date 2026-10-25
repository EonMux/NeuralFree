package com.eonmux.neuralfree.gguf.tokenizer;

/**
 * A read-only {@link CharSequence} that bounds the work one regex match call may do on it.
 *
 * <p>Tokenizer regexes can come from untrusted model files, and a backtracking engine can take
 * exponential time on a hostile pattern. Every {@link #charAt} call is charged against a budget of
 * {@code 1000 × input length + 1,000,000} character reads per match call; when it runs out the
 * read throws {@link IllegalStateException} naming the pattern and the input length, which aborts
 * the match. {@link #reset} starts the budget of the next match call. Views returned by
 * {@link #subSequence} share the budget of the sequence they came from.</p>
 *
 * <p>Instances are mutable (the budget) and confined to the thread running the match.</p>
 */
public final class BudgetedCharSequence implements CharSequence {

    /** Character reads allowed per input character in one match call. */
    public static final long READS_PER_CHAR = 1000L;
    /** Character reads every match call is allowed regardless of its input length. */
    public static final long BASE_READS = 1_000_000L;

    /** The budget shared by a sequence and its views. */
    private static final class Budget {
        final String pattern;
        long remaining;
        int inputLength;

        Budget(String pattern) {
            this.pattern = pattern;
        }
    }

    private final CharSequence text;
    private final int offset;
    private final int length;
    private final Budget budget;

    /**
     * Wraps {@code text} for matching {@code pattern}; the budget starts as that of a match call
     * over the whole text.
     *
     * @param pattern the regex text, used in the exhaustion message
     */
    public BudgetedCharSequence(CharSequence text, String pattern) {
        if (text == null) throw new IllegalArgumentException("text is null");
        if (pattern == null) throw new IllegalArgumentException("pattern is null");
        this.text = text;
        this.offset = 0;
        this.length = text.length();
        this.budget = new Budget(pattern);
        reset(length);
    }

    private BudgetedCharSequence(CharSequence text, int offset, int length, Budget budget) {
        this.text = text;
        this.offset = offset;
        this.length = length;
        this.budget = budget;
    }

    /** The read budget of one match call over {@code inputLength} characters. */
    public static long budgetFor(int inputLength) {
        if (inputLength < 0) throw new IllegalArgumentException("negative input length: " + inputLength);
        return READS_PER_CHAR * inputLength + BASE_READS;
    }

    /** Starts a new match call over {@code inputLength} characters, refilling the shared budget. */
    public void reset(int inputLength) {
        budget.remaining = budgetFor(inputLength);
        budget.inputLength = inputLength;
    }

    /** The reads left in the current match call. */
    public long remaining() {
        return budget.remaining;
    }

    /** The regex text this sequence is budgeted for. */
    public String pattern() {
        return budget.pattern;
    }

    /**
     * The character at {@code index}, charged against the budget.
     *
     * @throws IllegalStateException when the current match call has used its budget
     */
    @Override
    public char charAt(int index) {
        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException("index " + index + " outside [0, " + length + ")");
        }
        if (--budget.remaining < 0) {
            throw new IllegalStateException("tokenizer regex '" + budget.pattern + "' exceeded its step budget on a "
                    + budget.inputLength + "-char input");
        }
        return text.charAt(offset + index);
    }

    @Override
    public int length() {
        return length;
    }

    /** A view of {@code [start, end)} that shares this sequence's budget. */
    @Override
    public CharSequence subSequence(int start, int end) {
        if (start < 0 || end > length || start > end) {
            throw new IndexOutOfBoundsException("range [" + start + ", " + end + ") outside [0, " + length + ")");
        }
        return new BudgetedCharSequence(text, offset + start, end - start, budget);
    }

    /** The characters of this sequence; not charged, since it is not a regex read. */
    @Override
    public String toString() {
        return text.subSequence(offset, offset + length).toString();
    }
}
