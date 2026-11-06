package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.SequenceMemory;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.BooleanSupplier;

/**
 * One lookahead (Jacobi) decoding run over a group of {@code W + G + 1} sequences: the observed n-gram table, the
 * active verification n-grams and the {@code N - 1} Jacobi levels of {@code W} guesses, with the step batch and the
 * KV commit of llama.cpp's {@code examples/lookahead/lookahead.cpp}.
 *
 * <p>The run works on a base sequence id: sequence {@code base} is the accepted sequence, {@code base + 1 ..
 * base + W} are the Jacobi diagonals and {@code base + W + 1 .. base + W + G} verify one observed n-gram each.
 * {@link LookaheadDecoder} runs a group at id 0; the batch engine runs one group per running request. The caller
 * owns the output side ({@link SequenceRun}: sampling, stops, routing) and the memory (initializing the group,
 * committing the accepted sequence, freeing it). Mutable, owned by one thread at a time, never thread-safe.
 */
public final class LookaheadRun {

    /** One step's forward batch and the output-row indices the walk reads. */
    public static final class Step {
        private final ForwardBatch batch;
        private final int idRow;
        private final int[][] ngramRows;
        private final int[] lastLevelRows;
        private final int rows;
        private final int outputs;

        Step(ForwardBatch batch, int idRow, int[][] ngramRows, int[] lastLevelRows, int rows, int outputs) {
            this.batch = batch;
            this.idRow = idRow;
            this.ngramRows = ngramRows;
            this.lastLevelRows = lastLevelRows;
            this.rows = rows;
            this.outputs = outputs;
        }

        /** The step's forward batch, over the run's sequence group. */
        public ForwardBatch batch() {
            return batch;
        }

        /** The rows the step's batch holds. */
        public int rows() {
            return rows;
        }

        /** The logits rows the step's batch returns ({@link #rows()} minus the rows that ask no logits). */
        public int outputs() {
            return outputs;
        }
    }

    /** What one step emitted and confirmed. Immutable. */
    public static final class Result {
        private final int[] tokens;
        private final int accepted;
        private final int best;
        private final StopReason reason;

        Result(int[] tokens, int accepted, int best, StopReason reason) {
            this.tokens = tokens;
            this.accepted = accepted;
            this.best = best;
            this.reason = reason;
        }

        /** The tokens the step sampled, in emission order; the last one waits to be fed at the next position. */
        public int[] tokens() {
            return tokens.clone();
        }

        /** The tokens the step sampled. */
        public int emitted() {
            return tokens.length;
        }

        /**
         * The tokens an observed n-gram beyond the step's first emitted token confirmed ({@code emitted - 1}, 0
         * when the step emitted none or only the target's own token).
         */
        public int accepted() {
            return accepted;
        }

        /** The n-gram sequence offset the last confirmation kept, 0 when no n-gram matched. */
        public int best() {
            return best;
        }

        /** Why the step ended the generation, null while it continues. */
        public StopReason reason() {
            return reason;
        }

        /** Whether the step ended the generation. */
        public boolean stopped() {
            return reason != null;
        }

        /** Whether the step stopped on the caller's cancellation poll, before a token the run would have emitted. */
        public boolean cancelled() {
            return reason == StopReason.CANCELLED;
        }
    }

    private final SequenceMemory sequences;
    private final int base;
    private final int window;
    private final int ngramSize;
    private final int maxNgrams;
    private final LookaheadNgrams table;
    private final SequenceRun run;
    private final int[][] levels;
    private final int[] previous;
    private final boolean[] active;
    /** The active verification n-grams; only {@code [0, count)} are live. */
    private final int[][] current;
    private final int[] gram;
    private int count;
    private boolean initialized;

    /**
     * @param sequences the paged memory holding the group
     * @param base      the accepted sequence's id; the group spans {@code base .. base + sequenceCount - 1}
     * @param options   the lookahead sizes
     * @param run       the output side the walk samples and emits through
     * @param vocabSize the vocabulary size, for the initial Jacobi guesses
     * @throws IllegalArgumentException when the memory does not hold the whole group below its base
     */
    public LookaheadRun(SequenceMemory sequences, int base, LookaheadOptions options, SequenceRun run, int vocabSize) {
        this.sequences = Objects.requireNonNull(sequences, "sequences");
        this.run = Objects.requireNonNull(run, "run");
        Objects.requireNonNull(options, "options");
        if (!options.enabled()) {
            throw new IllegalArgumentException("lookahead options are off; a lookahead run needs them enabled");
        }
        int sequencesNeeded = options.sequenceCount();
        if (base < 0 || base + sequencesNeeded > sequences.maxSequences()) {
            throw new IllegalArgumentException("a lookahead group of " + sequencesNeeded + " sequences does not fit"
                    + " at id " + base + " of a memory holding " + sequences.maxSequences());
        }
        this.base = base;
        this.window = options.window();
        this.ngramSize = options.ngramSize();
        this.maxNgrams = options.maxNgrams();
        this.table = new LookaheadNgrams(ngramSize, maxNgrams);
        this.levels = new int[ngramSize - 1][window];
        this.previous = new int[window];
        Arrays.fill(previous, -1);
        this.active = new boolean[maxNgrams];
        this.current = new int[maxNgrams][ngramSize];
        this.gram = new int[ngramSize - 1];
        for (int j = 0; j < levels.length; j++) {
            for (int i = 0; i < window; i++) {
                // any in-vocabulary guess starts the iterations; the update replaces each after one step
                levels[j][i] = vocabSize > 0 ? i % vocabSize : 0;
            }
        }
    }

    /** Takes this run's observed n-grams and Jacobi grid to a group at {@code newBase}, for a re-admitted slot. */
    public LookaheadRun rebased(int newBase) {
        if (newBase < 0 || newBase + sequenceCount() > sequences.maxSequences()) {
            throw new IllegalArgumentException("a lookahead group of " + sequenceCount() + " sequences does not fit"
                    + " at id " + newBase + " of a memory holding " + sequences.maxSequences());
        }
        return new LookaheadRun(this, newBase);
    }

    /** A copy of {@code src} over another group base: the n-gram table and Jacobi grid carry over, the group's
     *  copied prefix does not (the slot re-initializes it after its recompute). */
    private LookaheadRun(LookaheadRun src, int newBase) {
        this.sequences = src.sequences;
        this.base = newBase;
        this.window = src.window;
        this.ngramSize = src.ngramSize;
        this.maxNgrams = src.maxNgrams;
        this.table = src.table;
        this.run = src.run;
        this.levels = src.levels;
        this.previous = src.previous;
        this.active = src.active;
        this.current = src.current;
        this.gram = src.gram;
        this.count = src.count;
        this.initialized = false;
    }

    /** The accepted sequence's id. */
    public int base() {
        return base;
    }

    /** The sequences the run's group spans, {@code W + G + 1}. */
    public int sequenceCount() {
        return window + maxNgrams + 1;
    }

    /** The Jacobi window {@code W}. */
    public int window() {
        return window;
    }

    /** The n-gram size {@code N}. */
    public int ngramSize() {
        return ngramSize;
    }

    /** The verification n-grams {@code G}. */
    public int maxNgrams() {
        return maxNgrams;
    }

    /** Whether {@link #initialize()} ran (the accepted sequence was copied over the group). */
    public boolean initialized() {
        return initialized;
    }

    /**
     * Copies the accepted sequence over the rest of the group, so every sequence holds the same prefix: llama.cpp's
     * {@code llama_memory_seq_cp} of the first step. The pages are shared until a sequence writes them.
     */
    public void initialize() {
        int n = sequenceCount();
        for (int s = 1; s < n; s++) {
            sequences.remove(base + s, 0, -1);
            sequences.copy(base, base + s, 0, -1);
        }
        initialized = true;
    }

    /**
     * Builds one step's batch: the input token at each sequence's next position, the active n-grams' tokens, the
     * {@code W - 1} shared first-level guesses and the {@code N - 2} later Jacobi levels, exactly the rows of
     * llama.cpp's example.
     */
    public Step step(int token, int nPast) {
        candidates(token);
        ForwardBatch.Builder b = ForwardBatch.builder(stepRows());
        int out = 0;
        int rows = 0;
        int idRow = -1;
        int n = sequenceCount();
        for (int s = 0; s < n; s++) {
            b.add(token, nPast, base + s, s == 0);
            rows++;
            if (s == 0) {
                idRow = out++;
            }
        }
        int[][] ngramRow = new int[maxNgrams][ngramSize];
        for (int g = 0; g < count; g++) {
            for (int j = 0; j < ngramSize - 1; j++) {
                b.add(current[g][j + 1], nPast + j + 1, base + window + 1 + g, true);
                ngramRow[g][j + 1] = out++;
                rows++;
            }
        }
        for (int i = 1; i < window; i++) {
            for (int s = i + 1; s <= window; s++) {
                b.add(levels[0][i], nPast + i, base + s, false);
                rows++;
            }
        }
        int[] lastLevel = new int[window];
        for (int j = 1; j < ngramSize - 1; j++) {
            boolean last = j == ngramSize - 2;
            for (int i = 0; i < window; i++) {
                b.add(levels[j][i], nPast + j + i, base + i + 1, last);
                rows++;
                if (last) {
                    lastLevel[i] = out++;
                }
            }
        }
        return new Step(b.build(), idRow, ngramRow, lastLevel, rows, out);
    }

    /** The positions one step spans at most, as {@link LookaheadOptions#positionSpan()}. */
    public int positionSpan() {
        return window >= 2 ? window + ngramSize - 2 : ngramSize;
    }

    /** The rows a step's batch holds at most, as {@link LookaheadOptions#stepRows()}. */
    private int stepRows() {
        return sequenceCount() + maxNgrams * (ngramSize - 1) + window * (window - 1) / 2
                + window * (ngramSize - 2);
    }

    /**
     * Walks one step's rows: samples the target's token from the input row, then keeps the active n-grams whose
     * next token the sample equals, sampling one token per confirmation and shifting the Jacobi grid. Returns the
     * emitted tokens, the confirmations and the stop the walk reached (a token-level stop from the run, or the
     * caller's cancellation poll).
     *
     * @param step      the step's batch and row indices
     * @param out       the batch's returned rows
     * @param cancelled the caller's cancellation poll, consulted before every emission
     */
    public Result walk(Step step, float[][] out, BooleanSupplier cancelled) {
        Objects.requireNonNull(step, "step");
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(cancelled, "cancelled");
        if (out.length != step.outputs) {
            throw new IllegalStateException("the lookahead step returned " + out.length + " logits rows for a batch"
                    + " of " + step.outputs);
        }
        int[] tokens = new int[ngramSize];
        int emitted = 0;
        int best = 0;
        StopReason reason = null;
        for (int v = 0; v < ngramSize; v++) {
            int row;
            if (v == 0) {
                row = step.idRow;
            } else {
                int g = firstActive();
                if (g < 0) {
                    break;
                }
                best = window + 1 + g;
                row = step.ngramRows[g][v];
            }
            if (cancelled.getAsBoolean()) {
                reason = StopReason.CANCELLED;
                break;
            }
            int token = run.next(out[row]);
            tokens[emitted++] = token;
            SequenceRun.Step offered = run.offer(token);
            if (offered.stopped()) {
                reason = offered.reason();
                break;
            }
            verify(v, token);
            advance(v, out, step.lastLevelRows);
        }
        return new Result(Arrays.copyOf(tokens, emitted), Math.max(0, emitted - 1), best, reason);
    }

    /**
     * Keeps the rows of the accepted prefix and makes the accepted sequence the group's base: the rows past
     * {@code nPast} leave every sequence, and when an n-gram matched, its sequence becomes the base and is copied
     * over the others. {@code best} stays 0 only for a step whose first sampled token matched no n-gram, where every
     * sequence already holds just the accepted prefix.
     */
    public void commit(int nPast, int best) {
        int n = sequenceCount();
        for (int s = 0; s < n; s++) {
            sequences.remove(base + s, nPast, -1);
        }
        if (best == 0) {
            return;
        }
        sequences.remove(base, 0, -1);
        sequences.copy(base + best, base, 0, -1);
        for (int s = 1; s < n; s++) {
            sequences.remove(base + s, 0, -1);
            sequences.copy(base, base + s, 0, -1);
        }
    }

    /** The first active verification n-gram, or -1. */
    private int firstActive() {
        for (int g = 0; g < count; g++) {
            if (active[g]) {
                return g;
            }
        }
        return -1;
    }

    /** Rebuilds the verification candidates from the table for {@code token}, the step's input token. */
    private void candidates(int token) {
        count = table.count(token);
        for (int g = 0; g < count; g++) {
            current[g][0] = token;
            int[] continuation = table.gram(token, g);
            System.arraycopy(continuation, 0, current[g], 1, continuation.length);
            active[g] = true;
        }
    }

    /** Deactivates the candidates whose next token differs from the sampled one (all after the last level). */
    private void verify(int v, int token) {
        if (v == ngramSize - 1) {
            for (int g = 0; g < count; g++) {
                active[g] = false;
            }
            return;
        }
        for (int g = 0; g < count; g++) {
            if (active[g] && current[g][v + 1] != token) {
                active[g] = false;
            }
        }
    }

    /** Shifts the Jacobi grid one diagonal, refills its last level and observes the step's n-gram. */
    private void advance(int v, float[][] out, int[] lastLevelRow) {
        shift(v, out, lastLevelRow);
        observe(v);
    }

    /** Shifts the Jacobi grid one diagonal and refills its last level, as llama.cpp's example does. */
    private void shift(int v, float[][] out, int[] lastLevelRow) {
        System.arraycopy(levels[0], 0, previous, 0, window);
        int last = levels.length - 1;
        // the shifted-out first level becomes the refilled last one: rotating references keeps the grids distinct
        // arrays, which copying the old last level back would alias at N = 3
        int[] first = levels[0];
        for (int j = 0; j < last; j++) {
            levels[j] = levels[j + 1];
        }
        levels[last] = first;
        if (v == 0) {
            for (int i = 0; i < window; i++) {
                levels[last][i] = run.sampler().sample(out[lastLevelRow[i]]);
            }
        } else {
            System.arraycopy(levels[0], 0, levels[last], 0, window);
        }
    }

    /** Feeds the diagonals into the n-gram table at a step's first level. */
    private void observe(int v) {
        if (v != 0) {
            return;
        }
        for (int f = 0; f < window; f++) {
            if (previous[f] < 0) {
                continue;
            }
            for (int j = 0; j < gram.length; j++) {
                gram[j] = levels[j][f];
            }
            table.add(previous[f], gram);
        }
    }

    @Override
    public String toString() {
        return "LookaheadRun{base=" + base + ", window=" + window + ", ngram=" + ngramSize + ", maxNgrams="
                + maxNgrams + '}';
    }
}
