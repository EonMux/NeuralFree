package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The pre-top-k logit edits a device sampling readout applies to whole vocabulary rows before it selects each
 * row's top candidates: the fixed logit-bias stage (banned tokens set to {@code -inf}, additive biases), the
 * repetition, presence and frequency penalties, and DRY, in the sampler chain's order, over each output row's own
 * history. A row here corresponds to an output row of the forward call, so a batch of sequences carries one row
 * per sequence and each uses its own window.
 *
 * <p>{@link Sampler#prefilterRow()} builds a row from a sampler's stages; a forward backend that reports
 * {@link ForwardBackend#batchPrefilterOnDevice()} (or {@link ForwardBackend#topKPrefilterOnDevice()} for the
 * single-row path) applies the rows on the device and reads only the top candidates back, then runs the rest of
 * the chain over them ({@link Sampler#applyTopK(int[], float[], boolean)}). Immutable.</p>
 */
public final class LogitPrefilter {

    /** No row: a call that prefilters nothing. */
    public static final LogitPrefilter EMPTY = new LogitPrefilter(new Row[0]);

    private final Row[] rows;

    private LogitPrefilter(Row[] rows) {
        this.rows = rows;
    }

    /**
     * A prefilter over {@code rows}, one per output row of the call; entries are copied defensively.
     *
     * @throws NullPointerException when {@code rows} or an entry is null
     */
    public static LogitPrefilter of(List<Row> rows) {
        Objects.requireNonNull(rows, "rows");
        Row[] copy = new Row[rows.size()];
        for (int i = 0; i < copy.length; i++) {
            copy[i] = Objects.requireNonNull(rows.get(i), "prefilter row " + i);
        }
        return new LogitPrefilter(copy);
    }

    /** A prefilter over one row. */
    public static LogitPrefilter single(Row row) {
        return of(java.util.Collections.singletonList(row));
    }

    /** Output rows. */
    public int rows() {
        return rows.length;
    }

    /** Row {@code i}. */
    public Row row(int i) {
        if (i < 0 || i >= rows.length) {
            throw new IndexOutOfBoundsException("prefilter row " + i + " outside [0, " + rows.length + ")");
        }
        return rows[i];
    }

    /** Whether every row prefilters nothing, so a device readout needs no prefilter pass. */
    public boolean isEmpty() {
        for (Row row : rows) {
            if (!row.isEmpty()) {
                return false;
            }
        }
        return true;
    }

    /**
     * The rows {@code indices} names, in that order: the rows of the output rows a forward step reads back.
     *
     * @throws IndexOutOfBoundsException when an index is outside this prefilter
     */
    public LogitPrefilter select(int[] indices) {
        Objects.requireNonNull(indices, "indices");
        Row[] picked = new Row[indices.length];
        for (int i = 0; i < indices.length; i++) {
            picked[i] = row(indices[i]);
        }
        return new LogitPrefilter(picked);
    }

    @Override
    public String toString() {
        return "LogitPrefilter{rows=" + rows.length + (isEmpty() ? ", empty" : "") + '}';
    }

    /**
     * One output row's pre-top-k edits. The bias values carry {@link Float#NaN} for a banned token (the logit is
     * set to {@code -inf}) and a finite additive bias otherwise; {@link #penaltyTokens()} lists the row's distinct
     * penalized tokens with their window counts, and {@link #dryTokens()} its DRY window oldest first. Immutable.
     */
    public static final class Row {

        private final int[] biasIds;
        private final float[] biasValues;
        private final boolean penalties;
        private final float repeatPenalty;
        private final float frequencyPenalty;
        private final float presencePenalty;
        private final int[] penaltyTokens;
        private final int[] penaltyCounts;
        private final boolean dry;
        private final float dryMultiplier;
        private final float dryBase;
        private final int dryAllowedLength;
        private final int[] dryTokens;
        /** Breaker entries, one per restart sequence: the head token first, the tail ids after it. */
        private final int[][] dryBreakers;
        private final boolean penaltiesFirst;

        private Row(Builder b) {
            this.biasIds = b.biasIds;
            this.biasValues = b.biasValues;
            this.penalties = b.penalties;
            this.repeatPenalty = b.repeatPenalty;
            this.frequencyPenalty = b.frequencyPenalty;
            this.presencePenalty = b.presencePenalty;
            this.penaltyTokens = b.penaltyTokens;
            this.penaltyCounts = b.penaltyCounts;
            this.dry = b.dry;
            this.dryMultiplier = b.dryMultiplier;
            this.dryBase = b.dryBase;
            this.dryAllowedLength = b.dryAllowedLength;
            this.dryTokens = b.dryTokens;
            this.dryBreakers = b.dryBreakers;
            this.penaltiesFirst = b.penaltiesFirst;
        }

        /** A builder over an empty row. */
        public static Builder builder() {
            return new Builder();
        }

        /** Distinct banned or biased token ids (a copy). */
        public int[] biasIds() {
            return biasIds.clone();
        }

        /** The bias of each id: NaN marks the {@code -inf} ban (a copy). */
        public float[] biasValues() {
            return biasValues.clone();
        }

        /** Whether the repetition, presence and frequency penalties apply before the readout. */
        public boolean penalties() {
            return penalties;
        }

        /** llama.cpp's repeat penalty. */
        public float repeatPenalty() {
            return repeatPenalty;
        }

        /** The frequency penalty per occurrence in the window. */
        public float frequencyPenalty() {
            return frequencyPenalty;
        }

        /** The presence penalty once per window token. */
        public float presencePenalty() {
            return presencePenalty;
        }

        /** Distinct penalized token ids (a copy). */
        public int[] penaltyTokens() {
            return penaltyTokens.clone();
        }

        /** Occurrences of each {@link #penaltyTokens()} id in the window (a copy). */
        public int[] penaltyCounts() {
            return penaltyCounts.clone();
        }

        /** Whether DRY applies before the readout. */
        public boolean dry() {
            return dry;
        }

        /** DRY's repetition strength. */
        public float dryMultiplier() {
            return dryMultiplier;
        }

        /** DRY's penalty base. */
        public float dryBase() {
            return dryBase;
        }

        /** The repeated suffix length DRY starts penalizing at. */
        public int dryAllowedLength() {
            return dryAllowedLength;
        }

        /** The DRY window, oldest first (a copy). */
        public int[] dryTokens() {
            return dryTokens.clone();
        }

        /** The row's restart sequences, each the head token followed by its tail ids (copies). */
        public int[][] dryBreakers() {
            int[][] copy = new int[dryBreakers.length][];
            for (int i = 0; i < dryBreakers.length; i++) {
                copy[i] = dryBreakers[i].clone();
            }
            return copy;
        }

        /** Whether the penalties stage precedes DRY in the chain (true when only one of them applies). */
        public boolean penaltiesFirst() {
            return penaltiesFirst;
        }

        /** Whether this row edits no logit. */
        public boolean isEmpty() {
            return biasIds.length == 0 && !penalties && !dry;
        }

        @Override
        public String toString() {
            return "Row{bias=" + biasIds.length + (penalties ? ", penalties=" + penaltyTokens.length : "")
                    + (dry ? ", dry=" + dryTokens.length + " tokens, " + dryBreakers.length + " breakers" : "")
                    + '}';
        }

        /** Configures a {@link Row}; every array is copied on {@link #build()}. */
        public static final class Builder {

            private int[] biasIds = new int[0];
            private float[] biasValues = new float[0];
            private boolean penalties;
            private float repeatPenalty = 1f;
            private float frequencyPenalty;
            private float presencePenalty;
            private int[] penaltyTokens = new int[0];
            private int[] penaltyCounts = new int[0];
            private boolean dry;
            private float dryMultiplier;
            private float dryBase = 1f;
            private int dryAllowedLength;
            private int[] dryTokens = new int[0];
            private int[][] dryBreakers = new int[0][];
            private boolean penaltiesFirst = true;

            private Builder() {
            }

            /** The logit-bias stage: each id takes its value, {@link Float#NaN} meaning the {@code -inf} ban. */
            public Builder bias(int[] ids, float[] values) {
                Objects.requireNonNull(ids, "bias ids");
                Objects.requireNonNull(values, "bias values");
                if (ids.length != values.length) {
                    throw new IllegalArgumentException(ids.length + " bias ids carry " + values.length + " values");
                }
                this.biasIds = ids.clone();
                this.biasValues = values.clone();
                return this;
            }

            /** The penalties over the row's distinct window tokens and their counts. */
            public Builder penalties(boolean enabled, float repeat, float frequency, float presence,
                                     int[] tokens, int[] counts) {
                Objects.requireNonNull(tokens, "penalty tokens");
                Objects.requireNonNull(counts, "penalty counts");
                if (tokens.length != counts.length) {
                    throw new IllegalArgumentException(tokens.length + " penalty tokens carry " + counts.length
                            + " counts");
                }
                this.penalties = enabled;
                this.repeatPenalty = repeat;
                this.frequencyPenalty = frequency;
                this.presencePenalty = presence;
                this.penaltyTokens = tokens.clone();
                this.penaltyCounts = counts.clone();
                return this;
            }

            /** DRY over the row's window and restart sequences. */
            public Builder dry(boolean enabled, float multiplier, float base, int allowedLength, int[] tokens,
                               int[][] breakers) {
                Objects.requireNonNull(tokens, "dry tokens");
                Objects.requireNonNull(breakers, "dry breakers");
                this.dry = enabled;
                this.dryMultiplier = multiplier;
                this.dryBase = base;
                this.dryAllowedLength = allowedLength;
                this.dryTokens = tokens.clone();
                int[][] copy = new int[breakers.length][];
                for (int i = 0; i < breakers.length; i++) {
                    if (breakers[i] == null || breakers[i].length == 0) {
                        throw new IllegalArgumentException("DRY breaker " + i + " carries no head token");
                    }
                    copy[i] = breakers[i].clone();
                }
                this.dryBreakers = copy;
                return this;
            }

            /** {@link Row#penaltiesFirst()}. */
            public Builder penaltiesFirst(boolean v) {
                this.penaltiesFirst = v;
                return this;
            }

            public Row build() {
                return new Row(this);
            }
        }
    }
}
