package com.eonmux.neuralfree.gguf.engine;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Objects;

/**
 * An immutable forward batch: the tokens or embedding rows of one step, each belonging to exactly one sequence
 * (llama.cpp {@code llama_batch_ext}). A single-sequence batch that continues the held positions runs on any
 * backend; a batch over several sequences needs a backend whose memory holds more than one sequence and is refused
 * otherwise ({@code ForwardBackend.batchRefusal()}).
 *
 * <p>Built by {@link Builder#build()}, which validates the llama.cpp batch rules: at least one row, non-negative
 * positions and sequence ids, per-sequence positions consecutive from one row of the batch to the next, no mixing
 * of token and embedding rows in one batch and one embedding width per batch. Accessors return copies.</p>
 */
public final class ForwardBatch {

    private final int[] token;    // token id per row; -1 marks an embedding row
    private final int[] pos;
    private final int[] seq;
    private final boolean[] output;
    private final int[] outputRows;   // batch indices of the output rows, in batch order
    private final float[][] embd;     // row copies for embedding rows (null for token rows); null for token batches
    private final int[][] ropeSection; // section-major M-RoPE positions ropeSection[k][row]; null when unset
    private final int[] adapterSlot;   // per-row adapter slot; null = every row runs in slot 0
    private final int[] kindBits;      // OutputKind bitmask per row

    private ForwardBatch(int[] token, int[] pos, int[] seq, boolean[] output, int[] outputRows,
            float[][] embd, int[][] ropeSection, int[] adapterSlot, int[] kindBits) {
        this.token = token;
        this.pos = pos;
        this.seq = seq;
        this.output = output;
        this.outputRows = outputRows;
        this.embd = embd;
        this.ropeSection = ropeSection;
        this.adapterSlot = adapterSlot;
        this.kindBits = kindBits;
    }

    /**
     * A fresh builder holding up to {@code capacity} rows.
     *
     * @throws IllegalArgumentException when {@code capacity} is not positive
     */
    public static Builder builder(int capacity) {
        return new Builder(capacity);
    }

    /** Rows in the batch. */
    public int nTokens() {
        return token.length;
    }

    /**
     * Token id of every row; a batch carries all token rows or all embedding rows ({@link Builder#build()}
     * rejects the mix), so {@code -1} appears on an embedding batch only.
     */
    public int[] tokens() {
        return token.clone();
    }

    /** Position of every row. */
    public int[] positions() {
        return pos.clone();
    }

    /** Sequence id of every row. */
    public int[] seqIds() {
        return seq.clone();
    }

    /** True when the row produces its requested output kinds ({@link #kinds(int)}). */
    public boolean output(int row) {
        return output[row];
    }

    /** Indices of the output rows, in batch order. */
    public int[] outputRows() {
        return outputRows.clone();
    }

    /**
     * Embedding rows, indexed as {@code embeddings()[row]} (each row a copy), or null when the batch is a token
     * batch.
     */
    public float[][] embeddings() {
        if (embd == null) {
            return null;
        }
        float[][] out = new float[embd.length][];
        for (int i = 0; i < embd.length; i++) {
            out[i] = embd[i] == null ? null : embd[i].clone();
        }
        return out;
    }

    /**
     * Embedding row at {@code row} (a copy), or null for a token row.
     *
     * @throws IndexOutOfBoundsException when {@code row} is outside the batch
     */
    public float[] embeddingRow(int row) {
        if (row < 0 || row >= nTokens()) {
            throw new IndexOutOfBoundsException("row " + row + " outside the batch of " + nTokens() + " rows");
        }
        if (embd == null || embd[row] == null) {
            return null;
        }
        return embd[row].clone();
    }

    /**
     * Section-major M-RoPE positions of every row ({@code [section][row]}, the layout of
     * {@code GpuStepInputs.ropePos()}), or null when the batch does not set them: then a single-section model uses
     * {@link #positions()} and a multi-section model refuses the batch.
     */
    public int[][] ropePositions() {
        if (ropeSection == null) {
            return null;
        }
        int[][] out = new int[ropeSection.length][];
        for (int k = 0; k < ropeSection.length; k++) {
            out[k] = ropeSection[k].clone();
        }
        return out;
    }

    /** Adapter slot of every row, or null when the batch does not set them (every row runs in slot 0, LR-2 fills). */
    public int[] adapterSlots() {
        return adapterSlot == null ? null : adapterSlot.clone();
    }

    /** The output kinds row {@code row} produces (a fresh set; empty when the row is not an output row). */
    public EnumSet<OutputKind> kinds(int row) {
        return OutputKind.kinds(kindBits[row]);
    }

    /** True when row {@code row} produces kind {@code kind}. */
    public boolean produces(int row, OutputKind kind) {
        return (kindBits[row] & kind.bit()) != 0;
    }

    /** True when every row belongs to one sequence. */
    public boolean isSingleSequence() {
        for (int i = 1; i < seq.length; i++) {
            if (seq[i] != seq[0]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Accumulates rows for one batched forward step. Not thread-safe; one caller drives a builder.
     *
     * @throws IllegalStateException when an optional per-row array is set twice or the batch overflows
     */
    public static final class Builder {

        private final int[] token;
        private final int[] pos;
        private final int[] seq;
        private final boolean[] output;
        private int n;
        private float[][] embd;     // entry set per addEmbedding
        private int embdWidth = -1; // width of the embedding rows added so far, -1 = none yet
        private int nEmbedding;
        private int[][] ropeSection;
        private int[] adapterSlot;
        private EnumSet<OutputKind> kinds; // kinds of the output rows; null = LOGITS
        private boolean ropeSet;
        private boolean adapterSet;
        private boolean kindsSet;

        private Builder(int capacity) {
            if (capacity <= 0) {
                throw new IllegalArgumentException("batch capacity must be positive, got " + capacity);
            }
            token = new int[capacity];
            pos = new int[capacity];
            seq = new int[capacity];
            output = new boolean[capacity];
        }

        /**
         * Adds one token row.
         *
         * @throws IllegalStateException when the builder holds its capacity already
         */
        public Builder add(int token, int pos, int seq, boolean output) {
            reserve();
            this.token[n] = token;
            this.pos[n] = pos;
            this.seq[n] = seq;
            this.output[n] = output;
            n++;
            return this;
        }

        /**
         * Adds one embedding row (a copy of it).
         *
         * @throws IllegalArgumentException when the row has another width than the rows added before
         * @throws IllegalStateException    when the builder holds its capacity already
         */
        public Builder addEmbedding(float[] row, int pos, int seq, boolean output) {
            Objects.requireNonNull(row, "row");
            reserve();
            if (embdWidth >= 0 && row.length != embdWidth) {
                throw new IllegalArgumentException("embedding rows share one width: row of " + row.length
                        + " floats after rows of " + embdWidth);
            }
            if (embd == null) {
                embd = new float[token.length][];
            }
            embd[n] = row.clone();
            if (embdWidth < 0) {
                embdWidth = row.length;
            }
            token[n] = -1;
            this.pos[n] = pos;
            this.seq[n] = seq;
            this.output[n] = output;
            nEmbedding++;
            n++;
            return this;
        }

        /**
         * M-RoPE positions of every row added so far: {@code sections[k]} holds section {@code k} of each row,
         * laid out as {@code GpuStepInputs.ropePos()} (pos of the row plus per-section positions). Optional.
         *
         * @throws IllegalArgumentException when no section or a section covers another row count than the batch
         * @throws IllegalStateException    when set twice
         */
        public Builder ropePositions(int[][] sections) {
            if (ropeSet) {
                throw new IllegalStateException("rope positions are set once");
            }
            Objects.requireNonNull(sections, "sections");
            if (sections.length == 0) {
                throw new IllegalArgumentException("rope positions need at least one section");
            }
            for (int k = 0; k < sections.length; k++) {
                Objects.requireNonNull(sections[k], "section " + k);
                if (sections[k].length != n) {
                    throw new IllegalArgumentException("rope section " + k + " covers " + sections[k].length
                            + " rows, the batch holds " + n);
                }
            }
            // a rejected call changes nothing: the set-once flag goes last, after the validation
            ropeSet = true;
            ropeSection = new int[sections.length][];
            for (int k = 0; k < sections.length; k++) {
                ropeSection[k] = sections[k].clone();
            }
            return this;
        }

        /**
         * Adapter slot of every row added so far. Optional; rows without set slots run in slot 0.
         *
         * @throws IllegalArgumentException when a slot is negative or the array covers another row count
         * @throws IllegalStateException    when set twice
         */
        public Builder adapterSlots(int[] slots) {
            if (adapterSet) {
                throw new IllegalStateException("adapter slots are set once");
            }
            Objects.requireNonNull(slots, "slots");
            if (slots.length != n) {
                throw new IllegalArgumentException("adapter slots cover " + slots.length + " rows, the batch holds "
                        + n);
            }
            for (int s : slots) {
                if (s < 0) {
                    throw new IllegalArgumentException("adapter slots are non-negative, got " + s);
                }
            }
            // a rejected call changes nothing: the set-once flag goes last, after the validation
            adapterSet = true;
            adapterSlot = slots.clone();
            return this;
        }

        /**
         * Output kinds the output rows produce; the default is {@link OutputKind#LOGITS} on every output row.
         *
         * @throws IllegalArgumentException for an empty set
         * @throws IllegalStateException    when set twice
         */
        public Builder outputKinds(EnumSet<OutputKind> kinds) {
            if (kindsSet) {
                throw new IllegalStateException("output kinds are set once");
            }
            kindsSet = true;
            Objects.requireNonNull(kinds, "kinds");
            if (kinds.isEmpty()) {
                throw new IllegalArgumentException("output kinds must not be empty");
            }
            this.kinds = EnumSet.copyOf(kinds);
            return this;
        }

        /**
         * Builds the batch, validating the llama.cpp batch rules.
         *
         * @throws IllegalArgumentException on every rule violation
         * @throws IllegalStateException    when no row was added
         */
        public ForwardBatch build() {
            if (n <= 0) {
                throw new IllegalStateException("a batch holds at least one row");
            }
            if (nEmbedding > 0 && nEmbedding != n) {
                throw new IllegalArgumentException("token and embedding rows are not mixed in one batch: "
                        + nEmbedding + " embedding rows of " + n);
            }
            int[] tokens = Arrays.copyOf(token, n);
            if (nEmbedding == 0) {
                for (int i = 0; i < n; i++) {
                    if (tokens[i] < 0) {
                        throw new IllegalArgumentException("row " + i + " carries no token id");
                    }
                }
            }
            int lastSeq = -1;
            int lastPos = -1;
            for (int i = 0; i < n; i++) {
                if (seq[i] < 0) {
                    throw new IllegalArgumentException("row " + i + " has sequence id " + seq[i]);
                }
                if (pos[i] < 0) {
                    throw new IllegalArgumentException("row " + i + " has position " + pos[i]);
                }
                if (seq[i] != lastSeq) {
                    lastSeq = seq[i];
                    lastPos = pos[i] - 1;
                }
                if (pos[i] != lastPos + 1) {
                    throw new IllegalArgumentException("sequence " + seq[i] + " positions are not consecutive from "
                            + "one row of the batch to the next: row " + i + " holds " + pos[i] + " after "
                            + lastPos);
                }
                lastPos = pos[i];
            }

            int[] outputRows = new int[0];
            int nOut = 0;
            for (int i = 0; i < n; i++) {
                nOut += output[i] ? 1 : 0;
            }
            outputRows = new int[nOut];
            int o = 0;
            for (int i = 0; i < n; i++) {
                if (output[i]) {
                    outputRows[o++] = i;
                }
            }

            int[] bits = new int[n];
            int kindsMask = kinds == null ? OutputKind.LOGITS.bit() : OutputKind.bits(kinds);
            for (int i = 0; i < n; i++) {
                bits[i] = output[i] ? kindsMask : 0;
            }

            float[][] embdCopy;
            if (embd == null) {
                embdCopy = null;
            } else {
                embdCopy = new float[n][];
                for (int i = 0; i < n; i++) {
                    embdCopy[i] = embd[i] == null ? null : embd[i].clone();
                }
            }

            return new ForwardBatch(tokens, Arrays.copyOf(pos, n), Arrays.copyOf(seq, n),
                    Arrays.copyOf(output, n), outputRows, embdCopy, ropeSection, adapterSlot, bits);
        }

        private void reserve() {
            if (n >= token.length) {
                throw new IllegalStateException("batch is full: builder capacity " + token.length);
            }
        }
    }
}
