package com.eonmux.neuralfree.gguf.memory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A copy of the DSV4 compressed streams of one sequence: every compressed cache's compressed rows (as float rows,
 * the memory decodes and re-encodes them) and both compressor tensors with all their planes, plus the sequence's
 * compressor rollback record. It is the compressed half of a context checkpoint: a compressor state accumulates
 * across positions, so a cached prefix of a DeepSeek-V4 memory is restorable only where such a copy was taken
 * beside the KV pages. {@code ModelMemory.restoreCompressedState} writes it into any sequence slot of the same
 * memory. Immutable.
 */
public final class CompressedStateCopy {

    /** The copy of a memory without compressed caches: no streams. */
    public static final CompressedStateCopy EMPTY = new CompressedStateCopy(List.of(), 0, false);

    /** The copied streams of one compressed cache, in the memory's allocation order. */
    public static final class Stream {

        private final float[][] rows;
        private final float[] compressorKv;
        private final float[] compressorScore;

        private Stream(float[][] rows, float[] compressorKv, float[] compressorScore) {
            this.rows = rows;
            this.compressorKv = compressorKv;
            this.compressorScore = compressorScore;
        }

        /**
         * A stream copy holding {@code rows} (one float row per compressed row), and the two compressor tensors
         * with all their planes.
         *
         * @throws NullPointerException for a null array or row
         */
        public static Stream of(float[][] rows, float[] compressorKv, float[] compressorScore) {
            Objects.requireNonNull(rows, "rows");
            float[][] copied = new float[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                copied[i] = Objects.requireNonNull(rows[i], "compressed row " + i).clone();
            }
            return new Stream(copied, Objects.requireNonNull(compressorKv, "compressorKv").clone(),
                    Objects.requireNonNull(compressorScore, "compressorScore").clone());
        }

        /** Compressed rows the copy holds. */
        public int rows() {
            return rows.length;
        }

        /** Floats of compressed row {@code i}. */
        public int floats(int i) {
            return rows[Objects.checkIndex(i, rows.length)].length;
        }

        /** Compressed row {@code i} (a copy). */
        public float[] row(int i) {
            return rows[Objects.checkIndex(i, rows.length)].clone();
        }

        /** The compressor KV tensor with all its planes (a copy). */
        public float[] compressorKv() {
            return compressorKv.clone();
        }

        /** The compressor score tensor with all its planes (a copy). */
        public float[] compressorScore() {
            return compressorScore.clone();
        }
    }

    private final List<Stream> streams;
    private final int lastStepTokens;
    private final boolean pending;
    private final long bytes;

    private CompressedStateCopy(List<Stream> streams, int lastStepTokens, boolean pending) {
        this.streams = List.copyOf(streams);
        this.lastStepTokens = lastStepTokens;
        this.pending = pending;
        long b = 0;
        for (Stream s : streams) {
            for (int i = 0; i < s.rows(); i++) {
                b += (long) s.floats(i) * Float.BYTES;
            }
            b += (long) s.compressorKv.length * Float.BYTES + (long) s.compressorScore.length * Float.BYTES;
        }
        this.bytes = b;
    }

    /**
     * A copy holding {@code streams} (one per compressed cache, in the memory's allocation order) and the
     * rollback record {@code lastStepTokens} / {@code pending}.
     *
     * @throws NullPointerException for a null stream
     */
    public static CompressedStateCopy of(List<Stream> streams, int lastStepTokens, boolean pending) {
        Objects.requireNonNull(streams, "streams");
        for (Stream s : streams) {
            Objects.requireNonNull(s, "stream");
        }
        return new CompressedStateCopy(new ArrayList<>(streams), lastStepTokens, pending);
    }

    /** Compressed caches the copy holds. */
    public int streams() {
        return streams.size();
    }

    /** The copied stream {@code i}, in the memory's allocation order. */
    public Stream stream(int i) {
        return streams.get(Objects.checkIndex(i, streams.size()));
    }

    /** The tokens of the step whose rollback snapshots the copy carries. */
    public int lastStepTokens() {
        return lastStepTokens;
    }

    /** Whether the copied state stood after a rollback (no new step had run since). */
    public boolean pending() {
        return pending;
    }

    /** Bytes the copy holds. */
    public long bytes() {
        return bytes;
    }

    @Override
    public String toString() {
        return "CompressedStateCopy[" + streams.size() + " streams, " + bytes + " bytes]";
    }
}
