package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Arrays;
import java.util.Objects;

/**
 * The sequences of a memoryless {@link ForwardBatch} as contiguous row runs (llama.cpp's per-sequence KQ mask over
 * a cache-less ubatch): each run is one sequence's rows, and a row attends only the rows of its own run. A packed
 * batch of several sequences lets an encoder run many short inputs as one step, so the projections and FFNs read
 * their weights once for all of them while every attention stays inside its sequence.
 *
 * <p>A batch of one sequence is one run over every row at the batch's own positions. A batch of several sequences
 * must keep each sequence's rows together and number them from position 0 (each run is an independent encoder
 * input), which is what the device attention's per-run launches read: run-local query positions against the run's
 * own key rows. Immutable.</p>
 */
public final class BatchSegments {

    /**
     * Sequences one packed memoryless batch holds. A device runs one attention launch (and its split merge) per
     * sequence and layer, so the bound keeps those launches below the per-sequence step overhead the packing saves.
     */
    public static final int MAX_PACKED = 64;

    private final int[] starts;
    private final int[] lengths;

    private BatchSegments(int[] starts, int[] lengths) {
        this.starts = starts;
        this.lengths = lengths;
    }

    /**
     * The runs of {@code batch}'s sequences.
     *
     * @throws IllegalArgumentException when a sequence's rows are split by another sequence's, or a batch of several
     *                                  sequences holds a run whose positions are not {@code 0, 1, ..} in row order
     */
    public static BatchSegments of(ForwardBatch batch) {
        Objects.requireNonNull(batch, "batch");
        int[] seq = batch.seqIds();
        int n = seq.length;
        int[] starts = new int[n];
        int count = 0;
        for (int t = 0; t < n; t++) {
            if (t == 0 || seq[t] != seq[t - 1]) {
                starts[count++] = t;
            }
        }
        int[] runStarts = Arrays.copyOf(starts, count);
        int[] runLengths = new int[count];
        for (int i = 0; i < count; i++) {
            runLengths[i] = (i + 1 < count ? runStarts[i + 1] : n) - runStarts[i];
        }
        if (count > 1) {
            checkRuns(seq, batch.positions(), runStarts, runLengths);
        }
        return new BatchSegments(runStarts, runLengths);
    }

    /**
     * The runs of a memoryless {@code batch} over {@code spec}'s layers, or null when the batch holds one sequence:
     * several sequences pack only where {@link #packingRefusal} allows and up to {@link #MAX_PACKED} of them.
     *
     * @throws UnsupportedOperationException when several sequences meet layers that cannot keep them apart
     * @throws IllegalArgumentException      when the batch's runs break the {@link #of} rules or exceed
     *                                       {@link #MAX_PACKED}
     */
    public static BatchSegments packed(ForwardBatch batch, ModelSpec spec) {
        BatchSegments segments = of(batch);
        if (segments.count() == 1) {
            return null;
        }
        String refusal = packingRefusal(spec);
        if (refusal != null) {
            throw new UnsupportedOperationException("a memoryless batch of " + segments.count() + " sequences: "
                    + refusal);
        }
        if (segments.count() > MAX_PACKED) {
            throw new IllegalArgumentException("a memoryless batch packs at most " + MAX_PACKED + " sequences, got "
                    + segments.count());
        }
        return segments;
    }

    private static void checkRuns(int[] seq, int[] pos, int[] starts, int[] lengths) {
        int[] seen = seq.clone();
        Arrays.sort(seen);
        int distinct = 0;
        for (int i = 0; i < seen.length; i++) {
            if (i == 0 || seen[i] != seen[i - 1]) {
                distinct++;
            }
        }
        if (distinct != starts.length) {
            throw new IllegalArgumentException("a memoryless batch keeps each sequence's rows together; "
                    + distinct + " sequences run in " + starts.length + " row runs");
        }
        for (int i = 0; i < starts.length; i++) {
            for (int t = 0; t < lengths[i]; t++) {
                if (pos[starts[i] + t] != t) {
                    throw new IllegalArgumentException("a memoryless batch of several sequences runs each from"
                            + " position 0; row " + (starts[i] + t) + " of sequence " + seq[starts[i]]
                            + " sits at position " + pos[starts[i] + t] + ", not " + t);
                }
            }
        }
    }

    /**
     * Why {@code spec}'s layers cannot run several sequences in one memoryless batch, or null when they can: every
     * layer must mix its rows through plain softmax attention alone (or no mixer) and run a row-wise FFN, so the
     * attention's per-run launches are the only place rows meet. Recurrent and convolutional mixers, dual-chunk,
     * loop and static-sink-token attention, cross-attention and token-shift FFNs carry state along the rows and
     * keep one sequence per batch.
     */
    public static String packingRefusal(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        String arch = spec.arch().ggufName();
        if (spec.memoryKind() != MemoryKind.NONE) {
            return arch + " keeps a memory; its sequences pack on the paged memory instead";
        }
        for (LayerSpec l : spec.layers()) {
            if (l.mixerKind() != MixerKind.ATTENTION && l.mixerKind() != MixerKind.NONE) {
                return arch + " layer " + l.il() + " mixes rows with " + l.mixerKind()
                        + ", which runs along the whole batch";
            }
            if (l.ffnKind() != FfnKind.DENSE && l.ffnKind() != FfnKind.MOE) {
                return arch + " layer " + l.il() + " runs a " + l.ffnKind() + " FFN";
            }
            if (l.crossAttention() != null || l.isRecurrent()) {
                return arch + " layer " + l.il() + " holds cross-attention or recurrent state";
            }
            AttentionSpec a = l.attention();
            if (l.mixerKind() == MixerKind.ATTENTION
                    && (a == null || a.dual() || a.hasDualChunk() || a.sinkTokens() > 0)) {
                return arch + " layer " + l.il() + " runs dual, dual-chunk or sink-token attention";
            }
        }
        return null;
    }

    /** The number of runs (sequences). */
    public int count() {
        return starts.length;
    }

    /** The first row of run {@code i}. */
    public int start(int i) {
        return starts[i];
    }

    /** The rows of run {@code i}. */
    public int length(int i) {
        return lengths[i];
    }

    /** The longest run's rows. */
    public int maxLength() {
        int max = 0;
        for (int len : lengths) {
            max = Math.max(max, len);
        }
        return max;
    }

    @Override
    public String toString() {
        return "BatchSegments{" + starts.length + " runs, starts " + Arrays.toString(starts) + '}';
    }
}
