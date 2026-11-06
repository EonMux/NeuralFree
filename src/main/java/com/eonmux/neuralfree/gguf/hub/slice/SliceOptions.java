package com.eonmux.neuralfree.gguf.hub.slice;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The choices a model slice is built from: how many layers and experts to keep (or which layers exactly), how far
 * to shave vocab-sized tensors, the size budget, the output directory and the request concurrency. Immutable;
 * built by {@link Builder}, which validates every value so the planner and the writer never see an impossible one.
 *
 * <p>The defaults mirror the reference skill: the first 2 layers, the first 8 experts, no vocab slicing, a 1 GiB
 * budget and 8 concurrent range requests.</p>
 */
public final class SliceOptions {

    /** Default number of layers kept from the front of the model. */
    public static final int DEFAULT_LAYERS = 2;
    /** Default number of experts kept per expert layer. */
    public static final int DEFAULT_EXPERTS = 8;
    /** Default size budget of a slice. */
    public static final long DEFAULT_MAX_BYTES = 1L << 30;
    /** Default number of concurrent range requests. */
    public static final int DEFAULT_JOBS = 8;

    private final int numLayers;
    private final List<Integer> keepLayers;
    private final int numExperts;
    private final int vocabSize;
    private final long maxBytes;
    private final int jobs;
    private final Path output;
    private final boolean overwrite;
    private final boolean inspect;
    private final boolean dryRun;

    private SliceOptions(Builder b) {
        this.numLayers = b.numLayers;
        this.keepLayers = b.keepLayers == null ? null : Collections.unmodifiableList(new ArrayList<>(b.keepLayers));
        this.numExperts = b.numExperts;
        this.vocabSize = b.vocabSize;
        this.maxBytes = b.maxBytes;
        this.jobs = b.jobs;
        this.output = b.output;
        this.overwrite = b.overwrite;
        this.inspect = b.inspect;
        this.dryRun = b.dryRun;
    }

    /** A builder with the reference defaults. */
    public static Builder builder() {
        return new Builder();
    }

    /** How many layers the front of the model contributes, when no explicit layer list is set. */
    public int numLayers() {
        return numLayers;
    }

    /** The explicit layer indices to keep, sorted and deduplicated; null keeps the first {@link #numLayers()}. */
    public List<Integer> keepLayers() {
        return keepLayers;
    }

    /** How many experts each expert layer keeps. */
    public int numExperts() {
        return numExperts;
    }

    /** The row count vocab-sized tensors are cut to; 0 leaves them whole. */
    public int vocabSize() {
        return vocabSize;
    }

    /** The size budget of the slice's weights in bytes. */
    public long maxBytes() {
        return maxBytes;
    }

    /** How many range requests run at once. */
    public int jobs() {
        return jobs;
    }

    /** The output directory; null only for a run that writes nothing ({@link #inspect()}, {@link #dryRun()}). */
    public Path output() {
        return output;
    }

    /** Whether a non-empty output directory may be replaced. */
    public boolean overwrite() {
        return overwrite;
    }

    /** Whether the run prints its analysis and plan and writes nothing. */
    public boolean inspect() {
        return inspect;
    }

    /** Whether the run plans and reports its size and writes nothing. */
    public boolean dryRun() {
        return dryRun;
    }

    /** Whether the run writes nothing at all. */
    public boolean writesNothing() {
        return inspect || dryRun;
    }

    /** Collects {@link SliceOptions}. Not thread-safe. */
    public static final class Builder {

        private int numLayers = DEFAULT_LAYERS;
        private List<Integer> keepLayers;
        private int numExperts = DEFAULT_EXPERTS;
        private int vocabSize;
        private long maxBytes = DEFAULT_MAX_BYTES;
        private int jobs = DEFAULT_JOBS;
        private Path output;
        private boolean overwrite;
        private boolean inspect;
        private boolean dryRun;

        private Builder() {
        }

        /** The number of leading layers to keep; 0 keeps none of the regular layers. */
        public Builder numLayers(int value) {
            this.numLayers = value;
            return this;
        }

        /** The exact layer indices to keep, replacing {@link #numLayers}; they are sorted and deduplicated. */
        public Builder keepLayers(List<Integer> values) {
            this.keepLayers = values == null ? null : new ArrayList<>(values);
            return this;
        }

        /** The number of experts each expert layer keeps. */
        public Builder numExperts(int value) {
            this.numExperts = value;
            return this;
        }

        /** The row count vocab-sized tensors are cut to; 0 leaves them whole. */
        public Builder vocabSize(int value) {
            this.vocabSize = value;
            return this;
        }

        /** The size budget of the slice's weights in bytes. */
        public Builder maxBytes(long value) {
            this.maxBytes = value;
            return this;
        }

        /** The number of concurrent range requests. */
        public Builder jobs(int value) {
            this.jobs = value;
            return this;
        }

        /** The output directory. */
        public Builder output(Path value) {
            this.output = value;
            return this;
        }

        /** Whether a non-empty output directory may be replaced. */
        public Builder overwrite(boolean value) {
            this.overwrite = value;
            return this;
        }

        /** Whether the run prints its analysis and plan and writes nothing. */
        public Builder inspect(boolean value) {
            this.inspect = value;
            return this;
        }

        /** Whether the run plans and reports its size and writes nothing. */
        public Builder dryRun(boolean value) {
            this.dryRun = value;
            return this;
        }

        /**
         * The options.
         *
         * @throws IllegalArgumentException when a value is out of its range, naming it
         */
        public SliceOptions build() {
            if (numLayers < 0) {
                throw new IllegalArgumentException("num-layers must be >= 0, got " + numLayers);
            }
            if (numExperts < 1) {
                throw new IllegalArgumentException("num-experts must be >= 1, got " + numExperts);
            }
            if (vocabSize < 0) {
                throw new IllegalArgumentException("vocab-size must be >= 0, got " + vocabSize);
            }
            if (maxBytes <= 0) {
                throw new IllegalArgumentException("max-size must be > 0, got " + maxBytes);
            }
            if (jobs < 1) {
                throw new IllegalArgumentException("jobs must be >= 1, got " + jobs);
            }
            if (keepLayers != null) {
                if (keepLayers.isEmpty()) {
                    throw new IllegalArgumentException("keep-layers needs at least one layer index");
                }
                List<Integer> sorted = new ArrayList<>(keepLayers);
                Collections.sort(sorted);
                for (int i = 0; i < sorted.size(); i++) {
                    int layer = sorted.get(i);
                    if (layer < 0) {
                        throw new IllegalArgumentException("keep-layers has the negative index " + layer);
                    }
                    if (i > 0 && sorted.get(i - 1).intValue() == layer) {
                        sorted.remove(i--);
                    }
                }
                keepLayers = sorted;
            }
            if (!inspect && !dryRun && output == null) {
                throw new IllegalArgumentException("an output directory is needed unless the run writes nothing");
            }
            return new SliceOptions(this);
        }
    }

    @Override
    public String toString() {
        return "SliceOptions{layers=" + (keepLayers != null ? keepLayers : "first " + numLayers)
                + ", experts=" + numExperts + ", vocab=" + vocabSize + ", maxBytes=" + maxBytes
                + ", jobs=" + jobs + ", output=" + output + "}";
    }
}
