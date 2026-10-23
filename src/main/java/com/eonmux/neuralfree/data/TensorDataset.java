package com.eonmux.neuralfree.data;

import com.eonmux.neuralfree.tensor.Tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * A dataset of (input, target) tensor pairs.
 *
 * <p><b>Ownership.</b> The array constructor keeps the caller's tensors, which stay the caller's to close. The batch
 * constructor creates one view per sample over the batch tensors (each holding a reference to the batch storage);
 * those views are the dataset's own and {@link #close()} releases them. {@link #close()} never closes a caller's
 * tensor.</p>
 */
public final class TensorDataset implements Dataset<TensorDataset.Sample>, AutoCloseable {
    private final Tensor[] inputs;
    private final Tensor[] targets;
    private final int size;
    private final boolean ownsSamples;

    /** A dataset over the caller's tensors, which it does not close. */
    public TensorDataset(Tensor[] inputs, Tensor[] targets) {
        Objects.requireNonNull(inputs, "Inputs cannot be null");
        Objects.requireNonNull(targets, "Targets cannot be null");
        
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException(
                "Inputs and targets must have the same length: " + 
                inputs.length + " vs " + targets.length);
        }
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Dataset cannot be empty");
        }

        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] == null) {
                throw new IllegalArgumentException("Input at index " + i + " is null");
            }
        }
        for (int i = 0; i < targets.length; i++) {
            if (targets[i] == null) {
                throw new IllegalArgumentException("Target at index " + i + " is null");
            }
        }

        this.inputs = inputs.clone();
        this.targets = targets.clone();
        this.size = inputs.length;
        this.ownsSamples = false;
    }

    /**
     * A dataset whose sample {@code i} is row {@code i} of {@code inputBatch} and {@code targetBatch} (views without
     * the leading dim, owned by the dataset); the batch tensors stay the caller's.
     */
    public TensorDataset(Tensor inputBatch, Tensor targetBatch) {
        Objects.requireNonNull(inputBatch, "Input batch cannot be null");
        Objects.requireNonNull(targetBatch, "Target batch cannot be null");

        if (inputBatch.shape().rank() == 0) {
            throw new IllegalArgumentException("Input batch must have at least 1 dimension");
        }
        if (targetBatch.shape().rank() == 0) {
            throw new IllegalArgumentException("Target batch must have at least 1 dimension");
        }

        long inputBatchSize = inputBatch.shape().dim(0);
        long targetBatchSize = targetBatch.shape().dim(0);

        if (inputBatchSize != targetBatchSize) {
            throw new IllegalArgumentException(
                "Input and target batch sizes must match: " + 
                inputBatchSize + " vs " + targetBatchSize);
        }

        this.size = (int) inputBatchSize;
        this.inputs = new Tensor[size];
        this.targets = new Tensor[size];
        this.ownsSamples = true;

        try {
            for (int i = 0; i < size; i++) {
                inputs[i] = row(inputBatch, i);
                targets[i] = row(targetBatch, i);
            }
        } catch (RuntimeException e) {
            try {
                close();
            } catch (RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /** Row {@code i} of {@code batch} without the leading dim, as a view; the intermediate slice is released. */
    private static Tensor row(Tensor batch, int i) {
        try (Tensor slice = batch.slice(0, i, i + 1)) {
            return slice.squeeze(0);
        }
    }

    /** Releases the per-sample views the batch constructor created; does nothing for the array constructor. */
    @Override
    public void close() {
        if (!ownsSamples) {
            return;
        }
        RuntimeException failure = null;
        for (Tensor[] tensors : new Tensor[][]{inputs, targets}) {
            for (Tensor t : tensors) {
                if (t == null) {
                    continue;
                }
                try {
                    t.close();
                } catch (RuntimeException e) {
                    if (failure == null) {
                        failure = e;
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public Sample get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " out of bounds for dataset size " + size);
        }
        return new Sample(inputs[index], targets[index]);
    }

    public Tensor input(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " out of bounds for dataset size " + size);
        }
        return inputs[index];
    }

    public Tensor target(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " out of bounds for dataset size " + size);
        }
        return targets[index];
    }

    public Tensor[] inputs() {
        return inputs.clone();
    }

    public Tensor[] targets() {
        return targets.clone();
    }

    @Override
    public String toString() {
        return "TensorDataset{size=" + size + "}";
    }

    /**
     * One (input, target) pair. A sample built with the public constructor only refers to its tensors. A sample a
     * {@link DataLoader} creates for a batch (a copy on the loader's backend) owns the copies it made, and
     * {@link #close()} releases them; the consumer of a batch closes its samples once it no longer needs them.
     */
    public static final class Sample implements AutoCloseable {
        private final Tensor input;
        private final Tensor target;
        private final boolean ownsInput;
        private final boolean ownsTarget;

        /** A sample that refers to {@code input} and {@code target} without owning them. */
        public Sample(Tensor input, Tensor target) {
            this(input, target, false, false);
        }

        Sample(Tensor input, Tensor target, boolean ownsInput, boolean ownsTarget) {
            this.input = Objects.requireNonNull(input, "Input cannot be null");
            this.target = Objects.requireNonNull(target, "Target cannot be null");
            this.ownsInput = ownsInput;
            this.ownsTarget = ownsTarget;
        }

        /** Closes the tensors this sample owns; idempotent, and nothing for a sample that owns none. */
        @Override
        public void close() {
            try {
                if (ownsInput) {
                    input.close();
                }
            } finally {
                if (ownsTarget) {
                    target.close();
                }
            }
        }

        public Tensor input() {
            return input;
        }

        public Tensor target() {
            return target;
        }

        @Override
        public String toString() {
            return "Sample{input=" + input.shape() + ", target=" + target.shape() + "}";
        }
    }
}
