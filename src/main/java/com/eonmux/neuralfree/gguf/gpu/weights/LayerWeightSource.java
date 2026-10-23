package com.eonmux.neuralfree.gguf.gpu.weights;

import java.util.Objects;

/**
 * Per-layer access to device weights, so the layer executor can overlap the upload of the next layer's weights with
 * the compute of the current one when weights are streamed.
 *
 * <p>The executor calls {@code prefetch(il + 1)} before running layer {@code il}, {@code await(il)} before the first
 * dispatch that reads layer {@code il}'s weights, and binds blocks to {@link #weightsFor(int)}. {@code il} is the
 * physical weight layer ({@code LayerSpec.weightLayer()}).</p>
 */
public interface LayerWeightSource extends AutoCloseable {

    /** Starts making layer {@code il}'s weights resident; returns without waiting. */
    void prefetch(int il);

    /**
     * Makes the compute queue wait for layer {@code il}'s weights: on return, every dispatch enqueued afterwards on
     * the compute queue observes them (an event wait, not a host wait, where the backend allows it).
     */
    void await(int il);

    /** The weights of layer {@code il} (and the model-level tensors). */
    GpuWeights weightsFor(int il);

    /** Releases streaming resources; a resident source has none. */
    @Override
    default void close() {
    }

    /** Source over weights that are all resident: prefetch and await do nothing. */
    static LayerWeightSource resident(GpuWeights weights) {
        Objects.requireNonNull(weights, "weights");
        return new LayerWeightSource() {
            @Override
            public void prefetch(int il) {
                checkLayer(il);
            }

            @Override
            public void await(int il) {
                checkLayer(il);
            }

            @Override
            public GpuWeights weightsFor(int il) {
                checkLayer(il);
                return weights;
            }

            private void checkLayer(int il) {
                if (il < 0) {
                    throw new IllegalArgumentException("weight layer index " + il + " is negative");
                }
            }

            @Override
            public String toString() {
                return "LayerWeightSource.resident{" + weights + "}";
            }
        };
    }
}
