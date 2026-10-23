package com.eonmux.neuralfree.data;

import java.util.List;

/** The order in which a {@link DataLoader} visits the indices of a dataset. */
public interface Sampler {
    /** The indices of one pass over a dataset of {@code datasetSize} items; its size is {@link #length(int)}. */
    List<Integer> sample(int datasetSize);

    /**
     * The number of indices {@link #sample(int)} returns for a dataset of {@code datasetSize} items, without drawing
     * them ({@code len(sampler)} in PyTorch).
     */
    int length(int datasetSize);

    /**
     * Restarts a seeded sampler's sequence, so the following passes repeat the orders drawn since it was seeded; the
     * {@link DataLoader} never calls it, so each pass otherwise takes the sampler's next order. A no-op for an unseeded
     * or deterministic sampler.
     */
    void reset();
}
