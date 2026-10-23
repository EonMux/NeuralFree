package com.eonmux.neuralfree.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class SequentialSampler implements Sampler {
    public SequentialSampler() {
    }

    @Override
    public List<Integer> sample(int datasetSize) {
        if (datasetSize < 0) {
            throw new IllegalArgumentException("Dataset size must be non-negative, got: " + datasetSize);
        }
        List<Integer> indices = new ArrayList<>(datasetSize);
        for (int i = 0; i < datasetSize; i++) {
            indices.add(i);
        }
        return indices;
    }

    /** Every index once: {@code datasetSize}. */
    @Override
    public int length(int datasetSize) {
        if (datasetSize < 0) {
            throw new IllegalArgumentException("Dataset size must be non-negative, got: " + datasetSize);
        }
        return datasetSize;
    }

    @Override
    public void reset() {
    }

    @Override
    public String toString() {
        return "SequentialSampler{}";
    }
}
