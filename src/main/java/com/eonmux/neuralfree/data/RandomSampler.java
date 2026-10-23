package com.eonmux.neuralfree.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public final class RandomSampler implements Sampler {
    private final Random random;
    private final Long seed;

    public RandomSampler() {
        this.random = new Random();
        this.seed = null;
    }

    public RandomSampler(long seed) {
        this.random = new Random(seed);
        this.seed = seed;
    }

    public RandomSampler(Random random) {
        this.random = Objects.requireNonNull(random, "Random cannot be null");
        this.seed = null;
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
        Collections.shuffle(indices, random);
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
        if (seed != null) {
            random.setSeed(seed);
        }
    }

    public Long seed() {
        return seed;
    }

    @Override
    public String toString() {
        return "RandomSampler{seed=" + seed + "}";
    }
}
