package com.eonmux.neuralfree.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Draws {@code numSamples} indices with probability proportional to their weights, following PyTorch's
 * {@code WeightedRandomSampler} ({@code torch.multinomial}). With replacement, every draw is independent. Without
 * replacement, each index appears at most once, zero-weight indices are never drawn, and {@code numSamples} may not
 * exceed the number of positive weights.
 */
public final class WeightedRandomSampler implements Sampler {
    private final double[] weights;
    private final double[] cumulativeWeights;
    private final int lastPositiveIndex;
    private final Random random;
    private final Long seed;
    private final int numSamples;
    private final boolean replacement;

    public WeightedRandomSampler(double[] weights, int numSamples, boolean replacement) {
        this(weights, numSamples, replacement, new Random());
    }

    public WeightedRandomSampler(double[] weights, int numSamples, boolean replacement, long seed) {
        this(weights, numSamples, replacement, new Random(seed), seed);
    }

    public WeightedRandomSampler(double[] weights, int numSamples, boolean replacement, Random random) {
        this(weights, numSamples, replacement, random, null);
    }

    private WeightedRandomSampler(double[] weights, int numSamples, boolean replacement, Random random, Long seed) {
        Objects.requireNonNull(weights, "Weights cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        if (weights.length == 0) {
            throw new IllegalArgumentException("Weights array cannot be empty");
        }
        if (numSamples <= 0) {
            throw new IllegalArgumentException("numSamples must be positive, got: " + numSamples);
        }

        double[] cumulative = new double[weights.length];
        double sum = 0.0;
        int positiveCount = 0;
        int lastPositive = -1;
        for (int i = 0; i < weights.length; i++) {
            double w = weights[i];
            if (!Double.isFinite(w) || w < 0) {
                throw new IllegalArgumentException("Weights must be finite and non-negative, got: " + w);
            }
            if (w > 0) {
                positiveCount++;
                lastPositive = i;
            }
            sum += w;
            cumulative[i] = sum;
        }
        if (!(sum > 0) || !Double.isFinite(sum)) {
            throw new IllegalArgumentException("Sum of weights must be positive and finite, got: " + sum);
        }
        if (!replacement && numSamples > positiveCount) {
            throw new IllegalArgumentException("Cannot draw " + numSamples
                + " samples without replacement from " + positiveCount + " positive weights");
        }

        this.weights = weights.clone();
        this.cumulativeWeights = cumulative;
        this.lastPositiveIndex = lastPositive;
        this.numSamples = numSamples;
        this.replacement = replacement;
        this.random = random;
        this.seed = seed;
    }

    @Override
    public List<Integer> sample(int datasetSize) {
        if (datasetSize < 0) {
            throw new IllegalArgumentException("Dataset size must be non-negative, got: " + datasetSize);
        }
        if (datasetSize != weights.length) {
            throw new IllegalArgumentException("Dataset size (" + datasetSize +
                ") must match weights length (" + weights.length + ")");
        }
        return replacement ? sampleWithReplacement() : sampleWithoutReplacement();
    }

    /** Independent inverse-CDF draws. */
    private List<Integer> sampleWithReplacement() {
        double totalWeight = cumulativeWeights[cumulativeWeights.length - 1];
        List<Integer> indices = new ArrayList<>(numSamples);
        for (int i = 0; i < numSamples; i++) {
            indices.add(indexOf(random.nextDouble() * totalWeight));
        }
        return indices;
    }

    /**
     * The first index whose cumulative weight is strictly greater than {@code value}. A zero-weight index has the same
     * cumulative weight as its predecessor (or 0), so it is never chosen. Rounding can make {@code value} reach the
     * total; that draw falls on the last positive weight.
     */
    private int indexOf(double value) {
        int low = 0;
        int high = cumulativeWeights.length;
        while (low < high) {
            int mid = (low + high) >>> 1;
            if (cumulativeWeights[mid] <= value) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low < cumulativeWeights.length ? low : lastPositiveIndex;
    }

    /**
     * Efraimidis-Spirakis weighted sampling: each positive-weight index gets the key {@code log(u) / w} with
     * {@code u} uniform on (0, 1]; the indices in descending key order are distributed exactly as successive draws
     * that each remove the chosen index.
     */
    private List<Integer> sampleWithoutReplacement() {
        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < weights.length; i++) {
            if (weights[i] > 0) {
                candidates.add(i);
            }
        }
        double[] keys = new double[weights.length];
        for (int i : candidates) {
            keys[i] = Math.log(1.0 - random.nextDouble()) / weights[i];
        }
        Integer[] order = candidates.toArray(new Integer[0]);
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> keys[i]).reversed());
        return new ArrayList<>(Arrays.asList(order).subList(0, numSamples));
    }

    /** {@code numSamples}, whatever the dataset size. */
    @Override
    public int length(int datasetSize) {
        return numSamples;
    }

    @Override
    public void reset() {
        if (seed != null) {
            random.setSeed(seed);
        }
    }

    public int numSamples() {
        return numSamples;
    }

    public boolean replacement() {
        return replacement;
    }

    @Override
    public String toString() {
        return "WeightedRandomSampler{numSamples=" + numSamples + ", replacement=" + replacement + "}";
    }
}
