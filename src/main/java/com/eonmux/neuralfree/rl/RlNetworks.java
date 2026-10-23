package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.nn.layers.Linear;

import java.util.Objects;
import java.util.Random;

/**
 * The networks of the reinforcement-learning algorithms: the multilayer perceptron with the layout of
 * Stable-Baselines3's default {@code net_arch} for actor, critic and Q-networks, two hidden layers of width
 * {@code hidden} with ReLU activations,
 * {@code Linear(in, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, out)}. It is a
 * {@link Sequential} whose five layers are its children {@code "0"} to {@code "4"}, so its parameters are named
 * {@code "0.weight"}, {@code "0.bias"}, {@code "2.weight"}, {@code "2.bias"}, {@code "4.weight"} and
 * {@code "4.bias"} and listed in that order by {@link Sequential#parameters()}.
 *
 * <p><b>Initialization.</b> The two hidden layers feed a ReLU and take He-normal weights {@code N(0, 2 / fanIn)}
 * ({@link Linear#initializeKaiming}); the output layer takes Glorot-normal weights
 * {@code N(0, 2 / (fanIn + fanOut))} ({@link Linear#initialize(ComputeBackend, Random)}); every bias starts at zero.
 * Each layer draws one seed from the algorithm's {@link Random}, in layer order, so a network is reproduced exactly by
 * the algorithm's seed and is identical on every backend.</p>
 */
final class RlNetworks {

    private RlNetworks() {
    }

    /**
     * The perceptron {@code inputs -> hidden -> hidden -> outputs} on {@code backend}, initialized from {@code random}
     * (see the class description). The caller owns the result and closes it.
     *
     * @throws IllegalArgumentException when a width is not positive
     */
    static Sequential mlp(ComputeBackend backend, int inputs, int hidden, int outputs, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        requirePositiveHiddenSize(hidden);
        Linear first = new Linear(inputs, hidden);
        Linear second = new Linear(hidden, hidden);
        Linear head = new Linear(hidden, outputs);
        Sequential network = new Sequential(first, new ReLU(), second, new ReLU(), head);
        try {
            first.initializeKaiming(backend, random);
            second.initializeKaiming(backend, random);
            head.initialize(backend, random);
        } catch (RuntimeException e) {
            try {
                network.close();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
        return network;
    }

    /**
     * Refuses a hidden width that is not positive.
     *
     * @throws IllegalArgumentException when {@code hiddenSize <= 0}
     */
    static void requirePositiveHiddenSize(int hiddenSize) {
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("Hidden size must be positive: " + hiddenSize);
        }
    }
}
