package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * A fixed-capacity ring of {@link Transition}s held on the host, sampled uniformly or by priority (proportional
 * prioritized replay with importance weights). Transitions carry a discrete action index or a continuous action
 * vector ({@link #add(float[], float[], float, float[], boolean)}); a batch returns both forms
 * ({@link SampleBatch#actions()}, {@link SampleBatch#continuousActions()}).
 */
public final class ReplayBuffer implements AutoCloseable {
    private final int capacity;
    private final int stateSize;
    private final ComputeBackend backend;
    private final boolean usePrioritizedReplay;
    private final float alpha;
    private float beta;
    private final float betaIncrement;
    private final float epsilon;

    private final Transition[] buffer;
    private final float[] priorities;
    private float maxPriority;

    private int size;
    private int position;
    private volatile boolean closed;

    public ReplayBuffer(int capacity, int stateSize, ComputeBackend backend) {
        this(capacity, stateSize, backend, false, 1.0f, 0.4f, 0.001f, 1e-6f);
    }

    public ReplayBuffer(int capacity, int stateSize, ComputeBackend backend, boolean usePrioritizedReplay,
                        float alpha, float beta, float betaIncrement, float epsilon) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("Capacity must be positive, got: " + capacity);
        }
        if (stateSize <= 0) {
            throw new IllegalArgumentException("State size must be positive, got: " + stateSize);
        }
        Objects.requireNonNull(backend, "Backend cannot be null");

        this.capacity = capacity;
        this.stateSize = stateSize;
        this.backend = backend;
        this.usePrioritizedReplay = usePrioritizedReplay;
        this.alpha = alpha;
        this.beta = beta;
        this.betaIncrement = betaIncrement;
        this.epsilon = epsilon;

        this.buffer = new Transition[capacity];
        this.priorities = new float[capacity];
        this.maxPriority = 1.0f;

        this.size = 0;
        this.position = 0;
        this.closed = false;
    }

    /**
     * Adds a transition with the discrete action {@code action}, at the current maximum priority. The states are
     * copied, so the caller may reuse its arrays.
     *
     * @throws IllegalArgumentException when a state does not have {@code stateSize} values
     */
    public synchronized void add(float[] state, int action, float reward, float[] nextState, boolean done) {
        checkNotClosed();
        float[] stateCopy = RlInputs.copyOfState(state, stateSize, "State");
        float[] nextStateCopy = RlInputs.copyOfState(nextState, stateSize, "Next state");
        store(Transition.ofOwnedStates(stateCopy, action, reward, nextStateCopy, done, maxPriority));
    }

    /**
     * Adds a transition with the continuous action {@code action}, at the current maximum priority; its
     * {@link Transition#action()} is {@link Transition#NO_DISCRETE_ACTION}. The states and the action are copied, so
     * the caller may reuse its arrays.
     *
     * @throws IllegalArgumentException when a state does not have {@code stateSize} values or the action is empty
     */
    public synchronized void add(float[] state, float[] action, float reward, float[] nextState, boolean done) {
        checkNotClosed();
        float[] stateCopy = RlInputs.copyOfState(state, stateSize, "State");
        float[] nextStateCopy = RlInputs.copyOfState(nextState, stateSize, "Next state");
        store(Transition.ofOwnedStates(stateCopy, action, reward, nextStateCopy, done, maxPriority));
    }

    /** Adds {@code transition}'s state, action (discrete or continuous), reward and flag at the maximum priority. */
    public synchronized void add(Transition transition) {
        Objects.requireNonNull(transition, "Transition cannot be null");
        float[] continuous = transition.continuousActionRef();
        if (continuous != null) {
            add(transition.stateRef(), continuous, transition.reward(), transition.nextStateRef(), transition.done());
        } else {
            add(transition.stateRef(), transition.action(), transition.reward(), transition.nextStateRef(),
                transition.done());
        }
    }

    private void store(Transition transition) {
        buffer[position] = transition;
        priorities[position] = maxPriority;
        position = (position + 1) % capacity;
        if (size < capacity) {
            size++;
        }
    }

    /** {@link #sample(int, Random)} with the calling thread's {@link ThreadLocalRandom}. */
    public synchronized SampleBatch sample(int batchSize) {
        return sample(batchSize, ThreadLocalRandom.current());
    }

    /**
     * Draws {@code batchSize} stored transitions, uniformly or by priority. The batch's state, next-state and
     * continuous-action rows are fresh copies the caller owns: modifying them does not change the stored transitions.
     *
     * @throws IllegalStateException when fewer than {@code batchSize} transitions are stored
     */
    public synchronized SampleBatch sample(int batchSize, Random random) {
        checkNotClosed();
        if (size < batchSize) {
            throw new IllegalStateException("Not enough transitions: have " + size + ", need " + batchSize);
        }
        Objects.requireNonNull(random, "Random cannot be null");

        int[] indices = new int[batchSize];
        float[] weights = new float[batchSize];
        float[] sampledPriorities = new float[batchSize];

        if (usePrioritizedReplay) {
            float sum = 0f;
            for (int i = 0; i < size; i++) {
                sum += (float) Math.pow(priorities[i] + epsilon, alpha);
            }

            List<Integer> sampledIndices = new ArrayList<>(batchSize);
            for (int j = 0; j < batchSize; j++) {
                float r = random.nextFloat() * sum;
                float cumulative = 0f;
                int idx = 0;
                for (int i = 0; i < size; i++) {
                    cumulative += (float) Math.pow(priorities[i] + epsilon, alpha);
                    if (cumulative >= r) {
                        idx = i;
                        break;
                    }
                }
                sampledIndices.add(idx);
                sampledPriorities[j] = priorities[idx];
            }

            for (int j = 0; j < batchSize; j++) {
                int idx = sampledIndices.get(j);
                indices[j] = idx;
                float probability = (float) Math.pow(priorities[idx] + epsilon, alpha) / sum;
                weights[j] = (float) Math.pow(size * probability, -beta);
            }

            float maxWeight = 0f;
            for (float w : weights) {
                if (w > maxWeight) maxWeight = w;
            }
            if (maxWeight == 0f) maxWeight = 1f;
            for (int j = 0; j < batchSize; j++) {
                weights[j] /= maxWeight;
            }
        } else {
            for (int j = 0; j < batchSize; j++) {
                indices[j] = random.nextInt(size);
                weights[j] = 1.0f;
                sampledPriorities[j] = 1.0f;
            }
        }

        float[][] states = new float[batchSize][];
        int[] actions = new int[batchSize];
        float[][] continuousActions = new float[batchSize][];
        float[] rewards = new float[batchSize];
        float[][] nextStates = new float[batchSize][];
        boolean[] dones = new boolean[batchSize];

        for (int j = 0; j < batchSize; j++) {
            Transition t = buffer[indices[j]];
            states[j] = t.state();
            actions[j] = t.action();
            continuousActions[j] = t.continuousAction();
            rewards[j] = t.reward();
            nextStates[j] = t.nextState();
            dones[j] = t.done();
        }

        return new SampleBatch(states, actions, continuousActions, rewards, nextStates, dones, indices, weights);
    }

    public synchronized void updatePriorities(int[] indices, float[] tdErrors) {
        checkNotClosed();
        Objects.requireNonNull(indices, "Indices cannot be null");
        Objects.requireNonNull(tdErrors, "TD errors cannot be null");

        if (indices.length != tdErrors.length) {
            throw new IllegalArgumentException("Indices and TD errors must have same length");
        }

        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            if (idx < 0 || idx >= size) {
                throw new IndexOutOfBoundsException("Index out of bounds: " + idx);
            }
            float newPriority = (float) (Math.abs(tdErrors[i]) + epsilon);
            priorities[idx] = newPriority;
            maxPriority = Math.max(maxPriority, newPriority);
        }
    }

    public void incrementBeta() {
        if (usePrioritizedReplay && beta < 1.0f) {
            this.beta = Math.min(1.0f, beta + betaIncrement);
        }
    }

    public synchronized void clear() {
        checkNotClosed();
        Arrays.fill(buffer, null);
        Arrays.fill(priorities, 0f);
        size = 0;
        position = 0;
        maxPriority = 1.0f;
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return capacity;
    }

    public boolean isFull() {
        return size >= capacity;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public float beta() {
        return beta;
    }

    public boolean usesPrioritizedReplay() {
        return usePrioritizedReplay;
    }

    public ComputeBackend backend() {
        return backend;
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("ReplayBuffer has been closed");
        }
    }

    /** Closes the buffer; later adds and samples throw. Idempotent. */
    @Override
    public synchronized void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "ReplayBuffer{capacity=" + capacity + ", size=" + size + 
               ", prioritized=" + usePrioritizedReplay + '}';
    }

    public static final class SampleBatch {
        private final float[][] states;
        private final int[] actions;
        private final float[][] continuousActions;
        private final float[] rewards;
        private final float[][] nextStates;
        private final boolean[] dones;
        private final int[] indices;
        private final float[] importanceWeights;

        /** A batch of discrete transitions (every continuous action null). */
        public SampleBatch(float[][] states, int[] actions, float[] rewards,
                          float[][] nextStates, boolean[] dones, int[] indices, float[] importanceWeights) {
            this(states, actions, new float[actions.length][], rewards, nextStates, dones, indices, importanceWeights);
        }

        /**
         * A batch whose entry {@code j} has the discrete action {@code actions[j]} or, when
         * {@code continuousActions[j]} is not null, that continuous action.
         */
        public SampleBatch(float[][] states, int[] actions, float[][] continuousActions, float[] rewards,
                           float[][] nextStates, boolean[] dones, int[] indices, float[] importanceWeights) {
            this.states = states;
            this.actions = actions;
            this.continuousActions = continuousActions;
            this.rewards = rewards;
            this.nextStates = nextStates;
            this.dones = dones;
            this.indices = indices;
            this.importanceWeights = importanceWeights;
        }

        public float[][] states() {
            return states;
        }

        public int[] actions() {
            return actions;
        }

        /** The continuous action of each entry, null for an entry whose action is discrete. */
        public float[][] continuousActions() {
            return continuousActions;
        }

        public float[] rewards() {
            return rewards;
        }

        public float[][] nextStates() {
            return nextStates;
        }

        public boolean[] dones() {
            return dones;
        }

        public int[] indices() {
            return indices;
        }

        public float[] importanceWeights() {
            return importanceWeights;
        }

        public int batchSize() {
            return states != null ? states.length : 0;
        }

        public float[] flattenedStates() {
            if (states == null || states.length == 0) return new float[0];
            int stateSize = states[0].length;
            float[] flat = new float[states.length * stateSize];
            for (int i = 0; i < states.length; i++) {
                System.arraycopy(states[i], 0, flat, i * stateSize, stateSize);
            }
            return flat;
        }

        public float[] flattenedNextStates() {
            if (nextStates == null || nextStates.length == 0) return new float[0];
            int stateSize = nextStates[0].length;
            float[] flat = new float[nextStates.length * stateSize];
            for (int i = 0; i < nextStates.length; i++) {
                System.arraycopy(nextStates[i], 0, flat, i * stateSize, stateSize);
            }
            return flat;
        }

        @Override
        public String toString() {
            return "SampleBatch{batchSize=" + batchSize() + '}';
        }
    }

    public static final class Builder {
        private int capacity;
        private int stateSize;
        private ComputeBackend backend;
        private boolean usePrioritizedReplay = false;
        private float alpha = 1.0f;
        private float beta = 0.4f;
        private float betaIncrement = 0.001f;
        private float epsilon = 1e-6f;

        public Builder capacity(int capacity) {
            this.capacity = capacity;
            return this;
        }

        public Builder stateSize(int stateSize) {
            this.stateSize = stateSize;
            return this;
        }

        public Builder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }

        public Builder usePrioritizedReplay(boolean use) {
            this.usePrioritizedReplay = use;
            return this;
        }

        public Builder alpha(float alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder beta(float beta) {
            this.beta = beta;
            return this;
        }

        public Builder betaIncrement(float increment) {
            this.betaIncrement = increment;
            return this;
        }

        public Builder epsilon(float epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        public ReplayBuffer build() {
            return new ReplayBuffer(capacity, stateSize, backend, usePrioritizedReplay,
                                   alpha, beta, betaIncrement, epsilon);
        }
    }
}
