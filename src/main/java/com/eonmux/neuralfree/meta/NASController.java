package com.eonmux.neuralfree.meta;

import com.eonmux.neuralfree.meta.NASSearchSpace.Edge;
import com.eonmux.neuralfree.meta.NASSearchSpace.Operation;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

public final class NASController implements AutoCloseable {
    private final NASSearchSpace searchSpace;
    private final SearchStrategy strategy;
    private final Random random;
    private final AtomicBoolean closed;
    private final List<Candidate> population;
    private final int populationSize;
    private final int tournamentSize;

    private List<Float> architectureWeights;
    private float learningRate;
    private float mutationRate;

    public NASController(NASSearchSpace searchSpace, SearchStrategy strategy) {
        this(searchSpace, strategy, 42);
    }

    public NASController(NASSearchSpace searchSpace, SearchStrategy strategy, long seed) {
        Objects.requireNonNull(searchSpace, "Search space cannot be null");
        Objects.requireNonNull(strategy, "Search strategy cannot be null");

        this.searchSpace = searchSpace;
        this.strategy = strategy;
        this.random = new Random(seed);
        this.closed = new AtomicBoolean(false);
        this.population = new ArrayList<>();
        this.populationSize = 50;
        this.mutationRate = 0.1f;
        this.tournamentSize = 5;
        this.learningRate = 0.01f;

        initializeArchitectureWeights();
    }

    private void initializeArchitectureWeights() {
        architectureWeights = new ArrayList<>();
        int numEdges = searchSpace.numEdges();
        int numOps = searchSpace.availableOperations().size();
        
        for (int i = 0; i < numEdges * numOps; i++) {
            architectureWeights.add(0.0f);
        }
    }

    public NASResult search(int numEvaluations, Evaluator evaluator) {
        ensureOpen();
        
        if (numEvaluations < 1) {
            throw new IllegalArgumentException("Number of evaluations must be at least 1");
        }
        Objects.requireNonNull(evaluator, "Evaluator cannot be null");

        List<Candidate> history = new ArrayList<>();

        switch (strategy) {
            case RANDOM:
                history = randomSearch(numEvaluations, evaluator);
                break;
            case EVOLUTIONARY:
                history = evolutionarySearch(numEvaluations, evaluator);
                break;
            case DARTS:
                history = dartsSearch(numEvaluations, evaluator);
                break;
            default:
                throw new IllegalStateException("Unknown search strategy: " + strategy);
        }

        Candidate best = findBest(history);
        return new NASResult(best.architecture, best.fitness, history);
    }

    private List<Candidate> randomSearch(int numEvaluations, Evaluator evaluator) {
        List<Candidate> results = new ArrayList<>();

        for (int i = 0; i < numEvaluations; i++) {
            searchSpace.setRandomArchitecture(random.nextLong());
            List<Integer> arch = searchSpace.getArchitecture();
            float fitness = evaluator.evaluate(arch);
            results.add(new Candidate(arch, fitness));
        }

        return results;
    }

    private List<Candidate> evolutionarySearch(int numEvaluations, Evaluator evaluator) {
        List<Candidate> results = new ArrayList<>();
        
        initializePopulation(evaluator);

        int evaluations = populationSize;
        results.addAll(population);

        while (evaluations < numEvaluations) {
            Candidate parent1 = tournamentSelect();
            Candidate parent2 = tournamentSelect();

            Candidate child = crossover(parent1, parent2);
            child = mutate(child, evaluator);

            evaluations++;
            results.add(child);

            addToPopulation(child);
        }

        return results;
    }

    private void initializePopulation(Evaluator evaluator) {
        population.clear();
        
        for (int i = 0; i < populationSize; i++) {
            searchSpace.setRandomArchitecture(random.nextLong());
            List<Integer> arch = searchSpace.getArchitecture();
            float fitness = evaluator.evaluate(arch);
            population.add(new Candidate(arch, fitness));
        }
    }

    private Candidate tournamentSelect() {
        List<Candidate> tournament = new ArrayList<>();
        
        for (int i = 0; i < tournamentSize; i++) {
            int idx = random.nextInt(population.size());
            tournament.add(population.get(idx));
        }

        return Collections.max(tournament, Comparator.comparingDouble(c -> c.fitness));
    }

    private Candidate crossover(Candidate parent1, Candidate parent2) {
        List<Integer> childArch = new ArrayList<>();
        List<Integer> arch1 = parent1.architecture;
        List<Integer> arch2 = parent2.architecture;

        for (int i = 0; i < arch1.size(); i++) {
            if (random.nextBoolean()) {
                childArch.add(arch1.get(i));
            } else {
                childArch.add(arch2.get(i));
            }
        }

        return new Candidate(childArch, Float.NEGATIVE_INFINITY);
    }

    private Candidate mutate(Candidate candidate, Evaluator evaluator) {
        List<Integer> mutatedArch = new ArrayList<>(candidate.architecture);
        int numOps = searchSpace.availableOperations().size();

        for (int i = 0; i < mutatedArch.size(); i++) {
            if (random.nextFloat() < mutationRate) {
                int newOp = random.nextInt(numOps);
                mutatedArch.set(i, newOp);
            }
        }

        float fitness = evaluator.evaluate(mutatedArch);
        return new Candidate(mutatedArch, fitness);
    }

    private void addToPopulation(Candidate candidate) {
        if (population.size() < populationSize) {
            population.add(candidate);
            return;
        }

        int worstIdx = 0;
        float worstFitness = population.get(0).fitness;
        
        for (int i = 1; i < population.size(); i++) {
            if (population.get(i).fitness < worstFitness) {
                worstFitness = population.get(i).fitness;
                worstIdx = i;
            }
        }

        if (candidate.fitness > worstFitness) {
            population.set(worstIdx, candidate);
        }
    }

    private List<Candidate> dartsSearch(int numIterations, Evaluator evaluator) {
        List<Candidate> results = new ArrayList<>();

        for (int iter = 0; iter < numIterations; iter++) {
            updateArchitectureWeights();
            
            List<Integer> arch = sampleFromWeights();
            float fitness = evaluator.evaluate(arch);
            results.add(new Candidate(arch, fitness));

            updateWeightsWithReward(arch, fitness);
        }

        return results;
    }

    private void updateArchitectureWeights() {
        int numEdges = searchSpace.numEdges();
        int numOps = searchSpace.availableOperations().size();

        for (int edge = 0; edge < numEdges; edge++) {
            float sum = 0.0f;
            for (int op = 0; op < numOps; op++) {
                int idx = edge * numOps + op;
                sum += (float) Math.exp(architectureWeights.get(idx));
            }

            for (int op = 0; op < numOps; op++) {
                int idx = edge * numOps + op;
                float weight = architectureWeights.get(idx);
                architectureWeights.set(idx, weight * 0.99f);
            }
        }
    }

    private List<Integer> sampleFromWeights() {
        List<Integer> arch = new ArrayList<>();
        int numEdges = searchSpace.numEdges();
        int numOps = searchSpace.availableOperations().size();

        for (int edge = 0; edge < numEdges; edge++) {
            List<Float> probs = new ArrayList<>();
            float sum = 0.0f;

            for (int op = 0; op < numOps; op++) {
                int idx = edge * numOps + op;
                float expWeight = (float) Math.exp(architectureWeights.get(idx));
                probs.add(expWeight);
                sum += expWeight;
            }

            float r = random.nextFloat();
            float cumProb = 0.0f;
            int selectedOp = 0;

            for (int op = 0; op < numOps; op++) {
                cumProb += probs.get(op) / sum;
                if (r <= cumProb) {
                    selectedOp = op;
                    break;
                }
            }

            arch.add(selectedOp);
        }

        return arch;
    }

    private void updateWeightsWithReward(List<Integer> arch, float fitness) {
        int numOps = searchSpace.availableOperations().size();

        for (int edge = 0; edge < arch.size(); edge++) {
            int op = arch.get(edge);
            int idx = edge * numOps + op;
            float current = architectureWeights.get(idx);
            architectureWeights.set(idx, current + learningRate * fitness);
        }
    }

    private Candidate findBest(List<Candidate> candidates) {
        if (candidates.isEmpty()) {
            return null;
        }
        
        return Collections.max(candidates, Comparator.comparingDouble(c -> c.fitness));
    }

    public List<Float> getArchitectureWeights() {
        return new ArrayList<>(architectureWeights);
    }

    public void setLearningRate(float lr) {
        if (lr <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = lr;
    }

    public void setMutationRate(float rate) {
        if (rate < 0 || rate > 1) {
            throw new IllegalArgumentException("Mutation rate must be between 0 and 1");
        }
        this.mutationRate = rate;
    }

    public NASSearchSpace searchSpace() {
        return searchSpace;
    }

    public SearchStrategy strategy() {
        return strategy;
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("NASController has been closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            population.clear();
            architectureWeights.clear();
        }
    }

    @Override
    public String toString() {
        return "NASController{" +
            "strategy=" + strategy +
            ", searchSpace=" + searchSpace.name() +
            ", populationSize=" + populationSize +
            ", closed=" + closed.get() +
            '}';
    }

    public enum SearchStrategy {
        RANDOM,
        EVOLUTIONARY,
        DARTS
    }

    public interface Evaluator {
        float evaluate(List<Integer> architecture);
    }

    private static final class Candidate {
        final List<Integer> architecture;
        final float fitness;

        Candidate(List<Integer> architecture, float fitness) {
            this.architecture = Collections.unmodifiableList(new ArrayList<>(architecture));
            this.fitness = fitness;
        }

        @Override
        public String toString() {
            return "Candidate{fitness=" + fitness + ", arch=" + architecture + "}";
        }
    }

    public static final class NASResult {
        private final List<Integer> bestArchitecture;
        private final float bestFitness;
        private final List<Candidate> history;

        public NASResult(List<Integer> bestArchitecture, float bestFitness, 
                        List<Candidate> history) {
            this.bestArchitecture = Collections.unmodifiableList(new ArrayList<>(bestArchitecture));
            this.bestFitness = bestFitness;
            this.history = Collections.unmodifiableList(new ArrayList<>(history));
        }

        public List<Integer> bestArchitecture() {
            return bestArchitecture;
        }

        public float bestFitness() {
            return bestFitness;
        }

        public int totalEvaluations() {
            return history.size();
        }

        public List<Float> fitnessHistory() {
            return history.stream()
                .map(c -> c.fitness)
                .collect(Collectors.toList());
        }

        public float averageFitness() {
            return (float) history.stream()
                .mapToDouble(c -> c.fitness)
                .average()
                .orElse(Double.NaN);
        }

        @Override
        public String toString() {
            return "NASResult{" +
                "bestFitness=" + bestFitness +
                ", evaluations=" + totalEvaluations() +
                ", bestArch=" + bestArchitecture +
                '}';
        }
    }
}
