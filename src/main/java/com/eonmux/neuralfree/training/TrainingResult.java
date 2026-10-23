package com.eonmux.neuralfree.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class TrainingResult {
    private final List<Float> trainLosses;
    private final List<Float> valLosses;
    private final List<Float> learningRates;
    private final int epochs;
    private final int totalSteps;
    private final long trainingTimeMs;
    private final Float bestValLoss;
    private final int bestEpoch;

    private TrainingResult(Builder builder) {
        this.trainLosses = Collections.unmodifiableList(new ArrayList<>(builder.trainLosses));
        this.valLosses = Collections.unmodifiableList(new ArrayList<>(builder.valLosses));
        this.learningRates = Collections.unmodifiableList(new ArrayList<>(builder.learningRates));
        this.epochs = builder.epochs;
        this.totalSteps = builder.totalSteps;
        this.trainingTimeMs = builder.trainingTimeMs;
        this.bestValLoss = builder.bestValLoss;
        this.bestEpoch = builder.bestEpoch;
    }

    public List<Float> trainLosses() {
        return trainLosses;
    }

    public List<Float> valLosses() {
        return valLosses;
    }

    public List<Float> learningRates() {
        return learningRates;
    }

    public int epochs() {
        return epochs;
    }

    public int totalSteps() {
        return totalSteps;
    }

    public long trainingTimeMs() {
        return trainingTimeMs;
    }

    public Float bestValLoss() {
        return bestValLoss;
    }

    public int bestEpoch() {
        return bestEpoch;
    }

    public float finalTrainLoss() {
        if (trainLosses.isEmpty()) {
            throw new IllegalStateException("No training losses recorded");
        }
        return trainLosses.get(trainLosses.size() - 1);
    }

    public Float finalValLoss() {
        if (valLosses.isEmpty()) {
            return null;
        }
        return valLosses.get(valLosses.size() - 1);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private List<Float> trainLosses = new ArrayList<>();
        private List<Float> valLosses = new ArrayList<>();
        private List<Float> learningRates = new ArrayList<>();
        private int epochs = 0;
        private int totalSteps = 0;
        private long trainingTimeMs = 0;
        private Float bestValLoss = null;
        private int bestEpoch = -1;

        public Builder trainLosses(List<Float> trainLosses) {
            this.trainLosses = new ArrayList<>(trainLosses);
            return this;
        }

        public Builder addTrainLoss(float loss) {
            this.trainLosses.add(loss);
            return this;
        }

        public Builder valLosses(List<Float> valLosses) {
            this.valLosses = new ArrayList<>(valLosses);
            return this;
        }

        public Builder addValLoss(float loss) {
            this.valLosses.add(loss);
            return this;
        }

        public Builder learningRates(List<Float> learningRates) {
            this.learningRates = new ArrayList<>(learningRates);
            return this;
        }

        public Builder addLearningRate(float lr) {
            this.learningRates.add(lr);
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder totalSteps(int totalSteps) {
            this.totalSteps = totalSteps;
            return this;
        }

        public Builder trainingTimeMs(long trainingTimeMs) {
            this.trainingTimeMs = trainingTimeMs;
            return this;
        }

        public Builder bestValLoss(Float bestValLoss) {
            this.bestValLoss = bestValLoss;
            return this;
        }

        public Builder bestEpoch(int bestEpoch) {
            this.bestEpoch = bestEpoch;
            return this;
        }

        public TrainingResult build() {
            return new TrainingResult(this);
        }
    }

    @Override
    public String toString() {
        return "TrainingResult{" +
            "epochs=" + epochs +
            ", totalSteps=" + totalSteps +
            ", trainingTimeMs=" + trainingTimeMs +
            ", finalTrainLoss=" + finalTrainLoss() +
            ", finalValLoss=" + finalValLoss() +
            ", bestValLoss=" + bestValLoss +
            ", bestEpoch=" + bestEpoch +
            '}';
    }
}
