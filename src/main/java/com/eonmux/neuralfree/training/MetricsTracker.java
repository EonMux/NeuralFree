package com.eonmux.neuralfree.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class MetricsTracker implements Callback {
    private final List<Float> trainLossHistory;
    private final List<Float> valLossHistory;
    private final List<Float> accuracyHistory;
    private final Map<String, List<Float>> customMetrics;
    private final boolean trackAccuracy;

    private EvalResult lastEvalResult;

    public MetricsTracker() {
        this(true);
    }

    public MetricsTracker(boolean trackAccuracy) {
        this.trainLossHistory = new ArrayList<>();
        this.valLossHistory = new ArrayList<>();
        this.accuracyHistory = new ArrayList<>();
        this.customMetrics = new HashMap<>();
        this.trackAccuracy = trackAccuracy;
    }

    @Override
    public void onTrainBegin(TrainingContext trainer) {
        trainLossHistory.clear();
        valLossHistory.clear();
        accuracyHistory.clear();
        customMetrics.clear();
        lastEvalResult = null;
    }

    @Override
    public void onEpochEnd(int epoch, float trainLoss, Float valLoss) {
        trainLossHistory.add(trainLoss);
        if (valLoss != null) {
            valLossHistory.add(valLoss);
        }
    }

    /**
     * Records the validation result. With accuracy tracking on, the accuracy is added to {@link #accuracyHistory()}
     * only when the result defines one ({@link Trainer#countCorrect} leaves it undefined for a single output per
     * sample).
     */
    @Override
    public void onValidationEnd(int epoch, EvalResult result) {
        lastEvalResult = result;

        Float accuracy = result.metric("accuracy");
        if (trackAccuracy && accuracy != null) {
            accuracyHistory.add(accuracy);
        }

        if (result.metrics() != null) {
            for (Map.Entry<String, Float> entry : result.metrics().entrySet()) {
                String metricName = entry.getKey();
                if (!metricName.equals("accuracy") || !trackAccuracy) {
                    customMetrics.computeIfAbsent(metricName, k -> new ArrayList<>())
                        .add(entry.getValue());
                }
            }
        }
    }

    public List<Float> trainLossHistory() {
        return Collections.unmodifiableList(trainLossHistory);
    }

    public List<Float> valLossHistory() {
        return Collections.unmodifiableList(valLossHistory);
    }

    public List<Float> accuracyHistory() {
        return Collections.unmodifiableList(accuracyHistory);
    }

    public Map<String, List<Float>> customMetrics() {
        Map<String, List<Float>> copy = new HashMap<>();
        for (Map.Entry<String, List<Float>> entry : customMetrics.entrySet()) {
            copy.put(entry.getKey(), Collections.unmodifiableList(new ArrayList<>(entry.getValue())));
        }
        return Collections.unmodifiableMap(copy);
    }

    public List<Float> metricHistory(String name) {
        if (name.equals("train_loss")) {
            return trainLossHistory();
        } else if (name.equals("val_loss")) {
            return valLossHistory();
        } else if (name.equals("accuracy")) {
            return accuracyHistory();
        }
        return Collections.unmodifiableList(customMetrics.getOrDefault(name, Collections.emptyList()));
    }

    public float bestTrainLoss() {
        if (trainLossHistory.isEmpty()) {
            throw new IllegalStateException("No train losses recorded");
        }
        float best = trainLossHistory.get(0);
        for (float loss : trainLossHistory) {
            if (loss < best) {
                best = loss;
            }
        }
        return best;
    }

    public float bestValLoss() {
        if (valLossHistory.isEmpty()) {
            throw new IllegalStateException("No val losses recorded");
        }
        float best = valLossHistory.get(0);
        for (float loss : valLossHistory) {
            if (loss < best) {
                best = loss;
            }
        }
        return best;
    }

    public float bestAccuracy() {
        if (accuracyHistory.isEmpty()) {
            throw new IllegalStateException("No accuracy recorded");
        }
        float best = accuracyHistory.get(0);
        for (float acc : accuracyHistory) {
            if (acc > best) {
                best = acc;
            }
        }
        return best;
    }

    public float latestTrainLoss() {
        if (trainLossHistory.isEmpty()) {
            throw new IllegalStateException("No train losses recorded");
        }
        return trainLossHistory.get(trainLossHistory.size() - 1);
    }

    public float latestValLoss() {
        if (valLossHistory.isEmpty()) {
            throw new IllegalStateException("No val losses recorded");
        }
        return valLossHistory.get(valLossHistory.size() - 1);
    }

    public float latestAccuracy() {
        if (accuracyHistory.isEmpty()) {
            throw new IllegalStateException("No accuracy recorded");
        }
        return accuracyHistory.get(accuracyHistory.size() - 1);
    }

    public EvalResult lastEvalResult() {
        return lastEvalResult;
    }

    public int epochCount() {
        return trainLossHistory.size();
    }

    public boolean hasValidationData() {
        return !valLossHistory.isEmpty();
    }

    public boolean trackAccuracy() {
        return trackAccuracy;
    }

    @Override
    public String name() {
        return "MetricsTracker";
    }

    @Override
    public String toString() {
        return "MetricsTracker{" +
            "epochs=" + epochCount() +
            ", hasValidation=" + hasValidationData() +
            ", trackAccuracy=" + trackAccuracy +
            ", latestTrainLoss=" + (trainLossHistory.isEmpty() ? "N/A" : latestTrainLoss()) +
            ", latestValLoss=" + (valLossHistory.isEmpty() ? "N/A" : latestValLoss()) +
            ", latestAccuracy=" + (accuracyHistory.isEmpty() ? "N/A" : latestAccuracy()) +
            '}';
    }
}
