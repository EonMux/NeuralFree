package com.eonmux.neuralfree.data;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Tensor;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class CSVDataset implements Dataset<CSVDataset.Sample> {
    private final Path csvPath;
    private final ComputeBackend backend;
    private final boolean hasHeader;
    private final String delimiter;
    private final List<String> columnNames;
    private final Set<Integer> featureColumns;
    private final Set<Integer> targetColumns;
    private final boolean normalizeFeatures;
    private final float[] featureMean;
    private final float[] featureStd;
    private final boolean oneHotEncodeTargets;
    private final int numClasses;

    private final List<float[]> featureRows;
    private final List<float[]> targetRows;
    private final int numFeatures;
    private final int numTargets;
    private final Map<String, Integer> categoricalMappings;

    private static final int MAX_CATEGORICAL_UNIQUE = 100;

    private CSVDataset(Builder builder) {
        this.csvPath = Objects.requireNonNull(builder.csvPath, "CSV path cannot be null");
        this.backend = Objects.requireNonNull(builder.backend, "Backend cannot be null");
        this.hasHeader = builder.hasHeader;
        this.delimiter = builder.delimiter;
        this.featureColumns = new HashSet<>(builder.featureColumns);
        this.targetColumns = new HashSet<>(builder.targetColumns);
        this.normalizeFeatures = builder.normalizeFeatures;
        this.oneHotEncodeTargets = builder.oneHotEncodeTargets;

        if (!Files.isRegularFile(csvPath)) {
            throw new IllegalArgumentException("CSV path must be a file: " + csvPath);
        }

        this.columnNames = new ArrayList<>();
        this.featureRows = new ArrayList<>();
        this.targetRows = new ArrayList<>();
        this.categoricalMappings = new HashMap<>();

        int[] featureTargetCounts = parseCSV();
        this.numFeatures = featureTargetCounts[0];
        this.numTargets = featureTargetCounts[1];

        if (normalizeFeatures) {
            this.featureMean = new float[numFeatures];
            this.featureStd = new float[numFeatures];
            computeFeatureStats();
            normalizeFeatures();
        } else {
            this.featureMean = null;
            this.featureStd = null;
        }

        if (oneHotEncodeTargets) {
            this.numClasses = countUniqueTargets();
        } else {
            this.numClasses = -1;
        }
    }

    private int[] parseCSV() {
        try (BufferedReader reader = Files.newBufferedReader(csvPath)) {
            String firstLine = reader.readLine();
            if (firstLine == null) {
                throw new IllegalStateException("CSV file is empty: " + csvPath);
            }

            String[] firstLineParts = parseCSVLine(firstLine);
            int numColumns = firstLineParts.length;

            if (hasHeader) {
                columnNames.addAll(Arrays.asList(firstLineParts));
            }

            if (featureColumns.isEmpty()) {
                if (targetColumns.isEmpty()) {
                    for (int i = 0; i < numColumns - 1; i++) {
                        featureColumns.add(i);
                    }
                    targetColumns.add(numColumns - 1);
                } else {
                    for (int i = 0; i < numColumns; i++) {
                        if (!targetColumns.contains(i)) {
                            featureColumns.add(i);
                        }
                    }
                }
            } else if (targetColumns.isEmpty()) {
                for (int i = 0; i < numColumns; i++) {
                    if (!featureColumns.contains(i)) {
                        targetColumns.add(i);
                    }
                }
            }

            List<Integer> sortedFeatures = new ArrayList<>(featureColumns);
            Collections.sort(sortedFeatures);
            List<Integer> sortedTargets = new ArrayList<>(targetColumns);
            Collections.sort(sortedTargets);

            int numFeatures = sortedFeatures.size();
            int numTargets = sortedTargets.size();

            if (!hasHeader) {
                float[] features = new float[numFeatures];
                float[] targets = new float[numTargets];

                for (int i = 0; i < sortedFeatures.size(); i++) {
                    int col = sortedFeatures.get(i);
                    features[i] = parseFloat(firstLineParts[col], col);
                }

                for (int i = 0; i < sortedTargets.size(); i++) {
                    int col = sortedTargets.get(i);
                    targets[i] = parseFloat(firstLineParts[col], col);
                }

                featureRows.add(features);
                targetRows.add(targets);
            }

            String line;
            int lineNum = hasHeader ? 1 : 0;
            while ((line = reader.readLine()) != null) {
                lineNum++;
                if (line.trim().isEmpty()) continue;

                String[] values = parseCSVLine(line);
                if (values.length != numColumns) {
                    throw new IllegalStateException(
                        "Line " + lineNum + " has " + values.length + 
                        " columns, expected " + numColumns);
                }

                float[] features = new float[numFeatures];
                float[] targets = new float[numTargets];

                for (int i = 0; i < sortedFeatures.size(); i++) {
                    int col = sortedFeatures.get(i);
                    features[i] = parseFloat(values[col], col);
                }

                for (int i = 0; i < sortedTargets.size(); i++) {
                    int col = sortedTargets.get(i);
                    targets[i] = parseFloat(values[col], col);
                }

                featureRows.add(features);
                targetRows.add(targets);
            }

            if (featureRows.isEmpty()) {
                throw new IllegalStateException("CSV file has no data rows: " + csvPath);
            }

            return new int[]{numFeatures, numTargets};

        } catch (IOException e) {
            throw new RuntimeException("Failed to read CSV: " + csvPath, e);
        }
    }

    private String[] parseCSVLine(String line) {
        List<String> values = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                values.add(current.toString().trim());
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        values.add(current.toString().trim());
        
        return values.toArray(new String[0]);
    }

    private float parseFloat(String value, int columnIndex) {
        if (value == null || value.isEmpty() || value.equalsIgnoreCase("na") || 
            value.equalsIgnoreCase("null") || value.equals("?")) {
            return 0f;
        }

        try {
            return Float.parseFloat(value);
        } catch (NumberFormatException e) {
            String key = columnIndex + ":" + value.toLowerCase(Locale.ROOT);
            return categoricalMappings.computeIfAbsent(key, k -> categoricalMappings.size());
        }
    }

    private void computeFeatureStats() {
        float[] sum = new float[numFeatures];
        float[] sumSq = new float[numFeatures];

        for (float[] row : featureRows) {
            for (int i = 0; i < numFeatures; i++) {
                sum[i] += row[i];
                sumSq[i] += row[i] * row[i];
            }
        }

        int n = featureRows.size();
        for (int i = 0; i < numFeatures; i++) {
            featureMean[i] = sum[i] / n;
            float variance = (sumSq[i] / n) - (featureMean[i] * featureMean[i]);
            featureStd[i] = (float) Math.sqrt(Math.max(variance, 1e-8));
        }
    }

    private void normalizeFeatures() {
        for (float[] row : featureRows) {
            for (int i = 0; i < numFeatures; i++) {
                row[i] = (row[i] - featureMean[i]) / featureStd[i];
            }
        }
    }

    private int countUniqueTargets() {
        Set<Integer> uniqueTargets = new HashSet<>();
        for (float[] target : targetRows) {
            if (target.length == 1) {
                uniqueTargets.add((int) target[0]);
            }
        }
        return uniqueTargets.size();
    }

    @Override
    public int size() {
        return featureRows.size();
    }

    @Override
    public Sample get(int index) {
        if (index < 0 || index >= featureRows.size()) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " out of bounds for dataset size " + featureRows.size());
        }

        float[] features = featureRows.get(index).clone();
        float[] targets = targetRows.get(index).clone();

        Tensor featureTensor = Tensor.fromArray(backend, features);

        Tensor targetTensor;
        if (oneHotEncodeTargets && targets.length == 1) {
            int labelIndex = (int) targets[0];
            float[] oneHot = new float[numClasses];
            if (labelIndex >= 0 && labelIndex < numClasses) {
                oneHot[labelIndex] = 1f;
            }
            targetTensor = Tensor.fromArray(backend, oneHot);
        } else {
            targetTensor = Tensor.fromArray(backend, targets);
        }

        return new Sample(featureTensor, targetTensor, features, targets);
    }

    public List<String> columnNames() {
        return Collections.unmodifiableList(columnNames);
    }

    public int numFeatures() {
        return numFeatures;
    }

    public int numTargets() {
        return numTargets;
    }

    public int numClasses() {
        return numClasses;
    }

    public float[] featureMean() {
        return featureMean != null ? featureMean.clone() : null;
    }

    public float[] featureStd() {
        return featureStd != null ? featureStd.clone() : null;
    }

    @Override
    public String toString() {
        return "CSVDataset{size=" + featureRows.size() + 
               ", features=" + numFeatures + 
               ", targets=" + numTargets + "}";
    }

    public static final class Sample {
        private final Tensor features;
        private final Tensor targets;
        private final float[] rawFeatures;
        private final float[] rawTargets;

        public Sample(Tensor features, Tensor targets, float[] rawFeatures, float[] rawTargets) {
            this.features = Objects.requireNonNull(features, "Features cannot be null");
            this.targets = Objects.requireNonNull(targets, "Targets cannot be null");
            this.rawFeatures = rawFeatures;
            this.rawTargets = rawTargets;
        }

        public Tensor features() {
            return features;
        }

        public Tensor targets() {
            return targets;
        }

        public float[] rawFeatures() {
            return rawFeatures.clone();
        }

        public float[] rawTargets() {
            return rawTargets.clone();
        }

        @Override
        public String toString() {
            return "Sample{features=" + features.shape() + ", targets=" + targets.shape() + "}";
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Path csvPath;
        private ComputeBackend backend;
        private boolean hasHeader = true;
        private String delimiter = ",";
        private Set<Integer> featureColumns = new HashSet<>();
        private Set<Integer> targetColumns = new HashSet<>();
        private boolean normalizeFeatures = false;
        private boolean oneHotEncodeTargets = false;

        public Builder csvPath(Path csvPath) {
            this.csvPath = csvPath;
            return this;
        }

        public Builder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }

        public Builder hasHeader(boolean hasHeader) {
            this.hasHeader = hasHeader;
            return this;
        }

        public Builder delimiter(String delimiter) {
            this.delimiter = delimiter;
            return this;
        }

        public Builder featureColumns(Integer... columns) {
            this.featureColumns = new HashSet<>(Arrays.asList(columns));
            return this;
        }

        public Builder targetColumns(Integer... columns) {
            this.targetColumns = new HashSet<>(Arrays.asList(columns));
            return this;
        }

        public Builder normalizeFeatures(boolean normalize) {
            this.normalizeFeatures = normalize;
            return this;
        }

        public Builder oneHotEncodeTargets(boolean oneHot) {
            this.oneHotEncodeTargets = oneHot;
            return this;
        }

        public CSVDataset build() {
            return new CSVDataset(this);
        }
    }
}
