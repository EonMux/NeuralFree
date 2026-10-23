package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.*;

/**
 * Random forest of CART trees for classification (Gini impurity, majority vote) and regression (variance reduction,
 * mean of the trees), with optional bootstrap sampling and a random feature subset per split, reproducible from
 * {@code seed}.
 *
 * <p><b>CPU only</b>: training is a recursive, data-dependent split search; a device version needs
 * histogram split kernels, which the train kernel library does not have, so it would be a new feature. The
 * {@link ComputeBackend} constructor parameter is kept for API compatibility; it must be non-null and is not
 * used.</p>
 */
public final class RandomForest implements AutoCloseable {
    
    private final int nTrees;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final int maxFeatures;
    private final boolean bootstrap;
    private final int seed;
    
    private List<TreeNode> trees;
    private int nFeatures;
    private int nClasses;
    private boolean isClassification;
    private boolean closed;
    
    public RandomForest(ComputeBackend backend, int nTrees, int maxDepth, 
                       int minSamplesSplit, int minSamplesLeaf, int maxFeatures,
                       boolean bootstrap, int seed) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (nTrees < 1) {
            throw new IllegalArgumentException("nTrees must be at least 1");
        }
        if (maxDepth < 1) {
            throw new IllegalArgumentException("maxDepth must be at least 1");
        }
        if (minSamplesSplit < 2) {
            throw new IllegalArgumentException("minSamplesSplit must be at least 2");
        }
        if (minSamplesLeaf < 1) {
            throw new IllegalArgumentException("minSamplesLeaf must be at least 1");
        }
        
        this.nTrees = nTrees;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.bootstrap = bootstrap;
        this.seed = seed;
        this.closed = false;
    }
    
    public RandomForest(ComputeBackend backend, int nTrees) {
        this(backend, nTrees, Integer.MAX_VALUE, 2, 1, -1, true, 42);
    }
    
    public void fitClassification(float[][] x, int[] y) {
        ensureOpen();
        if (x == null || y == null) {
            throw new IllegalArgumentException("x and y must not be null");
        }
        
        int features = validateRows(x, -1);
        if (y.length != x.length) {
            throw new IllegalArgumentException("y has " + y.length + " labels for " + x.length + " rows");
        }
        
        this.isClassification = true;
        this.nClasses = countUniqueClasses(y);
        this.nFeatures = features;
        
        int actualMaxFeatures = featuresPerSplit(true);
        
        trees = new ArrayList<>();
        Random rng = new Random(seed);
        
        for (int t = 0; t < nTrees; t++) {
            int[] sampleIndices = bootstrap ? bootstrapIndices(x.length, rng) : allIndices(x.length);
            TreeNode tree = buildTree(x, y, sampleIndices, 0, actualMaxFeatures, rng);
            trees.add(tree);
        }
    }
    
    public void fitRegression(float[][] x, float[] y) {
        ensureOpen();
        if (x == null || y == null) {
            throw new IllegalArgumentException("x and y must not be null");
        }
        
        int features = validateRows(x, -1);
        if (y.length != x.length) {
            throw new IllegalArgumentException("y has " + y.length + " targets for " + x.length + " rows");
        }
        
        this.isClassification = false;
        this.nClasses = 0;
        this.nFeatures = features;
        
        int actualMaxFeatures = featuresPerSplit(false);
        
        trees = new ArrayList<>();
        Random rng = new Random(seed);
        
        for (int t = 0; t < nTrees; t++) {
            int[] sampleIndices = bootstrap ? bootstrapIndices(x.length, rng) : allIndices(x.length);
            TreeNode tree = buildRegressionTree(x, y, sampleIndices, 0, actualMaxFeatures, rng);
            trees.add(tree);
        }
    }
    
    public int[] predictClassification(float[][] x) {
        ensureOpen();
        if (x == null) {
            throw new IllegalArgumentException("x must not be null");
        }
        if (trees == null) {
            throw new IllegalStateException("RandomForest not fitted yet");
        }
        if (!isClassification) {
            throw new IllegalStateException("Model was trained for regression");
        }
        validateRows(x, nFeatures);
        
        int n = x.length;
        int[] predictions = new int[n];
        
        for (int i = 0; i < n; i++) {
            Map<Integer, Integer> votes = new HashMap<>();
            for (TreeNode tree : trees) {
                int pred = predictTree(tree, x[i]);
                votes.merge(pred, 1, Integer::sum);
            }
            
            predictions[i] = votes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(0);
        }
        
        return predictions;
    }
    
    public float[] predictRegression(float[][] x) {
        ensureOpen();
        if (x == null) {
            throw new IllegalArgumentException("x must not be null");
        }
        if (trees == null) {
            throw new IllegalStateException("RandomForest not fitted yet");
        }
        if (isClassification) {
            throw new IllegalStateException("Model was trained for classification");
        }
        validateRows(x, nFeatures);
        
        int n = x.length;
        float[] predictions = new float[n];
        
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (TreeNode tree : trees) {
                sum += predictTreeRegression(tree, x[i]);
            }
            predictions[i] = sum / nTrees;
        }
        
        return predictions;
    }
    
    private TreeNode buildTree(float[][] x, int[] y, int[] indices, int depth, 
                               int maxFeats, Random rng) {
        if (indices.length == 0) {
            return new TreeNode(majorityClass(y, indices), true);
        }
        
        if (allSameClass(y, indices)) {
            return new TreeNode(y[indices[0]], true);
        }
        
        if (indices.length < minSamplesSplit || depth >= maxDepth) {
            return new TreeNode(majorityClass(y, indices), true);
        }
        
        int[] featureOrder = shuffledFeatures(rng);
        
        double bestGini = Double.MAX_VALUE;
        int bestFeature = -1;
        float bestThreshold = 0;
        List<Integer> bestLeft = null;
        List<Integer> bestRight = null;
        
        for (int k = 0; k < featureOrder.length; k++) {
            if (k >= maxFeats && bestFeature != -1) {
                break;
            }
            int feat = featureOrder[k];
            float[] thresholds = extractFeatureValues(x, indices, feat);
            
            for (float threshold : thresholds) {
                List<Integer> left = new ArrayList<>();
                List<Integer> right = new ArrayList<>();
                
                for (int idx : indices) {
                    if (x[idx][feat] <= threshold) {
                        left.add(idx);
                    } else {
                        right.add(idx);
                    }
                }
                
                if (left.size() < minSamplesLeaf || right.size() < minSamplesLeaf) {
                    continue;
                }
                
                double gini = weightedGini(y, left, right);
                
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = feat;
                    bestThreshold = threshold;
                    bestLeft = left;
                    bestRight = right;
                }
            }
        }
        
        if (bestFeature == -1) {
            return new TreeNode(majorityClass(y, indices), true);
        }
        
        TreeNode node = new TreeNode(bestFeature, bestThreshold, false);
        node.left = buildTree(x, y, bestLeft.stream().mapToInt(i -> i).toArray(), 
                             depth + 1, maxFeats, rng);
        node.right = buildTree(x, y, bestRight.stream().mapToInt(i -> i).toArray(), 
                              depth + 1, maxFeats, rng);
        
        return node;
    }
    
    private TreeNode buildRegressionTree(float[][] x, float[] y, int[] indices, 
                                         int depth, int maxFeats, Random rng) {
        if (indices.length == 0) {
            return new TreeNode(0.0f, true);
        }
        
        if (indices.length < minSamplesSplit || depth >= maxDepth) {
            return new TreeNode(meanValue(y, indices), true);
        }
        
        int[] featureOrder = shuffledFeatures(rng);
        
        double bestMse = Double.MAX_VALUE;
        int bestFeature = -1;
        float bestThreshold = 0;
        List<Integer> bestLeft = null;
        List<Integer> bestRight = null;
        
        for (int k = 0; k < featureOrder.length; k++) {
            if (k >= maxFeats && bestFeature != -1) {
                break;
            }
            int feat = featureOrder[k];
            float[] thresholds = extractFeatureValues(x, indices, feat);
            
            for (float threshold : thresholds) {
                List<Integer> left = new ArrayList<>();
                List<Integer> right = new ArrayList<>();
                
                for (int idx : indices) {
                    if (x[idx][feat] <= threshold) {
                        left.add(idx);
                    } else {
                        right.add(idx);
                    }
                }
                
                if (left.size() < minSamplesLeaf || right.size() < minSamplesLeaf) {
                    continue;
                }
                
                double mse = weightedMse(y, left, right);
                
                if (mse < bestMse) {
                    bestMse = mse;
                    bestFeature = feat;
                    bestThreshold = threshold;
                    bestLeft = left;
                    bestRight = right;
                }
            }
        }
        
        if (bestFeature == -1) {
            return new TreeNode(meanValue(y, indices), true);
        }
        
        TreeNode node = new TreeNode(bestFeature, bestThreshold, false);
        node.left = buildRegressionTree(x, y, bestLeft.stream().mapToInt(i -> i).toArray(), 
                                       depth + 1, maxFeats, rng);
        node.right = buildRegressionTree(x, y, bestRight.stream().mapToInt(i -> i).toArray(), 
                                        depth + 1, maxFeats, rng);
        
        return node;
    }
    
    private int predictTree(TreeNode node, float[] x) {
        if (node.isLeaf) {
            return node.prediction;
        }
        
        if (x[node.feature] <= node.threshold) {
            return predictTree(node.left, x);
        } else {
            return predictTree(node.right, x);
        }
    }
    
    private float predictTreeRegression(TreeNode node, float[] x) {
        if (node.isLeaf) {
            return node.value;
        }
        
        if (x[node.feature] <= node.threshold) {
            return predictTreeRegression(node.left, x);
        } else {
            return predictTreeRegression(node.right, x);
        }
    }
    
    private int[] bootstrapIndices(int n, Random rng) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = rng.nextInt(n);
        }
        return indices;
    }
    
    private int[] allIndices(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        return indices;
    }
    
    /**
     * Checks that every row of {@code x} is non-null and {@code expectedFeatures} long; with a negative
     * {@code expectedFeatures} (fitting), {@code x} must be non-empty and its rows share the first row's length of at
     * least one. Returns the row length.
     */
    private static int validateRows(float[][] x, int expectedFeatures) {
        if (x.length == 0 && expectedFeatures < 0) {
            throw new IllegalArgumentException("x must have at least one row");
        }
        int features = expectedFeatures >= 0 ? expectedFeatures : (x[0] == null ? 0 : x[0].length);
        if (features < 1) {
            throw new IllegalArgumentException("x rows must have at least one feature");
        }
        for (int i = 0; i < x.length; i++) {
            if (x[i] == null || x[i].length != features) {
                throw new IllegalArgumentException("x[" + i + "] must have " + features + " features");
            }
        }
        return features;
    }
    
    /**
     * Number of candidate features drawn per split: {@code maxFeatures} when positive, otherwise the R randomForest
     * default of {@code sqrt(nFeatures)} for classification and {@code nFeatures / 3} for regression, never below one
     * so that every split has a feature to test.
     */
    private int featuresPerSplit(boolean classification) {
        if (maxFeatures > 0) {
            return maxFeatures;
        }
        int byRule = classification ? (int) Math.sqrt(nFeatures) : nFeatures / 3;
        return Math.max(1, byRule);
    }
    
    /**
     * All feature indices in random order. A split examines the first {@code maxFeats} of them and, as scikit-learn
     * does, keeps examining further features while none of the examined ones yields a valid split.
     */
    private int[] shuffledFeatures(Random rng) {
        List<Integer> features = new ArrayList<>();
        for (int i = 0; i < nFeatures; i++) {
            features.add(i);
        }
        Collections.shuffle(features, rng);
        return features.stream().mapToInt(i -> i).toArray();
    }
    
    private int countUniqueClasses(int[] y) {
        return (int) Arrays.stream(y).distinct().count();
    }
    
    private boolean allSameClass(int[] y, int[] indices) {
        int first = y[indices[0]];
        for (int idx : indices) {
            if (y[idx] != first) return false;
        }
        return true;
    }
    
    private int majorityClass(int[] y, int[] indices) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int idx : indices) {
            counts.merge(y[idx], 1, Integer::sum);
        }
        return counts.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(0);
    }
    
    private float[] extractFeatureValues(float[][] x, int[] indices, int feature) {
        float[] values = new float[indices.length];
        for (int i = 0; i < indices.length; i++) {
            values[i] = x[indices[i]][feature];
        }
        Arrays.sort(values);
        List<Float> uniqueList = new ArrayList<>();
        float prev = Float.NaN;
        for (float v : values) {
            if (Float.isNaN(prev) || v != prev) {
                uniqueList.add(v);
                prev = v;
            }
        }
        float[] unique = new float[uniqueList.size()];
        for (int i = 0; i < uniqueList.size(); i++) {
            unique[i] = uniqueList.get(i);
        }
        return unique;
    }
    
    private double weightedGini(int[] y, List<Integer> left, List<Integer> right) {
        int total = left.size() + right.size();
        double leftGini = giniImpurity(y, left);
        double rightGini = giniImpurity(y, right);
        
        return (left.size() * leftGini + right.size() * rightGini) / total;
    }
    
    private double giniImpurity(int[] y, List<Integer> indices) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int idx : indices) {
            counts.merge(y[idx], 1, Integer::sum);
        }
        
        double sum = 0.0;
        int total = indices.size();
        for (int count : counts.values()) {
            double p = (double) count / total;
            sum += p * p;
        }
        
        return 1.0 - sum;
    }
    
    private double weightedMse(float[] y, List<Integer> left, List<Integer> right) {
        int total = left.size() + right.size();
        double leftMse = mse(y, left);
        double rightMse = mse(y, right);
        
        return (left.size() * leftMse + right.size() * rightMse) / total;
    }
    
    private double mse(float[] y, List<Integer> indices) {
        double mean = meanValue(y, indices);
        double sum = 0.0;
        for (int idx : indices) {
            double diff = y[idx] - mean;
            sum += diff * diff;
        }
        return sum / indices.size();
    }
    
    private float meanValue(float[] y, int[] indices) {
        double sum = 0.0;
        for (int idx : indices) {
            sum += y[idx];
        }
        return (float) (sum / indices.length);
    }
    
    private float meanValue(float[] y, List<Integer> indices) {
        double sum = 0.0;
        for (int idx : indices) {
            sum += y[idx];
        }
        return (float) (sum / indices.size());
    }
    
    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("RandomForest has been closed");
        }
    }
    
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (trees != null) {
            trees.clear();
        }
    }
    
    private static final class TreeNode {
        final int feature;
        final float threshold;
        final int prediction;
        final float value;
        final boolean isLeaf;
        
        TreeNode left;
        TreeNode right;
        
        TreeNode(int prediction, boolean isLeaf) {
            this.feature = -1;
            this.threshold = 0;
            this.prediction = prediction;
            this.value = 0;
            this.isLeaf = isLeaf;
        }
        
        TreeNode(float value, boolean isLeaf) {
            this.feature = -1;
            this.threshold = 0;
            this.prediction = 0;
            this.value = value;
            this.isLeaf = isLeaf;
        }
        
        TreeNode(int feature, float threshold, boolean isLeaf) {
            this.feature = feature;
            this.threshold = threshold;
            this.prediction = 0;
            this.value = 0;
            this.isLeaf = isLeaf;
        }
    }
}
