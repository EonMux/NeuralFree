package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.*;

/**
 * Gradient-boosted regression trees for squared-error loss: starts from the mean target and adds
 * {@code learningRate}-scaled trees fitted to the loss gradients, with optional row subsampling, reproducible from
 * {@code seed}.
 *
 * <p><b>CPU only</b>: each tree is grown by a recursive, data-dependent split search; a device
 * version needs histogram split kernels, which the train kernel library does not have, so it would be a new feature.
 * The {@link ComputeBackend} constructor parameter is kept for API compatibility; it must be non-null and is not
 * used.</p>
 */
public final class GradientBoostedTrees implements AutoCloseable {
    
    private final int nEstimators;
    private final float learningRate;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final float subsample;
    private final int maxBins;
    private final int seed;
    
    private List<RegTree> trees;
    private float initialPrediction;
    private int nFeatures;
    private boolean closed;
    
    public GradientBoostedTrees(ComputeBackend backend, int nEstimators, float learningRate,
                                int maxDepth, int minSamplesSplit, float subsample, 
                                int maxBins, int seed) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (nEstimators < 1) {
            throw new IllegalArgumentException("nEstimators must be at least 1");
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive");
        }
        
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.subsample = subsample;
        this.maxBins = maxBins;
        this.seed = seed;
        this.closed = false;
    }
    
    public GradientBoostedTrees(ComputeBackend backend) {
        this(backend, 100, 0.1f, 3, 2, 1.0f, 256, 42);
    }
    
    public void fit(float[][] x, float[] y) {
        ensureOpen();
        if (x == null || y == null) {
            throw new IllegalArgumentException("x and y must not be null");
        }
        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must have same length");
        }
        
        this.nFeatures = x[0].length;
        this.trees = new ArrayList<>();
        
        initialPrediction = mean(y);
        float[] predictions = new float[y.length];
        Arrays.fill(predictions, initialPrediction);
        
        Random rng = new Random(seed);
        
        for (int iter = 0; iter < nEstimators; iter++) {
            float[] gradients = new float[y.length];
            float[] hessians = new float[y.length];
            
            for (int i = 0; i < y.length; i++) {
                float residual = y[i] - predictions[i];
                gradients[i] = -2 * residual;
                hessians[i] = 2.0f;
            }
            
            int[] sampleIndices = subsample < 1.0f ? 
                sampleIndices(x.length, subsample, rng) : 
                allIndices(x.length);
            
            RegTree tree = buildTree(x, gradients, hessians, sampleIndices, 0, rng);
            trees.add(tree);
            
            for (int i = 0; i < y.length; i++) {
                predictions[i] += learningRate * predictTree(tree, x[i]);
            }
        }
    }
    
    public float[] predict(float[][] x) {
        ensureOpen();
        if (x == null) {
            throw new IllegalArgumentException("x must not be null");
        }
        if (trees == null) {
            throw new IllegalStateException("GradientBoostedTrees not fitted yet");
        }
        
        int n = x.length;
        float[] predictions = new float[n];
        Arrays.fill(predictions, initialPrediction);
        
        for (RegTree tree : trees) {
            for (int i = 0; i < n; i++) {
                predictions[i] += learningRate * predictTree(tree, x[i]);
            }
        }
        
        return predictions;
    }
    
    private RegTree buildTree(float[][] x, float[] gradients, float[] hessians, 
                              int[] indices, int depth, Random rng) {
        if (indices.length < minSamplesSplit || depth >= maxDepth) {
            return new RegTree(computeLeafValue(gradients, hessians, indices));
        }
        
        double bestGain = 0;
        int bestFeature = -1;
        float bestThreshold = 0;
        List<Integer> bestLeft = null;
        List<Integer> bestRight = null;
        
        float[][] binEdges = computeHistogramBinEdges(x, indices);
        
        for (int feat = 0; feat < nFeatures; feat++) {
            for (int bin = 0; bin < binEdges[feat].length; bin++) {
                float threshold = binEdges[feat][bin];
                
                List<Integer> left = new ArrayList<>();
                List<Integer> right = new ArrayList<>();
                
                for (int idx : indices) {
                    if (x[idx][feat] <= threshold) {
                        left.add(idx);
                    } else {
                        right.add(idx);
                    }
                }
                
                if (left.isEmpty() || right.isEmpty()) {
                    continue;
                }
                
                double gain = computeSplitGain(gradients, hessians, left, right);
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = feat;
                    bestThreshold = threshold;
                    bestLeft = left;
                    bestRight = right;
                }
            }
        }
        
        if (bestFeature == -1 || bestGain <= 0) {
            return new RegTree(computeLeafValue(gradients, hessians, indices));
        }
        
        RegTree node = new RegTree(bestFeature, bestThreshold);
        node.left = buildTree(x, gradients, hessians, 
                             bestLeft.stream().mapToInt(i -> i).toArray(), depth + 1, rng);
        node.right = buildTree(x, gradients, hessians, 
                              bestRight.stream().mapToInt(i -> i).toArray(), depth + 1, rng);
        
        return node;
    }
    
    private float[][] computeHistogramBinEdges(float[][] x, int[] indices) {
        float[][] binEdges = new float[nFeatures][];
        
        for (int feat = 0; feat < nFeatures; feat++) {
            float minVal = Float.MAX_VALUE;
            float maxVal = -Float.MAX_VALUE;
            
            for (int idx : indices) {
                minVal = Math.min(minVal, x[idx][feat]);
                maxVal = Math.max(maxVal, x[idx][feat]);
            }
            
            if (minVal == maxVal) {
                binEdges[feat] = new float[]{minVal};
            } else {
                int nBins = Math.min(maxBins, indices.length);
                float step = (maxVal - minVal) / nBins;
                binEdges[feat] = new float[nBins - 1];
                for (int b = 0; b < nBins - 1; b++) {
                    binEdges[feat][b] = minVal + (b + 1) * step;
                }
            }
        }
        
        return binEdges;
    }
    
    private double computeSplitGain(float[] gradients, float[] hessians,
                                   List<Integer> left, List<Integer> right) {
        double leftGrad = 0, leftHess = 0;
        for (int idx : left) {
            leftGrad += gradients[idx];
            leftHess += hessians[idx];
        }
        
        double rightGrad = 0, rightHess = 0;
        for (int idx : right) {
            rightGrad += gradients[idx];
            rightHess += hessians[idx];
        }
        
        double lambda = 1.0;
        
        double leftScore = (leftGrad * leftGrad) / (leftHess + lambda);
        double rightScore = (rightGrad * rightGrad) / (rightHess + lambda);
        double parentScore = ((leftGrad + rightGrad) * (leftGrad + rightGrad)) / 
                            (leftHess + rightHess + lambda);
        
        return 0.5 * (leftScore + rightScore - parentScore);
    }
    
    private float computeLeafValue(float[] gradients, float[] hessians, int[] indices) {
        double sumGrad = 0, sumHess = 0;
        for (int idx : indices) {
            sumGrad += gradients[idx];
            sumHess += hessians[idx];
        }
        
        double lambda = 1.0;
        return (float) (-sumGrad / (sumHess + lambda));
    }
    
    private float predictTree(RegTree node, float[] x) {
        if (node.isLeaf) {
            return node.value;
        }
        
        if (x[node.feature] <= node.threshold) {
            return predictTree(node.left, x);
        } else {
            return predictTree(node.right, x);
        }
    }
    
    private float mean(float[] arr) {
        double sum = 0;
        for (float v : arr) {
            sum += v;
        }
        return (float) (sum / arr.length);
    }
    
    private int[] sampleIndices(int n, float rate, Random rng) {
        int sampleSize = Math.max(1, (int) (n * rate));
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, rng);
        return indices.subList(0, sampleSize).stream().mapToInt(i -> i).toArray();
    }
    
    private int[] allIndices(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        return indices;
    }
    
    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GradientBoostedTrees has been closed");
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
    
    private static final class RegTree {
        final int feature;
        final float threshold;
        final float value;
        final boolean isLeaf;
        
        RegTree left;
        RegTree right;
        
        RegTree(float value) {
            this.feature = -1;
            this.threshold = 0;
            this.value = value;
            this.isLeaf = true;
        }
        
        RegTree(int feature, float threshold) {
            this.feature = feature;
            this.threshold = threshold;
            this.value = 0;
            this.isLeaf = false;
        }
    }
}
