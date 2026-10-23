package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Arrays;
import java.util.Random;

/**
 * Discrete hidden Markov model: Baum-Welch (expectation-maximization) training over observation sequences with
 * scaled forward-backward passes, Viterbi decoding ({@link #predict}) and the log-likelihood of a sequence
 * ({@link #score}). Parameters start from a fixed-seed random stochastic initialization.
 *
 * <p><b>CPU only</b>: every pass is a sequential recursion over time steps whose per-step work is
 * one {@code nStates x nStates} product, so a device version is latency-bound per step, and the train kernel library
 * has no formulation of it. The {@link ComputeBackend} constructor parameter is kept for API compatibility; it must
 * be non-null and is not used.</p>
 */
public final class HiddenMarkovModel implements AutoCloseable {
    
    private final int nStates;
    private final int nObservations;
    private final int maxIterations;
    private final float tolerance;
    
    private float[][] transition;
    private float[][] emission;
    private float[] initial;
    private boolean closed;
    private boolean fitted;
    
    public HiddenMarkovModel(ComputeBackend backend, int nStates, int nObservations,
                             int maxIterations, float tolerance) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (nStates < 1) {
            throw new IllegalArgumentException("nStates must be at least 1");
        }
        if (nObservations < 1) {
            throw new IllegalArgumentException("nObservations must be at least 1");
        }
        if (maxIterations < 0) {
            throw new IllegalArgumentException("maxIterations must not be negative");
        }
        if (!(tolerance >= 0.0f)) {
            throw new IllegalArgumentException("tolerance must be a non-negative number");
        }
        
        this.nStates = nStates;
        this.nObservations = nObservations;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.closed = false;
        this.fitted = false;
        
        initializeParameters();
    }
    
    public HiddenMarkovModel(ComputeBackend backend, int nStates, int nObservations) {
        this(backend, nStates, nObservations, 100, 1e-6f);
    }
    
    private void initializeParameters() {
        Random rng = new Random(42);
        
        transition = new float[nStates][nStates];
        for (int i = 0; i < nStates; i++) {
            float sum = 0.0f;
            for (int j = 0; j < nStates; j++) {
                transition[i][j] = rng.nextFloat() + 0.1f;
                sum += transition[i][j];
            }
            for (int j = 0; j < nStates; j++) {
                transition[i][j] /= sum;
            }
        }
        
        emission = new float[nStates][nObservations];
        for (int i = 0; i < nStates; i++) {
            float sum = 0.0f;
            for (int o = 0; o < nObservations; o++) {
                emission[i][o] = rng.nextFloat() + 0.1f;
                sum += emission[i][o];
            }
            for (int o = 0; o < nObservations; o++) {
                emission[i][o] /= sum;
            }
        }
        
        initial = new float[nStates];
        float sum = 0.0f;
        for (int i = 0; i < nStates; i++) {
            initial[i] = rng.nextFloat() + 0.1f;
            sum += initial[i];
        }
        for (int i = 0; i < nStates; i++) {
            initial[i] /= sum;
        }
    }
    
    public void fit(int[][] sequences) {
        ensureOpen();
        if (sequences == null || sequences.length == 0) {
            throw new IllegalArgumentException("sequences must not be null or empty");
        }
        for (int s = 0; s < sequences.length; s++) {
            validateSymbols(sequences[s], "sequences[" + s + "]");
        }
        
        float prevLogLikelihood = Float.NEGATIVE_INFINITY;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            float totalLogLikelihood = 0.0f;
            
            float[][] newTransitionNum = new float[nStates][nStates];
            float[] newTransitionDenom = new float[nStates];
            float[][] newEmissionNum = new float[nStates][nObservations];
            float[] newEmissionDenom = new float[nStates];
            float[] newInitialNum = new float[nStates];
            
            for (int[] seq : sequences) {
                if (seq.length == 0) continue;
                
                ForwardBackwardResult fb = forwardBackward(seq);
                totalLogLikelihood += fb.logLikelihood;
                
                float[][] gamma = computeGamma(seq, fb);
                float[][][] xi = computeXi(seq, fb);
                
                for (int i = 0; i < nStates; i++) {
                    newInitialNum[i] += gamma[0][i];
                    newEmissionDenom[i] += gamma[seq.length - 1][i];
                    
                    for (int o = 0; o < nObservations; o++) {
                        if (seq[seq.length - 1] == o) {
                            newEmissionNum[i][o] += gamma[seq.length - 1][i];
                        }
                    }
                }
                
                for (int t = 0; t < seq.length - 1; t++) {
                    for (int i = 0; i < nStates; i++) {
                        newTransitionDenom[i] += gamma[t][i];
                        newEmissionDenom[i] += gamma[t][i];
                        
                        for (int j = 0; j < nStates; j++) {
                            newTransitionNum[i][j] += xi[t][i][j];
                        }
                        
                        for (int o = 0; o < nObservations; o++) {
                            if (seq[t] == o) {
                                newEmissionNum[i][o] += gamma[t][i];
                            }
                        }
                    }
                }
            }
            
            float initialSum = 0.0f;
            for (int i = 0; i < nStates; i++) {
                initialSum += newInitialNum[i];
            }
            for (int i = 0; i < nStates; i++) {
                initial[i] = initialSum > 0 ? newInitialNum[i] / initialSum : 1.0f / nStates;
            }
            
            for (int i = 0; i < nStates; i++) {
                for (int j = 0; j < nStates; j++) {
                    transition[i][j] = newTransitionDenom[i] > 0 ?
                        newTransitionNum[i][j] / newTransitionDenom[i] : 1.0f / nStates;
                }
            }
            
            for (int i = 0; i < nStates; i++) {
                for (int o = 0; o < nObservations; o++) {
                    emission[i][o] = newEmissionDenom[i] > 0 ?
                        newEmissionNum[i][o] / newEmissionDenom[i] : 1.0f / nObservations;
                }
            }
            
            float improvement = totalLogLikelihood - prevLogLikelihood;
            if (Math.abs(improvement) < tolerance && iter > 0) {
                break;
            }
            prevLogLikelihood = totalLogLikelihood;
        }
        
        fitted = true;
    }
    
    public int[] predict(int[] sequence) {
        ensureOpen();
        if (sequence == null || sequence.length == 0) {
            throw new IllegalArgumentException("sequence must not be null or empty");
        }
        if (!fitted) {
            throw new IllegalStateException("HMM not fitted yet");
        }
        validateSymbols(sequence, "sequence");
        
        return viterbi(sequence);
    }
    
    public float score(int[] sequence) {
        ensureOpen();
        if (sequence == null || sequence.length == 0) {
            throw new IllegalArgumentException("sequence must not be null or empty");
        }
        if (!fitted) {
            throw new IllegalStateException("HMM not fitted yet");
        }
        validateSymbols(sequence, "sequence");
        
        ForwardBackwardResult fb = forwardBackward(sequence);
        return fb.logLikelihood;
    }
    
    private ForwardBackwardResult forwardBackward(int[] sequence) {
        int T = sequence.length;
        
        float[][] alpha = new float[T][nStates];
        float[] scale = new float[T];
        
        for (int i = 0; i < nStates; i++) {
            alpha[0][i] = initial[i] * emission[i][sequence[0]];
        }
        scale[0] = normalize(alpha[0]);
        
        for (int t = 1; t < T; t++) {
            for (int j = 0; j < nStates; j++) {
                float sum = 0.0f;
                for (int i = 0; i < nStates; i++) {
                    sum += alpha[t - 1][i] * transition[i][j];
                }
                alpha[t][j] = sum * emission[j][sequence[t]];
            }
            scale[t] = normalize(alpha[t]);
        }
        
        float[][] beta = new float[T][nStates];
        Arrays.fill(beta[T - 1], 1.0f / scale[T - 1]);
        
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < nStates; i++) {
                float sum = 0.0f;
                for (int j = 0; j < nStates; j++) {
                    sum += transition[i][j] * emission[j][sequence[t + 1]] * beta[t + 1][j];
                }
                beta[t][i] = sum / scale[t];
            }
        }
        
        float logLikelihood = 0.0f;
        for (int t = 0; t < T; t++) {
            logLikelihood += Math.log(scale[t]);
        }
        
        return new ForwardBackwardResult(alpha, beta, logLikelihood);
    }
    
    private float[][] computeGamma(int[] sequence, ForwardBackwardResult fb) {
        int T = sequence.length;
        float[][] gamma = new float[T][nStates];
        
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int i = 0; i < nStates; i++) {
                gamma[t][i] = fb.alpha[t][i] * fb.beta[t][i];
                sum += gamma[t][i];
            }
            if (sum > 0) {
                for (int i = 0; i < nStates; i++) {
                    gamma[t][i] /= sum;
                }
            }
        }
        
        return gamma;
    }
    
    private float[][][] computeXi(int[] sequence, ForwardBackwardResult fb) {
        int T = sequence.length;
        float[][][] xi = new float[T - 1][nStates][nStates];
        
        for (int t = 0; t < T - 1; t++) {
            float sum = 0.0f;
            for (int i = 0; i < nStates; i++) {
                for (int j = 0; j < nStates; j++) {
                    xi[t][i][j] = fb.alpha[t][i] * transition[i][j] * 
                                  emission[j][sequence[t + 1]] * fb.beta[t + 1][j];
                    sum += xi[t][i][j];
                }
            }
            if (sum > 0) {
                for (int i = 0; i < nStates; i++) {
                    for (int j = 0; j < nStates; j++) {
                        xi[t][i][j] /= sum;
                    }
                }
            }
        }
        
        return xi;
    }
    
    private int[] viterbi(int[] sequence) {
        int T = sequence.length;
        float[][] viterbi = new float[T][nStates];
        int[][] backpointer = new int[T][nStates];
        
        for (int i = 0; i < nStates; i++) {
            viterbi[0][i] = (float) Math.log(initial[i]) + (float) Math.log(emission[i][sequence[0]]);
            backpointer[0][i] = 0;
        }
        
        for (int t = 1; t < T; t++) {
            for (int j = 0; j < nStates; j++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                int maxIdx = 0;
                
                for (int i = 0; i < nStates; i++) {
                    float val = viterbi[t - 1][i] + (float) Math.log(transition[i][j]);
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = i;
                    }
                }
                
                viterbi[t][j] = maxVal + (float) Math.log(emission[j][sequence[t]]);
                backpointer[t][j] = maxIdx;
            }
        }
        
        int[] path = new int[T];
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < nStates; i++) {
            if (viterbi[T - 1][i] > maxVal) {
                maxVal = viterbi[T - 1][i];
                path[T - 1] = i;
            }
        }
        
        for (int t = T - 2; t >= 0; t--) {
            path[t] = backpointer[t + 1][path[t + 1]];
        }
        
        return path;
    }
    
    /** Rejects a null sequence and any symbol outside {@code [0, nObservations)}, naming the offending position. */
    private void validateSymbols(int[] sequence, String label) {
        if (sequence == null) {
            throw new IllegalArgumentException(label + " must not be null");
        }
        for (int t = 0; t < sequence.length; t++) {
            int symbol = sequence[t];
            if (symbol < 0 || symbol >= nObservations) {
                throw new IllegalArgumentException(label + "[" + t + "] = " + symbol
                        + " is outside [0, " + nObservations + ")");
            }
        }
    }
    
    private float normalize(float[] arr) {
        float sum = 0.0f;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        if (sum > 0) {
            for (int i = 0; i < arr.length; i++) {
                arr[i] /= sum;
            }
        }
        return sum;
    }
    
    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("HiddenMarkovModel has been closed");
        }
    }
    
    /** Marks the model closed; later calls to {@link #fit}, {@link #predict} and {@link #score} throw. */
    @Override
    public void close() {
        closed = true;
    }
    
    private static final class ForwardBackwardResult {
        final float[][] alpha;
        final float[][] beta;
        final float logLikelihood;
        
        ForwardBackwardResult(float[][] alpha, float[][] beta, float logLikelihood) {
            this.alpha = alpha;
            this.beta = beta;
            this.logLikelihood = logLikelihood;
        }
    }
}
