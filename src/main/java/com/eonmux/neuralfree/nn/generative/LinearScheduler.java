package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

public final class LinearScheduler implements NoiseScheduler {

    private final int numTimesteps;
    private final float betaStart;
    private final float betaEnd;
    private final float[] betas;
    private final float[] alphas;
    private final float[] alphasCumprod;
    private final float[] alphasCumprodPrev;
    private final float[] posteriorVariance;
    private final ComputeBackend backend;

    public LinearScheduler(ComputeBackend backend, int numTimesteps) {
        this(backend, numTimesteps, 1e-4f, 0.02f);
    }

    public LinearScheduler(ComputeBackend backend, int numTimesteps, float betaStart, float betaEnd) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        if (numTimesteps <= 0) {
            throw new IllegalArgumentException("numTimesteps must be positive, got: " + numTimesteps);
        }
        if (betaStart <= 0 || betaStart >= 1) {
            throw new IllegalArgumentException("betaStart must be in (0, 1), got: " + betaStart);
        }
        if (betaEnd <= betaStart || betaEnd >= 1) {
            throw new IllegalArgumentException("betaEnd must be in (betaStart, 1), got: " + betaEnd);
        }
        this.backend = backend;
        this.numTimesteps = numTimesteps;
        this.betaStart = betaStart;
        this.betaEnd = betaEnd;

        this.betas = NoiseScheduler.linearSchedule(numTimesteps, betaStart, betaEnd);

        this.alphas = new float[numTimesteps];
        this.alphasCumprod = new float[numTimesteps];
        this.alphasCumprodPrev = new float[numTimesteps];
        this.posteriorVariance = new float[numTimesteps];

        float cumulativeProd = 1.0f;
        for (int i = 0; i < numTimesteps; i++) {
            this.alphas[i] = 1.0f - this.betas[i];
            cumulativeProd *= this.alphas[i];
            this.alphasCumprod[i] = cumulativeProd;
            this.alphasCumprodPrev[i] = i > 0 ? this.alphasCumprod[i - 1] : 1.0f;

            this.posteriorVariance[i] = this.betas[i] * (1.0f - this.alphasCumprodPrev[i])
                / (1.0f - this.alphasCumprod[i]);
        }
    }

    @Override
    public int numTimesteps() {
        return numTimesteps;
    }

    @Override
    public float[] getBetas() {
        return betas.clone();
    }

    @Override
    public float[] getAlphas() {
        return alphas.clone();
    }

    @Override
    public float[] getAlphasCumprod() {
        return alphasCumprod.clone();
    }

    @Override
    public float getAlphaCumprod(int timestep) {
        validateTimestep(timestep);
        return alphasCumprod[timestep];
    }

    @Override
    public float getAlphaCumprodPrev(int timestep) {
        validateTimestep(timestep);
        return alphasCumprodPrev[timestep];
    }

    @Override
    public float getBeta(int timestep) {
        validateTimestep(timestep);
        return betas[timestep];
    }

    @Override
    public float getNoiseLevel(int timestep) {
        validateTimestep(timestep);
        return (float) Math.sqrt(1.0 - alphasCumprod[timestep]);
    }

    @Override
    public Tensor addNoise(Tensor xStart, Tensor noise, int timestep) {
        Objects.requireNonNull(xStart, "xStart cannot be null");
        Objects.requireNonNull(noise, "noise cannot be null");
        validateTimestep(timestep);

        float sqrtAlphaCumprod = (float) Math.sqrt(alphasCumprod[timestep]);
        float sqrtOneMinusAlphaCumprod = (float) Math.sqrt(1.0 - alphasCumprod[timestep]);

        Tensor scaledStart = TensorOps.mul(xStart, sqrtAlphaCumprod);
        Tensor scaledNoise = TensorOps.mul(noise, sqrtOneMinusAlphaCumprod);
        Tensor result = TensorOps.add(scaledStart, scaledNoise);

        scaledStart.close();
        scaledNoise.close();

        return result;
    }

    @Override
    public Tensor predictStartFromNoise(Tensor xNoisy, int timestep, Tensor predictedNoise) {
        Objects.requireNonNull(xNoisy, "xNoisy cannot be null");
        Objects.requireNonNull(predictedNoise, "predictedNoise cannot be null");
        validateTimestep(timestep);

        float alphaCumprod = alphasCumprod[timestep];
        float sqrtAlphaCumprod = (float) Math.sqrt(alphaCumprod);
        float sqrtOneMinusAlphaCumprod = (float) Math.sqrt(1.0 - alphaCumprod);

        Tensor scaledXNoisy = TensorOps.mul(xNoisy, 1.0f / sqrtAlphaCumprod);
        Tensor scaledNoise = TensorOps.mul(predictedNoise, sqrtOneMinusAlphaCumprod / sqrtAlphaCumprod);
        Tensor result = TensorOps.sub(scaledXNoisy, scaledNoise);

        scaledXNoisy.close();
        scaledNoise.close();

        return result;
    }

    @Override
    public float[] getPosteriorVariance() {
        return posteriorVariance.clone();
    }

    @Override
    public float getPosteriorVariance(int timestep) {
        validateTimestep(timestep);
        return posteriorVariance[timestep];
    }

    @Override
    public String name() {
        return "LinearScheduler";
    }

    public float betaStart() {
        return betaStart;
    }

    public float betaEnd() {
        return betaEnd;
    }

    public ComputeBackend backend() {
        return backend;
    }

    private void validateTimestep(int timestep) {
        if (timestep < 0 || timestep >= numTimesteps) {
            throw new IllegalArgumentException(
                "timestep must be in [0, " + (numTimesteps - 1) + "], got: " + timestep);
        }
    }

    @Override
    public String toString() {
        return "LinearScheduler{numTimesteps=" + numTimesteps
            + ", betaStart=" + betaStart
            + ", betaEnd=" + betaEnd + "}";
    }
}
