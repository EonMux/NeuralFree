package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.tensor.Tensor;

public interface NoiseScheduler {

    int numTimesteps();

    float[] getBetas();

    float[] getAlphas();

    float[] getAlphasCumprod();

    float getAlphaCumprod(int timestep);

    float getAlphaCumprodPrev(int timestep);

    float getBeta(int timestep);

    float getNoiseLevel(int timestep);

    Tensor addNoise(Tensor xStart, Tensor noise, int timestep);

    Tensor predictStartFromNoise(Tensor xNoisy, int timestep, Tensor predictedNoise);

    float[] getPosteriorVariance();

    float getPosteriorVariance(int timestep);

    String name();

    static float[] linearSchedule(int numTimesteps, float betaStart, float betaEnd) {
        float[] betas = new float[numTimesteps];
        float step = (betaEnd - betaStart) / (numTimesteps - 1);
        for (int i = 0; i < numTimesteps; i++) {
            betas[i] = betaStart + i * step;
        }
        return betas;
    }

    static float[] cosineSchedule(int numTimesteps, float s) {
        float[] betas = new float[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) {
            float t = (float) i / numTimesteps;
            float fT = (float) Math.cos((t + s) / (1.0f + s) * Math.PI / 2.0f);
            float fTm1 = i > 0
                ? (float) Math.cos(((float) (i - 1) / numTimesteps + s) / (1.0f + s) * Math.PI / 2.0f)
                : 1.0f;
            float alphaBar = fT * fT;
            float alphaBarPrev = fTm1 * fTm1;
            betas[i] = Math.min(1.0f - alphaBar / alphaBarPrev, 0.999f);
        }
        return betas;
    }

    static float[] squaredCosineSchedule(int numTimesteps) {
        float[] betas = new float[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) {
            float t = (float) i / numTimesteps;
            float alphaBar = (float) Math.pow(Math.cos(t * Math.PI / 2.0f), 2);
            float alphaBarPrev = i > 0
                ? (float) Math.pow(Math.cos(((float) (i - 1) / numTimesteps) * Math.PI / 2.0f), 2)
                : 1.0f;
            betas[i] = Math.min(1.0f - alphaBar / alphaBarPrev, 0.999f);
        }
        return betas;
    }
}
