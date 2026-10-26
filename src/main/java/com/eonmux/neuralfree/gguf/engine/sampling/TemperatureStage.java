package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's temperature sampler (llama_sampler_temp_apply): every surviving logit is divided by
 * {@code temperature}, which the final dist stage softmaxes over. A temperature {@code <= 0} applies nothing: the
 * chain then ends in the argmax (a greedy configuration), or in the diffusion chain it draws straight from the
 * unscaled logits, exactly as the fixed chain behaved. A scale of 1 divides by 1 exactly, so applying it is a no-op
 * that leaves the logits unchanged.
 */
public final class TemperatureStage extends SamplerStage {

    private final float temperature;

    public TemperatureStage(float temperature) {
        this.temperature = temperature;
    }

    @Override
    public StageKind kind() {
        return StageKind.TEMPERATURE;
    }

    @Override
    public void apply(Candidates candidates) {
        if (temperature <= 0f) {
            return;
        }
        for (int i = 0; i < candidates.size(); i++) {
            candidates.setLogit(i, candidates.logit(i) / temperature);
        }
    }
}
