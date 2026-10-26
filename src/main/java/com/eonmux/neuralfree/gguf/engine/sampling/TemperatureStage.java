package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's temperature sampler (llama_sampler_temp_apply) and with a {@code dynatempRange} above 0 its dynamic
 * extension (llama_sampler_temp_ext_apply, llama.cpp src/llama-sampler.cpp:2139, the CLI {@code --dynatemp-range} and
 * {@code --dynatemp-exponent}): every surviving logit is divided by a temperature, which the final dist stage softmaxes
 * over. The plain sampler divides by {@code temperature}; the dynamic one softmaxes over the sorted set (doubles for
 * the re-normalized probabilities), measures its normalized entropy, maps it into the
 * {@code [max(0, temperature - dynatempRange), temperature + dynatempRange]} window by
 * {@code dyn_temp = min + (max - min) * normalizedEntropy^exponent} and scales by that (a result of 0 masks everything
 * but the first maximum, as llama_sampler_temp_impl's temperature of 0 does). A temperature {@code <= 0} applies
 * nothing: the chain then ends in the argmax (a greedy configuration), or in the diffusion chain it draws straight
 * from the unscaled logits, exactly as the fixed chain behaved. A scale of 1 divides by 1 exactly, so applying it is a
 * no-op that leaves the logits unchanged.
 */
public final class TemperatureStage extends SamplerStage {

    private final float temperature;
    private final float dynatempRange;
    private final float dynatempExponent;

    /** llama.cpp {@code llama_sampler_init_temp}: plain scaling. */
    public TemperatureStage(float temperature) {
        this(temperature, 0.0f, 1.0f);
    }

    /** llama.cpp {@code llama_sampler_init_temp_ext}: {@code dynatempRange > 0} turns the sampler dynamic. */
    public TemperatureStage(float temperature, float dynatempRange, float dynatempExponent) {
        this.temperature = temperature;
        this.dynatempRange = dynatempRange;
        this.dynatempExponent = dynatempExponent;
    }

    @Override
    public StageKind kind() {
        return StageKind.TEMPERATURE;
    }

    @Override
    public void apply(Candidates candidates) {
        if (dynatempRange > 0f) {
            applyDynamic(candidates);
            return;
        }
        if (temperature <= 0f) {
            return;
        }
        for (int i = 0; i < candidates.size(); i++) {
            candidates.setLogit(i, candidates.logit(i) / temperature);
        }
    }

    /** llama_sampler_temp_ext_apply's dynamic branch (llama-sampler.cpp:2141-2207). */
    private void applyDynamic(Candidates candidates) {
        float minTemp = Math.max(0.0f, temperature - dynatempRange);
        float maxTemp = temperature + dynatempRange;

        // no need to do anything if there is only one (or zero) candidates
        if (candidates.size() <= 1) {
            return;
        }

        // Calculate maximum possible entropy
        float maxEntropy = (float) -Math.log(1.0f / candidates.size());

        candidates.softmaxSorted();

        // Calculate entropy of the softmax probabilities
        float entropy = 0.0f;
        for (int i = 0; i < candidates.size(); i++) {
            float prob = candidates.prob(i);
            if (prob > 0.0f) { // Ensure no log(0)
                entropy -= prob * (float) Math.log(prob);
            }
        }

        // Map the normalized entropy to the desired temperature range using the power function
        float normalizedEntropy = entropy / maxEntropy;
        float dynTemp = minTemp + (maxTemp - minTemp)
                * (float) Math.pow(normalizedEntropy, dynatempExponent);

        llamaTempImpl(candidates, dynTemp);

        // Re-compute softmax probabilities after scaling logits with dynamic temperature (in double)
        double maxL = candidates.logit(0);
        double cumSum = 0.0;
        for (int i = 0; i < candidates.size(); i++) {
            double p = Math.exp(candidates.logit(i) - maxL);
            candidates.setProb(i, (float) p); // Store the scaled probability
            cumSum += p;
        }
        for (int i = 0; i < candidates.size(); i++) {
            candidates.setProb(i, (float) (candidates.prob(i) / cumSum)); // Re-normalize
        }
    }

    /** llama_sampler_temp_impl (llama-sampler.cpp:265): a temperature of 0 or below keeps only the first maximum. */
    private static void llamaTempImpl(Candidates candidates, float temp) {
        int size = candidates.size();
        if (size == 0) {
            return;
        }
        if (temp <= 0.0f) {
            // find the token with the highest logit and set the rest to -inf
            int maxIndex = 0;
            float maxLogit = candidates.logit(0);
            for (int i = 1; i < size; i++) {
                if (candidates.logit(i) > maxLogit) {
                    candidates.setLogit(maxIndex, Float.NEGATIVE_INFINITY);
                    maxIndex = i;
                    maxLogit = candidates.logit(i);
                } else {
                    candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                }
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            candidates.setLogit(i, candidates.logit(i) / temp);
        }
    }
}
