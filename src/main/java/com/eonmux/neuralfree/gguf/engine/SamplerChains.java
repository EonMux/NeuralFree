package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.AdaptivePStage;
import com.eonmux.neuralfree.gguf.engine.sampling.BiasStage;
import com.eonmux.neuralfree.gguf.engine.sampling.DryStage;
import com.eonmux.neuralfree.gguf.engine.sampling.InfillStage;
import com.eonmux.neuralfree.gguf.engine.sampling.MinPStage;
import com.eonmux.neuralfree.gguf.engine.sampling.MirostatStage;
import com.eonmux.neuralfree.gguf.engine.sampling.PenaltiesStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.engine.sampling.TemperatureStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TopKStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TopNSigmaStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TopPStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TypicalStage;
import com.eonmux.neuralfree.gguf.engine.sampling.XtcStage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * The configured stage chain of a {@link Sampler}: llama.cpp's chain construction from a {@link SamplerConfig}, the
 * chain's constraint stages, and the built-in stage classes whose instances hold only their own sampler's state.
 */
final class SamplerChains {

    /**
     * The final stage classes whose instances hold only their own sampler's state
     * ({@link Sampler#samplesIndependently}).
     */
    static final Set<Class<? extends SamplerStage>> INDEPENDENT_STAGES = Set.of(TemperatureStage.class,
            MirostatStage.class, PenaltiesStage.class, DryStage.class, TopNSigmaStage.class, TopKStage.class,
            TypicalStage.class, TopPStage.class, MinPStage.class, XtcStage.class, AdaptivePStage.class,
            BiasStage.class);

    private SamplerChains() {
    }

    /** llama.cpp's chain construction (common/sampling.cpp:344-416). */
    static List<SamplerStage> build(SamplerConfig config, long seed, SamplerVocab vocab) {
        List<SamplerStage> chain = new ArrayList<>();
        if (config.mirostat() == 1 || config.mirostat() == 2) {
            // llama.cpp's mirostat chains are temperature + the mirostat sampler, nothing else
            chain.add(new TemperatureStage(config.temperature()));
            chain.add(new MirostatStage(config.mirostat(), config.mirostatTau(), config.mirostatEta(), seed));
            return chain;
        }
        boolean adaptiveP = false;
        for (StageKind kind : config.order()) {
            if (kind == StageKind.PENALTIES) {
                chain.add(new PenaltiesStage(config.repeatPenalty(), config.presencePenalty(),
                        config.frequencyPenalty(), config.penaltyLastN(), config.penaltiesEnabled()));
            } else if (kind == StageKind.DRY) {
                chain.add(new DryStage(config.dryMultiplier(), config.dryBase(), config.dryAllowedLength(),
                        config.dryPenaltyLastN(), config.drySequenceBreakers(), vocab));
            } else if (kind == StageKind.TOP_N_SIGMA) {
                chain.add(new TopNSigmaStage(config.topNSigma()));
            } else if (kind == StageKind.TOP_K) {
                chain.add(new TopKStage(config.topK(), config.minKeep()));
            } else if (kind == StageKind.TYPICAL_P) {
                chain.add(new TypicalStage(config.typicalP(), config.minKeep()));
            } else if (kind == StageKind.TOP_P) {
                chain.add(new TopPStage(config.topP(), config.minKeep()));
            } else if (kind == StageKind.MIN_P) {
                chain.add(new MinPStage(config.minP(), config.minKeep()));
            } else if (kind == StageKind.XTC) {
                chain.add(new XtcStage(config.xtcProbability(), config.xtcThreshold(), config.minKeep(),
                        stageSeed(config, seed)));
            } else if (kind == StageKind.TEMPERATURE) {
                chain.add(new TemperatureStage(config.temperature(), config.dynatempRange(), config.dynatempExponent()));
            } else if (kind == StageKind.INFILL) {
                chain.add(new InfillStage(vocab));
            } else if (kind == StageKind.ADAPTIVE_P) {
                // llama.cpp keeps the adaptive-p position in the order but builds it last, in place of the dist stage
                adaptiveP = true;
            }
        }
        if (adaptiveP) {
            chain.add(new AdaptivePStage(config.adaptiveTarget(), config.adaptiveDecay(), stageSeed(config, seed)));
        }
        return chain;
    }

    /** llama.cpp's per-sampler {@code get_rng_seed(seed)}: every stage draws its own seed unless the seed is fixed. */
    private static long stageSeed(SamplerConfig config, long seed) {
        return config.seeded() ? seed : Sampler.randomSeed();
    }

    /**
     * The chain's constraint stages ({@link SamplerStage#constrains}, {@link SamplerStage#masks}) in chain
     * order; empty when it has none.
     */
    static List<SamplerStage> constraintsOf(List<SamplerStage> chain) {
        List<SamplerStage> constraints = new ArrayList<>(1);
        for (SamplerStage stage : chain) {
            if (stage.constrains() || stage.masks()) {
                constraints.add(stage);
            }
        }
        return constraints.isEmpty() ? List.of() : Collections.unmodifiableList(constraints);
    }
}
