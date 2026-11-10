package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.engine.sampling.TokenGate;

import java.util.List;

/**
 * Which greedy chains {@link Sampler#sample} may run sample-first ({@link Sampler#sampleFirst}): those whose argmax
 * over the unmasked row, once every constraint-head stage keeps it, is the argmax of the masked row.
 */
final class SampleFirstRules {

    private SampleFirstRules() {
    }

    /**
     * Whether a greedy chain with constraint head {@code constraints} and configured stages {@code chain} qualifies:
     * the head is not empty and its every stage answers per token ({@link TokenGate}), and the configured stages
     * are, at this step, per-token edits — the penalties and DRY, the stages greedy decoding skips, and stages that
     * apply nothing (typical-p at 1, {@link SamplerStage#modifies} false). A stage that reads the whole candidate
     * set (typical-p, top-n-sigma, XTC, infill, a terminal selector, a logits processor, a forcing reasoning
     * budget) keeps the whole-row mask, as does every sampled chain, whose truncation and draw run over the
     * admitted set.
     */
    static boolean qualifies(List<SamplerStage> constraints, List<SamplerStage> chain, float typicalP) {
        if (constraints.isEmpty()) {
            return false;
        }
        for (SamplerStage stage : constraints) {
            if (!(stage instanceof TokenGate)) {
                return false;
            }
        }
        for (SamplerStage stage : chain) {
            StageKind kind = stage.kind();
            if (stage.constrains() || stage.masks() || skippedWhenGreedy(kind, typicalP)
                    || kind == StageKind.PENALTIES || kind == StageKind.DRY
                    || kind == StageKind.TYPICAL_P && typicalP >= 1f) {
                continue;
            }
            if (kind == StageKind.MIROSTAT || kind == StageKind.ADAPTIVE_P || stage.modifies()) {
                return false;
            }
        }
        return true;
    }

    /** Whether a greedy chain skips stages of {@code kind}: temperature, and the argmax-keeping truncations. */
    static boolean skippedWhenGreedy(StageKind kind, float typicalP) {
        return kind == StageKind.TEMPERATURE || typicalP >= 1f
                && (kind == StageKind.TOP_K || kind == StageKind.TOP_P || kind == StageKind.MIN_P);
    }
}
