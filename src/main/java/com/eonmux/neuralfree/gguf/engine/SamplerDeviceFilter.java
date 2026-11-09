package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.BiasStage;
import com.eonmux.neuralfree.gguf.engine.sampling.DryStage;
import com.eonmux.neuralfree.gguf.engine.sampling.PenaltiesStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A chain's device prefilter plan: which of its stages a pre-top-k device readout reproduces
 * ({@link LogitPrefilter}), the cone width that readout needs, and the row it uploads for one output row. The
 * plan reads the chain in place, so stages inserted into it later ({@code Sampler#insertExtras},
 * {@code Sampler#insertLogitsProcessors}) are seen after a {@link #refresh()}.
 *
 * <p>Owned by the sampler that built it; not thread-safe.</p>
 */
final class SamplerDeviceFilter {

    private final Sampler sampler;
    private final List<SamplerStage> chain;
    private final SamplerStage biasStage;
    private final int[] banned;
    private final int coneExtra;
    /** Whether a logit bias raises a token, so no device cone contains the chain's keeps. */
    private final boolean positiveBias;
    private final SamplerConfig config;

    private SamplerStage prefilteredPenalties;
    private SamplerStage prefilteredDry;

    SamplerDeviceFilter(Sampler sampler, List<SamplerStage> chain, SamplerStage biasStage, int[] banned,
                        int coneExtra, boolean positiveBias, SamplerConfig config) {
        this.sampler = sampler;
        this.chain = chain;
        this.biasStage = biasStage;
        this.banned = banned;
        this.coneExtra = coneExtra;
        this.positiveBias = positiveBias;
        this.config = config;
    }

    /** The chain's penalties stage the device prefilter reproduces, or null. */
    SamplerStage prefilteredPenalties() {
        return prefilteredPenalties;
    }

    /** The chain's DRY stage the device prefilter reproduces, or null. */
    SamplerStage prefilteredDry() {
        return prefilteredDry;
    }

    /**
     * Selects the chain's penalties and DRY stages a device prefilter reproduces before the readout: a stage
     * applies there only when it is the chain's one stage of its kind and precedes the top-k stage. A stage after
     * the top-k, or a repeated one, stays on the host and runs over the readout candidates.
     */
    void refresh() {
        prefilteredPenalties = null;
        prefilteredDry = null;
        int topKAt = topKIndex();
        if (topKAt < 0) {
            return;
        }
        int penaltiesAt = -1;
        int dryAt = -1;
        int penaltiesCount = 0;
        int dryCount = 0;
        for (int i = 0; i < chain.size(); i++) {
            StageKind kind = chain.get(i).kind();
            if (kind == StageKind.PENALTIES) {
                penaltiesCount++;
                if (penaltiesAt < 0) {
                    penaltiesAt = i;
                }
            } else if (kind == StageKind.DRY) {
                dryCount++;
                if (dryAt < 0) {
                    dryAt = i;
                }
            }
        }
        if (penaltiesCount == 1 && penaltiesAt < topKAt) {
            prefilteredPenalties = chain.get(penaltiesAt);
        }
        if (dryCount == 1 && dryAt < topKAt) {
            prefilteredDry = chain.get(dryAt);
        }
    }

    /** Index of the chain's first top-k stage, or -1. */
    private int topKIndex() {
        for (int i = 0; i < chain.size(); i++) {
            if (chain.get(i).kind() == StageKind.TOP_K) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Width of the chain's cone over a row's largest logits ({@code Sampler#coneWidth}): every stage that can
     * keep a candidate outside the cone refuses it, and without a top-k stage the cone is the largest {@code
     * coneExtra + 1} raw logits when the chain selects the argmax ({@link Sampler#argmaxChain()}).
     */
    int coneWidth() {
        if (positiveBias || sampler.hasConstraints()) {
            return -1;
        }
        if (config.topK() < 1) {
            return sampler.argmaxChain() ? coneExtra + 1 : -1;
        }
        int topKAt = topKIndex();
        if (topKAt < 0) {
            return -1;
        }
        for (int i = 0; i < chain.size(); i++) {
            SamplerStage stage = chain.get(i);
            StageKind kind = stage.kind();
            if (kind == null) {
                // a null-kind request stage (grammar, min-tokens, bad-words, reasoning-budget): only one that
                // leaves the chain's set alone keeps the cone
                if (stage.modifies()) {
                    return -1;
                }
                continue;
            }
            boolean beforeTopK = i < topKAt;
            switch (kind) {
                case TOP_K:
                    break;
                case TOP_P:
                    if (beforeTopK && config.topP() < 1f) {
                        return -1;
                    }
                    break;
                case MIN_P:
                    if (beforeTopK && config.minP() > 0f) {
                        return -1;
                    }
                    break;
                case TYPICAL_P:
                    if (beforeTopK && config.typicalP() < 1f) {
                        return -1;
                    }
                    break;
                case TOP_N_SIGMA:
                    if (beforeTopK && config.topNSigma() > 0f) {
                        return -1;
                    }
                    break;
                case XTC:
                    if (beforeTopK && config.xtcProbability() > 0f && config.xtcThreshold() <= 0.5f) {
                        return -1;
                    }
                    break;
                case TEMPERATURE:
                    if (beforeTopK && config.dynatempRange() > 0f) {
                        return -1;
                    }
                    break;
                case PENALTIES:
                    if (config.penaltiesEnabled()) {
                        return -1;
                    }
                    break;
                case DRY:
                    if (config.dryMultiplier() != 0f) {
                        return -1;
                    }
                    break;
                case MIROSTAT:
                case ADAPTIVE_P:
                case INFILL:
                    return -1;
                default:
                    break;
            }
        }
        return Math.max(config.topK(), config.minKeep()) + coneExtra;
    }

    /**
     * Width of the chain's cone over a device readout that first applies {@link #row(int[])} to the whole row, or
     * -1 when such a readout serves the chain no better than the whole row.
     */
    int deviceConeWidth() {
        if (config.topK() < 1 || sampler.hasConstraints()) {
            return -1;
        }
        int topKAt = topKIndex();
        if (topKAt < 0) {
            return -1;
        }
        for (int i = 0; i < chain.size(); i++) {
            SamplerStage stage = chain.get(i);
            StageKind kind = stage.kind();
            if (kind == null) {
                if (stage.modifies()) {
                    return -1;
                }
                continue;
            }
            boolean beforeTopK = i < topKAt;
            switch (kind) {
                case TOP_K:
                    break;
                case PENALTIES:
                    if (beforeTopK && config.penaltiesEnabled() && stage != prefilteredPenalties) {
                        return -1; // a pre-top-k penalties stage the prefilter row does not carry
                    }
                    break;
                case DRY:
                    if (beforeTopK && stage.modifies() && stage != prefilteredDry) {
                        return -1; // a pre-top-k DRY stage the prefilter row does not carry
                    }
                    break;
                case TOP_P:
                    if (beforeTopK && config.topP() < 1f) {
                        return -1;
                    }
                    break;
                case MIN_P:
                    if (beforeTopK && config.minP() > 0f) {
                        return -1;
                    }
                    break;
                case TYPICAL_P:
                    if (beforeTopK && config.typicalP() < 1f) {
                        return -1;
                    }
                    break;
                case TOP_N_SIGMA:
                    if (beforeTopK && config.topNSigma() > 0f) {
                        return -1;
                    }
                    break;
                case XTC:
                    if (beforeTopK && config.xtcProbability() > 0f && config.xtcThreshold() <= 0.5f) {
                        return -1;
                    }
                    break;
                case TEMPERATURE:
                    if (beforeTopK && config.dynatempRange() > 0f) {
                        return -1;
                    }
                    break;
                case MIROSTAT:
                case ADAPTIVE_P:
                case INFILL:
                    return -1;
                default:
                    break;
            }
        }
        if (prefilteredPenalties != null && config.penaltiesEnabled()) {
            if (!(prefilteredPenalties instanceof PenaltiesStage)) {
                return -1;
            }
            if (((PenaltiesStage) prefilteredPenalties).activeTokens().length
                    > Sampler.MAX_PREFILTER_PENALTY_TOKENS) {
                return -1;
            }
        }
        if (prefilteredDry != null && prefilteredDry.modifies()) {
            if (!(prefilteredDry instanceof DryStage)) {
                return -1;
            }
            DryStage dry = (DryStage) prefilteredDry;
            if (dry.windowSize() > Sampler.MAX_PREFILTER_DRY_WINDOW) {
                return -1;
            }
            if (dry.breakerCount() > Sampler.MAX_PREFILTER_DRY_BREAKERS
                    || dry.breakerTailCount() > Sampler.MAX_PREFILTER_DRY_TAIL_IDS) {
                return -1;
            }
        }
        return Math.max(config.topK(), config.minKeep());
    }

    /** Whether the chain's cone needs a device prefilter (a bias, penalties or DRY) rather than the plain cone. */
    boolean required() {
        return coneExtra > 0 || config.penaltiesEnabled() || config.dryMultiplier() != 0f;
    }

    /**
     * The pre-top-k edits this chain's device readout applies to one row: the fixed logit-bias table, the
     * penalties and DRY that precede the top-k stage over the row's current history, and their chain order. A
     * chain with no such edit yields an empty row.
     */
    LogitPrefilter.Row row() {
        return row(null);
    }

    /**
     * {@link #row()} with {@code pendingTokens} accepted first: the history a completing prefill will hold after
     * the forward (the batch engine accepts a prompt's tokens once its last chunk has sampled). The chain's own
     * state is unchanged.
     */
    LogitPrefilter.Row row(int[] pendingTokens) {
        LogitPrefilter.Row.Builder row = LogitPrefilter.Row.builder();
        buildBiasRow(row);
        boolean penaltiesApplied = prefilteredPenalties instanceof PenaltiesStage && config.penaltiesEnabled();
        boolean dryApplied = prefilteredDry instanceof DryStage && prefilteredDry.modifies();
        if (pendingTokens == null || pendingTokens.length == 0 || !penaltiesApplied && !dryApplied) {
            return buildPrefilterRow(row, penaltiesApplied, dryApplied);
        }
        Object state = sampler.saveState();
        try {
            sampler.accept(pendingTokens, 0, pendingTokens.length);
            return buildPrefilterRow(row, penaltiesApplied, dryApplied);
        } finally {
            sampler.restoreState(state);
        }
    }

    /** Fills {@code row} with the fixed bias table. */
    private void buildBiasRow(LogitPrefilter.Row.Builder row) {
        if (coneExtra == 0) {
            return;
        }
        LinkedHashMap<Integer, Float> table = new LinkedHashMap<>();
        for (int id : banned) {
            if (id >= 0) {
                table.put(id, Float.NaN);
            }
        }
        BiasStage bias = (BiasStage) biasStage;
        int[] ids = bias.biasedIds();
        float[] values = bias.biases();
        for (int i = 0; i < ids.length; i++) {
            int id = ids[i];
            if (id < 0) {
                continue;
            }
            Float current = table.get(id);
            if (current != null && current.isNaN()) {
                continue; // banned tokens keep their -inf, the additive bias never reaches them
            }
            table.put(id, values[i] <= -100f ? Float.NaN : Float.valueOf(values[i]));
        }
        int[] outIds = new int[table.size()];
        float[] outValues = new float[outIds.length];
        int n = 0;
        for (Map.Entry<Integer, Float> e : table.entrySet()) {
            outIds[n] = e.getKey();
            outValues[n] = e.getValue();
            n++;
        }
        row.bias(outIds, outValues);
    }

    /** Fills {@code row} with the penalties and DRY stages the device readout reproduces. */
    private LogitPrefilter.Row buildPrefilterRow(LogitPrefilter.Row.Builder row, boolean penaltiesApplied,
                                                 boolean dryApplied) {
        boolean penaltiesFirst = true;
        if (penaltiesApplied) {
            PenaltiesStage stage = (PenaltiesStage) prefilteredPenalties;
            row.penalties(true, stage.repeatPenalty(), stage.frequencyPenalty(), stage.presencePenalty(),
                    stage.activeTokens(), stage.activeCounts());
        }
        if (dryApplied) {
            DryStage stage = (DryStage) prefilteredDry;
            row.dry(true, stage.multiplier(), stage.base(), stage.allowedLength(), stage.windowTokens(),
                    stage.breakerEntries());
        }
        if (penaltiesApplied && dryApplied) {
            penaltiesFirst = chain.indexOf(prefilteredPenalties) < chain.indexOf(prefilteredDry);
        }
        return row.penaltiesFirst(penaltiesFirst).build();
    }
}
