package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.SamplerConfig;

/**
 * The status of one sequence slot ({@code BatchEngine.slotStatuses}, the data of llama.cpp's {@code GET /slots}):
 * the slot's id and context, whether the engine speculates and whether a sequence runs in the slot, and — while
 * the slot's request runs or once it left ({@code task_prev} of llama.cpp, recorded by
 * {@code BatchState.recordTask}) — the request's id, counters, sampler parameters and tokens. Immutable; token
 * arrays copy on the way in and out.
 */
public final class SlotStatus {

    private final int id;
    private final int nCtx;
    private final boolean processing;
    private final boolean speculative;
    /** The running or last request's engine id ({@code BatchSequence.submitId}), -1 when the slot holds none. */
    private final int idTask;
    private final int nPromptTokens;
    private final int nPromptTokensProcessed;
    private final int nPromptTokensCache;
    private final SamplerConfig params;
    private final int maxNewTokens;
    private final boolean hasNextToken;
    private final int nRemain;
    private final int nDecoded;
    private final int[] promptTokens;
    private final int[] generatedTokens;

    SlotStatus(int id, int nCtx, boolean processing, boolean speculative, int idTask, int nPromptTokens,
               int nPromptTokensProcessed, int nPromptTokensCache, SamplerConfig params, int maxNewTokens,
               boolean hasNextToken, int nRemain, int nDecoded, int[] promptTokens, int[] generatedTokens) {
        this.id = id;
        this.nCtx = nCtx;
        this.processing = processing;
        this.speculative = speculative;
        this.idTask = idTask;
        this.nPromptTokens = nPromptTokens;
        this.nPromptTokensProcessed = nPromptTokensProcessed;
        this.nPromptTokensCache = nPromptTokensCache;
        this.params = params;
        this.maxNewTokens = maxNewTokens;
        this.hasNextToken = hasNextToken;
        this.nRemain = nRemain;
        this.nDecoded = nDecoded;
        this.promptTokens = promptTokens == null ? null : promptTokens.clone();
        this.generatedTokens = generatedTokens == null ? null : generatedTokens.clone();
    }

    /** The bare status of a slot whose engine holds no task for it: no task fields, nothing to decode. */
    static SlotStatus idle(int id, int nCtx, boolean speculative) {
        return new SlotStatus(id, nCtx, false, speculative, -1, -1, -1, -1, null, -1, false, -1, 0, null, null);
    }

    /** The slot id, over the engine's slots. */
    public int id() {
        return id;
    }

    /** The positions the slot's sequence may hold (the plan's {@code nCtx}). */
    public int nCtx() {
        return nCtx;
    }

    /** Whether a sequence runs in the slot now. */
    public boolean processing() {
        return processing;
    }

    /** Whether the engine decodes with a drafter ({@code BatchDrafter}), as llama.cpp's {@code speculative}. */
    public boolean speculative() {
        return speculative;
    }

    /** The running or last request's engine id, -1 when the slot holds no task. */
    public int idTask() {
        return idTask;
    }

    /** The task's prompt tokens (the keyed rows it occupies, media rows included), -1 without a task. */
    public int nPromptTokens() {
        return nPromptTokens;
    }

    /** The task's prompt rows its prefills processed so far (a reused prefix included). */
    public int nPromptTokensProcessed() {
        return nPromptTokensProcessed;
    }

    /** The task's prompt rows adopted from the prefix cache instead of prefilled. */
    public int nPromptTokensCache() {
        return nPromptTokensCache;
    }

    /** The task's sampler parameters, null without a task. */
    public SamplerConfig params() {
        return params;
    }

    /** The task's token budget ({@code GenerationRequest.maxNewTokens}), -1 for unlimited and without a task. */
    public int maxNewTokens() {
        return maxNewTokens;
    }

    /** Whether a sampled token waits to be fed at the next position. */
    public boolean hasNextToken() {
        return hasNextToken;
    }

    /** The tokens the task may still decode within its budget, -1 for unlimited and without a task. */
    public int nRemain() {
        return nRemain;
    }

    /** The decode tokens the task fed so far, 0 without a task. */
    public int nDecoded() {
        return nDecoded;
    }

    /** Copy of the task's prompt tokens ({@code MediaPrompt.samplerTokens}, media rows left out), or null. */
    public int[] promptTokens() {
        return promptTokens == null ? null : promptTokens.clone();
    }

    /** Copy of the task's fed decode tokens, or null without a task. */
    public int[] generatedTokens() {
        return generatedTokens == null ? null : generatedTokens.clone();
    }

    @Override
    public String toString() {
        return "SlotStatus{id=" + id + ", nCtx=" + nCtx + ", processing=" + processing
                + ", speculative=" + speculative + ", idTask=" + idTask + ", nPromptTokens=" + nPromptTokens
                + ", nDecoded=" + nDecoded + "}";
    }
}
