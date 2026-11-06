package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.sampling.ReasoningBudgetStage;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The live chat completions the control endpoint ({@code POST /v1/chat/completions/control}) resolves by id: an
 * entry enters when its request starts and leaves when every choice of its batch reached a terminal state.
 */
final class LiveCompletions {

    /** The outcome of a control action. */
    enum Kind { FORCED, NO_COMPLETION, NOT_ARMED }

    /**
     * What a control action did: {@link Kind#FORCED} with the count of stages it moved into forcing,
     * {@link Kind#NO_COMPLETION} for an unknown or finished id, {@link Kind#NOT_ARMED} for a live completion whose
     * request armed no budget stage.
     */
    record Outcome(Kind kind, int forced) { }

    private final ConcurrentHashMap<String, GenerationBatch> live = new ConcurrentHashMap<>();

    /** Whether a live completion id is tracked here. */
    boolean tracks(String id) {
        return live.containsKey(id);
    }

    /** Tracks {@code id} until every choice of {@code batch} finished. */
    void track(String id, GenerationBatch batch) {
        live.put(id, batch);
        batch.onAllDone(() -> live.remove(id));
    }

    /**
     * The {@code reasoning_end} action on {@code id}: every still-running choice with an armed budget stage moves
     * into forcing the end sequence (llama.cpp's {@code SERVER_TASK_TYPE_CONTROL} handling).
     */
    Outcome endReasoning(String id) {
        GenerationBatch batch = live.get(id);
        if (batch == null) {
            return new Outcome(Kind.NO_COMPLETION, 0);
        }
        List<BatchSequence> sequences = batch.liveSequences();
        if (sequences.isEmpty()) {
            return new Outcome(Kind.NO_COMPLETION, 0);
        }
        int forced = 0;
        boolean armed = false;
        for (BatchSequence sequence : sequences) {
            ReasoningBudgetStage stage = sequence.budgetStage();
            if (stage == null) {
                continue;
            }
            armed = true;
            if (stage.force()) {
                forced++;
            }
        }
        return armed ? new Outcome(Kind.FORCED, forced) : new Outcome(Kind.NOT_ARMED, 0);
    }
}
