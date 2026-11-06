package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.StepSizes;
import com.eonmux.neuralfree.gguf.engine.StopReason;

import java.util.ArrayList;
import java.util.List;

/**
 * Schedules one {@link BatchEngine} step within its token budget ({@code neuralfree.max.batched.tokens}, vLLM's
 * {@code max_num_batched_tokens}): first-come first-served over the running sequences in admission order, each
 * decode round its rows ({@link DecodeRounds}) and each prefill chunk its rows ({@link PrefillChunks}), then hands
 * the pieces to the {@link StepExecutor}. With chunked prefill off the step keeps the older shape: the oldest
 * waiting prefill runs alone — its whole prompt tail in one forward batch — or, when none waits, one decode row
 * per running sequence. Over a memory with context checkpoints a text prefill's chunks end on the checkpoint rows.
 * A text prefill's chunks are multiples of the ubatch size the whole run to the next forced break splits into
 * equally, so no step holds a short trailing ubatch.
 */
final class StepScheduler {

    private final BatchState state;
    private final PrefixClaims claims;
    private final SlotLifecycle lifecycle;
    private final DecodeRounds rounds;
    private final StepExecutor executor;
    /** Whether a prompt prefill splits into budget-bounded chunks beside the decodes. */
    private final boolean chunkedPrefill;
    /** The token budget one step shares between its prefill chunk rows and decode rows. */
    private final int tokenBudget;
    /** The per-step cap on one request's prefill chunk while other requests run or wait; 0 sets no cap. */
    private final int longPrefillTokens;
    /** The media rows one step schedules over all its media prefills (vLLM's {@code encoder_compute_budget}). */
    private final int encoderBudget;
    /** The rows of one ubatch, the bound on a decode round with chunked prefill off. */
    private final int ubatch;
    /** Whether the model's attention is causal: a bidirectional model's pooling prompt prefills whole. */
    private final boolean causal;
    /** Whether decode rounds verify tree proposals: their steps share the row cap the per-row mask bounds. */
    private final boolean treeMode;

    StepScheduler(BatchState state, PrefixClaims claims, SlotLifecycle lifecycle, DecodeRounds rounds,
                  StepExecutor executor, boolean chunkedPrefill, int tokenBudget, int longPrefillTokens,
                  int encoderBudget, int ubatch, boolean causal, boolean treeMode) {
        this.state = state;
        this.claims = claims;
        this.lifecycle = lifecycle;
        this.rounds = rounds;
        this.executor = executor;
        this.chunkedPrefill = chunkedPrefill;
        this.tokenBudget = tokenBudget;
        this.longPrefillTokens = longPrefillTokens;
        this.encoderBudget = encoderBudget;
        this.ubatch = ubatch;
        this.causal = causal;
        this.treeMode = treeMode;
    }

    /** The pieces one step schedules and the budgets they leave. */
    private static final class StepPlan {
        final List<Piece> pieces = new ArrayList<>();
        int budgetLeft;
        /** The media rows the step's media prefills may still schedule (vLLM's {@code encoder_compute_budget}). */
        int encoderLeft;
        /** Whether a prefill chunk is capped by {@code neuralfree.long.prefill.tokens} this step. */
        boolean capChunks;
        /** Whether a running sequence has a runnable prefill (the unchunked step then decodes nothing). */
        boolean anyPrefill;
        /**
         * The logits-batch rows the step may still schedule while one of its rounds verifies a tree: the per-row
         * ancestor mask names every row of the shared batch, whose rows must also fit one ubatch whole (a tree
         * never splits). Unbounded when no round armed a tree.
         */
        int maskLeft;
    }

    /**
     * Schedules and runs one step over the active sequences: first-come first-served over the running sequences
     * in admission order within the token budget, each decode round its rows (one plain, or {@code 1 + drafted}
     * when the engine speculates: the step's proposals are drafted first, in one call over every decoding
     * sequence, {@link DecodeRounds#draftStep}, and a proposal the step does not schedule is discarded) and a
     * prefill chunk its rows, so a sequence ahead of a long prompt keeps
     * decoding and a later one waits (vLLM's FCFS loop). A prefill chunk takes the rows its prompt still lacks,
     * capped per step by {@code neuralfree.long.prefill.tokens} while other requests run or wait (vLLM
     * {@code long_prefill_token_threshold}); a decode round bounds its drafts to the budget left (vLLM budgets a
     * request's speculative rows too). A media prompt's pending prefill schedules whole-chunk pieces within the
     * step's encoder budget ({@code neuralfree.max.encoder.tokens}, vLLM's {@code encoder_compute_budget});
     * unscheduled chunks wait for a later step (see {@link PrefillChunks#chunkPieces}). With chunked prefill off
     * the oldest waiting prefill takes its whole tail alone and the decodes wait. A fork whose donor has
     * committed the shared rows copies them ({@link PrefixClaims#settleFork}) before its tail schedules; while it
     * waits the step schedules the others.
     */
    void step() {
        if (state.active.isEmpty()) {
            return;
        }
        StepPlan plan = new StepPlan();
        plan.budgetLeft = tokenBudget;
        plan.encoderLeft = encoderBudget;
        plan.capChunks = longPrefillTokens > 0
                && state.active.size() + state.queue.size() + state.preempted.size() > 1;
        plan.anyPrefill = hasWaitingPrefill();
        plan.maskLeft = Integer.MAX_VALUE;
        if (chunkedPrefill || !plan.anyPrefill) {
            rounds.draftStep(new ArrayList<>(state.active), chunkedPrefill ? tokenBudget : ubatch);
        }
        if (treeMode) {
            for (Slot slot : state.active) {
                if (slot.treeProposal != null) {
                    // a tree step's every logits row carries a mask bit and runs in one ubatch
                    plan.maskLeft = Math.min(DecodeRounds.MAX_TREE_ROWS, ubatch);
                    break;
                }
            }
        }
        List<Slot> order = new ArrayList<>(state.active);
        for (Slot slot : order) {
            boolean more = slot.needsPrefill ? schedulePrefill(slot, plan) : scheduleDecode(slot, plan);
            if (!more) {
                break;
            }
        }
        for (Slot slot : order) {
            rounds.discardProposal(slot);
        }
        if (!plan.pieces.isEmpty()) {
            executor.run(plan.pieces);
        }
    }

    /**
     * Schedules {@code slot}'s pending prefill: a cancelled request finishes, a fork waiting for its donor skips
     * this step, a text-only prompt takes a chunk of the token budget and a media prompt its whole chunks within
     * the step's encoder budget. False when the step schedules no further sequence (the budget is spent, or
     * chunked prefill is off and this prefill runs alone).
     */
    private boolean schedulePrefill(Slot slot, StepPlan plan) {
        if (state.cancelRequested(slot)) {
            lifecycle.finish(slot, StopReason.CANCELLED);
            return true;
        }
        if (claims.waitsForDonor(slot)) {
            return true; // the fork's donor prefills first (it was admitted first)
        }
        claims.settleFork(slot);
        if (slot.submission.prompt.isTextOnly()) {
            int rows = chunkRows(slot, Math.min(plan.budgetLeft, plan.maskLeft), plan.capChunks);
            if (chunkedPrefill && rows <= 0) {
                return false;
            }
            Piece piece = PrefillChunks.chunkPiece(slot, rows);
            if (slot.submission.pooling != null) {
                piece.wantsHidden = true; // the rows feed the pooling run instead of the sampler
            }
            plan.pieces.add(piece);
            plan.budgetLeft -= rows;
            if (!piece.wantsHidden) {
                plan.maskLeft -= rows;
            }
        } else {
            int media = PrefillChunks.chunkPieces(slot, plan.pieces, plan.encoderLeft, encoderBudget,
                    checkpointStop(slot));
            plan.encoderLeft -= media;
        }
        return chunkedPrefill;
    }

    /**
     * Schedules {@code slot}'s decode round: within the budget left with chunked prefill on, one ubatch at most
     * with it off — and then only when no prefill waits. False when the budget is spent.
     */
    private boolean scheduleDecode(Slot slot, StepPlan plan) {
        if (plan.maskLeft <= 0) {
            return false; // the step's masked rows are spent: the rest decodes on a later step
        }
        if (chunkedPrefill) {
            if (plan.budgetLeft <= 0) {
                return false;
            }
            if (slot.lookahead != null) {
                return scheduleLookahead(slot, plan);
            }
            Piece decode = rounds.decodePiece(slot, Math.min(plan.budgetLeft, plan.maskLeft));
            if (decode != null) {
                plan.pieces.add(decode);
                plan.budgetLeft -= decode.rows;
                plan.maskLeft -= decode.rows;
            }
        } else if (!plan.anyPrefill) {
            if (slot.lookahead != null) {
                return scheduleLookahead(slot, plan);
            }
            Piece decode = rounds.decodePiece(slot, Math.min(ubatch, plan.maskLeft));
            if (decode != null) {
                plan.pieces.add(decode);
                plan.maskLeft -= decode.rows;
            }
        }
        return true;
    }

    /**
     * Schedules {@code slot}'s lookahead step: its own forward batch over the slot's whole sequence group, so it
     * spends none of the step's token budget (like a media chunk's encoder batch). A request holding no room for
     * the step's position span stops with {@link StopReason#CONTEXT_FULL} before anything is written, and a
     * cancelled one finishes at once.
     */
    private boolean scheduleLookahead(Slot slot, StepPlan plan) {
        if (state.cancelRequested(slot)) {
            lifecycle.finish(slot, StopReason.CANCELLED);
            return true;
        }
        if (slot.nextPos + slot.lookahead.positionSpan() > slot.nCtx) {
            lifecycle.finish(slot, StopReason.CONTEXT_FULL);
            return true;
        }
        plan.pieces.add(Piece.lookahead(slot, slot.lookahead.step(slot.pendingToken, slot.nextPos)));
        return true;
    }

    /**
     * The rows {@code slot}'s prefill chunk takes this step: its whole remaining tail with chunked prefill off,
     * else the tail bounded by the long-prefill cap and the budget left, in as many ubatches of the run's equal
     * size as fit ({@link #ubatchAlignedRows}). A bidirectional model's pooling prompt takes its whole tail
     * whatever the budget (its rows attend each other, so a chunk boundary would change them, as vLLM's scheduler
     * keeps pooling requests whole without chunked prefill). Over a memory with context checkpoints the chunk ends
     * at the next checkpoint row ({@link ContextCheckpoints#nextTarget}: the next spaced row, the tail, or the
     * prompt's next message boundary), as llama.cpp's server ends a prompt batch where it takes a checkpoint, so
     * the checkpoint captures the state at exactly that row.
     */
    private int chunkRows(Slot slot, int budgetLeft, boolean capChunks) {
        int rows = slot.pendingTotal() - slot.prefillRows;
        if (slot.submission.pooling != null && !causal) {
            return rows;
        }
        int cap = rows;
        if (chunkedPrefill) {
            if (capChunks && longPrefillTokens < cap) {
                cap = longPrefillTokens;
            }
            cap = Math.min(cap, budgetLeft);
        }
        // the rows to the next forced break: the prompt's tail, or the checkpoint row the batch must end at
        int stop = rows;
        int checkpoint = -1;
        if (state.checkpoints != null && state.cache != null) {
            int committed = slot.committedRows();
            int target = checkpointStop(slot);
            if (target > committed && target - committed < stop) {
                stop = target - committed;
                checkpoint = target;
            }
        }
        // a spent budget schedules no prefill row this step (the caller stops the walk), and a run that fits the
        // step takes all of it
        int take = Math.min(cap, stop);
        if (chunkedPrefill && take > 0 && take < stop) {
            take = ubatchAlignedRows(take, stop);
        }
        if (checkpoint >= 0 && take == stop) {
            slot.pendingCheckpointRow = checkpoint;
        }
        return take;
    }

    /**
     * The rows of the step within {@code rows} (its budget, caps and tail) of a run of {@code stop} rows to the
     * next forced break: as many ubatches of the run's equal size as {@code rows} holds
     * ({@link StepSizes#next}), so every step up to the break runs ubatches of one size and none trails short — a
     * ubatch costs nearly as much whatever its rows when it pays a fixed cost (the expert upload of a streaming
     * model). A server batch that would hold a short trailing ubatch instead takes the rows that fill one of the
     * run's ubatches and lets the next step hold the rest (avalon tools/server/server-context.cpp:4126).
     *
     * <p>With chunked prefill off the run is one batch, which the backend splits into equal ubatches itself.
     */
    private int ubatchAlignedRows(int rows, int stop) {
        int unit = StepSizes.next(stop, Math.min(rows, ubatch));
        return rows >= unit ? (rows / unit) * unit : rows;
    }

    /**
     * The next checkpoint row of {@code slot}'s pending prefill (absolute), or -1 without a checkpointed cache:
     * the nearest of the spaced rows, the tail and the prompt's next message boundary past the committed rows.
     */
    private int checkpointStop(Slot slot) {
        if (state.checkpoints == null || state.cache == null) {
            return -1;
        }
        int committed = slot.committedRows();
        return state.checkpoints.nextTarget(committed, slot.prefillEnd(), slot.lastCheckpoint, slot.nextBoundary());
    }

    /** Whether a running sequence has a runnable prefill: a fork still waiting for its donor does not count,
     * so the step decodes instead of stalling on an empty step. */
    private boolean hasWaitingPrefill() {
        for (Slot slot : state.active) {
            if (slot.needsPrefill && !claims.waitsForDonor(slot)) {
                return true;
            }
        }
        return false;
    }
}
