package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.LookaheadRun;
import com.eonmux.neuralfree.gguf.engine.SequenceRun;
import com.eonmux.neuralfree.gguf.engine.draft.Verifier;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * One running sequence: its slot id, run pipeline and decode cursor, and the decode tokens it fed (with the
 * prompt, the history a recompute after preemption prefills again). {@code nextRow} counts the KV cells the
 * sequence stored (its prompt rows and fed decodes, one per embedding row too), {@code nextPos} the position
 * the next decode takes — the two differ after a media chunk that took fewer positions than rows, and after a
 * self-extend compression. {@code nextPromptPos} is the position the pending prefill's next row takes and
 * {@code gaI} the lowest position a self-extend has not grouped yet. A pending
 * prefill writes its rows in budget-bounded pieces: {@code prefillRows} counts the rows already written,
 * {@code prefillNanos} their forward time. A fork ({@code forkFrom >= 0}) still owes its prefill the copy of
 * the donor's first {@code forkLen} prompt rows before its prompt tail. A speculating sequence carries its
 * verifier, its verify and draft streams, and the proposal its step's draft call armed. A context shift
 * ({@link #contextShifted}) moves the cursors down and rebuilds the view {@code history()} replays with the
 * dropped tokens gone ({@code shifted}); a shifted slot no longer serves as a donor fork.
 */
final class Slot {
    final int seq;
    final Submission submission;
    final SequenceRun run;
    final int nCtx;
    int nextPos;
    int nextRow;
    /**
     * The position the pending prefill's next row takes. It equals the row index until a self-extend compression
     * moves the stored positions, and every prefill piece advances it past its own rows.
     */
    int nextPromptPos;
    /** The lowest position the self-extend has not grouped yet (llama.cpp's {@code ga_i}); 0 without group attention. */
    int gaI;
    int pendingToken = -1;
    boolean needsPrefill = true;
    /** Whether the pending prefill replays the history of a preempted run instead of a fresh prompt. */
    boolean recompute;
    /** The donor sequence a pending fork copies its {@code forkLen} prefix rows from, or -1. */
    int forkFrom = -1;
    int forkLen;
    /** Rows of the pending prefill or recompute already written; the rest schedules in later chunks. */
    int prefillRows;
    /** Forward time the pending prefill's chunks took so far, reported once by {@code SequenceRun.prefilled}. */
    long prefillNanos;
    /**
     * The rows the sequence stored before its fed window, once a context shift moved them ({@code shifted}): the
     * prompt and fed tokens compacted into one view, ended at the rows the shifts left. Null keeps the composition
     * of the submission's prompt and every fed token.
     */
    int[] historyView;
    /** Whether a context shift moved the stored rows off the submission's prompt view. */
    boolean shifted;
    /** The cached blocks the slot's committed rows extend; adoption sets it, each commit extends it. */
    PrefixCache.BlockChain chain = new PrefixCache.BlockChain();
    /** The rows of the last context checkpoint the slot took or restored (0 for none). */
    int lastCheckpoint;
    /** The sequence's verifier when the engine verifies by another mode than {@code MATCH}, else null. */
    Verifier verifier;
    /** The uniform stream of the sequence's verify rounds, seeded by its sampler seed. */
    SplittableRandom verifyRng;
    /** The stream a probabilistic drafter draws the sequence's draft tokens from. */
    SplittableRandom draftRng;
    /** The temperature a probabilistic drafter drafts the sequence with; 0 for a greedy request. */
    float draftTemperature;
    /** The proposal the step's draft call armed for the decode round, or null. */
    int[] proposal;
    /** The tree proposal the step's draft call armed for the decode round, or null (the chain proposal then). */
    TreeProposal treeProposal;
    /**
     * The request's lookahead (Jacobi) decoding run over the group of sequences this slot owns, or null for plain
     * decoding (and for a pooling request, which never decodes).
     */
    LookaheadRun lookahead;
    /**
     * The prompt row the pending prefill's next checkpoint lands on (a message boundary the prompt named, or a
     * checkpoint target a media chunk runs whole past), or -1 for none; {@code commitBlocks} consumes it.
     */
    int pendingCheckpointRow = -1;
    private int[] fed = new int[16];
    private int fedCount;

    Slot(int seq, Submission submission, SequenceRun run, int nCtx) {
        this.seq = seq;
        this.submission = submission;
        this.run = run;
        this.nCtx = nCtx;
        this.nextPromptPos = submission.reused;
    }

    /**
     * The prefix of the keyed prompt rows {@code prompt} this slot's prompt shares, short of {@code prompt}'s last
     * row and never ending inside a media chunk ({@link CacheRows#cut}): rows agree by token, or by media item
     * identity and index, or by prompt-embedding digest, so text, media and prompt-embedding prompts all fork.
     */
    int forkLength(CacheRows prompt) {
        int common = CacheRows.commonLength(submission.promptRows(), 0, prompt, 0, prompt.length() - 1);
        return prompt.cut(common);
    }

    /**
     * The rows the pending prefill writes in total: the prompt tail after a reused prefix, or the history a
     * recompute replays (the prompt's rows plus the fed decodes).
     */
    int pendingTotal() {
        return recompute ? nextRow : submission.prompt.nTokens() - submission.reused;
    }

    /** The row the pending prefill ends at: the prompt's rows, or the history a recompute replays. */
    int prefillEnd() {
        return recompute ? nextRow : submission.prompt.nTokens();
    }

    /**
     * The leading rows of the sequence whose tokens are known and final: the prompt rows written so far (a reused
     * prefix included) while the prompt prefills, the replayed rows while a recompute runs, and then the prompt
     * rows and fed decodes. Rows of a step still being verified, or of a sampled token not yet fed, are not among
     * them.
     */
    int committedRows() {
        if (!needsPrefill) {
            return nextRow;
        }
        return recompute ? prefillRows : submission.reused + prefillRows;
    }

    /**
     * The keyed rows of the sequence's committed rows (at least {@link #committedRows()} of them): the prompt's
     * rows while it prefills, then the prompt's rows and the fed decodes.
     */
    CacheRows committedKeys() {
        return needsPrefill && !recompute ? submission.promptRows() : historyRows();
    }

    /**
     * The prompt's first checkpoint row past both the committed rows and the last checkpoint taken, or -1: the
     * next message boundary of the prompt the prefill still owes a checkpoint.
     */
    int nextBoundary() {
        int[] rows = submission.prompt.checkpointRows();
        int after = Math.max(committedRows(), lastCheckpoint);
        for (int row : rows) {
            if (row > after) {
                return row;
            }
        }
        return -1;
    }

    /**
     * The keyed rows of the {@code nextRow} rows a recompute replays: the prompt's rows (media rows under their
     * keys) and the fed decode tokens at the positions after the prompt, the tokens the context shifts dropped
     * gone.
     */
    CacheRows historyRows() {
        if (shifted) {
            return CacheRows.ofTokens(history());
        }
        return submission.promptRows().withFed(fedTokens());
    }

    /** Records a decode token fed at the next position. */
    void fed(int token) {
        if (fedCount == fed.length) {
            fed = Arrays.copyOf(fed, fed.length * 2);
        }
        fed[fedCount++] = token;
    }

    /** Decode tokens fed so far, in feed order. */
    int fedCount() {
        return fedCount;
    }

    /** Copy of the fed decode tokens. */
    int[] fedTokens() {
        return Arrays.copyOf(fed, fedCount);
    }

    /**
     * The rows the sequence stores, with the tokens the context shifts dropped gone: the shifted view followed by
     * the decodes fed after it (or the prompt and every fed token before the first shift). These are the
     * {@code nextRow} rows a text prompt's recompute replays (a media prompt's recompute walks its chunks instead
     * and never reads this view).
     */
    int[] history() {
        int[] prompt = submission.prompt.samplerTokens();
        if (historyView == null) {
            int[] out = Arrays.copyOf(prompt, prompt.length + fedCount);
            System.arraycopy(fed, 0, out, prompt.length, fedCount);
            return out;
        }
        int[] out = Arrays.copyOf(historyView, historyView.length + fedCount);
        System.arraycopy(fed, 0, out, historyView.length, fedCount);
        return out;
    }

    /**
     * Settles a context shift's bookkeeping: the positions {@code [nKeep, nKeep + nDiscard)} left the sequence,
     * so the cursors move down by that count and the tokens they held leave the view — the middle of a prompt
     * drops, not its head. The stored tokens compact into {@code historyView} and the fed window restarts with
     * them, so every decode after the shift extends the view. A shifted slot keeps decoding in place and never
     * serves as a donor fork again:
     * its rows at positions past {@code nKeep} are not the prompt's.
     */
    void contextShifted(int nKeep, int nDiscard) {
        int[] cur = history();
        int[] out = new int[cur.length - nDiscard];
        System.arraycopy(cur, 0, out, 0, nKeep);
        System.arraycopy(cur, nKeep + nDiscard, out, nKeep, cur.length - nKeep - nDiscard);
        historyView = out;
        fedCount = 0;
        nextPos -= nDiscard;
        nextRow -= nDiscard;
        shifted = true;
    }

    /**
     * Marks a self-extend edit: the sequence's stored positions were compressed, so the slot no longer serves as a
     * donor fork (a fork copies positions it assumes are the prompt's). The tokens and the row cursor stay.
     */
    void selfExtended() {
        shifted = true;
    }

    /**
     * The same run re-admitted under {@code newSeq}, its pending prefill starting over: the old slot's pages
     * were freed, so a decode history replays as a recompute ({@code pendingToken} stays armed) and a prompt
     * that never prefilled starts from its first token (its adopted prefix rows went with the pages, and a
     * pending fork's donor relationship went with them: it re-admits as a plain prompt).
     */
    Slot moveTo(int newSeq) {
        Slot moved = new Slot(newSeq, submission, run, nCtx);
        moved.fed = fed.clone();
        moved.fedCount = fedCount;
        moved.nextPos = nextPos;
        moved.nextRow = nextRow;
        moved.pendingToken = pendingToken;
        moved.verifier = verifier;
        moved.verifyRng = verifyRng;
        moved.draftRng = draftRng;
        moved.draftTemperature = draftTemperature;
        moved.lookahead = lookahead == null ? null : lookahead.rebased(newSeq);
        moved.historyView = historyView;
        moved.shifted = shifted;
        moved.recompute = !needsPrefill || recompute;
        // a replay starts over at row 0 and position 0, so the self-extend restarts with it; a prompt that never
        // prefilled keeps its compressed positions from the reused prefix on
        moved.gaI = moved.recompute ? 0 : gaI;
        moved.nextPromptPos = moved.recompute ? 0 : submission.reused;
        if (!moved.recompute && submission.reused > 0) {
            submission.reused = 0;
            submission.handle.reused(0);
        }
        // the prefill starts over: a prompt-logprobs capture and a pooling run restart with it
        submission.promptLogprobs = null;
        if (submission.pooling != null) {
            submission.pooling.reset();
        }
        return moved;
    }
}
