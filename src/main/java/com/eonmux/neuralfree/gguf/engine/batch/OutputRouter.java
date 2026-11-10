package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.SequenceRun;
import com.eonmux.neuralfree.gguf.engine.StopReason;

import java.util.List;

/**
 * Routes one {@link BatchEngine} step's logits rows to the sequences that scheduled them, in batch order: a plain
 * decode samples its row, a verify round walks its rows ({@link DecodeRounds#verify}), a completing prompt chunk
 * samples the sequence's first token, and a chunk without a logits row advances its prefill cursor. A step that read
 * its candidates back on the device routes the same pieces over its top-k rows instead ({@link #routeTopK}). Each
 * sampled token goes through the sequence's own {@link SequenceRun} (sample, stop, route to the submission's sink).
 */
final class OutputRouter {

    private final BatchState state;
    private final SlotLifecycle lifecycle;
    private final DecodeRounds rounds;
    private final ContextShift shifts;

    OutputRouter(BatchState state, SlotLifecycle lifecycle, DecodeRounds rounds, ContextShift shifts) {
        this.state = state;
        this.lifecycle = lifecycle;
        this.rounds = rounds;
        this.shifts = shifts;
    }

    /**
     * Routes one step's outputs in batch order: each decode piece samples its row or walks its verify round, each
     * completing prompt chunk samples from its logits, a pooling piece's rows land in its submission's pooling run
     * (the completing one retires the request through its pooling completion), and the chunks without an output row
     * advance their cursors. A piece's rows sit at its {@code outIdx} in the step's outputs — one row, or all of a
     * decode round's, a prompt-logprobs piece's or a pooling piece's rows.
     */
    void route(List<Piece> pieces, float[][] outputs) {
        for (Piece p : pieces) {
            try {
                if (p.decode) {
                    if (p.rows == 1) {
                        sampleAndOffer(p.slot, outputs[p.outIdx], true);
                    } else if (p.tree != null) {
                        rounds.verifyTree(p.slot, p.tree, outputs, p.outIdx);
                    } else {
                        rounds.verify(p.slot, p.tokens, outputs, p.outIdx);
                    }
                } else if (p.wantsHidden) {
                    float[][] rows = new float[p.rows][];
                    for (int i = 0; i < p.rows; i++) {
                        rows[i] = outputs[p.outIdx + i];
                    }
                    p.slot.submission.pooling.append(rows);
                    PrefillChunks.chunkWritten(p.slot, p);
                    if (p.finalChunk) {
                        poolingDone(p.slot);
                    }
                } else if (p.output) {
                    p.slot.nextPromptPos = p.from + p.rows;
                    prefillDone(p.slot, outputs[p.readRows ? p.outIdx + p.rows - 1 : p.outIdx]);
                } else {
                    PrefillChunks.chunkWritten(p.slot, p);
                }
            } catch (RuntimeException e) {
                retireFrom(pieces, pieces.indexOf(p), e);
                return;
            }
        }
    }

    /**
     * Routes a step's device top-k readback in batch order: every piece with a {@code topRow} entry samples the
     * token the step's chain selects over that row of {@code top}. The step only takes this path when all of its
     * sampling pieces keep their candidates inside the cone ({@link StepExecutor}), so a piece is either a plain
     * decode or a completing prompt chunk.
     *
     * @param topRow for each piece, its row in {@code top}, or -1 when the piece samples nothing
     */
    void routeTopK(List<Piece> pieces, RowTopK top, int[] topRow) {
        routeTopK(pieces, top, topRow, false);
    }

    /**
     * {@link #routeTopK(List, RowTopK, int[])} where the readout applied each row's logit prefilter before its
     * top-k, so the chain runs over the candidates without repeating those edits.
     */
    void routeTopK(List<Piece> pieces, RowTopK top, int[] topRow, boolean prefiltered) {
        for (int i = 0; i < pieces.size(); i++) {
            Piece p = pieces.get(i);
            try {
                if (topRow[i] < 0) {
                    if (!p.decode && !p.wantsHidden) {
                        // a prompt chunk that does not complete the prompt only advances its cursor
                        PrefillChunks.chunkWritten(p.slot, p);
                    }
                    continue;
                }
                if (p.decode) {
                    sampleAndOfferTopK(p.slot, top, topRow[i], true, prefiltered);
                } else {
                    p.slot.nextPromptPos = p.from + p.rows;
                    prefillDoneTopK(p.slot, top, topRow[i], prefiltered);
                }
            } catch (RuntimeException e) {
                retireFrom(pieces, i, e);
                return;
            }
        }
    }

    /** Retires every sequence whose piece ran from {@code from} on: the earlier pieces already sampled. */
    private void retireFrom(List<Piece> pieces, int from, RuntimeException e) {
        lifecycle.retire(pieces.get(from).slot, null, false, e);
        for (int j = from + 1; j < pieces.size(); j++) {
            lifecycle.retire(pieces.get(j).slot, null, false, e);
        }
    }

    /**
     * Completes a pooling request at its prompt's last chunk: the row and position cursors take the prompt's
     * counts (they differ after a media chunk with fewer positions than rows), the reused prefix counts as cached
     * prompt work like a generation prefill's, and the sequence retires through the pooling completion, which
     * applies the pooling operator over the accumulated rows.
     */
    private void poolingDone(Slot slot) {
        Submission sub = slot.submission;
        slot.needsPrefill = false;
        slot.nextPos = sub.prompt.nPositions();
        slot.nextRow = sub.prompt.nTokens();
        state.stats.promptTokensCached += sub.reused;
        lifecycle.poolingRetire(slot);
    }

    /**
     * Completes a fresh prompt prefill from its final chunk's last-row logits: the run records the prompt phase
     * over the chunked forward time, the sampler takes the prompt's real tokens, and the row and position cursors
     * take the prompt's counts (they differ after a media chunk with fewer positions than rows). The sampled
     * token waits to be fed at {@code nextPos}.
     */
    private void prefillDone(Slot slot, float[] logits) {
        prefillAccounting(slot);
        sampleAndOffer(slot, logits, false);
    }

    /** The same prefill completion over the piece's row of a device top-k readback. */
    private void prefillDoneTopK(Slot slot, RowTopK top, int row, boolean prefiltered) {
        prefillAccounting(slot);
        sampleAndOfferTopK(slot, top, row, false, prefiltered);
    }

    /** The prefill bookkeeping of a completing prompt chunk, before either sampling path draws its first token. */
    private void prefillAccounting(Slot slot) {
        Submission sub = slot.submission;
        slot.needsPrefill = false;
        slot.run.prefilled(sub.prefilledTokens(), slot.prefillNanos);
        state.stats.promptTokensCached += sub.reused;
        int[] samplerTokens = sub.prompt.samplerTokens();
        slot.run.sampler().acceptPrompt(samplerTokens, 0, samplerTokens.length);
        // the prefill's own cursor: a self-extend compression left the positions below the prompt's row count
        slot.nextPos = slot.nextPromptPos;
        slot.nextRow = sub.prompt.nTokens();
    }

    /**
     * Samples one logits row through the sequence's chain, offers the token and retires or arms the slot. After a
     * prefill ({@code decoded} false) the sampled token waits to be fed at {@code nextPos}; after a decode the fed
     * token's row counts as committed first, so a sequence that stops here retires with it, and the sampled token
     * arms at the next position.
     */
    private void sampleAndOffer(Slot slot, float[] logits, boolean decoded) {
        if (!beginSample(slot, decoded)) {
            return;
        }
        long t0 = System.nanoTime();
        int token = slot.run.next(logits);
        state.stats.hostSampleNanos += System.nanoTime() - t0;
        offerSampled(slot, token);
    }

    /** The same offer over row {@code row} of a device top-k readback, the batched backend-sampled path. */
    private void sampleAndOfferTopK(Slot slot, RowTopK top, int row, boolean decoded, boolean prefiltered) {
        if (!beginSample(slot, decoded)) {
            return;
        }
        long t0 = System.nanoTime();
        int token = slot.run.sampleTopK(top, row, prefiltered);
        state.stats.hostSampleNanos += System.nanoTime() - t0;
        slot.run.accept(token);
        state.stats.deviceSampledRows++;
        offerSampled(slot, token);
    }

    /**
     * Counts a decode's fed token as committed and refuses to sample into a cancelled request.
     *
     * @return false when the request was cancelled and the slot finished; the caller samples nothing
     */
    private boolean beginSample(Slot slot, boolean decoded) {
        if (decoded) {
            slot.fed(slot.pendingToken);
            slot.nextPos++;
            slot.nextRow++;
        }
        if (state.cancelRequested(slot)) {
            lifecycle.finish(slot, StopReason.CANCELLED);
            return false;
        }
        return true;
    }

    /** Offers a sampled token to the run and retires or arms the slot. */
    private void offerSampled(Slot slot, int token) {
        state.stats.generatedTokens++;
        SequenceRun.Step step = slot.run.offer(token);
        if (step.stopped()) {
            // offer recorded the stop (end of generation, stop string, budget): the result carries its reason
            lifecycle.retire(slot, slot.run.result(), false, null);
            return;
        }
        slot.pendingToken = token;
        shifts.shiftOrFinish(slot);
    }
}
