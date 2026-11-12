package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.SequenceRun;
import com.eonmux.neuralfree.gguf.engine.StopReason;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Routes one {@link BatchEngine} step's logits rows to the sequences that scheduled them, in batch order: a plain
 * decode samples its row, a verify round walks its rows ({@link DecodeRounds#verify}), a completing prompt chunk
 * samples the sequence's first token, and a chunk without a logits row advances its prefill cursor. A piece whose row
 * the step read back as device candidates samples over them instead of a whole row, piece by piece, so one step can
 * mix both. Each sampled token goes through the sequence's own {@link SequenceRun} (sample, stop, route to the
 * submission's sink).
 *
 * <p>The host-sampled rows of a step sample in parallel: every plain decode and completing prompt chunk whose chain
 * holds only its own state ({@code Sampler#samplesIndependently}, no logprobs capture) first takes its bookkeeping
 * on the engine's thread in batch order, then all of them run their chains at once on worker threads, and the
 * engine's thread offers, stops and routes the tokens in batch order. Each chain sees exactly the row and history
 * it saw serially, so every sequence samples the same token.</p>
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
     * Routes one step's whole rows in batch order ({@link #route(List, float[][], RowTopK, int[], boolean)} with no
     * device candidates).
     */
    void route(List<Piece> pieces, float[][] outputs) {
        route(pieces, outputs, null, null, false);
    }

    /**
     * Routes one step's outputs in batch order: a piece with a {@code topRow} entry samples the token its chain
     * selects over that row of {@code top} (a verify round walks its rows from there on, one per round row); every
     * other decode piece samples its row or walks its verify round, a
     * completing prompt chunk samples from its logits, a pooling piece's rows land in its submission's pooling run
     * (the completing one retires the request through its pooling completion), and the chunks without an output row
     * advance their cursors. A piece's whole rows sit at its {@code outIdx} in the step's outputs — one row, or all
     * of a decode round's, a prompt-logprobs piece's or a pooling piece's rows.
     *
     * @param top         the device candidates of the shared batch's output rows, or null when every row was read
     *                    whole
     * @param topRow      for each piece its (first) row in {@code top}, or -1 when it routes over whole rows; null
     *                    without {@code top}
     * @param prefiltered whether the readout applied each candidate row's logit prefilter before its top-k, so the
     *                    chain runs over the candidates without repeating those edits
     */
    void route(List<Piece> pieces, float[][] outputs, RowTopK top, int[] topRow, boolean prefiltered) {
        Presampled presampled = presample(pieces, outputs, topRow);
        for (int i = 0; i < pieces.size(); i++) {
            Piece p = pieces.get(i);
            try {
                if (topRow != null && topRow[i] >= 0) {
                    if (p.decode && p.rows > 1) {
                        rounds.verify(p.slot, p.tokens, top, topRow[i], prefiltered);
                    } else if (p.decode) {
                        sampleAndOfferTopK(p.slot, top, topRow[i], true, prefiltered);
                    } else {
                        p.slot.nextPromptPos = p.from + p.rows;
                        prefillDoneTopK(p.slot, top, topRow[i], prefiltered);
                    }
                } else if (presampled != null && presampled.state[i] != Presampled.SERIAL) {
                    if (!p.decode) {
                        recordPrefill(p.slot, presampled.generationStart[i]);
                    }
                    if (presampled.state[i] == Presampled.FAILED) {
                        throw presampled.errors[i];
                    }
                    if (presampled.state[i] == Presampled.SAMPLED) {
                        state.stats.parallelSampledRows++;
                        offerSampled(p.slot, presampled.tokens[i]);
                    } else {
                        lifecycle.finish(p.slot, StopReason.CANCELLED);
                    }
                } else if (p.decode) {
                    if (p.rows == 1) {
                        sampleAndOffer(p.slot, outputs[p.outIdx], true);
                    } else if (p.tree != null) {
                        rounds.verifyTree(p.slot, p.tree, outputs, p.outIdx);
                    } else {
                        rounds.verify(p.slot, p.tokens, outputs, p.outIdx);
                    }
                } else if (p.wantsHidden) {
                    float[][] rows = new float[p.rows][];
                    for (int r = 0; r < p.rows; r++) {
                        rows[r] = outputs[p.outIdx + r];
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
                retireFrom(pieces, i, e);
                return;
            }
        }
    }

    /**
     * The tokens the parallel pass sampled, by piece: how each piece's sampling went. A completing prompt's prefill
     * is recorded on its run and in the engine's counters when the route reaches the piece ({@link #recordPrefill}),
     * so a failure of an earlier piece leaves them as the serial route leaves them.
     */
    private static final class Presampled {
        /** The piece routes serially in {@link #route}. */
        static final byte SERIAL = 0;
        /** The piece sampled {@code tokens[i]}, which waits to be offered. */
        static final byte SAMPLED = 1;
        /**
         * The request was cancelled at its bookkeeping: nothing is offered, and the route finishes the slot when it
         * reaches the piece, so a failure of an earlier piece retires it once, with that failure, as serially.
         */
        static final byte CANCELLED = 2;
        /** The bookkeeping or the chain threw {@code errors[i]}: the route retires from the piece on. */
        static final byte FAILED = 3;

        final byte[] state;
        final int[] tokens;
        final RuntimeException[] errors;
        /** A completing prompt's generation clock start ({@link System#nanoTime}), taken at its bookkeeping. */
        final long[] generationStart;

        Presampled(int pieces) {
            state = new byte[pieces];
            tokens = new int[pieces];
            errors = new RuntimeException[pieces];
            generationStart = new long[pieces];
        }
    }

    /** The row a piece samples on the host when it may sample in parallel, or null. */
    private static float[] parallelRow(Piece p, float[][] outputs) {
        boolean sampled = p.decode ? p.rows == 1 && p.tree == null : p.output && !p.wantsHidden;
        if (!sampled || p.slot.run.wantsHostSampling() || !p.slot.run.sampler().samplesIndependently()) {
            return null;
        }
        return outputs[p.decode || !p.readRows ? p.outIdx : p.outIdx + p.rows - 1];
    }

    /**
     * Samples the step's parallel-safe host rows ({@link #parallelRow}; a piece routed over device candidates has
     * none) at once, or null when fewer than two qualify (the route samples serially). The pieces' bookkeeping (a
     * decode's fed row, a completing prompt's cursors and prompt history, the cancellation check) runs first on this
     * thread in batch order, and stops at the first piece that throws, as the serial route stops there. A cancelled
     * piece is only recorded: the route finishes it in batch order.
     */
    private Presampled presample(List<Piece> pieces, float[][] outputs, int[] topRow) {
        int n = pieces.size();
        float[][] rows = new float[n][];
        int count = 0;
        for (int i = 0; i < n; i++) {
            rows[i] = topRow != null && topRow[i] >= 0 ? null : parallelRow(pieces.get(i), outputs);
            if (rows[i] != null) {
                count++;
            }
        }
        if (count < 2) {
            return null;
        }
        Presampled out = new Presampled(n);
        int[] run = new int[count];
        int runs = 0;
        for (int i = 0; i < n; i++) {
            if (rows[i] == null) {
                continue;
            }
            Piece p = pieces.get(i);
            try {
                if (!p.decode) {
                    p.slot.nextPromptPos = p.from + p.rows;
                    out.generationStart[i] = System.nanoTime();
                    prefillCursors(p.slot);
                }
                if (fedAndLive(p.slot, p.decode)) {
                    out.state[i] = Presampled.SAMPLED;
                    run[runs++] = i;
                } else {
                    out.state[i] = Presampled.CANCELLED;
                }
            } catch (RuntimeException e) {
                out.state[i] = Presampled.FAILED;
                out.errors[i] = e;
                break;
            }
        }
        long t0 = System.nanoTime();
        IntStream.range(0, runs).parallel().forEach(k -> {
            int i = run[k];
            try {
                out.tokens[i] = pieces.get(i).slot.run.next(rows[i]);
            } catch (RuntimeException e) {
                out.state[i] = Presampled.FAILED;
                out.errors[i] = e;
            }
        });
        state.stats.hostSampleNanos += System.nanoTime() - t0;
        return out;
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
        recordPrefill(slot, System.nanoTime());
        prefillCursors(slot);
    }

    /**
     * Records a completing prompt's prefill: the run takes the prompt phase and starts its generation clock at
     * {@code generationStart}, and the reused prefix counts as cached prompt work.
     */
    private void recordPrefill(Slot slot, long generationStart) {
        Submission sub = slot.submission;
        slot.run.prefilled(sub.prefilledTokens(), slot.prefillNanos, generationStart);
        state.stats.promptTokensCached += sub.reused;
    }

    /**
     * Ends a completing prompt's prefill on its slot: the sampler takes the prompt's real tokens and the row and
     * position cursors take the prompt's counts.
     */
    private void prefillCursors(Slot slot) {
        Submission sub = slot.submission;
        slot.needsPrefill = false;
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
        if (!fedAndLive(slot, decoded)) {
            lifecycle.finish(slot, StopReason.CANCELLED);
            return false;
        }
        return true;
    }

    /**
     * {@link #beginSample}'s bookkeeping without finishing: counts a decode's fed token as committed and says
     * whether the request is still live (not cancelled).
     */
    private boolean fedAndLive(Slot slot, boolean decoded) {
        if (decoded) {
            slot.fed(slot.pendingToken);
            slot.nextPos++;
            slot.nextRow++;
        }
        return !state.cancelRequested(slot);
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
