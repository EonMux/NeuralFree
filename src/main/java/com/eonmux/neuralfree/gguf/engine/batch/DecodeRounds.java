package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.SequenceRun;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.VerifyRound;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntUnaryOperator;

/**
 * The decode rounds of a {@link BatchEngine} step: a running sequence's armed token, plus the drafter's proposal
 * when the engine speculates (vLLM's {@code scheduled_spec_decode_tokens}, llama.cpp server's per-slot
 * speculative loop), and the settling of a verify round once its rows are scored. The step's proposals are drafted
 * in one {@link BatchDrafter#draftBatch} call over every sequence that decodes in it ({@link #draftStep}), so a
 * model drafter runs every sequence's draft in the same forward batches. Each sequence then verifies through its
 * own sampler chain: by {@code MATCH} — it keeps the longest draft prefix its chain would have sampled, the rule of
 * llama.cpp's {@code common_sampler_sample_and_accept_n} — or by the engine's other verify mode (vLLM's
 * {@code RejectionSampler}, block verification, FLy, synthetic acceptance) through a {@link VerifyRound} over the
 * sequence's rows, its own uniform stream and the drafter's proposal distributions ({@link BatchDrafter#draftProbs}).
 * A proposal is cut before its first token the sequence's grammar refuses
 * ({@code Sampler#admitted}, vLLM's {@code validate_tokens}), as {@code SpeculativeDecoder} cuts it, so a
 * constrained sequence verifies no row its grammar could never confirm. A {@code MATCH} round whose chain keeps its
 * candidates inside a device top-k cone walks its rows' device candidates rather than whole rows (the step reads
 * them back as a single-row decode's, {@code StepExecutor}), drawing the same tokens.
 *
 * <p>In tree verify ({@code neuralfree.draft.tree}, the batch form of SpecPilot's ragged verify, arXiv 2611.02501)
 * the step's draft call asks {@link BatchDrafter#draftTreeBatch} for a draft tree per sequence, the verify budget
 * ({@link TreeBudget}) cuts each tree to its node prefix, and the kept trees of the step verify in one masked
 * forward: every tree row runs under its ancestor-or-self mask at its depth rope position beside the step's other
 * rows (plain decodes and admitted prefill chunks), and the walk samples through the tree — the pending row, then
 * every matched node's row — keeping the path the target names, the batch form of
 * {@code SpeculativeDecoder}'s tree round. The commit compacts the accepted path onto consecutive positions
 * ({@code moveRows}, the host token record retokenized after), the rows above leave, and the round cost the step's
 * one shared forward. A tree verify runs by {@code MATCH} only (the walk samples through the chain), and a
 * watermarked request is refused at submission ({@code BatchEngine#requireWatermarkSupport}); a grammar-constrained
 * request keeps its tree, the target's grammar governing every sample the walk takes, so the round accepts no token
 * the sequence could not have sampled.
 */
final class DecodeRounds {

    /** Rows one verify step's shared batch holds at most: the per-row ancestor mask's reach. */
    static final int MAX_TREE_ROWS = 64;

    private final BatchState state;
    private final SlotLifecycle lifecycle;
    private final ContextShift shifts;
    /** The drafter proposing every round's tokens, or null for plain single-row decodes. */
    private final BatchDrafter drafter;
    private final BatchSpec spec;
    private final int ubatch;
    /** The backend whose row moves a tree commit's compaction queues onto the device. */
    private final ForwardBackend backend;

    DecodeRounds(BatchState state, SlotLifecycle lifecycle, ContextShift shifts, BatchDrafter drafter,
                 BatchSpec spec, int ubatch, ForwardBackend backend) {
        this.state = state;
        this.lifecycle = lifecycle;
        this.shifts = shifts;
        this.drafter = drafter;
        this.spec = spec;
        this.ubatch = ubatch;
        this.backend = backend;
    }

    /**
     * Drafts the proposals of the step's decoding sequences in one {@link BatchDrafter#draftBatch} call: every
     * running sequence past its prefill whose round can hold a draft of at least {@code draftMin} within the free
     * context after its pending token, one ubatch, {@code maxRows} (the step's token budget) and what is left of
     * {@code draftRows} (the rows the step leaves its drafts, handed out in {@code active} order) asks for one. Each
     * answer is validated ({@link #checkDraft}) and cut to the prefix the sequence's grammar admits after the
     * pending token, which its sampler already holds ({@code Sampler#admitted}, vLLM's {@code validate_tokens});
     * the cut proposal is what {@code draftMin} judges. A broken answer retires its sequence alone, before anything
     * was written, and a draft call that throws retires every sequence it drafted for (rare path, see the engine's
     * class doc). A proposal shorter than {@code draftMin} is not verified and the drafter hears
     * {@code accept(seq, 0)}.
     */
    void draftStep(List<Slot> active, int maxRows, int draftRows) {
        if (drafter == null) {
            return;
        }
        if (spec.treeVerify()) {
            draftTreeStep(active, maxRows, draftRows);
            return;
        }
        List<Slot> asking = new ArrayList<>();
        List<BatchDrafter.Request> requests = new ArrayList<>();
        int left = draftRows;
        for (Slot slot : active) {
            int k = Math.min(draftLimit(slot, maxRows), left);
            if (!slot.needsPrefill && k > 0 && k >= spec.draftMin()) {
                left -= k;
                asking.add(slot);
                requests.add(new BatchDrafter.Request(slot.seq, slot.pendingToken, k, slot.draftTemperature,
                        slot.draftRng, slot.run.sampler().watermarkConfig(),
                        slot.submission.prompt.nTokens()));
            }
        }
        if (requests.isEmpty()) {
            return;
        }
        int[][] drafts;
        try {
            drafts = drafter.draftBatch(requests);
            if (drafts == null || drafts.length != requests.size()) {
                throw new IllegalStateException("the batch drafter answered " + (drafts == null ? "no" : String
                        .valueOf(drafts.length)) + " proposals for " + requests.size() + " requests");
            }
        } catch (RuntimeException e) {
            for (Slot slot : asking) {
                lifecycle.retire(slot, null, false, e);
            }
            return;
        }
        for (int i = 0; i < drafts.length; i++) {
            settleProposal(asking.get(i), drafts[i], requests.get(i).limit());
        }
    }

    /**
     * The tree form of {@link #draftStep}: one {@link BatchDrafter#draftTreeBatch} call over every decoding
     * sequence, then the verify budget's cut. The step's tree rows share the logits batch with the plain decode
     * rows of the other sequences, so the budget's room is the mask-and-ubatch row cap over one row per decoding
     * sequence, and at most {@code draftRows} (the rows the step leaves its drafts); less room than a node per asking
     * sequence drafts nothing this step (the sequences decode plain, no
     * drafter call, no state to unwind). Each answer is validated, cut to the budget's per-request node count and
     * armed on its slot; a broken answer retires its sequence alone, a draft call that throws retires every
     * sequence of the call.
     */
    private void draftTreeStep(List<Slot> active, int maxRows, int draftRows) {
        List<Slot> asking = new ArrayList<>();
        List<BatchDrafter.Request> requests = new ArrayList<>();
        int decodingRows = 0;
        for (Slot slot : active) {
            if (slot.needsPrefill) {
                continue;
            }
            decodingRows++;
            int k = treeLimit(slot, maxRows);
            if (k > 0) {
                asking.add(slot);
                requests.add(new BatchDrafter.Request(slot.seq, slot.pendingToken, k, slot.draftTemperature,
                        slot.draftRng, slot.run.sampler().watermarkConfig(),
                        slot.submission.prompt.nTokens()));
            }
        }
        if (requests.isEmpty()) {
            return;
        }
        int room = Math.min(Math.min(MAX_TREE_ROWS, ubatch) - decodingRows, draftRows);
        if (room < requests.size()) {
            return; // the step holds no node row beyond the pending rows: plain decodes, nothing armed
        }
        int width = spec.treeWidth();
        int total = Math.min(spec.treeBudget().totalFor(requests.size(), width), room);
        TreeProposal[] answers;
        try {
            answers = drafter.draftTreeBatch(requests);
            if (answers == null || answers.length != requests.size()) {
                throw new IllegalStateException("the batch drafter answered " + (answers == null ? "no" : String
                        .valueOf(answers.length)) + " tree proposals for " + requests.size() + " requests");
            }
            for (TreeProposal answer : answers) {
                checkTree(answer);
            }
        } catch (RuntimeException e) {
            for (Slot slot : asking) {
                lifecycle.retire(slot, null, false, e);
            }
            return;
        }
        double[][] scores = new double[answers.length][];
        for (int i = 0; i < answers.length; i++) {
            scores[i] = answers[i].pathScores();
        }
        int[] kept = spec.treeBudget().cut(scores, total);
        for (int i = 0; i < answers.length; i++) {
            settleTree(asking.get(i), answers[i], kept[i], requests.get(i).limit());
        }
    }

    /** Validates one tree answer: a null proposal, or more nodes than the request asked, is a broken drafter. */
    private static void checkTree(TreeProposal answer) {
        if (answer == null) {
            throw new IllegalStateException("the batch drafter returned no tree proposal");
        }
    }

    /** Arms the cut tree of one answer on its slot, or drops it (the drafter hears {@code accept(seq, 0)}). */
    private void settleTree(Slot slot, TreeProposal answer, int kept, int k) {
        if (answer.tree().size() > k) {
            lifecycle.retire(slot, null, false, new IllegalStateException("the batch drafter proposed "
                    + answer.tree().size() + " tree nodes, at most " + k + " were asked"));
            return;
        }
        if (kept < 1 || kept < spec.draftMin()) {
            drafter.accept(slot.seq, 0);
            return;
        }
        slot.treeProposal = answer.cut(kept);
    }

    /**
     * The most nodes a tree round of {@code slot} may hold: {@code draftMax} within the tree width, the free
     * context after the pending token, one ubatch and {@code maxRows} (the verify stores {@code 1 + nodes} rows);
     * 0 when that leaves less than one node or less than {@code draftMin}.
     */
    private int treeLimit(Slot slot, int maxRows) {
        int k = Math.min(spec.draftMax(), slot.nCtx - slot.nextPos - 1);
        k = Math.min(k, spec.treeWidth());
        k = Math.min(k, ubatch - 1);
        k = Math.min(k, maxRows - 1);
        return k < 1 || k < spec.draftMin() ? 0 : k;
    }

    /** Validates one answer of {@link #draftStep} and arms it on its slot, or retires the slot for a broken one. */
    private void settleProposal(Slot slot, int[] draft, int k) {
        try {
            checkDraft(draft, k);
        } catch (RuntimeException e) {
            lifecycle.retire(slot, null, false, e);
            return;
        }
        draft = slot.run.sampler().admitted(draft);
        if (draft.length == 0 || draft.length < spec.draftMin()) {
            drafter.accept(slot.seq, 0);
            return;
        }
        slot.proposal = draft;
    }

    /**
     * The most tokens a round of {@code slot} may draft: {@code draftMax} within the free context after the pending
     * token, one ubatch, and {@code maxRows} (the verify stores {@code 1 + drafted} positions in one step); 0 when
     * that leaves less than one token or less than {@code draftMin}.
     */
    private int draftLimit(Slot slot, int maxRows) {
        int k = Math.min(spec.draftMax(), slot.nCtx - slot.nextPos - 1);
        k = Math.min(k, ubatch - 1);
        k = Math.min(k, maxRows - 1);
        return k < 1 || k < spec.draftMin() ? 0 : k;
    }

    /**
     * The step's decode piece of {@code slot}: its armed token at the next position plus its proposal of this step
     * cut to {@code maxRows - 1} tokens (the rows the step's token budget leaves); a proposal cut below
     * {@code draftMin}, and a slot without one, decode the plain single row. A tree proposal runs instead of the
     * chain proposal: its piece carries the cut tree, its rows the pending token then the nodes in node order, and
     * the router walks the tree over them.
     */
    Piece decodePiece(Slot slot, int maxRows) {
        TreeProposal tree = slot.treeProposal;
        slot.treeProposal = null;
        if (tree != null) {
            TreeProposal cut = tree.cut(Math.min(tree.tree().size(), maxRows - 1));
            if (cut.tree().size() > 0 && cut.tree().size() >= spec.draftMin()) {
                return treePiece(slot, cut);
            }
            drafter.accept(slot.seq, 0);
            return new Piece(slot, true, 1, slot.nextPos, slot.nextRow,
                    new int[] {slot.pendingToken}, true, false);
        }
        int[] round = new int[] {slot.pendingToken};
        int[] draft = slot.proposal;
        slot.proposal = null;
        if (draft != null) {
            int n = Math.min(draft.length, maxRows - 1);
            if (n >= 1 && n >= spec.draftMin()) {
                round = new int[n + 1];
                round[0] = slot.pendingToken;
                System.arraycopy(draft, 0, round, 1, n);
            } else {
                drafter.accept(slot.seq, 0);
            }
        }
        return new Piece(slot, true, round.length, slot.nextPos, slot.nextRow, round, true, false);
    }

    /** The tree round's piece of {@code slot}: the pending token then the cut tree's nodes in node order. */
    private static Piece treePiece(Slot slot, TreeProposal cut) {
        List<DraftTree.Node> nodes = cut.tree().nodes();
        int[] round = new int[nodes.size() + 1];
        round[0] = slot.pendingToken;
        for (int i = 0; i < nodes.size(); i++) {
            round[i + 1] = nodes.get(i).token();
        }
        return new Piece(slot, true, round.length, slot.nextPos, slot.nextRow, round, true, false, cut);
    }

    /** Drops the proposal of a slot the step did not schedule: the drafter hears {@code accept(seq, 0)}. */
    void discardProposal(Slot slot) {
        if (slot.proposal != null) {
            slot.proposal = null;
            drafter.accept(slot.seq, 0);
        }
        if (slot.treeProposal != null) {
            slot.treeProposal = null;
            drafter.accept(slot.seq, 0);
        }
    }

    /**
     * Validates a drafter's answer to a request for at most {@code k} tokens: a null array, more tokens than asked
     * or a negative id is an {@link IllegalStateException}, as {@code SpeculativeDecoder} reports a broken drafter.
     */
    private static void checkDraft(int[] draft, int k) {
        if (draft == null) {
            throw new IllegalStateException("the batch drafter returned no proposal array");
        }
        if (draft.length > k) {
            throw new IllegalStateException("the batch drafter proposed " + draft.length + " tokens, at most " + k
                    + " were asked");
        }
        for (int token : draft) {
            if (token < 0) {
                throw new IllegalStateException("the batch drafter proposed negative token id " + token);
            }
        }
    }

    /** Outcome of one verify round. */
    private static final class Verified {
        /** Draft tokens confirmed before the round ended. */
        final int accepted;
        /** The last emitted token (the target's next token), -1 when the round ended on a cancellation. */
        final int token;
        /** CANCELLED from the round's poll, a token-level stop from the run's offer, or null to continue. */
        final StopReason reason;
        /** The accepted path's node indices, {@code accepted} long; a chain round's is empty. */
        final int[] path;

        Verified(int accepted, int token, StopReason reason) {
            this(accepted, token, reason, new int[0]);
        }

        Verified(int accepted, int token, StopReason reason, int[] path) {
            this.accepted = accepted;
            this.token = token;
            this.reason = reason;
            this.path = path;
        }
    }

    /**
     * Verifies the slot's round over its scored rows and settles it: by {@code MATCH} ({@link #matchWalk}) or by
     * the slot's verifier ({@link #verifierRound}). The unverified tail then leaves the paged sequence, the
     * confirmed drafts and the pending token join the fed history, and the drafter hears the outcome. A stop inside
     * the round (end of generation, stop string, budget, cancel) leaves the sequence in the ordinary rejection
     * state: the verified positions past the stop are dropped, never left as an overshot state.
     */
    void verify(Slot slot, int[] round, float[][] rows, int base) {
        if (state.cancelRequested(slot)) {
            lifecycle.finish(slot, StopReason.CANCELLED);
            return;
        }
        Verified v = slot.verifier == null
                ? matchWalk(slot, round, i -> sample(slot, rows[base + i]))
                : verifierRound(slot, round, rows, base);
        settleRound(slot, round, v);
    }

    /**
     * {@link #verify(Slot, int[], float[][], int)} by {@code MATCH} over the round's rows of a device top-k readback,
     * rows {@code topBase .. topBase + round.length) of {@code top}: each row samples through the chain over its
     * candidates, which is the token the chain samples over the whole row (llama.cpp backend sampling over the
     * verified rows; the single-stream {@code SpeculativeDecoder}'s sampled verify).
     *
     * @param prefiltered whether the readout applied each row's logit prefilter before its top-k
     * @throws IllegalStateException when the slot verifies through a verifier, which reads whole rows
     */
    void verify(Slot slot, int[] round, RowTopK top, int topBase, boolean prefiltered) {
        if (slot.verifier != null) {
            throw new IllegalStateException("a verify round under " + slot.verifier
                    + " reads the processed distributions of whole rows, not device candidates");
        }
        if (state.cancelRequested(slot)) {
            lifecycle.finish(slot, StopReason.CANCELLED);
            return;
        }
        Verified v = matchWalk(slot, round, i -> sampleTopK(slot, top, topBase + i, prefiltered));
        settleRound(slot, round, v);
    }

    /** A verified chain round's bookkeeping: counters, the tail removal, the drafter's outcome, the fed rows. */
    private void settleRound(Slot slot, int[] round, Verified v) {
        int k = round.length - 1;
        state.stats.specRounds++;
        state.stats.specDrafted += k;
        state.stats.specAccepted += v.accepted;
        int drop = k - v.accepted;
        if (drop > 0) {
            // the memory stores positions nextPos..nextPos+k; the round keeps nextPos..nextPos+accepted
            state.sequences.remove(slot.seq, slot.nextPos + v.accepted + 1, -1);
        }
        drafter.accept(slot.seq, v.accepted);
        feed(slot, round, v.accepted);
        settleAfter(slot, v);
    }

    /**
     * Verifies the slot's tree round over its scored rows and settles it: the walk keeps the path the target's
     * samples name ({@link #treeWalk}), the commit compacts that path onto the consecutive positions after the
     * pending row ({@link #commitTree}), the rows above leave, and the pending token and the path's tokens join the
     * fed history. The drafter hears {@code accept(seq, acceptedPathNodes)}.
     */
    void verifyTree(Slot slot, TreeProposal proposal, float[][] rows, int base) {
        if (state.cancelRequested(slot)) {
            lifecycle.finish(slot, StopReason.CANCELLED);
            return;
        }
        int n = proposal.tree().size();
        Verified v = treeWalk(slot, proposal.tree(), rows, base);
        state.stats.specRounds++;
        state.stats.specDrafted += n;
        state.stats.specAccepted += v.accepted;
        commitTree(slot, proposal.tree(), v);
        drafter.accept(slot.seq, v.accepted);
        feedTree(slot, proposal.tree(), v);
        settleAfter(slot, v);
    }

    /** A settled round's tail: a stop finishes or retires the sequence, else the bonus token arms for the next step. */
    private void settleAfter(Slot slot, Verified v) {
        if (v.reason != null) {
            if (v.reason == StopReason.CANCELLED) {
                lifecycle.finish(slot, StopReason.CANCELLED);
            } else {
                // the run recorded the token-level stop: the result is complete
                lifecycle.retire(slot, slot.run.result(), false, null);
            }
            return;
        }
        slot.pendingToken = v.token;
        shifts.shiftOrFinish(slot);
    }

    /**
     * Feeds a settled round's kept rows into the slot, before a stop can retire it: the pending token and the
     * {@code accepted} drafts join the fed history and the position and row cursors move past them.
     */
    private static void feed(Slot slot, int[] round, int accepted) {
        slot.fed(slot.pendingToken);
        for (int i = 1; i <= accepted; i++) {
            slot.fed(round[i]);
        }
        slot.nextPos += 1 + accepted;
        slot.nextRow += 1 + accepted;
    }

    /** Feeds a settled tree round's kept path: the pending token and the accepted path's node tokens, in path order. */
    private static void feedTree(Slot slot, DraftTree tree, Verified v) {
        slot.fed(slot.pendingToken);
        for (int d = 1; d <= v.accepted; d++) {
            slot.fed(tree.nodes().get(v.path[d - 1]).token());
        }
        slot.nextPos += 1 + v.accepted;
        slot.nextRow += 1 + v.accepted;
    }

    /**
     * Compacts a settled tree round's kept path: the path's rows move from their verify slots onto the consecutive
     * positions after the pending row (a node roped at {@code pos + depth} lands on the position it already roped
     * at, so no rotation replants), the host token record follows the kept rows, and the tree rows past the path
     * leave. The moves run in list order, sources above their destinations, ascending — the memory's compaction
     * contract — and the backend carries them to the device before the tail removal frees the rows they read.
     */
    private void commitTree(Slot slot, DraftTree tree, Verified v) {
        int pos = slot.nextPos;
        if (v.accepted > 0) {
            List<DraftTree.Node> nodes = tree.nodes();
            int[] src = new int[v.accepted];
            int[] dst = new int[v.accepted];
            int moves = 0;
            for (int d = 1; d <= v.accepted; d++) {
                int s = pos + v.path[d - 1] + 1;
                int t = pos + d;
                if (s != t) {
                    src[moves] = s;
                    dst[moves] = t;
                    moves++;
                }
            }
            if (moves > 0) {
                state.sequences.moveRows(slot.seq, Arrays.copyOf(src, moves), Arrays.copyOf(dst, moves));
                backend.carryRowMoves();
                // the record holds the nodes' tokens at their verify slots; the compacted path's entries follow
                int[] positions = new int[v.accepted + 1];
                int[] tokens = new int[v.accepted + 1];
                positions[0] = pos;
                tokens[0] = slot.pendingToken;
                for (int d = 1; d <= v.accepted; d++) {
                    positions[d] = pos + d;
                    tokens[d] = nodes.get(v.path[d - 1]).token();
                }
                state.sequences.sequence(slot.seq).retokenize(positions, tokens);
            }
        }
        if (tree.size() > v.accepted) {
            state.sequences.remove(slot.seq, pos + v.accepted + 1, -1);
        }
    }

    /** One verify row through the slot's chain, its time counted as host sampling. */
    private int sample(Slot slot, float[] row) {
        long t0 = System.nanoTime();
        int token = slot.run.next(row);
        state.stats.hostSampleNanos += System.nanoTime() - t0;
        return token;
    }

    /** One verify row's device candidates through the slot's chain, accepted as {@link #sample} accepts a row's. */
    private int sampleTopK(Slot slot, RowTopK top, int row, boolean prefiltered) {
        long t0 = System.nanoTime();
        int token = slot.run.sampleTopK(top, row, prefiltered);
        slot.run.accept(token);
        state.stats.hostSampleNanos += System.nanoTime() - t0;
        state.stats.deviceSampledRows++;
        return token;
    }

    /**
     * {@code MATCH}: the slot's verify rows sampled in order through its own chain — row {@code i} answers the
     * token after {@code [pendingToken, drafts 0..i-1]}, so the walk continues while the chain's sample equals
     * draft {@code i}, and the first mismatching sample, or the last row's, is the target's next token.
     *
     * @param rowToken the chain's sample of verify row {@code i}, accepted into the chain, asked in ascending order
     */
    private Verified matchWalk(Slot slot, int[] round, IntUnaryOperator rowToken) {
        int k = round.length - 1;
        int accepted = 0;
        int token = -1;
        int i = 0;
        while (true) {
            if (i > 0 && state.cancelRequested(slot)) {
                return new Verified(accepted, -1, StopReason.CANCELLED);
            }
            token = rowToken.applyAsInt(i);
            state.stats.generatedTokens++;
            SequenceRun.Step step = slot.run.offer(token);
            if (step.stopped()) {
                return new Verified(accepted, token, step.reason());
            }
            if (i == k || token != round[i + 1]) {
                return new Verified(accepted, token, null);
            }
            accepted++;
            i++;
        }
    }

    /**
     * The tree walk: the pending row's sample offered, then the node the sample names among the predecessor's
     * children, then that node's row's sample, and so on — the first depth the target's sample finds no child of
     * (or a stop, or a cancellation) ends the round, the last sample being the target's next token. The path is the
     * target's own choice among the cut tree's node prefixes, so the round emits exactly the tokens the target
     * would have sampled decoding alone.
     */
    private Verified treeWalk(Slot slot, DraftTree tree, float[][] rows, int base) {
        List<DraftTree.Node> nodes = tree.nodes();
        int[] path = new int[nodes.size()];
        int accepted = 0;
        int parent = -1; // the node index the next candidates extend; -1 names the pending row's children
        int row = base;  // the predecessor row whose logits the next sample reads
        int token;
        while (true) {
            if (accepted > 0 && state.cancelRequested(slot)) {
                return new Verified(accepted, -1, StopReason.CANCELLED, path);
            }
            token = sample(slot, rows[row]);
            state.stats.generatedTokens++;
            SequenceRun.Step step = slot.run.offer(token);
            if (step.stopped()) {
                return new Verified(accepted, token, step.reason(), path);
            }
            int matched = -1;
            for (int i = 0; i < nodes.size(); i++) {
                DraftTree.Node nd = nodes.get(i);
                if (nd.parent() == parent && nd.token() == token) {
                    matched = i;
                    break;
                }
            }
            if (matched < 0) {
                return new Verified(accepted, token, null, path);
            }
            path[accepted++] = matched;
            parent = matched;
            row = base + matched + 1;
        }
    }

    /**
     * The slot's verifier over its rows ({@link VerifyRound}): the processed distributions under the sequence's
     * chain, the drafter's proposal distributions of the round (one-hot where it has none), and the slot's own
     * uniform stream, with every emission offered through the sequence's run. The proposal distributions stop at
     * the round's kept draft count: the grammar cut and the step's row budget keep a prefix of the drafter's
     * proposal, and the round verifies no position past it (vLLM cuts {@code draft_probs} with the proposal).
     */
    private Verified verifierRound(Slot slot, int[] round, float[][] rows, int base) {
        int[] draft = Arrays.copyOfRange(round, 1, round.length);
        VerifyRound vr = VerifyRound.forRun(slot.run, () -> state.cancelRequested(slot), draft, rows, base,
                slot.verifyRng, position -> position < draft.length ? drafter.draftProbs(slot.seq, position) : null);
        slot.verifier.verify(vr);
        vr.finish(slot.verifier);
        state.stats.generatedTokens += vr.emitted();
        return new Verified(vr.accepted(), vr.next(), vr.reason());
    }
}
