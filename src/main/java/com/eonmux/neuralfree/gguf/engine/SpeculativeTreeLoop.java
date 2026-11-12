package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.engine.lora.LoraPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The multi-branch (tree) path of {@link SpeculativeDecoder}: the refusals a proposal tree cannot serve, and the
 * generation loop that drafts a {@link DraftTree} every round, verifies it over the target's sequences — one
 * fixed-width forward under the tree mask for a single-pass drafter, the exact recurrent round, the linear
 * token-match walk, one forward per depth, or a two-level round through the quantized view — and commits the
 * accepted path. Owned by the decoder that built it; one generation runs at a time, on the backend's thread.
 */
final class SpeculativeTreeLoop {

    private final SpeculativeDecoder decoder;

    SpeculativeTreeLoop(SpeculativeDecoder decoder) {
        this.decoder = decoder;
    }

    /**
     * Whether this decoder verifies a multi-branch proposal tree: a tree drafter. The verify runs on the target's
     * paged sequences and on a default single-user contiguous memory alike — one fixed-width forward of the one
     * target sequence under the tree mask.
     */
    boolean treeMode() {
        return decoder.drafter instanceof TreeDraftModel && ((TreeDraftModel) decoder.drafter).branches() > 1;
    }

    /**
     * Refuses the request and configuration shapes the tree round cannot serve, naming each: a recurrent target
     * the round does not cover (see {@link #refuseRecurrentTree}), compressed memory, sliding-window trims, a
     * ubatch too small for one verify forward or a backend that refuses the batch the masked verify step runs
     * through, the context shift, per-request adapters, a verify mode other than {@code MATCH} and a logprobs
     * capture ({@code logprobs}, {@code logprob_token_ids}). A tree round moves the accepted rows onto their
     * committed positions and rolls the rejected verify rows out of the target sequence; those settings either
     * rewind state the round cannot restore (compressed memory, the shift) or read rows the round does not expose
     * (logprobs). A multi-section rope target ropes every row from its depth positions in all sections, so it
     * verifies like a single-section one.
     */
    void refuseTree(GenerationRequest request, ModelMemory memory, LoraPlan plan,
                            GenerationRequest effective) {
        if (memory.hasRecurrentState()) {
            refuseRecurrentTree(memory, (TreeDraftModel) decoder.drafter);
        }
        if (memory.plan().hasCompressed()) {
            throw new UnsupportedOperationException("multi-branch (tree) speculative decoding rewinds the target"
                    + " rows of a round, which a memory with compressed caches cannot restore");
        }
        com.eonmux.neuralfree.gguf.memory.paged.PagedSequences pages = memory.pagedSequences();
        if (pages != null) {
            for (int g = 0; g < pages.groups(); g++) {
                if (pages.trims(g)) {
                    throw new UnsupportedOperationException("multi-branch (tree) speculative decoding rewinds the"
                            + " target rows of a round, which a sliding-window memory trims away");
                }
            }
        }
        if (memory.plan().ubatch() < 2) {
            throw new IllegalArgumentException("multi-branch (tree) speculative decoding verifies the pending token"
                    + " and its tree nodes in one forward of at least two rows, more than the ubatch "
                    + memory.plan().ubatch() + "; raise neuralfree.ubatch");
        }
        boolean singlePass = ((TreeDraftModel) decoder.drafter).singlePassVerify();
        if (pages == null && !singlePass && decoder.backend.batchRefusal() != null) {
            throw new UnsupportedOperationException("multi-branch (tree) speculative decoding verifies one target"
                    + " forward per depth over page-shared sequences, which needs paged memory (neuralfree.parallel"
                    + " > 1 or neuralfree.kv.paged=true); the drafter " + decoder.drafter + " drafts no single-pass tree");
        }
        if (pages == null && singlePass && decoder.backend.batchRefusal() != null) {
            throw new UnsupportedOperationException("the single-pass tree verify of drafter " + decoder.drafter
                    + " runs its masked verify step through " + ForwardBackend.class.getSimpleName()
                    + ".forwardBatch, which this backend refuses: " + decoder.backend.batchRefusal());
        }
        if (decoder.contextShift) {
            throw new IllegalArgumentException("multi-branch (tree) speculative decoding runs over paged sequences"
                    + " and cannot shift the context: clear neuralfree.context.shift");
        }
        if (!plan.isDefault()) {
            throw new UnsupportedOperationException("lora: multi-branch (tree) speculative decoding runs the target"
                    + " and the draft model without per-request adapters (" + plan + ")");
        }
        if (decoder.verifyMode != VerifyMode.MATCH) {
            throw new IllegalArgumentException("neuralfree.draft.verify=" + decoder.verifyMode + " does not verify a"
                    + " multi-branch proposal tree; use match (the target samples and the round keeps the drafted"
                    + " tokens that sample, llama.cpp's token-match acceptance over the tree)");
        }
        if (request.wantsLogprobs()) {
            throw new UnsupportedOperationException("logprobs: multi-branch (tree) speculative decoding verifies"
                    + " per-depth target rows and does not score them; clear logprobs / logprob_token_ids");
        }
        if (effective.sampler().temperature() < 0f) {
            throw new IllegalArgumentException("sampling temperature must not be negative, got "
                    + effective.sampler().temperature());
        }
        if (decoder.quantizedBackend != null && !effective.sampler().isGreedy()) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_TLV + " serves greedy requests only:"
                    + " the quantized target view verifies a proposal with plain argmax and cannot reproduce a"
                    + " sampler's draw, so a sampled request's tokens would diverge from the target-only loop;"
                    + " request greedy decoding (or temperature=0), or clear " + SpeculativeOptions.PROP_TLV);
        }
    }

    /**
     * Lifts the tree round's recurrent refusal for the covered families and refuses the rest by name: the
     * property must not opt out ({@code neuralfree.draft.tree.recurrent=false}), the two-level quantized view
     * must be off (its chain re-decodes target rows through a second backend, which rewinds the recurrent state
     * the exact round's replay builds on), the drafter must draft a single-pass tree (the per-depth rounds
     * re-decode the target row by row, which rewinds recurrent state), and the backend must cover every recurrent
     * layer's mixer, which its probe names (an uncovered mixer kind, a non-paged GPU memory, a non-linear layer
     * schedule).
     */
    void refuseRecurrentTree(ModelMemory memory, TreeDraftModel tree) {
        if (!decoder.treeRecurrent) {
            throw new UnsupportedOperationException("the tree verify through this target's recurrent layers is"
                    + " disabled: set " + SpeculativeOptions.PROP_TREE_RECURRENT + "=true, or clear"
                    + " neuralfree.draft.tree to speculate without a tree");
        }
        if (decoder.quantizedBackend != null) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_TLV + " does not combine with a target"
                    + " with recurrent layers: the quantized view's chain re-decodes target rows, which rewinds the"
                    + " recurrent state the exact round's replay builds on; clear " + SpeculativeOptions.PROP_TLV);
        }
        if (!tree.singlePassVerify()) {
            throw new UnsupportedOperationException("multi-branch (tree) speculative decoding verifies a target"
                    + " with recurrent layers in one masked step, which the drafter " + tree + " does not draft:"
                    + " its per-depth rounds re-decode the target row by row, which rewinds recurrent state; use a"
                    + " drafter whose single-pass tree covers the model (neuralfree.draft.tree=true)");
        }
        String refusal = decoder.backend.recurrentTreeRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("multi-branch (tree) speculative decoding verifies a target"
                    + " with recurrent layers through per-node recurrent state, which this backend does not cover:"
                    + " " + refusal);
        }
    }

    /**
     * Generates with a multi-branch proposal: the prompt prefills target sequence 0, and every round drafts a
     * {@link DraftTree}, verifies it (one fixed-width forward of the one sequence under the tree mask for a
     * single-pass drafter, depth by depth over the paged sequences otherwise) and commits the accepted path.
     * Each generation resets the memory (the scheme keeps no reusable prompt prefix).
     */
    GenerationResult runTree(int[] prompt, ModelMemory memory, int nCtx, Sampler sampler,
                                     StopCriteria.Session session, GenerationRequest effective, String rendered) {
        SequenceMemory seqs = memory.sequences();
        TreeDraftModel tree = (TreeDraftModel) decoder.drafter;
        // a recurrent target verifies through the recurrent tree round: refuseTree lifted the refusal only when
        // the backend covers every recurrent layer's mixer and the drafter drafts a single-pass tree
        boolean recurrent = memory.hasRecurrentState() && decoder.backend.recurrentTreeRefusal() == null;
        // the single-pass round's acceptance walks on the device when the request asked for it, the sampler
        // reduces to the raw logits' argmax (the kernel sees no sampler mask or draw) and the backend walks there
        boolean deviceAccept = decoder.verifyOnDevice && sampler.argmaxOnly()
                && decoder.backend.treeAcceptOnDevice();
        // the verify runs one forward of width + 1 rows: the fixed width of neuralfree.draft.tree.nodes, clamped
        // once per generation to the ubatch budget, so the step keeps one stable shape throughout the run
        int width = Math.min(decoder.treeNodes, memory.plan().ubatch() - 1);
        long t0 = System.nanoTime();
        // the generation-start protocol: the tree round prefills the whole prompt (no prefix reuse), so the full
        // prefill covers whatever replay window the drafter asks over its stored rows
        int replay = tree.beginGeneration(prompt.length);
        if (replay < 0 || replay > prompt.length) {
            throw new IllegalStateException("the drafter asked to replay " + replay + " of the " + prompt.length
                    + " prompt positions");
        }
        decoder.backend.reset();
        float[] logits;
        try {
            logits = prefillTree(prompt);
        } catch (MemoryFullException e) {
            sampler.acceptPrompt(prompt, 0, prompt.length);
            long t1 = System.nanoTime();
            return session.finish(StopReason.CONTEXT_FULL, GenerationResult.Timings.of(prompt.length, t1 - t0, 0,
                    System.nanoTime() - t1));
        }
        long t1 = System.nanoTime();
        sampler.acceptPrompt(prompt, 0, prompt.length);
        SplittableRandom rng = new SplittableRandom(sampler.seed());
        int rounds = 0;
        int drafted = 0;
        int accepted = 0;
        int sampled = 0;
        int token = -1;
        StopReason reason;
        if (session.cancelled()) {
            reason = StopReason.CANCELLED;
        } else {
            token = sampler.sample(logits);
            sampler.accept(token);
            sampled++;
            reason = session.offer(token);
        }
        while (reason == null) {
            if (session.cancelled()) {
                reason = StopReason.CANCELLED;
                break;
            }
            int pos = seqs.sequence(0).nPast();
            if (pos >= nCtx) {
                reason = StopReason.CONTEXT_FULL;
                break;
            }
            int k = decoder.draftLength(memory, pos, nCtx, !recurrent);
            float temperature = sampler.greedy() ? 0f : effective.sampler().temperature();
            DraftTree proposal;
            try {
                // the single-pass verify holds the fixed width; the linear and per-depth rounds keep the ubatch
                // budget they verify under
                int maxRows = tree.singlePassVerify() ? width : memory.plan().ubatch();
                proposal = k <= 0 ? new DraftTree(java.util.Collections.emptyList())
                        : tree.draftTree(token, k, sampler.hasConstraints() ? 1 : tree.branches(),
                                maxRows, temperature, rng);
            } catch (MemoryFullException e) {
                reason = StopReason.CONTEXT_FULL;
                break;
            }
            if (proposal.size() == 0 || proposal.depth() < decoder.draftMin) {
                tree.acceptTree(0);
                try {
                    token = decodeTree(token, pos, sampler);
                } catch (MemoryFullException e) {
                    reason = StopReason.CONTEXT_FULL;
                    break;
                }
                sampled++;
                reason = session.offer(token);
                continue;
            }
            SpeculativeDecoder.Verified v;
            // a two-level round drafts twice: composedOut carries the tokens the second draft added to the
            // proposal the exact round verifies, on top of the first draft's own
            int[] composedOut = {0};
            try {
                v = decoder.quantizedBackend != null
                        ? twoLevelRound(seqs, tree, proposal, token, pos, sampler, session, width, temperature, rng,
                                composedOut, recurrent, deviceAccept)
                        : recurrent
                            ? TreeVerify.round(decoder.backend, seqs, proposal, token, pos, sampler, session, width,
                                    decoder.ropeType, decoder.textPositions, tree, true, deviceAccept)
                            : proposal.linear()
                                ? verifyLinearTree(seqs, proposal, token, pos, sampler, session)
                                : tree.singlePassVerify()
                                    ? TreeVerify.round(decoder.backend, seqs, proposal, token, pos, sampler, session, width,
                                            decoder.ropeType, decoder.textPositions, tree, false, deviceAccept)
                                    : verifyBranchingTree(seqs, proposal, token, pos, sampler, session,
                                            tree.branches());
            } catch (MemoryFullException e) {
                reason = StopReason.CONTEXT_FULL;
                break;
            }
            rounds++;
            drafted += proposal.size() + composedOut[0];
            accepted += v.accepted;
            sampled += v.emitted;
            token = v.next;
            reason = v.reason;
        }
        long t2 = System.nanoTime();
        decoder.statsSink.accept(new SpeculativeDecoder.Stats(rounds, drafted, accepted));
        return session.finish(reason, GenerationResult.Timings.of(prompt.length, t1 - t0, sampled, t2 - t1));
    }

    /**
     * One round of two-level verification: the quantized view verifies the proposal greedily (its chain and
     * bonus only steer a second draft, nothing commits), the committed chain plays back once through the view so
     * the drafter's cache receives the chain rows and the second draft anchors below the chain, and the exact
     * target verifies the composed proposal in one pass and commits — the emitted tokens all come from the exact
     * weights. The view's passes run inside the view's Q scope: on a nested W12 target the scope routes the
     * target's own projections through their 8-bit Q view, on a copied target it is a no-op, and either way it
     * closes before the exact round. The view's rows leave the sequence before the exact round: the exact verify
     * re-stores whatever it accepts, and a wrong chain costs only the width it occupied.
     */
    private SpeculativeDecoder.Verified twoLevelRound(SequenceMemory seqs, TreeDraftModel tree, DraftTree proposal, int token, int pos,
                                   Sampler sampler, StopCriteria.Session session, int width, float temperature,
                                   SplittableRandom rng, int[] composedOut, boolean recurrent, boolean deviceAccept)
            throws MemoryFullException {
        if (!sampler.greedy()) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_TLV + " serves greedy requests only: the"
                    + " quantized view verifies with plain argmax and cannot reproduce a sampler's draw; disable"
                    + " temperature for this request");
        }
        TwoLevelVerify.Chain chain;
        int chainLen;
        try (QuantizedView.QScope qScope = decoder.quantizedView.openQScope()) {
            chain = TwoLevelVerify.round(decoder.quantizedBackend, proposal, token, pos, width, decoder.ropeType, tree);
            // keep the bonus's subtree inside the fixed width: the chain may run at most width - 1 nodes deep
            chainLen = Math.min(chain.tokens.length, Math.max(1, width - 1));
            TwoLevelVerify.commitChain(decoder.quantizedBackend, seqs, token, pos, chain.tokens, chainLen);
        }
        int budget = width - chainLen;
        DraftTree sub;
        try {
            sub = budget <= 0 ? new DraftTree(java.util.Collections.emptyList())
                    : tree.draftTree(chain.bonus, budget, sampler.hasConstraints() ? 1 : tree.branches(), budget,
                            temperature, rng);
        } catch (MemoryFullException e) {
            seqs.remove(0, pos, -1);
            throw e;
        }
        DraftTree composed = TwoLevelVerify.compose(chain, chainLen, sub);
        composedOut[0] = composed.size();
        // the view's rows leave: the exact verify re-stores whatever it accepts with exact weights
        seqs.remove(0, pos, -1);
        return TreeVerify.round(decoder.backend, seqs, composed, token, pos, sampler, session, width, decoder.ropeType, decoder.textPositions,
                tree, recurrent, deviceAccept);
    }

    /** Prefills {@code prompt} into target sequence 0 through one batched forward and returns the last row's logits. */
    private float[] prefillTree(int[] prompt) {
        ForwardBatch.Builder b = ForwardBatch.builder(prompt.length);
        for (int i = 0; i < prompt.length; i++) {
            b.add(prompt[i], i, 0, i == prompt.length - 1);
        }
        float[][] rows = decoder.backend.forwardBatch(b.build());
        if (rows.length != 1) {
            throw new IllegalStateException("the prompt prefill returned " + rows.length + " logits rows for one"
                    + " requested row");
        }
        return rows[0];
    }

    /** Decodes {@code token} at {@code pos} on target sequence 0 and samples the next token. */
    private int decodeTree(int token, int pos, Sampler sampler) {
        ForwardBatch batch = ForwardBatch.builder(1).add(token, pos, 0, true).build();
        float[][] rows = decoder.backend.forwardBatch(batch);
        int next = sampler.sample(rows[0]);
        sampler.accept(next);
        return next;
    }

    /**
     * Verifies a linear proposal over one target sequence: the target samples a token after the pending token and
     * after each drafted token, and the round keeps the longest prefix whose samples equal the drafted tokens
     * (llama.cpp's token-match acceptance, its {@code common_sampler_sample_and_accept_n}; a greedy chain samples
     * argmax, a sampling chain draws from the target). The unconfirmed tail leaves the sequence.
     */
    private SpeculativeDecoder.Verified verifyLinearTree(SequenceMemory seqs, DraftTree proposal, int token, int pos, Sampler sampler,
                                      StopCriteria.Session session) {
        int[] draft = proposal.linearTokens();
        if (sampler.hasConstraints()) {
            draft = sampler.admitted(draft);
        }
        int k = draft.length;
        if (k < Math.max(1, decoder.draftMin)) {
            // the constraints left no chain worth verifying: decode the pending token's target row on its own
            int next = decodeTree(token, pos, sampler);
            return new SpeculativeDecoder.Verified(0, 1, next, session.offer(next));
        }
        int[] batchTokens = new int[k + 1];
        batchTokens[0] = token;
        System.arraycopy(draft, 0, batchTokens, 1, k);
        ForwardBatch.Builder b = ForwardBatch.builder(k + 1);
        for (int i = 0; i <= k; i++) {
            b.add(batchTokens[i], pos + i, 0, true);
        }
        float[][] rows = decoder.backend.forwardBatch(b.build());
        SpeculativeDecoder.Verified v = SpeculativeDecoder.matchVerify(sampler, session, draft,
                i -> sampler.sample(rows[i]));
        int drop = k - v.accepted;
        if (drop > 0) {
            seqs.remove(0, pos + v.accepted + 1, -1);
        }
        ((TreeDraftModel) decoder.drafter).acceptTree(v.accepted);
        return v;
    }

    /**
     * Verifies a branching proposal one target forward per depth: each depth's nodes run on their own page-shared
     * target sequences, the target samples after the accepted node of the previous depth, and the round keeps every
     * node at the depth whose token the sample equals (llama.cpp's {@code common_sampler_sample_and_accept_n} over
     * the tree). The first depth with no match emits the target's own token and ends the round; the accepted branch
     * becomes target sequence 0.
     */
    private SpeculativeDecoder.Verified verifyBranchingTree(SequenceMemory seqs, DraftTree proposal, int token, int pos,
                                         Sampler sampler, StopCriteria.Session session, int branches) {
        List<DraftTree.Node> nodes = proposal.nodes();
        float[] predecessor = decodeRow(token, pos, 0);
        List<Integer> frontier = java.util.Collections.emptyList();
        int acceptedDepth = 0;
        int emitted = 0;
        int next = -1;
        StopReason reason = null;
        int[] lastDepth = new int[branches];
        java.util.Arrays.fill(lastDepth, -1);
        // whether the last depth processed ended in a match: only then does the target sample the bonus token past
        // the accepted path; a mismatch already sampled the token that ends the round
        boolean advanced = false;
        for (int depth = 1; depth <= proposal.depth(); depth++) {
            List<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < nodes.size(); i++) {
                DraftTree.Node n = nodes.get(i);
                if (n.depth() == depth && (frontier.isEmpty() ? n.parent() == -1 : frontier.contains(n.parent()))) {
                    candidates.add(i);
                }
            }
            if (candidates.isEmpty()) {
                break;
            }
            for (int index : candidates) {
                DraftTree.Node n = nodes.get(index);
                int branch = n.branch();
                int parentBranch = n.parent() < 0 ? 0 : nodes.get(n.parent()).branch();
                if (lastDepth[branch] != depth - 1) {
                    if (branch != parentBranch) {
                        seqs.remove(branch, 0, -1);
                        seqs.copy(parentBranch, branch, 0, -1);
                    }
                    lastDepth[branch] = depth - 1;
                }
            }
            ForwardBatch.Builder b = ForwardBatch.builder(candidates.size());
            for (int index : candidates) {
                DraftTree.Node n = nodes.get(index);
                b.add(n.token(), pos + depth, n.branch(), true);
            }
            float[][] rows = decoder.backend.forwardBatch(b.build());
            for (int index : candidates) {
                lastDepth[nodes.get(index).branch()] = depth;
            }
            next = sampler.sample(predecessor);
            sampler.accept(next);
            emitted++;
            reason = session.offer(next);
            if (reason != null) {
                break;
            }
            List<Integer> matched = new ArrayList<>();
            float[] nextPredecessor = null;
            for (int j = 0; j < candidates.size(); j++) {
                if (nodes.get(candidates.get(j)).token() == next) {
                    matched.add(candidates.get(j));
                    if (nextPredecessor == null) {
                        nextPredecessor = rows[j];
                    }
                }
            }
            if (matched.isEmpty()) {
                advanced = false;
                break;
            }
            frontier = matched;
            acceptedDepth = depth;
            predecessor = nextPredecessor;
            advanced = true;
        }
        if (reason == null && advanced) {
            next = sampler.sample(predecessor);
            sampler.accept(next);
            emitted++;
            reason = session.offer(next);
        }
        commitTree(seqs, nodes, frontier, pos, acceptedDepth, branches);
        ((TreeDraftModel) decoder.drafter).acceptTree(acceptedDepth);
        return new SpeculativeDecoder.Verified(acceptedDepth, emitted, next, reason);
    }

    /**
     * Keeps the accepted path as sequence 0 and empties the other target sequences: rows past the accepted depth
     * leave every sequence, the accepted branch's pages copy over sequence 0 and every other branch sequence
     * empties.
     */
    private static void commitTree(SequenceMemory seqs, List<DraftTree.Node> nodes, List<Integer> frontier, int pos,
                                   int acceptedDepth, int branches) {
        int keep = pos + acceptedDepth;
        int best = frontier.isEmpty() ? -1 : nodes.get(frontier.get(0)).branch();
        for (int s = 0; s < branches; s++) {
            if (seqs.posMax(s) > keep) {
                seqs.remove(s, keep + 1, -1);
            }
        }
        if (best > 0) {
            seqs.remove(0, 0, -1);
            seqs.copy(best, 0, 0, -1);
        }
        for (int s = 1; s < branches; s++) {
            if (s != best) {
                seqs.remove(s, 0, -1);
            }
        }
    }

    /** One token's logits at {@code pos} on target sequence {@code seq}. */
    private float[] decodeRow(int token, int pos, int seq) {
        ForwardBatch batch = ForwardBatch.builder(1).add(token, pos, seq, true).build();
        float[][] rows = decoder.backend.forwardBatch(batch);
        return rows[0];
    }

}
