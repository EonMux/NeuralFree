package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * One single-pass verification of a branching proposal: the whole tree runs as ONE fixed-width forward of the ONE
 * target sequence, every tree node a row of a single batch under a tree attention mask (SpecPilot's
 * {@code TREE_MODE=ddtree} verify, arXiv 2611.02501). The batch's rows append at the sequence's next slots: row 0
 * holds the pending token (the tree's root) and rows {@code 1..width} hold the proposal's nodes, shorter trees
 * padded to the width, so the step's shape never changes and a decode-step replay captures it. Every row stores at
 * its slot position but ropes at its depth position ({@code pos + depth}, siblings sharing one position), and the
 * per-row ancestor bitmask — bit {@code j} of row {@code i}'s mask names row {@code j} when {@code j} is
 * {@code i} or its ancestor — replaces the position rules within the step's fresh rows: a node attends the
 * committed prefix and exactly its ancestors, never its siblings' subtrees. The logits equal a chain decode's over
 * the node's root path.
 *
 * <p>The acceptance walk samples from the pending row and then from each matched node's row, keeping every node
 * whose token the sample equals, exactly like the per-depth round ({@code SpeculativeDecoder#verifyBranchingTree},
 * llama.cpp's token-match acceptance): one sampler draw per emitted token, the bonus token sampled from the last
 * accepted row, so a sampling request consumes the same draws as the ordinary rounds.
 *
 * <p>The commit compacts: the accepted path's KV rows move from their verify slots down onto the consecutive
 * positions {@code pos+1..pos+acceptedDepth} (a row move per layer cache, the drafter's captured rows reinjected),
 * the rows above leave, and the round costs ONE target forward — no re-decode. A node roped at {@code pos + depth}
 * lands on the position it already roped at, so the moves replant no rotation; the sources ascend above their
 * destinations, so the moves run in list order on every memory. The committed KV equals a linear decode's on
 * contiguous and paged memory alike, and the drafter's hidden rows end where a chain step would have left them.
 */
final class TreeVerify {

    private TreeVerify() {
    }

    /**
     * Verifies {@code proposal} after the target's pending {@code token} at {@code pos}, and commits the outcome.
     *
     * @param width         the fixed verify width in tree nodes: the batch always lays out {@code 1 + width} rows
     *                      (the pending token plus the nodes, padded), whatever the proposal drafts
     * @param ropeType      the target's rope type, which decides whether the batch carries per-row rope sections
     * @param textPositions the target's text-token section pattern for a multi-section rope, unused otherwise
     * @param recurrent     whether the target keeps recurrent state ({@code memory.hasRecurrentState()}): the
     *                      round then opens a {@link RecurrentTreeRound} around the masked verify, whose per-node
     *                      mixer-input captures the commit replays along the accepted path from the unchanged
     *                      committed state
     * @param deviceAccept  whether the acceptance walks on the device ({@code LLM_TreeAccept} over the batch's
     *                      argmax rows, gated by the caller on {@code verifyOnDevice}, an argmax-only sampler and
     *                      {@code backend.treeAcceptOnDevice()}): the rows' logits never come back, and the round
     *                      replays the device walk's emissions through the sampler and the stop session
     * @throws IllegalArgumentException when the proposal drafts more nodes than the width holds
     */
    static SpeculativeDecoder.Verified round(ForwardBackend backend, SequenceMemory seqs, DraftTree proposal,
                                             int token, int pos, Sampler sampler, StopCriteria.Session session,
                                             int width, RopeType ropeType, MropeTextPositions textPositions,
                                             TreeDraftModel drafter, boolean recurrent, boolean deviceAccept) {
        List<DraftTree.Node> nodes = proposal.nodes();
        if (nodes.size() > width) {
            throw new IllegalArgumentException("the proposal drafts " + nodes.size() + " nodes, the fixed verify"
                    + " width holds " + width + " (neuralfree.draft.tree.nodes); narrow the tree or raise the"
                    + " width");
        }
        int rows = width + 1;

        // one sequence, slot order: row i stores at pos + i and ropes at pos + its node's depth
        int[] storePos = new int[rows];
        int sections = RopeMath.positionsPerToken(ropeType);
        if (sections > 1 && textPositions == null) {
            throw new IllegalArgumentException("the " + ropeType + " rope ropes every row from a text-token"
                    + " section pattern, which the round was not given");
        }
        int[][] rope = new int[sections][rows];
        long[] mask = new long[rows];
        mask[0] = 1L;
        storePos[0] = pos;
        ropeAt(rope, textPositions, 0, pos);
        ForwardBatch.Builder batch = ForwardBatch.builder(rows);
        batch.add(token, storePos[0], 0, true);
        for (int i = 0; i < nodes.size(); i++) {
            DraftTree.Node n = nodes.get(i);
            int row = i + 1;
            storePos[row] = pos + row;
            ropeAt(rope, textPositions, row, pos + n.depth());
            long bits = (1L << row) | 1L; // the pending root is every node's ancestor
            for (int p = n.parent(); p >= 0; p = nodes.get(p).parent()) {
                bits |= 1L << (p + 1);
            }
            mask[row] = bits;
            batch.add(n.token(), storePos[row], 0, true);
        }
        for (int row = nodes.size() + 1; row < rows; row++) {
            // a padding row keeps the shape: it stores its slot, its output is never read, and its all-ones mask
            // keeps the row's softmax non-empty; no real row's mask names a padding row (they sit last)
            storePos[row] = pos + row;
            ropeAt(rope, textPositions, row, pos + row);
            mask[row] = -1L;
            batch.add(token, storePos[row], 0, false);
        }
        batch.ropePositions(rope);
        batch.treeMask(mask);

        RecurrentTreeRound round = null;
        try {
            drafter.featureTapSuspended(true);
            float[][] rowOut = null;
            TreeAccept accept = null;
            try {
                if (recurrent) {
                    round = backend.openRecurrentTreeRound(rows, nodes.size() + 1);
                }
                if (deviceAccept) {
                    accept = backend.treeAcceptForward(batch.build(), proposal);
                } else {
                    rowOut = backend.forwardBatch(batch.build());
                }
            } finally {
                drafter.featureTapSuspended(false);
            }

            // greedy/token-match acceptance along the tree: sample from the pending row, keep the nodes the sample
            // names, continue through their children — the per-depth round's walk over one batch of rows
            List<Integer> path = new ArrayList<>();
            int acceptedDepth = 0;
            int emitted = 0;
            int next = -1;
            StopReason reason = null;
            if (accept != null) {
                // the device walked the raw logits' argmax ids (the sampler reduces to them on the device path),
                // so the round's emissions are a prefix of [firstId, the matched nodes' tokens from the second
                // match on, finalDrawId]: a matched node's token is the sample that took it, so the first match's
                // token is firstId itself and repeating it would emit one token more per round than the host
                // walk; the replay runs them through the sampler and the stop session, and a stop cuts the replay
                // exactly where the host walk would have stopped
                int a = accept.acceptedDepth();
                int[] devicePath = accept.path();
                int[] ids = new int[a >= 1 ? a + 1 : 1];
                ids[0] = accept.firstId();
                for (int d = 1; d < a; d++) {
                    ids[d] = nodes.get(devicePath[d]).token();
                }
                if (a >= 1) {
                    ids[a] = accept.finalDrawId();
                }
                for (int i = 0; i < ids.length && reason == null; i++) {
                    next = ids[i];
                    sampler.accept(next);
                    emitted++;
                    reason = session.offer(next);
                    // emission i is the depth-(i + 1) sample for i < a: once the session took it, the walk's
                    // (i + 1)-th node is confirmed exactly as the host walk's match confirms it; a session stop
                    // leaves the earlier depths, and the final draw (i == a) never advances the path
                    if (reason == null && i < a) {
                        acceptedDepth = i + 1;
                        path.add(devicePath[i]);
                    }
                }
            } else {
                boolean advanced = false;
                List<Integer> frontier = Collections.emptyList();
                float[] predecessor = rowOut[0];
                for (int depth = 1; depth <= proposal.depth(); depth++) {
                    List<Integer> candidates = new ArrayList<>();
                    for (int i = 0; i < nodes.size(); i++) {
                        DraftTree.Node n = nodes.get(i);
                        if (n.depth() == depth && (frontier.isEmpty() ? n.parent() == -1
                                : frontier.contains(n.parent()))) {
                            candidates.add(i);
                        }
                    }
                    if (candidates.isEmpty()) {
                        break;
                    }
                    Sampler.Candidates cur = sampler.apply(predecessor);
                    next = cur.id(cur.selected());
                    sampler.accept(next);
                    emitted++;
                    reason = session.offer(next);
                    if (reason != null) {
                        break;
                    }
                    List<Integer> matched = new ArrayList<>();
                    float[] nextPredecessor = null;
                    for (int index : candidates) {
                        if (nodes.get(index).token() == next) {
                            matched.add(index);
                            if (nextPredecessor == null) {
                                nextPredecessor = rowOut[index + 1];
                            }
                        }
                    }
                    if (matched.isEmpty()) {
                        advanced = false;
                        break;
                    }
                    frontier = matched;
                    path.add(matched.get(0));
                    acceptedDepth = depth;
                    predecessor = nextPredecessor;
                    advanced = true;
                }
                if (reason == null && advanced) {
                    Sampler.Candidates cur = sampler.apply(predecessor);
                    next = cur.id(cur.selected());
                    sampler.accept(next);
                    emitted++;
                    reason = session.offer(next);
                }
            }

            commit(backend, seqs, nodes, pos, path, acceptedDepth, drafter, round != null);
            if (round != null) {
                // the replay runs the accepted path (row 0 first, then the nodes ascending) from the unchanged
                // committed state with the arithmetic the verify used, so the committed state ends where the
                // accepted leaf's stack slot ended; row 0 replays even when nothing was accepted
                int[] replay = new int[acceptedDepth + 1];
                for (int d = 1; d <= acceptedDepth; d++) {
                    replay[d] = path.get(d - 1) + 1;
                }
                round.commit(replay);
            }
            drafter.acceptTree(acceptedDepth);
            return new SpeculativeDecoder.Verified(acceptedDepth, emitted, next, reason);
        } finally {
            if (round != null) {
                round.close();
            }
        }
    }

    /** Writes row {@code row}'s rope position {@code p} into every section: the pattern's for multi-section ropes. */
    private static void ropeAt(int[][] rope, MropeTextPositions textPositions, int row, int p) {
        for (int s = 0; s < rope.length; s++) {
            rope[s][row] = rope.length == 1 ? p : textPositions.position(s, p);
        }
    }

    /**
     * The fixed-width verify batch of {@code proposal} after the pending {@code token} at {@code pos}: one
     * sequence, slot order — row 0 stores the pending token at {@code pos}, rows {@code 1..width} hold the
     * proposal's nodes (shorter trees padded), every row stores at its slot position {@code pos + i} and ropes
     * at {@code pos +} its node's depth, and the per-row ancestor bitmask names the row and its ancestors. The
     * rows' outputs come back in slot order: row {@code i + 1} is node {@code i}'s.
     */
    static ForwardBatch batch(DraftTree proposal, int token, int pos, int width, RopeType ropeType) {
        List<DraftTree.Node> nodes = proposal.nodes();
        int rows = width + 1;

        int[] storePos = new int[rows];
        int[][] rope = RopeMath.positionsPerToken(ropeType) == 1 ? new int[1][rows] : null;
        long[] mask = new long[rows];
        mask[0] = 1L;
        storePos[0] = pos;
        if (rope != null) {
            rope[0][0] = pos;
        }
        ForwardBatch.Builder batch = ForwardBatch.builder(rows);
        batch.add(token, storePos[0], 0, true);
        for (int i = 0; i < nodes.size(); i++) {
            DraftTree.Node n = nodes.get(i);
            int row = i + 1;
            storePos[row] = pos + row;
            if (rope != null) {
                rope[0][row] = pos + n.depth();
            }
            long bits = 1L << row;
            for (int p = n.parent(); p >= 0; p = nodes.get(p).parent()) {
                bits |= 1L << (p + 1);
            }
            mask[row] = bits;
            batch.add(n.token(), storePos[row], 0, true);
        }
        for (int row = nodes.size() + 1; row < rows; row++) {
            // a padding row keeps the shape: it stores its slot, its output is never read, and its all-ones mask
            // keeps the row's softmax non-empty; no real row's mask names a padding row (they sit last)
            storePos[row] = pos + row;
            if (rope != null) {
                rope[0][row] = pos + row;
            }
            mask[row] = -1L;
            batch.add(token, storePos[row], 0, false);
        }
        if (rope != null) {
            batch.ropePositions(rope);
        }
        batch.treeMask(mask);
        return batch.build();
    }

    /**
     * Leaves exactly the accepted path committed: the whole tree step rolls back out of the sequence (its
     * {@code width + 1} rows sat at {@code pos..pos + width}), and the path re-decodes at consecutive positions
     * {@code pos..pos + acceptedDepth} with the drafter's injection running — the replay IS the injection, the
     * rows' store and rope positions coincide on a chain, so the committed KV equals a linear decode's and the
     * drafter's hidden rows end where a chain step would have left them. The pending token re-decodes even when
     * nothing was accepted: its KV belongs to the context the next round builds on. Under {@code recurrent} the
     * dead rows drop without touching the recurrent state ({@link SequenceMemory#removeKeepRecurrent(int, int)}):
     * the round's accepted-path replay right after re-derives it; without the round the plain removal keeps the
     * memory's rollback pairing of state and sequence.
     */
    private static void commit(ForwardBackend backend, SequenceMemory seqs, List<DraftTree.Node> nodes, int pos,
                               List<Integer> path, int acceptedDepth, TreeDraftModel drafter, boolean recurrent) {
        int[] srcPos = new int[acceptedDepth];
        int[] dstPos = new int[acceptedDepth];
        int moves = 0;
        for (int d = 1; d <= acceptedDepth; d++) {
            int src = pos + path.get(d - 1) + 1;
            int dst = pos + d;
            if (src != dst) {
                srcPos[moves] = src;
                dstPos[moves] = dst;
                moves++;
            }
        }
        if (moves > 0) {
            seqs.moveRows(0, java.util.Arrays.copyOf(srcPos, moves), java.util.Arrays.copyOf(dstPos, moves));
            backend.carryRowMoves();
        }
        if (recurrent) {
            seqs.removeKeepRecurrent(0, pos + acceptedDepth + 1);
        } else {
            seqs.remove(0, pos + acceptedDepth + 1, -1);
        }
        if (acceptedDepth > 0) {
            int[] positions = new int[acceptedDepth];
            int[] tokens = new int[acceptedDepth];
            for (int d = 1; d <= acceptedDepth; d++) {
                positions[d - 1] = pos + d;
                tokens[d - 1] = nodes.get(path.get(d - 1)).token();
            }
            seqs.sequence(0).retokenize(positions, tokens);
        }
        int[] verifyRow = new int[acceptedDepth + 1];
        for (int d = 1; d <= acceptedDepth; d++) {
            verifyRow[d] = path.get(d - 1) + 1;
        }
        drafter.injectVerifyRows(verifyRow, pos);
    }
}
