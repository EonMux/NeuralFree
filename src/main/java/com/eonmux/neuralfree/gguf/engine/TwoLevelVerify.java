package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * One round of two-level verification ({@code neuralfree.draft.tlv}, SpecPilot's TLV, arXiv 2611.02501): the
 * quantized view of the target verifies the drafter's tree once with plain argmax and commits nothing — the
 * accepted chain and the view's bonus token only steer a second, narrower draft — and the exact target verifies
 * the composed proposal in one pass and commits, so every emitted token stays the exact target's own greedy
 * choice whatever the copy proposes.
 *
 * <p>The Q round reuses the single-pass verify batch ({@link TreeVerify#batch}): the pending token plus the
 * fixed-width tree rows over the shared sequence memory, the drafter's injection suspended. Its walk keeps the
 * longest chain of nodes whose tokens the argmax names, then takes the bonus as the argmax after the chain —
 * no sampler, no stop criteria, no commit: the round's rows play back once through the view
 * ({@link #commitChain}) so the drafter's cache receives the chain rows from the view's hidden states, which is
 * what lets the second draft anchor below the chain. The exact round then rolls the view's rows back out
 * ({@link SpeculativeDecoder} does, before {@link TreeVerify#round}) and re-decides everything with exact
 * weights: a wrong chain only wastes the width it occupied.
 */
final class TwoLevelVerify {

    private TwoLevelVerify() {
    }

    /** The quantized view's greedy outcome: the chain it accepted and the bonus after it. */
    static final class Chain {

        /** The pending token followed by the accepted proposal tokens; at least the pending token. */
        final int[] tokens;
        /** The view's greedy token after the chain (its bonus, the second draft's anchor). */
        final int bonus;

        Chain(int[] tokens, int bonus) {
            this.tokens = tokens;
            this.bonus = bonus;
        }

        /** The nodes the exact proposal carries for the chain: the accepted tokens after the root. */
        int spine() {
            return tokens.length - 1;
        }
    }

    /**
     * The Q round: verifies {@code proposal} after the pending {@code token} at {@code pos} on the quantized
     * view's backend with plain argmax, one fixed-width batch over the shared sequence memory, the drafter's
     * injection suspended. Nothing is committed: the rows the forward stored stay until the caller plays the
     * chain back ({@link #commitChain}) or rolls them out.
     *
     * @param width    the fixed verify width in tree nodes, the same batch shape the exact round runs
     * @param ropeType the target's rope type, as {@link TreeVerify#batch} takes it
     */
    static Chain round(ForwardBackend viewBackend, DraftTree proposal, int token, int pos, int width,
                       RopeType ropeType, TreeDraftModel drafter) {
        if (proposal.nodes().size() > width) {
            throw new IllegalArgumentException("the proposal drafts " + proposal.nodes().size() + " nodes, the"
                    + " fixed verify width holds " + width + " (neuralfree.draft.tree.nodes)");
        }

        drafter.featureTapSuspended(true);
        float[][] rows;
        try {
            rows = viewBackend.forwardBatch(TreeVerify.batch(proposal, token, pos, width, ropeType));
        } finally {
            drafter.featureTapSuspended(false);
        }

        // the greedy walk keeps the longest chain of nodes whose tokens the argmax names: no sampler draw, no
        // stop criterion, no commit — the chain is a proposal, the exact round decides
        List<Integer> frontier = Collections.emptyList();
        List<Integer> path = new ArrayList<>();
        float[] predecessor = rows[0];
        for (int depth = 1; depth <= proposal.depth(); depth++) {
            List<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < proposal.nodes().size(); i++) {
                DraftTree.Node n = proposal.nodes().get(i);
                if (n.depth() == depth
                        && (frontier.isEmpty() ? n.parent() == -1 : frontier.contains(n.parent()))) {
                    candidates.add(i);
                }
            }
            if (candidates.isEmpty()) {
                break;
            }
            int next = argmax(predecessor);
            int matched = -1;
            for (int index : candidates) {
                if (proposal.nodes().get(index).token() == next) {
                    matched = index;
                    break;
                }
            }
            if (matched < 0) {
                break;
            }
            frontier = Collections.singletonList(matched);
            path.add(matched);
            predecessor = rows[matched + 1];
        }
        int bonus = argmax(predecessor);
        int[] tokens = new int[path.size() + 1];
        tokens[0] = token;
        for (int i = 0; i < path.size(); i++) {
            tokens[i + 1] = proposal.nodes().get(path.get(i)).token();
        }
        return new Chain(tokens, bonus);
    }

    /**
     * Plays the chain's first {@code chainLen} tokens back once through the view: the stored round rows leave
     * ({@code chainLen} rows sit at {@code pos..pos + chainLen - 1} when it returns), and the replay runs with
     * the drafter's injection active — the replay IS the injection, so the drafter's cache receives the chain
     * rows from the view's hidden states and the second draft can anchor below the chain. The view commits
     * nothing to the request: the exact round rolls these rows out and rebuilds them with exact weights.
     */
    static void commitChain(ForwardBackend viewBackend, SequenceMemory seqs, int token, int pos, int[] chain,
                            int chainLen) {
        seqs.remove(0, pos, -1);
        ForwardBatch.Builder replay = ForwardBatch.builder(chainLen);
        replay.add(token, pos, 0, false);
        for (int d = 1; d < chainLen; d++) {
            replay.add(chain[d], pos + d, 0, false);
        }
        viewBackend.forwardBatch(replay.build());
    }

    /**
     * Composes the exact round's proposal: the chain after its root as a spine of {@code chainLen - 1} nodes,
     * the view's bonus as their child, and the draft the bonus anchored hung below it (its roots children of
     * the bonus node, its depths shifted past the chain). The spine may be empty — a chain of only the pending
     * token composes to the anchored draft below the bonus alone.
     *
     * @param chain    the Q round's tokens: the pending token then the accepted proposals
     * @param chainLen how much of the chain the round carries, at most the width minus one so the bonus and the
     *                 anchored draft always fit the fixed width
     * @param sub      the draft the bonus anchored, over the view's rows; empty when the width left nothing
     */
    static DraftTree compose(Chain chain, int chainLen, DraftTree sub) {
        int spine = Math.min(chainLen - 1, chain.spine());
        List<DraftTree.Node> nodes = new ArrayList<>();
        for (int i = 0; i < spine; i++) {
            nodes.add(new DraftTree.Node(chain.tokens[i + 1], i - 1, i + 1, 0));
        }
        int bonusIndex = nodes.size();
        nodes.add(new DraftTree.Node(chain.bonus, spine - 1, spine + 1, 0));
        for (DraftTree.Node n : sub.nodes()) {
            // the sub node at index k lands at spine + 1 + k: its roots hang below the bonus, its depths shift
            // past the chain, and a sub parent maps to that sub node's new index
            int parent = n.parent() < 0 ? bonusIndex : spine + 1 + n.parent();
            nodes.add(new DraftTree.Node(n.token(), parent, n.depth() + spine + 1, n.branch()));
        }
        return new DraftTree(nodes);
    }

    /** The argmax of one logits row, the only decision a quantized view makes. */
    static int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }
}
