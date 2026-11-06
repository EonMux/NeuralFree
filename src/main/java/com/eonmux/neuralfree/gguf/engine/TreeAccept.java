package com.eonmux.neuralfree.gguf.engine;

import java.util.Arrays;

/**
 * The single-pass tree verify's acceptance, walked on the target's device ({@code LLM_TreeAccept} over
 * {@link ForwardBackend#treeAcceptForward}): the accepted depth, the path's node indices in depth order, and the
 * two greedy ids the round's emissions need besides the path's node tokens — the first emission (the pending row's
 * greedy id) and the final draw (the last accepted row's greedy id, the pending row's when nothing was accepted).
 * The emissions a round produces are a prefix of {@code [firstId, the matched nodes' tokens from the second match
 * on, finalDrawId]} — the first match's token is firstId itself — so the host replays them through the sampler and
 * the stop session and cuts the replay at a stop.
 *
 * <p>Immutable; the path array is copied at construction and never modified afterwards.</p>
 */
public final class TreeAccept {

    private final int acceptedDepth;
    private final int[] path;
    private final int firstId;
    private final int finalDrawId;

    /**
     * @param acceptedDepth the accepted path's depth, 0 when nothing was accepted
     * @param path          the accepted path's node indices in depth order, {@code acceptedDepth} entries
     * @param firstId       the greedy id of the verify batch's pending row, the round's first emission
     * @param finalDrawId   the greedy id of the last accepted node's row (the pending row's when nothing was
     *                      accepted), the round's final emission
     * @throws IllegalArgumentException when the path does not hold {@code acceptedDepth} entries or names a
     *                                  negative node
     */
    public TreeAccept(int acceptedDepth, int[] path, int firstId, int finalDrawId) {
        if (acceptedDepth < 0) {
            throw new IllegalArgumentException("negative accepted depth " + acceptedDepth);
        }
        if (path == null || path.length < acceptedDepth) {
            throw new IllegalArgumentException("an accepted depth of " + acceptedDepth + " needs that many path"
                    + " entries, got " + (path == null ? "none" : path.length));
        }
        for (int i = 0; i < acceptedDepth; i++) {
            if (path[i] < 0) {
                throw new IllegalArgumentException("path entry " + i + " names node " + path[i]);
            }
        }
        this.acceptedDepth = acceptedDepth;
        this.path = Arrays.copyOf(path, acceptedDepth);
        this.firstId = firstId;
        this.finalDrawId = finalDrawId;
    }

    /** The accepted path's depth, 0 when nothing was accepted. */
    public int acceptedDepth() {
        return acceptedDepth;
    }

    /** The accepted path's node indices in depth order, {@link #acceptedDepth} entries. */
    public int[] path() {
        return path.clone();
    }

    /** The greedy id of the verify batch's pending row, the round's first emission. */
    public int firstId() {
        return firstId;
    }

    /**
     * The greedy id of the last accepted node's row, the pending row's when nothing was accepted: the round's
     * final emission when the walk confirmed at least one node.
     */
    public int finalDrawId() {
        return finalDrawId;
    }

    @Override
    public String toString() {
        return "TreeAccept{depth " + acceptedDepth + ", path " + Arrays.toString(path)
                + ", first " + firstId + ", final " + finalDrawId + "}";
    }
}
