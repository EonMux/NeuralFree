package com.eonmux.neuralfree.gguf.engine;

/**
 * One single-pass tree verify's recurrent-side bookkeeping, opened by the forward backend around exactly one
 * masked verify forward of a target whose layers carry recurrent state (Gated DeltaNet, KDA, Mamba-2). The
 * forward's tree step runs every node against a per-node <b>stack</b> copy of the state, seeded from the committed
 * state along the node's root path, so the committed state itself is untouched by the verify (no snapshot, no
 * rewind); the round's {@link #commit} then replays the accepted rows — the pending token first, then the accepted
 * path — through the recurrent layers, advancing the committed state with the same one-token arithmetic the verify
 * used. The committed state after the round is therefore the state after exactly the accepted tokens.
 *
 * <p>The round keeps no per-node state: every node contributes one captured mixer-input row per recurrent layer
 * (the commit inputs of the replay), and the stack holds one state slot per verify row.
 */
public interface RecurrentTreeRound extends AutoCloseable {

    /** The verify step's row count: the pending token plus the fixed width in tree nodes. */
    int rows();

    /**
     * The rows the verify step computes: the pending token plus the proposal's nodes. The rows above it are the
     * step's padding, whose outputs no reader consumes and whose state the round skips.
     */
    int realRows();

    /** The step's per-row ancestor-or-self bitmask, padding rows zeroed. */
    long[] mask();

    /** Depth of the row's node: {@code Long.bitCount(mask[row]) - 1}, -1 for a padding row. */
    static int depth(long maskRow) {
        return maskRow == 0 ? -1 : Long.bitCount(maskRow) - 1;
    }

    /**
     * The verify row whose stack slot a node's state seeds from: its immediate parent, the largest row its mask
     * names. Rows ascend along every root path (a parent precedes its children), so the largest named row is the
     * parent and its slot was written before the child's step runs.
     */
    static int parentRow(long maskRow, int row) {
        return 63 - Long.numberOfLeadingZeros(maskRow ^ (1L << row));
    }

    /**
     * Replays the named verify rows, in the order given, through the model's recurrent layers: per row, every
     * recurrent layer runs its one-token step from its captured input row against the committed state. Called by
     * the tree commit after the KV compaction, with row 0 first and the accepted path ascending.
     *
     * @throws IllegalStateException when the round ran no verify step or was already committed
     */
    void commit(int[] replayRows);

    /** Releases the round's scratch; the committed state is live afterwards. */
    @Override
    void close();
}
