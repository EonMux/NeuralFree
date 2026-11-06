package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.memory.CompressorRollback;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

import java.util.Objects;

/**
 * The paged sequences of a memory whose per-sequence state the mixers read beside the pages: the recurrent-state
 * slots and the DeepSeek-V4 compressed streams. The sequence operations keep that state in step, as llama.cpp's
 * recurrent and DSV4 caches do (llama-kv-cache-dsv4.cpp:1595-1690):
 * <ul>
 *   <li>a removal, a keep or a clear that empties a sequence zeroes its state ({@link States#clear}), so the next
 *       generation in the slot starts fresh (the engine's evictions and preemptions reuse slots this way);</li>
 *   <li>with compressed streams, a tail removal restores the rollback snapshot plane it drops to
 *       ({@link States#restore}) and is refused beyond the snapshots, as is a removal inside the sequence; a copy
 *       takes whole sequences only and copies the streams ({@link States#copy}); position shifts and divisions are
 *       refused.</li>
 * </ul>
 * Everything else delegates to the {@link PagedSequences}. Both memories (heap and device) run their sequence
 * operations through this class with their own {@link States}.
 */
public final class StatefulPagedSequences implements SequenceMemory {

    /** The per-sequence state a memory keeps beside its pages. */
    public interface States {

        /** Zeroes sequence {@code seq}'s recurrent-state slot and compressed streams. */
        void clear(int seq);

        /** Copies rollback snapshot plane {@code plane} into the live compressor plane of {@code seq}'s streams. */
        void restore(int seq, int plane);

        /** Copies every compressed stream of {@code src}, rows and planes, into {@code dst}. */
        void copy(int src, int dst);
    }

    private final PagedSequences paged;
    private final States states;
    private final boolean compressed;
    private final CompressorRollback[] rollbacks;

    /**
     * @param compressed whether the memory keeps DSV4 compressed streams (the plan's {@code hasCompressed()})
     * @param rollbacks  the compressor rollback bookkeeping of every sequence, which the memory's steps advance
     * @throws IllegalArgumentException when {@code rollbacks} does not hold one entry per sequence
     */
    public StatefulPagedSequences(PagedSequences paged, States states, boolean compressed,
                                  CompressorRollback[] rollbacks) {
        this.paged = Objects.requireNonNull(paged, "paged");
        this.states = Objects.requireNonNull(states, "states");
        this.compressed = compressed;
        this.rollbacks = Objects.requireNonNull(rollbacks, "rollbacks");
        if (rollbacks.length != paged.maxSequences()) {
            throw new IllegalArgumentException(rollbacks.length + " rollback records for " + paged.maxSequences()
                    + " sequences");
        }
    }

    @Override
    public int maxSequences() {
        return paged.maxSequences();
    }

    @Override
    public SequenceState sequence(int seq) {
        return paged.sequence(seq);
    }

    @Override
    public int posMin(int seq) {
        return paged.posMin(seq);
    }

    @Override
    public int posMax(int seq) {
        return paged.posMax(seq);
    }

    @Override
    public String removeRefusal(int seq, int p0, int p1) {
        String refusal = paged.removeRefusal(seq, p0, p1);
        return refusal != null ? refusal : compressedRemoveRefusal(seq, p0, p1);
    }

    @Override
    public void remove(int seq, int p0, int p1) {
        String refusal = removeRefusal(seq, p0, p1);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        int dropped = compressedTail(seq, p0);
        paged.remove(seq, p0, p1);
        if (paged.posMax(seq) < 0) {
            clearSequence(seq);
        } else if (dropped > 0) {
            states.restore(seq, rollbacks[seq].onRollback(dropped));
        }
    }

    /**
     * The tree verify refuses a memory with DSV4 compressed caches up front, so the keep-recurrent drop never runs
     * over compressor planes here: the plain suffix removal of the paged rows is the whole story.
     */
    @Override
    public String removeKeepRecurrentRefusal(int seq, int p0) {
        return paged.removeKeepRecurrentRefusal(seq, p0);
    }

    @Override
    public void removeKeepRecurrent(int seq, int p0) {
        paged.removeKeepRecurrent(seq, p0);
    }

    @Override
    public String copyRefusal() {
        return paged.copyRefusal();
    }

    @Override
    public boolean sharesPages(int seq) {
        return paged.sharesPages(seq);
    }

    /**
     * @throws UnsupportedOperationException with DSV4 compressed caches, for a copy of part of {@code src}
     *                                       (llama-kv-cache-dsv4.cpp:1649-1651)
     */
    @Override
    public void copy(int src, int dst, int p0, int p1) {
        if (compressed && (p0 > paged.posMin(src) || (p1 >= 0 && p1 <= paged.posMax(src)))) {
            throw new UnsupportedOperationException("DeepSeek-V4 compressed caches copy whole sequences only"
                    + " (llama-kv-cache-dsv4.cpp:1649-1651), not positions [" + p0 + ", " + p1 + ") of sequence "
                    + src);
        }
        paged.copy(src, dst, p0, p1);
        if (compressed) {
            states.copy(src, dst);
            // the copy keeps no rollback window (llama-kv-cache-dsv4.cpp:1661-1664)
            rollbacks[dst].reset();
        }
    }

    @Override
    public void keep(int seq) {
        paged.keep(seq);
        for (int s = 0; s < paged.maxSequences(); s++) {
            if (s != seq) {
                clearSequence(s);
            }
        }
    }

    @Override
    public void clear() {
        paged.clear();
        for (int s = 0; s < paged.maxSequences(); s++) {
            clearSequence(s);
        }
    }

    @Override
    public String shiftRefusal() {
        return compressed ? "DeepSeek-V4 compressed caches cannot shift positions (llama-kv-cache-dsv4.cpp:1452)"
                : paged.shiftRefusal();
    }

    @Override
    public void add(int seq, int p0, int p1, int delta) {
        if (compressed && delta != 0) {
            throw new UnsupportedOperationException(shiftRefusal());
        }
        paged.add(seq, p0, p1, delta);
    }

    @Override
    public String divRefusal() {
        return compressed ? "DeepSeek-V4 compressed caches cannot re-position positions (llama-kv-cache-dsv4.cpp:1452)"
                : paged.divRefusal();
    }

    @Override
    public void div(int seq, int p0, int p1, int d) {
        if (compressed && d > 1) {
            throw new UnsupportedOperationException(divRefusal());
        }
        paged.div(seq, p0, p1, d);
    }

    @Override
    public String moveRowsRefusal(int seq, int[] srcPos, int[] dstPos) {
        return paged.moveRowsRefusal(seq, srcPos, dstPos);
    }

    /** A compaction keeps every row and position: no state follows the cells, so this delegates unchanged. */
    @Override
    public void moveRows(int seq, int[] srcPos, int[] dstPos) {
        paged.moveRows(seq, srcPos, dstPos);
    }

    @Override
    public int freeCells() {
        return paged.freeCells();
    }

    @Override
    public int totalCells() {
        return paged.totalCells();
    }

    private void clearSequence(int seq) {
        states.clear(seq);
        rollbacks[seq].reset();
    }

    /**
     * Why removing {@code [p0, p1)} of {@code seq} cannot keep its DSV4 compressed streams exact, or null: the
     * compressor state accumulates, so only a whole-sequence removal or a tail removal its rollback snapshots cover
     * can run (llama-kv-cache-dsv4.cpp:1600-1646).
     */
    private String compressedRemoveRefusal(int seq, int p0, int p1) {
        int max = paged.posMax(seq);
        if (!compressed || max < 0 || p0 > max || (p1 >= 0 && p1 <= p0)) {
            return null;
        }
        if (p1 >= 0 && p1 <= max) {
            return "DeepSeek-V4 compressed caches drop a sequence's tail only (llama-kv-cache-dsv4.cpp:1600-1603),"
                    + " not positions [" + p0 + ", " + p1 + ") of sequence " + seq + " (" + paged.posMin(seq) + ".."
                    + max + ")";
        }
        int tail = compressedTail(seq, p0);
        return tail == 0 ? null : rollbacks[seq].blocker(tail);
    }

    /**
     * The tail positions a removal from {@code p0} drops from the compressor state of {@code seq}: 0 when it drops
     * nothing or the whole sequence (the streams are zeroed then), else {@code posMax + 1 - p0}.
     */
    private int compressedTail(int seq, int p0) {
        int max = paged.posMax(seq);
        if (!compressed || max < 0 || p0 > max || p0 <= paged.posMin(seq)) {
            return 0;
        }
        return max + 1 - p0;
    }

    @Override
    public String toString() {
        return "StatefulPagedSequences{" + paged + (compressed ? ", compressed streams" : "") + '}';
    }
}
