package com.eonmux.neuralfree.gguf.memory;

/**
 * The {@link SequenceMemory} of a contiguous single-sequence {@link ModelMemory} ({@code GpuMemory} or
 * {@code CpuMemory}): the adapter that maps the sequence API onto today's storage, where rows hold positions in
 * order and only a suffix can be dropped ({@link ModelMemory#rollback(int)}). Sequences beyond 0 do not exist, so
 * {@code copy}, interior {@code remove} ranges and position shifts refuse, and every refusal names what would be
 * needed.
 */
public final class SingleSequenceMemory implements SequenceMemory {

    private final ModelMemory memory;

    /** The adapter over {@code memory}'s stored sequence ({@link ModelMemory#sequence()}). */
    public SingleSequenceMemory(ModelMemory memory) {
        this.memory = java.util.Objects.requireNonNull(memory, "memory");
    }

    @Override
    public int maxSequences() {
        return 1;
    }

    @Override
    public SequenceState sequence(int seq) {
        checkSeq(seq);
        return memory.sequence();
    }

    @Override
    public int posMin(int seq) {
        checkSeq(seq);
        SequenceState state = memory.sequence();
        if (state.nPast() == 0) {
            return -1;
        }
        return state.wrapped() ? Math.max(0, state.nPast() - state.nCtx()) : 0;
    }

    @Override
    public int posMax(int seq) {
        checkSeq(seq);
        SequenceState state = memory.sequence();
        return state.nPast() == 0 ? -1 : state.nPast() - 1;
    }

    @Override
    public String removeRefusal(int seq, int p0, int p1) {
        checkSeq(seq);
        if (p0 < 0) {
            return "remove starts before the sequence: position " + p0;
        }
        if (p1 >= 0) {
            return "contiguous single-sequence memory can only remove a suffix of positions";
        }
        int max = posMax(seq);
        if (max < 0 || p0 > max) {
            // removing nothing, or the sequence holds nothing: never a refusal of the storage
            return null;
        }
        int nDrop = max - p0 + 1;
        if (p0 == 0 || memory.canRollback(nDrop)) {
            return null;
        }
        return "contiguous single-sequence memory can drop at most " + nDrop + " stored positions only when "
                + "canRollback(" + nDrop + ") holds";
    }

    @Override
    public void remove(int seq, int p0, int p1) {
        String refusal = removeRefusal(seq, p0, p1);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        SequenceState state = memory.sequence();
        if (state.nPast() == 0 || p0 > posMax(seq)) {
            return;
        }
        if (p0 == 0) {
            memory.reset();
            return;
        }
        memory.rollback(posMax(seq) - p0 + 1);
    }

    @Override
    public String copyRefusal() {
        return "contiguous single-sequence memory cannot copy between sequences";
    }

    @Override
    public void copy(int src, int dst, int p0, int p1) {
        throw new UnsupportedOperationException(copyRefusal());
    }

    @Override
    public void keep(int seq) {
        checkSeq(seq);
        // only sequence 0 exists; nothing outside it to empty
    }

    @Override
    public void clear() {
        memory.reset();
    }

    @Override
    public String shiftRefusal() {
        return "position shifts of contiguous single-sequence memory run through ForwardBackend.shiftContext";
    }

    @Override
    public void add(int seq, int p0, int p1, int delta) {
        throw new UnsupportedOperationException(shiftRefusal());
    }

    @Override
    public int freeCells() {
        // one page pool per KV group of rows, rows after the stored positions are free
        return memory.sequence().nCtx() - memory.sequence().nPast();
    }

    @Override
    public int totalCells() {
        return memory.sequence().nCtx();
    }

    private void checkSeq(int seq) {
        if (seq != 0) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside the single held sequence 0");
        }
    }
}
