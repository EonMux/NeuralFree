package com.eonmux.neuralfree.distributed;

import java.util.Objects;
import java.util.concurrent.TimeUnit;

/**
 * What one rank's collective call is: its kind, its detail (the {@link ReduceOp} of an all-reduce, the source rank of
 * a broadcast) and its element count. Before a collective exchanges values, the ranks exchange their descriptors and
 * every rank checks them with {@link #requireAgreement}, so ranks whose call sequences diverge, or whose tensors differ
 * in size, all fail with the same {@link IllegalArgumentException} instead of pairing unrelated values or waiting for
 * each other until the timeout.
 *
 * <p>Descriptors cross the wire as {@link #INTS} ints ({@link #toInts}) or, in a shared-memory slot, as
 * {@link #FLOATS} floats ({@link #toFloats}): every field is split so that each float holds an integer below
 * {@code 2^16}, which a float represents exactly.</p>
 */
final class CollectiveDescriptor {
    /** How long a rank waits for another rank inside one collective before it fails. */
    static final long TIMEOUT_MINUTES = 10;
    static final long TIMEOUT_NANOS = TimeUnit.MINUTES.toNanos(TIMEOUT_MINUTES);
    /** The largest element count of one collective: its values fit one byte array. */
    static final int MAX_COUNT = (Integer.MAX_VALUE - 64) / Float.BYTES;
    static final int INTS = 3;
    static final int FLOATS = 4;

    private static final int LOW_BITS = 16;
    private static final int LOW_MASK = (1 << LOW_BITS) - 1;
    private static final int NO_DETAIL = -1;

    enum Kind {
        ALL_REDUCE("allReduce"),
        BROADCAST("broadcast"),
        ALL_GATHER("allGather"),
        BARRIER("barrier");

        private final String label;

        Kind(String label) {
            this.label = label;
        }
    }

    private final Kind kind;
    private final int detail;
    private final int count;

    private CollectiveDescriptor(Kind kind, int detail, int count) {
        this.kind = kind;
        this.detail = detail;
        this.count = count;
    }

    /** @throws IllegalArgumentException when {@code count} exceeds {@link #MAX_COUNT} */
    static CollectiveDescriptor allReduce(ReduceOp op, long count) {
        Objects.requireNonNull(op, "ReduceOp cannot be null");
        return new CollectiveDescriptor(Kind.ALL_REDUCE, op.ordinal(), checkedCount(count));
    }

    /** @throws IllegalArgumentException when {@code count} exceeds {@link #MAX_COUNT} */
    static CollectiveDescriptor broadcast(int srcRank, long count) {
        return new CollectiveDescriptor(Kind.BROADCAST, srcRank, checkedCount(count));
    }

    /** @throws IllegalArgumentException when {@code count} exceeds {@link #MAX_COUNT} */
    static CollectiveDescriptor allGather(long count) {
        return new CollectiveDescriptor(Kind.ALL_GATHER, NO_DETAIL, checkedCount(count));
    }

    static CollectiveDescriptor barrier() {
        return new CollectiveDescriptor(Kind.BARRIER, NO_DETAIL, 0);
    }

    Kind kind() {
        return kind;
    }

    int count() {
        return count;
    }

    int[] toInts() {
        return new int[]{kind.ordinal(), detail, count};
    }

    /**
     * The descriptor encoded at {@code offset} of {@code values} ({@link #toInts}).
     *
     * @throws IllegalArgumentException when the ints do not encode a descriptor
     */
    static CollectiveDescriptor fromInts(int[] values, int offset) {
        return decode(values[offset], values[offset + 1], values[offset + 2]);
    }

    float[] toFloats() {
        return new float[]{kind.ordinal(), detail + 1, count >>> LOW_BITS, count & LOW_MASK};
    }

    /**
     * The descriptor encoded in {@code values} ({@link #toFloats}).
     *
     * @throws IllegalArgumentException when the floats do not encode a descriptor
     */
    static CollectiveDescriptor fromFloats(float[] values) {
        int kindCode = exactInt(values[0]);
        int detail = exactInt(values[1]) - 1;
        int high = exactInt(values[2]);
        int low = exactInt(values[3]);
        if (high < 0 || low < 0 || low > LOW_MASK || high > (MAX_COUNT >>> LOW_BITS)) {
            throw new IllegalArgumentException("Corrupt collective descriptor: count fields " + values[2] + ", "
                + values[3]);
        }
        return decode(kindCode, detail, (high << LOW_BITS) | low);
    }

    /**
     * Checks that every rank describes the same collective.
     *
     * @throws IllegalArgumentException naming every rank's call when they differ
     */
    static void requireAgreement(CollectiveDescriptor[] perRank) {
        boolean agree = true;
        for (int r = 1; r < perRank.length; r++) {
            agree &= perRank[0].equals(perRank[r]);
        }
        if (agree) {
            return;
        }
        StringBuilder message = new StringBuilder("The ranks called different collectives:");
        for (int r = 0; r < perRank.length; r++) {
            message.append(r == 0 ? " " : ", ").append("rank ").append(r).append(' ').append(perRank[r]);
        }
        message.append("; every rank must call the same collectives in the same order with tensors of the same size");
        throw new IllegalArgumentException(message.toString());
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof CollectiveDescriptor)) {
            return false;
        }
        CollectiveDescriptor other = (CollectiveDescriptor) o;
        return kind == other.kind && detail == other.detail && count == other.count;
    }

    @Override
    public int hashCode() {
        return Objects.hash(kind, detail, count);
    }

    @Override
    public String toString() {
        switch (kind) {
            case ALL_REDUCE:
                return kind.label + "(" + ReduceOp.values()[detail] + ", " + count + " values)";
            case BROADCAST:
                return kind.label + "(from rank " + detail + ", " + count + " values)";
            case ALL_GATHER:
                return kind.label + "(" + count + " values)";
            default:
                return kind.label + "()";
        }
    }

    private static int checkedCount(long count) {
        if (count < 0 || count > MAX_COUNT) {
            throw new IllegalArgumentException("A collective exchanges at most " + MAX_COUNT + " values, got " + count);
        }
        return (int) count;
    }

    private static CollectiveDescriptor decode(int kindCode, int detail, int count) {
        Kind[] kinds = Kind.values();
        if (kindCode < 0 || kindCode >= kinds.length) {
            throw new IllegalArgumentException("Corrupt collective descriptor: kind " + kindCode);
        }
        Kind kind = kinds[kindCode];
        boolean detailValid;
        switch (kind) {
            case ALL_REDUCE:
                detailValid = detail >= 0 && detail < ReduceOp.values().length;
                break;
            case BROADCAST:
                detailValid = detail >= 0;
                break;
            default:
                detailValid = detail == NO_DETAIL;
        }
        if (!detailValid || count < 0 || count > MAX_COUNT || (kind == Kind.BARRIER && count != 0)) {
            throw new IllegalArgumentException("Corrupt collective descriptor: " + kind.label + " with detail "
                + detail + " and count " + count);
        }
        return new CollectiveDescriptor(kind, detail, count);
    }

    private static int exactInt(float value) {
        int asInt = (int) value;
        if (asInt != value) {
            throw new IllegalArgumentException("Corrupt collective descriptor field " + value);
        }
        return asInt;
    }
}
