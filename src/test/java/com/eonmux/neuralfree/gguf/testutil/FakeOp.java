package com.eonmux.neuralfree.gguf.testutil;

import java.util.Arrays;

/** Immutable record of one queue operation of a {@link FakeComputeBackend}. Fields that do not apply to the kind are null or 0. */
public final class FakeOp {
    private final int sequence;
    private final FakeOpKind kind;
    private final String queue;
    private final FakeDispatch dispatch;
    private final FakeGpuBuffer buffer;
    private final long offset;
    private final FakeGpuBuffer source;
    private final long sourceOffset;
    private final long bytes;
    private final int pattern;
    private final int waitCount;

    private FakeOp(Builder b, int sequence) {
        this.sequence = sequence;
        this.kind = b.kind;
        this.queue = b.queue;
        this.dispatch = b.dispatch;
        this.buffer = b.buffer;
        this.offset = b.offset;
        this.source = b.source;
        this.sourceOffset = b.sourceOffset;
        this.bytes = b.bytes;
        this.pattern = b.pattern;
        this.waitCount = b.dispatch != null ? b.dispatch.waitCount() : b.waitCount;
    }

    static Builder builder(FakeOpKind kind, String queue) {
        return new Builder(kind, queue);
    }

    public int sequence() {
        return sequence;
    }

    public FakeOpKind kind() {
        return kind;
    }

    public String queue() {
        return queue;
    }

    /** The dispatch for {@link FakeOpKind#DISPATCH}, else null. */
    public FakeDispatch dispatch() {
        return dispatch;
    }

    /** Destination of WRITE/COPY/FILL, source of READ. */
    public FakeGpuBuffer buffer() {
        return buffer;
    }

    public long offset() {
        return offset;
    }

    /** Source of COPY. */
    public FakeGpuBuffer source() {
        return source;
    }

    public long sourceOffset() {
        return sourceOffset;
    }

    public long bytes() {
        return bytes;
    }

    /** Fill pattern of FILL. */
    public int pattern() {
        return pattern;
    }

    public int waitCount() {
        return waitCount;
    }

    @Override
    public String toString() {
        if (kind == FakeOpKind.DISPATCH) {
            return dispatch.toString();
        }
        StringBuilder sb = new StringBuilder();
        sb.append('#').append(sequence).append(' ').append(queue).append(' ').append(kind);
        switch (kind) {
            case WRITE:
            case READ:
                sb.append(' ').append(buffer).append(" @").append(offset).append(" +").append(bytes).append('B');
                break;
            case COPY:
                sb.append(' ').append(source).append(" @").append(sourceOffset).append(" -> ").append(buffer)
                        .append(" @").append(offset).append(" +").append(bytes).append('B');
                break;
            case FILL:
                sb.append(' ').append(buffer).append(" @").append(offset).append(" +").append(bytes)
                        .append("B pattern=0x").append(Integer.toHexString(pattern));
                break;
            default:
                break;
        }
        if (waitCount > 0) {
            sb.append(" waits=").append(waitCount);
        }
        return sb.toString();
    }

    static final class Builder {
        private final FakeOpKind kind;
        private final String queue;
        private FakeDispatch dispatch;
        private FakeGpuBuffer buffer;
        private long offset;
        private FakeGpuBuffer source;
        private long sourceOffset;
        private long bytes;
        private int pattern;
        private int waitCount;

        private Builder(FakeOpKind kind, String queue) {
            this.kind = kind;
            this.queue = queue;
        }

        Builder dispatch(FakeDispatch d) {
            this.dispatch = d;
            return this;
        }

        Builder buffer(FakeGpuBuffer b, long off) {
            this.buffer = b;
            this.offset = off;
            return this;
        }

        Builder source(FakeGpuBuffer b, long off) {
            this.source = b;
            this.sourceOffset = off;
            return this;
        }

        Builder bytes(long n) {
            this.bytes = n;
            return this;
        }

        Builder pattern(int p) {
            this.pattern = p;
            return this;
        }

        Builder waits(int n) {
            this.waitCount = n;
            return this;
        }

        FakeOp build(int sequence) {
            return new FakeOp(this, sequence);
        }
    }
}
