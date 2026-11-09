package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.fused.GpuMultiTask;

import java.util.Objects;
import java.util.function.IntConsumer;

/** A prepared dispatch whose count argument and 1-D grid follow the step's token count. */
final class TokenStep {

    final PreparedDispatch dispatch;
    /** Persistent multi-task launch; when set the step runs it instead of {@link #dispatch}. */
    private final GpuMultiTask batch;
    /** Arbitrary per-step action (an embedding lookup); when set it replaces {@link #dispatch}. */
    private final IntConsumer action;
    private final int countArg;
    private final long countPerToken;
    private final long itemsPerToken;
    private final long itemsFixed;
    private final int posStrideArg;
    private final int local;
    private int lastTokens = -1;
    /** Runs the step over the step's segments instead of one launch over {@code n} tokens when set. */
    SegmentStep segments;

    /** A step whose launches follow the step's sequence segments (Inkling's short convolutions). */
    interface SegmentStep {
        void run(int n, PagedStep paged);
    }

    TokenStep(PreparedDispatch dispatch, int countArg, long countPerToken, long itemsPerToken,
              long itemsFixed, int posStrideArg, int local) {
        this(dispatch, null, null, countArg, countPerToken, itemsPerToken, itemsFixed, posStrideArg, local);
    }

    private TokenStep(PreparedDispatch dispatch, GpuMultiTask batch, IntConsumer action, int countArg,
                      long countPerToken, long itemsPerToken, long itemsFixed, int posStrideArg, int local) {
        this.dispatch = dispatch;
        this.batch = batch;
        this.action = action;
        this.countArg = countArg;
        this.countPerToken = countPerToken;
        this.itemsPerToken = itemsPerToken;
        this.itemsFixed = itemsFixed;
        this.posStrideArg = posStrideArg;
        this.local = local;
    }

    static TokenStep multiTask(GpuMultiTask batch) {
        return new TokenStep(null, batch, null, 0, 1, 1, 0, -1, 1);
    }

    /** A step that runs {@code action} with the token count and launches no kernel of its own. */
    static TokenStep action(IntConsumer action) {
        return new TokenStep(null, null, Objects.requireNonNull(action, "action"), 0, 1, 1, 0, -1, 1);
    }

    void run(int n) {
        run(n, null);
    }

    void run(int n, PagedStep paged) {
        if (segments != null) {
            segments.run(n, paged);
            return;
        }
        if (action != null) {
            action.accept(n);
            return;
        }
        if (batch != null) {
            batch.run(n);
            return;
        }
        if (n != lastTokens) {
            dispatch.setInt(countArg, (int) (n * countPerToken));
            if (posStrideArg >= 0) {
                dispatch.setInt(posStrideArg, n);
            }
            long items = n * itemsPerToken + itemsFixed;
            dispatch.setGlobal(Math.max(1, (items + local - 1) / local) * local);
            lastTokens = n;
        }
        dispatch.dispatch();
    }
}
