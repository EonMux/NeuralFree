package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Flash attention of a block of query rows over a KV cache layer its caller owns (a drafter's own cache): the
 * {@link GpuFlashAttention} launches of one layer bound to the cache's K and V rows and its {@code kv_pos} record,
 * reading Q rows from and writing the attention rows to the caller's buffers, on the context's queue. Visibility is
 * the kernels' rule ({@code common/attn_common.h} {@code nf_attn_visible}: {@code kv_pos}, the causal flag and the
 * window of the {@link AttentionSpec}), the same rule {@link CpuAttentionCore} applies. Not thread-safe.
 */
public final class GpuCacheAttention implements AutoCloseable {

    private final GpuMixerResources resources;
    private final GpuFlashAttention flash;
    private final ComputeQueue queue;
    private final GpuBuffer qPos;
    private final ByteBuffer staging;
    private final int ubatch;
    private boolean closed;

    private GpuCacheAttention(GpuMixerResources resources, GpuFlashAttention flash, ComputeQueue queue,
                              GpuBuffer qPos, int ubatch) {
        this.resources = resources;
        this.flash = flash;
        this.queue = queue;
        this.qPos = qPos;
        this.ubatch = ubatch;
        this.staging = ByteBuffer.allocate(ubatch * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Prepares the attention of up to {@code ubatch} query rows of {@code a} over {@code kv}.
     *
     * @param kvPos the position held by each cache row of {@code kv}, -1 for an empty row
     * @param q     {@code ubatch} Q rows of {@code a.qDim()} floats from element 0
     * @param out   {@code ubatch} output rows of {@code a.outDim()} floats from element 0
     * @param sinks the F32 per-head sink logits, or null without sinks
     * @throws IllegalArgumentException when the geometry fits no flash kernel
     */
    public static GpuCacheAttention bind(GpuForwardContext ctx, AttentionSpec a, GpuKvLayer kv, GpuBuffer kvPos,
                                         GpuBuffer q, GpuBuffer out, GpuBuffer sinks, int ubatch) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(kv, "kv");
        if (ubatch < 1) {
            throw new IllegalArgumentException("cache attention of " + ubatch + " query rows");
        }
        if (a.sinks() != (sinks != null)) {
            throw new IllegalArgumentException("attention " + a + (a.sinks() ? " needs" : " takes no") + " sinks");
        }
        if (a.sinkTokens() > 0) {
            throw new UnsupportedOperationException("attention " + a + " has learned static sink tokens, which"
                    + " the drafter cache attention does not carry");
        }
        GpuMixerResources res = new GpuMixerResources(ctx.ops(), ctx.backend());
        try {
            GpuBuffer qPos = res.allocate((long) ubatch * Integer.BYTES);
            FlashKeys keys = FlashKeys.cached(kv, Objects.requireNonNull(kvPos, "kvPos"));
            FlashPlan plan = FlashPlan.of(ctx.kernels(), a, keys, null, ubatch);
            GpuBuffer partial = ctx.scratch().partials((int) Math.max(1L, plan.partialCapacity()));
            FlashKernelArgs args = new FlashKernelArgs(a, keys, Objects.requireNonNull(q, "q"), qPos,
                    Objects.requireNonNull(out, "out"), partial, sinks, null, null, null,
                    ctx.ops().placeholders().kvScalesOnes(), ctx.ops().placeholders().readOnlyFloats());
            GpuFlashAttention flash = GpuFlashAttention.prepare(res, ctx.queue(), plan, args, null, null, ubatch,
                    null);
            return new GpuCacheAttention(res, flash, ctx.queue(), qPos, ubatch);
        } catch (RuntimeException e) {
            RuntimeException f = res.release(null);
            if (f != null) {
                e.addSuppressed(f);
            }
            throw e;
        }
    }

    /**
     * Enqueues the attention of the {@code n} query rows at positions {@code pos0 .. pos0 + n - 1} over the cache
     * rows up to {@code pos0 + n} (every row of a ring), each row seeing the keys its visibility rule admits.
     *
     * @throws IllegalArgumentException when {@code n} is outside {@code [1, ubatch]} or a position is negative
     */
    public void attend(int pos0, int n) {
        if (closed) {
            throw new IllegalStateException("cache attention is closed");
        }
        if (n < 1 || n > ubatch || pos0 < 0) {
            throw new IllegalArgumentException("cache attention of " + n + " rows at " + pos0 + " outside [1, "
                    + ubatch + "] rows from a non-negative position");
        }
        staging.clear();
        for (int t = 0; t < n; t++) {
            staging.putInt(pos0 + t);
        }
        GpuEvent e = queue.enqueueWriteBuffer(qPos, 0, (long) n * Integer.BYTES, staging.array());
        if (e != null) {
            e.close();
        }
        flash.prefill(pos0, n);
    }

    /** Releases the prepared launches and the query-position buffer. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException f = resources.release(null);
        if (f != null) {
            throw f;
        }
    }
}
