package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.List;

/**
 * The token mixer of one layer on the GPU (attention, recurrent or convolutional), enqueued on the context queue.
 *
 * <p>Lifecycle: {@link #bind} once with the layer's context (look up weights, reserve scratch and memory views,
 * prepare dispatches), then {@link #decode} or {@link #prefill} per step, then {@link #close()}. Per step a mixer
 * allocates nothing and never waits on the host: every buffer comes from {@code GpuScratch}, {@code GpuMemory} or
 * {@code GpuWeights}.</p>
 *
 * <p>{@code in} holds the normalized layer input ({@code nTokens} rows of the stream width); the mixer writes its
 * branch output, before the residual, to {@code out} in the same layout. {@code in} and {@code out} are distinct.</p>
 */
public interface GpuMixer extends AutoCloseable {

    /** Binds this mixer to a model context and its layer. Called once, before the first step. */
    void bind(GpuForwardContext ctx, LayerSpec layer);

    /**
     * Points the kernels' state-row routing buffer at {@code stateRow} before the next dispatch: a batched step
     * re-points it per segment, so every segment advances its sequence's own state row. A memory without
     * per-sequence state rows never routes (its single sequence is row 0).
     */
    static void routeStateRow(int stateRow, GpuBuffer stateRows, ComputeQueue queue) {
        java.nio.ByteBuffer one = java.nio.ByteBuffer.allocate(Integer.BYTES)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        one.putInt(stateRow);
        GpuEvent e = queue.enqueueWriteBuffer(stateRows, 0, Integer.BYTES, one.array());
        if (e != null) {
            e.close();
        }
    }

    /** One token at position {@code pos}. */
    void decode(int pos, ScratchSlot in, ScratchSlot out);

    /**
     * The patchers of this mixer's decode launches, in the order their groups enqueue: a graph replay's
     * plan claims each captured kernel dispatch with the patcher whose group it belongs to, and one
     * claimed patch re-sets the per-step kernel arguments the {@link #decode} body would set. Empty when
     * the mixer's decode launches have no per-step derivable values, or the mixer has none.
     */
    default List<GpuStepPatcher> decodeReplayPatchers() {
        return List.of();
    }

    /** {@code nTokens} consecutive tokens at positions {@code pos0 ..}. */
    void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out);

    /**
     * The cache writes of {@code nTokens} tokens at positions {@code pos0 ..} alone: the projections, norms and
     * rotations the stored K and V rows need, and the store; no query, attention, output projection or branch
     * residual, none of which a reader of these caches consumes. {@code in} holds the normalized layer input and is
     * left unread afterwards. The stored rows are the ones {@link #prefill} stores.
     *
     * <p>A mixer that writes no cache (a linear replacement, a reader of another layer's cache) does nothing.</p>
     *
     * @throws UnsupportedOperationException when this mixer has no cache-write-only step ({@link #kvRefusal()})
     */
    default void prefillKV(int pos0, int nTokens, ScratchSlot in) {
        throw new UnsupportedOperationException(kvRefusal());
    }

    /**
     * Why this mixer has no cache-write-only step ({@link #prefillKV}), or null when it has one. The layer executor
     * asks before the pass; a refusing mixer runs its whole attention part instead, so the cache rows are the same.
     */
    default String kvRefusal() {
        return getClass().getSimpleName() + " cannot run its cache writes without its attention";
    }

    /**
     * Drops cached positions {@code [nKeep, nKeep + nDiscard)} and shifts the later ones down (context shift).
     *
     * @throws UnsupportedOperationException when this mixer cannot shift its memory
     */
    default void shiftContext(int nKeep, int nDiscard) {
        throw new UnsupportedOperationException(shiftRefusal());
    }

    /**
     * Re-rotates this mixer's cached keys by the memory's last per-row shift deltas, without a keep/discard rule:
     * the position edits ({@code seq_add}, {@code seq_div}) and the paged context shift all re-rotate by the same
     * {@code shift_delta} buffer, so the default defers to {@link #shiftContext} with counts that carry no rule.
     *
     * @throws UnsupportedOperationException when this mixer cannot shift its memory
     */
    default void applyShiftDeltas() {
        shiftContext(0, 1);
    }

    /**
     * Why {@link #shiftContext} would refuse on this bound mixer, or null when it can shift. The forward backend asks
     * every cache-owning mixer before the memory moves any row, so a refused shift leaves the cache intact.
     */
    default String shiftRefusal() {
        return "context shift unsupported for " + getClass().getSimpleName();
    }

    /**
     * Why this mixer has no batched step over {@link StepLayout#contiguousSingle()} == false, or null when it can
     * run one. Default: refused, today's mixers run the contiguous step only.
     */
    default String batchRefusal() {
        return getClass().getSimpleName() + " has no multi-sequence step";
    }

    /**
     * One step over the rows of {@code s}: today's contiguous step ({@link #decode} / {@link #prefill} at
     * {@code pos0(0)}) when the step is contiguous and the memory un-paged, the backend's batched step otherwise.
     *
     * @throws UnsupportedOperationException when the step is not contiguous-un-paged and this mixer has no
     *                                       batched step ({@link #batchRefusal()})
     */
    default void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        if (s.contiguousSingle()) {
            if (s.nTokens() == 1) {
                decode(s.pos0(0), in, out);
            } else {
                prefill(s.pos0(0), s.nTokens(), in, out);
            }
            return;
        }
        throw new UnsupportedOperationException(batchRefusal());
    }

    /** Releases what {@link #bind} created; the context's shared resources stay. */
    @Override
    default void close() {
    }
}
