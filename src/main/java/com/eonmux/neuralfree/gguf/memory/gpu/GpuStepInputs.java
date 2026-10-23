package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

/**
 * Per-step device inputs shared by every layer: the addressing contract of the KV and recurrent caches.
 * {@link #upload(int, int)} writes each buffer once per ubatch with a single transfer:
 * <ul>
 *   <li>{@link #kvPos(int)}: {@code int[rows]} per KV group, the position stored in each row or -1 when empty;
 *       updated for the rows the step writes. Attention derives visibility (causal, SWA, chunked, symmetric) from
 *       it.</li>
 *   <li>{@link #cellIdx(int)}: {@code int[nTokens]} per KV group, {@code cell_idx[t] = rowOf(pos0 + t)}, the row each
 *       token's K/V is stored in.</li>
 *   <li>{@link #qPos()}: {@code int[nTokens]} query positions {@code pos0 + t}.</li>
 *   <li>{@link #ropePos()}: {@code int[4 * nTokens]} RoPE positions laid out {@code pos[t + k * nTokens]}; section
 *       {@code k} of a text token per {@code RopeSpec.textPositions()}; single-section types read only section 0,
 *       which holds {@code pos0 + t}.</li>
 *   <li>{@link #stateRows()}: {@code int[1] = {0}}, the recurrent-state row of the single sequence.</li>
 * </ul>
 * Buffers are sized for {@code ubatch} tokens and stay valid until {@code close()} of the owning memory.
 */
public interface GpuStepInputs {

    /** Number of KV groups. */
    int groups();

    /** {@code kv_pos[rows]} of a KV group. */
    GpuBuffer kvPos(int group);

    /** {@code cell_idx[nTokens]} of a KV group for the current step. */
    GpuBuffer cellIdx(int group);

    /** Query positions of the current step. */
    GpuBuffer qPos();

    /** Four-section RoPE positions of the current step. */
    GpuBuffer ropePos();

    /** Recurrent-state row indices ({@code {0}}). */
    GpuBuffer stateRows();

    /** First position of the current step. */
    int pos0();

    /** Tokens in the current step. */
    int nTokens();

    /**
     * Writes the inputs for tokens at positions {@code [pos0, pos0 + nTokens)} and marks their rows in every group's
     * {@code kv_pos}.
     *
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]} or a position is outside a
     *                                  non-ring group
     */
    void upload(int pos0, int nTokens);
}
