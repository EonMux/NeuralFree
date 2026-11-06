package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

/**
 * A hook that observes or replaces a contiguous attention step's raw head outputs before the mixer's output
 * stages run. Context-parallel attention installs one to copy each rank's normalized attention over its context
 * block, merge the ranks' partial softmaxes with their log-sum-exp, and hand the merged rows back to the rank that
 * owns the layer output.
 *
 * <p>{@link #afterAttention} runs after the flash attention (and over a cache, after the KV store) and before the
 * output rotation, the output norm/gate steps and the output projection, so a replacement is seen by all of
 * them. It runs for the contiguous {@code decode} / {@code prefill} paths only; a paged batch has no tap.</p>
 */
public interface AttentionTap {

    /**
     * Receives one step's raw attention outputs.
     *
     * @param pos0       first position of the step
     * @param nTokens    rows of the step
     * @param out        {@code [nTokens][nHead][headDimV]} dense F32 head outputs, rank-major; may be rewritten
     * @param lse        {@code [nTokens][nHead]} merged log-sum-exp of this rank's key range
     * @param nHead      query heads
     * @param headDimV   values per head
     */
    void afterAttention(int pos0, int nTokens, GpuBuffer out, GpuBuffer lse, int nHead, int headDimV);

    /**
     * Whether this tap's mixer must run the output stages (rotation, norm/gate and output projection) after
     * {@link #afterAttention}. A context-parallel peer that only contributes a partial to the rank owning the layer
     * output returns false, so its discarded projection is not computed.
     */
    default boolean ownsOutput() {
        return true;
    }
}
