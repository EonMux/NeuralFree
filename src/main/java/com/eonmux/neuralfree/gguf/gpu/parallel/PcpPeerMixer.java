package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/**
 * The attention mixer a peer rank's prefill context-parallel trunk binds ({@code PcpPrefillCoordinator}'s
 * per-rank registry): the coordinator drives the peer's per-layer phases through its executor, and the peer's
 * attention part is only the finish of its own chunk-pair rows. The projections, rope and cache stores of those
 * rows ran through the {@link PcpAttentionMixer}'s part for this rank (which also scattered the gathered queries
 * and every other rank's cache rows into this rank's cache), and the merged attention output arrived in this
 * rank's attention buffer through the coordinator's scatter, so the peer's prefill runs the part's
 * {@code finishStep} — the output stages and the output projection over its packed rows.
 *
 * <p>The peer trunk never runs a decode, a cache-write-only pass or a batched step: the coordinator calls the
 * trunk's norm, attention and post phases of prefill steps only, so the other mixer entries refuse.</p>
 */
final class PcpPeerMixer implements GpuMixer {

    private final TensorParallelGroup group;
    private final LayerSpec layer;
    private final int rank;

    PcpPeerMixer(TensorParallelGroup group, LayerSpec layer, int rank) {
        this.group = group;
        this.layer = layer;
        this.rank = rank;
    }

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        // nothing of its own: the pcp mixer's part for this rank holds the bound state
    }

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        throw new IllegalStateException("the prefill context-parallel trunk of rank " + rank + " runs prefill"
                + " phases only");
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        group.pcpCoordinator().finishPeerRows(rank, layer, nTokens, out);
    }

    @Override
    public void prefillKV(int pos0, int nTokens, ScratchSlot in) {
        throw new IllegalStateException("the prefill context-parallel trunk of rank " + rank + " runs prefill"
                + " phases only");
    }

    @Override
    public String kvRefusal() {
        return "the prefill context-parallel trunk of rank " + rank + " runs the coordinator's phases only";
    }

    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        throw new IllegalStateException("the prefill context-parallel trunk of rank " + rank + " runs prefill"
                + " phases only");
    }

    @Override
    public String toString() {
        return "PcpPeerMixer{rank " + rank + ", layer " + layer.il() + "}";
    }
}
