package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;

import java.util.Objects;

/**
 * The shared transformer block of a Zamba2 layer (vLLM {@code Zamba2HybridLayer} over
 * {@code Zamba2AttentionDecoderLayer}, vllm/model_executor/models/zamba2.py). A hybrid layer concatenates its input
 * with the model input embedding, normalizes the {@code 2 * n_embd} row, runs the shared attention and the gated
 * GELU MLP of one of {@code shared_blocks.count} bare blocks, projects the result back to {@code n_embd} with its
 * {@code shared_linear} and adds it to the Mamba-2 input. The bare block's weights are stored once under the
 * {@code shared_blk} stack; each use adds its own rank-{@code adapter_rank} adapters to the block's q/k/v and
 * gate_up projections.
 *
 * <p>The facet is also present on Mamba-only layers ({@link #sharedBlock()} {@link #NO_SHARED_BLOCK}), which run no
 * shared block but belong to the same model and carry the model's adapter rank. Immutable.
 */
public final class Zamba2Spec {

    /** {@link #sharedBlock()} of a Mamba-only layer. */
    public static final int NO_SHARED_BLOCK = -1;

    private final int sharedBlock;
    private final int adapterRank;
    private final boolean attentionAdapter;

    private Zamba2Spec(Builder b) {
        this.sharedBlock = b.sharedBlock;
        this.adapterRank = b.adapterRank;
        this.attentionAdapter = b.attentionAdapter;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Bare shared block this layer runs, or {@link #NO_SHARED_BLOCK} for a Mamba-only layer. */
    public int sharedBlock() { return sharedBlock; }

    /** True when the layer runs a shared transformer block before its Mamba-2 mixer. */
    public boolean hybrid() { return sharedBlock != NO_SHARED_BLOCK; }

    /** Rank of the per-use adapters (always positive: the shared MLP adds its adapter on every use). */
    public int adapterRank() { return adapterRank; }

    /** True when the shared attention's q/k/v projections add their per-use adapters ({@code use_shared_attention_adapter}). */
    public boolean attentionAdapter() { return attentionAdapter; }

    /**
     * The shared block's attention as a plain {@link MixerKind#ATTENTION} layer over the {@code shared_blk} stack,
     * bound by the attention mixer: it keeps the real layer's {@code il} (the per-use KV cache) and reads the bare
     * block's weights.
     *
     * @throws IllegalStateException when the layer runs no shared block
     */
    public LayerSpec attentionLayer(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        if (!hybrid()) {
            throw new IllegalStateException("layer " + layer.il() + " runs no shared transformer block");
        }
        return LayerSpec.builder(layer.il(), MixerKind.ATTENTION, FfnKind.NONE)
                .weightLayer(sharedBlock)
                .stack(TensorNames.STACK_SHARED_BLK)
                .norm(layer.norm())
                .residual(layer.residual())
                .attention(layer.attention())
                .rope(layer.rope())
                .build();
    }

    /**
     * This layer's Mamba-2 branch as a plain {@link MixerKind#MAMBA2} layer, bound by the Mamba-2 mixer: the same
     * {@code il} (its recurrent state), weight layer and stack, with the recurrent spec's Mamba-2 view.
     *
     * @throws IllegalStateException when the layer has no Mamba-2 recurrent spec
     */
    public LayerSpec mambaLayer(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        RecurrentSpec r = layer.recurrent();
        if (r == null || r.kind() != MixerKind.ZAMBA2) {
            throw new IllegalStateException("layer " + layer.il() + " has no Zamba2 recurrent spec");
        }
        return LayerSpec.builder(layer.il(), MixerKind.MAMBA2, FfnKind.NONE)
                .weightLayer(layer.weightLayer())
                .stack(layer.stack())
                .norm(layer.norm())
                .residual(layer.residual())
                .recurrent(r.asKind(MixerKind.MAMBA2))
                .build();
    }

    @Override
    public String toString() {
        return "Zamba2Spec{block=" + sharedBlock + ", rank=" + adapterRank
                + (attentionAdapter ? ", qkv adapters" : "") + '}';
    }

    public static final class Builder {
        private int sharedBlock = NO_SHARED_BLOCK;
        private int adapterRank;
        private boolean attentionAdapter;

        private Builder() {
        }

        /** Bare shared block of the layer, or {@link #NO_SHARED_BLOCK} for a Mamba-only layer. */
        public Builder sharedBlock(int v) { this.sharedBlock = v; return this; }
        /** Rank of the per-use adapters; positive. */
        public Builder adapterRank(int v) { this.adapterRank = v; return this; }
        /** The shared attention adds its per-use q/k/v adapters. */
        public Builder attentionAdapter(boolean v) { this.attentionAdapter = v; return this; }

        /**
         * @throws IllegalArgumentException when the block index is below {@link Zamba2Spec#NO_SHARED_BLOCK}, the
         *                                  adapter rank is not positive, or a Mamba-only layer claims attention
         *                                  adapters
         */
        public Zamba2Spec build() {
            if (sharedBlock < NO_SHARED_BLOCK) {
                throw new IllegalArgumentException("zamba2 shared block " + sharedBlock + " is below "
                        + NO_SHARED_BLOCK);
            }
            if (adapterRank < 1) {
                throw new IllegalArgumentException("zamba2 adapter rank " + adapterRank + " must be positive; the"
                        + " shared MLP adds its per-use adapter on every use");
            }
            if (attentionAdapter && sharedBlock == NO_SHARED_BLOCK) {
                throw new IllegalArgumentException("a Mamba-only zamba2 layer has no shared attention to adapt");
            }
            return new Zamba2Spec(this);
        }
    }
}
