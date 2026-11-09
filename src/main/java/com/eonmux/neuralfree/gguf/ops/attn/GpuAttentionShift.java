package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/**
 * The context shift of one GPU attention layer: whether the layer can take a shift and, for a layer that owns a
 * RoPE cache in a model llama.cpp can shift, the {@link GpuContextShift} that rotates its K rows. A layer without
 * projections of its own shifts nothing; a host-KV layer, a layer that attends only to the current batch, a model
 * llama.cpp cannot shift ({@link GpuContextShift#refusal}) and a cache whose geometry the shift kernel cannot address
 * refuse by name. Not thread-safe.
 */
final class GpuAttentionShift implements AutoCloseable {

    private final String refusal;
    private final GpuContextShift shift;

    private GpuAttentionShift(String refusal, GpuContextShift shift) {
        this.refusal = refusal;
        this.shift = shift;
    }

    /** The shift of a layer that projects no attention rows of its own: nothing to rotate and nothing to refuse. */
    static GpuAttentionShift none() {
        return new GpuAttentionShift(null, null);
    }

    /**
     * The shift of a layer with attention projections of its own over the cache {@code kv} (null for a layer
     * without a cache). A cache whose geometry the shift kernel cannot address (for example a block-type cache whose
     * head stride is not a multiple of 32) does not fail the bind, because the shift is optional:
     * {@link #refusal} reports it instead.
     */
    static GpuAttentionShift bind(GpuForwardContext ctx, LayerSpec layer, boolean hostKv, GpuKvLayer kv,
                                  GpuBuffer freqFactors) {
        if (hostKv) {
            return new GpuAttentionShift("layer " + layer.il() + " keeps its KV cache on the host; context shift"
                    + " over a host cache is not supported", null);
        }
        if (!layer.attention().usesMemory()) {
            return new GpuAttentionShift("layer " + layer.il() + " attends only to the current batch; it has no"
                    + " context to shift", null);
        }
        String modelRefusal = GpuContextShift.refusal(ctx.spec(), layer);
        if (modelRefusal != null) {
            return new GpuAttentionShift(modelRefusal, null);
        }
        if (kv == null || !GpuContextShift.shifts(layer)) {
            return none();
        }
        try {
            return new GpuAttentionShift(null, GpuContextShift.of(ctx.ops(), ctx.memory(), layer, kv, freqFactors));
        } catch (IllegalArgumentException e) {
            return new GpuAttentionShift("layer " + layer.il() + ": " + e.getMessage(), null);
        }
    }

    /** Why {@link #apply} cannot run, or null when it can. */
    String refusal() {
        return refusal;
    }

    /**
     * Applies the RoPE delta of a context shift the memory already applied to the layer's own K cache; a layer
     * with nothing to rotate does nothing.
     *
     * @throws IllegalArgumentException      when {@code nKeep < 0} or {@code nDiscard <= 0}
     * @throws UnsupportedOperationException when {@link #refusal} names a reason
     */
    void apply(int nKeep, int nDiscard) {
        if (nKeep < 0 || nDiscard <= 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard > 0, got " + nKeep + "/"
                    + nDiscard);
        }
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        if (shift != null) {
            shift.apply();
        }
    }

    @Override
    public void close() {
        if (shift != null) {
            shift.close();
        }
    }
}
