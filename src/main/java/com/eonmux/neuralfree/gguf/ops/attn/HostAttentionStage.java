package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * The attention of one device layer over its host-resident KV cache (llama.cpp {@code -nkvo}): the projected Q, K
 * and V rows cross from the device, the K and V rows are rotated for a quantized cache, stored at the rows of their
 * positions and attended with {@link HostKvAttention}, and the attention rows cross back for the device output
 * projection. The cache is the heap memory's cache of the layer ({@code CpuMemory#kv}); a shared-cache reader
 * attends over its owner's cache and passes no K or V rows.
 *
 * <p>The stage owns no buffers; the per-step arrays belong to the caller. Immutable. Not thread-safe.</p>
 */
final class HostAttentionStage {

    private final KvAccessor kv;
    private final HostKvAttention attention;
    private final CpuAttentionCore.Params params;
    private final CpuQuantizedKvRotation rotation;
    private final CpuQuantizedKvRotation storeRotation;
    private final int[] positions;

    private HostAttentionStage(KvAccessor kv, HostKvAttention attention, CpuAttentionCore.Params params,
                               CpuQuantizedKvRotation rotation, CpuQuantizedKvRotation storeRotation,
                               int[] positions) {
        this.kv = kv;
        this.attention = attention;
        this.params = params;
        this.rotation = rotation;
        this.storeRotation = storeRotation;
        this.positions = positions;
    }

    /**
     * Binds the stage of {@code layer} over its cache in {@code memory}.
     *
     * @param sinks   per-head attention sink logits, or null without sinks
     * @param sinkK   learned static sink keys {@code [n_sink][n_head_kv][d_k]}, or null without sink tokens
     * @param sinkV   learned static sink values {@code [n_sink][n_head_kv][d_v]}, or null without sink tokens
     * @param kScale  per-KV-head FP8 K cache scales, or null for an unscaled cache
     * @param vScale  per-KV-head FP8 V cache scales, or null for an unscaled cache
     * @param ubatch  the largest step the caller runs
     * @throws IllegalStateException         when the layer has no cache or attention spec
     * @throws UnsupportedOperationException for a latent cache or a T5 relative-position bias
     * @throws IllegalArgumentException      when the cache rows do not hold the layer's heads
     */
    static HostAttentionStage bind(CpuMemory memory, LayerSpec layer, float[] sinks, float[] sinkK, float[] sinkV,
                                   float[] kScale, float[] vScale, int ubatch) {
        Objects.requireNonNull(memory, "memory");
        Objects.requireNonNull(layer, "layer");
        AttentionSpec spec = layer.attention();
        if (spec == null || !spec.usesMemory()) {
            throw new IllegalStateException("layer " + layer.il() + " has no KV cache to keep on the host");
        }
        KvAccessor kv = memory.kv(layer.il());
        if (kv == null) {
            throw new IllegalStateException("layer " + layer.il() + " has no KV cache in the host memory plan");
        }
        KvLayout layout = kv.layout();
        if (layout.vAliasesK()) {
            throw new UnsupportedOperationException("layer " + layer.il() + " has a latent (MLA) cache; a host"
                    + " attention over it is not supported");
        }
        if (spec.biasMode() == AttentionSpec.BiasMode.T5_CAUSAL
                || spec.biasMode() == AttentionSpec.BiasMode.T5_BIDIR) {
            throw new UnsupportedOperationException("layer " + layer.il() + ": a T5 relative-position bias needs"
                    + " the bias table the host attention does not carry");
        }
        if (layout.widthK() != spec.kDim() || layout.widthV() != spec.vDim()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": host cache rows of K " + layout.widthK()
                    + " / V " + layout.widthV() + " do not hold " + spec.nHeadKv() + " heads of ("
                    + spec.headDimK() + ", " + spec.headDimV() + ")");
        }
        RotationPlan plan = memory.plan().layer(kv.layer()).rotation();
        CpuQuantizedKvRotation rotation = CpuQuantizedKvRotation.of(plan, spec.qDim(), spec.kDim(), spec.vDim(),
                spec.outDim());
        CpuQuantizedKvRotation storeRotation = CpuQuantizedKvRotation.forStore(plan, spec.kDim(), spec.vDim());
        int group = kv.group();
        IntUnaryOperator rowPosition = row -> memory.positionOfRow(group, row);
        CpuAttentionCore.Params params = CpuAttentionCore.Params.of(spec, sinks, null, kScale, vScale, sinkK, sinkV);
        HostKvAttention attention = new HostKvAttention(kv, rowPosition, spec.qDim(), spec.kDim(), spec.vDim(),
                spec.outDim(), spec.headDimK(), spec.headDimV(), kScale, vScale);
        return new HostAttentionStage(kv, attention, params, rotation, storeRotation, new int[ubatch]);
    }

    /** Stores projected K/V rows (null for a shared-cache reader) at the step's positions. */
    void store(int pos0, int nTokens, float[] k, float[] v) {
        if (k == null && v == null) {
            return;
        }
        int[] pos = positions(pos0, nTokens);
        if (storeRotation != null) {
            storeRotation.beforeStoreKV(nTokens, k, v);
        }
        attention.store(nTokens, pos, k, v);
    }

    /**
     * Stores the step's K/V rows and writes the attention of its rows to {@code out}: the cache rotation of Q/K/V,
     * the store, {@link HostKvAttention#run} and the V rotation of the output.
     */
    void attend(int pos0, int nTokens, float[] q, float[] k, float[] v, float[] out) {
        int[] pos = positions(pos0, nTokens);
        if (rotation != null) {
            rotation.beforeStore(nTokens, q, k, v);
        }
        attention.run(params, pos0, nTokens, pos, q, k, v, out);
        if (rotation != null) {
            rotation.afterAttention(nTokens, out);
        }
    }

    private int[] positions(int pos0, int nTokens) {
        if (nTokens > positions.length) {
            throw new IllegalArgumentException("step of " + nTokens + " tokens exceeds the host attention stage's "
                    + positions.length);
        }
        for (int t = 0; t < nTokens; t++) {
            positions[t] = pos0 + t;
        }
        return positions;
    }

    @Override
    public String toString() {
        return "HostAttentionStage{layer " + kv.layer() + ", " + params + '}';
    }
}
