package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * The per-use Q/K/V adapters of a shared attention block (Zamba2) in {@link GpuAttentionMixer}:
 * {@code blk.<qkvAdapterLayer>.attn_{q,k,v}_adapter_a} maps the block input to the adapter rank and {@code _b} back
 * to the projection width; their product is added to the base projections before RoPE (vLLM {@code Zamba2LoRA}).
 * Not thread-safe; driven by the owning mixer's step thread.
 */
final class GpuQkvAdapters {

    private static final String T_ADAPTER = "attn.adapter_rank";

    private final GpuAttentionMixer mixer;

    /** Per-use Q/K/V adapters of a shared attention block (Zamba2); null without them. */
    private GpuWeight wqAdapterA;
    private GpuWeight wqAdapterB;
    private GpuWeight wkAdapterA;
    private GpuWeight wkAdapterB;
    private GpuWeight wvAdapterA;
    private GpuWeight wvAdapterB;
    private ScratchSlot adapterRankBuf;

    GpuQkvAdapters(GpuAttentionMixer mixer) {
        this.mixer = mixer;
    }

    /**
     * Binds the adapters of the spec's shared attention layer and the rank temporary their products run in.
     *
     * @throws IllegalArgumentException when an adapter has the wrong shape
     */
    void bind(GpuScratch scratch) {
        int rank = mixer.spec().qkvAdapterRank();
        if (rank <= 0) {
            return;
        }
        GpuWeights w = mixer.ctx().weightSource().weightsFor(mixer.spec().qkvAdapterLayer());
        int il = mixer.spec().qkvAdapterLayer();
        int in = (int) mixer.wq().cols();
        wqAdapterA = adapter(w, il, "attn_q_adapter_a", in, rank);
        wqAdapterB = adapter(w, il, "attn_q_adapter_b", rank, mixer.spec().qDim());
        wkAdapterA = adapter(w, il, "attn_k_adapter_a", in, rank);
        wkAdapterB = adapter(w, il, "attn_k_adapter_b", rank, mixer.spec().kDim());
        wvAdapterA = adapter(w, il, "attn_v_adapter_a", in, rank);
        wvAdapterB = adapter(w, il, "attn_v_adapter_b", rank, mixer.spec().vDim());
        adapterRankBuf = scratch.temp(T_ADAPTER, (long) mixer.ubatch() * rank);
    }

    private GpuWeight adapter(GpuWeights w, int il, String base, long cols, long rows) {
        GpuWeight a = w.layer(il, TensorNames.weight(base));
        QkvProjection.checkShape(a.name(), a.cols(), a.rows(), cols, rows);
        return a;
    }

    /** Adds the per-use adapter products to the projected Q, K and V rows of the step. */
    void run(int n, ScratchSlot in) {
        if (adapterRankBuf == null) {
            return;
        }
        adapterRankBuf.require((long) n * mixer.spec().qkvAdapterRank());
        addAdapter(wqAdapterA, wqAdapterB, n, in, mixer.qProjBuf());
        addAdapter(wkAdapterA, wkAdapterB, n, in, mixer.kBuf());
        addAdapter(wvAdapterA, wvAdapterB, n, in, mixer.vBuf());
    }

    private void addAdapter(GpuWeight a, GpuWeight b, int n, ScratchSlot in, GpuBuffer target) {
        mixer.ops().matmul(a, in.buffer(), in.offset(), adapterRankBuf.buffer(), adapterRankBuf.offset(), n);
        mixer.ops().matvecAdd(b, adapterRankBuf.buffer(), adapterRankBuf.offset(), target, 0, n);
    }

    /** Whether the layer runs adapters (the spec sets {@code qkvAdapterRank}). */
    boolean bound() {
        return adapterRankBuf != null;
    }
}
