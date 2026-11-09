package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;

import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * The gated mixed attention of an IQuestLoopCoder layer ({@code spec.dual()}), the CPU twin of the mixer's dual
 * flash: the local attention over the layer's own cache with a sliding window, the global attention read-only over
 * the first pass's cache ({@code spec.dualReadLayer()}) with a full causal window, then the per-head gate
 * {@code sigmoid(dot(q_head, w_head) + b_head)} mixing them. Not thread-safe; driven by the owning mixer's step
 * thread.
 */
final class CpuLoopAttention {

    private static final String T_DUAL = "attn.dual";
    private static final String T_LOOP_GATE = "attn.loop_gate";

    private final CpuAttentionMixer mixer;
    private KvAccessor dualKv;
    private IntUnaryOperator dualRowPos;
    private HostKvAttention dualAttention;
    private CpuAttentionCore.Params localParams;
    private CpuAttentionCore.Params localNonCausalParams;
    private float[] loopGateW;
    private float[] loopGateB;

    CpuLoopAttention(CpuAttentionMixer mixer) {
        this.mixer = Objects.requireNonNull(mixer, "mixer");
    }

    /**
     * Binds the gated mixed attention of IQuestLoopCoder: the first pass's cache of the same block layer, read
     * read-only over a full causal window, its own cache over a sliding window, and the per-head gate projection.
     */
    void bind(CpuForwardContext ctx, LayerSpec layer, CpuWeights w) {
        if (mixer.kv() == null) {
            throw new IllegalStateException("layer " + layer.il() + " mixes two attentions but owns no KV cache");
        }
        CpuMemory memory = ctx.memory();
        dualKv = memory.kv(mixer.spec().dualReadLayer());
        if (dualKv == null) {
            throw new IllegalStateException("layer " + layer.il() + " reads the cache of layer "
                    + mixer.spec().dualReadLayer() + ", which the memory plan does not hold");
        }
        if (dualKv == mixer.kv()) {
            throw new IllegalStateException("layer " + layer.il() + " reads its own cache as the global attention");
        }
        mixer.checkCache(dualKv.layout());
        final int group = dualKv.group();
        dualRowPos = r -> memory.positionOfRow(group, r);
        dualAttention = new HostKvAttention(dualKv, dualRowPos, mixer.spec().qDim(), 0, 0, mixer.spec().outDim(),
                mixer.spec().headDimK(), mixer.spec().headDimV(), null, null);
        KvWindow window = KvWindow.of(SwaType.STANDARD, mixer.spec().dualWindow(), true);
        localParams = mixer.params().withWindow(window);
        localNonCausalParams = mixer.nonCausalParams().withWindow(window);
        String gateW = TensorNames.weight(TensorNames.ATTN_LOOP_GATE);
        loopGateW = w.layerFloatsOrNull(layer, gateW);
        if (loopGateW == null) {
            throw new MissingTensorException(ctx.spec().arch().ggufName(), layer.il(), gateW);
        }
        QkvProjection.checkLength(gateW, loopGateW.length, (long) mixer.spec().nHead() * mixer.spec().headDimK());
        String gateB = TensorNames.bias(TensorNames.ATTN_LOOP_GATE);
        loopGateB = w.layerFloatsOrNull(layer, gateB);
        if (loopGateB == null) {
            throw new MissingTensorException(ctx.spec().arch().ggufName(), layer.il(), gateB);
        }
        QkvProjection.checkLength(gateB, loopGateB.length, mixer.spec().nHead());
    }

    /** Why the gated mixed attention refuses an input: it reads its two caches over contiguous rows. */
    String pagedRefusal() {
        return "layer " + mixer.layer().il() + " mixes a global and a local attention; NeuralFree runs that pair over"
                + " the contiguous KV cache only, not over paged memory";
    }

    /**
     * The gated mixed attention of IQuestLoopCoder: the local attention over the layer's own cache with a sliding
     * window, the global attention read-only over the first pass's cache with a full causal window, then the per-head
     * gate {@code sigmoid(dot(q_head, w_head) + b_head)} mixing them.
     */
    void attend(CpuAttentionCore.Params stepParams, int pos0, int n, float[] q, int[] pos, float[] attn) {
        CpuAttentionCore.Params local = mixer.ctx().stepNonCausal() ? localNonCausalParams : localParams;
        KvAccessor kv = mixer.kv();
        int keys = Math.min(kv.layout().rows(), pos0 + n);
        CpuAttentionCore.attendRows(local, n, q, 0, mixer.spec().qDim(), pos, 0, kv, mixer.rowPos(), 0, keys, attn, 0,
                mixer.spec().outDim());
        float[] global = mixer.scratch().temp(T_DUAL, mixer.ubatch() * mixer.spec().outDim());
        int globalKeys = Math.min(dualKv.layout().rows(), pos0 + n);
        CpuAttentionCore.attendRows(stepParams, n, q, 0, mixer.spec().qDim(), pos, 0, dualKv, dualRowPos, 0,
                globalKeys, global, 0, mixer.spec().outDim());
        float[] gate = mixer.scratch().temp(T_LOOP_GATE, mixer.ubatch() * mixer.spec().nHead());
        int headDim = mixer.spec().headDimK();
        int valueDim = mixer.spec().headDimV();
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < mixer.spec().nHead(); h++) {
                float sum = loopGateB[h];
                for (int d = 0; d < headDim; d++) {
                    sum += q[t * mixer.spec().qDim() + h * headDim + d] * loopGateW[h * headDim + d];
                }
                gate[t * mixer.spec().nHead() + h] = ActMath.sigmoid(sum);
            }
        }
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < mixer.spec().nHead(); h++) {
                float g = gate[t * mixer.spec().nHead() + h];
                int base = t * mixer.spec().outDim() + h * valueDim;
                for (int d = 0; d < valueDim; d++) {
                    float localValue = attn[base + d];
                    attn[base + d] = localValue * (1.0f - g) + global[base + d] * g;
                }
            }
        }
    }
}
