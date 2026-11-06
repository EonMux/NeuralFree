package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;

/**
 * The mixer of a device layer whose KV cache stays in host RAM (llama.cpp {@code -nkvo}, common/arg.cpp;
 * llama-kv-cache.cpp chooses the buffers' device per layer): the device {@link GpuAttentionMixer} runs the layer's
 * projections, norms, RoPE and output, and this wrapper runs the attention over the host cache through a
 * {@link HostAttentionStage}. The Q/K/V rows of a step cross to the host after the device projections and the
 * attention rows cross back before the output projection, the same per-layer activation traffic the hybrid split's
 * host stages carry.
 *
 * <p>{@link #prefillKV} writes the cache rows alone (no query or attention), so the NextN draft passes that only
 * write a cache work unchanged. Context shift over the host cache is refused by name. The wrapper owns the device
 * mixer; {@link #close()} closes it. Not thread-safe.</p>
 */
public final class HostKvMixer implements GpuMixer {

    private final GpuAttentionMixer device;
    private final CpuForwardContext hostContext;
    private int ubatch;
    private HostAttentionStage stage;
    private float[] q;
    private float[] k;
    private float[] v;
    private float[] attn;
    private boolean closed;

    private HostKvMixer(GpuAttentionMixer device, CpuForwardContext hostContext) {
        this.device = Objects.requireNonNull(device, "device");
        this.hostContext = Objects.requireNonNull(hostContext, "hostContext");
    }

    /**
     * The host-KV wrapper of a device mixer, for a registry's mixer decorator.
     *
     * @throws UnsupportedOperationException when the device block is not an attention mixer a host attention can
     *                                       drive
     */
    public static GpuMixer wrap(GpuMixer deviceBlock, CpuForwardContext hostContext) {
        if (!(deviceBlock instanceof GpuAttentionMixer)) {
            throw new UnsupportedOperationException("a host KV cache needs the softmax attention mixer, got "
                    + deviceBlock.getClass().getSimpleName());
        }
        return new HostKvMixer((GpuAttentionMixer) deviceBlock, hostContext);
    }

    /**
     * @throws UnsupportedOperationException when the device mixer cannot split its attention from its projections,
     *                                       or the host memory has no cache for the layer
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        device.bind(ctx, layer);
        String refusal = device.hostAttentionRefusal();
        if (refusal != null) {
            close();
            throw new UnsupportedOperationException("layer " + layer.il() + ": " + refusal);
        }
        AttentionSpec spec = layer.attention();
        if (spec == null || !spec.usesMemory()) {
            close();
            throw new IllegalStateException("layer " + layer.il() + " has no KV cache to keep on the host");
        }
        this.ubatch = ctx.scratch().ubatch();
        this.stage = HostAttentionStage.bind(hostContext.memory(), layer, device.hostSinkValues(),
                device.hostSinkK(), device.hostSinkV(), device.hostKScale(), device.hostVScale(),
                hostContext.scratch().ubatch());
        this.q = new float[Math.multiplyExact(ubatch, spec.qDim())];
        this.attn = new float[Math.multiplyExact(ubatch, spec.outDim())];
        if (device.projectsKv()) {
            this.k = new float[Math.multiplyExact(ubatch, spec.kDim())];
            this.v = new float[Math.multiplyExact(ubatch, spec.vDim())];
        }
    }

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        run(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        run(pos0, nTokens, in, out);
    }

    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        Objects.requireNonNull(s, "s");
        if (!s.contiguousSingle()) {
            throw new UnsupportedOperationException("a host KV cache runs the contiguous step only, got " + s);
        }
        if (s.nTokens() == 1) {
            decode(s.pos0(0), in, out);
        } else {
            prefill(s.pos0(0), s.nTokens(), in, out);
        }
    }

    @Override
    public void prefillKV(int pos0, int nTokens, ScratchSlot in) {
        ensureBound();
        device.projectKvForHostAttention(pos0, nTokens, in, k, v);
        stage.store(pos0, nTokens, k, v);
    }

    @Override
    public String kvRefusal() {
        return null;
    }

    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        throw new UnsupportedOperationException(shiftRefusal());
    }

    @Override
    public String shiftRefusal() {
        return "the KV cache is in host RAM; context shift over a host cache is not supported";
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        device.close();
    }

    private void run(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        device.runWithHostAttention(pos0, nTokens, in, out, q, k, v, attn, stage::attend);
    }

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("the host KV mixer is closed");
        }
        if (stage == null) {
            throw new IllegalStateException("the host KV mixer is not bound");
        }
    }

    @Override
    public String toString() {
        return "HostKvMixer{" + device + '}';
    }
}
