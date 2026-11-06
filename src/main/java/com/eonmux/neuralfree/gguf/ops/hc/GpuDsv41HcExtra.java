package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.Arrays;
import java.util.Objects;

/**
 * Hyper-connections of one DeepSeek-V4.1 layer on the GPU ({@code LayerSpec.Extras.HC}), the twin of
 * {@link CpuDsv41HcExtra}. The V4.1 graph runs one mix computation per sub-block: the attention collapse uses the
 * previous sub-block's pre-mix, the FFN collapse uses the attention pre-mix of the same layer, and the output head
 * reuses the last FFN pre-mix instead of {@code output_hc} (models/deepseek41.cpp:830-868, 918-923). The first
 * layer's attention collapses with stream 0, the identity pre-mix. Steps:
 * <ol>
 *   <li>{@link #beforeBlock}: copy the stream, compute the attention coefficients, collapse it with the saved pre-mix
 *       of the previous layer's FFN (stream 0 on the first layer);</li>
 *   <li>{@link #afterMixer}: {@code stream = post(branch, stream)}, collapse it with the attention pre-mix, then
 *       compute the FFN coefficients and save their pre-mix for the next layer;</li>
 *   <li>{@link #afterFfn}: {@code x = post(branch, stream)} with the FFN coefficients.</li>
 * </ol>
 * Every hook is a per-row collapse or expand, exact over any step layout. Not thread-safe; one forward pass at a
 * time.
 */
public final class GpuDsv41HcExtra implements GpuLayerExtra {

    private GpuOps ops;
    private GpuHcSite attn;
    private GpuHcSite ffn;
    private ScratchSlot stream;
    /** Pre-mix of the previous layer's FFN, shared with the next layer's extra. */
    private ScratchSlot previousPre;
    private int hc;
    private int nEmbd;
    private int streamWidth;
    private boolean first;

    @Override
    public String batchRefusal() {
        return null;
    }

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (attn != null) {
            throw new IllegalStateException("DeepSeek-V4.1 hyper-connection extra of layer " + layer.il()
                    + " is already bound");
        }
        ModelSpec spec = ctx.spec();
        HcSubBlock attnBlock = HcSubBlock.forLayer(spec, layer, false);
        HcSubBlock ffnBlock = HcSubBlock.forLayer(spec, layer, true);
        this.hc = attnBlock.hc();
        this.nEmbd = attnBlock.nEmbd();
        this.streamWidth = attnBlock.streamWidth();
        this.first = spec.layers().isEmpty() || spec.layers().get(0).il() == layer.il();
        this.ops = ctx.ops();
        int ubatch = ctx.scratch().ubatch();
        this.stream = ctx.scratch().temp(HcSubBlock.TEMP_STREAM,
                Math.multiplyExact((long) ubatch, (long) streamWidth));
        this.previousPre = ctx.scratch().temp(HcSubBlock.TEMP_PREVIOUS_PRE,
                Math.multiplyExact((long) ubatch, (long) hc));
        com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights weights =
                ctx.weightSource().weightsFor(layer.weightLayer());
        GpuHcSite a = new GpuHcSite(ctx, weights, attnBlock);
        try {
            this.ffn = new GpuHcSite(ctx, weights, ffnBlock);
        } catch (RuntimeException e) {
            a.close();
            throw e;
        }
        this.attn = a;
    }

    @Override
    public void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
        requireBound();
        x.require((long) nTokens * streamWidth);
        ops.copy(x.buffer(), x.offset(), stream.buffer(), stream.offset(), nTokens * streamWidth);
        attn.computeCoefficients(nTokens, stream.buffer(), stream.offset());
        if (first) {
            ops.copy2d(stream.buffer(), stream.offset(), streamWidth, x.buffer(), x.offset(), nEmbd, nTokens, nEmbd);
        } else {
            attn.collapseWithPre(nTokens, stream.buffer(), stream.offset(), previousPre.buffer(),
                    previousPre.offset(), x.buffer(), x.offset());
        }
    }

    @Override
    public boolean afterMixer(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        requireBound();
        branch.require((long) nTokens * nEmbd);
        attn.expand(nTokens, branch.buffer(), branch.offset(), stream.buffer(), stream.offset(), stream.buffer(),
                stream.offset());
        attn.collapseWithPre(nTokens, stream.buffer(), stream.offset(), attn.preBuffer(), attn.preOffset(),
                x.buffer(), x.offset());
        ffn.computeCoefficients(nTokens, stream.buffer(), stream.offset());
        ops.copy2d(ffn.preBuffer(), ffn.preOffset(), hc, previousPre.buffer(), previousPre.offset(), hc, nTokens, hc);
        return true;
    }

    @Override
    public boolean afterFfn(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        requireBound();
        branch.require((long) nTokens * nEmbd);
        ffn.expand(nTokens, branch.buffer(), branch.offset(), stream.buffer(), stream.offset(), x.buffer(), x.offset());
        return true;
    }

    private void requireBound() {
        if (attn == null) {
            throw new IllegalStateException("DeepSeek-V4.1 hyper-connection extra used before bind()");
        }
    }

    @Override
    public void close() {
        GpuHcSite a = attn;
        GpuHcSite f = ffn;
        attn = null;
        ffn = null;
        RuntimeException first = Closeables.closeAll(Arrays.asList(a, f), null);
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuDsv41HcExtra{" + (attn != null ? attn.block() + ", " + ffn.block() : "unbound") + '}';
    }
}
