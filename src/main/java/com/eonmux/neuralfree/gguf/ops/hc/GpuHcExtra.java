package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.Arrays;
import java.util.Objects;

/**
 * Hyper-connections of one layer on the GPU ({@code LayerSpec.Extras.HC}; deepseek4, glm5-next, hy_v4, qwen4exp,
 * dflash): the residual stream holds {@code hc} streams per token ({@code S = hc*n_embd} values) and each sub-block
 * reads a collapse of them and writes back through its post map instead of a residual add.
 *
 * <p>Contract with the layer executor. At layer boundaries {@code x} holds {@code T} stream rows of width {@code S}
 * ({@code ResidualSpec.streamWidth()}). Between {@link #beforeBlock} and the return of {@link #afterFfn}, {@code x}
 * holds {@code T} collapsed rows of width {@code n_embd} packed from its offset, which the executor normalizes and
 * feeds to the mixer and the FFN; {@code branch} carries {@code T} rows of width {@code n_embd}. Both residual hooks
 * return true, so the executor adds no residual of its own:
 * <ol>
 *   <li>{@link #beforeBlock}: copy the stream of {@code x} to {@code hc.stream}; collapse it with the
 *       {@code hc_attn_*} site into {@code x} (deepseek4.cpp:1255-1259);</li>
 *   <li>{@link #afterMixer}: {@code hc.stream = post(branch, hc.stream)} in place (deepseek4.cpp:1266); collapse it
 *       with the {@code hc_ffn_*} site into {@code x} (:1269-1274);</li>
 *   <li>{@link #afterFfn}: {@code x = post(branch, hc.stream)}, the next layer's stream (:1325).</li>
 * </ol>
 * The per-site dispatches are listed in {@link GpuHcSite}. The stream temporary is shared by every hyper-connection
 * layer of the model (a layer uses it only between its own {@code beforeBlock} and {@code afterFfn}). Not
 * thread-safe; one forward pass at a time.</p>
 */
public final class GpuHcExtra implements GpuLayerExtra {

    private GpuOps ops;
    private GpuHcSite attn;
    private GpuHcSite ffn;
    private ScratchSlot stream;
    private int streamWidth;
    private int nEmbd;

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (attn != null) {
            throw new IllegalStateException("hyper-connection extra of layer " + layer.il() + " is already bound");
        }
        HcSubBlock attnBlock = HcSubBlock.forLayer(ctx.spec(), layer, false);
        HcSubBlock ffnBlock = HcSubBlock.forLayer(ctx.spec(), layer, true);
        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        this.ops = ctx.ops();
        this.streamWidth = attnBlock.streamWidth();
        this.nEmbd = attnBlock.nEmbd();
        this.stream = ctx.scratch().temp(HcSubBlock.TEMP_STREAM,
                Math.multiplyExact((long) ctx.scratch().ubatch(), (long) streamWidth));
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
        attn.collapse(nTokens, stream.buffer(), stream.offset(), x.buffer(), x.offset());
    }

    @Override
    public boolean afterMixer(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        requireBound();
        branch.require((long) nTokens * nEmbd);
        attn.expand(nTokens, branch.buffer(), branch.offset(), stream.buffer(), stream.offset(), stream.buffer(),
                stream.offset());
        ffn.collapse(nTokens, stream.buffer(), stream.offset(), x.buffer(), x.offset());
        return true;
    }

    @Override
    public boolean afterFfn(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        requireBound();
        branch.require((long) nTokens * nEmbd);
        x.require((long) nTokens * streamWidth);
        ffn.expand(nTokens, branch.buffer(), branch.offset(), stream.buffer(), stream.offset(), x.buffer(),
                x.offset());
        return true;
    }

    private void requireBound() {
        if (attn == null) {
            throw new IllegalStateException("hyper-connection extra used before bind()");
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
        return "GpuHcExtra{" + (attn != null ? attn.block() + ", " + ffn.block() : "unbound") + '}';
    }
}
