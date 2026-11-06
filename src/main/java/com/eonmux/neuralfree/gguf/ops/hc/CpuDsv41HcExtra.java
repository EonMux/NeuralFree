package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * Hyper-connections of one DeepSeek-V4.1 layer on the CPU ({@code LayerSpec.Extras.HC}). The V4.1 graph runs one
 * mix computation per sub-block: the attention collapse uses the previous sub-block's pre-mix, the FFN collapse uses
 * the attention pre-mix of the same layer, and the output head reuses the last FFN pre-mix instead of
 * {@code output_hc} (models/deepseek41.cpp:830-868, 918-923). The first layer's attention collapses with stream 0,
 * the identity pre-mix. Steps:
 * <ol>
 *   <li>{@link #beforeBlock}: copy the stream, compute the attention coefficients, collapse it with the saved pre-mix
 *       of the previous layer's FFN (stream 0 on the first layer);</li>
 *   <li>{@link #afterMixer}: {@code stream = post(branch, stream)}, collapse it with the attention pre-mix, then
 *       compute the FFN coefficients and save their pre-mix for the next layer;</li>
 *   <li>{@link #afterFfn}: {@code x = post(branch, stream)} with the FFN coefficients.</li>
 * </ol>
 * Not thread-safe; one forward pass at a time.
 */
public final class CpuDsv41HcExtra implements CpuLayerExtra {

    private CpuHcSite attn;
    private CpuHcSite ffn;
    private float[] stream;
    /** Pre-mix of the previous layer's FFN, shared with the next layer's extra. */
    private float[] previousPre;
    private int hc;
    private int nEmbd;
    private int streamWidth;
    private boolean first;

    /** Every hook is a per-row collapse or expand, exact over any step layout. */
    @Override
    public String batchRefusal() {
        return null;
    }

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
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
        long floats = Math.multiplyExact((long) ctx.scratch().ubatch(), (long) streamWidth);
        if (floats > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("hyper-connection stream of " + floats
                    + " floats exceeds a Java array; lower neuralfree.ubatch");
        }
        this.stream = ctx.scratch().temp(HcSubBlock.TEMP_STREAM, (int) floats);
        this.previousPre = ctx.scratch().temp(HcSubBlock.TEMP_PREVIOUS_PRE,
                Math.multiplyExact(ctx.scratch().ubatch(), hc));
        this.ffn = new CpuHcSite(ctx, ffnBlock);
        this.attn = new CpuHcSite(ctx, attnBlock);
    }

    @Override
    public void beforeBlock(int pos0, int n, float[] x) {
        requireBound();
        CpuHyperConnections.requireLength("x", x, (long) n * streamWidth);
        System.arraycopy(x, 0, stream, 0, n * streamWidth);
        attn.computeCoefficients(n, stream, 0);
        if (first) {
            for (int t = 0; t < n; t++) {
                System.arraycopy(stream, t * streamWidth, x, t * nEmbd, nEmbd);
            }
        } else {
            attn.collapseWithPre(n, stream, 0, previousPre, 0, x, 0);
        }
    }

    @Override
    public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
        requireBound();
        CpuHyperConnections.requireLength("branch", branch, (long) n * nEmbd);
        attn.expand(n, branch, 0, stream, 0, stream, 0);
        attn.collapseWithPre(n, stream, 0, attn.preCoefficients(), 0, x, 0);
        ffn.computeCoefficients(n, stream, 0);
        System.arraycopy(ffn.preCoefficients(), 0, previousPre, 0, n * hc);
        return true;
    }

    @Override
    public boolean afterFfn(int pos0, int n, float[] x, float[] branch) {
        requireBound();
        CpuHyperConnections.requireLength("branch", branch, (long) n * nEmbd);
        ffn.expand(n, branch, 0, stream, 0, x, 0);
        return true;
    }

    private void requireBound() {
        if (attn == null) {
            throw new IllegalStateException("DeepSeek-V4.1 hyper-connection extra used before bind()");
        }
    }

    @Override
    public void close() {
        attn = null;
        ffn = null;
        stream = null;
        previousPre = null;
    }

    @Override
    public String toString() {
        return "CpuDsv41HcExtra{" + (attn != null ? attn.block() + ", " + ffn.block() : "unbound") + '}';
    }
}
