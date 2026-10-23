package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;

/**
 * Hyper-connections of one layer on the CPU ({@code LayerSpec.Extras.HC}), the twin of {@link GpuHcExtra} with the
 * same executor contract: {@code x} holds {@code n} stream rows of width {@code hc*n_embd} at layer boundaries and
 * {@code n} collapsed rows of width {@code n_embd} between {@link #beforeBlock} and the return of {@link #afterFfn};
 * both residual hooks return true. Steps:
 * <ol>
 *   <li>{@link #beforeBlock}: copy the stream of {@code x} to {@code hc.stream}; collapse it with the
 *       {@code hc_attn_*} site into {@code x};</li>
 *   <li>{@link #afterMixer}: {@code hc.stream = post(branch, hc.stream)}; collapse it with the {@code hc_ffn_*} site
 *       into {@code x};</li>
 *   <li>{@link #afterFfn}: {@code x = post(branch, hc.stream)}.</li>
 * </ol>
 * Not thread-safe; one forward pass at a time.
 */
public final class CpuHcExtra implements CpuLayerExtra {

    private CpuHcSite attn;
    private CpuHcSite ffn;
    private float[] stream;
    private int streamWidth;
    private int nEmbd;

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (attn != null) {
            throw new IllegalStateException("hyper-connection extra of layer " + layer.il() + " is already bound");
        }
        HcSubBlock attnBlock = HcSubBlock.forLayer(ctx.spec(), layer, false);
        HcSubBlock ffnBlock = HcSubBlock.forLayer(ctx.spec(), layer, true);
        this.streamWidth = attnBlock.streamWidth();
        this.nEmbd = attnBlock.nEmbd();
        long floats = Math.multiplyExact((long) ctx.scratch().ubatch(), (long) streamWidth);
        if (floats > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("hyper-connection stream of " + floats
                    + " floats exceeds a Java array; lower neuralfree.ubatch");
        }
        this.stream = ctx.scratch().temp(HcSubBlock.TEMP_STREAM, (int) floats);
        CpuHcSite f = new CpuHcSite(ctx, ffnBlock);
        this.attn = new CpuHcSite(ctx, attnBlock);
        this.ffn = f;
    }

    @Override
    public void beforeBlock(int pos0, int n, float[] x) {
        requireBound();
        CpuHyperConnections.requireLength("x", x, (long) n * streamWidth);
        System.arraycopy(x, 0, stream, 0, n * streamWidth);
        attn.collapse(n, stream, 0, x, 0);
    }

    @Override
    public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
        requireBound();
        CpuHyperConnections.requireLength("branch", branch, (long) n * nEmbd);
        attn.expand(n, branch, 0, stream, 0, stream, 0);
        ffn.collapse(n, stream, 0, x, 0);
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
            throw new IllegalStateException("hyper-connection extra used before bind()");
        }
    }

    @Override
    public void close() {
        attn = null;
        ffn = null;
        stream = null;
    }

    @Override
    public String toString() {
        return "CpuHcExtra{" + (attn != null ? attn.block() + ", " + ffn.block() : "unbound") + '}';
    }
}
