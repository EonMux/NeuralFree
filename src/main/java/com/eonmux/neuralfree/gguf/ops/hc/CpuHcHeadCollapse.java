package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.head.CpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * Collapses the hyper-connection streams of the final hidden rows to {@code n_embd} before the output norm on the
 * CPU, the twin of {@link GpuHcHeadCollapse} ({@link HcSubBlock#forHead}: HC head, qwen4exp gated head or stream
 * mean). Per {@link #apply}: the site's collapse into {@code hc.collapsed}, then a copy back to index 0. Not
 * thread-safe.
 */
public final class CpuHcHeadCollapse implements CpuHeadPreprocessor {

    private CpuHcSite site;
    private float[] collapsed;
    private int ubatch;

    @Override
    public boolean appliesTo(ModelSpec spec) {
        return HcSubBlock.headApplies(Objects.requireNonNull(spec, "spec"));
    }

    @Override
    public void bind(CpuForwardContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        if (site != null) {
            throw new IllegalStateException("hyper-connection head is already bound");
        }
        HcSubBlock block = HcSubBlock.forHead(ctx.spec());
        this.ubatch = ctx.scratch().ubatch();
        long floats = Math.multiplyExact((long) ubatch, (long) block.nEmbd());
        if (floats > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("hyper-connection head rows of " + floats
                    + " floats exceed a Java array; lower neuralfree.ubatch");
        }
        this.collapsed = ctx.scratch().temp(HcSubBlock.TEMP_COLLAPSED, (int) floats);
        this.site = new CpuHcSite(ctx, block);
    }

    /**
     * @throws IllegalStateException    before {@link #bind}
     * @throws IllegalArgumentException when {@code nRows} is outside {@code [1, ubatch]} or {@code hidden} is too short
     */
    @Override
    public void apply(int nRows, float[] hidden) {
        if (site == null) {
            throw new IllegalStateException("hyper-connection head used before bind()");
        }
        if (nRows < 1 || nRows > ubatch) {
            throw new IllegalArgumentException("hyper-connection head: row count " + nRows + " is outside [1, "
                    + ubatch + "]");
        }
        HcSubBlock b = site.block();
        CpuHyperConnections.requireLength("hidden", hidden, (long) nRows * b.streamWidth());
        site.collapse(nRows, hidden, 0, collapsed, 0);
        System.arraycopy(collapsed, 0, hidden, 0, nRows * b.nEmbd());
    }

    @Override
    public void close() {
        site = null;
        collapsed = null;
    }

    @Override
    public String toString() {
        return "CpuHcHeadCollapse{" + (site != null ? site.block() : "unbound") + '}';
    }
}
