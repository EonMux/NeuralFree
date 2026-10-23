package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * Collapses the hyper-connection streams of the final hidden rows to {@code n_embd} before the output norm on the
 * GPU ({@link HcSubBlock#forHead}): the HC head of deepseek4, hy_v4 and dflash ({@code output_hc_fn/scale/base},
 * deepseek4.cpp:1352-1353), the qwen4exp gated head that also replaces the output norm (qwen4exp.cpp:503-511), or
 * the stream mean of glm5-next (glm5-next.cpp:791).
 *
 * <p>Per {@link #apply} of {@code R} rows: the site's collapse dispatches ({@link GpuHcSite}) from the hidden rows
 * into {@code hc.collapsed}, then one {@code LLM_Copy2D_F32} of {@code R*n_embd} floats back to the slot's start.
 * Each row is collapsed from its own streams, so the rows may be any subset of the step's tokens. Not thread-safe.</p>
 */
public final class GpuHcHeadCollapse implements GpuHeadPreprocessor {

    private GpuOps ops;
    private GpuHcSite site;
    private ScratchSlot collapsed;
    private int ubatch;

    @Override
    public boolean appliesTo(ModelSpec spec) {
        return HcSubBlock.headApplies(Objects.requireNonNull(spec, "spec"));
    }

    @Override
    public void bind(GpuForwardContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        if (site != null) {
            throw new IllegalStateException("hyper-connection head is already bound");
        }
        HcSubBlock block = HcSubBlock.forHead(ctx.spec());
        this.ops = ctx.ops();
        this.ubatch = ctx.scratch().ubatch();
        this.collapsed = ctx.scratch().temp(HcSubBlock.TEMP_COLLAPSED,
                Math.multiplyExact((long) ubatch, (long) block.nEmbd()));
        this.site = new GpuHcSite(ctx, ctx.weights(), block);
    }

    /**
     * @throws IllegalStateException    before {@link #bind}
     * @throws IllegalArgumentException when {@code nRows} is outside {@code [1, ubatch]}
     */
    @Override
    public void apply(int nRows, ScratchSlot hidden) {
        if (site == null) {
            throw new IllegalStateException("hyper-connection head used before bind()");
        }
        HcSubBlock b = site.block();
        if (nRows < 1 || nRows > ubatch) {
            throw new IllegalArgumentException("hyper-connection head: row count " + nRows + " is outside [1, "
                    + ubatch + "]");
        }
        hidden.require((long) nRows * b.streamWidth());
        site.collapse(nRows, hidden.buffer(), hidden.offset(), collapsed.buffer(), collapsed.offset());
        ops.copy(collapsed.buffer(), collapsed.offset(), hidden.buffer(), hidden.offset(), nRows * b.nEmbd());
    }

    @Override
    public void close() {
        GpuHcSite s = site;
        site = null;
        if (s != null) {
            s.close();
        }
    }

    @Override
    public String toString() {
        return "GpuHcHeadCollapse{" + (site != null ? site.block() : "unbound") + '}';
    }
}
