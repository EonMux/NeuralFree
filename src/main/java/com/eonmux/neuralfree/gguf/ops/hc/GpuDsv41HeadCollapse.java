package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.head.GpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.List;
import java.util.Objects;

/**
 * Collapses the final hyper-connection streams of a DeepSeek-V4.1 model with the last layer's FFN pre-mix, the role
 * {@code output_hc_fn} plays in DeepSeek-V4 (models/deepseek41.cpp:830, 918-923). The last layer's
 * {@link GpuDsv41HcExtra} saved that pre-mix ({@link HcSubBlock#TEMP_PREVIOUS_PRE}) after its attention block; the
 * head gathers the saved pre of each output row and collapses the row with it. Raw row {@code r} is collapsed into
 * {@code hc.collapsed} first because the output rows pack over the wider stream. Not thread-safe.
 */
public final class GpuDsv41HeadCollapse implements GpuHeadPreprocessor {

    private GpuOps ops;
    private GpuHcSite site;
    private ScratchSlot previousPre;
    private ScratchSlot collapsed;
    private int ubatch;

    @Override
    public boolean appliesTo(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return spec.arch() == Arch.DEEPSEEK41 && HcSubBlock.headApplies(spec);
    }

    @Override
    public void bind(GpuForwardContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        if (site != null) {
            throw new IllegalStateException("DeepSeek-V4.1 hyper-connection head is already bound");
        }
        ModelSpec spec = ctx.spec();
        List<LayerSpec> layers = spec.layers();
        LayerSpec last = layers.get(layers.size() - 1);
        HcSubBlock block = HcSubBlock.forLayer(spec, last, true);
        this.ops = ctx.ops();
        this.ubatch = ctx.scratch().ubatch();
        this.previousPre = ctx.scratch().temp(HcSubBlock.TEMP_PREVIOUS_PRE,
                Math.multiplyExact((long) ubatch, (long) block.hc()));
        this.collapsed = ctx.scratch().temp(HcSubBlock.TEMP_COLLAPSED,
                Math.multiplyExact((long) ubatch, (long) block.nEmbd()));
        this.site = new GpuHcSite(ctx, ctx.weightSource().weightsFor(last.weightLayer()), block);
    }

    @Override
    public void apply(int nRows, ScratchSlot hidden) {
        throw new IllegalStateException("the DeepSeek-V4.1 head needs the source rows of its last pre-mix");
    }

    /**
     * @throws IllegalStateException    before {@link #bind}
     * @throws IllegalArgumentException when {@code nRows} is outside {@code [1, ubatch]} or a row is outside the step
     */
    @Override
    public void apply(int nRows, ScratchSlot hidden, int[] rowIdx) {
        Objects.requireNonNull(hidden, "hidden");
        Objects.requireNonNull(rowIdx, "rowIdx");
        if (site == null) {
            throw new IllegalStateException("DeepSeek-V4.1 hyper-connection head used before bind()");
        }
        HcSubBlock b = site.block();
        if (nRows < 1 || nRows > ubatch || nRows > rowIdx.length) {
            throw new IllegalArgumentException("hyper-connection head over " + nRows + " rows outside [1, "
                    + Math.min(ubatch, rowIdx.length) + "]");
        }
        hidden.require((long) nRows * b.streamWidth());
        for (int r = 0; r < nRows; r++) {
            int src = rowIdx[r];
            if (src < 0 || src >= ubatch) {
                throw new IllegalArgumentException("output row " + src + " outside the step's " + ubatch + " rows");
            }
            site.collapseWithPre(1, hidden.buffer(), hidden.offset() + r * b.streamWidth(), previousPre.buffer(),
                    previousPre.offset() + src * b.hc(), collapsed.buffer(), collapsed.offset() + r * b.nEmbd());
        }
        ops.copy(collapsed.buffer(), collapsed.offset(), hidden.buffer(), hidden.offset(), nRows * b.nEmbd());
    }

    @Override
    public void close() {
        GpuHcSite s = site;
        site = null;
        ops = null;
        if (s != null) {
            s.close();
        }
    }

    @Override
    public String toString() {
        return "GpuDsv41HeadCollapse{" + (site != null ? site.block() : "unbound") + '}';
    }
}
