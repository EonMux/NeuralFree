package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.head.CpuHeadPreprocessor;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.List;
import java.util.Objects;

/**
 * Collapses the final hyper-connection streams of a DeepSeek-V4.1 model with the last layer's FFN pre-mix, the role
 * {@code output_hc_fn} plays in DeepSeek-V4 (models/deepseek41.cpp:830, 918-923). The last layer's
 * {@link CpuDsv41HcExtra} saved that pre-mix ({@link HcSubBlock#TEMP_PREVIOUS_PRE}) after its attention block; the
 * head gathers the saved pre of each output row and collapses the row with it. Not thread-safe.
 */
public final class CpuDsv41HeadCollapse implements CpuHeadPreprocessor {

    private HcSubBlock block;
    private float[] previousPre;
    private int ubatch;

    @Override
    public boolean appliesTo(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return spec.arch() == Arch.DEEPSEEK41 && HcSubBlock.headApplies(spec);
    }

    @Override
    public void bind(CpuForwardContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        if (block != null) {
            throw new IllegalStateException("DeepSeek-V4.1 hyper-connection head is already bound");
        }
        ModelSpec spec = ctx.spec();
        List<LayerSpec> layers = spec.layers();
        this.block = HcSubBlock.forLayer(spec, layers.get(layers.size() - 1), true);
        this.ubatch = ctx.scratch().ubatch();
        this.previousPre = ctx.scratch().temp(HcSubBlock.TEMP_PREVIOUS_PRE,
                Math.multiplyExact(ubatch, block.hc()));
    }

    @Override
    public void apply(int nRows, float[] hidden) {
        throw new IllegalStateException("the DeepSeek-V4.1 head needs the source rows of its last pre-mix");
    }

    @Override
    public void apply(int nRows, float[] hidden, int[] rowIdx) {
        Objects.requireNonNull(hidden, "hidden");
        Objects.requireNonNull(rowIdx, "rowIdx");
        if (block == null) {
            throw new IllegalStateException("DeepSeek-V4.1 hyper-connection head used before bind()");
        }
        if (nRows < 1 || nRows > ubatch || nRows > rowIdx.length) {
            throw new IllegalArgumentException("hyper-connection head over " + nRows + " rows outside [1, "
                    + Math.min(ubatch, rowIdx.length) + "]");
        }
        CpuHyperConnections.requireLength("hidden", hidden, (long) nRows * block.streamWidth());
        for (int r = 0; r < nRows; r++) {
            int src = rowIdx[r];
            if (src < 0 || src >= ubatch) {
                throw new IllegalArgumentException("output row " + src + " outside the step's " + ubatch + " rows");
            }
            CpuHyperConnections.pre(block.pre(), block.hc(), block.nEmbd(), 1, hidden, r * block.streamWidth(),
                    previousPre, src * block.hc(), hidden, r * block.nEmbd());
        }
    }

    @Override
    public void close() {
        block = null;
        previousPre = null;
    }

    @Override
    public String toString() {
        return "CpuDsv41HeadCollapse{" + (block != null ? block : "unbound") + '}';
    }
}
