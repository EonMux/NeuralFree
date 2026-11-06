package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * A mixer-level CPU fixture over a real heap memory: every layer of a hand-built spec gets its mixer bound to one
 * memory, contiguous or paged ({@code parallel(2)}), and steps run the way the CPU backend runs them: the contiguous
 * step marks its rows ({@code beginStep}) and calls {@code decode} / {@code prefill}; the paged step opens through
 * {@link HeapModelMemory#openPagedStep}, fills section 0 of the {@link CpuScratch#POS} slot with the rows' positions,
 * flags a non-causal step on the context and calls {@code batch} on every mixer. Every mixer reads the same input
 * rows; the outputs come back layer after layer ({@code [layer][row][width]}).
 */
public final class CpuMixerRig implements AutoCloseable {

    private final HeapModelMemory memory;
    private final CpuForwardContext ctx;
    private final List<CpuMixer> mixers = new ArrayList<>();
    private final int width;

    private CpuMixerRig(HeapModelMemory memory, CpuForwardContext ctx, int width) {
        this.memory = memory;
        this.ctx = ctx;
        this.width = width;
    }

    /**
     * Plans and allocates the memory of {@code spec} for {@code nCtx} positions, binds one mixer per layer.
     *
     * @param mixers creates the unbound mixer of a layer
     * @param width  values per row of the mixers' input and output
     */
    public static CpuMixerRig open(ModelSpec spec, CpuWeights weights, MemoryConfig cfg, int nCtx, int width,
                                   Function<LayerSpec, CpuMixer> mixers) {
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, nCtx);
        HeapModelMemory memory = HeapModelMemory.allocate(plan, spec, null);
        CpuForwardContext ctx = CpuForwardContext.builder(spec, weights, CpuScratch.forModel(spec, cfg.ubatch()))
                .memory(memory).build();
        CpuMixerRig rig = new CpuMixerRig(memory, ctx, width);
        for (LayerSpec layer : spec.layers()) {
            CpuMixer m = mixers.apply(layer);
            m.bind(ctx, layer);
            rig.mixers.add(m);
        }
        return rig;
    }

    public HeapModelMemory memory() {
        return memory;
    }

    public CpuForwardContext context() {
        return ctx;
    }

    /** The bound mixer of layer {@code layer}, for a step the rig's own drivers do not run. */
    public CpuMixer mixer(int layer) {
        return mixers.get(layer);
    }

    /** One contiguous step of {@code n} rows at positions {@code pos0 ..} through every layer. */
    public float[] contiguous(int pos0, int n, float[] in) {
        memory.beginStep(pos0, n);
        int[] pos = ctx.scratch().ints(CpuScratch.POS);
        for (int t = 0; t < n; t++) {
            pos[t] = pos0 + t;
        }
        float[] out = new float[mixers.size() * n * width];
        float[] layerOut = new float[n * width];
        for (int l = 0; l < mixers.size(); l++) {
            if (n == 1) {
                mixers.get(l).decode(pos0, in, layerOut);
            } else {
                mixers.get(l).prefill(pos0, n, in, layerOut);
            }
            System.arraycopy(layerOut, 0, out, l * n * width, n * width);
        }
        return out;
    }

    /**
     * One paged step: segment {@code i} holds {@code len[i]} rows of sequence {@code seq[i]} at positions
     * {@code pos0[i] ..}, the segments' rows in order in {@code in}.
     */
    public float[] paged(int[] seq, int[] len, int[] pos0, float[] in, boolean nonCausal) {
        int n = 0;
        int[] start = new int[seq.length];
        for (int i = 0; i < seq.length; i++) {
            start[i] = n;
            n += len[i];
        }
        int[] positions = new int[n];
        for (int i = 0; i < seq.length; i++) {
            for (int t = 0; t < len[i]; t++) {
                positions[start[i] + t] = pos0[i] + t;
            }
        }
        StepLayout layout = StepLayout.of(n, seq, start, len, pos0, new int[0], true);
        memory.openPagedStep(layout, positions);
        System.arraycopy(positions, 0, ctx.scratch().ints(CpuScratch.POS), 0, n);
        ctx.stepNonCausal(nonCausal);
        try {
            float[] out = new float[mixers.size() * n * width];
            float[] layerOut = new float[n * width];
            for (int l = 0; l < mixers.size(); l++) {
                mixers.get(l).batch(layout, in, layerOut);
                System.arraycopy(layerOut, 0, out, l * n * width, n * width);
            }
            return out;
        } finally {
            ctx.stepNonCausal(false);
            memory.endPagedStep();
        }
    }

    /** Rows {@code [row0, row0 + n)} of layer {@code layer} out of a step's outputs of {@code stepRows} rows. */
    public float[] rows(float[] out, int layer, int stepRows, int row0, int n) {
        float[] r = new float[n * width];
        System.arraycopy(out, (layer * stepRows + row0) * width, r, 0, n * width);
        return r;
    }

    @Override
    public void close() {
        for (CpuMixer m : mixers) {
            m.close();
        }
        memory.close();
    }
}
