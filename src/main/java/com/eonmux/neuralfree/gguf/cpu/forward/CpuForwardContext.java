package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.BatchSegments;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.ops.moe.MoeExpertHeat;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * Everything a CPU block needs to run: the model spec, its in-place weights, the matvec bound to the weights'
 * transforms, the preallocated scratch, the KV / recurrent memory, the hidden-state tap, the projection deltas
 * (LoRA) of the model's layers and the MoE expert-heat recorder ({@code null} without a profile save). Row-wise and
 * element-wise math is
 * {@code CpuOps} (static) and threading is {@code CpuParallel}, so neither is held here.
 *
 * <p>Immutable and shared by every block of one model; the twin of {@code GpuForwardContext}. The context does not
 * own its parts: the forward backend that builds it closes them. {@link #memory()} is absent for models without
 * memory (encoders, embedding models). {@link #hiddenTap()} is {@link HiddenTap#NONE} unless one was set; a forward
 * backend that receives a new tap rebuilds its context with {@link #withHiddenTap}. {@link #projectionDeltas()} is a
 * mutable registry created with the context and shared by every context derived from it ({@link #withHiddenTap}).</p>
 */
public final class CpuForwardContext {

    private final ModelSpec spec;
    private final CpuWeights weights;
    private final CpuMatVec matVec;
    private final CpuScratch scratch;
    private final CpuMemory memory;
    private final HiddenTap hiddenTap;
    private final CpuProjectionDeltas projectionDeltas;
    private final CpuForwardStepState stepState;
    private final MoeExpertHeat heat;

    private CpuForwardContext(Builder b) {
        this.spec = b.spec;
        this.weights = b.weights;
        this.matVec = b.matVec != null ? b.matVec : b.weights.matVec();
        this.scratch = b.scratch;
        this.memory = b.memory;
        this.hiddenTap = b.hiddenTap;
        this.projectionDeltas = b.projectionDeltas != null ? b.projectionDeltas : new CpuProjectionDeltas();
        this.stepState = b.stepState != null ? b.stepState : new CpuForwardStepState();
        this.heat = b.heat;
    }

    /** Builder over the model spec, its weights and the scratch sized for it. */
    public static Builder builder(ModelSpec spec, CpuWeights weights, CpuScratch scratch) {
        return new Builder(spec, weights, scratch);
    }

    public ModelSpec spec() {
        return spec;
    }

    /** The model's tensors, read in place. */
    public CpuWeights weights() {
        return weights;
    }

    /** Matvec applying the weights' PRISM input transforms and output scales. */
    public CpuMatVec matVec() {
        return matVec;
    }

    public CpuScratch scratch() {
        return scratch;
    }

    /** True when the model has KV or recurrent memory. */
    public boolean hasMemory() {
        return memory != null;
    }

    /**
     * The model's host memory.
     *
     * @throws IllegalStateException when the model has none
     */
    public CpuMemory memory() {
        if (memory == null) {
            throw new IllegalStateException("model " + spec.arch().ggufName() + " has no KV or recurrent memory");
        }
        return memory;
    }

    /** Observer of the residual stream at layer inputs; {@link HiddenTap#NONE} when nothing taps it. */
    public HiddenTap hiddenTap() {
        return hiddenTap;
    }

    /** Additive deltas (LoRA) on the layers' projections, shared by the blocks that apply and register them. */
    public CpuProjectionDeltas projectionDeltas() {
        return projectionDeltas;
    }

    /**
     * The per-step flags of the running forward pass: the backend sets them before it runs the slots and clears
     * them after. Shared by every context derived from this one.
     */
    CpuForwardStepState stepState() {
        return stepState;
    }

    /** Whether the running step attends without the causal mask (a media chunk's rows). */
    public boolean stepNonCausal() {
        return stepState.nonCausal();
    }

    /** Flags the running step non-causal or causal; the backend sets it around the step. */
    public void stepNonCausal(boolean nonCausal) {
        stepState.nonCausal(nonCausal);
    }

    /** The running step's tree mask ({@code fresh} fresh rows), or null when the step carries none. */
    public long[] stepTreeMask() {
        return stepState.treeMask();
    }

    /** The running step's fresh tree rows, or -1 when the step carries no tree mask. */
    public int stepTreeFresh() {
        return stepState.treeFresh();
    }

    /** Sets the running step's tree mask; the backend sets it around a tree verify's step. */
    public void stepTreeMask(long[] mask, int fresh) {
        stepState.treeMask(mask, fresh);
    }

    /** The running step's recurrent tree round, or null when the step runs no recurrent tree step. */
    public CpuRecurrentTreeRound stepTreeRound() {
        return stepState.treeRound();
    }

    /** Sets the running step's recurrent tree round; the backend sets it around a recurrent target's tree step. */
    public void stepTreeRound(CpuRecurrentTreeRound round) {
        stepState.treeRound(round);
    }

    /** The running step's KV store positions, or null when the rows store at their POS positions. */
    public int[] stepStorePositions() {
        return stepState.storePositions();
    }

    /** Sets the running step's KV store positions; the backend sets it around a contiguous tree step. */
    public void stepStorePositions(int[] positions) {
        stepState.storePositions(positions);
    }

    /**
     * The running memoryless step's sequence runs, each attending only its own rows, or null when the step's rows
     * form one sequence.
     */
    public BatchSegments stepSegments() {
        return stepState.segments();
    }

    /** Sets the running memoryless step's sequence runs; the backend sets it around a packed step. */
    public void stepSegments(BatchSegments segments) {
        stepState.segments(segments);
    }

    /** The MoE expert-heat recorder the routed blocks note their selections into; null without a profile save. */
    public MoeExpertHeat heat() {
        return heat;
    }

    /** This context with another hidden-state tap ({@link HiddenTap#NONE} to stop tapping). */
    public CpuForwardContext withHiddenTap(HiddenTap tap) {
        Builder b = new Builder(spec, weights, scratch);
        b.matVec = matVec;
        b.memory = memory;
        b.hiddenTap = Objects.requireNonNull(tap, "hiddenTap");
        b.projectionDeltas = projectionDeltas;
        b.stepState = stepState;
        b.heat = heat;
        return new CpuForwardContext(b);
    }

    @Override
    public String toString() {
        return "CpuForwardContext{" + spec.arch().ggufName() + ", ubatch=" + scratch.ubatch()
                + (memory != null ? ", memory" : "") + '}';
    }

    /** Builder of a {@link CpuForwardContext}. */
    public static final class Builder {
        private final ModelSpec spec;
        private final CpuWeights weights;
        private final CpuScratch scratch;
        private CpuMatVec matVec;
        private CpuMemory memory;
        private HiddenTap hiddenTap = HiddenTap.NONE;
        private CpuProjectionDeltas projectionDeltas;
        private CpuForwardStepState stepState;
        private MoeExpertHeat heat;

        private Builder(ModelSpec spec, CpuWeights weights, CpuScratch scratch) {
            this.spec = Objects.requireNonNull(spec, "spec");
            this.weights = Objects.requireNonNull(weights, "weights");
            this.scratch = Objects.requireNonNull(scratch, "scratch");
        }

        /** The model's KV or recurrent memory; leave unset for models without memory. */
        public Builder memory(CpuMemory v) {
            this.memory = Objects.requireNonNull(v, "memory");
            return this;
        }

        /** A matvec other than the weights' own (for example one with diagnostics); defaults to it. */
        public Builder matVec(CpuMatVec v) {
            this.matVec = Objects.requireNonNull(v, "matVec");
            return this;
        }

        /** Observer of the residual stream at layer inputs; defaults to {@link HiddenTap#NONE}. */
        public Builder hiddenTap(HiddenTap v) {
            this.hiddenTap = Objects.requireNonNull(v, "hiddenTap");
            return this;
        }

        /** The MoE expert-heat recorder the routed blocks note their selections into; null (the default) skips it. */
        public Builder heat(MoeExpertHeat v) {
            this.heat = v;
            return this;
        }

        public CpuForwardContext build() {
            return new CpuForwardContext(this);
        }
    }
}
