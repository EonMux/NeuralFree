package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSource;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertPrefetch;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.moe.EplbLoadWindow;
import com.eonmux.neuralfree.gguf.ops.moe.MoeExpertHeat;
import com.eonmux.neuralfree.gguf.ops.moe.MoeExpertRemap;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * Everything a GPU block needs to enqueue its work: the backend and queue, the kernel library, the dispatch helpers,
 * scratch, weights, recurrent/KV memory, the model spec, the placeholder buffers bound to absent optional kernel
 * arguments, the hidden-state tap, the projection deltas (LoRA) of the model's layers, the MoE expert cache of the
 * host-resident routed experts and its prefill expert prefetch pool, the MoE expert-heat recorder ({@code null} without a profile save) and, when a
 * placement rule keeps tensors in RAM, the host view of the model's tensors.
 *
 * <p>Immutable and shared by every block of one model. The context does not own its parts: the forward backend that
 * builds it closes them. {@link #memory()} is absent for models without memory (encoders, embedding models).
 * {@link #hiddenTap()} is {@link HiddenTap#NONE} unless one was set; a forward backend that receives a new tap
 * rebuilds its context with {@link #withHiddenTap}. {@link #projectionDeltas()} is a mutable registry created with
 * the context and shared by every context derived from it ({@link #withHiddenTap}).</p>
 */
public final class GpuForwardContext {

    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final LlmKernels kernels;
    private final GpuOps ops;
    private final GpuScratch scratch;
    private final GpuWeights weights;
    private final LayerWeightSource weightSource;
    private final GpuMemory memory;
    private final ModelSpec spec;
    private final GpuPlaceholders placeholders;
    private final ExecutionObserver observer;
    private final HiddenTap hiddenTap;
    private final GpuProjectionDeltas projectionDeltas;
    private final MoeExpertCache moeExpertCache;
    private final CpuWeights hostWeights;
    private final MoeExpertPrefetch moeExpertPrefetch;
    private final int moeTailSkipRank;
    private final MoeExpertHeat heat;
    private final EplbLoadWindow moeLoadWindow;
    private final MoeExpertRemap moeExpertRemap;
    private final IntPredicate hostKv;
    private final GpuStepSegments stepSegments;

    private GpuForwardContext(Builder b) {
        this.backend = b.kernels.backend();
        this.queue = b.ops.queue();
        this.kernels = b.kernels;
        this.ops = b.ops;
        this.scratch = b.ops.scratch();
        this.weights = b.weights;
        this.weightSource = b.weightSource != null ? b.weightSource : LayerWeightSource.resident(b.weights);
        this.memory = b.memory;
        this.spec = b.spec;
        this.placeholders = b.ops.placeholders();
        this.observer = b.observer;
        this.hiddenTap = b.hiddenTap;
        this.projectionDeltas = b.projectionDeltas != null ? b.projectionDeltas : new GpuProjectionDeltas();
        this.moeExpertCache = b.moeExpertCache;
        this.hostWeights = b.hostWeights;
        this.moeExpertPrefetch = b.moeExpertPrefetch;
        this.moeTailSkipRank = b.moeTailSkipRank;
        this.heat = b.heat;
        this.moeLoadWindow = b.moeLoadWindow;
        this.moeExpertRemap = b.moeExpertRemap;
        this.hostKv = b.hostKv;
        this.stepSegments = b.stepSegments != null ? b.stepSegments : new GpuStepSegments();
    }

    /**
     * Builder over the dispatch helpers (which carry the queue, scratch and placeholders), the model's weights and
     * its spec.
     */
    public static Builder builder(LlmKernels kernels, GpuOps ops, GpuWeights weights, ModelSpec spec) {
        return new Builder(kernels, ops, weights, spec);
    }

    public ComputeBackend backend() { return backend; }
    /** Queue every block enqueues on. */
    public ComputeQueue queue() { return queue; }
    public LlmKernels kernels() { return kernels; }
    public GpuOps ops() { return ops; }
    public GpuScratch scratch() { return scratch; }
    /** Model-level weights (and every layer's weights when they are all resident). */
    public GpuWeights weights() { return weights; }
    /** Per-layer weight access; resident over {@link #weights()} unless weights are streamed. */
    public LayerWeightSource weightSource() { return weightSource; }
    public ModelSpec spec() { return spec; }
    /**
     * Dummy buffers bound to optional kernel arguments whose {@code has_x} flag is off (a kernel never receives a
     * null buffer); the ones {@link #ops()} binds.
     */
    public GpuPlaceholders placeholders() { return placeholders; }
    public ExecutionObserver observer() { return observer; }
    /** Observer of the residual stream at layer inputs; {@link HiddenTap#NONE} when nothing taps it. */
    public HiddenTap hiddenTap() { return hiddenTap; }
    /** Additive deltas (LoRA) on the layers' projections, shared by the blocks that apply and register them. */
    public GpuProjectionDeltas projectionDeltas() { return projectionDeltas; }
    /** The MoE expert cache of the host-resident routed experts, or null when none is configured. */
    public MoeExpertCache moeExpertCache() { return moeExpertCache; }
    /**
     * Host view of the model's tensors, set when a placement rule keeps tensors in RAM ({@code --cpu-moe},
     * {@code --n-cpu-moe}, {@code -ot}, {@code neuralfree.moe.cache}); the MoE blocks read their hosted routed
     * experts from it. Null otherwise.
     */
    public CpuWeights hostWeights() { return hostWeights; }
    /** The prefill expert prefetch pool of the cache, or null when prefill prefetch is off. */
    public MoeExpertPrefetch moeExpertPrefetch() { return moeExpertPrefetch; }

    /**
     * The hosted tail-skip rank of the MoE expert cache ({@code neuralfree.moe.cache.tail.skip}): a missed expert
     * that every token of a step routes only at this 0-based top-k position or lower is not computed. 0 keeps every
     * miss. The MoE blocks apply it to trunk layers only.
     */
    public int moeTailSkipRank() { return moeTailSkipRank; }

    /** The MoE expert-heat recorder the routed blocks note their selections into; null without a profile save. */
    public MoeExpertHeat heat() { return heat; }

    /** The live routed-row window the runtime expert rearrangement reads; null without one. */
    public EplbLoadWindow moeLoadWindow() { return moeLoadWindow; }

    /** The routed experts' logical-to-physical replica mapping; null when the experts keep their logical ids. */
    public MoeExpertRemap moeExpertRemap() { return moeExpertRemap; }

    /**
     * True for a trunk layer whose KV cache lives in host RAM and whose attention runs on the host over it
     * ({@code neuralfree.kv.offload=false}, llama.cpp {@code -nkvo}); such a layer's device mixer binds without a
     * device cache. False for every layer by default.
     */
    public boolean hostKv(int il) { return hostKv.test(il); }

    /** True when the model has KV or recurrent memory. */
    public boolean hasMemory() {
        return memory != null;
    }

    /**
     * The model's device memory.
     *
     * @throws IllegalStateException when the model has none
     */
    public GpuMemory memory() {
        if (memory == null) {
            throw new IllegalStateException("model " + spec.arch().ggufName() + " has no KV or recurrent memory");
        }
        return memory;
    }

    /**
     * The sequence runs of the running packed memoryless step ({@link GpuStepSegments}), shared by every context
     * derived from this one; the forward driver closes it.
     */
    public GpuStepSegments stepSegments() { return stepSegments; }

    /** This context with another hidden-state tap ({@link HiddenTap#NONE} to stop tapping). */
    public GpuForwardContext withHiddenTap(HiddenTap tap) {
        Builder b = toBuilder();
        b.hiddenTap = Objects.requireNonNull(tap, "hiddenTap");
        return new GpuForwardContext(b);
    }

    private Builder toBuilder() {
        Builder b = new Builder(kernels, ops, weights, spec);
        b.weightSource = weightSource;
        b.memory = memory;
        b.observer = observer;
        b.hiddenTap = hiddenTap;
        b.projectionDeltas = projectionDeltas;
        b.moeExpertCache = moeExpertCache;
        b.hostWeights = hostWeights;
        b.moeExpertPrefetch = moeExpertPrefetch;
        b.moeTailSkipRank = moeTailSkipRank;
        b.heat = heat;
        b.moeLoadWindow = moeLoadWindow;
        b.moeExpertRemap = moeExpertRemap;
        b.hostKv = hostKv;
        b.stepSegments = stepSegments;
        return b;
    }

    @Override
    public String toString() {
        return "GpuForwardContext{" + spec.arch().ggufName() + ", " + backend.backendType()
                + (memory != null ? ", memory" : "") + '}';
    }

    /** Builder of a {@link GpuForwardContext}. */
    public static final class Builder {
        private final LlmKernels kernels;
        private final GpuOps ops;
        private final GpuWeights weights;
        private final ModelSpec spec;
        private LayerWeightSource weightSource;
        private GpuMemory memory;
        private ExecutionObserver observer = ExecutionObserver.NONE;
        private HiddenTap hiddenTap = HiddenTap.NONE;
        private GpuProjectionDeltas projectionDeltas;
        private MoeExpertCache moeExpertCache;
        private CpuWeights hostWeights;
        private MoeExpertPrefetch moeExpertPrefetch;
        private int moeTailSkipRank;
        private MoeExpertHeat heat;
        private EplbLoadWindow moeLoadWindow;
        private MoeExpertRemap moeExpertRemap;
        private IntPredicate hostKv = il -> false;
        private GpuStepSegments stepSegments;

        private Builder(LlmKernels kernels, GpuOps ops, GpuWeights weights, ModelSpec spec) {
            this.kernels = Objects.requireNonNull(kernels, "kernels");
            this.ops = Objects.requireNonNull(ops, "ops");
            this.weights = Objects.requireNonNull(weights, "weights");
            this.spec = Objects.requireNonNull(spec, "spec");
        }

        /** Streams per-layer weights through this source instead of the resident weights. */
        public Builder weightSource(LayerWeightSource v) {
            this.weightSource = Objects.requireNonNull(v, "weightSource");
            return this;
        }

        /**
         * Host view of the model's tensors for the MoE blocks that compute hosted routed experts on the CPU; not
         * owned (the loader closes it). Null leaves the blocks without the hosted path.
         */
        public Builder hostWeights(CpuWeights v) {
            this.hostWeights = v;
            return this;
        }

        /** The model's KV or recurrent memory; leave unset for models without memory. */
        public Builder memory(GpuMemory v) {
            this.memory = Objects.requireNonNull(v, "memory");
            return this;
        }

        public Builder observer(ExecutionObserver v) {
            this.observer = Objects.requireNonNull(v, "observer");
            return this;
        }

        /** Observer of the residual stream at layer inputs; defaults to {@link HiddenTap#NONE}. */
        public Builder hiddenTap(HiddenTap v) {
            this.hiddenTap = Objects.requireNonNull(v, "hiddenTap");
            return this;
        }

        /** The MoE expert cache of the host-resident routed experts; leave unset when there is none. */
        public Builder moeExpertCache(MoeExpertCache v) {
            this.moeExpertCache = Objects.requireNonNull(v, "moeExpertCache");
            return this;
        }

        /** The prefill expert prefetch pool of the cache; null keeps prefill prefetch off. */
        public Builder moeExpertPrefetch(MoeExpertPrefetch v) {
            this.moeExpertPrefetch = v;
            return this;
        }

        /** The hosted tail-skip rank of the expert cache; 0 (the default) keeps every miss. */
        public Builder moeTailSkipRank(int v) {
            this.moeTailSkipRank = v;
            return this;
        }

        /**
         * The trunk layers whose KV cache lives in host RAM and whose attention runs on the host; none by default.
         */
        public Builder hostKv(IntPredicate v) {
            this.hostKv = Objects.requireNonNull(v, "hostKv");
            return this;
        }

        /** The routed experts' logical-to-physical replica mapping; null (the default) keeps logical ids. */
        public Builder moeExpertRemap(MoeExpertRemap v) {
            this.moeExpertRemap = v;
            return this;
        }

        /** The MoE expert-heat recorder the routed blocks note their selections into; null (the default) skips it. */
        public Builder heat(MoeExpertHeat v) {
            this.heat = v;
            return this;
        }

        /** The live routed-row window of the runtime expert rearrangement; null (the default) skips it. */
        public Builder moeLoadWindow(EplbLoadWindow v) {
            this.moeLoadWindow = v;
            return this;
        }

        /**
         * @throws IllegalArgumentException when the dispatch helpers were made for another kernel library
         */
        public GpuForwardContext build() {
            if (ops.kernels() != kernels) {
                throw new IllegalArgumentException("GpuOps uses another kernel library than the context ("
                        + ops.kernels().backend().backendType() + " vs " + kernels.backend().backendType() + ")");
            }
            return new GpuForwardContext(this);
        }
    }
}
