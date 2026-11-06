package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.ops.heads.ControlVector;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * What a layer split loads ({@link GpuSplitSession#open}): the model file and its resolved specification, the engine
 * settings, and what the loading model already read for it: the LoRA adapters, the control vectors, the MoE ranking
 * session, host access to the model's tensors and the drafter the main device holds beside its context. Immutable.
 */
public final class SplitRequest {

    /** Host access to the model's tensors, opened on first use by the loading model, which owns it. */
    public interface HostWeights {
        CpuWeights get() throws IOException;
    }

    /**
     * The memory of the configured drafter, measured on the device that holds it (the main device): its weights as
     * that device's kernels upload them, and its caches.
     */
    public interface DrafterFit {
        ContextFitter.Companion on(ComputeBackend backend, ComputeQueue queue, LlmKernels kernels);
    }

    private final ModelSource source;
    private final TensorCatalog catalog;
    private final InputTransformLoader.Result transforms;
    private final GgufKv kv;
    private final EngineConfig cfg;
    private final ModelSpec initial;
    private final String modelName;
    private final MoeCacheRanking ranking;
    private final List<LoraAttachment.Binding> loras;
    private final ControlVector controlVectors;
    private final HostWeights hostWeights;
    private final DrafterFit drafter;

    private SplitRequest(Builder b) {
        this.source = Objects.requireNonNull(b.source, "source");
        this.catalog = Objects.requireNonNull(b.catalog, "catalog");
        this.transforms = b.transforms;
        this.kv = Objects.requireNonNull(b.kv, "kv");
        this.cfg = Objects.requireNonNull(b.cfg, "cfg");
        this.initial = Objects.requireNonNull(b.initial, "initial");
        this.modelName = b.modelName != null ? b.modelName : b.initial.arch().ggufName();
        this.ranking = b.ranking;
        this.loras = Collections.unmodifiableList(new ArrayList<>(b.loras));
        this.controlVectors = b.controlVectors;
        this.hostWeights = Objects.requireNonNull(b.hostWeights, "hostWeights");
        this.drafter = b.drafter;
    }

    /**
     * A request for the model {@code source} holds, resolved as {@code initial} under {@code cfg}.
     *
     * @param transforms the model's input transforms, or null for none
     * @param hostWeights host access to the model's tensors (host layers, hosted experts)
     */
    public static Builder builder(ModelSource source, TensorCatalog catalog, InputTransformLoader.Result transforms,
                                  GgufKv kv, EngineConfig cfg, ModelSpec initial, HostWeights hostWeights) {
        Builder b = new Builder();
        b.source = source;
        b.catalog = catalog;
        b.transforms = transforms;
        b.kv = kv;
        b.cfg = cfg;
        b.initial = initial;
        b.hostWeights = hostWeights;
        return b;
    }

    public ModelSource source() { return source; }
    public TensorCatalog catalog() { return catalog; }
    /** The model's input transforms, or null for none. */
    public InputTransformLoader.Result transforms() { return transforms; }
    public GgufKv kv() { return kv; }
    public EngineConfig cfg() { return cfg; }
    public ModelSpec initial() { return initial; }
    /** The model's name, for messages. */
    public String modelName() { return modelName; }
    /** The model's MoE ranking session (expert heat and cache preload), or null for none. */
    public MoeCacheRanking ranking() { return ranking; }
    /** The LoRA adapters loaded against the model, empty for none (unmodifiable). */
    public List<LoraAttachment.Binding> loras() { return loras; }
    /** The control vectors read against the model, or null for none. */
    public ControlVector controlVectors() { return controlVectors; }
    public HostWeights hostWeights() { return hostWeights; }

    /** The drafter's memory on the main device, {@link ContextFitter.Companion#NONE} without a drafter. */
    ContextFitter.Companion drafterOn(ComputeBackend backend, ComputeQueue queue, LlmKernels kernels) {
        return drafter == null ? ContextFitter.Companion.NONE
                : Objects.requireNonNull(drafter.on(backend, queue, kernels), "drafter companion");
    }

    /** Builder of a {@link SplitRequest}. */
    public static final class Builder {
        private ModelSource source;
        private TensorCatalog catalog;
        private InputTransformLoader.Result transforms;
        private GgufKv kv;
        private EngineConfig cfg;
        private ModelSpec initial;
        private String modelName;
        private MoeCacheRanking ranking;
        private List<LoraAttachment.Binding> loras = Collections.emptyList();
        private ControlVector controlVectors;
        private HostWeights hostWeights;
        private DrafterFit drafter;

        private Builder() {
        }

        public Builder modelName(String v) { this.modelName = v; return this; }
        public Builder ranking(MoeCacheRanking v) { this.ranking = v; return this; }
        public Builder loras(List<LoraAttachment.Binding> v) {
            this.loras = Objects.requireNonNull(v, "loras");
            return this;
        }
        public Builder controlVectors(ControlVector v) { this.controlVectors = v; return this; }
        /** The drafter the main device holds beside its context; none when unset. */
        public Builder drafter(DrafterFit v) { this.drafter = v; return this; }

        /** @throws NullPointerException when a required input is missing */
        public SplitRequest build() {
            return new SplitRequest(this);
        }
    }
}
