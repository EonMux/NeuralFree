package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.HostPlacement;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeCacheLoader;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hybrid.HybridFit;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;

import java.io.IOException;
import java.util.Objects;
import java.util.function.LongSupplier;

/**
 * Fits the parameters of one device load without loading: the context, the layer placement and the expert
 * overflow are planned exactly as {@link GGUFModelGPU} plans them at load — the weight placement rules
 * ({@link HostPlacement}), the automatic layer-split search and the streaming ladder ({@link GpuLoadPlacement},
 * {@link PlacementFitter} of {@code com.eonmux.neuralfree.gguf.gpu.weights}) — against the device budget the
 * load would see. The budget mirrors the load's prologue: the free VRAM measured before anything is planned,
 * less the compute buffers the load allocates before the placement (the model's scratch slots at the configured
 * {@code -ub} and the kernel placeholders), capped by {@code neuralfree.gpu.vram.limit}, less a fit margin.
 *
 * <p>The result is a {@link FitParams}: the fitted context, the placement mode and the arguments that reproduce
 * the load ({@code -c}, {@code -ngl}, {@code --cpu-moe} or {@code -ot}, {@code -nkvo}). Everything the load
 * refuses, the planner refuses the same way: a draft model, a hybrid split the model cannot serve, weights that
 * do not fit even streamed ({@code GpuOutOfMemoryException}), a context that does not fit either
 * ({@code ContextFitter.ContextBudgetException}). Nothing is allocated; the model file is only mapped, and what
 * the planner opens itself (a weight store, an uploader) it closes before returning.</p>
 */
public final class FitParamsPlanner {

    private final GGUFFileParser parser;
    private final ComputeBackend backend;
    private final EngineConfig cfg;
    private final GpuWeightUploader uploader;
    private final LlmKernels kernels;
    private final LongSupplier usableVram;
    private final ContextFitter.Companion drafter;
    private final long marginBytes;
    private final int contextFloor;

    private FitParamsPlanner(Builder b) {
        this.parser = b.parser;
        this.backend = b.backend;
        this.cfg = b.cfg;
        this.uploader = b.uploader;
        this.kernels = b.kernels;
        this.usableVram = b.usableVram;
        this.drafter = b.drafter;
        this.marginBytes = b.marginBytes;
        this.contextFloor = b.contextFloor;
    }

    /** Runs the fit over the builder's model, backend and configuration. */
    public FitParams plan() throws IOException {
        ModelSource source = parser;
        TensorCatalog catalog = TensorCatalog.of(source);
        GgufKv kv = GgufKv.of(source);
        ModelSpec initial = ModelSpecBuilder.build(kv, catalog, cfg.memory(), 0, cfg.pooling());
        if (initial.supportClass() == SupportClass.DRAFT) {
            throw new UnsupportedOperationException(initial.arch().ggufName()
                    + " is a draft model; fit the parameters of its target model, which loads the drafter beside"
                    + " its own context");
        }
        InputTransformLoader.Result transforms = InputTransformLoader.load(source, kv, catalog,
                cfg.activationQuant());
        MemoryConfig mem = cfg.memory().forModel(initial);
        // the free VRAM of the device, measured before anything plans against it, as the load measures it
        LongSupplier budget = usableVram != null ? usableVram : () -> usableBytes(initial, mem);
        boolean drafterPlanned = drafter == null && cfg.draftModelPath() != null && !cfg.mtp();
        LlmKernels policy = kernels;
        if (policy == null && (uploader == null || drafterPlanned)) {
            policy = LlmKernels.open(backend);
        }
        GpuWeightUploader upload = uploader != null ? uploader
                : ownedUploader(source, transforms, initial, policy);
        try {
            return fit(source, catalog, transforms, initial, mem, upload, policy, budget);
        } finally {
            if (upload != uploader) {
                upload.close();
            }
        }
    }

    /** The device bytes the load would place the weights into: the free VRAM less the load's reserves, the
     * compute buffers it allocates before the placement and the fit margin, capped by the VRAM limit. */
    private long usableBytes(ModelSpec initial, MemoryConfig mem) {
        long computeBytes = computeBytes(initial, mem);
        long budget = VRAMBudget.memoryBudgetBytes(backend.capabilities().getTotalVramBytes(),
                VRAMBudget.queryFreeVram(backend), 0L, computeBytes, 0L);
        long usable = VRAMBudget.limitedBytes(budget, cfg.gpuVramLimitBytes(), computeBytes, 0L);
        return Math.max(0L, usable - marginBytes);
    }

    /** The uploader the planner owns: the model's tensors planned with the kernels' type policy. */
    private GpuWeightUploader ownedUploader(ModelSource source, InputTransformLoader.Result transforms,
                                            ModelSpec initial, LlmKernels policy) throws IOException {
        WeightStore store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        try {
            ComputeQueue queue = backend.computeQueue();
            return new GpuWeightUploader(backend, queue, source.getTensors(), store, initial.arch().ggufName(),
                    GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(policy))
                            .transforms(transforms)
                            .w12(cfg.speculativeOptions().w12())
                            .w12Head(cfg.speculativeOptions().w12Head())
                            .w12F16(cfg.speculativeOptions().w12F16())
                            .build());
        } catch (RuntimeException e) {
            store.close();
            throw e;
        }
    }

    /**
     * The placement of the load, exactly as {@link GGUFModelGPU} resolves it at load: the placement rules of the
     * configuration, the requested layer split, the MoE expert cache, then the placement ladder — and its
     * translation into the fitted parameters.
     */
    private FitParams fit(ModelSource source, TensorCatalog catalog, InputTransformLoader.Result transforms,
                          ModelSpec initial, MemoryConfig mem, GpuWeightUploader upload, LlmKernels policy,
                          LongSupplier budget) throws IOException {
        ContextFitter.Companion companion = (drafter != null ? drafter
                : LLMModelFactory.draftCompanion(cfg, initial, false,
                        (draft, placed) -> GpuWeightUploader.plannedBytesOf(backend, backend.computeQueue(),
                                policy, draft, placed)))
                .plus(LLMModelFactory.twoLevelCompanion(cfg, source, cfg.speculativeOptions().w12()));
        LayerOffload requested = LayerOffload.requested(initial, cfg, source.getTensors());
        boolean loadNextn = GGUFModelGPU.loadsNextn(cfg, mem);
        WeightPlacement weights = HostPlacement.placement(cfg, initial, upload, budget, loadNextn,
                requested != null ? requested::hostTensor : n -> false);
        boolean moeCacheConfigured = HostPlacement.moeCacheRequested(cfg)
                && !MoeCacheLoader.routedMoeLayers(initial).isEmpty();
        GpuLoadPlacement.Placement placement = new GpuLoadPlacement(source, catalog, transforms, upload, backend,
                companion, budget, moeCacheConfigured, contextFloor)
                .place(initial, mem, cfg, weights, requested);
        long totalWeightBytes = upload.plannedBytes(t -> true);
        long deviceWeightBytes = upload.plannedBytes(placement.weights::onDevice);
        MemoryPlan plan = MemoryPlanner.planForContext(initial, mem, placement.nCtx);
        long contextBytes = plan.totalBytes();
        long deviceContextBytes = placement.split != null
                ? HybridFit.contextBytes(plan, HybridFit.deviceLayers(placement.split))
                : contextBytes;
        return FitParams.of(placement, budget.getAsLong(), contextBytes, deviceContextBytes,
                computeBytes(initial, mem), totalWeightBytes, deviceWeightBytes, initial.nLayer());
    }

    /** The compute buffers the load allocates before it places the weights: the model's scratch slots at the
     * configured {@code -ub} and the kernel placeholders. */
    private static long computeBytes(ModelSpec initial, MemoryConfig mem) {
        long scratchBytes = 0L;
        for (long floats : GpuScratch.modelSlotFloats(initial, mem.ubatch()).values()) {
            scratchBytes = Math.addExact(scratchBytes, Math.multiplyExact(floats, 4L));
        }
        return Math.addExact(GpuPlaceholders.TOTAL_BYTES, scratchBytes);
    }

    /** Builds one planner. */
    public static final class Builder {

        private final GGUFFileParser parser;
        private final ComputeBackend backend;
        private final EngineConfig cfg;
        private GpuWeightUploader uploader;
        private LlmKernels kernels;
        private LongSupplier usableVram;
        private ContextFitter.Companion drafter;
        private long marginBytes;
        private int contextFloor;

        /**
         * @param parser  the opened model file, whose tensors the fit plans over
         * @param backend the device the fit plans for, as the load would open it
         * @param cfg     the engine configuration the load would run with; the placement flags
         *                ({@code -ngl}, {@code -ot}, {@code --cpu-moe}, {@code --n-cpu-moe},
         *                {@code --n-cpu-ffn}, {@code --fit}) are honored as the load honors them, so a fit of
         *                parameters that a caller fixes is a fit of what is left
         */
        public Builder(GGUFFileParser parser, ComputeBackend backend, EngineConfig cfg) {
            this.parser = Objects.requireNonNull(parser, "parser");
            this.backend = Objects.requireNonNull(backend, "backend");
            this.cfg = Objects.requireNonNull(cfg, "cfg");
        }

        /** Plans the weights with the caller's uploader, already configured; not closed by the planner. */
        public Builder uploader(GpuWeightUploader v) {
            this.uploader = Objects.requireNonNull(v, "uploader");
            return this;
        }

        /** Plans with the caller's kernels, whose type policy sizes the device tensors; not closed. */
        public Builder kernels(LlmKernels v) {
            this.kernels = Objects.requireNonNull(v, "kernels");
            return this;
        }

        /** Plans against the caller's device budget instead of the free VRAM of {@code backend}. */
        public Builder usableVram(LongSupplier v) {
            this.usableVram = Objects.requireNonNull(v, "usableVram");
            return this;
        }

        /** Holds the caller's companion memory beside the context instead of the configured drafter's. */
        public Builder drafter(ContextFitter.Companion v) {
            this.drafter = Objects.requireNonNull(v, "drafter");
            return this;
        }

        /** Free device bytes the fit leaves on the device after the load (llama.cpp's {@code --fit-target}
         * margin); 0 by default, the load's own reserves are always held back. */
        public Builder marginBytes(long v) {
            if (v < 0) {
                throw new IllegalArgumentException("marginBytes must be >= 0, got " + v);
            }
            this.marginBytes = v;
            return this;
        }

        /** The smallest automatic context the placement may run with (llama.cpp's {@code --fit-ctx}); 0 for the
         * model's own floor. A higher floor moves weights off the device instead of shrinking the context. */
        public Builder contextFloor(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("contextFloor must be >= 0, got " + v);
            }
            this.contextFloor = v;
            return this;
        }

        /** Runs the fit. */
        public FitParams plan() throws IOException {
            return new FitParamsPlanner(this).plan();
        }
    }
}
