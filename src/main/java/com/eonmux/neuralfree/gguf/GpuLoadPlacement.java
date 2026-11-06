package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet;
import com.eonmux.neuralfree.gguf.gpu.weights.OffloadParams;
import com.eonmux.neuralfree.gguf.gpu.weights.PlacementFitter;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hybrid.HybridFit;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionBody;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.LongSupplier;
import java.util.function.Predicate;

/**
 * The context and weight placement of one {@link GGUFModelGPU} load, measured against the VRAM left after the
 * kernels, placeholders and scratch are allocated: a resolved hybrid split is honored as asked; otherwise the weights
 * placed on the device stay resident when the largest context that then fits the remaining VRAM is at least
 * {@value #AUTO_CONTEXT_FLOOR} tokens (or the training context when shorter); an automatic {@code -ngl} request then
 * offloads the largest suffix of trunk layers whose weights and context fit the VRAM; otherwise
 * {@link PlacementFitter} fits across the device and the host: the expert tensors of the layers that do not fit
 * stream first, then whole layers. A host-KV load ({@link LayerOffload#hostKvRequested}) resolves its split through
 * the same fit: an automatic {@code -ngl} tries a full offload first, since the KV caches no longer occupy VRAM.
 * Every context fit also holds the NextN draft caches and the configured drafter's
 * memory beside the context; the hybrid fit charges only the offloaded layers' plan bytes
 * ({@link HybridFit#contextSizer}). A load whose MoE expert cache hosts the routed experts
 * ({@link MoeCacheConfig#PROP_MOE_CACHE}) neither offloads trunk layers automatically nor streams the trunk: when the
 * weights left on the device do not fit resident, it is refused by name. A {@code neuralfree.stream.params} selection
 * suppresses the automatic suffix offload, so the fit streams the selected tensors instead of splitting whole layers.
 * Plans only; nothing is allocated.
 */
final class GpuLoadPlacement {

    private static final Logger LOGGER = LogManager.getLogger(GpuLoadPlacement.class);

    /** Smallest automatic context that keeps every weight resident before the trunk is streamed. */
    static final int AUTO_CONTEXT_FLOOR = PlacementFitter.AUTO_CONTEXT_FLOOR;
    private static final double MIB = 1024.0 * 1024.0;

    private final ModelSource source;
    private final TensorCatalog catalog;
    private final InputTransformLoader.Result transforms;
    private final GpuWeightUploader uploader;
    private final ComputeBackend backend;
    private final ContextFitter.Companion drafter;
    private final LongSupplier usableVram;
    private final boolean moeCache;
    private final int contextFloor;
    /** Tensors the load keeps in host-mapped memory ({@code neuralfree.cpu.offload.mb}); never on device. */
    private final Predicate<String> offloaded;

    /**
     * @param uploader   the weight uploader, whose plans size the device weights and the streamed sets
     * @param drafter    the memory a configured drafter reserves beside the context
     * @param usableVram the device bytes left for the model ({@code GGUFModelGPU.usableVram})
     * @param moeCache   true when the MoE expert cache is configured and the model has routed MoE layers, so the
     *                   routed experts stay in RAM instead of offloading or streaming trunk layers
     */
    GpuLoadPlacement(ModelSource source, TensorCatalog catalog, InputTransformLoader.Result transforms,
                     GpuWeightUploader uploader, ComputeBackend backend, ContextFitter.Companion drafter,
                     LongSupplier usableVram, boolean moeCache) {
        this(source, catalog, transforms, uploader, backend, drafter, usableVram, moeCache, 0, name -> false);
    }

    /**
     * @param contextFloor the smallest automatic context a placement runs with, 0 for the model's own
     *                     ({@link PlacementFitter#contextFloor}); a load that must not shrink its context below a
     *                     configured minimum passes it, so the fit moves weights off the device instead
     */
    GpuLoadPlacement(ModelSource source, TensorCatalog catalog, InputTransformLoader.Result transforms,
                     GpuWeightUploader uploader, ComputeBackend backend, ContextFitter.Companion drafter,
                     LongSupplier usableVram, boolean moeCache, int contextFloor) {
        this(source, catalog, transforms, uploader, backend, drafter, usableVram, moeCache, contextFloor,
                name -> false);
    }

    /**
     * @param offloaded the stored names of the tensors kept in host-mapped memory the device reads in place;
     *                  they count against neither the device weights nor the streamed sets
     */
    GpuLoadPlacement(ModelSource source, TensorCatalog catalog, InputTransformLoader.Result transforms,
                     GpuWeightUploader uploader, ComputeBackend backend, ContextFitter.Companion drafter,
                     LongSupplier usableVram, boolean moeCache, Predicate<String> offloaded) {
        this(source, catalog, transforms, uploader, backend, drafter, usableVram, moeCache, 0, offloaded);
    }

    /**
     * @param contextFloor the smallest automatic context a placement runs with, 0 for the model's own
     *                     ({@link PlacementFitter#contextFloor})
     * @param offloaded    the stored names of the tensors kept in host-mapped memory the device reads in place;
     *                     they count against neither the device weights nor the streamed sets
     */
    GpuLoadPlacement(ModelSource source, TensorCatalog catalog, InputTransformLoader.Result transforms,
                     GpuWeightUploader uploader, ComputeBackend backend, ContextFitter.Companion drafter,
                     LongSupplier usableVram, boolean moeCache, int contextFloor, Predicate<String> offloaded) {
        this.source = Objects.requireNonNull(source, "source");
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.transforms = transforms;
        this.uploader = Objects.requireNonNull(uploader, "uploader");
        this.backend = Objects.requireNonNull(backend, "backend");
        this.drafter = Objects.requireNonNull(drafter, "drafter");
        this.usableVram = Objects.requireNonNull(usableVram, "usableVram");
        this.moeCache = moeCache;
        this.contextFloor = contextFloor;
        this.offloaded = Objects.requireNonNull(offloaded, "offloaded");
    }

    /** The placement's context floor: the override when positive, else the model's own. */
    private static int floorOf(ModelSpec spec, int override) {
        if (override <= 0) {
            return PlacementFitter.contextFloor(spec);
        }
        int train = spec.hparams().nCtxTrain();
        return train > 0 ? Math.min(override, train) : override;
    }

    /**
     * The context and placement of the load.
     *
     * @param weights   the tensors the load places on the device before any split
     * @param requested the hybrid split the configuration resolved ({@link LayerOffload#requested}), or null
     * @throws UnsupportedOperationException when a layer split is requested beside {@code neuralfree.cpu.offload.mb}
     *                                       or {@code neuralfree.stream.params}, or the MoE expert cache is configured
     *                                       and the weights left on the device do not fit resident
     * @throws IllegalArgumentException      when a stream parameter segment matches no tensor the load places on the
     *                                       device
     */
    Placement place(ModelSpec initial, MemoryConfig mem, EngineConfig cfg, WeightPlacement weights,
                    LayerOffload requested) {
        validateStreamParams(initial, cfg, weights);
        if (requested != null) {
            refuseOffloadWithSplit(initial, cfg, requested);
            if (requested.hostKv() && cfg.gpuLayers() == EngineConfig.GPU_LAYERS_AUTO) {
                Placement fitted = autoHybrid(initial, mem, cfg, weights);
                if (fitted != null) {
                    return fitted;
                }
            }
            return placeHybrid(initial, mem, weights, requested, true);
        }
        long weightBytes = uploader.plannedBytes(t -> weights.onDevice(t) && !offloaded.test(t.name()));
        long usable = usableVram.getAsLong();
        int floor = floorOf(initial, contextFloor);
        boolean memoryless = initial.memoryKind() == MemoryKind.NONE;
        int residentCtx;
        if (memoryless) {
            residentCtx = mem.memorylessContext(initial);
            if (weightBytes <= usable) {
                return Placement.resident(residentCtx, weights);
            }
        } else {
            residentCtx = fitOrZero(initial, mem, usable - weightBytes, drafter);
            if (residentCtx > 0 && (!mem.autoContext() || residentCtx >= floor)) {
                return Placement.resident(residentCtx, weights);
            }
            if (cfg.gpuLayers() == EngineConfig.GPU_LAYERS_AUTO && !moeCache && cfg.cpuOffloadMb() == 0
                    && cfg.streamParams().isEmpty()) {
                Placement hybrid = autoHybrid(initial, mem, cfg, weights);
                if (hybrid != null) {
                    return hybrid;
                }
            }
        }
        return placeStreamed(initial, mem, cfg, weights, weightBytes, usable, floor, residentCtx);
    }

    /**
     * The placement that fits across the device and the host ({@link PlacementFitter}), or the resident placement
     * when the model has no trunk set to stream and its weights fit.
     */
    private Placement placeStreamed(ModelSpec initial, MemoryConfig mem, EngineConfig cfg, WeightPlacement weights,
                                    long weightBytes, long usable, int floor, int residentCtx) {
        boolean memoryless = initial.memoryKind() == MemoryKind.NONE;
        ModelSpec run = DecisionBody.forwardSpec(initial);
        List<LayerWeightSet> all = MemoryConfig.singleStepInput(initial) ? Collections.<LayerWeightSet>emptyList()
                : trunkSets(run, weights, cfg);
        if (moeCache && !all.isEmpty()) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + ": the weights leave too little"
                    + " VRAM for a resident load, and " + MoeCacheConfig.PROP_MOE_CACHE + " keeps the routed experts"
                    + " in host RAM instead of streaming them; lower the context, or clear"
                    + " " + MoeCacheConfig.PROP_MOE_CACHE + " to stream the trunk layers");
        }
        if (all.isEmpty()) {
            if (!memoryless && residentCtx > 0) {
                return Placement.resident(residentCtx, weights);
            }
            if (!memoryless && weightBytes <= usable) {
                return Placement.resident(ContextFitter.fit(initial, mem, usable - weightBytes, drafter), weights);
            }
            throw outOfMemory(initial, weightBytes, usable);
        }
        PlacementFitter.Fit fit = new PlacementFitter.Builder(initial, run, catalog, uploadOptions(cfg),
                ref -> weights.onDevice(ref.name()) && !offloaded.test(ref.name()))
                .usableBytes(usable)
                .weightBytes(weightBytes)
                .memory(mem)
                .companion(drafter)
                .contextFloor(floor)
                .backendType(backend.backendType().name())
                .build();
        return new Placement(fit.nCtx(), fit.streamedSets(), null, weights, fit);
    }

    /**
     * The largest suffix of trunk layers whose device weights and device-side context fit the usable VRAM, tried
     * from one below a full offload down to a single one; with the KV caches on the host ({@code -nkvo}) a full
     * offload is a candidate too, since the resident path would keep them on the device. The first that fits a
     * context wins. Null when none does, so the caller falls back to streaming. A candidate the model cannot serve
     * (a schedule or memory kind that cannot split) is skipped, as llama.cpp's fit skips layers it cannot offload.
     */
    private Placement autoHybrid(ModelSpec initial, MemoryConfig mem, EngineConfig cfg, WeightPlacement weights) {
        // a full offload is the resident path normally; with the KV on the host it is itself a candidate split
        int start = LayerOffload.hostKvRequested(initial, cfg) ? initial.nLayer() : initial.nLayer() - 1;
        for (int g = start; g >= 1; g--) {
            LayerOffload candidate;
            Placement placed;
            try {
                candidate = LayerOffload.auto(initial, cfg, source.getTensors(), g);
                placed = candidate == null ? null : placeHybrid(initial, mem, weights, candidate, false);
            } catch (UnsupportedOperationException | IllegalArgumentException e) {
                continue;
            }
            if (placed != null) {
                LOGGER.info("{}: offloading the last {} of {} trunk layers to {} ({})",
                        initial.arch().ggufName(), candidate.nGpuLayers(), initial.nLayer(), backend.backendType(),
                        candidate);
                return placed;
            }
        }
        return null;
    }

    /** Fits one candidate split ({@link HybridFit#placeHybrid}); on success the placement becomes the split's. */
    private Placement placeHybrid(ModelSpec initial, MemoryConfig mem, WeightPlacement weights,
                                  LayerOffload candidate, boolean required) {
        HybridFit.Fit fit = HybridFit.placeHybrid(initial, mem, candidate, weights, uploader,
                usableVram.getAsLong(), drafter, required);
        if (fit != null) {
            return new Placement(fit.nCtx, Collections.<LayerWeightSet>emptyList(), fit.split, fit.placement, null);
        }
        return null;
    }

    private static int fitOrZero(ModelSpec spec, MemoryConfig mem, long budget, ContextFitter.Companion companion) {
        if (budget < 0) {
            return 0;
        }
        try {
            return ContextFitter.fit(spec, mem, budget, companion);
        } catch (ContextFitter.ContextBudgetException e) {
            return 0;
        }
    }

    private GpuOutOfMemoryException outOfMemory(ModelSpec s, long weightBytes, long usable) {
        return new GpuOutOfMemoryException(s.arch().ggufName() + ": " + mib(weightBytes) + " MiB of weights do not fit"
                + " the " + mib(usable) + " MiB of usable VRAM" + (MemoryConfig.singleStepInput(s)
                ? " and this model keeps every weight resident" : ", even with the trunk layers streamed"),
                weightBytes, usable,
                backend.backendType().name());
    }

    /**
     * Upload planning of the streamed sets: the uploader's device types, shard limits and the input transforms,
     * plus the parameter selection of {@code neuralfree.stream.params} (vLLM's prefetch {@code offload_params}).
     */
    private LayerWeightSet.Options uploadOptions(EngineConfig cfg) {
        LayerWeightSet.Options.Builder b = LayerWeightSet.Options.builder(t -> uploader.deviceType(t.name()))
                .maxShardBytes(LayerWeightSet.Options.maxShardBytes(backend.capabilities()))
                .transforms(transforms);
        OffloadParams selection = OffloadParams.of(EngineConfig.PROP_STREAM_PARAMS, cfg.streamParams());
        if (!selection.isEmpty()) {
            b.filter(ref -> selection.matches(ref.name()));
        }
        return b.build();
    }

    /** Streamed sets of the trunk, with the uploader's device types, the model's input transforms, the stream
     * parameter selection and only the tensors {@code placed} puts on the device and no host-mapped tier holds (a
     * host expert tensor never streams, and neither does a host-mapped one). */
    private List<LayerWeightSet> trunkSets(ModelSpec s, WeightPlacement placed, EngineConfig cfg) {
        LayerWeightSet.Options options = uploadOptions(cfg);
        return LayerWeightSet.forTrunk(s, LayerWeightSet.Options.builder(options.policy())
                .filter(options.filter().and(ref -> placed.onDeviceName(ref.name()) && !offloaded.test(ref.name())))
                .maxShardBytes(options.maxShardBytes())
                .transforms(options.transforms())
                .build());
    }

    /**
     * Validates the {@code neuralfree.stream.params} selection against the tensors the load places on the device,
     * so a segment naming nothing is refused by the property it came from rather than silently streaming everything
     * or nothing.
     *
     * @throws IllegalArgumentException when a segment matches no device tensor, or the model keeps every weight
     *                                  resident so nothing can stream
     */
    private void validateStreamParams(ModelSpec initial, EngineConfig cfg, WeightPlacement weights) {
        if (cfg.streamParams().isEmpty()) {
            return;
        }
        OffloadParams selection = OffloadParams.of(EngineConfig.PROP_STREAM_PARAMS, cfg.streamParams());
        if (MemoryConfig.singleStepInput(initial)) {
            throw new IllegalArgumentException(EngineConfig.PROP_STREAM_PARAMS + " " + selection.segments()
                    + " select the streamed tensors, but this model keeps every weight resident; remove "
                    + EngineConfig.PROP_STREAM_PARAMS);
        }
        List<String> deviceNames = new ArrayList<>();
        for (String name : catalog.names()) {
            if (weights.onDeviceName(name) && !offloaded.test(name)) {
                deviceNames.add(name);
            }
        }
        List<String> unmatched = selection.unmatched(deviceNames);
        if (!unmatched.isEmpty()) {
            throw new IllegalArgumentException(EngineConfig.PROP_STREAM_PARAMS + " " + unmatched
                    + " match no tensor the load places on the device; name whole dotted parts of a stored tensor"
                    + " (for example experts or ffn_gate_exps)");
        }
    }

    /**
     * Refuses the offload settings beside an explicit layer split: the split's stages own their own weights and
     * uploads, so a host-mapped tensor would have to belong to exactly one stage and no stage streams. Naming the
     * properties keeps the configuration failure explicit.
     *
     * @throws UnsupportedOperationException when {@code neuralfree.cpu.offload.mb} or
     *                                       {@code neuralfree.stream.params} is set beside a layer split
     */
    private static void refuseOffloadWithSplit(ModelSpec spec, EngineConfig cfg, LayerOffload requested) {
        if (cfg.cpuOffloadMb() > 0) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + ": " + EngineConfig.PROP_CPU_OFFLOAD_MB
                    + "=" + cfg.cpuOffloadMb() + " keeps trunk weights in host-mapped memory, which a layer split"
                    + " (" + requested + ") cannot place; keep every layer on one device, or clear "
                    + EngineConfig.PROP_CPU_OFFLOAD_MB);
        }
        if (!cfg.streamParams().isEmpty()) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + ": " + EngineConfig.PROP_STREAM_PARAMS
                    + "=" + cfg.streamParams() + " selects the tensors of a streamed trunk, but a layer split ("
                    + requested + ") streams nothing; keep every layer on one device, or clear "
                    + EngineConfig.PROP_STREAM_PARAMS);
        }
    }

    /** Mebibytes with one decimal. */
    private static String mib(long bytes) {
        return String.format(Locale.ROOT, "%.1f", bytes / MIB);
    }

    /** Context length, streamed sets, the resolved hybrid split, the weight placement chosen at load and, for a
     * streaming placement, the {@link PlacementFitter.Fit} that chose it. */
    static final class Placement {
        final int nCtx;
        final List<LayerWeightSet> streamed;
        /** The hybrid split the fit resolved, or null. */
        final LayerOffload split;
        /** The tensors placed on the device: the given placement, or the split's when one was fitted. */
        final WeightPlacement weights;
        /** The streaming fit the placement resolved, or null for a resident or split placement. */
        final PlacementFitter.Fit fit;

        Placement(int nCtx, List<LayerWeightSet> streamed, LayerOffload split, WeightPlacement weights,
                  PlacementFitter.Fit fit) {
            this.nCtx = nCtx;
            this.streamed = streamed;
            this.split = split;
            this.weights = weights;
            this.fit = fit;
        }

        static Placement resident(int nCtx, WeightPlacement weights) {
            return new Placement(nCtx, Collections.<LayerWeightSet>emptyList(), null, weights, null);
        }
    }
}
