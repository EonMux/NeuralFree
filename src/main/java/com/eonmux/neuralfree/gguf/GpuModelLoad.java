package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.diag.GpuDiagnostics;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.HostMappedPlan;
import com.eonmux.neuralfree.gguf.gpu.weights.HostMappedWeightSource;
import com.eonmux.neuralfree.gguf.gpu.weights.HostPlacement;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSource;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeCacheLoader;
import com.eonmux.neuralfree.gguf.gpu.weights.StreamingLayerWeightSource;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hybrid.HybridFit;
import com.eonmux.neuralfree.gguf.hybrid.HybridForwardBackend;
import com.eonmux.neuralfree.gguf.hybrid.HybridMemory;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.attn.HostKvLayers;
import com.eonmux.neuralfree.gguf.ops.audio.GpuWavTokenizer;
import com.eonmux.neuralfree.gguf.ops.audio.TtsBackbone;
import com.eonmux.neuralfree.gguf.ops.encdec.GpuEncDecBackend;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.ops.heads.ControlVector;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionBody;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraBackend;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The single-device load assembly of a {@link GGUFModelGPU}: the backend and kernel setup, the weight placement
 * and upload, the memory allocation and the forward-path binding that {@link GGUFModelGPU#load} runs after the
 * file-level checks and the split-session shortcut. All state lives on the model; this class only sequences it.
 */
final class GpuModelLoad {

    private static final Logger LOGGER = LogManager.getLogger(GGUFModelGPU.class);

    private GpuModelLoad() {
    }

    /**
     * The single-device load: opens the backend, plans the weight placement, uploads the weights, allocates the
     * memory and binds the forward path ({@link GGUFModelGPU#load}). The parallel split plans are
     * {@link GpuSplitPlanner#select}'s.
     */
    static void load(GGUFModelGPU model, ComputeBackend sharedBackend, EngineConfig cfg, GgufKv kv,
                     ModelSpec initial, List<LoraAttachment.Binding> loraBindings) throws IOException {
        openBackend(model, sharedBackend, cfg);
        model.queue = model.backend.computeQueue();
        model.kernels = LlmKernels.open(model.backend);
        model.freeAtLoad = VRAMBudget.queryFreeVram(model.backend);
        model.allocatedAtLoad = model.backend.getAllocatedBytes();
        MemoryConfig mem = cfg.memory().forModel(initial);
        model.drafter = LLMModelFactory.draftCompanion(cfg, initial, false, model::draftDeviceWeightBytes)
                .plus(LLMModelFactory.twoLevelCompanion(cfg, model.source, cfg.speculativeOptions().w12()));
        model.placeholders = GpuPlaceholders.allocate(model.backend, model.queue);
        model.scratch = GpuScratch.forModel(model.backend, initial, mem.ubatch());
        model.store = model.source.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        model.uploader = new GpuWeightUploader(model.backend, model.queue, model.source.getTensors(), model.store, initial.arch().ggufName(),
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(model.kernels))
                        .transforms(model.transforms)
                        .w12(cfg.speculativeOptions().w12())
                        .w12Head(cfg.speculativeOptions().w12Head())
                        .w12F16(cfg.speculativeOptions().w12F16())
                        .build());
        model.hybridSplit = LayerOffload.requested(initial, cfg, model.source.getTensors());
        model.moeCacheLayers = MoeCacheLoader.routedMoeLayers(initial);
        model.moeCacheSettings = GGUFModelGPU.resolveMoeCache(cfg);
        model.moeCacheConfigured = HostPlacement.moeCacheRequested(cfg) && !model.moeCacheLayers.isEmpty();
        if (HostPlacement.moeCacheRequested(cfg)) {
            if (model.hybridSplit != null && model.hybridSplit.firstGpuLayer() > 0) {
                throw new UnsupportedOperationException(initial.arch().ggufName() + ": a hybrid load that offloads"
                        + " trunk layers (" + model.hybridSplit + ") cannot be combined with the MoE expert cache ("
                        + cfg.moeCache() + (cfg.moeCacheMb() > 0
                        ? ", " + EngineConfig.PROP_MOE_CACHE_MB + "=" + cfg.moeCacheMb() : "")
                        + "), which keeps the routed experts of the device layers in host RAM; clear "
                        + EngineConfig.PROP_CPU_MOE + " / " + EngineConfig.PROP_N_CPU_MOE + " / "
                        + EngineConfig.PROP_TENSOR_OVERRIDES + " / " + EngineConfig.PROP_N_GPU_LAYERS
                        + ", or " + MoeCacheConfig.PROP_MOE_CACHE);
            }
            if (model.moeCacheLayers.isEmpty()) {
                LOGGER.info("{}: the MoE expert cache is configured but the model has no routed MoE layers",
                        initial.arch().ggufName());
            }
        }
        if (cfg.tensorParallel() || cfg.contextMode()) {
            GpuSplitPlanner.Selected selected = GpuSplitPlanner.select(cfg, initial, model.backend,
                    model.uploader, model.modelName, model.eplb, loraBindings);
            if (selected != null) {
                model.parallelGroup = selected.group;
                model.rowSplitPlan = selected.rowSplit;
                model.attentionPlan = selected.attention;
                model.vocabPlan = selected.vocab;
                model.moePlan = selected.moe;
                model.eplbRebalancer = selected.rebalancer;
            }
        }
        Predicate<String> fixedHost = model.hybridSplit != null ? model.hybridSplit::hostTensor : n -> false;
        model.adapters = GpuAdapters.load(model.modelName, initial, cfg, mem, loraBindings,
                LoraAttachment.adaptableShapes(model.catalog, n -> !fixedHost.test(n)));
        model.pendingDeviceBytes = Math.addExact(model.adapters.deviceBytes(), model.adapters.slotBytes());
        model.weightPlacement = HostPlacement.placement(cfg, initial, model.uploader, model::usableVram,
                GGUFModelGPU.loadsNextn(cfg, mem), fixedHost);
        model.hostMappedPlan = planHostMapped(model, cfg, initial, model.weightPlacement);

        GpuLoadPlacement.Placement placement = new GpuLoadPlacement(model.source, model.catalog, model.transforms, model.uploader, model.backend,
                model.drafter, model::usableVram, model.moeCacheConfigured, model.hostMappedPlan::contains)
                .place(initial, mem, cfg, model.weightPlacement, model.hybridSplit);
        model.placement = placement;
        model.weightPlacement = placement.weights;
        model.hybridSplit = placement.split;
        if (model.parallelGroup != null && model.hybridSplit != null && model.hybridSplit.firstGpuLayer() > 0) {
            throw new UnsupportedOperationException(model.modelName + ": " + EngineConfig.PROP_SPLIT_MODE + "="
                    + cfg.splitMode().llamaName() + " keeps a shard of every split projection on every"
                    + " rank, so the " + model.hybridSplit.firstGpuLayer() + " trunk layer(s) left in host RAM cannot run;"
                    + " keep every layer on the device (" + EngineConfig.PROP_N_GPU_LAYERS + "=all), lower the"
                    + " context, or use " + EngineConfig.PROP_SPLIT_MODE + "=layer");
        }
        if (model.parallelGroup != null && !placement.streamed.isEmpty()) {
            throw new UnsupportedOperationException(model.modelName + ": " + EngineConfig.PROP_SPLIT_MODE + "="
                    + cfg.splitMode().llamaName() + " keeps the weights it shards resident on every"
                    + " rank, but the placement streams " + placement.streamed.size() + " layer set(s) through"
                    + " the device; raise the VRAM depth (" + EngineConfig.PROP_VRAM_DEPTH + "), lower the context,"
                    + " or use " + EngineConfig.PROP_SPLIT_MODE + "=layer");
        }
        model.hostedLayers = GpuHostedLayers.resolve(initial, model.catalog, model.weightPlacement, model.hybridSplit);
        if (model.hostedLayers.hasCached()) {
            model.cpuWeights = model.cpuWeights();
        }
        model.hostedLayers.log(model.modelName, model.catalog, model.moeCacheConfigured ? model.moeCacheSettings : MoeCacheConfig.off());
        int derived = mem.autoContext() ? initial.hparams().nCtxTrain() : mem.nCtx();
        model.spec = placement.nCtx == derived ? initial
                : ModelSpecBuilder.build(kv, model.catalog, mem, placement.nCtx, cfg.pooling());
        model.forwardSpec = DecisionBody.forwardSpec(model.spec);
        if (model.adapters.controlVectors() != null) {
            model.forwardSpec = ControlVector.applied(model.forwardSpec);
        }
        model.plan = MemoryPlanner.planForContext(model.spec, mem, placement.nCtx);
        model.config = cfg.toBuilder().memory(mem.toBuilder().ubatch(model.plan.ubatch()).build()).build();
        if (model.plan.ubatch() < model.scratch.ubatch()) {
            GpuScratch larger = model.scratch;
            model.scratch = null;
            larger.close();
            model.scratch = GpuScratch.forModel(model.backend, model.spec, model.plan);
        }
        model.ops = new GpuOps(model.queue, model.kernels, model.scratch, model.placeholders);
        if (model.plan.kind() != MemoryKind.NONE) {
            model.drafterDeviceBytes = ContextFitter.reservedBytes(model.spec, mem, model.plan, model.drafter);
            if (model.hybridSplit != null) {
                allocateHybridMemory(model);
            } else {
                // a tensor split keeps the attention caches on the ranks, so the main device allocates none of them
                java.util.function.IntPredicate mainLayers = model.attentionPlan == null
                        ? il -> true
                        : il -> model.attentionPlan.layerOf(model.spec.layers().get(il)) == null;
                model.memory = DeviceModelMemory.allocate(model.backend, model.queue, model.plan, model.spec.hparams().mropeTextPositions(),
                        null, mainLayers);
            }
        }
        if (model.parallelGroup != null) {
            model.parallelGroup.expertHeat(model.eplb == null ? null : model.eplb.heat());
            model.parallelGroup.moeLoadWindow(model.eplbRebalancer == null ? null : model.eplbRebalancer.window());
            model.parallelGroup.init(model.spec, model.plan.ubatch(), model.kernels, model.queue, model.placeholders, model.rowSplitPlan, model.attentionPlan,
                    model.vocabPlan, model.moePlan, model.config.memory(), model.plan.nCtx());
        }
        uploadWeights(model, placement.streamed);
        if (model.parallelGroup != null) {
            model.parallelGroup.upload(model.store, model.uploader);
        }
        model.moeCache = MoeCacheLoader.open(model.moeCacheSettings, model.source.getTensors(), model.store, model.uploader, model.backend, model.queue,
                model.plan.ubatch(), model.moeCacheLayers, model.usableVram(), model.ranking, model.modelName);
        model.moePrefetch = MoeCacheLoader.openPrefetch(model.moeCacheSettings, model.moeCache, model.backend, model.modelName);
        bindForward(model);
        model.loras = model.hybridSplit != null
                ? model.adapters.attachHybridLoras(model.modelName, model.kernels, model.ops, model.ctx.projectionDeltas(),
                        model.cpuContext.projectionDeltas())
                : model.adapters.attachLoras(model.modelName, model.kernels, model.ops, model.ctx.projectionDeltas());
        if (model.adapters.hasLoras()) {
            model.pendingDeviceBytes = model.adapters.slotBytes();
        }
        if (model.spec.arch() != Arch.WAVTOKENIZER_DEC) {
            model.loraRegistry = GpuAdapters.perRequest(model.loras, Collections.singletonList(model.kernels), cfg,
                    model::cpuWeights, new long[]{GpuAdapters.perSlot(cfg, model.adapters.slotBytes())},
                    model.scratch.ubatch());
        }
        LOGGER.info("{} ({}) on {} {}: context {}, ubatch {}, memory {} MiB, resident weights {} MiB{}",
                model.modelName, model.spec.arch().ggufName(), model.backend.backendType(), model.backend.capabilities().getDeviceName(),
                model.plan.nCtx(), model.plan.ubatch(), model.mib(model.plan.totalBytes()), model.mib(model.uploader.residentBytes()),
                (model.streaming == null ? "" : ", " + model.streaming.streamedLayers() + " layers streamed through "
                        + model.streaming.depth() + " slots (" + model.mib(model.streaming.deviceBytes()) + " MiB)")
                        + (model.hostMapped == null ? "" : ", " + model.hostMapped.tensorCount() + " tensors host-mapped ("
                        + model.mib(model.hostMapped.hostBytes()) + " MiB)"));
        if (model.hybridSplit != null) {
            LOGGER.info("{}: {}", model.modelName, model.hybridSplit);
        }
        if (model.weightPlacement.hasNextn() && !model.weightPlacement.nextnOnDevice()) {
            LOGGER.info("{}: {} NextN block(s) stay off the device ({} MiB); set {}=true to draft with them",
                    model.modelName, model.spec.nextnLayers().size(), model.mib(model.uploader.plannedBytes(model.weightPlacement::isNextn)),
                    EngineConfig.PROP_MTP);
        }
    }

    /**
     * Allocates the two sides of a hybrid load over the shared plan: the device memory of the offloaded layers, the
     * host scratch and heap memory of the earlier layers, and the host context the host executors bind against.
     */
    static void allocateHybridMemory(GGUFModelGPU model) throws IOException {
        model.memory = DeviceModelMemory.allocate(model.backend, model.queue, model.plan, model.spec.hparams().mropeTextPositions(), null,
                HybridFit.deviceLayers(model.hybridSplit));
        model.cpuScratch = CpuScratch.forModel(model.spec, model.plan.ubatch());
        HeapModelMemory host = HeapModelMemory.allocate(model.plan, model.spec, null, model.hybridSplit::hostMemoryLayer);
        model.hybridMemory = HybridMemory.of(model.memory, host);
        model.cpuContext = CpuForwardContext.builder(model.forwardSpec, model.cpuWeights(), model.cpuScratch).memory(host)
                .heat(model.ranking.heat()).build();
    }

    static void openBackend(GGUFModelGPU model, ComputeBackend sharedBackend, EngineConfig cfg) {
        if (sharedBackend != null) {
            model.backend = sharedBackend;
            return;
        }
        HWAcceleration acceleration = HWAcceleration.forMode(cfg.backend().toAccelerationMode());
        model.ownedAcceleration = acceleration;
        model.backend = cfg.picksMainGpu() ? GpuDevices.openMainGpu(acceleration, cfg).backend()
                : acceleration.createBackend(cfg.backend().toAccelerationMode(), LlmKernels.requirement());
    }

    /**
     * Plans the host-mapped tier: the trunk tensors the load places on the device that
     * {@code neuralfree.cpu.offload.mb} and {@code neuralfree.cpu.offload.params} select, in weight-layer and file
     * order up to the budget. Their bytes leave the device weight fit; the device reads them in place.
     *
     * @throws UnsupportedOperationException when the backend has no host-mapped buffers
     * @throws IllegalArgumentException      as {@link HostMappedPlan#plan}
     */
    private static HostMappedPlan planHostMapped(GGUFModelGPU model, EngineConfig cfg, ModelSpec initial,
                                                 WeightPlacement placement) {
        if (cfg.cpuOffloadMb() == 0) {
            return HostMappedPlan.empty();
        }
        if (!model.backend.supportsHostMappedBuffers()) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + ": "
                    + EngineConfig.PROP_CPU_OFFLOAD_MB + " keeps trunk weights in host-mapped memory, which the "
                    + model.backend.backendType() + " backend does not provide; clear " + EngineConfig.PROP_CPU_OFFLOAD_MB
                    + ", or use a CUDA or Vulkan backend");
        }
        LayerWeightSet.Options options = LayerWeightSet.Options.builder(t -> model.uploader.deviceType(t.name()))
                .maxShardBytes(LayerWeightSet.Options.maxShardBytes(model.backend.capabilities()))
                .transforms(model.transforms)
                .build();
        return HostMappedPlan.plan(cfg, initial, options, placement::onDeviceName);
    }

    /**
     * Uploads the resident weights, every tensor placed on the device ({@link #weightPlacement}) but the streamed
     * sets of the placement and the host-mapped tensors, then opens the streaming source over them and the usable
     * VRAM that is left after the drafter memory the context fit reserved (its depth from
     * {@code neuralfree.vram.depth}, else the largest that fits). The streaming source reads its layers through a
     * store of its own ({@link StoreUse#STREAMING}), released with it. The host-mapped tensors are filled once into
     * page-locked memory through a host tier of the same store and read in place from then on.
     */
    private static void uploadWeights(GGUFModelGPU model, List<LayerWeightSet> streamed)
            throws IOException {
        WeightPlacement placed = model.weightPlacement;
        Set<String> mapped = model.hostMappedPlan.names();
        if (streamed.isEmpty() && mapped.isEmpty()) {
            model.uploader.uploadResident(t -> !placed.onDevice(t)
                    || (model.rowSplitPlan != null && model.rowSplitPlan.splits(t.name()))
                    || (model.attentionPlan != null && model.attentionPlan.splits(t.name()))
                    || (model.vocabPlan != null && model.vocabPlan.splits(t.name()))
                    || (model.moePlan != null && model.moePlan.splits(t.name())));
            return;
        }
        Set<String> trunk = new HashSet<>(LayerWeightSet.allNames(streamed));
        model.uploader.uploadResident(t -> !placed.onDevice(t) || trunk.contains(t.name()) || mapped.contains(t.name()));
        model.streamingStore = model.source.createMappedStore(StoreUse.STREAMING);
        if (!streamed.isEmpty()) {
            model.streaming = StreamingLayerWeightSource.open(model.backend, model.queue, model.uploader, model.streamingStore,
                    model.source.getTensors(), model.forwardSpec, streamed, model.config, model.usableVram());
            model.streamedSets = streamed;
        }
        if (!mapped.isEmpty()) {
            WeightDataProvider data = new WeightDataProvider(model.streamingStore, model.source.getTensors(),
                    Collections.<List<String>>emptyList(), 0, 0L);
            LayerWeightSource resident = model.streaming != null ? model.streaming : LayerWeightSource.resident(model.uploader);
            model.hostMapped = HostMappedWeightSource.open(model.backend, model.queue, data, model.forwardSpec.arch().ggufName(),
                    model.hostMappedPlan, resident);
        }
    }

    /** Builds the context, the registry and the forward backend, and prewarms every kernel they launch. */
    private static void bindForward(GGUFModelGPU model) throws IOException {
        GpuDiagnostics.Config diag = GpuDiagnostics.Config.fromSystemProperties();
        if (diag.enabled()) {
            model.observer.setBase(new GpuDiagnostics(model.queue, diag, System.err, model.deviceName()));
        }
        GpuForwardContext.Builder b = GpuForwardContext.builder(model.kernels, model.ops, model.uploader, model.forwardSpec)
                .observer(model.observer);
        if (model.hostMapped != null) {
            b.weightSource(model.hostMapped);
        } else if (model.streaming != null) {
            b.weightSource(model.streaming);
        }
        if (model.moeCache != null) {
            b.moeExpertCache(model.moeCache);
            b.moeExpertPrefetch(model.moePrefetch);
        }
        b.moeTailSkipRank(model.moeCacheSettings.tailSkipRank());
        if (model.hostedLayers.hasCached() && model.cpuWeights != null) {
            b.hostWeights(model.cpuWeights);
        }
        if (model.memory != null) {
            b.memory(model.memory);
        }
        if (model.hybridSplit != null) {
            b.hostKv(model.hybridSplit::hostKvLayer);
        }
        b.heat(model.eplb != null && model.eplb.heat() != null ? model.eplb.heat() : model.ranking.heat());
        model.ctx = b.build();
        model.router = GpuFactoryCatalog.usesSwitchedLora(model.forwardSpec) ? SwitchedLoraExtra.Router.of(model.spec.hparams())
                : null;
        // the decorators compose: a layer whose expert set stays in RAM keeps its host FFN, and every planned
        // feed-forward takes the split block
        GpuBlockRegistry gpuRegistry = model.hostedLayers.decorate(
                GpuFactoryCatalog.defaults(model.router, model.adapters.controlVectors()), model.cpuContext);
        if (model.hybridSplit != null && model.hybridSplit.hostKv()) {
            gpuRegistry = HostKvLayers.decorate(gpuRegistry, model.cpuContext, model.hybridSplit::hostKvLayer);
        }
        if (model.parallelGroup != null) {
            if (model.attentionPlan != null) {
                gpuRegistry = gpuRegistry.withMixerDecorator(model.parallelGroup.attentionDecorator());
            }
            gpuRegistry = gpuRegistry.withFfnDecorator(model.parallelGroup.decorator());
        }
        model.registry = gpuRegistry;
        model.kernels.prewarm(GpuFactoryCatalog.kernels(model.forwardSpec));
        WeightKernelPrewarm.prewarm(model.uploader.plans(model.weightPlacement::onDevice), model.ops, model.kernels);
        if (model.spec.arch() == Arch.WAVTOKENIZER_DEC) {
            model.kernels.prewarm(GpuWavTokenizer.KERNELS);
            model.tasks.bindWavTokenizer(GpuWavTokenizer.bind(model.ctx));
            return;
        }
        if (model.hybridSplit != null) {
            model.rawBackend = HybridForwardBackend.bind(model.ctx, gpuRegistry, GpuFactoryCatalog.headPreprocessors(),
                    model.cpuContext, CpuFactoryCatalog.defaults(null, model.adapters.controlVectors()), model.hybridSplit,
                    model.hybridMemory);
        } else if (model.spec.supportClass() == SupportClass.ENC_DEC) {
            model.rawBackend = new GpuEncDecBackend(model.ctx, gpuRegistry, GpuFactoryCatalog.headPreprocessors(),
                    model.tokenizer.specialTokens().bos());
        } else {
            model.rawBackend = new GpuForwardBackend(model.ctx, gpuRegistry, GpuFactoryCatalog.headPreprocessors(),
                    model.parallelGroup == null ? null : model.parallelGroup.vocabEmbedding(),
                    model.parallelGroup == null ? null : model.parallelGroup.vocabHead());
        }
        if (model.parallelGroup != null && model.rawBackend instanceof com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend) {
            // the split blocks synchronize with peer queues between the main queue's dispatches; a one-queue graph
            // cannot record that, so their decode steps run directly
            ((com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend) model.rawBackend).disableDecodeReplay();
            if (model.attentionPlan != null && model.parallelGroup.pcpSize() > 1) {
                // the DualChunkSwap prefill: rank 0's executor drives the phases and the coordinator interlaces
                // the ranks layer by layer between them; over a paged memory it carries the whole batched step
                // instead (the segments' chunk pairs run over the ranks' translated page tables)
                com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend raw =
                        (com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend) model.rawBackend;
                if (model.memory != null && model.memory.pagedSequences() != null) {
                    raw.batchTrunkRunner(model.parallelGroup.pcpBatchTrunkRunner(model.ctx, raw.layerExecutor()));
                } else {
                    raw.trunkRunner(model.parallelGroup.pcpTrunkRunner(model.ctx, raw.layerExecutor()));
                }
            }
        }
        ForwardBackend f = model.rawBackend;
        if (model.parallelGroup != null && model.attentionPlan != null) {
            f = new com.eonmux.neuralfree.gguf.gpu.parallel.TensorParallelBackend(f, model.parallelGroup);
        }
        if (model.eplbRebalancer != null) {
            f = new com.eonmux.neuralfree.gguf.gpu.parallel.EplbStepBackend(f, model.eplbRebalancer);
        }
        if (model.spec.arch() == Arch.QWEN3TTS) {
            f = TtsBackbone.qwen3tts(f, model.spec);
        }
        if (model.router != null) {
            f = SwitchedLoraBackend.wrap(f, model.router);
        }
        model.forward = f;
        if (model.streaming != null) {
            model.streaming.warmup();
        }
    }
}
