package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.QCopySource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory.DraftRoute;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory.DraftWeightBytes;
import com.eonmux.neuralfree.gguf.engine.draft.DrafterChain;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.HostPlacement;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead;
import com.eonmux.neuralfree.gguf.ops.flow.NextnDraftModel;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DerivedDraftSource;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashDevice;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashDrafter;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashSpec;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ScratchWidths;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * The drafter binding of {@link LLMModelFactory}: which drafter a draft file drafts as ({@link DraftRoute}), how the
 * draft model's own load is placed ({@code -devd}, {@code -ngld}, {@code -otd}, {@code -cmoed}, {@code -ncmoed}),
 * the drafter of a configuration bound to an open target ({@code #drafters}: the target's NextN head, an MTP draft
 * file, the {@code dflash} / {@code eagle} / {@code eagle3} / {@code gemma4-assistant} drafters, or a separate draft
 * model), and the drafter memory a target's context fit reserves beside ({@code #draftCompanion}). The factory's
 * public {@code openSpeculative} / {@code speculate} / {@code draftCompanion} / {@code openBatchDraftModel} entry
 * points delegate here; the routes and placement helpers are shared with the tests through the same names on the
 * factory.
 */
final class DrafterBinding {

    private static final Logger LOGGER = LogManager.getLogger(DrafterBinding.class);

    private DrafterBinding() {
    }

    /**
     * A bound drafter with its kind, its own draft-length limit and the writes that run after it closes (the
     * dynamic lookup caches).
     */
    static final class Drafter {
        final DraftModel drafter;
        final String kind;
        final int maxDraft;
        final List<AutoCloseable> saves;

        Drafter(DraftModel drafter, String kind, int maxDraft) {
            this(drafter, kind, maxDraft, Collections.emptyList());
        }

        Drafter(DraftModel drafter, String kind, int maxDraft, List<AutoCloseable> saves) {
            this.drafter = drafter;
            this.kind = kind;
            this.maxDraft = maxDraft;
            this.saves = saves;
        }
    }

    /**
     * The drafter {@code cfg} configures, bound to {@code target}: the lookup drafters of its lookup types
     * ({@link LookupDrafters}) and the drafter of the draft file or of the MTP head, chained in llama.cpp's ask order
     * ({@link DrafterChain}, the lookup drafters first) when there are several. In a chain each member keeps its own
     * draft-length limit. A drafter that fails to bind closes the ones already bound.
     */
    static Drafter drafters(EngineModel target, Path targetPath, EngineConfig cfg, List<AutoCloseable> out)
            throws IOException {
        LookupDrafters lookups = LookupDrafters.open(target, cfg);
        Drafter file = null;
        try {
            if (cfg.mtp()) {
                file = nextn(target, cfg);
            } else if (cfg.draftModelPath() != null) {
                file = fromFile(target, targetPath, cfg, out);
            }
        } catch (IOException | RuntimeException | Error e) {
            Throwable first = e;
            for (DraftModel d : lookups.drafters()) {
                first = LLMModelFactory.closeStep(d, first);
            }
            throw e;
        }
        if (lookups.isEmpty()) {
            if (file == null) {
                throw new IllegalArgumentException("no drafter is configured for " + target.name());
            }
            return file;
        }
        List<DraftModel> members = new ArrayList<>(lookups.drafters());
        List<String> kinds = new ArrayList<>(lookups.names());
        if (file != null) {
            members.add(file.drafter);
            kinds.add(file.kind);
        }
        if (members.size() == 1) {
            return new Drafter(members.get(0), kinds.get(0), Integer.MAX_VALUE, lookups.saves());
        }
        int[] limits = new int[members.size()];
        Arrays.fill(limits, Integer.MAX_VALUE);
        if (file != null) {
            limits[limits.length - 1] = file.maxDraft;
        }
        return new Drafter(new DrafterChain(members, limits), String.join("+", kinds), Integer.MAX_VALUE,
                lookups.saves());
    }

    private static Drafter nextn(EngineModel target, EngineConfig cfg) {
        MtpHead head = targetHead(target);
        DraftModel d = DraftModelLoader.nextn(target, head, DraftModelLoader.Options.of(cfg.draftPMin()));
        return new Drafter(d, "mtp", mtpDraftMax(target.spec()));
    }

    /**
     * The draft-length limit of an MTP drafter over the NextN blocks of {@code mtp}: their count when there are
     * several, whose heads chain one block per drafted token (llama.cpp clamps {@code n_max} to
     * {@code n_layer_nextn} when {@code chain_heads}, common/speculative.cpp:1523-1528); none for a single head or
     * a cycling head, whose blocks repeat (vLLM {@code spec_step_idx % num_mtp_layers}).
     */
    private static int mtpDraftMax(ModelSpec mtp) {
        int blocks = mtp.nextnLayers().size();
        return blocks > 1 && !NextnDraftModel.cyclesBlocks(mtp.arch()) ? blocks : Integer.MAX_VALUE;
    }

    /**
     * The drafter of {@code cfg.draftModelPath()}, opened with the target {@code targetPath} the draft serves (null
     * when the target model was opened elsewhere and carries no path, so a Hugging Face draft of a drafter
     * architecture cannot name its target); resources the drafter does not own are appended to {@code out}.
     */
    private static Drafter fromFile(EngineModel target, Path targetPath, EngineConfig cfg, List<AutoCloseable> out)
            throws IOException {
        String path = cfg.draftModelPath();
        ModelSources.requireOpenable(path);
        MemoryConfig draftMemory = draftMemory(target, cfg);
        ModelSource dp = ModelSources.open(path, cfg.sourceOptions(), targetPath, cfg.activationQuant());
        Arch arch;
        DraftRoute route;
        try {
            arch = draftArch(dp, path);
            route = draftRoute(arch, GgufKv.of(dp), TensorCatalog.of(dp), target.spec(), cfg.draftType(), path);
        } catch (RuntimeException e) {
            LLMModelFactory.closeStep(dp::close, e);
            throw e;
        }
        if (route != DraftRoute.MODEL && cfg.speculativeOptions().heterogeneousVocab()) {
            IllegalArgumentException refusal = new IllegalArgumentException(SpeculativeOptions
                    .PROP_HETEROGENEOUS_VOCAB + " needs a separate draft model (draft-simple), but " + path + " is a "
                    + arch.ggufName() + " drafter");
            LLMModelFactory.closeStep(dp::close, refusal);
            throw refusal;
        }
        if (route == DraftRoute.MODEL) {
            dp.close();
            return modelDrafter(target, path, cfg, draftMemory);
        }
        refuseDraftPlacement(route, arch, path, cfg);
        out.add(dp::close);
        if (route == DraftRoute.MTP_FILE) {
            return mtpFileDrafter(target, dp, cfg, out);
        }
        if (route == DraftRoute.DFLASH) {
            return dflashDrafter(target, dp, arch, cfg, out);
        }
        refuseSplitTargetCaches(target, arch, path);
        GgufKv kv = GgufKv.of(dp);
        TensorCatalog catalog = TensorCatalog.of(dp);
        ModelSpec spec = ModelSpecBuilder.build(kv, catalog, draftMemory);
        MtpHead head = draftHead(target, dp, kv, catalog, spec, cfg.activationQuant(), t -> false, out);
        DraftModel d = DraftModelLoader.fromFile(target, spec, head, draftMemory,
                DraftModelLoader.Options.of(cfg.draftPMin()));
        return new Drafter(d, arch.ggufName(), Integer.MAX_VALUE);
    }

    /**
     * The {@code dflash} drafter of {@code dp} bound to {@code target}: on a GPU target with
     * {@link SpeculativeOptions#dflashOnDevice()} on the target's device ({@link DflashDevice}, the draft's matrices
     * uploaded beside the target's weights and appended to {@code out}); on the host otherwise, and when the device
     * cannot run the draft, which is logged with its reason (llama.cpp's draft backend-sampling fallback,
     * common/speculative.cpp:1127-1129).
     */
    private static Drafter dflashDrafter(EngineModel target, ModelSource dp, Arch arch, EngineConfig cfg,
                                         List<AutoCloseable> out) throws IOException {
        SpeculativeOptions spec = cfg.speculativeOptions();
        DraftHeadOptions head = spec.draftHead();
        if (head != null && head.enabled()) {
            dp = DerivedDraftSource.wrap(dp, LLMModelFactory.hostWeights(target), head);
        }
        DflashDrafter.Options options = DflashDrafter.Options.of(cfg.draftMin(), cfg.draftPMin(),
                cfg.memory().draftTypeK(), cfg.memory().kvRotation()).withActivationQuant(cfg.activationQuant())
                .withTree(spec.treeVerify(), spec.treeTemp());
        List<AutoCloseable> deviceResources = new ArrayList<>();
        try {
            DflashDevice device = dflashDevice(target, dp, cfg, options, deviceResources);
            if (device != null) {
                DflashDrafter d = boundDflash(DflashDrafter.open(dp, LLMModelFactory.hostWeights(target), options, device), target);
                out.addAll(deviceResources);
                LOGGER.info("dflash drafter {} drafts on {}{}", cfg.draftModelPath(), device,
                        spec.treeVerify() ? " with single-pass draft-tree verification" : "");
                return new Drafter(d, arch.ggufName(), d.spec().nDraftMax());
            }
        } catch (UnsupportedOperationException e) {
            Throwable closeFailure = null;
            for (int i = deviceResources.size() - 1; i >= 0; i--) {
                closeFailure = LLMModelFactory.closeStep(deviceResources.get(i), closeFailure);
            }
            if (closeFailure != null) {
                e.addSuppressed(closeFailure);
            }
            LOGGER.warn("dflash drafter {} drafts on the host: {}", cfg.draftModelPath(), LLMModelFactory.describe(e));
        } catch (IOException | RuntimeException | Error e) {
            out.addAll(deviceResources);
            throw e;
        }
        DflashDrafter d = boundDflash(DflashDrafter.open(dp, LLMModelFactory.hostWeights(target), options), target);
        LOGGER.info("dflash drafter {} drafts on the host{}", cfg.draftModelPath(),
                spec.treeVerify() ? " with single-pass draft-tree verification" : "");
        return new Drafter(d, arch.ggufName(), d.spec().nDraftMax());
    }

    /** {@code d} bound to {@code target}; a failed bind closes it. */
    private static DflashDrafter boundDflash(DflashDrafter d, EngineModel target) {
        try {
            d.bindTarget(target);
        } catch (RuntimeException e) {
            LLMModelFactory.closeStep(d, e);
            throw e;
        }
        return d;
    }

    /**
     * The device a {@code dflash} draft of {@code target} drafts on, with the draft's matrices uploaded to it
     * ({@link DflashDevice#reads}) and appended to {@code out}; null for a host target, with
     * {@link SpeculativeOptions#dflashOnDevice()} off, for a layer-split target and for a draft
     * {@link DflashDevice#refusal} keeps on the host (logged with the reason).
     */
    private static DflashDevice dflashDevice(EngineModel target, ModelSource dp, EngineConfig cfg,
                                             DflashDrafter.Options options, List<AutoCloseable> out)
            throws IOException {
        if (!(target instanceof GGUFModelGPU) || !cfg.speculativeOptions().dflashOnDevice()) {
            return null;
        }
        GGUFModelGPU g = (GGUFModelGPU) target;
        String refusal = g.isSplit()
                ? EngineConfig.PROP_SPLIT_MODE + "=layer spreads the target over several devices"
                : DflashDevice.refusal(DflashDrafter.readSpec(dp), options.kvType());
        if (refusal != null) {
            LOGGER.info("dflash drafter {} drafts on the host: {}", cfg.draftModelPath(), refusal);
            return null;
        }
        GgufKv kv = GgufKv.of(dp);
        TensorCatalog catalog = TensorCatalog.of(dp);
        GpuWeightUploader uploader = uploadDraft(g, dp, kv, catalog, Arch.DFLASH.ggufName(), cfg.activationQuant(),
                t -> !DflashDevice.reads(t.name()), out);
        return new DflashDevice(g.forwardContext(), uploader);
    }

    /**
     * Refuses a drafter that attends over the target's own KV caches ({@code gemma4-assistant}) on a target split over
     * several GPUs, whose caches are spread over the devices of their layers while the drafter runs on the main one.
     */
    private static void refuseSplitTargetCaches(EngineModel target, Arch arch, String path) {
        if (arch == Arch.GEMMA4_ASSISTANT && target instanceof GGUFModelGPU && ((GGUFModelGPU) target).isSplit()) {
            throw new UnsupportedOperationException("the " + arch.ggufName() + " drafter " + path + " attends over"
                    + " the target's KV caches on one device, but " + EngineConfig.PROP_SPLIT_MODE + "=layer spreads"
                    + " them over the devices of their layers; draft with it over " + EngineConfig.PROP_SPLIT_MODE
                    + "=none");
        }
    }

    static DraftRoute draftRoute(Arch arch, GgufKv kv, TensorCatalog tc, ModelSpec target,
                                 EngineConfig.DraftType type, String path) {
        Objects.requireNonNull(arch, "arch");
        Objects.requireNonNull(type, "type");
        boolean drafterArch = arch == Arch.DFLASH || arch == Arch.EAGLE || arch == Arch.EAGLE3
                || arch == Arch.EAGLE_MISTRAL_LARGE3 || arch == Arch.EAGLE_LLAMA4 || arch == Arch.EAGLE_MINICPM
                || arch == Arch.EAGLE_COHERE || arch == Arch.GEMMA4_ASSISTANT;
        if (drafterArch && type != EngineConfig.DraftType.AUTO) {
            throw new IllegalArgumentException("draft type " + type.llamaName() + ": the draft model " + path
                    + " is a " + arch.ggufName() + " drafter, which drafts only as itself; use draft type auto");
        }
        if (arch == Arch.DFLASH) {
            return DraftRoute.DFLASH;
        }
        if (drafterArch) {
            return DraftRoute.HEAD;
        }
        boolean nextn = ModelSpecBuilder.isNextnDraft(kv, tc);
        if (type == EngineConfig.DraftType.MODEL) {
            if (nextn && ModelSpecBuilder.isMtpOnly(kv, tc)) {
                throw new IllegalArgumentException("draft type " + type.llamaName() + ": the draft model " + path
                        + " holds only the MTP blocks of its model, which draft only as an MTP drafter (draft-mtp)");
            }
            return DraftRoute.MODEL;
        }
        if (!nextn && type == EngineConfig.DraftType.AUTO) {
            return DraftRoute.MODEL;
        }
        String mismatch = ModelSpecBuilder.mtpDraftMismatch(kv, tc, target);
        if (mismatch == null) {
            return DraftRoute.MTP_FILE;
        }
        boolean mtpOnly = nextn && ModelSpecBuilder.isMtpOnly(kv, tc);
        if (type == EngineConfig.DraftType.MTP || mtpOnly) {
            throw new IllegalArgumentException("draft type " + type.llamaName() + ": the draft model " + path
                    + " is not an MTP draft of the target " + target.arch().ggufName() + " (" + mismatch + ")"
                    + (mtpOnly ? "; it holds only MTP blocks and has no model of its own to draft with"
                    : "; draft type draft-simple runs it as a separate draft model"));
        }
        LOGGER.info("the draft model {} holds NextN / MTP blocks but is not an MTP draft of the target {} ({});"
                + " it drafts as a separate draft model", path, target.arch().ggufName(), mismatch);
        return DraftRoute.MODEL;
    }

    static void refuseDraftPlacement(DraftRoute route, Arch arch, String path, EngineConfig cfg) {
        if (route == DraftRoute.MODEL || !cfg.hasDraftPlacement()) {
            return;
        }
        throw new UnsupportedOperationException("the draft model placement settings ("
                + EngineConfig.PROP_DRAFT_GPU_DEVICES + ", " + EngineConfig.PROP_DRAFT_N_GPU_LAYERS + ", "
                + EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES + ", " + EngineConfig.PROP_DRAFT_CPU_MOE + ", "
                + EngineConfig.PROP_DRAFT_N_CPU_MOE + ": llama.cpp -devd, -ngld, -otd, -cmoed, -ncmoed) place the"
                + " load of a separate draft model, but " + path + " drafts as a " + arch.ggufName() + " "
                + (route == DraftRoute.MTP_FILE ? "MTP drafter over the target's hidden rows"
                : "drafter on the target's runtime") + "; draft with it as a separate model (draft type"
                + " draft-simple) to place it");
    }

    /**
     * The MTP drafter of an MTP draft file ({@link ModelSpecBuilder#isNextnDraft}): its spec over the target's trunk
     * ({@link ModelSpecBuilder#buildMtpDraft}) with the draft-context settings of the target plan, and the weights it
     * reads ({@link ModelSpecBuilder#mtpDraftReads}: never the file's trunk) uploaded to the target's device or mapped
     * ({@link #draftHead}).
     */
    private static Drafter mtpFileDrafter(EngineModel target, ModelSource dp, EngineConfig cfg,
                                          List<AutoCloseable> out) throws IOException {
        GgufKv kv = GgufKv.of(dp);
        TensorCatalog catalog = TensorCatalog.of(dp);
        ModelSpec spec = ModelSpecBuilder.buildMtpDraft(kv, catalog, mtpMemory(target), target.spec());
        Predicate<String> reads = ModelSpecBuilder.mtpDraftReads(spec);
        MtpHead head = draftHead(target, dp, kv, catalog, spec, cfg.activationQuant(), t -> !reads.test(t.name()),
                out);
        DraftModel d = DraftModelLoader.nextn(target, spec, head, DraftModelLoader.Options.of(cfg.draftPMin()));
        return new Drafter(d, "mtp draft file " + spec.arch().ggufName(), mtpDraftMax(spec));
    }

    /**
     * Memory settings of an MTP drafter's spec and caches: the target plan's settings as a draft context runs with
     * them ({@link MemoryConfig#forDraftContext()}), with the plan's context and the plan ubatch the draft ubatch
     * replaces when one is set, as llama.cpp creates the MTP context from the speculative parameters
     * (common/speculative.cpp:2714-2715, 2658-2669).
     */
    private static MemoryConfig mtpMemory(EngineModel target) {
        MemoryPlan plan = target.forwardBackend().memory().plan();
        MemoryConfig settings = plan.config();
        if (settings == null) {
            throw new IllegalStateException(target.architecture() + ": the target memory plan records no settings to"
                    + " plan an MTP drafter with");
        }
        return settings.forDraftContext().toBuilder().nCtx(plan.nCtx())
                .ubatch(settings.draftUbatchOr(plan.ubatch())).build();
    }

    static Predicate<GGUFTensorInfo> deviceTensors(DraftRoute route, GgufKv kv, TensorCatalog catalog,
                                                   MemoryConfig memory, List<GGUFTensorInfo> tensors,
                                                   EngineConfig draftPlacement) {
        switch (route) {
            case DFLASH:
                return t -> DflashDevice.reads(t.name());
            case HEAD:
                return t -> true;
            case MTP_FILE: {
                Predicate<String> reads = ModelSpecBuilder.mtpDraftReads(kv, catalog);
                return t -> reads.test(t.name());
            }
            default:
                return draftModelPlacement(kv, catalog, memory, tensors, draftPlacement)::onDevice;
        }
    }

    /**
     * What a separate draft model's own GPU load places: the default placement without draft placement settings,
     * otherwise the load of them — the draft configuration's layer count ({@code -ngld},
     * {@link LayerOffload#requested}), its tensor overrides and expert placement ({@code -otd}, {@code -cmoed},
     * {@code -ncmoed}; {@link HostPlacement#rulePlacement}) — with a layer split's host cut fixed, as
     * {@code GGUFModelGPU} resolves them.
     */
    private static WeightPlacement draftModelPlacement(GgufKv kv, TensorCatalog catalog, MemoryConfig memory,
                                                       List<GGUFTensorInfo> tensors, EngineConfig draftPlacement) {
        ModelSpec spec = ModelSpecBuilder.build(kv, catalog, memory);
        if (draftPlacement == null) {
            return WeightPlacement.of(spec, false);
        }
        LayerOffload offload = LayerOffload.requested(spec, draftPlacement, tensors);
        Predicate<String> fixedHost = offload == null ? n -> false : offload::hostTensor;
        return HostPlacement.rulePlacement(draftPlacement, spec, false, fixedHost);
    }

    public static ContextFitter.Companion draftCompanion(EngineConfig cfg, ModelSpec target, boolean hostTarget,
                                                         DraftWeightBytes weightBytes) {
        return draftCompanion(cfg, target, hostTarget, null, weightBytes);
    }

    public static ContextFitter.Companion draftCompanion(EngineConfig cfg, ModelSpec target, boolean hostTarget,
                                                         Path targetPath, DraftWeightBytes weightBytes) {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(weightBytes, "weightBytes");
        String path = cfg.draftModelPath();
        if (path == null || cfg.mtp() || target.memoryKind() == MemoryKind.NONE
                || !LLMModelFactory.hasTokenLoop(target.supportClass())) {
            return ContextFitter.Companion.NONE;
        }
        ModelSource dp = null;
        try {
            ModelSources.requireOpenable(path);
            dp = ModelSources.open(path, cfg.sourceOptions(), targetPath, cfg.activationQuant());
            ContextFitter.Companion c = companionOf(dp, path, targetPath, cfg, target, hostTarget, weightBytes);
            if (c != ContextFitter.Companion.NONE) {
                LOGGER.info("context fit reserves room for the drafter {}: {}", path, c);
            }
            return c;
        } catch (IOException | RuntimeException e) {
            LOGGER.warn("cannot measure the memory of the drafter {}: {}; the context is fitted without it", path,
                    LLMModelFactory.describe(e));
            return ContextFitter.Companion.NONE;
        } finally {
            if (dp != null) {
                Throwable closeFailure = LLMModelFactory.closeStep(dp::close, null);
                if (closeFailure != null) {
                    LOGGER.warn("closing the drafter {} after measuring it failed: {}", path, LLMModelFactory.describe(closeFailure));
                }
            }
        }
    }

    public static ContextFitter.Companion twoLevelCompanion(EngineConfig cfg, ModelSource targetSource,
                                                            boolean w12Target) {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(targetSource, "targetSource");
        SpeculativeOptions options = cfg.speculativeOptions();
        if (!options.twoLevelVerify() || w12Target) {
            return ContextFitter.Companion.NONE;
        }
        long bytes = QCopySource.plannedBytes(targetSource, options.tlvBits());
        if (bytes <= 0L) {
            return ContextFitter.Companion.NONE;
        }
        ContextFitter.Companion c = ContextFitter.Companion.of("two-level verify quantized target view", bytes, 0L);
        LOGGER.info("context fit reserves room for the quantized target view: {}", c);
        return c;
    }

    private static ContextFitter.Companion companionOf(ModelSource dp, String path, Path targetPath, EngineConfig cfg,
                                                        ModelSpec target, boolean hostTarget,
                                                        DraftWeightBytes weightBytes) throws IOException {
        Arch arch = draftArch(dp, path);
        GgufKv kv = GgufKv.of(dp);
        TensorCatalog catalog = TensorCatalog.of(dp);
        DraftRoute route = draftRoute(arch, kv, catalog, target, cfg.draftType(), path);
        if (route == DraftRoute.MODEL && draftOffTarget(cfg, hostTarget)) {
            return ContextFitter.Companion.NONE;
        }
        if (route == DraftRoute.DFLASH) {
            ModelSource derived = derivedFitSource(dp, cfg, catalog);
            if (derived != dp) {
                dp = derived;
                kv = GgufKv.of(dp);
                catalog = TensorCatalog.of(dp);
            }
            DflashSpec spec = DflashDrafter.readSpec(dp);
            KvDType kvType = cfg.memory().draftTypeK();
            if (hostTarget) {
                return ContextFitter.Companion.withCacheBytes("dflash drafter", 0L, 0L, "the dflash KV cache",
                        (nCtx, ubatch) -> DflashDrafter.cacheBytes(spec, nCtx, kvType));
            }
            if (!cfg.speculativeOptions().dflashOnDevice() || cfg.splitMode() == EngineConfig.SplitMode.LAYER
                    || DflashDevice.refusal(spec, kvType) != null) {
                return ContextFitter.Companion.NONE;
            }
            final TensorCatalog fitCatalog = catalog;
            long weights = weightBytes.of(dp, deviceTensors(route, kv, fitCatalog, cfg.memory(), dp.getTensors(),
                    null));
            return ContextFitter.Companion.withCacheBytes("dflash drafter", weights, 0L, "the dflash device KV cache",
                    (nCtx, ubatch) -> DflashDevice.deviceBytes(spec, fitCatalog, nCtx, kvType, cfg.draftPMin()));
        }
        MemoryConfig memory = cfg.memory().forDraftContext();
        long weights = weightBytes.of(dp, deviceTensors(route, kv, catalog, memory, dp.getTensors(),
                route == DraftRoute.MODEL && cfg.hasDraftPlacement() ? draftPlacementOf(cfg) : null));
        if (route == DraftRoute.MTP_FILE) {
            ModelSpec spec = ModelSpecBuilder.buildMtpDraft(kv, catalog, memory, target);
            return ContextFitter.Companion.withCacheBytes("MTP draft file " + arch.ggufName(), weights,
                    0L, "its NextN blocks", (nCtx, ubatch) -> MemoryPlanner.nextnBytes(spec, memory, nCtx, ubatch));
        }
        ModelSpec spec = ModelSpecBuilder.build(kv, catalog, memory);
        String name = arch.ggufName() + " drafter";
        switch (arch) {
            case EAGLE:
            case EAGLE3:
                return ContextFitter.Companion.withCaches(name, weights, 0L, spec, memory);
            case GEMMA4_ASSISTANT:
                return ContextFitter.Companion.of(name, weights, 0L);
            default:
                return ContextFitter.Companion.withCaches("draft model " + arch.ggufName(), weights,
                        scratchBytesPerToken(spec), spec, memory).withStepUbatch(cfg.memory().draftUbatch());
        }
    }

    /**
     * The derived draft source a memory fit can plan without the target's weights: a draft that carries its own
     * head, or options that only restate the drafter's own matrices. The file as it is when the drafter would
     * borrow the target's head, which a fit cannot reach; the bind derives it exactly.
     */
    private static ModelSource derivedFitSource(ModelSource dp, EngineConfig cfg, TensorCatalog catalog)
            throws IOException {
        DraftHeadOptions head = cfg.speculativeOptions().draftHead();
        if (head == null || !head.enabled()) {
            return dp;
        }
        if ((head.vocabSize() > 0 || head.headPrec() != null)
                && !catalog.has(TensorNames.weight(TensorNames.OUTPUT))) {
            return dp;
        }
        return DerivedDraftSource.wrap(dp, null, head);
    }

    /**
     * True when a separate draft model drafts off the target's device placement ({@code -devd}): a device list that
     * is not {@code none} beside a host target, or any list beside a GPU target, opens the draft on devices of its
     * own, whose budget the target's context fit does not share.
     */
    private static boolean draftOffTarget(EngineConfig cfg, boolean hostTarget) {
        List<String> devices = cfg.draftDevices();
        if (devices.isEmpty()) {
            return false;
        }
        return !hostTarget || !isNoDevice(devices);
    }

    /**
     * The draft model's placement settings as the configuration its own load resolves them from: the draft knobs on
     * an otherwise-default configuration, with the draft path kept, as the builder refuses placement without one.
     */
    private static EngineConfig draftPlacementOf(EngineConfig cfg) {
        return EngineConfig.builder()
                .draftModelPath(cfg.draftModelPath())
                .gpuLayers(cfg.draftGpuLayers())
                .tensorOverrides(cfg.draftTensorOverrides())
                .cpuMoe(cfg.draftCpuMoe())
                .nCpuMoe(cfg.draftNCpuMoe())
                .build();
    }

    static long scratchBytesPerToken(ModelSpec spec) {
        ScratchWidths w = ScratchWidths.of(spec);
        long[] floatWidths = {w.stream(), w.stream(), w.stream(), w.projection(), w.projection(), w.projection(),
                w.projection(), w.gateUp(), w.ffnOut(), w.logits(), w.input()};
        long elements = ScratchWidths.TOKEN_ID_INTS + ScratchWidths.POS_INTS;
        for (long width : floatWidths) {
            elements = Math.addExact(elements, Math.max(1L, width));
        }
        return Math.multiplyExact(elements, (long) Float.BYTES);
    }

    /**
     * The architecture of a draft file.
     *
     * @throws IllegalArgumentException when the file names no architecture
     * @throws UnsupportedOperationException when it is a drafter architecture no drafter here runs
     */
    static Arch draftArch(ModelSource dp, String path) {
        String name = dp.getString("general.architecture");
        if (name == null) {
            throw new IllegalArgumentException("draft model " + path + " has no general.architecture");
        }
        Arch arch = Arch.fromGgufName(name);
        if (arch == Arch.UNKNOWN) {
            throw new UnsupportedOperationException("draft model " + path + " has the unknown architecture " + name);
        }
        return arch;
    }

    /**
     * Memory settings of a drafter with caches of its own: the user's settings as a draft context runs with them
     * ({@link MemoryConfig#forDraftContext()}) with the target plan's context and the plan ubatch the draft ubatch
     * replaces when one is set.
     */
    private static MemoryConfig draftMemory(EngineModel target, EngineConfig cfg) {
        MemoryPlan plan = target.forwardBackend().memory().plan();
        return cfg.memory().forDraftContext().toBuilder()
                .nCtx(plan.nCtx())
                .ubatch(cfg.memory().draftUbatchOr(plan.ubatch()))
                .build();
    }

    /**
     * The NextN/MTP head of {@code target} over its own tensors. A GPU target opened without MTP left its NextN blocks
     * off the device, as llama.cpp skips them without {@code load_mtp} (llama-model.cpp:3915-3917); they are placed
     * now, before the head reads them ({@link GGUFModelGPU#ensureNextnWeights()}).
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when a GPU target's NextN blocks do not fit
     *                                                                    the VRAM it has left
     */
    private static MtpHead targetHead(EngineModel target) {
        if (target instanceof GGUFModelGPU) {
            GGUFModelGPU g = (GGUFModelGPU) target;
            g.ensureNextnWeights();
            return MtpHead.gpu(g.nextnContext(), g.blockRegistry(), null);
        }
        if (target instanceof GGUFModel) {
            GGUFModel c = (GGUFModel) target;
            return MtpHead.cpu(c.forwardContext(), c.blockRegistry(), null);
        }
        throw notFactoryModel(target);
    }

    private static UnsupportedOperationException notFactoryModel(EngineModel target) {
        return new UnsupportedOperationException("drafters bind to models opened through LLMModelFactory, not "
                + target.getClass().getName());
    }

    /**
     * The head of a draft file (eagle, eagle3, gemma4-assistant, an MTP draft file): on a GPU target the file's
     * tensors except {@code deferred} uploaded to the target's device with the target's kernel type policy and their
     * weight kernels prepared ({@link WeightKernelPrewarm}); on a CPU target the file mapped on the host, where
     * tensors the drafter never reads take no memory. The weights are appended to {@code out}; the head does not own
     * them.
     *
     * @param actQuant how the draft file's activation quantization is applied (the session's
     *                 {@link EngineConfig#activationQuant()})
     * @param deferred tensors the drafter never reads, left off the device
     */
    private static MtpHead draftHead(EngineModel target, ModelSource dp, GgufKv kv, TensorCatalog catalog,
                                     ModelSpec spec, ActivationQuant actQuant, Predicate<GGUFTensorInfo> deferred,
                                     List<AutoCloseable> out) throws IOException {
        if (target instanceof GGUFModelGPU) {
            GGUFModelGPU g = (GGUFModelGPU) target;
            GpuWeightUploader uploader = uploadDraft(g, dp, kv, catalog, spec.arch().ggufName(), actQuant, deferred,
                    out);
            return MtpHead.gpu(g.forwardContext(), g.blockRegistry(), uploader);
        }
        if (target instanceof GGUFModel) {
            GGUFModel c = (GGUFModel) target;
            CpuWeights weights = CpuWeights.open(dp, catalog, InputTransformLoader.load(dp, kv, catalog, actQuant));
            out.add(weights);
            return MtpHead.cpu(c.forwardContext(), c.blockRegistry(), weights);
        }
        throw notFactoryModel(target);
    }

    /**
     * The tensors of the draft file {@code dp} except {@code deferred} uploaded to the GPU target's device with the
     * target's kernel type policy and the draft's activation transforms, their weight kernels prepared
     * ({@link WeightKernelPrewarm}). The store and the uploader are appended to {@code out}.
     */
    private static GpuWeightUploader uploadDraft(GGUFModelGPU g, ModelSource dp, GgufKv kv, TensorCatalog catalog,
                                                 String name, ActivationQuant actQuant,
                                                 Predicate<GGUFTensorInfo> deferred, List<AutoCloseable> out)
            throws IOException {
        InputTransformLoader.Result transforms = InputTransformLoader.load(dp, kv, catalog, actQuant);
        GpuForwardContext ctx = g.forwardContext();
        WeightStore store = dp.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        out.add(store);
        GpuWeightUploader uploader = new GpuWeightUploader(g.backend(), ctx.queue(), dp.getTensors(), store, name,
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(ctx.kernels()))
                        .transforms(transforms)
                        .build());
        out.add(uploader);
        uploader.uploadResident(deferred);
        WeightKernelPrewarm.prewarm(uploader.plans(deferred.negate()), ctx.ops(), ctx.kernels());
        LOGGER.info("{} draft weights on {}: {} MiB resident", name, g.backend().backendType(),
                uploader.residentBytes() >> 20);
        return uploader;
    }

    /** A separate generative draft model on the target's backend (GPU: the target's device; CPU: the CPU engine). */
    private static Drafter modelDrafter(EngineModel target, String path, EngineConfig cfg, MemoryConfig memory)
            throws IOException {
        PagedSequences paged = target.forwardBackend().memory() == null ? null
                : target.forwardBackend().memory().pagedSequences();
        MemoryConfig draftMemory = paged == null ? memory
                : memory.toBuilder().parallel(paged.maxSequences()).kvPaged(true).build();
        EngineModel draft = openDraftModel(target, path, cfg, draftMemory);
        ModelDrafter d;
        try {
            d = new ModelDrafter(draft, cfg.draftPMin(), cfg.speculativeOptions().heterogeneousVocab(),
                    cfg.speculativeOptions().pSplit());
        } catch (RuntimeException e) {
            LLMModelFactory.closeStep(draft, e);
            throw e;
        }
        try {
            d.bindTarget(target);
        } catch (RuntimeException e) {
            LLMModelFactory.closeStep(d, e);
            throw e;
        }
        return new Drafter(d, "draft model " + draft.architecture(), Integer.MAX_VALUE);
    }

    /**
     * Opens the draft model {@code path} with the draft memory {@code memory} on the target's backend, or on GPUs of
     * its own when the draft placement names them ({@code -devd}): the lone {@code none} keeps the draft on the
     * host, an explicit list opens it on a GPU of its own — one selector the GPU of a one-device load, two or more
     * a layer split — as llama.cpp's draft devices replace the main ones only when set
     * (common/speculative.cpp:2631-2700); unset devices keep the draft beside the target's weights. The layer count,
     * tensor overrides and expert placement of the draft load ({@code -ngld}, {@code -otd}, {@code -cmoed},
     * {@code -ncmoed}) ride it as the target's ride its own.
     */
    private static EngineModel openDraftModel(EngineModel target, String path, EngineConfig cfg, MemoryConfig memory)
            throws IOException {
        ComputeBackend backend = LLMModelFactory.computeBackend(target);
        List<String> devices = cfg.draftDevices();
        if (isNoDevice(devices)) {
            return new GGUFModel(path, draftConfig(cfg, memory, EngineConfig.BackendMode.CPU, null));
        }
        if (!devices.isEmpty()) {
            return LLMModelFactory.openGpu(path, draftConfig(cfg, memory, draftDeviceMode(cfg), devices), null);
        }
        EngineConfig.BackendMode mode = backend == null ? EngineConfig.BackendMode.CPU
                : cfg.backend().usesGpu() ? cfg.backend() : EngineConfig.BackendMode.AUTO;
        EngineConfig draftCfg = draftConfig(cfg, memory, mode, null);
        return backend == null ? new GGUFModel(path, draftCfg) : new GGUFModelGPU(path, backend, draftCfg);
    }

    /** The draft model's configuration: the target's streaming and source settings with the draft's placement. */
    private static EngineConfig draftConfig(EngineConfig cfg, MemoryConfig memory, EngineConfig.BackendMode mode,
                                            List<String> devices) {
        EngineConfig.Builder b = EngineConfig.builder()
                .backend(mode)
                .memory(memory)
                .samplerDefaults(cfg.samplerDefaults())
                .ramCacheMb(cfg.ramCacheMb())
                .vramDepth(cfg.vramDepth())
                .sourceOptions(cfg.sourceOptions())
                .activationQuant(cfg.activationQuant())
                .gpuLayers(cfg.draftGpuLayers())
                .tensorOverrides(cfg.draftTensorOverrides())
                .cpuMoe(cfg.draftCpuMoe())
                .nCpuMoe(cfg.draftNCpuMoe());
        if (devices == null) {
            return b.build();
        }
        if (devices.size() == 1) {
            String selector = devices.get(0);
            if (EngineConfig.isDeviceIndex(selector)) {
                b.gpuDevices(Collections.singletonList(selector));
            } else {
                b.deviceFilter(selector);
            }
            return b.build();
        }
        return b.gpuDevices(devices).splitMode(EngineConfig.SplitMode.LAYER).build();
    }

    /** The backend mode of a draft load on the GPUs {@code -devd} names: the configured GPU backend, else auto. */
    private static EngineConfig.BackendMode draftDeviceMode(EngineConfig cfg) {
        return cfg.backend().usesGpu() ? cfg.backend() : EngineConfig.BackendMode.AUTO;
    }

    /** True for the lone {@link EngineConfig#DRAFT_NO_DEVICE} entry that keeps the draft model on the host. */
    private static boolean isNoDevice(List<String> devices) {
        return devices.size() == 1 && devices.get(0).equalsIgnoreCase(EngineConfig.DRAFT_NO_DEVICE);
    }

    public static EngineModel openBatchDraftModel(EngineModel target, EngineConfig cfg) throws IOException {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(cfg, "cfg");
        String path = Objects.requireNonNull(cfg.draftModelPath(), "the configuration names no draft model");
        PagedSequences paged = target.forwardBackend().memory() == null ? null
                : target.forwardBackend().memory().pagedSequences();
        if (paged == null) {
            throw new UnsupportedOperationException("a batch draft model drafts for a target over paged memory; "
                    + target.name() + " has none");
        }
        ModelSources.requireOpenable(path);
        DraftRoute route;
        Arch arch;
        try (ModelSource dp = ModelSources.open(path, cfg.sourceOptions(), null, cfg.activationQuant())) {
            arch = draftArch(dp, path);
            route = draftRoute(arch, GgufKv.of(dp), TensorCatalog.of(dp), target.spec(), cfg.draftType(), path);
        }
        if (route != DraftRoute.MODEL) {
            if (cfg.speculativeOptions().heterogeneousVocab()) {
                throw new IllegalArgumentException(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB + " needs a separate"
                        + " draft model (draft-simple), but " + path + " is a " + arch.ggufName() + " drafter");
            }
            throw new UnsupportedOperationException(EngineConfig.PROP_DRAFT + "=" + path + ": the batch engine drafts"
                    + " with a separate draft model, the n-gram and the suffix drafters; the " + arch.ggufName() + " "
                    + (route == DraftRoute.MTP_FILE ? "MTP draft file" : "drafter") + " runs single-sequence"
                    + " (LLMModelFactory.openSpeculative)");
        }
        MemoryConfig memory = draftMemory(target, cfg).toBuilder()
                .parallel(paged.maxSequences())
                .kvPaged(true)
                .build();
        return openDraftModel(target, path, cfg, memory);
    }
}
