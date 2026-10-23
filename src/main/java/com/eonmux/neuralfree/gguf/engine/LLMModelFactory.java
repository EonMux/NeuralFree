package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashDrafter;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashSpec;
import com.eonmux.neuralfree.gguf.quant.PrismHadamardLoader;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ScratchWidths;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * Opens GGUF models on the configured compute backend and binds speculative drafters to them.
 *
 * <p><b>Backend.</b> With a GPU backend mode ({@link EngineConfig#backend()} other than CPU) the model loads with
 * {@link GGUFModelGPU}, either on a caller-owned backend or on one it creates and owns:
 * {@code HWAcceleration.forMode(mode).createBackend(mode, KernelLibraryRequirement.of(KernelLibraries.LLM))}, so only
 * devices that can run the LLM kernel library qualify. Device selection reads the {@code neuralfree.device} name
 * filter; a {@link EngineConfig#deviceFilter()} is applied by setting that property for the duration of the backend
 * creation (and restoring it afterwards), under a lock shared by every factory load. When the GPU load fails for any
 * reason other than an unreadable file, the reason is logged and the model loads on the CPU engine
 * ({@link GGUFModel}) with the same configuration.
 *
 * <p><b>Speculative decoding.</b> {@link #openSpeculative} and {@link #speculate} bind exactly one drafter to a
 * target: the target's own NextN head with {@link EngineConfig#mtp()}, or the drafter of the draft file
 * ({@link #draftRoute}). With {@link EngineConfig.DraftType#AUTO} the drafter is chosen from the file as llama.cpp
 * chooses the speculative type of a draft GGUF when none is given (common/arg.cpp:566-572,
 * common/speculative.cpp:2398-2433), except that a file with NextN blocks that is not an MTP draft of the target's
 * model ({@link ModelSpecBuilder#isMtpDraftOf}) drafts as a separate draft model; {@code DraftType.MTP}
 * ({@code draft-mtp}) and {@code DraftType.MODEL} ({@code draft-simple}) force one route, as llama.cpp's
 * {@code --spec-type} does:
 * <ul>
 *   <li>{@code dflash}: {@link DflashDrafter} over the target's host weights; the draft length is clamped to its
 *       block ({@code DflashSpec.nDraftMax()}).</li>
 *   <li>{@code eagle3}, {@code gemma4-assistant}: {@link DraftModelLoader#fromFile} with an {@link MtpHead} on the
 *       target's runtime, the draft weights uploaded to the target's device (GPU) or mapped (CPU).</li>
 *   <li>MTP: {@link DraftModelLoader#nextn} over the target's NextN blocks; a GPU target opened without MTP places
 *       them on its device when the drafter attaches ({@link GGUFModelGPU#ensureNextnWeights()}).</li>
 *   <li>An MTP draft file of the target's model ({@link ModelSpecBuilder#isNextnDraft}: its last block holds
 *       {@code nextn.eh_proj}; {@link ModelSpecBuilder#isMtpDraftOf}), with or without its trunk blocks: a full
 *       model with NextN blocks, or the MTP blocks alone as written by {@code convert_hf_to_gguf.py --mtp}.
 *       {@link DraftModelLoader#nextn(EngineModel, ModelSpec, MtpHead, DraftModelLoader.Options)} runs that file's
 *       NextN blocks, embedding and head against the target's hidden rows ({@link ModelSpecBuilder#buildMtpDraft});
 *       the file's trunk is never run, uploaded or reserved ({@link #deviceTensors}).</li>
 *   <li>Any other generative model, NextN blocks or not: {@link ModelDrafter}, a separate draft model on the
 *       target's backend with a vocabulary compatible with the target's.</li>
 * </ul>
 * Every draft context, the NextN / MTP caches included, runs with {@link MemoryConfig#forDraftContext()}: F16 rows
 * (llama.cpp {@code --spec-draft-type-k/-v} default, common/common.h:348-349) and no recurrent-state snapshots
 * ({@code n_rs_seq = 0}). With several NextN blocks the heads chain and a draft holds at most one token per block, as
 * llama.cpp clamps the draft length of a chained MTP drafter (common/speculative.cpp:1523-1528). Each backend has one
 * hidden-state tap, so a target serves one drafter (or one pockettts reader) at a time ({@link HiddenTapSlot}). A
 * target whose configuration names a draft model fits its context beside that drafter's memory
 * ({@link #draftCompanion}), as llama.cpp fits the main and the draft context together.
 */
public final class LLMModelFactory {

    private static final Logger LOGGER = LogManager.getLogger(LLMModelFactory.class);

    /** Row type of the dflash drafter's host KV cache: the draft cache type ({@link MemoryConfig#DRAFT_KV_TYPE}). */
    static final KvDType DFLASH_KV_TYPE = MemoryConfig.DRAFT_KV_TYPE;

    /** Serializes loads that set the device-filter property, so concurrent loads cannot see each other's filter. */
    private static final Object DEVICE_PROPERTY_LOCK = new Object();

    private LLMModelFactory() {
    }

    // ================================================================== single models

    /**
     * Opens a model on the backend {@code cfg} selects, falling back to the CPU when the GPU load fails.
     *
     * @throws FileNotFoundException when {@code path} is not a readable file
     * @throws IOException           when the file cannot be read
     * @throws RuntimeException      when the CPU load fails too (the GPU failure is suppressed in it), or when
     *                               {@code cfg} selects the CPU and the CPU load fails
     */
    public static EngineModel open(String path, EngineConfig cfg) throws IOException {
        return open(path, cfg, null);
    }

    /**
     * Opens a model on {@code sharedBackend}, which stays owned by the caller, falling back to the CPU when the GPU
     * load fails. With a null backend this is {@link #open(String, EngineConfig)}. A device filter in {@code cfg} has
     * no effect on a backend the caller created. A configured draft file is read first to size the compressor
     * rollback snapshots its drafter needs ({@link MemoryConfig#rollbackSnapshots()}).
     *
     * @throws FileNotFoundException when {@code path} is not a readable file
     * @throws IOException           when the file cannot be read
     */
    public static EngineModel open(String path, EngineConfig cfg, ComputeBackend sharedBackend) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(cfg, "cfg");
        requireReadable(path);
        EngineConfig c = withDraftRollback(path, cfg);
        if (sharedBackend == null && !c.backend().usesGpu()) {
            return new GGUFModel(path, c);
        }
        try {
            return openGpu(path, c, sharedBackend);
        } catch (IOException e) {
            throw e;
        } catch (RuntimeException | LinkageError gpuFailure) {
            LOGGER.warn("GPU load of {} failed: {}; falling back to the CPU engine", path, describe(gpuFailure));
            try {
                return new GGUFModel(path, c.toBuilder().backend(EngineConfig.BackendMode.CPU).build());
            } catch (IOException | RuntimeException | Error cpuFailure) {
                cpuFailure.addSuppressed(gpuFailure);
                throw cpuFailure;
            }
        }
    }

    /**
     * {@code cfg} with the compressor rollback snapshots ({@link MemoryConfig#rollbackSnapshots()}, llama.cpp
     * {@code n_rs_seq}) its draft file needs, pinned before the target is planned. {@link EngineConfig} gives a draft
     * file under {@link EngineConfig.DraftType#AUTO} the draft max; here the file's {@link #draftRoute} decides, as
     * llama.cpp picks the speculative type from the file before sizing the target context
     * (common/speculative.cpp:2398-2433, common/common.h:401-407): an MTP, eagle3, gemma4-assistant or dflash
     * drafter keeps the draft max and a separate draft model gets 0. A draft file that is missing, cannot be read or
     * binds no drafter gets 0 and is logged, and the target still opens: attaching the drafter reports the failure,
     * as the context fit does ({@link #draftCompanion}). Other configurations are returned unchanged.
     */
    private static EngineConfig withDraftRollback(String targetPath, EngineConfig cfg) {
        String draftPath = cfg.draftModelPath();
        if (draftPath == null || cfg.mtp() || cfg.draftType() != EngineConfig.DraftType.AUTO) {
            return cfg;
        }
        DraftRoute route = null;
        try {
            requireReadable(draftPath);
            GGUFFileParser dp = new GGUFFileParser(draftPath);
            try {
                Arch arch = draftArch(dp, draftPath);
                GgufKv kv = GgufKv.of(dp);
                TensorCatalog tc = TensorCatalog.of(dp);
                ModelSpec target = ModelSpecBuilder.isNextnDraft(kv, tc) ? inspect(targetPath, cfg.memory()) : null;
                route = draftRoute(arch, kv, tc, target, cfg.draftType(), draftPath);
            } finally {
                dp.close();
            }
        } catch (IOException | RuntimeException refused) {
            LOGGER.warn("cannot resolve how the draft model {} drafts for {} ({}); the target keeps no compressor"
                    + " rollback snapshots", draftPath, targetPath, describe(refused));
        }
        int rollback = route == null || route == DraftRoute.MODEL ? 0 : cfg.draftMax();
        return cfg.toBuilder().memory(cfg.memory().pinRollback(rollback)).build();
    }

    private static GGUFModelGPU openGpu(String path, EngineConfig cfg, ComputeBackend sharedBackend)
            throws IOException {
        String filter = cfg.deviceFilter();
        if (sharedBackend != null || filter == null) {
            return new GGUFModelGPU(path, sharedBackend, cfg);
        }
        synchronized (DEVICE_PROPERTY_LOCK) {
            String previous = System.getProperty(EngineConfig.PROP_DEVICE);
            System.setProperty(EngineConfig.PROP_DEVICE, filter);
            try {
                return new GGUFModelGPU(path, null, cfg);
            } finally {
                if (previous == null) {
                    System.clearProperty(EngineConfig.PROP_DEVICE);
                } else {
                    System.setProperty(EngineConfig.PROP_DEVICE, previous);
                }
            }
        }
    }

    private static void requireReadable(String path) throws FileNotFoundException {
        if (!Files.isRegularFile(Paths.get(path)) || !Files.isReadable(Paths.get(path))) {
            throw new FileNotFoundException("model file not found or not readable: " + path);
        }
    }

    /** The exception's class and message, with the root cause when it differs. */
    static String describe(Throwable t) {
        StringBuilder sb = new StringBuilder(t.getClass().getSimpleName());
        if (t.getMessage() != null) {
            sb.append(": ").append(t.getMessage());
        }
        Throwable root = t;
        while (root.getCause() != null && root.getCause() != root) {
            root = root.getCause();
        }
        if (root != t) {
            sb.append(" (cause ").append(root.getClass().getSimpleName());
            if (root.getMessage() != null) {
                sb.append(": ").append(root.getMessage());
            }
            sb.append(')');
        }
        return sb.toString();
    }

    /**
     * The resolved specification of a GGUF file from its metadata and tensor shapes, without loading tensor data;
     * lets callers choose settings by architecture and support class before {@link #open}.
     *
     * @throws FileNotFoundException when {@code path} is not a readable file
     * @throws RuntimeException      when the architecture cannot be resolved
     */
    public static ModelSpec inspect(String path, MemoryConfig memory) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(memory, "memory");
        requireReadable(path);
        GGUFFileParser p = new GGUFFileParser(path);
        try {
            return ModelSpecBuilder.build(GgufKv.of(p), TensorCatalog.of(p), memory);
        } finally {
            p.close();
        }
    }

    // ================================================================== model accessors

    /**
     * The configuration {@code model} runs with: for the factory's model classes the effective one (its memory
     * settings carry the planned ubatch), otherwise {@code requested}.
     */
    public static EngineConfig effectiveConfig(EngineModel model, EngineConfig requested) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(requested, "requested");
        if (model instanceof GGUFModelGPU) {
            return ((GGUFModelGPU) model).engineConfig();
        }
        if (model instanceof GGUFModel) {
            return ((GGUFModel) model).engineConfig();
        }
        return requested;
    }

    /** The memory plan of {@code model}, or null when it runs without memory. */
    public static MemoryPlan memoryPlan(EngineModel model) {
        Objects.requireNonNull(model, "model");
        if (model instanceof GGUFModelGPU) {
            return ((GGUFModelGPU) model).memoryPlan();
        }
        if (model instanceof GGUFModel) {
            return ((GGUFModel) model).memoryPlan();
        }
        ModelMemory m = model.forwardBackend().memory();
        return m == null ? null : m.plan();
    }

    /** The compute backend of a GPU model, or null for a CPU model. */
    public static ComputeBackend computeBackend(EngineModel model) {
        Objects.requireNonNull(model, "model");
        return model instanceof GGUFModelGPU ? ((GGUFModelGPU) model).backend() : null;
    }

    /**
     * Host access to the tensors of a model opened by this factory (drafters and heads that read target tensors).
     *
     * @throws UnsupportedOperationException for a model class this factory does not create
     */
    static CpuWeights hostWeights(EngineModel model) throws IOException {
        if (model instanceof GGUFModelGPU) {
            return ((GGUFModelGPU) model).cpuWeights();
        }
        if (model instanceof GGUFModel) {
            CpuForwardContext ctx = ((GGUFModel) model).forwardContext();
            if (ctx == null) {
                throw new UnsupportedOperationException(model.architecture() + " has no host forward context");
            }
            return ctx.weights();
        }
        throw new UnsupportedOperationException("host weights of " + model.getClass().getName()
                + " are not reachable; open the model through LLMModelFactory");
    }

    // ================================================================== speculative decoding

    /**
     * Opens {@code targetPath} with {@link #open(String, EngineConfig)} and binds the drafter of {@code draftPath}
     * to it; the session owns and closes the target.
     *
     * @param draftPath the draft GGUF, or null to use {@code cfg}'s draft model path or its MTP setting
     * @throws IllegalArgumentException      when no drafter is configured, both a draft file and MTP are, or the
     *                                       drafter is incompatible with the target
     * @throws UnsupportedOperationException when the target cannot be speculated on or the draft architecture has no
     *                                       drafter
     * @throws IllegalStateException         when the target's hidden-state tap is in use
     */
    public static SpeculativeSession openSpeculative(String targetPath, String draftPath, EngineConfig cfg)
            throws IOException {
        Objects.requireNonNull(targetPath, "targetPath");
        Objects.requireNonNull(cfg, "cfg");
        EngineConfig c = draftPath == null ? cfg : cfg.toBuilder().draftModelPath(draftPath).mtp(false).build();
        requireOneDrafter(c);
        EngineModel target = open(targetPath, c);
        try {
            return attach(target, c, true);
        } catch (IOException | RuntimeException | Error e) {
            try {
                target.close();
            } catch (IOException | RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /**
     * Binds the drafter {@code cfg} configures ({@link EngineConfig#draftModelPath()} or {@link EngineConfig#mtp()})
     * to an open target; the target stays owned by the caller and must not run forward passes outside the session
     * until the session is closed.
     *
     * @throws IllegalArgumentException      when no drafter is configured, both a draft file and MTP are, or the
     *                                       drafter is incompatible with the target
     * @throws UnsupportedOperationException when the target cannot be speculated on or the draft architecture has no
     *                                       drafter
     * @throws IllegalStateException         when the target's hidden-state tap is in use
     */
    public static SpeculativeSession speculate(EngineModel target, EngineConfig cfg) throws IOException {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(cfg, "cfg");
        requireOneDrafter(cfg);
        return attach(target, cfg, false);
    }

    private static void requireOneDrafter(EngineConfig cfg) {
        if (cfg.draftModelPath() != null && cfg.mtp()) {
            throw new IllegalArgumentException("both a draft model (" + cfg.draftModelPath() + ") and the MTP head are"
                    + " configured; a target binds one drafter");
        }
        if (!cfg.speculative()) {
            throw new IllegalArgumentException("no drafter is configured: set a draft model ("
                    + EngineConfig.PROP_DRAFT + ") or the MTP head (" + EngineConfig.PROP_MTP + ")");
        }
    }

    private static SpeculativeSession attach(EngineModel target, EngineConfig cfg, boolean ownsTarget)
            throws IOException {
        requireSpeculatable(target);
        ForwardBackend fb = target.forwardBackend();
        HiddenTapSlot.Lease lease = HiddenTapSlot.acquire(fb, "a speculative drafter");
        List<AutoCloseable> resources = new ArrayList<>();
        Drafter built = null;
        try {
            built = cfg.mtp() ? nextn(target, cfg) : fromFile(target, cfg, resources);
            final DraftModel drafter = built.drafter;
            int draftMax = Math.min(cfg.draftMax(), built.maxDraft);
            int draftMin = Math.min(cfg.draftMin(), draftMax);
            GGUFTokenizer tok = target.tokenizer();
            ReasoningFilter rf = ReasoningFilter.forChatTemplate(tok.chatTemplate()).withTagTokens(tok::getTokenId);
            SpeculativeSession session = new SpeculativeSession(target, ownsTarget, drafter, built.kind, resources,
                    lease, sink -> SpeculativeDecoder.builder(fb, drafter, tok)
                            .reasoning(rf)
                            .draftMax(draftMax)
                            .draftMin(draftMin)
                            .statsSink(sink)
                            .build());
            LOGGER.info("speculative decoding on {}: {} drafter, draft max {} (requested {}), min {}, p_min {}",
                    target.name(), built.kind, draftMax, cfg.draftMax(), draftMin, cfg.draftPMin());
            MemoryPlan plan = fb.memory().plan();
            if (plan.hasCompressed() && plan.rollbackSnapshots() < draftMax) {
                LOGGER.info("{}: the DSV4 compressor state keeps {} rollback snapshots, so a round drafts at most {}"
                                + " tokens{}", target.name(), plan.rollbackSnapshots(), plan.rollbackSnapshots(),
                        plan.rollbackSnapshots() == 0 ? " (speculation is off: the target was planned without"
                                + " snapshots, as for a separate draft model in llama.cpp)" : "");
            }
            return session;
        } catch (IOException | RuntimeException | Error e) {
            Throwable first = e;
            if (built != null) {
                first = closeStep(built.drafter, first);
            }
            for (int i = resources.size() - 1; i >= 0; i--) {
                first = closeStep(resources.get(i), first);
            }
            closeStep(lease, first);
            throw e;
        }
    }

    /**
     * Rejects targets the speculative decoder cannot drive: models without a decoder-only token loop and backends
     * without memory.
     */
    private static void requireSpeculatable(EngineModel target) {
        SupportClass support = target.supportClass();
        if (!hasTokenLoop(support)) {
            throw new UnsupportedOperationException(target.architecture() + " (" + support
                    + ") has no decoder-only token loop to speculate on");
        }
        if (target.forwardBackend().memory() == null) {
            throw new UnsupportedOperationException("speculative decoding needs model memory; " + target.architecture()
                    + " runs without a cache");
        }
    }

    /** Whether models of {@code support} run a decoder-only token loop a drafter can speculate on. */
    private static boolean hasTokenLoop(SupportClass support) {
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
            case DIFF:
            case ENC_DEC:
            case DRAFT:
                return false;
            default:
                return true;
        }
    }

    /** A bound drafter with its kind and its own draft-length limit. */
    private static final class Drafter {
        final DraftModel drafter;
        final String kind;
        final int maxDraft;

        Drafter(DraftModel drafter, String kind, int maxDraft) {
            this.drafter = drafter;
            this.kind = kind;
            this.maxDraft = maxDraft;
        }
    }

    private static Drafter nextn(EngineModel target, EngineConfig cfg) {
        MtpHead head = targetHead(target);
        DraftModel d = DraftModelLoader.nextn(target, head, DraftModelLoader.Options.of(cfg.draftPMin()));
        return new Drafter(d, "mtp", mtpDraftMax(target.spec()));
    }

    /**
     * The draft-length limit of an MTP drafter over the NextN blocks of {@code mtp}: their count when there are
     * several, whose heads chain one block per drafted token (llama.cpp clamps {@code n_max} to
     * {@code n_layer_nextn} when {@code chain_heads}, common/speculative.cpp:1523-1528); none for a single head.
     */
    private static int mtpDraftMax(ModelSpec mtp) {
        int blocks = mtp.nextnLayers().size();
        return blocks > 1 ? blocks : Integer.MAX_VALUE;
    }

    /** The drafter of {@code cfg.draftModelPath()}; resources the drafter does not own are appended to {@code out}. */
    private static Drafter fromFile(EngineModel target, EngineConfig cfg, List<AutoCloseable> out)
            throws IOException {
        String path = cfg.draftModelPath();
        requireReadable(path);
        MemoryConfig draftMemory = draftMemory(target, cfg);
        GGUFFileParser dp = new GGUFFileParser(path);
        Arch arch;
        DraftRoute route;
        try {
            arch = draftArch(dp, path);
            route = draftRoute(arch, GgufKv.of(dp), TensorCatalog.of(dp), target.spec(), cfg.draftType(), path);
        } catch (RuntimeException e) {
            closeStep(dp::close, e);
            throw e;
        }
        if (route == DraftRoute.MODEL) {
            dp.close();
            return modelDrafter(target, path, cfg, draftMemory);
        }
        out.add(dp::close);
        if (route == DraftRoute.MTP_FILE) {
            return mtpFileDrafter(target, dp, cfg, out);
        }
        if (route == DraftRoute.DFLASH) {
            DflashDrafter d = DflashDrafter.open(dp, hostWeights(target), DflashDrafter.Options.of(
                    cfg.draftMin(), cfg.draftPMin(), DFLASH_KV_TYPE, cfg.memory().kvRotation()));
            try {
                d.bindTarget(target);
            } catch (RuntimeException e) {
                closeStep(d, e);
                throw e;
            }
            return new Drafter(d, arch.ggufName(), d.spec().nDraftMax());
        }
        GgufKv kv = GgufKv.of(dp);
        TensorCatalog catalog = TensorCatalog.of(dp);
        ModelSpec spec = ModelSpecBuilder.build(kv, catalog, draftMemory);
        MtpHead head = draftHead(target, dp, kv, catalog, spec, t -> false, out);
        DraftModel d = DraftModelLoader.fromFile(target, spec, head, draftMemory,
                DraftModelLoader.Options.of(cfg.draftPMin()));
        return new Drafter(d, arch.ggufName(), Integer.MAX_VALUE);
    }

    /** How a draft file drafts for its target ({@link #draftRoute}). */
    enum DraftRoute {
        /** {@code dflash}: {@link DflashDrafter} over the target's host weights. */
        DFLASH,
        /** {@code eagle3}, {@code gemma4-assistant}: {@link DraftModelLoader#fromFile} with the file's head. */
        HEAD,
        /** An MTP draft of the target's model: its NextN / MTP blocks over the target's hidden rows. */
        MTP_FILE,
        /** A separate draft model ({@link ModelDrafter}). */
        MODEL
    }

    /**
     * How the draft file {@code path} drafts for {@code target} under {@code type}; the attach ({@link #speculate})
     * and the context fit ({@link #draftCompanion}) both follow it.
     * <ul>
     *   <li>{@link EngineConfig.DraftType#AUTO}: {@code dflash}, {@code eagle3} and {@code gemma4-assistant} by
     *       architecture; a file whose last block holds {@code nextn.eh_proj} ({@link ModelSpecBuilder#isNextnDraft},
     *       llama.cpp's detection, common/speculative.cpp:2414-2420) drafts with its NextN blocks when it is an MTP
     *       draft of the target's model ({@link ModelSpecBuilder#isMtpDraftOf}), and otherwise, logged, as a
     *       separate draft model, since llama.cpp's converters keep the NextN blocks of a full model by default
     *       (conversion/qwen.py:309-315, conversion/glm.py:119-121) and a smaller model of the family is an
     *       ordinary draft model; any other file is a separate draft model;</li>
     *   <li>{@link EngineConfig.DraftType#MTP} ({@code draft-mtp}): the NextN blocks of an MTP draft of the target's
     *       model, anything else refused;</li>
     *   <li>{@link EngineConfig.DraftType#MODEL} ({@code draft-simple}): a separate draft model, whether or not
     *       the file holds NextN blocks.</li>
     * </ul>
     *
     * @throws IllegalArgumentException when {@code type} cannot run the file: a drafter architecture under
     *                                  {@code draft-simple} or {@code draft-mtp}, a file holding only MTP blocks as
     *                                  a separate draft model, or a file that is not an MTP draft of the target's
     *                                  model under {@code draft-mtp} (or holding only MTP blocks under auto), naming
     *                                  the difference
     * @throws RuntimeException         when the metadata of a file with NextN blocks cannot be resolved
     */
    static DraftRoute draftRoute(Arch arch, GgufKv kv, TensorCatalog tc, ModelSpec target,
                                 EngineConfig.DraftType type, String path) {
        Objects.requireNonNull(arch, "arch");
        Objects.requireNonNull(type, "type");
        boolean drafterArch = arch == Arch.DFLASH || arch == Arch.EAGLE3 || arch == Arch.GEMMA4_ASSISTANT;
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

    /**
     * The MTP drafter of an MTP draft file ({@link ModelSpecBuilder#isNextnDraft}): its spec over the target's trunk
     * ({@link ModelSpecBuilder#buildMtpDraft}) with the draft-context settings of the target plan, and the weights it
     * reads ({@link ModelSpecBuilder#mtpDraftReads}: never the file's trunk) uploaded to the target's device or mapped
     * ({@link #draftHead}).
     */
    private static Drafter mtpFileDrafter(EngineModel target, GGUFFileParser dp, EngineConfig cfg,
                                          List<AutoCloseable> out) throws IOException {
        GgufKv kv = GgufKv.of(dp);
        TensorCatalog catalog = TensorCatalog.of(dp);
        ModelSpec spec = ModelSpecBuilder.buildMtpDraft(kv, catalog, mtpMemory(target), target.spec());
        Predicate<String> reads = ModelSpecBuilder.mtpDraftReads(spec);
        MtpHead head = draftHead(target, dp, kv, catalog, spec, t -> !reads.test(t.name()), out);
        DraftModel d = DraftModelLoader.nextn(target, spec, head, DraftModelLoader.Options.of(cfg.draftPMin()));
        return new Drafter(d, "mtp draft file " + spec.arch().ggufName(), mtpDraftMax(spec));
    }

    /**
     * Memory settings of an MTP drafter's spec and caches: the target plan's settings as a draft context runs with
     * them ({@link MemoryConfig#forDraftContext()}), with the plan's context and ubatch, as llama.cpp creates the MTP
     * context from the speculative parameters with the target's context (common/speculative.cpp:2606-2607,
     * 2658-2669).
     */
    private static MemoryConfig mtpMemory(EngineModel target) {
        MemoryPlan plan = target.forwardBackend().memory().plan();
        MemoryConfig settings = plan.config();
        if (settings == null) {
            throw new IllegalStateException(target.architecture() + ": the target memory plan records no settings to"
                    + " plan an MTP drafter with");
        }
        return settings.forDraftContext().toBuilder().nCtx(plan.nCtx()).ubatch(plan.ubatch()).build();
    }

    /**
     * Bytes a drafter file's weights take where the target runs: on a GPU target the device bytes of the tensors the
     * drafter places on the device; on a CPU target 0, as its drafter reads its weights from the mapped file.
     */
    @FunctionalInterface
    public interface DraftWeightBytes {
        /**
         * @param draft  the draft file
         * @param placed the tensors of {@code draft} its drafter places on the target's device, for the route the
         *               file drafts with for the target under the configured draft type ({@link #draftRoute},
         *               {@link #deviceTensors})
         * @throws IOException when the drafter's tensor data cannot be mapped
         */
        long of(GGUFFileParser draft, Predicate<GGUFTensorInfo> placed) throws IOException;
    }

    /**
     * The device tensors of a draft file drafting by {@code route}, the set {@link DraftWeightBytes} measures:
     * <ul>
     *   <li>{@code dflash}: none; it drafts over host weights;</li>
     *   <li>an MTP draft file: every tensor but its trunk blocks ({@link ModelSpecBuilder#mtpDraftReads});</li>
     *   <li>{@code eagle3}, {@code gemma4-assistant}: every tensor;</li>
     *   <li>a separate draft model: what its own GPU load places ({@link WeightPlacement}), which leaves its NextN
     *       blocks off the device, as a draft context loads no MTP layers, and the tensors only host heads read.</li>
     * </ul>
     *
     * @param memory the draft model's memory settings ({@link MemoryConfig#forDraftContext()}), for its spec
     */
    static Predicate<GGUFTensorInfo> deviceTensors(DraftRoute route, GgufKv kv, TensorCatalog catalog,
                                                   MemoryConfig memory) {
        switch (route) {
            case DFLASH:
                return t -> false;
            case HEAD:
                return t -> true;
            case MTP_FILE: {
                Predicate<String> reads = ModelSpecBuilder.mtpDraftReads(kv, catalog);
                return t -> reads.test(t.name());
            }
            default: {
                WeightPlacement placement = WeightPlacement.of(ModelSpecBuilder.build(kv, catalog, memory), false);
                return placement::onDevice;
            }
        }
    }

    /**
     * The memory of the drafter {@code cfg} names ({@link EngineConfig#draftModelPath()}) that a target fits its
     * context beside, measured at every candidate context as llama.cpp's {@code common_fit_params} measures its extra
     * model (common/common.cpp:1252-1280, common/fit.cpp:203-258):
     * <ul>
     *   <li>none without a draft model, or with {@link EngineConfig#mtp()}: the NextN / MTP draft memory is reserved
     *       through {@link MemoryConfig#nextnMemory()} and its weights are the target's;</li>
     *   <li>none for a target no drafter binds to: one without memory or without a decoder-only token loop;</li>
     *   <li>{@code dflash} drafts over host weights (the target's, and its own mapped from the file) with host caches:
     *       on a host target its KV cache at the target plan's context ({@link DflashDrafter#cacheBytes} in
     *       {@link #DFLASH_KV_TYPE}); on a device target nothing, as none of it is device memory;</li>
     *   <li>{@code eagle3}: its weights plus its own caches with {@link MemoryConfig#forDraftContext()} at the target
     *       plan's context and ubatch;</li>
     *   <li>{@code gemma4-assistant}: its weights; it reads the target's caches
     *       ({@code MemoryPlanner.planDrafter});</li>
     *   <li>an MTP draft file of the target's model ({@link #draftRoute}), with or without its trunk: the weights its
     *       drafter reads ({@link #deviceTensors}) and the NextN caches of its blocks
     *       ({@link MemoryPlanner#nextnBytes(ModelSpec, MemoryConfig, int, int)}, draft-context settings); its trunk
     *       takes neither caches nor scratch;</li>
     *   <li>a separate draft model ({@link #draftRoute}, NextN blocks or not): its weights, its caches as for eagle3
     *       and its forward scratch (the well-known slots of {@link ScratchWidths}) for every token of the target
     *       plan's ubatch.</li>
     * </ul>
     * The drafter is the one the attach binds: {@link #draftRoute} with the configured
     * {@link EngineConfig#draftType()} against {@code target}.
     * When the draft file cannot be measured, a warning names the reason and the target fits alone, as llama.cpp
     * does (common/fit.cpp:225-230); the drafter reports the problem when it binds.
     *
     * @param target      the target's resolved specification
     * @param hostTarget  true when the target runs on the host, whose budget also holds a drafter's host caches
     * @param weightBytes the bytes the drafter's weights take where the target runs, given the tensors the drafter's
     *                    route places on the device
     */
    public static ContextFitter.Companion draftCompanion(EngineConfig cfg, ModelSpec target, boolean hostTarget,
                                                         DraftWeightBytes weightBytes) {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(weightBytes, "weightBytes");
        String path = cfg.draftModelPath();
        if (path == null || cfg.mtp() || target.memoryKind() == MemoryKind.NONE
                || !hasTokenLoop(target.supportClass())) {
            return ContextFitter.Companion.NONE;
        }
        GGUFFileParser dp = null;
        try {
            requireReadable(path);
            dp = new GGUFFileParser(path);
            ContextFitter.Companion c = companionOf(dp, path, cfg, target, hostTarget, weightBytes);
            if (c != ContextFitter.Companion.NONE) {
                LOGGER.info("context fit reserves room for the drafter {}: {}", path, c);
            }
            return c;
        } catch (IOException | RuntimeException e) {
            LOGGER.warn("cannot measure the memory of the drafter {}: {}; the context is fitted without it", path,
                    describe(e));
            return ContextFitter.Companion.NONE;
        } finally {
            if (dp != null) {
                Throwable closeFailure = closeStep(dp::close, null);
                if (closeFailure != null) {
                    LOGGER.warn("closing the drafter {} after measuring it failed: {}", path, describe(closeFailure));
                }
            }
        }
    }

    private static ContextFitter.Companion companionOf(GGUFFileParser dp, String path, EngineConfig cfg,
                                                       ModelSpec target, boolean hostTarget,
                                                       DraftWeightBytes weightBytes) throws IOException {
        Arch arch = draftArch(dp, path);
        GgufKv kv = GgufKv.of(dp);
        TensorCatalog catalog = TensorCatalog.of(dp);
        DraftRoute route = draftRoute(arch, kv, catalog, target, cfg.draftType(), path);
        if (route == DraftRoute.DFLASH) {
            if (!hostTarget) {
                return ContextFitter.Companion.NONE;
            }
            DflashSpec spec = DflashDrafter.readSpec(dp);
            return ContextFitter.Companion.withCacheBytes("dflash drafter", 0L, 0L, "the dflash KV cache",
                    (nCtx, ubatch) -> DflashDrafter.cacheBytes(spec, nCtx, DFLASH_KV_TYPE));
        }
        MemoryConfig memory = cfg.memory().forDraftContext();
        long weights = weightBytes.of(dp, deviceTensors(route, kv, catalog, memory));
        if (route == DraftRoute.MTP_FILE) {
            ModelSpec spec = ModelSpecBuilder.buildMtpDraft(kv, catalog, memory, target);
            return ContextFitter.Companion.withCacheBytes("MTP draft file " + arch.ggufName(), weights,
                    0L, "its NextN blocks", (nCtx, ubatch) -> MemoryPlanner.nextnBytes(spec, memory, nCtx, ubatch));
        }
        ModelSpec spec = ModelSpecBuilder.build(kv, catalog, memory);
        String name = arch.ggufName() + " drafter";
        switch (arch) {
            case EAGLE3:
                return ContextFitter.Companion.withCaches(name, weights, 0L, spec, memory);
            case GEMMA4_ASSISTANT:
                return ContextFitter.Companion.of(name, weights, 0L);
            default:
                return ContextFitter.Companion.withCaches("draft model " + arch.ggufName(), weights,
                        scratchBytesPerToken(spec), spec, memory);
        }
    }

    /**
     * Bytes per ubatch token of a model's forward scratch: the well-known float slots ({@code X}, {@code BRANCH},
     * {@code NORMED}, {@code Q}, {@code K}, {@code V}, {@code ATTN_OUT}, {@code FFN_GATE_UP}, {@code FFN_OUT},
     * {@code LOGITS}, {@code INPUT_EMBD}) and the {@code TOKEN_IDS} and {@code POS} ints that {@code GpuScratch} and
     * {@code CpuScratch} size from {@link ScratchWidths}, four bytes each.
     */
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
    private static Arch draftArch(GGUFFileParser dp, String path) {
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
     * ({@link MemoryConfig#forDraftContext()}) with the context and ubatch of the target's memory plan.
     */
    private static MemoryConfig draftMemory(EngineModel target, EngineConfig cfg) {
        MemoryPlan plan = target.forwardBackend().memory().plan();
        return cfg.memory().forDraftContext().toBuilder()
                .nCtx(plan.nCtx())
                .ubatch(plan.ubatch())
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
            return MtpHead.gpu(g.forwardContext(), g.blockRegistry(), null);
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
     * The head of a draft file (eagle3, gemma4-assistant, an MTP draft file): on a GPU target the file's tensors
     * except {@code deferred} uploaded to the target's device with the target's kernel type policy and their weight
     * kernels prepared ({@link WeightKernelPrewarm}); on a CPU target the file mapped on the host, where tensors the
     * drafter never reads take no memory. The weights are appended to {@code out}; the head does not own them.
     *
     * @param deferred tensors the drafter never reads, left off the device
     */
    private static MtpHead draftHead(EngineModel target, GGUFFileParser dp, GgufKv kv, TensorCatalog catalog,
                                     ModelSpec spec, Predicate<GGUFTensorInfo> deferred, List<AutoCloseable> out)
            throws IOException {
        PrismHadamardLoader.Result prism = PrismHadamardLoader.load(kv, n -> {
            TensorRef r = catalog.get(n);
            return r != null ? r.ne() : null;
        }, kv.u32(GgufKeys.SSM_TIME_STEP_RANK, 0), kv.u32(GgufKeys.SSM_GROUP_COUNT, 0));
        if (target instanceof GGUFModelGPU) {
            GGUFModelGPU g = (GGUFModelGPU) target;
            GpuForwardContext ctx = g.forwardContext();
            WeightStore store = dp.createMappedStore();
            out.add(store);
            GpuWeightUploader uploader = new GpuWeightUploader(g.backend(), ctx.queue(), dp.getTensors(), store,
                    spec.arch().ggufName(),
                    GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(ctx.kernels()))
                            .prism(prism)
                            .build());
            out.add(uploader);
            uploader.uploadResident(deferred);
            WeightKernelPrewarm.prewarm(uploader.plans(deferred.negate()), ctx.ops(), ctx.kernels());
            LOGGER.info("{} draft weights on {}: {} MiB resident", spec.arch().ggufName(), g.backend().backendType(),
                    uploader.residentBytes() >> 20);
            return MtpHead.gpu(ctx, g.blockRegistry(), uploader);
        }
        if (target instanceof GGUFModel) {
            GGUFModel c = (GGUFModel) target;
            CpuWeights weights = CpuWeights.open(dp, catalog, prism);
            out.add(weights);
            return MtpHead.cpu(c.forwardContext(), c.blockRegistry(), weights);
        }
        throw notFactoryModel(target);
    }

    /** A separate generative draft model on the target's backend (GPU: the target's device; CPU: the CPU engine). */
    private static Drafter modelDrafter(EngineModel target, String path, EngineConfig cfg, MemoryConfig memory)
            throws IOException {
        ComputeBackend backend = computeBackend(target);
        EngineConfig.BackendMode mode = backend == null ? EngineConfig.BackendMode.CPU
                : cfg.backend().usesGpu() ? cfg.backend() : EngineConfig.BackendMode.AUTO;
        EngineConfig draftCfg = EngineConfig.builder()
                .backend(mode)
                .memory(memory)
                .samplerDefaults(cfg.samplerDefaults())
                .ramCacheMb(cfg.ramCacheMb())
                .vramDepth(cfg.vramDepth())
                .build();
        EngineModel draft = backend == null ? new GGUFModel(path, draftCfg) : new GGUFModelGPU(path, backend, draftCfg);
        ModelDrafter d;
        try {
            d = new ModelDrafter(draft, cfg.draftPMin());
        } catch (RuntimeException e) {
            closeStep(draft, e);
            throw e;
        }
        try {
            d.bindTarget(target);
        } catch (RuntimeException e) {
            closeStep(d, e);
            throw e;
        }
        return new Drafter(d, "draft model " + draft.architecture(), Integer.MAX_VALUE);
    }

    // ================================================================== closing

    /** Closes {@code c}, recording a failure on {@code first} (or returning it when there is none). */
    static Throwable closeStep(AutoCloseable c, Throwable first) {
        try {
            c.close();
            return first;
        } catch (Exception | Error e) {
            if (first == null) {
                return e;
            }
            if (e != first) {
                first.addSuppressed(e);
            }
            return first;
        }
    }

    /** Throws {@code first} as is when it is unchecked or an IOException, else wrapped; nothing when null. */
    static void rethrow(Throwable first, String message) throws IOException {
        if (first == null) {
            return;
        }
        if (first instanceof IOException) {
            throw (IOException) first;
        }
        if (first instanceof RuntimeException) {
            throw (RuntimeException) first;
        }
        if (first instanceof Error) {
            throw (Error) first;
        }
        throw new IOException(message, first);
    }
}
