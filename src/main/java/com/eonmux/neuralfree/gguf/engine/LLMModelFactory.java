package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.QCopySource;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.draft.DrafterChain;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
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
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * Opens models on the configured compute backend and binds speculative drafters to them. Model, draft and adapter
 * files are opened through {@link ModelSources} with the configuration's {@link EngineConfig#sourceOptions()}.
 *
 * <p><b>Backend.</b> With a GPU backend mode ({@link EngineConfig#backend()} other than CPU) the model loads with
 * {@link GGUFModelGPU}, either on a caller-owned backend or on one it creates and owns:
 * {@code HWAcceleration.forMode(mode).createBackend(mode, KernelLibraryRequirement.of(KernelLibraries.LLM))}, so only
 * devices that can run the LLM kernel library qualify. Device selection reads the {@code neuralfree.device} name
 * filter; a {@link EngineConfig#deviceFilter()} is applied by setting that property for the duration of the backend
 * creation (and restoring it afterwards), under a lock shared by every factory load. When the GPU load fails for any
 * reason other than an unreadable file, the reason is logged and the model loads on the CPU engine
 * ({@link GGUFModel}) with the same configuration — unless the configuration is refused by name
 * ({@link IllegalArgumentException} / {@link UnsupportedOperationException}): a layer split, or a GPU-only setting
 * whose silent loss the fallback would hide ({@code neuralfree.weights.w12}, {@code neuralfree.gpu.chainReplay});
 * such a refusal is rethrown. Device, driver and allocation failures keep the fallback.
 *
 * <p><b>Speculative decoding.</b> {@link #openSpeculative} and {@link #speculate} bind the drafters of the
 * configuration to a target: the lookup drafters of {@link EngineConfig#lookupTypes()} ({@link LookupDrafters}:
 * {@code ngram-simple}, {@code ngram-map-k}, {@code ngram-map-k4v}, {@code ngram-mod}, {@code ngram-cache},
 * {@code suffix}, sized by {@link EngineConfig#speculativeOptions()}), and at most one hidden-fed drafter: the
 * target's own NextN head with {@link EngineConfig#mtp()}, or the drafter of the draft file ({@link #draftRoute}).
 * Several drafters chain ({@link DrafterChain}) in llama.cpp's ask order, the lookup drafters first; the decoder
 * verifies with {@link SpeculativeOptions#verifyMode()}. With {@link EngineConfig.DraftType#AUTO} the drafter is
 * chosen from the file as llama.cpp chooses the speculative type of a draft model file or Hugging Face directory
 * when none is given (common/arg.cpp:566-572, common/speculative.cpp:2398-2433), except that a file with NextN
 * blocks that is not an MTP draft of the target's model ({@link ModelSpecBuilder#isMtpDraftOf}) drafts as a
 * separate draft model; {@code DraftType.MTP} ({@code draft-mtp}) and {@code DraftType.MODEL}
 * ({@code draft-simple}) force one route, as llama.cpp's {@code --spec-type} does:
 * <ul>
 *   <li>{@code dflash}: {@link DflashDrafter}, on the device of a GPU target ({@link DflashDevice}) or over the
 *       target's host weights; the draft length is clamped to its block ({@code DflashSpec.nDraftMax()}).</li>
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
 * A separate draft model's own load carries the draft placement settings (llama.cpp's
 * {@code --spec-draft-device}, {@code --spec-draft-ngl}, {@code --spec-draft-override-tensor},
 * {@code --spec-draft-cpu-moe}, {@code --spec-draft-n-cpu-moe}; common/speculative.cpp:2631-2700), in place of the
 * target's placement: a device list ({@link EngineConfig#draftDevices()}, or {@code none} for no offload) opens the
 * draft on GPUs of its own — one GPU for a single selector, a layer split for two or more, the host for
 * {@code none} — and unset devices keep the draft beside the target's weights on the target's backend. The layer
 * count, tensor overrides and expert placement ({@link EngineConfig#draftGpuLayers()},
 * {@link EngineConfig#draftTensorOverrides()}, {@link EngineConfig#draftCpuMoe()},
 * {@link EngineConfig#draftNCpuMoe()}) ride the draft's load as the target's ride its own, and the context fit
 * reserves nothing for a draft on its own devices. The other drafter routes refuse the placement settings: they
 * place no load of their own.
 * Every draft context, the NextN / MTP caches included, runs with {@link MemoryConfig#forDraftContext()}: the draft
 * cache types (F16 unless {@code -ctkd}/{@code -ctvd} configure others, llama.cpp
 * {@code --spec-draft-type-k/-v}, common/common.h:348-349) and the draft ubatch
 * ({@link MemoryConfig#draftUbatchOr}, llama.cpp {@code --spec-draft-ubatch-size}, whose 0 keeps the target's
 * {@code -ub}) and no recurrent-state snapshots ({@code n_rs_seq = 0}). With several NextN blocks the heads chain and
 * a draft holds at most one token per block, as llama.cpp clamps the draft length of a chained MTP drafter
 * (common/speculative.cpp:1523-1528). Each backend has one
 * hidden-state tap, so a target serves one drafter (or one pockettts reader) at a time ({@link HiddenTapSlot}). A
 * target whose configuration names a draft model fits its context beside that drafter's memory
 * ({@link #draftCompanion}), as llama.cpp fits the main and the draft context together.
 */
public final class LLMModelFactory {

    private static final Logger LOGGER = LogManager.getLogger(LLMModelFactory.class);

    /** Serializes loads that set the device-filter property, so concurrent loads cannot see each other's filter. */
    private static final Object DEVICE_PROPERTY_LOCK = new Object();

    private LLMModelFactory() {
    }

    // ================================================================== single models

    /**
     * Opens a model on the backend {@code cfg} selects. A failure of the GPU load falls back to the CPU engine only
     * when it means the device cannot serve the model ({@link #fallbackReason}); every other failure stops the load
     * with the GPU error as its cause.
     *
     * @throws FileNotFoundException when {@code path} is not a readable file or directory
     * @throws IOException           when the file cannot be read
     * @throws RuntimeException      when the CPU load fails too (the GPU failure is suppressed in it), when
     *                               {@code cfg} selects the CPU and the CPU load fails, or when the GPU load fails
     *                               with an error that does not mean the device cannot serve the model
     */
    public static EngineModel open(String path, EngineConfig cfg) throws IOException {
        return open(path, cfg, null);
    }

    /**
     * Opens a model on {@code sharedBackend}, which stays owned by the caller. With a null backend this is
     * {@link #open(String, EngineConfig)}. A device filter in {@code cfg} has no effect on a backend the caller
     * created. A configured draft file is read first to size the compressor rollback snapshots its drafter needs
     * ({@link MemoryConfig#rollbackSnapshots()}).
     *
     * <p>The GPU load's failure falls back to the CPU engine only when it means this device cannot serve this model:
     * no usable device or driver, the LLM kernel library unavailable on it, device memory the fitter could not fit,
     * an architecture, weight type or layer feature the GPU engine does not implement
     * ({@link GpuModelUnsupportedException}, refused by name), or no context that fits the device memory budget
     * ({@link ContextFitter.ContextBudgetException}). The fallback is announced on the warning log with that reason,
     * and the CPU engine reports it ({@link EngineModel#gpuFallbackReason()}) so the demos' startup lines and the
     * server's {@code /props} carry it. Every other failure — a kernel argument left unbound, an assertion-style
     * state, a failed kernel launch, a broken native installation — is an engine defect, not a device limitation:
     * it stops the load with the GPU error as its cause, like llama.cpp's error on a backend that fails. A refusal
     * of the configuration never falls back either but is thrown: a layer split ({@link EngineConfig#split()}), and
     * every GPU-only setting a CPU load would silently ignore — the W12 streaming ({@code neuralfree.weights.w12},
     * whose lossless F16 refusal and W12-only refusals a CPU load cannot serve) and a requested chain replay
     * ({@code neuralfree.gpu.chainReplay}).</p>
     *
     * @throws FileNotFoundException         when {@code path} is not a readable file or directory
     * @throws IOException                   when the file cannot be read
     * @throws IllegalArgumentException      when the configuration is refused, naming the setting (a layer split's
     *                                       placement, an off-grid F16 weight under
     *                                       {@code neuralfree.weights.w12.f16 = refuse})
     * @throws UnsupportedOperationException when the configuration is refused, naming the setting (the model or a
     *                                       combination a layer split cannot serve, a W12 load with no target)
     */
    public static EngineModel open(String path, EngineConfig cfg, ComputeBackend sharedBackend) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(cfg, "cfg");
        ModelSources.requireOpenable(path);
        EngineConfig c = withDraftRollback(path, cfg);
        if (sharedBackend == null && !c.backend().usesGpu()) {
            return warmed(new GGUFModel(path, c), c);
        }
        boolean gpuOnlyRequested = c.split() || c.rowSplit() || c.contextMode()
                || c.speculativeOptions().w12() || GpuForwardBackend.chainReplayConfigured();
        try {
            return warmed(openGpu(path, c, sharedBackend), c);
        } catch (IOException e) {
            throw e;
        } catch (RuntimeException | LinkageError gpuFailure) {
            if (gpuOnlyRequested && (gpuFailure instanceof IllegalArgumentException
                    || gpuFailure instanceof UnsupportedOperationException)) {
                throw (RuntimeException) gpuFailure;
            }
            String reason = fallbackReason(gpuFailure);
            if (reason == null) {
                LOGGER.error("GPU load of {} failed: {}; this is not a device limitation, so the load fails"
                        + " instead of falling back to the CPU engine", path, describe(gpuFailure));
                throw gpuFailure;
            }
            LOGGER.warn("GPU load of {} fell back to the CPU engine ({}): {}", path, reason, describe(gpuFailure));
            try {
                return warmed(new GGUFModel(path, c.toBuilder().backend(EngineConfig.BackendMode.CPU)
                        .splitMode(EngineConfig.SplitMode.NONE).moeCachePool(false).build(), reason), c);
            } catch (IOException | RuntimeException | Error cpuFailure) {
                cpuFailure.addSuppressed(gpuFailure);
                throw cpuFailure;
            }
        }
    }

    /**
     * Why a failed GPU load falls back to the CPU engine — the failure means the device cannot serve the model — or
     * null when it is an engine defect whose propagation must stop the load: no usable device or driver
     * ({@link NoBackendAvailableException}), the LLM kernel library unavailable on the device
     * ({@link UnsupportedKernelException}), device memory the fitter could not avoid exhausting
     * ({@link GpuOutOfMemoryException}), an architecture or weight type the GPU engine does not implement
     * ({@link GpuModelUnsupportedException}), or no context that fits the device memory budget
     * ({@link ContextFitter.ContextBudgetException}).
     */
    static String fallbackReason(Throwable gpuFailure) {
        if (gpuFailure instanceof NoBackendAvailableException) {
            return "no usable GPU device or driver";
        }
        if (gpuFailure instanceof UnsupportedKernelException) {
            return "the LLM kernel library is unavailable on the device";
        }
        if (gpuFailure instanceof GpuOutOfMemoryException) {
            return "the device memory does not fit the model";
        }
        if (gpuFailure instanceof GpuModelUnsupportedException) {
            return "the GPU engine cannot serve the model: " + gpuFailure.getMessage();
        }
        if (gpuFailure instanceof ContextFitter.ContextBudgetException) {
            return "no context fits the device memory budget";
        }
        return null;
    }

    /**
     * The load's warmup ({@code llama_set_warmup}, the default of {@code --no-warmup} off): one decode on an
     * empty sequence, token 0 at position 0, then a reset of the cache, so the first real request runs on kernels
     * queued, blocks decoded and every lazy path touched once. A paged memory ({@code neuralfree.parallel > 1} or
     * {@code neuralfree.kv.paged}) takes that row through {@link ForwardBackend#forwardBatch} on sequence 0, the
     * path its sequences decode on. Models that cannot decode (no decoder-only token
     * loop, no model memory, {@code isDiffusion}) skip it, and {@code cfg.sourceOptions().warmup()} false skips
     * it ({@code --no-warmup}). The decode and the reset run on the model's backend as the engine owns the thread
     * does. A model whose warmup fails is closed before the failure propagates, so a GPU model releases its
     * weights, memory and backend instead of holding them on the device for the rest of the process.
     *
     * @throws java.io.IOException          when the model cannot be read
     * @throws IllegalStateException        from the backend when the decode fails: a warmup of a loadable model
     *                                      that fails stops the load, like llama.cpp's warmup failure does
     */
    private static EngineModel warmed(EngineModel model, EngineConfig cfg) throws IOException {
        if (!cfg.sourceOptions().warmup()) {
            return model;
        }
        try {
            ForwardBackend backend = model.forwardBackend();
            if (backend.memory() == null || !hasTokenLoop(model.supportClass())) {
                return model;
            }
            if (backend.memory().pagedSequences() != null) {
                // the contiguous decode refuses on paged memory: warm through the batched forward the engine runs
                backend.forwardBatch(ForwardBatch.builder(1).add(0, 0, 0, true).build());
            } else {
                backend.decode(0, 0);
            }
            model.resetKVCache();
            return model;
        } catch (RuntimeException | Error e) {
            try {
                model.close();
            } catch (IOException | RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /**
     * {@code cfg} with the compressor rollback snapshots ({@link MemoryConfig#rollbackSnapshots()}, llama.cpp
     * {@code n_rs_seq}) its draft file needs, pinned before the target is planned. {@link EngineConfig} gives a draft
     * file under {@link EngineConfig.DraftType#AUTO} the draft max; here the file's {@link #draftRoute} decides, as
     * llama.cpp picks the speculative type from the file before sizing the target context
     * (common/speculative.cpp:2398-2433, common/common.h:401-407): an MTP, eagle, eagle3, gemma4-assistant or dflash
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
        Path target = Paths.get(targetPath);
        try {
            ModelSources.requireOpenable(draftPath);
            ModelSource dp = ModelSources.open(draftPath, cfg.sourceOptions(), target, cfg.activationQuant());
            try {
                Arch arch = draftArch(dp, draftPath);
                GgufKv kv = GgufKv.of(dp);
                TensorCatalog tc = TensorCatalog.of(dp);
                route = draftRoute(arch, kv, tc,
                        ModelSpecBuilder.isNextnDraft(kv, tc)
                                ? inspect(targetPath, cfg.memory(), cfg.sourceOptions()) : null,
                        cfg.draftType(), draftPath);
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
     * The resolved specification of a model from its metadata and tensor shapes, without loading tensor data; lets
     * callers choose settings by architecture and support class before {@link #open}. Checkpoints that are not GGUF
     * are opened with the source options of the system properties ({@link ModelSourceOptions#fromSystemProperties()}).
     *
     * @throws FileNotFoundException when {@code path} is not a readable file or directory
     * @throws RuntimeException      when the architecture cannot be resolved
     */
    public static ModelSpec inspect(String path, MemoryConfig memory) throws IOException {
        return inspect(path, memory, ModelSourceOptions.fromSystemProperties());
    }

    /**
     * The resolved specification of a model opened with {@code config}: its memory settings, source options and
     * pooling override ({@code neuralfree.pooling}) apply, so a caller chooses settings by the resolved head
     * before {@link #open}.
     *
     * @throws FileNotFoundException when {@code path} is not a readable file or directory
     * @throws RuntimeException      when the architecture cannot be resolved or a pooling override is refused
     */
    public static ModelSpec inspect(String path, EngineConfig config) throws IOException {
        Objects.requireNonNull(config, "config");
        return inspect(path, config.memory(), config.sourceOptions(), config.pooling());
    }

    /**
     * The resolved specification of a model from its metadata and tensor shapes, without loading tensor data, opened
     * with {@code options}.
     *
     * @throws FileNotFoundException when {@code path} is not a readable file or directory
     * @throws RuntimeException      when the architecture cannot be resolved
     */
    public static ModelSpec inspect(String path, MemoryConfig memory, ModelSourceOptions options) throws IOException {
        return inspect(path, memory, options, null);
    }

    /**
     * The resolved specification of a model from its metadata and tensor shapes, without loading tensor data, opened
     * with {@code options} and the head pooling overridden by {@code pooling} when not null.
     *
     * @throws FileNotFoundException        when {@code path} is not a readable file or directory
     * @throws RuntimeException             when the architecture cannot be resolved
     * @throws IllegalArgumentException     when a {@code rank} pooling override finds no classifier head
     */
    public static ModelSpec inspect(String path, MemoryConfig memory, ModelSourceOptions options,
                                    PoolingKind pooling) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(memory, "memory");
        Objects.requireNonNull(options, "options");
        ModelSources.requireOpenable(path);
        ModelSource p = ModelSources.open(path, options);
        try {
            return ModelSpecBuilder.build(GgufKv.of(p), TensorCatalog.of(p), memory, 0, pooling);
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
     * to it; the session owns and closes the target. Its generations apply the target's sampling defaults unless
     * {@code cfg} turns them off ({@link SamplingDefaults#forModel}).
     *
     * @param draftPath the draft model file or Hugging Face directory, or null to use {@code cfg}'s draft model
     *                  path or its MTP setting
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
        checkDrafters(c);
        EngineModel target = open(targetPath, c);
        try {
            return attach(target, Paths.get(targetPath), c, true);
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
     * Binds the drafters {@code cfg} configures ({@link EngineConfig#lookupTypes()}, and
     * {@link EngineConfig#draftModelPath()} or {@link EngineConfig#mtp()}) to an open target; the target stays
     * owned by the caller and must not run forward passes outside the session until the session is closed. Its
     * generations apply the target's sampling defaults unless {@code cfg} turns them off
     * ({@link SamplingDefaults#forModel}).
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
        checkDrafters(cfg);
        return attach(target, null, cfg, false);
    }

    /**
     * Refuses a drafter configuration the single-sequence session cannot run: none at all, a draft file beside the
     * MTP head, or probabilistic drafting (the batch engine's).
     */
    private static void checkDrafters(EngineConfig cfg) {
        if (cfg.speculativeOptions().probabilisticDraft()) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD + "=probabilistic:"
                    + " the single-sequence drafters draft greedily; probabilistic drafting runs in the batch engine"
                    + " (BatchEngine, the server)");
        }
        if (cfg.draftModelPath() != null && cfg.mtp()) {
            throw new IllegalArgumentException("both a draft model (" + cfg.draftModelPath() + ") and the MTP head are"
                    + " configured; a target binds one drafter");
        }
        if (!cfg.speculative()) {
            throw new IllegalArgumentException("no drafter is configured: set a draft model ("
                    + EngineConfig.PROP_DRAFT + "), the MTP head (" + EngineConfig.PROP_MTP + ") or a lookup type ("
                    + EngineConfig.PROP_DRAFT_TYPE + ")");
        }
    }

    private static SpeculativeSession attach(EngineModel target, Path targetPath, EngineConfig cfg,
                                             boolean ownsTarget) throws IOException {
        requireSpeculatable(target);
        ForwardBackend fb = target.forwardBackend();
        HiddenTapSlot.Lease lease = HiddenTapSlot.acquire(fb, "a speculative drafter");
        List<AutoCloseable> resources = new ArrayList<>();
        Drafter built = null;
        try {
            built = drafters(target, targetPath, cfg, resources);
            final DraftModel drafter = built.drafter;
            int draftMax = Math.min(cfg.draftMax(), built.maxDraft);
            int draftMin = Math.min(cfg.draftMin(), draftMax);
            GGUFTokenizer tok = target.tokenizer();
            ReasoningFilter rf = ReasoningParsers.forModel(cfg.reasoningParser(), cfg.toolParser(), tok);
            SamplingDefaults defaults = SamplingDefaults.forModel(target, cfg);
            SpeculativeOptions options = cfg.speculativeOptions();
            QuantizedView qView = null;
            if (options.twoLevelVerify()) {
                qView = twoLevelView(target, drafter, options);
                resources.add(qView);
                // a nested W12 view's backend is the target's own, whose hidden tap the drafter already holds:
                // only a copied view binds a second backend
                if (drafter instanceof DflashDrafter && qView.backend() != fb) {
                    ((DflashDrafter) drafter).bindSecondBackend(qView.backend());
                }
            }
            List<AutoCloseable> sessionResources = new ArrayList<>(resources);
            sessionResources.addAll(built.saves);
            final QuantizedView view = qView;
            SpeculativeSession session = new SpeculativeSession(target, ownsTarget, drafter, built.kind,
                    sessionResources, lease, sink -> options.applyTo(SpeculativeDecoder.builder(fb, drafter, tok)
                            .reasoning(rf)
                            .adapters(target.loraRegistry())
                            .draftMax(draftMax)
                            .draftMin(draftMin)
                            .prefixReuse(cfg.memory().prefixReuse())
                            .contextShift(cfg.memory().contextShift())
                            .samplingDefaults(defaults)
                            .logitsProcessors(LogitsProcessors.forConfig(cfg))
                            .quantizedView(view)
                            .statsSink(sink))
                            .build());
            LOGGER.info("speculative decoding on {}: {} drafter, draft max {} (requested {}), min {}, p_min {},"
                            + " verify {}{}{}", target.name(), built.kind, draftMax, cfg.draftMax(), draftMin,
                    cfg.draftPMin(), options.verifyMode(), options.verifyOnDevice() ? "" : " (greedy rows on the host)",
                    qView == null ? "" : ", " + qView);
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
    /**
     * The quantized target view of two-level verification ({@code neuralfree.draft.tlv}): opens the model's
     * quantized copy over its memory, refusing by name the drafter shapes the Q round cannot serve — a drafter
     * that drafts no single-pass tree (the view verifies a proposal with the fixed-width tree batch the exact
     * round also runs) and a model without a quantized view.
     */
    private static QuantizedView twoLevelView(EngineModel target, DraftModel drafter, SpeculativeOptions options)
            throws IOException {
        String shape = "two-level verification (" + SpeculativeOptions.PROP_TLV + ") verifies the drafter's tree"
                + " once through a quantized copy of the target, which needs a tree drafter that verifies in one"
                + " pass (neuralfree.draft.tree with a single-pass verify, such as the dflash drafter's)";
        if (!(drafter instanceof TreeDraftModel)) {
            throw new UnsupportedOperationException(shape + "; the drafter " + drafter + " drafts no tree");
        }
        TreeDraftModel tree = (TreeDraftModel) drafter;
        if (!tree.singlePassVerify() || tree.branches() <= 1) {
            throw new UnsupportedOperationException(shape + "; the drafter " + drafter + " drafts "
                    + (tree.branches() <= 1 ? "no branching tree" : "a tree its rounds verify depth by depth"));
        }
        return target.openQuantizedView(options.tlvBits());
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

    /**
     * A bound drafter with its kind, its own draft-length limit and the writes that run after it closes (the
     * dynamic lookup caches).
     */
    private static final class Drafter {
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
    private static Drafter drafters(EngineModel target, Path targetPath, EngineConfig cfg, List<AutoCloseable> out)
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
                first = closeStep(d, first);
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
            closeStep(dp::close, e);
            throw e;
        }
        if (route != DraftRoute.MODEL && cfg.speculativeOptions().heterogeneousVocab()) {
            IllegalArgumentException refusal = new IllegalArgumentException(SpeculativeOptions
                    .PROP_HETEROGENEOUS_VOCAB + " needs a separate draft model (draft-simple), but " + path + " is a "
                    + arch.ggufName() + " drafter");
            closeStep(dp::close, refusal);
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
            dp = DerivedDraftSource.wrap(dp, hostWeights(target), head);
        }
        DflashDrafter.Options options = DflashDrafter.Options.of(cfg.draftMin(), cfg.draftPMin(),
                cfg.memory().draftTypeK(), cfg.memory().kvRotation()).withActivationQuant(cfg.activationQuant())
                .withTree(spec.treeVerify(), spec.treeTemp());
        List<AutoCloseable> deviceResources = new ArrayList<>();
        try {
            DflashDevice device = dflashDevice(target, dp, cfg, options, deviceResources);
            if (device != null) {
                DflashDrafter d = boundDflash(DflashDrafter.open(dp, hostWeights(target), options, device), target);
                out.addAll(deviceResources);
                LOGGER.info("dflash drafter {} drafts on {}{}", cfg.draftModelPath(), device,
                        spec.treeVerify() ? " with single-pass draft-tree verification" : "");
                return new Drafter(d, arch.ggufName(), d.spec().nDraftMax());
            }
        } catch (UnsupportedOperationException e) {
            Throwable closeFailure = null;
            for (int i = deviceResources.size() - 1; i >= 0; i--) {
                closeFailure = closeStep(deviceResources.get(i), closeFailure);
            }
            if (closeFailure != null) {
                e.addSuppressed(closeFailure);
            }
            LOGGER.warn("dflash drafter {} drafts on the host: {}", cfg.draftModelPath(), describe(e));
        } catch (IOException | RuntimeException | Error e) {
            out.addAll(deviceResources);
            throw e;
        }
        DflashDrafter d = boundDflash(DflashDrafter.open(dp, hostWeights(target), options), target);
        LOGGER.info("dflash drafter {} drafts on the host{}", cfg.draftModelPath(),
                spec.treeVerify() ? " with single-pass draft-tree verification" : "");
        return new Drafter(d, arch.ggufName(), d.spec().nDraftMax());
    }

    /** {@code d} bound to {@code target}; a failed bind closes it. */
    private static DflashDrafter boundDflash(DflashDrafter d, EngineModel target) {
        try {
            d.bindTarget(target);
        } catch (RuntimeException e) {
            closeStep(d, e);
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

    /** How a draft file drafts for its target ({@link #draftRoute}). */
    enum DraftRoute {
        /** {@code dflash}: {@link DflashDrafter}, on the GPU target's device or on the host. */
        DFLASH,
        /** {@code eagle}, {@code eagle3}, {@code gemma4-assistant}: {@link DraftModelLoader#fromFile} with the
         * file's head. */
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
     *   <li>{@link EngineConfig.DraftType#AUTO}: {@code dflash}, {@code eagle}, {@code eagle3} and
     *       {@code gemma4-assistant} by architecture; a file whose last block holds {@code nextn.eh_proj}
     *       ({@link ModelSpecBuilder#isNextnDraft}, llama.cpp's detection, common/speculative.cpp:2414-2420) drafts
     *       with its NextN blocks when it is an MTP draft of the target's model
     *       ({@link ModelSpecBuilder#isMtpDraftOf}), and otherwise, logged, as a separate draft model, since
     *       llama.cpp's converters keep the NextN blocks of a full model by default (conversion/qwen.py:309-315,
     *       conversion/glm.py:119-121) and a smaller model of the family is an ordinary draft model; any other file
     *       is a separate draft model;</li>
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

    /**
     * Refuses the draft model placement settings ({@code -devd}, {@code -ngld}, {@code -otd}, {@code -cmoed},
     * {@code -ncmoed}) for a draft file that drafts by another route: an MTP drafter runs over the target's hidden
     * rows and a hidden-fed drafter on the target's runtime, and neither loads a model of its own to place.
     *
     * @throws UnsupportedOperationException naming the settings and the route the file drafts with
     */
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

    /**
     * Bytes a drafter file's weights take where the target runs: on a GPU target the device bytes of the tensors the
     * drafter places on the device; on a CPU target 0, as its drafter reads its weights from the mapped file.
     */
    @FunctionalInterface
    public interface DraftWeightBytes {
        /**
         * @param draft  the draft model
         * @param placed the tensors of {@code draft} its drafter places on the target's device, for the route the
         *               file drafts with for the target under the configured draft type ({@link #draftRoute},
         *               {@link #deviceTensors})
         * @throws IOException when the drafter's tensor data cannot be mapped
         */
        long of(ModelSource draft, Predicate<GGUFTensorInfo> placed) throws IOException;
    }

    /**
     * The device tensors of a draft file drafting by {@code route}, the set {@link DraftWeightBytes} measures:
     * <ul>
     *   <li>{@code dflash}: the matrices its device stack reads ({@link DflashDevice#reads}) when it drafts on a
     *       GPU target's device;</li>
     *   <li>an MTP draft file: every tensor but its trunk blocks ({@link ModelSpecBuilder#mtpDraftReads});</li>
     *   <li>{@code eagle3}, {@code gemma4-assistant}: every tensor;</li>
     *   <li>a separate draft model: what its own GPU load places ({@link WeightPlacement}), which leaves its NextN
     *       blocks off the device, as a draft context loads no MTP layers, and the tensors only host heads read.</li>
     * </ul>
     *
     * @param memory         the draft model's memory settings ({@link MemoryConfig#forDraftContext()}), for its spec
     * @param tensors        the draft file's tensor list, for the placement a layer split cuts
     * @param draftPlacement the draft model's placement settings as their own load configuration
     *                       ({@code -ngld}, {@code -otd}, {@code -cmoed}, {@code -ncmoed}; {@link #draftPlacementOf}),
     *                       or null for the default placement without them
     */
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

    /**
     * The drafter's companion at the plan it is fitted beside, opened with no explicit target path: a GGUF draft
     * measures as with an explicit target, while a Hugging Face draft of a drafter architecture cannot name its
     * target and the measurement fails with a logged warning, so the context fits without it.
     *
     * @param target      the target's resolved specification
     * @param hostTarget  true when the target runs on the host, whose budget also holds a drafter's host caches
     * @param weightBytes the bytes the drafter's weights take where the target runs, given the tensors the drafter's
     *                    route places on the device
     */
    public static ContextFitter.Companion draftCompanion(EngineConfig cfg, ModelSpec target, boolean hostTarget,
                                                         DraftWeightBytes weightBytes) {
        return draftCompanion(cfg, target, hostTarget, null, weightBytes);
    }

    /**
     * The memory of the drafter {@code cfg} names ({@link EngineConfig#draftModelPath()}) that a target fits its
     * context beside, measured at every candidate context as llama.cpp's {@code common_fit_params} measures its extra
     * model (common/common.cpp:1252-1280, common/fit.cpp:203-258):
     * <ul>
     *   <li>none without a draft model, or with {@link EngineConfig#mtp()}: the NextN / MTP draft memory is reserved
     *       through {@link MemoryConfig#nextnMemory()} and its weights are the target's;</li>
     *   <li>none for a target no drafter binds to: one without memory or without a decoder-only token loop;</li>
     *   <li>{@code dflash} on a host target drafts over host weights with a host cache: its KV cache at the target
     *       plan's context ({@link DflashDrafter#cacheBytes} in the draft cache type,
     *       {@link MemoryConfig#draftTypeK()}); on a device target that drafts on its device
     *       ({@link SpeculativeOptions#dflashOnDevice()}, a draft {@link DflashDevice#refusal} admits, no layer
     *       split) the matrices it uploads ({@link #deviceTensors}) and its device cache and working buffers
     *       ({@link DflashDevice#deviceBytes}); on any other device target nothing, as none of it is device
     *       memory;</li>
     *   <li>{@code eagle}, {@code eagle3}: their weights plus their own caches with
     *       {@link MemoryConfig#forDraftContext()} at the target plan's context and ubatch (the draft ubatch when
     *       one is set);</li>
     *   <li>{@code gemma4-assistant}: its weights; it reads the target's caches
     *       ({@code MemoryPlanner.planDrafter});</li>
     *   <li>an MTP draft file of the target's model ({@link #draftRoute}), with or without its trunk: the weights its
     *       drafter reads ({@link #deviceTensors}) and the NextN caches of its blocks
     *       ({@link MemoryPlanner#nextnBytes(ModelSpec, MemoryConfig, int, int)}, draft-context settings); its trunk
     *       takes neither caches nor scratch;</li>
     *   <li>a separate draft model ({@link #draftRoute}, NextN blocks or not): its weights, its caches as for eagle3
     *       and its forward scratch (the well-known slots of {@link ScratchWidths}) for every token of its step
     *       ubatch, the draft ubatch when one is set, else the target plan's.</li>
     * </ul>
     * The drafter is the one the attach binds: {@link #draftRoute} with the configured
     * {@link EngineConfig#draftType()} against {@code target}.
     * When the draft file cannot be measured, a warning names the reason and the target fits alone, as llama.cpp
     * does (common/fit.cpp:225-230); the drafter reports the problem when it binds.
     *
     * @param targetPath the model the draft serves, as the target was opened, for Hugging Face drafts that read it
     *                   through the draft source; null when the caller gives no target path, which only a Hugging
     *                   Face draft of a drafter architecture misses (the fit then leaves it out)
     * @param target     the target's resolved specification
     * @param hostTarget true when the target runs on the host, whose budget also holds a drafter's host caches
     * @param weightBytes the bytes the drafter's weights take where the target runs, given the tensors the drafter's
     *                    route places on the device
     */
    public static ContextFitter.Companion draftCompanion(EngineConfig cfg, ModelSpec target, boolean hostTarget,
                                                         Path targetPath, DraftWeightBytes weightBytes) {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(weightBytes, "weightBytes");
        String path = cfg.draftModelPath();
        if (path == null || cfg.mtp() || target.memoryKind() == MemoryKind.NONE
                || !hasTokenLoop(target.supportClass())) {
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

    /**
     * The memory the quantized target view of two-level verification ({@code neuralfree.draft.tlv}) streams where
     * the target runs, counted into the context fit like a drafter's weights: the copy's restated matrices
     * ({@link QCopySource#plannedBytes}), metadata-planned, nothing per step token (the view reads the target's
     * caches). None when two-level verification is off, and none for a target whose copy the attach refuses with
     * its own message (routed experts, no matrix the copy undercuts) — the refusal surfaces when the view opens,
     * as the drafter measurement does. None too when {@code w12Target} holds: the view then reads the target's
     * own W12 rows through their nested 8-bit Q view, adding no copy and no memory.
     *
     * @param targetSource the target's model source, whose tensor table the copy is planned from
     * @param w12Target    whether the target streams its projection matrices as W12 rows ({@code
     *                     neuralfree.weights.w12}), which makes the view the nested 8-bit one
     */
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
    private static Arch draftArch(ModelSource dp, String path) {
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
        ComputeBackend backend = computeBackend(target);
        List<String> devices = cfg.draftDevices();
        if (isNoDevice(devices)) {
            return new GGUFModel(path, draftConfig(cfg, memory, EngineConfig.BackendMode.CPU, null));
        }
        if (!devices.isEmpty()) {
            return openGpu(path, draftConfig(cfg, memory, draftDeviceMode(cfg), devices), null);
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

    /**
     * The separate draft model of {@code cfg.draftModelPath()} for a batch engine over {@code target} (vLLM's
     * draft-model proposer): the draft-context settings of the single-sequence drafter ({@code -ctkd}, {@code -ctvd},
     * {@code -ubd}, the target's context) over paged memory with one sequence per target sequence slot, on the
     * target's backend. The caller owns the model.
     *
     * @throws UnsupportedOperationException when the target memory is not paged, or the draft file drafts by
     *                                       another route ({@code eagle}, {@code eagle3},
     *                                       {@code gemma4-assistant}, {@code dflash}, an MTP draft of the target),
     *                                       naming {@code neuralfree.draft}
     * @throws IllegalArgumentException      when the draft type cannot run the file
     * @throws IOException                   when the draft file cannot be read
     */
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
