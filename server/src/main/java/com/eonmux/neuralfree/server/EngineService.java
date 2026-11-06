package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.SpeechFormat;
import com.eonmux.neuralfree.gguf.speech.Lfm3aGenClip;
import com.eonmux.neuralfree.gguf.speech.Lfm3aSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.PocketTtsClip;
import com.eonmux.neuralfree.gguf.speech.PocketTtsSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsClip;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.SpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.SpeechOptions;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.TemplateCaps;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserAnalysis;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserFormat;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.chat.response.BeamGrammar;
import com.eonmux.neuralfree.gguf.chat.response.ResponseTemplate;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.chat.tools.ToolStrict;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.InfillPrompt;
import com.eonmux.neuralfree.gguf.engine.lora.LoraHandle;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.engine.PoolingOutput;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.batch.SchedulingPolicy;
import com.eonmux.neuralfree.gguf.engine.batch.SlotResult;
import com.eonmux.neuralfree.gguf.engine.batch.SlotStatus;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferParams;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.image.ImageDecodeException;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.mm.ClipDualEncoder;
import com.eonmux.neuralfree.gguf.mm.ClipFamily;
import com.eonmux.neuralfree.gguf.mm.MmEncoderTpMode;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.server.tools.ToolsRegistry;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The serving lifecycle as a data-parallel pool: {@code neuralfree.server.replicas} engine
 * replicas behind the one service, each an {@link EngineReplica} (its open {@link EngineModel} — the backend
 * {@code neuralfree.backend} selects, CPU or GPU —, its {@link BatchEngine}, its pump thread) on its device when
 * {@code neuralfree.server.devices} names one. The
 * {@link DpScheduler} routes every request by the {@code neuralfree.server.load.balance} method — by default to
 * the least-loaded replica, preferring one whose recent prompts share the request's prefix (its prefix cache reuses
 * the rows); one replica on the default device — the default layout — behaves exactly like the single-engine
 * service. The replicas may be some of the deployment's data-parallel ranks ({@link DpLayout}): then an external
 * balancer spreads the requests over several such servers, and each reports the ranks it serves. HTTP threads
 * call {@link #submit} and the caller waits on the returned {@link Submission}'s latch or streams from its sink
 * queue.
 *
 * <p>The model path comes from the {@code neuralfree.server.model} property; the rest of the engine reads its
 * {@code neuralfree.*} properties as everywhere else ({@link EngineConfig#fromProperties}). Without the path the
 * service starts unloaded and every request refuses by name — the module stays bootable for its health and
 * metrics endpoints.
 *
 * <p>Every replica also sleeps (vLLM's sleep mode, llama.cpp's {@code --sleep-idle-seconds}): a sleep of level 0
 * pauses the scheduling, level 1 drops the KV cache contents, level 2 closes the model and releases its weights
 * and memory too — the wake re-opens it from the model path without restarting the process. The lifecycle
 * operations apply to the whole pool: {@link #sleep}, {@link #wakeUp}, {@link #pause}, {@link #resume}, the KV
 * release and the checkpoint capture run on every replica, and an idle sleep of one replica wakes on the next
 * request. An explicit sleep refuses generation until {@code /wake_up}.
 *
 * <p>The pool serves per-request LoRA adapters ({@link LoraAdmin}): the startup adapters ({@link ServerLoras})
 * load into every replica's model before it serves, and the runtime loads and unloads apply to every replica.</p>
 *
 * <p>The pool also takes live weight updates ({@link EngineWeightTransfer}, vLLM's in-process weight transfer
 * engine): an RL trainer initializes the transfer, then drives update rounds over every replica — each write
 * lands on the replica's pump thread between steps, so the next request serves the new weights.</p>
 *
 * <p>The media projector the replicas share follows the models: once every replica released its weights (a sleep
 * of level 2, requested or idle) the projector releases its weights too, and it re-opens as soon as a model is open
 * again — on {@code /wake_up}, or on the request that wakes an idle pool. As in vLLM, a sleep of level 1 or more, the
 * KV cache release and a checkpoint preparation drop the cached media encodings.</p>
 *
 * <p>The pool's shape is controllable at runtime: {@link #scaleElasticEp} resizes it for vLLM's elastic data-parallel
 * scaling (growing opens the new replicas, shrinking drains and closes the removed ones), and the fault-tolerance
 * routes report each replica's health and recover failed ones or remove their ranks ({@link #submitFaultTolerance}).
 * The replicas' data-parallel ranks come from the layout and survive a rank removal, so a dropped rank's slot stays
 * the deployment's.</p>
 */
public final class EngineService implements AutoCloseable {

    /**
     * One submitted request: the engine handle plus the latch that fires on its terminal state, and the hook run
     * once at that moment on the replica's pump thread, after every routed piece reached the sink and before the
     * latch releases its waiters.
     */
    public static final class Submission {
        private final EngineReplica.Submission delegated;
        private final Supplier<Map<String, Object>> kvTransfer;

        Submission(EngineReplica.Submission delegated) {
            this(delegated, null);
        }

        Submission(EngineReplica.Submission delegated, Supplier<Map<String, Object>> kvTransfer) {
            this.delegated = delegated;
            this.kvTransfer = kvTransfer;
        }

        /** The engine handle, or null until the pump thread registered the request. */
        public BatchSequence sequence() {
            return delegated.sequence();
        }

        /** The engine refusal (oversize prompt, unusable constraint), or null once registered. */
        public Throwable refused() {
            return delegated.refused();
        }

        /** Waits for the terminal state or the registration refusal; false on timeout. */
        public boolean await(long timeout, TimeUnit unit) throws InterruptedException {
            return delegated.await(timeout, unit);
        }

        /**
         * The request's {@code kv_transfer_params} answer (the prefill instance's address and transfer id, or the
         * decode instance's completion), or null when it ran no transfer. Read it once the request is terminal;
         * a failed export throws {@link IllegalStateException} rather than advertising a missing transfer.
         */
        public Map<String, Object> kvTransferParams() {
            return kvTransfer == null ? null : kvTransfer.get();
        }
    }

    private static final Logger LOGGER = Logger.getLogger(EngineService.class.getName());

    private final EngineConfig config;
    /** The custom logits processors the request bodies' {@code vllm_xargs} configure, empty when none are loaded. */
    private final LogitsProcessors logitsProcessors;
    private final ServerOptions options;
    private final ServedModels served;
    private final ResponseStore responses;
    private final LiveCompletions completions = new LiveCompletions();
    /**
     * The replicas, in their pool order; empty while unloaded ({@code neuralfree.server.model} unset). Replaced
     * whole by the elastic scale and the fault-tolerance rank removal, so a reader iterates one consistent pool.
     */
    private volatile List<EngineReplica> replicas;
    private final DpScheduler scheduler;
    /** The layout the pool serves today: the launch's, replaced by an elastic scale. */
    private volatile DpLayout layout;
    /** The model path the replicas open, or null when the service starts unloaded. */
    private final String modelPath;
    /** The per-request adapter catalog the replicas' opens load. */
    private final LoraCatalog catalog;
    /** The device filters of {@code neuralfree.server.devices}, one per pool position; null for the default. */
    private final List<String> devices;
    /** Whether the pool serves runtime adapter changes ({@link ServerLoras#runtimeUpdating()}). */
    private final boolean runtimeAdapters;
    /** Serializes the pool changes: an elastic scale, a fault-tolerance retry or a rank removal. */
    private final Object poolLock = new Object();
    /** The elastic-scale lifecycle the status route reports ({@link ElasticPhase}). */
    private volatile ElasticPhase elasticPhase = ElasticPhase.IDLE;
    /** The epoch of the latest elastic scale, null before the first. */
    private volatile String elasticEpoch;
    /** The requested data-parallel size of the latest elastic scale, null before the first. */
    private volatile Integer elasticRequestedSize;
    /** The failure that ended the latest elastic scale, null when it succeeded. */
    private volatile String elasticError;
    /** How many elastic-scale epochs ran; the next epoch is its successor. */
    private long elasticEpochs;
    /** Whether a fault-tolerance scale_down removed ranks, which ends elastic scaling for the pool. */
    private volatile boolean ranksRemoved;
    /** The fault-tolerance state per data-parallel rank; removed ranks leave their entry behind as the record. */
    private final Map<Integer, FaultState> faultStates = new java.util.concurrent.ConcurrentHashMap<>();
    /** Runs a fault-tolerance instruction off the HTTP thread, one at a time. */
    private final java.util.concurrent.ExecutorService controlExecutor =
            java.util.concurrent.Executors.newSingleThreadExecutor(task -> {
                Thread thread = new Thread(task, "neuralfree-server-control");
                thread.setDaemon(true);
                return thread;
            });
    /** Listeners that follow the pool's shape, run whenever {@link #installPool} replaces it. */
    private final List<Runnable> poolListeners = new java.util.concurrent.CopyOnWriteArrayList<>();
    /** The media inputs: the projector of {@code neuralfree.server.mmproj}, or disabled without one. */
    private final ChatMedia media;
    /** The server-executed tools of the launch (built-ins and MCP servers), empty when none are enabled. */
    private final ToolsRegistry tools;
    /** The pooling, classification and scoring entry points. */
    private final EnginePooling pooling;
    /** The slot save, restore, erase and status operations. */
    private final EngineSlots slots;
    /** The sleep, pause, KV release and checkpoint operations of the pool. */
    private final EnginePower power;
    /** The live weight transfer operations of the pool (the RL trainer-to-inference weight sync). */
    private final EngineWeightTransfer weightTransfer;

    /** Disaggregated prefill/decode over the pool, or null when no {@code kv_transfer_config} is set. */
    private final KvTransferService kvTransfer;
    /** The chat, tokenizer and tool surfaces. */
    private final EngineChat chat;
    /** The per-request LoRA adapters of the pool. */
    private final LoraAdmin loras;
    /**
     * The CLIP dual encoder when the model path is a two-tower clip file or a Hugging Face {@code CLIPModel} /
     * {@code SiglipModel} checkpoint (vLLM's {@code CLIPEmbeddingModel} / {@code SiglipEmbeddingModel}): it serves
     * the pooling routes without replicas; null for every other model.
     */
    private final ClipDualEncoder dual;
    /** The pool wakes in flight: while one runs, no replica's idle sleep may fire (another may still re-open). */
    private final AtomicInteger waking = new AtomicInteger();
    /** Orders the projector's release against its reopen: each decides on the replicas' state under it. */
    private final Object projectorLock = new Object();
    /** The generator file of {@code neuralfree.server.mmproj} (llama.cpp's {@code --mmproj}), or null. */
    private final String mmprojPath;
    /** The text-to-speech generator over the primary replica's model; null until the first speech request. */
    private volatile SpeechGenerator speech;

    /**
     * The serving setup a sleeping engine keeps answering {@code /props} with (llama.cpp serves its cached meta
     * while the context is destroyed): captured while the model was open.
     *
     * @param chatTemplateCaps the capabilities of the chat template ({@link TemplateCaps})
     * @param contextLength the planned context of the engine, else the training one
     * @param maxModelLength the context a request may fill, as {@link #maxModelLength} reports while awake
     * @param backend the backend API the model ran on ({@code "opencl"}, {@code "cuda"}, {@code "cpu"}, ...)
     * @param gpuFallbackReason why the model fell back from a GPU load to the CPU engine, or null when it did not
     */
    public record Props(String architecture, String chatTemplate, TemplateCaps chatTemplateCaps, int vocabSize,
                        int slots, int contextLength, int maxModelLength, String backend,
                        String gpuFallbackReason) { }

    /** Opens {@code path} (null keeps the service unloaded) with {@code config} and starts the pool. */
    public EngineService(String path, EngineConfig config) throws IOException {
        this(path, config, ServerOptions.defaults());
    }

    /** {@link #EngineService(String, EngineConfig)} under the server {@code options}, without media input. */
    EngineService(String path, EngineConfig config, ServerOptions options) throws IOException {
        this(path, config, options, MediaOptions.none(), ServerLoras.NONE);
    }

    /**
     * {@link #EngineService(String, EngineConfig, ServerOptions)} with the media inputs of {@code mediaOptions}:
     * its projector opens beside the model, checked against the model's embedding width.
     *
     * @throws IllegalArgumentException when a projector is configured for an unloaded service, or does not fit
     *                                  the model
     */
    EngineService(String path, EngineConfig config, ServerOptions options, MediaOptions mediaOptions)
            throws IOException {
        this(path, config, options, mediaOptions, ServerLoras.NONE);
    }

    /**
     * {@link #EngineService(String, EngineConfig, ServerOptions)} serving the per-request adapters of
     * {@code adapters}: its startup modules load into every replica before it serves.
     *
     * @throws IOException              when the model or a startup adapter does not open
     * @throws IllegalArgumentException when a startup adapter takes a served model's name
     */
    EngineService(String path, EngineConfig config, ServerOptions options, ServerLoras adapters)
            throws IOException {
        this(path, config, options, MediaOptions.none(), adapters);
    }

    /**
     * The media inputs of {@code mediaOptions} and the per-request adapters of {@code adapters} beside the model:
     * the projector opens checked against the model's embedding width, the startup adapter modules load into every
     * replica before it serves.
     *
     * @throws IOException              when the model, a projector or a startup adapter does not open
     * @throws IllegalArgumentException when a projector is configured for an unloaded service or does not fit
     *                                  the model, or a startup adapter takes a served model's name
     */
    EngineService(String path, EngineConfig config, ServerOptions options, MediaOptions mediaOptions,
                  ServerLoras adapters) throws IOException {
        Objects.requireNonNull(mediaOptions, "mediaOptions");
        this.pooling = new EnginePooling(this);
        this.slots = new EngineSlots(this);
        this.power = new EnginePower(this);
        this.weightTransfer = new EngineWeightTransfer(this);
        this.chat = new EngineChat(this);
        boolean twoTower = path != null && !path.isBlank() && ClipDualEncoder.isTwoTower(path);
        EngineConfig configured = withAdapterSlots(Objects.requireNonNull(config, "config"), adapters);
        boolean adaptersConfigured = !configured.loras().isEmpty() || !adapters.modules().isEmpty()
                || adapters.runtimeUpdating();
        this.config = !twoTower && mediaOptions.mmproj() != null && adaptersConfigured
                ? withVisionCatalog(configured, mediaOptions.mmproj()) : configured;
        this.logitsProcessors = LogitsProcessors.forConfig(this.config);
        this.options = Objects.requireNonNull(options, "options");
        this.modelPath = path == null || path.isBlank() ? null : path;
        this.devices = options.devices();
        this.layout = options.dataParallel();
        this.runtimeAdapters = adapters.runtimeUpdating();
        this.mmprojPath = mediaOptions.mmproj();
        this.tools = ToolsRegistry.of(options.tools(), options.mcpServers());
        this.served = ServedModels.of(options.servedNames(), path);
        this.responses = new ResponseStore(options.responseStore());
        LoraCatalog catalog = new LoraCatalog();
        this.catalog = catalog;
        for (ServerLoras.Module m : adapters.modules()) {
            if (served.names().contains(m.name())) {
                throw new IllegalArgumentException("--lora-modules: the adapter name " + m.name()
                        + " conflicts with a served base model");
            }
            catalog.put(new LoraCatalog.Adapter(m.name(), m.path(), 0.0f));
        }
        if (path == null || path.isBlank()) {
            if (mediaOptions.mmproj() != null) {
                throw new IllegalArgumentException(MediaOptions.MMPROJ_PROPERTY + " needs a model to serve beside");
            }
            if (options.kvTransfer() != null) {
                throw new IllegalArgumentException(ServerOptions.KV_TRANSFER_CONFIG_PROPERTY + " needs a served model");
            }
            this.kvTransfer = null;
            this.replicas = List.of();
            this.scheduler = null;
            this.media = ChatMedia.disabled();
            this.loras = new LoraAdmin(this::replicas, served, catalog, adapters.runtimeUpdating());
            this.dual = null;
            return;
        }
        if (twoTower) {
            if (mediaOptions.mmproj() != null) {
                throw new IllegalArgumentException("'" + path + "' is a CLIP dual encoder that embeds images through"
                        + " its own vision tower; unset " + MediaOptions.MMPROJ_PROPERTY);
            }
            this.replicas = List.of();
            this.scheduler = null;
            if (!config.loras().isEmpty() || !adapters.modules().isEmpty() || adapters.runtimeUpdating()) {
                throw new IllegalArgumentException("'" + path + "' is a CLIP dual encoder, which serves no LoRA"
                        + " adapters; unset --lora (" + EngineConfig.PROP_LORA + "), --lora-modules and runtime"
                        + " LoRA updating");
            }
            if (options.kvTransfer() != null) {
                throw new IllegalArgumentException(ServerOptions.KV_TRANSFER_CONFIG_PROPERTY + " needs a generating"
                        + " model; '" + path + "' is a CLIP dual encoder");
            }
            this.kvTransfer = null;
            this.media = ChatMedia.open(mediaOptions, 0, false, token -> false);
            this.loras = new LoraAdmin(this::replicas, served, catalog, false);
            this.dual = ClipDualEncoder.open(path, config.sourceOptions(),
                    onEngineBackend(mediaOptions, config).encoder());
            return;
        }
        this.dual = null;
        List<EngineReplica> opened = new ArrayList<>(options.replicas());
        try {
            for (int i = 0; i < options.replicas(); i++) {
                opened.add(openReplica(i, options.globalRank(i)));
            }
        } catch (IOException | RuntimeException e) {
            closeAll(opened);
            throw e;
        }
        this.replicas = List.copyOf(opened);
        this.scheduler = new DpScheduler(opened.toArray(new EngineReplica[0]), options.dataParallel().balance());
        try {
            this.media = openMedia(onEngineBackend(mediaOptions, this.config), opened.get(0));
        } catch (IOException | RuntimeException e) {
            closeAll(opened);
            throw e;
        }
        releaseProjectorIfAsleep(); // a replica that idle-slept while the projector opened skipped the release
        this.loras = new LoraAdmin(this::replicas, served, catalog, adapters.runtimeUpdating());
        try {
            this.kvTransfer = KvTransferService.open(options.kvTransfer());
        } catch (IOException | RuntimeException e) {
            closeAll(opened);
            try {
                media.close();
            } catch (IOException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /**
     * Opens the replica at pool position {@code index} serving data-parallel rank {@code rank}: its device filter
     * from {@code neuralfree.server.devices} ({@link #devicesFor(int)}), the model path and settings of the service,
     * and the per-request adapter catalog.
     *
     * @throws IOException              when the model or a startup adapter does not open
     * @throws IllegalArgumentException when no device filter is configured for that pool position
     */
    private EngineReplica openReplica(int index, int rank) throws IOException {
        String device = devicesFor(index);
        EngineConfig replicaConfig = device == null ? this.config : this.config.toBuilder().deviceFilter(device).build();
        return new EngineReplica(index, rank, device, modelPath, replicaConfig, options,
                () -> waking.get() > 0, catalog, this::releaseProjectorIfAsleep);
    }

    /**
     * The device filter of pool position {@code index}, or null for the default device. A configured list names one
     * device per pool position; a position past its end (an elastic scale) refuses by name instead of opening on a
     * device the launch did not assign.
     */
    private String devicesFor(int index) {
        if (devices == null) {
            return null;
        }
        if (index >= devices.size()) {
            throw new IllegalArgumentException("replica " + index + " has no device: "
                    + ServerOptions.DEVICES_PROPERTY + " names " + devices.size() + " device(s), so the pool"
                    + " cannot grow past " + devices.size() + " replicas; list a device for every replica the"
                    + " deployment may scale to");
        }
        return devices.get(index);
    }

    /**
     * Installs a changed replica pool and its layout, and routes over it from now on. The caller holds
     * {@link #poolLock}.
     */
    private void installPool(List<EngineReplica> next, DpLayout nextLayout) {
        if (next.isEmpty()) {
            throw new IllegalArgumentException("the pool needs at least one replica");
        }
        this.replicas = List.copyOf(next);
        this.layout = nextLayout;
        if (scheduler != null) {
            scheduler.replace(next.toArray(new EngineReplica[0]));
        }
        for (Runnable listener : poolListeners) {
            listener.run();
        }
    }

    /**
     * Registers {@code listener} to run whenever the pool's shape changes (an elastic scale or a fault-tolerance
     * rank removal); it runs once now too, so a reader registers against the current pool.
     */
    void onPoolChange(Runnable listener) {
        Objects.requireNonNull(listener, "listener");
        poolListeners.add(listener);
        listener.run();
    }

    /**
     * {@code config} with the per-request adapter slots on ({@code LoraConfig.enabled}, vLLM's {@code --enable-lora})
     * when the service serves adapters per request — startup adapters a request rescales ({@code --lora}, as
     * llama.cpp's {@code lora} list does), modules or runtime updating — so a GPU load reserves their device memory;
     * unchanged otherwise.
     */
    static EngineConfig withAdapterSlots(EngineConfig config, ServerLoras adapters) {
        boolean perRequest = !config.loras().isEmpty() || !adapters.modules().isEmpty() || adapters.runtimeUpdating();
        if (!perRequest || config.loraConfig().enabled()) {
            return config;
        }
        return config.toBuilder().loraConfig(config.loraConfig().toBuilder().enabled(true).build()).build();
    }

    /**
     * {@code config} whose LoRA adapter loads resolve targets absent from the text model against the tensor table
     * of the mmproj file at {@code mmproj}: the adapter's vision-encoder and connector pairs then attach to the
     * media projector instead of failing as unknown base tensors.
     *
     * @throws IOException when the projector file cannot be read
     */
    static EngineConfig withVisionCatalog(EngineConfig config, String mmproj) throws IOException {
        TensorCatalog catalog;
        try (ModelSource source = ModelSources.parseGguf(mmproj)) {
            catalog = TensorCatalog.of(source);
        }
        ModelSourceOptions options = config.sourceOptions().toBuilder().loraVisionCatalog(catalog).build();
        return config.toBuilder().sourceOptions(options).build();
    }

    /**
     * {@code options} with the vision encoder following the engine's backend mode unless a device selector names
     * its device, as llama.cpp's mmproj device follows {@code --device}: a CPU engine keeps the encoder on the CPU,
     * a GPU engine offloads it through the same API ({@link MmprojOptions#withBackend}). A
     * {@link MmEncoderTpMode#DATA} encoder with no rank group of its own takes the engine's devices
     * ({@link EngineConfig#gpuDevices()}), the ranks the language model runs on, as vLLM spreads the encoder's
     * batch over its tensor-parallel ranks. A CLIP dual encoder's towers follow the same choice.
     */
    static MediaOptions onEngineBackend(MediaOptions options, EngineConfig config) {
        MmprojOptions encoder = options.encoder();
        MmprojOptions resolved = encoder.device() != null ? encoder : encoder.withBackend(config.backend());
        if (resolved.tpMode() == MmEncoderTpMode.DATA && resolved.devices().isEmpty() && resolved.offload()
                && !config.gpuDevices().isEmpty()) {
            resolved = resolved.withDevices(config.gpuDevices());
        }
        return resolved == encoder ? options : options.withEncoder(resolved);
    }

    /** The media inputs over the first replica's model: its embedding width, RoPE layout and vocabulary. */
    private static ChatMedia openMedia(MediaOptions options, EngineReplica replica) throws IOException {
        if (options.mmproj() == null) {
            return ChatMedia.open(options, 0, false, token -> false);
        }
        EngineModel model = replica.model();
        boolean mrope = model.spec().traits().ropeType().usesSections();
        GGUFTokenizer tokenizer = model.tokenizer();
        return ChatMedia.open(options, model.embeddingLength(), mrope, token -> tokenizer.getTokenId(token) >= 0,
                startupAdapters(model));
    }

    /**
     * The startup adapters ({@code --lora}) of the open {@code model} at their scales: the ones its vision encoder
     * applies to every encoded row. A per-request adapter applies to vision rows only when a request selects it,
     * which the media path does not do.
     */
    private static List<LoraAttachment.Binding> startupAdapters(EngineModel model) {
        LoraRegistry registry = model.loraRegistry();
        if (registry == null) {
            return List.of();
        }
        List<LoraAttachment.Binding> out = new ArrayList<>();
        for (LoraHandle handle : registry.list()) {
            if (handle.startup()) {
                out.add(new LoraAttachment.Binding(handle.adapter(), handle.scale()));
            }
        }
        return out;
    }

    /** The media inputs: image and audio content parts, {@code multimodal_data}. */
    ChatMedia media() {
        return media;
    }

    /** The CLIP dual encoder the pooling routes serve, or null for every other model. */
    ClipDualEncoder dual() {
        return dual;
    }

    /** The replicas, in their pool order as of the call; empty while unloaded. */
    List<EngineReplica> replicas() {
        return replicas;
    }

    /** The pool wakes in flight, which keep the idle sleep and the projector release off. */
    AtomicInteger waking() {
        return waking;
    }

    /**
     * Drops the projector's media cache, its encodings and its processed media (vLLM's {@code POST /reset_mm_cache});
     * nothing to do without a projector.
     */
    public void resetMediaCache() {
        media.clearCache();
    }

    /**
     * Drops every encoder output (vLLM's {@code POST /reset_encoder_cache}): the projector's cached encodings and,
     * on each replica's pump thread, its batch engine's shared encoder cache, so the next request of a medium runs
     * its encoder again (from the processed media the projector still holds). Running requests keep the rows they
     * hold, and the prefix cache's blocks stay. A replica without an engine (unloaded or released by a sleep) has
     * nothing to drop.
     *
     * @throws IllegalStateException when a pump did not answer within the request budget
     */
    public void resetEncoderCache() throws InterruptedException {
        media.clearEncodings();
        for (EngineReplica replica : replicas) {
            replica.resetEncoderCacheOp();
        }
    }

    /** The media items looked up in the projector's cache (vLLM's {@code mm_cache_queries}), 0 without one. */
    public long mediaCacheQueries() {
        return media.cacheQueries();
    }

    /** The media items the projector's cache served without decoding (vLLM's {@code mm_cache_hits}). */
    public long mediaCacheHits() {
        return media.cacheHits();
    }

    /** The bytes the projector's media cache holds, encodings and processed media. */
    public long mediaCacheUsageBytes() {
        return media.cacheUsageBytes();
    }

    /** The media items this consumer loaded from the shared encoder cache (vLLM's EC transfer loads). */
    public long mediaSharedLoads() {
        return media.sharedLoads();
    }

    /** The media items this producer saved to the shared encoder cache (vLLM's EC transfer saves). */
    public long mediaSharedSaves() {
        return media.sharedSaves();
    }

    /**
     * Releases the media projector and its cached encodings once no replica holds its model open and no pool wake
     * is in flight: the hook each replica runs on its pump thread after a sleep released its weights. A release
     * failure is logged; the projector then stays loaded.
     */
    void releaseProjectorIfAsleep() {
        ChatMedia current = media;
        if (current == null || replicas == null) {
            return; // a replica slept while the service was still opening; the projector opens after it
        }
        synchronized (projectorLock) {
            if (waking.get() > 0 || openReplica() != null) {
                return;
            }
            current.clearCache();
            SpeechGenerator generator = speech;
            speech = null;
            if (generator != null) {
                try {
                    generator.close();
                } catch (RuntimeException e) {
                    LOGGER.log(Level.WARNING, "speech generator release failed", e);
                }
            }
            try {
                current.release();
            } catch (IOException | RuntimeException e) {
                LOGGER.log(Level.WARNING, "media projector release failed", e);
            }
        }
    }

    /**
     * Re-opens the released media projector once a replica holds its model open again (the wake).
     *
     * @throws IllegalStateException when the projector does not reopen (the message names the failure)
     */
    void reopenProjector() {
        if (!media.released()) {
            return;
        }
        synchronized (projectorLock) {
            if (openReplica() == null) {
                return;
            }
            try {
                media.reopen();
            } catch (IOException | RuntimeException e) {
                throw new IllegalStateException("the media projector did not reopen: " + e.getMessage(), e);
            }
        }
    }

    /**
     * Re-opens the released projector inside a pool wake window, so no replica idle-sleeps while it loads. A
     * failure is logged and leaves the projector released: a text request serves on, a media request refuses with
     * the release's 503, and the next request retries.
     */
    private void reopenProjectorQuietly() {
        waking.incrementAndGet();
        try {
            reopenProjector();
        } catch (IllegalStateException e) {
            LOGGER.log(Level.WARNING, e.getMessage(), e);
        } finally {
            waking.decrementAndGet();
        }
    }

    private static void closeAll(List<EngineReplica> opened) {
        for (EngineReplica replica : opened) {
            try {
                replica.close();
            } catch (RuntimeException e) {
                LOGGER.log(Level.WARNING, "replica close failed", e);
            }
        }
    }

    /** The first replica, the pool's source for the model metadata every replica shares. */
    EngineReplica primary() {
        requireLoaded();
        if (dual != null) {
            throw new UnsupportedOperationException(dualRefusal());
        }
        return replicas.get(0);
    }

    /**
     * The media-session adapter key of the request's {@code selection}, resolved on the primary replica's pump
     * thread; null when the request names no adapter, so its media encodes and caches under the startup adapters'
     * keys (or none). The engine resolves adapter slots there, and the key names the adapters the projector's
     * session applies while the request's media encode.
     *
     * @throws EngineUnavailableException   when no model is loaded, or the pump missed its request budget
     * @throws UnsupportedOperationException when the model runs no per-request adapters
     * @throws IllegalArgumentException      when the selection names an adapter that is not resident
     */
    LoraSlotKey resolveMediaLoras(LoraSelection selection) {
        if (selection == null) {
            return null;
        }
        try {
            return primary().loras().resolveMediaLoras(selection);
        } catch (IOException e) {
            throw new EngineUnavailableException("resolving the request's LoRA adapters failed: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("resolving the request's LoRA adapters was interrupted", e);
        }
    }

    /** The opened model path, or null when unloaded ({@code neuralfree.server.model} unset). */
    public String modelPath() {
        if (dual != null) {
            return dual.path();
        }
        return replicas.isEmpty() ? null : replicas.get(0).modelPath();
    }

    /** The replicas' counters and state, aggregated (the counters add; vLLM's DP-wide figures). */
    public BatchStats stats() {
        BatchStats sum = BatchStats.EMPTY;
        for (EngineReplica replica : replicas) {
            sum = sum.plus(replica.stats());
        }
        return sum;
    }

    /** Each replica's own counters and state, in pool order. */
    public List<BatchStats> replicaStats() {
        List<BatchStats> out = new ArrayList<>(replicas.size());
        for (EngineReplica replica : replicas) {
            out.add(replica.stats());
        }
        return out;
    }

    /** One replica's own counters and state. */
    BatchStats replicaStats(int index) {
        return replicas.get(index).stats();
    }

    /** How many replicas the pool serves the model with, 0 while unloaded. */
    public int replicaCount() {
        return replicas.size();
    }

    /**
     * The device memory the pool's open models hold, in bytes: the sum of what every replica allocated on its
     * device (the {@code --models-memory-margin} gauge). 0 while unloaded, on the CPU, or when the backend does
     * not track allocations.
     */
    public long allocatedDeviceBytes() {
        long sum = 0;
        for (EngineReplica replica : replicas) {
            sum += replica.deviceBytesAllocated();
        }
        return sum;
    }

    /** A replica's device-name filter, or null when it runs on the default device. */
    String replicaDevice(int index) {
        return replicas.get(index).device();
    }

    /** The prompt tokens of the generations routed to a replica and not yet terminal. */
    long replicaPromptTokensInFlight(int index) {
        return replicas.get(index).promptTokensInFlight();
    }

    /** The deployment's global data-parallel size, 0 while unloaded (this server then serves no rank). */
    public int dataParallelSize() {
        return replicas.isEmpty() ? 0 : layout.size();
    }

    /** The global data-parallel rank a replica serves. */
    public int replicaRank(int index) {
        return replicas.get(index).rank();
    }

    /** The layout the pool serves today, as an elastic scale may have replaced the launch's. */
    DpLayout dataParallelLayout() {
        return layout;
    }

    /**
     * Clears the prefix cache of every replica on its pump thread ({@code BatchEngine.clearPrefixCache}): false
     * while sequences of some replica hold pages of its memory, as vLLM's reset reports (the idle replicas clear
     * regardless); true when every replica had nothing cached or cleared. A sleeping replica holds no cached
     * prefix and stays asleep.
     *
     * @throws IllegalStateException when the service is unloaded, or a pump did not answer within the request
     *                               budget
     */
    public boolean resetPrefixCache() throws InterruptedException {
        if (replicas.isEmpty()) {
            throw new IllegalStateException("no model loaded: set -Dneuralfree.server.model=<gguf path>");
        }
        boolean success = true;
        for (EngineReplica replica : replicas) {
            success &= replica.resetPrefixCacheOp();
        }
        return success;
    }

    /**
     * Aborts every live request of the pool (each submission carries its own abort switch): the engines stop them
     * with {@code CANCELLED} at their next step. Returns how many requests were queued, running or preempted.
     */
    public int abortAll() {
        int aborted = 0;
        for (EngineReplica replica : replicas) {
            aborted += replica.abortAll();
        }
        return aborted;
    }

    /**
     * The requests in flight over the pool, vLLM's {@code server_load}: generations submitted and not yet terminal
     * (queued for a pump, waiting, running or preempted) plus the pooling forwards callers wait on.
     */
    public int load() {
        int load = 0;
        for (EngineReplica replica : replicas) {
            load += replica.load();
        }
        return load;
    }

    /**
     * Whether the server is ready (the probes' fact): every replica holds its model open, or the pool sleeps and
     * a wake restores it — llama.cpp keeps a sleeping server ready. Asking does not wake an idle sleep.
     */
    public boolean ready() {
        return loaded() || power.anySleeping();
    }

    /** The server settings. */
    ServerOptions options() {
        return options;
    }

    /** The custom logits processors the engine configuration loads; empty when it loads none. */
    LogitsProcessors logitsProcessors() {
        return logitsProcessors;
    }

    /** The server-executed tools of the launch (built-ins and MCP servers), empty when the launch enabled none. */
    ToolsRegistry tools() {
        return tools;
    }

    /** The names the server answers to. */
    ServedModels served() {
        return served;
    }

    /** The per-request LoRA adapters of the pool. */
    LoraAdmin loras() {
        return loras;
    }

    /** The pool's live weight transfer operations (the RL trainer-to-inference weight sync). */
    EngineWeightTransfer weightTransfer() {
        return weightTransfer;
    }

    /** The Responses API store ({@link ServerOptions#responseStore}). */
    ResponseStore responses() {
        return responses;
    }

    /** The live chat completions the control endpoint resolves by id. */
    LiveCompletions completions() {
        return completions;
    }

    /** The engine configuration the service opened with. */
    EngineConfig config() {
        return config;
    }

    /** The order the replicas' batch engines serve requests in ({@code neuralfree.scheduling.policy}). */
    SchedulingPolicy schedulingPolicy() {
        return config.memory().schedulingPolicy();
    }

    /** The model's tokenizer ({@code GGUFTokenizer}), the primary replica's; a loaded service carries one. */
    GGUFTokenizer tokenizer() {
        if (dual != null) {
            return dual.tokenizer();
        }
        return primary().model().tokenizer();
    }

    /** The model's architecture name, or null while no model is open (unloaded, or a sleep released it). */
    public String architecture() {
        if (dual != null) {
            return dual.architecture();
        }
        return openReplica() == null ? null : openReplica().model().architecture();
    }

    /**
     * The backend API the open model runs on ({@code "opencl"}, {@code "cuda"}, {@code "vulkan"}, {@code "cpu"}), or
     * null while no model is open. The actual engine, not {@code config().backend()}: a requested GPU whose load fell
     * back reports {@code cpu}, with {@link #gpuFallbackReason()} naming why.
     */
    public String backendName() {
        EngineReplica open = openReplica();
        return open == null ? null : open.backendName();
    }

    /** Why the open model fell back from a GPU load to the CPU engine, or null when it did not. */
    public String gpuFallbackReason() {
        EngineReplica open = openReplica();
        return open == null ? null : open.gpuFallbackReason();
    }

    /** The pool's sequence-slot count, summed over the replicas; 0 before a first generation builds an engine. */
    public int slots() {
        int sum = 0;
        for (EngineReplica replica : replicas) {
            sum += replica.slots();
        }
        return sum;
    }

    /** Each replica's own sequence-slot count, in pool order (0 before its engine builds). */
    public int[] replicaSlots() {
        int[] out = new int[replicas.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = replicas.get(i).slots();
        }
        return out;
    }

    /** The memory's context length, or 0 while the batch engine is not built. */
    public int contextLength() {
        if (dual != null) {
            return dual.contextLength();
        }
        EngineReplica replica = replicas.isEmpty() ? null : primary();
        return replica == null ? 0 : replica.contextLength();
    }

    /** Whether every replica holds its model open and runs its pump thread. */
    public boolean loaded() {
        if (dual != null) {
            return true;
        }
        if (replicas.isEmpty()) {
            return false;
        }
        for (EngineReplica replica : replicas) {
            if (!replica.alive() || !replica.open()) {
                return false;
            }
        }
        return true;
    }

    // ------------------------------------------------------------------ cluster control

    /** The elastic-scale lifecycle of the pool, as vLLM's external elastic EP phases name it. */
    enum ElasticPhase {
        IDLE, PREPARING, COMMITTING, FAILED, COMPLETED;

        /** The phase's lowercase API name. */
        String label() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /** The elastic-scale state {@code POST /is_scaling_elastic_ep} reports. */
    record ElasticStatus(boolean scaling, String phase, String epoch, String error,
                         Integer requestedDataParallelSize) { }

    /**
     * One replica's fault-tolerance view: the engine's data-parallel rank, its health ({@code healthy},
     * {@code unhealthy} or {@code dead}), the failure that made it unhealthy, and the state of the fault-tolerance
     * instruction that last touched it ({@code recovering}, {@code failed} or null).
     */
    record ReplicaHealth(int id, String status, String faultInfo, String ftState, String lastFtRequestId,
                         String ftError) { }

    /** One rank's fault-tolerance bookkeeping; the instruction's state survives while the rank is served. */
    private static final class FaultState {
        private volatile String state;
        private volatile String lastRequestId;
        private volatile String error;
    }

    /** The elastic-scale state the route reports. */
    ElasticStatus elasticStatus() {
        ElasticPhase phase = elasticPhase;
        boolean scaling = phase == ElasticPhase.PREPARING || phase == ElasticPhase.COMMITTING;
        return new ElasticStatus(scaling, phase.label(), elasticEpoch, elasticError, elasticRequestedSize);
    }

    /** Whether an elastic scale may run: every rank of the deployment is local, and no scale_down removed any. */
    boolean elasticScalable() {
        List<EngineReplica> current = replicas;
        return dual == null && !current.isEmpty() && !ranksRemoved
                && layout.startRank() == 0 && current.size() == layout.size();
    }

    /**
     * Resizes the pool to {@code newSize} replicas behind the one service (vLLM's {@code POST /scale_elastic_ep}):
     * a shrink first stops routing to the removed replicas and drains their requests for at most
     * {@code drainMillis}, then closes them; a grow opens the new replicas before routing to them. The global
     * data-parallel size becomes {@code newSize} because the server holds every rank of the deployment. A failure
     * leaves the pool as it was (a shrink that timed out restores the routing) and records the phase as failed.
     *
     * @throws IllegalArgumentException when the pool cannot scale elastically (unloaded, a dual encoder, ranks held
     *                                  by other servers, or a rank removed by fault tolerance), or the new size is
     *                                  unusable
     * @throws java.util.concurrent.TimeoutException when the removed replicas did not drain within the timeout
     * @throws IllegalStateException when opening the new replicas fails
     */
    void scaleElasticEp(int newSize, long drainMillis) throws InterruptedException,
            java.util.concurrent.TimeoutException {
        if (newSize < 1) {
            throw new IllegalArgumentException("new_data_parallel_size must be a positive integer, got " + newSize);
        }
        if (drainMillis < 1) {
            throw new IllegalArgumentException("drain_timeout must be a positive integer, got " + drainMillis);
        }
        synchronized (poolLock) {
            if (dual != null) {
                throw new IllegalArgumentException("elastic scaling needs a generation model; this server serves a"
                        + " CLIP dual encoder");
            }
            if (replicas.isEmpty()) {
                throw new IllegalArgumentException("elastic scaling needs a loaded model: set "
                        + "-Dneuralfree.server.model=<gguf path>");
            }
            if (!elasticScalable()) {
                throw new IllegalArgumentException("elastic scaling resizes a server that holds every rank of the"
                        + " deployment: this one holds " + replicas.size() + " of " + layout.size() + " rank(s)"
                        + (ranksRemoved ? " after a fault-tolerance scale_down" : "")
                        + ", so its external deployment owns the topology");
            }
            elasticEpoch = Long.toString(++elasticEpochs);
            elasticRequestedSize = newSize;
            elasticError = null;
            elasticPhase = ElasticPhase.PREPARING;
            try {
                if (newSize < replicas.size()) {
                    shrinkPool(newSize, drainMillis);
                } else if (newSize > replicas.size()) {
                    growPool(newSize);
                }
                elasticPhase = ElasticPhase.COMMITTING;
                elasticPhase = ElasticPhase.COMPLETED;
            } catch (InterruptedException e) {
                elasticError = "the scale was interrupted";
                elasticPhase = ElasticPhase.FAILED;
                throw e;
            } catch (java.util.concurrent.TimeoutException | RuntimeException e) {
                elasticError = e.getMessage();
                elasticPhase = ElasticPhase.FAILED;
                throw e;
            }
        }
    }

    /** Stops routing to the trailing replicas, drains them, closes them, and installs the survivors. */
    private void shrinkPool(int newSize, long drainMillis)
            throws InterruptedException, java.util.concurrent.TimeoutException {
        List<EngineReplica> current = replicas;
        List<EngineReplica> survivors = new ArrayList<>(current.subList(0, newSize));
        List<EngineReplica> removed = new ArrayList<>(current.subList(newSize, current.size()));
        scheduler.replace(survivors.toArray(new EngineReplica[0]));
        try {
            waitForDrain(removed, drainMillis);
        } catch (InterruptedException | java.util.concurrent.TimeoutException | RuntimeException e) {
            scheduler.replace(current.toArray(new EngineReplica[0]));
            throw e;
        }
        closeAll(removed);
        installPool(survivors, new DpLayout(newSize, layout.startRank(), layout.balance()));
    }

    /**
     * Waits until none of {@code replicas} holds a request. A request that picked a removed replica just before the
     * scheduler swap needs a settling window to land, so the check runs again after a short pause.
     */
    private static void waitForDrain(List<EngineReplica> replicas, long drainMillis)
            throws InterruptedException, java.util.concurrent.TimeoutException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(drainMillis);
        while (draining(replicas)) {
            if (System.nanoTime() >= deadline) {
                throw new java.util.concurrent.TimeoutException("the requests on the replicas being removed did"
                        + " not drain within " + drainMillis + " ms");
            }
            Thread.sleep(10);
        }
        Thread.sleep(50);
        while (draining(replicas)) {
            if (System.nanoTime() >= deadline) {
                throw new java.util.concurrent.TimeoutException("the requests on the replicas being removed did"
                        + " not drain within " + drainMillis + " ms");
            }
            Thread.sleep(10);
        }
    }

    /** Whether any of {@code replicas} still holds a request. */
    private static boolean draining(List<EngineReplica> replicas) {
        for (EngineReplica replica : replicas) {
            if (replica.load() > 0) {
                return true;
            }
        }
        return false;
    }

    /** Opens the replicas past the pool's end, then installs the grown pool. */
    private void growPool(int newSize) {
        List<EngineReplica> current = replicas;
        List<EngineReplica> next = new ArrayList<>(current);
        try {
            for (int i = current.size(); i < newSize; i++) {
                next.add(openReplica(i, layout.startRank() + i));
            }
        } catch (IllegalArgumentException e) {
            closeFrom(next, current.size(), e);
            throw e;
        } catch (IOException | RuntimeException e) {
            closeFrom(next, current.size(), e);
            throw new IllegalStateException("the scale to " + newSize + " replicas failed: " + e.getMessage(), e);
        }
        installPool(next, new DpLayout(newSize, layout.startRank(), layout.balance()));
    }

    /** Closes the replicas {@code next} gained past {@code from}, suppressing their failures on {@code cause}. */
    private static void closeFrom(List<EngineReplica> next, int from, Exception cause) {
        for (int i = from; i < next.size(); i++) {
            try {
                next.get(i).close();
            } catch (RuntimeException closeFailure) {
                cause.addSuppressed(closeFailure);
            }
        }
    }

    /** The fault-tolerance status of every replica still served, in pool order (vLLM's engine status list). */
    List<ReplicaHealth> faultToleranceEngines() {
        List<EngineReplica> current = replicas;
        List<ReplicaHealth> out = new ArrayList<>(current.size());
        for (EngineReplica replica : current) {
            FaultState state = faultStates.get(replica.rank());
            String fault = replica.alive() ? replica.faultInfo() : "the replica's pump thread stopped";
            String status = !replica.alive() ? "dead" : fault == null ? "healthy" : "unhealthy";
            out.add(new ReplicaHealth(replica.rank(), status, fault,
                    state == null ? null : state.state,
                    state == null ? null : state.lastRequestId,
                    state == null ? null : state.error));
        }
        return out;
    }

    /**
     * Runs one fault-tolerance instruction off the HTTP thread (vLLM answers 202 and the caller polls the status).
     * {@code retry} recovers every unhealthy or dead replica; {@code scale_down} closes the ranks
     * {@code params.removed_dp_ranks} names, which must be unhealthy or dead, and drops them from the pool while
     * the deployment's global size keeps their slots.
     *
     * @throws IllegalArgumentException for an unknown instruction or unusable parameters, before anything changes
     */
    void submitFaultTolerance(String instruction, Map<?, ?> params, String requestId) {
        if ("retry".equals(instruction)) {
            submitRetry(requestId);
            return;
        }
        if ("scale_down".equals(instruction)) {
            submitScaleDown(params, requestId);
            return;
        }
        throw new IllegalArgumentException("Invalid instruction: '" + instruction + "'.");
    }

    /** Marks the unhealthy replicas recovering and schedules their recovery. */
    private void submitRetry(String requestId) {
        synchronized (poolLock) {
            List<EngineReplica> targets = new ArrayList<>();
            for (EngineReplica replica : replicas) {
                if (!replica.alive() || replica.faultInfo() != null) {
                    targets.add(replica);
                }
            }
            markFaultState(requestId, targets, null);
            controlExecutor.execute(() -> runRetry(targets, requestId));
        }
    }

    /** Recovers {@code targets} one by one; a failure records the error on the replicas it could not recover. */
    private void runRetry(List<EngineReplica> targets, String requestId) {
        synchronized (poolLock) {
            List<EngineReplica> live = new ArrayList<>(targets.size());
            for (EngineReplica replica : targets) {
                if (replicas.contains(replica)) {
                    live.add(replica);
                }
            }
            if (live.isEmpty()) {
                markFaultFailure(requestId, replicas, "no unhealthy engine to recover: every replica is healthy");
                return;
            }
            List<EngineReplica> failed = new ArrayList<>();
            String error = null;
            for (EngineReplica replica : live) {
                try {
                    if (!replica.alive()) {
                        replaceReplica(replica);
                    } else {
                        replica.recoverOp();
                    }
                } catch (IOException | InterruptedException | RuntimeException e) {
                    if (e instanceof InterruptedException) {
                        Thread.currentThread().interrupt();
                    }
                    failed.add(replica);
                    error = error == null ? e.getMessage() : error;
                }
            }
            if (failed.isEmpty()) {
                clearFaultState(live);
            } else {
                markFaultFailure(requestId, failed, error);
            }
        }
    }

    /** Opens a replacement for a dead replica at its pool position and rank, then closes the dead one. */
    private void replaceReplica(EngineReplica dead) throws IOException {
        List<EngineReplica> current = replicas;
        int position = current.indexOf(dead);
        if (position < 0) {
            throw new IllegalStateException("the replica of rank " + dead.rank() + " is no longer in the pool");
        }
        EngineReplica replacement = openReplica(position, dead.rank());
        List<EngineReplica> next = new ArrayList<>(current);
        next.set(position, replacement);
        installPool(next, layout);
        dead.close();
    }

    /** Marks the removed ranks recovering and schedules their removal. */
    private void submitScaleDown(Map<?, ?> params, String requestId) {
        synchronized (poolLock) {
            List<Integer> removed = removedRanks(params);
            List<EngineReplica> targets = new ArrayList<>();
            for (EngineReplica replica : replicas) {
                if (removed.contains(replica.rank())) {
                    targets.add(replica);
                }
            }
            markFaultState(requestId, targets, null);
            controlExecutor.execute(() -> runScaleDown(targets, requestId));
        }
    }

    /** The {@code removed_dp_ranks} of a scale_down, validated against the ranks still served and their health. */
    private List<Integer> removedRanks(Map<?, ?> params) {
        Object raw = params.get("removed_dp_ranks");
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw new IllegalArgumentException("scale_down needs 'removed_dp_ranks', a non-empty array of"
                    + " data-parallel ranks");
        }
        List<EngineReplica> current = replicas;
        List<Integer> removed = new ArrayList<>(list.size());
        for (Object entry : list) {
            if (!(entry instanceof Number number) || number.intValue() != number.doubleValue()) {
                throw new IllegalArgumentException("removed_dp_ranks must hold whole numbers, got " + entry);
            }
            int rank = number.intValue();
            EngineReplica replica = null;
            for (EngineReplica candidate : current) {
                if (candidate.rank() == rank) {
                    replica = candidate;
                    break;
                }
            }
            if (replica == null) {
                throw new IllegalArgumentException("removed_dp_ranks names rank " + rank + ", which this server does"
                        + " not serve (it serves " + ranks(current) + ")");
            }
            if (replica.alive() && replica.faultInfo() == null) {
                throw new IllegalArgumentException("removed_dp_ranks names rank " + rank + ", which is healthy;"
                        + " scale_down removes failed ranks only");
            }
            if (!removed.contains(rank)) {
                removed.add(rank);
            }
        }
        if (current.size() <= removed.size()) {
            throw new IllegalArgumentException("scale_down cannot remove every rank: the pool needs at least one"
                    + " replica to keep serving");
        }
        return removed;
    }

    /** Recovers {@code removed} as far as closing goes: the ranks leave the pool and the layout keeps theirs. */
    private void runScaleDown(List<EngineReplica> removed, String requestId) {
        synchronized (poolLock) {
            List<EngineReplica> current = replicas;
            List<EngineReplica> next = new ArrayList<>(current.size());
            for (EngineReplica replica : current) {
                if (!removed.contains(replica)) {
                    next.add(replica);
                }
            }
            if (next.isEmpty()) {
                markFaultFailure(requestId, removed, "scale_down would remove every replica");
                return;
            }
            for (EngineReplica replica : removed) {
                if (replicas.contains(replica) && replica.alive() && replica.faultInfo() == null) {
                    markFaultFailure(requestId, removed, "rank " + replica.rank() + " became healthy before the"
                            + " scale_down ran; failed ranks only");
                    return;
                }
            }
            closeAll(removed);
            for (EngineReplica replica : removed) {
                faultStates.remove(replica.rank());
            }
            ranksRemoved = true;
            installPool(next, layout);
        }
    }

    /** The ranks {@code replicas} serve, comma-separated for a refusal. */
    private static String ranks(List<EngineReplica> replicas) {
        StringBuilder out = new StringBuilder();
        for (EngineReplica replica : replicas) {
            if (out.length() > 0) {
                out.append(", ");
            }
            out.append(replica.rank());
        }
        return out.toString();
    }

    /** Records {@code state} ({@code recovering}) on {@code targets}; {@code error} null on the recovering mark. */
    private void markFaultState(String requestId, List<EngineReplica> targets, String error) {
        for (EngineReplica replica : targets) {
            FaultState state = faultStates.computeIfAbsent(replica.rank(), rank -> new FaultState());
            state.state = "recovering";
            state.lastRequestId = requestId;
            state.error = error;
        }
    }

    /** Records a failed instruction on {@code targets}. */
    private void markFaultFailure(String requestId, List<EngineReplica> targets, String error) {
        for (EngineReplica replica : targets) {
            FaultState state = faultStates.computeIfAbsent(replica.rank(), rank -> new FaultState());
            state.state = "failed";
            state.lastRequestId = requestId;
            state.error = error;
        }
    }

    /** Clears the instruction state of recovered {@code targets}, as vLLM's healthy push replaces it. */
    private void clearFaultState(List<EngineReplica> targets) {
        for (EngineReplica replica : targets) {
            faultStates.remove(replica.rank());
        }
    }

    // ------------------------------------------------------------------ chat, tokenizer and tools

    /**
     * Renders the chat request through the model's template, around the autoparser's derived markers when the
     * final message is continued; see {@link EngineChat#formatChat}.
     */
    public String formatChat(ChatRequest request) {
        return chat.formatChat(request, continuationAnalysis(request));
    }

    /** {@link #formatChat(ChatRequest)} for the multimodal path (media parts as media markers). */
    public String formatChat(ChatRequest request, boolean mediaMarkers) {
        return chat.formatChat(request, mediaMarkers, continuationAnalysis(request));
    }

    /**
     * The autoparser's analysis when the configured tool parser selects it for the model's template, else null:
     * the template is one no hand-written format recognizes, or {@code autoparser} is named. It brings the
     * template's derived reasoning markers, structured-output grammar and literal stops.
     */
    public AutoParserFormat autoparser() {
        requireLoaded();
        return AutoParserFormat.selected(config.toolParser(), primary().model().tokenizer().chatTemplate());
    }

    /** The analysis a continued final message is written around, or null without a continuation or autoparser. */
    private AutoParserAnalysis continuationAnalysis(ChatRequest request) {
        if (request == null || !request.continueFinalMessage()) {
            return null;
        }
        AutoParserFormat autoparser = autoparser();
        return autoparser == null ? null : autoparser.analysis();
    }

    /** The reasoning filter the engine routes output with. */
    public ReasoningFilter reasoning() {
        return chat.reasoning();
    }

    /** How the tools of {@code request} reach its output. */
    public ToolRouting toolRouting(ChatRequest request, String renderedPrompt,
                                   ParallelToolCalls parallelToolCalls) {
        return chat.toolRouting(request, renderedPrompt, parallelToolCalls);
    }

    /** {@link #toolRouting(ChatRequest, String, ParallelToolCalls)} over a per-request reasoning filter. */
    public ToolRouting toolRouting(ChatRequest request, String renderedPrompt,
                                   ParallelToolCalls parallelToolCalls, ReasoningFilter filter) {
        return chat.toolRouting(request, renderedPrompt, parallelToolCalls, filter);
    }

    /** The response-template parsing a request answers under ({@code hf} or {@code beam}), or null. */
    public TemplateParsing templateParsing(ReasoningFilter filter) {
        return chat.templateParsing(filter);
    }

    /** The grammar the {@code beam} parser holds a chat request's output to, or null. */
    public BeamGrammar beamGrammar(TemplateParsing parsing, ChatRequest chatRequest, String renderedPrompt,
                                   ParallelToolCalls parallelToolCalls, ConstraintSpec output,
                                   ReasoningFilter filter, boolean includeReasoning) {
        return chat.beamGrammar(parsing, chatRequest, renderedPrompt, parallelToolCalls, output, filter,
                includeReasoning);
    }

    /** The model's chat template source, or null when it carries none. */
    public String chatTemplate() {
        return chat.chatTemplate();
    }

    /** The capabilities of the model's chat template. */
    public TemplateCaps chatTemplateCaps() {
        return chat.chatTemplateCaps();
    }

    /** The {@code parallel_tool_calls} default of the model's template. */
    public boolean defaultParallelToolCalls() {
        return chat.defaultParallelToolCalls();
    }

    /** The {@code parallel_tool_calls} default of the template {@code request} renders with. */
    public boolean defaultParallelToolCalls(ChatRequest request) {
        return chat.defaultParallelToolCalls(request);
    }

    /** The capabilities of the template {@code request} renders with. */
    public TemplateCaps chatTemplateCaps(ChatRequest request) {
        return chat.chatTemplateCaps(request);
    }

    /** The vocabulary size, the bound of a token id. */
    public int vocabSize() {
        return chat.vocabSize();
    }

    /** Decodes token ids to text ({@code GGUFTokenizer.decode}). */
    public String decode(int[] tokens) {
        return chat.decode(tokens);
    }

    /** Tokenizes text ({@code GGUFTokenizer.encode(text, addSpecial, parseSpecial)}). */
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        return chat.encode(text, addSpecial, parseSpecial);
    }

    /** Decodes one token id to its text (the logprobs rendering). */
    public String tokenText(int tokenId) {
        return chat.tokenText(tokenId);
    }

    /** The raw bytes one token renders to, special tokens included ({@code GGUFTokenizer.pieceWithSpecial}). */
    public byte[] tokenPiece(int tokenId) {
        return chat.tokenPiece(tokenId);
    }

    /** The BOS token a raw prompt starts with when the vocabulary adds one, else -1. */
    int addedBos() {
        return chat.addedBos();
    }

    /** The FIM tokens the model lacks for {@code /infill}, or null when it has them all. */
    public String infillMissingTokens() {
        return chat.infillMissingTokens();
    }

    /** The fill-in-the-middle prompt of llama.cpp's {@code /infill}. */
    public int[] infillPrompt(int[] prefix, int[] suffix, List<InfillPrompt.Chunk> extra, int[] prompt,
                              int nPredict) {
        return chat.infillPrompt(prefix, suffix, extra, prompt, nPredict);
    }

    /** The context a request may fill: the planned memory context once the engine runs, else the training one. */
    public int maxModelLength() {
        requireLoaded();
        if (dual != null) {
            return dual.contextLength();
        }
        EngineReplica replica = primary();
        BatchEngine engine = replica.builtEngine();
        return engine != null ? engine.contextLength() : replica.model().kvCacheContext();
    }

    /** A slot id outside the pool's slots (llama.cpp's {@code Invalid slot ID}). */
    public static final class UnknownSlotException extends IllegalArgumentException {
        UnknownSlotException(int slot, int slots) {
            super("slot " + slot + " is outside the pool's " + slots + " slots");
        }
    }

    /** Writes slot {@code slot}'s state to {@code file}; see {@link EngineSlots#saveSlot}. */
    public SlotResult saveSlot(int slot, Path file) throws IOException {
        return slots.saveSlot(slot, file);
    }

    /** Reads the slot file at {@code file} into slot {@code slot}; see {@link EngineSlots#restoreSlot}. */
    public SlotResult restoreSlot(int slot, Path file) throws IOException {
        return slots.restoreSlot(slot, file);
    }

    /** Erases slot {@code slot}'s state; see {@link EngineSlots#eraseSlot}. */
    public int eraseSlot(int slot) throws IOException {
        return slots.eraseSlot(slot);
    }

    /** The status of every slot of the pool; see {@link EngineSlots#slotStatuses}. */
    public List<SlotStatus> slotStatuses() throws InterruptedException {
        return slots.slotStatuses();
    }

    /**
     * The speech decoding geometry the loaded model declares ({@link LLMModel#speechFormat}), or null when it
     * cannot turn its features into a waveform.
     */
    public SpeechFormat speechFormat() {
        if (dual != null) {
            return null;
        }
        return primary().model().speechFormat();
    }

    /**
     * The waveform of audio codec codes through the loaded model's speech decoder
     * ({@link LLMModel#synthesizeSpeech}).
     *
     * @throws UnsupportedOperationException when the model is not a speech decoder or declares no geometry
     */
    public AudioClip synthesizeSpeech(int[] codes) {
        if (dual != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        return primary().model().synthesizeSpeech(codes);
    }

    /**
     * Whether the loaded model turns text into audio, i.e. whether {@link #synthesizeSpeech(String, SpeechOptions)}
     * has a speech-generator backbone to reach. A qwen3tts, pockettts or lfm3a backbone answers true even before
     * its generator file is opened, so the route can prefer text over the integer-code path and let the generator
     * name the missing file.
     */
    public boolean supportsTextSpeech() {
        if (dual != null || replicas == null || replicas.isEmpty()) {
            return false;
        }
        Arch arch = primary().model().spec().arch();
        return arch == Arch.QWEN3TTS || arch == Arch.POCKETTTS || arch == Arch.LFM3;
    }

    /**
     * Synthesizes speech from {@code text} through the loaded model's speech generator: a qwen3tts backbone with
     * the code predictor and code2wav decoder of the {@code neuralfree.server.mmproj} generator file, a pockettts
     * backbone with that file's flow network and mimi decoder, or an lfm3a backbone with its MaskGIT depthformer
     * and liquid-codec decoder. Runs on the primary replica's engine thread once its batch engine is idle,
     * because the pipeline resets and drives the model's forward memory.
     *
     * @param text    the text to speak
     * @param options the language, voice, sampling and frame cap
     * @return the mono waveform at the generator file's sample rate
     * @throws UnsupportedOperationException when the model has no speech-generator backbone, no generator file is
     *                                       configured, or the lfm3 omni backbone carries no speech chat
     *                                       template (the message names the missing piece)
     * @throws IllegalArgumentException      when {@code text} or an option is invalid, or the rendered prompt does
     *                                       not open the omni pipeline's fresh turn
     * @throws IOException                   when the generation fails or the engine stays busy past its budget
     */
    public AudioClip synthesizeSpeech(String text, SpeechOptions options) throws IOException {
        if (dual != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        SpeechGenerator generator = speechGenerator();
        if (generator instanceof Lfm3aSpeechGenerator omni) {
            // llama.cpp speaks the tts template's render, which must end in the audio start token
            // (server-context.cpp:6477-6478, mtmd-helper-gen.cpp:1337): the render already carries it, so the
            // fresh turn runs over it directly instead of the generator appending one to the raw text
            GGUFTokenizer tokenizer = primary().model().tokenizer();
            if (tokenizer.ttsTemplate() == null) {
                throw new UnsupportedOperationException("lfm3a text-to-speech needs the "
                        + GGUFTokenizer.KEY_TTS_TEMPLATE + " chat template: the omni pipeline speaks the prompt"
                        + " rendered from it, and the render ends in the audio start token");
            }
            String rendered = tokenizer.formatTtsPrompt(text, options.language(), options.voice());
            int[] prompt = tokenizer.encode(rendered, false, true);
            return primary().onPump(() -> omni.generate(prompt, Lfm3aSpeechGenerator.Mode.AUDIO, false, options,
                    null).audio());
        }
        return primary().onPump(() -> generator.synthesize(text, options));
    }

    /**
     * {@link #synthesizeSpeech(String, SpeechOptions)} delivered in decoder chunks: the generator hands each
     * clip to {@code sink} as it decodes it, on the primary replica's engine thread. The clips concatenate to the
     * one-shot result. {@code sink} may throw to stop the generation (a disconnected client); the throw reaches
     * this call.
     *
     * @throws UnsupportedOperationException when the model has no speech-generator backbone, or no generator file
     *                                       is configured (the message names the missing piece)
     * @throws IllegalArgumentException      when {@code text} or an option is invalid
     * @throws IOException                   when the generation fails or the engine stays busy past its budget
     */
    public void synthesizeSpeech(String text, SpeechOptions options, Consumer<AudioClip> sink) throws IOException {
        if (dual != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        Objects.requireNonNull(sink, "sink");
        SpeechGenerator generator = speechGenerator();
        primary().onPump(() -> {
            generator.synthesize(text, options, sink);
            return null;
        });
    }

    /**
     * The sample rate of the text-to-speech generator's waveform; opens the generator file on first use, so a
     * missing backbone or file refuses here by name before a streamed response starts.
     *
     * @throws UnsupportedOperationException when the model has no speech-generator backbone, or no generator file
     *                                       is configured
     * @throws IOException                   when the generator file cannot be read
     */
    public int speechSampleRate() throws IOException {
        if (dual != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        return speechGenerator().sampleRate();
    }

    /**
     * {@link #synthesizeSpeech(int[])} delivered in chunks through the model's streaming speech decoder.
     *
     * @throws UnsupportedOperationException when the model is not a speech decoder or declares no geometry
     */
    public void synthesizeSpeech(int[] codes, Consumer<AudioClip> sink) {
        if (dual != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        Objects.requireNonNull(sink, "sink");
        primary().model().synthesizeSpeech(codes, sink);
    }

    /**
     * The sample rate of the code-to-waveform decoder, refusing like {@link #synthesizeSpeech(int[])} when the
     * model has none so a streamed response starts only for a model that can fill it.
     *
     * @throws UnsupportedOperationException when the model is not a speech decoder or declares no geometry
     */
    public int speechCodeSampleRate() {
        if (dual != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        return primary().model().requireSpeechFormat().sampleRate();
    }

    /**
     * One omni chat turn: the conversation renders through the model's conversation template
     * ({@link GGUFTokenizer#formatSpeechChat}, with the request's {@code language} and {@code voice} bound), and
     * the lfm3a omni pipeline answers it. An AUDIO-only turn is fresh and stops at the end-of-audio frame, its
     * prompt ending in the audio start token; a MIXED turn continues the conversation, the pipeline appending the
     * mixed start token, and interleaves text and audio blocks until {@code <|im_end|>} or the step cap
     * (mtmd-helper-gen.cpp:1320-1352, 1392-1450). Runs on the primary replica's engine thread like
     * {@link #synthesizeSpeech(String, SpeechOptions)}, because the pipeline drives the model's forward memory.
     *
     * @param chat       the conversation to answer
     * @param mode       {@link Lfm3aSpeechGenerator.Mode#AUDIO} for speech only, {@link Lfm3aSpeechGenerator.Mode#MIXED}
     *                   for interleaved text and speech
     * @param continuing true for a turn that continues the conversation (the pipeline appends the mode's switch
     *                   token and lets the response continue in text after the audio); false for a fresh AUDIO turn
     * @param options    the text-step sampler, the language and voice variables and the step cap
     * @param sink       receives the turn's text pieces and audio chunks as they happen, or null for none
     * @return the turn's text, whole waveform and prompt length
     * @throws UnsupportedOperationException when the loaded model has no speech-generator backbone, no generator
     *                                       file, or when its generator is not the lfm3a omni pipeline (the message
     *                                       names the missing piece)
     * @throws IllegalArgumentException      when the conversation does not render, or a fresh prompt is not an
     *                                       audio turn
     * @throws IOException                   when the generation fails or the engine stays busy past its budget
     */
    public Lfm3aSpeechGenerator.OmniTurn omniTurn(ChatRequest chat, Lfm3aSpeechGenerator.Mode mode, boolean continuing,
                                                SpeechOptions options, Lfm3aSpeechGenerator.OmniSink sink)
            throws IOException {
        if (dual != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        if (replicas == null || replicas.isEmpty()) {
            throw new UnsupportedOperationException("no model loaded: chat completions audio output needs the lfm3a"
                    + " omni pipeline");
        }
        Arch arch = primary().model().spec().arch();
        if (arch != Arch.LFM3) {
            throw new UnsupportedOperationException("chat completions audio output needs the lfm3a omni pipeline:"
                    + (arch == Arch.QWEN3TTS
                            ? " the loaded qwen3tts backbone speaks through /v1/audio/speech, which does not run"
                              + " the omni chat turn"
                            : " the loaded " + arch.ggufName() + " model has no speech-generator backbone; a"
                              + " qwen3tts or lfm3 backbone with its generator file speaks"));
        }
        SpeechGenerator generator = speechGenerator();
        if (!(generator instanceof Lfm3aSpeechGenerator omni)) {
            throw new UnsupportedOperationException("chat completions audio output needs the lfm3a omni pipeline;"
                    + " the loaded speech generator is " + generator.name() + ", which speaks through"
                    + " /v1/audio/speech");
        }
        GGUFTokenizer tokenizer = primary().model().tokenizer();
        String rendered = tokenizer.formatSpeechChat(chat, options.language(), options.voice());
        int[] prompt = tokenizer.encode(rendered, false, true);
        if (!continuing) {
            // the pipeline decides a fresh turn's mode from the last prompt token, and only an audio turn is valid
            // (mtmd-helper-gen.cpp:1336): the render must end in audio_start
            if (mode != Lfm3aSpeechGenerator.Mode.AUDIO) {
                throw new IllegalArgumentException("a fresh omni turn opens in audio; a " + mode
                        + " turn continues the conversation");
            }
            int audioStart = tokenizer.getTokenId(Lfm3aSpeechGenerator.AUDIO_START_PIECE);
            int[] withStart = Arrays.copyOf(prompt, prompt.length + 1);
            withStart[prompt.length] = audioStart;
            prompt = withStart;
        }
        int[] tokens = prompt;
        return primary().onPump(() -> omni.generate(tokens, mode, continuing, options, sink));
    }

    /**
     * The speech generator of the pool, opened on first use from the generator file and the primary replica's
     * model and cached; models without a speech-generator backbone are refused by name.
     *
     * @throws UnsupportedOperationException when the model has no speech-generator backbone, or no generator file
     *                                       is configured
     * @throws IOException                   when the generator file cannot be read
     */
    private SpeechGenerator speechGenerator() throws IOException {
        synchronized (projectorLock) {
            if (speech != null) {
                return speech;
            }
            if (replicas == null || replicas.isEmpty()) {
                throw new UnsupportedOperationException("no model loaded: speech generation needs a qwen3tts,"
                        + " pockettts or lfm3a backbone");
            }
            EngineModel model = primary().model();
            Arch arch = model.spec().arch();
            if (arch != Arch.QWEN3TTS && arch != Arch.POCKETTTS && arch != Arch.LFM3) {
                throw new UnsupportedOperationException("this server has no speech-generator backbone: the loaded "
                        + arch.ggufName() + " model cannot turn text into audio; a qwen3tts, pockettts or lfm3a"
                        + " backbone with its generator file does");
            }
            if (mmprojPath == null) {
                throw new UnsupportedOperationException("speech generation needs the generator file: set "
                        + MediaOptions.MMPROJ_PROPERTY + " (llama.cpp --mmproj) to the " + generatorFileOf(arch));
            }
            speech = openSpeechGenerator(model, arch);
            return speech;
        }
    }

    /** The generator file an arch's speech pipeline reads, for the missing-file message. */
    private static String generatorFileOf(Arch arch) {
        switch (arch) {
            case QWEN3TTS:
                return "qwen3tts code predictor and code2wav GGUF";
            case POCKETTTS:
                return "pockettts flow network and mimi decoder GGUF";
            case LFM3:
                return "lfm3a MaskGIT depthformer and liquid-codec GGUF";
            default:
                return arch.ggufName() + " generator GGUF";
        }
    }

    /** Opens the generator file of {@code arch} and binds it to {@code model}; closes it on a binding failure. */
    private SpeechGenerator openSpeechGenerator(EngineModel model, Arch arch) throws IOException {
        switch (arch) {
            case QWEN3TTS: {
                Qwen3TtsClip clip = Qwen3TtsClip.open(mmprojPath, config.sourceOptions());
                try {
                    return Qwen3TtsSpeechGenerator.open(model, clip);
                } catch (RuntimeException | Error e) {
                    clip.close();
                    throw e;
                }
            }
            case POCKETTTS: {
                PocketTtsClip clip = PocketTtsClip.open(mmprojPath, config.sourceOptions());
                try {
                    return PocketTtsSpeechGenerator.open(model, clip);
                } catch (IOException | RuntimeException | Error e) {
                    clip.close();
                    throw e;
                }
            }
            case LFM3: {
                Lfm3aGenClip clip = Lfm3aGenClip.open(mmprojPath, config.sourceOptions());
                try {
                    return Lfm3aSpeechGenerator.open(model, clip);
                } catch (IOException | RuntimeException | Error e) {
                    clip.close();
                    throw e;
                }
            }
            default:
                throw new UnsupportedOperationException("the " + arch.ggufName()
                        + " backbone has no speech generator");
        }
    }

    // ------------------------------------------------------------------ pooling, classification and scoring

    /** The pooled embedding of one input under {@code lora}; see {@link EnginePooling#embed}. */
    public float[] embed(LoraSelection lora, int[] tokens) throws IOException {
        return pooling.embed(lora, tokens);
    }

    /** The pooled embeddings of several inputs under {@code lora} over one replica. */
    public float[][] embedAll(LoraSelection lora, int[][] inputs) throws IOException {
        return pooling.embedAll(lora, inputs);
    }

    /** The embedding of one image through a CLIP dual encoder's vision tower. */
    public float[] embedImage(LoraSelection lora, String reference, String field) {
        return pooling.embedImage(lora, reference, field);
    }

    /** True when the model is a CLIP dual encoder. */
    public boolean dualEncoder() {
        return pooling.dualEncoder();
    }

    /** The bi-encoder score of a cosine similarity, through the dual encoder's activation when it has one. */
    public double biEncoderScore(double cosine) {
        return pooling.biEncoderScore(cosine);
    }

    /** The pooled embedding of one media prompt. */
    public float[] embedMedia(LoraSelection lora, MediaPrompt prompt) throws IOException {
        return pooling.embedMedia(lora, prompt);
    }

    /** The RANK classifier's relevance score of one media prompt. */
    public float rerankScoreMedia(LoraSelection lora, MediaPrompt prompt) throws IOException {
        return pooling.rerankScoreMedia(lora, prompt);
    }

    /** {@link #rerankScoreMedia} with the classifier's final probability activation applied or skipped. */
    public float rerankScoreMedia(LoraSelection lora, MediaPrompt prompt, boolean activation) throws IOException {
        return pooling.rerankScoreMedia(lora, prompt, activation);
    }

    /** Whether the model's embeddings are per token. */
    public boolean perTokenEmbeddings() {
        return pooling.perTokenEmbeddings();
    }

    /** The per-token embeddings of one input under {@code lora}, for NONE pooling. */
    public float[][] embedTokens(LoraSelection lora, int[] tokens) throws IOException {
        return pooling.embedTokens(lora, tokens);
    }

    /** The per-token embeddings of several inputs over one replica. */
    public float[][][] embedTokensAll(LoraSelection lora, int[][] inputs) throws IOException {
        return pooling.embedTokensAll(lora, inputs);
    }

    /** Whether the model classifies per token. */
    public boolean tokenClassifier() {
        return pooling.tokenClassifier();
    }

    /** Whether the model pools a vector or row before classifying. */
    public boolean sequenceClassifier() {
        return pooling.sequenceClassifier();
    }

    /** Whether the model answers {@code token_classify}. */
    public boolean canTokenClassify() {
        return pooling.canTokenClassify();
    }

    /**
     * Whether the model stores a separate per-token score projection ({@code cls.sparse}, vLLM's BGE-M3
     * {@code sparse_linear}); its {@code token_classify} rows are that projection's ReLU scores.
     */
    public boolean tokenScoreProjection() {
        return pooling.tokenScoreProjection();
    }

    /**
     * Whether the model stores a separate per-token embedding projection ({@code cls.colbert}, vLLM's BGE-M3
     * {@code colbert_linear}); {@code token_embed} answers its rows beside the pooled {@code embed}.
     */
    public boolean tokenEmbeddingProjection() {
        return pooling.tokenEmbeddingProjection();
    }

    /**
     * The process-reward step filter a {@code token_classify} request omits ({@code classifier.step_tag_id}), or
     * -1 when the model sets none.
     */
    public int defaultStepTagId() {
        return pooling.defaultStepTagId();
    }

    /** The leading per-token boundary token id ({@code classifier.bos_token}), or -1. */
    public int tokenBoundaryBos() {
        return pooling.tokenBoundaryBos();
    }

    /** The trailing per-token boundary token id ({@code classifier.eos_token}), or -1. */
    public int tokenBoundaryEos() {
        return pooling.tokenBoundaryEos();
    }

    /**
     * The kind of sparse pooling over the per-token classifier scores ({@code EncodeGate.sparsePooling}), the
     * vLLM SPLADE reduction, or null when the model pools none of it.
     */
    public String sparsePooling() {
        requireLoaded();
        if (dual != null) {
            return null;
        }
        return com.eonmux.neuralfree.gguf.ops.heads.EncodeGate.sparsePooling(primary().model().spec());
    }

    /** The per-token classifier scores of one input ({@code EngineModel.tokenClassify}) under {@code lora}. */
    public float[][] tokenClassify(LoraSelection lora, int[] tokens) throws IOException {
        return pooling.tokenClassify(lora, tokens);
    }

    /** The per-token classifier scores of several inputs over one replica. */
    public float[][][] tokenClassifyAll(LoraSelection lora, int[][] inputs) throws IOException {
        return pooling.tokenClassifyAll(lora, inputs);
    }

    /** The rerank input tokens of one query-document pair. */
    public int[] rerankInput(String query, String document) {
        return pooling.rerankInput(query, document);
    }

    /** The rerank input of an already tokenized query and document. */
    public int[] rerankInput(int[] query, int[] document) {
        return pooling.rerankInput(query, document);
    }

    /** The rerank input of a query, document and task instruction. */
    public int[] rerankInput(String query, String document, String instruction) {
        return pooling.rerankInput(query, document, instruction);
    }

    /** The relevance score of one rerank input under {@code lora}. */
    public float rerankScore(LoraSelection lora, int[] tokens) throws IOException {
        return pooling.rerankScore(lora, tokens);
    }

    /** {@link #rerankScore} with the classifier's final probability activation applied or skipped. */
    public float rerankScore(LoraSelection lora, int[] tokens, boolean activation) throws IOException {
        return pooling.rerankScore(lora, tokens, activation);
    }

    /** The relevance scores of several rerank inputs over one replica. */
    public float[] rerankScores(LoraSelection lora, int[][] inputs) throws IOException {
        return pooling.rerankScores(lora, inputs);
    }

    /** {@link #rerankScores} with the final probability activation applied or skipped per input. */
    public float[] rerankScores(LoraSelection lora, int[][] inputs, boolean activation) throws IOException {
        return pooling.rerankScores(lora, inputs, activation);
    }

    /** The model's pooling kind. */
    public com.eonmux.neuralfree.gguf.arch.PoolingKind poolingKind() {
        return pooling.poolingKind();
    }

    /** Whether the model carries a decision head scoring per token. */
    public boolean decisionHead() {
        return pooling.decisionHead();
    }

    /**
     * The decision head's type ({@code <arch>.decision.type}), or the architecture's own head family when the
     * model carries no metadata key; null when the model has no decision readout: the families of
     * {@link DecisionFamily} serve the decision routes.
     */
    public String decisionType() {
        requireLoaded();
        if (dual != null) {
            return null;
        }
        com.eonmux.neuralfree.gguf.spec.ModelSpec spec = primary().model().spec();
        String type = spec.hparams().decisionType();
        if (type != null) {
            return type;
        }
        return switch (spec.arch()) {
            case VLLM_DECISION2 -> "vllm-decision2";
            case CLEF -> "clef";
            case MODERN_BERT -> spec.hparams().nLayerDecision() > 0 ? "laya" : null;
            case LFM2 -> spec.hparams().nLayerDecision() > 0 ? "lfm2-d1" : null;
            default -> null;
        };
    }

    /** The marker head's question-and-options token budget ({@code decision.max_head_tokens}), 0 when absent. */
    public int decisionMaxHeadTokens() {
        requireLoaded();
        if (dual != null) {
            return 0;
        }
        return primary().model().spec().hparams().decisionMaxHeadTokens();
    }

    /** The decision head's per-type temperatures ({@code decision.temperature.<name>}), unmodifiable. */
    public Map<String, Float> decisionTemperatures() {
        requireLoaded();
        if (dual != null) {
            return Map.of();
        }
        return primary().model().spec().hparams().decisionTemperatures();
    }

    /** The decision head's score offsets by level count ({@code decision.score_bias.<n>}), unmodifiable. */
    public Map<Integer, float[]> decisionScoreBias() {
        requireLoaded();
        if (dual != null) {
            return Map.of();
        }
        return primary().model().spec().hparams().decisionScoreBias();
    }

    /** The decision head's option labels ({@code decision.labels}), unmodifiable. */
    public List<String> decisionLabels() {
        requireLoaded();
        if (dual != null) {
            return List.of();
        }
        return List.of(primary().model().spec().hparams().decisionLabels());
    }

    /** The classifier output rows of the head ({@code <arch>.n_embd_out}), the embedding size when unset. */
    public int nEmbdOut() {
        requireLoaded();
        if (dual != null) {
            return 0;
        }
        return primary().model().spec().hparams().nEmbdOut();
    }

    /**
     * The decision scores of one ordered prompt under {@code lora}: output row {@code i} holds the score of the
     * {@code i}-th option span of {@code order}.
     */
    public float[] decisionScores(LoraSelection lora, int[] tokens, int[] order) throws IOException {
        return pooling.decisionScores(lora, tokens, order);
    }

    /** What the model is for. */
    public com.eonmux.neuralfree.gguf.arch.SupportClass supportClass() {
        return pooling.supportClass();
    }

    /** The classifier output of one input under {@code lora}. */
    public ClassifierOutput classifyOutput(LoraSelection lora, int[] tokens) throws IOException {
        return pooling.classifyOutput(lora, tokens);
    }

    /** {@link #classifyOutput} with the final probability activation applied or skipped. */
    public ClassifierOutput classifyOutput(LoraSelection lora, int[] tokens, boolean activation) throws IOException {
        return pooling.classifyOutput(lora, tokens, activation);
    }

    /** The classifier outputs of several inputs over one replica. */
    public ClassifierOutput[] classifyOutputsAll(LoraSelection lora, int[][] inputs) throws IOException {
        return pooling.classifyOutputsAll(lora, inputs);
    }

    /** {@link #classifyOutputsAll} with the final probability activation applied or skipped per input. */
    public ClassifierOutput[] classifyOutputsAll(LoraSelection lora, int[][] inputs, boolean activation)
            throws IOException {
        return pooling.classifyOutputsAll(lora, inputs, activation);
    }

    /**
     * The classifier output over a media prompt under {@code lora}: the sequence classifier's scores over the pooled
     * row of the prompt's text and media rows.
     */
    public ClassifierOutput classifyOutputMedia(LoraSelection lora, MediaPrompt prompt, boolean activation)
            throws IOException {
        return pooling.classifyOutputMedia(lora, prompt, activation);
    }

    /** The classifier scores of one input under {@code lora}. */
    public float[] classifyScores(LoraSelection lora, int[] tokens) throws IOException {
        return pooling.classifyScores(lora, tokens);
    }

    /** The classifier scores of several inputs over one replica. */
    public float[][] classifyScoresAll(LoraSelection lora, int[][] inputs) throws IOException {
        return pooling.classifyScoresAll(lora, inputs);
    }

    /** The next-token logits of one prompt under {@code lora}. */
    public float[] nextLogits(LoraSelection lora, int[] tokens) throws IOException {
        return pooling.nextLogits(lora, tokens);
    }

    /** The next-token logits after the last row of a media prompt under {@code lora}. */
    public float[] nextLogits(LoraSelection lora, MediaPrompt prompt) throws IOException {
        return pooling.nextLogits(lora, prompt);
    }

    /**
     * The decision scores of one media prompt under {@code lora}: output row {@code i} holds the score of the
     * {@code i}-th option span of {@code order}, read over the prompt's text and media rows.
     */
    public float[] decisionScores(LoraSelection lora, MediaPrompt prompt, int[] order) throws IOException {
        return pooling.decisionScores(lora, prompt, order);
    }

    /**
     * The replica a pooling forward runs on: the least-loaded one, and when it just idle-slept, the pool wakes and
     * the forward takes an open replica instead.
     */
    EngineReplica poolingReplica() throws IOException {
        requireLoaded();
        EngineReplica replica = scheduler.pick(null);
        if (!replica.open()) {
            wakeIdleReplicas();
            for (EngineReplica open : replicas) {
                if (open.open()) {
                    return open;
                }
            }
            throw new IOException("no replica holds its model open: POST /wake_up to restore the pool");
        }
        return replica;
    }

    // ------------------------------------------------------------------ generation

    /**
     * Enqueues {@code prompt} with its request on the replica the {@link DpScheduler} routes it to; the replica's
     * pump thread registers it with its engine and the returned submission completes when the request reaches a
     * terminal state. An engine refusal (oversize prompt, unusable constraint) lands on {@link Submission#refused()}
     * instead.
     *
     * @throws EngineUnavailableException when the service is unloaded, sleeping behind an explicit sleep, or paused
     *                                     in a rejecting mode
     */
    public Submission submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                             String renderedPrompt, ConstraintSpec constraint) throws InterruptedException {
        return submit(prompt, request, sink, renderedPrompt, constraint, null);
    }

    /**
     * {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec)} with a hook run once
     * when the request reaches its terminal state or is refused, on the replica's pump thread. The replica's abort
     * (a sleep or pause in the abort mode) and {@link #abortAll} cancel the request beside the caller's own
     * cancellation hook.
     */
    public Submission submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                             String renderedPrompt, ConstraintSpec constraint, Runnable onDone)
            throws InterruptedException {
        return submit(prompt, request, sink, renderedPrompt, constraint, null, onDone);
    }

    /**
     * {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec, Runnable)} with a
     * per-request reasoning filter that replaces the configured one for this request (a
     * {@code reasoning_format}); null runs the configured one.
     */
    public Submission submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                             String renderedPrompt, ConstraintSpec constraint, ReasoningFilter filter,
                             Runnable onDone) throws InterruptedException {
        requireLoaded();
        String refusal = servingRefusal();
        if (refusal != null) {
            throw new EngineUnavailableException(refusal);
        }
        EngineReplica replica = scheduler.pick(prompt);
        KvTransferParams transfer = request.kvTransfer();
        if (transfer == null) {
            return new Submission(replica.submit(prompt, request, sink, renderedPrompt, constraint, filter, onDone));
        }
        return submitTransferred(replica, prompt, request, transfer,
                (prefill, terminal, chain) -> replica.submit(prompt, prefill, sink, renderedPrompt, constraint,
                        filter, onDone, terminal, chain));
    }

    /** Submits a request through the replica's token or media path, with the transfer's hooks. */
    @FunctionalInterface
    private interface TransferSubmitter {
        EngineReplica.Submission submit(GenerationRequest request, Consumer<BatchSequence> onTerminal,
                                        byte[] importChain) throws InterruptedException;
    }

    /**
     * The disaggregated prefill/decode path of {@link #submit} and {@link #submitMedia}: a prefill request forces a
     * one-token generation and exports the sequence's cached rows when it retires; a decode request pulls the
     * prefill instance's chain and has it imported into the replica's prefix cache before the request registers, so
     * the prompt adopts it. {@code submitter} is the replica's token or media submission, whichever the request
     * needs.
     *
     * @param prompt the request's prompt tokens (a text-only media prompt's sampler tokens)
     */
    private Submission submitTransferred(EngineReplica replica, int[] prompt, GenerationRequest request,
                                         KvTransferParams transfer, TransferSubmitter submitter)
            throws InterruptedException {
        KvTransferService service = kvTransfer;
        if (service == null) {
            throw new IllegalArgumentException(KvTransferParams.FIELD + " is set, but this server runs no KV"
                    + " transfer connector; start it with " + ServerOptions.KV_TRANSFER_CONFIG_FLAG + " (or set "
                    + ServerOptions.KV_TRANSFER_CONFIG_PROPERTY + ")");
        }
        if (transfer.doRemoteDecode()) {
            KvTransferParams assigned = service.producerParams(transfer, prompt);
            GenerationRequest prefill = request.toBuilder().maxNewTokens(1).build();
            Consumer<BatchSequence> onTerminal = service.exporter(assigned, replica, prompt);
            Supplier<Map<String, Object>> params = () -> service.responseParams(assigned.transferId());
            return new Submission(submitter.submit(prefill, onTerminal, null), params);
        }
        byte[] chain;
        try {
            chain = service.importFor(transfer, prompt);
        } catch (IOException e) {
            throw new java.io.UncheckedIOException("the KV transfer " + transfer.transferId() + " failed: "
                    + e.getMessage(), e);
        }
        boolean imported = chain != null;
        Map<String, Object> answer = service.consumerParams(transfer, imported);
        return new Submission(submitter.submit(request, null, chain), () -> answer);
    }

    /** {@link #submit} for a content-only callback and no constraint. */
    public Submission submit(int[] prompt, GenerationRequest request, TokenCallback callback)
            throws InterruptedException {
        return submit(prompt, request, callback == null ? null
                : ReasoningFilter.contentTo(callback), null, null);
    }

    /**
     * Enqueues a media prompt ({@link MediaPrompt}: text chunks with media or precomputed embedding rows between
     * them — the encoded images and audio of a chat, vLLM's {@code prompt_embeds} and the chat embeds parts): the
     * contract of {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec)}, with the
     * chunks the engine prefills, a per-request reasoning filter (null runs the configured one) and a hook run once
     * at the request's terminal state or refusal, on the replica's pump thread. The scheduler routes it like a
     * token prompt over its rows' prefix-cache keys ({@link DpScheduler#routingRows}), so a conversation's next
     * turn favours the replica whose prefix cache holds its images.
     *
     * @throws EngineUnavailableException when the service is unloaded, sleeping behind an explicit sleep, or paused
     *                                     in a rejecting mode
     */
    public Submission submitMedia(MediaPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                  String renderedPrompt, ConstraintSpec constraint, ReasoningFilter filter,
                                  Runnable onDone) throws InterruptedException {
        KvTransferParams transfer = request.kvTransfer();
        if (transfer != null && !prompt.isTextOnly()) {
            throw new IllegalArgumentException(KvTransferParams.FIELD + " is not supported with media or embedding"
                    + " prompts: a cached block holding media rows cannot leave the memory");
        }
        requireLoaded();
        String refusal = servingRefusal();
        if (refusal != null) {
            throw new EngineUnavailableException(refusal);
        }
        EngineReplica replica = scheduler.pick(DpScheduler.routingRows(prompt));
        if (transfer == null) {
            return new Submission(replica.submitMedia(prompt, request, sink, renderedPrompt, constraint, filter,
                    onDone));
        }
        return submitTransferred(replica, prompt.samplerTokens(), request, transfer,
                (media, terminal, chain) -> replica.submitMedia(prompt, media, sink, renderedPrompt, constraint,
                        filter, onDone, terminal, chain));
    }

    /** The model's embedding width ({@code n_embd}), the width a {@code prompt_embeds} row must hold. */
    public int embeddingLength() {
        return primary().model().embeddingLength();
    }

    // ------------------------------------------------------------------ sleep, pause and checkpoint (the pool)

    /** Sleeps every replica at {@code level}; see {@link EnginePower#sleep}. */
    public void sleep(int level, SleepControl.Mode mode) throws InterruptedException {
        power.sleep(level, mode);
    }

    /** Wakes every sleeping replica; see {@link EnginePower#wakeUp}. */
    public String wakeUp(List<String> tags) throws InterruptedException {
        return power.wakeUp(tags);
    }

    /** Pauses the generation of every replica. */
    public void pause(SleepControl.Mode mode, boolean clearCache) throws InterruptedException {
        power.pause(mode, clearCache);
    }

    /** Resumes the generation after a pause. */
    public void resume() throws InterruptedException {
        power.resume();
    }

    /** Drops the KV cache contents of every replica. */
    public void releaseKvCacheMemory() throws InterruptedException {
        power.releaseKvCacheMemory();
    }

    /** Starts a checkpoint capture on every replica. */
    public String prepareCheckpoint(SleepControl.Mode mode) throws InterruptedException {
        return power.prepareCheckpoint(mode);
    }

    /** Restores a prepared checkpoint on every replica. */
    public String restoreCheckpoint() throws InterruptedException {
        return power.restoreCheckpoint();
    }

    /** Whether any replica holds released memory (a sleep of level 1 or more). */
    public boolean isSleeping() {
        return power.isSleeping();
    }

    /** The deepest sleep level of the pool: 0 scheduling only, 1 the KV cache contents, 2 the weights too. */
    public int sleepLevel() {
        return power.sleepLevel();
    }

    /** Whether a pause is active on any replica. */
    public boolean isPaused() {
        return power.isPaused();
    }

    /** The pool's checkpoint capture state as of now. */
    public CheckpointStatus checkpointStatus() {
        return power.checkpointStatus();
    }

    /**
     * Wakes every idle-sleeping replica and waits for it: each model re-opens on its pump thread. Concurrent
     * callers share one wake per replica; each wait bounds at the request budget.
     */
    void wakeAndWait() {
        wakeIdleReplicas();
    }

    /**
     * Wakes each replica whose idle sleep a request may end (an explicit sleep stays for {@code /wake_up}), then
     * the media projector.
     */
    private void wakeIdleReplicas() {
        waking.incrementAndGet();
        try {
            for (EngineReplica replica : replicas) {
                if (replica.wakesOnRequest()) {
                    replica.wakeAndWait();
                }
            }
            reopenProjectorQuietly();
        } finally {
            waking.decrementAndGet();
        }
    }


    /** The first replica that still holds its model open, or null while none does. */
    private EngineReplica openReplica() {
        for (EngineReplica replica : replicas) {
            if (replica.open()) {
                return replica;
            }
        }
        return null;
    }


    /** Why a route outside the pooling ones refuses the CLIP dual encoder. */
    String dualRefusal() {
        boolean siglip = dual.family() == ClipFamily.SIGLIP;
        return "the model '" + dual.name() + "' is a " + (siglip ? "SigLIP" : "CLIP") + " dual encoder (vLLM's "
                + (siglip ? "SiglipEmbeddingModel" : "CLIPEmbeddingModel") + "); it serves the"
                + " pooling routes /v1/embeddings, /pooling, /score and /rerank only";
    }

    void requireLoaded() {
        if (!loaded() && wakesOnRequest()) {
            wakeIdleReplicas();
        }
        if (!loaded()) {
            String refusal = servingRefusal();
            throw new EngineUnavailableException(refusal != null ? refusal
                    : "no model loaded: set -Dneuralfree.server.model=<gguf path>");
        }
        if (media.released()) {
            reopenProjectorQuietly(); // a reopen that failed on the wake retries with the next request
        }
    }

    /** Why generation must refuse right now (an explicit sleep, a rejecting pause), or null when it is served. */
    String servingRefusal() {
        for (int i = 0; i < replicas.size(); i++) {
            String refusal = replicas.get(i).servingRefusal();
            if (refusal != null) {
                return replicas.size() > 1 ? "replica " + i + " " + refusal : refusal;
            }
        }
        return null;
    }

    /** Whether a request wakes the pool: only an idle sleep does; an explicit one waits for {@code /wake_up}. */
    boolean wakesOnRequest() {
        for (EngineReplica replica : replicas) {
            if (replica.wakesOnRequest()) {
                return true;
            }
        }
        return false;
    }


    /** The checkpoint capture state (vLLM's CheckpointState) and its failure, or a null error. */
    public record CheckpointStatus(String state, String error) { }


    /** The model setup captured while a model was open; null before the first open. */
    public Props capturedProps() {
        for (EngineReplica replica : replicas) {
            if (replica.capturedProps() != null) {
                return replica.capturedProps();
            }
        }
        return null;
    }

    /** The disaggregated prefill/decode service, or null when this server runs no KV transfer connector. */
    KvTransferService kvTransfer() {
        return kvTransfer;
    }

    /** Stops every replica's pump thread, clears the engines and closes the models. */
    @Override
    public void close() {
        controlExecutor.shutdownNow();
        synchronized (projectorLock) {
            SpeechGenerator generator = speech;
            speech = null;
            if (generator != null) {
                generator.close();
            }
        }
        synchronized (poolLock) {
            closeAll(replicas);
            replicas = List.of();
        }
        if (dual != null) {
            try {
                dual.close();
            } catch (IOException e) {
                LOGGER.log(Level.WARNING, "CLIP dual encoder close failed", e);
            }
        }
        try {
            media.close();
        } catch (IOException e) {
            LOGGER.log(Level.WARNING, "media projector close failed", e);
        }
        tools.close();
        if (kvTransfer != null) {
            kvTransfer.close();
        }
    }
}
