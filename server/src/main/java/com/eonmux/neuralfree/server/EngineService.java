package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
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
import java.util.List;
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
 * <p>The pool's shape is controllable at runtime: {@link EngineScaling#scaleElasticEp} resizes it for vLLM's elastic
 * data-parallel scaling (growing opens the new replicas, shrinking drains and closes the removed ones), and the
 * fault-tolerance routes report each replica's health and recover failed ones or remove their ranks
 * ({@link EngineScaling#submitFaultTolerance}).
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
    final Object poolLock = new Object();
    /** Listeners that follow the pool's shape, run whenever {@link #installPool} replaces it. */
    private final List<Runnable> poolListeners = new java.util.concurrent.CopyOnWriteArrayList<>();
    /** The per-request latencies the generation batches report and the metrics' timers receive. */
    private final RequestLatencies latencies = new RequestLatencies();
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
    /** The pool's elastic data-parallel scaling and fault tolerance. */
    private final EngineScaling scaling;
    /** The pool's text-to-speech and audio-decoding surface. */
    private final EngineSpeech speech;

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
    final Object projectorLock = new Object();
    /** The generator file of {@code neuralfree.server.mmproj} (llama.cpp's {@code --mmproj}), or null. */
    private final String mmprojPath;

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
        this.scaling = new EngineScaling(this);
        this.speech = new EngineSpeech(this);
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
        this.scheduler = new DpScheduler(opened.toArray(new EngineReplica[0]), options.dataParallel().balance(),
                options.maxQueuedRequests());
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
    EngineReplica openReplica(int index, int rank) throws IOException {
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
    void installPool(List<EngineReplica> next, DpLayout nextLayout) {
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

    /** The per-request latency observations of this service's generation requests. */
    RequestLatencies latencies() {
        return latencies;
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
            speech.closeGenerator();
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

    static void closeAll(List<EngineReplica> opened) {
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

    /** The pool's elastic data-parallel scaling and fault tolerance. */
    EngineScaling scaling() {
        return scaling;
    }

    /** The pool's text-to-speech and audio-decoding surface. */
    EngineSpeech speech() {
        return speech;
    }

    /** The generator file of {@code neuralfree.server.mmproj}, or null. */
    String mmprojPath() {
        return mmprojPath;
    }

    /** The scheduler routing requests across the pool, null while unloaded. */
    DpScheduler scheduler() {
        return scheduler;
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

    // ------------------------------------------------------------------ pooling, classification and scoring
    /** The pooling, classification and scoring entry points; see {@link EnginePooling}. */
    EnginePooling pooling() {
        return pooling;
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
     * @throws EngineSaturatedException    when every replica holds {@link ServerOptions#maxQueuedRequests} waiting
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
        return submit(prompt, request, sink, renderedPrompt, constraint, filter, onDone, null);
    }

    /**
     * {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec, ReasoningFilter,
     * Runnable)} as one generation of the request {@code admission} stands for under the queue limit
     * ({@link QueueAdmission}); null for a request of its own.
     */
    public Submission submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                             String renderedPrompt, ConstraintSpec constraint, ReasoningFilter filter,
                             Runnable onDone, QueueAdmission admission) throws InterruptedException {
        requireLoaded();
        String refusal = servingRefusal();
        if (refusal != null) {
            throw new EngineUnavailableException(refusal);
        }
        KvTransferParams transfer = request.kvTransfer();
        return scheduler.queue().submit(admission, scheduler.pick(prompt), prompt, replica -> transfer == null
                ? new Submission(replica.submit(prompt, request, sink, renderedPrompt, constraint, filter, onDone))
                : submitTransferred(replica, prompt, request, transfer,
                        (prefill, terminal, chain) -> replica.submit(prompt, prefill, sink, renderedPrompt,
                                constraint, filter, onDone, terminal, chain)));
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
     * turn favours the replica whose prefix cache holds its images. {@code admission} is the request the generation
     * belongs to under the queue limit ({@link QueueAdmission}), null for a request of its own.
     *
     * @throws EngineUnavailableException when the service is unloaded, sleeping behind an explicit sleep, or paused
     *                                     in a rejecting mode
     * @throws EngineSaturatedException    when every replica holds {@link ServerOptions#maxQueuedRequests} waiting
     */
    public Submission submitMedia(MediaPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                  String renderedPrompt, ConstraintSpec constraint, ReasoningFilter filter,
                                  Runnable onDone, QueueAdmission admission) throws InterruptedException {
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
        int[] routing = DpScheduler.routingRows(prompt);
        return scheduler.queue().submit(admission, scheduler.pick(routing), routing, replica -> transfer == null
                ? new Submission(replica.submitMedia(prompt, request, sink, renderedPrompt, constraint, filter,
                        onDone))
                : submitTransferred(replica, prompt.samplerTokens(), request, transfer,
                        (media, terminal, chain) -> replica.submitMedia(prompt, media, sink, renderedPrompt,
                                constraint, filter, onDone, terminal, chain)));
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
        scaling.close();
        speech.closeGenerator();
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
