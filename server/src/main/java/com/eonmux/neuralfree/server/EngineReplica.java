package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;
import com.eonmux.neuralfree.gguf.engine.PoolingOutput;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import com.eonmux.neuralfree.gguf.engine.batch.SlotStatus;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.engine.lora.LoraHandle;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * One data-parallel replica ({@link EngineService}): an open {@link EngineModel} (the backend
 * {@code neuralfree.backend} selects, CPU or GPU), its {@link BatchEngine} and the
 * single thread that submits into and pumps the engine. HTTP threads call {@link #submit} — the request joins the
 * engine's queue through the pump thread (the engine is not thread-safe) and the caller waits on the returned
 * {@link Submission}'s latch or streams from its sink queue. The replicas share the model path and settings and
 * differ, when set, in their device filter ({@code neuralfree.device}).
 *
 * <p>The pump loop alternates between registering queued submissions and running one engine step, so prompt
 * prefills interleave with decode steps of the running sequences (continuous batching; {@code BatchEngine.pump}
 * runs exactly one step).
 *
 * <p>The replica also sleeps (vLLM's sleep mode, llama.cpp's {@code --sleep-idle-seconds}): a sleep of level 0
 * pauses the scheduling, level 1 drops the KV cache contents, level 2 closes the model and releases its weights
 * and memory too — the wake re-opens it from the model path without restarting the process. {@link SleepControl}
 * holds the replica's sleep, pause and checkpoint state; the pump thread transitions it, skips the engine steps
 * while it is frozen, completes wait-mode drains once the engine quiesced, and enters the idle sleep after
 * {@code neuralfree.server.sleep.idle.seconds} without work. An idle sleep wakes on the next request; an explicit
 * one refuses generation until {@code /wake_up}.
 *
 * <p>The model's per-request LoRA adapters ({@code LoraRegistry}) change on the pump thread too
 * ({@link #adapterOp}), so a load or unload lands between engine steps; every open of the model — the first and
 * the wake from a full sleep — records its startup adapters in the service's {@link LoraCatalog} and loads the
 * catalog's per-request adapters. The live weight transfer writes ({@link EngineWeightTransfer}) land on the
 * pump thread the same way, between engine steps.</p>
 */
final class EngineReplica implements AutoCloseable {

    /**
     * One submitted request: the engine handle plus the latch that fires on its terminal state, the hooks run once
     * at that moment on the pump thread — {@code onTerminal} first, with the handle (the KV transfer export), then
     * {@code onDone}, after every routed piece reached the sink — and before the latch releases its waiters.
     */
    static final class Submission {
        private final CountDownLatch done = new CountDownLatch(1);
        private final AtomicBoolean terminal = new AtomicBoolean();
        private final Runnable onDone;
        private final Consumer<BatchSequence> onTerminal;
        private volatile BatchSequence sequence;
        private volatile Throwable refused;

        Submission(Runnable onDone) {
            this(onDone, null);
        }

        Submission(Runnable onDone, Consumer<BatchSequence> onTerminal) {
            this.onDone = onDone;
            this.onTerminal = onTerminal;
        }

        /** The engine handle, or null until the pump thread registered the request. */
        public BatchSequence sequence() {
            return sequence;
        }

        /** The engine refusal (oversize prompt, unusable constraint), or null once registered. */
        public Throwable refused() {
            return refused;
        }

        /** Waits for the terminal state or the registration refusal; false on timeout. */
        public boolean await(long timeout, TimeUnit unit) throws InterruptedException {
            return done.await(timeout, unit);
        }

        void registered(BatchSequence sequence) {
            this.sequence = sequence;
            sequence.onFinish(() -> finish(sequence));
            if (sequence.finished()) {
                finish(sequence);
            }
        }

        void refused(Throwable e) {
            this.refused = e;
            finish(null);
        }

        private void finish(BatchSequence finished) {
            if (terminal.compareAndSet(false, true)) {
                try {
                    if (onTerminal != null) {
                        try {
                            onTerminal.accept(finished);
                        } catch (RuntimeException e) {
                            LOGGER.log(Level.WARNING, "the engine's terminal hook failed", e);
                        }
                    }
                    if (onDone != null) {
                        onDone.run();
                    }
                } finally {
                    done.countDown();
                }
            }
        }
    }

    /** A change of the model's per-request adapters, run on the pump thread. */
    @FunctionalInterface
    interface RegistryCall {
        void run(LoraRegistry registry) throws IOException;
    }

    private static final Logger LOGGER = Logger.getLogger(EngineReplica.class.getName());

    /** How long the pump waits for work while nothing decodes. */
    private static final long IDLE_POLL_MILLIS = 50;

    /** The replica's position in the pool, naming its pump thread. */
    private final int index;
    /** The global data-parallel rank this replica serves (the pool position past the layout's start rank). */
    private final int rank;
    /** The replica's device-name filter ({@code neuralfree.device}), or null for the default device. */
    private final String device;
    private final String modelPath;
    private final EngineConfig config;
    private final ServerOptions options;
    /** The open model; null while a sleep of level 2 released it (the wake re-opens it). */
    private volatile EngineModel model;
    private final BlockingQueue<Runnable> work = new ArrayBlockingQueue<>(256);
    private final AtomicBoolean running = new AtomicBoolean(true);
    /** The pooling and idle-engine forwards of this replica. */
    private final ReplicaPooling pooling;
    /** The abort switch of every live request; a request removes its own when it leaves the engine. */
    private final Set<AtomicBoolean> aborts = ConcurrentHashMap.newKeySet();
    /** The per-request LoRA adapters and the model open. */
    private final ReplicaLoras loras;
    /** The live weight transfer updater ({@code EngineWeightTransfer}); null until initialized. */
    private volatile WeightUpdater weightTransfer;
    /** Pooling forwards a caller waits on, queued or running. */
    /** The prompt tokens of the generations submitted and not yet terminal or refused. */
    private final AtomicLong promptTokensInFlight = new AtomicLong();
    /** The replica's sleep, pause and checkpoint state; the pump thread transitions it, the routes read it. */
    private final SleepControl sleep = new SleepControl();
    /** The single-flight wake: HTTP threads waking an idle sleep share one reopen on the pump thread. */
    private final Object wakeLock = new Object();
    private final Thread pumpThread;
    private volatile BatchEngine engine;
    /** The engine's latest counters, published by the pump thread after each step. */
    private volatile BatchStats stats = BatchStats.EMPTY;
    /** The setup a sleeping engine still reports on {@code /props}, captured while the model was open. */
    private volatile EngineService.Props props;
    /** The wake failure of the last attempt, or null; a failed wake leaves the engine sleeping. */
    private volatile String wakeError;
    /** The engine failure of the last pump step, or null; recovery clears it. */
    private volatile String engineFailure;
    private CountDownLatch wakeLatch;
    /** Whether a pool wake is in flight ({@code EngineService}); while it runs, the idle sleep stays off. */
    private final BooleanSupplier poolWaking;
    /** The per-request adapters every open of the model loads. */
    private final LoraCatalog adapters;
    /** Run on the pump thread after a sleep released the weights: the pool's resources shared over the models. */
    private final Runnable weightsReleased;

    /**
     * Opens {@code path} as replica {@code index} serving data-parallel rank {@code rank} with {@code config}, loads
     * the adapters of {@code adapters} and starts the pump thread. {@code weightsReleased} runs on the pump thread
     * each time a sleep, requested or idle, released the weights (the pool then releases what it shares over the
     * models, such as the media projector, once no model is open).
     */
    EngineReplica(int index, int rank, String device, String path, EngineConfig config, ServerOptions options,
                  BooleanSupplier poolWaking, LoraCatalog adapters, Runnable weightsReleased) throws IOException {
        this.weightsReleased = Objects.requireNonNull(weightsReleased, "weightsReleased");
        this.index = index;
        this.rank = rank;
        this.device = device;
        this.modelPath = path;
        this.config = Objects.requireNonNull(config, "config");
        this.options = Objects.requireNonNull(options, "options");
        this.poolWaking = Objects.requireNonNull(poolWaking, "poolWaking");
        this.adapters = Objects.requireNonNull(adapters, "adapters");
        this.pooling = new ReplicaPooling(this);
        this.loras = new ReplicaLoras(this);
        this.model = loras.open();
        captureProps();
        // the batch engine builds lazily on first use: generation through a submission, a pooling request through
        // submitPooling — a model that cannot run what asks refuses there by name
        this.pumpThread = new Thread(this::pumpLoop, "neuralfree-server-pump-" + index);
        this.pumpThread.setDaemon(false);
        this.pumpThread.start();
    }

    /** The replica's device-name filter, or null for the default device. */
    String device() {
        return device;
    }

    /** The global data-parallel rank this replica serves. */
    int rank() {
        return rank;
    }

    /** The opened model path. */
    String modelPath() {
        return modelPath;
    }

    /** The replica's open model. */
    EngineModel model() {
        return model;
    }

    /** The device memory the open model holds, in bytes; 0 while unloaded, on the CPU, or when unmeasurable. */
    long deviceBytesAllocated() {
        EngineModel open = model;
        if (!(open instanceof com.eonmux.neuralfree.gguf.GGUFModelGPU gpu)) {
            return 0;
        }
        try {
            return Math.max(0, gpu.backend().getAllocatedBytes());
        } catch (RuntimeException e) {
            return 0;
        }
    }

    /** Whether the model is open (a sleep of level 2 released it). */
    boolean open() {
        return model != null;
    }

    /**
     * The batch engine over the model, built on first use (a non-generating model refuses here by name). Its KV
     * cache events go to the publisher {@code neuralfree.kv.events} names, stamped with the global data-parallel
     * rank this replica serves (a ZMQ publisher also offsets its endpoints' ports by the rank, so the replicas bind
     * their own addresses, as vLLM's per-rank publishers do).
     */
    synchronized BatchEngine engine() {
        if (engine == null) {
            engine = BatchEngine.forModel(model, config, KvEventPublishers.ranked(
                    KvEventPublishers.forConfig(config.memory(), rank), rank));
        }
        return engine;
    }

    /** The replica's published counters and state as of its latest step. */
    BatchStats stats() {
        return stats;
    }

    /** The engine's sequence-slot count, or 0 while the batch engine is not built (no generation ran yet). */
    int slots() {
        BatchEngine current = engine;
        return current == null ? 0 : current.maxSequences();
    }
    /** The built batch engine, or null while no generation ran (the peek, without building it). */
    BatchEngine builtEngine() {
        return engine;
    }

    /** The memory's context length, or 0 while the batch engine is not built. */
    int contextLength() {
        BatchEngine current = engine;
        return current == null ? 0 : current.contextLength();
    }

    /**
     * The batch engine over the model, built on first use; the slot actions ({@code EngineService.saveSlot} and
     * friends) run on it from the pump thread. A model that does not generate refuses here by name.
     */
    BatchEngine batchEngine() {
        return engine();
    }

    /** Whether the model is open and the pump thread runs. */
    boolean alive() {
        return running.get();
    }

    /**
     * Why this replica cannot serve, or null while it is healthy: the failure of the last wake or engine step. A
     * replica whose pump thread stopped is dead rather than unhealthy ({@link #alive()}).
     */
    String faultInfo() {
        String wake = wakeError;
        if (wake != null) {
            return wake;
        }
        return engineFailure;
    }

    // ------------------------------------------------------------------ requests


    /**
     * Enqueues {@code prompt} with its request; the pump thread registers it with the engine and the returned
     * submission completes when the request reaches a terminal state. An engine refusal (oversize prompt, unusable
     * constraint) lands on {@link Submission#refused()} instead, and so does an {@link EngineUnavailableException}
     * when a sleep or rejecting pause took effect on the pump between the caller's check and the registration. The
     * replica's abort (a sleep or pause in the abort mode) and {@link #abortAll} cancel the request beside the
     * caller's own cancellation hook. A non-null {@code filter} replaces the configured reasoning filter for this
     * request (a {@code reasoning_format}).
     *
     * @throws IllegalStateException when the replica is shutting down
     */
    Submission submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink, String renderedPrompt,
                      ConstraintSpec constraint, ReasoningFilter filter, Runnable onDone)
            throws InterruptedException {
        return submit(prompt, request, sink, renderedPrompt, constraint, filter, onDone, null, null);
    }

    /**
     * {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec, ReasoningFilter,
     * Runnable)} with the disaggregated-transfer hooks: {@code importChain} (a prefill instance's chain) is
     * imported into the engine on the pump thread before the request registers, and {@code onTerminal} runs on the
     * pump thread once the sequence reaches its terminal state, with its handle (the KV export).
     */
    Submission submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink, String renderedPrompt,
                      ConstraintSpec constraint, ReasoningFilter filter, Runnable onDone,
                      Consumer<BatchSequence> onTerminal, byte[] importChain) throws InterruptedException {
        return enqueue(request, prompt.length, onDone, onTerminal, importChain,
                (engine, cancellable) -> engine.submit(prompt, cancellable, sink, renderedPrompt, constraint,
                        filter));
    }

    /**
     * {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec, ReasoningFilter,
     * Runnable)} for a media prompt: text with encoded media or precomputed embedding rows (vLLM's
     * {@code prompt_embeds}, the chat embeds parts) between the text chunks.
     *
     * @throws IllegalStateException when the replica is shutting down
     */
    Submission submitMedia(MediaPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                            String renderedPrompt, ConstraintSpec constraint, ReasoningFilter filter,
                            Runnable onDone) throws InterruptedException {
        return submitMedia(prompt, request, sink, renderedPrompt, constraint, filter, onDone, null, null);
    }

    /**
     * {@link #submitMedia(MediaPrompt, GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec,
     * ReasoningFilter, Runnable)} with the disaggregated-transfer hooks of
     * {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec, ReasoningFilter,
     * Runnable, Consumer, byte[])}.
     */
    Submission submitMedia(MediaPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                           String renderedPrompt, ConstraintSpec constraint, ReasoningFilter filter, Runnable onDone,
                           Consumer<BatchSequence> onTerminal, byte[] importChain) throws InterruptedException {
        return enqueue(request, prompt.nTokens(), onDone, onTerminal, importChain,
                (engine, cancellable) -> engine.submit(prompt, cancellable, sink, renderedPrompt, constraint,
                        filter));
    }

    /**
     * Queues one registration for the pump thread: {@code register} submits the request — made cancellable by
     * the replica's abort and by the request's own abort switch ({@link #abortAll}) — to the engine there, and an
     * engine refusal lands on the submission, as does an {@link EngineUnavailableException} when the replica's
     * serving state refuses generation by the time the pump registers it. The switch leaves the live set, and the
     * {@code promptTokens} leave the in-flight count, when the request reaches its terminal state or is refused.
     */
    private Submission enqueue(GenerationRequest request, int promptTokens, Runnable onDone,
                               BiFunction<BatchEngine, GenerationRequest, BatchSequence> register)
            throws InterruptedException {
        return enqueue(request, promptTokens, onDone, null, null, register);
    }

    /**
     * {@link #enqueue(GenerationRequest, int, Runnable, BiFunction)} with the disaggregated-transfer hooks: the
     * chain imported into the engine before the request registers ({@code importChain}, null for none) and the hook
     * run on the pump thread when the sequence reaches its terminal state ({@code onTerminal}, the export).
     */
    private Submission enqueue(GenerationRequest request, int promptTokens, Runnable onDone,
                               Consumer<BatchSequence> onTerminal, byte[] importChain,
                               BiFunction<BatchEngine, GenerationRequest, BatchSequence> register)
            throws InterruptedException {
        AtomicBoolean abort = new AtomicBoolean();
        BooleanSupplier callerCancellation = request.cancellation();
        GenerationRequest cancellable = request.toBuilder()
                .cancellation(() -> sleep.abortRequested() || abort.get() || callerCancellation.getAsBoolean())
                .build();
        Submission submission = new Submission(() -> {
            aborts.remove(abort);
            promptTokensInFlight.addAndGet(-promptTokens);
            if (onDone != null) {
                onDone.run();
            }
        }, onTerminal);
        aborts.add(abort);
        promptTokensInFlight.addAndGet(promptTokens);
        if (!running.get()) {
            aborts.remove(abort);
            promptTokensInFlight.addAndGet(-promptTokens);
            throw new IllegalStateException("the replica is shutting down");
        }
        try {
            work.put(() -> {
                try {
                    String refusal = servingRefusal();
                    if (refusal != null) {
                        throw new EngineUnavailableException(refusal);
                    }
                    BatchEngine current = engine();
                    if (importChain != null) {
                        try {
                            current.importKv(importChain);
                        } catch (IOException e) {
                            throw new IllegalStateException("importing the transferred KV chain failed: "
                                    + e.getMessage(), e);
                        }
                    }
                    submission.registered(register.apply(current, cancellable));
                } catch (RuntimeException e) {
                    submission.refused(e);
                }
            });
        } catch (InterruptedException e) {
            aborts.remove(abort);
            promptTokensInFlight.addAndGet(-promptTokens);
            throw e;
        }
        return submission;
    }

    /**
     * Aborts every live request of this replica: each one's abort switch flips and the engine stops it with
     * {@code CANCELLED} at its next step. Returns how many requests were queued, running or preempted.
     */
    int abortAll() {
        int aborted = 0;
        for (AtomicBoolean abort : aborts) {
            if (abort.compareAndSet(false, true)) {
                aborted++;
            }
        }
        return aborted;
    }

    /**
     * The requests in flight on this replica: generations submitted and not yet terminal (queued for the pump,
     * waiting, running or preempted) plus the pooling forwards callers wait on.
     */
    int load() {
        return aborts.size() + pooling.inFlight();
    }

    /** The prompt tokens (or embedding rows) of the generations submitted here and not yet terminal or refused. */
    long promptTokensInFlight() {
        return promptTokensInFlight.get();
    }

    /**
     * Clears the prefix cache on the pump thread ({@code BatchEngine.clearPrefixCache}): false while sequences
     * hold pages of the memory, as vLLM's reset reports; true when nothing was cached (no engine built, or a sleep
     * released it) or the cache cleared.
     *
     * @throws IllegalStateException when the pump did not answer within the request budget
     */
    boolean resetPrefixCacheOp() throws InterruptedException {
        return resetPrefixCacheOp(-1);
    }

    /**
     * Clears the prefix cache on the pump thread with the given wait bound; see {@link #resetPrefixCacheOp()}.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     */
    boolean resetPrefixCacheOp(long waitMillis) throws InterruptedException {
        CountDownLatch done = new CountDownLatch(1);
        AtomicBoolean cleared = new AtomicBoolean();
        AtomicReference<RuntimeException> failure = new AtomicReference<>();
        work.put(() -> {
            try {
                BatchEngine current = engine;
                if (current == null || current.idle()) {
                    if (current != null) {
                        current.clearPrefixCache();
                    }
                    cleared.set(true);
                }
            } catch (RuntimeException e) {
                failure.set(e);
            } finally {
                done.countDown();
            }
        });
        awaitOperation(done, "the prefix cache reset", waitMillis);
        if (failure.get() != null) {
            throw failure.get();
        }
        return cleared.get();
    }

    /**
     * The batch engine's slot statuses, read on the pump thread between its steps (llama.cpp serves {@code
     * GET /slots} as a high-priority task while the slots run). A replica whose engine is not built (no
     * generation ran, or a sleep released it) reports none.
     *
     * @throws IllegalStateException when the pump did not answer within the request budget
     */
    List<SlotStatus> slotStatusesOp() throws InterruptedException {
        BatchEngine current = engine;
        if (current == null) {
            return List.of();
        }
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<List<SlotStatus>> result = new AtomicReference<>();
        AtomicReference<RuntimeException> failure = new AtomicReference<>();
        work.put(() -> {
            try {
                result.set(current.slotStatuses());
            } catch (RuntimeException e) {
                failure.set(e);
            } finally {
                done.countDown();
            }
        });
        awaitOperation(done, "the slot status");
        if (failure.get() != null) {
            throw failure.get();
        }
        List<SlotStatus> read = result.get();
        return read == null ? List.of() : read;
    }

    /**
     * Drops the batch engine's shared encoder cache on the pump thread ({@code BatchEngine.clearEncoderCache});
     * nothing to drop without an engine (none built, or a sleep released it).
     *
     * @throws IllegalStateException when the pump did not answer within the request budget
     */
    void resetEncoderCacheOp() throws InterruptedException {
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<RuntimeException> failure = new AtomicReference<>();
        work.put(() -> {
            try {
                BatchEngine current = engine;
                if (current != null) {
                    current.clearEncoderCache();
                }
            } catch (RuntimeException e) {
                failure.set(e);
            } finally {
                done.countDown();
            }
        });
        awaitOperation(done, "the encoder cache reset");
        if (failure.get() != null) {
            throw failure.get();
        }
    }

    // ------------------------------------------------------------------ live weight updates

    /**
     * Creates this replica's live weight updater over its open model on the pump thread ({@link
     * EngineWeightTransfer}); a model without a forward pass refuses here by name. Repeating an initialization
     * releases the previous updater first. {@code version} is the pool's committed weight label, which the fresh
     * updater reports from the start so every replica answers {@code get_weight_version} with the served round.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws UnsupportedOperationException when the model takes no live weight update
     * @throws IllegalStateException         when no model is open, or the pump did not answer in time
     */
    void initWeightTransferOp(String version, long waitMillis) throws InterruptedException {
        requireOpen();
        transferOp("the weight transfer initialization", () -> {
            WeightUpdater previous = weightTransfer;
            if (previous != null) {
                previous.shutdown();
            }
            WeightUpdater fresh = model().weightUpdater();
            if (version != null) {
                fresh.setVersion(version);
            }
            weightTransfer = fresh;
            return null;
        }, waitMillis);
    }

    /**
     * Releases this replica's live weight updater on the pump thread; none is no-op, so a shutdown without an
     * initialization stays idempotent.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     */
    void shutdownWeightTransferOp(long waitMillis) throws InterruptedException {
        transferOp("the weight transfer shutdown", () -> {
            WeightUpdater current = weightTransfer;
            weightTransfer = null;
            if (current != null) {
                current.shutdown();
            }
            return null;
        }, waitMillis);
    }

    /**
     * Opens an update round on the pump thread.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the transfer is not initialized, or the pump did not answer in time
     */
    void startWeightUpdateOp(long waitMillis) throws InterruptedException {
        WeightUpdater updater = requireTransfer();
        transferOp("the weight update start", () -> {
            updater.start();
            return null;
        }, waitMillis);
    }

    /**
     * Writes one tensor on the pump thread, so the write and the forward stack's rebuild land between engine
     * steps: the idle batch engine drops first (its cached prefixes go with it, as vLLM's
     * {@code reset_prefix_cache} does for an RL round), then the write lands.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the transfer is not initialized, the engine is running requests, or the
     *                               pump did not answer in time
     */
    void applyWeightOp(WeightUpdateTensor tensor, long waitMillis) throws InterruptedException {
        WeightUpdater updater = requireTransfer();
        transferOp("the weight update", () -> {
            requireIdleEngine();
            releaseIdleEngine();
            updater.apply(tensor);
            return null;
        }, waitMillis);
    }

    /**
     * Commits the update round on the pump thread, dropping the idle batch engine for the same reasons as
     * {@link #applyWeightOp}.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the transfer is not initialized, the engine is running requests, or the
     *                               pump did not answer in time
     */
    void finishWeightUpdateOp(String version, long waitMillis) throws InterruptedException {
        WeightUpdater updater = requireTransfer();
        transferOp("the weight update finish", () -> {
            requireIdleEngine();
            releaseIdleEngine();
            updater.finish(version);
            return null;
        }, waitMillis);
    }

    /**
     * Stamps the weight version on the pump thread; it writes no weights, so a busy engine may stay running.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the transfer is not initialized, or the pump did not answer in time
     */
    void setWeightVersionOp(String version, long waitMillis) throws InterruptedException {
        WeightUpdater updater = requireTransfer();
        transferOp("the weight version change", () -> {
            updater.setVersion(version);
            return null;
        }, waitMillis);
    }

    /**
     * The version this replica's updater reports ({@code get_weight_version}), or null while the transfer is not
     * initialized (the pool then reports its committed label).
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the pump did not answer in time
     */
    String weightVersionOp(long waitMillis) throws InterruptedException {
        WeightUpdater updater = weightTransfer;
        if (updater == null) {
            return null;
        }
        return transferOp("the weight version read", updater::version, waitMillis);
    }

    /** The replica's live weight transfer updater, or null while none is initialized. */
    WeightUpdater weightTransfer() {
        return weightTransfer;
    }

    /** The replica's updater, refusing by name when the transfer was never initialized or was shut down. */
    private WeightUpdater requireTransfer() {
        WeightUpdater updater = weightTransfer;
        if (updater == null) {
            throw new IllegalStateException("the weight transfer engine is not initialized;"
                    + " POST /init_weight_transfer_engine first");
        }
        return updater;
    }

    /** Refuses a live weight update on a busy engine: the writes must land between steps. */
    private void requireIdleEngine() {
        BatchEngine current = engine;
        if (current != null && !current.idle()) {
            throw new IllegalStateException("the engine is running requests; pause or wait for them before a"
                    + " live weight update (POST /pause)");
        }
    }

    /**
     * Drops the idle batch engine before a weight update lands: its forward pass holds the tensors the update
     * replaces (the CPU rebuilds the model's forward stack, so a captured pass would serve closed views), and
     * its cached prefixes were computed under the old weights anyway. The engine rebuilds on the next
     * submission over the new weights.
     */
    private void releaseIdleEngine() {
        BatchEngine current = engine;
        if (current != null && current.idle()) {
            current.close();
            engine = null;
        }
    }

    /**
     * Runs one live weight update call on the pump thread and waits for it; past the wait bound it refuses by
     * name, and a refusal of the call itself reaches the caller as its own exception.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     */
    private <T> T transferOp(String what, Supplier<T> call, long waitMillis) throws InterruptedException {
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<RuntimeException> failure = new AtomicReference<>();
        AtomicReference<T> result = new AtomicReference<>();
        work.put(() -> {
            try {
                result.set(call.get());
            } catch (RuntimeException e) {
                failure.set(e);
            } finally {
                done.countDown();
            }
        });
        awaitOperation(done, what, waitMillis);
        if (failure.get() != null) {
            throw failure.get();
        }
        return result.get();
    }

    // ------------------------------------------------------------------ sleep, pause and checkpoint

    /**
     * Sleeps at {@code level} — 0 pauses the scheduling only, 1 also drops the KV cache contents, 2 also closes
     * the model and releases its weights — treating the running requests as {@code mode} says; blocks until this
     * replica sleeps. The level bounds and the keep-mode refusal are the caller's ({@code EngineService#sleep}).
     *
     * @throws IllegalStateException         when no model is open, or the engine stayed busy past the request
     *                                       budget
     * @throws UnsupportedOperationException when a checkpoint capture is under way
     */
    void sleepOp(int level, SleepControl.Mode mode) throws InterruptedException {
        requireOpen();
        if (sleep.checkpoint() != SleepControl.Checkpoint.RUNNING) {
            throw new IllegalStateException("cannot sleep while the checkpoint state is " + sleep.checkpoint());
        }
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<String> failure = new AtomicReference<>();
        work.put(() -> beginSleep(level, mode, done, failure));
        awaitOperation(done, "the sleep");
        raise(failure);
    }

    /** The pump-thread sleep: drain, release, then enter the sleeping state — now, or once the engine quiesced. */
    private void beginSleep(int level, SleepControl.Mode mode, CountDownLatch done,
                            AtomicReference<String> failure) {
        try {
            if (sleep.draining()) {
                done.countDown();
                throw new IllegalStateException("a wait-mode operation is already draining the engine;"
                        + " it completes first");
            }
            if (sleep.sleeping() && sleep.level() >= level) {
                done.countDown();
                return;
            }
            sleep.pauseNew();
            switch (mode) {
                case ABORT -> {
                    sleep.requestAbort();
                    drainForStop();
                    sleep.clearAbort();
                    finishSleep(level, SleepControl.Source.REQUEST);
                    done.countDown();
                }
                case KEEP -> {
                    finishSleep(level, SleepControl.Source.REQUEST);
                    done.countDown();
                }
                case WAIT -> sleep.onQuiesce(() -> {
                    try {
                        finishSleep(level, SleepControl.Source.REQUEST);
                    } catch (RuntimeException e) {
                        failure.compareAndSet(null, "the engine did not sleep: " + e.getMessage());
                        throw e;
                    } finally {
                        done.countDown();
                    }
                });
            }
        } catch (RuntimeException e) {
            failure.compareAndSet(null, "the engine did not sleep: " + e.getMessage());
            done.countDown();
            throw e;
        }
    }

    /** The pump-thread release of a sleep at {@code level}: the KV cache contents, and from level 2 the weights. */
    private void finishSleep(int level, SleepControl.Source source) {
        BatchEngine current = engine;
        if (level >= 1 && current != null) {
            // the running sequences a keep mode froze preempt now and recompute on the wake; a drained engine
            // preempts nothing, and only the idle engine can spare its cached prefixes
            current.preemptAll();
            if (current.idle()) {
                current.clearPrefixCache();
            }
        }
        if (level >= 2) {
            releaseWeightsAndEngine();
        }
        sleep.slept(level, source);
        captureProps();
        if (level >= 2) {
            weightsReleased.run();
        }
    }

    /** Closes the batch engine and the model: the weights, the KV cache and the scratch all release. */
    private void releaseWeightsAndEngine() {
        BatchEngine current = engine;
        if (current != null) {
            current.close();
        }
        engine = null;
        weightTransfer = null; // the updater wrote into the released weights; the wake initializes a fresh one
        EngineModel open = model;
        if (open != null) {
            try {
                open.close();
            } catch (IOException e) {
                LOGGER.log(Level.WARNING, "model close failed", e);
            }
        }
        model = null;
    }

    /**
     * Wakes this sleeping replica ({@code POST /wake_up}): the memory a sleep of level 1 or more released restores
     * — the model re-opens from the model path on the pump thread — and the scheduling resumes. The tag
     * validation is the caller's ({@code EngineService#wakeUp}).
     *
     * @return {@code awake} when the replica serves again, {@code sleeping} when a scheduling-only wake left
     *         memory released
     * @throws IllegalStateException when the wake outlived the request budget or failed (the message names it)
     */
    String wakeUp(List<String> tags) throws InterruptedException {
        if (!sleep.sleeping()) {
            return "awake";
        }
        boolean schedulingOnly = !tags.isEmpty() && tags.stream().allMatch("scheduling"::equals);
        if (schedulingOnly && sleep.level() >= 1) {
            return "sleeping";
        }
        if (sleep.level() >= 1) {
            wakeAndWait();
        } else {
            CountDownLatch done = new CountDownLatch(1);
            work.put(() -> {
                try {
                    sleep.woke();
                } finally {
                    done.countDown();
                }
            });
            awaitOperation(done, "the wake");
        }
        if (sleep.sleeping() && wakeError != null) {
            throw new IllegalStateException(wakeError);
        }
        return sleep.sleeping() ? "sleeping" : "awake";
    }

    /**
     * Wakes an idle-sleeping replica and waits for it: the model re-opens on the pump thread. Concurrent callers
     * share one wake; the wait bounds at the request budget.
     */
    void wakeAndWait() {
        CountDownLatch latch;
        synchronized (wakeLock) {
            if (!sleep.sleeping()) {
                return;
            }
            if (wakeLatch == null) {
                wakeLatch = new CountDownLatch(1);
                CountDownLatch mine = wakeLatch;
                try {
                    work.put(() -> wakeOnPump(mine));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    wakeLatch = null;
                    return;
                }
            }
            latch = wakeLatch;
        }
        try {
            if (!latch.await(options.requestTimeout().toMillis(), TimeUnit.MILLISECONDS)) {
                LOGGER.warning("the wake of replica " + index + " outlived the " + options.budgetText()
                        + " request budget");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        synchronized (wakeLock) {
            if (wakeLatch == latch) {
                wakeLatch = null;
            }
        }
    }

    /** The pump-thread wake: re-opens the model a full sleep released, then clears the sleeping state. */
    private void wakeOnPump(CountDownLatch done) {
        try {
            if (sleep.sleeping() && model == null) {
                reopen();
            }
            sleep.woke();
            wakeError = null;
        } catch (IOException | RuntimeException e) {
            wakeError = "the engine did not wake: " + e.getMessage();
            LOGGER.log(Level.WARNING, "the engine wake failed", e);
        } finally {
            done.countDown();
        }
    }

    /** Re-opens the model a sleep released (the engine rebuilds on the next submission). */
    private void reopen() throws IOException {
        this.model = loras.open();
        this.engine = null;
        captureProps();
    }

    /**
     * Recovers this replica after a fault ({@code POST /v1/fault_tolerance/apply} with {@code retry}): aborts the
     * live requests, releases the batch engine and the model, then re-opens the model from its path and clears the
     * failure state. The replica serves again once the pump thread finishes; a repair that fails leaves the failure
     * recorded and the next retry tries again.
     *
     * @throws IllegalStateException when the pump did not finish within the request budget, or the reopen failed
     *                               (the message names it)
     */
    void recoverOp() throws InterruptedException {
        abortAll();
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<String> failure = new AtomicReference<>();
        work.put(() -> {
            try {
                releaseWeightsAndEngine();
                sleep.woke();
                sleep.unpause();
                reopen();
                wakeError = null;
                engineFailure = null;
            } catch (IOException | RuntimeException e) {
                failure.set("the engine did not recover: " + e.getMessage());
                engineFailure = failure.get();
                LOGGER.log(Level.WARNING, "the engine recovery failed", e);
            } finally {
                done.countDown();
            }
        });
        awaitOperation(done, "the engine recovery");
        if (failure.get() != null) {
            throw new IllegalStateException(failure.get());
        }
    }


    /**
     * Pauses the generation ({@code POST /pause}): the abort and wait modes reject new requests with the pause
     * refusal until {@link #resumeOp()}; the keep mode keeps accepting and queueing them but schedules nothing.
     * {@code clearCache} drops the KV cache contents once the engine drained (ignored for the keep mode, whose
     * frozen requests hold their pages).
     *
     * @throws IllegalStateException when no model is open, or the engine stayed busy past the request budget
     */
    void pauseOp(SleepControl.Mode mode, boolean clearCache) throws InterruptedException {
        requireOpen();
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<String> failure = new AtomicReference<>();
        work.put(() -> beginPause(mode, clearCache, done, failure));
        awaitOperation(done, "the pause");
        raise(failure);
    }

    /** The pump-thread pause: reject or freeze, drain as the mode says, then optionally drop the KV cache. */
    private void beginPause(SleepControl.Mode mode, boolean clearCache, CountDownLatch done,
                            AtomicReference<String> failure) {
        try {
            if (sleep.draining()) {
                done.countDown();
                throw new IllegalStateException("a wait-mode operation is already draining the engine;"
                        + " it completes first");
            }
            if (mode == SleepControl.Mode.KEEP) {
                sleep.pauseAll();
                done.countDown();
                return;
            }
            sleep.pauseNew();
            if (mode == SleepControl.Mode.ABORT) {
                sleep.requestAbort();
                drainForStop();
                sleep.clearAbort();
                dropKvIfIdle();
                done.countDown();
            } else {
                sleep.onQuiesce(() -> {
                    try {
                        dropKvIfIdle();
                    } finally {
                        done.countDown();
                    }
                });
            }
        } catch (RuntimeException e) {
            failure.compareAndSet(null, "the engine did not pause: " + e.getMessage());
            done.countDown();
            throw e;
        }
    }

    /** Resumes the generation after a pause ({@code POST /resume}). */
    void resumeOp() throws InterruptedException {
        CountDownLatch done = new CountDownLatch(1);
        work.put(() -> {
            try {
                sleep.unpause();
            } finally {
                done.countDown();
            }
        });
        awaitOperation(done, "the resume");
    }

    /**
     * Drops the KV cache contents — every cached prefix and retired sequence — so its memory serves new requests
     * ({@code POST /release_kv_cache_memory}). The weights stay resident; a sleep of level 2 releases them too.
     *
     * @throws UnsupportedOperationException when a request still runs, holds or waits for pages
     * @throws IllegalStateException         when no model is open
     */
    void releaseKvOp() throws InterruptedException {
        requireOpen();
        BatchEngine current = engine;
        if (current != null && (current.running() > 0 || current.queued() > 0)) {
            throw new UnsupportedOperationException("the KV cache releases only on an idle engine;"
                    + " pause or wait for the requests first");
        }
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<String> failure = new AtomicReference<>();
        work.put(() -> {
            try {
                BatchEngine live = engine;
                if (live != null) {
                    live.clearPrefixCache();
                }
            } catch (RuntimeException e) {
                failure.compareAndSet(null, e.getMessage());
                throw e;
            } finally {
                done.countDown();
            }
        });
        awaitOperation(done, "the KV cache release");
        if (failure.get() != null) {
            throw new UnsupportedOperationException(failure.get());
        }
    }

    /** Drops every cached prefix when the engine is idle; a busy engine keeps its cache (the best-effort paths). */
    private void dropKvIfIdle() {
        BatchEngine current = engine;
        if (current != null && current.idle()) {
            current.clearPrefixCache();
        }
    }

    /** Re-throws a pump-thread operation failure the caller waited on, as its own exception. */
    private static void raise(AtomicReference<String> failure) {
        if (failure.get() != null) {
            throw new IllegalStateException(failure.get());
        }
    }

    /**
     * Starts a checkpoint capture ({@code POST /checkpoint/prepare}): drain as {@code mode} says, then release the
     * transient state a capture must not contain (the KV cache contents) and keep new work out until
     * {@link #restoreCheckpointOp()}. Returns once the preparation started; repeating one is a no-op.
     *
     * @return the replica's checkpoint state name at the return
     * @throws IllegalStateException when no model is open, the engine sleeps, a preparation cannot start from the
     *                               current state, or the abort drain stayed busy past the request budget
     */
    String prepareCheckpointOp(SleepControl.Mode mode) throws InterruptedException {
        requireOpen();
        if (sleep.sleeping()) {
            throw new IllegalStateException("the engine is sleeping; wake it before preparing a checkpoint");
        }
        SleepControl.Checkpoint current = sleep.checkpoint();
        if (current == SleepControl.Checkpoint.RESTORING || current == SleepControl.Checkpoint.FAILED) {
            throw new IllegalStateException("cannot prepare a checkpoint from " + current);
        }
        if (current == SleepControl.Checkpoint.RUNNING) {
            CountDownLatch done = new CountDownLatch(1);
            work.put(() -> beginCheckpointPrepare(mode, done));
            // the abort drain finishes now; the wait drain returns once started (status polls until prepared)
            awaitOperation(done, "the checkpoint preparation");
        }
        return sleep.checkpoint().name().toLowerCase(java.util.Locale.ROOT);
    }

    /** The pump-thread preparation: PREPARING, drain, then release the transient state into PREPARED. */
    private void beginCheckpointPrepare(SleepControl.Mode mode, CountDownLatch done) {
        try {
            if (sleep.draining()) {
                done.countDown();
                throw new IllegalStateException("a wait-mode operation is already draining the engine;"
                        + " it completes first");
            }
            sleep.checkpointPreparing();
            if (mode == SleepControl.Mode.ABORT) {
                sleep.requestAbort();
                drainForStop();
                sleep.clearAbort();
                prepareCaptured();
                done.countDown();
            } else {
                sleep.pauseNew();
                sleep.onQuiesce(() -> {
                    try {
                        prepareCaptured();
                    } catch (RuntimeException e) {
                        sleep.checkpointFailed(String.valueOf(e.getMessage()));
                        throw e;
                    }
                });
                done.countDown();
            }
        } catch (RuntimeException e) {
            sleep.checkpointFailed(String.valueOf(e.getMessage()));
            done.countDown();
            throw e;
        }
    }

    /** The prepared state: the transient KV state gone, new work kept out until the restore. */
    private void prepareCaptured() {
        dropKvIfIdle();
        sleep.checkpointPrepared();
        sleep.pauseNew();
    }

    /**
     * Restores a prepared checkpoint ({@code POST /checkpoint/restore}): the released state comes back and the
     * engine serves again. A restore in progress or a running engine is a no-op.
     *
     * @return the replica's checkpoint state name at the return
     * @throws IllegalStateException when a preparation is still running or the last one failed
     */
    String restoreCheckpointOp() throws InterruptedException {
        if (!open()) {
            throw new IllegalStateException("no model loaded: set -Dneuralfree.server.model=<gguf path>");
        }
        SleepControl.Checkpoint current = sleep.checkpoint();
        if (current == SleepControl.Checkpoint.PREPARING || current == SleepControl.Checkpoint.FAILED) {
            throw new IllegalStateException("cannot restore a checkpoint from " + current
                    + (sleep.checkpointError() == null ? "" : ": " + sleep.checkpointError()));
        }
        if (current == SleepControl.Checkpoint.PREPARED) {
            CountDownLatch done = new CountDownLatch(1);
            work.put(() -> {
                try {
                    sleep.checkpointRestoring();
                    sleep.unpause();
                    sleep.checkpointRunning();
                } finally {
                    done.countDown();
                }
            });
            awaitOperation(done, "the checkpoint restore");
        }
        return sleep.checkpoint().name().toLowerCase(java.util.Locale.ROOT);
    }

    /** Waits for the pump thread to complete an operation; past the request budget it refuses by name. */
    void awaitOperation(CountDownLatch done, String what) throws InterruptedException {
        awaitOperation(done, what, -1);
    }

    /**
     * Waits for the pump thread to complete an operation; past {@code waitMillis}, or the request budget when it
     * is 0 or less, it refuses by name.
     */
    void awaitOperation(CountDownLatch done, String what, long waitMillis) throws InterruptedException {
        long budget = waitMillis > 0 ? waitMillis : options.requestTimeout().toMillis();
        if (!done.await(budget, TimeUnit.MILLISECONDS)) {
            String bound = waitMillis > 0 ? "the " + waitMillis + " ms collective RPC timeout"
                    : "the " + options.budgetText() + " request budget";
            throw new IllegalStateException(what + " outlived " + bound);
        }
    }

    /** Pumps until nothing is queued or running (an abort or wait drain), then once over the idle engine. */
    private void drainForStop() {
        BatchEngine current = engine;
        if (current == null) {
            return;
        }
        current.cancelAll();
        while (current.running() > 0 || current.queued() > 0) {
            current.pump();
        }
        current.pump();
    }

    /** Captures the model setup a sleeping engine keeps reporting: the live values while the model is open. */
    private void captureProps() {
        EngineModel open = model;
        if (open == null) {
            return;
        }
        BatchEngine current = engine;
        int context = current != null ? current.contextLength() : open.kvCacheContext();
        this.props = new EngineService.Props(open.architecture(), open.tokenizer().chatTemplate(),
                open.tokenizer().chatTemplateCaps(), open.tokenizer().vocabSize(), current != null ? current.maxSequences() : 0, context, context,
                backendName(), open.gpuFallbackReason());
    }

    /**
     * The backend API the model runs on ({@code "opencl"}, {@code "cuda"}, {@code "vulkan"}, {@code "cpu"}), or null
     * before the first open.
     */
    String backendName() {
        if (model instanceof GGUFModelGPU gpu) {
            return gpu.backend().backendType().name().toLowerCase(Locale.ROOT);
        }
        return model == null ? null : "cpu";
    }

    /** Why the model fell back from a GPU load to the CPU engine, or null when it did not. */
    String gpuFallbackReason() {
        return model == null ? null : model.gpuFallbackReason();
    }

    // ------------------------------------------------------------------ state views

    /** Why generation must refuse right now on this replica, or null when it is served. */
    String servingRefusal() {
        String refusal = sleep.schedulingRefusal();
        if (refusal != null && wakeError != null) {
            return refusal + " (" + wakeError + ")";
        }
        return refusal;
    }

    /** Whether a request wakes this replica: only the idle sleep does; an explicit one waits for {@code /wake_up}. */
    boolean wakesOnRequest() {
        return sleep.sleeping() && sleep.source() == SleepControl.Source.IDLE;
    }

    /** Whether this replica holds released memory (a sleep of level 1 or more). */
    boolean sleeping() {
        return sleep.sleeping();
    }

    /** The sleep level: 0 scheduling only, 1 the KV cache contents, 2 the weights too; only while sleeping. */
    int sleepLevel() {
        return sleep.level();
    }

    /** Whether a pause is active on this replica. */
    boolean paused() {
        return sleep.paused();
    }

    /** This replica's checkpoint capture state. */
    SleepControl.Checkpoint checkpoint() {
        return sleep.checkpoint();
    }

    /** This replica's checkpoint failure, or null. */
    String checkpointError() {
        return sleep.checkpointError();
    }

    /** The model setup captured while the model was open; null before the first open. */
    EngineService.Props capturedProps() {
        return props;
    }

    /** Refuses by name when a sleep released the model ({@code EngineService} wakes it first). */
    private void requireOpen() {
        if (!open()) {
            throw new IllegalStateException("no model loaded: set -Dneuralfree.server.model=<gguf path>");
        }
    }

    // ------------------------------------------------------------------ pump thread and close

    /** Stops the pump thread, clears the engine and closes the model; any sleep or pause state dies with it. */
    @Override
    public void close() {
        running.set(false);
        sleep.woke();
        sleep.unpause();
        pumpThread.interrupt();
        try {
            pumpThread.join(5_000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        if (!pumpThread.isAlive()) {
            // the pump is gone: queued submissions register now so the engine's close fails them, and waiting
            // pooling callers learn the service closed instead of timing out
            Runnable task;
            while ((task = work.poll()) != null) {
                try {
                    task.run();
                } catch (RuntimeException e) {
                    LOGGER.log(Level.FINE, "queued task failed at close", e);
                }
            }
            pooling.close();
        }
        BatchEngine current = engine;
        if (current != null) {
            current.close();
        }
        weightTransfer = null;
        EngineModel open = model;
        if (open != null) {
            try {
                open.close();
            } catch (IOException e) {
                LOGGER.log(Level.WARNING, "model close failed", e);
            }
        }
    }

    /**
     * The pump thread: registers queued work, runs the pooling forwards while the engine is idle, and steps the
     * engine. It waits for new work only while there is nothing to decode, so decode steps run back to back.
     * While the sleep or pause state freezes the engine nothing is scheduled; a wait-mode operation completes
     * once the engine quiesced; the idle sleep enters after {@code neuralfree.server.sleep.idle.seconds} without
     * work.
     */
    private void pumpLoop() {
        long idleSince = System.nanoTime();
        while (running.get()) {
            boolean acted = false;
            try {
                acted = drainWork(busy() && !sleep.frozen() ? 0 : IDLE_POLL_MILLIS);
            } catch (InterruptedException e) {
                if (!running.get()) {
                    return;
                }
            }
            acted |= pooling.runWaiting(busy());
            BatchEngine current = engine;
            if (!sleep.frozen() && current != null && !current.idle()) {
                try {
                    current.pump();
                    acted = true;
                } catch (RuntimeException e) {
                    engineFailure = "the engine step failed: " + e;
                    LOGGER.log(Level.WARNING, "engine step failed", e);
                }
            }
            if (current != null) {
                stats = current.stats();
            }
            sleep.completeDrainIfQuiesced(quiesced());
            long now = System.nanoTime();
            if (acted || poolWaking.getAsBoolean()) {
                // a pool wake in flight counts as work: the idle clock restarts once the whole pool is open
                idleSince = now;
            } else {
                tryIdleSleep(now - idleSince);
            }
        }
    }

    private boolean busy() {
        BatchEngine current = engine;
        return current != null && !current.idle();
    }

    /**
     * Whether the engine holds no work a wait-mode operation would wait for: no queued task, no pooling forward,
     * and no running or queued submission. Preempted sequences hold no pages and wait for a wake, so they do not
     * count.
     */
    private boolean quiesced() {
        if (!work.isEmpty() || !pooling.isEmpty()) {
            return false;
        }
        BatchEngine current = engine;
        return current == null || (current.running() == 0 && current.queued() == 0);
    }

    /**
     * The idle sleep (llama.cpp's {@code --sleep-idle-seconds}): after the configured seconds without work the
     * engine releases everything at sleep level 2; the next request wakes it. A pause, a sleep, a checkpoint
     * capture, a pending drain or a pool wake in flight (another replica may still be re-opening) keeps the engine
     * awake.
     */
    private void tryIdleSleep(long idleNanos) {
        if (poolWaking.getAsBoolean() || options.sleepIdleSeconds() < 0 || sleep.sleeping() || sleep.paused()
                || sleep.draining() || sleep.checkpoint() != SleepControl.Checkpoint.RUNNING || !quiesced()) {
            return;
        }
        if (idleNanos < options.sleepIdleSeconds() * 1_000_000_000L) {
            return;
        }
        releaseWeightsAndEngine();
        sleep.slept(2, SleepControl.Source.IDLE);
        weightsReleased.run();
        LOGGER.info("replica " + index + " slept after " + options.sleepIdleSeconds() + " idle second(s);"
                + " the next request wakes it");
    }

    /** Runs every queued task: the first after waiting up to {@code millis}, the rest without waiting. */
    private boolean drainWork(long millis) throws InterruptedException {
        Runnable task = millis > 0 ? work.poll(millis, TimeUnit.MILLISECONDS) : work.poll();
        if (task == null) {
            return false;
        }
        do {
            try {
                task.run();
            } catch (RuntimeException e) {
                LOGGER.log(Level.WARNING, "submit task failed", e);
            }
            task = work.poll();
        } while (task != null);
        return true;
    }

    /**
     * Runs a pooling forward on the pump thread once the batch engine is idle; see {@link ReplicaPooling#onPump}.
     */
    <T> T onPump(ReplicaPooling.ModelCall<T> call) throws IOException {
        return pooling.onPump(call);
    }

    /** {@link #onPump(ReplicaPooling.ModelCall)} with the forward's adapter slot from {@code selection}. */
    <T> T onPump(LoraSelection selection, ReplicaPooling.ModelCall<T> call) throws IOException {
        return pooling.onPump(selection, call);
    }

    /** Runs a call on the batch engine's state once it is idle, keeping the prefix cache. */
    <T> T onIdleEngine(ReplicaPooling.ModelCall<T> call) throws IOException {
        return pooling.onIdleEngine(call);
    }

    /** Runs one pooling request for {@code output} over {@code tokens}. */
    Object pooling(int[] tokens, PoolingOutput output, LoraSelection selection) throws IOException {
        return pooling.pooling(tokens, output, selection);
    }

    /** {@link #pooling} with the classifier's final probability activation applied or skipped. */
    Object pooling(int[] tokens, PoolingOutput output, LoraSelection selection, boolean activation)
            throws IOException {
        return pooling.pooling(tokens, output, selection, activation);
    }

    /** Runs {@code requests} pooling requests over one replica. */
    Object[] poolingAll(int[][] requests, PoolingOutput output, LoraSelection selection) throws IOException {
        return pooling.poolingAll(requests, output, selection);
    }

    /** {@link #poolingAll} with the classifier's final probability activation applied or skipped per request. */
    Object[] poolingAll(int[][] requests, PoolingOutput output, LoraSelection selection, boolean activation)
            throws IOException {
        return pooling.poolingAll(requests, output, selection, activation);
    }

    /** Runs a change of the model's per-request adapters on the pump thread. */
    void adapterOp(String what, RegistryCall call) throws IOException, InterruptedException {
        loras.adapterOp(what, call);
    }

    /** The PEFT metadata of the resident adapter {@code name}. */
    LoraCatalog.Metadata adapterMetadata(String name) throws IOException, InterruptedException {
        return loras.adapterMetadata(name);
    }

    // ------------------------------------------------------------------ collaborator accessors

    /** The pump thread's work queue. */
    BlockingQueue<Runnable> work() {
        return work;
    }

    /** The replica's launch options. */
    ServerOptions options() {
        return options;
    }

    /** The per-request adapters every open of the model loads. */
    LoraCatalog adapters() {
        return adapters;
    }

    /** The replica's per-request adapter sessions over the loaded model. */
    ReplicaLoras loras() {
        return loras;
    }

    /** The replica's engine configuration. */
    EngineConfig config() {
        return config;
    }
}
