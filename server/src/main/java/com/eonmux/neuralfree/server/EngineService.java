package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.InfillPrompt;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The serving lifecycle (PARITY plan P2): one open {@link GGUFModel}, its {@link BatchEngine} and the single
 * thread that submits into and pumps the engine. HTTP threads call {@link #submit} — the request joins the
 * engine's queue through the pump thread (the engine is not thread-safe) and the caller waits on the returned
 * {@link Submission}'s latch or streams from its sink queue.
 *
 * <p>The model path comes from the {@code neuralfree.server.model} property; the rest of the engine reads its
 * {@code neuralfree.*} properties as everywhere else ({@link EngineConfig#fromProperties}). Without the path the
 * service starts unloaded and every request refuses by name — the module stays bootable for its health and
 * metrics endpoints.
 *
 * <p>The pump loop alternates between registering queued submissions and running one engine step, so prompt
 * prefills interleave with decode steps of the running sequences (continuous batching; {@code BatchEngine.pump}
 * runs exactly one step).
 */
public final class EngineService implements AutoCloseable {

    /**
     * One submitted request: the engine handle plus the latch that fires on its terminal state, and the hook run
     * once at that moment on the pump thread, after every routed piece reached the sink and before the latch
     * releases its waiters.
     */
    public static final class Submission {
        private final CountDownLatch done = new CountDownLatch(1);
        private final AtomicBoolean terminal = new AtomicBoolean();
        private final Runnable onDone;
        private volatile BatchSequence sequence;
        private volatile Throwable refused;

        Submission(Runnable onDone) {
            this.onDone = onDone;
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
            sequence.onFinish(this::finish);
            if (sequence.finished()) {
                finish();
            }
        }

        void refused(Throwable e) {
            this.refused = e;
            finish();
        }

        private void finish() {
            if (terminal.compareAndSet(false, true)) {
                try {
                    if (onDone != null) {
                        onDone.run();
                    }
                } finally {
                    done.countDown();
                }
            }
        }
    }

    private static final Logger LOGGER = Logger.getLogger(EngineService.class.getName());

    /** llama.cpp's default logical batch, the share of the context an infill's prefix and suffix fit. */
    private static final int INFILL_BATCH = 2048;

    /** How long the pump waits for work while nothing decodes. */
    private static final long IDLE_POLL_MILLIS = 50;

    private final String modelPath;
    private final ServedModels served;
    private final ServerOptions options;
    private final ResponseStore responses;
    private final EngineConfig config;
    private final GGUFModel model;
    private final BlockingQueue<Runnable> work = new ArrayBlockingQueue<>(256);
    private final AtomicBoolean running = new AtomicBoolean(true);
    /** Pooling forwards waiting for an idle engine; touched by the pump thread only (and by close after it). */
    private final Deque<PoolingCall<?>> pooling = new ArrayDeque<>();
    private final Thread pumpThread;
    private volatile BatchEngine engine;
    /** The engine's latest counters, published by the pump thread after each step. */
    private volatile BatchStats stats = BatchStats.EMPTY;
    private volatile ReasoningFilter reasoning;

    /** Opens {@code path} (null keeps the service unloaded) with {@code config} and starts the pump thread. */
    public EngineService(String path, EngineConfig config) throws IOException {
        this(path, config, ServerOptions.defaults());
    }

    /** {@link #EngineService(String, EngineConfig)} under the server {@code options}. */
    EngineService(String path, EngineConfig config, ServerOptions options) throws IOException {
        this.modelPath = path;
        this.options = Objects.requireNonNull(options, "options");
        this.served = ServedModels.of(options.servedNames(), path);
        this.responses = new ResponseStore(options.responseStore());
        this.config = Objects.requireNonNull(config, "config");
        if (path == null || path.isBlank()) {
            this.model = null;
            this.pumpThread = null;
            return;
        }
        this.model = new GGUFModel(path, config);
        // the batch engine builds lazily: an embedding or classifier model loads for /v1/embeddings and
        // /v1/rerank, and a generation request then refuses through BatchEngine.forModel by name
        this.pumpThread = new Thread(this::pumpLoop, "neuralfree-server-pump");
        this.pumpThread.setDaemon(false);
        this.pumpThread.start();
    }

    /** The batch engine over the model, built on first use (a non-generating model refuses here by name). */
    private synchronized BatchEngine engine() {
        if (engine == null) {
            engine = BatchEngine.forModel(model, config);
        }
        return engine;
    }

    /** The opened model path, or null when unloaded ({@code neuralfree.server.model} unset). */
    public String modelPath() {
        return modelPath;
    }

    /** The engine's counters and state as of its latest step; {@link BatchStats#EMPTY} before the first. */
    public BatchStats stats() {
        return stats;
    }

    /** The server settings. */
    ServerOptions options() {
        return options;
    }

    /** The names the server answers to. */
    ServedModels served() {
        return served;
    }

    /** The Responses API store ({@link ServerOptions#responseStore}). */
    ResponseStore responses() {
        return responses;
    }

    /** The model's architecture name, or null when unloaded. */
    public String architecture() {
        return model == null ? null : model.architecture();
    }

    /** The engine's sequence-slot count, or 0 while the batch engine is not built (no generation ran yet). */
    public int slots() {
        BatchEngine current = engine;
        return current == null ? 0 : current.maxSequences();
    }

    /** The memory's context length, or 0 while the batch engine is not built. */
    public int contextLength() {
        BatchEngine current = engine;
        return current == null ? 0 : current.contextLength();
    }

    /** Whether a model is open and the pump thread runs. */
    public boolean loaded() {
        return model != null && running.get();
    }

    /** Renders the chat request through the model's template ({@code GGUFTokenizer.formatChat}). */
    public String formatChat(ChatRequest request) {
        requireLoaded();
        return model.tokenizer().formatChat(request);
    }

    /**
     * The reasoning filter the engine routes output with: the configured parser
     * ({@code neuralfree.reasoning.parser}), else the chat template's.
     */
    public ReasoningFilter reasoning() {
        requireLoaded();
        ReasoningFilter current = reasoning;
        if (current == null) {
            current = ReasoningParsers.forModel(config.reasoningParser(), model.tokenizer());
            reasoning = current;
        }
        return current;
    }

    /**
     * How the tools of {@code request} reach its output ({@link ToolRouting}): the configured tool parser
     * ({@code neuralfree.tool.parser}) over the model's template and reasoning filter.
     *
     * @param renderedPrompt    the rendered prompt, which tells whether reasoning is left open
     * @param parallelToolCalls the request's {@code parallel_tool_calls}
     * @throws IllegalArgumentException for a tool choice naming an undeclared function
     */
    public ToolRouting toolRouting(ChatRequest request, String renderedPrompt, boolean parallelToolCalls) {
        requireLoaded();
        return ToolRouting.plan(request, config.toolParser(), model.tokenizer().chatTemplate(), reasoning(),
                renderedPrompt, parallelToolCalls);
    }

    /** The context a request may fill: the planned memory context once the engine runs, else the training one. */
    public int maxModelLength() {
        requireLoaded();
        BatchEngine current = engine;
        return current != null ? current.contextLength() : model.kvCacheContext();
    }

    /** The model's chat template source, or null when it carries none. */
    public String chatTemplate() {
        requireLoaded();
        return model.tokenizer().chatTemplate();
    }

    /** The vocabulary size, the bound of a token id. */
    public int vocabSize() {
        requireLoaded();
        return model.tokenizer().vocabSize();
    }

    /** Decodes token ids to text ({@code GGUFTokenizer.decode}). */
    public String decode(int[] tokens) {
        requireLoaded();
        return model.tokenizer().decode(tokens);
    }

    /** Tokenizes text ({@code GGUFTokenizer.encode(text, addSpecial, parseSpecial)}). */
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        requireLoaded();
        return model.tokenizer().encode(text, addSpecial, parseSpecial);
    }

    /** The FIM tokens the model lacks for {@code /infill}, or null when it has them all. */
    public String infillMissingTokens() {
        requireLoaded();
        return InfillPrompt.missingTokens(model.tokenizer());
    }

    /**
     * The fill-in-the-middle prompt of llama.cpp's {@code /infill} ({@link InfillPrompt}) over the model's FIM
     * tokens, sized to the context and to llama.cpp's default 2048-token batch.
     *
     * @throws IllegalArgumentException when the model has no FIM tokens
     */
    public int[] infillPrompt(int[] prefix, int[] suffix, List<InfillPrompt.Chunk> extra, int[] prompt,
                              int nPredict) {
        requireLoaded();
        int nCtx = maxModelLength();
        return InfillPrompt.build(model.tokenizer(), prefix, suffix, extra, prompt, Math.min(INFILL_BATCH, nCtx),
                nPredict, nCtx, options.spmInfill());
    }

    /** Decodes one token id to its text (the logprobs rendering). */
    public String tokenText(int tokenId) {
        requireLoaded();
        return model.tokenizer().decode(tokenId);
    }

    /** The pooled embedding of one input ({@code GGUFModel.encode}); an EMBED or pooling model carries it. */
    public float[] embed(int[] tokens) throws IOException {
        return onPump(() -> model.encode(tokens));
    }

    /**
     * Whether the model's embeddings are per token ({@link #embedTokens}): no pooling on an embedding model or
     * pockettts. A generative model without pooling has none, and {@link #embed} refuses it by name.
     */
    public boolean perTokenEmbeddings() {
        requireLoaded();
        return com.eonmux.neuralfree.gguf.ops.heads.EncodeGate.embedsPerToken(model.spec());
    }

    /** The per-token embeddings of one input ({@code GGUFModel.encodeTokens}), for NONE pooling. */
    public float[][] embedTokens(int[] tokens) throws IOException {
        return onPump(() -> model.encodeTokens(tokens));
    }

    /** The rerank input tokens of one query-document pair ({@code ClassifierHead.rerankInput}). */
    public int[] rerankInput(String query, String document) {
        requireLoaded();
        return com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.rerankInput(model.tokenizer(), query, document);
    }

    /** The relevance score of one rerank input ({@code GGUFModel.classify} over a RANK model). */
    public float rerankScore(int[] tokens) throws IOException {
        return onPump(() -> model.classify(tokens)[0]);
    }

    /** A model forward run on the pump thread. */
    @FunctionalInterface
    interface ModelCall<T> {
        T run() throws IOException;
    }

    /** A queued pooling forward and the future its caller waits on. */
    private record PoolingCall<T>(ModelCall<T> call, CompletableFuture<T> result) {
        void run() {
            try {
                result.complete(call.run());
            } catch (IOException | RuntimeException e) {
                result.completeExceptionally(e);
            }
        }
    }

    /**
     * Runs a pooling forward on the pump thread — the model's only thread — once the batch engine is idle: the
     * forward resets the shared memory, which would corrupt running sequences, so it waits for them to finish.
     *
     * @throws IOException                   when the forward fails, the wait outlives the request budget, or the
     *                                       caller is interrupted
     * @throws UnsupportedOperationException when the model does not pool or classify
     */
    private <T> T onPump(ModelCall<T> call) throws IOException {
        requireLoaded();
        PoolingCall<T> task = new PoolingCall<>(call, new CompletableFuture<>());
        try {
            work.put(() -> pooling.add(task));
            return task.result().get(options.requestTimeout().toNanos(), TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while waiting for the model", e);
        } catch (TimeoutException e) {
            throw new IOException("the model stayed busy past the " + options.budgetText() + " request budget", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException io) {
                throw io;
            }
            if (cause instanceof RuntimeException re) {
                throw re;
            }
            throw new IOException(cause);
        }
    }

    /**
     * Enqueues {@code prompt} with its request; the pump thread registers it with the engine and the returned
     * submission completes when the request reaches a terminal state. An engine refusal (oversize prompt, unusable
     * constraint) lands on {@link Submission#refused()} instead.
     *
     * @throws IllegalStateException when the service is unloaded or shutting down
     */
    public Submission submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                             String renderedPrompt, ConstraintSpec constraint) throws InterruptedException {
        return submit(prompt, request, sink, renderedPrompt, constraint, null);
    }

    /**
     * {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec)} with a hook run once
     * when the request reaches its terminal state or is refused, on the pump thread.
     */
    public Submission submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                             String renderedPrompt, ConstraintSpec constraint, Runnable onDone)
            throws InterruptedException {
        requireLoaded();
        Submission submission = new Submission(onDone);
        work.put(() -> {
            try {
                submission.registered(engine().submit(prompt, request, sink, renderedPrompt, constraint));
            } catch (RuntimeException e) {
                submission.refused(e);
            }
        });
        return submission;
    }

    /** {@link #submit} for a content-only callback and no constraint. */
    public Submission submit(int[] prompt, GenerationRequest request, TokenCallback callback)
            throws InterruptedException {
        return submit(prompt, request, callback == null ? null
                : ReasoningFilter.contentTo(callback), null, null);
    }

    /** Stops the pump thread, clears the engine and closes the model. */
    @Override
    public void close() {
        running.set(false);
        if (pumpThread != null) {
            pumpThread.interrupt();
            try {
                pumpThread.join(5_000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        if (pumpThread == null || !pumpThread.isAlive()) {
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
            PoolingCall<?> call;
            while ((call = pooling.poll()) != null) {
                call.result().completeExceptionally(new IOException("the server is shutting down"));
            }
        }
        if (engine != null) {
            engine.close();
        }
        if (model != null) {
            try {
                model.close();
            } catch (IOException e) {
                LOGGER.log(Level.WARNING, "model close failed", e);
            }
        }
    }

    private void requireLoaded() {
        if (!loaded()) {
            throw new IllegalStateException("no model loaded: set -Dneuralfree.server.model=<gguf path>");
        }
    }

    /**
     * The pump thread: registers queued work, runs the pooling forwards while the engine is idle, and steps the
     * engine. It waits for new work only while there is nothing to decode, so decode steps run back to back.
     */
    private void pumpLoop() {
        while (running.get()) {
            try {
                drainWork(busy() ? 0 : IDLE_POLL_MILLIS);
            } catch (InterruptedException e) {
                if (!running.get()) {
                    return;
                }
            }
            runPooling();
            BatchEngine current = engine;
            try {
                if (current != null && !current.idle()) {
                    current.pump();
                }
            } catch (RuntimeException e) {
                LOGGER.log(Level.WARNING, "engine step failed", e);
            }
            if (current != null) {
                stats = current.stats();
            }
        }
    }

    private boolean busy() {
        BatchEngine current = engine;
        return current != null && !current.idle();
    }

    /** Runs every queued task: the first after waiting up to {@code millis}, the rest without waiting. */
    private void drainWork(long millis) throws InterruptedException {
        Runnable task = millis > 0 ? work.poll(millis, TimeUnit.MILLISECONDS) : work.poll();
        while (task != null) {
            try {
                task.run();
            } catch (RuntimeException e) {
                LOGGER.log(Level.WARNING, "submit task failed", e);
            }
            task = work.poll();
        }
    }

    /** Runs the waiting pooling forwards when no sequence uses the memory, then drops the stale cached prefixes. */
    private void runPooling() {
        if (pooling.isEmpty() || busy()) {
            return;
        }
        while (!pooling.isEmpty()) {
            pooling.poll().run();
        }
        BatchEngine current = engine;
        if (current != null) {
            current.clearPrefixCache();
        }
    }
}
