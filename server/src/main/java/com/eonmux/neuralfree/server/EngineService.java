package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;

import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
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

    /** One submitted request: the engine handle plus the latch that fires on its terminal state. */
    public static final class Submission {
        private final CountDownLatch done = new CountDownLatch(1);
        private volatile BatchSequence sequence;
        private volatile Throwable refused;

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
            sequence.onFinish(done::countDown);
            if (sequence.finished()) {
                done.countDown();
            }
        }

        void refused(Throwable e) {
            this.refused = e;
            done.countDown();
        }
    }

    private static final Logger LOGGER = Logger.getLogger(EngineService.class.getName());

    private final String modelPath;
    private final EngineConfig config;
    private final GGUFModel model;
    private final BlockingQueue<Runnable> work = new ArrayBlockingQueue<>(256);
    private final AtomicBoolean running = new AtomicBoolean(true);
    private final Thread pumpThread;
    private volatile BatchEngine engine;

    /** Opens {@code path} (null keeps the service unloaded) with {@code config} and starts the pump thread. */
    public EngineService(String path, EngineConfig config) throws IOException {
        this.modelPath = path;
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
    public String formatChat(com.eonmux.neuralfree.gguf.chat.ChatRequest request) {
        requireLoaded();
        return model.tokenizer().formatChat(request);
    }

    /** Tokenizes text ({@code GGUFTokenizer.encode(text, addSpecial, parseSpecial)}). */
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        requireLoaded();
        return model.tokenizer().encode(text, addSpecial, parseSpecial);
    }

    /** Decodes one token id to its text (the logprobs rendering). */
    public String tokenText(int tokenId) {
        requireLoaded();
        return model.tokenizer().decode(tokenId);
    }

    /** The pooled embedding of one input ({@code GGUFModel.encode}); an EMBED or pooling model carries it. */
    public float[] embed(int[] tokens) throws IOException {
        requireLoaded();
        return model.encode(tokens);
    }

    /** The rerank input tokens of one query-document pair ({@code ClassifierHead.rerankInput}). */
    public int[] rerankInput(String query, String document) {
        requireLoaded();
        return com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead.rerankInput(model.tokenizer(), query, document);
    }

    /** The relevance score of one rerank input ({@code GGUFModel.classify} over a RANK model). */
    public float rerankScore(int[] tokens) throws IOException {
        requireLoaded();
        return model.classify(tokens)[0];
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
        requireLoaded();
        Submission submission = new Submission();
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

    private void pumpLoop() {
        while (running.get()) {
            try {
                Runnable task = work.poll(50, TimeUnit.MILLISECONDS);
                if (task != null) {
                    task.run();
                }
            } catch (InterruptedException e) {
                if (!running.get()) {
                    return;
                }
            } catch (RuntimeException e) {
                LOGGER.log(Level.WARNING, "submit task failed", e);
            }
            try {
                BatchEngine current = engine;
                if (current != null && !current.idle()) {
                    current.pump();
                }
            } catch (RuntimeException e) {
                LOGGER.log(Level.WARNING, "engine step failed", e);
            }
        }
    }
}
