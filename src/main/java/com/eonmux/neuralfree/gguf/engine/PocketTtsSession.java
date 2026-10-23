package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.ops.audio.TtsBackbone;

import java.io.IOException;
import java.util.Objects;

/**
 * Caller-side access to the pockettts backbone: the hidden states {@code t_embd} after the biased output LayerNorm
 * (llama.cpp src/models/pockettts.cpp:132-138), which the flow network of a pockettts pipeline consumes. KV-cached
 * {@link #prefill} and {@link #decode} read the residual stream through {@link TtsBackbone.PocketHidden}, which holds
 * the model's hidden-state tap ({@link HiddenTapSlot}) until {@link #close()}; {@link #hiddenRows} evaluates a whole
 * sequence without the cache.
 *
 * <p>Positions follow the model memory: a call at {@code pos0} must continue the stored sequence
 * ({@link #nPast()}), as for {@link ForwardBackend#prefill}. Not thread-safe; the model stays open after
 * {@link #close()}.
 */
public final class PocketTtsSession implements AutoCloseable {

    private final EngineModel model;
    private final ForwardBackend backend;
    private final HiddenTapSlot.Lease lease;
    private final TtsBackbone.PocketHidden hidden;
    private boolean closed;

    private PocketTtsSession(EngineModel model, HiddenTapSlot.Lease lease, TtsBackbone.PocketHidden hidden) {
        this.model = model;
        this.backend = lease.backend();
        this.lease = lease;
        this.hidden = hidden;
    }

    /**
     * Binds a session to a pockettts model opened through {@link LLMModelFactory}.
     *
     * @throws IllegalArgumentException when the model is not pockettts
     * @throws IllegalStateException    when another user (a speculative drafter, another session) holds the model's
     *                                  hidden-state tap
     * @throws IOException              when the output-norm tensors cannot be read
     */
    public static PocketTtsSession open(EngineModel model) throws IOException {
        Objects.requireNonNull(model, "model");
        if (model.spec().arch() != Arch.POCKETTTS) {
            throw new IllegalArgumentException("pockettts hidden states need a " + Arch.POCKETTTS.ggufName()
                    + " model, not " + model.architecture());
        }
        ForwardBackend fb = model.forwardBackend();
        HiddenTapSlot.Lease lease = HiddenTapSlot.acquire(fb, "the pockettts hidden-state reader");
        try {
            TtsBackbone.PocketHidden h = TtsBackbone.PocketHidden.attach(fb, model.spec(),
                    LLMModelFactory.hostWeights(model));
            return new PocketTtsSession(model, lease, h);
        } catch (IOException | RuntimeException e) {
            lease.close();
            throw e;
        }
    }

    /** {@code n_embd}, the width of every returned row. */
    public int width() {
        return hidden.width();
    }

    /** Positions stored in the model memory; the next call continues at this position. */
    public int nPast() {
        ensureOpen();
        ModelMemory m = backend.memory();
        return m == null ? 0 : m.sequence().nPast();
    }

    /** Evaluates {@code tokens} at {@code pos0 ..} into the cache; returns {@code t_embd} of the last token. */
    public float[] prefill(int[] tokens, int pos0) {
        ensureOpen();
        return hidden.prefill(tokens, pos0);
    }

    /** Evaluates {@code tokens} at {@code pos0 ..} into the cache; returns {@code t_embd} of every token. */
    public float[][] prefillAll(int[] tokens, int pos0) {
        ensureOpen();
        return hidden.prefillAll(tokens, pos0);
    }

    /** Evaluates one token at {@code pos} into the cache; returns its {@code t_embd}. */
    public float[] decode(int token, int pos) {
        ensureOpen();
        return hidden.decode(token, pos);
    }

    /**
     * {@code t_embd} of a whole sequence from position 0 without the cache. The memory is emptied first, because a
     * memoryless pass runs over an empty memory.
     */
    public float[][] hiddenRows(int[] tokens) {
        ensureOpen();
        if (backend.memory() != null) {
            backend.reset();
        }
        return hidden.hiddenRows(tokens);
    }

    /** Empties the model memory. */
    public void reset() {
        ensureOpen();
        backend.reset();
    }

    /** The model this session reads. */
    public EngineModel model() {
        return model;
    }

    /** Removes the tap and frees the model's tap slot; the model stays open. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            hidden.close();
        } finally {
            lease.close();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the pockettts session is closed");
        }
    }

    @Override
    public String toString() {
        return "PocketTtsSession{" + model.architecture() + ", width=" + hidden.width() + (closed ? ", closed" : "")
                + '}';
    }
}
