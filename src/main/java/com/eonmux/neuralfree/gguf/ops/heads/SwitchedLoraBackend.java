package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra.Router;

import java.util.Objects;
import java.util.function.Function;

/**
 * A {@link ForwardBackend} that feeds a graniteswitch model's switched-LoRA {@link Router} (one router per model,
 * granite-switch.cpp:139-145, 216-221, 260-288), shared by the GPU and CPU orchestrators.
 *
 * <p>Every call that stores positions records its tokens in the router at their positions before the forward pass
 * and embeds the router's substitutes for the adapter-activation tokens; rollback, context shift and reset are
 * mirrored, so the router and the memory always describe the same positions. When a call fails, the router is
 * brought back to the positions the memory holds. The memoryless evaluations ({@link #forwardAll},
 * {@link #hiddenRows}) record their tokens from position 0 and clear the router afterwards; on a model whose memory
 * holds positions they are passed through untouched, so the delegate's refusal leaves the router intact. Not
 * thread-safe.</p>
 */
public final class SwitchedLoraBackend implements ForwardBackend {

    private final ForwardBackend delegate;
    private final Router router;

    private SwitchedLoraBackend(ForwardBackend delegate, Router router) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.router = Objects.requireNonNull(router, "router");
    }

    /** {@code delegate} fed through {@code router}; the returned backend closes the delegate. */
    public static ForwardBackend wrap(ForwardBackend delegate, Router router) {
        return new SwitchedLoraBackend(delegate, router);
    }

    @Override
    public float[] prefill(int[] tokens, int pos0) {
        return routed(pos0, tokens, t -> delegate.prefill(t, pos0));
    }

    @Override
    public float[] decode(int token, int pos) {
        return routed(pos, new int[]{token}, t -> delegate.decode(t[0], pos));
    }

    @Override
    public int decodeGreedyOnDevice(int token, int pos) {
        return routed(pos, new int[]{token}, t -> delegate.decodeGreedyOnDevice(t[0], pos));
    }

    @Override
    public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
        return routed(pos0, tokens, t -> delegate.forwardRows(t, pos0, outputRows));
    }

    /** Records {@code tokens} at {@code pos0}, runs {@code call} on their substitutes, resyncs on failure. */
    private <T> T routed(int pos0, int[] tokens, Function<int[], T> call) {
        Objects.requireNonNull(tokens, "tokens");
        router.accept(pos0, tokens);
        try {
            return call.apply(router.substitute(tokens));
        } catch (RuntimeException e) {
            resync(e);
            throw e;
        }
    }

    @Override
    public float[][] forwardAll(int[] tokens, int[] outputRows) {
        Objects.requireNonNull(tokens, "tokens");
        if (delegate.memory() != null) {
            return delegate.forwardAll(tokens, outputRows);
        }
        router.accept(0, tokens);
        try {
            return delegate.forwardAll(router.substitute(tokens), outputRows);
        } finally {
            router.reset();
        }
    }

    @Override
    public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
        Objects.requireNonNull(tokens, "tokens");
        if (delegate.memory() != null && delegate.memory().sequence().nPast() != 0) {
            return delegate.hiddenRows(tokens, tokenTypes);
        }
        router.accept(0, tokens);
        try {
            return delegate.hiddenRows(router.substitute(tokens), tokenTypes);
        } finally {
            router.reset();
        }
    }

    @Override
    public void reset() {
        delegate.reset();
        router.reset();
    }

    @Override
    public boolean canRollback(int nTokens) {
        return delegate.canRollback(nTokens);
    }

    @Override
    public void rollback(int nTokens) {
        delegate.rollback(nTokens);
        router.rollback(nTokens);
    }

    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        delegate.shiftContext(nKeep, nDiscard);
        router.shift(nKeep, nDiscard);
    }

    @Override
    public ModelMemory memory() {
        return delegate.memory();
    }

    @Override
    public void setHiddenTap(HiddenTap tap) {
        delegate.setHiddenTap(tap);
    }

    @Override
    public void close() {
        delegate.close();
    }

    /** Brings the router back to the memory's positions after a failed call. */
    private void resync(RuntimeException failure) {
        try {
            int past = delegate.memory() != null ? delegate.memory().sequence().nPast() : 0;
            int recorded = router.length();
            if (recorded > past) {
                router.rollback(recorded - past);
            } else if (recorded < past) {
                if (delegate.canRollback(past - recorded)) {
                    delegate.rollback(past - recorded);
                } else {
                    delegate.reset();
                    router.reset();
                }
            }
        } catch (RuntimeException e) {
            failure.addSuppressed(e);
        }
    }

    @Override
    public String toString() {
        return "switchedLora(" + delegate + ')';
    }
}
