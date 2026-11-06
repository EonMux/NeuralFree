package com.eonmux.neuralfree.gguf.gpu.diag;

import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;

import java.util.Objects;

/**
 * The execution observer of one device's forward context: the diagnostics from {@code neuralfree.diag} (or
 * nothing) as its base, temporarily replaced by another observer, a profiler while a benchmark runs. A one-device
 * model has one slot; every stage of a layer split has its own, so each observes the stages of its own device.
 * Executors consult {@link #enabled()} per step. Not thread-safe.
 */
public final class ObserverSlot implements ExecutionObserver {

    private ExecutionObserver base = ExecutionObserver.NONE;
    private ExecutionObserver active = ExecutionObserver.NONE;

    /** Makes {@code o} the base observer and the active one. */
    public void setBase(ExecutionObserver o) {
        base = Objects.requireNonNull(o, "observer");
        active = base;
    }

    /** The base observer, which {@link #restore()} makes active again. */
    public ExecutionObserver base() {
        return base;
    }

    /** Replaces the active observer with {@code o} until {@link #restore()}. */
    public void use(ExecutionObserver o) {
        active = Objects.requireNonNull(o, "observer");
    }

    /** Makes the base observer active again. */
    public void restore() {
        active = base;
    }

    @Override
    public boolean enabled() {
        return active.enabled();
    }

    @Override
    public void onStepStart(int pos0, int nTokens) {
        active.onStepStart(pos0, nTokens);
    }

    @Override
    public void onStageStart(int layer, Stage stage) {
        active.onStageStart(layer, stage);
    }

    @Override
    public void onStageEnd(int layer, Stage stage, ScratchSlot result, int nRows, int width) {
        active.onStageEnd(layer, stage, result, nRows, width);
    }

    @Override
    public void onStepEnd(int pos0, int nTokens) {
        active.onStepEnd(pos0, nTokens);
    }

    @Override
    public String toString() {
        return "ObserverSlot{" + active + '}';
    }
}
