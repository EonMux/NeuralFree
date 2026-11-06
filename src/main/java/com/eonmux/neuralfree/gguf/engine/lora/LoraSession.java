package com.eonmux.neuralfree.gguf.engine.lora;

import java.util.Objects;

/**
 * The adapter slots one request holds while its passes run alone on the model — a single-sequence generation or a
 * pooling call — over passes whose batches carry no per-row slots: {@link #open} acquires the slots of a
 * {@link LoraPlan}, {@link #runKey} and {@link #runPre} put the following passes in the slot of the rows from the
 * plan's start on, or before it ({@link LoraRegistry#useSlot}), and {@link #close} returns the passes to slot 0
 * and releases the slots.
 *
 * <p>Mutable and owned by the engine thread the registry runs on, for the duration of one request.</p>
 */
public final class LoraSession implements AutoCloseable {

    private final LoraRegistry registry;
    private final int slot;
    private final int preSlot;
    private boolean closed;

    private LoraSession(LoraRegistry registry, int slot, int preSlot) {
        this.registry = registry;
        this.slot = slot;
        this.preSlot = preSlot;
    }

    /**
     * Acquires the slots of {@code plan} on {@code registry}; the passes stay in slot 0 until {@link #runKey} or
     * {@link #runPre}.
     *
     * @throws IllegalStateException when every adapter slot serves running requests of other keys, or an adapter
     *                               of the plan left the residents
     */
    public static LoraSession open(LoraRegistry registry, LoraPlan plan) {
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(plan, "plan");
        int slot = acquire(registry, plan.key());
        int pre;
        try {
            pre = plan.preKey() == null ? 0 : plan.preKey().equals(plan.key()) ? slot
                    : acquire(registry, plan.preKey());
        } catch (RuntimeException e) {
            if (slot > 0) {
                registry.release(slot);
            }
            throw e;
        }
        return new LoraSession(registry, slot, pre);
    }

    private static int acquire(LoraRegistry registry, LoraSlotKey key) {
        if (key == null) {
            return 0;
        }
        int slot = registry.acquire(key);
        if (slot < 0) {
            throw new IllegalStateException("lora: every one of the " + registry.slotCount() + " adapter slots ("
                    + LoraConfig.PROP_MAX + ") serves running requests of other adapters; " + key.label()
                    + " finds none free");
        }
        return slot;
    }

    /** The slot of the rows from the plan's start on. */
    public int slot() {
        return slot;
    }

    /** The slot of the rows before the plan's start. */
    public int preSlot() {
        return preSlot;
    }

    /** Runs the following passes in the slot of the rows from the plan's start on. */
    public void runKey() {
        registry.useSlot(slot);
    }

    /** Runs the following passes in the slot of the rows before the plan's start. */
    public void runPre() {
        registry.useSlot(preSlot);
    }

    /** Returns the passes to slot 0 and releases the slots. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            registry.useSlot(0);
        } finally {
            if (slot > 0) {
                registry.release(slot);
            }
            if (preSlot > 0 && preSlot != slot) {
                registry.release(preSlot);
            }
        }
    }
}
