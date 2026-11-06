package com.eonmux.neuralfree.gguf.engine.lora;

/**
 * The runtime adapter changes of a {@link LoraRegistry} the engine applies before the change ends
 * (llama.cpp server {@code /lora-adapters}, vLLM {@code /v1/load_lora_adapter}):
 * <ul>
 *   <li>{@link #loaded}: an adapter became resident; an attachment begins holding its deltas at the handle's
 *       scale;</li>
 *   <li>{@link #unloaded}: an adapter left the residents, by {@link LoraRegistry#unload} or by eviction to the
 *       host cache; an attachment removes its deltas and releases the device uploads;</li>
 *   <li>{@link #scaleChanged}: the scale of a resident adapter changed. In-process identity follows the handle's
 *       revision: a static adapter's scale change creates a new epoch (its handle's {@link LoraHandle#revision()}
 *       rises, also when the value returns to an earlier one); a
 *       per-request default scale is one value of later requests' contexts and rewrites no held
 *       buffer.</li>
 *   <li>{@link #slotOpened} / {@link #slotRescaled} / {@link #slotClosed}: an adapter slot of the per-request slot
 *       table opened for a request's adapters ({@link LoraRegistry#acquire}) or closed (displaced by another key,
 *       or an adapter of it left the residents); an attachment registers or removes the slot's deltas, which apply
 *       to the rows a forward batch runs in that slot. A key whose adapters are the startup set at other scales
 *       opens rescaled: the slot runs the slot-0 deltas at the key's factor of their bound scales instead of
 *       uploading a copy, so a startup adapter above {@code neuralfree.lora.max.rank} rescales within the slot
 *       reservation too;</li>
 *   <li>{@link #sessionSlot}: the slot of the passes whose batch sets no per-row slots changed
 *       ({@link LoraRegistry#useSlot}, a single-sequence run or a pooling call of one request);</li>
 *   <li>{@link #checkSlot}: a request's key is checked before it is accepted, so a slot the listener could not
 *       open refuses at submission.</li>
 * </ul>
 * */
public interface LoraListener {

    /** An adapter became resident and is attached from now on, at {@code handle.scale()}. */
    void loaded(LoraHandle handle);

    /**
     * An adapter left the residents and is detached from now on: its deltas are unregistered and the queues that
     * ran them must be finished before its device uploads are reused.
     */
    void unloaded(LoraHandle handle);

    /** The scale of {@code handle} changed from {@code previous} to {@code handle.scale()}. */
    void scaleChanged(LoraHandle handle, float previous);

    /**
     * Slot {@code slot} (1 and up) opened for {@code key}: the rows a forward batch runs in that slot take the
     * key's adapters at their scales from now on. A listener that cannot serve the slot throws, and the acquire
     * that opened it fails. Default: nothing to register.
     */
    default void slotOpened(int slot, LoraSlotKey key) {
    }

    /**
     * Slot {@code slot} (1 and up) opened as a rescale of the startup set: the key's adapters are the slot-0
     * adapters at other scales, so the slot runs the slot-0 deltas at the key's factor of their bound scales and
     * uploads nothing (the deltas stay shared with slot 0). A listener that cannot serve the slot throws, and the
     * acquire that opened it fails; the slot closes like any other. Default: nothing to register.
     */
    default void slotRescaled(int slot, LoraSlotKey key) {
    }

    /**
     * Slot {@code slot} closed: its deltas are removed and the queues that ran them must be finished before their
     * device uploads are reused. Default: nothing registered.
     */
    default void slotClosed(int slot, LoraSlotKey key) {
    }

    /**
     * The rows of the passes whose batch sets no per-row slots run in {@code slot} from now on (0 for the startup
     * adapters). Default: nothing to set.
     */
    default void sessionSlot(int slot) {
    }

    /**
     * Checks that a slot of {@code key} could open on this listener; throws {@link IllegalArgumentException}
     * naming the reason when it could not (a slot larger than the reserved device memory). Default: every key can.
     */
    default void checkSlot(LoraSlotKey key) {
    }
}
