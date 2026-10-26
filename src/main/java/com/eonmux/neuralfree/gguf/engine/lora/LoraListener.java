package com.eonmux.neuralfree.gguf.engine.lora;

/**
 * The runtime adapter changes of a {@link LoraRegistry} the engine applies before the change ends
 * (llama.cpp server {@code /lora-adapters}, vLLM {@code /v1/load_lora_adapter}):
 * <ul>
 *   <li>{@link #loaded}: an adapter became resident; an attachment begins holding its deltas at the handle's
 *       scale;</li>
 *   <li>{@link #unloaded}: an adapter left the residents, by {@link LoraRegistry#unload} or by eviction to the
 *       host cache; an attachment removes its deltas and releases the device uploads;</li>
 *   <li>{@link #scaleChanged}: the scale of a resident adapter changed. In-process identity follows §5.5.1 N-2:
 *       a static adapter's scale change creates a new epoch (its handle's {@link LoraHandle#revision()} rises,
 *       also when the value returns to an earlier one) and PC-0 mounts the epoch ids on {@link LoraHandle}; a
 *       per-request default scale is one value of later requests' contexts (LR-2) and rewrites no held
 *       buffer.</li>
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
}
