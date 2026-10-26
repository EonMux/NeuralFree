package com.eonmux.neuralfree.gguf.engine.lora;

import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;

/**
 * One runtime adapter of a {@link LoraRegistry}: its registry name, its loaded {@link LoraAdapter}, the scale the
 * requests use and its adapter identity — the content digest of §5.5.1 N-2 (the canonical view metadata, the tensor
 * table and the tensor bytes under the exact-bytes rule) as the persistent identity of (digest, scale bits) in the
 * cache contexts, and the in-process revision of the held state. PC-0 mounts the N-2 epoch ids ({@code LoadIds}) on
 * the revision: every mutation — a reload, a static {@code setScale} also back at an earlier value, a slot change —
 * creates a new revision; two loads never share one, so no context of an earlier state is recreated by value.
 * A scale change of a per-request adapter rewrites no held state (LR-2 keys its slots by adapter and scale).
 *
 * <p>Mutable: the registry and the attachment it serves mutate {@code scale}, {@code revision} and {@code holds} on
 * the engine thread, like {@code Sampler} (§3.1); the registry never hands a handle out while an event runs. The
 * adapter's content is immutable; the revision and the hold count do not distinguish two loads of the same
 * name (use a different name for that).</p>
 * */
public final class LoraHandle {

    private final String name;
    private final LoraAdapter adapter;
    private long revision;
    private float scale;
    private int holds;

    LoraHandle(String name, LoraAdapter adapter, float scale) {
        this.name = name;
        this.adapter = adapter;
        this.scale = scale;
    }

    /** The registry name of the adapter (unique among the residents). */
    public String name() {
        return name;
    }

    /** The file the adapter was loaded from. */
    public String path() {
        return adapter.path();
    }

    /** The adapter's content. */
    public LoraAdapter adapter() {
        return adapter;
    }

    /** The scale the requests use: the static scale of an attached adapter, or the default of its later requests. */
    public float scale() {
        return scale;
    }

    /** The revision of the held state: 0 at load, raised by every new epoch (PC-0 mounts {@code LoadIds}). */
    public long revision() {
        return revision;
    }

    /** The content digest of the adapter (§5.5.1 N-2), as 64 lowercase hex characters. */
    public String digest() {
        return adapter.digest();
    }

    /** The number of bytes the digest covers ({@code adapter.contentLength()}). */
    public long contentLength() {
        return adapter.contentLength();
    }

    /** How many attachments hold the adapter; 0 for a detached or merely resident one. */
    public int holds() {
        return holds;
    }

    /**
     * Sets the scale of a held (statically applied) adapter: a new epoch of the held state, unlike the default
     * scale set by a per-request adapter ({@link #useDefaultScale}).
     */
    public void setScale(float scale) {
        this.scale = scale;
        revision++;
    }

    /**
     * Sets the scale of a per-request adapter without an epoch of held state: LR-2's slots key each request side by
     * the adapter and this scale.
     */
    public void useDefaultScale(float scale) {
        this.scale = scale;
    }

    /** An attachment began holding the adapter on the engine thread. */
    public void hold() {
        holds++;
    }

    /** An attachment released the adapter on the engine thread. */
    public void release() {
        if (holds == 0) {
            throw new IllegalStateException(name + " is held by no attachment; release balances hold");
        }
        holds--;
    }

    @Override
    public String toString() {
        return "LoraHandle{" + name + ", " + adapter.path() + ", scale=" + scale + ", rev=" + revision + ", holds="
                + holds + "}";
    }
}
