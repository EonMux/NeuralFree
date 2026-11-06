package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.lora.LoraHandle;
import com.eonmux.neuralfree.gguf.engine.lora.LoraListener;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;
import java.util.function.Predicate;

/**
 * The generic LoRA adapters of one model attached to its forward pass (llama.cpp {@code --lora} /
 * {@code --lora-scaled}, applied by {@code build_lora_mm}, {@code build_lora_mm_id} and {@code build_lora_get_rows},
 * src/llama-graph.cpp:1629-1749, 2694-2709): every pair of every adapter becomes a delta of its kind in the model's
 * {@link GpuProjectionDeltas} or {@link CpuProjectionDeltas}, keyed by the pair's base tensor name. Several adapters
 * on one tensor add up in load order, as llama.cpp's loop over {@code loras} does.
 *
 * <p>Attaching happens after the forward blocks are bound. A pair whose base tensor no bound block applies deltas to
 * (a product the model's graph does not run, or one llama.cpp leaves without LoRA such as a pooling head) fails the
 * whole attachment with the adapter and tensor named, as does a block that refuses deltas (a PRISM-rotated input,
 * the absorbed MLA {@code attn_kv_b}); nothing is registered then. A device forward pass split over several GPUs
 * ({@link #gpu(List, List)}, one {@link GpuDevice} per stage) registers every pair on the device whose bound blocks
 * serve its base tensor, so each stage applies the pairs of its own layers and the main stage those of the embedding
 * and the output head. {@link #follow} connects the attachment to a
 * {@link LoraRegistry}: every runtime change (llama.cpp {@code /lora-adapters}, vLLM
 * {@code /v1/load_lora_adapter}) is applied synchronously on the engine thread the registry runs on — a loaded
 * adapter becomes attached, an unloaded or evicted one is detached with its device uploads released (the queues that
 * ran them must be finished, as on {@link #close}), and a scale change rescales its {@code scale * B} in place
 * between steps. A hot change fails the registry call that caused it and leaves nothing registered.</p>
 *
 * <p>{@link #serve} connects it to a registry of per-request adapters instead (vLLM's multi-LoRA serving): the
 * static bindings become the registry's startup adapters ({@link LoraRegistry#adoptStartup}, llama.cpp lists them
 * first) and stay registered in slot 0 for the rows of requests that select no adapter; a loaded adapter is
 * validated against the bound blocks but attached to no row; when the registry opens an adapter slot for a
 * request's adapters ({@link LoraListener#slotOpened}) their pairs — startup adapters included when the key lists
 * them — are registered at that slot, at the slot key's scales, so they apply only to the rows a forward batch runs
 * in the slot, in place of slot 0's; a closed slot's deltas go and their device uploads are released. The bare slot
 * ({@link LoraRegistry#bareSlot}) opens holding nothing. On a device forward pass, {@link #reserveSlots} bounds the
 * uploads of one slot by the device memory the load reserved for it, so a slot open never allocates past the
 * budget.</p>
 *
 * <p>{@link #hybrid} attaches to a hybrid CPU+GPU layer split: a pair goes to the device whose bound blocks serve its
 * base tensor, else to the host blocks that do (the layers kept on the CPU).</p>
 *
 * <p>The {@link LoraAdapter.Side} overloads of {@link #cpu} and {@link #gpu} serve only that side's pairs: the text
 * forward pass attaches the text-model pairs of an adapter, and a vision encoder attaches its mmproj pairs, so one
 * adapter converted for both models runs on both. {@link #session} opens a private adapter slot for one caller
 * without a registry: a media encoder runs a request's adapters only while it encodes that request's media,
 * registering their pairs of its side at the session slot and pointing its rows there (a bare session runs no
 * adapters where slot 0 has some); closing the attachment drops them.</p>
 *
 * <p>Mutable, owned by the engine thread: it belongs to the forward pass it attaches its deltas to, and the
 * registry events it subscribes to are delivered on that same thread. {@link #follow} and the events balance
 * the handles' hold counts of the registry.</p>
 */
public final class LoraAttachment implements AutoCloseable, LoraListener {

    /** One loaded adapter and its user scale. Immutable. */
    public static final class Binding {
        private final LoraAdapter adapter;
        private final float scale;

        public Binding(LoraAdapter adapter, float scale) {
            this.adapter = Objects.requireNonNull(adapter, "adapter");
            if (!Float.isFinite(scale)) {
                throw new IllegalArgumentException("LoRA adapter " + adapter.path() + " has a non-finite scale "
                        + scale);
            }
            this.scale = scale;
        }

        /**
         * Loads the adapter at {@code path} against the base model's weights ({@link LoraAdapterLoader#load}).
         *
         * @param options how a checkpoint that is not GGUF is opened
         * @throws IOException when the file cannot be read
         */
        public static Binding load(String path, float scale, CpuWeights base, ModelSourceOptions options)
                throws IOException {
            return new Binding(LoraAdapterLoader.load(path, base, options), scale);
        }

        public LoraAdapter adapter() { return adapter; }
        public float scale() { return scale; }

        @Override
        public String toString() {
            return adapter.path() + ":" + scale;
        }
    }

    /**
     * One device of a device forward pass: the dispatch helpers its deltas enqueue on and the projection deltas its
     * bound blocks serve. Immutable.
     */
    public static final class GpuDevice {
        private final GpuOps ops;
        private final GpuProjectionDeltas deltas;

        public GpuDevice(GpuOps ops, GpuProjectionDeltas deltas) {
            this.ops = Objects.requireNonNull(ops, "ops");
            this.deltas = Objects.requireNonNull(deltas, "deltas");
        }

        public GpuOps ops() { return ops; }
        public GpuProjectionDeltas deltas() { return deltas; }
    }

    /**
     * One registered delta: its adapter, the device it is registered on (-1 on the host), its adapter slot and its
     * delta. A {@code shared} delta belongs to slot 0 (which owns and closes it); its slot registration is the
     * rescale's reference to it, dropped without a close when the slot closes.
     */
    private static final class Registered {
        final LoraAdapter adapter;
        final String name;
        final int device;
        final int slot;
        final Object delta;
        final boolean shared;

        Registered(LoraAdapter adapter, String name, int device, int slot, Object delta, boolean shared) {
            this.adapter = adapter;
            this.name = name;
            this.device = device;
            this.slot = slot;
            this.delta = delta;
            this.shared = shared;
        }
    }

    private final List<Binding> bindings = new ArrayList<>();
    /** The registry handles this attachment holds, to release them on the events of {@link #follow}. */
    private final Map<LoraAdapter, LoraHandle> held = new HashMap<>();
    /** The devices of a device forward pass, in stage order; empty for the host forward pass. */
    private final List<GpuDevice> devices;
    private final CpuProjectionDeltas cpu;
    /** The model whose pairs this attachment serves: the text forward pass, or a vision encoder. */
    private final LoraAdapter.Side side;
    private final List<Registered> registered = new ArrayList<>();
    private final List<AutoCloseable> owned = new ArrayList<>();
    /** Whether the followed registry's adapters run per request ({@link #serve}) rather than on every row. */
    private boolean perRequest;
    /** Device bytes one adapter slot may upload per device ({@link #reserveSlots}); null for no bound. */
    private long[] slotBudget;
    /** The scratch ubatch the device deltas allocate their zero slots for, with {@link #slotBudget}. */
    private int slotUbatch;
    /** How {@link #slotBudget} was set, named in a refusal. */
    private String budgetSource;
    /** Whether a registry is connected, by {@link #follow} or {@link #serve}. */
    private boolean connected;
    private boolean closed;

    private LoraAttachment(List<Binding> bindings, List<GpuDevice> devices, CpuProjectionDeltas cpu,
                           LoraAdapter.Side side) {
        this.bindings.addAll(bindings);
        this.devices = Collections.unmodifiableList(new ArrayList<>(devices));
        this.cpu = cpu;
        this.side = Objects.requireNonNull(side, "side");
    }

    /**
     * Uploads and registers every pair of the {@code side} model in {@code bindings} on the device forward pass
     * whose blocks share {@code deltas}, dispatching on the queue of {@code ops}. Every pair is validated before
     * the first upload.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment gpu(List<Binding> bindings, GpuOps ops, GpuProjectionDeltas deltas,
                                     LoraAdapter.Side side) {
        Objects.requireNonNull(ops, "ops");
        return gpu(bindings, Collections.singletonList(new GpuDevice(ops, deltas)), side);
    }

    /**
     * Uploads and registers every pair of {@code bindings} on the device forward pass whose blocks share
     * {@code deltas}, dispatching on the queue of {@code ops}. Every pair is validated before the first upload.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment gpu(List<Binding> bindings, GpuOps ops, GpuProjectionDeltas deltas) {
        return gpu(bindings, ops, deltas, LoraAdapter.Side.TEXT);
    }

    /**
     * {@link #gpu(List, List)} for the pairs of the {@code side} model: a vision encoder attaches its adapter's
     * {@link LoraAdapter.Side#VISION} pairs and ignores the text-model ones, which the text forward pass serves.
     *
     * @throws IllegalArgumentException      when {@code devices} is empty
     * @throws IllegalStateException         when a pair targets a product no bound block of any device applies to,
     *                                       blocks of two devices serve it, or the served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment gpu(List<Binding> bindings, List<GpuDevice> devices, LoraAdapter.Side side) {
        Objects.requireNonNull(devices, "devices");
        if (devices.isEmpty()) {
            throw new IllegalArgumentException("a device LoRA attachment needs at least one device");
        }
        return attachAll(new LoraAttachment(Objects.requireNonNull(bindings, "bindings"), devices, null, side));
    }

    /**
     * Uploads and registers every pair of {@code bindings} on the device forward pass of {@code devices}, the stages
     * of a layer split: each pair goes to the device whose bound blocks serve its base tensor and is uploaded and
     * dispatched there. Every pair is validated before the first upload.
     *
     * @throws IllegalArgumentException      when {@code devices} is empty
     * @throws IllegalStateException         when a pair targets a product no bound block of any device applies to,
     *                                       blocks of two devices serve it, or the served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment gpu(List<Binding> bindings, List<GpuDevice> devices) {
        return gpu(bindings, devices, LoraAdapter.Side.TEXT);
    }

    /**
     * Uploads and registers every pair of {@code bindings} on a hybrid CPU+GPU layer split: each pair goes to the
     * device of {@code devices} whose bound blocks serve its base tensor, else to the host blocks sharing
     * {@code host}. Every pair is validated before the first registration.
     *
     * @throws IllegalArgumentException      when {@code devices} is empty
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, blocks of two
     *                                       sides serve it, or the served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment hybrid(List<Binding> bindings, List<GpuDevice> devices, CpuProjectionDeltas host) {
        Objects.requireNonNull(devices, "devices");
        Objects.requireNonNull(host, "host");
        if (devices.isEmpty()) {
            throw new IllegalArgumentException("a hybrid LoRA attachment needs at least one device");
        }
        return attachAll(new LoraAttachment(Objects.requireNonNull(bindings, "bindings"), devices, host,
                LoraAdapter.Side.TEXT));
    }

    /** Validates every pair of the attachment's bindings, then registers them in slot 0. */
    private static LoraAttachment attachAll(LoraAttachment att) {
        try {
            for (Binding b : att.bindings) {
                att.checkAdapter(b.adapter);
            }
            for (Binding b : att.bindings) {
                for (String target : att.targetsOf(b.adapter)) {
                    att.attach(b.adapter, target, b.scale(), 0);
                }
            }
            return att;
        } catch (RuntimeException e) {
            att.closeAfterFailure(e);
            throw e;
        }
    }

    /**
     * Registers every pair of {@code bindings} on the host forward pass whose blocks share {@code deltas}. Every
     * pair is validated before the first registration.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment cpu(List<Binding> bindings, CpuProjectionDeltas deltas) {
        return cpu(bindings, deltas, LoraAdapter.Side.TEXT);
    }

    /**
     * {@link #cpu(List, CpuProjectionDeltas)} for the pairs of the {@code side} model: a vision encoder attaches its
     * adapter's {@link LoraAdapter.Side#VISION} pairs and ignores the text-model ones, which the text forward pass
     * serves.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment cpu(List<Binding> bindings, CpuProjectionDeltas deltas, LoraAdapter.Side side) {
        Objects.requireNonNull(deltas, "deltas");
        return attachAll(new LoraAttachment(Objects.requireNonNull(bindings, "bindings"),
                Collections.<GpuDevice>emptyList(), deltas, side));
    }

    /**
     * Opens a one-caller adapter session of the {@code side} model on the host forward pass whose blocks share
     * {@code deltas}: every pair of {@code bindings} is registered in a private adapter slot, and the rows the
     * session's caller runs apply that slot instead of slot 0, so they see the session's pairs and no static
     * attachment's. An empty list opens the session bare: its rows run no adapters where slot 0 has some. Not
     * connected to a registry; the caller closes the session ({@link #close}) when its rows are done.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment session(List<Binding> bindings, CpuProjectionDeltas deltas,
                                         LoraAdapter.Side side) {
        Objects.requireNonNull(deltas, "deltas");
        return session(new LoraAttachment(Objects.requireNonNull(bindings, "bindings"),
                Collections.<GpuDevice>emptyList(), deltas, side));
    }

    /**
     * {@link #session(List, CpuProjectionDeltas, LoraAdapter.Side)} on the device forward pass whose blocks share
     * {@code deltas}, dispatching on the queue of {@code ops}.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment session(List<Binding> bindings, GpuOps ops, GpuProjectionDeltas deltas,
                                         LoraAdapter.Side side) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(deltas, "deltas");
        return session(new LoraAttachment(Objects.requireNonNull(bindings, "bindings"),
                Collections.singletonList(new GpuDevice(ops, deltas)), null, side));
    }

    /**
     * The adapter slot one session registers in: slots 1 and above are the private slots a registry hands out per
     * request, and a session is the encoder's own single one.
     */
    private static final int SESSION_SLOT = 1;

    /** Whether this attachment is a one-caller session ({@link #session}), holding {@link #sessionSlot}. */
    private boolean session;
    /** The adapter slot a session registers in ({@code 0} when this is not a session). */
    private int sessionSlot;

    /**
     * Opens the session attachment: the session slot is opened first (an empty slot runs nothing, and the slot's
     * allocator exists before any pair uploads into it), every pair is validated, the session's pairs are registered
     * into the slot, and the rows the caller runs are set to it.
     */
    private static LoraAttachment session(LoraAttachment att) {
        att.session = true;
        att.sessionSlot = SESSION_SLOT;
        try {
            att.openSlot(SESSION_SLOT);
            for (Binding b : att.bindings) {
                att.checkAdapter(b.adapter);
            }
            for (Binding b : att.bindings) {
                for (String target : att.targetsOf(b.adapter)) {
                    att.attach(b.adapter, target, b.scale(), SESSION_SLOT);
                }
            }
            att.sessionSlot(SESSION_SLOT);
            return att;
        } catch (RuntimeException e) {
            att.closeAfterFailure(e);
            throw e;
        }
    }

    /**
     * Applies every following runtime change of {@code registry} to this forward pass: an adapter the registry
     * loads becomes attached (at {@code LoraHandle.scale()}), an unloaded or evicted one detached, a scale change
     * rescaled. The registry delivers the events synchronously on the engine thread this attachment's deltas serve.
     */
    public void follow(LoraRegistry registry) {
        connect(registry, false);
    }

    /**
     * Serves the per-request adapters of {@code registry}: the static bindings become its startup adapters, held in
     * slot 0 ({@link LoraRegistry#adoptStartup}); an adapter it loads is validated against the bound blocks (an
     * unrunnable pair fails the load) and attached to no row; the slots it opens for requests register their
     * adapters' pairs at the slot, the slots it closes remove them, and its bare slot opens empty. The events are
     * delivered synchronously on the engine thread this attachment's deltas serve.
     *
     * @throws IllegalStateException when the attachment already follows or serves a registry, or the registry holds
     *                               per-request adapters already
     */
    public void serve(LoraRegistry registry) {
        connect(registry, true);
        for (Binding b : bindings) {
            LoraHandle handle = registry.adoptStartup(b.adapter(), b.scale());
            handle.hold();
            held.put(b.adapter(), handle);
        }
        int bare = registry.bareSlot();
        if (bare > 0) {
            openSlot(bare);
        }
    }

    /**
     * Bounds the device uploads of one per-request slot: {@code perDevice[d]} bytes on device {@code d}, the share
     * of the slot reservation the load held back from the VRAM budget ({@code maxLoras} slots of
     * {@code maxLoraRank} pairs over the adapted products), with the device deltas' zero slots sized for
     * {@code ubatch}. A key whose uploads exceed it is refused when it resolves ({@link LoraListener#checkSlot});
     * a budget of 0 bytes refuses every slot that uploads, naming {@code source}.
     *
     * @throws IllegalArgumentException when the array does not hold one entry per device, or for a negative entry
     */
    public void reserveSlots(long[] perDevice, int ubatch, String source) {
        Objects.requireNonNull(perDevice, "perDevice");
        if (perDevice.length != devices.size()) {
            throw new IllegalArgumentException("the slot reservation names " + perDevice.length + " devices, the"
                    + " attachment has " + devices.size());
        }
        for (long b : perDevice) {
            if (b < 0) {
                throw new IllegalArgumentException("a slot reservation is non-negative, got " + b);
            }
        }
        if (ubatch < 1) {
            throw new IllegalArgumentException("ubatch must be positive, got " + ubatch);
        }
        this.slotBudget = perDevice.clone();
        this.slotUbatch = ubatch;
        this.budgetSource = Objects.requireNonNull(source, "source");
    }

    private void connect(LoraRegistry registry, boolean perRequestAdapters) {
        Objects.requireNonNull(registry, "registry");
        if (connected) {
            throw new IllegalStateException("the attachment is connected to a registry already");
        }
        connected = true;
        perRequest = perRequestAdapters;
        registry.addListener(this);
    }

    // ------------------------------------------------------------------ the registry events

    /** {@link LoraRegistry} loaded one adapter: it becomes attached, or the load fails. */
    @Override
    public void loaded(LoraHandle handle) {
        Objects.requireNonNull(handle, "handle");
        if (closed) {
            throw new IllegalStateException("the attachment is closed; it cannot attach " + handle);
        }
        if (held.containsKey(handle.adapter())) {
            throw new IllegalStateException("the adapter at " + handle.adapter().path() + " is already attached,"
                    + " the registry cannot load it twice");
        }
        if (perRequest) {
            checkAdapter(handle.adapter());
            return;
        }
        try {
            attachAdapter(handle.adapter(), handle.scale(), 0);
        } catch (RuntimeException e) {
            detach(handle.adapter());
            throw e;
        }
        handle.hold();
        held.put(handle.adapter(), handle);
        bindings.add(new Binding(handle.adapter(), handle.scale()));
    }

    /** {@link LoraRegistry} unloaded or evicted one adapter: its deltas go, and it is no longer attached. */
    @Override
    public void unloaded(LoraHandle handle) {
        Objects.requireNonNull(handle, "handle");
        LoraHandle heldHandle = held.remove(handle.adapter());
        if (heldHandle == null) {
            return;
        }
        detach(handle.adapter());
        handle.release();
        bindings.removeIf(b -> b.adapter() == handle.adapter());
    }

    /**
     * {@link LoraRegistry} changed the scale of one adapter: its pairs are rescaled to {@code handle.scale()}. An
     * adapter without deltas held by this attachment has nothing to rewrite (the default scale of its later
     * requests).
     */
    @Override
    public void scaleChanged(LoraHandle handle, float previous) {
        Objects.requireNonNull(handle, "handle");
        if (held.containsKey(handle.adapter())) {
            rescale(handle.adapter(), handle.scale());
        }
    }

    /**
     * {@link LoraRegistry} opened slot {@code slot} for a request's adapters: each adapter of {@code key} is
     * registered at the slot, at the key's scale, or nothing is when one fails.
     */
    @Override
    public void slotOpened(int slot, LoraSlotKey key) {
        Objects.requireNonNull(key, "key");
        if (closed) {
            throw new IllegalStateException("the attachment is closed; it cannot open adapter slot " + slot);
        }
        if (slot < 1) {
            throw new IllegalArgumentException("per-request adapter slots start at 1, got " + slot);
        }
        try {
            checkBudget(key);
            openSlot(slot);
            for (LoraSlotKey.Part part : key.parts()) {
                attachAdapter(part.handle().adapter(), part.scale(), slot);
            }
        } catch (RuntimeException e) {
            detachSlot(slot);
            closeSlot(slot);
            throw e;
        }
    }

    /** {@link LoraRegistry} closed slot {@code slot}: its deltas go and their device uploads are released. */
    @Override
    public void slotClosed(int slot, LoraSlotKey key) {
        detachSlot(slot);
        closeSlot(slot);
    }

    /**
     * {@link LoraRegistry} opened slot {@code slot} as a rescale of the startup set: every adapter of {@code key}
     * is resident in slot 0 at a non-zero scale, so the slot registers slot 0's deltas of each adapter again — at
     * the key's factor of their bound scale ({@code requested / resident}), shared with slot 0, uploaded nowhere —
     * and the rows of the slot run the startup factors at the requested scales. Nothing is copied, so a startup
     * adapter whose rank exceeds {@code neuralfree.lora.max.rank} rescales within the slot reservation too. The
     * registration drops without a close when the slot closes ({@link #slotClosed}).
     *
     * @throws IllegalArgumentException when an adapter of {@code key} is not resident in slot 0 at a non-zero scale
     */
    @Override
    public void slotRescaled(int slot, LoraSlotKey key) {
        Objects.requireNonNull(key, "key");
        if (closed) {
            throw new IllegalStateException("the attachment is closed; it cannot open adapter slot " + slot);
        }
        if (slot < 1) {
            throw new IllegalArgumentException("per-request adapter slots start at 1, got " + slot);
        }
        for (LoraSlotKey.Part part : key.parts()) {
            LoraHandle resident = held.get(part.handle().adapter());
            if (resident == null || resident != part.handle() || resident.scale() <= 0.0f) {
                throw new IllegalArgumentException("lora: " + part.handle().name() + " is not a startup adapter"
                        + " resident in slot 0 at a non-zero scale, so slot " + slot + " cannot rescale it");
            }
        }
        try {
            openSlot(slot);
            for (LoraSlotKey.Part part : key.parts()) {
                float factor = part.scale() / part.handle().scale();
                rescaleAdapter(part.handle().adapter(), slot, factor);
            }
        } catch (RuntimeException e) {
            detachSlot(slot);
            closeSlot(slot);
            throw e;
        }
    }

    /**
     * Registers every slot-0 delta of {@code adapter} again in slot {@code slot} at {@code factor}, shared with
     * slot 0 (which owns the delta), on the side that serves each pair.
     */
    private void rescaleAdapter(LoraAdapter adapter, int slot, float factor) {
        List<Registered> resident = new ArrayList<>();
        for (Registered r : registered) {
            if (r.adapter == adapter && r.slot == 0) {
                resident.add(r);
            }
        }
        for (Registered r : resident) {
            LoraAdapter.Pair pair = adapter.pair(r.name);
            register(adapter, r.name, pair, r.device, slot, factor, r.delta, true);
        }
    }

    /** Registers one shared reference to {@code d} in slot {@code slot} at {@code scale}; nothing is owned. */
    private void register(LoraAdapter adapter, String target, LoraAdapter.Pair pair, int device, int slot,
                          float scale, Object d, boolean shared) {
        GpuProjectionDeltas gpu = device >= 0 ? devices.get(device).deltas() : null;
        switch (pair.kind()) {
            case EXPERTS:
                if (gpu != null) {
                    gpu.registerExperts(target, slot, scale, (GpuProjectionDeltas.ExpertDelta) d);
                } else {
                    cpu.registerExperts(target, slot, scale, (CpuProjectionDeltas.ExpertDelta) d);
                }
                break;
            case EMBEDDING:
                if (gpu != null) {
                    gpu.registerEmbedding(target, slot, scale, (GpuProjectionDeltas.EmbeddingDelta) d);
                } else {
                    cpu.registerEmbedding(target, slot, scale, (CpuProjectionDeltas.EmbeddingDelta) d);
                }
                break;
            default:
                if (gpu != null) {
                    gpu.register(target, slot, scale, (GpuProjectionDeltas.Delta) d);
                } else {
                    cpu.register(target, slot, scale, (CpuProjectionDeltas.Delta) d);
                }
                break;
        }
        registered.add(new Registered(adapter, target, device, slot, d, shared));
    }

    /** The passes whose batch sets no per-row slots run in {@code slot} on every side of the forward pass. */
    @Override
    public void sessionSlot(int slot) {
        for (GpuDevice d : devices) {
            d.deltas().rows().session(slot);
        }
        if (cpu != null) {
            cpu.rows().session(slot);
        }
    }

    /**
     * Checks that a slot of {@code key} fits the reserved device memory of every device ({@link #reserveSlots}).
     *
     * @throws IllegalArgumentException naming the device, the bytes and the reservation when it does not
     */
    @Override
    public void checkSlot(LoraSlotKey key) {
        checkBudget(key);
    }

    private void checkBudget(LoraSlotKey key) {
        if (slotBudget == null || key.isBare()) {
            return;
        }
        long[] need = new long[devices.size()];
        for (LoraSlotKey.Part part : key.parts()) {
            LoraAdapter adapter = part.handle().adapter();
            for (String target : targetsOf(adapter)) {
                int d = placeOf(adapter, target);
                if (d >= 0) {
                    need[d] = Math.addExact(need[d], pairDeviceBytes(adapter.pair(target), slotUbatch));
                }
            }
        }
        for (int d = 0; d < need.length; d++) {
            if (need[d] > slotBudget[d]) {
                throw new IllegalArgumentException("lora: the adapters " + key.label() + " upload " + need[d]
                        + " bytes to device " + d + ", above the " + slotBudget[d] + " bytes one adapter slot"
                        + " reserves there (" + budgetSource + ")");
            }
        }
    }

    private void openSlot(int slot) {
        for (GpuDevice d : devices) {
            d.deltas().openSlot(slot);
        }
        if (cpu != null) {
            cpu.openSlot(slot);
        }
    }

    private void closeSlot(int slot) {
        for (GpuDevice d : devices) {
            d.deltas().closeSlot(slot);
        }
        if (cpu != null) {
            cpu.closeSlot(slot);
        }
    }

    /**
     * Rewrites {@code scale * B} of every registered pair of {@code adapter} to the effective scale of
     * {@code userScale}: the host arrays and, on the device path, the buffer bytes, in place
     * between steps. The registry runs this on the engine thread.
     *
     * @throws IllegalArgumentException when the adapter is not attached or the scale is not finite
     */
    public void rescale(LoraAdapter adapter, float userScale) {
        Objects.requireNonNull(adapter, "adapter");
        if (!Float.isFinite(userScale)) {
            throw new IllegalArgumentException("LoRA adapter " + adapter.path() + " cannot rescale to the"
                    + " non-finite scale " + userScale);
        }
        boolean any = false;
        for (Registered r : registered) {
            if (r.adapter != adapter || r.slot != 0) {
                continue;
            }
            any = true;
            if (r.delta instanceof GpuLoraDeltas.Rescalable) {
                ((GpuLoraDeltas.Rescalable) r.delta).rescale(userScale);
            } else if (r.delta instanceof CpuLoraDeltas.Rescalable) {
                ((CpuLoraDeltas.Rescalable) r.delta).rescale(userScale);
            }
        }
        if (!any) {
            throw new IllegalArgumentException("LoRA adapter " + adapter.path() + " has no registered pair to"
                    + " rescale");
        }
    }

    /** The attached adapters in load order. A registry event removes and appends its adapters. */
    public List<Binding> bindings() {
        return Collections.unmodifiableList(new ArrayList<>(bindings));
    }

    /**
     * Device bytes {@link #gpu} allocates for {@code bindings} with a scratch ubatch of at most {@code ubatch}: every
     * pair's {@code A} and {@code scale * B} as F32, plus one ubatch of zero slots per projection and embedding pair.
     *
     * @throws IllegalArgumentException for a non-positive {@code ubatch}
     */
    public static long gpuDeviceBytes(List<Binding> bindings, int ubatch) {
        return gpuDeviceBytes(bindings, ubatch, target -> true);
    }

    /**
     * Device bytes {@link #gpu} allocates for the pairs of {@code bindings} whose base tensor {@code targets} accepts,
     * the share of one device of a layer split; otherwise as {@link #gpuDeviceBytes(List, int)}.
     *
     * @throws IllegalArgumentException for a non-positive {@code ubatch}
     */
    public static long gpuDeviceBytes(List<Binding> bindings, int ubatch, Predicate<String> targets) {
        Objects.requireNonNull(bindings, "bindings");
        Objects.requireNonNull(targets, "targets");
        if (ubatch < 1) {
            throw new IllegalArgumentException("ubatch must be positive, got " + ubatch);
        }
        long floats = 0;
        for (Binding b : bindings) {
            for (String target : textTargets(b.adapter)) {
                if (!targets.test(target)) {
                    continue;
                }
                floats = Math.addExact(floats, pairDeviceBytes(b.adapter.pair(target), ubatch) / Float.BYTES);
            }
        }
        return Math.multiplyExact(floats, (long) Float.BYTES);
    }

    /**
     * Device bytes the device delta of {@code pair} allocates with a scratch ubatch of {@code ubatch}: its {@code A}
     * and {@code scale * B} as F32, plus one ubatch of zero slots for a projection or embedding pair.
     */
    public static long pairDeviceBytes(LoraAdapter.Pair pair, int ubatch) {
        long floats = Math.addExact((long) pair.a.length, pair.b.length);
        if (pair.kind() != LoraAdapter.Target.EXPERTS) {
            floats = Math.addExact(floats, ubatch);
        }
        return Math.multiplyExact(floats, (long) Float.BYTES);
    }

    /**
     * Device bytes of one worst-case adapter slot over the products {@code shapes} name: a pair of rank
     * {@code maxRank} on every one ({@code maxRank * (in + out)} floats per expert, plus one ubatch of zero slots
     * for a dense product or table), the footprint vLLM reserves per {@code max_loras} slot at
     * {@code max_lora_rank}.
     *
     * @throws IllegalArgumentException for a non-positive rank or ubatch, or a shape without positive dimensions
     */
    public static long slotReserveBytes(List<long[]> shapes, int maxRank, int ubatch) {
        Objects.requireNonNull(shapes, "shapes");
        if (maxRank < 1 || ubatch < 1) {
            throw new IllegalArgumentException("the slot reservation needs a positive rank and ubatch, got rank "
                    + maxRank + ", ubatch " + ubatch);
        }
        long floats = 0;
        for (long[] s : shapes) {
            if (s.length != 3 || s[0] < 1 || s[1] < 1 || s[2] < 1) {
                throw new IllegalArgumentException("a reserved product needs {in, out, experts} > 0, got "
                        + Arrays.toString(s));
            }
            long pair = Math.multiplyExact(Math.multiplyExact((long) maxRank, s[0] + s[1]), s[2]);
            floats = Math.addExact(floats, s[2] > 1 ? pair : Math.addExact(pair, ubatch));
        }
        return Math.multiplyExact(floats, (long) Float.BYTES);
    }

    /**
     * The {@code {in, out, experts}} of every product of {@code catalog} an adapter can target where
     * {@code placed} accepts its tensor name: each {@code *.weight} matrix or 3-D expert tensor. A superset of the
     * products the bound blocks serve, so a reservation over it ({@link #slotReserveBytes}) covers every adapter a
     * slot can run.
     */
    public static List<long[]> adaptableShapes(TensorCatalog catalog, Predicate<String> placed) {
        Objects.requireNonNull(catalog, "catalog");
        Objects.requireNonNull(placed, "placed");
        List<long[]> out = new ArrayList<>();
        for (String name : new TreeSet<>(catalog.names())) {
            TensorRef t = catalog.get(name);
            if (t == null || !name.endsWith(".weight") || (t.rank() != 2 && t.rank() != 3) || !placed.test(name)) {
                continue;
            }
            out.add(new long[]{t.ne(0), t.ne(1), t.rank() == 3 ? t.ne(2) : 1L});
        }
        return out;
    }

    /** Number of registered deltas (pairs times adapters). */
    public int deltaCount() {
        return registered.size();
    }

    // ------------------------------------------------------------------ attach and detach

    /**
     * Validates and uploads/registers one adapter's every pair at {@code userScale} in adapter slot {@code slot}.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    private void attachAdapter(LoraAdapter adapter, float userScale, int slot) {
        checkAdapter(adapter);
        for (String target : targetsOf(adapter)) {
            attach(adapter, target, userScale, slot);
        }
    }

    /**
     * The targets of {@code adapter} this attachment serves: its {@link LoraAdapter.Side#TEXT} pairs on a text
     * forward pass, its {@link LoraAdapter.Side#VISION} pairs on a vision encoder.
     */
    private List<String> targetsOf(LoraAdapter adapter) {
        return side == LoraAdapter.Side.VISION ? visionTargets(adapter) : textTargets(adapter);
    }

    /** True when one of {@code bindings} adapts a pair of {@code side}. */
    public static boolean adapts(List<Binding> bindings, LoraAdapter.Side side) {
        Objects.requireNonNull(bindings, "bindings");
        Objects.requireNonNull(side, "side");
        for (Binding b : bindings) {
            for (String target : b.adapter().targets()) {
                if (b.adapter().pair(target).side() == side) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * The targets of {@code adapter} this attachment serves: its text-model pairs. A vision-encoder pair is served
     * by the vision encoder's own attachment ({@link #visionTargets}).
     */
    private static List<String> textTargets(LoraAdapter adapter) {
        List<String> out = new ArrayList<>();
        for (String target : adapter.targets()) {
            if (adapter.pair(target).side() == LoraAdapter.Side.TEXT) {
                out.add(target);
            }
        }
        return out;
    }

    /** The targets of {@code adapter} the vision encoder serves: its {@link LoraAdapter.Side#VISION} pairs. */
    public static List<String> visionTargets(LoraAdapter adapter) {
        Objects.requireNonNull(adapter, "adapter");
        List<String> out = new ArrayList<>();
        for (String target : adapter.targets()) {
            if (adapter.pair(target).side() == LoraAdapter.Side.VISION) {
                out.add(target);
            }
        }
        return out;
    }

    /** Uploads/registers one validated pair on the side that serves it. */
    private void attach(LoraAdapter adapter, String target, float userScale, int slot) {
        int d = placeOf(adapter, target);
        if (d >= 0) {
            attachGpu(adapter, target, d, userScale, slot);
        } else {
            attachCpu(adapter, target, userScale, slot);
        }
    }

    /**
     * Validates every pair of {@code adapter} against the bound blocks, registering nothing.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    private void checkAdapter(LoraAdapter adapter) {
        for (String target : targetsOf(adapter)) {
            if (placeOf(adapter, target) == HOST) {
                checkCpu(adapter, target);
            } else {
                checkGpu(adapter, target);
            }
        }
    }

    /** {@link #placeOf}: the pair goes to the host deltas. */
    private static final int HOST = -1;

    /**
     * Where the pair of {@code target} registers: the index of the device whose bound blocks serve it, {@link #HOST}
     * when the host deltas serve it (a host forward pass, or a hybrid's host layers), else -2 (no block serves it,
     * which {@link #checkGpu} then refuses by name).
     *
     * @throws IllegalStateException when blocks of two sides serve it
     */
    private int placeOf(LoraAdapter adapter, String target) {
        int device = devices.isEmpty() ? -2 : servingDevice(adapter, target);
        boolean host = cpu != null && cpu.served(target) != null;
        if (device >= 0 && host) {
            throw new IllegalStateException("LoRA adapter " + adapter.path() + ", tensor " + target
                    + ": blocks of device " + device + " and of the host both serve it");
        }
        if (device >= 0) {
            return device;
        }
        return host || devices.isEmpty() ? HOST : -2;
    }

    /** Detaches everything registered in slot 0 for {@code adapter}: unregisters its deltas and releases the uploads. */
    private void detach(LoraAdapter adapter) {
        detachWhere(r -> r.adapter == adapter && r.slot == 0);
    }

    /** Detaches everything registered in adapter slot {@code slot}. */
    private void detachSlot(int slot) {
        detachWhere(r -> r.slot == slot);
    }

    private void detachWhere(Predicate<Registered> match) {
        for (int i = registered.size() - 1; i >= 0; i--) {
            Registered r = registered.get(i);
            if (!match.test(r)) {
                continue;
            }
            registered.remove(i);
            unregister(r);
            if (r.shared) {
                continue; // a rescale registration borrows slot 0's delta: slot 0 owns and closes it
            }
            if (r.delta instanceof AutoCloseable) {
                owned.remove(r.delta);
            }
            close(r.delta);
        }
    }

    private void attachGpu(LoraAdapter adapter, String target, int device, float userScale, int slot) {
        LoraAdapter.Pair pair = adapter.pair(target);
        Object d = GpuLoraDeltas.of(devices.get(device).ops(), adapter.delta(target, userScale));
        try {
            register(adapter, target, pair, device, slot, d);
        } catch (IllegalStateException | UnsupportedOperationException e) {
            close(d);
            throw rethrow(adapter, target, e);
        }
    }

    private void attachCpu(LoraAdapter adapter, String target, float userScale, int slot) {
        LoraAdapter.Pair pair = adapter.pair(target);
        Object d = CpuLoraDeltas.of(adapter.delta(target, userScale));
        try {
            register(adapter, target, pair, -1, slot, d);
        } catch (IllegalStateException | UnsupportedOperationException e) {
            throw rethrow(adapter, target, e);
        }
    }

    private void register(LoraAdapter adapter, String target, LoraAdapter.Pair pair, int device, int slot,
                          Object d) {
        register(adapter, target, pair, device, slot, 1.0f, d);
    }

    /** Registers one validated pair's delta in slot {@code slot} at {@code scale} on the side that serves it. */
    private void register(LoraAdapter adapter, String target, LoraAdapter.Pair pair, int device, int slot,
                          float scale, Object d) {
        GpuProjectionDeltas gpu = device >= 0 ? devices.get(device).deltas() : null;
        switch (pair.kind()) {
            case EXPERTS:
                if (gpu != null) {
                    gpu.registerExperts(target, slot, scale, (GpuProjectionDeltas.ExpertDelta) d);
                } else {
                    cpu.registerExperts(target, slot, scale, (CpuProjectionDeltas.ExpertDelta) d);
                }
                break;
            case EMBEDDING:
                if (gpu != null) {
                    gpu.registerEmbedding(target, slot, scale, (GpuProjectionDeltas.EmbeddingDelta) d);
                } else {
                    cpu.registerEmbedding(target, slot, scale, (CpuProjectionDeltas.EmbeddingDelta) d);
                }
                break;
            default:
                if (gpu != null) {
                    gpu.register(target, slot, scale, (GpuProjectionDeltas.Delta) d);
                } else {
                    cpu.register(target, slot, scale, (CpuProjectionDeltas.Delta) d);
                }
                break;
        }
        registered.add(new Registered(adapter, target, device, slot, d, false));
        if (d instanceof AutoCloseable) {
            owned.add((AutoCloseable) d);
        }
    }

    private static void close(Object delta) {
        if (!(delta instanceof AutoCloseable)) {
            return;
        }
        try {
            ((AutoCloseable) delta).close();
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("releasing a LoRA delta failed", e);
        }
    }

    private void checkIn(Binding b, LoraAdapter.Pair pair, long[] shape, String refusal, String kind) {
        String where = "LoRA adapter " + b.adapter.path() + ", tensor " + pair.target() + ": ";
        if (shape == null) {
            throw new IllegalStateException(where + "no bound block applies deltas to this tensor; the model's"
                    + " forward pass does not run it as a LoRA-adaptable product");
        }
        if (refusal != null) {
            throw new UnsupportedOperationException(where + refusal);
        }
        String want = pair.kind() == LoraAdapter.Target.EXPERTS ? GpuProjectionDeltas.Kind.EXPERTS.name()
                : pair.kind() == LoraAdapter.Target.EMBEDDING ? GpuProjectionDeltas.Kind.EMBEDDING.name()
                : GpuProjectionDeltas.Kind.PROJECTION.name();
        if (!want.equals(kind)) {
            throw new IllegalStateException(where + "the forward pass runs this tensor as a " + kind
                    + " product, the adapter pair is a " + want + " pair");
        }
        if (shape[0] >= 0 && (shape[0] != pair.inDim() || shape[1] != pair.outDim()
                || (pair.kind() == LoraAdapter.Target.EXPERTS && shape[2] != pair.experts()))) {
            throw new IllegalStateException(where + "the forward pass runs it as {" + shape[0] + " -> " + shape[1]
                    + ", " + shape[2] + " experts}, the adapter pair is " + pair);
        }
    }

    /** Validates one pair against the device that serves its base tensor. */
    private void checkGpu(LoraAdapter adapter, String target) {
        int device = servingDevice(adapter, target);
        GpuProjectionDeltas.Served<?> s = device < 0 ? null : devices.get(device).deltas().served(target);
        checkIn(new Binding(adapter, 0.0f), adapter.pair(target),
                s == null ? null : new long[]{s.inDim(), s.outDim(), s.experts()},
                s == null ? null : s.refusal(), s == null ? null : s.kind().name());
    }

    /**
     * The device whose bound blocks serve {@code target}, or -1 when none does.
     *
     * @throws IllegalStateException when blocks of two devices serve it
     */
    private int servingDevice(LoraAdapter adapter, String target) {
        int found = -1;
        for (int d = 0; d < devices.size(); d++) {
            if (devices.get(d).deltas().served(target) == null) {
                continue;
            }
            if (found >= 0) {
                throw new IllegalStateException("LoRA adapter " + adapter.path() + ", tensor " + target
                        + ": blocks of devices " + found + " and " + d + " both serve it");
            }
            found = d;
        }
        return found;
    }

    private void unregister(Registered r) {
        if (r.device >= 0) {
            devices.get(r.device).deltas().unregister(r.name, r.delta);
        } else {
            cpu.unregister(r.name, r.delta);
        }
    }

    private void checkCpu(LoraAdapter adapter, String target) {
        CpuProjectionDeltas.Served<?> s = cpu.served(target);
        checkIn(new Binding(adapter, 0.0f), adapter.pair(target),
                s == null ? null : new long[]{s.inDim(), s.outDim(), s.experts()},
                s == null ? null : s.refusal(), s == null ? null : s.kind().name());
    }

    private static RuntimeException rethrow(LoraAdapter adapter, String target, RuntimeException e) {
        String msg = "LoRA adapter " + adapter.path() + ", tensor " + target + ": " + e.getMessage();
        if (e instanceof UnsupportedOperationException) {
            return new UnsupportedOperationException(msg, e);
        }
        return new IllegalStateException(msg, e);
    }

    private void closeAfterFailure(RuntimeException e) {
        try {
            close();
        } catch (RuntimeException suppressed) {
            e.addSuppressed(suppressed);
        }
    }

    /**
     * Unregisters every delta and releases the device uploads, and releases the handles it holds; a session also
     * closes its slot and returns the rows of the pass to slot 0. Idempotent.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (int i = registered.size() - 1; i >= 0; i--) {
            unregister(registered.get(i));
        }
        registered.clear();
        for (LoraHandle h : held.values()) {
            h.release();
        }
        held.clear();
        bindings.clear();
        RuntimeException first = null;
        for (AutoCloseable c : owned) {
            try {
                c.close();
            } catch (Exception e) {
                RuntimeException re = e instanceof RuntimeException ? (RuntimeException) e
                        : new IllegalStateException("releasing a LoRA delta failed", e);
                if (first == null) {
                    first = re;
                } else {
                    first.addSuppressed(re);
                }
            }
        }
        owned.clear();
        if (session) {
            try {
                closeSlot(sessionSlot);
            } catch (RuntimeException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
            sessionSlot(0);
        }
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "LoraAttachment{" + bindings + ", " + registered.size() + " deltas" + (closed ? ", closed" : "") + '}';
    }
}
