package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.lora.LoraHandle;
import com.eonmux.neuralfree.gguf.engine.lora.LoraListener;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

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
 * the absorbed MLA {@code attn_kv_b}); nothing is registered then. {@link #follow} connects the attachment to a
 * {@link LoraRegistry}: every runtime change (LR-1 J3, llama.cpp {@code /lora-adapters}, vLLM
 * {@code /v1/load_lora_adapter}) is applied synchronously on the engine thread the registry runs on — a loaded
 * adapter becomes attached, an unloaded or evicted one is detached with its device uploads released (the queues that
 * ran them must be finished, as on {@link #close}), and a scale change rescales its {@code scale * B} in place
 * between steps. A hot change fails the registry call that caused it and leaves nothing registered.</p>
 *
 * <p>Mutable, owned by the engine thread: it belongs to the forward pass it attaches its deltas to, and the
 * registry events it subscribes to are delivered on that same thread (§3.3). {@link #follow} and the events balance
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

    /** One registered delta, its adapter and how its removal and rescale go through. */
    private static final class Registered {
        final LoraAdapter adapter;
        final String name;
        final Object delta;

        Registered(LoraAdapter adapter, String name, Object delta) {
            this.adapter = adapter;
            this.name = name;
            this.delta = delta;
        }
    }

    private final List<Binding> bindings = new ArrayList<>();
    /** The registry handles this attachment holds, to release them on the events of {@link #follow}. */
    private final Map<LoraAdapter, LoraHandle> held = new HashMap<>();
    private final GpuProjectionDeltas gpu;
    private final CpuProjectionDeltas cpu;
    private final GpuOps ops;
    private final List<Registered> registered = new ArrayList<>();
    private final List<AutoCloseable> owned = new ArrayList<>();
    private boolean closed;

    private LoraAttachment(List<Binding> bindings, GpuProjectionDeltas gpu, CpuProjectionDeltas cpu, GpuOps ops) {
        this.bindings.addAll(bindings);
        this.gpu = gpu;
        this.cpu = cpu;
        this.ops = ops;
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
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(deltas, "deltas");
        LoraAttachment att = new LoraAttachment(Objects.requireNonNull(bindings, "bindings"), deltas, null, ops);
        try {
            for (Binding b : att.bindings) {
                for (String target : b.adapter.targets()) {
                    GpuProjectionDeltas.Served<?> site = att.gpu.served(target);
                    att.checkIn(b, b.adapter.pair(target),
                            site == null ? null : new long[]{site.inDim(), site.outDim(), site.experts()},
                            site == null ? null : site.refusal(), site == null ? null : site.kind().name());
                }
                for (String target : b.adapter.targets()) {
                    att.attachGpu(b.adapter, target, b.scale());
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
        Objects.requireNonNull(deltas, "deltas");
        LoraAttachment att = new LoraAttachment(Objects.requireNonNull(bindings, "bindings"), null, deltas, null);
        try {
            for (Binding b : att.bindings) {
                for (String target : b.adapter.targets()) {
                    att.checkCpu(b.adapter, target);
                }
                for (String target : b.adapter.targets()) {
                    att.attachCpu(b.adapter, target, b.scale());
                }
            }
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
        Objects.requireNonNull(registry, "registry");
        registry.addListener(this);
    }

    // ------------------------------------------------------------------ the registry events

    /** {@link LoraRegistry} loaded (LR-1 {@code load}) one adapter: it becomes attached, or the load fails. */
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
        try {
            attachAdapter(handle.adapter(), handle.scale());
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
     * requests, LR-2).
     */
    @Override
    public void scaleChanged(LoraHandle handle, float previous) {
        Objects.requireNonNull(handle, "handle");
        if (held.containsKey(handle.adapter())) {
            rescale(handle.adapter(), handle.scale());
        }
    }

    /**
     * Rewrites {@code scale * B} of every registered pair of {@code adapter} to the effective scale of
     * {@code userScale} (LR-1 {@code rescale}): the host arrays and, on the device path, the buffer bytes, in place
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
            if (r.adapter != adapter) {
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
        Objects.requireNonNull(bindings, "bindings");
        if (ubatch < 1) {
            throw new IllegalArgumentException("ubatch must be positive, got " + ubatch);
        }
        long floats = 0;
        for (Binding b : bindings) {
            for (String target : b.adapter.targets()) {
                LoraAdapter.Pair pair = b.adapter.pair(target);
                floats = Math.addExact(floats, (long) pair.a.length + pair.b.length);
                if (pair.kind() != LoraAdapter.Target.EXPERTS) {
                    floats = Math.addExact(floats, ubatch);
                }
            }
        }
        return Math.multiplyExact(floats, (long) Float.BYTES);
    }

    /** Number of registered deltas (pairs times adapters). */
    public int deltaCount() {
        return registered.size();
    }

    // ------------------------------------------------------------------ attach and detach

    /**
     * Validates and uploads/registers one adapter's every pair at {@code userScale}.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    private void attachAdapter(LoraAdapter adapter, float userScale) {
        for (String target : adapter.targets()) {
            if (gpu != null) {
                GpuProjectionDeltas.Served<?> site = gpu.served(target);
                checkAdapter(adapter, adapter.pair(target),
                        site == null ? null : new long[]{site.inDim(), site.outDim(), site.experts()},
                        site == null ? null : site.refusal(),
                        site == null ? null : site.kind().name());
            } else {
                CpuProjectionDeltas.Served<?> site = cpu.served(target);
                checkAdapter(adapter, adapter.pair(target),
                        site == null ? null : new long[]{site.inDim(), site.outDim(), site.experts()},
                        site == null ? null : site.refusal(),
                        site == null ? null : site.kind().name());
            }
        }
        for (String target : adapter.targets()) {
            if (gpu != null) {
                attachGpu(adapter, target, userScale);
            } else {
                attachCpu(adapter, target, userScale);
            }
        }
    }

    /** Detaches everything registered for {@code adapter}: unregisters its deltas and releases the uploads. */
    private void detach(LoraAdapter adapter) {
        for (int i = registered.size() - 1; i >= 0; i--) {
            Registered r = registered.get(i);
            if (r.adapter != adapter) {
                continue;
            }
            registered.remove(i);
            if (gpu != null) {
                gpu.unregister(r.name, r.delta);
            } else {
                cpu.unregister(r.name, r.delta);
            }
            close(r.delta);
        }
    }

    private void attachGpu(LoraAdapter adapter, String target, float userScale) {
        LoraAdapter.Pair pair = adapter.pair(target);
        Object d = GpuLoraDeltas.of(ops, adapter.delta(target, userScale));
        try {
            register(adapter, target, pair, d);
        } catch (IllegalStateException | UnsupportedOperationException e) {
            throw rethrow(adapter, target, e);
        }
    }

    private void attachCpu(LoraAdapter adapter, String target, float userScale) {
        LoraAdapter.Pair pair = adapter.pair(target);
        Object d = CpuLoraDeltas.of(adapter.delta(target, userScale));
        try {
            register(adapter, target, pair, d);
        } catch (IllegalStateException | UnsupportedOperationException e) {
            throw rethrow(adapter, target, e);
        }
    }

    private void register(LoraAdapter adapter, String target, LoraAdapter.Pair pair, Object d) {
        switch (pair.kind()) {
            case EXPERTS:
                if (gpu != null) {
                    gpu.registerExperts(target, (GpuProjectionDeltas.ExpertDelta) d);
                } else {
                    cpu.registerExperts(target, (CpuProjectionDeltas.ExpertDelta) d);
                }
                break;
            case EMBEDDING:
                if (gpu != null) {
                    gpu.registerEmbedding(target, (GpuProjectionDeltas.EmbeddingDelta) d);
                } else {
                    cpu.registerEmbedding(target, (CpuProjectionDeltas.EmbeddingDelta) d);
                }
                break;
            default:
                if (gpu != null) {
                    gpu.register(target, (GpuProjectionDeltas.Delta) d);
                } else {
                    cpu.register(target, (CpuProjectionDeltas.Delta) d);
                }
                break;
        }
        registered.add(new Registered(adapter, target, d));
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

    private void checkAdapter(LoraAdapter adapter, LoraAdapter.Pair pair, long[] shape, String refusal, String kind) {
        checkIn(new Binding(adapter, 0.0f), pair, shape, refusal, kind);
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

    /** Unregisters every delta and releases the device uploads, and releases the handles it holds. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (int i = registered.size() - 1; i >= 0; i--) {
            Registered r = registered.get(i);
            if (gpu != null) {
                gpu.unregister(r.name, r.delta);
            } else {
                cpu.unregister(r.name, r.delta);
            }
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
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "LoraAttachment{" + bindings + ", " + registered.size() + " deltas" + (closed ? ", closed" : "") + '}';
    }
}
