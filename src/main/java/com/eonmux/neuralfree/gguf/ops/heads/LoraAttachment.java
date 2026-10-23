package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
 * the absorbed MLA {@code attn_kv_b}); nothing is registered then. {@link #close} unregisters every delta and releases
 * the device uploads; the queues that ran them must be finished. Not thread-safe.</p>
 */
public final class LoraAttachment implements AutoCloseable {

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
         * @throws IOException when the file cannot be read
         */
        public static Binding load(String path, float scale, CpuWeights base) throws IOException {
            return new Binding(LoraAdapterLoader.load(path, base), scale);
        }

        public LoraAdapter adapter() { return adapter; }
        public float scale() { return scale; }

        @Override
        public String toString() {
            return adapter.path() + ":" + scale;
        }
    }

    /** One registered delta and how to remove it. */
    private static final class Registered {
        final String name;
        final Object delta;

        Registered(String name, Object delta) {
            this.name = name;
            this.delta = delta;
        }
    }

    private final List<Binding> bindings;
    private final GpuProjectionDeltas gpu;
    private final CpuProjectionDeltas cpu;
    private final List<Registered> registered = new ArrayList<>();
    private final List<AutoCloseable> owned = new ArrayList<>();
    private boolean closed;

    private LoraAttachment(List<Binding> bindings, GpuProjectionDeltas gpu, CpuProjectionDeltas cpu) {
        this.bindings = Collections.unmodifiableList(new ArrayList<>(bindings));
        this.gpu = gpu;
        this.cpu = cpu;
    }

    /**
     * Uploads and registers every pair of {@code bindings} on the device forward pass whose blocks share
     * {@code deltas}, dispatching on the queue of {@code ops}. Every pair is validated before the first upload.
     *
     * @throws IllegalStateException         when a pair targets a product no bound block applies deltas to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment gpu(List<Binding> bindings, GpuOps ops, GpuProjectionDeltas deltas) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(deltas, "deltas");
        LoraAttachment att = new LoraAttachment(Objects.requireNonNull(bindings, "bindings"), deltas, null);
        for (Binding b : att.bindings) {
            for (String target : b.adapter.targets()) {
                att.checkGpu(b, target);
            }
        }
        try {
            for (Binding b : att.bindings) {
                for (String target : b.adapter.targets()) {
                    att.attachGpu(b, target, ops);
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
     * @throws IllegalStateException         when a pair targets a product no bound block applies deltas to, or the
     *                                       served shape differs from the pair's
     * @throws UnsupportedOperationException when the serving block refuses deltas
     */
    public static LoraAttachment cpu(List<Binding> bindings, CpuProjectionDeltas deltas) {
        Objects.requireNonNull(deltas, "deltas");
        LoraAttachment att = new LoraAttachment(Objects.requireNonNull(bindings, "bindings"), null, deltas);
        for (Binding b : att.bindings) {
            for (String target : b.adapter.targets()) {
                att.checkCpu(b, target);
            }
        }
        try {
            for (Binding b : att.bindings) {
                for (String target : b.adapter.targets()) {
                    att.attachCpu(b, target);
                }
            }
            return att;
        } catch (RuntimeException e) {
            att.closeAfterFailure(e);
            throw e;
        }
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

    /** The attached adapters in load order. */
    public List<Binding> bindings() {
        return bindings;
    }

    /** Number of registered deltas (pairs times adapters). */
    public int deltaCount() {
        return registered.size();
    }

    /** Validates one pair against the device site serving its tensor. */
    private void checkGpu(Binding b, String target) {
        GpuProjectionDeltas.Served<?> site = gpu.served(target);
        check(b, b.adapter.pair(target), site == null ? null : new long[]{site.inDim(), site.outDim(), site.experts()},
                site == null ? null : site.refusal(), site == null ? null : site.kind().name());
    }

    /** Validates one pair against the host site serving its tensor. */
    private void checkCpu(Binding b, String target) {
        CpuProjectionDeltas.Served<?> site = cpu.served(target);
        check(b, b.adapter.pair(target), site == null ? null : new long[]{site.inDim(), site.outDim(), site.experts()},
                site == null ? null : site.refusal(), site == null ? null : site.kind().name());
    }

    /** Uploads and registers one pair that {@link #checkGpu} accepted. */
    private void attachGpu(Binding b, String target, GpuOps ops) {
        LoraAdapter.Pair pair = b.adapter.pair(target);
        AutoCloseable d = GpuLoraDeltas.of(ops, b.adapter.delta(target, b.scale));
        owned.add(d);
        try {
            switch (pair.kind()) {
                case EXPERTS:
                    gpu.registerExperts(target, (GpuProjectionDeltas.ExpertDelta) d);
                    break;
                case EMBEDDING:
                    gpu.registerEmbedding(target, (GpuProjectionDeltas.EmbeddingDelta) d);
                    break;
                default:
                    gpu.register(target, (GpuProjectionDeltas.Delta) d);
                    break;
            }
        } catch (IllegalStateException | UnsupportedOperationException e) {
            throw rethrow(b, target, e);
        }
        registered.add(new Registered(target, d));
    }

    /** Registers one pair that {@link #checkCpu} accepted. */
    private void attachCpu(Binding b, String target) {
        LoraAdapter.Pair pair = b.adapter.pair(target);
        Object d = CpuLoraDeltas.of(b.adapter.delta(target, b.scale));
        try {
            switch (pair.kind()) {
                case EXPERTS:
                    cpu.registerExperts(target, (CpuProjectionDeltas.ExpertDelta) d);
                    break;
                case EMBEDDING:
                    cpu.registerEmbedding(target, (CpuProjectionDeltas.EmbeddingDelta) d);
                    break;
                default:
                    cpu.register(target, (CpuProjectionDeltas.Delta) d);
                    break;
            }
        } catch (IllegalStateException | UnsupportedOperationException e) {
            throw rethrow(b, target, e);
        }
        registered.add(new Registered(target, d));
    }

    /**
     * Fails when the target is unserved, refused, of another kind, or served with another shape. {@code shape} is
     * {@code {in, out, experts}} of the site (-1 entries when the block gave no shape).
     */
    private static void check(Binding b, LoraAdapter.Pair pair, long[] shape, String refusal, String kind) {
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

    private static RuntimeException rethrow(Binding b, String target, RuntimeException e) {
        String msg = "LoRA adapter " + b.adapter.path() + ", tensor " + target + ": " + e.getMessage();
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

    /** Unregisters every delta and releases the device uploads. Idempotent. */
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
