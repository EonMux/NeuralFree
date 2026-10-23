package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * GPU layer extra {@code LayerSpec.Extras.SWITCHED_LORA}: the graniteswitch per-token LoRA delta of every layer
 * projection (models/granite-switch.cpp:242-427).
 *
 * <p>Every projection {@link Projection} of a graniteswitch layer adds {@code B[s_t] (A[s_t] x_t)} to its base
 * output, with the stacked {@code <proj>.lora_a {in, rank, n_slots}} and {@code <proj>.lora_b {rank, out, n_slots}}
 * of the layer and the token's adapter slot {@code s_t} (slot 0 is the zero delta). The inputs are those of the base
 * projections: {@code attn_norm(x)} for Q/K/V (:346-348), the attention output before {@code wo} for O (:376), the
 * FFN norm output for gate/up (:397-398) and {@code silu(gate) * up} for down (:401). The slots come from the
 * model's {@link Router}.</p>
 *
 * <p>{@link #bind} resolves the 14 tensors, keeps F32 device copies of those stored in another type (buffers owned by
 * this extra, filled once by an embedding lookup of every row and released by {@link #close}), prepares the two
 * {@value HeadsKernels#SWITCHED_LORA_ADD} stages per projection and registers {@link #addDelta} of each projection in
 * the context's {@link GpuProjectionDeltas} under {@code <base>.weight}, so the layer's attention mixer and dense FFN
 * add it right after their base products (a projection no bound block serves fails the bind).
 * {@link #beforeBlock} uploads the slots of the step's rows; {@link #addDelta} enqueues stage 0 then stage 1 on the
 * context queue. {@link #close} unregisters the deltas and expects the queues that used the extra to be finished.
 * Not thread-safe.</p>
 */
public final class SwitchedLoraExtra implements GpuLayerExtra {

    /** The adapted projections of a graniteswitch layer and their tensor base names. */
    public enum Projection {
        Q("attn_q"), K("attn_k"), V("attn_v"), O("attn_output"), GATE("ffn_gate"), UP("ffn_up"), DOWN("ffn_down");

        private final String base;

        Projection(String base) {
            this.base = base;
        }

        /** Tensor base name, as in {@code blk.N.<base>.lora_a}. */
        public String base() {
            return base;
        }

        /** Suffix of the stacked {@code A} tensor. */
        public String loraA() {
            return base + LoraAdapterLoader.SUFFIX_A;
        }

        /** Suffix of the stacked {@code B} tensor. */
        public String loraB() {
            return base + LoraAdapterLoader.SUFFIX_B;
        }
    }

    static final String TEMP_SLOTS = "switched_lora.slots";

    private final Router router;
    private final SlotUpload upload;
    private final Map<Projection, GpuDelta> deltas = new EnumMap<>(Projection.class);
    /** The projection deltas registered in {@link #registry}, removed by {@link #close}. */
    private final Map<Projection, GpuProjectionDeltas.Delta> registered = new EnumMap<>(Projection.class);
    private GpuProjectionDeltas registry;
    /** F32 device copies of the stacked tensors stored in another type. */
    private final List<GpuBuffer> owned = new ArrayList<>();
    private GpuOps ops;
    private LayerSpec layer;
    private ScratchSlot slots;
    private int ubatch;

    SwitchedLoraExtra(Router router, SlotUpload upload) {
        this.router = Objects.requireNonNull(router, "router");
        this.upload = Objects.requireNonNull(upload, "upload");
    }

    /** The factory of {@code LayerSpec.Extras.SWITCHED_LORA}; {@code router} null makes creation fail. */
    static GpuLayerExtraFactory factory(Router router) {
        SlotUpload upload = new SlotUpload();
        return new GpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return LayerSpec.Extras.SWITCHED_LORA;
            }

            @Override
            public GpuLayerExtra create(LayerSpec layer) {
                if (router == null) {
                    throw new UnsupportedOperationException("switched LoRA layer " + layer.il() + " needs the model's"
                            + " SwitchedLoraExtra.Router; build the extras with HeadsBlocks.gpuExtras(router)");
                }
                return new SwitchedLoraExtra(router, upload);
            }
        };
    }

    /** The model's adapter router shared by the layers. */
    public Router router() {
        return router;
    }

    /**
     * Binds the extra to a layer; a second bind releases the resources of the first.
     *
     * @throws IllegalArgumentException when a stacked tensor does not have the router's slot count
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        close();
        this.layer = layer;
        this.ops = ctx.ops();
        this.ubatch = ctx.scratch().ubatch();
        int nSlots = router.nSlots();
        try {
            for (Projection p : Projection.values()) {
                GpuWeight a = ctx.weights().layer(layer, p.loraA());
                GpuWeight b = ctx.weights().layer(layer, p.loraB());
                long[] an = a.ne();
                long[] bn = b.ne();
                if (an.length != 3 || bn.length != 3 || an[1] != bn[0] || an[2] != nSlots || bn[2] != nSlots) {
                    throw new IllegalArgumentException("layer " + layer.il() + " " + p.base() + ": lora_a "
                            + Arrays.toString(an) + " and lora_b " + Arrays.toString(bn) + " are not stacked {in,"
                            + " rank, " + nSlots + "} / {rank, out, " + nSlots + "}");
                }
                int in = HeadsBlocks.toInt(an[0], a.name());
                int rank = HeadsBlocks.toInt(an[1], a.name());
                int out = HeadsBlocks.toInt(bn[1], b.name());
                HeadsKernels.checkLoraShape(in, out, rank, nSlots);
                deltas.put(p, new GpuDelta(ops, f32(ops, a, owned), f32(ops, b, owned), in, out, rank, nSlots));
            }
            this.slots = ctx.scratch().temp(TEMP_SLOTS, ubatch);
            upload.reserve(ubatch);
            this.registry = ctx.projectionDeltas();
            for (Projection p : Projection.values()) {
                GpuProjectionDeltas.Delta d = (n, x, xOff, xStride, y, yOff, yStride) ->
                        addDelta(p, n, x, xOff, xStride, y, yOff, yStride);
                registry.register(layer, TensorNames.weight(p.base()), d);
                registered.put(p, d);
            }
        } catch (RuntimeException e) {
            close();
            this.ops = null;
            throw e;
        }
    }

    /**
     * An F32 single-buffer copy of {@code w}: the tensor's own buffer when it already is one, otherwise a new buffer
     * filled by an embedding lookup of every row and added to {@code owned}.
     */
    static GpuBuffer f32(GpuOps ops, GpuWeight w, List<GpuBuffer> owned) {
        if (w.type() == GGMLType.F32 && w.isSingleShard() && !w.hasScale()) {
            return w.shard(0);
        }
        int rows = HeadsBlocks.toInt(w.rows(), w.name());
        GpuBuffer copy = allocateFloats(ops, w.name() + " (F32 copy)", w.rows() * w.cols(), false);
        owned.add(copy);
        int[] ids = new int[rows];
        for (int i = 0; i < rows; i++) {
            ids[i] = i;
        }
        ops.embed(w, ids, copy, 0);
        if (w.hasScale()) {
            ops.scale(copy, 0, HeadsBlocks.toInt(w.rows() * w.cols(), w.name()), w.scale());
        }
        return copy;
    }

    /**
     * A device buffer of {@code floats} floats for the caller to own and close.
     *
     * @throws GpuOutOfMemoryException when the buffer exceeds the device's allocation or binding limit
     */
    static GpuBuffer allocateFloats(GpuOps ops, String what, long floats, boolean readOnly) {
        if (floats <= 0) {
            throw new IllegalArgumentException(what + " needs a positive size, got " + floats + " floats");
        }
        ComputeBackend backend = ops.kernels().backend();
        long bytes = Math.multiplyExact(floats, (long) Float.BYTES);
        DeviceCapabilities caps = backend.capabilities();
        long limit = Long.MAX_VALUE;
        if (caps.getMaxAllocSizeBytes() > 0) {
            limit = caps.getMaxAllocSizeBytes();
        }
        if (caps.getMaxStorageBufferRange() > 0) {
            limit = Math.min(limit, caps.getMaxStorageBufferRange());
        }
        if (bytes > limit) {
            throw new GpuOutOfMemoryException(what + " needs " + bytes + " bytes, above the device limit of " + limit
                    + " bytes per buffer", bytes, limit, String.valueOf(backend.backendType()));
        }
        return backend.allocateBuffer(bytes, readOnly);
    }

    /** Closes every buffer of {@code buffers} and empties the list; the first failure is rethrown. */
    static void closeAll(List<GpuBuffer> buffers) {
        RuntimeException first = Closeables.closeAll(buffers, null);
        buffers.clear();
        if (first != null) {
            throw first;
        }
    }

    /**
     * Uploads the adapter slots of rows {@code [pos0, pos0 + nTokens)} for this step; the layers share one device
     * copy, written by the first layer of the step.
     */
    @Override
    public void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
        ensureBound();
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("switched LoRA step of " + nTokens + " rows, ubatch is " + ubatch);
        }
        upload.write(ops, router, pos0, nTokens, slots);
    }

    /**
     * {@code y[t] += B[s_t] (A[s_t] x[t])} for the {@code nTokens} rows of the current step after projection
     * {@code p}'s base product: row {@code t} of {@code x} at {@code xOff + t*xStride}, of {@code y} at
     * {@code yOff + t*yStride} (floats).
     */
    public void addDelta(Projection p, int nTokens, GpuBuffer x, int xOff, int xStride, GpuBuffer y, int yOff,
                         int yStride) {
        ensureBound();
        deltas.get(Objects.requireNonNull(p, "projection")).add(nTokens, slots.buffer(), x, xOff, xStride, y, yOff,
                yStride);
    }

    private void ensureBound() {
        if (ops == null) {
            throw new IllegalStateException("switched LoRA extra is not bound");
        }
    }

    /** Unregisters the projection deltas and releases the prepared dispatches and the owned F32 copies. Idempotent. */
    @Override
    public void close() {
        if (registry != null) {
            for (Map.Entry<Projection, GpuProjectionDeltas.Delta> e : registered.entrySet()) {
                registry.unregister(layer, TensorNames.weight(e.getKey().base()), e.getValue());
            }
            registered.clear();
            registry = null;
        }
        for (GpuDelta d : deltas.values()) {
            d.close();
        }
        deltas.clear();
        closeAll(owned);
    }

    @Override
    public String toString() {
        return "SwitchedLoraExtra{" + (layer == null ? "unbound" : "layer " + layer.il()) + '}';
    }

    /**
     * The host side of the shared slot upload of one model's layers: the encoded slots and the step they were last
     * written for, so one write serves every layer of a step.
     */
    static final class SlotUpload {
        private int[] host = new int[0];
        private byte[] bytes = new byte[0];
        private long revision = -1L;
        private int pos0 = -1;
        private int nTokens;
        private GpuBuffer target;

        void reserve(int ubatch) {
            if (host.length < ubatch) {
                host = new int[ubatch];
                bytes = new byte[ubatch * Integer.BYTES];
            }
        }

        void write(GpuOps ops, Router router, int pos0, int nTokens, ScratchSlot dst) {
            if (revision == router.revision() && this.pos0 == pos0 && this.nTokens == nTokens
                    && target == dst.buffer()) {
                return;
            }
            router.slots(pos0, nTokens, host, 0);
            for (int t = 0; t < nTokens; t++) {
                int v = host[t];
                int b = t * Integer.BYTES;
                bytes[b] = (byte) v;
                bytes[b + 1] = (byte) (v >>> 8);
                bytes[b + 2] = (byte) (v >>> 16);
                bytes[b + 3] = (byte) (v >>> 24);
            }
            GpuEvent e = ops.queue().enqueueWriteBuffer(dst.buffer(), (long) dst.offset() * Float.BYTES,
                    (long) nTokens * Integer.BYTES, bytes);
            if (e != null) {
                e.close();
            }
            this.revision = router.revision();
            this.pos0 = pos0;
            this.nTokens = nTokens;
            this.target = dst.buffer();
        }
    }

    // ================================================================== device delta

    /**
     * {@value HeadsKernels#SWITCHED_LORA_ADD} over one stacked F32 pair: stage 0 {@code tmp = A[s] x}, then stage 1
     * {@code y += B[s] tmp}, on the queue of {@code ops}. The two stages are prepared once; a call re-sets the token
     * count, the views and the global sizes. A generic adapter is a pair with one slot ({@link #ofAdapter}), whose
     * device copies of {@code A}, {@code scale * B} and the zero slots the delta owns and releases on {@link #close}.
     */
    public static final class GpuDelta implements AutoCloseable {

        static final String TEMP_TMP = "switched_lora.tmp";

        private static final int ARG_X_STRIDE = 6;
        private static final int ARG_X_OFF = 7;
        private static final int ARG_Y_STRIDE = 8;
        private static final int ARG_Y_OFF = 9;
        private static final int ARG_TMP_OFF = 10;
        private static final int ARG_SLOTS = 11;
        private static final int ARG_X = 14;
        private static final int ARG_TMP = 15;
        private static final int ARG_Y = 16;

        private final GpuOps ops;
        private final int inDim;
        private final int outDim;
        private final int rank;
        private final int ubatch;
        private final PreparedDispatch down;
        private final PreparedDispatch up;
        /** Device zero slots of a one-slot adapter delta, null for a switched delta. */
        private GpuBuffer zeroSlots;
        /** Buffers this delta allocated (the uploads of a one-slot adapter delta). */
        private final List<GpuBuffer> owned = new ArrayList<>();
        /** The {@code tmp} view the stages are bound to; re-bound when the shared temporary is regrown. */
        private ScratchSlot tmp;

        GpuDelta(GpuOps ops, GpuBuffer a, GpuBuffer b, int inDim, int outDim, int rank, int nSlots) {
            HeadsKernels.checkLoraShape(inDim, outDim, rank, nSlots);
            this.ops = ops;
            this.inDim = inDim;
            this.outDim = outDim;
            this.rank = rank;
            this.ubatch = ops.scratch().ubatch();
            this.tmp = ops.scratch().temp(TEMP_TMP, (long) ubatch * rank);
            GpuBuffer ph = ops.placeholders().scratchFloats();
            GpuBuffer phInts = ops.placeholders().readOnlyInts();
            long[] local = {HeadsKernels.LORA_LOCAL};
            this.down = ops.prepare(HeadsKernels.SWITCHED_LORA_ADD,
                    new long[]{HeadsKernels.loraGlobal(HeadsKernels.LORA_STAGE_DOWN, 1, rank, outDim)}, local,
                    args(HeadsKernels.LORA_STAGE_DOWN, nSlots, tmp, phInts, a, b, ph));
            this.up = ops.prepare(HeadsKernels.SWITCHED_LORA_ADD,
                    new long[]{HeadsKernels.loraGlobal(HeadsKernels.LORA_STAGE_UP, 1, rank, outDim)}, local,
                    args(HeadsKernels.LORA_STAGE_UP, nSlots, tmp, phInts, a, b, ph));
        }

        private Object[] args(int stage, int nSlots, ScratchSlot tmp, GpuBuffer slots, GpuBuffer a, GpuBuffer b,
                              GpuBuffer ph) {
            return new Object[]{1, inDim, outDim, rank, stage, nSlots, inDim, 0, outDim, 0, tmp.offset(),
                    slots, a, b, ph, tmp.buffer(), ph};
        }

        /**
         * A generic LoRA delta: uploads {@code A} and {@code scale * B} to F32 device buffers owned by the returned
         * delta and adds them with one slot of zeros.
         *
         * @throws IllegalArgumentException for an embedding pair, which is applied per looked-up row
         */
        public static GpuDelta ofAdapter(GpuOps ops, LoraAdapter.Delta d) {
            Objects.requireNonNull(ops, "ops");
            Objects.requireNonNull(d, "delta");
            LoraAdapter.Pair p = d.pair();
            if (p.embedding()) {
                throw new IllegalArgumentException(p.target() + " is an embedding table, not a projection");
            }
            List<GpuBuffer> uploads = new ArrayList<>();
            try {
                GpuBuffer a = upload(ops, p.target() + LoraAdapterLoader.SUFFIX_A, d.a(), uploads);
                GpuBuffer b = upload(ops, p.target() + LoraAdapterLoader.SUFFIX_B, d.scaledB(), uploads);
                int ubatch = ops.scratch().ubatch();
                GpuBuffer zeros = allocateFloats(ops, p.target() + " LoRA slots", ubatch, false);
                uploads.add(zeros);
                ops.fill(zeros, 0L, (long) ubatch * Integer.BYTES, 0);
                GpuDelta delta = new GpuDelta(ops, a, b, p.inDim(), p.outDim(), p.rank(), 1);
                delta.down.setBuffer(ARG_SLOTS, zeros);
                delta.up.setBuffer(ARG_SLOTS, zeros);
                delta.zeroSlots = zeros;
                delta.owned.addAll(uploads);
                return delta;
            } catch (RuntimeException e) {
                closeAll(uploads);
                throw e;
            }
        }

        private static GpuBuffer upload(GpuOps ops, String what, float[] values, List<GpuBuffer> owned) {
            GpuBuffer buf = allocateFloats(ops, what, values.length, true);
            owned.add(buf);
            GpuEvent e = ops.queue().enqueueWriteBuffer(buf, 0L, (long) values.length * Float.BYTES, values);
            if (e != null) {
                e.close();
            }
            return buf;
        }

        /** {@link #add(int, GpuBuffer, GpuBuffer, int, int, GpuBuffer, int, int)} of a one-slot adapter delta. */
        public void add(int nTokens, GpuBuffer x, int xOff, int xStride, GpuBuffer y, int yOff, int yStride) {
            if (zeroSlots == null) {
                throw new IllegalStateException("a switched delta needs the step's slot buffer");
            }
            add(nTokens, zeroSlots, x, xOff, xStride, y, yOff, yStride);
        }

        /**
         * Enqueues both stages for {@code nTokens} rows with the slot ids in {@code slots} (ints from element 0).
         *
         * @throws IllegalArgumentException for a row count outside {@code [1, ubatch]} or strides below the widths
         */
        public void add(int nTokens, GpuBuffer slots, GpuBuffer x, int xOff, int xStride, GpuBuffer y, int yOff,
                        int yStride) {
            if (nTokens < 1 || nTokens > ubatch) {
                throw new IllegalArgumentException("LoRA delta of " + nTokens + " rows, ubatch is " + ubatch);
            }
            if ((nTokens > 1 && (xStride < inDim || yStride < outDim)) || xOff < 0 || yOff < 0) {
                throw new IllegalArgumentException("LoRA delta views: x stride " + xStride + " (in " + inDim
                        + "), y stride " + yStride + " (out " + outDim + "), offsets " + xOff + "/" + yOff);
            }
            Objects.requireNonNull(slots, "slots");
            Objects.requireNonNull(x, "x");
            Objects.requireNonNull(y, "y");
            ScratchSlot current = ops.scratch().temp(TEMP_TMP, (long) ubatch * rank);
            if (current.buffer() != tmp.buffer() || current.offset() != tmp.offset()) {
                tmp = current;
                for (PreparedDispatch d : new PreparedDispatch[]{down, up}) {
                    d.setInt(ARG_TMP_OFF, tmp.offset());
                    d.setBuffer(ARG_TMP, tmp.buffer());
                }
            }
            stage(down, HeadsKernels.LORA_STAGE_DOWN, nTokens, slots, x, xOff, xStride, y, yOff, yStride);
            stage(up, HeadsKernels.LORA_STAGE_UP, nTokens, slots, x, xOff, xStride, y, yOff, yStride);
        }

        private void stage(PreparedDispatch d, int stage, int nTokens, GpuBuffer slots, GpuBuffer x, int xOff,
                           int xStride, GpuBuffer y, int yOff, int yStride) {
            d.setInt(HeadsKernels.ARG_LORA_N_TOKENS, nTokens);
            d.setInt(ARG_X_STRIDE, xStride);
            d.setInt(ARG_X_OFF, xOff);
            d.setInt(ARG_Y_STRIDE, yStride);
            d.setInt(ARG_Y_OFF, yOff);
            d.setBuffer(ARG_SLOTS, slots);
            d.setBuffer(ARG_X, x);
            d.setBuffer(ARG_Y, y);
            d.setGlobal(HeadsKernels.loraGlobal(stage, nTokens, rank, outDim));
            d.dispatch();
        }

        /** Releases the prepared stages and the owned uploads. Idempotent. */
        @Override
        public void close() {
            down.close();
            up.close();
            closeAll(owned);
        }
    }

    // ================================================================== router

    /**
     * The graniteswitch adapter router of one model (granite-switch.cpp:27-72, 152-196, 260-288).
     *
     * <p>llama.cpp computes the slot of token {@code p} with a single causal attention head at the extra router layer:
     * {@code q = 1}, {@code k_j = +gain} for an adapter-activation token and {@code -gain} otherwise,
     * {@code v_j} = the token's slot ({@code i + 1} for adapter {@code i}) or 0, scale 1, then
     * {@code round(clamp(out, 0, n_adapters))}. Over the visible positions {@code j <= p} that attention equals the
     * closed form
     * <pre>
     *   out_p = A_p / (nA_p + e^(-2 gain) nB_p)
     * </pre>
     * with {@code A_p} the sum of the adapter slots, {@code nA_p} the adapter-token count and {@code nB_p} the count
     * of other tokens; the router evaluates it in float, as the graph does, from the token history, so it needs no
     * router KV layer. The rounding is {@code roundf} (half away from zero, ggml_round). It also maps adapter-activation tokens to their substitute ids for the embedding (:139-145, :216-221).</p>
     *
     * <p>The history follows the sequence: {@link #accept} records the tokens at their positions (dropping any later
     * history, as a rollback does), {@link #shift} mirrors a context shift and {@link #reset} clears it. Not
     * thread-safe.</p>
     */
    public static final class Router {

        private final int nAdapters;
        private final float decay;
        private final Map<Integer, int[]> adapterTokens;
        private int[] value = new int[0];
        private long[] slotSum = new long[0];
        private int[] adapterCount = new int[0];
        private int length;
        private long revision;

        private Router(int nAdapters, float gain, int[] activate, int[] substitute) {
            if (nAdapters < 0) {
                throw new IllegalArgumentException("adapter count must be non-negative, got " + nAdapters);
            }
            if (activate.length != nAdapters || substitute.length != nAdapters) {
                throw new IllegalArgumentException("adapter token id arrays (" + activate.length + " activate, "
                        + substitute.length + " substitute) do not match adapter count " + nAdapters);
            }
            if (!Float.isFinite(gain)) {
                throw new IllegalArgumentException("router gain must be finite, got " + gain);
            }
            this.nAdapters = nAdapters;
            this.decay = (float) Math.exp(-2.0 * gain);
            this.adapterTokens = new HashMap<>();
            for (int i = 0; i < nAdapters; i++) {
                adapterTokens.put(activate[i], new int[]{i + 1, substitute[i]});
            }
        }

        /** The router of a graniteswitch model. */
        public static Router of(ModelHparams hp) {
            Objects.requireNonNull(hp, "hparams");
            return new Router(hp.adapterCount(), hp.adapterRouterGain(), hp.adapterTokenIdsActivate(),
                    hp.adapterTokenIdsSubstitute());
        }

        /**
         * Explicit adapters: token {@code activate[i]} selects slot {@code i + 1} and embeds as
         * {@code substitute[i]}.
         */
        public static Router of(float gain, int[] activate, int[] substitute) {
            Objects.requireNonNull(activate, "activate");
            Objects.requireNonNull(substitute, "substitute");
            return new Router(activate.length, gain, activate.clone(), substitute.clone());
        }

        /** Slots of the stacked LoRA tensors: the adapters plus the zero slot 0. */
        public int nSlots() {
            return nAdapters + 1;
        }

        /** Number of recorded positions. */
        public int length() {
            return length;
        }

        /** A counter that changes whenever the recorded history changes. */
        public long revision() {
            return revision;
        }

        /** The ids to embed: adapter-activation tokens replaced by their substitutes (a new array). */
        public int[] substitute(int[] tokens) {
            int[] out = tokens.clone();
            for (int i = 0; i < out.length; i++) {
                int[] a = adapterTokens.get(out[i]);
                if (a != null) {
                    out[i] = a[1];
                }
            }
            return out;
        }

        /**
         * Records {@code tokens} at positions {@code pos0 ..}; history after them is dropped.
         *
         * @throws IllegalArgumentException when {@code pos0} leaves a gap after the recorded positions
         */
        public void accept(int pos0, int[] tokens) {
            Objects.requireNonNull(tokens, "tokens");
            if (pos0 < 0 || pos0 > length) {
                throw new IllegalArgumentException("router position " + pos0 + " is outside [0, " + length + "]");
            }
            ensureCapacity(pos0 + tokens.length);
            for (int t = 0; t < tokens.length; t++) {
                int[] a = adapterTokens.get(tokens[t]);
                value[pos0 + t] = a == null ? 0 : a[0];
            }
            length = pos0 + tokens.length;
            revision++;
            recompute(pos0);
        }

        /** The slot of position {@code pos}. */
        public int slot(int pos) {
            if (pos < 0 || pos >= length) {
                throw new IllegalArgumentException("router position " + pos + " is outside [0, " + length + ")");
            }
            int nA = adapterCount[pos];
            if (nA == 0) {
                return 0;
            }
            float out = (float) slotSum[pos] / ((float) nA + decay * (float) (pos + 1 - nA));
            float clamped = Math.min(Math.max(out, 0.0f), (float) nAdapters);
            return Math.round(clamped);
        }

        /** Slots of positions {@code [pos0, pos0 + n)} into {@code dst[dstOff ..]}. */
        public void slots(int pos0, int n, int[] dst, int dstOff) {
            for (int t = 0; t < n; t++) {
                dst[dstOff + t] = slot(pos0 + t);
            }
        }

        /** Drops the last {@code nTokens} recorded positions. */
        public void rollback(int nTokens) {
            if (nTokens < 0 || nTokens > length) {
                throw new IllegalArgumentException("cannot roll back " + nTokens + " of " + length + " positions");
            }
            length -= nTokens;
            revision++;
        }

        /** Removes positions {@code [nKeep, nKeep + nDiscard)} and moves the later ones down. */
        public void shift(int nKeep, int nDiscard) {
            if (nKeep < 0 || nDiscard < 0 || nKeep + nDiscard > length) {
                throw new IllegalArgumentException("context shift keep " + nKeep + " discard " + nDiscard
                        + " exceeds " + length + " positions");
            }
            System.arraycopy(value, nKeep + nDiscard, value, nKeep, length - nKeep - nDiscard);
            length -= nDiscard;
            revision++;
            recompute(nKeep);
        }

        /** Clears the history. */
        public void reset() {
            length = 0;
            revision++;
        }

        private void recompute(int from) {
            for (int p = from; p < length; p++) {
                long sum = p > 0 ? slotSum[p - 1] : 0L;
                int cnt = p > 0 ? adapterCount[p - 1] : 0;
                if (value[p] > 0) {
                    sum += value[p];
                    cnt++;
                }
                slotSum[p] = sum;
                adapterCount[p] = cnt;
            }
        }

        private void ensureCapacity(int n) {
            if (n <= value.length) {
                return;
            }
            int cap = Math.max(n, Math.max(16, value.length * 2));
            value = Arrays.copyOf(value, cap);
            slotSum = Arrays.copyOf(slotSum, cap);
            adapterCount = Arrays.copyOf(adapterCount, cap);
        }
    }
}
