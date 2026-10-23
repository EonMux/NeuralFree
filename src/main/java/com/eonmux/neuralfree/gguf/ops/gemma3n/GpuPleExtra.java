package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nBlocks.PleLayout;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * Per-layer embedding of gemma4 and gemma-embedding2 on the GPU ({@code LayerSpec.Extras.PLE}).
 *
 * <p>{@link #beforeBlock} of layer 0 computes the per-layer inputs of every layer from the embedding output
 * ({@link PerLayerInputs}, {@link PleLayout}); {@link #afterBlock} of layer {@code il} adds
 * {@code rms_norm(per_layer_proj . (gelu(per_layer_inp_gate . x) * inp_per_layer[il])) * per_layer_post_norm} to the
 * residual stream {@code x} (gemma4.cpp:376-395, gemma-embedding2.cpp:172-193).</p>
 *
 * <p>Dispatches per {@link #afterBlock} of {@code T} tokens ({@link LayerGate}): the {@code inp_gate} projection,
 * {@code LLM_GLU} (GEGLU_TANH) against the layer's input view, the {@code proj} projection, {@code LLM_RowNorm}
 * (RMS), and the residual add ({@code LLM_Axpby_F32}). Every buffer is a scratch slot or temporary reserved at bind;
 * a step enqueues work only. Not thread-safe; one forward pass at a time.</p>
 */
public final class GpuPleExtra implements GpuLayerExtra {

    /** Scratch temporary: per-layer inputs of the step, {@code ubatch * n_layer * width}, shared by every layer. */
    static final String TEMP_INPUTS = "gemma3n.ple.inputs";
    /** Scratch temporary: per-layer token rows of the step, {@code ubatch * n_layer * width}. */
    static final String TEMP_TOKENS = "gemma3n.ple.tokens";
    /** Scratch temporary: gate activations, {@code ubatch * width}. */
    static final String TEMP_GATE = "gemma3n.ple.gate";
    /** Scratch temporary: per-layer embedding output, {@code ubatch * n_embd}. */
    static final String TEMP_OUT = "gemma3n.ple.out";

    private GpuOps ops;
    private int ubatch;
    private int nEmbd;
    private PerLayerInputs producer;
    private ScratchSlot inputs;
    private LayerGate gate;

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (ops != null) {
            throw new IllegalStateException("per-layer embedding extra is already bound");
        }
        ModelSpec spec = ctx.spec();
        String arch = spec.arch().ggufName();
        if (layer.hasExtra(LayerSpec.Extras.ALTUP)) {
            throw new IllegalStateException(arch + " layer " + layer.il() + " lists both '" + LayerSpec.Extras.PLE
                    + "' and '" + LayerSpec.Extras.ALTUP + "'; the AltUp extra applies the gemma3n per-layer embedding");
        }
        requireFirstLayer(spec, LayerSpec.Extras.PLE);
        PleLayout layout = PleLayout.resolve(spec);
        layout.requireLayerTensors(spec, layer);
        if (layer.residual().streamWidth() != layout.nEmbd()) {
            throw new IllegalArgumentException(arch + " layer " + layer.il() + ": the per-layer embedding needs a "
                    + "residual stream of n_embd " + layout.nEmbd() + ", got " + layer.residual().streamWidth());
        }
        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        this.ubatch = ctx.scratch().ubatch();
        this.nEmbd = layout.nEmbd();
        this.producer = layer.il() == 0 ? new PerLayerInputs(ctx, weights, layout) : null;
        this.inputs = PerLayerInputs.slot(ctx.scratch(), layout);
        this.gate = new LayerGate(ctx, weights, layer, layout);
        this.ops = ctx.ops();
    }

    @Override
    public void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
        checkStep(nTokens, x);
        if (producer != null) {
            producer.compute(nTokens, x);
        }
    }

    @Override
    public void afterBlock(int pos0, int nTokens, ScratchSlot x) {
        checkStep(nTokens, x);
        ScratchSlot out = gate.apply(nTokens, x.buffer(), x.offset(), inputs);
        ops.add(x.buffer(), x.offset(), out.buffer(), out.offset(), x.buffer(), x.offset(), nTokens * nEmbd);
    }

    private void checkStep(int nTokens, ScratchSlot x) {
        if (ops == null) {
            throw new IllegalStateException("per-layer embedding extra is not bound");
        }
        checkTokens(nTokens, ubatch, "per-layer embedding");
        Objects.requireNonNull(x, "x").require((long) nTokens * nEmbd);
    }

    /**
     * Requires layer 0 of the model to list {@code id}: its extra computes the per-layer inputs every layer reads.
     *
     * @throws IllegalStateException otherwise
     */
    static void requireFirstLayer(ModelSpec spec, String id) {
        if (spec.nLayer() == 0 || !spec.layer(0).hasExtra(id)) {
            throw new IllegalStateException(spec.arch().ggufName() + ": layer 0 must list '" + id
                    + "', whose extra computes the per-layer inputs of every layer");
        }
    }

    /**
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    static void checkTokens(int nTokens, int ubatch, String what) {
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException(what + ": token count " + nTokens + " is outside [1, " + ubatch + "]");
        }
    }

    /**
     * A resident tensor with the catalog's shape.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when it is not resident
     * @throws IllegalArgumentException      when its resident shape differs
     */
    static GpuWeight resident(GpuWeights weights, String name, long cols, long rows) {
        GpuWeight w = weights.get(name);
        if (w.cols() != cols || w.rows() != rows) {
            throw new IllegalArgumentException("tensor " + name + " is resident as " + w + ", expected " + cols
                    + " columns and " + rows + " rows");
        }
        return w;
    }

    /**
     * A resident F32 vector of {@code n} floats in one buffer from byte 0, as the kernels that take a raw weight
     * buffer read it.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when it is not resident
     * @throws IllegalArgumentException      when it is not a single F32 buffer of {@code n} floats
     */
    static GpuWeight residentVector(GpuWeights weights, String name, long n) {
        GpuWeight w = weights.get(name);
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.cols() * w.rows() != n) {
            throw new IllegalArgumentException("tensor " + name + " must be one F32 buffer of " + n + " floats, got "
                    + w);
        }
        return w;
    }

    @Override
    public String toString() {
        return "GpuPleExtra{" + (gate != null ? "layer " + gate.il : "unbound") + (producer != null ? ", inputs" : "")
                + '}';
    }

    /**
     * Computes the per-layer inputs of a step into {@link #TEMP_INPUTS} ({@link PleLayout}). Dispatches for
     * {@code T} tokens: the {@code per_layer_model_proj} projection, a scale by {@code 1/sqrt(n_embd)},
     * {@code LLM_RowNorm} (RMS, {@code T*n_layer} rows of {@code width}); with a token table, the row lookup
     * ({@link RowLookup}), a scale by {@code sqrt(width)}, the add and a scale by {@code 1/sqrt(2)}.
     */
    static final class PerLayerInputs {
        private final PleLayout layout;
        private final GpuOps ops;
        private final int ubatch;
        private final GpuWeight modelProj;
        private final GpuWeight projNorm;
        private final ScratchSlot inputs;
        private final ScratchSlot tokens;
        private final RowLookup lookup;

        /**
         * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is not resident
         * @throws IllegalArgumentException      when a resident shape differs from the layout
         */
        PerLayerInputs(GpuForwardContext ctx, GpuWeights weights, PleLayout layout) {
            this.layout = Objects.requireNonNull(layout, "layout");
            this.ops = ctx.ops();
            this.ubatch = ctx.scratch().ubatch();
            this.modelProj = resident(weights, layout.modelProj().name(), layout.nEmbd(), layout.rowStride());
            this.projNorm = residentVector(weights, layout.projNorm().name(), layout.width());
            this.inputs = slot(ctx.scratch(), layout);
            if (layout.tokenTable() != null) {
                GpuWeight table = resident(weights, layout.tokenTable().name(), layout.rowStride(),
                        layout.tokenTable().rows());
                this.tokens = ctx.scratch().temp(TEMP_TOKENS, layout.floats(ubatch));
                this.lookup = RowLookup.forStepTokens(ctx, table);
            } else {
                this.tokens = null;
                this.lookup = null;
            }
        }

        /** The shared inputs temporary of a model ({@link #TEMP_INPUTS}). */
        static ScratchSlot slot(GpuScratch scratch, PleLayout layout) {
            return scratch.temp(TEMP_INPUTS, layout.floats(scratch.ubatch()));
        }

        /** Computes the inputs of the step's {@code nTokens} tokens from the embedding rows {@code x}. */
        void compute(int nTokens, ScratchSlot x) {
            checkTokens(nTokens, ubatch, "per-layer inputs");
            int n = nTokens * layout.rowStride();
            GpuBuffer in = inputs.buffer();
            int off = inputs.offset();
            ops.matmul(modelProj, x.buffer(), x.offset(), in, off, nTokens);
            ops.scale(in, off, n, layout.projScale());
            ops.norm(NormKind.RMS, projNorm, null, layout.eps(), in, off, in, off, nTokens * layout.nLayer(),
                    layout.width());
            if (lookup == null) {
                return;
            }
            lookup.stepTokens(nTokens, tokens.buffer(), tokens.offset());
            ops.scale(tokens.buffer(), tokens.offset(), n, layout.tokenScale());
            ops.add(in, off, tokens.buffer(), tokens.offset(), in, off, n);
            ops.scale(in, off, n, layout.inputScale());
        }
    }

    /**
     * The per-layer embedding gate of one layer:
     * {@code out = rms_norm(per_layer_proj . (gelu(per_layer_inp_gate . in) * inp_per_layer[il])) * per_layer_post_norm}
     * (gemma3n.cpp:236-249, gemma4.cpp:379-393) into {@link #TEMP_OUT}.
     */
    static final class LayerGate {
        private final GpuOps ops;
        private final PleLayout layout;
        private final int il;
        private final int ubatch;
        private final GpuWeight inpGate;
        private final GpuWeight proj;
        /** LoRA deltas of the gate and its projection ({@code build_lora_mm}, gemma3n.cpp:237, 245). */
        private final GpuProjectionDeltas.Site inpGateDeltas;
        private final GpuProjectionDeltas.Site projDeltas;
        private final GpuWeight postNorm;
        private final ScratchSlot gate;
        private final ScratchSlot out;

        /**
         * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is not resident
         * @throws IllegalArgumentException      when a resident shape differs from the layout
         */
        LayerGate(GpuForwardContext ctx, GpuWeights weights, LayerSpec layer, PleLayout layout) {
            this.ops = ctx.ops();
            this.layout = layout;
            this.il = layer.il();
            this.ubatch = ctx.scratch().ubatch();
            int w = layout.width();
            int e = layout.nEmbd();
            this.inpGate = resident(weights, layer.tensorName(TensorNames.weight(TensorNames.PER_LAYER_INP_GATE)), e, w);
            this.proj = resident(weights, layer.tensorName(TensorNames.weight(TensorNames.PER_LAYER_PROJ)), w, e);
            this.inpGateDeltas = ctx.projectionDeltas().serve(layer,
                    TensorNames.weight(TensorNames.PER_LAYER_INP_GATE), inpGate);
            this.projDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.PER_LAYER_PROJ), proj);
            this.postNorm = residentVector(weights,
                    layer.tensorName(TensorNames.weight(TensorNames.PER_LAYER_POST_NORM)), e);
            this.gate = ctx.scratch().temp(TEMP_GATE, (long) ubatch * w);
            this.out = ctx.scratch().temp(TEMP_OUT, (long) ubatch * e);
        }

        /**
         * Applies the gate to {@code nTokens} rows of {@code n_embd} at {@code in[inOff]} (which must not overlap the
         * gate temporaries) and returns the slot holding the {@code [nTokens][n_embd]} result.
         */
        ScratchSlot apply(int nTokens, GpuBuffer in, int inOff, ScratchSlot inputs) {
            int w = layout.width();
            ops.matmul(inpGate, in, inOff, gate.buffer(), gate.offset(), nTokens);
            inpGateDeltas.applyRows(nTokens, in, inOff, gate.buffer(), gate.offset());
            ops.glu(GluOp.GEGLU_TANH, ActParams.NONE, gate.buffer(), gate.offset(), w,
                    inputs.buffer(), inputs.offset() + layout.layerOffset(il), layout.rowStride(),
                    gate.buffer(), gate.offset(), w, nTokens, w);
            ops.matmul(proj, gate.buffer(), gate.offset(), out.buffer(), out.offset(), nTokens);
            projDeltas.applyRows(nTokens, gate.buffer(), gate.offset(), out.buffer(), out.offset());
            ops.norm(NormKind.RMS, postNorm, null, layout.eps(), out.buffer(), out.offset(), out.buffer(), out.offset(),
                    nTokens, layout.nEmbd());
            return out;
        }
    }

    /**
     * The current step's token ids on the host: the copy the embedding stage records in the scratch
     * ({@link GpuScratch#recordStepTokens}) when it writes {@link ScratchSlots#TOKEN_IDS}, so reading them never waits
     * on the queue.
     */
    static final class StepTokens {
        private final GpuScratch scratch;

        StepTokens(GpuForwardContext ctx) {
            this.scratch = ctx.scratch();
        }

        /**
         * Writes the ids of the step's {@code nTokens} tokens to {@code dst[dstOff ..]}.
         *
         * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]} or {@code dst} is too
         *                                  short
         * @throws IllegalStateException    when the embedding stage recorded a step of another size
         */
        void read(int nTokens, int[] dst, int dstOff) {
            checkTokens(nTokens, scratch.ubatch(), "step token read");
            scratch.readStepTokens(dst, dstOff, nTokens);
        }
    }

    /**
     * Row lookup in an embedding table on the device. A table held in one buffer is read through device token ids:
     * the step's {@link ScratchSlots#TOKEN_IDS} (ids from element 0) or ids this lookup writes to its own
     * temporary from a reused host array. A table split into row shards is read through host ids
     * ({@link GpuOps#embed(GpuWeight, int[], GpuBuffer, int)}), the step's tokens then coming from
     * {@link StepTokens}. Host id arrays are kept per count, so steady-state steps allocate nothing.
     */
    static final class RowLookup {
        private final GpuOps ops;
        private final ComputeQueue queue;
        private final GpuWeight table;
        private final ScratchSlot deviceIds;
        private final byte[] idBytes;
        private final StepTokens stepTokens;
        private final int[][] hostIds;

        private RowLookup(GpuForwardContext ctx, GpuWeight table, ScratchSlot deviceIds, byte[] idBytes,
                          StepTokens stepTokens, int maxIds) {
            this.ops = ctx.ops();
            this.queue = ctx.queue();
            this.table = table;
            this.deviceIds = deviceIds;
            this.idBytes = idBytes;
            this.stepTokens = stepTokens;
            this.hostIds = new int[maxIds + 1][];
        }

        /**
         * A lookup of the step's tokens ({@link #stepTokens}).
         *
         * @throws IllegalStateException when a single-buffer table meets a token-id slot that does not start at
         *                               element 0 of its buffer
         */
        static RowLookup forStepTokens(GpuForwardContext ctx, GpuWeight table) {
            int ubatch = ctx.scratch().ubatch();
            if (!table.isSingleShard()) {
                return new RowLookup(ctx, table, null, null, new StepTokens(ctx), ubatch);
            }
            ScratchSlot ids = ctx.scratch().slot(ScratchSlots.TOKEN_IDS);
            if (ids.offset() != 0) {
                throw new IllegalStateException("the token-id slot must start at element 0 of its buffer for the "
                        + "lookup in " + table.name() + ", got offset " + ids.offset());
            }
            return new RowLookup(ctx, table, ids, null, null, ubatch);
        }

        /**
         * A lookup of host-computed row ids ({@link #rows}), at most {@code maxIds} per call; single-buffer tables
         * read them from the temporary {@code idTemp}.
         */
        static RowLookup forHostIds(GpuForwardContext ctx, GpuWeight table, String idTemp, int maxIds) {
            if (maxIds < 1) {
                throw new IllegalArgumentException("row lookup in " + table.name() + " needs a positive id capacity");
            }
            if (!table.isSingleShard()) {
                return new RowLookup(ctx, table, null, null, null, maxIds);
            }
            ScratchSlot ids = ctx.scratch().temp(idTemp, maxIds);
            if (ids.offset() != 0) {
                throw new IllegalStateException("row-id temporary " + idTemp + " must start at element 0 of its buffer");
            }
            return new RowLookup(ctx, table, ids, new byte[maxIds * Integer.BYTES], null, maxIds);
        }

        /**
         * Looks up the rows of the step's {@code nTokens} tokens into {@code out[outOff + t*cols ...]}.
         *
         * @throws IllegalStateException when this lookup was made for host ids
         */
        void stepTokens(int nTokens, GpuBuffer out, int outOff) {
            if (deviceIds != null && idBytes == null) {
                ops.embed(table, deviceIds.buffer(), nTokens, out, outOff);
                return;
            }
            if (stepTokens == null) {
                throw new IllegalStateException("row lookup in " + table.name() + " was made for host row ids");
            }
            int[] ids = hostArray(nTokens);
            stepTokens.read(nTokens, ids, 0);
            ops.embed(table, ids, out, outOff);
        }

        /**
         * Looks up rows {@code ids[0 .. count)} into {@code out[outOff + i*cols ...]}.
         *
         * @throws IllegalArgumentException when {@code count} exceeds the capacity or an id is outside the table
         */
        void rows(int[] ids, int count, GpuBuffer out, int outOff) {
            if (count < 1 || count >= hostIds.length || ids.length < count) {
                throw new IllegalArgumentException("row lookup of " + count + " ids in " + table.name()
                        + " exceeds the capacity " + (hostIds.length - 1) + " or the id array");
            }
            if (deviceIds == null) {
                int[] host = hostArray(count);
                System.arraycopy(ids, 0, host, 0, count);
                ops.embed(table, host, out, outOff);
                return;
            }
            if (idBytes == null) {
                throw new IllegalStateException("row lookup in " + table.name() + " reads the step's token ids");
            }
            long rows = table.rows();
            for (int i = 0; i < count; i++) {
                int v = ids[i];
                if (v < 0 || v >= rows) {
                    throw new IllegalArgumentException("row id " + v + " at " + i + " is outside the " + rows
                            + " rows of " + table.name());
                }
                int b = i * Integer.BYTES;
                idBytes[b] = (byte) v;
                idBytes[b + 1] = (byte) (v >>> 8);
                idBytes[b + 2] = (byte) (v >>> 16);
                idBytes[b + 3] = (byte) (v >>> 24);
            }
            GpuEvent e = queue.enqueueWriteBuffer(deviceIds.buffer(), (long) deviceIds.offset() * Integer.BYTES,
                    (long) count * Integer.BYTES, idBytes);
            if (e != null) {
                e.close();
            }
            ops.embed(table, deviceIds.buffer(), count, out, outOff);
        }

        private int[] hostArray(int n) {
            int[] a = hostIds[n];
            if (a == null) {
                a = new int[n];
                hostIds[n] = a;
            }
            return a;
        }
    }
}
