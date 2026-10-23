package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nBlocks.PleLayout;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU twin of {@link GpuPleExtra}: the per-layer embedding of gemma4 and gemma-embedding2
 * ({@code LayerSpec.Extras.PLE}). {@link #beforeBlock} of layer 0 computes the per-layer inputs of every layer
 * ({@link PerLayerInputs}); {@link #afterBlock} adds the layer's gated per-layer embedding ({@link LayerGate}) to the
 * residual stream (gemma4.cpp:376-395, gemma-embedding2.cpp:172-193). Working arrays are scratch temporaries
 * reserved at bind; the per-layer inputs carry data from layer 0 to every later layer under a name only the Gemma3n
 * extras use. Not thread-safe.
 */
public final class CpuPleExtra implements CpuLayerExtra {

    private int ubatch;
    private int nEmbd;
    private PerLayerInputs producer;
    private float[] inputs;
    private LayerGate gate;

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (gate != null) {
            throw new IllegalStateException("per-layer embedding extra is already bound");
        }
        ModelSpec spec = ctx.spec();
        String arch = spec.arch().ggufName();
        if (layer.hasExtra(LayerSpec.Extras.ALTUP)) {
            throw new IllegalStateException(arch + " layer " + layer.il() + " lists both '" + LayerSpec.Extras.PLE
                    + "' and '" + LayerSpec.Extras.ALTUP + "'; the AltUp extra applies the gemma3n per-layer embedding");
        }
        GpuPleExtra.requireFirstLayer(spec, LayerSpec.Extras.PLE);
        PleLayout layout = PleLayout.resolve(spec);
        layout.requireLayerTensors(spec, layer);
        if (layer.residual().streamWidth() != layout.nEmbd()) {
            throw new IllegalArgumentException(arch + " layer " + layer.il() + ": the per-layer embedding needs a "
                    + "residual stream of n_embd " + layout.nEmbd() + ", got " + layer.residual().streamWidth());
        }
        this.ubatch = ctx.scratch().ubatch();
        this.nEmbd = layout.nEmbd();
        this.producer = layer.il() == 0 ? new PerLayerInputs(ctx, layout) : null;
        this.inputs = PerLayerInputs.array(ctx.scratch(), layout);
        this.gate = new LayerGate(ctx, layer, layout);
    }

    @Override
    public void beforeBlock(int pos0, int n, float[] x) {
        checkStep(n, x);
        if (producer != null) {
            producer.compute(n, x);
        }
    }

    @Override
    public void afterBlock(int pos0, int n, float[] x) {
        checkStep(n, x);
        float[] out = gate.apply(n, x, 0, inputs);
        CpuOps.add(x, 0, out, 0, x, 0, n * nEmbd);
    }

    private void checkStep(int n, float[] x) {
        if (gate == null) {
            throw new IllegalStateException("per-layer embedding extra is not bound");
        }
        GpuPleExtra.checkTokens(n, ubatch, "per-layer embedding");
        requireLength(x, (long) n * nEmbd, "residual stream");
    }

    /** @throws IllegalArgumentException when {@code a} holds fewer than {@code n} floats */
    static void requireLength(float[] a, long n, String what) {
        if (Objects.requireNonNull(a, what).length < n) {
            throw new IllegalArgumentException(what + " holds " + a.length + " floats, the step needs " + n);
        }
    }

    /**
     * A weight matrix of {@code rows x cols}.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when it is absent
     * @throws IllegalArgumentException      when its shape differs
     */
    static CpuTensorRef matrix(CpuWeights weights, String name, long cols, long rows) {
        CpuTensorRef w = weights.get(name);
        if (w.cols() != cols || w.rows() != rows) {
            throw new IllegalArgumentException("tensor " + name + " is " + w + ", expected " + cols + " columns and "
                    + rows + " rows");
        }
        if (!CpuMatVec.supports(w.type())) {
            throw new UnsupportedOperationException("tensor " + name + " has type " + w.type()
                    + ", which the CPU projections do not support");
        }
        return w;
    }

    /**
     * A vector tensor of exactly {@code n} floats, decoded to F32 (shared, read only).
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when it is absent
     * @throws IllegalArgumentException      when its length differs
     */
    static float[] vector(CpuWeights weights, String name, int n) {
        float[] v = weights.floats(name);
        if (v.length != n) {
            throw new IllegalArgumentException("tensor " + name + " holds " + v.length + " floats, expected " + n);
        }
        return v;
    }

    @Override
    public String toString() {
        return "CpuPleExtra{" + (gate != null ? "layer " + gate.il : "unbound") + (producer != null ? ", inputs" : "")
                + '}';
    }

    /**
     * CPU twin of {@link GpuPleExtra.PerLayerInputs}: the per-layer inputs of a step into
     * {@link GpuPleExtra#TEMP_INPUTS} ({@code [t][layer][width]}), token rows from {@link CpuScratch#TOKEN_IDS}.
     */
    static final class PerLayerInputs {
        private final PleLayout layout;
        private final CpuMatVec matVec;
        private final CpuWeights weights;
        private final CpuScratch scratch;
        private final int ubatch;
        private final CpuTensorRef modelProj;
        private final float[] projNorm;
        private final CpuTensorRef table;
        private final float[] inputs;
        private final float[] tokens;

        /**
         * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is absent
         * @throws IllegalArgumentException      when a shape differs from the layout
         */
        PerLayerInputs(CpuForwardContext ctx, PleLayout layout) {
            this.layout = Objects.requireNonNull(layout, "layout");
            this.matVec = ctx.matVec();
            this.weights = ctx.weights();
            this.scratch = ctx.scratch();
            this.ubatch = scratch.ubatch();
            this.modelProj = matrix(weights, layout.modelProj().name(), layout.nEmbd(), layout.rowStride());
            this.projNorm = vector(weights, layout.projNorm().name(), layout.width());
            this.inputs = array(scratch, layout);
            if (layout.tokenTable() != null) {
                this.table = weights.get(layout.tokenTable().name());
                if (table.cols() != layout.rowStride()) {
                    throw new IllegalArgumentException("tensor " + layout.tokenTable().name() + " rows hold " + table.cols()
                            + " floats, expected " + layout.rowStride());
                }
                this.tokens = scratch.temp(GpuPleExtra.TEMP_TOKENS, layout.floats(ubatch));
            } else {
                this.table = null;
                this.tokens = null;
            }
        }

        /** The shared inputs temporary of a model ({@link GpuPleExtra#TEMP_INPUTS}). */
        static float[] array(CpuScratch scratch, PleLayout layout) {
            return scratch.temp(GpuPleExtra.TEMP_INPUTS, layout.floats(scratch.ubatch()));
        }

        /** Computes the inputs of {@code n} tokens from the embedding rows {@code x}. */
        void compute(int n, float[] x) {
            GpuPleExtra.checkTokens(n, ubatch, "per-layer inputs");
            int stride = layout.rowStride();
            int total = n * stride;
            matVec.matmul(modelProj, x, 0, inputs, 0, n);
            CpuOps.scale(inputs, 0, total, layout.projScale());
            CpuOps.norm(NormKind.RMS, layout.eps(), inputs, 0, inputs, 0, n * layout.nLayer(), layout.width(),
                    projNorm, null);
            if (table == null) {
                return;
            }
            int[] ids = scratch.ints(CpuScratch.TOKEN_IDS);
            for (int t = 0; t < n; t++) {
                weights.embedRow(table, ids[t], tokens, t * stride);
            }
            CpuOps.scale(tokens, 0, total, layout.tokenScale());
            CpuOps.add(inputs, 0, tokens, 0, inputs, 0, total);
            CpuOps.scale(inputs, 0, total, layout.inputScale());
        }
    }

    /** CPU twin of {@link GpuPleExtra.LayerGate}. */
    static final class LayerGate {
        private final CpuMatVec matVec;
        private final PleLayout layout;
        private final int il;
        private final CpuTensorRef inpGate;
        private final CpuTensorRef proj;
        /** LoRA deltas of the gate and its projection ({@code build_lora_mm}, gemma3n.cpp:237, 245). */
        private final CpuProjectionDeltas.Site inpGateDeltas;
        private final CpuProjectionDeltas.Site projDeltas;
        private final float[] postNorm;
        private final float[] gate;
        private final float[] out;

        /**
         * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is absent
         * @throws IllegalArgumentException      when a shape differs from the layout
         */
        LayerGate(CpuForwardContext ctx, LayerSpec layer, PleLayout layout) {
            CpuWeights weights = ctx.weights();
            int ubatch = ctx.scratch().ubatch();
            int w = layout.width();
            int e = layout.nEmbd();
            this.matVec = ctx.matVec();
            this.layout = layout;
            this.il = layer.il();
            this.inpGate = matrix(weights, layer.tensorName(TensorNames.weight(TensorNames.PER_LAYER_INP_GATE)), e, w);
            this.proj = matrix(weights, layer.tensorName(TensorNames.weight(TensorNames.PER_LAYER_PROJ)), w, e);
            this.inpGateDeltas = ctx.projectionDeltas().serve(layer,
                    TensorNames.weight(TensorNames.PER_LAYER_INP_GATE), inpGate, matVec);
            this.projDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.PER_LAYER_PROJ), proj,
                    matVec);
            this.postNorm = vector(weights, layer.tensorName(TensorNames.weight(TensorNames.PER_LAYER_POST_NORM)), e);
            this.gate = ctx.scratch().temp(GpuPleExtra.TEMP_GATE, ubatch * w);
            this.out = ctx.scratch().temp(GpuPleExtra.TEMP_OUT, ubatch * e);
        }

        /**
         * Applies the gate to {@code n} rows of {@code n_embd} at {@code in[inOff]} and returns the array holding the
         * {@code [n][n_embd]} result from index 0.
         */
        float[] apply(int n, float[] in, int inOff, float[] inputs) {
            int w = layout.width();
            matVec.matmul(inpGate, in, inOff, gate, 0, n);
            inpGateDeltas.applyRows(n, in, inOff, gate, 0);
            CpuOps.gluRows(GluOp.GEGLU_TANH, ActParams.NONE, n, w, gate, 0, w,
                    inputs, layout.layerOffset(il), layout.rowStride(), gate, 0, w);
            matVec.matmul(proj, gate, 0, out, 0, n);
            projDeltas.applyRows(n, gate, 0, out, 0);
            CpuOps.norm(NormKind.RMS, layout.eps(), out, 0, out, 0, n, layout.nEmbd(), postNorm, null);
            return out;
        }
    }
}
