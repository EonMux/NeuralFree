package com.eonmux.neuralfree.gguf.ops.act;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.act.FfnBlocks.DenseLayout;
import com.eonmux.neuralfree.gguf.ops.gemma3n.CpuActivationSparsity;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * The dense feed-forward block ({@code FfnKind.DENSE}) on the CPU, the twin of {@link GpuDenseFfn}: llama.cpp
 * {@code build_ffn} (src/llama-graph.cpp:1922-2139) with the same stages in the same order.
 *
 * <ol>
 *   <li>Projections through the context's {@link CpuMatVec} (PRISM input rotation and output scale {@code w_s} of
 *       the weights), each followed by the {@link CpuProjectionDeltas} registered on it (LoRA). A projection with a
 *       bias (read only when {@link FfnSpec#biases()} holds) or a delta computes {@code (W x + delta + b) * s}: its
 *       product is taken without the scale, the delta and the bias row added ({@link CpuOps#addRowBias}), then the
 *       scale applied, as llama.cpp orders them (llama-graph.cpp:1959-1995, 2116-2136).</li>
 *   <li>Gemma3n activation sparsity on the gate rows ({@link CpuActivationSparsity#gaussianTopK}).</li>
 *   <li>{@link CpuOps#gluRows} of gate and up, or {@link CpuOps#unary} on the gate or up rows, then the division by
 *       {@code ffn.act.scales} after a GELU (mpt).</li>
 *   <li>bitnet's {@code ffn_sub_norm} (RMS, {@link CpuOps#norm}).</li>
 *   <li>The down projection into {@code out}.</li>
 * </ol>
 * Working rows come from {@link CpuScratch}: the {@code FFN_GATE_UP} slot and the {@link DenseLayout} temporaries,
 * all reserved at bind time, so a step allocates nothing. Not thread-safe; each stage is threaded by
 * {@link CpuParallel} inside the call.
 */
public final class CpuDenseFfn implements CpuFfnBlock {

    /** Rows per parallel task of the activation-scale division, about 32K elements per task. */
    private static final int DIVIDE_GRAIN_ELEMENTS = 1 << 15;

    private DenseLayout layout;
    private String where;
    private int ubatch;
    private float[] gateUp;
    private float[] upTemp;
    private float[] actTemp;
    private Projection gate;
    private Projection up;
    private Projection down;
    private float[] actScales;
    private float[] subNormWeight;
    private boolean bound;
    private boolean closed;

    /**
     * Binds the block: resolves the layer's tensors, validates them against its spec and reserves the working
     * arrays.
     *
     * @throws IllegalStateException   when already bound or closed, or the tensors contradict the spec
     * @throws MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        if (closed || bound) {
            throw new IllegalStateException("dense FFN block is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(ctx, "ctx");
        FfnSpec spec = FfnBlocks.denseSpec(layer);
        try {
            bindLayer(ctx, layer, spec);
        } catch (RuntimeException e) {
            unbind();
            throw e;
        }
        bound = true;
    }

    private void bindLayer(CpuForwardContext ctx, LayerSpec layer, FfnSpec spec) {
        CpuWeights weights = ctx.weights();
        where = ctx.spec().arch().ggufName() + " layer " + layer.stack() + "." + layer.il();
        ubatch = ctx.scratch().ubatch();
        CpuTensorRef upW = weights.layer(layer, spec.upWeight());
        CpuTensorRef gateW = weights.layerOrNull(layer, spec.gateWeight());
        CpuTensorRef downW = weights.layer(layer, spec.downWeight());
        layout = DenseLayout.of(layer, shape(upW), gateW != null ? shape(gateW) : null, shape(downW));

        CpuScratch scratch = ctx.scratch();
        gateUp = require(scratch.slot(CpuScratch.FFN_GATE_UP), layout.gateUpWidth(), CpuScratch.FFN_GATE_UP);
        if (layout.upTempWidth() > 0) {
            upTemp = scratch.temp(DenseLayout.TEMP_UP, rows(layout.upTempWidth()));
        }
        if (layout.form() == DenseLayout.Form.FUSED_GLU) {
            actTemp = scratch.temp(DenseLayout.TEMP_ACT, rows(layout.nFf()));
        }

        CpuMatVec base = ctx.matVec();
        CpuMatVec unscaled = new CpuMatVec(w -> withoutScale(base.transformsOf(w)));
        CpuProjectionDeltas deltas = ctx.projectionDeltas();
        boolean biases = spec.biases();
        if (layout.hasGate()) {
            gate = projection(weights, layer, gateW, layout.gateBase(), biases, base, unscaled, deltas);
        }
        up = projection(weights, layer, upW, layout.upBase(), biases, base, unscaled, deltas);
        down = projection(weights, layer, downW, layout.downBase(), biases, base, unscaled, deltas);

        if (layout.actScalesApply()) {
            actScales = weights.layerFloatsOrNull(layer, DenseLayout.ACT_SCALES);
            if (actScales != null) {
                checkVector(actScales, layout.nFf(), DenseLayout.ACT_SCALES);
            }
        }
        if (layout.subNorm()) {
            subNormWeight = weights.layerFloatsOrNull(layer, DenseLayout.SUB_NORM);
            if (subNormWeight == null) {
                throw new MissingTensorException(weights.archName(), layer.il(),
                        layer.tensorName(DenseLayout.SUB_NORM));
            }
            checkVector(subNormWeight, layout.nFf(), DenseLayout.SUB_NORM);
        }
    }

    /** Drops every reference a failed bind may have left, so the block reads as unbound. */
    private void unbind() {
        layout = null;
        where = null;
        gateUp = null;
        upTemp = null;
        actTemp = null;
        gate = null;
        up = null;
        down = null;
        actScales = null;
        subNormWeight = null;
    }

    /**
     * One projection of the layer; {@code biases} false ({@link FfnSpec#biases()}) leaves {@code <base>.bias} unread,
     * as the graph passes {@code NULL} for it.
     */
    private Projection projection(CpuWeights weights, LayerSpec layer, CpuTensorRef w, String base, boolean biases,
                                  CpuMatVec scaled, CpuMatVec unscaled, CpuProjectionDeltas deltas) {
        int rows = (int) w.rows();
        CpuMatVec.Transforms t = scaled.transformsOf(w);
        String refusal = t.input() != null ? layer.tensorName(TensorNames.weight(base)) + " rotates its input (PRISM"
                + " Hadamard) inside the matvec; a LoRA delta would need the rotated rows" : null;
        CpuProjectionDeltas.Site site = deltas.serve(layer, TensorNames.weight(base), refusal);
        float[] bias = biases ? weights.layerFloatsOrNull(layer, TensorNames.bias(base)) : null;
        if (bias != null) {
            checkVector(bias, rows, layer.tensorName(TensorNames.bias(base)));
        }
        return new Projection(w, rows, scaled, t.hasScale() ? unscaled : scaled, bias,
                t.hasScale() ? t.scale(0) : 1.0f, site);
    }

    /** The same transforms without the output scale, for projections whose scale follows their bias. */
    private static CpuMatVec.Transforms withoutScale(CpuMatVec.Transforms t) {
        return t.hasScale() ? CpuMatVec.Transforms.of(t.input(), null, t.experts()) : t;
    }

    // ------------------------------------------------------------------ step

    /**
     * Runs the FFN over {@code n} rows of {@code in} (the up projection's input width, from index 0) into {@code out}
     * (the down projection's output width, from index 0).
     *
     * @throws IllegalStateException    when not bound or closed
     * @throws IllegalArgumentException for a row count outside {@code [1, ubatch]}, arrays too small for the rows,
     *                                  or an output array that is the input or one of the FFN's working arrays
     */
    @Override
    public void forward(int n, float[] in, float[] out) {
        if (closed || !bound) {
            throw new IllegalStateException("dense FFN block is " + (closed ? "closed" : "not bound"));
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException(where + ": FFN row count " + n + " outside [1, " + ubatch + "]");
        }
        checkLength("input", in, (long) n * layout.nIn());
        checkLength("output", out, (long) n * layout.nOut());
        float[] act = activated();
        if (out == in || out == act || out == gateUp || out == upTemp) {
            throw new IllegalArgumentException(where + ": FFN output array aliases its input or working rows");
        }
        int nFf = layout.nFf();
        switch (layout.form()) {
            case PARALLEL_GLU:
                gate.run(n, in, gateUp);
                sparsity(n);
                up.run(n, in, upTemp);
                CpuOps.gluRows(layout.glu(), layout.params(), n, nFf, gateUp, 0, nFf, upTemp, 0, nFf,
                        gateUp, 0, nFf);
                break;
            case FUSED_GLU:
                up.run(n, in, gateUp);
                sparsity(n);
                CpuOps.gluRows(layout.glu(), layout.params(), n, nFf, gateUp, 0, 2 * nFf, gateUp, nFf, 2 * nFf,
                        actTemp, 0, nFf);
                break;
            case SEQUENTIAL:
                up.run(n, in, upTemp);
                gate.run(n, upTemp, gateUp);
                activate(n);
                break;
            default:
                up.run(n, in, gateUp);
                activate(n);
                break;
        }
        if (subNormWeight != null) {
            CpuOps.norm(NormKind.RMS, layout.subNormEps(), act, 0, act, 0, n, nFf, subNormWeight, null);
        }
        down.run(n, act, out);
    }

    private void sparsity(int n) {
        float k = layout.sparsity();
        if (k > 0f) {
            CpuActivationSparsity.gaussianTopK(n, layout.nFf(), layout.gateStride(), 0, k, gateUp);
        }
    }

    /** Unary activation in place on the gate (sequential) or up (ungated) rows, then the GELU activation scales. */
    private void activate(int n) {
        int nFf = layout.nFf();
        CpuOps.unary(layout.act(), layout.params(), gateUp, 0, gateUp, 0, n * nFf);
        if (actScales != null) {
            final float[] x = gateUp;
            final float[] s = actScales;
            CpuParallel.parallelFor(n, Math.max(1, DIVIDE_GRAIN_ELEMENTS / nFf), (from, to) -> {
                for (int r = from; r < to; r++) {
                    int base = r * nFf;
                    for (int c = 0; c < nFf; c++) {
                        x[base + c] = x[base + c] / s[c];
                    }
                }
            });
        }
    }

    /** Array holding the activated {@code [n][n_ff]} rows from index 0. */
    private float[] activated() {
        return actTemp != null ? actTemp : gateUp;
    }

    // ------------------------------------------------------------------ helpers

    /**
     * One projection with its deltas, optional bias and the weight's output scale. The scale is folded into the
     * matvec unless a bias or a delta must come before it. Immutable.
     */
    private static final class Projection {
        private final CpuTensorRef weight;
        private final int rows;
        private final int cols;
        private final CpuMatVec scaled;
        private final CpuMatVec unscaled;
        private final float[] bias;
        private final float scale;
        private final CpuProjectionDeltas.Site deltas;

        Projection(CpuTensorRef weight, int rows, CpuMatVec scaled, CpuMatVec unscaled, float[] bias, float scale,
                   CpuProjectionDeltas.Site deltas) {
            this.weight = weight;
            this.rows = rows;
            this.cols = (int) weight.cols();
            this.scaled = scaled;
            this.unscaled = unscaled;
            this.bias = bias;
            this.scale = scale;
            this.deltas = Objects.requireNonNull(deltas, "deltas");
        }

        /** {@code y[t*rows + r] = (W x_t + delta(x_t) + b)[r] * s} for {@code t < n}, rows dense from index 0. */
        void run(int n, float[] x, float[] y) {
            boolean hasDeltas = !deltas.isEmpty();
            boolean postScale = scale != 1.0f && (bias != null || hasDeltas);
            (postScale ? unscaled : scaled).matmul(weight, x, 0, y, 0, n);
            if (hasDeltas) {
                deltas.apply(n, x, 0, cols, y, 0, rows);
            }
            if (bias != null) {
                CpuOps.addRowBias(y, 0, n, rows, bias, 0);
            }
            if (postScale) {
                CpuOps.scale(y, 0, n * rows, scale);
            }
        }
    }

    private float[] require(float[] slot, int width, String name) {
        if (slot.length < rows(width)) {
            throw new IllegalStateException(where + ": scratch slot " + name + " holds " + slot.length
                    + " floats, " + rows(width) + " are needed");
        }
        return slot;
    }

    private int rows(int width) {
        long n = (long) ubatch * width;
        if (n > Integer.MAX_VALUE) {
            throw new IllegalStateException(where + ": " + ubatch + " rows of " + width + " floats exceed one array");
        }
        return (int) n;
    }

    private void checkVector(float[] v, int n, String name) {
        if (v.length != n) {
            throw new IllegalStateException(where + ": " + name + " holds " + v.length + " values, " + n
                    + " are needed");
        }
    }

    private void checkLength(String what, float[] a, long needed) {
        if (a.length < needed) {
            throw new IllegalArgumentException(where + ": FFN " + what + " array holds " + a.length + " floats, "
                    + needed + " are needed");
        }
    }

    private static long[] shape(CpuTensorRef w) {
        return new long[]{w.cols(), w.rows()};
    }

    /** Drops the bound state; the scratch arrays belong to the context. Idempotent. */
    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuDenseFfn{" + (layout != null ? where + ", " + layout : "unbound") + (closed ? ", closed" : "") + '}';
    }
}
