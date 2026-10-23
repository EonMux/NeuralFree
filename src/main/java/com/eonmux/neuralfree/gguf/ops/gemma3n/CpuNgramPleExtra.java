package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nBlocks.NgramLayout;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.ssm.CpuSsmConv;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU twin of {@link GpuNgramPleExtra}: the qwen4exp n-gram per-layer embedding ({@code LayerSpec.Extras.NGRAM_PLE},
 * qwen4exp.cpp:1203-1420) in {@link #beforeBlock} of the PLE layer. The step's tokens come from
 * {@link CpuScratch#TOKEN_IDS}, their predecessors from the memory's {@link SequenceState}; the score is the
 * {@code ggml_sum_rows} of the products in double precision ({@code ggml_vec_sum_f32}), the gate
 * {@code sigmoid(sgn(s) * sqrtf(clamp(|s|, 1e-6, inf)))} element-wise, and the dilated causal convolution
 * {@link CpuSsmConv#forward} over the same dense kernel as the GPU form (tap {@code k} at {@code k*ngram}, zeros
 * between), updating the {@code [D][hist]} state in place. Working arrays are reserved at bind. Not thread-safe.
 */
public final class CpuNgramPleExtra implements CpuLayerExtra {

    private static final int ROW_GRAIN = 16;

    private CpuMatVec matVec;
    private CpuWeights weights;
    private CpuScratch scratch;
    private CpuMemory memory;
    private NgramLayout layout;
    private int il;
    private int ubatch;
    private int width;
    private CpuTensorRef table;
    private CpuTensorRef key;
    private CpuTensorRef value;
    /** LoRA deltas of {@code ple_key} / {@code ple_value} ({@code build_lora_mm}, qwen4exp.cpp:1388-1389). */
    private CpuProjectionDeltas.Site keyDeltas;
    private CpuProjectionDeltas.Site valueDeltas;
    private float[] normKey;
    private float[] normQuery;
    private float[] normConv;
    private float[] dense;
    private float[] state;
    private int[] window;
    private int[] ids;
    private float[] emb;
    private float[] keys;
    private float[] values;
    private float[] query;
    private float[] gates;
    private float[] gated;
    private float[] normed;
    private float[] conv;

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (matVec != null) {
            throw new IllegalStateException("n-gram PLE extra is already bound");
        }
        ModelSpec spec = ctx.spec();
        String arch = spec.arch().ggufName();
        NgramLayout l = NgramLayout.resolve(spec, layer);
        if (l.kernel() < 2) {
            throw new UnsupportedOperationException(arch + ": a PLE convolution of " + l.kernel()
                    + " tap keeps no history; the n-gram PLE needs ple_conv_kernel >= 2");
        }
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(arch + " layer " + layer.il() + ": the n-gram PLE needs the model memory"
                    + " (convolution history and token record)");
        }
        CpuMemory mem = ctx.memory();
        RecurrentStateAccessor rs = mem.recurrent(layer.il());
        if (rs == null) {
            throw new IllegalStateException(arch + " layer " + layer.il() + " has no recurrent state for the PLE"
                    + " convolution history");
        }
        int d = l.streamWidth();
        int e = l.nEmbd();
        int hist = l.history();
        float[] st = rs.tensor(LayerMemoryPlan.STATE_PLE_CONV);
        if (st.length != (long) hist * d) {
            throw new IllegalStateException(arch + " layer " + layer.il() + ": state " + LayerMemoryPlan.STATE_PLE_CONV
                    + " holds " + st.length + " floats, expected (K-1)*ngram*hc*n_embd = " + ((long) hist * d));
        }
        CpuWeights w = ctx.weights();
        CpuScratch s = ctx.scratch();
        int u = s.ubatch();
        CpuTensorRef tbl = w.get(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD));
        if (tbl.cols() != l.headDim()) {
            throw new IllegalArgumentException(arch + ": " + TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD)
                    + " rows hold " + tbl.cols() + " floats, expected ple_head_dim " + l.headDim());
        }
        this.key = CpuPleExtra.matrix(w, layer.tensorName(TensorNames.weight(TensorNames.PLE_KEY)), e, d);
        this.value = CpuPleExtra.matrix(w, layer.tensorName(TensorNames.weight(TensorNames.PLE_VALUE)), e, e);
        this.keyDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.PLE_KEY), key,
                ctx.matVec());
        this.valueDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.PLE_VALUE), value,
                ctx.matVec());
        this.normKey = CpuPleExtra.vector(w, layer.tensorName(TensorNames.weight(TensorNames.PLE_NORM_KEY)), d);
        this.normQuery = CpuPleExtra.vector(w, layer.tensorName(TensorNames.weight(TensorNames.PLE_NORM_QUERY)), d);
        this.normConv = CpuPleExtra.vector(w, layer.tensorName(TensorNames.weight(TensorNames.PLE_NORM_CONV)), d);
        this.dense = denseKernel(CpuPleExtra.vector(w, layer.tensorName(TensorNames.weight(TensorNames.PLE_CONV1D)),
                Math.multiplyExact(d, l.kernel())), l, d);
        this.table = tbl;
        this.layout = l;
        this.il = layer.il();
        this.ubatch = u;
        this.width = d;
        this.state = st;
        this.window = new int[u + l.lookBehind()];
        this.ids = new int[Math.multiplyExact(u, l.nHeads())];
        this.emb = s.temp(GpuNgramPleExtra.TEMP_EMB, u * e);
        this.keys = s.temp(GpuNgramPleExtra.TEMP_KEY, u * d);
        this.values = s.temp(GpuNgramPleExtra.TEMP_VALUE, u * e);
        this.query = s.temp(GpuNgramPleExtra.TEMP_QUERY, u * d);
        this.gates = s.temp(GpuNgramPleExtra.TEMP_SCORE, u * l.hc());
        this.gated = s.temp(GpuNgramPleExtra.TEMP_GATED, u * d);
        this.normed = s.temp(GpuNgramPleExtra.TEMP_NORMED, u * d);
        this.conv = s.temp(GpuNgramPleExtra.TEMP_CONV, u * d);
        this.weights = w;
        this.scratch = s;
        this.memory = mem;
        this.matVec = ctx.matVec();
    }

    /** {@code w'[c][k*ngram] = w[c][k]} over {@code hist+1} taps per channel, zero elsewhere. */
    private static float[] denseKernel(float[] w, NgramLayout l, int d) {
        int k = l.kernel();
        int taps = l.history() + 1;
        float[] out = new float[Math.multiplyExact(d, taps)];
        for (int c = 0; c < d; c++) {
            for (int j = 0; j < k; j++) {
                out[c * taps + j * l.dilation()] = w[c * k + j];
            }
        }
        return out;
    }

    @Override
    public void beforeBlock(int pos0, int n, float[] x) {
        if (matVec == null) {
            throw new IllegalStateException("n-gram PLE extra is not bound");
        }
        GpuPleExtra.checkTokens(n, ubatch, "n-gram PLE");
        int d = width;
        int e = layout.nEmbd();
        int hc = layout.hc();
        int len = n * d;
        CpuPleExtra.requireLength(x, len, "residual stream");
        fillWindow(pos0, n);
        layout.hash(window, n, ids);
        int nIds = n * layout.nHeads();
        int hd = layout.headDim();
        for (int i = 0; i < nIds; i++) {
            weights.embedRow(table, ids[i], emb, i * hd);
        }
        matVec.matmul(key, emb, 0, keys, 0, n);
        keyDeltas.applyRows(n, emb, 0, keys, 0);
        matVec.matmul(value, emb, 0, values, 0, n);
        valueDeltas.applyRows(n, emb, 0, values, 0);
        int rows = n * hc;
        float eps = layout.eps();
        NormMath.rowNorm(NormKind.RMS, eps, rows, e, e, hc, keys, 0, normKey, null, keys, 0);
        NormMath.rowNorm(NormKind.RMS, eps, rows, e, e, hc, x, 0, normQuery, null, query, 0);
        float scale = layout.scoreScale();
        CpuParallel.parallelFor(rows, ROW_GRAIN, (from, to) -> {
            for (int r = from; r < to; r++) {
                gates[r] = gate(score(keys, query, r * e, e) * scale);
            }
        });
        CpuParallel.parallelFor(n, 1, (from, to) -> {
            for (int t = from; t < to; t++) {
                for (int h = 0; h < hc; h++) {
                    float g = gates[t * hc + h];
                    int o = t * d + h * e;
                    for (int i = 0; i < e; i++) {
                        gated[o + i] = values[t * e + i] * g;
                    }
                }
            }
        });
        NormMath.rowNorm(NormKind.RMS, eps, rows, e, e, hc, gated, 0, normConv, null, normed, 0);
        CpuSsmConv.forward(d, layout.history() + 1, n, SsmKernels.CONV_FLAG_SILU, normed, 0, d, dense, null,
                state, 0, 0, conv, 0, d);
        CpuOps.add(gated, 0, conv, 0, conv, 0, len);
        CpuOps.add(x, 0, conv, 0, x, 0, len);
    }

    /** {@code sum_i (key[i] * query[i])}: products rounded to float, summed in double (ggml_vec_sum_f32). */
    private static float score(float[] k, float[] q, int off, int n) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += k[off + i] * q[off + i];
        }
        return (float) sum;
    }

    /** {@code sigmoid(sgn(s) * sqrtf(clamp(|s|, 1e-6, inf)))} (qwen4exp.cpp:1357-1358). */
    static float gate(float s) {
        float sgn = s > 0f ? 1f : (s < 0f ? -1f : 0f);
        float mag = (float) Math.sqrt(Math.max(Math.abs(s), GpuNgramPleExtra.MIN_MAGNITUDE));
        return ActMath.sigmoid(sgn * mag);
    }

    /**
     * Tokens of positions {@code [pos0 - (ngram-1), pos0 + n)} into {@link #window}: predecessors from the sequence
     * record (-1 before position 0), the step's own tokens from {@link CpuScratch#TOKEN_IDS}.
     *
     * @throws IllegalStateException when the record does not reach {@code pos0}
     */
    private void fillWindow(int pos0, int n) {
        int back = layout.lookBehind();
        SequenceState seq = memory.sequence();
        if (pos0 > 0 && seq.nPast() < pos0) {
            throw new IllegalStateException("n-gram PLE at position " + pos0 + " needs the tokens before it in the "
                    + "sequence record, which holds " + seq.nPast());
        }
        for (int w = 0; w < back; w++) {
            int pos = pos0 - back + w;
            window[w] = pos < 0 ? -1 : seq.token(pos);
        }
        int[] step = scratch.ints(CpuScratch.TOKEN_IDS);
        System.arraycopy(step, 0, window, back, n);
    }

    @Override
    public String toString() {
        return "CpuNgramPleExtra{" + (layout == null ? "unbound" : "layer " + il + ", " + layout) + '}';
    }
}
