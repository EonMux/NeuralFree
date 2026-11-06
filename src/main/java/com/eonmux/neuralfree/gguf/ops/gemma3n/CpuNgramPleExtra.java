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
 * {@link CpuScratch#TOKEN_IDS} (an embedding row hashes the stand-in id), their predecessors from the memory's
 * {@link SequenceState} ({@link NgramLayout#completeWindow}); the score is the
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
    /** Floats of one state slot and the slots the memory keeps ({@code RecurrentStateAccessor}). */
    private int stateRow;
    private int stateSlots;
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
        state = rs.tensor(LayerMemoryPlan.STATE_PLE_CONV);
        stateRow = rs.slotFloats(LayerMemoryPlan.STATE_PLE_CONV);
        if (stateRow != (long) hist * d) {
            throw new IllegalStateException(arch + " layer " + layer.il() + ": state " + LayerMemoryPlan.STATE_PLE_CONV
                    + " holds " + stateRow + " floats per sequence, expected (K-1)*ngram*hc*n_embd = "
                    + ((long) hist * d));
        }
        stateSlots = rs.slots();
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
        this.state = state;
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
        checkBound();
        GpuPleExtra.checkTokens(n, ubatch, "n-gram PLE");
        CpuPleExtra.requireLength(x, (long) n * width, "residual stream");
        runSegment(pos0, n, 0, scratch.ints(CpuScratch.TOKEN_IDS), 0, 0, memory.sequence(), true, x);
    }

    /**
     * One batched step: the score, gate and convolution of every row read that row's predecessors and update that
     * row's sequence's history, so the segments run one at a time, each against its own sequence's state row and
     * token record (the paged memory's per-sequence record, positions per entry). Every segment's rows sit at its
     * {@code tokStart} offset of {@code x}, its tokens at the same offset of {@code CpuScratch#TOKEN_IDS}.
     */
    @Override
    public void beforeBlock(com.eonmux.neuralfree.gguf.memory.StepLayout s, float[] x) {
        checkBound();
        GpuPleExtra.checkTokens(s.nTokens(), ubatch, "n-gram PLE");
        CpuPleExtra.requireLength(x, (long) s.nTokens() * width, "residual stream");
        int[] step = scratch.ints(CpuScratch.TOKEN_IDS);
        boolean paged = memory.pagedSequences() != null;
        for (int seg = 0; seg < s.nSeg(); seg++) {
            int seq = s.seq(seg);
            SequenceState record = paged ? memory.pagedSequences().sequence(seq) : memory.sequence();
            runSegment(s.pos0(seg), s.len(seg), s.tokStart(seg) * width, step, s.tokStart(seg), seq, record, !paged,
                    x);
        }
    }

    private void checkBound() {
        if (matVec == null) {
            throw new IllegalStateException("n-gram PLE extra is not bound");
        }
    }

    /**
     * One segment of {@code n} rows at {@code x[xOff]} whose tokens are {@code step[tokOff ..]}: the predecessors
     * come from sequence {@code seq}'s token record {@code record} ({@code contiguous}: the contiguous record), the
     * convolution updates sequence {@code seq}'s history row.
     */
    private void runSegment(int pos0, int n, int xOff, int[] step, int tokOff, int seq, SequenceState record,
                            boolean contiguous, float[] x) {
        int d = width;
        int e = layout.nEmbd();
        int hc = layout.hc();
        int len = n * d;
        int base = stateSlot(seq) * stateRow;
        System.arraycopy(step, tokOff, window, layout.lookBehind(), n);
        layout.completeWindow(record, contiguous, pos0, n, window);
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
        NormMath.rowNorm(NormKind.RMS, eps, rows, e, e, hc, x, xOff, normQuery, null, query, 0);
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
                state, base, 0, conv, 0, d);
        CpuOps.add(gated, 0, conv, 0, conv, 0, len);
        CpuOps.add(x, xOff, conv, 0, x, xOff, len);
    }

    /** The state slot of {@code seq}, validated against the memory's slots. */
    private int stateSlot(int seq) {
        if (seq < 0 || seq >= stateSlots) {
            throw new UnsupportedOperationException("n-gram PLE runs sequence " + seq + ", the memory keeps "
                    + stateSlots + " state slots");
        }
        return seq;
    }

    /** The hooks run per segment over the real rows, so batching is supported. */
    @Override
    public String batchRefusal() {
        return null;
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

    @Override
    public String toString() {
        return "CpuNgramPleExtra{" + (layout == null ? "unbound" : "layer " + il + ", " + layout) + '}';
    }
}
