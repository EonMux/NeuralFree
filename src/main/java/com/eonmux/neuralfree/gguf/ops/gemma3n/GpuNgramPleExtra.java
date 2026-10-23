package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nBlocks.NgramLayout;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The qwen4exp n-gram per-layer embedding on the GPU ({@code LayerSpec.Extras.NGRAM_PLE}, qwen4exp.cpp:1203-1420),
 * applied in {@link #beforeBlock} of the PLE layer to the hyper-connection stream {@code x} ({@code hc} streams of
 * {@code n_embd}, {@code D = hc*n_embd} per token) before the attention collapse (qwen4exp.cpp:452-461):
 * <pre>
 *   emb   = per_layer_token_embd[hash rows]                       ([n_heads][head_dim] = n_embd per token)
 *   key   = gn(ple_key . emb, ple_norm_key),  query = gn(x, ple_norm_query)     gn = rms per stream * w[hc][n_embd]
 *   s     = sum_i(key * query) / sqrt(n_embd)                      (per stream)
 *   gated = (ple_value . emb) * sigmoid(sgn(s) * sqrt(clamp(|s|, 1e-6, inf)))  (value broadcast over the streams)
 *   conv  = silu(sum_k ple_conv1d[c][k] * concat(state, gn(gated, ple_norm_conv))[t + k*ngram][c])
 *   x     = x + (gated + conv)
 * </pre>
 * The row ids are hashed on the host ({@link NgramLayout#hash}) from the step's tokens
 * ({@link GpuPleExtra.StepTokens}) and their {@code ngram-1} predecessors from the memory's {@link SequenceState}
 * (llama.cpp reads both from the KV cells after {@code apply_ubatch}).
 *
 * <p>Kernel mapping: the per-stream dot product is a projection of {@code key*query} by a ones row (scaled by
 * {@code 1/sqrt(n_embd)}) over {@code T*hc} rows; {@code sgn(s)*sqrt(clamp(|s|, 1e-6))} is computed exactly with
 * power-of-two scalings: {@code c = clamp(s*2^254, -2^26, 2^26) = sgn(s)*2^26} (every non-zero float saturates),
 * {@code a = sqrt(max(c*s, 1e-6*2^26)) = sqrt(clamp(|s|, 1e-6))*2^13} ({@code SQRT_SOFTPLUS} is the identity root
 * above 20), {@code (c*2^-39)*a}; the sigmoid and the broadcast multiply are one {@code LLM_HC_Post} (post act
 * sigmoid, scale and magnitude 1) onto a zeroed buffer; the dilated causal convolution is {@code LLM_SSM_Conv}
 * (fused SiLU, in-place {@code [D][hist]} state update) with the taps spread to a dense kernel of {@code hist+1}
 * where tap {@code k} sits at {@code k*ngram} and the others are zero, which adds exact zeros to the ascending tap
 * sum. The ones row is built at bind into a temporary only this extra uses; the dense kernel of each PLE layer
 * lives in a temporary named after the layer and is built at bind from resident weights, or before every step from
 * streamed weights (whose slot buffers hold the layer's data only after the executor's {@code await}). A step
 * re-sets token counts and work sizes only. Not thread-safe.</p>
 */
public final class GpuNgramPleExtra implements GpuLayerExtra {

    static final String TEMP_IDS = "gemma3n.ngram.ids";
    static final String TEMP_EMB = "gemma3n.ngram.emb";
    static final String TEMP_KEY = "gemma3n.ngram.key";
    static final String TEMP_VALUE = "gemma3n.ngram.value";
    static final String TEMP_QUERY = "gemma3n.ngram.query";
    static final String TEMP_SCORE = "gemma3n.ngram.score";
    static final String TEMP_SIGN = "gemma3n.ngram.sign";
    static final String TEMP_GATED = "gemma3n.ngram.gated";
    static final String TEMP_NORMED = "gemma3n.ngram.normed";
    static final String TEMP_CONV = "gemma3n.ngram.conv";
    static final String TEMP_ONES = "gemma3n.ngram.ones";
    /** Prefix of the per-layer dense convolution kernels: {@code TEMP_KERNEL + "." + il}. */
    static final String TEMP_KERNEL = "gemma3n.ngram.kernel";
    static final String TEMP_TAPS = "gemma3n.ngram.taps";
    static final String TEMP_CHANNELS = "gemma3n.ngram.channels";

    /** {@code 2^127}: two scalings saturate every non-zero float past the clamp bound. */
    static final float SIGN_SCALE = 0x1p127f;
    /** {@code 2^26}: the sign magnitude after the clamp. */
    static final float SIGN_BOUND = 0x1p26f;
    /** {@code 2^-39}: turns {@code sgn*2^26} into {@code sgn*2^-13}, undoing the {@code 2^13} of the root. */
    static final float SIGN_UNSCALE = 0x1p-39f;
    /** Lower clamp of {@code |s|} (qwen4exp.cpp:1357). */
    static final float MIN_MAGNITUDE = 1e-6f;

    private static final ActParams SATURATE = ActParams.builder().scaleBias(SIGN_SCALE, 0f).build();
    private static final ActParams SIGN_CLAMP = ActParams.builder().clamp(-SIGN_BOUND, SIGN_BOUND).build();
    private static final ActParams MAGNITUDE_CLAMP = ActParams.builder()
            .clamp(MIN_MAGNITUDE * SIGN_BOUND, Float.POSITIVE_INFINITY).build();
    private static final ActParams UNSCALE = ActParams.builder().scaleBias(SIGN_UNSCALE, 0f).build();

    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private GpuForwardContext ctx;
    private GpuOps ops;
    private NgramLayout layout;
    private int il;
    private int ubatch;
    private int width;
    private GpuWeight key;
    private GpuWeight value;
    /** LoRA deltas of {@code ple_key} / {@code ple_value} ({@code build_lora_mm}, qwen4exp.cpp:1388-1389). */
    private GpuProjectionDeltas.Site keyDeltas;
    private GpuProjectionDeltas.Site valueDeltas;
    private GpuWeight normKey;
    private GpuWeight normQuery;
    private GpuWeight normConv;
    private GpuWeight onesRow;
    private GpuPleExtra.RowLookup lookup;
    private GpuPleExtra.StepTokens stepTokens;
    private GpuMemory memory;
    private GpuWeight convTaps;
    private ScratchSlot dense;
    private ScratchSlot tapRows;
    private GpuPleExtra.RowLookup channelLookup;
    private int[] channels;
    private boolean rebuildKernel;
    private ScratchSlot emb;
    private ScratchSlot keys;
    private ScratchSlot values;
    private ScratchSlot query;
    private ScratchSlot score;
    private ScratchSlot sign;
    private ScratchSlot gated;
    private ScratchSlot normed;
    private ScratchSlot conv;
    private PreparedDispatch post;
    private PreparedDispatch convolve;
    private int[] window;
    private int[] ids;
    private int local;

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (ops != null) {
            throw new IllegalStateException("n-gram PLE extra is already bound");
        }
        NgramLayout l = NgramLayout.resolve(ctx.spec(), layer);
        GpuRecurrentState state = convolutionState(ctx, layer, l);
        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        this.ctx = ctx;
        this.ops = ctx.ops();
        this.layout = l;
        this.il = layer.il();
        this.ubatch = ctx.scratch().ubatch();
        this.width = l.streamWidth();
        this.local = ops.local();
        try {
            bindWeights(weights, layer);
            bindScratch();
            prepareDispatches(weights, layer, state);
        } catch (RuntimeException ex) {
            close();
            this.ops = null;
            throw ex;
        }
    }

    /**
     * The layer's convolution-history state, checked to hold {@code (K-1) * ngram * hc * n_embd} floats.
     *
     * @throws UnsupportedOperationException when the convolution has fewer than 2 taps
     * @throws IllegalStateException         when the model has no memory or the state is missing or mis-sized
     */
    private static GpuRecurrentState convolutionState(GpuForwardContext ctx, LayerSpec layer, NgramLayout l) {
        String arch = ctx.spec().arch().ggufName();
        if (l.kernel() < 2) {
            throw new UnsupportedOperationException(arch + ": a PLE convolution of " + l.kernel()
                    + " tap keeps no history; the n-gram PLE needs ple_conv_kernel >= 2");
        }
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(arch + " layer " + layer.il() + ": the n-gram PLE needs the model memory"
                    + " (convolution history and token record)");
        }
        GpuRecurrentState state = ctx.memory().recurrent(layer.il());
        if (state == null) {
            throw new IllegalStateException(arch + " layer " + layer.il() + " has no recurrent state for the PLE"
                    + " convolution history");
        }
        long expected = (long) l.history() * l.streamWidth();
        long stateFloats = stateFloats(state, arch, layer.il());
        if (stateFloats != expected) {
            throw new IllegalStateException(arch + " layer " + layer.il() + ": state " + LayerMemoryPlan.STATE_PLE_CONV
                    + " holds " + stateFloats + " floats, expected (K-1)*ngram*hc*n_embd = " + expected);
        }
        return state;
    }

    /** The key / value projections and their LoRA sites, the grouped norms and the per-layer embedding lookup. */
    private void bindWeights(GpuWeights weights, LayerSpec layer) {
        NgramLayout l = layout;
        int e = l.nEmbd();
        String arch = ctx.spec().arch().ggufName();
        this.key = GpuPleExtra.resident(weights, layer.tensorName(TensorNames.weight(TensorNames.PLE_KEY)), e, width);
        this.value = GpuPleExtra.resident(weights, layer.tensorName(TensorNames.weight(TensorNames.PLE_VALUE)), e, e);
        this.keyDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.PLE_KEY), key);
        this.valueDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.PLE_VALUE), value);
        this.normKey = groupedNorm(weights, layer, TensorNames.PLE_NORM_KEY);
        this.normQuery = groupedNorm(weights, layer, TensorNames.PLE_NORM_QUERY);
        this.normConv = groupedNorm(weights, layer, TensorNames.PLE_NORM_CONV);
        GpuWeight table = weights.get(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD));
        if (table.cols() != l.headDim()) {
            throw new IllegalArgumentException(arch + ": " + table.name() + " is resident as " + table
                    + ", expected rows of ple_head_dim " + l.headDim());
        }
        this.lookup = GpuPleExtra.RowLookup.forHostIds(ctx, table, TEMP_IDS, Math.multiplyExact(ubatch, l.nHeads()));
        this.stepTokens = new GpuPleExtra.StepTokens(ctx);
        this.memory = ctx.memory();
    }

    /** Host id windows and the scratch temporaries of one ubatch. */
    private void bindScratch() {
        NgramLayout l = layout;
        GpuScratch scratch = ctx.scratch();
        int u = ubatch;
        int d = width;
        int e = l.nEmbd();
        this.window = new int[u + l.lookBehind()];
        this.ids = new int[Math.multiplyExact(u, l.nHeads())];
        this.emb = scratch.temp(TEMP_EMB, (long) u * e);
        this.keys = scratch.temp(TEMP_KEY, (long) u * d);
        this.values = scratch.temp(TEMP_VALUE, (long) u * e);
        this.query = scratch.temp(TEMP_QUERY, (long) u * d);
        this.score = scratch.temp(TEMP_SCORE, (long) u * l.hc());
        this.sign = scratch.temp(TEMP_SIGN, (long) u * l.hc());
        this.gated = scratch.temp(TEMP_GATED, (long) u * d);
        this.normed = scratch.temp(TEMP_NORMED, (long) u * d);
        this.conv = scratch.temp(TEMP_CONV, (long) u * d);
    }

    /** The ones row, the dense convolution kernel, the residual post-mix and the causal convolution. */
    private void prepareDispatches(GpuWeights weights, LayerSpec layer, GpuRecurrentState state) {
        NgramLayout l = layout;
        int u = ubatch;
        int d = width;
        int e = l.nEmbd();
        int hist = l.history();
        this.onesRow = buildOnes(e, d);
        prepareDenseKernel(weights, layer, d);
        this.rebuildKernel = weights != ctx.weights();
        if (!rebuildKernel) {
            buildDenseKernel();
        }
        this.post = prepare(HcKernels.HC_POST, new long[]{HcKernels.elementGlobal(u, e, local)}, local,
                postArgs(u, e, l.hc()));
        Object[] convArgs = {d, hist + 1, u, SsmKernels.CONV_FLAG_SILU, d, normed.offset(), d, conv.offset(), 0,
            hist * d, normed.buffer(), dense.buffer(), ctx.placeholders().readOnlyFloats(),
            state.tensor(LayerMemoryPlan.STATE_PLE_CONV), memory.stepInputs().stateRows(), conv.buffer()};
        SsmKernels.checkConvShape(d, hist + 1);
        if (convArgs.length != SsmKernels.CONV_ARG_COUNT) {
            throw new IllegalStateException(SsmKernels.SSM_CONV + " takes " + SsmKernels.CONV_ARG_COUNT
                    + " arguments, built " + convArgs.length);
        }
        this.convolve = prepare(SsmKernels.SSM_CONV, new long[]{SsmKernels.elementwiseGlobal(d,
                SsmKernels.ELEMENTWISE_LOCAL)}, SsmKernels.ELEMENTWISE_LOCAL, convArgs);
    }

    private static long stateFloats(GpuRecurrentState state, String arch, int il) {
        for (LayerMemoryPlan.StateTensor t : state.tensors()) {
            if (LayerMemoryPlan.STATE_PLE_CONV.equals(t.name())) {
                return t.floats();
            }
        }
        throw new IllegalStateException(arch + " layer " + il + " has no '" + LayerMemoryPlan.STATE_PLE_CONV
                + "' state tensor");
    }

    /** A {@code [hc][n_embd]} view of a grouped-norm weight of {@code D} floats, for row norms over streams. */
    private GpuWeight groupedNorm(GpuWeights weights, LayerSpec layer, String base) {
        String name = layer.tensorName(TensorNames.weight(base));
        GpuWeight w = GpuPleExtra.residentVector(weights, name, width);
        return GpuWeight.of(name, GGMLType.F32, new long[]{layout.nEmbd(), layout.hc()}, w.shard(0));
    }

    /**
     * Fills the ones temporary ({@code max(n_embd, D)} floats of 1) and returns it as the {@code {n_embd, 1}} F32
     * row of the per-stream sum, scaled by {@code 1/sqrt(n_embd)}.
     */
    private GpuWeight buildOnes(int e, int d) {
        int n = Math.max(e, d);
        ScratchSlot ones = ctx.scratch().temp(TEMP_ONES, n);
        requireOrigin(ones);
        ops.fill(ones.buffer(), 0L, (long) n * Float.BYTES, Float.floatToIntBits(1.0f));
        return GpuWeight.of(TEMP_ONES, GGMLType.F32, new long[]{e, 1}, ones.buffer()).withScale(layout.scoreScale());
    }

    /**
     * Reserves the dense causal kernel {@code w'[c][j]} ({@code j <= hist}) of this layer and the sources
     * {@link #buildDenseKernel} reads: {@code ple_conv1d} directly when it is one F32 buffer, else its rows looked up
     * (converted to F32 as {@code ggml_cast} does, qwen4exp.cpp:1451-1455) into {@link #TEMP_TAPS}.
     */
    private void prepareDenseKernel(GpuWeights weights, LayerSpec layer, int d) {
        int k = layout.kernel();
        this.convTaps = GpuPleExtra.resident(weights, layer.tensorName(TensorNames.weight(TensorNames.PLE_CONV1D)),
                k, d);
        this.dense = ctx.scratch().temp(TEMP_KERNEL + "." + layer.il(), (long) d * (layout.history() + 1));
        requireOrigin(dense);
        if (convTaps.type() != GGMLType.F32 || !convTaps.isSingleShard()) {
            this.tapRows = ctx.scratch().temp(TEMP_TAPS, (long) d * k);
            requireOrigin(tapRows);
            this.channels = new int[d];
            for (int c = 0; c < d; c++) {
                channels[c] = c;
            }
            this.channelLookup = GpuPleExtra.RowLookup.forHostIds(ctx, convTaps, TEMP_CHANNELS, d);
        }
    }

    /** Fills the dense kernel: {@code w'[c][k*ngram] = ple_conv1d[c][k]}, zero elsewhere. */
    private void buildDenseKernel() {
        int k = layout.kernel();
        int d = width;
        int taps = layout.history() + 1;
        ops.fill(dense.buffer(), 0L, (long) d * taps * Float.BYTES, 0);
        GpuBuffer src;
        if (channelLookup == null) {
            src = convTaps.shard(0);
        } else {
            channelLookup.rows(channels, d, tapRows.buffer(), 0);
            src = tapRows.buffer();
        }
        for (int j = 0; j < k; j++) {
            ops.copy2d(src, j, k, dense.buffer(), j * layout.dilation(), taps, d, 1);
        }
    }

    private static void requireOrigin(ScratchSlot s) {
        if (s.offset() != 0) {
            throw new IllegalStateException("temporary " + s.name() + " must start at element 0 of its buffer");
        }
    }

    private Object[] postArgs(int nTokens, int e, int hc) {
        Object[] a = new Object[HcKernels.POST_ARG_COUNT];
        a[HcKernels.POST_ARG_N_TOKENS] = nTokens;
        a[HcKernels.POST_ARG_N_EMBD] = e;
        a[HcKernels.POST_ARG_HC] = hc;
        a[HcKernels.POST_ARG_HAS_COMB] = 0;
        a[HcKernels.POST_ARG_POST_ACT] = HcKernels.POST_ACT_SIGMOID;
        a[HcKernels.POST_ARG_POST_IN_SCALE] = 1.0f;
        a[HcKernels.POST_ARG_POST_MAG] = 1.0f;
        a[HcKernels.POST_ARG_Y_OFF] = values.offset();
        a[HcKernels.POST_ARG_X_OFF] = gated.offset();
        a[HcKernels.POST_ARG_POST_OFF] = sign.offset();
        a[HcKernels.POST_ARG_COMB_OFF] = 0;
        a[HcKernels.POST_ARG_XOUT_OFF] = gated.offset();
        a[HcKernels.POST_ARG_Y] = values.buffer();
        a[HcKernels.POST_ARG_X] = gated.buffer();
        a[HcKernels.POST_ARG_POST] = sign.buffer();
        a[HcKernels.POST_ARG_COMB] = ctx.placeholders().readOnlyFloats();
        a[HcKernels.POST_ARG_XOUT] = gated.buffer();
        return a;
    }

    private PreparedDispatch prepare(String kernel, long[] global, int l, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, new long[]{l}, args);
        prepared.add(d);
        return d;
    }

    @Override
    public void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
        if (ops == null) {
            throw new IllegalStateException("n-gram PLE extra is not bound");
        }
        GpuPleExtra.checkTokens(nTokens, ubatch, "n-gram PLE");
        int d = width;
        int e = layout.nEmbd();
        int hc = layout.hc();
        int n = nTokens * d;
        Objects.requireNonNull(x, "x").require(n);
        if (rebuildKernel) {
            buildDenseKernel();
        }
        fillWindow(pos0, nTokens);
        layout.hash(window, nTokens, ids);
        lookup.rows(ids, nTokens * layout.nHeads(), emb.buffer(), emb.offset());
        ops.matmul(key, emb.buffer(), emb.offset(), keys.buffer(), keys.offset(), nTokens);
        keyDeltas.applyRows(nTokens, emb.buffer(), emb.offset(), keys.buffer(), keys.offset());
        ops.matmul(value, emb.buffer(), emb.offset(), values.buffer(), values.offset(), nTokens);
        valueDeltas.applyRows(nTokens, emb.buffer(), emb.offset(), values.buffer(), values.offset());
        int rows = nTokens * hc;
        ops.norm(NormKind.RMS, normKey, null, layout.eps(), keys.buffer(), keys.offset(), keys.buffer(),
                keys.offset(), rows, e);
        ops.norm(NormKind.RMS, normQuery, null, layout.eps(), x.buffer(), x.offset(), query.buffer(), query.offset(),
                rows, e);
        ops.glu(GluOp.MUL, ActParams.NONE, keys.buffer(), keys.offset(), d, query.buffer(), query.offset(), d,
                keys.buffer(), keys.offset(), d, nTokens, d);
        ops.matvec(onesRow, keys.buffer(), keys.offset(), score.buffer(), score.offset(), rows);
        gate(rows);
        ops.fill(gated.buffer(), (long) gated.offset() * Float.BYTES, (long) n * Float.BYTES, 0);
        post.setInt(HcKernels.POST_ARG_N_TOKENS, nTokens);
        post.setGlobal(HcKernels.elementGlobal(nTokens, e, local));
        post.dispatch();
        ops.norm(NormKind.RMS, normConv, null, layout.eps(), gated.buffer(), gated.offset(), normed.buffer(),
                normed.offset(), rows, e);
        convolve.setInt(SsmKernels.ARG_CONV_N_TOKENS, nTokens);
        convolve.dispatch();
        ops.add(gated.buffer(), gated.offset(), conv.buffer(), conv.offset(), conv.buffer(), conv.offset(), n);
        ops.add(x.buffer(), x.offset(), conv.buffer(), conv.offset(), x.buffer(), x.offset(), n);
    }

    /**
     * {@code score = sgn(s) * sqrt(clamp(|s|, 1e-6, inf))} in place over {@code rows} scores, exactly (class
     * comment); the sigmoid is applied by {@code LLM_HC_Post}.
     */
    private void gate(int rows) {
        GpuBuffer s = score.buffer();
        int sOff = score.offset();
        GpuBuffer c = sign.buffer();
        int cOff = sign.offset();
        ops.unary(ActOp.SCALE_BIAS, SATURATE, s, sOff, c, cOff, rows);
        ops.unary(ActOp.SCALE_BIAS, SATURATE, c, cOff, c, cOff, rows);
        ops.unary(ActOp.CLAMP, SIGN_CLAMP, c, cOff, c, cOff, rows);
        ops.glu(GluOp.MUL, ActParams.NONE, c, cOff, rows, s, sOff, rows, s, sOff, rows, 1, rows);
        ops.unary(ActOp.CLAMP, MAGNITUDE_CLAMP, s, sOff, s, sOff, rows);
        ops.unary(ActOp.SQRT_SOFTPLUS, ActParams.NONE, s, sOff, s, sOff, rows);
        ops.unary(ActOp.SCALE_BIAS, UNSCALE, c, cOff, c, cOff, rows);
        ops.glu(GluOp.MUL, ActParams.NONE, c, cOff, rows, s, sOff, rows, c, cOff, rows, 1, rows);
    }

    /**
     * Tokens of positions {@code [pos0 - (ngram-1), pos0 + nTokens)} into {@link #window}: the predecessors from the
     * sequence record (-1 before position 0), the step's own tokens from {@link GpuPleExtra.StepTokens}.
     *
     * @throws IllegalStateException when the record does not reach {@code pos0}
     */
    private void fillWindow(int pos0, int nTokens) {
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
        stepTokens.read(nTokens, window, back);
    }

    @Override
    public void close() {
        RuntimeException failure = Closeables.closeAll(prepared, null);
        prepared.clear();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuNgramPleExtra{" + (layout == null ? "unbound" : "layer " + il + ", " + layout) + '}';
    }
}
