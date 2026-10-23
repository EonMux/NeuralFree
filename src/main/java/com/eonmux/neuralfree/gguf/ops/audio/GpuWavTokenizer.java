package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.flow.FlowKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * GPU executor of {@link WavTokenizerDecoder}; {@link CpuWavTokenizer} is its reference twin with the same step
 * order and layouts.
 *
 * <p>{@link #bind} resolves every tensor, converts the tensors the float kernels read ({@code LLM_Conv1dSame},
 * {@code LLM_Conv1dDepthwise}, {@code LLM_GroupNorm}, {@code LLM_RowNorm}, {@code LLM_Batch_BiasAdd},
 * {@code LLM_MulRowScale}) to F32 when the file stores them otherwise, reserves every work buffer from
 * {@link GpuScratch} for the capacity, uploads the key positions of the posnet attention and plans its flash
 * attention. A {@link #decode} call then uploads the codes and three transpose index tables, enqueues the graph and
 * reads back the features with one {@code finish()}. Every kernel geometry depends on the sequence length, so the
 * dispatches are enqueued per call rather than prepared.
 *
 * <p>Per call over {@code T} codes ({@code E} embedding width, {@code C} model width, {@code F} ConvNeXt hidden
 * width, {@code O} feature width; channel-major signals {@code [C][T]}, token-major rows {@code [T][C]}):
 * <ol>
 *   <li>{@link GpuOps#embed} of the codes into {@code emb [T][E]}; {@code LLM_GatherRows} transposes it to
 *       {@code a [E][T]}; {@code LLM_Conv1dSame} K=7 into the residual {@code x [C][T]}.</li>
 *   <li>Posnet ResNet layer: {@code LLM_GroupNorm} x to a, {@code LLM_Unary} SiLU, {@code LLM_Conv1dSame} K=3 a to
 *       b, {@code LLM_GroupNorm} b to a, SiLU, conv a to b, {@link GpuOps#add} {@code x += b}.</li>
 *   <li>Posnet attention layer: {@code LLM_GroupNorm} x to a; gather to {@code tok [T][C]}; three
 *       {@link GpuOps#matmul} of the 1x1 projections viewed as {@code [C][C]} matrices plus
 *       {@code LLM_Batch_BiasAdd} into q, k, v; {@code LLM_FlashAttn_F32} (one head, {@code d = C}, non-causal,
 *       scale {@code 1/sqrt(C)}) into tok, with {@code LLM_FlashAttn_Combine} when the plan splits the keys; output
 *       projection tok to q plus bias; gather to {@code a [C][T]}; {@code x += a}.</li>
 *   <li>Posnet norm layer: {@code LLM_GroupNorm} x to a, {@link GpuOps#copy} back to x.</li>
 *   <li>Token LayerNorm: gather x to tok, {@code LLM_RowNorm} in place, gather back to x.</li>
 *   <li>ConvNeXt layer: {@code LLM_Conv1dDepthwise} x to a; gather to tok; {@code LLM_RowNorm}; {@code pw1}
 *       matmul into {@code ff [T][F]} plus bias; {@code LLM_Unary} GELU (tanh); {@code pw2} matmul into tok plus
 *       bias; {@code LLM_MulRowScale} by gamma; gather to a; {@code x += a}.</li>
 *   <li>Output: gather x to tok, {@code LLM_RowNorm}, {@code output} matmul into {@code out [T][O]} plus bias,
 *       read back.</li>
 * </ol>
 * Not thread-safe; one decode at a time per model scratch.
 */
public final class GpuWavTokenizer implements WavTokenizerDecoder.Backend {

    /** {@code data[s*dim + d] += bias[d]} for {@code s < seq_len} ({@code kernels/llm/eltwise.cl:122}). */
    public static final String BATCH_BIAS_ADD = GpuOps.BATCH_BIAS_ADD;

    /** Every kernel a decode launches, for prewarming. */
    public static final List<String> KERNELS = Collections.unmodifiableList(Arrays.asList(
            AudioKernels.CONV1D_SAME, AudioKernels.CONV1D_DEPTHWISE, NormKernels.GROUP_NORM, NormKernels.ROW_NORM,
            NormKernels.MUL_ROW_SCALE, ActKernels.UNARY, GpuOps.AXPBY, GpuOps.COPY2D, FlowKernels.GATHER_ROWS,
            BATCH_BIAS_ADD, AttnKernels.flashAttn(KvDType.F32), AttnKernels.FLASH_COMBINE));

    /** Splits of the posnet attention the partial buffer holds at full capacity. */
    static final int PARTIAL_SPLITS_AT_CAPACITY = 2;

    private static final String TEMP = "wavtok.";

    /** Resolved device tensors of one posnet layer; unused fields are null for the layer's kind. */
    private static final class PosnetLayer {
        final WavTokenizerDecoder.PosnetKind kind;
        final GpuBuffer normW;
        final GpuBuffer normB;
        final GpuBuffer conv1W;
        final GpuBuffer conv1B;
        final GpuBuffer norm2W;
        final GpuBuffer norm2B;
        final GpuBuffer conv2W;
        final GpuBuffer conv2B;
        final GpuWeight[] proj;
        final GpuBuffer[] projB;

        PosnetLayer(WavTokenizerDecoder.PosnetKind kind, GpuBuffer normW, GpuBuffer normB, GpuBuffer conv1W,
                    GpuBuffer conv1B, GpuBuffer norm2W, GpuBuffer norm2B, GpuBuffer conv2W, GpuBuffer conv2B,
                    GpuWeight[] proj, GpuBuffer[] projB) {
            this.kind = kind;
            this.normW = normW;
            this.normB = normB;
            this.conv1W = conv1W;
            this.conv1B = conv1B;
            this.norm2W = norm2W;
            this.norm2B = norm2B;
            this.conv2W = conv2W;
            this.conv2B = conv2B;
            this.proj = proj;
            this.projB = projB;
        }
    }

    /** Resolved device tensors of one ConvNeXt layer. */
    private static final class ConvnextLayer {
        final GpuBuffer dwW;
        final GpuBuffer dwB;
        final GpuWeight normW;
        final GpuWeight normB;
        final GpuWeight pw1;
        final GpuBuffer pw1B;
        final GpuWeight pw2;
        final GpuBuffer pw2B;
        final GpuBuffer gamma;

        ConvnextLayer(GpuBuffer dwW, GpuBuffer dwB, GpuWeight normW, GpuWeight normB, GpuWeight pw1, GpuBuffer pw1B,
                      GpuWeight pw2, GpuBuffer pw2B, GpuBuffer gamma) {
            this.dwW = dwW;
            this.dwB = dwB;
            this.normW = normW;
            this.normB = normB;
            this.pw1 = pw1;
            this.pw1B = pw1B;
            this.pw2 = pw2;
            this.pw2B = pw2B;
            this.gamma = gamma;
        }
    }

    private final WavTokenizerDecoder dec;
    private final GpuOps ops;
    private final ComputeQueue queue;
    private final GpuScratch scratch;
    private final GpuPlaceholders placeholders;
    private final int capacity;
    private final int local;

    private final GpuWeight tokEmbd;
    private final GpuBuffer convW;
    private final GpuBuffer convB;
    private final List<PosnetLayer> posnet;
    private final GpuWeight tokNormW;
    private final GpuWeight tokNormB;
    private final List<ConvnextLayer> convnext;
    private final GpuWeight outNormW;
    private final GpuWeight outNormB;
    private final GpuWeight output;
    private final GpuBuffer outputB;

    private final ScratchSlot emb;
    private final ScratchSlot x;
    private final ScratchSlot a;
    private final ScratchSlot b;
    private final ScratchSlot tok;
    private final ScratchSlot q;
    private final ScratchSlot k;
    private final ScratchSlot v;
    private final ScratchSlot ff;
    private final ScratchSlot out;
    private final ScratchSlot idxToChannelE;
    private final ScratchSlot idxToToken;
    private final ScratchSlot idxToChannel;
    private final ScratchSlot pos;
    private final FlashAttnTiling plan;
    private final GpuBuffer partials;
    private final long partialFloats;
    private final byte[] hostIndex;
    private boolean closed;

    private GpuWavTokenizer(WavTokenizerDecoder dec, GpuOps ops, GpuWeights weights, int capacity,
                            int computeUnits) {
        this.dec = Objects.requireNonNull(dec, "decoder");
        this.ops = Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(weights, "weights");
        dec.checkCapacity(capacity);
        this.queue = ops.queue();
        this.scratch = ops.scratch();
        this.placeholders = ops.placeholders();
        this.capacity = capacity;
        this.local = ops.local();
        NormKernels.checkReduceLocal(local);
        int c = dec.width();
        int e = dec.nEmbd();

        this.tokEmbd = weights.get(TensorNames.weight(TensorNames.TOKEN_EMBD));
        this.convW = f32(weights.get(TensorNames.weight(TensorNames.CONV1D)));
        this.convB = f32(weights.get(TensorNames.bias(TensorNames.CONV1D)));
        List<PosnetLayer> pn = new ArrayList<>();
        for (int il = 0; il < dec.posnetLayers(); il++) {
            pn.add(posnetLayer(weights, il));
        }
        this.posnet = Collections.unmodifiableList(pn);
        this.tokNormW = f32Vector(weights.get(TensorNames.weight(TensorNames.TOKEN_EMBD_NORM)));
        this.tokNormB = f32Vector(weights.get(TensorNames.bias(TensorNames.TOKEN_EMBD_NORM)));
        List<ConvnextLayer> cn = new ArrayList<>();
        for (int il = 0; il < dec.convnextLayers(); il++) {
            cn.add(convnextLayer(weights, il));
        }
        this.convnext = Collections.unmodifiableList(cn);
        this.outNormW = f32Vector(weights.get(TensorNames.weight(TensorNames.OUTPUT_NORM)));
        this.outNormB = f32Vector(weights.get(TensorNames.bias(TensorNames.OUTPUT_NORM)));
        this.output = weights.get(TensorNames.weight(TensorNames.OUTPUT));
        this.outputB = f32(weights.get(TensorNames.bias(TensorNames.OUTPUT)));

        long cap = capacity;
        this.emb = temp("emb", e * cap);
        this.x = temp("x", c * cap);
        this.a = temp("a", Math.max(e, c) * cap);
        this.b = temp("b", c * cap);
        this.tok = temp("tok", c * cap);
        this.out = temp("out", dec.nOut() * cap);
        this.idxToChannelE = temp("idx.channel_e", e * cap);
        this.idxToToken = temp("idx.token", c * cap);
        this.idxToChannel = temp("idx.channel", c * cap);
        this.ff = dec.convnextLayers() > 0 ? temp("ff", (long) dec.nFf() * cap) : null;
        this.hostIndex = new byte[Math.multiplyExact(Math.max(e, c) * capacity, Integer.BYTES)];

        if (dec.posnetLayers() > WavTokenizerDecoder.POSNET_ATTENTION_LAYER) {
            this.q = temp("q", c * cap);
            this.k = temp("k", c * cap);
            this.v = temp("v", c * cap);
            this.pos = temp("pos", cap);
            int[] positions = new int[capacity];
            for (int t = 0; t < capacity; t++) {
                positions[t] = t;
            }
            writeInts(pos, positions, capacity);
            this.plan = attentionPlan(c, capacity, ops.kernels().faSmemFloats(), computeUnits);
            this.partialFloats = FlashAttnTiling.partialFloats(cap, PARTIAL_SPLITS_AT_CAPACITY, c);
            this.partials = scratch.partials(Math.toIntExact(partialFloats));
        } else {
            this.q = null;
            this.k = null;
            this.v = null;
            this.pos = null;
            this.plan = null;
            this.partials = null;
            this.partialFloats = 0;
        }
        prewarmKernels();
    }

    /**
     * Prepares each kernel a decode launches at the work-group size it launches it with ({@link GpuOps#prewarm}),
     * so no pipeline is built during the first decode.
     */
    private void prewarmKernels() {
        long[] row = {local};
        long[] conv = {AudioKernels.CONV_LOCAL};
        ops.prewarm(AudioKernels.CONV1D_SAME, conv);
        ops.prewarm(FlowKernels.GATHER_ROWS, new long[]{FlowKernels.GATHER_LOCAL});
        ops.prewarm(BATCH_BIAS_ADD, row);
        if (dec.posnetLayers() > 0) {
            ops.prewarm(NormKernels.GROUP_NORM, row);
        }
        if (dec.convnextLayers() > 0) {
            ops.prewarm(AudioKernels.CONV1D_DEPTHWISE, conv);
            ops.prewarm(NormKernels.MUL_ROW_SCALE, row);
        }
        if (plan != null) {
            ops.prewarm(AttnKernels.flashAttn(KvDType.F32), FlashAttnTiling.prefillLocal());
            ops.prewarm(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineLocal());
        }
    }

    /** Binds to a model's GPU context with the scratch ubatch as capacity. */
    public static GpuWavTokenizer bind(GpuForwardContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        return bind(WavTokenizerDecoder.of(ctx.spec()), ctx.ops(), ctx.weights(), ctx.scratch().ubatch());
    }

    /**
     * Binds to explicit device weights.
     *
     * @param capacity largest code sequence per call
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is missing
     * @throws UnsupportedOperationException when no flash-attention tiling fits the posnet width on this device
     */
    public static GpuWavTokenizer bind(WavTokenizerDecoder dec, GpuOps ops, GpuWeights weights, int capacity) {
        Objects.requireNonNull(ops, "ops");
        return new GpuWavTokenizer(dec, ops, weights, capacity, ops.kernels().caps().getComputeUnits());
    }

    /** Single-head non-causal flash plan over up to {@code capacity} keys of width {@code c}. */
    static FlashAttnTiling attentionPlan(int c, int capacity, int smemFloats, int computeUnits) {
        if (c % 4 != 0 || c > FlashAttnTiling.ACC_PER_GROUP) {
            throw new UnsupportedOperationException("wavtokenizer posnet attention of width " + c + " needs a width"
                    + " that is a multiple of 4 and at most " + FlashAttnTiling.ACC_PER_GROUP
                    + " for the flash-attention kernels");
        }
        FlashAttnTiling p = FlashAttnTiling.prefill(1, 1, c, c, capacity, smemFloats, Math.max(1, computeUnits));
        if (p.plan() != FlashAttnTiling.Plan.TILED) {
            throw new UnsupportedOperationException("no tiled flash-attention plan for the wavtokenizer posnet"
                    + " attention (d=" + c + ", " + smemFloats + " shared floats): " + p);
        }
        return p;
    }

    // ------------------------------------------------------------------ bind helpers

    private PosnetLayer posnetLayer(GpuWeights w, int il) {
        WavTokenizerDecoder.PosnetKind kind = WavTokenizerDecoder.posnetKind(il);
        switch (kind) {
            case RESNET:
                return new PosnetLayer(kind,
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_NORM1, false))),
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_NORM1, true))),
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_CONV1, false))),
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_CONV1, true))),
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_NORM2, false))),
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_NORM2, true))),
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_CONV2, false))),
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_CONV2, true))),
                        null, null);
            case ATTENTION: {
                String[] names = CpuWavTokenizer.ATTENTION_PROJECTIONS;
                GpuWeight[] proj = new GpuWeight[names.length];
                GpuBuffer[] projB = new GpuBuffer[names.length];
                for (int i = 0; i < names.length; i++) {
                    proj[i] = projection(w.get(WavTokenizerDecoder.posnet(il, names[i], false)));
                    projB[i] = f32(w.get(WavTokenizerDecoder.posnet(il, names[i], true)));
                }
                return new PosnetLayer(kind,
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_ATTN_NORM, false))),
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_ATTN_NORM, true))),
                        null, null, null, null, null, null, proj, projB);
            }
            case NORM:
                return new PosnetLayer(kind,
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_ATTN_NORM, false))),
                        f32(w.get(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_ATTN_NORM, true))),
                        null, null, null, null, null, null, null, null);
            default:
                throw new IllegalStateException("unhandled posnet kind " + kind);
        }
    }

    private ConvnextLayer convnextLayer(GpuWeights w, int il) {
        return new ConvnextLayer(
                f32(w.get(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_DW, false))),
                f32(w.get(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_DW, true))),
                f32Vector(w.get(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_NORM, false))),
                f32Vector(w.get(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_NORM, true))),
                w.get(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_PW1, false)),
                f32(w.get(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_PW1, true))),
                w.get(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_PW2, false)),
                f32(w.get(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_PW2, true))),
                f32(w.get(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_GAMMA, false))));
    }

    /**
     * A buffer holding the tensor as dense F32 from element 0: the tensor's own buffer when it is a single F32
     * shard, else a scratch copy decoded on the device ({@link GpuOps#embed} of each shard viewed as one row).
     */
    private GpuBuffer f32(GpuWeight w) {
        if (w.type() == GGMLType.F32 && w.isSingleShard()) {
            return w.shard(0);
        }
        long cols = w.cols();
        ScratchSlot dst = temp("f32." + w.name(), w.rows() * cols);
        for (int s = 0; s < w.shardCount(); s++) {
            long n = w.shardRows(s) * cols;
            GpuWeight flat = GpuWeight.of(w.name(), w.type(), new long[]{n}, w.shard(s));
            ops.embed(flat, new int[]{0}, dst.buffer(), Math.toIntExact(w.shardRowStart(s) * cols));
        }
        return dst.buffer();
    }

    /** A 1-D norm vector as the single-buffer F32 weight {@link GpuOps#norm} reads. */
    private GpuWeight f32Vector(GpuWeight w) {
        if (w.type() == GGMLType.F32 && w.isSingleShard()) {
            return w;
        }
        return GpuWeight.of(w.name(), GGMLType.F32, new long[]{w.cols() * w.rows()}, f32(w));
    }

    /**
     * A posnet 1x1 projection, ggml {@code {1, C_in, C_out}}, viewed as the {@code [C_out][C_in]} matrix its bytes
     * already are ({@code w[oc][ic][0]}), so {@link GpuOps#matmul} applies it to token-major rows.
     */
    private GpuWeight projection(GpuWeight w) {
        int c = dec.width();
        long[] ne = {c, c};
        if (w.isSingleShard()) {
            return GpuWeight.of(w.name(), w.type(), ne, w.shard(0));
        }
        List<GpuBuffer> bufs = new ArrayList<>();
        long[] rows = new long[w.shardCount()];
        for (int s = 0; s < w.shardCount(); s++) {
            if (w.shardRows(s) % c != 0) {
                throw new IllegalArgumentException("tensor " + w.name() + ": shard " + s + " holds " + w.shardRows(s)
                        + " rows, not whole output channels of " + c);
            }
            bufs.add(w.shard(s));
            rows[s] = w.shardRows(s) / c;
        }
        return GpuWeight.sharded(w.name(), w.type(), ne, bufs, rows);
    }

    private ScratchSlot temp(String name, long floats) {
        ScratchSlot s = scratch.temp(TEMP + name, floats);
        if (s.offset() != 0) {
            throw new IllegalStateException("scratch temporary " + s.name() + " does not start at element 0");
        }
        return s;
    }

    // ------------------------------------------------------------------ decode

    @Override
    public float[][] decode(int[] codes) {
        ensureOpen();
        dec.checkCodes(codes, capacity);
        int t = codes.length;
        int e = dec.nEmbd();
        int c = dec.width();
        int n = c * t;
        writeTransposeIndex(idxToChannelE, t, e);
        writeTransposeIndex(idxToToken, c, t);
        writeTransposeIndex(idxToChannel, t, c);

        ops.embed(tokEmbd, codes, emb.buffer(), 0);
        gather(idxToChannelE, emb, a, e * t);
        conv(t, e, c, WavTokenizerDecoder.INPUT_CONV_K, a, convW, convB, x);

        for (PosnetLayer layer : posnet) {
            posnet(layer, t);
        }

        gather(idxToToken, x, tok, n);
        ops.norm(NormKind.LAYER, tokNormW, tokNormB, dec.normEps(), tok.buffer(), 0, tok.buffer(), 0, t, c);
        gather(idxToChannel, tok, x, n);

        for (ConvnextLayer layer : convnext) {
            convnext(layer, t);
        }

        gather(idxToToken, x, tok, n);
        ops.norm(NormKind.LAYER, outNormW, outNormB, dec.normEps(), tok.buffer(), 0, tok.buffer(), 0, t, c);
        int o = dec.nOut();
        ops.matmul(output, tok.buffer(), 0, out.buffer(), 0, t);
        biasAdd(out, outputB, t, o);
        return readRows(t, o);
    }

    private void posnet(PosnetLayer l, int t) {
        int c = dec.width();
        int n = c * t;
        switch (l.kind) {
            case RESNET:
                groupNorm(t, x, l.normW, l.normB, a);
                ops.unary(ActOp.SILU, ActParams.NONE, a.buffer(), 0, a.buffer(), 0, n);
                conv(t, c, c, WavTokenizerDecoder.POSNET_CONV_K, a, l.conv1W, l.conv1B, b);
                groupNorm(t, b, l.norm2W, l.norm2B, a);
                ops.unary(ActOp.SILU, ActParams.NONE, a.buffer(), 0, a.buffer(), 0, n);
                conv(t, c, c, WavTokenizerDecoder.POSNET_CONV_K, a, l.conv2W, l.conv2B, b);
                ops.add(b.buffer(), 0, x.buffer(), 0, x.buffer(), 0, n);
                return;
            case ATTENTION:
                groupNorm(t, x, l.normW, l.normB, a);
                gather(idxToToken, a, tok, n);
                project(l, CpuWavTokenizer.PROJ_Q, tok, q, t);
                project(l, CpuWavTokenizer.PROJ_K, tok, k, t);
                project(l, CpuWavTokenizer.PROJ_V, tok, v, t);
                attention(t);
                project(l, CpuWavTokenizer.PROJ_OUT, tok, q, t);
                gather(idxToChannel, q, a, n);
                ops.add(a.buffer(), 0, x.buffer(), 0, x.buffer(), 0, n);
                return;
            case NORM:
                groupNorm(t, x, l.normW, l.normB, a);
                ops.copy(a.buffer(), 0, x.buffer(), 0, n);
                return;
            default:
                throw new IllegalStateException("unhandled posnet kind " + l.kind);
        }
    }

    private void convnext(ConvnextLayer l, int t) {
        int c = dec.width();
        int f = dec.nFf();
        int n = c * t;
        int ol = AudioKernels.checkDepthwiseShape(t, c, WavTokenizerDecoder.CONVNEXT_DW_K);
        ops.enqueue(AudioKernels.CONV1D_DEPTHWISE, new long[]{AudioKernels.convGlobal(c, ol)},
                new long[]{AudioKernels.CONV_LOCAL},
                t, c, WavTokenizerDecoder.CONVNEXT_DW_K, 0, 0, x.buffer(), l.dwW, l.dwB, a.buffer());
        gather(idxToToken, a, tok, n);
        ops.norm(NormKind.LAYER, l.normW, l.normB, dec.normEps(), tok.buffer(), 0, tok.buffer(), 0, t, c);
        ops.matmul(l.pw1, tok.buffer(), 0, ff.buffer(), 0, t);
        biasAdd(ff, l.pw1B, t, f);
        ops.unary(ActOp.GELU_TANH, ActParams.NONE, ff.buffer(), 0, ff.buffer(), 0, f * t);
        ops.matmul(l.pw2, ff.buffer(), 0, tok.buffer(), 0, t);
        biasAdd(tok, l.pw2B, t, c);
        ops.enqueue(NormKernels.MUL_ROW_SCALE, ActKernels.global1d(n, local), new long[]{local},
                NormKernels.mulRowScaleArgs(t, c, l.gamma, c, tok.buffer()));
        gather(idxToChannel, tok, a, n);
        ops.add(a.buffer(), 0, x.buffer(), 0, x.buffer(), 0, n);
    }

    /** {@code dst [T][C] = src [T][C] . W_p^T + b_p} for posnet projection {@code p}. */
    private void project(PosnetLayer l, int p, ScratchSlot src, ScratchSlot dst, int t) {
        ops.matmul(l.proj[p], src.buffer(), 0, dst.buffer(), 0, t);
        biasAdd(dst, l.projB[p], t, dec.width());
    }

    /**
     * Single-head non-causal attention of q over k, v (all {@code [T][C]}) into tok: {@code LLM_FlashAttn_F32}
     * with the AttnKernels argument layout, and {@code LLM_FlashAttn_Combine} when the keys are split.
     */
    private void attention(int t) {
        int c = dec.width();
        FlashAttnTiling.Split split = plan.prefillSplit(t, t, partialFloats);
        boolean direct = split.direct();
        Object[] fa = new Object[AttnKernels.FA_ARG_COUNT];
        fa[AttnKernels.ARG_FA_N_TOKENS] = t;
        fa[AttnKernels.ARG_FA_N_HEAD] = 1;
        fa[AttnKernels.ARG_FA_N_HEAD_KV] = 1;
        fa[AttnKernels.ARG_FA_D_K] = c;
        fa[AttnKernels.ARG_FA_D_V] = c;
        fa[AttnKernels.ARG_FA_BQ] = plan.bq();
        fa[AttnKernels.ARG_FA_BK] = plan.bk();
        fa[AttnKernels.ARG_FA_G_TILE] = plan.gTile();
        fa[AttnKernels.ARG_FA_N_SPLIT] = split.count();
        fa[AttnKernels.ARG_FA_SPLIT_LEN] = split.length();
        fa[AttnKernels.ARG_FA_KV_START] = 0;
        fa[AttnKernels.ARG_FA_KV_END] = t;
        fa[AttnKernels.ARG_FA_Q_ROW_STRIDE] = c;
        fa[AttnKernels.ARG_FA_OUT_ROW_STRIDE] = c;
        fa[AttnKernels.ARG_FA_K_ROW_STRIDE] = c;
        fa[AttnKernels.ARG_FA_K_HEAD_STRIDE] = c;
        fa[AttnKernels.ARG_FA_V_ROW_STRIDE] = c;
        fa[AttnKernels.ARG_FA_V_HEAD_STRIDE] = c;
        fa[AttnKernels.ARG_FA_SCALE] = dec.attentionScale();
        fa[AttnKernels.ARG_FA_SOFTCAP] = 0.0f;
        fa[AttnKernels.ARG_FA_MAX_BIAS] = 0.0f;
        fa[AttnKernels.ARG_FA_CAUSAL] = 0;
        fa[AttnKernels.ARG_FA_SWA_TYPE] = SwaType.NONE.code();
        fa[AttnKernels.ARG_FA_N_SWA] = 0;
        fa[AttnKernels.ARG_FA_BIAS_MODE] = AttentionSpec.BiasMode.NONE.code();
        fa[AttnKernels.ARG_FA_N_BUCKETS] = 0;
        fa[AttnKernels.ARG_FA_MAX_DISTANCE] = 0;
        fa[AttnKernels.ARG_FA_HAS_SINKS] = 0;
        fa[AttnKernels.ARG_FA_DIRECT_OUT] = direct ? 1 : 0;
        fa[AttnKernels.ARG_FA_Q_OFF] = 0;
        fa[AttnKernels.ARG_FA_OUT_OFF] = 0;
        fa[AttnKernels.ARG_FA_BUF_Q] = q.buffer();
        fa[AttnKernels.ARG_FA_BUF_K] = k.buffer();
        fa[AttnKernels.ARG_FA_BUF_V] = v.buffer();
        fa[AttnKernels.ARG_FA_BUF_Q_POS] = pos.buffer();
        fa[AttnKernels.ARG_FA_BUF_KV_POS] = pos.buffer();
        fa[AttnKernels.ARG_FA_BUF_SINKS] = placeholders.readOnlyFloats();
        fa[AttnKernels.ARG_FA_BUF_REL_BIAS] = placeholders.readOnlyFloats();
        fa[AttnKernels.ARG_FA_BUF_OUT] = tok.buffer();
        fa[AttnKernels.ARG_FA_BUF_PARTIAL] = direct ? placeholders.scratchFloats() : partials;
        ops.enqueue(AttnKernels.flashAttn(KvDType.F32), plan.prefillGlobal(t, split.count()), FlashAttnTiling.prefillLocal(),
                fa);
        if (direct) {
            return;
        }
        Object[] fc = new Object[AttnKernels.FC_ARG_COUNT];
        fc[AttnKernels.ARG_FC_N_ROWS] = t;
        fc[AttnKernels.ARG_FC_D_V] = c;
        fc[AttnKernels.ARG_FC_N_SPLIT] = split.count();
        fc[AttnKernels.ARG_FC_N_HEAD] = 1;
        fc[AttnKernels.ARG_FC_HAS_SINKS] = 0;
        fc[AttnKernels.ARG_FC_OUT_OFF] = 0;
        fc[AttnKernels.ARG_FC_BUF_PARTIAL] = partials;
        fc[AttnKernels.ARG_FC_BUF_SINKS] = placeholders.readOnlyFloats();
        fc[AttnKernels.ARG_FC_BUF_OUT] = tok.buffer();
        ops.enqueue(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(t), FlashAttnTiling.combineLocal(), fc);
    }

    private void groupNorm(int t, ScratchSlot src, GpuBuffer w, GpuBuffer bias, ScratchSlot dst) {
        ops.enqueue(NormKernels.GROUP_NORM,
                NormKernels.reduceGlobal(NormKernels.groupNormGroups(dec.nGroups()), local), new long[]{local},
                NormKernels.groupNormArgs(dec.width(), t, dec.nGroups(), dec.groupEps(), src.buffer(), w, bias,
                        dst.buffer()));
    }

    private void conv(int t, int ic, int oc, int kw, ScratchSlot src, GpuBuffer w, GpuBuffer bias, ScratchSlot dst) {
        int ol = AudioKernels.checkConvShape(t, ic, oc, kw, 1);
        ops.enqueue(AudioKernels.CONV1D_SAME, new long[]{AudioKernels.convGlobal(oc, ol)},
                new long[]{AudioKernels.CONV_LOCAL},
                t, ic, oc, kw, 1, 0, 0, src.buffer(), w, bias, dst.buffer());
    }

    /** {@code dst[r] = src[idx[r]]} for {@code r < n}: one transpose through an index table. */
    private void gather(ScratchSlot idx, ScratchSlot src, ScratchSlot dst, int n) {
        ops.enqueue(FlowKernels.GATHER_ROWS, new long[]{FlowKernels.gatherGlobal(n, 1)},
                new long[]{FlowKernels.GATHER_LOCAL},
                n, 1, 1, 1, 0, 0, idx.buffer(), src.buffer(), dst.buffer());
    }

    /** {@code data[r*dim + d] += bias[d]} for {@code r < rows}. */
    private void biasAdd(ScratchSlot data, GpuBuffer bias, int rows, int dim) {
        ops.enqueue(BATCH_BIAS_ADD, ActKernels.global1d(dim, local), new long[]{local}, dim, rows, bias,
                data.buffer());
    }

    /**
     * Uploads the gather table that transposes a dense {@code [rows][cols]} tensor to {@code [cols][rows]}:
     * {@code idx[col*rows + r] = r*cols + col}.
     */
    private void writeTransposeIndex(ScratchSlot idx, int rows, int cols) {
        ByteBuffer bb = ByteBuffer.wrap(hostIndex).order(ByteOrder.LITTLE_ENDIAN);
        for (int col = 0; col < cols; col++) {
            for (int r = 0; r < rows; r++) {
                bb.putInt(r * cols + col);
            }
        }
        closeEvent(queue.enqueueWriteBuffer(idx.buffer(), 0, (long) rows * cols * Integer.BYTES, hostIndex));
    }

    private void writeInts(ScratchSlot dst, int[] values, int n) {
        ByteBuffer bb = ByteBuffer.allocate(n * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            bb.putInt(values[i]);
        }
        closeEvent(queue.enqueueWriteBuffer(dst.buffer(), 0, (long) n * Integer.BYTES, bb.array()));
    }

    private float[][] readRows(int t, int o) {
        float[] host = new float[t * o];
        closeEvent(queue.enqueueReadBuffer(out.buffer(), 0, (long) t * o * Float.BYTES, host));
        queue.finish();
        float[][] rows = new float[t][o];
        for (int i = 0; i < t; i++) {
            System.arraycopy(host, i * o, rows[i], 0, o);
        }
        return rows;
    }

    private static void closeEvent(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    @Override
    public int capacity() {
        return capacity;
    }

    @Override
    public WavTokenizerDecoder decoder() {
        return dec;
    }

    /** The flash plan of the posnet attention; null without an attention layer. */
    FlashAttnTiling attentionPlan() {
        return plan;
    }

    /**
     * Ends use of this decoder; further decodes fail. The buffers belong to the model's scratch and weights and are
     * released with them. Idempotent.
     */
    @Override
    public void close() {
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("wavtokenizer GPU decoder is closed");
        }
    }

    @Override
    public String toString() {
        return "GpuWavTokenizer{" + dec + ", capacity=" + capacity + ", attention=" + plan + '}';
    }
}
