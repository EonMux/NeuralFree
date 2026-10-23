package com.eonmux.neuralfree.gguf.ops.rwkv;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Block factories of the RWKV family, exposed to the block catalogs: the RWKV6 / QRWKV time mix ({@link GpuRwkv6Mixer},
 * {@link CpuRwkv6Mixer}) for {@code rwkv6} and {@code rwkv6qwen2}, the RWKV7 time mix ({@link GpuRwkv7Mixer},
 * {@link CpuRwkv7Mixer}) for {@code rwkv7} and {@code arwkv7}, and the channel-mix FFN ({@link GpuRwkvChannelMix},
 * {@link CpuRwkvChannelMix}) of {@code rwkv6} and {@code rwkv7}. {@code rwkv6qwen2} and {@code arwkv7} use the dense
 * SwiGLU FFN of the FFN family. The family has no layer extras.
 *
 * <p>Shared conventions of the blocks:
 * <ul>
 *   <li>The recurrent memory of layer {@code il} holds the token-shift row
 *       {@code [att_shift(n_embd) | ffn_shift(n_embd)]} ({@code LayerMemoryPlan.STATE_SHIFT}) and the WKV state
 *       ({@code STATE_SCAN}) of the single sequence; the time mix uses shift offset 0, the channel mix
 *       {@code n_embd}.</li>
 *   <li>Working rows are scratch temporaries named {@code rwkv6.*}, {@code rwkv7.*} and {@code rwkv.cmix.*}.
 *       {@link #V_FIRST} additionally carries the first RWKV7 layer's value rows to the later layers of the same
 *       step, so no other block may use that name.</li>
 * </ul>
 */
public final class RwkvBlocks {

    /** Scratch temporary holding the first RWKV7 layer's {@code v} rows for the value residual of later layers. */
    public static final String V_FIRST = "rwkv7.v_first";

    /** Group-norm epsilon of the time-mix output (models/rwkv6-base.cpp:152, models/rwkv7-base.cpp:119). */
    public static final float GROUP_NORM_EPS = 64e-5f;

    /**
     * {@code data[s*dim + d] += bias[d]} for {@code s < seq_len} ({@code kernels/llm/eltwise.cl}); arguments
     * {@code i dim, i seq_len(dyn), R bias, RW data}, global {@code roundUp(dim, L)}.
     */
    public static final String BATCH_BIAS_ADD = GpuOps.BATCH_BIAS_ADD;
    /** {@code seq_len} of {@link #BATCH_BIAS_ADD}. */
    public static final int ARG_BIAS_SEQ_LEN = 1;

    /** Every kernel the RWKV blocks launch, for prewarming. */
    public static final List<String> KERNELS = kernels();

    private RwkvBlocks() {
    }

    private static List<String> kernels() {
        List<String> k = new ArrayList<>(RwkvKernels.ALL);
        k.addAll(Arrays.asList(BATCH_BIAS_ADD, GpuOps.UNARY, GpuOps.GLU, GpuOps.ROW_NORM, GpuOps.COPY2D,
                GpuOps.AXPBY));
        return Collections.unmodifiableList(k);
    }

    // ================================================================== factory exposure

    /** GPU time-mix mixers: RWKV6 and RWKV6_QWEN2 share {@link GpuRwkv6Mixer}; RWKV7 uses {@link GpuRwkv7Mixer}. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.unmodifiableList(Arrays.asList(
                gpuMixer(MixerKind.RWKV6), gpuMixer(MixerKind.RWKV6_QWEN2), gpuMixer(MixerKind.RWKV7)));
    }

    /** CPU time-mix mixers, the twins of {@link #gpuMixers()}. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.unmodifiableList(Arrays.asList(
                cpuMixer(MixerKind.RWKV6), cpuMixer(MixerKind.RWKV6_QWEN2), cpuMixer(MixerKind.RWKV7)));
    }

    /** The GPU channel-mix FFN of {@link FfnKind#RWKV_CHANNEL_MIX}. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.<GpuFfnFactory>singletonList(new GpuFfnFactory() {
            @Override
            public FfnKind kind() {
                return FfnKind.RWKV_CHANNEL_MIX;
            }

            @Override
            public GpuFfnBlock create(LayerSpec layer) {
                checkFfn(layer);
                return new GpuRwkvChannelMix();
            }
        });
    }

    /** The CPU channel-mix FFN of {@link FfnKind#RWKV_CHANNEL_MIX}. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.<CpuFfnFactory>singletonList(new CpuFfnFactory() {
            @Override
            public FfnKind kind() {
                return FfnKind.RWKV_CHANNEL_MIX;
            }

            @Override
            public CpuFfnBlock create(LayerSpec layer) {
                checkFfn(layer);
                return new CpuRwkvChannelMix();
            }
        });
    }

    /** The RWKV family has no layer extras. */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.emptyList();
    }

    /** The RWKV family has no layer extras. */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.emptyList();
    }

    private static GpuMixerFactory gpuMixer(MixerKind kind) {
        return new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return kind;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                checkMixer(kind, layer);
                return kind == MixerKind.RWKV7 ? new GpuRwkv7Mixer() : new GpuRwkv6Mixer();
            }
        };
    }

    private static CpuMixerFactory cpuMixer(MixerKind kind) {
        return new CpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return kind;
            }

            @Override
            public CpuMixer create(LayerSpec layer) {
                checkMixer(kind, layer);
                return kind == MixerKind.RWKV7 ? new CpuRwkv7Mixer() : new CpuRwkv6Mixer();
            }
        };
    }

    // ================================================================== spec checks

    private static void checkMixer(MixerKind kind, LayerSpec layer) {
        if (layer.mixerKind() != kind) {
            throw new IllegalArgumentException("the " + kind + " factory cannot serve layer " + layer.il() + " of kind "
                    + layer.mixerKind());
        }
        recurrent(layer, kind);
    }

    private static void checkFfn(LayerSpec layer) {
        if (layer.ffnKind() != FfnKind.RWKV_CHANNEL_MIX) {
            throw new IllegalArgumentException("the RWKV channel-mix factory cannot serve layer " + layer.il()
                    + " with FFN " + layer.ffnKind());
        }
        channelMixSpec(layer);
    }

    /**
     * The layer's RWKV recurrent spec, checked against the mixer kind, the kernels' head-size limit and the head
     * layout {@code n_embd = n_head * head_size} (models/rwkv6-base.cpp:43, models/rwkv7-base.cpp:42).
     *
     * @throws IllegalArgumentException when the layer has no recurrent spec of {@code kind}, its head size is
     *                                  unsupported, or {@code n_embd} is not a positive multiple of the head size
     */
    static RecurrentSpec recurrent(LayerSpec layer, MixerKind kind) {
        RecurrentSpec r = layer.recurrent();
        if (r == null || r.kind() != kind) {
            throw new IllegalArgumentException("layer " + layer.il() + " (" + layer.mixerKind()
                    + ") needs a " + kind + " recurrent spec, got " + r);
        }
        RwkvKernels.checkHeadSize(r.wkvHeadSize());
        if (r.nEmbd() < 1 || r.nEmbd() % r.wkvHeadSize() != 0) {
            throw new IllegalArgumentException("layer " + layer.il() + " (" + kind + "): n_embd " + r.nEmbd()
                    + " is not a positive multiple of the WKV head size " + r.wkvHeadSize());
        }
        return r;
    }

    /**
     * The recurrent spec of a channel-mix layer: an RWKV6 or RWKV7 time mix with two token-shift slots, the second
     * holding the channel mix's previous token.
     *
     * @throws IllegalArgumentException when the layer's mixer is not RWKV6 / RWKV7 or has one token-shift slot
     */
    static RecurrentSpec channelMixSpec(LayerSpec layer) {
        RecurrentSpec r = layer.recurrent();
        if (r == null || (r.kind() != MixerKind.RWKV6 && r.kind() != MixerKind.RWKV7)) {
            throw new IllegalArgumentException("RWKV channel mix of layer " + layer.il()
                    + " needs an RWKV6 or RWKV7 recurrent spec, got " + r);
        }
        if (r.tokenShiftCount() != 2) {
            throw new IllegalArgumentException("RWKV channel mix of layer " + layer.il()
                    + " needs two token-shift slots, the spec has " + r.tokenShiftCount());
        }
        return r;
    }

    // ================================================================== GPU helpers

    /**
     * Bind-time services of one GPU block: tensor resolution and checks, scratch temporaries, the layer's recurrent
     * tensors, prepared dispatches of the element-wise, row-norm and RWKV kernels, and device buffers derived from
     * weights (row blocks, packed vectors, F32 forms). Everything it prepares or allocates is released by
     * {@link #close()}. Not thread-safe.
     */
    static final class GpuKit implements AutoCloseable {

        /** {@code cur_off} of {@link RwkvKernels#TOKEN_SHIFT_LERP}. */
        static final int ARG_SHIFT_CUR_OFF = 5;
        /** {@code cur} of {@link RwkvKernels#TOKEN_SHIFT_LERP}. */
        static final int ARG_SHIFT_CUR = 6;

        final GpuForwardContext ctx;
        final GpuOps ops;
        final GpuWeights weights;
        final LayerSpec layer;
        final String where;
        final int ubatch;
        final int local;
        private final List<PreparedDispatch> prepared = new ArrayList<>();
        private final List<GpuBuffer> owned = new ArrayList<>();
        private boolean weightsAwaited;
        private boolean closed;

        GpuKit(GpuForwardContext ctx, LayerSpec layer, String block) {
            this.ctx = Objects.requireNonNull(ctx, "ctx");
            this.layer = Objects.requireNonNull(layer, "layer");
            this.ops = ctx.ops();
            this.where = ctx.spec().arch().ggufName() + " layer " + layer.il() + " " + block;
            this.ubatch = ctx.scratch().ubatch();
            this.local = ops.local();
            this.weights = ctx.weightSource().weightsFor(layer.weightLayer());
        }

        // -------------------------------------------------------------- tensors

        /** Required matrix {@code <base>.weight} with {@code cols} inputs; {@code rows <= 0} accepts any row count. */
        GpuWeight matrix(String base, long cols, long rows) {
            GpuWeight w = weights.layer(layer, TensorNames.weight(base));
            checkMatrix(w, cols, rows);
            return w;
        }

        /** {@link #matrix} or null when the tensor is absent. */
        GpuWeight matrixOrNull(String base, long cols, long rows) {
            GpuWeight w = weights.layerOrNull(layer, TensorNames.weight(base));
            if (w != null) {
                checkMatrix(w, cols, rows);
            }
            return w;
        }

        private void checkMatrix(GpuWeight w, long cols, long rows) {
            if (w.cols() != cols || (rows > 0 && w.rows() != rows)) {
                throw new IllegalStateException(where + ": " + w.name() + " is " + w.cols() + "x" + w.rows()
                        + ", expected " + cols + "x" + (rows > 0 ? String.valueOf(rows) : "*"));
            }
        }

        /** Required vector {@code <base>.weight} of {@code n} values as a dense F32 buffer. */
        GpuBuffer vector(String base, long n) {
            return f32(weights.layer(layer, TensorNames.weight(base)), n);
        }

        /** {@link #vector} or null when the tensor is absent. */
        GpuBuffer vectorOrNull(String base, long n) {
            GpuWeight w = weights.layerOrNull(layer, TensorNames.weight(base));
            return w != null ? f32(w, n) : null;
        }

        /** Optional bias {@code <base>.bias} of {@code n} values as a dense F32 buffer, or null. */
        GpuBuffer biasOrNull(String base, long n) {
            GpuWeight w = weights.layerOrNull(layer, TensorNames.bias(base));
            return w != null ? f32(w, n) : null;
        }

        /**
         * The tensor's {@code n} values as a dense F32 buffer: its own buffer when it is a single-buffer F32 tensor,
         * otherwise an owned buffer filled by the row lookup kernels ({@link GpuOps#embed}).
         */
        GpuBuffer f32(GpuWeight w, long n) {
            if (w.rows() * w.cols() != n) {
                throw new IllegalStateException(where + ": " + w.name() + " holds " + w.rows() * w.cols()
                        + " values, " + n + " expected");
            }
            if (w.type() == GGMLType.F32 && w.isSingleShard()) {
                return w.shard(0);
            }
            if (w.rows() > Integer.MAX_VALUE) {
                throw new IllegalStateException(where + ": " + w.name() + " has too many rows for an F32 copy");
            }
            int[] rows = new int[(int) w.rows()];
            for (int r = 0; r < rows.length; r++) {
                rows[r] = r;
            }
            GpuBuffer b = own(n * Float.BYTES);
            awaitWeights();
            ops.embed(w, rows, b, 0);
            return b;
        }

        /**
         * Rows {@code [rowStart, rowStart + rows)} of {@code w} copied byte for byte into an owned buffer, as a
         * weight of shape {@code {w.cols(), rows}} with {@code w}'s input transform and output scale, so it computes
         * those rows of {@code w}'s product (models/rwkv6-base.cpp:64-65 slices {@code time_mix_w2} this way per mix
         * plane; the CPU twin reads the same rows with {@code CpuMatVec.matvecRows}).
         */
        GpuWeight rowBlock(GpuWeight w, long rowStart, long rows, String name) {
            if (rowStart < 0 || rows < 1 || rowStart + rows > w.rows()) {
                throw new IllegalArgumentException(where + ": rows [" + rowStart + ", " + (rowStart + rows)
                        + ") outside " + w.name() + " with " + w.rows() + " rows");
            }
            long rowBytes = w.rowBytes();
            GpuBuffer dst = own(rows * rowBytes);
            awaitWeights();
            long end = rowStart + rows;
            long row = rowStart;
            while (row < end) {
                int s = w.shardOfRow(row);
                long shardStart = w.shardRowStart(s);
                long take = Math.min(end, shardStart + w.shardRows(s)) - row;
                ctx.queue().enqueueCopyBuffer(w.shard(s), (row - shardStart) * rowBytes, dst,
                        (row - rowStart) * rowBytes, take * rowBytes);
                row += take;
            }
            GpuWeight block = GpuWeight.of(name, w.type(), new long[]{w.cols(), rows}, dst);
            if (w.hadamard() != null) {
                block = block.withHadamard(w.hadamard());
            }
            return w.hasScale() ? block.withScale(w.scale()) : block;
        }

        /** The F32 vectors of {@code n} values each, packed as {@code [parts.size()][n]} into an owned buffer. */
        GpuBuffer pack(List<GpuBuffer> parts, long n) {
            GpuBuffer dst = own(parts.size() * n * Float.BYTES);
            awaitWeights();
            for (int i = 0; i < parts.size(); i++) {
                ctx.queue().enqueueCopyBuffer(parts.get(i), 0, dst, i * n * Float.BYTES, n * Float.BYTES);
            }
            return dst;
        }

        private GpuBuffer own(long bytes) {
            GpuBuffer b = ctx.backend().allocateBuffer(bytes);
            owned.add(b);
            return b;
        }

        /** Makes the compute queue wait for the layer's weights before a bind-time copy reads them. */
        private void awaitWeights() {
            if (!weightsAwaited) {
                ctx.weightSource().prefetch(layer.weightLayer());
                ctx.weightSource().await(layer.weightLayer());
                weightsAwaited = true;
            }
        }

        // -------------------------------------------------------------- scratch and memory

        /** Temporary of {@code ubatch} rows of {@code width} floats at element 0 of its buffer. */
        ScratchSlot temp(String name, long width) {
            ScratchSlot s = ctx.scratch().temp(name, Math.multiplyExact((long) ubatch, width));
            if (s.offset() != 0) {
                throw new IllegalStateException(where + ": temporary " + s + " must start at element 0");
            }
            return s;
        }

        /** Recurrent tensor {@code name} of this layer, checked to hold at least {@code minFloats} floats. */
        GpuBuffer state(String name, long minFloats) {
            if (!ctx.hasMemory()) {
                throw new IllegalStateException(where + ": the RWKV blocks need recurrent memory");
            }
            GpuRecurrentState rs = ctx.memory().recurrent(layer.il());
            if (rs == null) {
                throw new IllegalStateException(where + ": the memory plan gives this layer no recurrent state");
            }
            GpuBuffer b = rs.tensor(name);
            if (b == null) {
                throw new IllegalStateException(where + ": recurrent memory has no '" + name + "' tensor");
            }
            if (b.sizeInBytes() < minFloats * Float.BYTES) {
                throw new IllegalStateException(where + ": recurrent tensor '" + name + "' holds "
                        + b.sizeInBytes() / Float.BYTES + " floats, " + minFloats + " needed");
            }
            return b;
        }

        /** The {@code int[1]} state-row index of the sequence being decoded. */
        GpuBuffer stateRows() {
            return ctx.memory().stepInputs().stateRows();
        }

        GpuBuffer placeholder() {
            return ctx.placeholders().readOnlyFloats();
        }

        // -------------------------------------------------------------- prepared dispatches

        PreparedDispatch prepare(String kernel, long global, int localSize, Object... args) {
            PreparedDispatch d = ops.prepare(kernel, new long[]{global}, new long[]{localSize}, args);
            prepared.add(d);
            return d;
        }

        /** {@code LLM_Unary} in place over {@code ubatch * width} floats from element 0 of {@code buf}. */
        PreparedDispatch unary(ActOp op, GpuBuffer buf, int width) {
            int n = ActKernels.elementCount(ubatch, width);
            return prepare(GpuOps.UNARY, ActKernels.global1d(n, local)[0], local,
                    ActKernels.unaryArgs(op, ActParams.NONE, n, buf, 0, buf, 0));
        }

        void runUnary(PreparedDispatch d, int nTokens, int width) {
            int n = nTokens * width;
            d.setInt(ActKernels.ARG_UNARY_N, n);
            d.setGlobal(ActKernels.global1d(n, local));
            d.dispatch();
        }

        /**
         * {@code LLM_GLU} over {@code [ubatch][width]} rows at element 0 of each buffer; {@code out} may be
         * {@code gate} or {@code up}.
         */
        PreparedDispatch glu(GluOp op, GpuBuffer gate, GpuBuffer up, GpuBuffer out, int width) {
            return prepare(GpuOps.GLU, ActKernels.global1d((long) ubatch * width, local)[0], local,
                    ActKernels.gluArgs(op, ActParams.NONE, ubatch, width, gate, 0, width, up, 0, width, out, 0,
                            width));
        }

        void runGlu(PreparedDispatch d, int nTokens, int width) {
            d.setInt(ActKernels.ARG_GLU_N_ROWS, nTokens);
            d.setGlobal(ActKernels.global1d((long) nTokens * width, local));
            d.dispatch();
        }

        /** {@value #BATCH_BIAS_ADD} of {@code bias} onto rows of {@code width} at element 0, or null without bias. */
        PreparedDispatch biasAdd(GpuBuffer bias, GpuBuffer data, int width) {
            if (bias == null) {
                return null;
            }
            ActKernels.checkFloatView("bias", bias, 0, width, 1, width);
            ActKernels.checkFloatView("biased rows", data, 0, width, ubatch, width);
            return prepare(BATCH_BIAS_ADD, ActKernels.global1d(width, local)[0], local, width, ubatch, bias, data);
        }

        /** Runs a {@link #biasAdd} dispatch over {@code nTokens} rows; nothing for a null dispatch. */
        void runBias(PreparedDispatch bias, int nTokens) {
            if (bias != null) {
                bias.setInt(ARG_BIAS_SEQ_LEN, nTokens);
                bias.dispatch();
            }
        }

        /** Projection {@code out[t] = W in[t] (+ b)} into rows at element 0 of {@code out}. */
        void project(GpuWeight w, PreparedDispatch bias, GpuBuffer in, int inOff, GpuBuffer out, int nTokens) {
            ops.matmul(w, in, inOff, out, 0, nTokens);
            runBias(bias, nTokens);
        }

        /**
         * {@link RwkvKernels#TOKEN_SHIFT_LERP} into {@code out}; the input rows are set per step by
         * {@link #runShift}. Absent lerp operands bind the read-only placeholder.
         */
        PreparedDispatch tokenShift(int nEmbd, int nMix, int mode, GpuBuffer shift, int shiftOff, GpuBuffer lerpX,
                                    GpuBuffer lerpFused, GpuBuffer out) {
            GpuBuffer ph = placeholder();
            return prepare(RwkvKernels.TOKEN_SHIFT_LERP, RwkvKernels.tokenShiftGlobalSize(ubatch, nEmbd),
                    RwkvKernels.ELEMENTWISE_LOCAL, ubatch, nEmbd, nMix, mode, shiftOff, 0, ph, shift,
                    lerpX != null ? lerpX : ph, lerpFused != null ? lerpFused : ph, out);
        }

        void runShift(PreparedDispatch d, int nTokens, int nEmbd, ScratchSlot in) {
            d.setInt(RwkvKernels.ARG_SHIFT_N_TOKENS, nTokens);
            d.setInt(ARG_SHIFT_CUR_OFF, in.offset());
            d.setBuffer(ARG_SHIFT_CUR, in.buffer());
            d.setGlobal(RwkvKernels.tokenShiftGlobalSize(nTokens, nEmbd));
            d.dispatch();
        }

        /** {@link RwkvKernels#DECAY6} / {@link RwkvKernels#DECAY7} over {@code [ubatch][nCols]}. */
        PreparedDispatch decay(String kernel, int nCols, GpuBuffer in, GpuBuffer base, GpuBuffer out) {
            int n = ActKernels.elementCount(ubatch, nCols);
            return prepare(kernel, RwkvKernels.elementwiseGlobalSize(n), RwkvKernels.ELEMENTWISE_LOCAL, n, nCols,
                    in, base, out);
        }

        void runDecay(PreparedDispatch d, int nTokens, int nCols) {
            int n = nTokens * nCols;
            d.setInt(RwkvKernels.ARG_DECAY_N, n);
            d.setGlobal(RwkvKernels.elementwiseGlobalSize(n));
            d.dispatch();
        }

        /** A WKV6 / GLA / WKV7 recurrence; its global size does not depend on the token count. */
        PreparedDispatch wkv(String kernel, int nHead, int headSize, Object... args) {
            return prepare(kernel, RwkvKernels.wkvGlobalSize(nHead, headSize), RwkvKernels.wkvLocalSize(headSize),
                    args);
        }

        void runWkv(PreparedDispatch d, int nTokens) {
            d.setInt(RwkvKernels.ARG_WKV_N_TOKENS, nTokens);
            d.dispatch();
        }

        /** {@link RwkvKernels#KK_PREP} / {@link RwkvKernels#BONUS}: one work-group per (token, head). */
        PreparedDispatch headRows(String kernel, int nHead, int headSize, Object... args) {
            return prepare(kernel, RwkvKernels.headRowGlobalSize(ubatch, nHead, headSize),
                    RwkvKernels.headRowLocalSize(headSize), args);
        }

        void runHeadRows(PreparedDispatch d, int nTokens, int nHead, int headSize) {
            d.setInt(RwkvKernels.ARG_HEAD_ROW_N_TOKENS, nTokens);
            d.setGlobal(RwkvKernels.headRowGlobalSize(nTokens, nHead, headSize));
            d.dispatch();
        }

        /**
         * The per-head group norm of the WKV output in place (ggml_norm over each head, eps {@link #GROUP_NORM_EPS},
         * then {@code * ln + ln_b} with per-head rows of the {@code [n_head][S]} weights).
         */
        PreparedDispatch groupNorm(GpuBuffer y, int nHead, int headSize, GpuBuffer ln, GpuBuffer lnB) {
            NormKernels.checkReduceLocal(local);
            int nRows = ActKernels.elementCount(ubatch, nHead);
            return prepare(GpuOps.ROW_NORM, NormKernels.reduceGlobal(NormKernels.rowNormGroups(nRows), local)[0],
                    local, NormKernels.rowNormArgs(NormKind.LAYER, GROUP_NORM_EPS, nRows, headSize, headSize, y, 0,
                            y, 0, ln, lnB, nHead, placeholder()));
        }

        void runGroupNorm(PreparedDispatch d, int nTokens, int nHead) {
            int nRows = nTokens * nHead;
            d.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, nRows);
            d.setGlobal(NormKernels.reduceGlobal(NormKernels.rowNormGroups(nRows), local));
            d.dispatch();
        }

        // -------------------------------------------------------------- step checks

        void checkStep(int nTokens, ScratchSlot in, ScratchSlot out, int nEmbd, List<GpuBuffer> working) {
            Objects.requireNonNull(in, "in");
            Objects.requireNonNull(out, "out");
            if (nTokens < 1 || nTokens > ubatch) {
                throw new IllegalArgumentException(where + ": token count " + nTokens + " outside [1, " + ubatch
                        + "]");
            }
            in.require((long) nTokens * nEmbd);
            out.require((long) nTokens * nEmbd);
            for (GpuBuffer b : working) {
                if (in.buffer() == b || out.buffer() == b) {
                    throw new IllegalArgumentException(where + ": the input / output slots " + in + ", " + out
                            + " share a buffer with the block's working rows");
                }
            }
        }

        /** Releases the prepared dispatches and owned buffers. Idempotent. */
        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            RuntimeException first = null;
            for (PreparedDispatch d : prepared) {
                try {
                    d.close();
                } catch (RuntimeException e) {
                    first = first == null ? e : first;
                }
            }
            for (GpuBuffer b : owned) {
                try {
                    b.close();
                } catch (RuntimeException e) {
                    first = first == null ? e : first;
                }
            }
            prepared.clear();
            owned.clear();
            if (first != null) {
                throw first;
            }
        }
    }

    // ================================================================== CPU helpers

    /**
     * Bind-time services of one CPU block: tensor resolution and checks, decoded F32 vectors, scratch temporaries
     * and the layer's recurrent tensors.
     */
    static final class CpuKit {

        final CpuForwardContext ctx;
        final CpuWeights weights;
        final CpuMatVec mv;
        final LayerSpec layer;
        final String where;
        final int ubatch;

        CpuKit(CpuForwardContext ctx, LayerSpec layer, String block) {
            this.ctx = Objects.requireNonNull(ctx, "ctx");
            this.layer = Objects.requireNonNull(layer, "layer");
            this.weights = ctx.weights();
            this.mv = ctx.matVec();
            this.where = ctx.spec().arch().ggufName() + " layer " + layer.il() + " " + block;
            this.ubatch = ctx.scratch().ubatch();
        }

        /** Required matrix {@code <base>.weight} with {@code cols} inputs; {@code rows <= 0} accepts any row count. */
        CpuTensorRef matrix(String base, long cols, long rows) {
            CpuTensorRef w = weights.layer(layer, TensorNames.weight(base));
            checkMatrix(w, base, cols, rows);
            return w;
        }

        /** {@link #matrix} or null when the tensor is absent. */
        CpuTensorRef matrixOrNull(String base, long cols, long rows) {
            CpuTensorRef w = weights.layerOrNull(layer, TensorNames.weight(base));
            if (w != null) {
                checkMatrix(w, base, cols, rows);
            }
            return w;
        }

        private void checkMatrix(CpuTensorRef w, String base, long cols, long rows) {
            if (w.cols() != cols || (rows > 0 && w.rows() != rows)) {
                throw new IllegalStateException(where + ": " + layer.tensorName(TensorNames.weight(base)) + " is "
                        + w.cols() + "x" + w.rows() + ", expected " + cols + "x" + (rows > 0 ? String.valueOf(rows)
                        : "*"));
            }
        }

        /** Required vector {@code <base>.weight} of {@code n} decoded values (shared, read only). */
        float[] vector(String base, int n) {
            float[] v = vectorOrNull(base, n);
            if (v == null) {
                throw new MissingTensorException(weights.archName(), layer.il(),
                        layer.tensorName(TensorNames.weight(base)));
            }
            return v;
        }

        /** {@link #vector} or null when the tensor is absent. */
        float[] vectorOrNull(String base, int n) {
            return checkVector(weights.layerFloatsOrNull(layer, TensorNames.weight(base)), TensorNames.weight(base),
                    n);
        }

        /** Optional bias {@code <base>.bias} of {@code n} values, or null. */
        float[] biasOrNull(String base, int n) {
            return checkVector(weights.layerFloatsOrNull(layer, TensorNames.bias(base)), TensorNames.bias(base), n);
        }

        private float[] checkVector(float[] v, String suffix, int n) {
            if (v != null && v.length != n) {
                throw new IllegalStateException(where + ": " + layer.tensorName(suffix) + " holds " + v.length
                        + " values, " + n + " expected");
            }
            return v;
        }

        /** Float temporary of {@code ubatch} rows of {@code width}. */
        float[] temp(String name, int width) {
            return ctx.scratch().temp(name, Math.multiplyExact(ubatch, width));
        }

        /** Recurrent tensor {@code name} of this layer (live array, single sequence from index 0). */
        float[] state(String name) {
            if (!ctx.hasMemory()) {
                throw new IllegalStateException(where + ": the RWKV blocks need recurrent memory");
            }
            RecurrentStateAccessor rs = ctx.memory().recurrent(layer.il());
            if (rs == null) {
                throw new IllegalStateException(where + ": the memory plan gives this layer no recurrent state");
            }
            float[] s = rs.tensor(name);
            if (s == null) {
                throw new IllegalStateException(where + ": recurrent memory has no '" + name + "' tensor");
            }
            return s;
        }

        /** Projection {@code out[t] = W in[t] (+ b)} into rows from index 0 of {@code out}. */
        void project(CpuTensorRef w, float[] bias, float[] in, int inOff, float[] out, int nTokens) {
            mv.matmul(w, in, inOff, out, 0, nTokens);
            if (bias != null) {
                CpuOps.addRowBias(out, 0, nTokens, (int) w.rows(), bias, 0);
            }
        }

        void checkStep(int nTokens, float[] in, float[] out, int nEmbd) {
            Objects.requireNonNull(in, "in");
            Objects.requireNonNull(out, "out");
            if (nTokens < 1 || nTokens > ubatch) {
                throw new IllegalArgumentException(where + ": token count " + nTokens + " outside [1, " + ubatch
                        + "]");
            }
            long need = (long) nTokens * nEmbd;
            if (in.length < need || out.length < need) {
                throw new IllegalArgumentException(where + ": input / output rows hold " + in.length + " / "
                        + out.length + " floats, " + need + " are needed");
            }
            if (in == out) {
                throw new IllegalArgumentException(where + ": input and output rows must be distinct arrays");
            }
        }
    }
}
