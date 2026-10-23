package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

/**
 * Block factories of the delta-rule family, exposed to the block catalogs: the gated DeltaNet mixer of qwen3next,
 * qwen35, qwen35moe and clef ({@link GpuGdnMixer}, {@link CpuGdnMixer}), the Kimi delta attention mixer of kimi-linear,
 * kimi-k3, glm5-next and bailingmoe3 ({@link GpuKdaMixer}, {@link CpuKdaMixer}) and the MiniMax-01 lightning attention
 * mixer ({@link GpuLightningMixer}, {@link CpuLightningMixer}). The family has no FFN blocks and no layer extras.
 *
 * <p>The package-private helpers below are shared by the six mixers: tensor lookup with shape checks, F32 forms of
 * the vectors the kernels read raw, the recurrent-state views, and the argument arrays of {@code LLM_SSM_Conv},
 * {@code LLM_GDN_GatePrep}, {@code LLM_GDN_Scan} and {@code LLM_LightningAttn} in their kernel signature order.</p>
 */
public final class DeltaNetBlocks {

    /** Temporary holding the decay gate written by {@code LLM_GDN_GatePrep} (dense from element 0). */
    static final String TEMP_G = "deltanet.g";
    /** Temporary holding the update rate written by {@code LLM_GDN_GatePrep} (dense from element 0). */
    static final String TEMP_BETA = "deltanet.beta";
    /** Zeroed conv state of a layer run without memory. */
    static final String TEMP_CONV_STATE = "deltanet.conv_state";
    /** Zeroed scan state of a layer run without memory. */
    static final String TEMP_SCAN_STATE = "deltanet.scan_state";

    /** L2 epsilon of the glm5-next KDA q/k norm, a constant there (models/glm5-next.cpp:865). */
    static final float GLM5_NEXT_L2_EPS = 1e-6f;

    private DeltaNetBlocks() {
    }

    // ================================================================== factory exposure

    /** GPU mixers of the {@link MixerKind#GDN}, {@link MixerKind#KDA} and {@link MixerKind#LIGHTNING} kinds. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.unmodifiableList(Arrays.asList(
                gpu(MixerKind.GDN, GpuGdnMixer::new),
                gpu(MixerKind.KDA, GpuKdaMixer::new),
                gpu(MixerKind.LIGHTNING, GpuLightningMixer::new)));
    }

    /** CPU mixers of the {@link MixerKind#GDN}, {@link MixerKind#KDA} and {@link MixerKind#LIGHTNING} kinds. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.unmodifiableList(Arrays.asList(
                cpu(MixerKind.GDN, CpuGdnMixer::new),
                cpu(MixerKind.KDA, CpuKdaMixer::new),
                cpu(MixerKind.LIGHTNING, CpuLightningMixer::new)));
    }

    /** The delta-rule family has no FFN blocks. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.emptyList();
    }

    /** The delta-rule family has no FFN blocks. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.emptyList();
    }

    /** The delta-rule family has no layer extras. */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.emptyList();
    }

    /** The delta-rule family has no layer extras. */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.emptyList();
    }

    private static GpuMixerFactory gpu(MixerKind kind, Supplier<GpuMixer> ctor) {
        return new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return kind;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                checkKind(kind, layer);
                return ctor.get();
            }

            @Override
            public String toString() {
                return "DeltaNetBlocks.gpu(" + kind + ")";
            }
        };
    }

    private static CpuMixerFactory cpu(MixerKind kind, Supplier<CpuMixer> ctor) {
        return new CpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return kind;
            }

            @Override
            public CpuMixer create(LayerSpec layer) {
                checkKind(kind, layer);
                return ctor.get();
            }

            @Override
            public String toString() {
                return "DeltaNetBlocks.cpu(" + kind + ")";
            }
        };
    }

    // ================================================================== layer checks

    /**
     * Checks that a layer is of the mixer kind a factory or mixer serves and carries its recurrent spec.
     *
     * @throws IllegalArgumentException otherwise
     */
    static void checkKind(MixerKind kind, LayerSpec layer) {
        if (layer == null || layer.mixerKind() != kind || layer.recurrent() == null
                || layer.recurrent().kind() != kind) {
            throw new IllegalArgumentException(kind + " mixer cannot run layer "
                    + (layer == null ? "null" : describe(layer) + " (" + layer.mixerKind() + ")"));
        }
    }

    /** Layer name used in messages. */
    static String describe(LayerSpec layer) {
        return layer.stack() + "." + layer.il();
    }

    /**
     * Values per token of the mixer input and output: the residual stream width divided by the hyper-connection
     * count, the per-layer width the layer executors pass ({@code n_embd}; glm5-next and qwen4exp run their mixers
     * on one {@code n_embd} row mixed out of the {@code hcMult * n_embd} stream).
     *
     * @throws IllegalStateException when the stream width is not a multiple of the hyper-connection count
     */
    static int mixerWidth(LayerSpec layer) {
        ResidualSpec r = layer.residual();
        if (r.hcMult() < 1 || r.streamWidth() % r.hcMult() != 0) {
            throw new IllegalStateException("layer " + describe(layer) + ": stream width " + r.streamWidth()
                    + " is not a multiple of the hyper-connection count " + r.hcMult());
        }
        return r.streamWidth() / r.hcMult();
    }

    /**
     * Epsilon of the q/k L2 norm ({@code build_gdn_l2_norm}): {@code f_norm_rms_eps} (qwen3next.cpp:501,
     * qwen35.cpp:439, kimi-linear.cpp:345, kimi-k3.cpp:453, bailingmoe3.cpp:282), except the constant {@code 1e-6} of
     * glm5-next (glm5-next.cpp:865).
     */
    static float l2Eps(ModelSpec spec) {
        return spec.arch() == Arch.GLM5_NEXT ? GLM5_NEXT_L2_EPS : spec.hparams().normRmsEps();
    }

    /** Epsilon of the RMS norms inside the mixers ({@code build_norm(..., LLM_NORM_RMS)}: {@code f_norm_rms_eps}). */
    static float rmsEps(ModelSpec spec) {
        return spec.hparams().normRmsEps();
    }

    /**
     * Narrows a size to the kernels' {@code int} range.
     *
     * @throws IllegalStateException naming {@code what} when it does not fit
     */
    static int toInt(long v, String what) {
        if (v < 0 || v > Integer.MAX_VALUE) {
            throw new IllegalStateException(what + " = " + v + " is outside the kernels' 32-bit range");
        }
        return (int) v;
    }

    // ================================================================== GPU tensors

    /**
     * Required matrix {@code <base>.weight} with {@code cols} inputs and {@code rows} outputs.
     *
     * @throws MissingTensorException when it is absent
     * @throws IllegalStateException  when its shape differs
     */
    static GpuWeight gpuMatrix(GpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        GpuWeight w = ws.layer(layer, TensorNames.weight(base));
        checkMatrix(layer, w.name(), w.cols(), w.rows(), cols, rows);
        return w;
    }

    /** {@link #gpuMatrix} for an optional matrix; null when absent. */
    static GpuWeight gpuMatrixOrNull(GpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        GpuWeight w = ws.layerOrNull(layer, TensorNames.weight(base));
        if (w != null) {
            checkMatrix(layer, w.name(), w.cols(), w.rows(), cols, rows);
        }
        return w;
    }

    /**
     * Required matrix {@code <base>.weight} with {@code cols} inputs and any output count.
     *
     * @throws MissingTensorException when it is absent
     * @throws IllegalStateException  when its input width differs
     */
    static GpuWeight gpuMatrixCols(GpuWeights ws, LayerSpec layer, String base, long cols) {
        GpuWeight w = ws.layer(layer, TensorNames.weight(base));
        if (w.cols() != cols) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + w.name() + " reads " + w.cols()
                    + " inputs, the " + layer.mixerKind() + " mixer feeds it " + cols);
        }
        return w;
    }

    /**
     * Tensor stored as {@code <base>.weight} or as bare {@code <base>} (llama.cpp creates {@code ssm_a} without the
     * suffix, LLM_TENSOR_SSM_A_NOSCAN).
     *
     * @throws MissingTensorException naming the bare tensor when neither form exists
     */
    static GpuWeight gpuEither(GpuWeights ws, LayerSpec layer, String base) {
        GpuWeight w = ws.layerOrNull(layer, TensorNames.weight(base));
        if (w == null) {
            w = ws.layerOrNull(layer, base);
        }
        if (w == null) {
            throw new MissingTensorException(ws.archName(), layer.il(), layer.tensorName(base));
        }
        return w;
    }

    /**
     * The tensor as a dense F32 device buffer of {@code floats} values in row order, as the kernels read
     * {@code ssm_conv1d}, {@code ssm_dt.bias} and {@code ssm_a}: its own buffer when it is a single-buffer F32
     * tensor, otherwise a buffer filled at bind time by looking up every row with {@link GpuOps#embed} (the row
     * dequantization of the lookup kernels), added to {@code owned}.
     *
     * @throws IllegalStateException         when the tensor does not hold exactly {@code floats} values
     * @throws UnsupportedOperationException when no lookup kernel exists for its type
     */
    static GpuBuffer gpuF32(GpuForwardContext ctx, GpuWeight w, long floats, List<GpuBuffer> owned) {
        long n = w.rows() * w.cols();
        if (n != floats) {
            throw new IllegalStateException("tensor " + w.name() + " holds " + n + " values, " + floats + " expected");
        }
        if (w.type() == GGMLType.F32 && w.isSingleShard()) {
            return w.shard(0);
        }
        int nRows = toInt(w.rows(), "rows of " + w.name());
        int[] rows = new int[nRows];
        for (int r = 0; r < nRows; r++) {
            rows[r] = r;
        }
        GpuBuffer b = ctx.backend().allocateBuffer(Math.multiplyExact(floats, (long) Float.BYTES));
        owned.add(b);
        ctx.ops().embed(w, rows, b, 0);
        return b;
    }

    /**
     * A norm weight of {@code width} values as the single-buffer F32 {@link GpuWeight} {@link GpuOps#norm} needs:
     * the tensor itself when it already is one, otherwise its {@link #gpuF32} form as a {@code {width}} tensor.
     *
     * @throws IllegalStateException when the tensor does not hold exactly {@code width} values
     */
    static GpuWeight gpuNormWeight(GpuForwardContext ctx, GpuWeight w, int width, List<GpuBuffer> owned) {
        if (w.type() == GGMLType.F32 && w.isSingleShard() && w.cols() == width && w.rows() == 1) {
            return w;
        }
        GpuBuffer b = gpuF32(ctx, w, width, owned);
        return GpuWeight.of(w.name(), GGMLType.F32, new long[]{width}, b);
    }

    private static void checkMatrix(LayerSpec layer, String name, long gotCols, long gotRows, long cols, long rows) {
        if (gotCols != cols || gotRows != rows) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + name + " is " + gotCols + "x"
                    + gotRows + ", the " + layer.mixerKind() + " mixer needs " + cols + "x" + rows);
        }
    }

    /**
     * Checks that a scratch view starts at element 0 of its buffer, as kernels without offset arguments require.
     *
     * @throws IllegalStateException otherwise
     */
    static ScratchSlot atZero(ScratchSlot s, String kernel) {
        if (s.offset() != 0) {
            throw new IllegalStateException(kernel + " takes no offset for " + s.name() + ", which starts at element "
                    + s.offset());
        }
        return s;
    }

    /**
     * Checks a step's input and output against the scratch buffers a mixer works in.
     *
     * @throws IllegalArgumentException when either shares a buffer with a working slot
     */
    static void checkStepViews(String mixer, ScratchSlot in, ScratchSlot out, GpuBuffer[] working) {
        for (GpuBuffer w : working) {
            if (in.buffer() == w || out.buffer() == w) {
                throw new IllegalArgumentException(mixer + " works in the buffer of " + (in.buffer() == w ? in : out)
                        + "; pass the layer input and output in other slots");
            }
        }
    }

    /**
     * CPU twin of {@link #checkStepViews}.
     *
     * @throws IllegalArgumentException when the input or output array is one of the working arrays
     */
    static void checkStepArrays(String mixer, float[] in, float[] out, float[][] working) {
        for (float[] w : working) {
            if (in == w || out == w) {
                throw new IllegalArgumentException(mixer + " works in the " + (in == w ? "input" : "output")
                        + " array; pass the layer input and output in other arrays");
            }
        }
    }

    // ================================================================== CPU tensors

    /**
     * Required matrix {@code <base>.weight} read in place, with {@code cols} inputs and {@code rows} outputs.
     *
     * @throws MissingTensorException when it is absent
     * @throws IllegalStateException  when its shape differs
     */
    static CpuTensorRef cpuMatrix(CpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        String suffix = TensorNames.weight(base);
        CpuTensorRef w = ws.layer(layer, suffix);
        checkMatrix(layer, layer.tensorName(suffix), w.cols(), w.rows(), cols, rows);
        return w;
    }

    /** {@link #cpuMatrix} for an optional matrix; null when absent. */
    static CpuTensorRef cpuMatrixOrNull(CpuWeights ws, LayerSpec layer, String base, long cols, long rows) {
        String suffix = TensorNames.weight(base);
        CpuTensorRef w = ws.layerOrNull(layer, suffix);
        if (w != null) {
            checkMatrix(layer, layer.tensorName(suffix), w.cols(), w.rows(), cols, rows);
        }
        return w;
    }

    /**
     * Required matrix {@code <base>.weight} with {@code cols} inputs and any output count.
     *
     * @throws MissingTensorException when it is absent
     * @throws IllegalStateException  when its input width differs
     */
    static CpuTensorRef cpuMatrixCols(CpuWeights ws, LayerSpec layer, String base, long cols) {
        String suffix = TensorNames.weight(base);
        CpuTensorRef w = ws.layer(layer, suffix);
        if (w.cols() != cols) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + layer.tensorName(suffix) + " reads "
                    + w.cols() + " inputs, the " + layer.mixerKind() + " mixer feeds it " + cols);
        }
        return w;
    }

    /**
     * Required F32 values of the exact tensor {@code suffix} (shared, read only).
     *
     * @throws MissingTensorException when it is absent
     * @throws IllegalStateException  when it does not hold {@code floats} values
     */
    static float[] cpuFloats(CpuWeights ws, LayerSpec layer, String suffix, long floats) {
        float[] v = ws.layerFloatsOrNull(layer, suffix);
        if (v == null) {
            throw new MissingTensorException(ws.archName(), layer.il(), layer.tensorName(suffix));
        }
        return checkLength(layer, suffix, v, floats);
    }

    /**
     * Required F32 values of {@code <base>.weight} or bare {@code <base>} (shared, read only).
     *
     * @throws MissingTensorException naming the bare tensor when neither form exists
     * @throws IllegalStateException  when it does not hold {@code floats} values
     */
    static float[] cpuEither(CpuWeights ws, LayerSpec layer, String base, long floats) {
        String suffix = TensorNames.weight(base);
        float[] v = ws.layerFloatsOrNull(layer, suffix);
        if (v == null) {
            suffix = base;
            v = ws.layerFloatsOrNull(layer, base);
        }
        if (v == null) {
            throw new MissingTensorException(ws.archName(), layer.il(), layer.tensorName(base));
        }
        return checkLength(layer, suffix, v, floats);
    }

    private static float[] checkLength(LayerSpec layer, String suffix, float[] v, long floats) {
        if (v.length != floats) {
            throw new IllegalStateException("layer " + describe(layer) + ": " + layer.tensorName(suffix) + " holds "
                    + v.length + " values, " + floats + " expected");
        }
        return v;
    }

    // ================================================================== recurrent state

    /**
     * Checks the size of a recurrent state tensor of the layer's plan.
     *
     * @throws IllegalStateException when the tensor is missing or its size differs from {@code expected}
     */
    static void checkStateTensor(List<LayerMemoryPlan.StateTensor> tensors, String name, long expected,
                                 LayerSpec layer) {
        for (LayerMemoryPlan.StateTensor t : tensors) {
            if (t.name().equals(name)) {
                if (t.floats() != expected) {
                    throw new IllegalStateException("layer " + describe(layer) + ": recurrent state '" + name
                            + "' holds " + t.floats() + " floats, the " + layer.mixerKind() + " mixer needs "
                            + expected);
                }
                return;
            }
        }
        throw new IllegalStateException("layer " + describe(layer) + " has no recurrent state '" + name + "'");
    }

    /**
     * Device recurrent state of one layer: the conv and scan buffers of the layer's
     * {@link GpuRecurrentState} with the {@code stateRows} input, or, for a model run without memory, zeroed
     * temporaries with the read-only int placeholder as {@code stateRows} (row 0), cleared at the start of every step
     * so each call starts from the zero state llama.cpp uses for memory-less graphs.
     */
    static final class GpuStates {
        /** Conv state ({@code conv[c][j]}), or null when the layer has none. */
        final GpuBuffer conv;
        /** Scan state (delta-rule or lightning matrices). */
        final GpuBuffer scan;
        /** {@code int} buffer whose element 0 is the state row of the sequence. */
        final GpuBuffer rows;
        private final long convFloats;
        private final long scanFloats;
        private final boolean transientState;

        private GpuStates(GpuBuffer conv, GpuBuffer scan, GpuBuffer rows, long convFloats, long scanFloats,
                          boolean transientState) {
            this.conv = conv;
            this.scan = scan;
            this.rows = rows;
            this.convFloats = convFloats;
            this.scanFloats = scanFloats;
            this.transientState = transientState;
        }

        /**
         * @param convFloats conv-state floats per sequence, 0 when the layer has no conv state
         * @param scanFloats scan-state floats per sequence
         * @throws IllegalStateException when the memory has no matching state for the layer
         */
        static GpuStates bind(GpuForwardContext ctx, LayerSpec layer, long convFloats, long scanFloats) {
            toInt(convFloats, "conv state of layer " + describe(layer));
            toInt(scanFloats, "scan state of layer " + describe(layer));
            if (!ctx.hasMemory()) {
                GpuBuffer conv = null;
                if (convFloats > 0) {
                    conv = atZero(ctx.scratch().temp(TEMP_CONV_STATE, convFloats), "the recurrent kernels").buffer();
                }
                GpuBuffer scan = atZero(ctx.scratch().temp(TEMP_SCAN_STATE, scanFloats), "the recurrent kernels")
                        .buffer();
                return new GpuStates(conv, scan, ctx.placeholders().readOnlyInts(), convFloats, scanFloats, true);
            }
            GpuMemory memory = ctx.memory();
            GpuRecurrentState state = memory.recurrent(layer.il());
            if (state == null) {
                throw new IllegalStateException("layer " + describe(layer) + " has no recurrent state");
            }
            GpuBuffer conv = null;
            if (convFloats > 0) {
                checkStateTensor(state.tensors(), LayerMemoryPlan.STATE_CONV, convFloats, layer);
                conv = state.tensor(LayerMemoryPlan.STATE_CONV);
            }
            checkStateTensor(state.tensors(), LayerMemoryPlan.STATE_SCAN, scanFloats, layer);
            GpuBuffer scan = state.tensor(LayerMemoryPlan.STATE_SCAN);
            return new GpuStates(conv, scan, memory.stepInputs().stateRows(), convFloats, scanFloats, false);
        }

        /** Zeroes the temporaries of a memory-less layer; nothing to do when the state lives in memory. */
        void clearTransient(GpuOps ops) {
            if (!transientState) {
                return;
            }
            if (conv != null) {
                ops.fill(conv, 0, convFloats * Float.BYTES, 0);
            }
            ops.fill(scan, 0, scanFloats * Float.BYTES, 0);
        }
    }

    /**
     * Heap recurrent state of one layer, the twin of {@link GpuStates}: the live arrays of the layer's
     * {@link RecurrentStateAccessor}, or zeroed temporaries cleared at every step for a model run without memory.
     */
    static final class CpuStates {
        /** Conv state ({@code conv[c][j]}), or null when the layer has none. */
        final float[] conv;
        /** Scan state (delta-rule or lightning matrices). */
        final float[] scan;
        private final int convFloats;
        private final int scanFloats;
        private final boolean transientState;

        private CpuStates(float[] conv, float[] scan, int convFloats, int scanFloats, boolean transientState) {
            this.conv = conv;
            this.scan = scan;
            this.convFloats = convFloats;
            this.scanFloats = scanFloats;
            this.transientState = transientState;
        }

        /**
         * @param convFloats conv-state floats, 0 when the layer has no conv state
         * @param scanFloats scan-state floats
         * @throws IllegalStateException when the memory has no matching state for the layer
         */
        static CpuStates bind(CpuForwardContext ctx, LayerSpec layer, long convFloats, long scanFloats) {
            int nConv = toInt(convFloats, "conv state of layer " + describe(layer));
            int nScan = toInt(scanFloats, "scan state of layer " + describe(layer));
            if (!ctx.hasMemory()) {
                float[] conv = nConv > 0 ? ctx.scratch().temp(TEMP_CONV_STATE, nConv) : null;
                float[] scan = ctx.scratch().temp(TEMP_SCAN_STATE, nScan);
                return new CpuStates(conv, scan, nConv, nScan, true);
            }
            RecurrentStateAccessor state = ctx.memory().recurrent(layer.il());
            if (state == null) {
                throw new IllegalStateException("layer " + describe(layer) + " has no recurrent state");
            }
            float[] conv = null;
            if (nConv > 0) {
                checkStateTensor(state.tensors(), LayerMemoryPlan.STATE_CONV, nConv, layer);
                conv = state.tensor(LayerMemoryPlan.STATE_CONV);
            }
            checkStateTensor(state.tensors(), LayerMemoryPlan.STATE_SCAN, nScan, layer);
            float[] scan = state.tensor(LayerMemoryPlan.STATE_SCAN);
            return new CpuStates(conv, scan, nConv, nScan, false);
        }

        /** Zeroes the temporaries of a memory-less layer; nothing to do when the state lives in memory. */
        void clearTransient() {
            if (!transientState) {
                return;
            }
            if (conv != null) {
                Arrays.fill(conv, 0, convFloats, 0.0f);
            }
            Arrays.fill(scan, 0, scanFloats, 0.0f);
        }
    }

    // ================================================================== kernel arguments

    /**
     * {@code LLM_SSM_Conv(n_ch, d_conv, n_tokens, flags, x_stride, x_off, out_stride, out_off, state_off,
     * state_stride, x, w, b, conv_state, state_rows, out)}; {@code b} is a placeholder when flags has no bias bit.
     */
    static Object[] convArgs(int nCh, int dConv, int nTokens, int flags, int xStride, int xOff, int outStride,
                             int outOff, int stateOff, int stateStride, GpuBuffer x, GpuBuffer w, GpuBuffer b,
                             GpuBuffer convState, GpuBuffer stateRows, GpuBuffer out) {
        Object[] a = {nCh, dConv, nTokens, flags, xStride, xOff, outStride, outOff, stateOff, stateStride,
                x, w, b, convState, stateRows, out};
        checkCount(a, SsmKernels.CONV_ARG_COUNT, SsmKernels.SSM_CONV);
        return a;
    }

    /**
     * {@code LLM_GDN_GatePrep(n_tokens, n_v_heads, n_k_heads, layout, kda_mode, head_dim, lb, a_stride, a_off,
     * b_stride, b_off, a_in, b_in, dt_bias, ssm_a, g_out, beta_out)}.
     */
    static Object[] gatePrepArgs(CpuGatedDeltaNet.GateShape shape, int nTokens, int aStride, int aOff, int bStride,
                                 int bOff, GpuBuffer aIn, GpuBuffer bIn, GpuBuffer dtBias, GpuBuffer ssmA,
                                 GpuBuffer gOut, GpuBuffer betaOut) {
        Object[] a = {nTokens, shape.nVHeads(), shape.nKHeads(), shape.layout(), shape.kdaMode(), shape.headDim(),
                shape.lb(), aStride, aOff, bStride, bOff, aIn, bIn, dtBias, ssmA, gOut, betaOut};
        checkCount(a, DeltaNetKernels.GATE_ARG_COUNT, DeltaNetKernels.GDN_GATE_PREP);
        return a;
    }

    /**
     * {@code LLM_GDN_Scan(S_k, S_v, n_v_heads, n_k_heads, n_tokens, kv_head_mode, gate_mode, q_stride, q_off,
     * k_stride, k_off, v_stride, v_off, o_stride, o_off, scale, q, k, v, g, beta, state, state_rows, out)}.
     */
    static Object[] scanArgs(CpuGatedDeltaNet.ScanShape shape, int nTokens, int qStride, int qOff, int kStride,
                             int kOff, int vStride, int vOff, int oStride, int oOff, GpuBuffer q, GpuBuffer k,
                             GpuBuffer v, GpuBuffer g, GpuBuffer beta, GpuBuffer state, GpuBuffer stateRows,
                             GpuBuffer out) {
        Object[] a = {shape.sK(), shape.sV(), shape.nVHeads(), shape.nKHeads(), nTokens, shape.headMap().code(),
                shape.gateMode(), qStride, qOff, kStride, kOff, vStride, vOff, oStride, oOff, shape.scale(),
                q, k, v, g, beta, state, stateRows, out};
        checkCount(a, DeltaNetKernels.SCAN_ARG_COUNT, DeltaNetKernels.GDN_SCAN);
        return a;
    }

    /** {@code LLM_LightningAttn(n_tokens, n_head, D, layer_scale, slopes, qkv, state, state_rows, out)}. */
    static Object[] lightningArgs(int nTokens, int nHead, int headDim, float layerScale, GpuBuffer slopes,
                                  GpuBuffer qkv, GpuBuffer state, GpuBuffer stateRows, GpuBuffer out) {
        Object[] a = {nTokens, nHead, headDim, layerScale, slopes, qkv, state, stateRows, out};
        checkCount(a, DeltaNetKernels.LIGHTNING_ARG_COUNT, DeltaNetKernels.LIGHTNING_ATTN);
        return a;
    }

    /** Local size of {@code LLM_GDN_Scan} and {@code LLM_LightningAttn} on this device. */
    static int recurrenceLocal(GpuForwardContext ctx) {
        return DeltaNetKernels.recurrenceLocal(ctx.kernels().preferredMatvecLocal());
    }

    /** 1-D global size of an element-wise launch of {@code items} work-items with local size {@code local}. */
    static long[] global1d(long items, int local) {
        return new long[]{(items + local - 1) / local * local};
    }

    private static void checkCount(Object[] args, int expected, String kernel) {
        if (args.length != expected) {
            throw new IllegalStateException(kernel + " takes " + expected + " arguments, built " + args.length);
        }
    }
}
