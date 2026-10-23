package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * GPU selective-scan mixer of {@link MixerKind#MAMBA1} (mamba, falcon-mamba, jamba), {@link MixerKind#MAMBA2}
 * (mamba2, granitehybrid, nemotron_h, falcon-h1) and {@link MixerKind#PLAMO2_MAMBA}, following
 * llm_build_mamba_base::build_mamba_layer (models/mamba-base.cpp:9-151), build_mamba2_layer
 * (models/mamba-base.cpp:153-304) and build_plamo2_mamba_layer (models/plamo2.cpp:256-424) for one sequence. Per step,
 * with {@code T} tokens:
 *
 * <pre>
 * MAMBA1        in_proj -> Q[T][2*d_inner] = [x | z]
 *               LLM_SSM_Conv(x, bias, SiLU, conv state) -> K[T][d_inner]
 *               x_proj(K) -> V[T][dt_rank + 2*d_state] = [dt | B | C]; LLM_RowNorm on dt, B, C when the layer
 *               normalizes them (ssm_dt_b_c_rms, or all three norm weights present)
 *               dt_proj(dt) -> temp ssm.dt[T][d_inner] (dt rows compacted into ATTN_OUT first when T > 1)
 *               LLM_SSM_Scan(x = K, dt + dt_bias, A {d_state, d_inner}, B, C, scan state) -> ATTN_OUT[T][d_inner]
 *               LLM_MambaGateOut((y + D*x) * silu(z)) -> V; out_proj(V) -> out
 * MAMBA2        in_proj -> Q[T][2*d_inner + 2*n_group*d_state + n_head] = [z | xBC | dt]
 *               LLM_SSM_Conv(xBC, optional bias, SiLU, conv state) -> K[T][d_inner + 2*n_group*d_state]
 *               LLM_SSM_Scan(x, B, C from K; dt from Q + dt_bias; A {1, n_head}) -> ATTN_OUT[T][d_inner]
 *               LLM_MambaGateOut -> V; LLM_GroupRms(ssm_norm) in place when present; out_proj(V) -> out
 * PLAMO2_MAMBA  in_proj -> Q[T][n_head][z_h | x_h]; LLM_Copy2D_F32 de-interleaves x -> K and z -> V
 *               LLM_SSM_Conv(K, SiLU, conv state) -> ATTN_OUT[T][d_inner]
 *               x_proj(ATTN_OUT) -> Q[T][2*d_state + dt_dim] = [B | C | dt]; LLM_RowNorm on B, C, dt
 *               dt_proj(dt) -> temp ssm.dt[T][n_head] (dt rows compacted into K first when T > 1)
 *               LLM_SSM_Scan(x = ATTN_OUT, n_group 1, A {n_head}) -> K; LLM_MambaGateOut(z = V) -> Q
 *               out_proj(Q) -> out
 * </pre>
 *
 * Q, K, V and ATTN_OUT are the well-known scratch slots; the D skip and z gate run in one kernel, the dt bias and
 * softplus are folded into the scan, and every kernel launch except the projections and copies is a
 * {@link PreparedDispatch} bound once whose token count is re-set per step. A falcon-h1 layer
 * ({@link LayerSpec#parallelAttention()}) runs only its Mamba-2 branch here; the layer executor adds the attention
 * branch (models/falcon-h1.cpp:133-163).
 *
 * <p>State layouts: conv {@code conv[c][j]} with {@code dConv - 1} steps, scan
 * {@code s[(h*head_dim + p)*d_state + n]}, both in row {@code stateRows[0]} of the layer's recurrent state. Not
 * thread-safe.</p>
 */
public final class GpuMambaMixer implements GpuMixer {

    /** Temporary holding the dt projection. */
    static final String TEMP_DT = "ssm.dt";

    private static final int EL_LOCAL = SsmKernels.ELEMENTWISE_LOCAL;

    private final MixerKind kind;
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private GpuOps ops;
    private int ubatch;
    private int reduceLocal;
    private int nEmbd;
    private int dInner;
    private int dState;
    private int nHead;
    private int headDim;
    private int nGroup;
    private int dtRank;
    private int convCh;
    private int inWidth;
    private int xdbWidth;
    private boolean dtBcNorms;

    private GpuWeight ssmIn;
    private GpuWeight ssmX;
    private GpuWeight ssmDt;
    private GpuWeight ssmOut;
    /** LoRA deltas of {@code ssm_in}, {@code ssm_x}, {@code ssm_dt} and {@code ssm_out} ({@code build_lora_mm}). */
    private GpuProjectionDeltas.Site inDeltas;
    private GpuProjectionDeltas.Site xDeltas;
    private GpuProjectionDeltas.Site dtDeltas;
    private GpuProjectionDeltas.Site outDeltas;

    private ScratchSlot q;
    private ScratchSlot k;
    private ScratchSlot v;
    private ScratchSlot attnOut;
    private ScratchSlot dt;

    private PreparedDispatch conv;
    private PreparedDispatch scan;
    private PreparedDispatch gate;
    private PreparedDispatch normDt;
    private PreparedDispatch normB;
    private PreparedDispatch normC;
    private PreparedDispatch groupNorm;

    private boolean bound;
    private boolean closed;

    /**
     * @param kind MAMBA1, MAMBA2 or PLAMO2_MAMBA
     * @throws IllegalArgumentException for another kind
     */
    GpuMambaMixer(MixerKind kind) {
        this.kind = Objects.requireNonNull(kind, "kind");
        if (kind != MixerKind.MAMBA1 && kind != MixerKind.MAMBA2 && kind != MixerKind.PLAMO2_MAMBA) {
            throw new IllegalArgumentException("GpuMambaMixer does not serve " + kind);
        }
    }

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException      when the layer is not of this mixer's kind
     * @throws IllegalStateException         when the model has no recurrent state for the layer, a tensor or state
     *                                       has the wrong size, a scratch slot is too small, or the mixer is bound
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException(kind + " mixer is already bound or closed");
        }
        SsmBlocks.checkKind(kind, layer);
        try {
            bindChecked(ctx, layer);
            bound = true;
        } catch (RuntimeException e) {
            closeQuietly(e);
            throw e;
        }
    }

    private void bindChecked(GpuForwardContext ctx, LayerSpec layer) {
        RecurrentSpec r = layer.recurrent();
        ops = ctx.ops();
        GpuScratch scratch = ctx.scratch();
        ubatch = scratch.ubatch();
        reduceLocal = Math.min(ops.local(), LlmKernelConstants.LLM_DIMX);
        NormKernels.checkReduceLocal(reduceLocal);
        nEmbd = r.nEmbd();
        if (layer.residual().streamWidth() != nEmbd) {
            throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + ": " + kind + " expects a stream of "
                    + nEmbd + " floats, the layer streams " + layer.residual().streamWidth());
        }
        dimensions(r);
        SsmKernels.ScanGeometry geo = SsmKernels.ScanGeometry.of(dState, headDim, nHead, nGroup);
        SsmKernels.checkConvShape(convCh, r.dConv());

        GpuMemory memory = ctx.memory();
        GpuRecurrentState state = memory.recurrent(layer.il());
        if (state == null) {
            throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + " has no recurrent state");
        }
        int convStride = (int) SsmBlocks.stateFloats(state.tensors(), LayerMemoryPlan.STATE_CONV,
                (long) (r.dConv() - 1) * convCh, layer);
        SsmBlocks.stateFloats(state.tensors(), LayerMemoryPlan.STATE_SCAN, geo.stateFloats(), layer);
        GpuBuffer convState = state.tensor(LayerMemoryPlan.STATE_CONV);
        GpuBuffer scanState = state.tensor(LayerMemoryPlan.STATE_SCAN);
        GpuBuffer stateRows = memory.stepInputs().stateRows();
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();

        q = slot(scratch, ScratchSlots.Q, Math.max(inWidth, Math.max(xdbWidth, dInner)));
        k = slot(scratch, ScratchSlots.K, Math.max(convCh, Math.max(dInner, dtRank)));
        v = slot(scratch, ScratchSlots.V, Math.max(dInner, xdbWidth));
        attnOut = slot(scratch, ScratchSlots.ATTN_OUT, Math.max(dInner, dtRank));
        if (kind != MixerKind.MAMBA2) {
            dt = scratch.temp(TEMP_DT, (long) ubatch * nHead);
        }

        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        float eps = layer.norm().eps();
        ssmIn = SsmBlocks.gpuMatrix(ws, layer, TensorNames.SSM_IN, nEmbd, inWidth);
        ssmOut = SsmBlocks.gpuMatrix(ws, layer, TensorNames.SSM_OUT, dInner, nEmbd);
        inDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.SSM_IN), ssmIn);
        outDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.SSM_OUT), ssmOut);
        GpuBuffer convW = SsmBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.weight(TensorNames.SSM_CONV1D)),
                (long) convCh * r.dConv(), owned);
        GpuBuffer convB = null;
        if (kind != MixerKind.PLAMO2_MAMBA && r.convBias()) {
            convB = SsmBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.bias(TensorNames.SSM_CONV1D)), convCh, owned);
        }
        GpuBuffer dtBias = SsmBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.bias(TensorNames.SSM_DT)), nHead, owned);
        boolean aPerState = kind == MixerKind.MAMBA1;
        GpuBuffer a = SsmBlocks.gpuF32(ctx, SsmBlocks.gpuEither(ws, layer, TensorNames.SSM_A),
                aPerState ? (long) nHead * dState : nHead, owned);
        GpuBuffer d = SsmBlocks.gpuF32(ctx, SsmBlocks.gpuEither(ws, layer, TensorNames.SSM_D), nHead, owned);

        switch (kind) {
            case MAMBA1:
                bindMamba1(ctx, ws, layer, r, eps, ph);
                break;
            case MAMBA2:
                bindMamba2(ctx, ws, layer, eps);
                break;
            default:
                bindPlamo2(ctx, ws, layer, eps, ph);
                break;
        }
        prepareConv(r.dConv(), convW, convB, ph, convState, convStride, stateRows);
        prepareScan(geo, aPerState, dtBias, a, scanState, stateRows);
        prepareGate(d);
    }

    /** Kind-specific dimensions; scan heads, groups and conv channels as the kernels see them. */
    private void dimensions(RecurrentSpec r) {
        dInner = r.dInner();
        dState = r.dState();
        switch (kind) {
            case MAMBA1:
                nHead = dInner;
                headDim = 1;
                nGroup = 1;
                dtRank = r.dtRank();
                convCh = dInner;
                inWidth = 2 * dInner;
                xdbWidth = dtRank + 2 * dState;
                break;
            case MAMBA2:
                nHead = r.nHead();
                headDim = r.headDim();
                nGroup = r.nGroup();
                dtRank = 0;
                convCh = dInner + 2 * nGroup * dState;
                inWidth = 2 * dInner + 2 * nGroup * dState + nHead;
                xdbWidth = 0;
                break;
            default:
                nHead = r.nHead();
                headDim = r.headDim();
                nGroup = 1;
                dtRank = r.dtRank();
                convCh = dInner;
                inWidth = 2 * dInner;
                xdbWidth = 2 * dState + dtRank;
                break;
        }
        if (kind != MixerKind.MAMBA2 && dtRank <= 0) {
            throw new IllegalStateException(kind + " mixer needs a positive dt rank, got " + dtRank);
        }
        if (convCh != r.convChannels()) {
            throw new IllegalStateException(kind + " mixer convolves " + convCh + " channels, the recurrent spec "
                    + r.convChannels() + " (n_group " + r.nGroup() + ")");
        }
    }

    /**
     * x_proj, dt_proj, and the dt/B/C norms of Mamba-1 when the layer normalizes them
     * (models/mamba-base.cpp:88-109).
     */
    private void bindMamba1(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, RecurrentSpec r, float eps,
                            GpuBuffer ph) {
        ssmX = SsmBlocks.gpuMatrix(ws, layer, TensorNames.SSM_X, dInner, xdbWidth);
        ssmDt = SsmBlocks.gpuMatrix(ws, layer, TensorNames.SSM_DT, dtRank, dInner);
        serveXDt(ctx, layer);
        GpuWeight dtN = SsmBlocks.gpuEitherOrNull(ws, layer, TensorNames.SSM_DT_NORM);
        GpuWeight bN = SsmBlocks.gpuEitherOrNull(ws, layer, TensorNames.SSM_B_NORM);
        GpuWeight cN = SsmBlocks.gpuEitherOrNull(ws, layer, TensorNames.SSM_C_NORM);
        dtBcNorms = r.dtBcRms() || (dtN != null && bN != null && cN != null);
        if (dtBcNorms) {
            normDt = prepareRowNorm(ctx, dtN, eps, dtRank, v, 0, ph);
            normB = prepareRowNorm(ctx, bN, eps, dState, v, dtRank, ph);
            normC = prepareRowNorm(ctx, cN, eps, dState, v, dtRank + dState, ph);
        }
    }

    private void serveXDt(GpuForwardContext ctx, LayerSpec layer) {
        xDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.SSM_X), ssmX);
        dtDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.SSM_DT), ssmDt);
    }

    /** The grouped RMS norm of the gated output when the layer has {@code ssm_norm} (models/mamba-base.cpp:290-294). */
    private void bindMamba2(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, float eps) {
        GpuWeight norm = ws.layerOrNull(layer, TensorNames.weight(TensorNames.SSM_NORM));
        if (norm != null) {
            GpuBuffer w = SsmBlocks.gpuF32(ctx, norm, dInner, owned);
            requireZeroOffset(v);
            Object[] args = NormKernels.groupRmsArgs(ubatch, dInner, nGroup, eps, v.buffer(), 0, dInner, w,
                    v.buffer());
            args[NormKernels.ARG_GROUP_RMS_N_TOKENS] = 1;
            groupNorm = prepare(NormKernels.GROUP_RMS,
                    NormKernels.reduceGlobal(NormKernels.groupRmsGroups(1, nGroup), reduceLocal), reduceLocal, args);
        }
    }

    /** x_proj, dt_proj and the always-present B, C and dt norms of PLaMo-2 (models/plamo2.cpp:336-363). */
    private void bindPlamo2(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, float eps, GpuBuffer ph) {
        ssmX = SsmBlocks.gpuMatrix(ws, layer, TensorNames.SSM_X, dInner, xdbWidth);
        ssmDt = SsmBlocks.gpuMatrix(ws, layer, TensorNames.SSM_DT, dtRank, nHead);
        serveXDt(ctx, layer);
        dtBcNorms = true;
        normB = prepareRowNorm(ctx, SsmBlocks.gpuEither(ws, layer, TensorNames.SSM_B_NORM), eps, dState, q, 0, ph);
        normC = prepareRowNorm(ctx, SsmBlocks.gpuEither(ws, layer, TensorNames.SSM_C_NORM), eps, dState, q, dState,
                ph);
        normDt = prepareRowNorm(ctx, SsmBlocks.gpuEither(ws, layer, TensorNames.SSM_DT_NORM), eps, dtRank, q,
                2 * dState, ph);
    }

    /**
     * In-place RMS norm of one field of the x_proj rows ({@code len} floats at {@code fieldOff} of every
     * {@code xdbWidth}-float row of {@code rows}), weighted when {@code w} is present.
     */
    private PreparedDispatch prepareRowNorm(GpuForwardContext ctx, GpuWeight w, float eps, int len, ScratchSlot rows,
                                            int fieldOff, GpuBuffer ph) {
        GpuBuffer wb = w != null ? SsmBlocks.gpuF32(ctx, w, len, owned) : null;
        int off = rows.offset() + fieldOff;
        Object[] args = NormKernels.rowNormArgs(NormKind.RMS, eps, ubatch, len, xdbWidth, rows.buffer(), off,
                rows.buffer(), off, wb, null, 1, ph);
        args[NormKernels.ARG_ROW_NORM_N_ROWS] = 1;
        return prepare(NormKernels.ROW_NORM, NormKernels.reduceGlobal(NormKernels.rowNormGroups(1), reduceLocal),
                reduceLocal, args);
    }

    /** {@link SsmKernels#SSM_CONV}: MAMBA1 / MAMBA2 read Q, PLaMo-2 reads the de-interleaved x in K. */
    private void prepareConv(int dConv, GpuBuffer w, GpuBuffer b, GpuBuffer ph, GpuBuffer convState, int convStride,
                             GpuBuffer stateRows) {
        ScratchSlot src;
        int xOff;
        int xStride;
        ScratchSlot dst;
        switch (kind) {
            case MAMBA1:
                src = q;
                xOff = q.offset();
                xStride = inWidth;
                dst = k;
                break;
            case MAMBA2:
                src = q;
                xOff = q.offset() + dInner;
                xStride = inWidth;
                dst = k;
                break;
            default:
                src = k;
                xOff = k.offset();
                xStride = dInner;
                dst = attnOut;
                break;
        }
        int flags = SsmKernels.CONV_FLAG_SILU | (b != null ? SsmKernels.CONV_FLAG_BIAS : 0);
        Object[] args = {convCh, dConv, 1, flags, xStride, xOff, convCh, dst.offset(), 0, convStride,
            src.buffer(), w, b != null ? b : ph, convState, stateRows, dst.buffer()};
        checkCount(SsmKernels.SSM_CONV, args, SsmKernels.CONV_ARG_COUNT);
        conv = prepare(SsmKernels.SSM_CONV, new long[]{SsmKernels.elementwiseGlobal(convCh, EL_LOCAL)}, EL_LOCAL,
                args);
    }

    /** {@link SsmKernels#SSM_SCAN} over the conv output, writing y contiguously ({@code y_stride = d_inner}). */
    private void prepareScan(SsmKernels.ScanGeometry geo, boolean aPerState, GpuBuffer dtBias, GpuBuffer a,
                             GpuBuffer scanState, GpuBuffer stateRows) {
        ScratchSlot x;
        int xStride;
        GpuBuffer dtBuf;
        int dtOff;
        int dtStride;
        ScratchSlot bc;
        int bcStride;
        int bOff;
        ScratchSlot y;
        switch (kind) {
            case MAMBA1:
                x = k;
                xStride = dInner;
                dtBuf = dt.buffer();
                dtOff = dt.offset();
                dtStride = dInner;
                bc = v;
                bcStride = xdbWidth;
                bOff = v.offset() + dtRank;
                y = attnOut;
                break;
            case MAMBA2:
                x = k;
                xStride = convCh;
                dtBuf = q.buffer();
                dtOff = q.offset() + 2 * dInner + 2 * nGroup * dState;
                dtStride = inWidth;
                bc = k;
                bcStride = convCh;
                bOff = k.offset() + dInner;
                y = attnOut;
                break;
            default:
                x = attnOut;
                xStride = dInner;
                dtBuf = dt.buffer();
                dtOff = dt.offset();
                dtStride = nHead;
                bc = q;
                bcStride = xdbWidth;
                bOff = q.offset();
                y = k;
                break;
        }
        requireZeroOffset(y);
        int cOff = bOff + nGroup * dState;
        Object[] args = {dState, headDim, nHead, nGroup, 1, aPerState ? 1 : 0, geo.channelsPerGroup(),
            xStride, x.offset(), dtStride, dtOff, bcStride, bcStride, bOff, cOff, dInner, 1,
            x.buffer(), dtBuf, dtBias, a, bc.buffer(), bc.buffer(), scanState, stateRows, y.buffer()};
        checkCount(SsmKernels.SSM_SCAN, args, SsmKernels.SCAN_ARG_COUNT);
        scan = prepare(SsmKernels.SSM_SCAN, new long[]{geo.globalSize()}, geo.localSize(), args);
    }

    /** {@link SsmKernels#MAMBA_GATE_OUT}: {@code (y + D*x) * silu(z)} into a slot distinct from y, x and z. */
    private void prepareGate(GpuBuffer d) {
        ScratchSlot y;
        ScratchSlot x;
        int xStride;
        ScratchSlot z;
        int zOff;
        int zStride;
        ScratchSlot out;
        switch (kind) {
            case MAMBA1:
                y = attnOut;
                x = k;
                xStride = dInner;
                z = q;
                zOff = q.offset() + dInner;
                zStride = inWidth;
                out = v;
                break;
            case MAMBA2:
                y = attnOut;
                x = k;
                xStride = convCh;
                z = q;
                zOff = q.offset();
                zStride = inWidth;
                out = v;
                break;
            default:
                y = k;
                x = attnOut;
                xStride = dInner;
                z = v;
                zOff = v.offset();
                zStride = dInner;
                out = q;
                break;
        }
        requireZeroOffset(out);
        Object[] args = {dInner, dInner, headDim, xStride, x.offset(), y.offset(), zStride, zOff,
            y.buffer(), x.buffer(), d, z.buffer(), out.buffer()};
        checkCount(SsmKernels.MAMBA_GATE_OUT, args, SsmKernels.GATE_ARG_COUNT);
        gate = prepare(SsmKernels.MAMBA_GATE_OUT, new long[]{SsmKernels.elementwiseGlobal(dInner, EL_LOCAL)}, EL_LOCAL,
                args);
    }

    private PreparedDispatch prepare(String kernel, long[] global, int local, Object[] args) {
        PreparedDispatch p = ops.prepare(kernel, global, new long[]{local}, args);
        prepared.add(p);
        return p;
    }

    private ScratchSlot slot(GpuScratch scratch, String name, int width) {
        return scratch.slot(name).require((long) ubatch * width);
    }

    private static void requireZeroOffset(ScratchSlot s) {
        if (s.offset() != 0) {
            throw new IllegalStateException("scratch slot " + s.name() + " starts at element " + s.offset()
                    + "; the SSM kernels write it from element 0");
        }
    }

    private static void checkCount(String kernel, Object[] args, int expected) {
        if (args.length != expected) {
            throw new IllegalStateException(kernel + " takes " + expected + " arguments, " + args.length + " built");
        }
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        forward(1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        forward(nTokens, in, out);
    }

    private void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
        if (!bound || closed) {
            throw new IllegalStateException(kind + " mixer is " + (closed ? "closed" : "not bound"));
        }
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException(kind + " step of " + nTokens + " tokens is outside [1, " + ubatch + "]");
        }
        long n = (long) nTokens * nEmbd;
        in.require(n);
        out.require(n);
        checkExternal(in);
        checkExternal(out);
        switch (kind) {
            case MAMBA1:
                stepMamba1(nTokens, in, out);
                break;
            case MAMBA2:
                stepMamba2(nTokens, in, out);
                break;
            default:
                stepPlamo2(nTokens, in, out);
                break;
        }
    }

    private void stepMamba1(int t, ScratchSlot in, ScratchSlot out) {
        projectIn(t, in);
        runTokens(conv, SsmKernels.ARG_CONV_N_TOKENS, t);
        ops.matmul(ssmX, k.buffer(), k.offset(), v.buffer(), v.offset(), t);
        xDeltas.applyRows(t, k.buffer(), k.offset(), v.buffer(), v.offset());
        if (dtBcNorms) {
            runRows(normDt, t);
            runRows(normB, t);
            runRows(normC, t);
        }
        projectDt(t, v, attnOut);
        runTokens(scan, SsmKernels.ARG_SCAN_N_TOKENS, t);
        runElements(gate, t * dInner);
        projectOut(t, v, out);
    }

    private void stepMamba2(int t, ScratchSlot in, ScratchSlot out) {
        projectIn(t, in);
        runTokens(conv, SsmKernels.ARG_CONV_N_TOKENS, t);
        runTokens(scan, SsmKernels.ARG_SCAN_N_TOKENS, t);
        runElements(gate, t * dInner);
        if (groupNorm != null) {
            groupNorm.setInt(NormKernels.ARG_GROUP_RMS_N_TOKENS, t);
            groupNorm.setGlobal(NormKernels.reduceGlobal(NormKernels.groupRmsGroups(t, nGroup), reduceLocal));
            groupNorm.dispatch();
        }
        projectOut(t, v, out);
    }

    private void stepPlamo2(int t, ScratchSlot in, ScratchSlot out) {
        projectIn(t, in);
        int rows = t * nHead;
        ops.copy2d(q.buffer(), q.offset() + headDim, 2 * headDim, k.buffer(), k.offset(), headDim, rows, headDim);
        ops.copy2d(q.buffer(), q.offset(), 2 * headDim, v.buffer(), v.offset(), headDim, rows, headDim);
        runTokens(conv, SsmKernels.ARG_CONV_N_TOKENS, t);
        ops.matmul(ssmX, attnOut.buffer(), attnOut.offset(), q.buffer(), q.offset(), t);
        xDeltas.applyRows(t, attnOut.buffer(), attnOut.offset(), q.buffer(), q.offset());
        runRows(normB, t);
        runRows(normC, t);
        runRows(normDt, t);
        projectDt(t, q, k);
        runTokens(scan, SsmKernels.ARG_SCAN_N_TOKENS, t);
        runElements(gate, t * dInner);
        projectOut(t, q, out);
    }

    /** {@code ssm_in} of the step's rows into {@code q}, plus its deltas. */
    private void projectIn(int t, ScratchSlot in) {
        ops.matmul(ssmIn, in.buffer(), in.offset(), q.buffer(), q.offset(), t);
        inDeltas.applyRows(t, in.buffer(), in.offset(), q.buffer(), q.offset());
    }

    /** {@code ssm_out} of the gated rows {@code y} into {@code out}, plus its deltas. */
    private void projectOut(int t, ScratchSlot y, ScratchSlot out) {
        ops.matmul(ssmOut, y.buffer(), y.offset(), out.buffer(), out.offset(), t);
        outDeltas.applyRows(t, y.buffer(), y.offset(), out.buffer(), out.offset());
    }

    /**
     * dt_proj over the dt field of the x_proj rows in {@code xdb} into the dt temporary. One row is read in place;
     * several rows are first compacted into {@code compact} because the projections take dense input rows.
     */
    private void projectDt(int t, ScratchSlot xdb, ScratchSlot compact) {
        int field = kind == MixerKind.MAMBA1 ? 0 : 2 * dState;
        if (t == 1) {
            ops.matmul(ssmDt, xdb.buffer(), xdb.offset() + field, dt.buffer(), dt.offset(), 1);
            dtDeltas.applyRows(1, xdb.buffer(), xdb.offset() + field, dt.buffer(), dt.offset());
            return;
        }
        ops.copy2d(xdb.buffer(), xdb.offset() + field, xdbWidth, compact.buffer(), compact.offset(), dtRank, t, dtRank);
        ops.matmul(ssmDt, compact.buffer(), compact.offset(), dt.buffer(), dt.offset(), t);
        dtDeltas.applyRows(t, compact.buffer(), compact.offset(), dt.buffer(), dt.offset());
    }

    private static void runTokens(PreparedDispatch d, int argIndex, int nTokens) {
        d.setInt(argIndex, nTokens);
        d.dispatch();
    }

    private void runRows(PreparedDispatch d, int nRows) {
        d.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, nRows);
        d.setGlobal(NormKernels.reduceGlobal(NormKernels.rowNormGroups(nRows), reduceLocal));
        d.dispatch();
    }

    private static void runElements(PreparedDispatch d, int n) {
        d.setInt(SsmKernels.ARG_GATE_N, n);
        d.setGlobal(SsmKernels.elementwiseGlobal(n, EL_LOCAL));
        d.dispatch();
    }

    /** Rejects a caller view living in one of this mixer's working buffers. */
    private void checkExternal(ScratchSlot s) {
        GpuBuffer b = s.buffer();
        if (b == q.buffer() || b == k.buffer() || b == v.buffer() || b == attnOut.buffer()
                || (dt != null && b == dt.buffer())) {
            throw new IllegalArgumentException(kind + " mixer input/output " + s + " shares a buffer with its working"
                    + " slots");
        }
    }

    // ================================================================== lifecycle

    /** Releases the prepared dispatches and the F32 copies made at bind. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = release();
        if (failure != null) {
            throw failure;
        }
    }

    private void closeQuietly(RuntimeException cause) {
        closed = true;
        RuntimeException failure = release();
        if (failure != null) {
            cause.addSuppressed(failure);
        }
    }

    private RuntimeException release() {
        RuntimeException first = Closeables.closeAll(prepared, null);
        prepared.clear();
        first = Closeables.closeAll(owned, first);
        owned.clear();
        return first;
    }

    @Override
    public String toString() {
        return "GpuMambaMixer{" + kind + (bound ? ", d_inner=" + dInner + ", d_state=" + dState + ", heads=" + nHead
                + "x" + headDim + ", groups=" + nGroup : ", unbound") + (closed ? ", closed" : "") + '}';
    }
}
