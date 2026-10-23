package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU twin of {@link GpuMambaMixer}: the Mamba-1, Mamba-2 and PLaMo-2 selective-scan mixers
 * (models/mamba-base.cpp:9-304, models/plamo2.cpp:256-424) for one sequence, step for step on the same working layout:
 *
 * <pre>
 * MAMBA1        in_proj -> Q[T][2*d_inner]; CpuSsmConv(x, bias, SiLU) -> K; x_proj(K) -> V[T][dt_rank + 2*d_state];
 *               RMS norms on dt, B, C when the layer normalizes them; dt_proj(dt rows of V) -> ssm.dt[T][d_inner];
 *               CpuSsmScan.scan -> ATTN_OUT; gateOut(z = Q + d_inner) -> V; out_proj(V) -> out
 * MAMBA2        in_proj -> Q[T][2*d_inner + 2*n_group*d_state + n_head]; CpuSsmConv(xBC) -> K;
 *               scan(x, B, C from K, dt from Q) -> ATTN_OUT; gateOut(z = Q) -> V; groupRms(ssm_norm) in place;
 *               out_proj(V) -> out
 * PLAMO2_MAMBA  in_proj -> Q[T][n_head][z_h | x_h]; de-interleave x -> K, z -> V; CpuSsmConv(K, SiLU) -> ATTN_OUT;
 *               x_proj(ATTN_OUT) -> Q[T][2*d_state + dt_dim]; RMS norms on B, C, dt; dt_proj -> ssm.dt[T][n_head];
 *               scan(x = ATTN_OUT) -> K; gateOut(z = V) -> Q; out_proj(Q) -> out
 * </pre>
 *
 * A falcon-h1 layer runs only its Mamba-2 branch here; the layer executor adds the attention branch
 * (models/falcon-h1.cpp:133-163). Conv and scan state live in row 0 of the layer's recurrent state with the layouts of
 * {@link GpuMambaMixer}. Every working array comes from {@link CpuScratch}; a step allocates nothing. Not thread-safe.
 */
public final class CpuMambaMixer implements CpuMixer {

    private final MixerKind kind;

    private CpuMatVec mv;
    private int ubatch;
    private int nEmbd;
    private int dInner;
    private int dState;
    private int nHead;
    private int headDim;
    private int nGroup;
    private int dtRank;
    private int dConv;
    private int convCh;
    private int inWidth;
    private int xdbWidth;
    private int convFlags;
    private float eps;

    private CpuTensorRef ssmIn;
    private CpuTensorRef ssmX;
    private CpuTensorRef ssmDt;
    private CpuTensorRef ssmOut;
    /** LoRA deltas of {@code ssm_in}, {@code ssm_x}, {@code ssm_dt} and {@code ssm_out} ({@code build_lora_mm}). */
    private CpuProjectionDeltas.Site inDeltas;
    private CpuProjectionDeltas.Site xDeltas;
    private CpuProjectionDeltas.Site dtDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private float[] convW;
    private float[] convB;
    private float[] dtBias;
    private float[] a;
    private float[] d;
    private boolean dtBcNorms;
    private float[] dtNorm;
    private float[] bNorm;
    private float[] cNorm;
    private float[] groupNorm;

    private float[] convState;
    private float[] scanState;
    private float[] q;
    private float[] k;
    private float[] v;
    private float[] attnOut;
    private float[] dt;

    private boolean bound;
    private boolean closed;

    /**
     * @param kind MAMBA1, MAMBA2 or PLAMO2_MAMBA
     * @throws IllegalArgumentException for another kind
     */
    CpuMambaMixer(MixerKind kind) {
        this.kind = Objects.requireNonNull(kind, "kind");
        if (kind != MixerKind.MAMBA1 && kind != MixerKind.MAMBA2 && kind != MixerKind.PLAMO2_MAMBA) {
            throw new IllegalArgumentException("CpuMambaMixer does not serve " + kind);
        }
    }

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException when the layer is not of this mixer's kind
     * @throws IllegalStateException    when the model has no recurrent state for the layer, a tensor or state has the
     *                                  wrong size, a scratch slot is too small, or the mixer is bound or closed
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException(kind + " mixer is already bound or closed");
        }
        SsmBlocks.checkKind(kind, layer);
        RecurrentSpec r = layer.recurrent();
        mv = ctx.matVec();
        CpuScratch scratch = ctx.scratch();
        ubatch = scratch.ubatch();
        nEmbd = r.nEmbd();
        if (layer.residual().streamWidth() != nEmbd) {
            throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + ": " + kind + " expects a stream of "
                    + nEmbd + " floats, the layer streams " + layer.residual().streamWidth());
        }
        dimensions(r);
        SsmKernels.checkScanShape(dState, headDim, nHead, nGroup);
        SsmKernels.checkConvShape(convCh, dConv);
        eps = layer.norm().eps();

        if (!ctx.hasMemory()) {
            throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + " needs recurrent memory");
        }
        RecurrentStateAccessor state = ctx.memory().recurrent(layer.il());
        if (state == null) {
            throw new IllegalStateException("layer " + SsmBlocks.describe(layer) + " has no recurrent state");
        }
        SsmBlocks.stateFloats(state.tensors(), LayerMemoryPlan.STATE_CONV, (long) (dConv - 1) * convCh, layer);
        SsmBlocks.stateFloats(state.tensors(), LayerMemoryPlan.STATE_SCAN, (long) nHead * headDim * dState, layer);
        convState = state.tensor(LayerMemoryPlan.STATE_CONV);
        scanState = state.tensor(LayerMemoryPlan.STATE_SCAN);

        q = slot(scratch, CpuScratch.Q, Math.max(inWidth, Math.max(xdbWidth, dInner)));
        k = slot(scratch, CpuScratch.K, Math.max(convCh, dInner));
        v = slot(scratch, CpuScratch.V, Math.max(dInner, xdbWidth));
        attnOut = slot(scratch, CpuScratch.ATTN_OUT, dInner);
        if (kind != MixerKind.MAMBA2) {
            dt = scratch.temp(GpuMambaMixer.TEMP_DT, Math.multiplyExact(ubatch, nHead));
        }

        bindWeights(ctx.weights(), layer, r);
        bindDeltas(ctx.projectionDeltas(), layer);
        bound = true;
    }

    private void dimensions(RecurrentSpec r) {
        dInner = r.dInner();
        dState = r.dState();
        dConv = r.dConv();
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

    private void bindDeltas(CpuProjectionDeltas deltas, LayerSpec layer) {
        inDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_IN), ssmIn, mv);
        outDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_OUT), ssmOut, mv);
        if (ssmX != null) {
            xDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_X), ssmX, mv);
            dtDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.SSM_DT), ssmDt, mv);
        }
    }

    private void bindWeights(CpuWeights ws, LayerSpec layer, RecurrentSpec r) {
        ssmIn = SsmBlocks.cpuMatrix(ws, layer, TensorNames.SSM_IN, nEmbd, inWidth);
        ssmOut = SsmBlocks.cpuMatrix(ws, layer, TensorNames.SSM_OUT, dInner, nEmbd);
        convW = SsmBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.SSM_CONV1D), (long) convCh * dConv,
                true);
        convB = kind != MixerKind.PLAMO2_MAMBA && r.convBias()
                ? SsmBlocks.cpuFloats(ws, layer, TensorNames.bias(TensorNames.SSM_CONV1D), convCh, true) : null;
        convFlags = SsmKernels.CONV_FLAG_SILU | (convB != null ? SsmKernels.CONV_FLAG_BIAS : 0);
        dtBias = SsmBlocks.cpuFloats(ws, layer, TensorNames.bias(TensorNames.SSM_DT), nHead, true);
        a = SsmBlocks.cpuEither(ws, layer, TensorNames.SSM_A,
                kind == MixerKind.MAMBA1 ? (long) nHead * dState : nHead);
        d = SsmBlocks.cpuEither(ws, layer, TensorNames.SSM_D, nHead);
        switch (kind) {
            case MAMBA1:
                ssmX = SsmBlocks.cpuMatrix(ws, layer, TensorNames.SSM_X, dInner, xdbWidth);
                ssmDt = SsmBlocks.cpuMatrix(ws, layer, TensorNames.SSM_DT, dtRank, dInner);
                dtNorm = SsmBlocks.cpuEitherOrNull(ws, layer, TensorNames.SSM_DT_NORM, dtRank);
                bNorm = SsmBlocks.cpuEitherOrNull(ws, layer, TensorNames.SSM_B_NORM, dState);
                cNorm = SsmBlocks.cpuEitherOrNull(ws, layer, TensorNames.SSM_C_NORM, dState);
                dtBcNorms = r.dtBcRms() || (dtNorm != null && bNorm != null && cNorm != null);
                break;
            case MAMBA2:
                groupNorm = SsmBlocks.cpuFloats(ws, layer, TensorNames.weight(TensorNames.SSM_NORM), dInner, false);
                break;
            default:
                ssmX = SsmBlocks.cpuMatrix(ws, layer, TensorNames.SSM_X, dInner, xdbWidth);
                ssmDt = SsmBlocks.cpuMatrix(ws, layer, TensorNames.SSM_DT, dtRank, nHead);
                dtNorm = SsmBlocks.cpuEither(ws, layer, TensorNames.SSM_DT_NORM, dtRank);
                bNorm = SsmBlocks.cpuEither(ws, layer, TensorNames.SSM_B_NORM, dState);
                cNorm = SsmBlocks.cpuEither(ws, layer, TensorNames.SSM_C_NORM, dState);
                dtBcNorms = true;
                break;
        }
    }

    private float[] slot(CpuScratch scratch, String name, int width) {
        float[] s = scratch.slot(name);
        long need = (long) ubatch * width;
        if (s.length < need) {
            throw new IllegalStateException(kind + " mixer needs " + need + " floats in scratch slot " + name
                    + ", it holds " + s.length);
        }
        return s;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, float[] in, float[] out) {
        forward(1, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        forward(n, in, out);
    }

    private void forward(int t, float[] in, float[] out) {
        if (!bound || closed) {
            throw new IllegalStateException(kind + " mixer is " + (closed ? "closed" : "not bound"));
        }
        if (t < 1 || t > ubatch) {
            throw new IllegalArgumentException(kind + " step of " + t + " tokens is outside [1, " + ubatch + "]");
        }
        int n = t * nEmbd;
        if (in.length < n || out.length < n) {
            throw new IllegalArgumentException(kind + " step of " + t + " tokens needs " + n
                    + " floats of input and output");
        }
        if (in == out || isWorking(in) || isWorking(out)) {
            throw new IllegalArgumentException(kind + " mixer input and output must be distinct caller arrays");
        }
        switch (kind) {
            case MAMBA1:
                stepMamba1(t, in, out);
                break;
            case MAMBA2:
                stepMamba2(t, in, out);
                break;
            default:
                stepPlamo2(t, in, out);
                break;
        }
    }

    private void stepMamba1(int t, float[] in, float[] out) {
        projectIn(t, in);
        CpuSsmConv.forward(convCh, dConv, t, convFlags, q, 0, inWidth, convW, convB, convState, 0, 0, k, 0, dInner);
        mv.matmul(ssmX, k, 0, v, 0, t);
        xDeltas.applyRows(t, k, 0, v, 0);
        if (dtBcNorms) {
            NormMath.rowNorm(NormKind.RMS, eps, t, dtRank, xdbWidth, 1, v, 0, dtNorm, null, v, 0);
            NormMath.rowNorm(NormKind.RMS, eps, t, dState, xdbWidth, 1, v, dtRank, bNorm, null, v, dtRank);
            NormMath.rowNorm(NormKind.RMS, eps, t, dState, xdbWidth, 1, v, dtRank + dState, cNorm, null, v,
                    dtRank + dState);
        }
        mv.matmul(ssmDt, v, 0, xdbWidth, dt, 0, dInner, t);
        dtDeltas.apply(t, v, 0, xdbWidth, dt, 0, dInner);
        CpuSsmScan.scan(dState, 1, dInner, 1, t, true, k, 0, dInner, dt, 0, dInner, dtBias, a,
                v, dtRank, xdbWidth, v, dtRank + dState, xdbWidth, scanState, 0, attnOut, 0, dInner);
        CpuSsmScan.gateOut(t, dInner, 1, attnOut, 0, k, 0, dInner, d, q, dInner, inWidth, v, 0);
        projectOut(t, v, out);
    }

    private void stepMamba2(int t, float[] in, float[] out) {
        int gs = nGroup * dState;
        projectIn(t, in);
        CpuSsmConv.forward(convCh, dConv, t, convFlags, q, dInner, inWidth, convW, convB, convState, 0, 0,
                k, 0, convCh);
        CpuSsmScan.scan(dState, headDim, nHead, nGroup, t, false, k, 0, convCh, q, 2 * dInner + 2 * gs, inWidth,
                dtBias, a, k, dInner, convCh, k, dInner + gs, convCh, scanState, 0, attnOut, 0, dInner);
        CpuSsmScan.gateOut(t, dInner, headDim, attnOut, 0, k, 0, convCh, d, q, 0, inWidth, v, 0);
        if (groupNorm != null) {
            NormMath.groupRms(t, dInner, nGroup, eps, v, 0, dInner, groupNorm, v, 0);
        }
        projectOut(t, v, out);
    }

    private void stepPlamo2(int t, float[] in, float[] out) {
        projectIn(t, in);
        int rows = t * nHead;
        for (int r = 0; r < rows; r++) {
            int src = r * 2 * headDim;
            int dst = r * headDim;
            System.arraycopy(q, src + headDim, k, dst, headDim);
            System.arraycopy(q, src, v, dst, headDim);
        }
        CpuSsmConv.forward(convCh, dConv, t, convFlags, k, 0, dInner, convW, null, convState, 0, 0,
                attnOut, 0, dInner);
        mv.matmul(ssmX, attnOut, 0, q, 0, t);
        xDeltas.applyRows(t, attnOut, 0, q, 0);
        NormMath.rowNorm(NormKind.RMS, eps, t, dState, xdbWidth, 1, q, 0, bNorm, null, q, 0);
        NormMath.rowNorm(NormKind.RMS, eps, t, dState, xdbWidth, 1, q, dState, cNorm, null, q, dState);
        NormMath.rowNorm(NormKind.RMS, eps, t, dtRank, xdbWidth, 1, q, 2 * dState, dtNorm, null, q, 2 * dState);
        mv.matmul(ssmDt, q, 2 * dState, xdbWidth, dt, 0, nHead, t);
        dtDeltas.apply(t, q, 2 * dState, xdbWidth, dt, 0, nHead);
        CpuSsmScan.scan(dState, headDim, nHead, 1, t, false, attnOut, 0, dInner, dt, 0, nHead, dtBias, a,
                q, 0, xdbWidth, q, dState, xdbWidth, scanState, 0, k, 0, dInner);
        CpuSsmScan.gateOut(t, dInner, headDim, k, 0, attnOut, 0, dInner, d, v, 0, dInner, q, 0);
        projectOut(t, q, out);
    }

    /** {@code ssm_in} of the step's rows into {@code q}, plus its deltas. */
    private void projectIn(int t, float[] in) {
        mv.matmul(ssmIn, in, 0, q, 0, t);
        inDeltas.applyRows(t, in, 0, q, 0);
    }

    /** {@code ssm_out} of the gated rows {@code y} into {@code out}, plus its deltas. */
    private void projectOut(int t, float[] y, float[] out) {
        mv.matmul(ssmOut, y, 0, out, 0, t);
        outDeltas.applyRows(t, y, 0, out, 0);
    }

    private boolean isWorking(float[] arr) {
        return arr == q || arr == k || arr == v || arr == attnOut || arr == dt;
    }

    // ================================================================== lifecycle

    /** Marks the mixer closed; the arrays belong to the context. */
    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuMambaMixer{" + kind + (bound ? ", d_inner=" + dInner + ", d_state=" + dState + ", heads=" + nHead
                + "x" + headDim + ", groups=" + nGroup : ", unbound") + (closed ? ", closed" : "") + '}';
    }
}
