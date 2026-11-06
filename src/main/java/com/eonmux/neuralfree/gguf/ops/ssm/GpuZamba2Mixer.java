package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.ops.attn.GpuAttentionMixer;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.Zamba2Spec;

import java.util.Objects;

/**
 * GPU twin of {@link CpuZamba2Mixer}: a Zamba2 layer (vLLM {@code Zamba2HybridLayer} /
 * {@code Zamba2MambaDecoderLayer}, vllm/model_executor/models/zamba2.py). Every layer normalizes its input with
 * {@code attn_norm} and runs the Mamba-2 mixer ({@link GpuMambaMixer}); a hybrid layer first runs the shared
 * transformer block of its bare block index:
 *
 * <pre>
 * concat = [x | original input embedding]                 (2 * n_embd)
 * h = RMSNorm(concat) * shared_blk.&lt;b&gt;.attn_norm
 * h = shared attention(h)                                 (GpuAttentionMixer over the shared stack, per-use KV cache)
 * h = RMSNorm(h) * shared_blk.&lt;b&gt;.attn_pre_ff_norm
 * h = shared_blk.&lt;b&gt;.ffn_down · GEGLU_ERF(gate_up(h) + adapter(h))
 * t = shared_linear(h)
 * mamba_in = x + t
 * out = mamba2(RMSNorm(mamba_in) * attn_norm)
 * </pre>
 *
 * The per-use adapters ({@code blk.<il>.ffn_gate_up_adapter_{a,b}}) are added to the fused gate/up product before the
 * activation; the attention mixer adds its own q/k/v adapters. Every step is a sequence of {@link GpuOps} calls on the
 * context queue; a step allocates nothing and never waits on the host. Not thread-safe.
 */
public final class GpuZamba2Mixer implements GpuMixer {

    private static final String T_CONCAT = "zamba2.concat";
    private static final String T_NORMED = "zamba2.normed";
    private static final String T_ATTN = "zamba2.attn";
    private static final String T_PRE_FF = "zamba2.pre_ff";
    private static final String T_GATE_UP = "zamba2.gate_up";
    private static final String T_RANK = "zamba2.rank";
    private static final String T_DELTA = "zamba2.delta";
    private static final String T_ACT = "zamba2.act";
    private static final String T_MLP = "zamba2.mlp";
    private static final String T_LINEAR = "zamba2.linear";
    private static final String T_MAMBA_IN = "zamba2.mamba_in";
    private static final String T_MAMBA_NORMED = "zamba2.mamba_normed";

    private GpuOps ops;
    private GpuScratch scratch;
    private LayerSpec layer;
    private Zamba2Spec zamba2;
    private GpuAttentionMixer attention;
    private GpuMambaMixer mamba;
    private int ubatch;
    private int nEmbd;
    private int inputWidth;
    private int nFf;
    private float eps;
    private boolean bound;
    private boolean closed;

    private GpuWeight mambaNorm;
    private GpuWeight sharedNorm;
    private GpuWeight preFfNorm;
    private GpuWeight gateUpWeight;
    private GpuWeight downWeight;
    private GpuWeight linearWeight;
    private GpuWeight ffnAdapterA;
    private GpuWeight ffnAdapterB;
    private ScratchSlot orig;
    private ScratchSlot concat;
    private ScratchSlot normed;
    private ScratchSlot attnOut;
    private ScratchSlot preFf;
    private ScratchSlot gateUp;
    private ScratchSlot rankRows;
    private ScratchSlot delta;
    private ScratchSlot act;
    private ScratchSlot mlpOut;
    private ScratchSlot linear;
    private ScratchSlot mambaIn;
    private ScratchSlot mambaNormed;

    /** Runs the shared attention over {@code n} rows: the contiguous step or the batched step of the layout. */
    @FunctionalInterface
    private interface AttentionStep {
        void run(int n, ScratchSlot in, ScratchSlot out);
    }

    /** The Mamba-2 step over {@code n} rows: the contiguous step or the layout's batched step. */
    @FunctionalInterface
    private interface MambaStep {
        void run(int n, ScratchSlot in, ScratchSlot out);
    }

    /** An unbound mixer; {@link #bind} reads the layer's Zamba2 facet. */
    public GpuZamba2Mixer() {
    }

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException when the layer has no Zamba2 facet or a tensor has the wrong shape
     * @throws IllegalStateException    when the mixer is bound or closed, or the layer has no recurrent state
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (bound || closed) {
            throw new IllegalStateException("zamba2 mixer is already bound or closed");
        }
        Objects.requireNonNull(ctx, "ctx");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.zamba2 = layer.zamba2();
        if (zamba2 == null || layer.recurrent() == null || layer.recurrent().kind() != MixerKind.ZAMBA2) {
            throw new IllegalArgumentException("GpuZamba2Mixer cannot run layer " + layer.stack() + "." + layer.il()
                    + " (" + layer.mixerKind() + ")");
        }
        this.ops = ctx.ops();
        this.scratch = ctx.scratch();
        this.ubatch = scratch.ubatch();
        this.nEmbd = ctx.spec().hparams().nEmbd();
        this.inputWidth = Math.max(ctx.spec().embedding().nEmbdInp(), nEmbd);
        this.eps = layer.norm().eps();
        this.mamba = new GpuMambaMixer(MixerKind.MAMBA2);
        mamba.bind(ctx, zamba2.mambaLayer(layer));
        GpuWeights w = ctx.weightSource().weightsFor(layer.weightLayer());
        if (zamba2.hybrid()) {
            attention = new GpuAttentionMixer();
            attention.bind(ctx, zamba2.attentionLayer(layer));
            bindSharedBlock(w);
        }
        mambaNorm = matrix(w, layer, TensorNames.weight(TensorNames.ATTN_NORM), nEmbd, 1);
        mambaNormed = scratch.temp(T_MAMBA_NORMED, (long) ubatch * nEmbd);
        bound = true;
    }

    private void bindSharedBlock(GpuWeights w) {
        int block = zamba2.sharedBlock();
        sharedNorm = shared(w, block, TensorNames.weight(TensorNames.ATTN_NORM), 2L * nEmbd, 1);
        preFfNorm = shared(w, block, "attn_pre_ff_norm.weight", nEmbd, 1);
        gateUpWeight = shared(w, block, "ffn_gate_up.weight", nEmbd, -1);
        if (gateUpWeight.rows() % 2 != 0) {
            throw new IllegalArgumentException(TensorNames.STACK_SHARED_BLK + "." + block + ": " + gateUpWeight.name()
                    + " holds " + gateUpWeight.rows() + " rows, expected twice the feed-forward length");
        }
        nFf = (int) (gateUpWeight.rows() / 2);
        downWeight = shared(w, block, "ffn_down.weight", nFf, nEmbd);
        linearWeight = matrix(w, layer, TensorNames.weight("shared_linear"), nEmbd, nEmbd);
        int rank = zamba2.adapterRank();
        ffnAdapterA = matrix(w, layer, TensorNames.weight("ffn_gate_up_adapter_a"), nEmbd, rank);
        ffnAdapterB = matrix(w, layer, TensorNames.weight("ffn_gate_up_adapter_b"), rank, 2L * nFf);
        orig = scratch.slot(ScratchSlots.INPUT_EMBD).require((long) ubatch * inputWidth);
        concat = scratch.temp(T_CONCAT, (long) ubatch * 2 * nEmbd);
        normed = scratch.temp(T_NORMED, (long) ubatch * 2 * nEmbd);
        attnOut = scratch.temp(T_ATTN, (long) ubatch * nEmbd);
        preFf = scratch.temp(T_PRE_FF, (long) ubatch * nEmbd);
        gateUp = scratch.temp(T_GATE_UP, (long) ubatch * 2 * nFf);
        rankRows = scratch.temp(T_RANK, (long) ubatch * rank);
        delta = scratch.temp(T_DELTA, (long) ubatch * 2 * nFf);
        act = scratch.temp(T_ACT, (long) ubatch * nFf);
        mlpOut = scratch.temp(T_MLP, (long) ubatch * nEmbd);
        linear = scratch.temp(T_LINEAR, (long) ubatch * nEmbd);
        mambaIn = scratch.temp(T_MAMBA_IN, (long) ubatch * nEmbd);
    }

    private GpuWeight matrix(GpuWeights w, LayerSpec l, String suffix, long cols, long rows) {
        GpuWeight t = w.layer(l, suffix);
        if (t.cols() != cols || rows >= 0 && t.rows() != rows) {
            throw new IllegalArgumentException(l.stack() + "." + l.il() + ": " + t.name() + " is [" + t.cols() + " x "
                    + t.rows() + "], expected [" + cols + " x " + (rows < 0 ? "2 * n_ff" : rows) + "]");
        }
        return t;
    }

    private GpuWeight shared(GpuWeights w, int block, String suffix, long cols, long rows) {
        GpuWeight t = w.stackOrNull(TensorNames.STACK_SHARED_BLK, block, suffix);
        if (t == null) {
            throw new IllegalStateException(TensorNames.STACK_SHARED_BLK + "." + block + ": tensor " + suffix
                    + " is missing");
        }
        if (t.cols() != cols || rows >= 0 && t.rows() != rows) {
            throw new IllegalArgumentException(TensorNames.STACK_SHARED_BLK + "." + block + ": " + t.name() + " is ["
                    + t.cols() + " x " + t.rows() + "], expected [" + cols + " x "
                    + (rows < 0 ? "2 * n_ff" : rows) + "]");
        }
        return t;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        run(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        run(pos0, nTokens, in, out);
    }

    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        Objects.requireNonNull(s, "s");
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        int n = s.nTokens();
        checkStep(n);
        ScratchSlot inSlot = zamba2.hybrid() ? sharedBlock(n, in, (nn, src, dst) -> attention.batch(s, src, dst))
                : in;
        runMamba(n, inSlot, out, (nn, src, dst) -> mamba.batch(s, src, dst));
    }

    /** Why this mixer has no cache-write-only step: the shared attention's output feeds the Mamba-2 input. */
    @Override
    public String kvRefusal() {
        return "a zamba2 layer's cache writes need the shared attention's output, which needs its queries";
    }

    /** A Zamba2 layer keeps Mamba-2 recurrent state, so its memory cannot be re-positioned. */
    @Override
    public String shiftRefusal() {
        return "a zamba2 layer keeps Mamba-2 recurrent state; the context cannot be shifted";
    }

    @Override
    public String batchRefusal() {
        if (zamba2.hybrid() && attention.batchRefusal() != null) {
            return attention.batchRefusal();
        }
        return mamba.batchRefusal();
    }

    private void run(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        checkStep(n);
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + layer.il() + ": negative position " + pos0);
        }
        ScratchSlot inSlot = zamba2.hybrid()
                ? sharedBlock(n, in, (nn, src, dst) -> attention.prefill(pos0, nn, src, dst)) : in;
        runMamba(n, inSlot, out, (nn, src, dst) -> mamba.prefill(pos0, nn, src, dst));
    }

    private void checkStep(int n) {
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + n + " tokens outside [1, "
                    + ubatch + "]");
        }
    }

    /**
     * The shared transformer block: concatenates the residual stream with the model input embedding, normalizes and
     * attends, then runs the gated MLP and the {@code shared_linear} projection and adds the result to the layer
     * input.
     */
    private ScratchSlot sharedBlock(int n, ScratchSlot in, AttentionStep step) {
        ops.copy2d(in.buffer(), in.offset(), nEmbd, concat.buffer(), concat.offset(), 2 * nEmbd, n, nEmbd);
        ops.copy2d(orig.buffer(), orig.offset(), inputWidth, concat.buffer(), concat.offset() + nEmbd, 2 * nEmbd,
                n, nEmbd);
        ops.norm(NormKind.RMS, sharedNorm, null, eps, concat.buffer(), concat.offset(), normed.buffer(),
                normed.offset(), n, 2 * nEmbd);
        step.run(n, normed, attnOut);
        ops.norm(NormKind.RMS, preFfNorm, null, eps, attnOut.buffer(), attnOut.offset(), preFf.buffer(),
                preFf.offset(), n, nEmbd);
        mlp(n);
        ops.add(in.buffer(), in.offset(), linear.buffer(), linear.offset(), mambaIn.buffer(), mambaIn.offset(),
                n * nEmbd);
        return mambaIn;
    }

    /** The gated GELU MLP with its per-use adapter, then the {@code shared_linear} projection into {@link #T_LINEAR}. */
    private void mlp(int n) {
        // The gate-up and adapter projections of a single-token step both read this row, so its Q8_1 blocks
        // are quantized once. The scope stays inside the MLP: the shared attention of the block opens its own.
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(preFf.offset(), nEmbd)
                ? ops.shareQ8Activations(preFf.buffer(), preFf.offset(), nEmbd) : null;
        try {
            ops.matmul(gateUpWeight, preFf.buffer(), preFf.offset(), gateUp.buffer(), gateUp.offset(), n);
            ops.matmul(ffnAdapterA, preFf.buffer(), preFf.offset(), rankRows.buffer(), rankRows.offset(), n);
            ops.matmul(ffnAdapterB, rankRows.buffer(), rankRows.offset(), delta.buffer(), delta.offset(), n);
            ops.add(gateUp.buffer(), gateUp.offset(), delta.buffer(), delta.offset(), gateUp.buffer(), gateUp.offset(),
                    n * 2 * nFf);
            ops.glu(GluOp.GEGLU_ERF, ActParams.NONE, gateUp.buffer(), gateUp.offset(), 2 * nFf, gateUp.buffer(),
                    gateUp.offset() + nFf, 2 * nFf, act.buffer(), act.offset(), nFf, n, nFf);
            ops.matmul(downWeight, act.buffer(), act.offset(), mlpOut.buffer(), mlpOut.offset(), n);
            ops.matmul(linearWeight, mlpOut.buffer(), mlpOut.offset(), linear.buffer(), linear.offset(), n);
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    private void runMamba(int n, ScratchSlot in, ScratchSlot out, MambaStep step) {
        ops.norm(NormKind.RMS, mambaNorm, null, eps, in.buffer(), in.offset(), mambaNormed.buffer(),
                mambaNormed.offset(), n, nEmbd);
        step.run(n, mambaNormed, out);
    }

    // ================================================================== lifecycle

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("zamba2 mixer is closed");
        }
        if (!bound) {
            throw new IllegalStateException("zamba2 mixer is not bound");
        }
    }

    @Override
    public void close() {
        closed = true;
        if (attention != null) {
            attention.close();
        }
        if (mamba != null) {
            mamba.close();
        }
    }

    @Override
    public String toString() {
        return "GpuZamba2Mixer{" + (layer == null ? "unbound" : "layer " + layer.il() + ", " + zamba2) + '}';
    }
}
