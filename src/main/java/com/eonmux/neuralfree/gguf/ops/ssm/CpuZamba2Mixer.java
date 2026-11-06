package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.ops.act.GluMath;
import com.eonmux.neuralfree.gguf.ops.attn.CpuAttentionMixer;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.Zamba2Spec;

import java.util.Objects;

/**
 * CPU twin of {@link GpuZamba2Mixer}: a Zamba2 layer (vLLM {@code Zamba2HybridLayer} /
 * {@code Zamba2MambaDecoderLayer}, vllm/model_executor/models/zamba2.py). Every layer normalizes its input with
 * {@code attn_norm} and runs the Mamba-2 mixer ({@link CpuMambaMixer}); a hybrid layer first runs the shared
 * transformer block of its bare block index:
 *
 * <pre>
 * concat = [x | original input embedding]                 (2 * n_embd)
 * h = RMSNorm(concat) * shared_blk.&lt;b&gt;.attn_norm
 * h = shared attention(h)                                 (CpuAttentionMixer over the shared stack, per-use KV cache)
 * h = RMSNorm(h) * shared_blk.&lt;b&gt;.attn_pre_ff_norm
 * h = shared_blk.&lt;b&gt;.ffn_down · GEGLU_ERF(gate_up(h) + adapter(h))
 * t = shared_linear(h)
 * mamba_in = x + t
 * out = mamba2(RMSNorm(mamba_in) * attn_norm)
 * </pre>
 *
 * The per-use adapters ({@code blk.<il>.ffn_gate_up_adapter_{a,b}}) are added to the fused gate/up product before the
 * activation; the attention mixer adds its own q/k/v adapters. Working arrays come from {@link CpuScratch}; a step
 * allocates nothing. Not thread-safe.
 */
public final class CpuZamba2Mixer implements CpuMixer {

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
    /** Elements per parallel task of the token-wise loops. */
    private static final int ELEMENT_GRAIN = 1 << 15;

    private CpuScratch scratch;
    private CpuMatVec matVec;
    private LayerSpec layer;
    private Zamba2Spec zamba2;
    private CpuAttentionMixer attention;
    private CpuMambaMixer mamba;
    private int ubatch;
    private int nEmbd;
    private int inputWidth;
    private int nFf;
    private float eps;
    private boolean bound;
    private boolean closed;

    private float[] mambaNorm;
    private float[] sharedNorm;
    private float[] preFfNorm;
    private CpuTensorRef gateUpWeight;
    private CpuTensorRef downWeight;
    private CpuTensorRef linearWeight;
    private CpuTensorRef ffnAdapterA;
    private CpuTensorRef ffnAdapterB;

    /** Runs the shared attention over {@code n} rows: the contiguous step or the batched step of the layout. */
    @FunctionalInterface
    private interface AttentionStep {
        void run(int n, float[] concat, float[] out);
    }

    /** An unbound mixer; {@link #bind} reads the layer's Zamba2 facet. */
    public CpuZamba2Mixer() {
    }

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException when the layer has no Zamba2 facet or a tensor has the wrong shape
     * @throws IllegalStateException    when the mixer is bound or closed, or the layer has no recurrent state
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        if (bound || closed) {
            throw new IllegalStateException("zamba2 mixer is already bound or closed");
        }
        Objects.requireNonNull(ctx, "ctx");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.zamba2 = layer.zamba2();
        if (zamba2 == null || layer.recurrent() == null || layer.recurrent().kind() != MixerKind.ZAMBA2) {
            throw new IllegalArgumentException("CpuZamba2Mixer cannot run layer " + layer.stack() + "." + layer.il()
                    + " (" + layer.mixerKind() + ")");
        }
        this.scratch = ctx.scratch();
        this.matVec = ctx.matVec();
        this.ubatch = scratch.ubatch();
        this.nEmbd = ctx.spec().hparams().nEmbd();
        this.inputWidth = Math.max(ctx.spec().embedding().nEmbdInp(), nEmbd);
        this.eps = layer.norm().eps();
        this.mamba = new CpuMambaMixer(MixerKind.MAMBA2);
        mamba.bind(ctx, zamba2.mambaLayer(layer));
        if (zamba2.hybrid()) {
            attention = new CpuAttentionMixer();
            attention.bind(ctx, zamba2.attentionLayer(layer));
            bindSharedBlock(ctx.weights());
        }
        mambaNorm = vector(ctx.weights(), layer, TensorNames.weight(TensorNames.ATTN_NORM), nEmbd);
        scratch.temp(T_MAMBA_NORMED, ubatch * nEmbd);
        bound = true;
    }

    private void bindSharedBlock(CpuWeights w) {
        int block = zamba2.sharedBlock();
        sharedNorm = sharedVector(w, block, TensorNames.weight(TensorNames.ATTN_NORM), 2L * nEmbd);
        preFfNorm = sharedVector(w, block, "attn_pre_ff_norm.weight", nEmbd);
        gateUpWeight = shared(w, block, "ffn_gate_up.weight");
        downWeight = shared(w, block, "ffn_down.weight");
        if (gateUpWeight.cols() != nEmbd || gateUpWeight.rows() % 2 != 0) {
            throw new IllegalArgumentException(layer.stack() + "." + layer.il() + ": " + gateUpWeight
                    + " must read n_embd " + nEmbd + " values and hold twice the feed-forward length");
        }
        nFf = (int) (gateUpWeight.rows() / 2);
        if (downWeight.cols() != nFf || downWeight.rows() != nEmbd) {
            throw new IllegalArgumentException(layer.stack() + "." + layer.il() + ": " + downWeight + " is not a ["
                    + nFf + " x " + nEmbd + "] projection");
        }
        linearWeight = w.layer(layer, TensorNames.weight("shared_linear"));
        if (linearWeight.cols() != nEmbd || linearWeight.rows() != nEmbd) {
            throw new IllegalArgumentException(layer.stack() + "." + layer.il() + ": " + linearWeight + " is not a ["
                    + nEmbd + " x " + nEmbd + "] projection");
        }
        int rank = zamba2.adapterRank();
        ffnAdapterA = adapter(w, "ffn_gate_up_adapter_a", nEmbd, rank);
        ffnAdapterB = adapter(w, "ffn_gate_up_adapter_b", rank, 2L * nFf);
        scratch.temp(T_CONCAT, ubatch * 2 * nEmbd);
        scratch.temp(T_NORMED, ubatch * 2 * nEmbd);
        scratch.temp(T_ATTN, ubatch * nEmbd);
        scratch.temp(T_PRE_FF, ubatch * nEmbd);
        scratch.temp(T_GATE_UP, ubatch * 2 * nFf);
        scratch.temp(T_RANK, ubatch * rank);
        scratch.temp(T_DELTA, ubatch * 2 * nFf);
        scratch.temp(T_ACT, ubatch * nFf);
        scratch.temp(T_MLP, ubatch * nEmbd);
        scratch.temp(T_LINEAR, ubatch * nEmbd);
        scratch.temp(T_MAMBA_IN, ubatch * nEmbd);
    }

    private float[] vector(CpuWeights w, LayerSpec l, String suffix, long n) {
        float[] v = w.layerFloatsOrNull(l, suffix);
        if (v == null) {
            throw new IllegalStateException(l.stack() + "." + l.il() + ": tensor " + suffix + " is missing");
        }
        if (v.length != n) {
            throw new IllegalArgumentException(l.stack() + "." + l.il() + ": tensor " + suffix + " holds "
                    + v.length + " values, expected " + n);
        }
        return v;
    }

    private float[] sharedVector(CpuWeights w, int block, String suffix, long n) {
        float[] v = w.stackFloatsOrNull(TensorNames.STACK_SHARED_BLK, block, suffix);
        if (v == null) {
            throw new IllegalStateException(TensorNames.STACK_SHARED_BLK + "." + block + ": tensor " + suffix
                    + " is missing");
        }
        if (v.length != n) {
            throw new IllegalArgumentException(TensorNames.STACK_SHARED_BLK + "." + block + ": tensor " + suffix
                    + " holds " + v.length + " values, expected " + n);
        }
        return v;
    }

    private CpuTensorRef shared(CpuWeights w, int block, String suffix) {
        CpuTensorRef t = w.stackOrNull(TensorNames.STACK_SHARED_BLK, block, suffix);
        if (t == null) {
            throw new IllegalStateException(TensorNames.STACK_SHARED_BLK + "." + block + ": tensor " + suffix
                    + " is missing");
        }
        return t;
    }

    private CpuTensorRef adapter(CpuWeights w, String base, long cols, long rows) {
        CpuTensorRef t = w.layer(layer, TensorNames.weight(base));
        if (t.cols() != cols || t.rows() != rows) {
            throw new IllegalArgumentException(layer.stack() + "." + layer.il() + ": " + t + " is not a [" + cols
                    + " x " + rows + "] adapter");
        }
        return t;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, float[] in, float[] out) {
        run(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        run(pos0, n, in, out);
    }

    @Override
    public void batch(StepLayout s, float[] in, float[] out) {
        ensureBound();
        Objects.requireNonNull(s, "s");
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        int n = s.nTokens();
        checkStep(n);
        float[] mambaIn = zamba2.hybrid() ? sharedBlock(n, in, (nn, c, o) -> attention.batch(s, c, o)) : in;
        runMamba(n, mambaIn, out, (nn, src, dst) -> mamba.batch(s, src, dst));
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

    private void run(int pos0, int n, float[] in, float[] out) {
        ensureBound();
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        checkStep(n);
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + layer.il() + ": negative position " + pos0);
        }
        float[] mambaIn = zamba2.hybrid() ? sharedBlock(n, in, (nn, c, o) -> attention.prefill(pos0, nn, c, o)) : in;
        runMamba(n, mambaIn, out, (nn, src, dst) -> mamba.prefill(pos0, nn, src, dst));
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
    private float[] sharedBlock(int n, float[] in, AttentionStep step) {
        float[] concat = scratch.temp(T_CONCAT, ubatch * 2 * nEmbd);
        float[] normed = scratch.temp(T_NORMED, ubatch * 2 * nEmbd);
        float[] attnOut = scratch.temp(T_ATTN, ubatch * nEmbd);
        concat(n, in, concat);
        norm(n, concat, normed, 2 * nEmbd, sharedNorm);
        step.run(n, normed, attnOut);
        float[] preFf = scratch.temp(T_PRE_FF, ubatch * nEmbd);
        norm(n, attnOut, preFf, nEmbd, preFfNorm);
        mlp(n, preFf);
        float[] mambaIn = scratch.temp(T_MAMBA_IN, ubatch * nEmbd);
        CpuOps.add(in, 0, scratch.temp(T_LINEAR, ubatch * nEmbd), 0, mambaIn, 0, n * nEmbd);
        return mambaIn;
    }

    private void concat(int n, float[] in, float[] concat) {
        float[] orig = scratch.slot(CpuScratch.INPUT_EMBD);
        int w = nEmbd;
        int iw = inputWidth;
        CpuParallel.parallelFor(n, Math.max(1, ELEMENT_GRAIN / (2 * w)), (from, to) -> {
            for (int t = from; t < to; t++) {
                System.arraycopy(in, t * w, concat, t * 2 * w, w);
                System.arraycopy(orig, t * iw, concat, t * 2 * w + w, w);
            }
        });
    }

    private void norm(int n, float[] in, float[] out, int width, float[] weight) {
        NormMath.rowNorm(NormKind.RMS, eps, n, width, width, 1, in, 0, weight, null, out, 0);
    }

    /** The gated GELU MLP with its per-use adapter, then the {@code shared_linear} projection into {@link #T_LINEAR}. */
    private void mlp(int n, float[] preFf) {
        float[] gateUp = scratch.temp(T_GATE_UP, ubatch * 2 * nFf);
        matVec.matmul(gateUpWeight, preFf, 0, gateUp, 0, n);
        float[] rankRows = scratch.temp(T_RANK, ubatch * zamba2.adapterRank());
        matVec.matmul(ffnAdapterA, preFf, 0, rankRows, 0, n);
        float[] delta = scratch.temp(T_DELTA, ubatch * 2 * nFf);
        matVec.matmul(ffnAdapterB, rankRows, 0, delta, 0, n);
        CpuOps.add(gateUp, 0, delta, 0, gateUp, 0, n * 2 * nFf);
        float[] act = scratch.temp(T_ACT, ubatch * nFf);
        GluMath.applyRows(GluOp.GEGLU_ERF, ActParams.NONE, n, nFf, gateUp, 0, 2 * nFf, gateUp, nFf, 2 * nFf,
                act, 0, nFf);
        float[] mlpOut = scratch.temp(T_MLP, ubatch * nEmbd);
        matVec.matmul(downWeight, act, 0, mlpOut, 0, n);
        matVec.matmul(linearWeight, mlpOut, 0, scratch.temp(T_LINEAR, ubatch * nEmbd), 0, n);
    }

    private void runMamba(int n, float[] in, float[] out, MambaStep step) {
        float[] normed = scratch.temp(T_MAMBA_NORMED, ubatch * nEmbd);
        norm(n, in, normed, nEmbd, mambaNorm);
        step.run(n, normed, out);
    }

    /** The Mamba-2 step over {@code n} rows: the contiguous step or the layout's batched step. */
    @FunctionalInterface
    private interface MambaStep {
        void run(int n, float[] in, float[] out);
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
        return "CpuZamba2Mixer{" + (layer == null ? "unbound" : "layer " + layer.il() + ", " + zamba2) + '}';
    }
}
