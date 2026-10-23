package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.Arrays;
import java.util.Objects;

/**
 * kimi-k3 cross-layer attention residuals of one layer on the GPU ({@code LayerSpec.Extras.ATTN_RES};
 * kimi-k3.cpp:188-235, 280-337). The layer input {@code prefix_sum} is kept in {@code hc.res_prefix}; the mixer and
 * the FFN read a softmax-weighted mix of the checkpoint stack and the current prefix sum, scored with
 * {@code attn_res_score} / {@code ffn_res_score}. Every {@code attn_res_block_size} layers the raw layer input is
 * banked on the stack and the prefix sum restarts from the attention output.
 *
 * <p>The checkpoint stack {@code hc.res_stack} is shared by every layer of a step and laid out
 * {@code stack[(c*T + t)*n_embd + i]}. Layer {@code il} (block size {@code B}) mixes over the
 * {@code ceil(il / B)} checkpoints banked by layers {@code 0, B, 2B, ...} before it and, when {@code il % B == 0},
 * banks its input as checkpoint {@code il / B}; layers run in order within a step, so these counts are exact without
 * shared mutable state.</p>
 *
 * <p>Dispatches ({@code n} = checkpoints visible to the mix; "mix into" is {@code LLM_ResStack_Score} with
 * {@code n_ckpt = n} then {@code LLM_ResStack_Mix}):
 * <ol>
 *   <li>{@link #beforeBlock}: banked layer: copy {@code x} to checkpoint {@code il/B}; otherwise copy {@code x} to
 *       {@code hc.res_prefix}. Then, when {@code ceil(il/B) > 0}, mix {@code x} into {@code x} in place
 *       (kimi-k3.cpp:284-291: the mix precedes the push, so it excludes the new checkpoint);</li>
 *   <li>{@link #afterMixer}: {@code prefix = branch} (banked, :299) or {@code prefix += branch}; mix
 *       {@code prefix} into {@code x} with {@code ffn_res_score} (:309-310); returns true;</li>
 *   <li>{@link #afterFfn}: {@code x = prefix + branch} (:326); returns true;</li>
 *   <li>{@link #afterBlock} of the last trunk layer: mix {@code x} into {@code x} with {@code output_res_score}
 *       over every checkpoint (:335-337), so the head sees the final mix before its output norm.</li>
 * </ol>
 * {@code x} and {@code branch} are {@code n_embd} wide. Not thread-safe; one forward pass at a time.</p>
 */
public final class GpuAttnResidualExtra implements GpuLayerExtra {

    private GpuOps ops;
    private int ubatch;
    private int local;
    private int scoreLocal;
    private int nEmbd;
    private int il;
    private boolean banked;
    private int ckptBefore;
    private int ckptAfter;
    private int ckptFinal;
    private boolean last;
    private ScratchSlot stack;
    private ScratchSlot prefix;
    private ScratchSlot probs;
    private PreparedDispatch scoreAttn;
    private PreparedDispatch scoreFfn;
    private PreparedDispatch scoreOut;
    private PreparedDispatch mix;

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (mix != null) {
            throw new IllegalStateException("attention-residual extra of layer " + layer.il() + " is already bound");
        }
        String arch = ctx.spec().arch().ggufName();
        int blockSize = ctx.spec().hparams().attnResBlockSize();
        if (blockSize <= 0) {
            throw new UnsupportedOperationException(arch + " layer " + layer.il()
                    + ": attention residuals need attn_res_block_size > 0, got " + blockSize);
        }
        int nLayer = ctx.spec().nLayer();
        this.il = layer.il();
        if (!TensorNames.STACK_BLK.equals(layer.stack()) || il >= nLayer) {
            throw new UnsupportedOperationException(arch + ": attention residuals apply to the trunk layers only, got "
                    + layer);
        }
        this.ops = ctx.ops();
        GpuScratch scratch = ctx.scratch();
        GpuPlaceholders ph = ctx.placeholders();
        this.ubatch = scratch.ubatch();
        this.local = ops.local();
        this.scoreLocal = HcKernels.scoreLocal(local);
        this.nEmbd = ctx.spec().hparams().nEmbd();
        HcKernels.validateResStack(nEmbd, 0);
        this.banked = HcKernels.isCheckpointLayer(il, blockSize);
        this.ckptBefore = HcKernels.checkpointCount(il, blockSize);
        this.ckptAfter = banked ? ckptBefore + 1 : ckptBefore;
        this.ckptFinal = HcKernels.checkpointCount(nLayer, blockSize);
        this.last = il == nLayer - 1;
        float eps = ctx.spec().hparams().normRmsEps();

        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        GpuWeight attnScore = scoreWeight(weights.layer(layer, TensorNames.weight(TensorNames.ATTN_RES_SCORE)));
        GpuWeight ffnScore = scoreWeight(weights.layer(layer, TensorNames.weight(TensorNames.FFN_RES_SCORE)));
        GpuWeight outScore = last ? scoreWeight(ctx.weights().get(TensorNames.weight(TensorNames.OUTPUT_RES_SCORE)))
                : null;

        this.stack = scratch.temp(HcSubBlock.TEMP_RES_STACK, HcKernels.stackFloats(ckptFinal, ubatch, nEmbd));
        this.prefix = scratch.temp(HcSubBlock.TEMP_RES_PREFIX, (long) ubatch * nEmbd);
        this.probs = scratch.temp(HcSubBlock.TEMP_RES_PROBS, (long) ubatch * (ckptFinal + 1));

        PreparedDispatch sa = null;
        PreparedDispatch sf = null;
        PreparedDispatch so = null;
        try {
            sa = prepareScore(attnScore, eps, ph);
            sf = prepareScore(ffnScore, eps, ph);
            so = outScore != null ? prepareScore(outScore, eps, ph) : null;
            this.mix = prepareMix(ph);
        } catch (RuntimeException e) {
            for (PreparedDispatch d : new PreparedDispatch[]{sa, sf, so}) {
                if (d != null) {
                    d.close();
                }
            }
            throw e;
        }
        this.scoreAttn = sa;
        this.scoreFfn = sf;
        this.scoreOut = so;
    }

    private GpuWeight scoreWeight(GpuWeight w) {
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.cols() != nEmbd || w.rows() != 1) {
            throw new IllegalArgumentException("attention-residual score weight " + w.name() + " must be one F32 "
                    + "vector of " + nEmbd + " values, got " + w);
        }
        return w;
    }

    private PreparedDispatch prepareScore(GpuWeight scoreW, float eps, GpuPlaceholders ph) {
        Object[] a = new Object[HcKernels.SCORE_ARG_COUNT];
        a[HcKernels.SCORE_ARG_N_TOKENS] = ubatch;
        a[HcKernels.SCORE_ARG_N_EMBD] = nEmbd;
        a[HcKernels.SCORE_ARG_N_CKPT] = 0;
        a[HcKernels.SCORE_ARG_EPS] = eps;
        a[HcKernels.SCORE_ARG_STACK_OFF] = stack.offset();
        a[HcKernels.SCORE_ARG_CUR_OFF] = 0;
        a[HcKernels.SCORE_ARG_PROBS_OFF] = probs.offset();
        a[HcKernels.SCORE_ARG_STACK] = stack.buffer();
        a[HcKernels.SCORE_ARG_CUR] = ph.readOnlyFloats();
        a[HcKernels.SCORE_ARG_SCORE_W] = scoreW.shard(0);
        a[HcKernels.SCORE_ARG_PROBS] = probs.buffer();
        return ops.prepare(HcKernels.RES_STACK_SCORE, new long[]{HcKernels.scoreGlobal(ubatch, scoreLocal)},
                new long[]{scoreLocal}, a);
    }

    private PreparedDispatch prepareMix(GpuPlaceholders ph) {
        Object[] a = new Object[HcKernels.MIX_ARG_COUNT];
        a[HcKernels.MIX_ARG_N_TOKENS] = ubatch;
        a[HcKernels.MIX_ARG_N_EMBD] = nEmbd;
        a[HcKernels.MIX_ARG_N_CKPT] = 0;
        a[HcKernels.MIX_ARG_STACK_OFF] = stack.offset();
        a[HcKernels.MIX_ARG_CUR_OFF] = 0;
        a[HcKernels.MIX_ARG_PROBS_OFF] = probs.offset();
        a[HcKernels.MIX_ARG_OUT_OFF] = 0;
        a[HcKernels.MIX_ARG_STACK] = stack.buffer();
        a[HcKernels.MIX_ARG_CUR] = ph.readOnlyFloats();
        a[HcKernels.MIX_ARG_PROBS] = probs.buffer();
        a[HcKernels.MIX_ARG_OUT] = ph.scratchFloats();
        return ops.prepare(HcKernels.RES_STACK_MIX, new long[]{HcKernels.elementGlobal(ubatch, nEmbd, local)},
                new long[]{local}, a);
    }

    @Override
    public void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
        checkStep(nTokens);
        int n = nTokens * nEmbd;
        x.require(n);
        if (banked) {
            int ckpt = Math.toIntExact(stack.offset() + HcKernels.checkpointOffset(ckptBefore, nTokens, nEmbd));
            ops.copy(x.buffer(), x.offset(), stack.buffer(), ckpt, n);
        } else {
            ops.copy(x.buffer(), x.offset(), prefix.buffer(), prefix.offset(), n);
        }
        if (ckptBefore > 0) {
            mixInto(scoreAttn, ckptBefore, nTokens, x.buffer(), x.offset(), x.buffer(), x.offset());
        }
    }

    @Override
    public boolean afterMixer(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        checkStep(nTokens);
        int n = nTokens * nEmbd;
        branch.require(n);
        if (banked) {
            ops.copy(branch.buffer(), branch.offset(), prefix.buffer(), prefix.offset(), n);
        } else {
            ops.add(prefix.buffer(), prefix.offset(), branch.buffer(), branch.offset(), prefix.buffer(),
                    prefix.offset(), n);
        }
        if (ckptAfter > 0) {
            mixInto(scoreFfn, ckptAfter, nTokens, prefix.buffer(), prefix.offset(), x.buffer(), x.offset());
        } else {
            ops.copy(prefix.buffer(), prefix.offset(), x.buffer(), x.offset(), n);
        }
        return true;
    }

    @Override
    public boolean afterFfn(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        checkStep(nTokens);
        int n = nTokens * nEmbd;
        branch.require(n);
        ops.add(prefix.buffer(), prefix.offset(), branch.buffer(), branch.offset(), x.buffer(), x.offset(), n);
        return true;
    }

    @Override
    public void afterBlock(int pos0, int nTokens, ScratchSlot x) {
        checkStep(nTokens);
        if (last && ckptFinal > 0) {
            mixInto(scoreOut, ckptFinal, nTokens, x.buffer(), x.offset(), x.buffer(), x.offset());
        }
    }

    /** {@code out = mix(stack[0, nCkpt), cur)} with the given score dispatch; {@code out} may be {@code cur}. */
    private void mixInto(PreparedDispatch score, int nCkpt, int nTokens, GpuBuffer cur, int curOff, GpuBuffer out,
                         int outOff) {
        score.setInt(HcKernels.SCORE_ARG_N_TOKENS, nTokens);
        score.setInt(HcKernels.SCORE_ARG_N_CKPT, nCkpt);
        score.setBuffer(HcKernels.SCORE_ARG_CUR, cur);
        score.setInt(HcKernels.SCORE_ARG_CUR_OFF, curOff);
        score.setGlobal(HcKernels.scoreGlobal(nTokens, scoreLocal));
        score.dispatch();
        mix.setInt(HcKernels.MIX_ARG_N_TOKENS, nTokens);
        mix.setInt(HcKernels.MIX_ARG_N_CKPT, nCkpt);
        mix.setBuffer(HcKernels.MIX_ARG_CUR, cur);
        mix.setInt(HcKernels.MIX_ARG_CUR_OFF, curOff);
        mix.setBuffer(HcKernels.MIX_ARG_OUT, out);
        mix.setInt(HcKernels.MIX_ARG_OUT_OFF, outOff);
        mix.setGlobal(HcKernels.elementGlobal(nTokens, nEmbd, local));
        mix.dispatch();
    }

    private void checkStep(int nTokens) {
        if (mix == null) {
            throw new IllegalStateException("attention-residual extra used before bind()");
        }
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("attention residuals of layer " + il + ": token count " + nTokens
                    + " is outside [1, " + ubatch + "]");
        }
    }

    @Override
    public void close() {
        PreparedDispatch[] all = {scoreAttn, scoreFfn, scoreOut, mix};
        scoreAttn = null;
        scoreFfn = null;
        scoreOut = null;
        mix = null;
        RuntimeException first = Closeables.closeAll(Arrays.asList(all), null);
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuAttnResidualExtra{layer " + il + (banked ? ", banks" : "") + ", ckpt " + ckptBefore + "->"
                + ckptAfter + (last ? ", final " + ckptFinal : "") + '}';
    }
}
