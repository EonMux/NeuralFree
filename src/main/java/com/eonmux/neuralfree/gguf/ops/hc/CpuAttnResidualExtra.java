package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * kimi-k3 cross-layer attention residuals of one layer on the CPU ({@code LayerSpec.Extras.ATTN_RES}), the twin of
 * {@link GpuAttnResidualExtra} with the same checkpoint arithmetic, the same steps and the shared stack
 * {@code hc.res_stack} ({@code stack[(c*n + t)*n_embd + i]}). The score and mix are
 * {@link CpuAttnResidualStack#scores} and {@link CpuAttnResidualStack#mix}, split over tokens with
 * {@link CpuParallel}. {@code x} and {@code branch} are {@code n_embd} wide. Not thread-safe.
 */
public final class CpuAttnResidualExtra implements CpuLayerExtra {

    /** Minimum tokens per parallel range of the score and mix. */
    private static final int TOKEN_GRAIN = 1;

    private int ubatch;
    private int nEmbd;
    private int il;
    private float eps;
    private boolean banked;
    private int ckptBefore;
    private int ckptAfter;
    private int ckptFinal;
    private boolean last;
    private float[] attnScore;
    private float[] ffnScore;
    private float[] outScore;
    private float[] stack;
    private float[] prefix;
    private float[] probs;

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (stack != null) {
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
        CpuScratch scratch = ctx.scratch();
        CpuWeights weights = ctx.weights();
        this.ubatch = scratch.ubatch();
        this.nEmbd = ctx.spec().hparams().nEmbd();
        HcKernels.validateResStack(nEmbd, 0);
        this.eps = ctx.spec().hparams().normRmsEps();
        this.banked = HcKernels.isCheckpointLayer(il, blockSize);
        this.ckptBefore = HcKernels.checkpointCount(il, blockSize);
        this.ckptAfter = banked ? ckptBefore + 1 : ckptBefore;
        this.ckptFinal = HcKernels.checkpointCount(nLayer, blockSize);
        this.last = il == nLayer - 1;

        this.attnScore = scoreWeight(arch, weights.layerFloatsOrNull(layer,
                TensorNames.weight(TensorNames.ATTN_RES_SCORE)), layer.tensorName(TensorNames.weight(
                TensorNames.ATTN_RES_SCORE)));
        this.ffnScore = scoreWeight(arch, weights.layerFloatsOrNull(layer,
                TensorNames.weight(TensorNames.FFN_RES_SCORE)), layer.tensorName(TensorNames.weight(
                TensorNames.FFN_RES_SCORE)));
        String outName = TensorNames.weight(TensorNames.OUTPUT_RES_SCORE);
        this.outScore = last ? scoreWeight(arch, weights.floatsOrNull(outName), outName) : null;

        this.prefix = scratch.temp(HcSubBlock.TEMP_RES_PREFIX, arrayLength((long) ubatch * nEmbd));
        this.probs = scratch.temp(HcSubBlock.TEMP_RES_PROBS, arrayLength((long) ubatch * (ckptFinal + 1)));
        this.stack = scratch.temp(HcSubBlock.TEMP_RES_STACK,
                arrayLength(HcKernels.stackFloats(ckptFinal, ubatch, nEmbd)));
    }

    private float[] scoreWeight(String arch, float[] w, String name) {
        if (w == null) {
            throw new MissingTensorException(arch, il, name);
        }
        if (w.length != nEmbd) {
            throw new IllegalArgumentException("attention-residual score weight " + name + " has " + w.length
                    + " values, needs " + nEmbd);
        }
        return w;
    }

    private static int arrayLength(long floats) {
        if (floats > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("attention-residual buffer of " + floats
                    + " floats exceeds a Java array; lower neuralfree.ubatch");
        }
        return (int) floats;
    }

    @Override
    public void beforeBlock(int pos0, int n, float[] x) {
        checkStep(n);
        int len = n * nEmbd;
        CpuHyperConnections.requireLength("x", x, len);
        if (banked) {
            System.arraycopy(x, 0, stack, (int) HcKernels.checkpointOffset(ckptBefore, n, nEmbd), len);
        } else {
            System.arraycopy(x, 0, prefix, 0, len);
        }
        if (ckptBefore > 0) {
            mixInto(attnScore, ckptBefore, n, x, x);
        }
    }

    @Override
    public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
        checkStep(n);
        int len = n * nEmbd;
        CpuHyperConnections.requireLength("branch", branch, len);
        if (banked) {
            System.arraycopy(branch, 0, prefix, 0, len);
        } else {
            CpuOps.add(prefix, 0, branch, 0, prefix, 0, len);
        }
        if (ckptAfter > 0) {
            mixInto(ffnScore, ckptAfter, n, prefix, x);
        } else {
            System.arraycopy(prefix, 0, x, 0, len);
        }
        return true;
    }

    @Override
    public boolean afterFfn(int pos0, int n, float[] x, float[] branch) {
        checkStep(n);
        int len = n * nEmbd;
        CpuHyperConnections.requireLength("branch", branch, len);
        CpuOps.add(prefix, 0, branch, 0, x, 0, len);
        return true;
    }

    @Override
    public void afterBlock(int pos0, int n, float[] x) {
        checkStep(n);
        if (last && ckptFinal > 0) {
            mixInto(outScore, ckptFinal, n, x, x);
        }
    }

    /**
     * {@code out[0, n*n_embd) = mix(stack[0, nCkpt), cur)} scored with {@code scoreW}; {@code out} may be
     * {@code cur}.
     */
    private void mixInto(float[] scoreW, int nCkpt, int n, float[] cur, float[] out) {
        final float[] st = stack;
        final float[] pr = probs;
        CpuParallel.parallelFor(n, TOKEN_GRAIN, (from, to) -> {
            CpuAttnResidualStack.scores(n, nEmbd, nCkpt, eps, st, 0, cur, 0, scoreW, pr, 0, from, to);
            CpuAttnResidualStack.mix(n, nEmbd, nCkpt, st, 0, cur, 0, pr, 0, out, 0, from, to);
        });
    }

    private void checkStep(int n) {
        if (stack == null) {
            throw new IllegalStateException("attention-residual extra used before bind()");
        }
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("attention residuals of layer " + il + ": token count " + n
                    + " is outside [1, " + ubatch + "]");
        }
    }

    @Override
    public void close() {
        stack = null;
        prefix = null;
        probs = null;
    }

    @Override
    public String toString() {
        return "CpuAttnResidualExtra{layer " + il + (banked ? ", banks" : "") + ", ckpt " + ckptBefore + "->"
                + ckptAfter + (last ? ", final " + ckptFinal : "") + '}';
    }
}
