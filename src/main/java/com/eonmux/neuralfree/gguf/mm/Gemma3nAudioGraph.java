package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.Gemma3nAudioModel.block;

/**
 * CPU executor of the Gemma 3n audio encoder over {@link Gemma3nAudioModel}'s tensors, a transcription of Hugging
 * Face transformers {@code Gemma3nAudioEncoder} followed by {@code Gemma3nMultimodalEmbedder} and the padding of
 * {@code Gemma3nModel.forward} (models/gemma3n/modeling_gemma3n.py). One call encodes one log-mel window:
 * <ol>
 *   <li>two sub-sample blocks, each a 3x3 stride-2 convolution (frequency padded by one on both sides, time by two
 *       at the end), the cumulative group norm over the {@code [freq x channel]} values of every frame so far, and
 *       ReLU; then {@code a.pre_encode.out} over the frequency-major flatten;</li>
 *   <li>per conformer block: half a SiLU feed-forward, chunked local self-attention (each frame attends itself and
 *       the 12 before it, with sinusoidal relative position logits through {@code linear_pos}, a softplus
 *       per-dimension query scale and a tanh logit cap of 50), the light conv (GLU, causal depthwise conv, RMS
 *       norm, SiLU), the second half feed-forward and the block RMS norm;</li>
 *   <li>every fourth row through the soft-token embedder (RMS norm, {@code mm.a.input_projection}, RMS norm
 *       without scale), padded to {@value Gemma3nAudioModel#TOKENS_PER_WINDOW} rows with the embedding of the last
 *       hard token.</li>
 * </ol>
 * Matmuls run through {@link CpuMatVec} in the stored weight types; norms and statistics accumulate in double.
 * Not thread-safe.
 */
final class Gemma3nAudioGraph implements FamilyEncoder {

    /** im2col floats one sub-sample convolution batch may hold. */
    private static final int IM2COL_FLOATS = 1 << 22;

    private final Gemma3nAudioModel model;
    private final CpuMatVec matVec;
    private final int hidden;
    private final int nHead;
    private final int headDim;
    private final float eps;
    private final CpuTensorRef conv0;
    private final float[] conv0Norm;
    private final CpuTensorRef conv1;
    private final float[] conv1Norm;
    private final CpuTensorRef inputProj;
    private final Block[] blocks;
    private final float[] softNorm;
    private final CpuTensorRef projection;
    private final float[] paddingRow;

    /** The weights of one conformer block, with the query scale and relative position rows precomputed. */
    private static final class Block {
        final float[] ffnNorm;
        final CpuTensorRef ffnUp;
        final CpuTensorRef ffnDown;
        final float[] ffnPostNorm;
        final float[] attnNorm;
        final CpuTensorRef q;
        final CpuTensorRef k;
        final CpuTensorRef v;
        final CpuTensorRef out;
        final float[] attnPostNorm;
        final float[] qScale;
        final float[] rel;
        final float[] convNorm;
        final CpuTensorRef pw1;
        final float[] dw;
        final float[] convPostNorm;
        final CpuTensorRef pw2;
        final float[] ffnNorm1;
        final CpuTensorRef ffnUp1;
        final CpuTensorRef ffnDown1;
        final float[] ffnPostNorm1;
        final float[] blockNorm;

        Block(int il, Function<String, float[]> floats, Function<String, CpuTensorRef> refs, float[] qScale,
              float[] rel) {
            ffnNorm = floats.apply(block(il, "ffn_norm.weight"));
            ffnUp = refs.apply(block(il, "ffn_up.weight"));
            ffnDown = refs.apply(block(il, "ffn_down.weight"));
            ffnPostNorm = floats.apply(block(il, "ffn_post_norm.weight"));
            attnNorm = floats.apply(block(il, "ln1.weight"));
            q = refs.apply(block(il, "attn_q.weight"));
            k = refs.apply(block(il, "attn_k.weight"));
            v = refs.apply(block(il, "attn_v.weight"));
            out = refs.apply(block(il, "attn_out.weight"));
            attnPostNorm = floats.apply(block(il, "ln2.weight"));
            this.qScale = qScale;
            this.rel = rel;
            convNorm = floats.apply(block(il, "conv_norm.weight"));
            pw1 = refs.apply(block(il, "conv_pw1.weight"));
            dw = floats.apply(block(il, "conv_dw.weight"));
            convPostNorm = floats.apply(block(il, "norm_conv.weight"));
            pw2 = refs.apply(block(il, "conv_pw2.weight"));
            ffnNorm1 = floats.apply(block(il, "ffn_norm_1.weight"));
            ffnUp1 = refs.apply(block(il, "ffn_up_1.weight"));
            ffnDown1 = refs.apply(block(il, "ffn_down_1.weight"));
            ffnPostNorm1 = floats.apply(block(il, "ffn_post_norm_1.weight"));
            blockNorm = floats.apply(block(il, "layer_pre_norm.weight"));
        }
    }

    private Gemma3nAudioGraph(Gemma3nAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                              Function<String, CpuTensorRef> refs) {
        this.model = model;
        this.matVec = matVec;
        ClipHparams hp = model.hparams();
        this.hidden = hp.nEmbd();
        this.nHead = hp.nHead();
        this.headDim = hidden / nHead;
        this.eps = hp.eps();
        int k2 = Gemma3nAudioModel.SSCP_KERNEL * Gemma3nAudioModel.SSCP_KERNEL;
        this.conv0 = CpuVitTower.matrix(refs.apply("a.conv1d.0.weight"), model.c0(), k2);
        this.conv0Norm = floats.apply("a.conv1d.0.norm.weight");
        this.conv1 = CpuVitTower.matrix(refs.apply("a.conv1d.1.weight"), model.c1(), (long) k2 * model.c0());
        this.conv1Norm = floats.apply("a.conv1d.1.norm.weight");
        this.inputProj = refs.apply("a.pre_encode.out.weight");
        float[] timing = timingSignal(hidden);
        this.blocks = new Block[hp.nLayer()];
        for (int il = 0; il < blocks.length; il++) {
            float[] qScale = queryScale(floats.apply(block(il, "per_dim_scale")));
            float[] rel = new float[(Gemma3nAudioModel.MAX_PAST + 1) * hidden];
            matVec.matmul(refs.apply(block(il, "linear_pos.weight")), timing, 0, rel, 0,
                    Gemma3nAudioModel.MAX_PAST + 1);
            blocks[il] = new Block(il, floats, refs, qScale, rel);
        }
        this.softNorm = floats.apply("mm.a.soft_emb_norm.weight");
        this.projection = refs.apply("mm.a.input_projection.weight");
        float[] hard = new float[hidden];
        System.arraycopy(floats.apply("mm.a.embedding.weight"), (model.vocab() - 1) * hidden, hard, 0, hidden);
        CpuOps.rmsNorm(hard, 0, hard, 0, 1, hidden, floats.apply("mm.a.hard_emb_norm.weight"), eps);
        this.paddingRow = embed(hard, 1)[0];
    }

    /** Binds an executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static Gemma3nAudioGraph bind(Gemma3nAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                                  Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(matVec, "matVec");
        Objects.requireNonNull(floats, "floats");
        Objects.requireNonNull(refs, "refs");
        return new Gemma3nAudioGraph(model, matVec, floats, refs);
    }

    /**
     * {@code Gemma3nAudioRelativePositionEmbedding._get_timing_signal_1d_pos} for the relative positions
     * {@code 12 .. 0}: row {@code f} holds {@code [sin(p * inv), cos(p * inv)]} of position {@code p = 12 - f}.
     */
    private static float[] timingSignal(int hidden) {
        int half = hidden / 2;
        double increment = Math.log(1.0e4) / Math.max(half - 1, 1);
        int nPos = Gemma3nAudioModel.MAX_PAST + 1;
        float[] out = new float[nPos * hidden];
        for (int f = 0; f < nPos; f++) {
            int p = Gemma3nAudioModel.MAX_PAST - f;
            for (int i = 0; i < half; i++) {
                double angle = p * Math.exp(-i * increment);
                out[f * hidden + i] = (float) Math.sin(angle);
                out[f * hidden + half + i] = (float) Math.cos(angle);
            }
        }
        return out;
    }

    /** {@code head_dim^-0.5 / softplus(0) * softplus(per_dim_scale)}, the per-dimension query scale. */
    private static float[] queryScale(float[] perDimScale) {
        double base = 1.0 / Math.sqrt(perDimScale.length) / Math.log(2.0);
        float[] out = new float[perDimScale.length];
        for (int d = 0; d < out.length; d++) {
            out[d] = (float) (base * softplus(perDimScale[d]));
        }
        return out;
    }

    private static double softplus(double x) {
        return x > 20.0 ? x : Math.log1p(Math.exp(x));
    }

    /**
     * Encodes one mel window ({@code [nMel][nLen]}, mel-major) to
     * {@value Gemma3nAudioModel#TOKENS_PER_WINDOW} text-model rows: the encoder's rows, then the padding row.
     *
     * @throws IllegalArgumentException when the chunk has another mel width, or the window gives more rows than
     *                                  the prompt reserves (more than 30 s)
     */
    @Override
    public float[][] encode(MelChunk chunk) {
        if (chunk.nMel() != model.nMel()) {
            throw new IllegalArgumentException("mel chunk has " + chunk.nMel() + " bins, the encoder needs "
                    + model.nMel());
        }
        int nFrames = chunk.nLen();
        int rows = Gemma3nAudioModel.encoderRows(nFrames);
        if (rows > Gemma3nAudioModel.TOKENS_PER_WINDOW) {
            throw new IllegalArgumentException("mel window of " + nFrames + " frames gives " + rows + " gemma3na"
                    + " rows, above the " + Gemma3nAudioModel.TOKENS_PER_WINDOW + " one audio window reserves"
                    + " (30 s; the gemma3na frontend splits longer clips)");
        }
        int nMel = model.nMel();
        float[] img = new float[nFrames * nMel];
        for (int m = 0; m < nMel; m++) {
            for (int t = 0; t < nFrames; t++) {
                img[t * nMel + m] = chunk.data()[m * nFrames + t];
            }
        }
        int t0 = Gemma3nAudioModel.convOut(nFrames);
        img = subsample(img, nFrames, nMel, 1, conv0, model.c0(), conv0Norm);
        img = subsample(img, t0, model.f0(), model.c0(), conv1, model.c1(), conv1Norm);
        int n = Gemma3nAudioModel.convOut(t0);
        float[] x = new float[n * hidden];
        matVec.matmul(inputProj, img, 0, x, 0, n);
        for (Block b : blocks) {
            conformer(b, x, n);
        }
        float[] reduced = new float[rows * hidden];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(x, r * Gemma3nAudioModel.REDUCTION * hidden, reduced, r * hidden, hidden);
        }
        CpuOps.rmsNorm(reduced, 0, reduced, 0, rows, hidden, softNorm, eps);
        float[][] body = embed(reduced, rows);
        float[][] out = new float[Gemma3nAudioModel.TOKENS_PER_WINDOW][];
        for (int r = 0; r < out.length; r++) {
            out[r] = r < rows ? body[r] : paddingRow.clone();
        }
        return out;
    }

    /** {@code mm.a.input_projection} then the scale-free RMS norm, over normed rows. */
    private float[][] embed(float[] normed, int rows) {
        int d = model.hparams().projectionDim();
        float[] y = new float[rows * d];
        matVec.matmul(projection, normed, 0, y, 0, rows);
        CpuOps.rmsNorm(y, 0, y, 0, rows, d, null, eps);
        return VisionRows.tokenRows(y, d);
    }

    // ------------------------------------------------------------------ sub-sample conv projection

    /**
     * One {@code Gemma3nAudioSSCPConvBlock} over a time-major image ({@code (t * f + freq) * cIn + c}): the
     * convolution as batched im2col matmuls (slot {@code (c * 3 + dt) * 3 + df}), the cumulative group norm and
     * ReLU. Returns the {@code [t' x f' x cOut]} image.
     */
    private float[] subsample(float[] in, int t, int f, int cIn, CpuTensorRef kernel, int cOut, float[] norm) {
        int k = Gemma3nAudioModel.SSCP_KERNEL;
        int s = Gemma3nAudioModel.SSCP_STRIDE;
        int to = Gemma3nAudioModel.convOut(t);
        int fo = Gemma3nAudioModel.convOut(f);
        int span = cIn * k * k;
        int batch = Math.max(1, Math.min(to, IM2COL_FLOATS / (fo * span)));
        float[] cols = new float[batch * fo * span];
        float[] out = new float[to * fo * cOut];
        for (int start = 0; start < to; start += batch) {
            int nt = Math.min(batch, to - start);
            for (int tt = 0; tt < nt; tt++) {
                for (int fi = 0; fi < fo; fi++) {
                    int base = (tt * fo + fi) * span;
                    for (int c = 0; c < cIn; c++) {
                        for (int dt = 0; dt < k; dt++) {
                            int sy = (start + tt) * s + dt;
                            for (int df = 0; df < k; df++) {
                                int sx = fi * s + df - 1;
                                boolean inside = sy < t && sx >= 0 && sx < f;
                                cols[base + (c * k + dt) * k + df] = inside ? in[(sy * f + sx) * cIn + c] : 0f;
                            }
                        }
                    }
                }
            }
            matVec.matmul(kernel, cols, 0, out, start * fo * cOut, nt * fo);
        }
        cumulativeGroupNorm(out, to, fo * cOut, cOut, norm);
        return out;
    }

    /**
     * {@code Gemma3nAudioCumulativeGroupNorm} then ReLU, in place: frame {@code t} is normalized by the mean and
     * variance of every value of frames {@code 0..t} (each frame's squared deviations taken from its own running
     * mean), then scaled per channel.
     */
    private static void cumulativeGroupNorm(float[] x, int nT, int width, int channels, float[] weight) {
        double sum = 0;
        double sq = 0;
        long count = 0;
        for (int t = 0; t < nT; t++) {
            int base = t * width;
            for (int i = 0; i < width; i++) {
                sum += x[base + i];
            }
            count += width;
            double mean = sum / count;
            for (int i = 0; i < width; i++) {
                double d = x[base + i] - mean;
                sq += d * d;
            }
            double inv = 1.0 / Math.sqrt(sq / count + Gemma3nAudioModel.GROUP_NORM_EPS);
            for (int i = 0; i < width; i++) {
                double v = (x[base + i] - mean) * inv * weight[i % channels];
                x[base + i] = v > 0 ? (float) v : 0f;
            }
        }
    }

    // ------------------------------------------------------------------ conformer

    /** One {@code Gemma3nAudioConformerBlock} over the {@code n} rows of {@code x}, in place. */
    private void conformer(Block b, float[] x, int n) {
        feedForward(x, n, b.ffnNorm, b.ffnUp, b.ffnDown, b.ffnPostNorm);
        attention(b, x, n);
        lightConv(b, x, n);
        feedForward(x, n, b.ffnNorm1, b.ffnUp1, b.ffnDown1, b.ffnPostNorm1);
        clamp(x);
        CpuOps.rmsNorm(x, 0, x, 0, n, hidden, b.blockNorm, eps);
    }

    /** {@code x += 0.5 * norm(clamp(down(silu(up(norm(clamp(x)))))))}. */
    private void feedForward(float[] x, int n, float[] preNorm, CpuTensorRef up, CpuTensorRef down,
                             float[] postNorm) {
        float[] y = x.clone();
        clamp(y);
        CpuOps.rmsNorm(y, 0, y, 0, n, hidden, preNorm, eps);
        float[] h = new float[n * model.ff()];
        matVec.matmul(up, y, 0, h, 0, n);
        VisionRows.activate(ActOp.SILU, h);
        matVec.matmul(down, h, 0, y, 0, n);
        clamp(y);
        CpuOps.rmsNorm(y, 0, y, 0, n, hidden, postNorm, eps);
        for (int i = 0; i < x.length; i++) {
            x[i] += Gemma3nAudioModel.RESIDUAL_WEIGHT * y[i];
        }
    }

    /** {@code Gemma3nAudioConformerAttention}: {@code x += norm(clamp(attn_out(attention(norm(clamp(x))))))}. */
    private void attention(Block b, float[] x, int n) {
        float[] y = x.clone();
        clamp(y);
        CpuOps.rmsNorm(y, 0, y, 0, n, hidden, b.attnNorm, eps);
        float[] q = new float[n * hidden];
        float[] k = new float[n * hidden];
        float[] v = new float[n * hidden];
        matVec.matmul(b.q, y, 0, q, 0, n);
        matVec.matmul(b.k, y, 0, k, 0, n);
        matVec.matmul(b.v, y, 0, v, 0, n);
        for (int i = 0; i < q.length; i++) {
            q[i] *= b.qScale[i % headDim];
        }
        float[] ctx = new float[n * hidden];
        CpuParallel.parallelFor(n, 1, (from, to) -> {
            double[] p = new double[Gemma3nAudioModel.MAX_PAST + 1];
            for (int t = from; t < to; t++) {
                for (int head = 0; head < nHead; head++) {
                    attendRow(b.rel, q, k, v, t, head, p, ctx);
                }
            }
        });
        matVec.matmul(b.out, ctx, 0, y, 0, n);
        clamp(y);
        CpuOps.rmsNorm(y, 0, y, 0, n, hidden, b.attnPostNorm, eps);
        for (int i = 0; i < x.length; i++) {
            x[i] += y[i];
        }
    }

    /**
     * Query {@code t} of head {@code head} over the keys {@code t - 12 .. t}: content plus relative position
     * logits ({@code rel} row {@code 12 + s - t}), the tanh cap, softmax, and the weighted values into
     * {@code ctx}.
     */
    private void attendRow(float[] rel, float[] q, float[] k, float[] v, int t, int head, double[] p,
                           float[] ctx) {
        int first = Math.max(0, t - Gemma3nAudioModel.MAX_PAST);
        int qOff = t * hidden + head * headDim;
        double max = Double.NEGATIVE_INFINITY;
        for (int s = first; s <= t; s++) {
            int kOff = s * hidden + head * headDim;
            int rOff = (Gemma3nAudioModel.MAX_PAST + s - t) * hidden + head * headDim;
            double dot = 0;
            for (int d = 0; d < headDim; d++) {
                dot += (double) q[qOff + d] * ((double) k[kOff + d] + rel[rOff + d]);
            }
            double logit = Gemma3nAudioModel.LOGIT_CAP * Math.tanh(dot / Gemma3nAudioModel.LOGIT_CAP);
            p[s - first] = logit;
            max = Math.max(max, logit);
        }
        double sum = 0;
        for (int s = first; s <= t; s++) {
            p[s - first] = Math.exp(p[s - first] - max);
            sum += p[s - first];
        }
        for (int d = 0; d < headDim; d++) {
            double acc = 0;
            for (int s = first; s <= t; s++) {
                acc += p[s - first] * v[s * hidden + head * headDim + d];
            }
            ctx[qOff + d] = (float) (acc / sum);
        }
    }

    /**
     * {@code Gemma3nAudioConformerLightConv1d}: {@code x += pw2(silu(norm(clamp(dw(glu(pw1(norm(x))))))))}, the
     * depthwise convolution causal over the time axis.
     */
    private void lightConv(Block b, float[] x, int n) {
        float[] y = new float[n * hidden];
        CpuOps.rmsNorm(x, 0, y, 0, n, hidden, b.convNorm, eps);
        float[] h = new float[n * 2 * hidden];
        matVec.matmul(b.pw1, y, 0, h, 0, n);
        float[] glu = new float[n * hidden];
        CpuOps.gluRows(GluOp.SIGMUL, ActParams.NONE, n, hidden, h, hidden, 2 * hidden, h, 0, 2 * hidden, glu, 0,
                hidden);
        int kernel = model.convKernel();
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < hidden; c++) {
                double acc = 0;
                for (int j = 0; j < kernel; j++) {
                    int src = t + j - (kernel - 1);
                    if (src >= 0) {
                        acc += (double) b.dw[c * kernel + j] * glu[src * hidden + c];
                    }
                }
                y[t * hidden + c] = (float) acc;
            }
        }
        clamp(y);
        CpuOps.rmsNorm(y, 0, y, 0, n, hidden, b.convPostNorm, eps);
        VisionRows.activate(ActOp.SILU, y);
        matVec.matmul(b.pw2, y, 0, h, 0, n);
        for (int i = 0; i < x.length; i++) {
            x[i] += h[i];
        }
    }

    /** {@code torch.clamp(x, -gradient_clipping, gradient_clipping)} in place. */
    private static void clamp(float[] x) {
        float lim = Gemma3nAudioModel.GRADIENT_CLIP;
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.max(-lim, Math.min(lim, x[i]));
        }
    }

    @Override
    public String toString() {
        return "Gemma3nAudioGraph{" + model + "}";
    }
}
