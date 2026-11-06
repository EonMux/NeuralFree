package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;

import java.util.Objects;

/**
 * The stateful executor of {@link PocketTtsMimiDecoder}: one session carries the sliding transformer's key/value
 * window and RoPE position counter, the depthwise upsample's overlap tail and every causal convolution's left
 * context or overlap tail across calls (tools/mtmd/models/pockettts-gen.cpp:190-294, pockettts-seanet.cpp).
 *
 * <p>All signals are frame major ({@code [frames][channels]}, element {@code (t, c)} at
 * {@code v[t * channels + c]}); convolution weights are ggml arrays read at bind. A call decodes {@code 1..window}
 * latents and emits exactly {@code n * hop()} samples, so consecutive windows concatenate to the waveform of the
 * whole sequence. Not thread-safe.
 */
public final class PocketTtsMimiSession implements PocketTtsMimiDecoder.Decoder {

    /** GELU's tanh-approximation constants ({@code ggml_gelu}). */
    private static final double GELU_C1 = 0.7978845608028654;
    private static final double GELU_C2 = 0.044715;

    private final PocketTtsMimiDecoder geo;
    private final CpuMatVec matVec;
    private final int width;
    private final int latent;
    private final int nLayer;
    private final int nHead;
    private final int nHeadKv;
    private final int dHead;
    private final int kvCh;
    private final int prefix;
    private final int context;
    private final int hop;

    private int tfmPos;
    private final float[][] kPrefix;
    private final float[][] vPrefix;
    private final float[][] upState;
    private final float[][] decInState;
    private final float[][][] decUpStates;
    private final float[][][] decResStates;
    private final float[][] decOutState;
    private boolean closed;

    PocketTtsMimiSession(PocketTtsMimiDecoder geo) {
        this.geo = Objects.requireNonNull(geo, "geo");
        this.matVec = geo.weights().matVec();
        this.width = geo.width();
        this.latent = geo.latent();
        this.nLayer = geo.nLayer();
        this.nHead = geo.nHead();
        this.nHeadKv = geo.nHeadKv();
        this.dHead = geo.dHead();
        this.kvCh = nHeadKv * dHead;
        this.prefix = geo.context() - 1;
        this.context = geo.context();
        this.hop = geo.hop();
        this.kPrefix = new float[nLayer][prefix * kvCh];
        this.vPrefix = new float[nLayer][prefix * kvCh];
        this.upState = new float[geo.upsampleK() - PocketTtsMimiDecoder.DOWNSAMPLE][width];
        this.decInState = new float[geo.convInK() - 1][geo.convInChannels()];
        PocketTtsMimiDecoder.Stage[] stages = geo.stages();
        this.decUpStates = new float[stages.length][][];
        this.decResStates = new float[stages.length][][];
        for (int i = 0; i < stages.length; i++) {
            PocketTtsMimiDecoder.Stage stage = stages[i];
            decUpStates[i] = new float[stage.scaleConvK - stage.stride][stage.outChannels];
            decResStates[i] = new float[stage.resConv1K - 1][stage.channels];
        }
        this.decOutState = new float[geo.convOutK() - 1]
                [geo.stages()[geo.stages().length - 1].outChannels];
    }

    /**
     * Decodes {@code frames} continuous latents to PCM.
     *
     * @param frames {@code [n][latent()]} latents, {@code n} in {@code [1, windowFrames()]}
     * @return {@code n * hop()} mono samples in {@code [-1, 1]}
     * @throws IllegalArgumentException when the batch is empty, longer than a window, a row is another width, or a
     *                                  value is not finite
     * @throws IllegalStateException    when the session is closed
     */
    @Override
    public float[] decode(float[][] frames) {
        ensureOpen();
        Objects.requireNonNull(frames, "frames");
        int n = frames.length;
        if (n < 1 || n > PocketTtsMimiDecoder.WINDOW_FRAMES) {
            throw new IllegalArgumentException("a mimi batch holds 1 to " + PocketTtsMimiDecoder.WINDOW_FRAMES
                    + " frames, got " + n);
        }
        for (int f = 0; f < n; f++) {
            if (frames[f] == null || frames[f].length != latent) {
                throw new IllegalArgumentException("frame " + f + " holds "
                        + (frames[f] == null ? "no" : frames[f].length) + " values, a latent holds " + latent);
            }
            for (int i = 0; i < latent; i++) {
                if (!Float.isFinite(frames[f][i])) {
                    throw new IllegalArgumentException("frame " + f + " value " + i + " is " + frames[f][i]);
                }
            }
        }

        float[][] x = quantize(frames, n);
        x = depthwiseTranspose(x, geo.upsampleW(), geo.upsampleK(), PocketTtsMimiDecoder.DOWNSAMPLE, width, upState);
        x = transformer(x);
        x = seanet(x);
        float[] out = new float[n * hop];
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(-1f, Math.min(1f, x[i][0]));
        }
        return out;
    }

    /** {@link #decode(float[][])} as an {@link AudioClip}. */
    @Override
    public AudioClip decodeClip(float[][] frames) {
        return new AudioClip(decode(frames), geo.sampleRate());
    }

    /** The RoPE position of the next call's first upsampled frame. */
    @Override
    public int position() {
        return tfmPos;
    }

    /** Frees nothing (the weights belong to {@link PocketTtsMimiDecoder}); marks the session closed. */
    @Override
    public int sampleRate() {
        return geo.sampleRate();
    }

    @Override
    public void close() {
        closed = true;
    }

    // ------------------------------------------------------------------ quantizer

    /** Denormalizes the latents by the checkpoint's mean and standard deviation, then projects them to the width. */
    private float[][] quantize(float[][] frames, int n) {
        float[] feats = new float[n * latent];
        float[] std = geo.embStd();
        float[] mean = geo.embMean();
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < latent; i++) {
                feats[t * latent + i] = frames[t][i] * std[i] + mean[i];
            }
        }
        float[] projected = new float[n * width];
        matVec.matmul(geo.quantOut(), feats, 0, latent, projected, 0, width, n);
        return reshape(projected, n, width);
    }

    // ------------------------------------------------------------------ transformer

    /** The sliding-window transformer over the upsampled frames, with the persisted key/value window. */
    private float[][] transformer(float[][] input) {
        int nPos = input.length;
        int base = tfmPos;
        for (int il = 0; il < nLayer; il++) {
            input = layer(input, geo.layers()[il], il, base);
        }
        tfmPos += nPos;
        return input;
    }

    private float[][] layer(float[][] x, PocketTtsMimiDecoder.TransformerLayer layer, int il, int base) {
        int nPos = x.length;
        float[] normed = new float[nPos * width];
        for (int i = 0; i < nPos; i++) {
            layerNorm(x[i], layer.ln1W, layer.ln1B, geo.eps(), normed, i * width);
        }
        float[] q = matVec(layer.q, normed, width, nHead * dHead);
        float[] k = matVec(layer.k, normed, width, nHeadKv * dHead);
        float[] v = matVec(layer.v, normed, width, nHeadKv * dHead);
        for (int i = 0; i < nPos; i++) {
            rope(q, i * nHead * dHead, nHead, dHead, base + i);
            rope(k, i * nHeadKv * dHead, nHeadKv, dHead, base + i);
        }
        float[][] keys = new float[prefix + nPos][kvCh];
        float[][] values = new float[prefix + nPos][kvCh];
        for (int j = 0; j < prefix; j++) {
            System.arraycopy(kPrefix[il], j * kvCh, keys[j], 0, kvCh);
            System.arraycopy(vPrefix[il], j * kvCh, values[j], 0, kvCh);
        }
        for (int i = 0; i < nPos; i++) {
            System.arraycopy(k, i * kvCh, keys[prefix + i], 0, kvCh);
            System.arraycopy(v, i * kvCh, values[prefix + i], 0, kvCh);
        }
        float[] attn = attention(q, keys, values, base);
        float[] projected = matVec(layer.out, attn, nHead * dHead, width);
        float[][] out = new float[nPos][width];
        for (int i = 0; i < nPos; i++) {
            for (int c = 0; c < width; c++) {
                out[i][c] = x[i][c] + projected[i * width + c] * layer.ls1[c];
            }
        }
        float[] normed2 = new float[nPos * width];
        for (int i = 0; i < nPos; i++) {
            layerNorm(out[i], layer.ln2W, layer.ln2B, geo.eps(), normed2, i * width);
        }
        int nFf = geo.nFf();
        float[] up = matVec(layer.up, normed2, width, nFf);
        for (int i = 0; i < up.length; i++) {
            up[i] = geluTanh(up[i]);
        }
        float[] down = matVec(layer.down, up, nFf, width);
        for (int i = 0; i < nPos; i++) {
            for (int c = 0; c < width; c++) {
                out[i][c] += down[i * width + c] * layer.ls2[c];
            }
        }
        // the window keeps the most recent prefix frames of the concatenated keys and values
        for (int j = 0; j < prefix; j++) {
            int src = nPos + j;
            System.arraycopy(keys[src], 0, kPrefix[il], j * kvCh, kvCh);
            System.arraycopy(values[src], 0, vPrefix[il], j * kvCh, kvCh);
        }
        return out;
    }

    /**
     * Banded causal attention over the persisted window and the current chunk: query {@code i} attends to key
     * {@code j} when {@code 0 <= prefix + i - j < context} and the key's real position {@code j - prefix + base}
     * is not negative (the cold-start mask of pockettts-gen.cpp:225-233).
     */
    private float[] attention(float[] q, float[][] keys, float[][] values, int base) {
        int nPos = q.length / (nHead * dHead);
        int total = keys.length;
        int headsPerKv = nHead / nHeadKv;
        float scale = (float) (1.0 / Math.sqrt(dHead));
        float[] out = new float[nPos * nHead * dHead];
        float[] scores = new float[total];
        for (int i = 0; i < nPos; i++) {
            for (int h = 0; h < nHead; h++) {
                int kvHead = h / headsPerKv;
                int qOff = i * nHead * dHead + h * dHead;
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < total; j++) {
                    int keyPos = j - prefix + base;
                    int diff = prefix + i - j;
                    if (keyPos < 0 || diff < 0 || diff >= context) {
                        scores[j] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    float dot = 0f;
                    for (int d = 0; d < dHead; d++) {
                        dot += q[qOff + d] * keys[j][kvHead * dHead + d];
                    }
                    scores[j] = dot * scale;
                    if (scores[j] > max) {
                        max = scores[j];
                    }
                }
                double sum = 0.0;
                for (int j = 0; j < total; j++) {
                    if (scores[j] == Float.NEGATIVE_INFINITY) {
                        scores[j] = 0f;
                        continue;
                    }
                    double e = Math.exp(scores[j] - max);
                    scores[j] = (float) e;
                    sum += e;
                }
                float inv = sum > 0.0 ? (float) (1.0 / sum) : 0f;
                for (int d = 0; d < dHead; d++) {
                    float acc = 0f;
                    for (int j = 0; j < total; j++) {
                        if (scores[j] != 0f) {
                            acc += scores[j] * values[j][kvHead * dHead + d];
                        }
                    }
                    out[qOff + d] = acc * inv;
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ SEANet decoder

    private float[][] seanet(float[][] input) {
        float[][] x = causalConv(input, geo.convInW(), geo.convInB(), geo.convInK(), geo.convInChannels(),
                geo.convInChannels(), decInState);
        PocketTtsMimiDecoder.Stage[] stages = geo.stages();
        for (int i = 0; i < stages.length; i++) {
            PocketTtsMimiDecoder.Stage stage = stages[i];
            x = elu(x);
            x = causalTranspose(x, stage.scaleConvW, stage.scaleConvB, stage.scaleConvK, stage.stride,
                    stage.scaleInChannels, stage.outChannels, decUpStates[i]);
            x = residualUnit(x, stage, decResStates[i]);
        }
        x = elu(x);
        return causalConv(x, geo.convOutW(), geo.convOutB(), geo.convOutK(), geo.stages()[stages.length - 1]
                .outChannels, 1, decOutState);
    }

    /** {@code elu -> dilated causal conv -> elu -> pointwise conv -> residual} (pockettts-seanet.cpp:126-136). */
    private float[][] residualUnit(float[][] x, PocketTtsMimiDecoder.Stage stage, float[][] state) {
        float[][] h = elu(x);
        h = causalConv(h, stage.resConv1W, stage.resConv1B, stage.resConv1K, stage.channels, stage.channels, state);
        h = elu(h);
        h = causalConv(h, stage.resConv2W, stage.resConv2B, 1, stage.channels, stage.channels, null);
        float[][] out = new float[x.length][stage.channels];
        for (int i = 0; i < x.length; i++) {
            for (int c = 0; c < stage.channels; c++) {
                out[i][c] = x[i][c] + h[i][c];
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ convolution primitives

    /**
     * Causal convolution {@code y[t][oc] = b[oc] + sum_{k,ic} w[k][ic][oc] * xfull[t+k][ic]} whose left context is
     * the persisted {@code state} ({@code K-1} rows); the state becomes the input's last row.
     */
    private static float[][] causalConv(float[][] x, float[] w, float[] b, int k, int in, int out,
                                        float[][] state) {
        int t = x.length;
        int pad = k - 1;
        float[][] full = new float[pad + t][in];
        for (int i = 0; i < pad; i++) {
            System.arraycopy(state[i], 0, full[i], 0, in);
        }
        for (int i = 0; i < t; i++) {
            System.arraycopy(x[i], 0, full[pad + i], 0, in);
        }
        float[][] y = new float[t][out];
        for (int i = 0; i < t; i++) {
            for (int oc = 0; oc < out; oc++) {
                float acc = b[oc];
                for (int ic = 0; ic < in; ic++) {
                    int wBase = ic + in * oc;
                    for (int j = 0; j < k; j++) {
                        acc += w[j + k * wBase] * full[i + j][ic];
                    }
                }
                y[i][oc] = acc;
            }
        }
        if (state != null) {
            for (int i = 0; i < pad; i++) {
                System.arraycopy(full[t + i], 0, state[i], 0, in);
            }
        }
        return y;
    }

    /**
     * Causal depthwise ConvTranspose1d ({@code w[K][1][C]}): the overlap tail of the transposed convolution is
     * added from {@code state} and the new tail kept; the emitted samples ({@code T * stride}) exclude both tails.
     */
    private static float[][] depthwiseTranspose(float[][] x, float[] w, int k, int stride, int channels,
                                                float[][] state) {
        int t = x.length;
        int trim = k - stride;
        int fullLen = (t - 1) * stride + k;
        float[][] full = new float[fullLen][channels];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < k; j++) {
                int row = i * stride + j;
                for (int c = 0; c < channels; c++) {
                    full[row][c] += w[j + k * c] * x[i][c];
                }
            }
        }
        return overlapAdd(full, trim, t * stride, channels, state);
    }

    /**
     * Causal ConvTranspose1d over {@code [K, OC, IC]} weights: the overlap tail of the transposed convolution is
     * added from {@code state} and the new tail kept; the emitted samples ({@code T * stride}) exclude both tails.
     */
    private static float[][] causalTranspose(float[][] x, float[] w, float[] b, int k, int stride, int in, int out,
                                             float[][] state) {
        int t = x.length;
        int trim = k - stride;
        int fullLen = (t - 1) * stride + k;
        float[][] full = new float[fullLen][out];
        for (int i = 0; i < t; i++) {
            for (int ic = 0; ic < in; ic++) {
                float v = x[i][ic];
                for (int j = 0; j < k; j++) {
                    int row = i * stride + j;
                    for (int oc = 0; oc < out; oc++) {
                        full[row][oc] += w[j + k * (oc + out * ic)] * v;
                    }
                }
            }
        }
        float[][] y = overlapAdd(full, trim, t * stride, out, state);
        for (int i = 0; i < y.length; i++) {
            for (int oc = 0; oc < out; oc++) {
                y[i][oc] += b[oc];
            }
        }
        return y;
    }

    /**
     * The streaming half of a transposed convolution: the first {@code trim} rows add the previous call's tail,
     * the emitted rows are the first {@code emit} rows, and the trailing {@code trim} rows become the new tail.
     */
    private static float[][] overlapAdd(float[][] full, int trim, int emit, int channels, float[][] state) {
        float[][] y = new float[emit][channels];
        for (int i = 0; i < trim; i++) {
            for (int c = 0; c < channels; c++) {
                y[i][c] = full[i][c] + state[i][c];
            }
        }
        for (int i = trim; i < emit; i++) {
            System.arraycopy(full[i], 0, y[i], 0, channels);
        }
        for (int i = 0; i < trim; i++) {
            System.arraycopy(full[emit + i], 0, state[i], 0, channels);
        }
        return y;
    }

    // ------------------------------------------------------------------ normalization

    /** LayerNorm over the channel axis of one frame, written at {@code out[off]}. */
    private static void layerNorm(float[] x, float[] weight, float[] bias, float eps, float[] out, int off) {
        int n = x.length;
        double mean = 0.0;
        for (float v : x) {
            mean += v;
        }
        mean /= n;
        double var = 0.0;
        for (float v : x) {
            double d = v - mean;
            var += d * d;
        }
        var /= n;
        float scale = (float) (1.0 / Math.sqrt(var + eps));
        for (int i = 0; i < n; i++) {
            out[off + i] = (float) ((x[i] - mean) * scale) * weight[i] + bias[i];
        }
    }

    // ------------------------------------------------------------------ helpers

    /** NORMAL RoPE of one head: adjacent pairs {@code (2i, 2i+1)} rotated by {@code pos * theta^(-2i/dHead)}. */
    private static void rope(float[] x, int off, int nHead, int dHead, int pos) {
        int half = dHead / 2;
        for (int h = 0; h < nHead; h++) {
            int base = off + h * dHead;
            for (int i = 0; i < half; i++) {
                double angle = pos * Math.pow(PocketTtsMimiDecoder.ROPE_THETA, -2.0 * i / dHead);
                float cos = (float) Math.cos(angle);
                float sin = (float) Math.sin(angle);
                float x0 = x[base + 2 * i];
                float x1 = x[base + 2 * i + 1];
                x[base + 2 * i] = x0 * cos - x1 * sin;
                x[base + 2 * i + 1] = x0 * sin + x1 * cos;
            }
        }
    }

    private static float[][] elu(float[][] x) {
        float[][] y = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            float[] row = new float[x[i].length];
            for (int c = 0; c < row.length; c++) {
                row[c] = x[i][c] > 0f ? x[i][c] : (float) (Math.expm1(x[i][c]));
            }
            y[i] = row;
        }
        return y;
    }

    private static float geluTanh(float x) {
        double inner = GELU_C1 * (x + GELU_C2 * x * x * x);
        return (float) (0.5 * x * (1.0 + Math.tanh(inner)));
    }

    private static float[] flatten(float[][] rows, int width) {
        float[] out = new float[rows.length * width];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, out, i * width, width);
        }
        return out;
    }

    private static float[][] reshape(float[] flat, int rows, int width) {
        float[][] out = new float[rows][width];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * width, out[i], 0, width);
        }
        return out;
    }

    private float[] matVec(CpuTensorRef w, float[] x, int in, int out) {
        float[] y = new float[x.length / in * out];
        matVec.matmul(w, x, 0, in, y, 0, out, x.length / in);
        return y;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the mimi decoder session is closed");
        }
    }

    @Override
    public String toString() {
        return "PocketTtsMimiSession{position=" + tfmPos + ", layers=" + nLayer + ", hop=" + hop + '}';
    }
}
