package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.audio.AudioClip;

import java.util.Objects;

/**
 * The stateful executor of {@link Qwen3TtsCode2Wav}: one session carries the pre-transformer key/value prefix,
 * the RoPE position counter and every convolution's left-context or overlap tail across calls, so consecutive
 * windows concatenate to the waveform of the whole sequence (qwen3tts-gen.cpp:400-700).
 *
 * <p>All signals are token/frame major ({@code [frames][channels]}, element {@code (t, c)} at
 * {@code v[t * channels + c]}); convolution weights are the ggml {@code [K, IC, OC]} arrays read at bind. A call
 * decodes {@code 1..window} frames, rear-padding the batch with code 0 like the reference (clip.cpp:6278-6305)
 * and dropping the padded samples from the output. Not thread-safe.
 */
public final class Qwen3TtsCode2WavSession implements Qwen3TtsCode2Wav.Decoder {

    /** LayerNorm epsilon of the ConvNeXt blocks (qwen3tts-gen.cpp:500, {@code ggml_norm(ctx, hc, 1e-6f)}). */
    private static final float CONVNEXT_EPS = 1e-6f;
    /** GELU's tanh-approximation constants ({@code ggml_gelu}). */
    private static final double GELU_C1 = 0.7978845608028654;
    private static final double GELU_C2 = 0.044715;

    private final Qwen3TtsCode2Wav geo;
    private final CpuWeights weights;
    private final CpuMatVec matVec;
    private final int window;
    private final int prefix;
    private final int kvCh;

    private float tfmPos;
    private final float[][] kPrefix;
    private final float[][] vPrefix;
    private final float[][] preConvState;
    private final float[][][] upStates;
    private final float[][] dacEntryState;
    private final float[][][] dacTailState;
    private final float[][][] resStates;
    private final float[][] postConvState;
    private boolean closed;

    Qwen3TtsCode2WavSession(Qwen3TtsCode2Wav geo) {
        this.geo = Objects.requireNonNull(geo, "geo");
        this.weights = geo.weights();
        this.matVec = weights.matVec();
        this.window = geo.window();
        this.prefix = window - 1;
        this.kvCh = geo.nHeadKv() * geo.dHead();
        int layers = geo.nLayer();
        this.kPrefix = new float[layers][prefix * kvCh];
        this.vPrefix = new float[layers][prefix * kvCh];
        CpuWeights w = weights;
        this.preConvState = new float[geo.preConvK() - 1][geo.preConvIn()];
        Qwen3TtsCode2Wav.UpBlock[] ups = geo.upsample();
        this.upStates = new float[ups.length][][];
        for (int i = 0; i < ups.length; i++) {
            upStates[i] = new float[ups[i].dwK - 1][ups[i].outChannels];
        }
        this.dacEntryState = new float[geo.dacEntryK() - 1][geo.dacEntryIn()];
        Qwen3TtsCode2Wav.DacBlock[] dac = geo.dac();
        this.dacTailState = new float[dac.length][][];
        int resCount = 0;
        for (Qwen3TtsCode2Wav.DacBlock block : dac) {
            resCount += block.res.size();
        }
        this.resStates = new float[resCount][][];
        int at = 0;
        for (int ib = 0; ib < dac.length; ib++) {
            Qwen3TtsCode2Wav.DacBlock block = dac[ib];
            dacTailState[ib] = new float[block.stride][block.outChannels];
            for (int ir = 0; ir < block.res.size(); ir++) {
                Qwen3TtsCode2Wav.DacRes res = block.res.get(ir);
                int pad = (res.conv1K - 1) * Qwen3TtsCode2Wav.DAC_DILATIONS[ir];
                resStates[at++] = new float[pad][res.channels];
            }
        }
        this.postConvState = new float[geo.postConvK() - 1][geo.postConvIn()];
    }

    /**
     * Decodes {@code frames} RVQ frames to PCM.
     *
     * @param frames {@code [n][codesPerFrame()]} frame codes, {@code n} in {@code [1, window()]}
     * @return {@code n * hop()} mono samples in {@code [-1, 1]}
     * @throws IllegalArgumentException when the batch is empty, longer than the window, or a code is outside its
     *                                  codebook
     * @throws IllegalStateException    when the session is closed
     */
    public float[] decode(int[][] frames) {
        ensureOpen();
        Objects.requireNonNull(frames, "frames");
        int n = frames.length;
        if (n < 1 || n > window) {
            throw new IllegalArgumentException("a code2wav batch holds 1 to " + window + " frames, got " + n);
        }
        int codes = geo.codesPerFrame();
        for (int f = 0; f < n; f++) {
            if (frames[f] == null || frames[f].length != codes) {
                throw new IllegalArgumentException("frame " + f + " holds "
                        + (frames[f] == null ? "no" : frames[f].length) + " codes, the quantizer takes " + codes);
            }
            for (int g = 0; g < codes; g++) {
                int limit = g == 0 ? geo.firstVocab() : geo.restVocab();
                int code = frames[f][g];
                if (code < 0 || code >= limit) {
                    throw new IllegalArgumentException("code " + code + " of frame " + f + ", codebook " + g
                            + " is outside [0, " + limit + ")");
                }
            }
        }
        int[][] codesByGroup = groupMajor(frames, n, codes);

        float[] hidden = quantize(codesByGroup);
        float[][] x = reshape(hidden, window, geo.decoderWidth());
        x = causalConv(x, geo.preConvW(), geo.preConvB(), geo.preConvK(), geo.preConvIn(), geo.preConvOut(), 1,
                preConvState);
        x = transformer(x);
        for (int i = 0; i < geo.upsample().length; i++) {
            x = upsample(x, geo.upsample()[i], i);
        }
        x = causalConv(x, geo.dacEntryW(), geo.dacEntryB(), geo.dacEntryK(), geo.dacEntryIn(), geo.dacEntryOut(), 1,
                dacEntryState);
        int delta = 0;
        for (int ib = 0; ib < geo.dac().length; ib++) {
            Qwen3TtsCode2Wav.DacBlock block = geo.dac()[ib];
            x = snake(x, block.snakeAlpha, block.snakeBeta);
            x = causalTranspose(x, block.convW, block.convB, block.stride * 2, block.stride, block.inChannels,
                    block.outChannels, dacTailState[ib]);
            for (int ir = 0; ir < block.res.size(); ir++) {
                Qwen3TtsCode2Wav.DacRes res = block.res.get(ir);
                x = residualUnit(x, res, Qwen3TtsCode2Wav.DAC_DILATIONS[ir], resStates[delta++]);
            }
        }
        x = snake(x, geo.postSnakeAlpha(), geo.postSnakeBeta());
        x = causalConv(x, geo.postConvW(), geo.postConvB(), geo.postConvK(), geo.postConvIn(), geo.postConvOut(), 1,
                postConvState);
        float[] out = new float[n * geo.hop()];
        int keep = out.length;
        for (int i = 0; i < keep && i < x.length; i++) {
            out[i] = Math.max(-1f, Math.min(1f, x[i][0]));
        }
        return out;
    }

    /** {@link #decode(int[][])} as an {@link AudioClip}. */
    public AudioClip decodeClip(int[][] frames) {
        return new AudioClip(decode(frames), geo.sampleRate());
    }

    /** The decoded waveform's sample rate. */
    @Override
    public int sampleRate() {
        return geo.sampleRate();
    }

    /** The persisted RoPE position of the next window's first frame. */
    public float position() {
        return tfmPos;
    }

    /** Frees nothing (the weights belong to {@link Qwen3TtsCode2Wav}); marks the session closed. */
    @Override
    public void close() {
        closed = true;
    }

    // ------------------------------------------------------------------ quantizer

    /** RVQ decode: {@code [codesPerFrame][window]} group-major codes to {@code [window][decoderWidth]}. */
    private float[] quantize(int[][] groupCodes) {
        int dimFirst = geo.firstDim();
        int dimRest = geo.restDim();
        float[] sem = new float[window * dimFirst];
        float[] acc = new float[window * dimRest];
        float[] row = new float[Math.max(dimFirst, dimRest)];
        for (int f = 0; f < window; f++) {
            weights.embedRow(geo.quantFirstCb(), groupCodes[0][f], sem, f * dimFirst);
        }
        for (int g = 1; g < geo.codesPerFrame(); g++) {
            for (int f = 0; f < window; f++) {
                weights.embedRow(geo.quantRestCb(), (long) (g - 1) * geo.restVocab() + groupCodes[g][f], row, 0);
                for (int d = 0; d < dimRest; d++) {
                    acc[f * dimRest + d] += row[d];
                }
            }
        }
        float[] firstOut = new float[window * geo.decoderWidth()];
        float[] restOut = new float[window * geo.decoderWidth()];
        matVec.matmul(geo.quantFirstOut(), sem, 0, dimFirst, firstOut, 0, geo.decoderWidth(), window);
        matVec.matmul(geo.quantRestOut(), acc, 0, dimRest, restOut, 0, geo.decoderWidth(), window);
        CpuOps.add(firstOut, 0, restOut, 0, firstOut, 0, firstOut.length);
        return firstOut;
    }

    /** Re-arranges frame-major {@code [n][codes]} codes into {@code [codes][window]} with zero padding. */
    private int[][] groupMajor(int[][] frames, int n, int codes) {
        int[][] out = new int[codes][window];
        for (int g = 0; g < codes; g++) {
            for (int f = 0; f < n; f++) {
                out[g][f] = frames[f][g];
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ pre-transformer

    private float[][] transformer(float[][] input) {
        int width = geo.nEmbd();
        float[] flat = new float[window * width];
        matVec.matmul(geo.tfmIn(), flatten(input, geo.preConvOut()), 0, geo.preConvOut(), flat, 0, width, window);
        float[][] x = reshape(flat, window, width);
        addBias(x, geo.tfmInB());
        int base = (int) tfmPos;
        int[] qPos = new int[window];
        for (int i = 0; i < window; i++) {
            qPos[i] = base + i;
        }
        for (int il = 0; il < geo.layers().length; il++) {
            x = layer(x, geo.layers()[il], il, base, qPos);
        }
        rmsNorm(x, geo.tfmOutNorm(), x);
        float[] projected = new float[window * geo.preConvOut()];
        matVec.matmul(geo.tfmOut(), flatten(x, width), 0, width, projected, 0, geo.preConvOut(), window);
        float[][] out = reshape(projected, window, geo.preConvOut());
        addBias(out, geo.tfmOutB());
        // the reference advances the counter by a whole window even when the last batch was shorter
        tfmPos += window;
        return out;
    }

    private float[][] layer(float[][] x, Qwen3TtsCode2Wav.TfmLayer layer, int il, int base, int[] qPos) {
        int width = geo.nEmbd();
        int nHead = geo.nHead();
        int nHeadKv = geo.nHeadKv();
        int dHead = geo.dHead();
        int total = prefix + window;
        float[][] out = new float[window][width];
        float[][] normed = new float[window][width];
        rmsNorm(x, layer.ln1, normed);
        float[] q = matVec(layer.q, flatten(normed, width), width, nHead * dHead);
        float[] k = matVec(layer.k, flatten(normed, width), width, nHeadKv * dHead);
        float[] v = matVec(layer.v, flatten(normed, width), width, nHeadKv * dHead);
        // RoPE over the real positions of this window
        for (int i = 0; i < window; i++) {
            rope(q, i * nHead * dHead, nHead, dHead, qPos[i]);
            rope(k, i * nHeadKv * dHead, nHeadKv, dHead, qPos[i]);
        }
        float[][] keys = new float[total][kvCh];
        float[][] values = new float[total][kvCh];
        for (int j = 0; j < prefix; j++) {
            System.arraycopy(kPrefix[il], j * kvCh, keys[j], 0, kvCh);
            System.arraycopy(vPrefix[il], j * kvCh, values[j], 0, kvCh);
        }
        for (int i = 0; i < window; i++) {
            System.arraycopy(k, i * kvCh, keys[prefix + i], 0, kvCh);
            System.arraycopy(v, i * kvCh, values[prefix + i], 0, kvCh);
        }
        float[] attn = attention(q, keys, values, base);
        float[] projected = matVec(layer.out, attn, nHead * dHead, width);
        for (int i = 0; i < window; i++) {
            for (int c = 0; c < width; c++) {
                out[i][c] = x[i][c] + projected[i * width + c] * layer.ls1[c];
            }
        }
        float[][] normed2 = new float[window][width];
        rmsNorm(out, layer.ln2, normed2);
        int ff = geo.nFf();
        float[] gate = matVec(layer.gate, flatten(normed2, width), width, ff);
        float[] up = matVec(layer.up, flatten(normed2, width), width, ff);
        float[] down = new float[window * width];
        float[] swiglu = new float[window * ff];
        for (int i = 0; i < window * ff; i++) {
            swiglu[i] = silu(gate[i]) * up[i];
        }
        matVec.matmul(layer.down, swiglu, 0, ff, down, 0, width, window);
        for (int i = 0; i < window; i++) {
            for (int c = 0; c < width; c++) {
                out[i][c] += down[i * width + c] * layer.ls2[c];
            }
        }
        // next window's prefix: the last (window - 1) frames of this batch's keys and values
        for (int j = 0; j < prefix; j++) {
            System.arraycopy(keys[window + j], 0, kPrefix[il], j * kvCh, kvCh);
            System.arraycopy(values[window + j], 0, vPrefix[il], j * kvCh, kvCh);
        }
        return out;
    }

    /**
     * Banded causal attention of one window over the persisted prefix ({@code prefix} keys) and the window's keys:
     * query {@code i} attends to key {@code j} when {@code 0 <= prefix + i - j < window} and the key's real
     * position {@code j - prefix + base} is not negative (the cold-start warm mask).
     */
    private float[] attention(float[] q, float[][] keys, float[][] values, int base) {
        int nHead = geo.nHead();
        int nHeadKv = geo.nHeadKv();
        int dHead = geo.dHead();
        int headsPerKv = nHead / nHeadKv;
        int total = prefix + window;
        float scale = (float) (1.0 / Math.sqrt(dHead));
        float[] out = new float[window * nHead * dHead];
        float[] scores = new float[total];
        for (int i = 0; i < window; i++) {
            for (int h = 0; h < nHead; h++) {
                int kvHead = h / headsPerKv;
                int qOff = i * nHead * dHead + h * dHead;
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < total; j++) {
                    int keyPos = j - prefix + base;
                    int diff = prefix + i - j;
                    if (keyPos < 0 || diff < 0 || diff >= window) {
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

    // ------------------------------------------------------------------ upsample and DAC

    private float[][] upsample(float[][] input, Qwen3TtsCode2Wav.UpBlock block, int index) {
        float[][] x = causalTranspose(input, block.convW, block.convB, 2, 2, block.inChannels, block.outChannels,
                null);
        return convNext(x, block, upStates[index]);
    }

    /** {@code dwconv -> LayerNorm -> pwconv1 -> GELU -> pwconv2 -> layer scale -> residual}. */
    private float[][] convNext(float[][] x, Qwen3TtsCode2Wav.UpBlock block, float[][] state) {
        int t = x.length;
        int c = x.length == 0 ? 0 : x[0].length;
        float[][] dw = depthwiseCausal(x, block.dwconvW, block.dwconvB, block.dwK, state);
        layerNorm(dw, block.normW, block.normB, CONVNEXT_EPS);
        float[] g = matVec(block.pw1, flatten(dw, c), c, (int) block.pw1.rows());
        for (int i = 0; i < g.length; i++) {
            g[i] = geluTanh(g[i] + block.pw1B[i % block.pw1B.length]);
        }
        int ff = block.pw1B.length;
        float[] back = matVec(block.pw2, g, ff, c);
        float[][] out = new float[t][c];
        for (int i = 0; i < t; i++) {
            for (int ch = 0; ch < c; ch++) {
                out[i][ch] = x[i][ch] + (back[i * c + ch] + block.pw2B[ch]) * block.gamma[ch];
            }
        }
        return out;
    }

    private float[][] residualUnit(float[][] x, Qwen3TtsCode2Wav.DacRes res, int dilation, float[][] state) {
        float[][] h = snake(x, res.act1Alpha, res.act1Beta);
        h = causalConv(h, res.conv1W, res.conv1B, res.conv1K, res.channels, res.channels, dilation, state);
        h = snake(h, res.act2Alpha, res.act2Beta);
        h = causalConv(h, res.conv2W, res.conv2B, 1, res.channels, res.channels, 1, null);
        float[][] out = new float[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int c = 0; c < x[i].length; c++) {
                out[i][c] = x[i][c] + h[i][c];
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ convolution primitives

    /**
     * Causal convolution {@code y[t][oc] = b[oc] + sum_{k,ic} w[k][ic][oc] * xfull[t+k][ic]} whose left context is
     * the persisted {@code state} ({@code (K-1)*dilation} rows); the state becomes the input's last rows.
     */
    private float[][] causalConv(float[][] x, float[] w, float[] b, int k, int in, int out, int dilation,
                                 float[][] state) {
        int t = x.length;
        int pad = (k - 1) * dilation;
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
                        acc += w[j + k * wBase] * full[i + j * dilation][ic];
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

    /** Depthwise causal convolution, {@code w[K][C]}, left context persisted in {@code state}. */
    private float[][] depthwiseCausal(float[][] x, float[] w, float[] b, int k, float[][] state) {
        int t = x.length;
        int c = x.length == 0 ? 0 : x[0].length;
        int pad = k - 1;
        float[][] full = new float[pad + t][c];
        for (int i = 0; i < pad; i++) {
            System.arraycopy(state[i], 0, full[i], 0, c);
        }
        for (int i = 0; i < t; i++) {
            System.arraycopy(x[i], 0, full[pad + i], 0, c);
        }
        float[][] y = new float[t][c];
        for (int i = 0; i < t; i++) {
            for (int ch = 0; ch < c; ch++) {
                float acc = b[ch];
                for (int j = 0; j < k; j++) {
                    acc += w[j + k * ch] * full[i + j][ch];
                }
                y[i][ch] = acc;
            }
        }
        for (int i = 0; i < pad; i++) {
            System.arraycopy(full[t + i], 0, state[i], 0, c);
        }
        return y;
    }

    /**
     * Causal ConvTranspose1d: the transposed convolution's overlap-add tail is added from {@code state} and the new
     * tail kept; the emitted samples ({@code T * stride}) exclude both tails.
     */
    private float[][] causalTranspose(float[][] x, float[] w, float[] b, int k, int stride, int in, int out,
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
        int emit = t * stride;
        float[][] y = new float[emit][out];
        for (int i = 0; i < trim; i++) {
            for (int oc = 0; oc < out; oc++) {
                y[i][oc] = full[i][oc] + state[i][oc];
            }
        }
        for (int i = trim; i < emit; i++) {
            System.arraycopy(full[i], 0, y[i], 0, out);
        }
        for (int i = 0; i < emit; i++) {
            for (int oc = 0; oc < out; oc++) {
                y[i][oc] += b[oc];
            }
        }
        for (int i = 0; i < trim; i++) {
            System.arraycopy(full[emit + i], 0, state[i], 0, out);
        }
        return y;
    }

    /** SnakeBeta: {@code x + sin(alpha * x)^2 * beta} (alpha/beta folded at conversion time). */
    private static float[][] snake(float[][] x, float[] alpha, float[] beta) {
        float[][] y = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            float[] row = new float[x[i].length];
            for (int c = 0; c < row.length; c++) {
                double s = Math.sin(alpha[c] * x[i][c]);
                row[c] = (float) (x[i][c] + s * s * beta[c]);
            }
            y[i] = row;
        }
        return y;
    }

    // ------------------------------------------------------------------ normalization

    /** RMS norm over the channel axis of every frame, in place. */
    private void rmsNorm(float[][] x, float[] weight, float[][] out) {
        int width = weight.length;
        for (int i = 0; i < x.length; i++) {
            double sum = 0.0;
            for (int c = 0; c < width; c++) {
                sum += (double) x[i][c] * x[i][c];
            }
            float scale = (float) (1.0 / Math.sqrt(sum / width + geo.eps()));
            for (int c = 0; c < width; c++) {
                out[i][c] = x[i][c] * scale * weight[c];
            }
        }
    }

    /** LayerNorm over the channel axis of every frame, in place. */
    private static void layerNorm(float[][] x, float[] weight, float[] bias, float eps) {
        int width = weight.length;
        for (int i = 0; i < x.length; i++) {
            double mean = 0.0;
            for (int c = 0; c < width; c++) {
                mean += x[i][c];
            }
            mean /= width;
            double var = 0.0;
            for (int c = 0; c < width; c++) {
                double d = x[i][c] - mean;
                var += d * d;
            }
            var /= width;
            float scale = (float) (1.0 / Math.sqrt(var + eps));
            for (int c = 0; c < width; c++) {
                x[i][c] = (float) ((x[i][c] - mean) * scale) * weight[c] + bias[c];
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    private void rope(float[] x, int off, int nHead, int dHead, int pos) {
        int half = dHead / 2;
        for (int h = 0; h < nHead; h++) {
            int base = off + h * dHead;
            for (int i = 0; i < half; i++) {
                double angle = pos * Math.pow(geo.ropeTheta(), -2.0 * i / dHead);
                float cos = (float) Math.cos(angle);
                float sin = (float) Math.sin(angle);
                float x0 = x[base + i];
                float x1 = x[base + i + half];
                x[base + i] = x0 * cos - x1 * sin;
                x[base + i + half] = x0 * sin + x1 * cos;
            }
        }
    }

    private static float silu(float x) {
        return (float) (x / (1.0 + Math.exp(-x)));
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

    private static void addBias(float[][] x, float[] bias) {
        for (float[] row : x) {
            for (int c = 0; c < row.length; c++) {
                row[c] += bias[c];
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the code2wav session is closed");
        }
    }

    @Override
    public String toString() {
        return "Qwen3TtsCode2WavSession{window=" + window + ", position=" + tfmPos + ", layers="
                + geo.nLayer() + '}';
    }
}
