package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.StreamingIstft;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.speech.Lfm3aCodec.Layer;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * One decode stream of the lfm3a liquid-codec decoder, the state slots of llama.cpp's
 * {@code clip_graph_lfm3a_gen} (tools/mtmd/models/lfm3a-gen.cpp:172-251): the position counter and, per decoder
 * layer, the short convolution's taps or the sliding attention's key/value window, ending in the shared
 * {@link StreamingIstft}.
 *
 * <p>{@link #decodePolar} is the graph seam: a batch of codec frames (each a stack of {@code nCodebooks()} codes)
 * becomes one polar spectrum per decoder position — the log magnitudes, then the phases — with the position slot,
 * the taps and the key/value windows advanced exactly as the reference's tensors advance when the host feeds the
 * next frame. The layer math is the LFM2 stack of {@code build_lfm_layer}: RMS norms, the block, the sandwich
 * norm when present, and the residual. {@link #decodeClip} runs the same graph and hands the spectra to the ISTFT
 * as interleaved complex bins (tools/mtmd/mtmd.cpp:2359-2418). {@link #flushClip} drains the ISTFT tail and drops
 * its last {@code (n_fft - hop) / 2} samples, as the reference's flush branch does (mtmd.cpp:2370-2373).
 *
 * <p>Not thread-safe; one stream per concurrent synthesis.
 */
public final class Lfm3aCodecSession implements Lfm3aCodec.Decoder {

    private final Lfm3aCodec codec;
    private final CpuMatVec matVec;
    private final CpuWeights weights;
    private final StreamingIstft istft;
    private final int prefix;
    private final int nTrim;
    private final CpuTensorRef quantFirstCb;
    private final CpuTensorRef quantFirstOut;
    private final float[] quantFirstOutB;
    private final CpuTensorRef tfmIn;
    private final float[] tfmInB;
    private final CpuTensorRef tfmOut;
    private final float[] tfmOutB;
    private final float[] tfmOutNorm;
    private final float[][] dwCache;
    private final float[][] convState;
    private final float[][] kState;
    private final float[][] vState;
    private int pos;
    private boolean pendingTail;

    Lfm3aCodecSession(Lfm3aCodec codec) {
        this.codec = codec;
        this.weights = codec.weights();
        this.matVec = weights.matVec();
        this.istft = new StreamingIstft(codec.nFft(), codec.hop());
        this.prefix = codec.window() - 1;
        this.nTrim = (codec.nFft() - codec.hop()) / 2;
        this.quantFirstCb = weights.get(Lfm3aCodec.QUANT_FIRST_CB + ".weight");
        this.quantFirstOut = weights.get(Lfm3aCodec.QUANT_FIRST_OUT + ".weight");
        this.quantFirstOutB = weights.floats(Lfm3aCodec.QUANT_FIRST_OUT + ".bias");
        this.tfmIn = weights.get(Lfm3aCodec.TFM_IN_PROJ + ".weight");
        this.tfmInB = weights.floats(Lfm3aCodec.TFM_IN_PROJ + ".bias");
        this.tfmOut = weights.get(Lfm3aCodec.TFM_OUT_PROJ + ".weight");
        this.tfmOutB = weights.floats(Lfm3aCodec.TFM_OUT_PROJ + ".bias");
        this.tfmOutNorm = weights.floats(Lfm3aCodec.TFM_OUTPUT_NORM + ".weight");
        Layer[] layers = codec.layers();
        this.dwCache = new float[layers.length][];
        this.convState = new float[layers.length][];
        this.kState = new float[layers.length][];
        this.vState = new float[layers.length][];
        for (int il = 0; il < layers.length; il++) {
            if (layers[il].isConv()) {
                convState[il] = new float[(layers[il].taps - 1) * codec.nEmbdDec()];
                dwCache[il] = weights.floats("a.gen.wav.tfm.blk." + il + ".conv_dw.weight");
            } else {
                int kvWidth = layers[il].nHeadKv * layers[il].dHead;
                kState[il] = new float[prefix * kvWidth];
                vState[il] = new float[prefix * kvWidth];
            }
        }
    }

    // ------------------------------------------------------------------ accessors

    /** The decoder positions consumed so far (the reference's {@code inp_pos} state slot). */
    public int pos() {
        return pos;
    }

    @Override
    public int position() {
        return pos;
    }

    /** Spectrum bins per STFT frame. */
    public int nBins() {
        return codec.nBins();
    }

    /** The waveform's sample rate. */
    public int sampleRate() {
        return codec.sampleRate();
    }

    /** True once at least one codec frame was decoded since the last flush or reset. */
    public boolean pendingTail() {
        return pendingTail;
    }

    /** The persisted state slot names, in the reference's {@code list_lfm3a_gen_state_slots} order. */
    public List<String> stateSlots() {
        return codec.stateSlots();
    }

    /** Back to a cold stream: position 0, zeroed taps and windows, a fresh ISTFT. */
    @Override
    public void reset() {
        pos = 0;
        pendingTail = false;
        zeroAll(convState);
        zeroAll(kState);
        zeroAll(vState);
        istft.reset();
    }

    /** Releases the stream's buffers; the session holds no external resources. */
    @Override
    public void close() {
        reset();
    }

    private static void zeroAll(float[][] states) {
        for (float[] state : states) {
            if (state != null) {
                Arrays.fill(state, 0f);
            }
        }
    }

    // ------------------------------------------------------------------ the graph

    /**
     * One decoder pass: {@code codes} holds whole codec frames of {@code nCodebooks()} codes each, frame-major
     * ({@code codes[f * nCodebooks() + codebook]}), and the result holds one polar spectrum per decoder position —
     * the {@code nBins()} log magnitudes, then the {@code nBins()} phases — in position order.
     *
     * @throws IllegalArgumentException when the codes do not form whole frames or leave the FSQ vocabulary
     */
    @Override
    public float[] decodePolar(int[] codes) {
        Objects.requireNonNull(codes, "codes");
        int nCb = codec.nCodebooks();
        if (codes.length == 0 || codes.length % nCb != 0) {
            throw new IllegalArgumentException("the codes must form whole frames of " + nCb + ", got "
                    + codes.length);
        }
        int vocab = codec.vocab();
        for (int c : codes) {
            if (c < 0 || c >= vocab) {
                throw new IllegalArgumentException("code " + c + " is outside the FSQ vocabulary [0, " + vocab
                        + ")");
            }
        }
        int nFrames = codes.length / nCb;
        int nEmbdDec = codec.nEmbdDec();
        int nLvl = codec.nLvl();
        int frameStack = codec.frameStack();
        int nPos = nFrames * frameStack;
        int base = pos;

        // FSQ level vectors, the first-codebook projection and the in-projection, then nearest upsampling
        float[] inpL = new float[nPos * nEmbdDec];
        float[] q = new float[nLvl * nCb];
        float[] levels = new float[nLvl];
        float[] projected = new float[nEmbdDec];
        for (int f = 0; f < nFrames; f++) {
            for (int cb = 0; cb < nCb; cb++) {
                weights.embedRow(quantFirstCb, codes[f * nCb + cb], levels, 0);
                System.arraycopy(levels, 0, q, cb * nLvl, nLvl);
            }
            matVec.matmul(quantFirstOut, q, 0, projected, 0, 1);
            CpuOps.addRowBias(projected, 0, 1, nEmbdDec, quantFirstOutB, 0);
            matVec.matmul(tfmIn, projected, 0, projected, 0, 1);
            CpuOps.addRowBias(projected, 0, 1, nEmbdDec, tfmInB, 0);
            for (int s = 0; s < frameStack; s++) {
                System.arraycopy(projected, 0, inpL, (f * frameStack + s) * nEmbdDec, nEmbdDec);
            }
        }

        // the decoder layers under the causal window mask over [cached window | this batch]
        float[] mask = windowMask(nPos, base);
        float[] normed = new float[nPos * nEmbdDec];
        float[] result = new float[nPos * nEmbdDec];
        Layer[] layers = codec.layers();
        for (int il = 0; il < layers.length; il++) {
            Layer l = layers[il];
            rmsNormRows(inpL, l.ln1, normed);
            if (l.isConv()) {
                shortConv(l, il, normed, result, nPos);
            } else {
                windowAttention(l, il, normed, result, mask, nPos, base);
            }
            if (l.attnPostNorm != null) {
                rmsNormRows(result, l.attnPostNorm, result);
            }
            addResidual(inpL, result);
            rmsNormRows(inpL, l.ln2, normed);
            ffn(l, normed, result, nPos);
            if (l.ffPostNorm != null) {
                rmsNormRows(result, l.ffPostNorm, result);
            }
            addResidual(inpL, result);
        }
        return decodeOutput(inpL, nPos);
    }

    /** The output norm and projection of build_gen_wav's tail; advances the position slot. */
    private float[] decodeOutput(float[] inpL, int nPos) {
        int width = codec.nFft() + 2;
        int nEmbdDec = codec.nEmbdDec();
        float[] feats = new float[nPos * width];
        float[] row = new float[nEmbdDec];
        for (int p = 0; p < nPos; p++) {
            rmsNormRow(inpL, p * nEmbdDec, tfmOutNorm, row, 0);
            matVec.matmul(tfmOut, row, 0, feats, p * width, 1);
            CpuOps.addRowBias(feats, p * width, 1, width, tfmOutB, 0);
        }
        pos += nPos;
        pendingTail = true;
        return feats;
    }

    /**
     * Decodes codec frames to PCM: every polar spectrum becomes interleaved complex bins
     * ({@code exp(mag) * cos/sin(phase)}, tools/mtmd/mtmd.cpp:2368-2371) for the ISTFT, whose emitted hop-sized
     * chunks are concatenated. The first stream chunks carry the ISTFT's leading pad and may be empty.
     */
    @Override
    public AudioClip decodeClip(int[] codes) {
        float[] feats = decodePolar(codes);
        int nBins = codec.nBins();
        int width = 2 * nBins;
        int nPos = feats.length / width;
        float[] spec = new float[width];
        PcmBuffer pcm = new PcmBuffer();
        for (int p = 0; p < nPos; p++) {
            int off = p * width;
            for (int b = 0; b < nBins; b++) {
                float mag = (float) Math.exp(feats[off + b]);
                float phase = feats[off + nBins + b];
                spec[2 * b] = mag * (float) Math.cos(phase);
                spec[2 * b + 1] = mag * (float) Math.sin(phase);
            }
            pcm.append(istft.processFrame(spec));
        }
        return new AudioClip(pcm.toArray(), codec.sampleRate());
    }

    /**
     * Drains the ISTFT tail once frames were decoded: the flushed tail keeps its head and loses its last
     * {@code (n_fft - hop) / 2} samples (mtmd.cpp:2370-2373); without decoded frames this is an empty clip.
     */
    @Override
    public AudioClip flushClip() {
        if (!pendingTail) {
            return new AudioClip(new float[0], codec.sampleRate());
        }
        float[] tail = istft.flush();
        int keep = tail.length - Math.min(tail.length, nTrim);
        float[] kept = new float[keep];
        System.arraycopy(tail, 0, kept, 0, keep);
        istft.reset();
        pendingTail = false;
        return new AudioClip(kept, codec.sampleRate());
    }

    // ------------------------------------------------------------------ layer operators

    /** The causal short convolution of build_lfm_conv: pw1, b*x, the taps slot, c*conv, pw2. */
    private void shortConv(Layer l, int il, float[] normed, float[] result, int nPos) {
        int n = codec.nEmbdDec();
        int taps = l.taps;
        int dConv = taps - 1;
        float[] dw = dwCache[il];
        float[] v = new float[3 * n];
        float[] bx = new float[nPos * n];
        float[] cArr = new float[nPos * n];
        for (int p = 0; p < nPos; p++) {
            matVec.matmul(l.convPw1, normed, p * n, v, 0, 1);
            System.arraycopy(v, n, cArr, p * n, n);
            for (int i = 0; i < n; i++) {
                bx[p * n + i] = v[i] * v[2 * n + i];
            }
        }
        float[] all = new float[(dConv + nPos) * n];
        System.arraycopy(convState[il], 0, all, 0, dConv * n);
        System.arraycopy(bx, 0, all, dConv * n, nPos * n);
        for (int t = 0; t < nPos; t++) {
            for (int i = 0; i < n; i++) {
                float acc = 0f;
                for (int k = 0; k < taps; k++) {
                    acc += all[(t + k) * n + i] * dw[k + i * taps];
                }
                result[t * n + i] = cArr[t * n + i] * acc;
            }
        }
        matVec.matmul(l.convPw2, result, 0, n, result, 0, n, nPos);
        // new taps: the last dConv rows of [state | bx]
        for (int i = 0; i < dConv; i++) {
            System.arraycopy(all, (nPos + i) * n, convState[il], i * n, n);
        }
    }

    /** The sliding-window gated attention of build_lfm_attn, with the key/value slots advanced. */
    private void windowAttention(Layer l, int il, float[] normed, float[] result, float[] mask, int nPos,
                                 int base) {
        int n = codec.nEmbdDec();
        int dHead = l.dHead;
        int nHead = l.nHead;
        int nHeadKv = l.nHeadKv;
        int headsPerKv = nHead / nHeadKv;
        int qWidth = nHead * dHead;
        int kvWidth = nHeadKv * dHead;
        int nKv = prefix + nPos;
        float[] q = new float[nPos * qWidth];
        float[] k = new float[nPos * kvWidth];
        float[] v = new float[nPos * kvWidth];
        float[] scratch = new float[Math.max(qWidth, kvWidth)];
        for (int p = 0; p < nPos; p++) {
            int off = p * n;
            matVec.matmul(l.q, normed, off, q, p * qWidth, 1);
            matVec.matmul(l.k, normed, off, k, p * kvWidth, 1);
            matVec.matmul(l.v, normed, off, v, p * kvWidth, 1);
            for (int h = 0; h < nHead; h++) {
                rmsNormRange(q, p * qWidth + h * dHead, l.qNorm, scratch, dHead);
                ropeNeox(scratch, base + p, dHead);
                System.arraycopy(scratch, 0, q, p * qWidth + h * dHead, dHead);
            }
            for (int h = 0; h < nHeadKv; h++) {
                rmsNormRange(k, p * kvWidth + h * dHead, l.kNorm, scratch, dHead);
                ropeNeox(scratch, base + p, dHead);
                System.arraycopy(scratch, 0, k, p * kvWidth + h * dHead, dHead);
            }
        }
        float[] kAll = new float[nKv * kvWidth];
        float[] vAll = new float[nKv * kvWidth];
        System.arraycopy(kState[il], 0, kAll, 0, prefix * kvWidth);
        System.arraycopy(k, 0, kAll, prefix * kvWidth, nPos * kvWidth);
        System.arraycopy(vState[il], 0, vAll, 0, prefix * kvWidth);
        System.arraycopy(v, 0, vAll, prefix * kvWidth, nPos * kvWidth);

        float scale = (float) (1.0 / Math.sqrt(dHead));
        float[] scores = new float[nKv];
        float[] attn = new float[qWidth];
        float[] gate = new float[nHead];
        for (int p = 0; p < nPos; p++) {
            for (int h = 0; h < nHead; h++) {
                int kvHead = h / headsPerKv;
                float max = Float.NEGATIVE_INFINITY;
                int qOff = p * qWidth + h * dHead;
                int kvOff = kvHead * dHead;
                for (int j = 0; j < nKv; j++) {
                    float dot = 0f;
                    int kRow = j * kvWidth + kvOff;
                    for (int d = 0; d < dHead; d++) {
                        dot += q[qOff + d] * kAll[kRow + d];
                    }
                    scores[j] = dot * scale + mask[p * nKv + j];
                    if (scores[j] > max) {
                        max = scores[j];
                    }
                }
                double sum = 0.0;
                for (int j = 0; j < nKv; j++) {
                    double e = Math.exp(scores[j] - max);
                    scores[j] = (float) e;
                    sum += e;
                }
                float inv = (float) (1.0 / sum);
                for (int d = 0; d < dHead; d++) {
                    float acc = 0f;
                    for (int j = 0; j < nKv; j++) {
                        acc += scores[j] * vAll[j * kvWidth + kvOff + d];
                    }
                    attn[h * dHead + d] = acc * inv;
                }
            }
            // the per-head sigmoid gate reads the normed layer input, as the reference's build_lfm_attn does
            matVec.matmul(l.attnGate, normed, p * n, gate, 0, 1);
            for (int h = 0; h < nHead; h++) {
                float g = sigmoid(gate[h]);
                for (int d = 0; d < dHead; d++) {
                    attn[h * dHead + d] *= g;
                }
            }
            matVec.matmul(l.out, attn, 0, result, p * n, 1);
        }
        // new windows: the last prefix rows of [cache | chunk]
        rollWindow(kAll, kState[il], nPos, kvWidth);
        rollWindow(vAll, vState[il], nPos, kvWidth);
    }

    /** Keeps the last {@code prefix} rows of {@code [cache | chunk]} in {@code cache}. */
    private void rollWindow(float[] all, float[] cache, int nPos, int width) {
        int rows = cache.length / width;
        for (int i = 0; i < rows; i++) {
            System.arraycopy(all, (nPos + i) * width, cache, i * width, width);
        }
    }

    /** The SwiGLU FFN over every position. */
    private void ffn(Layer l, float[] x, float[] out, int nPos) {
        int n = codec.nEmbdDec();
        int ff = (int) l.gate.rows();
        float[] gate = new float[ff];
        float[] up = new float[ff];
        for (int p = 0; p < nPos; p++) {
            matVec.matmul(l.gate, x, p * n, gate, 0, 1);
            matVec.matmul(l.up, x, p * n, up, 0, 1);
            for (int i = 0; i < ff; i++) {
                gate[i] = silu(gate[i]) * up[i];
            }
            matVec.matmul(l.down, gate, 0, out, p * n, 1);
        }
    }

    // ------------------------------------------------------------------ math

    /**
     * The SWA mask of clip.cpp's set_lfm3a_window_inputs: query {@code q} (absolute position {@code base + q})
     * sees key rows {@code j} in {@code [max(0, q), q + prefix]}, and only while the key's absolute position
     * {@code base - prefix + j} has started; everything else is negative infinity.
     */
    private float[] windowMask(int nPos, int base) {
        int nKv = prefix + nPos;
        float[] mask = new float[nKv * nPos];
        Arrays.fill(mask, Float.NEGATIVE_INFINITY);
        for (int q = 0; q < nPos; q++) {
            int jHi = q + prefix;
            for (int j = q; j <= jHi; j++) {
                if (base - prefix + j >= 0) {
                    mask[q * nKv + j] = 0f;
                }
            }
        }
        return mask;
    }

    /** RMS norm over every row of {@code x} ({@code nEmbdDec} wide), scaled by one shared weight. */
    private void rmsNormRows(float[] x, float[] weight, float[] out) {
        int n = codec.nEmbdDec();
        for (int p = 0; p * n < x.length; p++) {
            rmsNormRow(x, p * n, weight, out, p * n);
        }
    }

    /** RMS norm of one row: {@code out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]}. */
    private void rmsNormRow(float[] x, int xOff, float[] weight, float[] out, int outOff) {
        int n = weight.length;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += (double) x[xOff + i] * x[xOff + i];
        }
        float inv = (float) (1.0 / Math.sqrt(sum / n + Lfm3aCodec.EPS));
        for (int i = 0; i < n; i++) {
            out[outOff + i] = x[xOff + i] * inv * weight[i];
        }
    }

    /** RMS norm of a head-sized range of one row. */
    private static void rmsNormRange(float[] x, int off, float[] weight, float[] out, int width) {
        double sum = 0.0;
        for (int i = 0; i < width; i++) {
            sum += (double) x[off + i] * x[off + i];
        }
        float inv = (float) (1.0 / Math.sqrt(sum / width + Lfm3aCodec.EPS));
        for (int i = 0; i < width; i++) {
            out[i] = x[off + i] * inv * weight[i];
        }
    }

    /** NEOX RoPE of one head: pairs {@code (i, i + width/2)} rotated by {@code pos * theta^(-2i/width)}. */
    private static void ropeNeox(float[] x, int pos, int width) {
        int half = width / 2;
        for (int i = 0; i < half; i++) {
            double angle = pos * Math.pow(Lfm3aCodec.ROPE_THETA, -2.0 * i / width);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            float x0 = x[i];
            float x1 = x[i + half];
            x[i] = x0 * cos - x1 * sin;
            x[i + half] = x0 * sin + x1 * cos;
        }
    }

    private static void addResidual(float[] residual, float[] sum) {
        CpuOps.add(residual, 0, sum, 0, residual, 0, residual.length);
    }

    private static float silu(float x) {
        return (float) (x / (1.0 + Math.exp(-x)));
    }

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    /** Minimal growable float buffer for the streamed PCM. */
    private static final class PcmBuffer {
        private float[] data = new float[64];
        private int size;

        void append(float[] chunk) {
            if (chunk.length == 0) {
                return;
            }
            if (size + chunk.length > data.length) {
                data = Arrays.copyOf(data, Math.max(data.length * 2, size + chunk.length));
            }
            System.arraycopy(chunk, 0, data, size, chunk.length);
            size += chunk.length;
        }

        float[] toArray() {
            return Arrays.copyOf(data, size);
        }
    }
}
