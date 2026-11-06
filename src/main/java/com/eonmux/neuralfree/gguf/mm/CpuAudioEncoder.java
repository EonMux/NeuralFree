package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.AudioResampler;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.audio.MelFrontend;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.CpuVitTower.floatsOrNull;
import static com.eonmux.neuralfree.gguf.mm.CpuVitTower.refOrNull;

/**
 * CPU executor of one audio encoder and its projector over {@link ClipAudioModel}'s tensors (the counterpart of
 * llama.cpp tools/mtmd/models/whisper-enc.cpp and qwen3a.cpp, and of the Hugging Face Gemma 3n audio tower).
 * One call encodes one log-mel chunk to text-model embedding rows, one row per audio token, ready for
 * {@code ForwardBatch.Builder#addEmbedding}:
 * <ul>
 *   <li>whisper: conv1d (stride 1) + GELU-erf and conv1d (stride 2) + GELU-erf as im2col matmuls, the learned
 *       position rows, the shared {@link CpuVitTower} with its average pool for qwen2a / voxtral /
 *       musicflamingo, the zero-padded frame stack for ultravox / voxtral / meralion, then the projector;</li>
 *   <li>qwen3a: the mel viewed as 100-frame batches, three stride-2 conv2d + GELU-erf blocks per batch, the
 *       {@code [freq x channel]} flatten through {@code a.conv_out}, the first 13 position rows repeated per batch,
 *       the tower, then the GELU-erf MLP;</li>
 *   <li>the other audio graphs: one {@link FamilyEncoder} per family, each following its
 *       tools/mtmd/models graph ({@link ConformerTower} for lfm2a / d1omni_a, {@link Lfm3aTower},
 *       {@link Gemma4aTower}, {@link Gemma4uaTower}, {@link ParakeetTower}, {@link GraniteSpeechTower},
 *       {@link VoxtralRtTower}, {@link MimoAudioTower}, {@link MuseSparkTower} and {@link Dots3NoteATower}), and
 *       for gemma3na the sub-sample convolutions, the conformer blocks and the soft-token embedder of
 *       {@link Gemma3nAudioGraph}, padded to 188 rows per 30 s window.</li>
 * </ul>
 * {@link #embed} runs the whole input path: resampling to the checkpoint's rate, the projector's log-mel
 * frontend ({@link MelFrontend}) and one encode per mel chunk. Matmuls run through {@link CpuMatVec} in the
 * stored weight types. Not thread-safe; a closed encoder refuses calls.
 */
public final class CpuAudioEncoder implements AutoCloseable {

    /** Epsilon of the ultravox pre/mid RMS norms (whisper-enc.cpp). */
    private static final float ULTRAVOX_EPS = 1e-6f;

    private final ClipAudioModel model;
    private final ClipHparams hp;
    private final CpuMatVec matVec;
    private final Function<String, float[]> floats;
    private final CpuVitTower tower;
    private final Front front;
    private final Projector proj;
    private final FamilyEncoder family;
    private boolean closed;

    /** The conv front of either graph: kernels as im2col matrices, biases and the position rows. */
    private static final class Front {
        final CpuTensorRef conv1;
        final float[] conv1B;
        final CpuTensorRef conv2;
        final float[] conv2B;
        final CpuTensorRef conv3;
        final float[] conv3B;
        final CpuTensorRef convOut;
        final float[] convOutB;
        final float[] posEmbd;

        Front(ClipAudioModel m, Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            ClipAudioModel.Tensors t = m.tensors();
            int e = m.hparams().nEmbd();
            int k = ClipAudioModel.CONV_K;
            if (m.isWhisper()) {
                conv1 = CpuVitTower.matrix(refs.apply(t.conv1W.name()), e, (long) k * m.hparams().audio().nMelBins());
                conv2 = CpuVitTower.matrix(refs.apply(t.conv2W.name()), e, (long) k * e);
                conv3 = null;
                convOut = null;
            } else {
                int[] c = m.convChannels();
                conv1 = CpuVitTower.matrix(refs.apply(t.conv1W.name()), c[0], (long) k * k);
                conv2 = CpuVitTower.matrix(refs.apply(t.conv2W.name()), c[1], (long) k * k * c[0]);
                conv3 = CpuVitTower.matrix(refs.apply(t.conv3W.name()), c[2], (long) k * k * c[1]);
                convOut = refs.apply(t.convOutW.name());
            }
            conv1B = floats.apply(t.conv1B.name());
            conv2B = floats.apply(t.conv2B.name());
            conv3B = floatsOrNull(t.conv3B, floats);
            convOutB = floatsOrNull(t.convOutB, floats);
            posEmbd = floats.apply(t.posEmbd.name());
        }
    }

    /** The projector weights; absent ones are null. */
    private static final class Projector {
        final float[] normPreW;
        final float[] normPreB;
        final float[] normMidW;
        final CpuTensorRef fc;
        final float[] fcB;
        final CpuTensorRef mm0;
        final float[] mm0B;
        final CpuTensorRef mm1;
        final float[] mm1B;
        final CpuTensorRef mm2;
        final float[] mm2B;
        final CpuTensorRef mm3;
        final float[] mm3B;
        final float[] boi;
        final float[] eoi;

        Projector(ClipAudioModel.Tensors t, Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            normPreW = floatsOrNull(t.normPreW, floats);
            normPreB = floatsOrNull(t.normPreB, floats);
            normMidW = floatsOrNull(t.normMidW, floats);
            fc = refOrNull(t.fcW, refs);
            fcB = floatsOrNull(t.fcB, floats);
            mm0 = refOrNull(t.mm0W, refs);
            mm0B = floatsOrNull(t.mm0B, floats);
            mm1 = refOrNull(t.mm1W, refs);
            mm1B = floatsOrNull(t.mm1B, floats);
            mm2 = refOrNull(t.mm2W, refs);
            mm2B = floatsOrNull(t.mm2B, floats);
            mm3 = refOrNull(t.mm3W, refs);
            mm3B = floatsOrNull(t.mm3B, floats);
            boi = floatsOrNull(t.boi, floats);
            eoi = floatsOrNull(t.eoi, floats);
        }
    }

    private CpuAudioEncoder(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                            Function<String, CpuTensorRef> refs) {
        this.model = model;
        this.hp = model.hparams();
        this.matVec = Objects.requireNonNull(matVec, "matVec");
        this.floats = floats;
        ClipAudioModel.Tensors t = model.tensors();
        if (!model.isWhisper() && !ProjectorEntry.ENCODER_QWEN3A.equals(model.encoderGraph())) {
            this.tower = null;
            this.front = null;
            this.proj = null;
            this.family = bindFamily(model, matVec, floats, refs);
            return;
        }
        this.tower = new CpuVitTower(hp, matVec, model.layers(), t.preLnW, t.preLnB, t.postLnW, t.postLnB,
                floats, refs, null);
        this.front = new Front(model, floats, refs);
        this.proj = new Projector(t, floats, refs);
        this.family = null;
    }

    /** The family executor of {@code model} (every graph but the whisper and qwen3a encoders). */
    private static FamilyEncoder bindFamily(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                                            Function<String, CpuTensorRef> refs) {
        switch (model.encoderGraph()) {
            case ProjectorEntry.ENCODER_CONFORMER:
                return ConformerTower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_LFM3A:
                return Lfm3aTower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_GEMMA4A:
                return Gemma4aTower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_GEMMA4UA:
                return Gemma4uaTower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_PARAKEET:
                return ParakeetTower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_GRANITE_SPEECH:
                return GraniteSpeechTower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_VOXTRAL_RT:
                return VoxtralRtTower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_MIMO_AUDIO:
                return MimoAudioTower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_MUSE_SPARK_AUDIO:
                return MuseSparkTower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_DOTS3NOTE_A:
                return Dots3NoteATower.bind(model, matVec, floats, refs);
            case ProjectorEntry.ENCODER_GEMMA3N_USM:
                return Gemma3nAudioGraph.bind(model.gemma3n(), matVec, floats, refs);
            default:
                throw new IllegalStateException("unknown audio encoder graph " + model.encoderGraph());
        }
    }

    /**
     * Binds an executor over {@code model} with the weights of {@code floats} / {@code refs}; matmuls run
     * through {@code matVec} in the tensors' stored types.
     */
    public static CpuAudioEncoder bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                                       Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(floats, "floats");
        Objects.requireNonNull(refs, "refs");
        return new CpuAudioEncoder(model, matVec, floats, refs);
    }

    /** The geometry this executor runs. */
    public ClipAudioModel model() {
        return model;
    }

    /**
     * The mel chunks of {@code clip}: resampled to the checkpoint's rate, then framed by the projector's log-mel
     * frontend (3000-frame chunks for the whisper family, 800-frame windows padded to 100-frame batches for
     * qwen3a, one window per 30 s for gemma3na, and the other flavors of the other towers).
     *
     * @throws IllegalArgumentException when the clip is empty
     */
    public List<MelChunk> chunks(AudioClip clip) {
        Objects.requireNonNull(clip, "clip");
        AudioClip resampled = AudioResampler.resample(clip, hp.audio().sampleRate());
        MelFrontend.Config config = model.melConfig();
        if (model.melFilters() != null) {
            config = config.withFrontend(floats.apply(model.melFilters().name()),
                    floats.apply(model.melWindow().name()));
        }
        return MelFrontend.chunks(model.melFlavor(), resampled, config);
    }

    /**
     * Every embedding of {@code clip}, one {@code [tokens][outputWidth]} block per mel chunk of
     * {@link #chunks}, in order.
     *
     * @throws IllegalStateException when the encoder is closed
     */
    public List<float[][]> embed(AudioClip clip) {
        ensureOpen();
        List<MelChunk> chunks = chunks(clip);
        List<float[][]> out = new ArrayList<>(chunks.size());
        for (MelChunk chunk : chunks) {
            out.add(encode(chunk));
        }
        return out;
    }

    /** Embedding rows {@link #embed} returns for {@code clip} in total. */
    public int nTokens(AudioClip clip) {
        int total = 0;
        for (MelChunk chunk : chunks(clip)) {
            total += model.nTokens(chunk.nLen());
        }
        return total;
    }

    /**
     * Encodes one mel chunk ({@code [nMel][nLen]}, mel-major) to embedding rows.
     *
     * @return {@code [nTokens][outputWidth]} rows (fresh arrays owned by the caller)
     * @throws IllegalArgumentException when the chunk has another mel width, more frames than the position table
     *                                  covers, too few frames for the pool, or (qwen3a) a frame count that is not
     *                                  a multiple of the 100-frame conv batch, or (gemma3na) more frames than one
     *                                  30 s window
     * @throws IllegalStateException    when the encoder is closed
     */
    public float[][] encode(MelChunk chunk) {
        ensureOpen();
        Objects.requireNonNull(chunk, "chunk");
        if (family != null) {
            return family.encode(chunk);
        }
        int nMel = hp.audio().nMelBins();
        if (chunk.nMel() != nMel) {
            throw new IllegalArgumentException("mel chunk has " + chunk.nMel() + " bins, the encoder needs " + nMel);
        }
        return model.isWhisper() ? encodeWhisper(chunk) : encodeQwen3a(chunk);
    }

    // ------------------------------------------------------------------ whisper

    private float[][] encodeWhisper(MelChunk chunk) {
        int len = chunk.nLen();
        int e = hp.nEmbd();
        int nPos = (len - 1) / 2 + 1;
        if (nPos > model.maxPositions()) {
            throw new IllegalArgumentException("mel chunk of " + len + " frames gives " + nPos + " positions, the"
                    + " position table holds " + model.maxPositions());
        }
        if (model.avgPool() && nPos < 2) {
            throw new IllegalArgumentException("mel chunk of " + len + " frames is too short for the average"
                    + " pool of " + hp.projectorType());
        }
        // conv1d_ph stride 1 over the mel-major chunk, then stride 2; im2col rows are token-major, so the
        // results come out transposed already ([T][E], the ggml_transpose of whisper-enc.cpp)
        int nMel = chunk.nMel();
        float[] cols = new float[len * nMel * ClipAudioModel.CONV_K];
        for (int t = 0; t < len; t++) {
            im2col1d(chunk.data(), len, nMel, t, 1, cols, t * nMel * ClipAudioModel.CONV_K, true);
        }
        float[] h1 = convRows(front.conv1, cols, len, e, front.conv1B);
        cols = new float[nPos * e * ClipAudioModel.CONV_K];
        for (int t = 0; t < nPos; t++) {
            im2col1d(h1, len, e, t, 2, cols, t * e * ClipAudioModel.CONV_K, false);
        }
        float[] rows = convRows(front.conv2, cols, nPos, e, front.conv2B);
        VisionRows.addPrefix(front.posEmbd, rows, nPos * e);
        rows = tower.forward(rows, nPos, NormKind.LAYER,
                CpuVitTower.Options.NONE.withAvgPool(model.avgPool()));
        int n = model.avgPool() ? (nPos - 2) / 2 + 1 : nPos;
        if (model.stackFrames()) {
            rows = stack(rows, n, e);
            n = rows.length / (e * model.stackFactor());
        }
        return project(rows, n);
    }

    /**
     * One im2col vector of a kernel-3 half-padded conv1d at output {@code t}: slot {@code ic * 3 + k} reads input
     * channel {@code ic} at frame {@code t * stride + k - 1}, zero outside. The input is channel-major
     * ({@code x[ic * len + frame]}) when {@code channelMajor}, token-major ({@code x[frame * nCh + ic]}) otherwise.
     */
    private static void im2col1d(float[] x, int len, int nCh, int t, int stride, float[] dst, int dstOff,
                                 boolean channelMajor) {
        int k = ClipAudioModel.CONV_K;
        for (int ic = 0; ic < nCh; ic++) {
            for (int j = 0; j < k; j++) {
                int src = t * stride + j - k / 2;
                float v = 0f;
                if (src >= 0 && src < len) {
                    v = channelMajor ? x[ic * len + src] : x[src * nCh + ic];
                }
                dst[dstOff + ic * k + j] = v;
            }
        }
    }

    /** {@code nRows} im2col rows through a conv matrix, plus the per-channel bias, then GELU-erf. */
    private float[] convRows(CpuTensorRef w, float[] cols, int nRows, int nOut, float[] bias) {
        float[] y = new float[nRows * nOut];
        matVec.matmul(w, cols, 0, y, 0, nRows);
        VisionRows.addBias(y, bias);
        VisionRows.activate(ActOp.GELU_ERF, y);
        return y;
    }

    /**
     * {@code build_stack}: the flat token-major rows zero-padded to a multiple of {@code factor * width} and
     * re-viewed as rows of that width.
     */
    private float[] stack(float[] rows, int nRows, int width) {
        int stride = width * model.stackFactor();
        int total = nRows * width;
        int padded = (total + stride - 1) / stride * stride;
        float[] out = new float[padded];
        System.arraycopy(rows, 0, out, 0, total);
        return out;
    }

    // ------------------------------------------------------------------ whisper projectors

    private float[][] project(float[] rows, int n) {
        int d = model.outputWidth();
        switch (model.projectorGraph()) {
            case ProjectorEntry.PROJECTOR_ULTRAVOX:
                return ultravox(rows, n, d);
            case ProjectorEntry.PROJECTOR_QWEN2A: {
                float[] out = new float[n * d];
                matVec.matmul(proj.fc, rows, 0, out, 0, n);
                VisionRows.addBias(out, proj.fcB);
                return VisionRows.tokenRows(out, d);
            }
            case ProjectorEntry.PROJECTOR_VOXTRAL:
            case ProjectorEntry.PROJECTOR_MUSIC_FLAMINGO:
                return VisionRows.tokenRows(geluMlp(rows, n), d);
            case ProjectorEntry.PROJECTOR_MERALION:
                return meralion(rows, n, d);
            case ProjectorEntry.PROJECTOR_GLMA:
                return glma(rows, n, d);
            default:
                throw new IllegalStateException("projector graph " + model.projectorGraph()
                        + " does not run on the whisper encoder");
        }
    }

    /** RMS pre-norm, {@code mm.a.mlp.1}, SwiGLU with SiLU on the second half, RMS mid-norm, {@code mm.a.mlp.2}. */
    private float[][] ultravox(float[] rows, int n, int d) {
        int in = (int) proj.mm1.cols();
        int ff = (int) proj.mm1.rows();
        int half = ff / 2;
        CpuOps.rmsNorm(rows, 0, rows, 0, n, in, proj.normPreW, ULTRAVOX_EPS);
        float[] h = new float[n * ff];
        matVec.matmul(proj.mm1, rows, 0, h, 0, n);
        float[] g = new float[n * half];
        CpuOps.gluRows(GluOp.SWIGLU, ActParams.NONE, n, half, h, half, ff, h, 0, ff, g, 0, half);
        CpuOps.rmsNorm(g, 0, g, 0, n, half, proj.normMidW, ULTRAVOX_EPS);
        float[] out = new float[n * d];
        matVec.matmul(proj.mm2, g, 0, out, 0, n);
        return VisionRows.tokenRows(out, d);
    }

    /** {@code build_ffn(mm.a.mlp.1, mm.a.mlp.2, GELU_ERF)}, biases where the projector has them. */
    private float[] geluMlp(float[] rows, int n) {
        int ff = (int) proj.mm1.rows();
        float[] h = new float[n * ff];
        matVec.matmul(proj.mm1, rows, 0, h, 0, n);
        VisionRows.addBias(h, proj.mm1B);
        VisionRows.activate(ActOp.GELU_ERF, h);
        float[] out = new float[n * model.outputWidth()];
        matVec.matmul(proj.mm2, h, 0, out, 0, n);
        VisionRows.addBias(out, proj.mm2B);
        return out;
    }

    /** Layer norm, SiLU {@code mlp.0}, {@code silu(mlp.1) * mlp.2}, {@code mlp.3} (whisper-enc.cpp). */
    private float[][] meralion(float[] rows, int n, int d) {
        int in = (int) proj.mm0.cols();
        int h = (int) proj.mm0.rows();
        CpuOps.layerNorm(rows, 0, rows, 0, n, in, proj.normPreW, proj.normPreB, hp.eps());
        float[] h0 = new float[n * h];
        matVec.matmul(proj.mm0, rows, 0, h0, 0, n);
        VisionRows.addBias(h0, proj.mm0B);
        VisionRows.activate(ActOp.SILU, h0);
        float[] gate = new float[n * h];
        matVec.matmul(proj.mm1, h0, 0, gate, 0, n);
        VisionRows.addBias(gate, proj.mm1B);
        float[] pool = new float[n * h];
        matVec.matmul(proj.mm2, h0, 0, pool, 0, n);
        VisionRows.addBias(pool, proj.mm2B);
        CpuOps.glu(GluOp.SWIGLU, ActParams.NONE, gate, 0, pool, 0, gate, 0, n * h);
        float[] out = new float[n * d];
        matVec.matmul(proj.mm3, gate, 0, out, 0, n);
        VisionRows.addBias(out, proj.mm3B);
        return VisionRows.tokenRows(out, d);
    }

    /** Layer norm over the tower width, the frame stack, the GELU-erf MLP, then the BOI and EOI rows. */
    private float[][] glma(float[] rows, int n, int d) {
        int e = hp.nEmbd();
        CpuOps.layerNorm(rows, 0, rows, 0, n, e, proj.normPreW, proj.normPreB, hp.eps());
        float[] stacked = stack(rows, n, e);
        int ns = stacked.length / (e * model.stackFactor());
        float[] body = geluMlp(stacked, ns);
        float[][] out = new float[ns + 2][];
        out[0] = proj.boi.clone();
        for (int r = 0; r < ns; r++) {
            out[r + 1] = new float[d];
            System.arraycopy(body, r * d, out[r + 1], 0, d);
        }
        out[ns + 1] = proj.eoi.clone();
        return out;
    }

    // ------------------------------------------------------------------ qwen3a

    private float[][] encodeQwen3a(MelChunk chunk) {
        int len = chunk.nLen();
        int batch = ClipAudioModel.QWEN3A_CHUNK_FRAMES;
        if (len % batch != 0) {
            throw new IllegalArgumentException("mel chunk has " + len + " frames, not a multiple of the " + batch
                    + "-frame conv batch (the qwen3a frontend pads them)");
        }
        int e = hp.nEmbd();
        int perBatch = ClipAudioModel.QWEN3A_TOKENS_PER_CHUNK;
        int nBatches = len / batch;
        int n = nBatches * perBatch;
        float[] rows = new float[n * e];
        for (int b = 0; b < nBatches; b++) {
            convBatch(chunk, b, rows, b * perBatch * e);
            for (int p = 0; p < perBatch; p++) {
                CpuOps.add(front.posEmbd, p * e, rows, (b * perBatch + p) * e, rows, (b * perBatch + p) * e, e);
            }
        }
        rows = tower.forward(rows, n, NormKind.LAYER, CpuVitTower.Options.NONE);
        return VisionRows.tokenRows(geluMlp(rows, n), model.outputWidth());
    }

    /**
     * One 100-frame batch through the three conv2d blocks and {@code a.conv_out} into {@code dst} at
     * {@code dstOff}, 13 token rows. Images are pixel-major: element {@code (x = time, y = mel, c)} at
     * {@code (y * w + x) * channels + c}.
     */
    private void convBatch(MelChunk chunk, int b, float[] dst, int dstOff) {
        int w = ClipAudioModel.QWEN3A_CHUNK_FRAMES;
        int h = chunk.nMel();
        float[] img = new float[w * h];
        for (int m = 0; m < h; m++) {
            System.arraycopy(chunk.data(), m * chunk.nLen() + b * w, img, m * w, w);
        }
        int[] c = model.convChannels();
        img = conv2d(img, w, h, 1, front.conv1, front.conv1B);
        w = ClipAudioModel.convOut(w);
        h = ClipAudioModel.convOut(h);
        img = conv2d(img, w, h, c[0], front.conv2, front.conv2B);
        w = ClipAudioModel.convOut(w);
        h = ClipAudioModel.convOut(h);
        img = conv2d(img, w, h, c[1], front.conv3, front.conv3B);
        int ow = ClipAudioModel.convOut(w);
        int oh = ClipAudioModel.convOut(h);
        // permute [OW, OH, OC] -> [OH, OC, OW]: feature f + OH * c per time step (qwen3a.cpp)
        int flat = oh * c[2];
        float[] feats = new float[ow * flat];
        for (int t = 0; t < ow; t++) {
            for (int f = 0; f < oh; f++) {
                for (int ch = 0; ch < c[2]; ch++) {
                    feats[t * flat + f + oh * ch] = img[(f * ow + t) * c[2] + ch];
                }
            }
        }
        matVec.matmul(front.convOut, feats, 0, dst, dstOff, ow);
        if (front.convOutB != null) {
            CpuOps.addRowBias(dst, dstOff, ow, hp.nEmbd(), front.convOutB, 0);
        }
    }

    /**
     * {@code ggml_conv_2d} with kernel 3x3, stride 2, padding 1 over a pixel-major image of {@code cIn} channels
     * as an im2col matmul (slot {@code (ci * 3 + ky) * 3 + kx}), plus the bias and GELU-erf.
     */
    private float[] conv2d(float[] img, int w, int h, int cIn, CpuTensorRef kernel, float[] bias) {
        int k = ClipAudioModel.CONV_K;
        int ow = ClipAudioModel.convOut(w);
        int oh = ClipAudioModel.convOut(h);
        int span = cIn * k * k;
        float[] cols = new float[ow * oh * span];
        for (int f = 0; f < oh; f++) {
            for (int t = 0; t < ow; t++) {
                int base = (f * ow + t) * span;
                for (int ci = 0; ci < cIn; ci++) {
                    for (int ky = 0; ky < k; ky++) {
                        int sy = 2 * f + ky - 1;
                        for (int kx = 0; kx < k; kx++) {
                            int sx = 2 * t + kx - 1;
                            boolean in = sy >= 0 && sy < h && sx >= 0 && sx < w;
                            cols[base + (ci * k + ky) * k + kx] = in ? img[(sy * w + sx) * cIn + ci] : 0f;
                        }
                    }
                }
            }
        }
        return convRows(kernel, cols, ow * oh, (int) kernel.rows(), bias);
    }

    @Override
    public void close() {
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the audio encoder is closed");
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "CpuAudioEncoder{%s, n_mel=%d, out=%d}", hp.projectorType(),
                hp.audio().nMelBins(), model.outputWidth());
    }
}
