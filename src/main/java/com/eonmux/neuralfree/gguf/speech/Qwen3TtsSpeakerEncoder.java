package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.AudioResampler;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.audio.MelFrontend;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * The qwen3tts speaker encoder of llama.cpp's {@code clip_graph_qwen3tts_spkenc}
 * (tools/mtmd/models/qwen3tts-spkenc.cpp:1-197): an ECAPA-TDNN that turns a reference waveform into one x-vector of
 * the talker's hidden width.
 *
 * <ol>
 *   <li>the waveform becomes a 128-bin log-mel at 24 kHz ({@link MelFrontend.Flavor#QWEN3TTS_SPKENC});</li>
 *   <li>a TDNN stem convolution widens the mel channels and a ReLU follows;</li>
 *   <li>three SE-Res2Net blocks at dilations 2, 3 and 4 run a pointwise TDNN, an 8-way Res2Net, a pointwise TDNN,
 *       a squeeze-and-excitation gate and a residual add;</li>
 *   <li>the three blocks are concatenated and reduced by the multi-layer feature aggregation TDNN;</li>
 *   <li>attentive statistics pooling with temporal softmax attention gives a weighted mean and standard
 *       deviation, concatenated into 2C values;</li>
 *   <li>the final 1x1 convolution projects them to the talker width.</li>
 * </ol>
 *
 * <p>The convolutions are "same" padded by reflection, as {@code conv1d_same} of the reference graph. Tensors use
 * the clip generator names of tools/mtmd/clip-impl.h in ggml dimension order ({@code ne[0]} first); every kernel is
 * decoded to an F32 host array at open. Immutable and thread-safe over the file's weights.</p>
 */
public final class Qwen3TtsSpeakerEncoder implements SpeakerEncoder {

    /** The {@code clip.audio.projector_type} of the speaker encoder this class runs. */
    public static final String PROJECTOR_TYPE = "qwen3tts_spkenc";
    /** The reference waveform's sample rate (clip.cpp:2157). */
    public static final int SAMPLE_RATE = 24000;
    /** The mel front-end's FFT size (clip.cpp:2158). */
    public static final int N_FFT = 1024;
    /** The mel front-end's hop (clip.cpp:2159). */
    public static final int HOP = 256;
    /** The mel front-end's window length (clip.cpp:2158). */
    public static final int WINDOW = 1024;
    /** The reference mel bin count (qwen3tts.py:238). */
    public static final int N_MEL = 128;
    /** The fixed stage count of the reference ECAPA-TDNN backbone. */
    public static final int BLOCKS = 3;
    /** Res2Net chunks per block ({@code enc_res2net_scale}, qwen3tts-spkenc.cpp:3). */
    public static final int RES2_SCALE = 8;
    /** The block dilations ({@code enc_dilations[1..3]}, qwen3tts-spkenc.cpp:4). */
    public static final int[] DILATIONS = {2, 3, 4};
    /** The pooling variance floor (qwen3tts-spkenc.cpp:125, 148). */
    static final float VARIANCE_EPS = 1e-12f;

    /** Tensor names of the clip generator file ({@code TN_CONV1D}, {@code TN_CONV_PW1}, ...). */
    public static final String STEM = "a.conv1d.0";
    public static final String BLOCK = "a.blk.";
    public static final String MFA = "a.conv_out";
    public static final String ASP_TDNN = "a.asp_tdnn";
    public static final String ASP_ATTN = "a.asp_attn";
    public static final String FC = "mm.a.fc";

    private final int width;
    private final int melBins;
    private final int stemK;
    private final int stemIn;
    private final int channels;
    private final float[] stemW;
    private final float[] stemB;
    private final Block[] blocks;
    private final int mfaK;
    private final float[] mfaW;
    private final float[] mfaB;
    private final int aspTdnnK;
    private final int aspHidden;
    private final float[] aspTdnnW;
    private final float[] aspTdnnB;
    private final int aspAttnK;
    private final float[] aspAttnW;
    private final float[] aspAttnB;
    private final int fcK;
    private final float[] fcW;
    private final float[] fcB;

    /** One SE-Res2Net block: both pointwise TDNNs, the squeeze-and-excitation pair and the seven Res2Net convolutions. */
    private static final class Block {
        final int tdnn1K;
        final float[] tdnn1W;
        final float[] tdnn1B;
        final int tdnn2K;
        final float[] tdnn2W;
        final float[] tdnn2B;
        final int se1K;
        final int seHidden;
        final float[] se1W;
        final float[] se1B;
        final int se2K;
        final float[] se2W;
        final float[] se2B;
        final int[] res2K;
        final float[][] res2W;
        final float[][] res2B;

        Block(int tdnn1K, float[] tdnn1W, float[] tdnn1B, int tdnn2K, float[] tdnn2W, float[] tdnn2B, int se1K,
              int seHidden, float[] se1W, float[] se1B, int se2K, float[] se2W, float[] se2B, int[] res2K,
              float[][] res2W, float[][] res2B) {
            this.tdnn1K = tdnn1K;
            this.tdnn1W = tdnn1W;
            this.tdnn1B = tdnn1B;
            this.tdnn2K = tdnn2K;
            this.tdnn2W = tdnn2W;
            this.tdnn2B = tdnn2B;
            this.se1K = se1K;
            this.seHidden = seHidden;
            this.se1W = se1W;
            this.se1B = se1B;
            this.se2K = se2K;
            this.se2W = se2W;
            this.se2B = se2B;
            this.res2K = res2K;
            this.res2W = res2W;
            this.res2B = res2B;
        }
    }

    private Qwen3TtsSpeakerEncoder(Builder b) {
        this.width = b.width;
        this.melBins = b.melBins;
        this.stemK = b.stemK;
        this.stemIn = b.stemIn;
        this.channels = b.channels;
        this.stemW = b.stemW;
        this.stemB = b.stemB;
        this.blocks = b.blocks;
        this.mfaK = b.mfaK;
        this.mfaW = b.mfaW;
        this.mfaB = b.mfaB;
        this.aspTdnnK = b.aspTdnnK;
        this.aspHidden = b.aspHidden;
        this.aspTdnnW = b.aspTdnnW;
        this.aspTdnnB = b.aspTdnnB;
        this.aspAttnK = b.aspAttnK;
        this.aspAttnW = b.aspAttnW;
        this.aspAttnB = b.aspAttnB;
        this.fcK = b.fcK;
        this.fcW = b.fcW;
        this.fcB = b.fcB;
    }

    /**
     * Resolves the speaker encoder of an opened clip generator file.
     *
     * @throws IllegalArgumentException when the file declares another projector type
     * @throws GgufFormatException      when a key or a tensor differs from the reference graph's geometry
     */
    public static Qwen3TtsSpeakerEncoder open(GgufKv kv, TensorCatalog tc, CpuWeights weights) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(weights, "weights");
        String type = kv.str(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, null);
        if (!PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE + " is "
                    + (type == null ? "absent" : "'" + type + "'")
                    + ", a qwen3tts speaker encoder declares '" + PROJECTOR_TYPE + "'");
        }
        Builder b = new Builder();
        b.melBins = kv.u32(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, N_MEL);
        if (b.melBins <= 0) {
            throw new GgufFormatException("qwen3tts speaker encoder: " + GgufKeys.CLIP_AUDIO_NUM_MEL_BINS
                    + " must be positive, got " + b.melBins);
        }
        int declaredBlocks = kv.u32(GgufKeys.CLIP_AUDIO_BLOCK_COUNT, BLOCKS);
        if (declaredBlocks != BLOCKS) {
            throw new GgufFormatException("qwen3tts speaker encoder: " + GgufKeys.CLIP_AUDIO_BLOCK_COUNT + " is "
                    + declaredBlocks + ", the ECAPA-TDNN backbone has " + BLOCKS + " stages");
        }

        TensorRef stemRef = tc.require(STEM + ".weight");
        b.stemK = dim(stemRef, 0);
        b.stemIn = dim(stemRef, 1);
        b.channels = dim(stemRef, 2);
        if (b.stemIn != b.melBins) {
            throw new GgufFormatException("qwen3tts speaker encoder: " + STEM + ".weight takes " + b.stemIn
                    + " mel bins, the front-end writes " + b.melBins);
        }
        if (b.channels % RES2_SCALE != 0) {
            throw new GgufFormatException("qwen3tts speaker encoder: the stem writes " + b.channels
                    + " channels, the Res2Net split needs a multiple of " + RES2_SCALE);
        }
        requireSameKernel(stemRef, STEM, 1);
        b.stemW = weights.floats(STEM + ".weight");
        b.stemB = weights.floats(STEM + ".bias");
        vector(b.stemB, b.channels, STEM + ".bias");

        int chunk = b.channels / RES2_SCALE;
        b.blocks = new Block[BLOCKS];
        for (int il = 0; il < BLOCKS; il++) {
            int bid = il + 1;
            String base = BLOCK + bid + ".";
            TensorRef tdnn1 = tc.require(base + "conv_pw1.weight");
            TensorRef tdnn2 = tc.require(base + "conv_pw2.weight");
            checkConv(tdnn1, base + "conv_pw1.weight", b.channels, b.channels);
            checkConv(tdnn2, base + "conv_pw2.weight", b.channels, b.channels);
            TensorRef se1 = tc.require(base + "se_conv1.weight");
            TensorRef se2 = tc.require(base + "se_conv2.weight");
            checkConv(se1, base + "se_conv1.weight", b.channels, -1);
            int seHidden = dim(se1, 2);
            checkConv(se2, base + "se_conv2.weight", seHidden, b.channels);
            requireSameKernel(tdnn1, base + "conv_pw1", 1);
            requireSameKernel(tdnn2, base + "conv_pw2", 1);
            requireSameKernel(se1, base + "se_conv1", 1);
            requireSameKernel(se2, base + "se_conv2", 1);
            int[] res2K = new int[RES2_SCALE - 1];
            float[][] res2W = new float[RES2_SCALE - 1][];
            float[][] res2B = new float[RES2_SCALE - 1][];
            for (int xid = 0; xid < RES2_SCALE - 1; xid++) {
                String name = base + "res2." + xid;
                TensorRef res = tc.require(name + ".weight");
                checkConv(res, name + ".weight", chunk, chunk);
                requireSameKernel(res, name, DILATIONS[il]);
                res2K[xid] = dim(res, 0);
                res2W[xid] = weights.floats(name + ".weight");
                res2B[xid] = weights.floats(name + ".bias");
                vector(res2B[xid], chunk, name + ".bias");
            }
            b.blocks[il] = new Block(dim(tdnn1, 0), weights.floats(base + "conv_pw1.weight"),
                    weights.floats(base + "conv_pw1.bias"), dim(tdnn2, 0),
                    weights.floats(base + "conv_pw2.weight"), weights.floats(base + "conv_pw2.bias"),
                    dim(se1, 0), seHidden, weights.floats(base + "se_conv1.weight"),
                    weights.floats(base + "se_conv1.bias"), dim(se2, 0), weights.floats(base + "se_conv2.weight"),
                    weights.floats(base + "se_conv2.bias"), res2K, res2W, res2B);
            vector(b.blocks[il].tdnn1B, b.channels, base + "conv_pw1.bias");
            vector(b.blocks[il].tdnn2B, b.channels, base + "conv_pw2.bias");
            vector(b.blocks[il].se1B, seHidden, base + "se_conv1.bias");
            vector(b.blocks[il].se2B, b.channels, base + "se_conv2.bias");
        }

        TensorRef mfaRef = tc.require(MFA + ".weight");
        checkConv(mfaRef, MFA + ".weight", BLOCKS * b.channels, b.channels);
        requireSameKernel(mfaRef, MFA, 1);
        b.mfaK = dim(mfaRef, 0);
        b.mfaW = weights.floats(MFA + ".weight");
        b.mfaB = weights.floats(MFA + ".bias");
        vector(b.mfaB, b.channels, MFA + ".bias");

        TensorRef tdnn = tc.require(ASP_TDNN + ".weight");
        checkConv(tdnn, ASP_TDNN + ".weight", 3 * b.channels, -1);
        requireSameKernel(tdnn, ASP_TDNN, 1);
        b.aspTdnnK = dim(tdnn, 0);
        b.aspHidden = dim(tdnn, 2);
        b.aspTdnnW = weights.floats(ASP_TDNN + ".weight");
        b.aspTdnnB = weights.floats(ASP_TDNN + ".bias");
        vector(b.aspTdnnB, b.aspHidden, ASP_TDNN + ".bias");

        TensorRef attn = tc.require(ASP_ATTN + ".weight");
        checkConv(attn, ASP_ATTN + ".weight", b.aspHidden, b.channels);
        requireSameKernel(attn, ASP_ATTN, 1);
        b.aspAttnK = dim(attn, 0);
        b.aspAttnW = weights.floats(ASP_ATTN + ".weight");
        b.aspAttnB = weights.floats(ASP_ATTN + ".bias");
        vector(b.aspAttnB, b.channels, ASP_ATTN + ".bias");

        TensorRef fcRef = tc.require(FC + ".weight");
        checkConv(fcRef, FC + ".weight", 2 * b.channels, -1);
        requireSameKernel(fcRef, FC, 1);
        b.fcK = dim(fcRef, 0);
        b.width = dim(fcRef, 2);
        b.fcW = weights.floats(FC + ".weight");
        b.fcB = weights.floats(FC + ".bias");
        vector(b.fcB, b.width, FC + ".bias");
        return new Qwen3TtsSpeakerEncoder(b);
    }

    /** The expected feature count of the mel front-end of a file declaring {@code melBins} bins. */
    public static MelFrontend.Config melConfig(int melBins) {
        return MelFrontend.Config.builder(melBins, SAMPLE_RATE, N_FFT, WINDOW, HOP).build();
    }

    // ------------------------------------------------------------------ encoding

    @Override
    public String name() {
        return PROJECTOR_TYPE;
    }

    @Override
    public int sampleRate() {
        return SAMPLE_RATE;
    }

    @Override
    public int width() {
        return width;
    }

    /** The clip's log-mel, frame major: {@code [T][melBins]}. */
    float[][] logMel(AudioClip clip) {
        AudioClip mono = AudioResampler.resample(Objects.requireNonNull(clip, "clip"), SAMPLE_RATE);
        if (mono.length() == 0) {
            throw new IllegalArgumentException("the speaker reference is empty");
        }
        MelChunk mel = MelFrontend.chunks(MelFrontend.Flavor.QWEN3TTS_SPKENC, mono, melConfig(melBins)).get(0);
        int frames = mel.nLen();
        float[][] out = new float[frames][melBins];
        for (int t = 0; t < frames; t++) {
            for (int m = 0; m < melBins; m++) {
                out[t][m] = mel.value(m, t);
            }
        }
        return out;
    }

    /**
     * {@inheritDoc}
     *
     * @throws IllegalArgumentException when the reference is empty, shorter than the front-end's reflections, or
     *                                  shorter than a same-padded kernel's reflection
     */
    @Override
    public float[][] encode(AudioClip clip) {
        float[][] mel = logMel(clip);
        int frames = mel.length;
        if (frames < 1) {
            throw new IllegalArgumentException("the speaker reference yields no mel frame");
        }
        float[][] cur = relu(conv(mel, stemW, stemB, stemK, stemIn, channels, 1, "stem"));
        float[][][] blockOut = new float[BLOCKS][][];
        for (int il = 0; il < BLOCKS; il++) {
            cur = block(cur, blocks[il], DILATIONS[il]);
            blockOut[il] = cur;
        }
        float[][] cat = new float[frames][BLOCKS * channels];
        for (int t = 0; t < frames; t++) {
            for (int il = 0; il < BLOCKS; il++) {
                System.arraycopy(blockOut[il][t], 0, cat[t], il * channels, channels);
            }
        }
        float[][] mfa = relu(conv(cat, mfaW, mfaB, mfaK, BLOCKS * channels, channels, 1, "mfa"));
        float[] stats = attentiveStatsPool(mfa);
        float[][] projected = conv(new float[][]{stats}, fcW, fcB, fcK, 2 * channels, width, 1, "fc");
        return new float[][]{projected[0].clone()};
    }

    /** {@code tdnn1 -> res2net -> tdnn2 -> se}, plus the block's residual. */
    private float[][] block(float[][] x, Block b, int dilation) {
        float[][] h = relu(conv(x, b.tdnn1W, b.tdnn1B, b.tdnn1K, channels, channels, 1, "tdnn1"));
        h = res2net(h, b, dilation);
        h = relu(conv(h, b.tdnn2W, b.tdnn2B, b.tdnn2K, channels, channels, 1, "tdnn2"));
        h = squeezeExcite(h, b);
        float[][] out = new float[x.length][channels];
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < channels; c++) {
                out[t][c] = h[t][c] + x[t][c];
            }
        }
        return out;
    }

    /** The 8-way Res2Net chain: chunk 0 is passed through, chunk i adds the previous output before its convolution. */
    private float[][] res2net(float[][] x, Block b, int dilation) {
        int frames = x.length;
        int chunk = channels / RES2_SCALE;
        float[][] out = new float[frames][channels];
        float[][] previous = null;
        for (int i = 0; i < RES2_SCALE; i++) {
            float[][] part = new float[frames][chunk];
            for (int t = 0; t < frames; t++) {
                System.arraycopy(x[t], i * chunk, part[t], 0, chunk);
            }
            if (i == 0) {
                for (int t = 0; t < frames; t++) {
                    System.arraycopy(part[t], 0, out[t], 0, chunk);
                }
                continue;
            }
            if (i >= 2) {
                for (int t = 0; t < frames; t++) {
                    for (int c = 0; c < chunk; c++) {
                        part[t][c] += previous[t][c];
                    }
                }
            }
            float[][] y = relu(conv(part, b.res2W[i - 1], b.res2B[i - 1], b.res2K[i - 1], chunk, chunk, dilation,
                    "res2"));
            for (int t = 0; t < frames; t++) {
                System.arraycopy(y[t], 0, out[t], i * chunk, chunk);
            }
            previous = y;
        }
        return out;
    }

    /** The squeeze-and-excitation gate: temporal mean, two convolutions, sigmoid, channel scaling. */
    private float[][] squeezeExcite(float[][] x, Block b) {
        int frames = x.length;
        float[] mean = new float[channels];
        for (int t = 0; t < frames; t++) {
            for (int c = 0; c < channels; c++) {
                mean[c] += x[t][c];
            }
        }
        for (int c = 0; c < channels; c++) {
            mean[c] /= frames;
        }
        float[][] gate = relu(conv(new float[][]{mean}, b.se1W, b.se1B, b.se1K, channels, b.seHidden, 1, "se"));
        gate = conv(gate, b.se2W, b.se2B, b.se2K, b.seHidden, channels, 1, "se");
        float[][] out = new float[frames][channels];
        for (int c = 0; c < channels; c++) {
            float g = sigmoid(gate[0][c]);
            for (int t = 0; t < frames; t++) {
                out[t][c] = x[t][c] * g;
            }
        }
        return out;
    }

    /** Attentive statistics pooling: temporal softmax attention, weighted mean and weighted std, concatenated. */
    private float[] attentiveStatsPool(float[][] x) {
        int frames = x.length;
        float[] mean = new float[channels];
        float[] std = new float[channels];
        for (int t = 0; t < frames; t++) {
            for (int c = 0; c < channels; c++) {
                mean[c] += x[t][c];
            }
        }
        for (int c = 0; c < channels; c++) {
            mean[c] /= frames;
        }
        for (int t = 0; t < frames; t++) {
            for (int c = 0; c < channels; c++) {
                float d = x[t][c] - mean[c];
                std[c] += d * d;
            }
        }
        for (int c = 0; c < channels; c++) {
            std[c] = (float) Math.sqrt(Math.max(std[c] / frames, VARIANCE_EPS));
        }
        float[][] cat = new float[frames][3 * channels];
        for (int t = 0; t < frames; t++) {
            System.arraycopy(x[t], 0, cat[t], 0, channels);
            for (int c = 0; c < channels; c++) {
                cat[t][channels + c] = mean[c];
                cat[t][2 * channels + c] = std[c];
            }
        }
        float[][] hidden = relu(conv(cat, aspTdnnW, aspTdnnB, aspTdnnK, 3 * channels, aspHidden, 1, "asp_tdnn"));
        for (int t = 0; t < frames; t++) {
            for (int c = 0; c < aspHidden; c++) {
                hidden[t][c] = (float) Math.tanh(hidden[t][c]);
            }
        }
        float[][] attention = conv(hidden, aspAttnW, aspAttnB, aspAttnK, aspHidden, channels, 1, "asp_attn");
        softmaxOverTime(attention);
        float[] weightedMean = new float[channels];
        for (int t = 0; t < frames; t++) {
            for (int c = 0; c < channels; c++) {
                weightedMean[c] += attention[t][c] * x[t][c];
            }
        }
        float[] weightedStd = new float[channels];
        for (int t = 0; t < frames; t++) {
            for (int c = 0; c < channels; c++) {
                float d = x[t][c] - weightedMean[c];
                weightedStd[c] += attention[t][c] * d * d;
            }
        }
        for (int c = 0; c < channels; c++) {
            weightedStd[c] = (float) Math.sqrt(Math.max(weightedStd[c], VARIANCE_EPS));
        }
        float[] out = new float[2 * channels];
        System.arraycopy(weightedMean, 0, out, 0, channels);
        System.arraycopy(weightedStd, 0, out, channels, channels);
        return out;
    }

    /** Softmax of each channel's attention scores over the time axis, in place. */
    private static void softmaxOverTime(float[][] scores) {
        int frames = scores.length;
        int channels = frames == 0 ? 0 : scores[0].length;
        for (int c = 0; c < channels; c++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < frames; t++) {
                max = Math.max(max, scores[t][c]);
            }
            double sum = 0.0;
            for (int t = 0; t < frames; t++) {
                double e = Math.exp(scores[t][c] - max);
                scores[t][c] = (float) e;
                sum += e;
            }
            float inv = sum > 0.0 ? (float) (1.0 / sum) : 0f;
            for (int t = 0; t < frames; t++) {
                scores[t][c] *= inv;
            }
        }
    }

    // ------------------------------------------------------------------ convolution primitives

    /**
     * A "same" convolution over frame-major {@code x[T][in]} with reflection padding on both ends.
     *
     * @throws IllegalArgumentException when the reflection needs more frames than the input holds
     */
    private static float[][] conv(float[][] x, float[] w, float[] b, int k, int in, int out, int dilation,
                                  String what) {
        int frames = x.length;
        int pad = ((k - 1) * dilation) / 2;
        if (pad >= frames && pad > 0) {
            throw new IllegalArgumentException("the speaker reference yields " + frames + " frames, fewer than the "
                    + pad + " reflected frames the " + what + " convolution needs");
        }
        int span = (k - 1) * dilation + 1;
        int padded = frames + 2 * pad;
        int outFrames = padded - span + 1;
        float[][] y = new float[outFrames][out];
        for (int t = 0; t < outFrames; t++) {
            for (int oc = 0; oc < out; oc++) {
                float acc = b[oc];
                for (int j = 0; j < k; j++) {
                    int at = t + j * dilation;
                    if (at >= padded) {
                        break;
                    }
                    int src = at - pad;
                    if (src < 0) {
                        src = -src;
                        if (src >= frames) {
                            src = 2 * frames - 2 - src;
                        }
                    } else if (src >= frames) {
                        src = 2 * frames - 2 - src;
                    }
                    if (src < 0 || src >= frames) {
                        continue;
                    }
                    for (int ic = 0; ic < in; ic++) {
                        acc += w[j + k * (ic + in * oc)] * x[src][ic];
                    }
                }
                y[t][oc] = acc;
            }
        }
        return y;
    }

    private static float[][] relu(float[][] x) {
        for (float[] row : x) {
            for (int i = 0; i < row.length; i++) {
                if (row[i] < 0f) {
                    row[i] = 0f;
                }
            }
        }
        return x;
    }

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    // ------------------------------------------------------------------ validation

    private static int dim(TensorRef ref, int i) {
        long v = ref.ne(i);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new GgufFormatException("qwen3tts speaker encoder: tensor " + ref.name() + " dimension " + i
                    + " is " + v);
        }
        return (int) v;
    }

    private static void checkConv(TensorRef ref, String name, int in, int out) {
        if (ref.rank() != 3) {
            throw new GgufFormatException("qwen3tts speaker encoder: " + name + " must be a 3-D convolution weight,"
                    + " found " + ref);
        }
        if (dim(ref, 1) != in) {
            throw new GgufFormatException("qwen3tts speaker encoder: " + name + " takes " + dim(ref, 1)
                    + " channels, expected " + in);
        }
        if (out > 0 && dim(ref, 2) != out) {
            throw new GgufFormatException("qwen3tts speaker encoder: " + name + " writes " + dim(ref, 2)
                    + " channels, expected " + out);
        }
    }

    /** A kernel whose {@code (k - 1) * dilation} is odd would shrink the frame count, unlike the reference. */
    private static void requireSameKernel(TensorRef ref, String name, int dilation) {
        int k = dim(ref, 0);
        if (((k - 1) * dilation) % 2 != 0) {
            throw new GgufFormatException("qwen3tts speaker encoder: " + name + " has kernel " + k + " with dilation "
                    + dilation + ", which does not preserve the frame count");
        }
    }

    private static void vector(float[] v, int n, String name) {
        if (v == null || v.length != n) {
            throw new GgufFormatException("qwen3tts speaker encoder: " + name + " must hold " + n + " floats, got "
                    + (v == null ? "none" : v.length));
        }
    }

    private static final class Builder {
        int width;
        int melBins;
        int stemK;
        int stemIn;
        int channels;
        float[] stemW;
        float[] stemB;
        Block[] blocks;
        int mfaK;
        float[] mfaW;
        float[] mfaB;
        int aspTdnnK;
        int aspHidden;
        float[] aspTdnnW;
        float[] aspTdnnB;
        int aspAttnK;
        float[] aspAttnW;
        float[] aspAttnB;
        int fcK;
        float[] fcW;
        float[] fcB;
    }

    @Override
    public String toString() {
        return "Qwen3TtsSpeakerEncoder{mel=" + melBins + ", channels=" + channels + ", width=" + width + '}';
    }
}
