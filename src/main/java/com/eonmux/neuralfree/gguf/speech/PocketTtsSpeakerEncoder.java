package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.AudioResampler;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * The pockettts speaker encoder of llama.cpp's {@code clip_graph_pockettts_spkenc}
 * (tools/mtmd/models/pockettts-spkenc.cpp and pockettts-seanet.cpp): the mimi encoder turns a reference waveform
 * into one conditioning row per 12.5 Hz model frame.
 *
 * <ol>
 *   <li>the raw 24 kHz waveform is zero-padded to whole 1920-sample frames and runs through the causal SEANet
 *       encoder: an input convolution, three residual and strided stages (strides 4, 5 and 6) and an output
 *       convolution;</li>
 *   <li>a pre-norm transformer with RoPE and a 250-frame causal sliding window refines the frames, with a layer
 *       scale on both residual paths;</li>
 *   <li>a strided convolution downsamples 16 encoder frames to one model frame, repeating the first frame as its
 *       left context;</li>
 *   <li>{@code a.speaker_proj} projects the 32-value voice latent to the backbone's embedding width.</li>
 * </ol>
 *
 * <p>The reference hardcodes the mimi geometry (clip.cpp:2208-2222): 24 kHz, encoder strides 4/5/6, downsample 16,
 * a 250-frame attention window and 10000 RoPE. A reference longer than {@value #MAX_SECONDS} seconds is truncated to
 * its first {@value #MAX_SECONDS} seconds, as the reference preprocessor does. Tensors use the clip generator names
 * of tools/mtmd/clip-impl.h in ggml dimension order. Immutable and thread-safe over the file's weights.</p>
 */
public final class PocketTtsSpeakerEncoder implements SpeakerEncoder {

    /** The {@code clip.audio.projector_type} of the speaker encoder this class runs. */
    public static final String PROJECTOR_TYPE = "pockettts_spkenc";
    /** The reference waveform's sample rate (clip.cpp:2212). */
    public static final int SAMPLE_RATE = 24000;
    /** Model frames per second, the mimi frame rate the rows are emitted at. */
    public static final double FRAME_RATE = 12.5;
    /** Samples per model frame: {@code DOWNSAMPLE * 120}. */
    public static final int SAMPLES_PER_FRAME = 1920;
    /** Encoder frames folded into one model frame (clip.cpp:2216). */
    public static final int DOWNSAMPLE = 16;
    /** SEANet encoder strides, in encoder stage order (clip.cpp:2214). */
    public static final int[] STAGE_STRIDES = {4, 5, 6};
    /** The sliding attention window in encoder frames (clip.cpp:2218). */
    public static final int CONTEXT = 250;
    /** The transformer's RoPE base (clip.cpp:2219). */
    public static final float ROPE_THETA = 10000.0f;
    /** The longest reference the reference preprocessor keeps ({@code pockettts_max_spk_seconds}). */
    public static final int MAX_SECONDS = 30;

    /** Tensor names of the clip generator file. */
    public static final String SEANET = "a.seanet.";
    public static final String DOWNSAMPLE_CONV = "a.downsample.conv";
    public static final String SPEAKER_PROJ = "a.speaker_proj";
    public static final String BLOCK = "a.blk.";

    /** GELU's tanh-approximation constants ({@code ggml_gelu}). */
    private static final double GELU_C1 = 0.7978845608028654;
    private static final double GELU_C2 = 0.044715;

    /** One SEANet stage: the residual convolutions and the strided scale convolution. */
    private static final class Stage {
        final int res1K;
        final float[] res1W;
        final float[] res1B;
        final float[] res2W;
        final float[] res2B;
        final int scaleK;
        final float[] scaleW;
        final float[] scaleB;
        final int stride;
        final int in;
        final int out;

        Stage(int res1K, float[] res1W, float[] res1B, float[] res2W, float[] res2B, int scaleK, float[] scaleW,
              float[] scaleB, int stride, int in, int out) {
            this.res1K = res1K;
            this.res1W = res1W;
            this.res1B = res1B;
            this.res2W = res2W;
            this.res2B = res2B;
            this.scaleK = scaleK;
            this.scaleW = scaleW;
            this.scaleB = scaleB;
            this.stride = stride;
            this.in = in;
            this.out = out;
        }
    }

    /** One pre-norm transformer layer with layer scales on both residual paths. */
    private static final class Layer {
        final float[] ln1W;
        final float[] ln1B;
        final CpuTensorRef q;
        final CpuTensorRef k;
        final CpuTensorRef v;
        final CpuTensorRef out;
        final float[] ls1;
        final float[] ln2W;
        final float[] ln2B;
        final CpuTensorRef up;
        final CpuTensorRef down;
        final float[] ls2;

        Layer(float[] ln1W, float[] ln1B, CpuTensorRef q, CpuTensorRef k, CpuTensorRef v, CpuTensorRef out,
              float[] ls1, float[] ln2W, float[] ln2B, CpuTensorRef up, CpuTensorRef down, float[] ls2) {
            this.ln1W = ln1W;
            this.ln1B = ln1B;
            this.q = q;
            this.k = k;
            this.v = v;
            this.out = out;
            this.ls1 = ls1;
            this.ln2W = ln2W;
            this.ln2B = ln2B;
            this.up = up;
            this.down = down;
            this.ls2 = ls2;
        }
    }

    private final CpuWeights weights;
    private final CpuMatVec matVec;
    private final int width;
    private final int embedding;
    private final int nLayer;
    private final int nHead;
    private final int dHead;
    private final int nFf;
    private final float eps;
    private final int convInK;
    private final int convInIn;
    private final float[] convInW;
    private final float[] convInB;
    private final Stage[] stages;
    private final int convOutK;
    private final float[] convOutW;
    private final float[] convOutB;
    private final int downsampleK;
    private final float[] downsampleW;
    private final CpuTensorRef speakerProj;
    private final int latent;
    private final Layer[] layers;

    private PocketTtsSpeakerEncoder(Builder b) {
        this.weights = b.weights;
        this.matVec = b.weights.matVec();
        this.width = b.width;
        this.embedding = b.embedding;
        this.nLayer = b.nLayer;
        this.nHead = b.nHead;
        this.dHead = b.dHead;
        this.nFf = b.nFf;
        this.eps = b.eps;
        this.convInK = b.convInK;
        this.convInIn = b.convInIn;
        this.convInW = b.convInW;
        this.convInB = b.convInB;
        this.stages = b.stages;
        this.convOutK = b.convOutK;
        this.convOutW = b.convOutW;
        this.convOutB = b.convOutB;
        this.downsampleK = b.downsampleK;
        this.downsampleW = b.downsampleW;
        this.speakerProj = b.speakerProj;
        this.latent = b.latent;
        this.layers = b.layers;
    }

    /**
     * Resolves the speaker encoder of an opened pockettts generator file.
     *
     * @throws IllegalArgumentException when the file declares another projector type
     * @throws GgufFormatException      when a key or a tensor differs from the reference graph's geometry
     */
    public static PocketTtsSpeakerEncoder open(GgufKv kv, TensorCatalog tc, CpuWeights weights) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(weights, "weights");
        String type = kv.str(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE, null);
        if (!PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE + " is "
                    + (type == null ? "absent" : "'" + type + "'")
                    + ", a pockettts speaker encoder declares '" + PROJECTOR_TYPE + "'");
        }
        Builder b = new Builder();
        b.weights = weights;
        b.width = positive(kv, GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH);
        b.nLayer = positive(kv, GgufKeys.CLIP_AUDIO_BLOCK_COUNT);
        b.nHead = positive(kv, GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT);
        b.nFf = positive(kv, GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH);
        b.eps = kv.f32Required(GgufKeys.CLIP_AUDIO_ATTENTION_LAYER_NORM_EPSILON);
        if (b.width % b.nHead != 0) {
            throw new GgufFormatException("pockettts speaker encoder: " + b.nHead + " heads do not divide the width "
                    + b.width);
        }
        b.dHead = b.width / b.nHead;
        if (b.dHead % 2 != 0) {
            throw new GgufFormatException("pockettts speaker encoder: head dim " + b.dHead + " must be even for RoPE");
        }
        if (!(b.eps > 0f) || Float.isInfinite(b.eps)) {
            throw new GgufFormatException("pockettts speaker encoder: layer norm epsilon must be positive, got "
                    + b.eps);
        }

        TensorRef inRef = tc.require(SEANET + "conv_in.weight");
        b.convInK = PocketTtsTensors.convKernel(inRef);
        b.convInIn = PocketTtsTensors.convIn(inRef);
        if (b.convInIn != 1) {
            throw new GgufFormatException("pockettts speaker encoder: " + SEANET + "conv_in.weight takes "
                    + b.convInIn + " channels, the mono waveform is one");
        }
        int channels = PocketTtsTensors.convOut(inRef);
        if (channels != b.width) {
            throw new GgufFormatException("pockettts speaker encoder: " + SEANET + "conv_in.weight writes "
                    + channels + " channels, the transformer runs " + b.width);
        }
        b.convInW = weights.floats(SEANET + "conv_in.weight");
        b.convInB = weights.floats(SEANET + "conv_in.bias");
        PocketTtsTensors.vector(b.convInB, channels, SEANET + "conv_in.bias");

        b.stages = new Stage[STAGE_STRIDES.length];
        for (int i = 0; i < b.stages.length; i++) {
            String base = SEANET + "blk." + i + ".";
            TensorRef res1 = tc.require(base + "res_conv1.weight");
            TensorRef res2 = tc.require(base + "res_conv2.weight");
            TensorRef scale = tc.require(base + "scale_conv.weight");
            if (PocketTtsTensors.convIn(res1) != channels || PocketTtsTensors.convOut(res1) != channels) {
                throw new GgufFormatException("pockettts speaker encoder: " + base + "res_conv1.weight must run over"
                        + " the " + channels + " residual channels, found " + res1);
            }
            if (PocketTtsTensors.convKernel(res2) != 1 || PocketTtsTensors.convIn(res2) != channels
                    || PocketTtsTensors.convOut(res2) != channels) {
                throw new GgufFormatException("pockettts speaker encoder: " + base + "res_conv2.weight must be a"
                        + " pointwise convolution over the " + channels + " residual channels");
            }
            if (PocketTtsTensors.convIn(scale) != channels) {
                throw new GgufFormatException("pockettts speaker encoder: " + base + "scale_conv.weight takes "
                        + PocketTtsTensors.convIn(scale) + " channels, the stage runs " + channels);
            }
            int outChannels = PocketTtsTensors.convOut(scale);
            float[] scaleB = weights.floats(base + "scale_conv.bias");
            PocketTtsTensors.vector(scaleB, outChannels, base + "scale_conv.bias");
            float[] res1B = weights.floats(base + "res_conv1.bias");
            PocketTtsTensors.vector(res1B, channels, base + "res_conv1.bias");
            float[] res2B = weights.floats(base + "res_conv2.bias");
            PocketTtsTensors.vector(res2B, channels, base + "res_conv2.bias");
            b.stages[i] = new Stage(PocketTtsTensors.convKernel(res1), weights.floats(base + "res_conv1.weight"),
                    res1B, weights.floats(base + "res_conv2.weight"), res2B,
                    PocketTtsTensors.convKernel(scale), weights.floats(base + "scale_conv.weight"), scaleB,
                    STAGE_STRIDES[i], channels, outChannels);
            channels = outChannels;
        }
        if (channels != b.width) {
            throw new GgufFormatException("pockettts speaker encoder: the SEANet stages write " + channels
                    + " channels, the transformer runs " + b.width);
        }
        TensorRef outRef = tc.require(SEANET + "conv_out.weight");
        if (PocketTtsTensors.convIn(outRef) != b.width) {
            throw new GgufFormatException("pockettts speaker encoder: " + SEANET + "conv_out.weight takes "
                    + PocketTtsTensors.convIn(outRef) + " channels, the last stage writes " + b.width);
        }
        b.convOutK = PocketTtsTensors.convKernel(outRef);
        b.convOutW = weights.floats(SEANET + "conv_out.weight");
        b.convOutB = weights.floats(SEANET + "conv_out.bias");
        PocketTtsTensors.vector(b.convOutB, b.width, SEANET + "conv_out.bias");

        TensorRef downRef = tc.require(DOWNSAMPLE_CONV + ".weight");
        if (PocketTtsTensors.convIn(downRef) != b.width) {
            throw new GgufFormatException("pockettts speaker encoder: " + DOWNSAMPLE_CONV + ".weight takes "
                    + PocketTtsTensors.convIn(downRef) + " channels, the transformer writes " + b.width);
        }
        b.downsampleK = PocketTtsTensors.convKernel(downRef);
        b.downsampleW = weights.floats(DOWNSAMPLE_CONV + ".weight");

        TensorRef projRef = tc.require(SPEAKER_PROJ + ".weight");
        b.latent = PocketTtsTensors.dim(projRef, 0);
        b.embedding = PocketTtsTensors.convOut(projRef);
        if (PocketTtsTensors.convOut(downRef) != b.latent) {
            throw new GgufFormatException("pockettts speaker encoder: " + DOWNSAMPLE_CONV + ".weight writes "
                    + PocketTtsTensors.convOut(downRef) + " latents, " + SPEAKER_PROJ + " takes " + b.latent);
        }
        if (b.embedding <= 0) {
            throw new GgufFormatException("pockettts speaker encoder: " + SPEAKER_PROJ + ".weight writes "
                    + b.embedding + " embedding values");
        }
        PocketTtsTensors.matrix(weights.get(SPEAKER_PROJ + ".weight"), b.latent, b.embedding);
        if (PocketTtsTensors.convKernel(downRef) < DOWNSAMPLE) {
            throw new GgufFormatException("pockettts speaker encoder: " + DOWNSAMPLE_CONV + ".weight has kernel "
                    + b.downsampleK + ", shorter than its stride " + DOWNSAMPLE);
        }
        b.speakerProj = weights.get(SPEAKER_PROJ + ".weight");

        int tensorLayers = PocketTtsTensors.countIndexed(tc, BLOCK);
        if (tensorLayers != b.nLayer) {
            throw new GgufFormatException("pockettts speaker encoder: the file declares " + b.nLayer
                    + " encoder transformer layers but holds " + tensorLayers);
        }
        b.layers = new Layer[b.nLayer];
        for (int il = 0; il < b.nLayer; il++) {
            b.layers[il] = layer(weights, tc, il, b);
        }
        return new PocketTtsSpeakerEncoder(b);
    }

    private static Layer layer(CpuWeights weights, TensorCatalog tc, int il, Builder b) {
        String base = BLOCK + il + ".";
        float[] ln1W = weights.floats(base + "ln1.weight");
        float[] ln1B = weights.floats(base + "ln1.bias");
        float[] ln2W = weights.floats(base + "ln2.weight");
        float[] ln2B = weights.floats(base + "ln2.bias");
        PocketTtsTensors.vector(ln1W, b.width, base + "ln1.weight");
        PocketTtsTensors.vector(ln1B, b.width, base + "ln1.bias");
        PocketTtsTensors.vector(ln2W, b.width, base + "ln2.weight");
        PocketTtsTensors.vector(ln2B, b.width, base + "ln2.bias");
        CpuTensorRef q = weights.get(base + "attn_q.weight");
        CpuTensorRef k = weights.get(base + "attn_k.weight");
        CpuTensorRef v = weights.get(base + "attn_v.weight");
        CpuTensorRef o = weights.get(base + "attn_out.weight");
        CpuTensorRef up = weights.get(base + "ffn_up.weight");
        CpuTensorRef down = weights.get(base + "ffn_down.weight");
        int qDim = b.nHead * b.dHead;
        PocketTtsTensors.matrix(q, b.width, qDim);
        PocketTtsTensors.matrix(k, b.width, qDim);
        PocketTtsTensors.matrix(v, b.width, qDim);
        PocketTtsTensors.matrix(o, qDim, b.width);
        PocketTtsTensors.matrix(up, b.width, b.nFf);
        PocketTtsTensors.matrix(down, b.nFf, b.width);
        float[] ls1 = weights.floats(base + "ls1.weight");
        float[] ls2 = weights.floats(base + "ls2.weight");
        PocketTtsTensors.vector(ls1, b.width, base + "ls1.weight");
        PocketTtsTensors.vector(ls2, b.width, base + "ls2.weight");
        return new Layer(ln1W, ln1B, q, k, v, o, ls1, ln2W, ln2B, up, down, ls2);
    }

    private static int positive(GgufKv kv, String key) {
        int v = kv.u32Required(key);
        if (v <= 0) {
            throw new GgufFormatException("pockettts speaker encoder: " + key + " must be positive, got " + v);
        }
        return v;
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
        return embedding;
    }

    /** {@inheritDoc} */
    @Override
    public float[][] encode(AudioClip clip) {
        Objects.requireNonNull(clip, "clip");
        AudioClip mono = AudioResampler.resample(clip, SAMPLE_RATE);
        float[] samples = mono.samples();
        int n = Math.min(samples.length, MAX_SECONDS * SAMPLE_RATE);
        if (n == 0) {
            throw new IllegalArgumentException("the speaker reference is empty");
        }
        int nPadded = ((n + SAMPLES_PER_FRAME - 1) / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME;
        float[][] x = new float[nPadded][1];
        for (int i = 0; i < n; i++) {
            x[i][0] = samples[i];
        }
        x = conv(x, convInW, convInB, convInK, convInIn, width, 1, 1, false);
        for (Stage stage : stages) {
            x = residualUnit(x, stage);
            x = elu(x);
            x = conv(x, stage.scaleW, stage.scaleB, stage.scaleK, stage.in, stage.out, stage.stride, 1, false);
        }
        x = elu(x);
        x = conv(x, convOutW, convOutB, convOutK, width, width, 1, 1, false);
        x = transformer(x);
        x = conv(x, downsampleW, null, downsampleK, width, latent, DOWNSAMPLE, 1, true);
        float[] flat = flatten(x, latent);
        float[] projected = new float[x.length * embedding];
        matVec.matmul(speakerProj, flat, 0, latent, projected, 0, embedding, x.length);
        return reshape(projected, x.length, embedding);
    }

    /** {@code elu -> dilated causal conv -> elu -> pointwise conv -> residual}. */
    private float[][] residualUnit(float[][] x, Stage stage) {
        float[][] h = elu(x);
        h = conv(h, stage.res1W, stage.res1B, stage.res1K, stage.in, stage.out, 1, 1, false);
        h = elu(h);
        h = conv(h, stage.res2W, stage.res2B, 1, stage.out, stage.out, 1, 1, false);
        float[][] out = new float[x.length][stage.out];
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < stage.out; c++) {
                out[t][c] = x[t][c] + h[t][c];
            }
        }
        return out;
    }

    /** The causal pre-norm transformer with RoPE, a sliding-window mask and layer scales. */
    private float[][] transformer(float[][] input) {
        float[][] x = input;
        for (int il = 0; il < nLayer; il++) {
            x = layer(x, layers[il]);
        }
        return x;
    }

    private float[][] layer(float[][] x, Layer layer) {
        int n = x.length;
        int qDim = nHead * dHead;
        float[] normed = new float[n * width];
        for (int t = 0; t < n; t++) {
            layerNorm(x[t], layer.ln1W, layer.ln1B, normed, t * width);
        }
        float[] q = matVec(layer.q, normed, width, qDim);
        float[] k = matVec(layer.k, normed, width, qDim);
        float[] v = matVec(layer.v, normed, width, qDim);
        for (int t = 0; t < n; t++) {
            rope(q, t * qDim, nHead, dHead, t);
            rope(k, t * qDim, nHead, dHead, t);
        }
        float[] attn = attention(q, k, v, n);
        float[] projected = matVec(layer.out, attn, qDim, width);
        float[][] out = new float[n][width];
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < width; c++) {
                out[t][c] = x[t][c] + projected[t * width + c] * layer.ls1[c];
            }
        }
        float[] normed2 = new float[n * width];
        for (int t = 0; t < n; t++) {
            layerNorm(out[t], layer.ln2W, layer.ln2B, normed2, t * width);
        }
        float[] up = matVec(layer.up, normed2, width, nFf);
        for (int i = 0; i < up.length; i++) {
            up[i] = geluTanh(up[i]);
        }
        float[] down = matVec(layer.down, up, nFf, width);
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < width; c++) {
                out[t][c] += down[t * width + c] * layer.ls2[c];
            }
        }
        return out;
    }

    /** Causal attention where query {@code t} sees keys {@code max(0, t - CONTEXT + 1)..t}. */
    private float[] attention(float[] q, float[] k, float[] v, int n) {
        int qDim = nHead * dHead;
        float scale = (float) (1.0 / Math.sqrt(dHead));
        float[] out = new float[n * qDim];
        float[] scores = new float[n];
        for (int t = 0; t < n; t++) {
            int first = Math.max(0, t - CONTEXT + 1);
            for (int h = 0; h < nHead; h++) {
                int off = t * qDim + h * dHead;
                float max = Float.NEGATIVE_INFINITY;
                for (int s = first; s <= t; s++) {
                    float dot = 0f;
                    int keyOff = s * qDim + h * dHead;
                    for (int d = 0; d < dHead; d++) {
                        dot += q[off + d] * k[keyOff + d];
                    }
                    scores[s] = dot * scale;
                    if (scores[s] > max) {
                        max = scores[s];
                    }
                }
                double sum = 0.0;
                for (int s = first; s <= t; s++) {
                    double e = Math.exp(scores[s] - max);
                    scores[s] = (float) e;
                    sum += e;
                }
                float inv = sum > 0.0 ? (float) (1.0 / sum) : 0f;
                for (int d = 0; d < dHead; d++) {
                    float acc = 0f;
                    for (int s = first; s <= t; s++) {
                        acc += scores[s] * v[s * qDim + h * dHead + d];
                    }
                    out[off + d] = acc * inv;
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ convolution primitives

    /**
     * A causal convolution over frame-major {@code x[T][in]}: the left context is {@code k - stride} frames (zeros,
     * or a repeat of the first frame for the downsample convolution) and the tail is zero padding, so the output has
     * {@code ceil(T / stride)} rows.
     */
    private static float[][] conv(float[][] x, float[] w, float[] b, int k, int in, int out, int stride,
                                  int dilation, boolean replicateFirst) {
        int frames = x.length;
        int span = (k - 1) * dilation + 1;
        int left = Math.max(0, span - stride);
        int outFrames = (frames + stride - 1) / stride;
        float[][] y = new float[outFrames][out];
        for (int t = 0; t < outFrames; t++) {
            for (int oc = 0; oc < out; oc++) {
                float acc = b == null ? 0f : b[oc];
                for (int j = 0; j < k; j++) {
                    int at = t * stride + j * dilation;
                    int src = at - left;
                    if (src < 0 || src >= frames) {
                        if (replicateFirst && src < 0) {
                            for (int ic = 0; ic < in; ic++) {
                                acc += w[j + k * (ic + in * oc)] * x[0][ic];
                            }
                        }
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

    private static float[][] elu(float[][] x) {
        float[][] y = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            float[] row = new float[x[i].length];
            for (int c = 0; c < row.length; c++) {
                row[c] = x[i][c] > 0f ? x[i][c] : (float) Math.expm1(x[i][c]);
            }
            y[i] = row;
        }
        return y;
    }

    private static float geluTanh(float x) {
        double inner = GELU_C1 * (x + GELU_C2 * x * x * x);
        return (float) (0.5 * x * (1.0 + Math.tanh(inner)));
    }

    private void layerNorm(float[] x, float[] weight, float[] bias, float[] out, int off) {
        int n = x.length;
        double mean = 0.0;
        for (float value : x) {
            mean += value;
        }
        mean /= n;
        double var = 0.0;
        for (float value : x) {
            double d = value - mean;
            var += d * d;
        }
        var /= n;
        float scale = (float) (1.0 / Math.sqrt(var + eps));
        for (int i = 0; i < n; i++) {
            out[off + i] = (float) ((x[i] - mean) * scale) * weight[i] + bias[i];
        }
    }

    /** NORMAL RoPE of one head: adjacent pairs rotated by {@code pos * theta^(-2i/dHead)}. */
    private static void rope(float[] x, int off, int nHead, int dHead, int pos) {
        int half = dHead / 2;
        for (int h = 0; h < nHead; h++) {
            int base = off + h * dHead;
            for (int i = 0; i < half; i++) {
                double angle = pos * Math.pow(ROPE_THETA, -2.0 * i / dHead);
                float cos = (float) Math.cos(angle);
                float sin = (float) Math.sin(angle);
                float x0 = x[base + 2 * i];
                float x1 = x[base + 2 * i + 1];
                x[base + 2 * i] = x0 * cos - x1 * sin;
                x[base + 2 * i + 1] = x0 * sin + x1 * cos;
            }
        }
    }

    private float[] matVec(CpuTensorRef w, float[] x, int in, int out) {
        float[] y = new float[x.length / in * out];
        matVec.matmul(w, x, 0, in, y, 0, out, x.length / in);
        return y;
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

    private static final class Builder {
        CpuWeights weights;
        int width;
        int embedding;
        int nLayer;
        int nHead;
        int dHead;
        int nFf;
        float eps;
        int convInK;
        int convInIn;
        float[] convInW;
        float[] convInB;
        Stage[] stages;
        int convOutK;
        float[] convOutW;
        float[] convOutB;
        int downsampleK;
        float[] downsampleW;
        CpuTensorRef speakerProj;
        int latent;
        Layer[] layers;
    }

    @Override
    public String toString() {
        return "PocketTtsSpeakerEncoder{width=" + width + ", layers=" + nLayer + ", latent=" + latent + '}';
    }
}
