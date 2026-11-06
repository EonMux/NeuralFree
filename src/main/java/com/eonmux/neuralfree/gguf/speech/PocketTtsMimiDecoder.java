package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The pockettts streaming mimi decoder of llama.cpp's {@code clip_graph_pockettts_gen} GEN_WAV graph
 * (tools/mtmd/models/pockettts-gen.cpp:100-294, pockettts-seanet.cpp): a window of continuous latents becomes PCM.
 *
 * <ol>
 *   <li>the latents are denormalized by the checkpoint's mean and standard deviation and projected from the latent
 *       width to the decoder width ({@code DummyQuantizer});</li>
 *   <li>a depthwise transposed convolution takes the model frame rate (12.5 Hz) to the mimi encoder frame rate;</li>
 *   <li>a sliding-window transformer (250 cached frames, banded causal attention, continuing RoPE positions)
 *       pre-processes the upsampled frames;</li>
 *   <li>the causal SEANet stack mirrors the encoder: a causal input convolution, three transposed-convolution and
 *       residual stages, and a causal output convolution, every convolution taking its left context from a state
 *       slot.</li>
 * </ol>
 *
 * <p>The reference hardcodes the mimi geometry (clip.cpp:2210-2222): 24 kHz, one latent per 16 encoder frames,
 * decoder strides 6/5/4, a 250-frame attention window, 10000 RoPE, and one window of 8 latents per decoder batch.
 * Tensors use the clip generator names of tools/mtmd/clip-impl.h:311-317 in ggml dimension order. Immutable over
 * shared host weights; {@link #newSession()} carries the streaming state.
 */
public final class PocketTtsMimiDecoder {

    /** The waveform's sample rate (clip.cpp:2211). */
    public static final int SAMPLE_RATE = 24000;

    /** Model frames to encoder frames (clip.cpp:2216). */
    public static final int DOWNSAMPLE = 16;
    /** SEANet decoder strides, in decoder stage order (the encoder's {4, 5, 6} reversed, clip.cpp:2214). */
    public static final int[] STAGE_STRIDES = {6, 5, 4};
    /** The sliding attention window in encoder frames (clip.cpp:2218). */
    public static final int CONTEXT = 250;
    /** Latents decoded per GEN_WAV call (mtmd-helper-gen.cpp:1076). */
    public static final int WINDOW_FRAMES = 8;
    /** The transformer's RoPE base (clip.cpp:2219). */
    public static final float ROPE_THETA = 10000.0f;

    /** ConvNeXt LayerNorm epsilon of the DAC-style blocks ({@code ggml_norm(..., 1e-6f)}). */
    static final float BLOCK_EPS = 1e-6f;

    /** Tensor base names of tools/mtmd/clip-impl.h:311-317. */
    public static final String QUANT_OUT = "a.gen.wav.quant_out";
    public static final String UPSAMPLE = "a.gen.wav.upsample";
    public static final String SEANET = "a.gen.wav.seanet.";
    public static final String TFM = "a.gen.wav.tfm.";

    /**
     * What the speech pipeline decodes through: a stateful window decoder. Implemented by the host
     * {@link PocketTtsMimiSession} and by the device executor of a GPU engine.
     */
    public interface Decoder extends AutoCloseable {
        /** {@link PocketTtsMimiSession#decode(float[][])}. */
        float[] decode(float[][] frames);

        /** {@link #decode(float[][])} as an {@link com.eonmux.neuralfree.gguf.audio.AudioClip}. */
        default com.eonmux.neuralfree.gguf.audio.AudioClip decodeClip(float[][] frames) {
            return new com.eonmux.neuralfree.gguf.audio.AudioClip(decode(frames), sampleRate());
        }

        /** The RoPE position of the next call's first upsampled frame. */
        int position();

        /** The decoded waveform's sample rate. */
        int sampleRate();

        /** Releases the decoder's state; idempotent. */
        @Override
        void close();
    }

    /** The GGUF name of one transformer-layer tensor. */
    static String layerTensor(int index, String part) {
        return TFM + "blk." + index + "." + part;
    }

    /** The GGUF name of one SEANet-stage tensor. */
    static String stageTensor(int index, String part) {
        return SEANET + "blk." + index + "." + part;
    }

    /** One SEANet decoder stage. */
    static final class Stage {
        final float[] resConv1W;
        final float[] resConv1B;
        final float[] resConv2W;
        final float[] resConv2B;
        final float[] scaleConvW;
        final float[] scaleConvB;
        final int resConv1K;
        final int channels;
        final int scaleInChannels;
        final int scaleConvK;
        final int stride;
        final int outChannels;

        Stage(float[] resConv1W, float[] resConv1B, float[] resConv2W, float[] resConv2B, float[] scaleConvW,
              float[] scaleConvB, int resConv1K, int channels, int scaleInChannels, int scaleConvK, int stride,
              int outChannels) {
            this.resConv1W = resConv1W;
            this.resConv1B = resConv1B;
            this.resConv2W = resConv2W;
            this.resConv2B = resConv2B;
            this.scaleConvW = scaleConvW;
            this.scaleConvB = scaleConvB;
            this.resConv1K = resConv1K;
            this.channels = channels;
            this.scaleInChannels = scaleInChannels;
            this.scaleConvK = scaleConvK;
            this.stride = stride;
            this.outChannels = outChannels;
        }
    }

    /** One sliding-window transformer layer. */
    static final class TransformerLayer {
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

        TransformerLayer(float[] ln1W, float[] ln1B, CpuTensorRef q, CpuTensorRef k, CpuTensorRef v,
                        CpuTensorRef out, float[] ls1, float[] ln2W, float[] ln2B, CpuTensorRef up,
                        CpuTensorRef down, float[] ls2) {            this.ln1W = ln1W;
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

    private final int latent;
    private final int width;
    private final CpuTensorRef quantOut;
    private final float[] embMean;
    private final float[] embStd;
    private final float[] upsampleW;
    private final int upsampleK;
    private final float[] convInW;
    private final float[] convInB;
    private final int convInK;
    private final int convInChannels;
    private final Stage[] stages;
    private final float[] convOutW;
    private final float[] convOutB;
    private final int convOutK;
    private final int convOutIn;
    private final int nLayer;
    private final int nHead;
    private final int nHeadKv;
    private final int dHead;
    private final int nFf;
    private final float eps;
    private final TransformerLayer[] layers;
    private final int hop;
    private final CpuWeights weights;

    private PocketTtsMimiDecoder(Builder b) {
        this.latent = b.latent;
        this.width = b.width;
        this.quantOut = b.quantOut;
        this.embMean = b.embMean;
        this.embStd = b.embStd;
        this.upsampleW = b.upsampleW;
        this.upsampleK = b.upsampleK;
        this.convInW = b.convInW;
        this.convInB = b.convInB;
        this.convInK = b.convInK;
        this.convInChannels = b.convInChannels;
        this.stages = b.stages;
        this.convOutW = b.convOutW;
        this.convOutB = b.convOutB;
        this.convOutK = b.convOutK;
        this.convOutIn = b.convOutIn;
        this.nLayer = b.nLayer;
        this.nHead = b.nHead;
        this.nHeadKv = b.nHeadKv;
        this.dHead = b.dHead;
        this.nFf = b.nFf;
        this.eps = b.eps;
        this.layers = b.layers;
        this.hop = b.hop;
        this.weights = b.weights;
    }

    /**
     * Resolves the mimi decoder of an opened pockettts generator file.
     *
     * @throws IllegalArgumentException when the file is another generator family
     * @throws GgufFormatException      when a key or a tensor shape differs from the reference graph's
     */
    public static PocketTtsMimiDecoder open(GgufKv kv, TensorCatalog tc, CpuWeights weights) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(weights, "weights");
        String type = kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!PocketTtsFlow.PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is "
                    + (type == null ? "absent" : "'" + type + "'") + ", a pockettts generator declares '"
                    + PocketTtsFlow.PROJECTOR_TYPE + "'");
        }
        Builder b = new Builder();
        b.weights = weights;
        b.width = positive(kv, GgufKeys.CLIP_GEN_AUDIO_EMBEDDING_LENGTH);
        b.nLayer = positive(kv, GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT);
        b.nHead = positive(kv, GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT);
        b.nHeadKv = kv.u32(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_HEAD_COUNT_KV, b.nHead);
        b.nFf = positive(kv, GgufKeys.CLIP_GEN_AUDIO_FEED_FORWARD_LENGTH);
        b.eps = kv.f32Required(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON);
        if (b.nHeadKv <= 0 || b.nHead % b.nHeadKv != 0 || b.width % b.nHead != 0) {
            throw new GgufFormatException("pockettts mimi: heads " + b.nHead + "/" + b.nHeadKv
                    + " do not divide the transformer width " + b.width);
        }
        b.dHead = b.width / b.nHead;
        if (b.dHead % 2 != 0) {
            throw new GgufFormatException("pockettts mimi: head dim " + b.dHead + " must be even for RoPE");
        }
        if (!(b.eps > 0f) || Float.isInfinite(b.eps)) {
            throw new GgufFormatException("pockettts mimi: layer norm epsilon must be positive, got " + b.eps);
        }

        TensorRef quantRef = tc.require(QUANT_OUT + ".weight");
        b.latent = PocketTtsTensors.dim(quantRef, 0);
        if (PocketTtsTensors.dim(quantRef, 1) != b.width) {
            throw new GgufFormatException("pockettts mimi: " + QUANT_OUT + ".weight writes "
                    + PocketTtsTensors.dim(quantRef, 1) + " channels, the transformer runs " + b.width);
        }
        b.quantOut = weights.get(QUANT_OUT + ".weight");
        b.embMean = weights.floats("a.gen.emb_mean");
        b.embStd = weights.floats("a.gen.emb_std");
        PocketTtsTensors.vector(b.embMean, b.latent, "a.gen.emb_mean");
        PocketTtsTensors.vector(b.embStd, b.latent, "a.gen.emb_std");

        TensorRef upRef = tc.require(UPSAMPLE + ".weight");
        if (upRef.rank() != 3 || upRef.ne(1) != 1 || upRef.ne(2) != b.width) {
            throw new GgufFormatException("pockettts mimi: " + UPSAMPLE + ".weight must be a depthwise transposed"
                    + " convolution over " + b.width + " channels, found " + upRef);
        }
        b.upsampleK = PocketTtsTensors.dim(upRef, 0);
        if (b.upsampleK < DOWNSAMPLE || b.upsampleK > 2 * DOWNSAMPLE) {
            throw new GgufFormatException("pockettts mimi: " + UPSAMPLE + ".weight has kernel " + b.upsampleK
                    + "; the streaming " + DOWNSAMPLE + "-frame upsample keeps one overlap tail, so its kernel must"
                    + " lie in [" + DOWNSAMPLE + ", " + (2 * DOWNSAMPLE) + "]");
        }
        b.upsampleW = weights.floats(UPSAMPLE + ".weight");
        PocketTtsTensors.vector(b.upsampleW, b.upsampleK * b.width, UPSAMPLE + ".weight");

        TensorRef inRef = tc.require(SEANET + "conv_in.weight");
        if (PocketTtsTensors.convIn(inRef) != b.width) {
            throw new GgufFormatException("pockettts mimi: " + SEANET + "conv_in.weight takes "
                    + PocketTtsTensors.convIn(inRef) + " channels, the upsample writes " + b.width);
        }
        b.convInK = PocketTtsTensors.convKernel(inRef);
        b.convInChannels = PocketTtsTensors.convOut(inRef);
        b.convInW = weights.floats(SEANET + "conv_in.weight");
        b.convInB = weights.floats(SEANET + "conv_in.bias");
        PocketTtsTensors.vector(b.convInB, b.convInChannels, SEANET + "conv_in.bias");

        b.stages = stages(weights, tc, b);
        TensorRef outRef = tc.require(SEANET + "conv_out.weight");
        b.convOutK = PocketTtsTensors.convKernel(outRef);
        b.convOutIn = PocketTtsTensors.convIn(outRef);
        if (PocketTtsTensors.convOut(outRef) != 1) {
            throw new GgufFormatException("pockettts mimi: " + SEANET + "conv_out.weight writes "
                    + PocketTtsTensors.convOut(outRef) + " channels, the waveform is mono");
        }
        int lastChannels = b.stages[b.stages.length - 1].outChannels;
        if (b.convOutIn != lastChannels) {
            throw new GgufFormatException("pockettts mimi: " + SEANET + "conv_out.weight takes " + b.convOutIn
                    + " channels, the last stage writes " + lastChannels);
        }
        b.convOutW = weights.floats(SEANET + "conv_out.weight");
        b.convOutB = weights.floats(SEANET + "conv_out.bias");
        PocketTtsTensors.vector(b.convOutB, 1, SEANET + "conv_out.bias");

        int nTensorLayer = PocketTtsTensors.countIndexed(tc, TFM + "blk.");
        if (nTensorLayer != b.nLayer) {
            throw new GgufFormatException("pockettts mimi: the file declares " + b.nLayer + " decoder transformer"
                    + " layers but holds " + nTensorLayer);
        }
        b.layers = new TransformerLayer[b.nLayer];
        for (int il = 0; il < b.nLayer; il++) {
            b.layers[il] = layer(weights, tc, il, b.width, b.nFf, b.nHead, b.nHeadKv, b.dHead);
        }
        b.hop = DOWNSAMPLE;
        for (int stride : STAGE_STRIDES) {
            b.hop *= stride;
        }
        return new PocketTtsMimiDecoder(b);
    }

    private static Stage[] stages(CpuWeights weights, TensorCatalog tc, Builder b) {
        Stage[] out = new Stage[STAGE_STRIDES.length];
        int in = b.convInChannels;
        for (int i = 0; i < out.length; i++) {
            String base = SEANET + "blk." + i + ".";
            TensorRef scaleRef = tc.require(base + "scale_conv.weight");
            if (scaleRef.rank() != 3 || scaleRef.ne(2) != in) {
                throw new GgufFormatException("pockettts mimi: " + base + "scale_conv.weight must be a grouped"
                        + " transposed convolution from " + in + " channels, found " + scaleRef);
            }
            int kernel = PocketTtsTensors.dim(scaleRef, 0);
            boolean depthwise = scaleRef.ne(1) == 1;
            int outChannels = depthwise ? PocketTtsTensors.dim(scaleRef, 2) : PocketTtsTensors.dim(scaleRef, 1);
            int stride = STAGE_STRIDES[i];
            if (kernel != 2 * stride) {
                throw new GgufFormatException("pockettts mimi: " + base + "scale_conv.weight has kernel " + kernel
                        + ", the reference stage upsamples " + stride + "x with a kernel of " + (2 * stride));
            }
            float[] scaleW = weights.floats(base + "scale_conv.weight");
            float[] scaleB = weights.floats(base + "scale_conv.bias");
            PocketTtsTensors.vector(scaleB, outChannels, base + "scale_conv.bias");

            TensorRef res1Ref = tc.require(base + "res_conv1.weight");
            if (PocketTtsTensors.convIn(res1Ref) != outChannels || PocketTtsTensors.convOut(res1Ref) != outChannels) {
                throw new GgufFormatException("pockettts mimi: " + base + "res_conv1.weight must run over the "
                        + outChannels + " residual channels, found " + res1Ref);
            }
            int res1K = PocketTtsTensors.convKernel(res1Ref);
            float[] res1W = weights.floats(base + "res_conv1.weight");
            float[] res1B = weights.floats(base + "res_conv1.bias");
            PocketTtsTensors.vector(res1B, outChannels, base + "res_conv1.bias");
            TensorRef res2Ref = tc.require(base + "res_conv2.weight");
            if (PocketTtsTensors.convKernel(res2Ref) != 1 || PocketTtsTensors.convIn(res2Ref) != outChannels
                    || PocketTtsTensors.convOut(res2Ref) != outChannels) {
                throw new GgufFormatException("pockettts mimi: " + base + "res_conv2.weight must be a pointwise"
                        + " convolution over the " + outChannels + " residual channels");
            }
            float[] res2W = weights.floats(base + "res_conv2.weight");
            float[] res2B = weights.floats(base + "res_conv2.bias");
            PocketTtsTensors.vector(res2B, outChannels, base + "res_conv2.bias");
            out[i] = new Stage(res1W, res1B, res2W, res2B, scaleW, scaleB, res1K, outChannels, in, kernel, stride,
                    outChannels);
            in = outChannels;
        }
        return out;
    }

    private static TransformerLayer layer(CpuWeights weights, TensorCatalog tc, int il, int width, int nFf,
                                         int nHead, int nHeadKv, int dHead) {
        String base = TFM + "blk." + il + ".";
        float[] ln1W = weights.floats(base + "ln1.weight");
        float[] ln1B = weights.floats(base + "ln1.bias");
        float[] ln2W = weights.floats(base + "ln2.weight");
        float[] ln2B = weights.floats(base + "ln2.bias");
        PocketTtsTensors.vector(ln1W, width, base + "ln1.weight");
        PocketTtsTensors.vector(ln1B, width, base + "ln1.bias");
        PocketTtsTensors.vector(ln2W, width, base + "ln2.weight");
        PocketTtsTensors.vector(ln2B, width, base + "ln2.bias");
        CpuTensorRef q = weights.get(base + "attn_q.weight");
        CpuTensorRef k = weights.get(base + "attn_k.weight");
        CpuTensorRef v = weights.get(base + "attn_v.weight");
        CpuTensorRef o = weights.get(base + "attn_out.weight");
        CpuTensorRef up = weights.get(base + "ffn_up.weight");
        CpuTensorRef down = weights.get(base + "ffn_down.weight");
        PocketTtsTensors.matrix(q, width, nHead * dHead);
        PocketTtsTensors.matrix(k, width, nHeadKv * dHead);
        PocketTtsTensors.matrix(v, width, nHeadKv * dHead);
        PocketTtsTensors.matrix(o, nHead * dHead, width);
        PocketTtsTensors.matrix(up, width, nFf);
        PocketTtsTensors.matrix(down, nFf, width);
        float[] ls1 = weights.floats(base + "ls1.weight");
        float[] ls2 = weights.floats(base + "ls2.weight");
        PocketTtsTensors.vector(ls1, width, base + "ls1.weight");
        PocketTtsTensors.vector(ls2, width, base + "ls2.weight");
        return new TransformerLayer(ln1W, ln1B, q, k, v, o, ls1, ln2W, ln2B, up, down, ls2);
    }

    private static int positive(GgufKv kv, String key) {
        int v = kv.u32Required(key);
        if (v <= 0) {
            throw new GgufFormatException("pockettts mimi: " + key + " must be positive, got " + v);
        }
        return v;
    }

    /** A fresh decoder state: empty attention windows and zeroed convolution left contexts. */
    public PocketTtsMimiSession newSession() {
        return new PocketTtsMimiSession(this);
    }

    /** A fresh decoder over {@code device}, or a host {@link #newSession()} when {@code device} is null. */
    public Decoder newDecoder(SpeechDeviceWeights device) {
        return device == null ? newSession() : new GpuPocketTtsMimiDecoder(this, device);
    }

    // ------------------------------------------------------------------ accessors

    /** The latent width the decoder takes per frame (32 in the reference). */
    public int latent() { return latent; }
    /** The decoder width the quantizer writes (512 in the reference). */
    public int width() { return width; }
    /** Decoder transformer layers. */
    public int nLayer() { return nLayer; }
    /** Query heads of the decoder transformer. */
    public int nHead() { return nHead; }
    /** Key/value heads of the decoder transformer. */
    public int nHeadKv() { return nHeadKv; }
    /** Head dim of the decoder transformer. */
    int dHead() { return dHead; }
    /** FFN width of the decoder transformer. */
    int nFf() { return nFf; }
    /** The decoder transformer's LayerNorm epsilon. */
    float eps() { return eps; }
    /** The sliding attention window in encoder frames. */
    public int context() { return CONTEXT; }
    /** Samples per latent frame. */
    public int hop() { return hop; }
    /** The waveform's sample rate. */
    public int sampleRate() { return SAMPLE_RATE; }
    /** Latents one decode call takes at most. */
    public int windowFrames() { return WINDOW_FRAMES; }

    /** The persisted state slot names, in the reference's {@code list_pockettts_state_slots} order. */
    public List<String> stateSlots() {
        List<String> names = new ArrayList<>();
        names.add("tfm_pos");
        for (int il = 0; il < nLayer; il++) {
            names.add("tfm_k_" + il);
            names.add("tfm_v_" + il);
        }
        names.add("up");
        names.add("dec_in");
        for (int i = 0; i < stages.length; i++) {
            names.add("dec_up_" + i);
            names.add("dec_res_" + i);
        }
        names.add("dec_out");
        return Collections.unmodifiableList(names);
    }

    CpuWeights weights() { return weights; }
    CpuTensorRef quantOut() { return quantOut; }
    float[] embMean() { return embMean; }
    float[] embStd() { return embStd; }
    float[] upsampleW() { return upsampleW; }
    int upsampleK() { return upsampleK; }
    float[] convInW() { return convInW; }
    float[] convInB() { return convInB; }
    int convInK() { return convInK; }
    int convInChannels() { return convInChannels; }
    Stage[] stages() { return stages; }
    float[] convOutW() { return convOutW; }
    float[] convOutB() { return convOutB; }
    int convOutK() { return convOutK; }
    TransformerLayer[] layers() { return layers; }

    private static final class Builder {
        int latent;
        int width;
        CpuTensorRef quantOut;
        float[] embMean;
        float[] embStd;
        float[] upsampleW;
        int upsampleK;
        float[] convInW;
        float[] convInB;
        int convInK;
        int convInChannels;
        Stage[] stages;
        float[] convOutW;
        float[] convOutB;
        int convOutK;
        int convOutIn;
        int nLayer;
        int nHead;
        int nHeadKv;
        int dHead;
        int nFf;
        float eps;
        TransformerLayer[] layers;
        int hop;
        CpuWeights weights;
    }

    @Override
    public String toString() {
        return "PocketTtsMimiDecoder{latent=" + latent + ", width=" + width + ", layers=" + nLayer + ", stages="
                + stages.length + ", hop=" + hop + '}';
    }
}
