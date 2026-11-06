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
 * The qwen3tts <b>code2wav</b> DAC decoder of llama.cpp's {@code clip_graph_qwen3tts_gen::code2wav}
 * (tools/mtmd/models/qwen3tts-gen.cpp:400-700): a window of RVQ code frames becomes raw PCM.
 *
 * <ol>
 *   <li>the RVQ quantizer sums the 16 codebook embeddings per frame, projects the semantic and acoustic groups
 *       separately and adds them ({@code quant_decode});</li>
 *   <li>a causal convolution widens the 512-value frames to 1024 values, the sliding-window pre-transformer (72
 *       frames, banded causal attention with a persisted key/value prefix and RoPE position counter) runs 8 layers
 *       and the output projection narrows them back;</li>
 *   <li>two causal ConvTranspose1d + ConvNeXt upsample stages (2x each) and four DAC blocks (SnakeBeta, causal
 *       ConvTranspose1d, three dilated residual units) expand the frame rate to samples;</li>
 *   <li>a final SnakeBeta and convolution produce the mono waveform, clamped to {@code [-1, 1]}.</li>
 * </ol>
 *
 * <p>The stateful convolutions and the attention prefix persist between windows ({@link Qwen3TtsCode2WavSession}
 * holds the slots {@code list_c2w_state_slots} names), so a sequence of windows produces the same PCM as one
 * pass. Tensors use the clip generator names of tools/mtmd/clip-impl.h:259-286 in ggml dimension order,
 * {@code ne[0]} first; convolution weights are read as F32 host arrays, the attention and projection weights stay
 * as {@link CpuTensorRef} for {@link com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec}. Immutable geometry over
 * shared host weights. The reference hardcodes the transformer head counts (16/16), the 10000 RoPE base and the
 * 72-frame window; {@code clip.gen.wav.attention.head_count}, {@code ...head_count_kv}, {@code clip.gen.wav.rope.freq_base}
 * and {@code clip.gen.wav.window} override them when a file declares a different decoder geometry.
 */
public final class Qwen3TtsCode2Wav {

    /** The sample rate of the decoded waveform (mtmd.cpp:2234-2275, qwen3tts 24 kHz). */
    public static final int SAMPLE_RATE = 24000;

    /**
     * What the speech pipeline decodes through: a stateful window decoder. Implemented by the host
     * {@link Qwen3TtsCode2WavSession} and by the device executor of a GPU engine.
     */
    public interface Decoder extends AutoCloseable {
        /** {@link Qwen3TtsCode2WavSession#decode(int[][])}. */
        float[] decode(int[][] frames);

        /** {@link #decode(int[][])} as an {@link com.eonmux.neuralfree.gguf.audio.AudioClip}. */
        default com.eonmux.neuralfree.gguf.audio.AudioClip decodeClip(int[][] frames) {
            return new com.eonmux.neuralfree.gguf.audio.AudioClip(decode(frames), sampleRate());
        }

        /** The decoded waveform's sample rate. */
        int sampleRate();

        /** The persisted RoPE position of the next window's first frame. */
        float position();

        /** Releases the decoder's state; idempotent. */
        @Override
        void close();
    }
    /** The reference decoder's sliding window in frames (clip.cpp:2205, speech_tokenizer config). */
    public static final int DEFAULT_WINDOW = 72;
    /** The reference transformer head count (clip.cpp:2198-2199). */
    public static final int DEFAULT_HEAD_COUNT = 16;
    /** The reference transformer RoPE base (clip.cpp:2201). */
    public static final float DEFAULT_ROPE_THETA = 10000.0f;

    /** Optional geometry keys of an override (llama.cpp hardcodes the values). */
    public static final String HEAD_COUNT_KEY = "clip.gen.wav.attention.head_count";
    public static final String HEAD_COUNT_KV_KEY = "clip.gen.wav.attention.head_count_kv";
    public static final String ROPE_THETA_KEY = "clip.gen.wav.rope.freq_base";
    public static final String WINDOW_KEY = "clip.gen.wav.window";

    /** Tensor base names of tools/mtmd/clip-impl.h:261-286. */
    public static final String QUANT_FIRST_IN = "a.gen.wav.quant.first.in_proj";
    public static final String QUANT_FIRST_OUT = "a.gen.wav.quant.first.out_proj";
    public static final String QUANT_FIRST_CB = "a.gen.wav.quant.first.codebook";
    public static final String QUANT_REST_IN = "a.gen.wav.quant.rest.in_proj";
    public static final String QUANT_REST_OUT = "a.gen.wav.quant.rest.out_proj";
    public static final String QUANT_REST_CB = "a.gen.wav.quant.rest.codebook";
    public static final String PRE_CONV = "a.gen.wav.pre_conv";
    public static final String TFM_IN_PROJ = "a.gen.wav.tfm.in_proj";
    public static final String TFM_OUT_PROJ = "a.gen.wav.tfm.out_proj";
    public static final String TFM_OUTPUT_NORM = "a.gen.wav.tfm.output_norm";
    public static final String UP = "a.gen.wav.up.blk.";
    public static final String DAC_ENTRY = "a.gen.wav.dac.entry";
    public static final String DAC = "a.gen.wav.dac.blk.";
    public static final String DAC_POST_SNAKE = "a.gen.wav.dac.post_snake";
    public static final String DAC_POST_CONV = "a.gen.wav.dac.post_conv";

    /** The DAC residual unit dilations (qwen3tts-gen.cpp:580). */
    public static final int[] DAC_DILATIONS = {1, 3, 9};

    /** Tensor part names of one pre-transformer layer. */
    private static final String[] TFM_PARTS = {"ln1.weight", "ln2.weight", "attn_q.weight", "attn_k.weight",
            "attn_v.weight", "attn_out.weight", "ls1.weight", "ls2.weight", "ffn_gate.weight", "ffn_up.weight",
            "ffn_down.weight"};

    /** One sliding-window pre-transformer layer. */
    static final class TfmLayer {
        final float[] ln1;
        final CpuTensorRef q;
        final CpuTensorRef k;
        final CpuTensorRef v;
        final CpuTensorRef out;
        final float[] ls1;
        final float[] ln2;
        final CpuTensorRef gate;
        final CpuTensorRef up;
        final CpuTensorRef down;
        final float[] ls2;

        TfmLayer(float[] ln1, CpuTensorRef q, CpuTensorRef k, CpuTensorRef v, CpuTensorRef out, float[] ls1,
                 float[] ln2, CpuTensorRef gate, CpuTensorRef up, CpuTensorRef down, float[] ls2) {
            this.ln1 = ln1;
            this.q = q;
            this.k = k;
            this.v = v;
            this.out = out;
            this.ls1 = ls1;
            this.ln2 = ln2;
            this.gate = gate;
            this.up = up;
            this.down = down;
            this.ls2 = ls2;
        }
    }

    /** One ConvNeXt upsample block (ConvTranspose1d carries the stride). */
    static final class UpBlock {
        final float[] convW;
        final float[] convB;
        final float[] dwconvW;
        final float[] dwconvB;
        final float[] normW;
        final float[] normB;
        final CpuTensorRef pw1;
        final float[] pw1B;
        final CpuTensorRef pw2;
        final float[] pw2B;
        final float[] gamma;
        final int inChannels;
        final int outChannels;
        final int convK;
        final int dwK;

        UpBlock(float[] convW, float[] convB, float[] dwconvW, float[] dwconvB, float[] normW, float[] normB,
                CpuTensorRef pw1, float[] pw1B, CpuTensorRef pw2, float[] pw2B, float[] gamma, int inChannels,
                int outChannels, int convK, int dwK) {
            this.convW = convW;
            this.convB = convB;
            this.dwconvW = dwconvW;
            this.dwconvB = dwconvB;
            this.normW = normW;
            this.normB = normB;
            this.pw1 = pw1;
            this.pw1B = pw1B;
            this.pw2 = pw2;
            this.pw2B = pw2B;
            this.gamma = gamma;
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.convK = convK;
            this.dwK = dwK;
        }
    }

    /** One DAC residual unit. */
    static final class DacRes {
        final float[] act1Alpha;
        final float[] act1Beta;
        final float[] conv1W;
        final float[] conv1B;
        final float[] act2Alpha;
        final float[] act2Beta;
        final float[] conv2W;
        final float[] conv2B;
        final int channels;
        final int conv1K;

        DacRes(float[] act1Alpha, float[] act1Beta, float[] conv1W, float[] conv1B, float[] act2Alpha,
               float[] act2Beta, float[] conv2W, float[] conv2B, int channels, int conv1K) {
            this.act1Alpha = act1Alpha;
            this.act1Beta = act1Beta;
            this.conv1W = conv1W;
            this.conv1B = conv1B;
            this.act2Alpha = act2Alpha;
            this.act2Beta = act2Beta;
            this.conv2W = conv2W;
            this.conv2B = conv2B;
            this.channels = channels;
            this.conv1K = conv1K;
        }
    }

    /** One DAC block. */
    static final class DacBlock {
        final float[] snakeAlpha;
        final float[] snakeBeta;
        final float[] convW;
        final float[] convB;
        final int inChannels;
        final int outChannels;
        final int stride;
        final List<DacRes> res;

        DacBlock(float[] snakeAlpha, float[] snakeBeta, float[] convW, float[] convB, int inChannels, int outChannels,
                 int stride, List<DacRes> res) {
            this.snakeAlpha = snakeAlpha;
            this.snakeBeta = snakeBeta;
            this.convW = convW;
            this.convB = convB;
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.stride = stride;
            this.res = res;
        }
    }

    private final int nEmbd;
    private final int nLayer;
    private final int nHead;
    private final int nHeadKv;
    private final int dHead;
    private final int nFf;
    private final float eps;
    private final float ropeTheta;
    private final int window;
    private final int hop;
    private final int firstDim;
    private final int firstVocab;
    private final int restDim;
    private final int restVocab;
    private final int nCodebooks;
    private final int decoderWidth;
    private final CpuWeights weights;
    private final CpuTensorRef firstCb;
    private final CpuTensorRef firstOut;
    private final CpuTensorRef restCb;
    private final CpuTensorRef restOut;
    private final float[] preConvW;
    private final float[] preConvB;
    private final int preConvK;
    private final int preConvIn;
    private final int preConvOut;
    private final CpuTensorRef tfmIn;
    private final float[] tfmInB;
    private final CpuTensorRef tfmOut;
    private final float[] tfmOutB;
    private final float[] tfmOutNorm;
    private final TfmLayer[] layers;
    private final UpBlock[] upsample;
    private final float[] dacEntryW;
    private final float[] dacEntryB;
    private final int dacEntryK;
    private final int dacEntryIn;
    private final int dacEntryOut;
    private final DacBlock[] dac;
    private final float[] postSnakeAlpha;
    private final float[] postSnakeBeta;
    private final float[] postConvW;
    private final float[] postConvB;
    private final int postConvK;
    private final int postConvIn;
    private final int postConvOut;

    private Qwen3TtsCode2Wav(Builder b) {
        this.nEmbd = b.nEmbd;
        this.nLayer = b.nLayer;
        this.nHead = b.nHead;
        this.nHeadKv = b.nHeadKv;
        this.dHead = b.dHead;
        this.nFf = b.nFf;
        this.eps = b.eps;
        this.ropeTheta = b.ropeTheta;
        this.window = b.window;
        this.hop = b.hop;
        this.firstDim = b.firstDim;
        this.firstVocab = b.firstVocab;
        this.restDim = b.restDim;
        this.restVocab = b.restVocab;
        this.nCodebooks = b.nCodebooks;
        this.decoderWidth = b.decoderWidth;
        this.weights = b.weights;
        this.firstCb = b.firstCb;
        this.firstOut = b.firstOut;
        this.restCb = b.restCb;
        this.restOut = b.restOut;
        this.preConvW = b.preConvW;
        this.preConvB = b.preConvB;
        this.preConvK = b.preConvK;
        this.preConvIn = b.preConvIn;
        this.preConvOut = b.preConvOut;
        this.tfmIn = b.tfmIn;
        this.tfmInB = b.tfmInB;
        this.tfmOut = b.tfmOut;
        this.tfmOutB = b.tfmOutB;
        this.tfmOutNorm = b.tfmOutNorm;
        this.layers = b.layers;
        this.upsample = b.upsample;
        this.dacEntryW = b.dacEntryW;
        this.dacEntryB = b.dacEntryB;
        this.dacEntryK = b.dacEntryK;
        this.dacEntryIn = b.dacEntryIn;
        this.dacEntryOut = b.dacEntryOut;
        this.dac = b.dac;
        this.postSnakeAlpha = b.postSnakeAlpha;
        this.postSnakeBeta = b.postSnakeBeta;
        this.postConvW = b.postConvW;
        this.postConvB = b.postConvB;
        this.postConvK = b.postConvK;
        this.postConvIn = b.postConvIn;
        this.postConvOut = b.postConvOut;
    }

    /**
     * Resolves the code2wav decoder of an opened clip generator file.
     *
     * @throws IllegalArgumentException when the file is not a qwen3tts code2wav generator
     * @throws GgufFormatException      when a tensor is missing or a shape differs from the reference graph's
     */
    public static Qwen3TtsCode2Wav open(GgufKv kv, TensorCatalog tc, CpuWeights weights) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(weights, "weights");
        String type = kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!Qwen3TtsCodePredictor.PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is "
                    + (type == null ? "absent" : "'" + type + "'") + ", a qwen3tts code2wav decoder declares '"
                    + Qwen3TtsCodePredictor.PROJECTOR_TYPE + "'");
        }
        Builder b = new Builder();
        b.weights = weights;
        b.window = kv.u32(WINDOW_KEY, DEFAULT_WINDOW);
        if (b.window <= 0) {
            throw new GgufFormatException("qwen3tts code2wav: " + WINDOW_KEY + " must be positive, got " + b.window);
        }
        b.ropeTheta = kv.f32(ROPE_THETA_KEY, DEFAULT_ROPE_THETA);
        b.eps = kv.f32(GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON, 1e-5f);
        if (!(b.eps > 0f) || Float.isInfinite(b.eps)) {
            throw new GgufFormatException("qwen3tts code2wav: layer norm epsilon must be positive, got " + b.eps);
        }

        TensorRef firstCbRef = tc.require(QUANT_FIRST_CB + ".weight");
        TensorRef firstOutRef = tc.require(QUANT_FIRST_OUT + ".weight");
        TensorRef restCbRef = tc.require(QUANT_REST_CB + ".weight");
        TensorRef restOutRef = tc.require(QUANT_REST_OUT + ".weight");
        b.firstCb = weights.get(QUANT_FIRST_CB + ".weight");
        b.firstOut = weights.get(QUANT_FIRST_OUT + ".weight");
        b.restCb = weights.get(QUANT_REST_CB + ".weight");
        b.restOut = weights.get(QUANT_REST_OUT + ".weight");
        b.firstDim = dim(firstCbRef, 0);
        b.firstVocab = dim(firstCbRef, 1);
        b.restDim = dim(restCbRef, 0);
        b.restVocab = dim(restCbRef, 1);
        b.decoderWidth = dim(firstOutRef, 1);
        shape(firstCbRef, b.firstDim, b.firstVocab);
        shape(firstOutRef, b.firstDim, b.decoderWidth);
        shape(restCbRef, b.restDim, b.restVocab, dim(restCbRef, 2));
        shape(restOutRef, b.restDim, b.decoderWidth);
        if (dim(restCbRef, 2) <= 0) {
            throw new GgufFormatException("qwen3tts code2wav: " + QUANT_REST_CB + ".weight holds no codebook page");
        }
        b.nCodebooks = dim(restCbRef, 2);

        TensorRef preRef = tc.require(PRE_CONV + ".weight");
        b.preConvW = weights.floats(PRE_CONV + ".weight");
        b.preConvB = weights.floats(PRE_CONV + ".bias");
        b.preConvK = convK(preRef);
        b.preConvIn = convIn(preRef);
        b.preConvOut = convOut(preRef);
        if (b.preConvIn != b.decoderWidth) {
            throw new GgufFormatException("qwen3tts code2wav: " + PRE_CONV + ".weight takes " + b.preConvIn
                    + " channels, the quantizer writes " + b.decoderWidth);
        }
        vector(b.preConvB, "pre_conv bias", b.preConvOut);

        b.tfmIn = weights.get(TFM_IN_PROJ + ".weight");
        b.tfmInB = weights.floats(TFM_IN_PROJ + ".bias");
        b.tfmOut = weights.get(TFM_OUT_PROJ + ".weight");
        b.tfmOutB = weights.floats(TFM_OUT_PROJ + ".bias");
        b.tfmOutNorm = weights.floats(TFM_OUTPUT_NORM + ".weight");
        TensorRef tfmInRef = tc.require(TFM_IN_PROJ + ".weight");
        b.nEmbd = dim(tfmInRef, 1);
        if (dim(tfmInRef, 0) != b.preConvOut) {
            throw new GgufFormatException("qwen3tts code2wav: " + TFM_IN_PROJ + ".weight takes "
                    + dim(tfmInRef, 0) + " channels, the pre-convolution writes " + b.preConvOut);
        }
        checkShape(b.tfmIn, b.preConvOut, b.nEmbd);
        checkShape(b.tfmOut, b.nEmbd, b.preConvOut);
        vector(b.tfmInB, TFM_IN_PROJ + " bias", b.nEmbd);
        vector(b.tfmOutB, TFM_OUT_PROJ + " bias", b.preConvOut);
        vector(b.tfmOutNorm, TFM_OUTPUT_NORM, b.nEmbd);

        b.nHead = (int) kv.u32(HEAD_COUNT_KEY, DEFAULT_HEAD_COUNT);
        b.nHeadKv = (int) kv.u32(HEAD_COUNT_KV_KEY, b.nHead);
        if (b.nHead <= 0 || b.nHeadKv <= 0 || b.nEmbd % b.nHead != 0 || b.nHead % b.nHeadKv != 0) {
            throw new GgufFormatException("qwen3tts code2wav: heads " + b.nHead + "/" + b.nHeadKv
                    + " do not divide the transformer width " + b.nEmbd);
        }
        b.dHead = b.nEmbd / b.nHead;
        if (b.dHead % 2 != 0) {
            throw new GgufFormatException("qwen3tts code2wav: head dim " + b.dHead + " must be even for RoPE");
        }

        b.nLayer = countIndexed(tc, "a.gen.wav.tfm.blk.");
        if (b.nLayer == 0) {
            throw new GgufFormatException("qwen3tts code2wav: the file holds no a.gen.wav.tfm.blk.* layer");
        }
        b.nFf = dim(tc.require(tfmTensor(0, "ffn_gate.weight")), 1);
        b.layers = new TfmLayer[b.nLayer];
        for (int il = 0; il < b.nLayer; il++) {
            b.layers[il] = tfmLayer(weights, il, b.nEmbd, b.nFf, b.nHead, b.nHeadKv, b.dHead);
        }

        b.upsample = upsample(weights, tc, b);
        if (b.upsample.length == 0) {
            throw new GgufFormatException("qwen3tts code2wav: the file holds no a.gen.wav.up.blk.* block");
        }
        TensorRef entryRef = tc.require(DAC_ENTRY + ".weight");
        b.dacEntryW = weights.floats(DAC_ENTRY + ".weight");
        b.dacEntryB = weights.floats(DAC_ENTRY + ".bias");
        b.dacEntryK = convK(entryRef);
        b.dacEntryIn = convIn(entryRef);
        b.dacEntryOut = convOut(entryRef);
        int upOut = b.upsample[b.upsample.length - 1].outChannels;
        if (b.dacEntryIn != upOut) {
            throw new GgufFormatException("qwen3tts code2wav: " + DAC_ENTRY + ".weight takes " + b.dacEntryIn
                    + " channels, the upsample stack writes " + upOut);
        }
        vector(b.dacEntryB, "dac entry bias", b.dacEntryOut);
        b.dac = dac(weights, tc, b);
        b.postSnakeAlpha = weights.floats(DAC_POST_SNAKE + ".alpha");
        b.postSnakeBeta = weights.floats(DAC_POST_SNAKE + ".beta");
        TensorRef postRef = tc.require(DAC_POST_CONV + ".weight");
        b.postConvW = weights.floats(DAC_POST_CONV + ".weight");
        b.postConvB = weights.floats(DAC_POST_CONV + ".bias");
        b.postConvK = convK(postRef);
        b.postConvIn = convIn(postRef);
        b.postConvOut = convOut(postRef);
        int dacOut = b.dac[b.dac.length - 1].res.get(b.dac[b.dac.length - 1].res.size() - 1).channels;
        if (b.postConvIn != dacOut) {
            throw new GgufFormatException("qwen3tts code2wav: " + DAC_POST_CONV + ".weight takes " + b.postConvIn
                    + " channels, the DAC stack writes " + dacOut);
        }
        vector(b.postSnakeAlpha, DAC_POST_SNAKE + ".alpha", b.postConvIn);
        vector(b.postSnakeBeta, DAC_POST_SNAKE + ".beta", b.postConvIn);
        vector(b.postConvB, "dac post bias", b.postConvOut);
        if (b.postConvOut != 1) {
            throw new GgufFormatException("qwen3tts code2wav: " + DAC_POST_CONV + ".weight writes "
                    + b.postConvOut + " channels, the waveform is mono");
        }
        b.hop = upsampleStrideProduct(b.upsample) * dacStrideProduct(b.dac);
        return new Qwen3TtsCode2Wav(b);
    }

    private static TfmLayer tfmLayer(CpuWeights weights, int il, int nEmbd, int nFf, int nHead, int nHeadKv,
                                     int dHead) {
        float[] ln1 = weights.floats(tfmTensor(il, "ln1.weight"));
        CpuTensorRef q = weights.get(tfmTensor(il, "attn_q.weight"));
        CpuTensorRef k = weights.get(tfmTensor(il, "attn_k.weight"));
        CpuTensorRef v = weights.get(tfmTensor(il, "attn_v.weight"));
        CpuTensorRef out = weights.get(tfmTensor(il, "attn_out.weight"));
        float[] ls1 = weights.floats(tfmTensor(il, "ls1.weight"));
        float[] ln2 = weights.floats(tfmTensor(il, "ln2.weight"));
        CpuTensorRef gate = weights.get(tfmTensor(il, "ffn_gate.weight"));
        CpuTensorRef up = weights.get(tfmTensor(il, "ffn_up.weight"));
        CpuTensorRef down = weights.get(tfmTensor(il, "ffn_down.weight"));
        float[] ls2 = weights.floats(tfmTensor(il, "ls2.weight"));
        vector(ln1, tfmTensor(il, "ln1.weight"), nEmbd);
        checkShape(q, nEmbd, nHead * dHead);
        checkShape(k, nEmbd, nHeadKv * dHead);
        checkShape(v, nEmbd, nHeadKv * dHead);
        checkShape(out, nHead * dHead, nEmbd);
        vector(ls1, tfmTensor(il, "ls1.weight"), nEmbd);
        vector(ln2, tfmTensor(il, "ln2.weight"), nEmbd);
        checkShape(gate, nEmbd, nFf);
        checkShape(up, nEmbd, nFf);
        checkShape(down, nFf, nEmbd);
        vector(ls2, tfmTensor(il, "ls2.weight"), nEmbd);
        return new TfmLayer(ln1, q, k, v, out, ls1, ln2, gate, up, down, ls2);
    }

    /** One plus the highest {@code N} among tensors named {@code prefix + N + ".*"}; 0 when there are none. */
    static int countIndexed(TensorCatalog tc, String prefix) {
        int highest = -1;
        for (String name : tc.names()) {
            if (!name.startsWith(prefix)) {
                continue;
            }
            int end = name.indexOf('.', prefix.length());
            if (end < 0) {
                continue;
            }
            int value = 0;
            boolean digits = true;
            for (int i = prefix.length(); i < end; i++) {
                char c = name.charAt(i);
                if (c < '0' || c > '9' || value > (Integer.MAX_VALUE - 9) / 10) {
                    digits = false;
                    break;
                }
                value = value * 10 + (c - '0');
            }
            if (digits) {
                highest = Math.max(highest, value);
            }
        }
        return highest + 1;
    }

    private static UpBlock[] upsample(CpuWeights weights, TensorCatalog tc, Builder b) {
        List<UpBlock> blocks = new ArrayList<>();
        for (int il = 0; il < countIndexed(tc, UP); il++) {
            String base = UP + il + ".";
            TensorRef convRef = tc.require(base + "conv.weight");
            float[] convW = weights.floats(base + "conv.weight");
            float[] convB = weights.floats(base + "conv.bias");
            int convK = convK(convRef);
            int in = transposeIn(convRef);
            int out = transposeOut(convRef);
            if (convK != 2) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "conv.weight has kernel " + convK
                        + ", the upsample stages convolve 2x");
            }
            if (il > 0 && in != blocks.get(il - 1).outChannels) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "conv.weight takes " + in
                        + " channels, the previous stage writes " + blocks.get(il - 1).outChannels);
            }
            if (il == 0 && in != b.preConvOut) {
                throw new GgufFormatException("qwen3tts code2wav: the first upsample stage takes " + in
                        + " channels, the pre-transformer writes " + b.preConvOut);
            }
            vector(convB, base + "conv.bias", out);
            TensorRef dwRef = tc.require(base + "dwconv.weight");
            float[] dw = weights.floats(base + "dwconv.weight");
            float[] dwB = weights.floats(base + "dwconv.bias");
            int dwK = convK(dwRef);
            if (convIn(dwRef) != 1 || convOut(dwRef) != out) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "dwconv.weight must be depthwise over "
                        + out + " channels");
            }
            vector(dwB, base + "dwconv.bias", out);
            float[] normW = weights.floats(base + "norm.weight");
            float[] normB = weights.floats(base + "norm.bias");
            vector(normW, base + "norm.weight", out);
            vector(normB, base + "norm.bias", out);
            CpuTensorRef pw1 = weights.get(base + "pw1.weight");
            float[] pw1B = weights.floats(base + "pw1.bias");
            int ff = (int) pw1.rows();
            checkShape(pw1, out, ff);
            vector(pw1B, base + "pw1.bias", ff);
            CpuTensorRef pw2 = weights.get(base + "pw2.weight");
            float[] pw2B = weights.floats(base + "pw2.bias");
            checkShape(pw2, ff, out);
            vector(pw2B, base + "pw2.bias", out);
            float[] gamma = weights.floats(base + "gamma");
            vector(gamma, base + "gamma", out);
            blocks.add(new UpBlock(convW, convB, dw, dwB, normW, normB, pw1, pw1B, pw2, pw2B, gamma, in, out, convK,
                    dwK));
        }
        return blocks.toArray(new UpBlock[0]);
    }

    private static DacBlock[] dac(CpuWeights weights, TensorCatalog tc, Builder b) {
        List<DacBlock> blocks = new ArrayList<>();
        int nBlock = countIndexed(tc, DAC);
        if (nBlock == 0) {
            throw new GgufFormatException("qwen3tts code2wav: the file holds no " + DAC + "* DAC block");
        }
        for (int ib = 0; ib < nBlock; ib++) {
            String base = DAC + ib + ".";
            float[] alpha = weights.floats(base + "snake.alpha");
            float[] beta = weights.floats(base + "snake.beta");
            TensorRef convRef = tc.require(base + "conv.weight");
            float[] convW = weights.floats(base + "conv.weight");
            float[] convB = weights.floats(base + "conv.bias");
            int k = convK(convRef);
            int in = transposeIn(convRef);
            int out = transposeOut(convRef);
            int stride = k / 2;
            if (k != stride * 2 || stride <= 0) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "conv.weight has kernel " + k
                        + ", the DAC blocks use kernel == 2 * stride");
            }
            int expect = ib == 0 ? b.dacEntryOut
                    : blocks.get(ib - 1).res.get(blocks.get(ib - 1).res.size() - 1).channels;
            if (in != expect) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "conv.weight takes " + in
                        + " channels, the previous stage writes " + expect);
            }
            vector(alpha, base + "snake.alpha", in);
            vector(beta, base + "snake.beta", in);
            vector(convB, base + "conv.bias", out);
            List<DacRes> res = new ArrayList<>();
            int nRes = countIndexed(tc, base + "res.");
            if (nRes == 0) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "res.* holds no residual unit");
            }
            if (nRes > DAC_DILATIONS.length) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "res.* holds " + nRes
                        + " residual units; the DAC dilations cover " + DAC_DILATIONS.length);
            }
            for (int ir = 0; ir < nRes; ir++) {
                String r = base + "res." + ir + ".";
                TensorRef c1Ref = tc.require(r + "conv1.weight");
                TensorRef c2Ref = tc.require(r + "conv2.weight");
                float[] a1 = weights.floats(r + "act1.alpha");
                float[] b1 = weights.floats(r + "act1.beta");
                float[] c1 = weights.floats(r + "conv1.weight");
                float[] c1B = weights.floats(r + "conv1.bias");
                int c1K = convK(c1Ref);
                int channels = convIn(c1Ref);
                int c1Out = convOut(c1Ref);
                float[] a2 = weights.floats(r + "act2.alpha");
                float[] b2 = weights.floats(r + "act2.beta");
                float[] c2 = weights.floats(r + "conv2.weight");
                float[] c2B = weights.floats(r + "conv2.bias");
                if (convK(c2Ref) != 1 || convIn(c2Ref) != c1Out || convOut(c2Ref) != c1Out
                        || channels != c1Out) {
                    throw new GgufFormatException("qwen3tts code2wav: " + r + "conv2.weight must be a pointwise"
                            + " convolution over the residual channels");
                }
                vector(a1, r + "act1.alpha", channels);
                vector(b1, r + "act1.beta", channels);
                vector(c1B, r + "conv1.bias", c1Out);
                vector(a2, r + "act2.alpha", channels);
                vector(b2, r + "act2.beta", channels);
                vector(c2B, r + "conv2.bias", c1Out);
                if (ir > 0 && channels != res.get(ir - 1).channels) {
                    throw new GgufFormatException("qwen3tts code2wav: " + r + "conv1.weight changes the residual"
                            + " channel count");
                }
                res.add(new DacRes(a1, b1, c1, c1B, a2, b2, c2, c2B, channels, c1K));
            }
            if (res.isEmpty()) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "holds no residual unit");
            }
            if (res.size() > DAC_DILATIONS.length) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "holds " + res.size()
                        + " residual units, the reference runs at most " + DAC_DILATIONS.length);
            }
            if (res.get(0).channels != out) {
                throw new GgufFormatException("qwen3tts code2wav: " + base + "conv.weight writes " + out
                        + " channels, its residual units run " + res.get(0).channels);
            }
            blocks.add(new DacBlock(alpha, beta, convW, convB, in, out, stride,
                    Collections.unmodifiableList(res)));
        }
        return blocks.toArray(new DacBlock[0]);
    }

    private static int upsampleStrideProduct(UpBlock[] blocks) {
        int product = 1;
        for (UpBlock block : blocks) {
            product *= block.convK == 0 ? 0 : 2;
        }
        return product;
    }

    private static int dacStrideProduct(DacBlock[] blocks) {
        int product = 1;
        for (DacBlock block : blocks) {
            product *= block.stride;
        }
        return product;
    }

    /** A fresh decoder state: zeroed key/value prefixes, convolution tails and the RoPE position counter. */
    public Qwen3TtsCode2WavSession newSession() {
        return new Qwen3TtsCode2WavSession(this);
    }

    /** The {@code a.gen.wav.tfm.blk.<il>.<part>} tensor name. */
    static String tfmTensor(int il, String part) {
        return "a.gen.wav.tfm.blk." + il + "." + part;
    }

    // ------------------------------------------------------------------ accessors

    /** Transformer width (512 in the reference). */
    public int nEmbd() { return nEmbd; }
    /** Pre-transformer layer count. */
    public int nLayer() { return nLayer; }
    /** Attention heads per query. */
    public int nHead() { return nHead; }
    /** Attention key/value heads. */
    public int nHeadKv() { return nHeadKv; }
    /** Head dim of the pre-transformer attention. */
    int dHead() { return dHead; }
    /** FFN hidden width of the pre-transformer. */
    int nFf() { return nFf; }
    /** The banded attention window and decoder batch size in frames. */
    public int window() { return window; }
    /** Samples per frame of the decoded waveform. */
    public int hop() { return hop; }
    /** The waveform's sample rate. */
    public int sampleRate() { return SAMPLE_RATE; }
    /** Codebook pages of the acoustic quantizer (15 in the reference). */
    public int codebooks() { return nCodebooks; }
    /** Codes per frame: the semantic codebook plus the acoustic pages. */
    public int codesPerFrame() { return nCodebooks + 1; }

    /** The persisted state slot names, in the reference's {@code list_c2w_state_slots} order. */
    public List<String> stateSlots() {
        List<String> names = new ArrayList<>();
        names.add("tfm_pos");
        for (int il = 0; il < nLayer; il++) {
            names.add("tfm_k_" + il);
            names.add("tfm_v_" + il);
        }
        names.add("pre_conv");
        for (int il = 0; il < upsample.length; il++) {
            names.add("up" + il + "_dwconv");
        }
        names.add("dac_entry");
        for (int ib = 0; ib < dac.length; ib++) {
            names.add("dac" + ib + "_tail");
            for (int ir = 0; ir < dac[ib].res.size(); ir++) {
                names.add("dac" + ib + "_res" + ir);
            }
        }
        names.add("dac_post_conv");
        return Collections.unmodifiableList(names);
    }

    /** The bound weights, for the session's embed and matvec calls. */
    CpuWeights weights() { return weights; }
    CpuTensorRef quantFirstCb() { return firstCb; }
    CpuTensorRef quantFirstOut() { return firstOut; }
    CpuTensorRef quantRestCb() { return restCb; }
    CpuTensorRef quantRestOut() { return restOut; }
    int firstDim() { return firstDim; }
    int firstVocab() { return firstVocab; }
    int restDim() { return restDim; }
    int restVocab() { return restVocab; }
    int decoderWidth() { return decoderWidth; }
    float[] preConvW() { return preConvW; }
    float[] preConvB() { return preConvB; }
    int preConvK() { return preConvK; }
    int preConvIn() { return preConvIn; }
    int preConvOut() { return preConvOut; }
    CpuTensorRef tfmIn() { return tfmIn; }
    float[] tfmInB() { return tfmInB; }
    CpuTensorRef tfmOut() { return tfmOut; }
    float[] tfmOutB() { return tfmOutB; }
    float[] tfmOutNorm() { return tfmOutNorm; }
    TfmLayer[] layers() { return layers; }
    UpBlock[] upsample() { return upsample; }
    float[] dacEntryW() { return dacEntryW; }
    float[] dacEntryB() { return dacEntryB; }
    int dacEntryK() { return dacEntryK; }
    int dacEntryIn() { return dacEntryIn; }
    int dacEntryOut() { return dacEntryOut; }
    DacBlock[] dac() { return dac; }
    float[] postSnakeAlpha() { return postSnakeAlpha; }
    float[] postSnakeBeta() { return postSnakeBeta; }
    float[] postConvW() { return postConvW; }
    float[] postConvB() { return postConvB; }
    int postConvK() { return postConvK; }
    int postConvIn() { return postConvIn; }
    int postConvOut() { return postConvOut; }
    float eps() { return eps; }
    float ropeTheta() { return ropeTheta; }

    // ------------------------------------------------------------------ validation helpers

    private static int dim(TensorRef ref, int i) {
        long v = ref.ne(i);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new GgufFormatException("qwen3tts code2wav: tensor " + ref.name() + " dimension " + i + " is " + v);
        }
        return (int) v;
    }

    private static void shape(TensorRef ref, long... ne) {
        boolean ok = ref.rank() >= ne.length;
        for (int i = 0; i < ne.length && ok; i++) {
            ok = ref.ne(i) == ne[i];
        }
        for (int i = ne.length; i < ref.rank() && ok; i++) {
            ok = ref.ne(i) == 1;
        }
        if (!ok) {
            throw new GgufFormatException("qwen3tts code2wav: tensor " + ref.name() + " has shape " + ref
                    + ", expected " + java.util.Arrays.toString(ne) + " (ggml order)");
        }
    }

    private static void checkShape(CpuTensorRef ref, long cols, long rows) {
        if (ref.cols() != cols || ref.rows() != rows) {
            throw new GgufFormatException("qwen3tts code2wav: tensor " + ref + " is " + ref.rows() + " x "
                    + ref.cols() + ", expected " + rows + " x " + cols);
        }
    }

    private static void vector(float[] v, String name, int n) {
        if (v == null || v.length != n) {
            throw new GgufFormatException("qwen3tts code2wav: " + name + " must hold " + n + " floats, got "
                    + (v == null ? "none" : v.length));
        }
    }

    /** Kernel width of a convolution weight {@code [K, IC, OC]} (ggml order). */
    private static int convK(TensorRef ref) {
        return dim(ref, 0);
    }

    /** Input channels of a convolution weight {@code [K, IC, OC]} (ggml order). */
    private static int convIn(TensorRef ref) {
        return ref.rank() < 3 ? 1 : dim(ref, 1);
    }

    /** Output channels of a convolution weight {@code [K, IC, OC]} (ggml order). */
    private static int convOut(TensorRef ref) {
        return ref.rank() < 3 ? dim(ref, 1) : dim(ref, 2);
    }

    /** Input channels of a transposed convolution weight {@code [K, OC, IC]} (ggml order). */
    private static int transposeIn(TensorRef ref) {
        return ref.rank() < 3 ? 1 : dim(ref, 2);
    }

    /** Output channels of a transposed convolution weight {@code [K, OC, IC]} (ggml order). */
    private static int transposeOut(TensorRef ref) {
        return dim(ref, 1);
    }

    /** Builder that keeps the resolution readable within the file limits. */
    private static final class Builder {
        int nEmbd;
        int nLayer;
        int nHead;
        int nHeadKv;
        int dHead;
        int nFf;
        float eps;
        float ropeTheta;
        int window;
        int hop;
        int firstDim;
        int firstVocab;
        int restDim;
        int restVocab;
        int nCodebooks;
        int decoderWidth;
        CpuWeights weights;
        CpuTensorRef firstCb;
        CpuTensorRef firstOut;
        CpuTensorRef restCb;
        CpuTensorRef restOut;
        float[] preConvW;
        float[] preConvB;
        int preConvK;
        int preConvIn;
        int preConvOut;
        CpuTensorRef tfmIn;
        float[] tfmInB;
        CpuTensorRef tfmOut;
        float[] tfmOutB;
        float[] tfmOutNorm;
        TfmLayer[] layers;
        UpBlock[] upsample;
        float[] dacEntryW;
        float[] dacEntryB;
        int dacEntryK;
        int dacEntryIn;
        int dacEntryOut;
        DacBlock[] dac;
        float[] postSnakeAlpha;
        float[] postSnakeBeta;
        float[] postConvW;
        float[] postConvB;
        int postConvK;
        int postConvIn;
        int postConvOut;
    }
}
