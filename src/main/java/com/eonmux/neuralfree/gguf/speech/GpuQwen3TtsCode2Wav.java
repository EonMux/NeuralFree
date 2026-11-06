package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.audio.SpeechKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.arch.RopeType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Device executor of {@link Qwen3TtsCode2Wav}: the same windowed decode graph as {@link Qwen3TtsCode2WavSession},
 * run as kernels on the engine's queue over the clip weights the {@link SpeechDeviceWeights} uploaded.
 *
 * <p>A call runs the RVQ quantization ({@link SpeechKernels#RVQ_SUM_ROWS} over the acoustic pages, then the two
 * projections), the causal pre-convolution, the sliding-window pre-transformer (RMS norms, q/k/v projections, NEOX
 * RoPE, {@link SpeechKernels#TTS_WINDOW_ATTN} over the persisted key/value prefix), the ConvNeXt upsample stages,
 * and the DAC stack's SnakeBeta activations ({@link SpeechKernels#SNAKE_BETA}) and causal transposed/dilated
 * convolutions. The key/value prefix, every convolution's left context and the RoPE position counter live in device
 * buffers, so consecutive windows concatenate to the samples of the whole sequence.
 *
 * <p>Owns its activation and state buffers; not thread-safe.
 */
final class GpuQwen3TtsCode2Wav implements Qwen3TtsCode2Wav.Decoder {

    /** LayerNorm epsilon of the ConvNeXt blocks ({@link Qwen3TtsCode2WavSession}). */
    private static final float CONVNEXT_EPS = 1e-6f;

    /** Left context or overlap tail of one causal convolution, double-buffered across calls. */
    private final class ConvState {
        private GpuBuffer cur;
        private GpuBuffer next;
        private final boolean active;
        private final long floats;

        ConvState(String name, int rows, int channels) {
            this.floats = (long) rows * channels;
            this.active = rows > 0 && channels > 0;
            if (active) {
                cur = own(name + ".cur", floats);
                next = own(name + ".next", floats);
            }
        }

        GpuBuffer read() {
            return active ? cur : ops.placeholders().readOnlyFloats();
        }

        GpuBuffer write() {
            return active ? next : ops.placeholders().scratchFloats();
        }

        void swap() {
            if (active) {
                GpuBuffer t = cur;
                cur = next;
                next = t;
            }
        }
    }

    /** Device buffers of one pre-transformer layer. */
    private static final class TfmWeights {
        final GpuWeight ln1;
        final GpuWeight ln2;
        final GpuWeight q;
        final GpuWeight k;
        final GpuWeight v;
        final GpuWeight out;
        final GpuWeight gate;
        final GpuWeight up;
        final GpuWeight down;
        final GpuBuffer ls1;
        final GpuBuffer ls2;
        final GpuBuffer kvK;
        final GpuBuffer kvV;

        TfmWeights(GpuWeight ln1, GpuWeight ln2, GpuWeight q, GpuWeight k, GpuWeight v, GpuWeight out,
                   GpuWeight gate, GpuWeight up, GpuWeight down, GpuBuffer ls1, GpuBuffer ls2,
                   GpuBuffer kvK, GpuBuffer kvV) {
            this.ln1 = ln1;
            this.ln2 = ln2;
            this.q = q;
            this.k = k;
            this.v = v;
            this.out = out;
            this.gate = gate;
            this.up = up;
            this.down = down;
            this.ls1 = ls1;
            this.ls2 = ls2;
            this.kvK = kvK;
            this.kvV = kvV;
        }
    }

    /** Device buffers of one ConvNeXt upsample stage. */
    private static final class UpWeights {
        final GpuBuffer convW;
        final GpuBuffer convB;
        final GpuBuffer dwW;
        final GpuBuffer dwB;
        final GpuWeight normW;
        final GpuWeight normB;
        final GpuWeight pw1;
        final GpuBuffer pw1B;
        final GpuWeight pw2;
        final GpuBuffer pw2B;
        final GpuBuffer gamma;

        UpWeights(GpuBuffer convW, GpuBuffer convB, GpuBuffer dwW, GpuBuffer dwB, GpuWeight normW, GpuWeight normB,
                  GpuWeight pw1, GpuBuffer pw1B, GpuWeight pw2, GpuBuffer pw2B, GpuBuffer gamma) {
            this.convW = convW;
            this.convB = convB;
            this.dwW = dwW;
            this.dwB = dwB;
            this.normW = normW;
            this.normB = normB;
            this.pw1 = pw1;
            this.pw1B = pw1B;
            this.pw2 = pw2;
            this.pw2B = pw2B;
            this.gamma = gamma;
        }
    }

    /** Device buffers of one DAC residual unit. */
    private static final class ResWeights {
        final GpuBuffer act1Alpha;
        final GpuBuffer act1Beta;
        final GpuBuffer conv1W;
        final GpuBuffer conv1B;
        final GpuBuffer act2Alpha;
        final GpuBuffer act2Beta;
        final GpuBuffer conv2W;
        final GpuBuffer conv2B;

        ResWeights(GpuBuffer act1Alpha, GpuBuffer act1Beta, GpuBuffer conv1W, GpuBuffer conv1B,
                   GpuBuffer act2Alpha, GpuBuffer act2Beta, GpuBuffer conv2W, GpuBuffer conv2B) {
            this.act1Alpha = act1Alpha;
            this.act1Beta = act1Beta;
            this.conv1W = conv1W;
            this.conv1B = conv1B;
            this.act2Alpha = act2Alpha;
            this.act2Beta = act2Beta;
            this.conv2W = conv2W;
            this.conv2B = conv2B;
        }
    }

    /** Device buffers of one DAC block. */
    private static final class DacWeights {
        final GpuBuffer snakeAlpha;
        final GpuBuffer snakeBeta;
        final GpuBuffer convW;
        final GpuBuffer convB;
        final ResWeights[] res;

        DacWeights(GpuBuffer snakeAlpha, GpuBuffer snakeBeta, GpuBuffer convW, GpuBuffer convB, ResWeights[] res) {
            this.snakeAlpha = snakeAlpha;
            this.snakeBeta = snakeBeta;
            this.convW = convW;
            this.convB = convB;
            this.res = res;
        }
    }

    private final Qwen3TtsCode2Wav geo;
    private final SpeechDeviceWeights device;
    private final GpuOps ops;
    private final int window;
    private final int prefix;
    private final int nEmbd;
    private final int nLayer;
    private final int nHead;
    private final int nHeadKv;
    private final int dHead;
    private final int kvCh;
    private final int nFf;
    private final int decoderWidth;
    private final int firstDim;
    private final int firstVocab;
    private final int restDim;
    private final int restVocab;
    private final int nCodebooks;
    private final int codes;
    private final float eps;
    private final int hop;
    private final int sampleRate;

    private final GpuWeight firstCb;
    private final GpuWeight restCb;
    private final GpuWeight firstOut;
    private final GpuWeight restOut;
    private final GpuWeight tfmIn;
    private final GpuWeight tfmOut;
    private final GpuWeight tfmOutNorm;
    private final GpuBuffer preConvW;
    private final GpuBuffer preConvB;
    private final int preConvK;
    private final GpuBuffer tfmInB;
    private final GpuBuffer tfmOutB;
    private final TfmWeights[] layers;
    private final UpWeights[] ups;
    private final GpuBuffer dacEntryW;
    private final GpuBuffer dacEntryB;
    private final int dacEntryK;
    private final int dacEntryIn;
    private final int dacEntryOut;
    private final DacWeights[] dac;
    private final GpuBuffer postSnakeAlpha;
    private final GpuBuffer postSnakeBeta;
    private final GpuBuffer postConvW;
    private final GpuBuffer postConvB;
    private final int postConvK;
    private final int postConvIn;
    private final RopeSpec rope;
    private final GpuBuffer positions;

    private final ConvState preConvState;
    private final ConvState[] upStates;
    private final ConvState dacEntryState;
    private final ConvState[] dacTailStates;
    private final ConvState[][] resStates;
    private final ConvState postConvState;

    private final GpuBuffer sem;
    private final GpuBuffer restRows;
    private final GpuBuffer restSum;
    private final GpuBuffer hidden;
    private final GpuBuffer quantTmp;
    private final GpuBuffer normed;
    private final GpuBuffer q;
    private final GpuBuffer k;
    private final GpuBuffer v;
    private final GpuBuffer attn;
    private final GpuBuffer projected;
    private final GpuBuffer gateBuf;
    private final GpuBuffer upBuf;
    private final GpuBuffer swiglu;
    private final GpuBuffer xBuf;
    private final GpuBuffer yBuf;
    private final GpuBuffer dacTmp;
    private final GpuBuffer dacH1;
    private final GpuBuffer dacH2;
    private final GpuBuffer dacH3;
    private final GpuBuffer[] upTransposes;
    private final GpuBuffer[] upDws;
    private final GpuBuffer[] upFfs;
    private final GpuBuffer[] upBacks;
    private final GpuBuffer[] upOuts;
    private final GpuBuffer finalOut;
    private final List<GpuBuffer> owned = new ArrayList<>();
    private int tfmPos;
    private boolean closed;

    GpuQwen3TtsCode2Wav(Qwen3TtsCode2Wav geo, SpeechDeviceWeights device) {
        this.geo = Objects.requireNonNull(geo, "geo");
        this.device = Objects.requireNonNull(device, "device");
        this.ops = device.ops();
        this.window = geo.window();
        this.prefix = window - 1;
        this.nEmbd = geo.nEmbd();
        this.nLayer = geo.nLayer();
        this.nHead = geo.nHead();
        this.nHeadKv = geo.nHeadKv();
        this.dHead = geo.dHead();
        this.kvCh = nHeadKv * dHead;
        this.nFf = geo.nFf();
        this.decoderWidth = geo.decoderWidth();
        this.firstDim = geo.firstDim();
        this.firstVocab = geo.firstVocab();
        this.restDim = geo.restDim();
        this.restVocab = geo.restVocab();
        this.nCodebooks = geo.codebooks();
        this.codes = geo.codesPerFrame();
        this.eps = geo.eps();
        this.hop = geo.hop();
        this.sampleRate = geo.sampleRate();
        if (window <= 1) {
            throw new IllegalArgumentException("qwen3tts code2wav: the decoder window must exceed 1, got " + window);
        }

        device.requireShape(Qwen3TtsCode2Wav.QUANT_FIRST_CB + ".weight", firstVocab, firstDim);
        device.requireShape(Qwen3TtsCode2Wav.QUANT_REST_CB + ".weight", (long) restVocab * nCodebooks, restDim);
        this.firstCb = device.weight(Qwen3TtsCode2Wav.QUANT_FIRST_CB + ".weight");
        this.restCb = device.weight(Qwen3TtsCode2Wav.QUANT_REST_CB + ".weight");
        this.firstOut = matrix(Qwen3TtsCode2Wav.QUANT_FIRST_OUT, decoderWidth, firstDim);
        this.restOut = matrix(Qwen3TtsCode2Wav.QUANT_REST_OUT, decoderWidth, restDim);
        this.tfmIn = matrix(Qwen3TtsCode2Wav.TFM_IN_PROJ, nEmbd, geo.preConvOut());
        this.tfmOut = matrix(Qwen3TtsCode2Wav.TFM_OUT_PROJ, geo.preConvOut(), nEmbd);
        this.tfmOutNorm = device.f32Weight(Qwen3TtsCode2Wav.TFM_OUTPUT_NORM + ".weight", geo.tfmOutNorm());
        this.preConvW = vector(Qwen3TtsCode2Wav.PRE_CONV + ".weight", geo.preConvW());
        this.preConvB = vector(Qwen3TtsCode2Wav.PRE_CONV + ".bias", geo.preConvB());
        this.preConvK = geo.preConvK();
        this.tfmInB = vector(Qwen3TtsCode2Wav.TFM_IN_PROJ + ".bias", geo.tfmInB());
        this.tfmOutB = vector(Qwen3TtsCode2Wav.TFM_OUT_PROJ + ".bias", geo.tfmOutB());

        this.layers = new TfmWeights[nLayer];
        for (int il = 0; il < nLayer; il++) {
            layers[il] = tfmLayer(il);
        }
        this.ups = new UpWeights[geo.upsample().length];
        for (int i = 0; i < ups.length; i++) {
            ups[i] = upWeights(i);
        }
        this.dacEntryW = vector(Qwen3TtsCode2Wav.DAC_ENTRY + ".weight", geo.dacEntryW());
        this.dacEntryB = vector(Qwen3TtsCode2Wav.DAC_ENTRY + ".bias", geo.dacEntryB());
        this.dacEntryK = geo.dacEntryK();
        this.dacEntryIn = geo.dacEntryIn();
        this.dacEntryOut = geo.dacEntryOut();
        this.dac = new DacWeights[geo.dac().length];
        for (int i = 0; i < dac.length; i++) {
            dac[i] = dacWeights(i);
        }
        this.postSnakeAlpha = vector(Qwen3TtsCode2Wav.DAC_POST_SNAKE + ".alpha", geo.postSnakeAlpha());
        this.postSnakeBeta = vector(Qwen3TtsCode2Wav.DAC_POST_SNAKE + ".beta", geo.postSnakeBeta());
        this.postConvW = vector(Qwen3TtsCode2Wav.DAC_POST_CONV + ".weight", geo.postConvW());
        this.postConvB = vector(Qwen3TtsCode2Wav.DAC_POST_CONV + ".bias", geo.postConvB());
        this.postConvK = geo.postConvK();
        this.postConvIn = geo.postConvIn();
        this.rope = RopeSpec.builder(RopeType.NEOX).nDims(dHead).freqBase(geo.ropeTheta()).build();
        this.positions = own("positions", window);

        this.preConvState = new ConvState("pre_conv", preConvK - 1, decoderWidth);
        this.upStates = new ConvState[ups.length];
        for (int i = 0; i < ups.length; i++) {
            upStates[i] = new ConvState("up" + i + ".dwconv", geo.upsample()[i].dwK - 1,
                    geo.upsample()[i].outChannels);
        }
        this.dacEntryState = new ConvState("dac_entry", Math.max(0, dacEntryK - 1), dacEntryIn);
        this.dacTailStates = new ConvState[dac.length];
        this.resStates = new ConvState[dac.length][];
        for (int i = 0; i < dac.length; i++) {
            Qwen3TtsCode2Wav.DacBlock block = geo.dac()[i];
            this.dacTailStates[i] = new ConvState("dac" + i + ".tail", trim(2 * block.stride, block.stride),
                    block.outChannels);
            this.resStates[i] = new ConvState[block.res.size()];
            for (int r = 0; r < block.res.size(); r++) {
                Qwen3TtsCode2Wav.DacRes res = block.res.get(r);
                this.resStates[i][r] = new ConvState("dac" + i + ".res" + r, (res.conv1K - 1) * dilations(r),
                        res.channels);
            }
        }
        this.postConvState = new ConvState("dac_post_conv", postConvK - 1, postConvIn);

        this.sem = own("quant.sem", (long) window * firstDim);
        this.restRows = own("quant.rest", (long) nCodebooks * window * restDim);
        this.restSum = own("quant.sum", (long) window * restDim);
        this.hidden = own("quant.hidden", (long) window * decoderWidth);
        this.quantTmp = own("quant.tmp", (long) window * decoderWidth);

        long transformer = (long) window * Math.max(Math.max(nEmbd, decoderWidth),
                Math.max(nHead * dHead, Math.max(kvCh, nFf)));
        long widest = Math.max(transformer, (long) window * geo.preConvOut());
        this.normed = own("tfm.normed", widest);
        this.q = own("tfm.q", (long) window * nHead * dHead);
        this.k = own("tfm.k", (long) window * kvCh);
        this.v = own("tfm.v", (long) window * kvCh);
        this.attn = own("tfm.attn", (long) window * nHead * dHead);
        this.projected = own("tfm.projected", (long) window * nEmbd);
        this.gateBuf = own("tfm.gate", (long) window * nFf);
        this.upBuf = own("tfm.up", (long) window * nFf);
        this.swiglu = own("tfm.swiglu", (long) window * nFf);
        this.yBuf = own("tfm.out", (long) window * geo.preConvOut());

        long maxDac = maxDacFloats();
        this.xBuf = own("signal.x", Math.max(widest, maxDac));
        this.dacTmp = own("dac.t", maxDac);
        this.dacH1 = own("dac.h1", maxDac);
        this.dacH2 = own("dac.h2", maxDac);
        this.dacH3 = own("dac.h3", maxDac);

        int upCount = ups.length;
        this.upTransposes = new GpuBuffer[upCount];
        this.upDws = new GpuBuffer[upCount];
        this.upFfs = new GpuBuffer[upCount];
        this.upBacks = new GpuBuffer[upCount];
        this.upOuts = new GpuBuffer[upCount];
        int rows = window;
        for (int i = 0; i < upCount; i++) {
            rows *= 2;
            Qwen3TtsCode2Wav.UpBlock block = geo.upsample()[i];
            int channels = block.outChannels;
            int ff = block.pw1B.length;
            upTransposes[i] = own("up" + i + ".transpose", (long) rows * channels);
            upDws[i] = own("up" + i + ".dw", (long) rows * channels);
            upFfs[i] = own("up" + i + ".ff", (long) rows * ff);
            upBacks[i] = own("up" + i + ".back", (long) rows * channels);
            upOuts[i] = own("up" + i + ".out", (long) rows * channels);
        }
        for (int i = 0; i < dac.length; i++) {
            rows *= geo.dac()[i].stride;
        }
        this.finalOut = own("dac.out", rows);
        this.tfmPos = 0;
    }

    /** Largest activation of the DAC stage: the entry output and every block's output. */
    private long maxDacFloats() {
        int rows = window;
        for (Qwen3TtsCode2Wav.UpBlock block : geo.upsample()) {
            rows *= 2;
        }
        long max = (long) rows * dacEntryOut;
        for (Qwen3TtsCode2Wav.DacBlock block : geo.dac()) {
            rows *= block.stride;
            max = Math.max(max, (long) rows * block.outChannels);
        }
        return max;
    }

    private TfmWeights tfmLayer(int il) {
        String base = "a.gen.wav.tfm.blk." + il + ".";
        GpuWeight q = matrix(base + "attn_q", nHead * dHead, nEmbd);
        GpuWeight k = matrix(base + "attn_k", kvCh, nEmbd);
        GpuWeight v = matrix(base + "attn_v", kvCh, nEmbd);
        GpuWeight out = matrix(base + "attn_out", nEmbd, nHead * dHead);
        GpuWeight gate = matrix(base + "ffn_gate", nFf, nEmbd);
        GpuWeight up = matrix(base + "ffn_up", nFf, nEmbd);
        GpuWeight down = matrix(base + "ffn_down", nEmbd, nFf);
        return new TfmWeights(
                device.f32Weight(base + "ln1.weight", geo.layers()[il].ln1),
                device.f32Weight(base + "ln2.weight", geo.layers()[il].ln2),
                q, k, v, out, gate, up, down,
                vector(base + "ls1.weight", geo.layers()[il].ls1),
                vector(base + "ls2.weight", geo.layers()[il].ls2),
                own("tfm.kv.k." + il, (long) (prefix + window) * kvCh),
                own("tfm.kv.v." + il, (long) (prefix + window) * kvCh));
    }

    private UpWeights upWeights(int i) {
        String base = Qwen3TtsCode2Wav.UP + i + ".";
        Qwen3TtsCode2Wav.UpBlock block = geo.upsample()[i];
        return new UpWeights(
                vector(base + "conv.weight", block.convW),
                vector(base + "conv.bias", block.convB),
                vector(base + "dwconv.weight", block.dwconvW),
                vector(base + "dwconv.bias", block.dwconvB),
                device.f32Weight(base + "norm.weight", block.normW),
                device.f32Weight(base + "norm.bias", block.normB),
                matrix(base + "pw1", block.pw1B.length, block.outChannels),
                vector(base + "pw1.bias", block.pw1B),
                matrix(base + "pw2", block.outChannels, block.pw1B.length),
                vector(base + "pw2.bias", block.pw2B),
                vector(base + "gamma", block.gamma));
    }

    private DacWeights dacWeights(int i) {
        String base = Qwen3TtsCode2Wav.DAC + i + ".";
        Qwen3TtsCode2Wav.DacBlock block = geo.dac()[i];
        ResWeights[] res = new ResWeights[block.res.size()];
        for (int r = 0; r < res.length; r++) {
            String rb = base + "res." + r + ".";
            Qwen3TtsCode2Wav.DacRes unit = block.res.get(r);
            res[r] = new ResWeights(
                    vector(rb + "act1.alpha", unit.act1Alpha),
                    vector(rb + "act1.beta", unit.act1Beta),
                    vector(rb + "conv1.weight", unit.conv1W),
                    vector(rb + "conv1.bias", unit.conv1B),
                    vector(rb + "act2.alpha", unit.act2Alpha),
                    vector(rb + "act2.beta", unit.act2Beta),
                    vector(rb + "conv2.weight", unit.conv2W),
                    vector(rb + "conv2.bias", unit.conv2B));
        }
        return new DacWeights(
                vector(base + "snake.alpha", block.snakeAlpha),
                vector(base + "snake.beta", block.snakeBeta),
                vector(base + "conv.weight", block.convW),
                vector(base + "conv.bias", block.convB),
                res);
    }

    private GpuWeight matrix(String name, long rows, long cols) {
        device.requireShape(name + ".weight", rows, cols);
        return device.weight(name + ".weight");
    }

    private GpuBuffer vector(String name, float[] values) {
        return device.f32(name, values);
    }

    private GpuBuffer own(String name, long floats) {
        GpuBuffer buffer = device.allocate("code2wav." + name, floats);
        owned.add(buffer);
        return buffer;
    }

    private static int trim(int k, int stride) {
        return k - stride;
    }

    private static int dilations(int res) {
        return Qwen3TtsCode2Wav.DAC_DILATIONS[res];
    }

    @Override
    public float[] decode(int[][] frames) {
        ensureOpen();
        Objects.requireNonNull(frames, "frames");
        int n = frames.length;
        if (n < 1 || n > window) {
            throw new IllegalArgumentException("a code2wav batch holds 1 to " + window + " frames, got " + n);
        }
        for (int f = 0; f < n; f++) {
            if (frames[f] == null || frames[f].length != codes) {
                throw new IllegalArgumentException("frame " + f + " holds "
                        + (frames[f] == null ? "no" : frames[f].length) + " codes, the quantizer takes " + codes);
            }
            for (int g = 0; g < codes; g++) {
                int limit = g == 0 ? firstVocab : restVocab;
                int code = frames[f][g];
                if (code < 0 || code >= limit) {
                    throw new IllegalArgumentException("code " + code + " of frame " + f + ", codebook " + g
                            + " is outside [0, " + limit + ")");
                }
            }
        }

        quantize(frames, n);
        causalConv(hidden, preConvW, preConvB, window, decoderWidth, geo.preConvOut(), preConvK, 1, preConvState,
                xBuf);
        transformer();

        int rows = window;
        GpuBuffer signal = yBuf;
        for (int i = 0; i < ups.length; i++) {
            int nextRows = rows * 2;
            Qwen3TtsCode2Wav.UpBlock block = geo.upsample()[i];
            causalTranspose(signal, ups[i].convW, ups[i].convB, rows, block.inChannels, block.outChannels,
                    block.convK, 2, null, upTransposes[i]);
            convNext(i, upTransposes[i], nextRows, block.outChannels, upOuts[i]);
            signal = upOuts[i];
            rows = nextRows;
        }

        causalConv(signal, dacEntryW, dacEntryB, rows, dacEntryIn, dacEntryOut, dacEntryK, 1, dacEntryState, xBuf);
        GpuBuffer x = xBuf;
        GpuBuffer t = dacTmp;
        int dacRows = rows;
        int dacChannels = dacEntryOut;
        for (int i = 0; i < dac.length; i++) {
            Qwen3TtsCode2Wav.DacBlock block = geo.dac()[i];
            snake(x, dacRows * dacChannels, dacChannels, dac[i].snakeAlpha, dac[i].snakeBeta, x);
            int nextRows = dacRows * block.stride;
            causalTranspose(x, dac[i].convW, dac[i].convB, dacRows, block.inChannels, block.outChannels,
                    block.stride * 2, block.stride, dacTailStates[i], t);
            for (int r = 0; r < block.res.size(); r++) {
                residualUnit(t, nextRows, block.outChannels, dac[i].res[r], dilations(r), resStates[i][r],
                        block.res.get(r).conv1K);
            }
            GpuBuffer swap = x;
            x = t;
            t = swap;
            dacRows = nextRows;
            dacChannels = block.outChannels;
        }

        snake(x, dacRows * dacChannels, dacChannels, postSnakeAlpha, postSnakeBeta, x);
        causalConv(x, postConvW, postConvB, dacRows, postConvIn, 1, postConvK, 1, postConvState, finalOut);
        ops.unary(ActOp.CLAMP, ActParams.builder().clamp(-1f, 1f).build(), finalOut, 0, finalOut, 0, dacRows);

        int keep = n * hop;
        float[] out = new float[keep];
        device.readFloats(finalOut, out, keep);
        return out;
    }

    @Override
    public AudioClip decodeClip(int[][] frames) {
        return new AudioClip(decode(frames), sampleRate);
    }

    @Override
    public int sampleRate() {
        return sampleRate;
    }

    @Override
    public float position() {
        return tfmPos;
    }

    // ------------------------------------------------------------------ quantizer, transformer, upsample, DAC

    private void quantize(int[][] frames, int n) {
        int[] firstIds = new int[window];
        int[] restIds = new int[nCodebooks * window];
        for (int f = 0; f < n; f++) {
            firstIds[f] = frames[f][0];
            for (int p = 0; p < nCodebooks; p++) {
                restIds[p * window + f] = p * restVocab + frames[f][p + 1];
            }
        }
        ops.embed(firstCb, firstIds, sem, 0);
        ops.embed(restCb, restIds, restRows, 0);
        if (nCodebooks > 0) {
            ops.enqueue(SpeechKernels.RVQ_SUM_ROWS, new long[]{SpeechKernels.elementGlobal((long) window * restDim)},
                    new long[]{SpeechKernels.LOCAL}, nCodebooks, window, restDim, restRows, restSum);
        } else {
            ops.copy(restRows, 0, restSum, 0, window * restDim);
        }
        ops.matmul(firstOut, sem, 0, hidden, 0, window);
        ops.matmul(restOut, restSum, 0, quantTmp, 0, window);
        ops.add(hidden, 0, quantTmp, 0, hidden, 0, window * decoderWidth);
    }

    private void transformer() {
        int base = tfmPos;
        int[] pos = new int[window];
        for (int i = 0; i < window; i++) {
            pos[i] = base + i;
        }
        device.writeInts(positions, pos);

        ops.matmul(tfmIn, xBuf, 0, yBuf, 0, window);
        biasAdd(yBuf, tfmInB, window, nEmbd);
        GpuBuffer x = yBuf;
        for (int il = 0; il < nLayer; il++) {
            x = tfmLayerForward(x, layers[il], base);
        }
        ops.norm(NormKind.RMS, tfmOutNorm, null, eps, x, 0, normed, 0, window, nEmbd);
        ops.matmul(tfmOut, normed, 0, yBuf, 0, window);
        biasAdd(yBuf, tfmOutB, window, geo.preConvOut());
        tfmPos += window;
    }

    /** One pre-transformer layer; returns the residual stream's buffer for the next layer. */
    private GpuBuffer tfmLayerForward(GpuBuffer x, TfmWeights l, int base) {
        ops.norm(NormKind.RMS, l.ln1, null, eps, x, 0, normed, 0, window, nEmbd);
        ops.matmul(l.q, normed, 0, q, 0, window);
        ops.matmul(l.k, normed, 0, k, 0, window);
        ops.matmul(l.v, normed, 0, v, 0, window);
        rope(q, nHead, nHead * dHead);
        rope(k, nHeadKv, kvCh);
        ops.copy2d(k, 0, kvCh, l.kvK, prefix * kvCh, kvCh, window, kvCh);
        ops.copy2d(v, 0, kvCh, l.kvV, prefix * kvCh, kvCh, window, kvCh);
        ops.enqueue(SpeechKernels.TTS_WINDOW_ATTN,
                new long[]{SpeechKernels.windowAttentionGlobal(window, nHead)}, new long[]{SpeechKernels.LOCAL},
                window, nHead, nHeadKv, dHead, prefix, window, base, q, l.kvK, l.kvV, attn);
        ops.matmul(l.out, attn, 0, projected, 0, window);
        mulRowScale(projected, l.ls1, window, nEmbd);
        ops.add(projected, 0, x, 0, x, 0, window * nEmbd);

        ops.norm(NormKind.RMS, l.ln2, null, eps, x, 0, normed, 0, window, nEmbd);
        ops.matmul(l.gate, normed, 0, gateBuf, 0, window);
        ops.matmul(l.up, normed, 0, upBuf, 0, window);
        ops.glu(GluOp.SWIGLU, ActParams.NONE, gateBuf, 0, nFf, upBuf, 0, nFf, swiglu, 0, nFf, window, nFf);
        ops.matmul(l.down, swiglu, 0, projected, 0, window);
        mulRowScale(projected, l.ls2, window, nEmbd);
        ops.add(projected, 0, x, 0, x, 0, window * nEmbd);

        // the next window's prefix: the last (window - 1) rows of this batch's keys and values
        ops.copy2d(l.kvK, window * kvCh, kvCh, l.kvK, 0, kvCh, prefix, kvCh);
        ops.copy2d(l.kvV, window * kvCh, kvCh, l.kvV, 0, kvCh, prefix, kvCh);
        return x;
    }

    private void convNext(int index, GpuBuffer input, int rows, int channels, GpuBuffer out) {
        UpWeights l = ups[index];
        Qwen3TtsCode2Wav.UpBlock block = geo.upsample()[index];
        causalDepthwise(input, l.dwW, l.dwB, rows, channels, block.dwK, upStates[index], upDws[index]);
        ops.norm(NormKind.LAYER, l.normW, l.normB, CONVNEXT_EPS, upDws[index], 0, upDws[index], 0, rows,
                channels);
        int ff = block.pw1B.length;
        ops.matmul(l.pw1, upDws[index], 0, upFfs[index], 0, rows);
        biasAdd(upFfs[index], l.pw1B, rows, ff);
        ops.unary(ActOp.GELU_TANH, ActParams.NONE, upFfs[index], 0, upFfs[index], 0, rows * ff);
        ops.matmul(l.pw2, upFfs[index], 0, upBacks[index], 0, rows);
        biasAdd(upBacks[index], l.pw2B, rows, channels);
        mulRowScale(upBacks[index], l.gamma, rows, channels);
        ops.add(upBacks[index], 0, input, 0, out, 0, rows * channels);
    }

    private void residualUnit(GpuBuffer x, int rows, int channels, ResWeights r, int dilation, ConvState state,
                              int conv1K) {
        snake(x, rows * channels, channels, r.act1Alpha, r.act1Beta, dacH1);
        causalConv(dacH1, r.conv1W, r.conv1B, rows, channels, channels, conv1K, dilation, state, dacH2);
        snake(dacH2, rows * channels, channels, r.act2Alpha, r.act2Beta, dacH2);
        causalConv(dacH2, r.conv2W, r.conv2B, rows, channels, channels, 1, 1, null, dacH3);
        ops.add(x, 0, dacH3, 0, x, 0, rows * channels);
    }

    // ------------------------------------------------------------------ kernel helpers

    private void causalConv(GpuBuffer x, GpuBuffer w, GpuBuffer b, int n, int ic, int oc, int k, int dil,
                            ConvState state, GpuBuffer y) {
        int pad = state == null ? 0 : (k - 1) * dil;
        ConvState st = state;
        ops.enqueue(SpeechKernels.CONV1D_CAUSAL, new long[]{SpeechKernels.convGlobal(n, ic, oc, pad)},
                new long[]{SpeechKernels.LOCAL}, n, ic, oc, k, dil, pad, x, w, b,
                st == null ? ops.placeholders().readOnlyFloats() : st.read(), y,
                st == null ? ops.placeholders().scratchFloats() : st.write());
        if (st != null) {
            st.swap();
        }
    }

    private void causalDepthwise(GpuBuffer x, GpuBuffer w, GpuBuffer b, int n, int channels, int k,
                                 ConvState state, GpuBuffer y) {
        int pad = k - 1;
        ops.enqueue(SpeechKernels.CONV1D_DEPTHWISE_CAUSAL,
                new long[]{SpeechKernels.depthwiseGlobal(n, channels, pad)}, new long[]{SpeechKernels.LOCAL},
                n, channels, k, pad, x, w, b, state.read(), y, state.write());
        state.swap();
    }

    /** Causal ConvTranspose1d; a null state is the upsample case where {@code k == stride} (no overlap). */
    private void causalTranspose(GpuBuffer x, GpuBuffer w, GpuBuffer b, int n, int ic, int oc, int k, int stride,
                                 ConvState state, GpuBuffer y) {
        ops.enqueue(SpeechKernels.CONV_TRANSPOSE1D_CAUSAL,
                new long[]{SpeechKernels.transposeGlobal(n, oc, k, stride)}, new long[]{SpeechKernels.LOCAL},
                n, ic, oc, k, stride, x, w, b,
                state == null ? ops.placeholders().readOnlyFloats() : state.read(), y,
                state == null ? ops.placeholders().scratchFloats() : state.write());
        if (state != null) {
            state.swap();
        }
    }

    private void snake(GpuBuffer x, int n, int channels, GpuBuffer alpha, GpuBuffer beta, GpuBuffer y) {
        ops.enqueue(SpeechKernels.SNAKE_BETA, new long[]{SpeechKernels.elementGlobal(n)},
                new long[]{SpeechKernels.LOCAL}, n, channels, x, alpha, beta, y);
    }

    private void rope(GpuBuffer x, int nHeads, int rowStride) {
        Object[] args = RopeKernels.ropeExtArgs(rope, window, nHeads, dHead, rowStride, 0, window, 0, false,
                positions, ops.placeholders().readOnlyFloats(), x);
        long items = RopeKernels.ropeExtWorkItems(RopeType.NEOX, window, nHeads, dHead);
        ops.enqueue(RopeKernels.ROPE_EXT, ActKernels.global1d(items, ops.local()), new long[]{ops.local()}, args);
    }

    private void biasAdd(GpuBuffer data, GpuBuffer bias, int rows, int dim) {
        ops.enqueue(GpuOps.BATCH_BIAS_ADD, ActKernels.global1d(dim, ops.local()), new long[]{ops.local()},
                dim, rows, bias, data);
    }

    private void mulRowScale(GpuBuffer x, GpuBuffer scale, int rows, int cols) {
        ops.enqueue(NormKernels.MUL_ROW_SCALE, ActKernels.global1d((long) rows * cols, ops.local()),
                new long[]{ops.local()}, NormKernels.mulRowScaleArgs(rows, cols, scale, cols, x));
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the qwen3tts GPU code2wav decoder is closed");
        }
    }

    /** Releases this session's activation and state buffers. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (GpuBuffer buffer : owned) {
            try {
                buffer.close();
            } catch (RuntimeException e) {
                failure = failure == null ? e : failure;
            }
        }
        owned.clear();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuQwen3TtsCode2Wav{window=" + window + ", position=" + tfmPos + ", layers=" + nLayer + '}';
    }
}
