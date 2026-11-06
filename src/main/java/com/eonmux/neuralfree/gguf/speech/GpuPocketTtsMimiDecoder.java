package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.ops.audio.SpeechKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Device executor of {@link PocketTtsMimiDecoder}: the same windowed decode graph as
 * {@link PocketTtsMimiSession}, run as kernels on the engine's queue over the clip weights the
 * {@link SpeechDeviceWeights} uploaded.
 *
 * <p>A call denormalizes and projects the latents, runs the depthwise transposed upsample, the sliding-window
 * transformer (LayerNorm, NORMAL RoPE, banded attention over the persisted key/value window with
 * {@link SpeechKernels#TTS_WINDOW_ATTN}) and the SEANet stack (ELU, causal and transposed
 * convolutions with their persisted contexts, the dilated residual units). The attention prefix, RoPE position and
 * every convolution's left context or overlap tail live in device buffers, so consecutive windows concatenate to the
 * samples of the whole sequence. The final clamp to {@code [-1, 1]} runs on the host after the readback.
 *
 * <p>Owns its activation and state buffers; not thread-safe.
 */
final class GpuPocketTtsMimiDecoder implements PocketTtsMimiDecoder.Decoder {

    /** Left context or overlap tail of one causal convolution, double-buffered across calls. */
    private final class ConvState {
        private GpuBuffer cur;
        private GpuBuffer next;
        private final boolean active;

        ConvState(String name, int rows, int channels) {
            this.active = rows > 0 && channels > 0;
            if (active) {
                cur = own(name + ".cur", (long) rows * channels);
                next = own(name + ".next", (long) rows * channels);
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

    /** Device tensors and buffers of one SEANet stage. */
    private final class StageDevice {
        final int k;
        final int stride;
        final int inChannels;
        final int outChannels;
        final int resK;
        final GpuBuffer scaleW;
        final GpuBuffer scaleB;
        final GpuBuffer res1W;
        final GpuBuffer res1B;
        final GpuBuffer res2W;
        final GpuBuffer res2B;
        final ConvState scaleState;
        final ConvState resState;
        final GpuBuffer elu;
        final GpuBuffer scaleOut;
        final GpuBuffer resElu1;
        final GpuBuffer resConv1;
        final GpuBuffer resElu2;
        final GpuBuffer resConv2;
        final GpuBuffer resOut;

        StageDevice(PocketTtsMimiDecoder.Stage src, int index, int maxRows) {
            this.k = src.scaleConvK;
            this.stride = src.stride;
            this.inChannels = src.scaleInChannels;
            this.outChannels = src.outChannels;
            this.resK = src.resConv1K;
            String prefix = PocketTtsMimiDecoder.stageTensor(index, "");
            this.scaleW = device.f32(prefix + "scale_conv.weight", src.scaleConvW);
            this.scaleB = device.f32(prefix + "scale_conv.bias", src.scaleConvB);
            this.res1W = device.f32(prefix + "res_conv1.weight", src.resConv1W);
            this.res1B = device.f32(prefix + "res_conv1.bias", src.resConv1B);
            this.res2W = device.f32(prefix + "res_conv2.weight", src.resConv2W);
            this.res2B = device.f32(prefix + "res_conv2.bias", src.resConv2B);
            this.scaleState = new ConvState(prefix + "scale_conv", k - stride, outChannels);
            this.resState = new ConvState(prefix + "res_conv1", resK - 1, outChannels);
            this.elu = own(prefix + "elu", (long) maxRows * inChannels);
            this.scaleOut = own(prefix + "scale", (long) maxRows * outChannels);
            this.resElu1 = own(prefix + "res.elu1", (long) maxRows * outChannels);
            this.resConv1 = own(prefix + "res.conv1", (long) maxRows * outChannels);
            this.resElu2 = own(prefix + "res.elu2", (long) maxRows * outChannels);
            this.resConv2 = own(prefix + "res.conv2", (long) maxRows * outChannels);
            this.resOut = own(prefix + "res.out", (long) maxRows * outChannels);
        }
    }

    /** Device tensors of one transformer layer. */
    private static final class LayerDevice {
        final GpuWeight ln1;
        final GpuWeight ln1B;
        final GpuWeight ln2;
        final GpuWeight ln2B;
        final GpuWeight q;
        final GpuWeight k;
        final GpuWeight v;
        final GpuWeight out;
        final GpuWeight up;
        final GpuWeight down;
        final GpuWeight ls1;
        final GpuWeight ls2;

        LayerDevice(GpuWeight ln1, GpuWeight ln1B, GpuWeight ln2, GpuWeight ln2B, GpuWeight q, GpuWeight k,
                    GpuWeight v, GpuWeight out, GpuWeight up, GpuWeight down, GpuWeight ls1, GpuWeight ls2) {
            this.ln1 = ln1;
            this.ln1B = ln1B;
            this.ln2 = ln2;
            this.ln2B = ln2B;
            this.q = q;
            this.k = k;
            this.v = v;
            this.out = out;
            this.up = up;
            this.down = down;
            this.ls1 = ls1;
            this.ls2 = ls2;
        }
    }

    private final PocketTtsMimiDecoder geo;
    private final SpeechDeviceWeights device;
    private final GpuOps ops;
    private final int latent;
    private final int width;
    private final int nLayer;
    private final int nHead;
    private final int nHeadKv;
    private final int dHead;
    private final int qCh;
    private final int kvCh;
    private final int nFf;
    private final float eps;
    private final int prefix;
    private final int context;
    private final int windowFrames;
    private final int downsample;
    private final int hop;
    private final float ropeScale;
    private final int maxPos;
    private final int maxRows;

    private final GpuWeight quantOut;
    private final GpuBuffer embMean;
    private final GpuBuffer embStd;
    private final GpuBuffer upsampleW;
    private final int upsampleK;
    private final GpuBuffer convInW;
    private final GpuBuffer convInB;
    private final int convInK;
    private final int convInChannels;
    private final GpuBuffer convOutW;
    private final GpuBuffer convOutB;
    private final int convOutK;
    private final int lastChannels;
    private final GpuBuffer zeroBias;
    private final ConvState upsampleState;
    private final ConvState convInState;
    private final ConvState convOutState;
    private final StageDevice[] stages;
    private final LayerDevice[] layers;

    private final GpuBuffer frames;
    private final GpuBuffer feats;
    private final GpuBuffer quantized;
    private final GpuBuffer upsampled;
    private final GpuBuffer normed;
    private final GpuBuffer qRaw;
    private final GpuBuffer q;
    private final GpuBuffer kRaw;
    private final GpuBuffer k;
    private final GpuBuffer v;
    private final GpuBuffer attn;
    private final GpuBuffer projected;
    private final GpuBuffer tfmA;
    private final GpuBuffer tfmB;
    private final GpuBuffer ffUp;
    private final GpuBuffer ffDown;
    private final GpuBuffer[] keys;
    private final GpuBuffer[] values;
    private final GpuBuffer[] keysNext;
    private final GpuBuffer[] valuesNext;
    private final GpuBuffer convIn;
    private final GpuBuffer eluLast;
    private final GpuBuffer pcm;
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final float[] framesHost;
    private final float[] pcmHost;
    private int tfmPos;
    private boolean closed;

    GpuPocketTtsMimiDecoder(PocketTtsMimiDecoder geo, SpeechDeviceWeights device) {
        this.geo = Objects.requireNonNull(geo, "geo");
        this.device = Objects.requireNonNull(device, "device");
        this.ops = device.ops();
        this.latent = geo.latent();
        this.width = geo.width();
        this.nLayer = geo.nLayer();
        this.nHead = geo.nHead();
        this.nHeadKv = geo.nHeadKv();
        this.dHead = geo.dHead();
        this.qCh = nHead * dHead;
        this.kvCh = nHeadKv * dHead;
        this.nFf = geo.nFf();
        this.eps = geo.eps();
        this.context = geo.context();
        this.prefix = context - 1;
        this.windowFrames = geo.windowFrames();
        this.downsample = PocketTtsMimiDecoder.DOWNSAMPLE;
        this.hop = geo.hop();
        this.ropeScale = (float) Math.pow(PocketTtsMimiDecoder.ROPE_THETA, -2.0 / dHead);
        this.maxPos = windowFrames * downsample;
        this.maxRows = windowFrames * hop;

        this.quantOut = device.weight(PocketTtsMimiDecoder.QUANT_OUT + ".weight");
        requireMatrix(quantOut, width, latent, PocketTtsMimiDecoder.QUANT_OUT);
        this.embMean = device.f32("a.gen.emb_mean", geo.embMean());
        this.embStd = device.f32("a.gen.emb_std", geo.embStd());
        this.upsampleW = device.f32(PocketTtsMimiDecoder.UPSAMPLE + ".weight", geo.upsampleW());
        this.upsampleK = geo.upsampleK();
        this.convInW = device.f32(PocketTtsMimiDecoder.SEANET + "conv_in.weight", geo.convInW());
        this.convInB = device.f32(PocketTtsMimiDecoder.SEANET + "conv_in.bias", geo.convInB());
        this.convInK = geo.convInK();
        this.convInChannels = geo.convInChannels();
        this.convOutW = device.f32(PocketTtsMimiDecoder.SEANET + "conv_out.weight", geo.convOutW());
        this.convOutB = device.f32(PocketTtsMimiDecoder.SEANET + "conv_out.bias", geo.convOutB());
        this.convOutK = geo.convOutK();
        this.lastChannels = geo.stages()[geo.stages().length - 1].outChannels;
        this.zeroBias = device.f32("pockettts.mimi.zero_bias", new float[width]);
        this.upsampleState = new ConvState("up", upsampleK - downsample, width);
        this.convInState = new ConvState("dec.in", convInK - 1, convInChannels);
        this.convOutState = new ConvState("dec.out", convOutK - 1, lastChannels);

        PocketTtsMimiDecoder.Stage[] sourceStages = geo.stages();
        this.stages = new StageDevice[sourceStages.length];
        for (int i = 0; i < sourceStages.length; i++) {
            stages[i] = new StageDevice(sourceStages[i], i, maxRows);
        }

        PocketTtsMimiDecoder.TransformerLayer[] sourceLayers = geo.layers();
        this.layers = new LayerDevice[sourceLayers.length];
        for (int il = 0; il < sourceLayers.length; il++) {
            PocketTtsMimiDecoder.TransformerLayer l = sourceLayers[il];
            this.layers[il] = new LayerDevice(
                    device.f32Weight(PocketTtsMimiDecoder.layerTensor(il, "ln1.weight"), l.ln1W),
                    device.f32Weight(PocketTtsMimiDecoder.layerTensor(il, "ln1.bias"), l.ln1B),
                    device.f32Weight(PocketTtsMimiDecoder.layerTensor(il, "ln2.weight"), l.ln2W),
                    device.f32Weight(PocketTtsMimiDecoder.layerTensor(il, "ln2.bias"), l.ln2B),
                    device.weight(PocketTtsMimiDecoder.layerTensor(il, "attn_q.weight")),
                    device.weight(PocketTtsMimiDecoder.layerTensor(il, "attn_k.weight")),
                    device.weight(PocketTtsMimiDecoder.layerTensor(il, "attn_v.weight")),
                    device.weight(PocketTtsMimiDecoder.layerTensor(il, "attn_out.weight")),
                    device.weight(PocketTtsMimiDecoder.layerTensor(il, "ffn_up.weight")),
                    device.weight(PocketTtsMimiDecoder.layerTensor(il, "ffn_down.weight")),
                    device.f32Weight(PocketTtsMimiDecoder.layerTensor(il, "ls1.weight"), l.ls1),
                    device.f32Weight(PocketTtsMimiDecoder.layerTensor(il, "ls2.weight"), l.ls2));
            requireMatrix(layers[il].q, qCh, width, PocketTtsMimiDecoder.layerTensor(il, "attn_q"));
            requireMatrix(layers[il].k, kvCh, width, PocketTtsMimiDecoder.layerTensor(il, "attn_k"));
            requireMatrix(layers[il].v, kvCh, width, PocketTtsMimiDecoder.layerTensor(il, "attn_v"));
            requireMatrix(layers[il].out, width, qCh, PocketTtsMimiDecoder.layerTensor(il, "attn_out"));
            requireMatrix(layers[il].up, nFf, width, PocketTtsMimiDecoder.layerTensor(il, "ffn_up"));
            requireMatrix(layers[il].down, width, nFf, PocketTtsMimiDecoder.layerTensor(il, "ffn_down"));
        }

        this.frames = own("frames", (long) windowFrames * latent);
        this.feats = own("feats", (long) windowFrames * latent);
        this.quantized = own("quantized", (long) windowFrames * width);
        this.upsampled = own("upsampled", (long) maxPos * width);
        this.normed = own("normed", (long) maxPos * width);
        this.qRaw = own("q.raw", (long) maxPos * qCh);
        this.q = own("q", (long) maxPos * qCh);
        this.kRaw = own("k.raw", (long) maxPos * kvCh);
        this.k = own("k", (long) maxPos * kvCh);
        this.v = own("v", (long) maxPos * kvCh);
        this.attn = own("attn", (long) maxPos * qCh);
        this.projected = own("projected", (long) maxPos * width);
        this.tfmA = own("tfm.a", (long) maxPos * width);
        this.tfmB = own("tfm.b", (long) maxPos * width);
        this.ffUp = own("ffn.up", (long) maxPos * nFf);
        this.ffDown = own("ffn.down", (long) maxPos * width);
        this.keys = new GpuBuffer[nLayer];
        this.values = new GpuBuffer[nLayer];
        this.keysNext = new GpuBuffer[nLayer];
        this.valuesNext = new GpuBuffer[nLayer];
        long window = (long) (prefix + maxPos) * kvCh;
        for (int il = 0; il < nLayer; il++) {
            keys[il] = own("tfm.k." + il, window);
            values[il] = own("tfm.v." + il, window);
            keysNext[il] = own("tfm.k.next." + il, window);
            valuesNext[il] = own("tfm.v.next." + il, window);
        }
        this.convIn = own("dec.in.out", (long) maxRows * convInChannels);
        this.eluLast = own("dec.last.elu", (long) maxRows * lastChannels);
        this.pcm = own("pcm", (long) windowFrames * hop);
        this.framesHost = new float[windowFrames * latent];
        this.pcmHost = new float[windowFrames * hop];
    }

    private GpuBuffer own(String name, long floats) {
        GpuBuffer buffer = device.allocate("mimi." + name, floats);
        owned.add(buffer);
        return buffer;
    }

    private static void requireMatrix(GpuWeight w, long rows, long cols, String what) {
        if (w.rows() != rows || w.cols() != cols) {
            throw new IllegalArgumentException("pockettts mimi tensor " + what + " is " + w.rows() + " x "
                    + w.cols() + " on the device, the graph needs " + rows + " x " + cols);
        }
    }

    @Override
    public float[] decode(float[][] window) {
        ensureOpen();
        Objects.requireNonNull(window, "frames");
        int n = window.length;
        if (n < 1 || n > windowFrames) {
            throw new IllegalArgumentException("a mimi batch holds 1 to " + windowFrames + " frames, got " + n);
        }
        for (int f = 0; f < n; f++) {
            if (window[f] == null || window[f].length != latent) {
                throw new IllegalArgumentException("frame " + f + " holds "
                        + (window[f] == null ? "no" : window[f].length) + " values, a latent holds " + latent);
            }
            for (int i = 0; i < latent; i++) {
                if (!Float.isFinite(window[f][i])) {
                    throw new IllegalArgumentException("frame " + f + " value " + i + " is " + window[f][i]);
                }
            }
            System.arraycopy(window[f], 0, framesHost, f * latent, latent);
        }
        device.writeFloats(frames, java.util.Arrays.copyOf(framesHost, n * latent));

        int nPos = n * downsample;
        quantize(n);
        depthwiseUpsample(n, nPos);
        transformer(nPos);
        seanet(nPos);
        device.readFloats(pcm, pcmHost, n * hop);
        float[] out = new float[n * hop];
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(-1f, Math.min(1f, pcmHost[i]));
        }
        return out;
    }

    /** The RoPE position of the next call's first upsampled frame. */
    @Override
    public int position() {
        return tfmPos;
    }

    @Override
    public int sampleRate() {
        return geo.sampleRate();
    }

    /** Denormalizes the latents by the checkpoint's mean and standard deviation, then projects them to the width. */
    private void quantize(int n) {
        ops.enqueue(SpeechKernels.ROW_AFFINE, new long[]{SpeechKernels.roundUp((long) n * latent)},
                new long[]{SpeechKernels.LOCAL}, n, latent, frames, embStd, embMean, feats);
        ops.matmul(quantOut, feats, 0, quantized, 0, n);
    }

    /** The depthwise transposed convolution to the encoder frame rate. */
    private void depthwiseUpsample(int n, int nPos) {
        ops.enqueue(SpeechKernels.CONV_TRANSPOSE1D_DEPTHWISE_CAUSAL,
                new long[]{SpeechKernels.depthwiseTransposeGlobal(n, width, upsampleK, downsample)},
                new long[]{SpeechKernels.LOCAL}, n, width, upsampleK, downsample, quantized, upsampleW,
                zeroBias, upsampleState.read(), upsampled, upsampleState.write());
        upsampleState.swap();
    }

    /** The sliding-window transformer over the upsampled frames, with the persisted key/value window. */
    private void transformer(int nPos) {
        int base = tfmPos;
        ops.copy(upsampled, 0, tfmA, 0, nPos * width);
        GpuBuffer input = tfmA;
        GpuBuffer output = tfmB;
        for (int il = 0; il < nLayer; il++) {
            LayerDevice l = layers[il];
            ops.norm(NormKind.LAYER, l.ln1, l.ln1B, eps, input, 0, normed, 0, nPos, width);
            ops.matmul(l.q, normed, 0, qRaw, 0, nPos);
            ops.matmul(l.k, normed, 0, kRaw, 0, nPos);
            ops.matmul(l.v, normed, 0, v, 0, nPos);
            ops.enqueue(SpeechKernels.NORMAL_ROPE_ROWS,
                    new long[]{SpeechKernels.ropeRowsGlobal(nPos, nHead, dHead)}, new long[]{SpeechKernels.LOCAL},
                    nPos, nHead, dHead, base, 0, ropeScale, qRaw, q);
            ops.enqueue(SpeechKernels.NORMAL_ROPE_ROWS,
                    new long[]{SpeechKernels.ropeRowsGlobal(nPos, nHeadKv, dHead)}, new long[]{SpeechKernels.LOCAL},
                    nPos, nHeadKv, dHead, base, 0, ropeScale, kRaw, k);
            ops.copy(k, 0, keys[il], prefix * kvCh, nPos * kvCh);
            ops.copy(v, 0, values[il], prefix * kvCh, nPos * kvCh);
            ops.enqueue(SpeechKernels.TTS_WINDOW_ATTN,
                    new long[]{SpeechKernels.windowAttentionGlobal(nPos, nHead)}, new long[]{SpeechKernels.LOCAL},
                    nPos, nHead, nHeadKv, dHead, prefix, context, base, q, keys[il], values[il], attn);
            ops.matmul(l.out, attn, 0, projected, 0, nPos);
            ops.enqueue(SpeechKernels.ROW_SCALE_ADD, new long[]{SpeechKernels.roundUp((long) nPos * width)},
                    new long[]{SpeechKernels.LOCAL}, nPos, width, 0, input, projected, l.ls1.shard(0), output);

            ops.norm(NormKind.LAYER, l.ln2, l.ln2B, eps, output, 0, normed, 0, nPos, width);
            ops.matmul(l.up, normed, 0, ffUp, 0, nPos);
            ops.unary(ActOp.GELU_TANH, ActParams.NONE, ffUp, 0, ffUp, 0, nPos * nFf);
            ops.matmul(l.down, ffUp, 0, ffDown, 0, nPos);
            ops.enqueue(SpeechKernels.ROW_SCALE_ADD, new long[]{SpeechKernels.roundUp((long) nPos * width)},
                    new long[]{SpeechKernels.LOCAL}, nPos, width, 0, output, ffDown, l.ls2.shard(0), input);
            rollWindow(keys[il], keysNext[il], nPos);
            rollWindow(values[il], valuesNext[il], nPos);
            GpuBuffer t = keys[il];
            keys[il] = keysNext[il];
            keysNext[il] = t;
            t = values[il];
            values[il] = valuesNext[il];
            valuesNext[il] = t;
        }
        tfmPos += nPos;
        // the last layer wrote into input; copy it where the SEANet reads when it is not tfmA
        if (input != tfmA) {
            ops.copy(input, 0, tfmA, 0, nPos * width);
        }
    }

    /** Keeps the last {@code prefix} rows of the window as the next call's prefix. */
    private void rollWindow(GpuBuffer all, GpuBuffer next, int nPos) {
        ops.copy(all, nPos * kvCh, next, 0, prefix * kvCh);
    }

    private void seanet(int nPos) {
        int rows = nPos;
        ops.enqueue(SpeechKernels.CONV1D_CAUSAL,
                new long[]{SpeechKernels.convGlobal(rows, convInChannels, convInChannels, convInK - 1)},
                new long[]{SpeechKernels.LOCAL}, rows, convInChannels, convInChannels, convInK, 1, convInK - 1,
                tfmA, convInW, convInB, convInState.read(), convIn, convInState.write());
        convInState.swap();
        GpuBuffer input = convIn;
        int channels = convInChannels;
        for (StageDevice stage : stages) {
            ops.enqueue(SpeechKernels.ELU, new long[]{SpeechKernels.roundUp((long) rows * channels)},
                    new long[]{SpeechKernels.LOCAL}, rows * channels, input, stage.elu);
            ops.enqueue(SpeechKernels.CONV_TRANSPOSE1D_CAUSAL,
                    new long[]{SpeechKernels.transposeGlobal(rows, stage.outChannels, stage.k, stage.stride)},
                    new long[]{SpeechKernels.LOCAL}, rows, stage.inChannels, stage.outChannels, stage.k,
                    stage.stride, stage.elu, stage.scaleW, stage.scaleB, stage.scaleState.read(), stage.scaleOut,
                    stage.scaleState.write());
            stage.scaleState.swap();
            rows *= stage.stride;
            ops.enqueue(SpeechKernels.ELU, new long[]{SpeechKernels.roundUp((long) rows * stage.outChannels)},
                    new long[]{SpeechKernels.LOCAL}, rows * stage.outChannels, stage.scaleOut, stage.resElu1);
            ops.enqueue(SpeechKernels.CONV1D_CAUSAL,
                    new long[]{SpeechKernels.convGlobal(rows, stage.outChannels, stage.outChannels, stage.resK - 1)},
                    new long[]{SpeechKernels.LOCAL}, rows, stage.outChannels, stage.outChannels, stage.resK, 1,
                    stage.resK - 1, stage.resElu1, stage.res1W, stage.res1B, stage.resState.read(),
                    stage.resConv1, stage.resState.write());
            stage.resState.swap();
            ops.enqueue(SpeechKernels.ELU, new long[]{SpeechKernels.roundUp((long) rows * stage.outChannels)},
                    new long[]{SpeechKernels.LOCAL}, rows * stage.outChannels, stage.resConv1, stage.resElu2);
            ops.enqueue(SpeechKernels.CONV1D_CAUSAL,
                    new long[]{SpeechKernels.convGlobal(rows, stage.outChannels, stage.outChannels, 0)},
                    new long[]{SpeechKernels.LOCAL}, rows, stage.outChannels, stage.outChannels, 1, 1, 0,
                    stage.resElu2, stage.res2W, stage.res2B, ops.placeholders().readOnlyFloats(), stage.resConv2,
                    ops.placeholders().scratchFloats());
            ops.add(stage.scaleOut, 0, stage.resConv2, 0, stage.resOut, 0, rows * stage.outChannels);
            input = stage.resOut;
            channels = stage.outChannels;
        }
        ops.enqueue(SpeechKernels.ELU, new long[]{SpeechKernels.roundUp((long) rows * channels)},
                new long[]{SpeechKernels.LOCAL}, rows * channels, input, eluLast);
        ops.enqueue(SpeechKernels.CONV1D_CAUSAL,
                new long[]{SpeechKernels.convGlobal(rows, lastChannels, 1, convOutK - 1)},
                new long[]{SpeechKernels.LOCAL}, rows, lastChannels, 1, convOutK, 1, convOutK - 1, eluLast, convOutW,
                convOutB, convOutState.read(), pcm, convOutState.write());
        convOutState.swap();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the pockettts GPU mimi decoder is closed");
        }
    }

    /** Releases this decoder's activation and state buffers. Idempotent. */
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
        return "GpuPocketTtsMimiDecoder{" + geo + ", position=" + tfmPos + '}';
    }
}
