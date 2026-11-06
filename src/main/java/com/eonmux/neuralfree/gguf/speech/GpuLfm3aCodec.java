package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.StreamingIstft;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.audio.SpeechKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Device executor of {@link Lfm3aCodec}: the same liquid-codec graph as {@link Lfm3aCodecSession}, run as kernels on
 * the engine's queue over the clip weights the {@link SpeechDeviceWeights} uploaded.
 *
 * <p>A call maps the codes through the FSQ level table and the two input projections, repeats each frame over the
 * decoder frame stack, runs the LFM2 decoder layers (RMS norms, the short convolution with its persisted taps, or
 * the gated sliding-window attention over the persisted key/value window with {@link SpeechKernels#TTS_WINDOW_ATTN}
 * and per-head sigmoid gates), and projects the output norm onto the log magnitudes and phases. The spectra are
 * read back for the shared host {@link StreamingIstft}; the position counter, convolution taps and attention
 * windows live in device buffers.
 *
 * <p>Activation buffers are allocated per call, so a batch of any length is accepted; the persisted state buffers
 * live until the decoder closes. Not thread-safe.
 */
final class GpuLfm3aCodec implements Lfm3aCodec.Decoder {

    /** Left context of one short-convolution layer, double-buffered across calls. */
    private final class ConvState {
        private GpuBuffer cur;
        private GpuBuffer next;
        private final boolean active;

        ConvState(String name, int rows, int channels) {
            this.active = rows > 0 && channels > 0;
            if (active) {
                this.cur = state(name + ".cur", (long) rows * channels);
                this.next = state(name + ".next", (long) rows * channels);
            }
        }

        void reset(GpuOps ops) {
            if (active) {
                ops.fill(cur, 0, cur.sizeInBytes(), 0);
                ops.fill(next, 0, next.sizeInBytes(), 0);
            }
        }

        GpuBuffer read() {
            return active ? cur : null;
        }

        GpuBuffer write() {
            return active ? next : null;
        }

        void swap() {
            if (active) {
                GpuBuffer t = cur;
                cur = next;
                next = t;
            }
        }
    }

    /** Device tensors of one attention layer. */
    private static final class AttnWeights {
        final int nHead;
        final int nHeadKv;
        final int dHead;
        final int qWidth;
        final int kvWidth;
        final GpuWeight q;
        final GpuWeight qNorm;
        final GpuWeight k;
        final GpuWeight kNorm;
        final GpuWeight v;
        final GpuWeight out;
        final GpuWeight gate;
        GpuBuffer prefixK;
        GpuBuffer prefixV;
        GpuBuffer nextK;
        GpuBuffer nextV;
        float thetaScale;

        AttnWeights(int nHead, int nHeadKv, int dHead, GpuWeight q, GpuWeight qNorm, GpuWeight k, GpuWeight kNorm,
                    GpuWeight v, GpuWeight out, GpuWeight gate, GpuBuffer prefixK, GpuBuffer prefixV,
                    GpuBuffer nextK, GpuBuffer nextV) {
            this.nHead = nHead;
            this.nHeadKv = nHeadKv;
            this.dHead = dHead;
            this.qWidth = nHead * dHead;
            this.kvWidth = nHeadKv * dHead;
            this.q = q;
            this.qNorm = qNorm;
            this.k = k;
            this.kNorm = kNorm;
            this.v = v;
            this.out = out;
            this.gate = gate;
            this.prefixK = prefixK;
            this.prefixV = prefixV;
            this.nextK = nextK;
            this.nextV = nextV;
        }
    }

    /** Device tensors of one short-convolution layer. */
    private static final class ConvWeights {
        final int taps;
        final GpuWeight pw1;
        final GpuBuffer dw;
        final GpuWeight pw2;
        final ConvState state;

        ConvWeights(int taps, GpuWeight pw1, GpuBuffer dw, GpuWeight pw2, ConvState state) {
            this.taps = taps;
            this.pw1 = pw1;
            this.dw = dw;
            this.pw2 = pw2;
            this.state = state;
        }
    }

    /** Device tensors shared by both layer kinds. */
    private static final class Common {
        final GpuWeight ln1;
        final GpuWeight ln2;
        final GpuWeight gate;
        final GpuWeight up;
        final GpuWeight down;
        final GpuWeight attnPostNorm;
        final GpuWeight ffPostNorm;

        Common(GpuWeight ln1, GpuWeight ln2, GpuWeight gate, GpuWeight up, GpuWeight down,
               GpuWeight attnPostNorm, GpuWeight ffPostNorm) {
            this.ln1 = ln1;
            this.ln2 = ln2;
            this.gate = gate;
            this.up = up;
            this.down = down;
            this.attnPostNorm = attnPostNorm;
            this.ffPostNorm = ffPostNorm;
        }
    }

    private final Lfm3aCodec geo;
    private final SpeechDeviceWeights device;
    private final GpuOps ops;
    private final int nCb;
    private final int nLvl;
    private final int vocab;
    private final int width;
    private final int frameStack;
    private final int window;
    private final int prefix;
    private final int nFft;
    private final int hop;
    private final int nBins;
    private final float eps;
    private final StreamingIstft istft;
    private final GpuBuffer zeroBias;

    private final GpuWeight quantFirstCb;
    private final GpuWeight quantFirstOut;
    private final GpuBuffer quantFirstOutB;
    private final GpuWeight tfmIn;
    private final GpuBuffer tfmInB;
    private final GpuWeight tfmOut;
    private final GpuBuffer tfmOutB;
    private final GpuWeight tfmOutNorm;
    private final Common[] common;
    private final ConvWeights[] conv;
    private final AttnWeights[] attn;

    private final List<GpuBuffer> states = new ArrayList<>();
    private int pos;
    private boolean pendingTail;
    private boolean closed;

    GpuLfm3aCodec(Lfm3aCodec geo, SpeechDeviceWeights device) {
        this.geo = Objects.requireNonNull(geo, "geo");
        this.device = Objects.requireNonNull(device, "device");
        this.ops = device.ops();
        this.nCb = geo.nCodebooks();
        this.nLvl = geo.nLvl();
        this.vocab = geo.vocab();
        this.width = geo.nEmbdDec();
        this.frameStack = geo.frameStack();
        this.window = geo.window();
        this.prefix = window - 1;
        this.nFft = geo.nFft();
        this.hop = geo.hop();
        this.nBins = geo.nBins();
        this.eps = Lfm3aCodec.EPS;
        this.istft = new StreamingIstft(nFft, hop);
        this.zeroBias = device.f32("lfm3a.codec.zero_bias", new float[width]);

        this.quantFirstCb = device.weight(Lfm3aCodec.QUANT_FIRST_CB + ".weight");
        this.quantFirstOut = device.weight(Lfm3aCodec.QUANT_FIRST_OUT + ".weight");
        this.quantFirstOutB = device.f32(Lfm3aCodec.QUANT_FIRST_OUT + ".bias", geo.quantFirstOutBiasValues());
        this.tfmIn = device.weight(Lfm3aCodec.TFM_IN_PROJ + ".weight");
        this.tfmInB = device.f32(Lfm3aCodec.TFM_IN_PROJ + ".bias", geo.tfmInBiasValues());
        this.tfmOut = device.weight(Lfm3aCodec.TFM_OUT_PROJ + ".weight");
        this.tfmOutB = device.f32(Lfm3aCodec.TFM_OUT_PROJ + ".bias", geo.tfmOutBiasValues());
        this.tfmOutNorm = device.f32Weight(Lfm3aCodec.TFM_OUTPUT_NORM + ".weight", geo.tfmOutNormValues());
        requireMatrix(quantFirstOut, width, (long) nLvl * nCb, Lfm3aCodec.QUANT_FIRST_OUT);
        requireMatrix(tfmIn, width, width, Lfm3aCodec.TFM_IN_PROJ);
        requireMatrix(tfmOut, nFft + 2, width, Lfm3aCodec.TFM_OUT_PROJ);

        Lfm3aCodec.Layer[] layers = geo.layers();
        this.common = new Common[layers.length];
        this.conv = new ConvWeights[layers.length];
        this.attn = new AttnWeights[layers.length];
        for (int il = 0; il < layers.length; il++) {
            Lfm3aCodec.Layer l = layers[il];
            String tensorBase = Lfm3aCodec.layerTensor(il, "");
            common[il] = new Common(
                    device.f32Weight(tensorBase + Lfm3aCodec.LN1, l.ln1),
                    device.f32Weight(tensorBase + Lfm3aCodec.LN2, l.ln2),
                    device.weight(tensorBase + Lfm3aCodec.FFN_GATE),
                    device.weight(tensorBase + Lfm3aCodec.FFN_UP),
                    device.weight(tensorBase + Lfm3aCodec.FFN_DOWN),
                    l.attnPostNorm == null ? null
                            : device.f32Weight(tensorBase + Lfm3aCodec.ATTN_POST_NORM, l.attnPostNorm),
                    l.ffPostNorm == null ? null
                            : device.f32Weight(tensorBase + Lfm3aCodec.FFN_POST_NORM, l.ffPostNorm));
            if (l.isConv()) {
                float[] dw = geo.weights().floats(tensorBase + Lfm3aCodec.CONV_DW);
                conv[il] = new ConvWeights(l.taps,
                        device.weight(tensorBase + Lfm3aCodec.CONV_PW1),
                        device.f32(tensorBase + Lfm3aCodec.CONV_DW, dw),
                        device.weight(tensorBase + Lfm3aCodec.CONV_PW2),
                        new ConvState("lfm3a.conv." + il, l.taps - 1, width));
                requireMatrix(conv[il].pw1, 3L * width, width, tensorBase + Lfm3aCodec.CONV_PW1);
                requireMatrix(conv[il].pw2, width, width, tensorBase + Lfm3aCodec.CONV_PW2);
            } else {
                int kvWidth = l.nHeadKv * l.dHead;
                int qWidth = l.nHead * l.dHead;
                GpuBuffer prefixK = state("lfm3a.k." + il, prefix * kvWidth);
                GpuBuffer prefixV = state("lfm3a.v." + il, prefix * kvWidth);
                GpuBuffer nextK = state("lfm3a.k.next." + il, prefix * kvWidth);
                GpuBuffer nextV = state("lfm3a.v.next." + il, prefix * kvWidth);
                GpuWeight q = device.weight(tensorBase + Lfm3aCodec.ATTN_Q);
                GpuWeight k = device.weight(tensorBase + Lfm3aCodec.ATTN_K);
                GpuWeight v = device.weight(tensorBase + Lfm3aCodec.ATTN_V);
                GpuWeight out = device.weight(tensorBase + Lfm3aCodec.ATTN_OUT);
                GpuWeight gate = device.weight(tensorBase + Lfm3aCodec.ATTN_GATE);
                requireMatrix(q, qWidth, width, tensorBase + Lfm3aCodec.ATTN_Q);
                requireMatrix(k, kvWidth, width, tensorBase + Lfm3aCodec.ATTN_K);
                requireMatrix(v, kvWidth, width, tensorBase + Lfm3aCodec.ATTN_V);
                requireMatrix(out, width, qWidth, tensorBase + Lfm3aCodec.ATTN_OUT);
                requireMatrix(gate, l.nHead, width, tensorBase + Lfm3aCodec.ATTN_GATE);
                attn[il] = new AttnWeights(l.nHead, l.nHeadKv, l.dHead, q,
                        device.f32Weight(tensorBase + Lfm3aCodec.ATTN_Q_NORM, l.qNorm), k,
                        device.f32Weight(tensorBase + Lfm3aCodec.ATTN_K_NORM, l.kNorm), v, out, gate,
                        prefixK, prefixV, nextK, nextV);
                attn[il].thetaScale = (float) Math.pow(Lfm3aCodec.ROPE_THETA, -2.0 / l.dHead);
            }
        }
    }

    private GpuBuffer state(String name, long floats) {
        GpuBuffer buffer = device.allocate(name, Math.max(floats, 1));
        states.add(buffer);
        return buffer;
    }

    private static void requireMatrix(GpuWeight w, long rows, long cols, String what) {
        if (w.rows() != rows || w.cols() != cols) {
            throw new IllegalArgumentException("lfm3a codec tensor " + what + " is " + w.rows() + " x "
                    + w.cols() + " on the device, the graph needs " + rows + " x " + cols);
        }
    }

    @Override
    public float[] decodePolar(int[] codes) {
        ensureOpen();
        Objects.requireNonNull(codes, "codes");
        if (codes.length == 0 || codes.length % nCb != 0) {
            throw new IllegalArgumentException("the codes must form whole frames of " + nCb + ", got "
                    + codes.length);
        }
        for (int c : codes) {
            if (c < 0 || c >= vocab) {
                throw new IllegalArgumentException("code " + c + " is outside the FSQ vocabulary [0, " + vocab
                        + ")");
            }
        }
        int nFrames = codes.length / nCb;
        int nPos = nFrames * frameStack;
        int base = pos;
        List<GpuBuffer> act = new ArrayList<>();
        try {
            GpuBuffer levels = alloc(act, "levels", (long) nFrames * nCb * nLvl);
            ops.embed(quantFirstCb, codes, levels, 0);
            GpuBuffer projected = alloc(act, "quant", (long) nFrames * width);
            ops.matmul(quantFirstOut, levels, 0, projected, 0, nFrames);
            bias(projected, quantFirstOutB, width, nFrames);
            GpuBuffer projected2 = alloc(act, "quant.in", (long) nFrames * width);
            ops.matmul(tfmIn, projected, 0, projected2, 0, nFrames);
            bias(projected2, tfmInB, width, nFrames);
            GpuBuffer inpL = alloc(act, "inp", (long) nPos * width);
            ops.enqueue(SpeechKernels.REPEAT_ROWS, new long[]{SpeechKernels.roundUp((long) nPos * width)},
                    new long[]{SpeechKernels.LOCAL}, nFrames, frameStack, width, projected2, inpL);

            GpuBuffer normed = alloc(act, "normed", (long) nPos * width);
            GpuBuffer result = alloc(act, "result", (long) nPos * width);
            for (int il = 0; il < common.length; il++) {
                Common c = common[il];
                ops.norm(NormKind.RMS, c.ln1, null, eps, inpL, 0, normed, 0, nPos, width);
                if (conv[il] != null) {
                    shortConv(act, conv[il], normed, result, nPos);
                } else {
                    windowAttention(act, attn[il], normed, result, nPos, base);
                }
                if (c.attnPostNorm != null) {
                    ops.norm(NormKind.RMS, c.attnPostNorm, null, eps, result, 0, result, 0, nPos, width);
                }
                ops.add(inpL, 0, result, 0, inpL, 0, nPos * width);
                ops.norm(NormKind.RMS, c.ln2, null, eps, inpL, 0, normed, 0, nPos, width);
                ffn(act, c, normed, result, nPos);
                if (c.ffPostNorm != null) {
                    ops.norm(NormKind.RMS, c.ffPostNorm, null, eps, result, 0, result, 0, nPos, width);
                }
                ops.add(inpL, 0, result, 0, inpL, 0, nPos * width);
            }

            int outWidth = nFft + 2;
            GpuBuffer outRow = alloc(act, "out.row", (long) nPos * width);
            GpuBuffer feats = alloc(act, "feats", (long) nPos * outWidth);
            ops.norm(NormKind.RMS, tfmOutNorm, null, eps, inpL, 0, outRow, 0, nPos, width);
            ops.matmul(tfmOut, outRow, 0, feats, 0, nPos);
            bias(feats, tfmOutB, outWidth, nPos);
            float[] host = new float[nPos * outWidth];
            device.readFloats(feats, host, host.length);
            pos += nPos;
            pendingTail = true;
            return host;
        } finally {
            closeAll(act);
        }
    }

    /** The FSQ levels through the two projections, the short convolution with its persisted taps. */
    private void shortConv(List<GpuBuffer> act, ConvWeights cw, GpuBuffer normed, GpuBuffer result, int nPos) {
        GpuBuffer v = alloc(act, "conv.v", (long) nPos * 3 * width);
        ops.matmul(cw.pw1, normed, 0, v, 0, nPos);
        GpuBuffer bx = alloc(act, "conv.bx", (long) nPos * width);
        ops.glu(GluOp.MUL, ActParams.NONE, v, 0, 3 * width, v, 2 * width, 3 * width, bx, 0, width, nPos, width);
        GpuBuffer conv = alloc(act, "conv.out", (long) nPos * width);
        ops.enqueue(SpeechKernels.CONV1D_DEPTHWISE_CAUSAL,
                new long[]{SpeechKernels.depthwiseGlobal(nPos, width, cw.taps - 1)},
                new long[]{SpeechKernels.LOCAL}, nPos, width, cw.taps, cw.taps - 1, bx, cw.dw, zeroBias,
                cw.state.read(), conv, cw.state.write());
        cw.state.swap();
        ops.glu(GluOp.MUL, ActParams.NONE, v, width, 3 * width, conv, 0, width, result, 0, width, nPos, width);
        GpuBuffer pw2 = alloc(act, "conv.pw2", (long) nPos * width);
        ops.matmul(cw.pw2, result, 0, pw2, 0, nPos);
        ops.copy(pw2, 0, result, 0, nPos * width);
    }

    /** The sliding-window gated attention, its key/value window advanced. */
    private void windowAttention(List<GpuBuffer> act, AttnWeights aw, GpuBuffer normed, GpuBuffer result, int nPos,
                                 int base) {
        GpuBuffer qRaw = alloc(act, "q.raw", (long) nPos * aw.qWidth);
        GpuBuffer q = alloc(act, "q", (long) nPos * aw.qWidth);
        GpuBuffer kRaw = alloc(act, "k.raw", (long) nPos * aw.kvWidth);
        GpuBuffer k = alloc(act, "k", (long) nPos * aw.kvWidth);
        GpuBuffer v = alloc(act, "v", (long) nPos * aw.kvWidth);
        ops.matmul(aw.q, normed, 0, qRaw, 0, nPos);
        ops.matmul(aw.k, normed, 0, kRaw, 0, nPos);
        ops.matmul(aw.v, normed, 0, v, 0, nPos);
        ops.enqueue(SpeechKernels.RMS_NORM_ROPE_ROWS,
                new long[]{SpeechKernels.ropeRowsGlobal(nPos, aw.nHead, aw.dHead)}, new long[]{SpeechKernels.LOCAL},
                nPos, aw.nHead, aw.dHead, base, 0, aw.thetaScale, eps, qRaw, aw.qNorm.shard(0), q);
        ops.enqueue(SpeechKernels.RMS_NORM_ROPE_ROWS,
                new long[]{SpeechKernels.ropeRowsGlobal(nPos, aw.nHeadKv, aw.dHead)},
                new long[]{SpeechKernels.LOCAL}, nPos, aw.nHeadKv, aw.dHead, base, 0, aw.thetaScale, eps, kRaw,
                aw.kNorm.shard(0), k);
        int total = prefix + nPos;
        GpuBuffer keys = alloc(act, "keys", (long) total * aw.kvWidth);
        GpuBuffer values = alloc(act, "values", (long) total * aw.kvWidth);
        ops.copy(aw.prefixK, 0, keys, 0, prefix * aw.kvWidth);
        ops.copy(aw.prefixV, 0, values, 0, prefix * aw.kvWidth);
        ops.copy(k, 0, keys, prefix * aw.kvWidth, nPos * aw.kvWidth);
        ops.copy(v, 0, values, prefix * aw.kvWidth, nPos * aw.kvWidth);
        GpuBuffer attnOut = alloc(act, "attn", (long) nPos * aw.qWidth);
        ops.enqueue(SpeechKernels.TTS_WINDOW_ATTN,
                new long[]{SpeechKernels.windowAttentionGlobal(nPos, aw.nHead)}, new long[]{SpeechKernels.LOCAL},
                nPos, aw.nHead, aw.nHeadKv, aw.dHead, prefix, window, base, q, keys, values, attnOut);
        GpuBuffer gate = alloc(act, "gate", (long) nPos * aw.nHead);
        ops.matmul(aw.gate, normed, 0, gate, 0, nPos);
        GpuBuffer gated = alloc(act, "attn.gated", (long) nPos * aw.qWidth);
        ops.enqueue(SpeechKernels.HEAD_SIGMOID_GATE, new long[]{SpeechKernels.headGateGlobal(nPos, aw.nHead)},
                new long[]{SpeechKernels.LOCAL}, nPos, aw.nHead, aw.dHead, attnOut, gate, gated);
        ops.matmul(aw.out, gated, 0, result, 0, nPos);
        // keep the last prefix rows of [cache | chunk]
        ops.copy(keys, nPos * aw.kvWidth, aw.nextK, 0, prefix * aw.kvWidth);
        ops.copy(values, nPos * aw.kvWidth, aw.nextV, 0, prefix * aw.kvWidth);
        swap(aw);
    }

    private static void swap(AttnWeights aw) {
        GpuBuffer t = aw.prefixK;
        aw.prefixK = aw.nextK;
        aw.nextK = t;
        t = aw.prefixV;
        aw.prefixV = aw.nextV;
        aw.nextV = t;
    }

    /** The SwiGLU FFN over every position. */
    private void ffn(List<GpuBuffer> act, Common c, GpuBuffer x, GpuBuffer out, int nPos) {
        int ff = (int) c.gate.rows();
        GpuBuffer gate = alloc(act, "ffn.gate", (long) nPos * ff);
        GpuBuffer up = alloc(act, "ffn.up", (long) nPos * ff);
        ops.matmul(c.gate, x, 0, gate, 0, nPos);
        ops.matmul(c.up, x, 0, up, 0, nPos);
        GpuBuffer swiglu = alloc(act, "ffn.swiglu", (long) nPos * ff);
        ops.glu(GluOp.SWIGLU, ActParams.NONE, gate, 0, ff, up, 0, ff, swiglu, 0, ff, nPos, ff);
        ops.matmul(c.down, swiglu, 0, out, 0, nPos);
    }

    private void bias(GpuBuffer data, GpuBuffer bias, int width, int rows) {
        ops.enqueue(GpuOps.BATCH_BIAS_ADD, ActKernels.global1d(width, ops.local()), new long[]{ops.local()},
                width, rows, bias, data);
    }

    private GpuBuffer alloc(List<GpuBuffer> act, String name, long floats) {
        GpuBuffer buffer = device.allocate("lfm3a.codec." + name, Math.max(floats, 1));
        act.add(buffer);
        return buffer;
    }

    private static void closeAll(List<GpuBuffer> buffers) {
        RuntimeException failure = null;
        for (GpuBuffer b : buffers) {
            try {
                b.close();
            } catch (RuntimeException e) {
                failure = failure == null ? e : failure;
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public AudioClip decodeClip(int[] codes) {
        float[] feats = decodePolar(codes);
        int width = 2 * nBins;
        int nPos = feats.length / width;
        float[] spec = new float[width];
        float[] pcm = new float[nPos * hop];
        int at = 0;
        for (int p = 0; p < nPos; p++) {
            int off = p * width;
            for (int b = 0; b < nBins; b++) {
                float mag = (float) Math.exp(feats[off + b]);
                float phase = feats[off + nBins + b];
                spec[2 * b] = mag * (float) Math.cos(phase);
                spec[2 * b + 1] = mag * (float) Math.sin(phase);
            }
            float[] chunk = istft.processFrame(spec);
            if (at + chunk.length > pcm.length) {
                pcm = Arrays.copyOf(pcm, at + chunk.length);
            }
            System.arraycopy(chunk, 0, pcm, at, chunk.length);
            at += chunk.length;
        }
        return new AudioClip(Arrays.copyOf(pcm, at), geo.sampleRate());
    }

    @Override
    public AudioClip flushClip() {
        if (!pendingTail) {
            return new AudioClip(new float[0], geo.sampleRate());
        }
        float[] tail = istft.flush();
        int trim = (nFft - hop) / 2;
        int keep = tail.length - Math.min(tail.length, trim);
        float[] kept = new float[keep];
        System.arraycopy(tail, 0, kept, 0, keep);
        istft.reset();
        pendingTail = false;
        return new AudioClip(kept, geo.sampleRate());
    }

    @Override
    public int position() {
        return pos;
    }

    @Override
    public void reset() {
        pos = 0;
        pendingTail = false;
        istft.reset();
        for (ConvWeights cw : conv) {
            if (cw != null) {
                cw.state.reset(ops);
            }
        }
        for (AttnWeights aw : attn) {
            if (aw != null) {
                ops.fill(aw.prefixK, 0, aw.prefixK.sizeInBytes(), 0);
                ops.fill(aw.prefixV, 0, aw.prefixV.sizeInBytes(), 0);
                ops.fill(aw.nextK, 0, aw.nextK.sizeInBytes(), 0);
                ops.fill(aw.nextV, 0, aw.nextV.sizeInBytes(), 0);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the lfm3a GPU codec is closed");
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (GpuBuffer buffer : states) {
            try {
                buffer.close();
            } catch (RuntimeException e) {
                failure = failure == null ? e : failure;
            }
        }
        states.clear();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuLfm3aCodec{" + geo + ", position=" + pos + '}';
    }
}
