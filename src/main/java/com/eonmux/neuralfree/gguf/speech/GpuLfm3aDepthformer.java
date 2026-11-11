package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.audio.SpeechKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Device executor of one {@link Lfm3aDepthformer} pass: the same graph as
 * {@link Lfm3aDepthformer#pass(float[], int[])}, run as kernels on the engine's queue over the clip weights the
 * {@link SpeechDeviceWeights} uploaded.
 *
 * <p>The codebook positions are the projected talker state plus each codebook's code or mask embedding; the
 * bidirectional LFM2 stack uses per-head RMS norm plus NEOX RoPE ({@link SpeechKernels#RMS_NORM_ROPE_ROWS}),
 * full attention ({@link SpeechKernels#TTS_FULL_ATTN}) and the per-head sigmoid gate
 * ({@link SpeechKernels#HEAD_SIGMOID_GATE}). The per-codebook head norm and the tied embedding head produce the
 * logits ({@link SpeechKernels#MATVEC_ROWS_F32}); the feedback row sums the output-embedding rows with
 * {@link SpeechKernels#RVQ_SUM_ROWS} and is read back for the backbone. Sampling stays on the host.
 *
 * <p>Owns its activation buffers; not thread-safe.
 */
final class GpuLfm3aDepthformer implements Lfm3aDepthformer.Engine {

    private final Lfm3aDepthformer geo;
    private final SpeechDeviceWeights device;
    private final GpuOps ops;
    private final int nEmbd;
    private final int talkerWidth;
    private final int nLayer;
    private final int nHead;
    private final int nHeadKv;
    private final int dHead;
    private final int kvCh;
    private final int qWidth;
    private final int nFf;
    private final int nCodebooks;
    private final int vocab;
    private final float eps;
    private final float thetaScale;

    private final GpuWeight projIn;
    private final GpuBuffer projInBias;
    private final GpuWeight embd;
    private final GpuWeight outEmbd;
    private final GpuBuffer headF32;
    private final GpuWeight outputNorm;
    private final GpuWeight[] headNorm;
    private final GpuBuffer maskEmbd;
    private final Layer[] layers;

    private final GpuBuffer hidden;
    private final GpuBuffer base;
    private final GpuBuffer cur;
    private final GpuBuffer residual;
    private final GpuBuffer normed;
    private final GpuBuffer qRaw;
    private final GpuBuffer qNormed;
    private final GpuBuffer kRaw;
    private final GpuBuffer kNormed;
    private final GpuBuffer vRaw;
    private final GpuBuffer attn;
    private final GpuBuffer gated;
    private final GpuBuffer gate;
    private final GpuBuffer projected;
    private final GpuBuffer ffnGate;
    private final GpuBuffer ffnUp;
    private final GpuBuffer swiglu;
    private final GpuBuffer row;
    private final GpuBuffer logits;
    private final GpuBuffer logitsPage;
    private final GpuBuffer embRows;
    private final GpuBuffer pageRows;
    private final GpuBuffer feedback;
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final float[] feedbackHost;
    private final float[] logitsHost;
    private final float[] pageHost;
    private boolean closed;

    /** Resolved device tensors of one depthformer layer. */
    private static final class Layer {
        final GpuWeight q;
        final GpuWeight qNorm;
        final GpuWeight k;
        final GpuWeight kNorm;
        final GpuWeight v;
        final GpuWeight out;
        final GpuWeight attnGate;
        final GpuWeight attnPostNorm;
        final GpuWeight ln1;
        final GpuWeight ln2;
        final GpuWeight gate;
        final GpuWeight up;
        final GpuWeight down;
        final GpuWeight ffPostNorm;

        Layer(GpuWeight q, GpuWeight qNorm, GpuWeight k, GpuWeight kNorm, GpuWeight v, GpuWeight out,
              GpuWeight attnGate, GpuWeight attnPostNorm, GpuWeight ln1, GpuWeight ln2, GpuWeight gate,
              GpuWeight up, GpuWeight down, GpuWeight ffPostNorm) {
            this.q = q;
            this.qNorm = qNorm;
            this.k = k;
            this.kNorm = kNorm;
            this.v = v;
            this.out = out;
            this.attnGate = attnGate;
            this.attnPostNorm = attnPostNorm;
            this.ln1 = ln1;
            this.ln2 = ln2;
            this.gate = gate;
            this.up = up;
            this.down = down;
            this.ffPostNorm = ffPostNorm;
        }
    }

    GpuLfm3aDepthformer(Lfm3aDepthformer geo, SpeechDeviceWeights device) {
        this.geo = Objects.requireNonNull(geo, "geo");
        this.device = Objects.requireNonNull(device, "device");
        this.ops = device.ops();
        this.nEmbd = geo.nEmbd();
        this.talkerWidth = geo.talkerWidth();
        this.nLayer = geo.nLayer();
        this.nHead = geo.nHeadValue();
        this.nHeadKv = geo.nHeadKvValue();
        this.dHead = geo.dHeadValue();
        this.kvCh = nHeadKv * dHead;
        this.qWidth = nHead * dHead;
        this.nFf = geo.nFfValue();
        this.nCodebooks = geo.nCodebooks();
        this.vocab = geo.vocab();
        this.eps = geo.epsValue();
        this.thetaScale = (float) Math.pow(Lfm3aDepthformer.ROPE_THETA, -2.0 / dHead);
        if (nHead % nHeadKv != 0) {
            throw new IllegalArgumentException("lfm3a depthformer: " + nHead + " heads do not divide " + nHeadKv
                    + " key/value heads");
        }

        this.projIn = device.weight(Lfm3aDepthformer.PROJ_IN + ".weight");
        requireMatrix(projIn, (long) nCodebooks * nEmbd, talkerWidth, Lfm3aDepthformer.PROJ_IN);
        this.projInBias = device.f32(Lfm3aDepthformer.PROJ_IN + ".bias", geo.projInBiasValues());
        this.embd = device.weight(Lfm3aDepthformer.EMBD + ".weight");
        this.outEmbd = device.weight(Lfm3aDepthformer.OUT_EMBD + ".weight");
        this.headF32 = device.f32Rows(embd);
        this.outputNorm = device.f32Weight(Lfm3aDepthformer.OUTPUT_NORM + ".weight",
                geo.outputNormValues());
        this.headNorm = new GpuWeight[nCodebooks];
        for (int p = 0; p < nCodebooks; p++) {
            float[] page = new float[nEmbd];
            System.arraycopy(geo.headNormValues(), p * nEmbd, page, 0, nEmbd);
            headNorm[p] = device.f32Weight(Lfm3aDepthformer.HEAD_NORM + ".weight." + p, page);
        }
        this.maskEmbd = device.f32(Lfm3aDepthformer.MASK_EMBD, geo.maskEmbdValues());

        this.layers = new Layer[nLayer];
        Lfm3aDepthformer.Layer[] source = geo.layerTensors();
        for (int il = 0; il < nLayer; il++) {
            Lfm3aDepthformer.Layer l = source[il];
            String prefix = "a.gen.code.blk." + il + ".";
            GpuWeight q = device.weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.ATTN_Q));
            GpuWeight k = device.weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.ATTN_K));
            GpuWeight v = device.weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.ATTN_V));
            GpuWeight out = device.weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.ATTN_OUT));
            GpuWeight attnGate = device.weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.ATTN_GATE));
            GpuWeight gateW = device.weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.FFN_GATE));
            GpuWeight upW = device.weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.FFN_UP));
            GpuWeight downW = device.weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.FFN_DOWN));
            requireMatrix(q, qWidth, nEmbd, prefix + "attn_q");
            requireMatrix(k, kvCh, nEmbd, prefix + "attn_k");
            requireMatrix(v, kvCh, nEmbd, prefix + "attn_v");
            requireMatrix(out, nEmbd, qWidth, prefix + "attn_out");
            requireMatrix(attnGate, nHead, nEmbd, prefix + "attn_gate");
            requireMatrix(gateW, nFf, nEmbd, prefix + "ffn_gate");
            requireMatrix(upW, nFf, nEmbd, prefix + "ffn_up");
            requireMatrix(downW, nEmbd, nFf, prefix + "ffn_down");
            layers[il] = new Layer(q,
                    device.f32Weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.ATTN_Q_NORM), l.qNorm),
                    k,
                    device.f32Weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.ATTN_K_NORM), l.kNorm),
                    v, out, attnGate,
                    l.attnPostNorm == null ? null
                            : device.f32Weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.ATTN_POST_NORM),
                                    l.attnPostNorm),
                    device.f32Weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.LN1), l.ln1),
                    device.f32Weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.LN2), l.ln2),
                    gateW, upW, downW,
                    l.ffPostNorm == null ? null
                            : device.f32Weight(Lfm3aDepthformer.layerTensor(il, Lfm3aDepthformer.FFN_POST_NORM),
                                    l.ffPostNorm));
        }

        this.hidden = own("hidden", talkerWidth);
        this.base = own("base", (long) nCodebooks * nEmbd);
        this.cur = own("cur", (long) nCodebooks * nEmbd);
        this.residual = own("residual", (long) nCodebooks * nEmbd);
        this.normed = own("normed", (long) nCodebooks * nEmbd);
        this.qRaw = own("q", (long) nCodebooks * qWidth);
        this.qNormed = own("q.normed", (long) nCodebooks * qWidth);
        this.kRaw = own("k", (long) nCodebooks * kvCh);
        this.kNormed = own("k.normed", (long) nCodebooks * kvCh);
        this.vRaw = own("v", (long) nCodebooks * kvCh);
        this.attn = own("attn", (long) nCodebooks * qWidth);
        this.gated = own("attn.gated", (long) nCodebooks * qWidth);
        this.gate = own("gate", (long) nCodebooks * nHead);
        this.projected = own("projected", (long) nCodebooks * nEmbd);
        this.ffnGate = own("ffn.gate", (long) nCodebooks * nFf);
        this.ffnUp = own("ffn.up", (long) nCodebooks * nFf);
        this.swiglu = own("ffn.swiglu", (long) nCodebooks * nFf);
        this.row = own("row", nEmbd);
        this.logits = own("logits", (long) nCodebooks * vocab);
        this.logitsPage = own("logits.page", vocab);
        this.embRows = own("emb.rows", (long) nCodebooks * nEmbd);
        this.pageRows = own("page.rows", (long) nCodebooks * talkerWidth);
        this.feedback = own("feedback", talkerWidth);
        this.feedbackHost = new float[talkerWidth];
        this.logitsHost = new float[nCodebooks * vocab];
        this.pageHost = new float[vocab];
    }

    private GpuBuffer own(String name, long floats) {
        GpuBuffer buffer = device.allocate("depthformer." + name, floats);
        owned.add(buffer);
        return buffer;
    }

    private static void requireMatrix(GpuWeight w, long rows, long cols, String what) {
        if (w.rows() != rows || w.cols() != cols) {
            throw new IllegalArgumentException("lfm3a depthformer tensor " + what + " is " + w.rows() + " x "
                    + w.cols() + " on the device, the graph needs " + rows + " x " + cols);
        }
    }

    @Override
    public Lfm3aDepthformer.Frame pass(float[] hiddenState, int[] codes) {
        ensureOpen();
        Objects.requireNonNull(hiddenState, "hiddenState");
        Objects.requireNonNull(codes, "codes");
        if (hiddenState.length != talkerWidth) {
            throw new IllegalArgumentException("the hidden state is " + hiddenState.length + " wide, the"
                    + " depthformer projection takes " + talkerWidth);
        }
        if (codes.length != nCodebooks) {
            throw new IllegalArgumentException("the frame holds " + codes.length + " codes, the depthformer runs "
                    + nCodebooks + " codebooks");
        }
        for (int p = 0; p < nCodebooks; p++) {
            if (codes[p] < -1 || codes[p] >= vocab) {
                throw new IllegalArgumentException("codebook " + p + " code " + codes[p] + " is outside [-1, "
                        + vocab + ")");
            }
        }

        float[] feedbackValues = feedback(codes);
        boolean allKnown = true;
        for (int c : codes) {
            allKnown &= c >= 0;
        }
        if (allKnown) {
            // the reference returns before building the transformer when nothing is masked
            return new Lfm3aDepthformer.Frame(null, feedbackValues);
        }

        device.writeFloats(hidden, hiddenState);
        // one projection row per codebook position, as the host graph's matmul over nCodebooks * nEmbd rows does
        ops.matvec(projIn, hidden, 0, base, 0, 1);
        ops.enqueue(GpuOps.BATCH_BIAS_ADD, ActKernels.global1d(nCodebooks * nEmbd, ops.local()), new long[]{ops.local()},
                nCodebooks * nEmbd, 1, projInBias, base);
        fillEmbeddingRows(codes);
        ops.copy(base, 0, cur, 0, nCodebooks * nEmbd);
        ops.add(cur, 0, embRows, 0, cur, 0, nCodebooks * nEmbd);

        for (int il = 0; il < nLayer; il++) {
            layer(layers[il]);
        }

        // output norm, then the per-codebook head norm and the tied embedding head
        ops.norm(NormKind.RMS, outputNorm, null, eps, cur, 0, normed, 0, nCodebooks, nEmbd);
        for (int p = 0; p < nCodebooks; p++) {
            ops.norm(NormKind.RMS, headNorm[p], null, eps, normed, p * nEmbd, row, 0, 1, nEmbd);
            ops.enqueue(SpeechKernels.MATVEC_ROWS_F32, new long[]{SpeechKernels.rowGroupsGlobal(vocab)},
                    new long[]{SpeechKernels.LOCAL}, nEmbd, vocab, p * vocab, headF32, row, logitsPage);
            ops.copy(logitsPage, 0, logits, p * vocab, vocab);
        }
        device.readFloats(logits, logitsHost, nCodebooks * vocab);
        return new Lfm3aDepthformer.Frame(logitsHost.clone(), feedbackValues);
    }

    /** The feedback row of the revealed codes, read back from the device. */
    private float[] feedback(int[] codes) {
        ops.fill(pageRows, 0, (long) nCodebooks * talkerWidth * Float.BYTES, 0);
        for (int p = 0; p < nCodebooks; p++) {
            int c = codes[p];
            if (c < 0) {
                continue;
            }
            if (c >= vocab) {
                throw new IllegalArgumentException("codebook " + p + " code " + c + " is outside [0, " + vocab + ")");
            }
            ops.embed(outEmbd, new int[]{p * vocab + c}, pageRows, p * talkerWidth);
        }
        ops.enqueue(SpeechKernels.RVQ_SUM_ROWS, new long[]{SpeechKernels.elementGlobal(talkerWidth)},
                new long[]{SpeechKernels.LOCAL}, nCodebooks, 1, talkerWidth, pageRows, feedback);
        device.readFloats(feedback, feedbackHost, talkerWidth);
        return feedbackHost.clone();
    }

    /** One row per codebook: the code's embedding row, or the shared mask row when the codebook is masked. */
    private void fillEmbeddingRows(int[] codes) {
        for (int p = 0; p < nCodebooks; p++) {
            if (codes[p] >= 0) {
                ops.embed(embd, new int[]{p * vocab + codes[p]}, embRows, p * nEmbd);
            } else {
                ops.copy(maskEmbd, 0, embRows, p * nEmbd, nEmbd);
            }
        }
    }

    private void layer(Layer l) {
        ops.copy(cur, 0, residual, 0, nCodebooks * nEmbd);
        ops.norm(NormKind.RMS, l.ln1, null, eps, cur, 0, normed, 0, nCodebooks, nEmbd);
        ops.matmul(l.q, normed, 0, qRaw, 0, nCodebooks);
        ops.matmul(l.k, normed, 0, kRaw, 0, nCodebooks);
        ops.matmul(l.v, normed, 0, vRaw, 0, nCodebooks);
        ops.enqueue(SpeechKernels.RMS_NORM_ROPE_ROWS,
                new long[]{SpeechKernels.ropeRowsGlobal(nCodebooks, nHead, dHead)}, new long[]{SpeechKernels.LOCAL},
                nCodebooks, nHead, dHead, 0, 0, thetaScale, eps, qRaw, l.qNorm.shard(0), qNormed);
        ops.enqueue(SpeechKernels.RMS_NORM_ROPE_ROWS,
                new long[]{SpeechKernels.ropeRowsGlobal(nCodebooks, nHeadKv, dHead)},
                new long[]{SpeechKernels.LOCAL}, nCodebooks, nHeadKv, dHead, 0, 0, thetaScale, eps, kRaw, l.kNorm.shard(0),
                kNormed);
        ops.enqueue(SpeechKernels.TTS_FULL_ATTN, new long[]{SpeechKernels.fullAttentionGlobal(nCodebooks, nHead)},
                new long[]{SpeechKernels.LOCAL}, nCodebooks, nHead, nHeadKv, dHead, qNormed, kNormed, vRaw, attn);
        ops.matmul(l.attnGate, normed, 0, gate, 0, nCodebooks);
        ops.enqueue(SpeechKernels.HEAD_SIGMOID_GATE, new long[]{SpeechKernels.headGateGlobal(nCodebooks, nHead)},
                new long[]{SpeechKernels.LOCAL}, nCodebooks, nHead, dHead, attn, gate, gated);
        ops.matmul(l.out, gated, 0, projected, 0, nCodebooks);
        if (l.attnPostNorm != null) {
            ops.norm(NormKind.RMS, l.attnPostNorm, null, eps, projected, 0, projected, 0, nCodebooks, nEmbd);
        }
        ops.add(residual, 0, projected, 0, cur, 0, nCodebooks * nEmbd);

        ops.copy(cur, 0, residual, 0, nCodebooks * nEmbd);
        ops.norm(NormKind.RMS, l.ln2, null, eps, cur, 0, normed, 0, nCodebooks, nEmbd);
        ops.matmul(l.gate, normed, 0, ffnGate, 0, nCodebooks);
        ops.matmul(l.up, normed, 0, ffnUp, 0, nCodebooks);
        ops.glu(GluOp.SWIGLU, ActParams.NONE, ffnGate, 0, nFf, ffnUp, 0, nFf, swiglu, 0, nFf, nCodebooks, nFf);
        ops.matmul(l.down, swiglu, 0, projected, 0, nCodebooks);
        if (l.ffPostNorm != null) {
            ops.norm(NormKind.RMS, l.ffPostNorm, null, eps, projected, 0, projected, 0, nCodebooks, nEmbd);
        }
        ops.add(residual, 0, projected, 0, cur, 0, nCodebooks * nEmbd);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the lfm3a GPU depthformer is closed");
        }
    }

    /** Releases this engine's activation buffers. Idempotent. */
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
        return "GpuLfm3aDepthformer{" + geo + '}';
    }
}
