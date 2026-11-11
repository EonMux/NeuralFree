package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
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
import java.util.Random;

/**
 * Device executor of one {@link Qwen3TtsCodePredictor} session: the same per-frame graph as
 * {@link Qwen3TtsCodePredictor.Session}, run as kernels on the engine's queue over the clip weights the
 * {@link SpeechDeviceWeights} uploaded.
 *
 * <p>Every step is a single-token transformer: row norms, the q/k/v/out and gate/up/down projections, the per-head
 * RMS norm and NEOX RoPE ({@link SpeechKernels#HEAD_RMS_NORM_ROPE}), causal attention over the key/value cache rows
 * {@code 0..pos} ({@link SpeechKernels#TTS_CAUSAL_ATTN}) and the SwiGLU product. The per-codebook LM head page is
 * applied by {@link SpeechKernels#MATVEC_ROWS_F32} and sampled on the host by {@link SpeechSampling}, so the draw
 * sequence of a session is the host session's, and the frame's feedback row is one codebook-group sum
 * ({@link SpeechKernels#RVQ_SUM_ROWS}) read back for the backbone.
 *
 * <p>Owns its activation buffers and key/value caches; not thread-safe.
 */
final class GpuQwen3TtsCodePredictor implements Qwen3TtsCodePredictor.Engine {

    private final Qwen3TtsCodePredictor geo;
    private final SpeechDeviceWeights device;
    private final GpuOps ops;
    private final int nEmbd;
    private final int talkerWidth;
    private final int nLayer;
    private final int nHead;
    private final int nHeadKv;
    private final int dHead;
    private final int kvCh;
    private final int nFf;
    private final int nCodebooks;
    private final int vocab;
    private final int vocab0;
    private final int nKv;
    private final float eps;
    private final float thetaScale;
    private final int topK;
    private final float topP;

    private final GpuWeight projIn;
    private final GpuBuffer projBias;
    private final GpuWeight embd;
    private final GpuWeight outEmbd;
    private final GpuWeight[] q;
    private final GpuWeight[] k;
    private final GpuWeight[] v;
    private final GpuWeight[] out;
    private final GpuWeight[] gate;
    private final GpuWeight[] up;
    private final GpuWeight[] down;
    private final GpuWeight[] ln1;
    private final GpuWeight[] ln2;
    private final GpuWeight[] qNorm;
    private final GpuWeight[] kNorm;
    private final GpuWeight outputNorm;
    private final GpuBuffer head;

    private final GpuBuffer cur;
    private final GpuBuffer residual;
    private final GpuBuffer normed;
    private final GpuBuffer work;
    private final GpuBuffer qRaw;
    private final GpuBuffer kRaw;
    private final GpuBuffer vRaw;
    private final GpuBuffer qNormed;
    private final GpuBuffer kNormed;
    private final GpuBuffer attn;
    private final GpuBuffer projected;
    private final GpuBuffer gateBuf;
    private final GpuBuffer upBuf;
    private final GpuBuffer swiglu;
    private final GpuBuffer tokenRow;
    private final GpuBuffer logits;
    private final GpuBuffer feedback;
    private final GpuBuffer pageRows;
    private final GpuBuffer pageSum;
    private final GpuBuffer[] keys;
    private final GpuBuffer[] values;
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final float[] logitsHost;
    private final float[] feedbackHost;
    private boolean closed;

    GpuQwen3TtsCodePredictor(Qwen3TtsCodePredictor geo, SpeechDeviceWeights device, int topK, float topP) {
        this.geo = Objects.requireNonNull(geo, "geo");
        this.device = Objects.requireNonNull(device, "device");
        this.ops = device.ops();
        if (topK < 0) {
            throw new IllegalArgumentException("top-k must not be negative, got " + topK);
        }
        if (!(topP > 0f) || topP > 1f || Float.isNaN(topP)) {
            throw new IllegalArgumentException("top-p must be in (0, 1], got " + topP);
        }
        this.topK = topK;
        this.topP = topP;
        this.nEmbd = geo.nEmbd();
        this.talkerWidth = geo.talkerWidth();
        this.nLayer = geo.nLayer();
        this.nHead = geo.nHead();
        this.nHeadKv = geo.nHeadKv();
        this.dHead = geo.dHead();
        this.kvCh = nHeadKv * dHead;
        this.nFf = geo.nFf();
        this.nCodebooks = geo.nCodebooks();
        this.vocab = geo.vocab();
        this.vocab0 = geo.vocab0();
        this.nKv = geo.codesPerFrame();
        this.eps = geo.eps();
        this.thetaScale = (float) Math.pow(Qwen3TtsCodePredictor.ROPE_THETA, -2.0 / dHead);

        device.requireShape(Qwen3TtsCodePredictor.EMBD + ".weight", (long) vocab * nCodebooks, talkerWidth);
        device.requireShape(Qwen3TtsCodePredictor.OUT_EMBD + ".weight", vocab0, talkerWidth);
        device.requireShape(Qwen3TtsCodePredictor.HEAD + ".weight", (long) vocab * nCodebooks, nEmbd);
        this.embd = device.weight(Qwen3TtsCodePredictor.EMBD + ".weight");
        this.outEmbd = device.weight(Qwen3TtsCodePredictor.OUT_EMBD + ".weight");
        this.head = device.f32Rows(device.weight(Qwen3TtsCodePredictor.HEAD + ".weight"));
        this.projIn = geo.hasProjIn() ? device.weight(Qwen3TtsCodePredictor.PROJ_IN + ".weight") : null;
        this.projBias = geo.hasProjIn()
                ? device.f32Weight(Qwen3TtsCodePredictor.PROJ_IN + ".bias", geo.projInBias()).shard(0)
                : null;
        if (projIn != null) {
            if (projIn.cols() != talkerWidth || projIn.rows() != nEmbd) {
                throw new IllegalArgumentException("qwen3tts code predictor projection is " + projIn.rows() + " x "
                        + projIn.cols() + " on the device, the graph needs " + nEmbd + " x " + talkerWidth);
            }
        } else if (talkerWidth != nEmbd) {
            throw new IllegalArgumentException("qwen3tts code predictor: no projection and talker width "
                    + talkerWidth + " differs from the predictor width " + nEmbd);
        }

        this.q = new GpuWeight[nLayer];
        this.k = new GpuWeight[nLayer];
        this.v = new GpuWeight[nLayer];
        this.out = new GpuWeight[nLayer];
        this.gate = new GpuWeight[nLayer];
        this.up = new GpuWeight[nLayer];
        this.down = new GpuWeight[nLayer];
        this.ln1 = new GpuWeight[nLayer];
        this.ln2 = new GpuWeight[nLayer];
        this.qNorm = new GpuWeight[nLayer];
        this.kNorm = new GpuWeight[nLayer];
        for (int il = 0; il < nLayer; il++) {
            ln1[il] = norm(il, Qwen3TtsCodePredictor.ATTN_NORM, geo.attnNorm(il));
            q[il] = matrix(il, Qwen3TtsCodePredictor.ATTN_Q, nHead * dHead, nEmbd);
            qNorm[il] = norm(il, Qwen3TtsCodePredictor.ATTN_Q_NORM, geo.qNorm(il));
            k[il] = matrix(il, Qwen3TtsCodePredictor.ATTN_K, kvCh, nEmbd);
            kNorm[il] = norm(il, Qwen3TtsCodePredictor.ATTN_K_NORM, geo.kNorm(il));
            v[il] = matrix(il, Qwen3TtsCodePredictor.ATTN_V, kvCh, nEmbd);
            out[il] = matrix(il, Qwen3TtsCodePredictor.ATTN_OUT, nEmbd, nHead * dHead);
            ln2[il] = norm(il, Qwen3TtsCodePredictor.FFN_NORM, geo.ffnNorm(il));
            gate[il] = matrix(il, Qwen3TtsCodePredictor.FFN_GATE, nFf, nEmbd);
            up[il] = matrix(il, Qwen3TtsCodePredictor.FFN_UP, nFf, nEmbd);
            down[il] = matrix(il, Qwen3TtsCodePredictor.FFN_DOWN, nEmbd, nFf);
        }
        this.outputNorm = device.f32Weight(Qwen3TtsCodePredictor.OUTPUT_NORM + ".weight",
                geo.outputNormWeights());

        this.cur = own("cur", nEmbd);
        this.residual = own("residual", nEmbd);
        this.normed = own("normed", nEmbd);
        this.work = own("work", nEmbd);
        this.qRaw = own("q", nHead * dHead);
        this.kRaw = own("k", kvCh);
        this.vRaw = own("v", kvCh);
        this.qNormed = own("q.normed", nHead * dHead);
        this.kNormed = own("k.normed", kvCh);
        this.attn = own("attn", nHead * dHead);
        this.projected = own("projected", nEmbd);
        this.gateBuf = own("ffn.gate", nFf);
        this.upBuf = own("ffn.up", nFf);
        this.swiglu = own("ffn.swiglu", nFf);
        this.tokenRow = own("token.row", talkerWidth);
        this.logits = own("logits", vocab);
        this.feedback = own("feedback", talkerWidth);
        this.pageRows = own("page.rows", (long) nCodebooks * talkerWidth);
        this.pageSum = own("page.sum", talkerWidth);
        this.keys = new GpuBuffer[nLayer];
        this.values = new GpuBuffer[nLayer];
        for (int il = 0; il < nLayer; il++) {
            keys[il] = own("kv.k." + il, (long) kvCh * nKv);
            values[il] = own("kv.v." + il, (long) kvCh * nKv);
        }
        this.logitsHost = new float[vocab];
        this.feedbackHost = new float[talkerWidth];
    }

    private GpuWeight matrix(int il, String part, long rows, long cols) {
        String name = Qwen3TtsCodePredictor.layerTensor(il, part);
        device.requireShape(name, rows, cols);
        return device.weight(name);
    }

    private GpuWeight norm(int il, String part, float[] values) {
        return device.f32Weight(Qwen3TtsCodePredictor.layerTensor(il, part), values);
    }

    private GpuBuffer own(String name, long floats) {
        GpuBuffer buffer = device.allocate("predictor." + name, floats);
        owned.add(buffer);
        return buffer;
    }

    @Override
    public Qwen3TtsCodePredictor.Frame process(float[] hiddenState, int code0, Random rng) {
        ensureOpen();
        Objects.requireNonNull(hiddenState, "hiddenState");
        Objects.requireNonNull(rng, "rng");
        if (hiddenState.length != talkerWidth) {
            throw new IllegalArgumentException("the talker hidden state is " + hiddenState.length + " wide, the"
                    + " code predictor takes " + talkerWidth);
        }
        if (code0 < 0 || code0 >= vocab0) {
            throw new IllegalArgumentException("codec-0 code " + code0 + " is outside [0, " + vocab0 + ")");
        }
        int[] codes = new int[nKv];
        codes[0] = code0;
        device.writeFloats(tokenRow, hiddenState);
        project();

        step(0);
        embed(outEmbd, code0, tokenRow);
        project();
        step(1);
        codes[1] = samplePage(0, rng);

        for (int g = 1; g < nCodebooks; g++) {
            embed(embd, (g - 1) * vocab + codes[g], tokenRow);
            project();
            step(g + 1);
            codes[g + 1] = samplePage(g, rng);
        }

        embed(outEmbd, code0, feedback);
        int[] ids = new int[nCodebooks];
        for (int g = 0; g < nCodebooks; g++) {
            ids[g] = g * vocab + codes[g + 1];
        }
        ops.embed(embd, ids, pageRows, 0);
        ops.enqueue(SpeechKernels.RVQ_SUM_ROWS, new long[]{SpeechKernels.elementGlobal(talkerWidth)},
                new long[]{SpeechKernels.LOCAL}, nCodebooks, 1, talkerWidth, pageRows, pageSum);
        ops.add(feedback, 0, pageSum, 0, feedback, 0, talkerWidth);
        device.readFloats(feedback, feedbackHost, talkerWidth);
        return new Qwen3TtsCodePredictor.Frame(codes, feedbackHost.clone());
    }

    /** The current token row through the predictor projection (identity when the widths agree). */
    private void project() {
        if (projIn == null) {
            ops.copy(tokenRow, 0, cur, 0, nEmbd);
            return;
        }
        ops.matvec(projIn, tokenRow, 0, cur, 0, 1);
        ops.enqueue(GpuOps.BATCH_BIAS_ADD, ActKernels.global1d(nEmbd, ops.local()),
                new long[]{ops.local()}, nEmbd, 1, projBias, cur);
    }

    private void embed(GpuWeight table, int token, GpuBuffer dst) {
        ops.embed(table, new int[]{token}, dst, 0);
    }

    /** One position through every layer, leaving the normalized output row in {@link #work}. */
    private void step(int pos) {
        for (int il = 0; il < nLayer; il++) {
            ops.copy(cur, 0, residual, 0, nEmbd);
            ops.norm(NormKind.RMS, ln1[il], null, eps, cur, 0, normed, 0, 1, nEmbd);
            ops.matvec(q[il], normed, 0, qRaw, 0, 1);
            ops.matvec(k[il], normed, 0, kRaw, 0, 1);
            ops.matvec(v[il], normed, 0, vRaw, 0, 1);
            headNormRope(qRaw, qNorm[il], qNormed, pos);
            headNormRope(kRaw, kNorm[il], kNormed, pos);
            ops.copy2d(kNormed, 0, kvCh, keys[il], pos * kvCh, kvCh, 1, kvCh);
            ops.copy2d(vRaw, 0, kvCh, values[il], pos * kvCh, kvCh, 1, kvCh);
            ops.enqueue(SpeechKernels.TTS_CAUSAL_ATTN, new long[]{SpeechKernels.causalAttentionGlobal(nHead)},
                    new long[]{SpeechKernels.LOCAL}, nHead, nHeadKv, dHead, pos, qNormed, keys[il], values[il],
                    attn);
            ops.matvec(out[il], attn, 0, projected, 0, 1);
            ops.add(residual, 0, projected, 0, cur, 0, nEmbd);

            ops.copy(cur, 0, residual, 0, nEmbd);
            ops.norm(NormKind.RMS, ln2[il], null, eps, cur, 0, normed, 0, 1, nEmbd);
            ops.matvec(gate[il], normed, 0, gateBuf, 0, 1);
            ops.matvec(up[il], normed, 0, upBuf, 0, 1);
            ops.glu(GluOp.SWIGLU, ActParams.NONE, gateBuf, 0, nFf, upBuf, 0, nFf, swiglu, 0, nFf, 1, nFf);
            ops.matvec(down[il], swiglu, 0, projected, 0, 1);
            ops.add(residual, 0, projected, 0, cur, 0, nEmbd);
        }
        ops.norm(NormKind.RMS, outputNorm, null, eps, cur, 0, work, 0, 1, nEmbd);
    }

    private void headNormRope(GpuBuffer in, GpuWeight weight, GpuBuffer dst, int pos) {
        ops.enqueue(SpeechKernels.HEAD_RMS_NORM_ROPE,
                new long[]{SpeechKernels.headNormRopeGlobal(nHead, dHead)}, new long[]{SpeechKernels.LOCAL},
                nHead, dHead, pos, thetaScale, eps, in, weight.shard(0), dst);
    }

    /** The page's logits over {@link #work}, sampled on the host with the session's draw. */
    private int samplePage(int page, Random rng) {
        ops.enqueue(SpeechKernels.MATVEC_ROWS_F32, new long[]{SpeechKernels.elementGlobal(vocab)},
                new long[]{SpeechKernels.LOCAL}, nEmbd, vocab, page * vocab, head, work, logits);
        device.readFloats(logits, logitsHost, vocab);
        return SpeechSampling.draw(logitsHost, topK, topP, rng.nextFloat());
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the qwen3tts GPU code predictor is closed");
        }
    }

    /** Releases this session's activation buffers and key/value caches. Idempotent. */
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
        return "GpuQwen3TtsCodePredictor{" + geo + ", topK=" + topK + ", topP=" + topP + '}';
    }
}
