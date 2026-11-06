package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.StreamingIstft;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Lfm3aCodecSession} against an independent double-precision evaluation of llama.cpp's
 * {@code build_gen_wav} (tools/mtmd/models/lfm3a-gen.cpp:172-251): the FSQ table lookup and the first-codebook
 * projection, the repeated pre-transformer, the sliding-window mask with its cold-start guard over the key slot,
 * the short-convolution layer with its taps slot and the gated attention layer with its key/value slots, the
 * tail norm and projection, and the ISTFT decode of the polar spectra with the flushed tail's last-sample trim.
 */
public class Lfm3aCodecSessionTest {

    private static final int CB = Lfm3aClipFixture.CODEBOOKS;
    private static final int LVL = Lfm3aClipFixture.LVL;
    private static final int VOCAB = Lfm3aClipFixture.VOCAB;
    private static final int DEC = Lfm3aClipFixture.DEC;
    private static final int FF = Lfm3aClipFixture.DEC_FF;
    private static final int WINDOW = Lfm3aClipFixture.WINDOW;
    private static final int PREFIX = WINDOW - 1;
    private static final int STACK = Lfm3aClipFixture.FRAME_STACK;
    private static final int N_FFT = Lfm3aClipFixture.N_FFT;
    private static final int WIDTH = N_FFT + 2;
    private static final int TAPS = Lfm3aClipFixture.TAPS;
    private static final double EPS = 1e-5;
    private static final double ROPE = 1e6;

    /** The fixture's tensors as reference-friendly doubles, plus the layer bookkeeping the reference needs. */
    private static final class Ref {
        final float[] table;
        final double[][] quantOut; // [DEC][LVL*CB]
        final double[] quantOutB;
        final double[][] tfmIn;
        final double[] tfmInB;
        final double[] outNorm;
        final double[][] tfmOut;   // [WIDTH][DEC]
        final double[] tfmOutB;
        final boolean[] conv;
        final double[][] ln1;      // [layer][DEC]
        final double[][] ln2;
        final double[][] attnPost;
        final double[][] ffPost;
        final double[][][] gate;
        final double[][][] up;
        final double[][][] down;
        final double[][][] pw1;
        final double[][][] dw;
        final double[][][] pw2;
        final double[][][] q;
        final double[][] qNorm;    // [layer][dHead]
        final double[][][] k;
        final double[][] kNorm;
        final double[][][] v;
        final double[][][] o;
        final double[][][] attnGate;

        Ref(GgufTestWriter w) {
            table = data(w, "a.gen.wav.quant.first.codebook.weight");
            quantOut = mat(w, "a.gen.wav.quant.first.out_proj.weight", DEC, LVL * CB);
            quantOutB = vec(w, "a.gen.wav.quant.first.out_proj.bias", DEC);
            tfmIn = mat(w, "a.gen.wav.tfm.in_proj.weight", DEC, DEC);
            tfmInB = vec(w, "a.gen.wav.tfm.in_proj.bias", DEC);
            outNorm = vec(w, "a.gen.wav.tfm.output_norm.weight", DEC);
            tfmOut = mat(w, "a.gen.wav.tfm.out_proj.weight", WIDTH, DEC);
            tfmOutB = vec(w, "a.gen.wav.tfm.out_proj.bias", WIDTH);
            int layers = 2;
            conv = new boolean[layers];
            conv[0] = true;
            ln1 = new double[layers][DEC];
            ln2 = new double[layers][DEC];
            attnPost = new double[layers][DEC];
            ffPost = new double[layers][DEC];
            gate = new double[layers][DEC][];
            up = new double[layers][DEC][];
            down = new double[layers][FF][];
            pw1 = new double[layers][][];
            dw = new double[layers][][];
            pw2 = new double[layers][][];
            q = new double[layers][][];
            qNorm = new double[layers][2];
            k = new double[layers][][];
            kNorm = new double[layers][2];
            v = new double[layers][][];
            o = new double[layers][][];
            attnGate = new double[layers][][];
            for (int il = 0; il < layers; il++) {
                String p = "a.gen.wav.tfm.blk." + il + ".";
                ln1[il] = vec(w, p + "ln1.weight", DEC);
                ln2[il] = vec(w, p + "ln2.weight", DEC);
                attnPost[il] = vec(w, p + "attn_post_norm.weight", DEC);
                ffPost[il] = vec(w, p + "ffn_post_norm.weight", DEC);
                gate[il] = mat(w, p + "ffn_gate.weight", FF, DEC);
                up[il] = mat(w, p + "ffn_up.weight", FF, DEC);
                down[il] = mat(w, p + "ffn_down.weight", DEC, FF);
                if (conv[il]) {
                    pw1[il] = mat(w, p + "conv_pw1.weight", 3 * DEC, DEC);
                    float[] dwFlat = data(w, p + "conv_dw.weight");
                    dw[il] = new double[DEC][TAPS];
                    for (int i = 0; i < DEC; i++) {
                        for (int t = 0; t < TAPS; t++) {
                            dw[il][i][t] = dwFlat[t + i * TAPS];
                        }
                    }
                    pw2[il] = mat(w, p + "conv_pw2.weight", DEC, DEC);
                } else {
                    q[il] = mat(w, p + "attn_q.weight", 2 * 2, DEC);
                    qNorm[il] = vec(w, p + "attn_q_norm.weight", 2);
                    k[il] = mat(w, p + "attn_k.weight", 1 * 2, DEC);
                    kNorm[il] = vec(w, p + "attn_k_norm.weight", 2);
                    v[il] = mat(w, p + "attn_v.weight", 1 * 2, DEC);
                    o[il] = mat(w, p + "attn_out.weight", DEC, 2 * 2);
                    attnGate[il] = mat(w, p + "attn_gate.weight", 2, DEC);
                }
            }
        }

        private static double[][] mat(GgufTestWriter w, String name, int rows,
                                      int cols) {
            float[] data = data(w, name);
            double[][] m = new double[rows][cols];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    m[r][c] = data[c + r * cols];
                }
            }
            return m;
        }

        private static double[] vec(GgufTestWriter w, String name, int n) {
            float[] data = data(w, name);
            double[] out = new double[n];
            for (int i = 0; i < n; i++) {
                out[i] = data[i];
            }
            return out;
        }
    }

    /**
     * The reference decode: two successive batches against the running state; {@code guardKeys} selects whether
     * masked key rows before the stream start are excluded (the reference's guard) or read from the zeroed slot.
     */
    private static final class RefSession {
        final Ref r;
        double[] convState0 = new double[(TAPS - 1) * DEC];
        double[] kState1 = new double[PREFIX * 2];
        double[] vState1 = new double[PREFIX * 2];
        int pos;

        RefSession(Ref r) {
            this.r = r;
        }

        double[] decode(int[] codes, boolean guardKeys) {
            int nPos = codes.length / CB * STACK;
            int nKv = PREFIX + nPos;
            // FSQ, first projection, pre-transformer, frame stacking
            double[] inpL = new double[nPos * DEC];
            for (int f = 0; f < codes.length / CB; f++) {
                double[] qv = new double[LVL * CB];
                for (int cb = 0; cb < CB; cb++) {
                    for (int l = 0; l < LVL; l++) {
                        qv[cb * LVL + l] = r.table[l + codes[f * CB + cb] * LVL];
                    }
                }
                double[] projected = matvec(r.quantOut, qv);
                add(projected, r.quantOutB);
                projected = matvec(r.tfmIn, projected);
                add(projected, r.tfmInB);
                for (int s = 0; s < STACK; s++) {
                    System.arraycopy(projected, 0, inpL, (f * STACK + s) * DEC, DEC);
                }
            }
            // the running residual: the stream each block's result adds back into (build_lfm_layer)
            double[][] residual = new double[nPos][DEC];
            for (int t = 0; t < nPos; t++) {
                residual[t] = Arrays.copyOfRange(inpL, t * DEC, (t + 1) * DEC);
            }
            double[][] projected = new double[nPos][DEC];
            for (int il = 0; il < 2; il++) {
                double[][] cur = new double[nPos][DEC];
                for (int t = 0; t < nPos; t++) {
                    cur[t] = rmsNorm(inpL, t * DEC, r.ln1[il]);
                }
                if (r.conv[il]) {
                    shortConv(il, cur, projected, nPos);
                } else {
                    attention(il, cur, projected, nPos, nKv, guardKeys);
                }
                if (r.attnPost[il] != null) {
                    for (int t = 0; t < nPos; t++) {
                        projected[t] = rmsNormRow(projected[t], r.attnPost[il]);
                    }
                }
                addResidual(inpL, residual, projected, nPos);
                for (int t = 0; t < nPos; t++) {
                    cur[t] = rmsNorm(inpL, t * DEC, r.ln2[il]);
                }
                ffn(il, cur, projected, nPos);
                if (r.ffPost[il] != null) {
                    for (int t = 0; t < nPos; t++) {
                        projected[t] = rmsNormRow(projected[t], r.ffPost[il]);
                    }
                }
                addResidual(inpL, residual, projected, nPos);
            }
            double[] feats = new double[nPos * WIDTH];
            for (int t = 0; t < nPos; t++) {
                double[] row = rmsNormRow(Arrays.copyOfRange(inpL, t * DEC, (t + 1) * DEC), r.outNorm);
                double[] out = matvec(r.tfmOut, row);
                add(out, r.tfmOutB);
                for (int i = 0; i < WIDTH; i++) {
                    feats[t * WIDTH + i] = out[i];
                }
            }
            pos += nPos;
            return feats;
        }

        private void shortConv(int il, double[][] cur, double[][] out, int nPos) {
            int dConv = TAPS - 1;
            double[][] all = new double[dConv + nPos][DEC];
            for (int t = 0; t < dConv; t++) {
                System.arraycopy(convState0, t * DEC, all[t], 0, DEC);
            }
            double[][] bx = new double[nPos][DEC];
            for (int t = 0; t < nPos; t++) {
                double[] v3 = matvec(r.pw1[il], cur[t]);
                for (int i = 0; i < DEC; i++) {
                    bx[t][i] = v3[i] * v3[2 * DEC + i];
                    all[dConv + t][i] = bx[t][i];
                }
                for (int i = 0; i < DEC; i++) {
                    bx[t][i] = v3[DEC + i];
                }
            }
            for (int t = 0; t < nPos; t++) {
                for (int i = 0; i < DEC; i++) {
                    double acc = 0;
                    for (int kk = 0; kk < TAPS; kk++) {
                        acc += all[t + kk][i] * r.dw[il][i][kk];
                    }
                    out[t][i] = bx[t][i] * acc;
                }
            }
            for (int t = 0; t < nPos; t++) {
                out[t] = matvec(r.pw2[il], out[t]);
            }
            for (int i = 0; i < dConv; i++) {
                System.arraycopy(all[nPos + i], 0, convState0, i * DEC, DEC);
            }
        }

        private void attention(int il, double[][] cur, double[][] out, int nPos, int nKv, boolean guardKeys) {
            double[][] qq = new double[nPos][4];
            double[][] kk = new double[nPos][2];
            double[][] vv = new double[nPos][2];
            for (int t = 0; t < nPos; t++) {
                qq[t] = matvec(r.q[il], cur[t]);
                kk[t] = matvec(r.k[il], cur[t]);
                vv[t] = matvec(r.v[il], cur[t]);
                for (int h = 0; h < 2; h++) {
                    double[] head = rmsNormRow(range(qq[t], h * 2, 2), r.qNorm[il]);
                    rope(head, pos + t);
                    System.arraycopy(head, 0, qq[t], h * 2, 2);
                }
                double[] head = rmsNormRow(kk[t], r.kNorm[il]);
                rope(head, pos + t);
                kk[t] = head;
            }
            double[][] kAll = new double[nKv][2];
            double[][] vAll = new double[nKv][2];
            for (int j = 0; j < PREFIX; j++) {
                System.arraycopy(kState1, j * 2, kAll[j], 0, 2);
                System.arraycopy(vState1, j * 2, vAll[j], 0, 2);
            }
            for (int t = 0; t < nPos; t++) {
                kAll[PREFIX + t] = kk[t];
                vAll[PREFIX + t] = vv[t];
            }
            for (int t = 0; t < nPos; t++) {
                double[] gates = matvec(r.attnGate[il], cur[t]);
                double[] attn = new double[4];
                for (int h = 0; h < 2; h++) {
                    double max = Double.NEGATIVE_INFINITY;
                    double[] scores = new double[nKv];
                    for (int j = 0; j < nKv; j++) {
                        boolean visible = j >= t && j <= t + PREFIX;
                        if (guardKeys && visible && pos - PREFIX + j < 0) {
                            visible = false;
                        }
                        double s = visible ? (qq[t][h * 2] * kAll[j][0] + qq[t][h * 2 + 1] * kAll[j][1])
                                / Math.sqrt(2) : Double.NEGATIVE_INFINITY;
                        scores[j] = s;
                        if (s > max) {
                            max = s;
                        }
                    }
                    double sum = 0;
                    double[] weight = new double[nKv];
                    for (int j = 0; j < nKv; j++) {
                        weight[j] = Math.exp(scores[j] - max);
                        sum += weight[j];
                    }
                    double g = 1.0 / (1.0 + Math.exp(-gates[h]));
                    for (int d = 0; d < 2; d++) {
                        double acc = 0;
                        for (int j = 0; j < nKv; j++) {
                            acc += weight[j] / sum * vAll[j][d];
                        }
                        attn[h * 2 + d] = acc * g;
                    }
                }
                out[t] = matvec(r.o[il], attn);
            }
            // the slots keep the last PREFIX rows of [slot | chunk]
            for (int j = 0; j < PREFIX; j++) {
                System.arraycopy(kAll[nPos + j], 0, kState1, j * 2, 2);
                System.arraycopy(vAll[nPos + j], 0, vState1, j * 2, 2);
            }
        }

        private void ffn(int il, double[][] cur, double[][] out, int nPos) {
            for (int t = 0; t < nPos; t++) {
                double[] g = matvec(r.gate[il], cur[t]);
                double[] u = matvec(r.up[il], cur[t]);
                for (int i = 0; i < FF; i++) {
                    g[i] = g[i] / (1.0 + Math.exp(-g[i])) * u[i];
                }
                out[t] = matvec(r.down[il], g);
            }
        }
    }

    /** The tensor's payload as little-endian floats. */
    private static float[] data(GgufTestWriter w, String name) {
        byte[] raw = w.tensorData(name);
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.wrap(raw).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[raw.length / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = buf.getFloat(i * 4);
        }
        return out;
    }

    // ------------------------------------------------------------------ small math

    private static double[] matvec(double[][] w, double[] x) {
        double[] y = new double[w.length];
        for (int r = 0; r < w.length; r++) {
            double acc = 0;
            for (int c = 0; c < w[r].length; c++) {
                acc += w[r][c] * x[c];
            }
            y[r] = acc;
        }
        return y;
    }

    private static void add(double[] x, double[] b) {
        for (int i = 0; i < x.length; i++) {
            x[i] += b[i];
        }
    }

    private static double[] rmsNorm(double[] flat, int off, double[] w) {
        return rmsNormRow(Arrays.copyOfRange(flat, off, off + DEC), w);
    }

    private static double[] rmsNormRow(double[] x, double[] w) {
        double sum = 0;
        for (double v : x) {
            sum += v * v;
        }
        double inv = 1.0 / Math.sqrt(sum / x.length + EPS);
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] * inv * w[i];
        }
        return out;
    }

    private static void addResidual(double[] flat, double[][] residual, double[][] sum, int nPos) {
        for (int t = 0; t < nPos; t++) {
            for (int i = 0; i < DEC; i++) {
                flat[t * DEC + i] = residual[t][i] + sum[t][i];
            }
            residual[t] = Arrays.copyOfRange(flat, t * DEC, (t + 1) * DEC);
        }
    }

    private static double[] range(double[] x, int off, int n) {
        double[] out = new double[n];
        System.arraycopy(x, off, out, 0, n);
        return out;
    }

    private static void rope(double[] head, int position) {
        int half = head.length / 2;
        for (int i = 0; i < half; i++) {
            double angle = position * Math.pow(ROPE, -2.0 * i / head.length);
            double x0 = head[i];
            double x1 = head[i + half];
            head[i] = x0 * Math.cos(angle) - x1 * Math.sin(angle);
            head[i + half] = x0 * Math.sin(angle) + x1 * Math.cos(angle);
        }
    }

    private static float[] toFloat(double[] v) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = (float) v[i];
        }
        return out;
    }

    // ------------------------------------------------------------------ the tests

    // the codec takes the codebook only: its FSQ table has no row for the end code (vocabulary - 1)
    private static final int[] BATCH_A = {0, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 1, 1, 1, 0, 2, 1, 0, 2, 1, 1, 1, 0};
    private static final int[] BATCH_B = {1, 1, 0, 2, 2, 1, 1, 0, 0, 0, 1, 1};

    @Test
    public void successiveBatchesMatchTheReferenceWithTheirState() throws IOException {
        GgufTestWriter writer = Lfm3aClipFixture.randomWeights(71L);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(writer.writeTempFile().toString())) {
            Lfm3aCodecSession session = clip.codec().newSession();
            Ref ref = new Ref(writer);
            RefSession refSession = new RefSession(ref);

            // the graph accumulates in float32 as ggml does while the reference runs double, so the bound is
            // an accumulation guard; a wrong tap, mask direction or position shows up an order above it
            float[] first = session.decodePolar(BATCH_A);
            double[] refFirst = refSession.decode(BATCH_A, true);
            assertArrayEquals(toFloat(refFirst), first, 1.5e-3f);
            assertEquals(4, session.pos());
            assertEquals(BATCH_A.length / CB * STACK * WIDTH, first.length);

            float[] second = session.decodePolar(BATCH_B);
            double[] refSecond = refSession.decode(BATCH_B, true);
            assertArrayEquals(toFloat(refSecond), second, 1.5e-3f);
            assertEquals(6, session.pos());
        }
    }

    @Test
    public void theColdStartGuardExcludesKeysBeforeTheStreamStart() throws IOException {
        GgufTestWriter writer = Lfm3aClipFixture.randomWeights(83L);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(writer.writeTempFile().toString())) {
            Lfm3aCodecSession session = clip.codec().newSession();
            Ref ref = new Ref(writer);
            RefSession guarded = new RefSession(ref);
            RefSession unguarded = new RefSession(ref);
            float[] got = session.decodePolar(BATCH_A);
            double[] withGuard = guarded.decode(BATCH_A, true);
            double[] withoutGuard = unguarded.decode(BATCH_A, false);
            assertArrayEquals(toFloat(withGuard), got, 1.5e-3f);
            // the guard must matter at position 0: five slot rows sit before the stream start
            assertFalse(Arrays.equals(toFloat(withGuard), toFloat(withoutGuard)));
        }
    }

    @Test
    public void resetRestoresTheFreshState() throws IOException {
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(
                Lfm3aClipFixture.randomWeights(97L).writeTempFile().toString())) {
            Lfm3aCodec codec = clip.codec();
            assertEquals(Arrays.asList("pos", "conv_0", "k_1", "v_1"), codec.stateSlots());
            Lfm3aCodecSession session = codec.newSession();
            float[] fresh = session.decodePolar(BATCH_A);
            assertEquals(4, session.pos());
            assertTrue(session.pendingTail());
            session.decodePolar(BATCH_B);
            session.reset();
            assertEquals(0, session.pos());
            assertFalse(session.pendingTail());
            assertArrayEquals(fresh, session.decodePolar(BATCH_A), 0f);
        }
    }

    @Test
    public void decodeClipConcatenatesTheIstftAndTheFlushTrimsTheTail() throws IOException {
        GgufTestWriter writer = Lfm3aClipFixture.randomWeights(103L);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(writer.writeTempFile().toString())) {
            int nBins = clip.codec().nBins();
            assertEquals((N_FFT / 2) + 1, nBins);
            assertEquals(24000, clip.codec().sampleRate());

            // the reference drive: one session yields the polar spectra, an independent StreamingIstft turns
            // them into the expected PCM, then a second session's decodeClip must reproduce it
            float[] feats = clip.codec().newSession().decodePolar(BATCH_A);
            StreamingIstft istft = new StreamingIstft(N_FFT, Lfm3aClipFixture.HOP);
            java.nio.ByteBuffer expected = java.nio.ByteBuffer.allocate(4 * (feats.length / WIDTH)
                    * Lfm3aClipFixture.HOP).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (int p = 0; p < feats.length / WIDTH; p++) {
                float[] spec = new float[2 * nBins];
                for (int b = 0; b < nBins; b++) {
                    float mag = (float) Math.exp(feats[p * WIDTH + b]);
                    float phase = feats[p * WIDTH + nBins + b];
                    spec[2 * b] = mag * (float) Math.cos(phase);
                    spec[2 * b + 1] = mag * (float) Math.sin(phase);
                }
                for (float s : istft.processFrame(spec)) {
                    expected.putFloat(s);
                }
            }

            Lfm3aCodecSession session = clip.codec().newSession();
            AudioClip audio = session.decodeClip(BATCH_A);
            assertEquals(24000, audio.sampleRate());
            float[] got = audio.samples();
            assertEquals(expected.position() / 4, got.length);
            for (int i = 0; i < got.length; i++) {
                assertEquals(expected.getFloat(i * 4), got[i], 5e-4f);
            }

            // the flush keeps the head and loses the last (n_fft - hop) / 2 samples
            AudioClip tail = session.flushClip();
            float[] rawTail = istft.flush();
            int trim = (N_FFT - Lfm3aClipFixture.HOP) / 2;
            int keep = rawTail.length - Math.min(rawTail.length, trim);
            assertEquals(keep, tail.length());
            for (int i = 0; i < keep; i++) {
                assertEquals(rawTail[i], tail.samples()[i], 0f);
            }
            assertEquals(0, session.flushClip().length());
        }
    }

    @Test
    public void refusesBadFrameInputs() throws IOException {
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(
                Lfm3aClipFixture.randomWeights(109L).writeTempFile().toString())) {
            Lfm3aCodecSession session = clip.codec().newSession();
            assertThrows(IllegalArgumentException.class, () -> session.decodePolar(new int[0]));
            assertThrows(IllegalArgumentException.class, () -> session.decodePolar(new int[CB + 1]));
            int[] negative = new int[CB];
            Arrays.fill(negative, -1);
            assertThrows(IllegalArgumentException.class, () -> session.decodePolar(negative));
            int[] beyond = new int[CB];
            Arrays.fill(beyond, VOCAB);
            assertThrows(IllegalArgumentException.class, () -> session.decodePolar(beyond));
            // the end code (the depthformer's last vocabulary row) has no FSQ row and the decoder refuses it
            int[] endCode = new int[CB];
            Arrays.fill(endCode, VOCAB - 1);
            assertThrows(IllegalArgumentException.class, () -> session.decodePolar(endCode));
            assertThrows(NullPointerException.class, () -> session.decodePolar(null));
            assertThrows(IllegalArgumentException.class, () -> session.decodeClip(new int[3]));
        }
    }
}
