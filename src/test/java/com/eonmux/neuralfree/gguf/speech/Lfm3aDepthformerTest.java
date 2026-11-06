package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Lfm3aDepthformer} against an independent double-precision evaluation of llama.cpp's
 * {@code build_gen_code} (tools/mtmd/models/lfm3a-gen.cpp:113-170): the replicated projection base, the code or
 * mask embedding per codebook position, the bidirectional LFM2 stack with per-head norms, full-width NEOX RoPE and
 * the sigmoid gate, the double RMS norm with the per-codebook head page, and the tied head's per-page logits.
 */
public class Lfm3aDepthformerTest {

    private static final double EPS = 1e-5;
    private static final double ROPE = 1e6;

    // fixture geometry
    private static final int TALKER = Lfm3aClipFixture.TALKER;
    private static final int EMBD = Lfm3aClipFixture.EMBD;
    private static final int LAYERS = Lfm3aClipFixture.LAYERS;
    private static final int HEADS = Lfm3aClipFixture.HEADS;
    private static final int KV = Lfm3aClipFixture.KV;
    private static final int DHEAD = Lfm3aClipFixture.DHEAD;
    private static final int FF = Lfm3aClipFixture.FF;
    private static final int CB = Lfm3aClipFixture.CODEBOOKS;
    private static final int VOCAB = Lfm3aClipFixture.VOCAB;

    /** The fixture's tensors, transposed into reference-friendly {@code [row][col]} arrays. */
    private static final class Ref {
        final double[][] projIn;   // [CB * EMBD][TALKER]
        final double[] projInB;    // [CB * EMBD]
        final double[][][] embd;   // [CB][VOCAB][EMBD]
        final double[][] outEmbd;  // [CB * VOCAB][TALKER], page p at row p * VOCAB
        final double[] outNorm;
        final double[][] headNorm; // [CB][EMBD]
        final double[] mask;
        final double[][] ln1;
        final double[][][] qw;     // [layer][EMBD][HEADS*DHEAD]
        final double[][] qn;
        final double[][][] kw;
        final double[][] kn;
        final double[][][] vw;
        final double[][][] ow;
        final double[][][] gateW;  // [layer][EMBD][HEADS]
        final double[][] ln2;
        final double[][][] ffnGate;
        final double[][][] ffnUp;
        final double[][][] ffnDown;

        Ref(GgufTestWriter w) {
            projIn = mat(w, "a.gen.code.proj_in.weight", EMBD * CB, TALKER);
            projInB = vec(w, "a.gen.code.proj_in.bias", EMBD * CB);
            embd = cube(w, "a.gen.code.embd.weight", CB, VOCAB, EMBD);
            outEmbd = mat(w, "a.gen.code.out_embd.weight", CB * VOCAB, TALKER);
            outNorm = vec(w, "a.gen.code.output_norm.weight", EMBD);
            headNorm = new double[CB][EMBD];
            float[] hn = data(w, "a.gen.code.head_norm.weight");
            for (int p = 0; p < CB; p++) {
                for (int d = 0; d < EMBD; d++) {
                    headNorm[p][d] = hn[d + p * EMBD];
                }
            }
            mask = vec(w, "a.gen.code.mask_embd", EMBD);
            ln1 = new double[LAYERS][EMBD];
            qw = new double[LAYERS][EMBD][HEADS * DHEAD];
            qn = new double[LAYERS][DHEAD];
            kw = new double[LAYERS][EMBD][KV * DHEAD];
            kn = new double[LAYERS][DHEAD];
            vw = new double[LAYERS][EMBD][KV * DHEAD];
            ow = new double[LAYERS][HEADS * DHEAD][EMBD];
            gateW = new double[LAYERS][EMBD][HEADS];
            ln2 = new double[LAYERS][EMBD];
            ffnGate = new double[LAYERS][EMBD][FF];
            ffnUp = new double[LAYERS][EMBD][FF];
            ffnDown = new double[LAYERS][FF][EMBD];
            for (int il = 0; il < LAYERS; il++) {
                String p = "a.gen.code.blk." + il + ".";
                ln1[il] = vec(w, p + "ln1.weight", EMBD);
                qw[il] = mat(w, p + "attn_q.weight", HEADS * DHEAD, EMBD);
                qn[il] = vec(w, p + "attn_q_norm.weight", DHEAD);
                kw[il] = mat(w, p + "attn_k.weight", KV * DHEAD, EMBD);
                kn[il] = vec(w, p + "attn_k_norm.weight", DHEAD);
                vw[il] = mat(w, p + "attn_v.weight", KV * DHEAD, EMBD);
                ow[il] = mat(w, p + "attn_out.weight", EMBD, HEADS * DHEAD);
                gateW[il] = mat(w, p + "attn_gate.weight", HEADS, EMBD);
                ln2[il] = vec(w, p + "ln2.weight", EMBD);
                ffnGate[il] = mat(w, p + "ffn_gate.weight", FF, EMBD);
                ffnUp[il] = mat(w, p + "ffn_up.weight", FF, EMBD);
                ffnDown[il] = mat(w, p + "ffn_down.weight", EMBD, FF);
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
            double[] v = new double[n];
            for (int i = 0; i < n; i++) {
                v[i] = data[i];
            }
            return v;
        }

        private static double[][][] cube(GgufTestWriter w, String name, int pages,
                                         int rows, int cols) {
            float[] data = data(w, name);
            double[][][] c = new double[pages][rows][cols];
            for (int p = 0; p < pages; p++) {
                for (int r = 0; r < rows; r++) {
                    for (int i = 0; i < cols; i++) {
                        c[p][r][i] = data[i + r * cols + p * rows * cols];
                    }
                }
            }
            return c;
        }
    }

    // ------------------------------------------------------------------ the reference

    /** The reference {@code pass}: logits codebook-major or null when every code is known, and the feedback. */
    private static Object[] referencePass(Ref r, double[] hidden, int[] codes) {
        double[][] cur = new double[CB][EMBD];
        for (int p = 0; p < CB; p++) {
            for (int d = 0; d < EMBD; d++) {
                double base = r.projInB[p * EMBD + d];
                for (int c = 0; c < TALKER; c++) {
                    base += r.projIn[p * EMBD + d][c] * hidden[c];
                }
                cur[p][d] = base + (codes[p] >= 0 ? r.embd[p][codes[p]][d] : r.mask[d]);
            }
        }
        boolean allKnown = true;
        for (int c : codes) {
            allKnown &= c >= 0;
        }
        if (allKnown) {
            return new Object[]{null, feedback(r, codes)};
        }
        // as build_lfm_layer does, each block reads the normed stream and its result adds back into the
        // stream it was normed from
        double[][] projected = new double[CB][EMBD];
        for (int il = 0; il < LAYERS; il++) {
            double[][] pre = new double[CB][];
            for (int p = 0; p < CB; p++) {
                pre[p] = cur[p];
                cur[p] = rmsNorm(pre[p], r.ln1[il]);
            }
            attention(r, il, cur, projected);
            addInto(cur, pre, projected);
            double[][] pre2 = new double[CB][];
            for (int p = 0; p < CB; p++) {
                pre2[p] = cur[p];
                cur[p] = rmsNorm(pre2[p], r.ln2[il]);
            }
            ffn(r, il, cur, projected);
            addInto(cur, pre2, projected);
        }
        double[] logits = new double[CB * VOCAB];
        for (int p = 0; p < CB; p++) {
            // the output norm over every row, then the per-codebook head page as the second norm's weight
            double[] normed = rmsNorm(cur[p], r.outNorm);
            double[] row = rmsNorm(normed, r.headNorm[p]);
            for (int v = 0; v < VOCAB; v++) {
                double dot = 0;
                for (int d = 0; d < EMBD; d++) {
                    dot += r.embd[p][v][d] * row[d];
                }
                logits[p * VOCAB + v] = dot;
            }
        }
        return new Object[]{logits, feedback(r, codes)};
    }

    /** The bidirectional gated GQA attention with per-head Q/K norms and full-width NEOX RoPE. */
    private static void attention(Ref r, int il, double[][] cur, double[][] out) {
        double[][] q = new double[CB][HEADS * DHEAD];
        double[][] k = new double[CB][KV * DHEAD];
        double[][] v = new double[CB][KV * DHEAD];
        for (int p = 0; p < CB; p++) {
            matvec(r.qw[il], cur[p], q[p]);
            matvec(r.kw[il], cur[p], k[p]);
            matvec(r.vw[il], cur[p], v[p]);
            for (int h = 0; h < HEADS; h++) {
                double[] head = rmsNorm(range(q[p], h * DHEAD, DHEAD), r.qn[il]);
                rope(head, p);
                copyInto(head, q[p], h * DHEAD);
            }
            for (int h = 0; h < KV; h++) {
                double[] head = rmsNorm(range(k[p], h * DHEAD, DHEAD), r.kn[il]);
                rope(head, p);
                copyInto(head, k[p], h * DHEAD);
            }
        }
        for (int p = 0; p < CB; p++) {
            double[] attn = new double[HEADS * DHEAD];
            double[] gates = new double[HEADS];
            matvec(r.gateW[il], cur[p], gates);
            for (int h = 0; h < HEADS; h++) {
                int kvHead = h / (HEADS / KV);
                double[] scores = new double[CB];
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < CB; j++) {
                    double dot = 0;
                    for (int d = 0; d < DHEAD; d++) {
                        dot += q[p][h * DHEAD + d] * k[j][kvHead * DHEAD + d];
                    }
                    scores[j] = dot / Math.sqrt(DHEAD);
                    if (scores[j] > max) {
                        max = scores[j];
                    }
                }
                double sum = 0;
                double[] weight = new double[CB];
                for (int j = 0; j < CB; j++) {
                    weight[j] = Math.exp(scores[j] - max);
                    sum += weight[j];
                }
                double g = 1.0 / (1.0 + Math.exp(-gates[h]));
                for (int d = 0; d < DHEAD; d++) {
                    double acc = 0;
                    for (int j = 0; j < CB; j++) {
                        acc += weight[j] / sum * v[j][kvHead * DHEAD + d];
                    }
                    attn[h * DHEAD + d] = acc * g;
                }
            }
            matvec(r.ow[il], attn, out[p]);
        }
    }

    private static void ffn(Ref r, int il, double[][] cur, double[][] out) {
        for (int p = 0; p < CB; p++) {
            double[] g = new double[FF];
            double[] u = new double[FF];
            matvec(r.ffnGate[il], cur[p], g);
            matvec(r.ffnUp[il], cur[p], u);
            for (int i = 0; i < FF; i++) {
                g[i] = g[i] / (1.0 + Math.exp(-g[i])) * u[i];
            }
            matvec(r.ffnDown[il], g, out[p]);
        }
    }

    private static double[] feedback(Ref r, int[] codes) {
        double[] fb = new double[TALKER];
        for (int p = 0; p < CB; p++) {
            if (codes[p] < 0) {
                continue;
            }
            for (int c = 0; c < TALKER; c++) {
                fb[c] += r.outEmbd[p * VOCAB + codes[p]][c];
            }
        }
        return fb;
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

    private static double[] rmsNorm(double[] x, double[] w) {
        double inv = 1.0 / norm(x);
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] * inv * w[i];
        }
        return out;
    }

    private static double norm(double[] x) {
        double sum = 0;
        for (double v : x) {
            sum += v * v;
        }
        return Math.sqrt(sum / x.length + EPS);
    }

    /** Full-width NEOX RoPE at position {@code pos} over {@code [i, i + half]} pairs. */
    private static void rope(double[] head, int pos) {
        int half = head.length / 2;
        for (int i = 0; i < half; i++) {
            double angle = pos * Math.pow(ROPE, -2.0 * i / head.length);
            double cos = Math.cos(angle);
            double sin = Math.sin(angle);
            double x0 = head[i];
            double x1 = head[i + half];
            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x0 * sin + x1 * cos;
        }
    }

    private static double[] range(double[] x, int off, int n) {
        double[] out = new double[n];
        System.arraycopy(x, off, out, 0, n);
        return out;
    }

    private static void copyInto(double[] src, double[] dst, int dstOff) {
        System.arraycopy(src, 0, dst, dstOff, src.length);
    }

    private static void matvec(double[][] w, double[] x, double[] y) {
        for (int r = 0; r < w.length; r++) {
            double acc = 0;
            for (int c = 0; c < w[r].length; c++) {
                acc += w[r][c] * x[c];
            }
            y[r] = acc;
        }
    }

    private static void addInto(double[][] target, double[][] residual, double[][] sum) {
        for (int p = 0; p < target.length; p++) {
            for (int i = 0; i < target[p].length; i++) {
                target[p][i] = residual[p][i] + sum[p][i];
            }
        }
    }

    // ------------------------------------------------------------------ the tests

    @Test
    public void maskedPassMatchesTheReferenceGraph() throws IOException {
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(
                Lfm3aClipFixture.randomWeights(11L).writeTempFile().toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            Ref ref = new Ref(Lfm3aClipFixture.randomWeights(11L));
            double[] hidden = hidden(0.3, -0.7, 1.1, 0.2);
            int[] codes = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
            Object[] expected = referencePass(ref, hidden, codes);
            Lfm3aDepthformer.Frame frame = df.pass(toFloat(hidden), codes);
            assertNotNull(frame.logits());
            assertArrayEquals(toFloat((double[]) expected[0]), frame.logits(), 2e-4f);
            assertArrayEquals(toFloat((double[]) expected[1]), frame.feedback(), 2e-4f);
            assertEquals(CB * VOCAB, frame.logits().length);
            assertEquals(TALKER, frame.feedback().length);
        }
    }

    @Test
    public void mixedMaskingMatchesTheReferenceGraph() throws IOException {
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(
                Lfm3aClipFixture.randomWeights(23L).writeTempFile().toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            Ref ref = new Ref(Lfm3aClipFixture.randomWeights(23L));
            double[] hidden = hidden(-0.2, 0.9, 0.4, -1.3);
            int[] codes = {2, -1, 0, 3, -1, 1, 0, -1, 2, 1, -1, 0};
            Object[] expected = referencePass(ref, hidden, codes);
            Lfm3aDepthformer.Frame frame = df.pass(toFloat(hidden), codes);
            assertNotNull(frame.logits());
            assertArrayEquals(toFloat((double[]) expected[0]), frame.logits(), 2e-4f);
            assertArrayEquals(toFloat((double[]) expected[1]), frame.feedback(), 2e-4f);
        }
    }

    @Test
    public void fullyKnownPassSkipsTheTransformer() throws IOException {
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(
                Lfm3aClipFixture.randomWeights(37L).writeTempFile().toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            Ref ref = new Ref(Lfm3aClipFixture.randomWeights(37L));
            int[] codes = {1, 2, 0, 3, 1, 1, 2, 0, 0, 2, 3, 1};
            Object[] expected = referencePass(ref, hidden(0.1, 0.2, 0.3, 0.4), codes);
            Lfm3aDepthformer.Frame frame = df.pass(toFloat(hidden(0.1, 0.2, 0.3, 0.4)), codes);
            assertNull("an all-known pass returns before running the transformer", frame.logits());
            assertArrayEquals(toFloat((double[]) expected[1]), frame.feedback(), 2e-4f);
            // the feedback equals the reference sum of the shared out-embedding rows
            float[] direct = df.feedback(codes);
            assertArrayEquals(frame.feedback(), direct, 0f);
        }
    }

    @Test
    public void headNormPagesDifferPerCodebook() throws IOException {
        // the fixture's head_norm pages are independent draws; a reference forced to page 0 on every codebook
        // must disagree with the graph, which proves the comparison is sensitive to the per-page selection
        GgufTestWriter shared = Lfm3aClipFixture.randomWeights(53L);
        float[] page0 = Arrays.copyOf(data(shared, "a.gen.code.head_norm.weight"), EMBD);
        float[] tiled = new float[EMBD * CB];
        for (int p = 0; p < CB; p++) {
            System.arraycopy(page0, 0, tiled, p * EMBD, EMBD);
        }
        shared.tensorF32("a.gen.code.head_norm.weight", new long[]{EMBD, CB}, tiled);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(
                Lfm3aClipFixture.randomWeights(53L).writeTempFile().toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            double[] hidden = hidden(0.5, -0.5, 0.5, -0.5);
            int[] codes = new int[CB];
            java.util.Arrays.fill(codes, -1);
            Lfm3aDepthformer.Frame frame = df.pass(toFloat(hidden), codes);
            double[] own = (double[]) referencePass(new Ref(Lfm3aClipFixture.randomWeights(53L)), hidden, codes)[0];
            double[] forced = (double[]) referencePass(new Ref(shared), hidden, codes)[0];
            boolean pageSensitive = false;
            for (int i = 0; i < own.length; i++) {
                if (Math.abs(own[i] - forced[i]) > 1e-3) {
                    pageSensitive = true;
                }
            }
            assertTrue("the fixture's head pages must differ for the test to be meaningful", pageSensitive);
            assertArrayEquals(toFloat(own), frame.logits(), 2e-4f);
        }
    }

    @Test
    public void acceptsHeadNormPagesWithALeadingSingletonDimension() throws IOException {
        // a checkpoint whose norm pages are [1, width] stacks to [width, 1, codebooks] instead of
        // [width, codebooks]; both carry the same per-codebook vectors and must drive the same graph
        GgufTestWriter w = Lfm3aClipFixture.randomWeights(67L);
        float[] hn = data(w, "a.gen.code.head_norm.weight");
        w.tensorF32("a.gen.code.head_norm.weight", new long[]{EMBD, 1, CB}, hn);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(w.writeTempFile().toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            Ref ref = new Ref(Lfm3aClipFixture.randomWeights(67L));
            double[] hidden = hidden(-0.4, 0.6, 1.2, 0.1);
            int[] codes = {-1, 2, 0, -1, 3, 1, -1, 0, -1, 2, 1, -1};
            Object[] expected = referencePass(ref, hidden, codes);
            Lfm3aDepthformer.Frame frame = df.pass(toFloat(hidden), codes);
            assertArrayEquals(toFloat((double[]) expected[0]), frame.logits(), 2e-4f);
            assertArrayEquals(toFloat((double[]) expected[1]), frame.feedback(), 2e-4f);
        }
    }

    @Test
    public void refusesWrongShapes() throws IOException {
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(
                Lfm3aClipFixture.randomWeights(61L).writeTempFile().toString())) {
            Lfm3aDepthformer df = clip.depthformer();
            assertThrows(IllegalArgumentException.class,
                    () -> df.pass(new float[]{1f, 2f}, new int[CB]));
            assertThrows(IllegalArgumentException.class,
                    () -> df.pass(new float[TALKER], new int[CB - 1]));
            int[] bad = new int[CB];
            java.util.Arrays.fill(bad, VOCAB);
            assertThrows(IllegalArgumentException.class, () -> df.pass(new float[TALKER], bad));
            int[] unknown = new int[CB];
            java.util.Arrays.fill(unknown, -2);
            assertThrows(IllegalArgumentException.class, () -> df.pass(new float[TALKER], unknown));
            assertThrows(IllegalArgumentException.class, () -> df.feedback(new int[CB - 1]));
        }
    }

    /** A hidden state of the backbone width with the given leading values and zeros after them. */
    private static double[] hidden(double... values) {
        double[] h = new double[TALKER];
        System.arraycopy(values, 0, h, 0, Math.min(values.length, TALKER));
        return h;
    }

    private static float[] toFloat(double[] v) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = (float) v[i];
        }
        return out;
    }
}
