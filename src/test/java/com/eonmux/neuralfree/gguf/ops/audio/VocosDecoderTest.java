package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.audio.StreamingIstft;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Vocos head geometry (llama.cpp tools/mtmd/models/soprano.cpp, the loader's shapes at clip.cpp:3662-3684) and
 * the CPU executor against an independent double-precision transcription of the graph, with the polar head and
 * the iSTFT conventions of {@code decode_spectrum}.
 */
public class VocosDecoderTest {

    private static final int INP = 3;
    private static final int C = 4;
    private static final int FF = 6;
    private static final int FRAME_VALS = 6; // n_bins = 3, n_fft = 4
    private static final int BLOCKS = 2;
    private static final float EPS = 1e-6f;

    // ================================================================== fixture

    /** Shapes (ggml order) and values of every head tensor. */
    private static final class Fixture {
        final Map<String, long[]> shapes = new LinkedHashMap<>();
        final Map<String, float[]> values = new HashMap<>();
        private final Random rnd;

        Fixture(long seed) {
            rnd = new Random(seed);
        }

        void put(String name, float lo, float hi, long... ne) {
            long n = 1;
            for (long d : ne) {
                n *= d;
            }
            float[] v = new float[(int) n];
            for (int i = 0; i < v.length; i++) {
                v[i] = lo + rnd.nextFloat() * (hi - lo);
            }
            shapes.put(name, ne);
            values.put(name, v);
        }

        float[] get(String name) {
            return values.get(name);
        }

        com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef ref(String name) {
            float[] v = get(name);
            long[] ne = shapes.get(name);
            long cols = 1;
            for (int i = 0; i < ne.length - 1; i++) {
                cols *= ne[i];
            }
            ByteBuffer bb = ByteBuffer.allocate(v.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            for (float f : v) {
                bb.putFloat(f);
            }
            bb.flip();
            return com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef.of(GGMLType.F32, bb, ne[ne.length - 1], cols);
        }

        TensorCatalog catalog() {
            List<GGUFTensorInfo> infos = new ArrayList<>();
            for (Map.Entry<String, long[]> e : shapes.entrySet()) {
                infos.add(new GGUFTensorInfo(e.getKey(), e.getValue(), GGMLType.F32, 0L));
            }
            return TensorCatalog.of("clip", infos);
        }

        /** A matmul head over a pointwise input projection. */
        static Fixture matmul(long seed) {
            Fixture f = new Fixture(seed);
            f.put(VocosDecoder.weight(VocosDecoder.INPUT), -0.5f, 0.5f, INP, C);
            common(f);
            return f;
        }

        /** A conv head over a 3-D input projection, kernel 3. */
        static Fixture conv(long seed) {
            Fixture f = new Fixture(seed);
            f.put(VocosDecoder.weight(VocosDecoder.INPUT), -0.5f, 0.5f, 3, INP, C);
            common(f);
            return f;
        }

        private static void common(Fixture f) {
            f.put(VocosDecoder.bias(VocosDecoder.INPUT), -0.1f, 0.1f, C);
            f.put(VocosDecoder.weight(VocosDecoder.NORM), 0.8f, 1.2f, C);
            f.put(VocosDecoder.bias(VocosDecoder.NORM), -0.1f, 0.1f, C);
            for (int il = 0; il < BLOCKS; il++) {
                f.put(VocosDecoder.dwconv(il, "weight"), -0.3f, 0.3f, VocosDecoder.DW_K, 1, C);
                f.put(VocosDecoder.dwconv(il, "bias"), -0.05f, 0.05f, C);
                f.put(VocosDecoder.block(il, "norm", "weight"), 0.8f, 1.2f, C);
                f.put(VocosDecoder.block(il, "norm", "bias"), -0.1f, 0.1f, C);
                f.put(VocosDecoder.block(il, "pw1", "weight"), -0.5f, 0.5f, C, FF);
                f.put(VocosDecoder.block(il, "pw1", "bias"), -0.1f, 0.1f, FF);
                f.put(VocosDecoder.block(il, "pw2", "weight"), -0.5f, 0.5f, FF, C);
                f.put(VocosDecoder.block(il, "pw2", "bias"), -0.1f, 0.1f, C);
                f.put("a.gen.wav.up.blk." + il + ".gamma", 0.3f, 0.7f, C);
            }
            f.put(VocosDecoder.weight(VocosDecoder.OUTPUT_NORM), 0.8f, 1.2f, C);
            f.put(VocosDecoder.bias(VocosDecoder.OUTPUT_NORM), -0.1f, 0.1f, C);
            f.put(VocosDecoder.weight(VocosDecoder.OUTPUT), -0.5f, 0.5f, C, FRAME_VALS);
            f.put(VocosDecoder.bias(VocosDecoder.OUTPUT), -0.1f, 0.1f, FRAME_VALS);
        }
    }

    private static VocosDecoder decoder(Fixture f) {
        return VocosDecoder.fromTensors(f.catalog(), BLOCKS, EPS);
    }

    private static CpuVocosDecoder executor(Fixture f, VocosDecoder d) {
        return new CpuVocosDecoder(d, CpuVocosDecoder.resolve(d, f::get, f::ref),
                com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec.PLAIN);
    }

    // ================================================================== geometry

    @Test
    public void theGeometryReadsTheLoaderShapes() {
        VocosDecoder d = decoder(Fixture.matmul(1));
        assertEquals(INP, d.nInp());
        assertEquals(C, d.width());
        assertEquals(FF, d.nFf());
        assertEquals(FRAME_VALS, d.nFrameVals());
        assertEquals(3, d.nBins());
        assertEquals(4, d.nFft());
        assertEquals(0, d.convK());
        assertEquals(BLOCKS, d.blocks());
        VocosDecoder conv = decoder(Fixture.conv(2));
        assertEquals(3, conv.convK());
        assertEquals(VocosDecoder.UPSCALE * (5 - 1) + 1, conv.upsampledFrames(5));
    }

    @Test
    public void refusalsNameTheShape() {
        Fixture missing = Fixture.matmul(3);
        missing.shapes.remove(VocosDecoder.weight(VocosDecoder.NORM));
        assertThrows(MissingTensorException.class, () -> decoder(missing));
        // a valid-shape but odd output width reaches the even-count check (soprano.cpp:46-47)
        Fixture odd = Fixture.matmul(4);
        odd.shapes.get(VocosDecoder.weight(VocosDecoder.OUTPUT))[1] = 5;
        odd.shapes.get(VocosDecoder.bias(VocosDecoder.OUTPUT))[0] = 5;
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> decoder(odd));
        assertTrue(e.getMessage().contains("even count"));
        assertThrows(GgufFormatException.class, () -> VocosDecoder.fromTensors(Fixture.matmul(5).catalog(), 0, EPS));
        assertThrows(GgufFormatException.class,
                () -> VocosDecoder.fromTensors(Fixture.matmul(6).catalog(), BLOCKS, -1f));
        Fixture narrow = Fixture.matmul(7);
        narrow.shapes.get(VocosDecoder.block(0, "pw1", "weight"))[1] = 5;
        assertThrows(GgufFormatException.class, () -> decoder(narrow));
    }

    // ================================================================== executor

    @Test
    public void theMatmulHeadMatchesTheGraph() {
        parity(Fixture.matmul(11));
    }

    @Test
    public void theConvHeadMatchesTheGraph() {
        parity(Fixture.conv(12));
    }

    private static void parity(Fixture f) {
        VocosDecoder d = decoder(f);
        CpuVocosDecoder cpu = executor(f, d);
        int frames = 5;
        float[][] feats = new float[frames][INP];
        Random rnd = new Random(13);
        for (float[] row : feats) {
            for (int i = 0; i < row.length; i++) {
                row[i] = rnd.nextFloat() * 2 - 1;
            }
        }
        float[][] got = cpu.spectrum(feats);
        float[][] want = Reference.spectrum(f, d, feats);
        assertEquals(d.upsampledFrames(frames), got.length);
        assertEquals(want.length, got.length);
        for (int t = 0; t < got.length; t++) {
            assertArrayEquals("frame " + t, want[t], got[t], 2e-4f);
        }
        cpu.close();
    }

    @Test
    public void theDecodeConventionsMatchDecodeSpectrum() throws java.io.IOException {
        Fixture f = Fixture.matmul(21);
        VocosDecoder d = decoder(f);
        CpuVocosDecoder cpu = executor(f, d);
        int frames = 4;
        float[][] feats = new float[frames][INP];
        for (float[] row : feats) {
            java.util.Arrays.fill(row, 0f);
        }
        // the graph's per-upsampled-frame rows; decode_spectrum runs one iSTFT frame per row
        float[][] spectra = cpu.spectrum(feats);
        assertEquals(d.upsampledFrames(frames), spectra.length);
        // decode_spectrum (soprano.cpp:62-78) with the test's own hop: hop/2 more head samples than the shared
        // iSTFT drops, and (frames - 1) * hop samples kept
        int hop = 2;
        StreamingIstft istft = new StreamingIstft(d.nFft(), hop);
        java.io.ByteArrayOutputStream pcm = new java.io.ByteArrayOutputStream();
        java.io.DataOutputStream out = new java.io.DataOutputStream(pcm);
        for (float[] frame : spectra) {
            for (float v : istft.processFrame(frame)) {
                out.writeFloat(v);
            }
        }
        for (float v : istft.flush()) {
            out.writeFloat(v);
        }
        java.nio.ByteBuffer in = ByteBuffer.wrap(pcm.toByteArray());
        float[] all = new float[in.remaining() / Float.BYTES];
        for (int i = 0; i < all.length; i++) {
            all[i] = in.getFloat();
        }
        int head = hop / 2;
        int kept = Math.min((frames - 1) * hop, Math.max(all.length - head, 0));
        assertEquals((frames - 1) * hop, kept);
        cpu.close();
    }

    @Test
    public void refusalsNameTheBatchContract() {
        VocosDecoder d = decoder(Fixture.matmul(31));
        CpuVocosDecoder cpu = executor(Fixture.matmul(32), d);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> cpu.spectrum(new float[1][INP]));
        assertTrue(e.getMessage().contains("at least 2 frames"));
        IllegalArgumentException w = assertThrows(IllegalArgumentException.class,
                () -> cpu.spectrum(new float[3][INP + 1]));
        assertTrue(w.getMessage().contains("the model takes " + INP));
        assertThrows(NullPointerException.class, () -> cpu.spectrum(null));
        cpu.close();
        IllegalStateException closed = assertThrows(IllegalStateException.class,
                () -> cpu.spectrum(new float[3][INP]));
        assertTrue(closed.getMessage().contains("closed"));
    }

    /**
     * The soprano.cpp graph in double precision: upsample (align_corners), projection, per-frame LayerNorms, the
     * Vocos blocks, the output head and the polar spectrum.
     */
    private static final class Reference {

        static float[][] spectrum(Fixture f, VocosDecoder d, float[][] feats) {
            int t = feats.length;
            int inp = INP;
            int c = C;
            int up = d.upsampledFrames(t);
            // channel-major [inp][t] -> [inp][up]
            double[][] x = new double[inp][up];
            for (int ch = 0; ch < inp; ch++) {
                for (int u = 0; u < up; u++) {
                    double src = (double) u * (t - 1) / (up - 1);
                    int i0 = (int) Math.floor(src);
                    double frac = src - i0;
                    double a = feats[i0][ch];
                    double b = feats[Math.min(i0 + 1, t - 1)][ch];
                    x[ch][u] = a + frac * (b - a);
                }
            }
            // projection: conv over time or the pointwise product, then bias and the LayerNorm (token-major rows)
            double[][] cur = new double[up][c];
            if (d.convK() > 0) {
                float[] w = f.get(VocosDecoder.weight(VocosDecoder.INPUT));
                float[] b = f.get(VocosDecoder.bias(VocosDecoder.INPUT));
                int k = 3;
                int pad = k / 2;
                for (int u = 0; u < up; u++) {
                    for (int oc = 0; oc < c; oc++) {
                        double acc = b[oc];
                        for (int ic = 0; ic < inp; ic++) {
                            for (int j = 0; j < k; j++) {
                                int src = u + j - pad;
                                if (src >= 0 && src < up) {
                                    acc += w[(oc * inp + ic) * k + j] * x[ic][src];
                                }
                            }
                        }
                        cur[u][oc] = acc;
                    }
                }
            } else {
                float[] w = f.get(VocosDecoder.weight(VocosDecoder.INPUT));
                for (int u = 0; u < up; u++) {
                    for (int oc = 0; oc < c; oc++) {
                        double acc = f.get(VocosDecoder.bias(VocosDecoder.INPUT))[oc];
                        for (int ic = 0; ic < inp; ic++) {
                            acc += w[oc * inp + ic] * x[ic][u];
                        }
                        cur[u][oc] = acc;
                    }
                }
            }
            layerNorm(cur, f.get(VocosDecoder.weight(VocosDecoder.NORM)),
                    f.get(VocosDecoder.bias(VocosDecoder.NORM)), EPS);
            for (int il = 0; il < BLOCKS; il++) {
                block(f, d, il, cur);
            }
            layerNorm(cur, f.get(VocosDecoder.weight(VocosDecoder.OUTPUT_NORM)),
                    f.get(VocosDecoder.bias(VocosDecoder.OUTPUT_NORM)), EPS);
            float[] ow = f.get(VocosDecoder.weight(VocosDecoder.OUTPUT));
            float[] ob = f.get(VocosDecoder.bias(VocosDecoder.OUTPUT));
            int out = d.nFrameVals();
            int nBins = d.nBins();
            float[][] rows = new float[up][2 * nBins];
            for (int u = 0; u < up; u++) {
                double[] head = new double[out];
                for (int r = 0; r < out; r++) {
                    double acc = ob[r];
                    for (int ch = 0; ch < c; ch++) {
                        acc += ow[r * c + ch] * cur[u][ch];
                    }
                    head[r] = acc;
                }
                for (int j = 0; j < nBins; j++) {
                    double mag = Math.exp(head[j]);
                    mag = Math.max(0.0, Math.min(100.0, mag));
                    double phase = head[nBins + j];
                    rows[u][2 * j] = (float) (mag * Math.cos(phase));
                    rows[u][2 * j + 1] = (float) (mag * Math.sin(phase));
                }
            }
            // the graph runs over the upsampled frames; the caller maps frames -> PCM separately, so this
            // reference answers the executor's per-upsampled-frame rows
            return rows;
        }

        private static void block(Fixture f, VocosDecoder d, int il, double[][] cur) {
            int c = C;
            int t = cur.length;
            double[][] res = new double[t][c];
            for (int u = 0; u < t; u++) {
                System.arraycopy(cur[u], 0, res[u], 0, c);
            }
            float[] dw = f.get(VocosDecoder.dwconv(il, "weight"));
            float[] db = f.get(VocosDecoder.dwconv(il, "bias"));
            int k = VocosDecoder.DW_K;
            int pad = k / 2;
            for (int u = 0; u < t; u++) {
                for (int ch = 0; ch < c; ch++) {
                    double acc = db[ch];
                    for (int j = 0; j < k; j++) {
                        int src = u + j - pad;
                        if (src >= 0 && src < t) {
                            acc += dw[ch * k + j] * res[src][ch];
                        }
                    }
                    cur[u][ch] = acc;
                }
            }
            layerNorm(cur, f.get(VocosDecoder.block(il, "norm", "weight")),
                    f.get(VocosDecoder.block(il, "norm", "bias")), EPS);
            int ff = FF;
            float[] w1 = f.get(VocosDecoder.block(il, "pw1", "weight"));
            float[] b1 = f.get(VocosDecoder.block(il, "pw1", "bias"));
            float[] w2 = f.get(VocosDecoder.block(il, "pw2", "weight"));
            float[] b2 = f.get(VocosDecoder.block(il, "pw2", "bias"));
            float[] gamma = f.get("a.gen.wav.up.blk." + il + ".gamma");
            for (int u = 0; u < t; u++) {
                double[] h = new double[ff];
                for (int r = 0; r < ff; r++) {
                    double acc = b1[r];
                    for (int ch = 0; ch < c; ch++) {
                        acc += w1[r * c + ch] * cur[u][ch];
                    }
                    h[r] = geluErf(acc);
                }
                double[] back = new double[c];
                for (int r = 0; r < c; r++) {
                    double acc = b2[r];
                    for (int f2 = 0; f2 < ff; f2++) {
                        acc += w2[r * ff + f2] * h[f2];
                    }
                    back[r] = acc;
                }
                for (int ch = 0; ch < c; ch++) {
                    cur[u][ch] = res[u][ch] + gamma[ch] * back[ch];
                }
            }
        }

        private static void layerNorm(double[][] x, float[] w, float[] b, float eps) {
            for (double[] row : x) {
                double mean = 0;
                for (double v : row) {
                    mean += v;
                }
                mean /= row.length;
                double variance = 0;
                for (double v : row) {
                    variance += (v - mean) * (v - mean);
                }
                double inv = 1.0 / Math.sqrt(variance / row.length + eps);
                for (int i = 0; i < row.length; i++) {
                    row[i] = (row[i] - mean) * inv * w[i] + b[i];
                }
            }
        }

        /** 0.5 * x * (1 + erf(x / sqrt(2))), the erf the CPU kernel approximates. */
        static double geluErf(double v) {
            return 0.5 * v * (1 + erf(v / Math.sqrt(2)));
        }

        /** Abramowitz-Stegun 7.1.26, exact to 1.5e-7. */
        static double erf(double v) {
            boolean neg = v < 0;
            double a = Math.abs(v);
            double t = 1.0 / (1.0 + 0.3275911 * a);
            double poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                    + 0.254829592) * t;
            double y = 1.0 - poly * Math.exp(-a * a);
            return neg ? -y : y;
        }
    }
}
