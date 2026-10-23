package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.flow.FlowKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.OpKind;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * wavtokenizer-dec geometry, the CPU executor against an independent transcription of the ggml graph
 * (llama.cpp src/models/wavtokenizer-dec.cpp:118-262), the GPU dispatch sequence on the host-only fake backend with
 * strict LLM argument rules, and the qwen3tts / pockettts output contracts of {@link TtsBackbone}.
 */
public class WavTokenizerDecoderTest {

    private static final int E = 4;
    private static final int C = 8;
    private static final int GROUPS = 2;
    private static final int FF = 16;
    private static final int OUT = 6;
    private static final int CODES = 10;
    private static final int POSNET = 6;
    private static final int CONVNEXT = 2;
    private static final int CAP = 8;
    private static final float GROUP_EPS = 1e-6f;
    private static final float NORM_EPS = 1e-5f;
    private static final String ARCH = "wavtokenizer-dec";

    private FakeComputeBackend backend;
    private GpuScratch gpuScratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    @After
    public void tearDown() {
        if (ops != null) {
            ops.close();
        }
        if (placeholders != null) {
            placeholders.close();
        }
        if (gpuScratch != null) {
            gpuScratch.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    // ================================================================== fixture

    /** Shapes (ggml order) and values of every decoder tensor, in loader order. */
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

        void weight(String name, long... ne) {
            put(name, -0.5f, 0.5f, ne);
        }

        void norm(String weight, String bias, long... ne) {
            put(weight, 0.8f, 1.2f, ne);
            put(bias, -0.1f, 0.1f, ne);
        }

        float[] get(String name) {
            float[] v = values.get(name);
            if (v == null) {
                throw new MissingTensorException(ARCH, -1, name);
            }
            return v;
        }

        CpuTensorRef ref(String name) {
            float[] v = get(name);
            long[] ne = shapes.get(name);
            ByteBuffer bb = ByteBuffer.allocate(v.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            for (float f : v) {
                bb.putFloat(f);
            }
            bb.flip();
            return CpuTensorRef.of(GGMLType.F32, bb, v.length / ne[0], ne[0]);
        }

        TensorCatalog catalog() {
            List<GGUFTensorInfo> infos = new ArrayList<>();
            for (Map.Entry<String, long[]> e : shapes.entrySet()) {
                infos.add(new GGUFTensorInfo(e.getKey(), e.getValue(), GGMLType.F32, 0L));
            }
            return TensorCatalog.of(ARCH, infos);
        }
    }

    private static String pn(int il, String base, boolean bias) {
        return WavTokenizerDecoder.posnet(il, base, bias);
    }

    private static String cn(int il, String base, boolean bias) {
        return WavTokenizerDecoder.convnext(il, base, bias);
    }

    /** Every tensor of wavtokenizer-dec.cpp:9-113 with the loader's shapes. */
    private static Fixture fixture(long seed) {
        Fixture f = new Fixture(seed);
        f.weight(TensorNames.weight(TensorNames.TOKEN_EMBD), E, CODES);
        f.weight(TensorNames.weight(TensorNames.CONV1D), 7, E, C);
        f.weight(TensorNames.bias(TensorNames.CONV1D), 1, C);
        for (int il = 0; il < POSNET; il++) {
            switch (WavTokenizerDecoder.posnetKind(il)) {
                case RESNET:
                    f.norm(pn(il, TensorNames.POS_NET_NORM1, false), pn(il, TensorNames.POS_NET_NORM1, true), 1, C);
                    f.weight(pn(il, TensorNames.POS_NET_CONV1, false), 3, C, C);
                    f.weight(pn(il, TensorNames.POS_NET_CONV1, true), 1, C);
                    f.norm(pn(il, TensorNames.POS_NET_NORM2, false), pn(il, TensorNames.POS_NET_NORM2, true), 1, C);
                    f.weight(pn(il, TensorNames.POS_NET_CONV2, false), 3, C, C);
                    f.weight(pn(il, TensorNames.POS_NET_CONV2, true), 1, C);
                    break;
                case ATTENTION:
                    f.norm(pn(il, TensorNames.POS_NET_ATTN_NORM, false), pn(il, TensorNames.POS_NET_ATTN_NORM, true),
                            1, C);
                    for (String p : CpuWavTokenizer.ATTENTION_PROJECTIONS) {
                        f.weight(pn(il, p, false), 1, C, C);
                        f.weight(pn(il, p, true), 1, C);
                    }
                    break;
                default:
                    f.norm(pn(il, TensorNames.POS_NET_ATTN_NORM, false), pn(il, TensorNames.POS_NET_ATTN_NORM, true),
                            1, C);
                    break;
            }
        }
        f.norm(TensorNames.weight(TensorNames.TOKEN_EMBD_NORM), TensorNames.bias(TensorNames.TOKEN_EMBD_NORM), C);
        for (int il = 0; il < CONVNEXT; il++) {
            f.weight(cn(il, TensorNames.CONVNEXT_DW, false), 7, 1, C);
            f.weight(cn(il, TensorNames.CONVNEXT_DW, true), 1, C);
            f.norm(cn(il, TensorNames.CONVNEXT_NORM, false), cn(il, TensorNames.CONVNEXT_NORM, true), C);
            f.weight(cn(il, TensorNames.CONVNEXT_PW1, false), C, FF);
            f.weight(cn(il, TensorNames.CONVNEXT_PW1, true), FF);
            f.weight(cn(il, TensorNames.CONVNEXT_PW2, false), FF, C);
            f.weight(cn(il, TensorNames.CONVNEXT_PW2, true), C);
            f.put(cn(il, TensorNames.CONVNEXT_GAMMA, false), 0.05f, 0.5f, C);
        }
        f.norm(TensorNames.weight(TensorNames.OUTPUT_NORM), TensorNames.bias(TensorNames.OUTPUT_NORM), C);
        f.weight(TensorNames.weight(TensorNames.OUTPUT), C, OUT);
        f.weight(TensorNames.bias(TensorNames.OUTPUT), OUT);
        return f;
    }

    private static WavTokenizerDecoder decoder(Fixture f) {
        return WavTokenizerDecoder.fromTensors(f.catalog(), POSNET, CONVNEXT, GROUPS, GROUP_EPS, NORM_EPS);
    }

    // ================================================================== geometry

    @Test
    public void geometryComesFromTheTensors() {
        WavTokenizerDecoder d = decoder(fixture(1));
        assertEquals(CODES, d.nCodes());
        assertEquals(E, d.nEmbd());
        assertEquals(C, d.width());
        assertEquals(FF, d.nFf());
        assertEquals(OUT, d.nOut());
        assertEquals(POSNET, d.posnetLayers());
        assertEquals(CONVNEXT, d.convnextLayers());
        assertEquals(GROUPS, d.nGroups());
        assertEquals(1.0f / (float) Math.sqrt(C), d.attentionScale(), 0f);
    }

    @Test
    public void posnetLayoutFollowsTheLoader() {
        assertEquals(WavTokenizerDecoder.PosnetKind.RESNET, WavTokenizerDecoder.posnetKind(0));
        assertEquals(WavTokenizerDecoder.PosnetKind.RESNET, WavTokenizerDecoder.posnetKind(1));
        assertEquals(WavTokenizerDecoder.PosnetKind.ATTENTION, WavTokenizerDecoder.posnetKind(2));
        assertEquals(WavTokenizerDecoder.PosnetKind.RESNET, WavTokenizerDecoder.posnetKind(3));
        assertEquals(WavTokenizerDecoder.PosnetKind.RESNET, WavTokenizerDecoder.posnetKind(4));
        assertEquals(WavTokenizerDecoder.PosnetKind.NORM, WavTokenizerDecoder.posnetKind(5));
        assertThrows(IllegalArgumentException.class, () -> WavTokenizerDecoder.posnetKind(6));
        assertEquals("posnet.2.attn_output.bias", pn(2, TensorNames.POS_NET_ATTN_OUT, true));
        assertEquals("convnext.1.gamma.weight", cn(1, TensorNames.CONVNEXT_GAMMA, false));
    }

    @Test
    public void shapesDifferentFromTheLoaderAreRejected() {
        Fixture f = fixture(2);
        f.shapes.put(TensorNames.weight(TensorNames.CONV1D), new long[]{5, E, C});
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> decoder(f));
        assertTrue(e.getMessage().contains("conv1d.weight"));

        Fixture g = fixture(2);
        g.shapes.put(TensorNames.bias(TensorNames.CONV1D), new long[]{C});
        assertThrows(GgufFormatException.class, () -> decoder(g));

        Fixture h = fixture(2);
        h.shapes.put(cn(1, TensorNames.CONVNEXT_PW1, false), new long[]{C, FF + 4});
        h.shapes.put(cn(1, TensorNames.CONVNEXT_PW1, true), new long[]{FF + 4});
        h.shapes.put(cn(1, TensorNames.CONVNEXT_PW2, false), new long[]{FF + 4, C});
        assertThrows(GgufFormatException.class, () -> decoder(h));
    }

    @Test
    public void missingTensorsAndBadCountsAreRejected() {
        Fixture f = fixture(3);
        f.shapes.remove(pn(2, TensorNames.POS_NET_ATTN_Q, false));
        assertThrows(MissingTensorException.class, () -> decoder(f));
        assertThrows(GgufFormatException.class,
                () -> WavTokenizerDecoder.fromTensors(fixture(3).catalog(), 7, CONVNEXT, GROUPS, GROUP_EPS, NORM_EPS));
        assertThrows(GgufFormatException.class,
                () -> WavTokenizerDecoder.fromTensors(fixture(3).catalog(), POSNET, CONVNEXT, 0, GROUP_EPS, NORM_EPS));
    }

    @Test
    public void codesAreCheckedAgainstCodebookAndCapacity() {
        WavTokenizerDecoder d = decoder(fixture(4));
        d.checkCodes(new int[]{0, CODES - 1}, CAP);
        assertThrows(IllegalArgumentException.class, () -> d.checkCodes(new int[0], CAP));
        assertThrows(IllegalArgumentException.class, () -> d.checkCodes(new int[CAP + 1], CAP));
        assertThrows(IllegalArgumentException.class, () -> d.checkCodes(new int[]{CODES}, CAP));
        assertThrows(IllegalArgumentException.class, () -> d.checkCodes(new int[]{-1}, CAP));
        assertThrows(IllegalArgumentException.class, () -> d.checkCapacity(0));
    }

    // ================================================================== CPU executor vs the ggml graph

    private static CpuWavTokenizer cpu(Fixture f, WavTokenizerDecoder d) {
        float[] table = f.get(TensorNames.weight(TensorNames.TOKEN_EMBD));
        CpuWavTokenizer.Weights w = CpuWavTokenizer.Weights.resolve(d, f::get, f::ref,
                (code, dst, off) -> System.arraycopy(table, code * E, dst, off, E));
        return new CpuWavTokenizer(d, w, CpuMatVec.PLAIN, CpuScratch.builder(CAP).build(), CAP);
    }

    @Test
    public void cpuDecodeMatchesTheGraphTranscription() {
        Fixture f = fixture(5);
        WavTokenizerDecoder d = decoder(f);
        CpuWavTokenizer dec = cpu(f, d);
        int[][] sequences = {{3}, {1, 4, 1, 5, 9}, {9, 2, 6, 5, 3, 5, 8, 0}};
        for (int[] codes : sequences) {
            float[][] got = dec.decode(codes);
            float[][] want = Reference.decode(f, codes);
            assertEquals(codes.length, got.length);
            for (int t = 0; t < codes.length; t++) {
                assertArrayEquals("row " + t + " of " + Arrays.toString(codes), want[t], got[t], 2e-4f);
            }
        }
    }

    @Test
    public void cpuDecodeIsRepeatableAndClosable() {
        Fixture f = fixture(6);
        CpuWavTokenizer dec = cpu(f, decoder(f));
        int[] codes = {7, 7, 2};
        float[][] first = dec.decode(codes);
        dec.decode(new int[]{1, 2, 3, 4, 5, 6, 7, 8});
        float[][] again = dec.decode(codes);
        for (int t = 0; t < codes.length; t++) {
            assertArrayEquals(first[t], again[t], 0f);
        }
        assertThrows(IllegalArgumentException.class, () -> dec.decode(new int[CAP + 1]));
        dec.close();
        assertThrows(IllegalStateException.class, () -> dec.decode(codes));
    }

    @Test
    public void transposeSwapsTheAxes() {
        float[] src = {0, 1, 2, 3, 4, 5};
        float[] dst = new float[6];
        CpuWavTokenizer.transpose(src, 2, 3, dst);
        assertArrayEquals(new float[]{0, 3, 1, 4, 2, 5}, dst, 0f);
        assertThrows(IllegalArgumentException.class, () -> CpuWavTokenizer.transpose(src, 2, 3, src));
    }

    /** Direct transcription of the ggml graph in double precision, channel-major like ggml. */
    private static final class Reference {

        static float[][] decode(Fixture f, int[] codes) {
            int t = codes.length;
            float[] table = f.get(TensorNames.weight(TensorNames.TOKEN_EMBD));
            double[][] in = new double[E][t];
            for (int i = 0; i < t; i++) {
                for (int e = 0; e < E; e++) {
                    in[e][i] = table[codes[i] * E + e];
                }
            }
            double[][] cur = conv(in, f.get(TensorNames.weight(TensorNames.CONV1D)),
                    f.get(TensorNames.bias(TensorNames.CONV1D)), C, 7);
            for (int il = 0; il < POSNET; il++) {
                double[][] inp = cur;
                switch (WavTokenizerDecoder.posnetKind(il)) {
                    case RESNET: {
                        double[][] h = silu(groupNorm(cur, f.get(pn(il, TensorNames.POS_NET_NORM1, false)),
                                f.get(pn(il, TensorNames.POS_NET_NORM1, true))));
                        h = conv(h, f.get(pn(il, TensorNames.POS_NET_CONV1, false)),
                                f.get(pn(il, TensorNames.POS_NET_CONV1, true)), C, 3);
                        h = silu(groupNorm(h, f.get(pn(il, TensorNames.POS_NET_NORM2, false)),
                                f.get(pn(il, TensorNames.POS_NET_NORM2, true))));
                        h = conv(h, f.get(pn(il, TensorNames.POS_NET_CONV2, false)),
                                f.get(pn(il, TensorNames.POS_NET_CONV2, true)), C, 3);
                        cur = add(h, inp);
                        break;
                    }
                    case ATTENTION: {
                        double[][] h = groupNorm(cur, f.get(pn(il, TensorNames.POS_NET_ATTN_NORM, false)),
                                f.get(pn(il, TensorNames.POS_NET_ATTN_NORM, true)));
                        double[][] q = conv(h, f.get(pn(il, TensorNames.POS_NET_ATTN_Q, false)),
                                f.get(pn(il, TensorNames.POS_NET_ATTN_Q, true)), C, 1);
                        double[][] k = conv(h, f.get(pn(il, TensorNames.POS_NET_ATTN_K, false)),
                                f.get(pn(il, TensorNames.POS_NET_ATTN_K, true)), C, 1);
                        double[][] v = conv(h, f.get(pn(il, TensorNames.POS_NET_ATTN_V, false)),
                                f.get(pn(il, TensorNames.POS_NET_ATTN_V, true)), C, 1);
                        double scale = 1.0f / (float) Math.sqrt(C);
                        double[][] o = new double[C][t];
                        for (int tq = 0; tq < t; tq++) {
                            double[] kq = new double[t];
                            double max = Double.NEGATIVE_INFINITY;
                            for (int tk = 0; tk < t; tk++) {
                                double s = 0;
                                for (int c = 0; c < C; c++) {
                                    s += k[c][tk] * q[c][tq];
                                }
                                kq[tk] = s * scale;
                                max = Math.max(max, kq[tk]);
                            }
                            double sum = 0;
                            for (int tk = 0; tk < t; tk++) {
                                kq[tk] = Math.exp(kq[tk] - max);
                                sum += kq[tk];
                            }
                            for (int c = 0; c < C; c++) {
                                double s = 0;
                                for (int tk = 0; tk < t; tk++) {
                                    s += kq[tk] / sum * v[c][tk];
                                }
                                o[c][tq] = s;
                            }
                        }
                        o = conv(o, f.get(pn(il, TensorNames.POS_NET_ATTN_OUT, false)),
                                f.get(pn(il, TensorNames.POS_NET_ATTN_OUT, true)), C, 1);
                        cur = add(o, inp);
                        break;
                    }
                    default:
                        cur = groupNorm(cur, f.get(pn(il, TensorNames.POS_NET_ATTN_NORM, false)),
                                f.get(pn(il, TensorNames.POS_NET_ATTN_NORM, true)));
                        break;
                }
            }
            cur = transpose(layerNorm(transpose(cur), f.get(TensorNames.weight(TensorNames.TOKEN_EMBD_NORM)),
                    f.get(TensorNames.bias(TensorNames.TOKEN_EMBD_NORM))));
            for (int il = 0; il < CONVNEXT; il++) {
                double[][] dw = depthwise(cur, f.get(cn(il, TensorNames.CONVNEXT_DW, false)),
                        f.get(cn(il, TensorNames.CONVNEXT_DW, true)));
                double[][] h = layerNorm(transpose(dw), f.get(cn(il, TensorNames.CONVNEXT_NORM, false)),
                        f.get(cn(il, TensorNames.CONVNEXT_NORM, true)));
                h = linear(h, f.get(cn(il, TensorNames.CONVNEXT_PW1, false)),
                        f.get(cn(il, TensorNames.CONVNEXT_PW1, true)), FF);
                for (double[] row : h) {
                    for (int j = 0; j < row.length; j++) {
                        double x = row[j];
                        row[j] = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * x * (1 + 0.044715 * x * x)));
                    }
                }
                h = linear(h, f.get(cn(il, TensorNames.CONVNEXT_PW2, false)),
                        f.get(cn(il, TensorNames.CONVNEXT_PW2, true)), C);
                float[] gamma = f.get(cn(il, TensorNames.CONVNEXT_GAMMA, false));
                for (double[] row : h) {
                    for (int c = 0; c < C; c++) {
                        row[c] *= gamma[c];
                    }
                }
                cur = add(transpose(h), cur);
            }
            double[][] h = layerNorm(transpose(cur), f.get(TensorNames.weight(TensorNames.OUTPUT_NORM)),
                    f.get(TensorNames.bias(TensorNames.OUTPUT_NORM)));
            h = linear(h, f.get(TensorNames.weight(TensorNames.OUTPUT)), f.get(TensorNames.bias(TensorNames.OUTPUT)),
                    OUT);
            float[][] out = new float[t][OUT];
            for (int i = 0; i < t; i++) {
                for (int o = 0; o < OUT; o++) {
                    out[i][o] = (float) h[i][o];
                }
            }
            return out;
        }

        /** ggml_conv_1d_ph (stride 1, padding K/2) plus the channel bias. */
        static double[][] conv(double[][] x, float[] w, float[] b, int oc, int k) {
            int ic = x.length;
            int len = x[0].length;
            double[][] y = new double[oc][len];
            for (int o = 0; o < oc; o++) {
                for (int t = 0; t < len; t++) {
                    double s = b[o];
                    for (int i = 0; i < ic; i++) {
                        for (int j = 0; j < k; j++) {
                            int src = t + j - k / 2;
                            if (src >= 0 && src < len) {
                                s += w[(o * ic + i) * k + j] * x[i][src];
                            }
                        }
                    }
                    y[o][t] = s;
                }
            }
            return y;
        }

        static double[][] depthwise(double[][] x, float[] w, float[] b) {
            int len = x[0].length;
            double[][] y = new double[x.length][len];
            for (int c = 0; c < x.length; c++) {
                for (int t = 0; t < len; t++) {
                    double s = b[c];
                    for (int j = 0; j < 7; j++) {
                        int src = t + j - 3;
                        if (src >= 0 && src < len) {
                            s += w[c * 7 + j] * x[c][src];
                        }
                    }
                    y[c][t] = s;
                }
            }
            return y;
        }

        /** ggml_group_norm over {T, 1, C}: groups of ceil(C / G) channels, then * w[c] + b[c]. */
        static double[][] groupNorm(double[][] x, float[] w, float[] b) {
            int len = x[0].length;
            int cpg = (C + GROUPS - 1) / GROUPS;
            double[][] y = new double[C][len];
            for (int g = 0; g < GROUPS; g++) {
                int c0 = g * cpg;
                int c1 = Math.min(c0 + cpg, C);
                double sum = 0;
                for (int c = c0; c < c1; c++) {
                    for (int t = 0; t < len; t++) {
                        sum += x[c][t];
                    }
                }
                double n = (double) (c1 - c0) * len;
                double mean = sum / n;
                double variance = 0;
                for (int c = c0; c < c1; c++) {
                    for (int t = 0; t < len; t++) {
                        variance += (x[c][t] - mean) * (x[c][t] - mean);
                    }
                }
                double inv = 1.0 / Math.sqrt(variance / n + GROUP_EPS);
                for (int c = c0; c < c1; c++) {
                    for (int t = 0; t < len; t++) {
                        y[c][t] = (x[c][t] - mean) * inv * w[c] + b[c];
                    }
                }
            }
            return y;
        }

        static double[][] layerNorm(double[][] rows, float[] w, float[] b) {
            double[][] y = new double[rows.length][];
            for (int r = 0; r < rows.length; r++) {
                double[] x = rows[r];
                double mean = 0;
                for (double v : x) {
                    mean += v;
                }
                mean /= x.length;
                double variance = 0;
                for (double v : x) {
                    variance += (v - mean) * (v - mean);
                }
                double inv = 1.0 / Math.sqrt(variance / x.length + NORM_EPS);
                y[r] = new double[x.length];
                for (int c = 0; c < x.length; c++) {
                    y[r][c] = (x[c] - mean) * inv * w[c] + b[c];
                }
            }
            return y;
        }

        static double[][] linear(double[][] rows, float[] w, float[] b, int outRows) {
            int cols = rows[0].length;
            double[][] y = new double[rows.length][outRows];
            for (int t = 0; t < rows.length; t++) {
                for (int o = 0; o < outRows; o++) {
                    double s = b[o];
                    for (int c = 0; c < cols; c++) {
                        s += w[o * cols + c] * rows[t][c];
                    }
                    y[t][o] = s;
                }
            }
            return y;
        }

        static double[][] silu(double[][] x) {
            double[][] y = new double[x.length][];
            for (int i = 0; i < x.length; i++) {
                y[i] = new double[x[i].length];
                for (int j = 0; j < x[i].length; j++) {
                    y[i][j] = x[i][j] / (1 + Math.exp(-x[i][j]));
                }
            }
            return y;
        }

        static double[][] add(double[][] a, double[][] b) {
            double[][] y = new double[a.length][];
            for (int i = 0; i < a.length; i++) {
                y[i] = new double[a[i].length];
                for (int j = 0; j < a[i].length; j++) {
                    y[i][j] = a[i][j] + b[i][j];
                }
            }
            return y;
        }

        static double[][] transpose(double[][] x) {
            double[][] y = new double[x[0].length][x.length];
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++) {
                    y[j][i] = x[i][j];
                }
            }
            return y;
        }
    }

    // ================================================================== GPU dispatch sequence

    private static final Set<String> DECODER_KERNELS = new HashSet<>(GpuWavTokenizer.KERNELS);

    private void openGpu() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        gpuScratch = GpuScratch.builder(backend, CAP).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), gpuScratch, placeholders);
    }

    /** Device copies of the fixture: F32 everywhere except the listed tensors, stored as zero F16. */
    private Map<String, GpuWeight> upload(Fixture f, Set<String> f16) {
        Map<String, GpuWeight> m = new HashMap<>();
        for (Map.Entry<String, long[]> e : f.shapes.entrySet()) {
            float[] v = f.get(e.getKey());
            if (f16.contains(e.getKey())) {
                GpuBuffer b = backend.allocateBuffer((long) v.length * 2, true);
                m.put(e.getKey(), GpuWeight.of(e.getKey(), GGMLType.F16, e.getValue(), b));
            } else {
                FakeGpuBuffer b = (FakeGpuBuffer) backend.allocateBuffer((long) v.length * Float.BYTES, true);
                b.putFloats(0, v);
                m.put(e.getKey(), GpuWeight.of(e.getKey(), GGMLType.F32, e.getValue(), b));
            }
        }
        return m;
    }

    private static List<String> expectedSequence() {
        List<String> s = new ArrayList<>();
        s.add(FlowKernels.GATHER_ROWS);
        s.add(AudioKernels.CONV1D_SAME);
        for (int il = 0; il < POSNET; il++) {
            switch (WavTokenizerDecoder.posnetKind(il)) {
                case RESNET:
                    s.addAll(Arrays.asList(NormKernels.GROUP_NORM, ActKernels.UNARY, AudioKernels.CONV1D_SAME,
                            NormKernels.GROUP_NORM, ActKernels.UNARY, AudioKernels.CONV1D_SAME, GpuOps.AXPBY));
                    break;
                case ATTENTION:
                    s.addAll(Arrays.asList(NormKernels.GROUP_NORM, FlowKernels.GATHER_ROWS,
                            GpuWavTokenizer.BATCH_BIAS_ADD, GpuWavTokenizer.BATCH_BIAS_ADD,
                            GpuWavTokenizer.BATCH_BIAS_ADD, AttnKernels.flashAttn(KvDType.F32), GpuWavTokenizer.BATCH_BIAS_ADD,
                            FlowKernels.GATHER_ROWS, GpuOps.AXPBY));
                    break;
                default:
                    s.addAll(Arrays.asList(NormKernels.GROUP_NORM, GpuOps.COPY2D));
                    break;
            }
        }
        s.addAll(Arrays.asList(FlowKernels.GATHER_ROWS, NormKernels.ROW_NORM, FlowKernels.GATHER_ROWS));
        for (int il = 0; il < CONVNEXT; il++) {
            s.addAll(Arrays.asList(AudioKernels.CONV1D_DEPTHWISE, FlowKernels.GATHER_ROWS, NormKernels.ROW_NORM,
                    GpuWavTokenizer.BATCH_BIAS_ADD, ActKernels.UNARY, GpuWavTokenizer.BATCH_BIAS_ADD,
                    NormKernels.MUL_ROW_SCALE, FlowKernels.GATHER_ROWS, GpuOps.AXPBY));
        }
        s.addAll(Arrays.asList(FlowKernels.GATHER_ROWS, NormKernels.ROW_NORM, GpuWavTokenizer.BATCH_BIAS_ADD));
        return s;
    }

    @Test
    public void gpuDecodeEnqueuesTheGraphInOrder() {
        openGpu();
        Fixture f = fixture(7);
        WavTokenizerDecoder d = decoder(f);
        String f16Conv = pn(0, TensorNames.POS_NET_CONV1, false);
        Map<String, GpuWeight> dev = upload(f, new HashSet<>(Arrays.asList(f16Conv)));
        GpuWeights weights = dev::get;

        backend.clearLog();
        GpuWavTokenizer gpu = GpuWavTokenizer.bind(d, ops, weights, CAP);
        boolean converted = false;
        for (Dispatch disp : backend.dispatches()) {
            converted |= disp.bufferArgs().contains((FakeGpuBuffer) dev.get(f16Conv).shard(0));
        }
        assertTrue("the F16 conv weight is decoded to F32 at bind", converted);
        FlashAttnTiling plan = gpu.attentionPlan();
        assertEquals(FlashAttnTiling.Plan.TILED, plan.plan());

        int[] codes = {1, 4, 1, 5, 9};
        int t = codes.length;
        backend.clearLog();
        float[][] rows = gpu.decode(codes);
        assertEquals(t, rows.length);
        assertEquals(OUT, rows[0].length);

        List<Dispatch> all = backend.dispatches();
        List<Dispatch> own = new ArrayList<>();
        int projections = 0;
        for (Dispatch disp : all) {
            if (DECODER_KERNELS.contains(disp.kernelName())) {
                own.add(disp);
            } else {
                projections++;
            }
        }
        List<String> names = new ArrayList<>();
        for (Dispatch disp : own) {
            names.add(disp.kernelName());
        }
        assertEquals(expectedSequence(), names);
        assertEquals("embedding, q/k/v/output projections, pw1/pw2 per ConvNeXt layer, output head",
                1 + 4 + 2 * CONVNEXT + 1, projections);

        Dispatch firstGather = own.get(0);
        assertArrayEquals(new int[]{E * t, 1, 1, 1, 0, 0}, ints(firstGather, 6));
        int[] idx = firstGather.bufferArg(6).getInts(0, E * t);
        for (int e = 0; e < E; e++) {
            for (int i = 0; i < t; i++) {
                assertEquals(i * E + e, idx[e * t + i]);
            }
        }

        Dispatch inputConv = own.get(1);
        assertArrayEquals(new int[]{t, E, C, 7, 1, 0, 0}, ints(inputConv, 7));
        assertArrayEquals(new long[]{AudioKernels.convGlobal(C, t)}, inputConv.global());

        Dispatch gn = own.get(2);
        assertArrayEquals(new int[]{C, t, GROUPS}, ints(gn, 3));
        assertEquals(GROUP_EPS, gn.floatArg(NormKernels.ARG_GROUP_NORM_EPS), 0f);
        Dispatch posnet0Conv1 = own.get(4);
        assertNotSame(dev.get(f16Conv).shard(0), posnet0Conv1.bufferArg(8));
        assertArrayEquals(new int[]{t, C, C, 3, 1, 0, 0}, ints(posnet0Conv1, 7));

        Dispatch fa = backend.dispatchesOf(AttnKernels.flashAttn(KvDType.F32)).get(0);
        assertEquals(AttnKernels.FA_ARG_COUNT, fa.argCount());
        assertEquals(t, fa.intArg(AttnKernels.ARG_FA_N_TOKENS));
        assertEquals(1, fa.intArg(AttnKernels.ARG_FA_N_HEAD));
        assertEquals(C, fa.intArg(AttnKernels.ARG_FA_D_K));
        assertEquals(C, fa.intArg(AttnKernels.ARG_FA_D_V));
        assertEquals(t, fa.intArg(AttnKernels.ARG_FA_KV_END));
        assertEquals(0, fa.intArg(AttnKernels.ARG_FA_CAUSAL));
        assertEquals(1, fa.intArg(AttnKernels.ARG_FA_DIRECT_OUT));
        assertEquals(C, fa.intArg(AttnKernels.ARG_FA_Q_ROW_STRIDE));
        assertEquals(C, fa.intArg(AttnKernels.ARG_FA_K_ROW_STRIDE));
        assertEquals(d.attentionScale(), fa.floatArg(AttnKernels.ARG_FA_SCALE), 0f);
        assertArrayEquals(plan.prefillGlobal(t, 1), fa.global());
        assertSame(fa.bufferArg(AttnKernels.ARG_FA_BUF_Q_POS), fa.bufferArg(AttnKernels.ARG_FA_BUF_KV_POS));
        assertArrayEquals(new int[]{0, 1, 2, 3, 4}, fa.bufferArg(AttnKernels.ARG_FA_BUF_KV_POS).getInts(0, t));
        assertTrue(backend.dispatchesOf(AttnKernels.FLASH_COMBINE).isEmpty());

        Dispatch bias = backend.dispatchesOf(GpuWavTokenizer.BATCH_BIAS_ADD).get(0);
        assertArrayEquals(new int[]{C, t}, ints(bias, 2));
        Dispatch scale = backend.dispatchesOf(NormKernels.MUL_ROW_SCALE).get(0);
        assertArrayEquals(new int[]{t, C, C}, ints(scale, 3));
        Dispatch dw = backend.dispatchesOf(AudioKernels.CONV1D_DEPTHWISE).get(0);
        assertArrayEquals(new int[]{t, C, 7, 0, 0}, ints(dw, 5));

        List<FakeComputeBackend.Op> log = backend.ops();
        assertEquals(OpKind.READ, log.get(log.size() - 2).kind());
        assertEquals(OpKind.FINISH, log.get(log.size() - 1).kind());
        gpu.close();
        assertThrows(IllegalStateException.class, () -> gpu.decode(codes));
    }

    @Test
    public void gpuRejectsSequencesAboveCapacityBeforeEnqueueing() {
        openGpu();
        Fixture f = fixture(8);
        Map<String, GpuWeight> dev = upload(f, new HashSet<>());
        GpuWavTokenizer gpu = GpuWavTokenizer.bind(decoder(f), ops, dev::get, CAP);
        backend.clearLog();
        assertThrows(IllegalArgumentException.class, () -> gpu.decode(new int[CAP + 1]));
        assertTrue(backend.ops().isEmpty());
    }

    @Test
    public void attentionPlanNeedsAKernelFriendlyWidth() {
        assertThrows(UnsupportedOperationException.class, () -> GpuWavTokenizer.attentionPlan(6, 8, 8192, 32));
        assertEquals(FlashAttnTiling.Plan.TILED, GpuWavTokenizer.attentionPlan(768, 512, 8192, 32).plan());
    }

    private static int[] ints(Dispatch d, int n) {
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
            v[i] = d.intArg(i);
        }
        return v;
    }

    // ================================================================== qwen3tts logits

    @Test
    public void qwen3ttsPaddingIsAPrefixOfNegativeInfinity() {
        TtsBackbone.LogitLayout l = TtsBackbone.LogitLayout.of(10, 4);
        assertEquals(6, l.pad());
        assertTrue(l.padded());
        float[] head = {0.5f, -2f, 3f, 1f};
        float[] row = l.toVocabRow(head);
        assertEquals(10, row.length);
        for (int i = 0; i < 6; i++) {
            assertEquals(Float.NEGATIVE_INFINITY, row[i], 0f);
        }
        assertArrayEquals(head, Arrays.copyOfRange(row, 6, 10), 0f);
        assertSame(row, l.toVocabRow(row));
        assertEquals(ForwardBackend.argmax(row), l.argmaxOfHead(head));
        assertEquals(8, l.argmaxOfHead(head));
        float[] dead = {Float.NEGATIVE_INFINITY, Float.NaN, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY};
        assertEquals(ForwardBackend.argmax(l.toVocabRow(dead)), l.argmaxOfHead(dead));

        float[] finitePad = new float[10];
        assertThrows(IllegalStateException.class, () -> l.toVocabRow(finitePad));
        assertThrows(IllegalStateException.class, () -> l.toVocabRow(new float[5]));
        assertThrows(GgufFormatException.class, () -> TtsBackbone.LogitLayout.of(3, 4));
        TtsBackbone.LogitLayout same = TtsBackbone.LogitLayout.of(4, 4);
        assertFalse(same.padded());
        assertSame(head, same.toVocabRow(head));
    }

    @Test
    public void qwen3ttsBackendPadsEveryLogitsRow() {
        float[] head = {0.25f, 2f, -1f};
        ForwardBackend inner = new StubBackend(head);
        TtsBackbone.LogitLayout l = TtsBackbone.LogitLayout.of(7, 3);
        ForwardBackend padded = new TtsBackbone.PaddedLogitsBackend(inner, l);
        float[] row = padded.decode(1, 0);
        assertEquals(7, row.length);
        assertEquals(Float.NEGATIVE_INFINITY, row[3], 0f);
        assertEquals(2f, row[5], 0f);
        assertEquals(5, padded.decodeGreedyOnDevice(1, 0));
        float[][] rows = padded.forwardAll(new int[]{1, 2}, null);
        assertEquals(2, rows.length);
        assertEquals(7, rows[1].length);
        assertEquals(-1f, rows[1][6], 0f);
    }

    @Test
    public void qwen3ttsDevicePaddingFillsThenCopies() {
        openGpu();
        TtsBackbone.LogitLayout l = TtsBackbone.LogitLayout.of(10, 4);
        FakeGpuBuffer head = (FakeGpuBuffer) backend.allocateBuffer(2 * 4 * Float.BYTES);
        FakeGpuBuffer dst = (FakeGpuBuffer) backend.allocateBuffer(2 * 10 * Float.BYTES);
        backend.clearLog();
        l.padOnDevice(ops, head, 0, dst, 0, 2);
        List<FakeComputeBackend.Op> log = backend.ops();
        assertEquals(OpKind.FILL, log.get(0).kind());
        Dispatch copy = backend.dispatchesOf(GpuOps.COPY2D).get(0);
        assertArrayEquals(new int[]{2, 4, 4, 0, 10, 6}, ints(copy, 6));
        assertEquals(Float.NEGATIVE_INFINITY, dst.getFloat(0), 0f);
        assertEquals(Float.NEGATIVE_INFINITY, dst.getFloat(15), 0f);
    }

    // ================================================================== pockettts hidden capture

    @Test
    public void captureCollectsTheTapRowsOfEveryForwardCall() {
        TtsBackbone.Capture cap = new TtsBackbone.Capture(3, 2);
        assertTrue(cap.wants(3));
        assertFalse(cap.wants(2));
        cap.onHidden(3, new float[]{1, 2, 3, 4}, 2, 2);
        cap.onHidden(3, new float[]{5, 6}, 1, 2);
        assertEquals(3, cap.rows());
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, Arrays.copyOf(cap.data(), 6), 0f);
        assertThrows(IllegalStateException.class, () -> cap.onHidden(3, new float[3], 1, 3));
        cap.clear();
        assertEquals(0, cap.rows());
    }

    /** Backend returning one fixed head-wide logits row for every token. */
    private static final class StubBackend implements ForwardBackend {
        private final float[] row;

        StubBackend(float[] row) {
            this.row = row;
        }

        @Override
        public float[] prefill(int[] tokens, int pos0) {
            return row.clone();
        }

        @Override
        public float[] decode(int token, int pos) {
            return row.clone();
        }

        @Override
        public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            return forwardAll(tokens, outputRows);
        }

        @Override
        public float[][] forwardAll(int[] tokens, int[] outputRows) {
            float[][] out = new float[outputRows == null ? tokens.length : outputRows.length][];
            for (int i = 0; i < out.length; i++) {
                out[i] = row.clone();
            }
            return out;
        }

        @Override
        public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            throw new UnsupportedOperationException("not used by the fixed-row test backend");
        }

        @Override
        public void reset() {
        }

        @Override
        public boolean canRollback(int nTokens) {
            return false;
        }

        @Override
        public void rollback(int nTokens) {
            throw new UnsupportedOperationException("not used by the fixed-row test backend");
        }

        @Override
        public void shiftContext(int nKeep, int nDiscard) {
            throw new UnsupportedOperationException("not used by the fixed-row test backend");
        }

        @Override
        public ModelMemory memory() {
            return null;
        }

        @Override
        public void setHiddenTap(HiddenTap tap) {
        }

        @Override
        public void close() {
        }
    }
}
