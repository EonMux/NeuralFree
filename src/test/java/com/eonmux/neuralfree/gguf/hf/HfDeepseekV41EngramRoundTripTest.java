package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfDeepseekV41Fixtures;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The DeepSeek-V4.1 Engram checkpoint round trip: a synthetic Hugging Face checkpoint with {@code engram_layer_ids}
 * converts through the real converter ({@link GgufWriter} over {@link ModelSources#open}), the exported GGUF's Engram
 * metadata and concatenated {@code per_layer_token_embd} table match an independent evaluation of the converter's
 * layout and E4M3/E8M0 dequantization, and the loaded model's forward logits match an independent implementation of
 * the Engram math.
 *
 * <p>The fixture ({@link HfDeepseekV41Fixtures#deepseekV41EngramRunnable()}) zeroes every attention and FFN output
 * projection and pins the hyper-connection coefficients to a fixed collapse onto stream 0 and a fixed Sinkhorn comb,
 * so a token's residual streams are its token embedding plus each Engram layer's gated value under that comb, and the
 * final RMS norm makes the output analytic in the head's collapsed stream. The reference reimplements the n-gram hash,
 * the unsigned rolling hash, the row lookup, the {@code engram_kv} projection, the RMS-normalized gate, the gated add
 * and the fixed hyper-connection mix from the checkpoint tensors and the golden layout; the GPU path runs the same
 * fixture where a device is available.</p>
 */
public class HfDeepseekV41EngramRoundTripTest {

    /** {@code engram_primes(2, 3, 2, 100)} rows per layer. */
    private static final int[] ENGRAM_ROWS = HfDeepseekV41Fixtures.RUNNABLE_ENGRAM_ROWS;
    /** The flattened bucket moduli: layer order, then n-gram order 2..3, then head. */
    private static final long[] HEAD_VOCAB = {101, 103, 107, 109, 113, 127, 131, 137};
    /** The absolute row offsets of {@link #HEAD_VOCAB} into the concatenated table. */
    private static final long[] HEAD_OFFSETS = {0, 101, 204, 311, 420, 533, 660, 791};
    /** numpy's {@code default_rng(10007 * layer)} multipliers, {@code 3} per layer, computed for the 7-key map. */
    private static final long[] MULTIPLIERS = {
            839276373626933875L, 355477604520660829L, 53987722132071145L,
            1084803951630661891L, 68513285976946179L, 509045121352709105L};
    /** The compressed token map of the seven-token synthetic tokenizer is the identity. */
    private static final int[] TOKEN_MAP = {0, 1, 2, 3, 4, 5, 6};
    private static final int NGRAM = 3;
    private static final int HEADS_PER_NGRAM = 2;
    private static final int COLS = (NGRAM - 1) * HEADS_PER_NGRAM;
    private static final int HEAD_DIM = HfDeepseekV41Fixtures.RUNNABLE_ENGRAM_DIM;
    private static final int N_EMBD = HfDeepseekV41Fixtures.RUNNABLE_DIM;
    private static final int HC = HfDeepseekV41Fixtures.RUNNABLE_HC;
    private static final float HC_EPS = HfDeepseekV41Fixtures.RUNNABLE_HC_EPS;
    private static final int HC_ITERS = HfDeepseekV41Fixtures.RUNNABLE_HC_ITERS;
    private static final int PAD_ID = 0;
    private static final float EPS = 1e-6f;
    private static final int[] PROMPT = {1, 2, 3, 4, 5};

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private HWAcceleration acceleration;
    private ComputeBackend backend;

    @Before
    public void setUpGpu() {
        if (GpuTestUtil.isOpenCLAvailable()) {
            acceleration = new HWAcceleration();
            backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
        }
    }

    @After
    public void tearDownGpu() {
        if (backend != null) {
            backend.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(32).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    @Test
    public void theConvertedCheckpointLoadsAndItsLogitsMatchTheEngramReference() throws IOException {
        Path dir = HfDeepseekV41Fixtures.deepseekV41EngramRunnable()
                .write(tmp.newFolder("hf-engram").toPath());
        Path gguf = tmp.getRoot().toPath().resolve("engram.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache").toPath()).build();
        try (ModelSource hf = ModelSources.open(dir.toString(), options, null)) {
            GgufWriter.write(hf, gguf, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }

        Reference reference = Reference.of(dir);
        assertConvertedLayout(gguf, reference);

        float[][] actual;
        try (GGUFModel model = new GGUFModel(gguf.toString(), config())) {
            int[] rows = new int[PROMPT.length];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = i;
            }
            actual = model.forwardBackend().forwardRows(PROMPT, 0, rows);
        }
        float[][] expected = reference.logits(PROMPT);
        assertEquals(PROMPT.length, actual.length);
        for (int t = 0; t < PROMPT.length; t++) {
            assertArrayEquals("logits of prompt row " + t, expected[t], actual[t], 5e-4f);
        }
    }

    @Test
    public void theGpuEngramPathMatchesTheReference() throws IOException {
        Assume.assumeTrue("OpenCL GPU not available", backend != null);
        Path dir = HfDeepseekV41Fixtures.deepseekV41EngramRunnable()
                .write(tmp.newFolder("hf-engram-gpu").toPath());
        Path gguf = tmp.getRoot().toPath().resolve("engram-gpu.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache-gpu").toPath()).build();
        try (ModelSource hf = ModelSources.open(dir.toString(), options, null)) {
            GgufWriter.write(hf, gguf, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        float[][] expected = Reference.of(dir).logits(PROMPT);
        float[][] actual;
        try (GGUFModelGPU model = new GGUFModelGPU(gguf.toString(), backend, config())) {
            int[] rows = new int[PROMPT.length];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = i;
            }
            actual = model.forwardBackend().forwardRows(PROMPT, 0, rows);
        }
        assertEquals(PROMPT.length, actual.length);
        for (int t = 0; t < PROMPT.length; t++) {
            assertArrayEquals("GPU logits of prompt row " + t, expected[t], actual[t], 5e-3f);
        }
    }

    /** The exported file's Engram keys and concatenated table, against the independent layout and dequantization. */
    private static void assertConvertedLayout(Path gguf, Reference reference) throws IOException {
        try (GGUFFileParser file = new GGUFFileParser(gguf.toString())) {
            assertEquals(java.util.Arrays.asList(0, 1), file.getMetadata().get("deepseek41.ple.layers"));
            assertNumber(3, file.getMetadata().get("deepseek41.ple.ngram_size"));
            assertNumber(2, file.getMetadata().get("deepseek41.ple.heads_per_ngram"));
            assertNumber(HEAD_DIM, file.getMetadata().get("deepseek41.embedding_length_per_layer_input"));
            assertNumbers(HEAD_OFFSETS, file.getMetadata().get("deepseek41.ple.head_offsets"));
            assertNumbers(HEAD_VOCAB, file.getMetadata().get("deepseek41.ple.head_vocab_sizes"));
            assertNumbers(MULTIPLIERS, file.getMetadata().get("deepseek41.ple.layer_multipliers"));
            assertNumber(0, file.getMetadata().get("deepseek41.engram.pad_token_id"));
            assertNumbers(toLongs(TOKEN_MAP), file.getMetadata().get("deepseek41.engram.token_map"));
            // The types llama.cpp's conversion/deepseek41.py writes (arrays included).
            assertEquals(GGUFValueType.ARRAY,
                    file.getValueType("deepseek41.ple.layers"));
            assertEquals(GGUFValueType.UINT32,
                    file.getArrayElementType("deepseek41.ple.layers"));
            assertEquals(GGUFValueType.UINT32,
                    file.getValueType("deepseek41.ple.ngram_size"));
            assertEquals(GGUFValueType.UINT32,
                    file.getValueType("deepseek41.ple.heads_per_ngram"));
            assertEquals(GGUFValueType.UINT32,
                    file.getValueType("deepseek41.embedding_length_per_layer_input"));
            assertEquals(GGUFValueType.UINT32,
                    file.getValueType("deepseek41.engram.pad_token_id"));
            assertEquals(GGUFValueType.UINT64,
                    file.getArrayElementType("deepseek41.ple.layer_multipliers"));
            assertEquals(GGUFValueType.UINT64,
                    file.getArrayElementType("deepseek41.ple.head_offsets"));
            assertEquals(GGUFValueType.UINT64,
                    file.getArrayElementType("deepseek41.ple.head_vocab_sizes"));
            assertEquals(GGUFValueType.INT32,
                    file.getArrayElementType("deepseek41.engram.token_map"));
            // The tensor shapes the runtime binds: engram_kv [n_embd, (hc + 1) * n_embd], engram_gate
            // [n_embd, hc], the shared table [ple_head_dim, rows].
            assertArrayEquals(new long[]{HEAD_DIM, 928}, file.getTensor("per_layer_token_embd.weight").dims());
            assertArrayEquals(new long[]{N_EMBD, (long) (HC + 1) * N_EMBD},
                    file.getTensor("blk.0.engram_kv.weight").dims());
            assertArrayEquals(new long[]{N_EMBD, HC}, file.getTensor("blk.0.engram_gate.weight").dims());
            assertArrayEquals(new long[]{N_EMBD, (long) (HC + 1) * N_EMBD},
                    file.getTensor("blk.1.engram_kv.weight").dims());
        }
        try (ModelSource source = ModelSources.open(gguf.toString());
             WeightStore store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            GGUFTensorInfo info = source.getTensors().stream()
                    .filter(t -> t.name().equals("per_layer_token_embd.weight")).findFirst().orElseThrow();
            assertEquals(GGMLType.BF16, info.type());
            byte[] bytes = new byte[(int) info.sizeInBytes()];
            store.readTensor(info.offset(), bytes.length, bytes);
            ByteBuffer b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            float[][] expected = reference.table();
            for (int r = 0; r < expected.length; r++) {
                for (int c = 0; c < HEAD_DIM; c++) {
                    assertEquals("table row " + r + ", column " + c, expected[r][c], bf16(b.getShort()), 0f);
                }
            }
        }
    }

    private static void assertNumber(long expected, Object actual) {
        assertNotNull(actual);
        assertEquals(expected, ((Number) actual).longValue());
    }

    private static void assertNumbers(long[] expected, Object actual) {
        assertNotNull(actual);
        List<?> list = (List<?>) actual;
        assertEquals(expected.length, list.size());
        for (int i = 0; i < expected.length; i++) {
            assertEquals("element " + i, expected[i], ((Number) list.get(i)).longValue());
        }
    }

    private static long[] toLongs(int[] values) {
        long[] out = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out;
    }

    private static float bf16(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    /**
     * The independent reference: the checkpoint's tensors (read from the safetensors files, not from the converted
     * GGUF), the golden Engram layout, and the Engram math of models/deepseek41.cpp:240-292, 372-410.
     */
    private static final class Reference {

        private final float[][] tokenEmbd;
        private final float[][] head;
        private final float[] outNorm;
        private final float[][][] kv;      // [layer][row][column], rows = key then value
        private final float[][] gateW;     // [layer][n_embd]
        private final float[][] table;     // [global row][head dim]
        private final int pad;

        private Reference(float[][] tokenEmbd, float[][] head, float[] outNorm, float[][][] kv,
                          float[][] gateW, float[][] table) {
            this.tokenEmbd = tokenEmbd;
            this.head = head;
            this.outNorm = outNorm;
            this.kv = kv;
            this.gateW = gateW;
            this.table = table;
            this.pad = TOKEN_MAP[PAD_ID];
        }

        static Reference of(Path dir) throws IOException {
            try (CheckpointContainer ckpt = CheckpointContainers.open(dir, dir, false)) {
                HfTensorIndex index = HfTensorIndex.of(ckpt);
                float[][] embed = matrix(index.get("embed.weight"));
                float[][] head = matrix(index.get("head.weight"));
                float[] norm = vector(index.get("norm.weight"));
                int layers = ENGRAM_ROWS.length;
                float[][][] kv = new float[layers][][];
                float[][] gateW = new float[layers][];
                float[][] table = new float[totalRows()][];
                int at = 0;
                for (int e = 0; e < layers; e++) {
                    String prefix = "layers." + e + ".engram.";
                    kv[e] = matrix(index.get(prefix + "wkv.weight"));
                    float[] q = vector(index.get(prefix + "q_weight"));
                    float[] k = vector(index.get(prefix + "k_weight"));
                    gateW[e] = new float[HC * N_EMBD];
                    for (int i = 0; i < gateW[e].length; i++) {
                        gateW[e][i] = q[i] * k[i];
                    }
                    float[][] rows = dequantise(index.get(prefix + "embed.weight"),
                            index.get(prefix + "embed.scale"), ENGRAM_ROWS[e]);
                    for (float[] row : rows) {
                        table[at++] = row;
                    }
                }
                return new Reference(embed, head, norm, kv, gateW, table);
            }
        }

        private static int totalRows() {
            int n = 0;
            for (int rows : ENGRAM_ROWS) {
                n += rows;
            }
            return n;
        }

        float[][] table() {
            return table;
        }

        /**
         * The logits of every prompt row. The fixture's hyper-connection coefficients collapse with {@code pre} and
         * mix the streams with the constant Sinkhorn {@code comb} twice per layer (the attention and FFN branches are
         * zero), and each Engram layer gates its value against every stream independently.
         */
        float[][] logits(int[] prompt) {
            int n = prompt.length;
            float[][][] x = new float[n][HC][];
            for (int t = 0; t < n; t++) {
                for (int h = 0; h < HC; h++) {
                    x[t][h] = tokenEmbd[prompt[t]].clone();
                }
            }
            float[] pre = pre();
            float[] comb = comb();
            for (int e = 0; e < ENGRAM_ROWS.length; e++) {
                for (int t = 0; t < n; t++) {
                    int[] ids = hash(e, prompt, t);
                    float[] emb = new float[COLS * HEAD_DIM];
                    for (int c = 0; c < COLS; c++) {
                        System.arraycopy(table[ids[c]], 0, emb, c * HEAD_DIM, HEAD_DIM);
                    }
                    float[] value = project(kv[e], emb, HC * N_EMBD);
                    for (int h = 0; h < HC; h++) {
                        float[] key = project(kv[e], emb, h * N_EMBD);
                        float[] nx = rms(x[t][h]);
                        float[] nk = rms(key);
                        double dot = 0;
                        for (int i = 0; i < N_EMBD; i++) {
                            dot += (double) nx[i] * nk[i] * gateW[e][h * N_EMBD + i];
                        }
                        dot /= Math.sqrt(N_EMBD);
                        float gate = gate((float) dot);
                        for (int i = 0; i < N_EMBD; i++) {
                            x[t][h][i] += value[i] * gate;
                        }
                    }
                    x[t] = applyComb(applyComb(x[t], comb), comb);
                }
            }
            float[][] logits = new float[n][head.length];
            for (int t = 0; t < n; t++) {
                float[] hidden = new float[N_EMBD];
                for (int h = 0; h < HC; h++) {
                    for (int i = 0; i < N_EMBD; i++) {
                        hidden[i] += pre[h] * x[t][h][i];
                    }
                }
                float[] normed = rms(hidden);
                for (int v = 0; v < head.length; v++) {
                    double sum = 0;
                    for (int i = 0; i < N_EMBD; i++) {
                        sum += (double) head[v][i] * normed[i] * outNorm[i];
                    }
                    logits[t][v] = (float) sum;
                }
            }
            return logits;
        }

        /**
         * The collapse coefficients of the fixture's zero {@code hc_*_fn}/{@code hc_*_scale}: {@code sigmoid(base)} +
         * {@code hc_eps} per stream, so the first stream survives and the others carry only {@code hc_eps}.
         */
        private static float[] pre() {
            float[] out = new float[HC];
            for (int h = 0; h < HC; h++) {
                double logit = h == 0 ? HfDeepseekV41Fixtures.HC_HIGH : HfDeepseekV41Fixtures.HC_LOW;
                out[h] = (float) (1.0 / (1.0 + Math.exp(-logit))) + HC_EPS;
            }
            return out;
        }

        /**
         * The Sinkhorn mixing matrix of the fixture's comb logits ({@code +100} on the identity, {@code -100}
         * elsewhere), the {@code LLM_HC_PrePostComb} math of models/deepseek4.cpp.
         */
        private static float[] comb() {
            float high = HfDeepseekV41Fixtures.HC_HIGH;
            float low = HfDeepseekV41Fixtures.HC_LOW;
            float[] base = {0f, 0f, 0f, 0f, high, low, low, high};
            float[] c = new float[HC * HC];
            for (int src = 0; src < HC; src++) {
                float max = Float.NEGATIVE_INFINITY;
                for (int dst = 0; dst < HC; dst++) {
                    c[dst + HC * src] = base[2 * HC + dst + HC * src];
                    max = Math.max(max, c[dst + HC * src]);
                }
                float sum = 0;
                for (int dst = 0; dst < HC; dst++) {
                    float v = (float) Math.exp(c[dst + HC * src] - max);
                    c[dst + HC * src] = v;
                    sum += v;
                }
                for (int dst = 0; dst < HC; dst++) {
                    c[dst + HC * src] = c[dst + HC * src] / sum + HC_EPS;
                }
            }
            normalizeCols(c);
            for (int it = 1; it < HC_ITERS; it++) {
                normalizeRows(c);
                normalizeCols(c);
            }
            return c;
        }

        private static void normalizeCols(float[] c) {
            for (int dst = 0; dst < HC; dst++) {
                float sum = HC_EPS;
                for (int src = 0; src < HC; src++) {
                    sum += c[dst + HC * src];
                }
                for (int src = 0; src < HC; src++) {
                    c[dst + HC * src] /= sum;
                }
            }
        }

        private static void normalizeRows(float[] c) {
            for (int src = 0; src < HC; src++) {
                float sum = HC_EPS;
                for (int dst = 0; dst < HC; dst++) {
                    sum += c[dst + HC * src];
                }
                for (int dst = 0; dst < HC; dst++) {
                    c[dst + HC * src] /= sum;
                }
            }
        }

        private static float[][] applyComb(float[][] x, float[] comb) {
            float[][] out = new float[HC][];
            for (int dst = 0; dst < HC; dst++) {
                out[dst] = new float[N_EMBD];
                for (int src = 0; src < HC; src++) {
                    for (int i = 0; i < N_EMBD; i++) {
                        out[dst][i] += comb[dst + HC * src] * x[src][i];
                    }
                }
            }
            return out;
        }

        /**
         * The n-gram row ids of prompt position {@code t} (engram.py {@code NgramHashState.forward}): shift 0 is the
         * token, shift s the token s positions back; a missing token blocks that shift and every longer one, and a
         * blocked slot hashes as the padding token. The unsigned 64-bit rolling hash is evaluated with {@link
         * BigInteger} so it is independent of the production long arithmetic.
         */
        private int[] hash(int layer, int[] prompt, int t) {
            long[] toks = new long[NGRAM];
            int cur = mapped(prompt[t]);
            boolean blocked = cur < 0;
            toks[0] = blocked ? pad : cur;
            for (int s = 1; s < NGRAM; s++) {
                int past = t - s;
                int c = past < 0 ? -1 : mapped(prompt[past]);
                blocked |= c < 0;
                toks[s] = blocked ? pad : c;
            }
            BigInteger rolling = BigInteger.valueOf(toks[0]).multiply(BigInteger.valueOf(MULTIPLIERS[layer * NGRAM]));
            int[] ids = new int[COLS];
            for (int o = 1; o < NGRAM; o++) {
                rolling = rolling.xor(BigInteger.valueOf(toks[o])
                        .multiply(BigInteger.valueOf(MULTIPLIERS[layer * NGRAM + o])));
                for (int h = 0; h < HEADS_PER_NGRAM; h++) {
                    int col = layer * COLS + (o - 1) * HEADS_PER_NGRAM + h;
                    ids[(o - 1) * HEADS_PER_NGRAM + h] = (int) unsignedMod(rolling, HEAD_VOCAB[col])
                            + (int) HEAD_OFFSETS[col];
                }
            }
            return ids;
        }

        private int mapped(int token) {
            return token < 0 || token >= TOKEN_MAP.length ? -1 : TOKEN_MAP[token];
        }

        private static long unsignedMod(BigInteger value, long modulus) {
            return value.mod(BigInteger.valueOf(modulus)).longValue();
        }

        private static float[] project(float[][] w, float[] emb, int from) {
            float[] out = new float[N_EMBD];
            for (int r = 0; r < N_EMBD; r++) {
                double sum = 0;
                for (int c = 0; c < emb.length; c++) {
                    sum += (double) w[from + r][c] * emb[c];
                }
                out[r] = (float) sum;
            }
            return out;
        }

        /** {@code 1/sqrt(mean(x^2) + eps)} (no weight), as {@code NormMath}'s RMS row norm. */
        private static float[] rms(float[] x) {
            double sum = 0;
            for (float v : x) {
                sum += (double) v * v;
            }
            float factor = (float) (1.0 / Math.sqrt(sum / x.length + EPS));
            float[] out = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                out[i] = x[i] * factor;
            }
            return out;
        }

        /** {@code sigmoid(copysign(sqrt(max(|s|, 1e-6)), s))}; zero takes the positive sign. */
        private static float gate(float s) {
            double mag = Math.sqrt(Math.max(Math.abs((double) s), 1e-6));
            double signed = s < 0f ? -mag : mag;
            return (float) (1.0 / (1.0 + Math.exp(-signed)));
        }

        /** A checkpoint matrix as row-major floats. */
        private static float[][] matrix(RowSource src) throws IOException {
            long[] shape = src.shape();
            assertEquals(2, shape.length);
            int rows = Math.toIntExact(shape[0]);
            int cols = Math.toIntExact(shape[1]);
            float[] flat = floats(src, rows, cols);
            float[][] out = new float[rows][];
            for (int r = 0; r < rows; r++) {
                out[r] = java.util.Arrays.copyOfRange(flat, r * cols, (r + 1) * cols);
            }
            return out;
        }

        private static float[] vector(RowSource src) throws IOException {
            long[] shape = src.shape();
            int n = 1;
            for (long d : shape) {
                n *= Math.toIntExact(d);
            }
            int rows = Math.toIntExact(src.rows());
            return floats(src, rows, n / rows);
        }

        private static float[] floats(RowSource src, int rows, int cols) throws IOException {
            StDType dtype = src.format().stDType();
            ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(src.rowBytes()) * rows)
                    .order(ByteOrder.LITTLE_ENDIAN);
            src.readRows(0, rows, b);
            b.flip();
            float[] out = new float[rows * cols];
            if (dtype == StDType.BF16) {
                for (int i = 0; i < out.length; i++) {
                    out[i] = bf16(b.getShort());
                }
            } else if (dtype == StDType.F32) {
                for (int i = 0; i < out.length; i++) {
                    out[i] = b.getFloat();
                }
            } else {
                throw new IOException("unexpected dtype " + dtype);
            }
            return out;
        }

        /**
         * The converter's Engram table: {@code code * 2^(e - 127)} with one E8M0 scale per 32 values, rows in the
         * checkpoint order.
         */
        private static float[][] dequantise(RowSource weight, RowSource scale, int rows) throws IOException {
            int cols = Math.toIntExact(weight.shape()[1]);
            ByteBuffer w = ByteBuffer.allocate(Math.toIntExact(weight.rowBytes()) * rows)
                    .order(ByteOrder.LITTLE_ENDIAN);
            weight.readRows(0, rows, w);
            w.flip();
            ByteBuffer s = ByteBuffer.allocate(Math.toIntExact(scale.rowBytes()) * rows)
                    .order(ByteOrder.LITTLE_ENDIAN);
            scale.readRows(0, rows, s);
            s.flip();
            float[][] out = new float[rows][cols];
            for (int r = 0; r < rows; r++) {
                float scaleValue = 1f;
                for (int c = 0; c < cols; c++) {
                    if (c % 32 == 0) {
                        int exponent = s.get() & 0xFF;
                        assertTrue("E8M0 exponent " + exponent, exponent >= 3 && exponent <= 246);
                        scaleValue = Float.intBitsToFloat(exponent << 23);
                    }
                    out[r][c] = FpBlockDecoders.e4m3ToFp32(w.get() & 0xFF) * scaleValue;
                }
            }
            return out;
        }
    }
}
