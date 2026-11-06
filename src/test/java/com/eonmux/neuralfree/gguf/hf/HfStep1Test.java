package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Step1 checkpoint loading: a rotary-free decoder with vLLM's square-root ALiBi, RMSNorm and a biased gated SiLU
 * feed-forward. Forward parity against an independent scalar implementation of {@code step1.py} on CPU and OpenCL,
 * the converted layout, the export round trip and the named refusals.
 */
public class HfStep1Test {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 4;
    private static final int V = 7;
    private static final int HEADS = 2;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = D / HEADS;
    private static final int FFN = 4;
    private static final float EPS = 1e-5f;

    private static final String CONFIG = SyntheticHf.json("{'architectures':['Step1ForCausalLM'],"
            + "'model_type':'step1','hidden_size':" + D + ",'intermediate_size':" + FFN
            + ",'num_hidden_layers':2,'num_attention_heads':" + HEADS + ",'num_attention_groups':" + KV_HEADS
            + ",'max_position_embeddings':16,'rms_norm_eps':" + EPS
            + ",'vocab_size':" + V + ",'attention_bias':true,'mlp_bias':true,'hidden_act':'silu'"
            + ",'tie_word_embeddings':false}");

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        assertForward(path, reference(new int[]{2, 3, 4}));
    }

    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(checkpoint().toString(), cfg)) {
            assertArrayEquals(reference(new int[]{2, 3, 4}),
                    model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone(), 3e-4f);
        }
    }

    @Test
    public void conversionWritesTheStep1Layout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            Map<String, Object> md = src.getMetadata();
            assertEquals("step1", md.get("general.architecture"));
            assertEquals(2, ((Number) md.get("step1.block_count")).intValue());
            assertEquals(2, ((Number) md.get("step1.attention.head_count")).intValue());
            assertEquals(2, ((Number) md.get("step1.attention.head_count_kv")).intValue());
            assertEquals(8.0, ((Number) md.get("step1.attention.max_alibi_bias")).doubleValue(), 1e-9);
            assertEquals(EPS, ((Number) md.get("step1.attention.layer_norm_rms_epsilon")).doubleValue(), 1e-9);
            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            for (String expected : new String[]{"token_embd.weight", "output_norm.weight", "output.weight",
                    "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_q.bias", "blk.0.attn_k.weight",
                    "blk.0.attn_k.bias", "blk.0.attn_v.weight", "blk.0.attn_v.bias", "blk.0.attn_output.weight",
                    "blk.0.attn_output.bias", "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_gate.bias",
                    "blk.0.ffn_up.weight", "blk.0.ffn_up.bias", "blk.0.ffn_down.weight", "blk.0.ffn_down.bias"}) {
                assertTrue("missing " + expected, tensors.contains(expected));
            }
        }
    }

    @Test
    public void exportedGgufLoadsAsStep1() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("step1.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        assertForward(out, reference(new int[]{2, 3, 4}));
    }

    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("Step1ForCausalLM"));
        assertEquals(RopeType.NONE, ArchRegistry.traits(Arch.STEP1).ropeType());
    }

    @Test
    public void refusesANonSiluActivation() throws Exception {
        String config = CONFIG.substring(0, CONFIG.length() - 1) + ",\"hidden_act\":\"gelu\"}";
        Path dir = builder(config).write(tmp.newFolder("rejected").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), names(e, "hidden_act"));
    }

    private static boolean names(Throwable e, String field) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t.getMessage() != null && t.getMessage().contains(field)) {
                return true;
            }
        }
        return false;
    }

    private void assertForward(Path path, float[] expected) throws Exception {
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            assertArrayEquals(expected, model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone(), 3e-4f);
        }
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("step1").toPath());
    }

    private static SyntheticHf.Builder builder() {
        return builder(CONFIG);
    }

    private static SyntheticHf.Builder builder(String config) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config);
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            int seed = 2 + 20 * l;
            tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "self_attn.q_proj.weight", D, D, matrix(D, D, seed));
            tensor(b, p + "self_attn.q_proj.bias", D, 1, bias(seed + 1));
            tensor(b, p + "self_attn.k_proj.weight", D, D, matrix(D, D, seed + 2));
            tensor(b, p + "self_attn.k_proj.bias", D, 1, bias(seed + 3));
            tensor(b, p + "self_attn.v_proj.weight", D, D, matrix(D, D, seed + 4));
            tensor(b, p + "self_attn.v_proj.bias", D, 1, bias(seed + 5));
            tensor(b, p + "self_attn.o_proj.weight", D, D, matrix(D, D, seed + 6));
            tensor(b, p + "self_attn.o_proj.bias", D, 1, bias(seed + 7));
            tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "mlp.gate_proj.weight", FFN, D, matrix(FFN, D, seed + 8));
            tensor(b, p + "mlp.gate_proj.bias", FFN, 1, bias(seed + 9, FFN));
            tensor(b, p + "mlp.up_proj.weight", FFN, D, matrix(FFN, D, seed + 10));
            tensor(b, p + "mlp.up_proj.bias", FFN, 1, bias(seed + 11, FFN));
            tensor(b, p + "mlp.down_proj.weight", D, FFN, matrix(D, FFN, seed + 12));
            tensor(b, p + "mlp.down_proj.bias", D, 1, bias(seed + 13));
        }
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 40));
        return b;
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = cols == 1 ? new long[]{rows} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 5) % 19 - 9) * 0.025f;
        }
        return x;
    }

    private static float[] ones(int n) {
        float[] x = new float[n];
        java.util.Arrays.fill(x, 1.0f);
        return x;
    }

    private static float[] bias(int seed) {
        return bias(seed, D);
    }

    private static float[] bias(int seed, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (i - seed % 3) * 0.02f;
        }
        return x;
    }

    // ------------------------------------------------------------------ scalar reference

    private static float[] linear(float[] x, float[] w, float[] bias, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            y[i] = bias == null ? 0 : bias[i];
            for (int j = 0; j < x.length; j++) {
                y[i] += x[j] * w[i * x.length + j];
            }
        }
        return y;
    }

    private static float[] rms(float[] x, float[] w) {
        double ss = 0;
        for (float v : x) {
            ss += (double) v * v;
        }
        float scale = (float) (1.0 / Math.sqrt(ss / x.length + EPS));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * scale * w[i];
        }
        return y;
    }

    /** vLLM {@code _get_step_alibi_slopes} for two heads and maximum bias 8: 2^(-8/2)^(h+1). */
    private static float[] slopes() {
        float base = (float) Math.pow(2.0, -8.0 / HEADS);
        float[] s = new float[HEADS];
        for (int h = 0; h < HEADS; h++) {
            s[h] = (float) Math.pow(base, h + 1);
        }
        return s;
    }

    private static float[] siluMul(float[] gate, float[] up) {
        float[] y = new float[gate.length];
        for (int i = 0; i < gate.length; i++) {
            y[i] = gate[i] / (1.0f + (float) Math.exp(-gate[i])) * up[i];
        }
        return y;
    }

    private static float[][] layerAttention(float[][] x, int layer) {
        int t = x.length;
        float[] slopes = slopes();
        int qSize = HEADS * HEAD_DIM;
        int seed = 2 + 20 * layer;
        float[][][] q = new float[t][HEADS][HEAD_DIM];
        float[][][] k = new float[t][HEADS][HEAD_DIM];
        float[][][] v = new float[t][HEADS][HEAD_DIM];
        for (int p = 0; p < t; p++) {
            float[] n = rms(x[p], ones(D));
            float[] qv = linear(n, matrix(D, D, seed), bias(seed + 1), D);
            float[] kv = linear(n, matrix(D, D, seed + 2), bias(seed + 3), D);
            float[] vv = linear(n, matrix(D, D, seed + 4), bias(seed + 5), D);
            for (int h = 0; h < HEADS; h++) {
                System.arraycopy(qv, h * HEAD_DIM, q[p][h], 0, HEAD_DIM);
                System.arraycopy(kv, h * HEAD_DIM, k[p][h], 0, HEAD_DIM);
                System.arraycopy(vv, h * HEAD_DIM, v[p][h], 0, HEAD_DIM);
            }
        }
        float[][] out = new float[t][D];
        float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        for (int p = 0; p < t; p++) {
            float[] heads = new float[qSize];
            for (int h = 0; h < HEADS; h++) {
                float[] scores = new float[p + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j <= p; j++) {
                    float dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dot += q[p][h][d] * k[j][h][d];
                    }
                    // vLLM use_alibi_sqrt: slope * -sqrt(query - key) for past keys.
                    scores[j] = dot * scale + slopes[h] * -(float) Math.sqrt(p - j);
                    max = Math.max(max, scores[j]);
                }
                float sum = 0;
                for (int j = 0; j <= p; j++) {
                    scores[j] = (float) Math.exp(scores[j] - max);
                    sum += scores[j];
                }
                for (int j = 0; j <= p; j++) {
                    for (int d = 0; d < HEAD_DIM; d++) {
                        heads[h * HEAD_DIM + d] += scores[j] / sum * v[j][h][d];
                    }
                }
            }
            out[p] = linear(heads, matrix(D, D, seed + 6), bias(seed + 7), D);
        }
        return out;
    }

    private static float[] reference(int[] tokens) {
        int t = tokens.length;
        float[][] x = new float[t][D];
        float[] emb = matrix(V, D, 1);
        for (int p = 0; p < t; p++) {
            System.arraycopy(emb, tokens[p] * D, x[p], 0, D);
        }
        for (int l = 0; l < 2; l++) {
            float[][] attn = layerAttention(x, l);
            int seed = 2 + 20 * l;
            for (int p = 0; p < t; p++) {
                for (int i = 0; i < D; i++) {
                    x[p][i] += attn[p][i];
                }
                float[] n = rms(x[p], ones(D));
                float[] gate = linear(n, matrix(FFN, D, seed + 8), bias(seed + 9, FFN), FFN);
                float[] up = linear(n, matrix(FFN, D, seed + 10), bias(seed + 11, FFN), FFN);
                float[] down = linear(siluMul(gate, up), matrix(D, FFN, seed + 12), bias(seed + 13), D);
                for (int i = 0; i < D; i++) {
                    x[p][i] += down[i];
                }
            }
        }
        return linear(rms(x[t - 1], ones(D)), matrix(V, D, 40), null, V);
    }
}
