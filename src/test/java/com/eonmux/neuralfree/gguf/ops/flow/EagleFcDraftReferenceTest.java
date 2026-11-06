package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The concat-fc drafters' first proposal against independent references of vLLM's forward: the fusion projection
 * over the normed target hidden and the token embedding, then the family's decoder layer at position 0 (where RoPE
 * is the identity and a one-token attention is the value projection), its output norm and its LM head. The four
 * cases cover the Cohere parallel LayerNorm block and its biased {@code fc} with {@code logit_scale}, MiniCPM's two
 * fusion-part norms and {@code scale_depth} residual, Llama 4's chunked-local layer with the weightless Q/K norm,
 * and Mistral Large 3's multi-head latent attention.
 */
public class EagleFcDraftReferenceTest {

    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int HEAD_DIM = 16;
    private static final int N_FF = 128;
    private static final int N_VOCAB = 320;
    private static final int TARGET_LAYERS = 3;
    private static final int TAP = TARGET_LAYERS;
    private static final int PROMPT_TOKEN = 7;
    private static final float EPS = 1e-5f;

    private static final int MLA_HEAD_DIM = 32;
    private static final int MLA_ROT = 16;
    private static final int MLA_NOPE = MLA_HEAD_DIM - MLA_ROT;
    private static final int KV_LORA = 32;
    private static final int Q_LORA = 32;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theCohereDraftMatchesTheReferenceForward() throws IOException {
        String arch = "eagle-cohere";
        Weights ref = new Weights(arch);
        GgufTestWriter w = new GgufTestWriter().architecture(arch);
        base(w, arch, TAP);
        w.f32(w.archKey("attention.layer_norm_epsilon"), EPS);
        w.f32(w.archKey("logit_scale"), 0.5f);
        ref.matrix(w, "token_embd.weight", N_EMBD, N_VOCAB);
        ref.matrix(w, "fc.weight", 2 * N_EMBD, N_EMBD);
        ref.vector(w, "fc.bias", N_EMBD);
        ref.ones(w, "output_norm.weight", N_EMBD);
        ref.matrix(w, "output.weight", N_EMBD, N_VOCAB);
        ref.ones(w, "blk.0.attn_norm.weight", N_EMBD);
        ref.qkvo(w, 0);
        ref.ffn(w, 0);
        assertProposal(w.writeTempFile(), ref);
    }

    @Test
    public void theMiniCpmDraftMatchesTheReferenceForward() throws IOException {
        String arch = "eagle-minicpm";
        float scaleEmb = 12f;
        float residual = 1.4f / 2f;
        float hiddenScale = 0.5f;
        Weights ref = new Weights(arch);
        GgufTestWriter w = new GgufTestWriter().architecture(arch);
        base(w, arch, TAP);
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), EPS);
        w.f32(w.archKey("embedding_scale"), scaleEmb);
        w.f32(w.archKey("residual_scale"), residual);
        w.f32(w.archKey("output_hidden_scale"), hiddenScale);
        ref.matrix(w, "token_embd.weight", N_EMBD, N_VOCAB);
        ref.matrix(w, "fc.weight", 2 * N_EMBD, N_EMBD);
        ref.ones(w, "enc.input_norm1.weight", N_EMBD);
        ref.ones(w, "enc.input_norm2.weight", N_EMBD);
        ref.matrix(w, "output.weight", N_EMBD, N_VOCAB);
        ref.ones(w, "blk.0.attn_norm.weight", N_EMBD);
        ref.qkvo(w, 0);
        ref.ones(w, "blk.0.ffn_norm.weight", N_EMBD);
        ref.ffn(w, 0);
        assertProposal(w.writeTempFile(), ref);
    }

    @Test
    public void theLlama4DraftMatchesTheReferenceForward() throws IOException {
        String arch = "eagle-llama4";
        Weights ref = new Weights(arch);
        GgufTestWriter w = new GgufTestWriter().architecture(arch);
        base(w, arch, TAP);
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), EPS);
        w.u32(w.archKey("attention.sliding_window"), 8192);
        w.boolArray(w.archKey("attention.sliding_window_pattern"), true);
        w.boolArray(w.archKey("attention.rope_pattern"), true);
        w.bool(w.archKey("attention.use_qk_norm"), true);
        ref.matrix(w, "token_embd.weight", N_EMBD, N_VOCAB);
        ref.matrix(w, "fc.weight", 2 * N_EMBD, N_EMBD);
        ref.ones(w, "output_norm.weight", N_EMBD);
        ref.matrix(w, "output.weight", N_EMBD, N_VOCAB);
        ref.ones(w, "blk.0.attn_norm.weight", N_EMBD);
        ref.qkvo(w, 0);
        ref.ones(w, "blk.0.ffn_norm.weight", N_EMBD);
        ref.ffn(w, 0);
        assertProposal(w.writeTempFile(), ref);
    }

    @Test
    public void theMistralLarge3DraftMatchesTheReferenceForward() throws IOException {
        String arch = "eagle-mistral-large3";
        Weights ref = new Weights(arch);
        GgufTestWriter w = new GgufTestWriter().architecture(arch);
        base(w, arch, TAP);
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), EPS);
        w.u32(w.archKey("feed_forward_length"), N_FF);
        w.u32(w.archKey("attention.head_count"), N_HEAD);
        w.u32(w.archKey("attention.head_count_kv"), 1);
        w.u32(w.archKey("attention.key_length"), KV_LORA + MLA_ROT);
        w.u32(w.archKey("attention.value_length"), KV_LORA);
        w.u32(w.archKey("attention.key_length_mla"), MLA_HEAD_DIM);
        w.u32(w.archKey("attention.value_length_mla"), MLA_HEAD_DIM);
        w.u32(w.archKey("attention.q_lora_rank"), Q_LORA);
        w.u32(w.archKey("attention.kv_lora_rank"), KV_LORA);
        w.u32(w.archKey("rope.dimension_count"), MLA_ROT);
        ref.matrix(w, "token_embd.weight", N_EMBD, N_VOCAB);
        ref.matrix(w, "fc.weight", 2 * N_EMBD, N_EMBD);
        ref.ones(w, "output_norm.weight", N_EMBD);
        ref.matrix(w, "output.weight", N_EMBD, N_VOCAB);
        ref.ones(w, "blk.0.attn_norm.weight", N_EMBD);
        ref.matrix(w, "blk.0.attn_q_a.weight", N_EMBD, Q_LORA);
        ref.ones(w, "blk.0.attn_q_a_norm.weight", Q_LORA);
        ref.matrix(w, "blk.0.attn_q_b.weight", Q_LORA, N_HEAD * MLA_HEAD_DIM);
        ref.matrix(w, "blk.0.attn_kv_a_mqa.weight", N_EMBD, KV_LORA + MLA_ROT);
        ref.ones(w, "blk.0.attn_kv_a_norm.weight", KV_LORA);
        ref.tensor(w, "blk.0.attn_k_b.weight", MLA_NOPE, KV_LORA, N_HEAD);
        ref.tensor(w, "blk.0.attn_v_b.weight", KV_LORA, MLA_HEAD_DIM, N_HEAD);
        ref.matrix(w, "blk.0.attn_output.weight", N_HEAD * MLA_HEAD_DIM, N_EMBD);
        ref.ones(w, "blk.0.ffn_norm.weight", N_EMBD);
        ref.ffn(w, 0);
        assertProposal(w.writeTempFile(), ref);
    }

    // ------------------------------------------------------------------ the engine run

    /** Runs the target once and the drafter once, and compares the first proposal with {@code referenceArgmax}. */
    private void assertProposal(Path draftPath, Weights ref) throws IOException {
        Path targetPath = ref.target().writeTempFile();
        int[] proposal;
        float[] tapsAtLast;
        int lastToken;
        try (EngineModel target = LLMModelFactory.open(targetPath.toString(), config())) {
            ForwardBackend fb = target.forwardBackend();
            RecordingTap rec = new RecordingTap();
            fb.setHiddenTap(rec);
            lastToken = argmax(fb.forwardRows(new int[]{PROMPT_TOKEN}, 0, new int[]{0})[0]);
            tapsAtLast = rec.row();
            fb.setHiddenTap(HiddenTap.NONE);
            target.resetKVCache();

            try (ModelSource dp = ModelSources.open(draftPath.toString(), ModelSourceOptions.defaults(), null,
                    ActivationQuant.WEIGHT_ONLY)) {
                GgufKv kv = GgufKv.of(dp);
                TensorCatalog catalog = TensorCatalog.of(dp);
                ModelSpec spec = ModelSpecBuilder.build(kv, catalog, memory());
                try (CpuWeights weights = CpuWeights.open(dp)) {
                    MtpHead head = MtpHead.cpu(((GGUFModel) target).forwardContext(),
                            ((GGUFModel) target).blockRegistry(), weights);
                    DraftModel drafter = DraftModelLoader.fromFile(target, spec, head, memory(),
                            DraftModelLoader.Options.DEFAULT);
                    try {
                        fb.forwardRows(new int[]{PROMPT_TOKEN}, 0, new int[]{0});
                        proposal = drafter.draft(lastToken, 1);
                    } finally {
                        drafter.close();
                    }
                }
            }
        }
        assertEquals("one proposed token", 1, proposal.length);
        float[] hidden = rms(tapsAtLast, ref.targetEps);
        assertEquals("the concat-fc draft's argmax matches the reference", ref.referenceArgmax(lastToken, hidden),
                proposal[0]);
    }

    private static EngineConfig config() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory()).build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    private static void base(GgufTestWriter w, String arch, int targetLayer) {
        w.str("general.name", "synthetic-" + arch);
        w.u32(w.archKey("vocab_size"), N_VOCAB);
        w.u32(w.archKey("context_length"), 64);
        w.u32(w.archKey("embedding_length"), N_EMBD);
        w.u32(w.archKey("block_count"), 1);
        w.u32(w.archKey("feed_forward_length"), N_FF);
        w.u32(w.archKey("attention.head_count"), N_HEAD);
        w.u32(w.archKey("attention.head_count_kv"), N_HEAD_KV);
        w.u32(w.archKey("attention.key_length"), HEAD_DIM);
        w.u32(w.archKey("attention.value_length"), HEAD_DIM);
        w.u32(w.archKey("rope.dimension_count"), HEAD_DIM);
        w.f32(w.archKey("rope.freq_base"), 10000f);
        w.i32Array(w.archKey("target_layers"), targetLayer);
        w.u32(w.archKey("target_hidden_size"), N_EMBD);
    }

    // ------------------------------------------------------------------ reference weights

    /**
     * The synthetic weights of one draft file, and the reference's copy: an LCG keyed by the tensor name, so both
     * sides compute from the same values.
     */
    private static final class Weights {
        private final String family;
        private final float targetEps;
        private final Map<String, float[]> values = new HashMap<>();

        Weights(String family) {
            this.family = family;
            this.targetEps = family.equals("eagle-mistral-large3") ? 1e-6f : EPS;
        }

        GgufTestWriter target() {
            if (family.equals("eagle-mistral-large3")) {
                return SyntheticModels.deepseek2(SyntheticModels.Config.mistralLarge4().toBuilder()
                        .nLayer(TARGET_LAYERS).build());
            }
            return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(TARGET_LAYERS).build());
        }

        void matrix(GgufTestWriter w, String name, int... ne) {
            tensor(w, name, ne);
        }

        void tensor(GgufTestWriter w, String name, int... ne) {
            long count = 1;
            long[] shape = new long[ne.length];
            for (int i = 0; i < ne.length; i++) {
                count *= ne[i];
                shape[i] = ne[i];
            }
            w.tensorF32(name, shape, get(name, Math.toIntExact(count)));
        }

        void vector(GgufTestWriter w, String name, int n) {
            tensor(w, name, n);
        }

        void ones(GgufTestWriter w, String name, int n) {
            w.tensorFilled(name, GGMLType.F32, 1.0f, n);
            values.put(name, filled(n));
        }

        void qkvo(GgufTestWriter w, int il) {
            matrix(w, "blk." + il + ".attn_q.weight", N_EMBD, N_HEAD * HEAD_DIM);
            matrix(w, "blk." + il + ".attn_k.weight", N_EMBD, N_HEAD_KV * HEAD_DIM);
            matrix(w, "blk." + il + ".attn_v.weight", N_EMBD, N_HEAD_KV * HEAD_DIM);
            matrix(w, "blk." + il + ".attn_output.weight", N_HEAD * HEAD_DIM, N_EMBD);
        }

        void ffn(GgufTestWriter w, int il) {
            matrix(w, "blk." + il + ".ffn_gate.weight", N_EMBD, N_FF);
            matrix(w, "blk." + il + ".ffn_up.weight", N_EMBD, N_FF);
            matrix(w, "blk." + il + ".ffn_down.weight", N_FF, N_EMBD);
        }

        private static float[] filled(int n) {
            float[] out = new float[n];
            java.util.Arrays.fill(out, 1.0f);
            return out;
        }

        float[] get(String name, int count) {
            float[] cached = values.get(name);
            if (cached != null) {
                if (cached.length != count) {
                    throw new IllegalStateException(name + " has " + cached.length + " values, asked for " + count);
                }
                return cached;
            }
            float[] out = new float[count];
            long s = name.hashCode() * 0x9E3779B97F4A7C15L + 0x1234;
            for (int i = 0; i < count; i++) {
                s = s * 6364136223846793005L + 1442695040888963407L;
                out[i] = ((s >>> 40) & 0xFFFF) / 65535f * 0.2f - 0.1f;
            }
            values.put(name, out);
            return out;
        }

        // -------------------------------------------------------------- the references

        int referenceArgmax(int lastToken, float[] hidden) {
            switch (family) {
                case "eagle-cohere":
                    return cohere(lastToken, hidden);
                case "eagle-minicpm":
                    return miniCpm(lastToken, hidden);
                case "eagle-llama4":
                    return llama4(lastToken, hidden);
                case "eagle-mistral-large3":
                    return mistral(lastToken, hidden);
                default:
                    throw new IllegalStateException("no reference for " + family);
            }
        }

        private int cohere(int lastToken, float[] hidden) {
            float[] fcIn = concat(embedRow(lastToken), hidden);
            float[] g = add(matVec("fc.weight", 2 * N_EMBD, N_EMBD, fcIn), get("fc.bias", N_EMBD));
            float[] normed = lnW("blk.0.attn_norm.weight", N_EMBD, g);
            float[] attn = valueOnly("blk.0", normed, N_HEAD_KV);
            float[] ffn = ffn("blk.0", normed);
            float[] pre = add(add(g, attn), ffn);
            float[] out = lnW("output_norm.weight", N_EMBD, pre);
            float[] lg = scale(matVec("output.weight", N_EMBD, N_VOCAB, out), 0.5f);
            return argmax(lg);
        }

        private int miniCpm(int lastToken, float[] hidden) {
            float[] e = rmsW("enc.input_norm1.weight", N_EMBD, scale(embedRow(lastToken), 12f));
            float[] h = rmsW("enc.input_norm2.weight", N_EMBD, hidden);
            float[] g = matVec("fc.weight", 2 * N_EMBD, N_EMBD, concat(e, h));
            float[] normed = rmsW("blk.0.attn_norm.weight", N_EMBD, g);
            float[] x1 = add(g, scale(valueOnly("blk.0", normed, N_HEAD_KV), 0.7f));
            float[] fn = rmsW("blk.0.ffn_norm.weight", N_EMBD, x1);
            float[] x2 = add(x1, scale(ffn("blk.0", fn), 0.7f));
            return argmax(matVec("output.weight", N_EMBD, N_VOCAB, scale(x2, 0.5f)));
        }

        private int llama4(int lastToken, float[] hidden) {
            float[] g = matVec("fc.weight", 2 * N_EMBD, N_EMBD, concat(embedRow(lastToken), hidden));
            float[] normed = rmsW("blk.0.attn_norm.weight", N_EMBD, g);
            float[] q = matVec("blk.0.attn_q.weight", N_EMBD, N_HEAD * HEAD_DIM, normed);
            float[] k = matVec("blk.0.attn_k.weight", N_EMBD, N_HEAD_KV * HEAD_DIM, normed);
            // the weightless per-head norm of the RoPE layer, applied before the rotation (identity at position 0)
            q = perHeadRms(q, N_HEAD, EPS);
            k = perHeadRms(k, N_HEAD_KV, EPS);
            if (!Float.isFinite(q[0]) || !Float.isFinite(k[0])) {
                throw new AssertionError("the reference Q/K norm is not finite");
            }
            float[] attn = valueOnly("blk.0", normed, N_HEAD_KV);
            float[] x1 = add(g, attn);
            float[] fn = rmsW("blk.0.ffn_norm.weight", N_EMBD, x1);
            float[] x2 = add(x1, ffn("blk.0", fn));
            return argmax(matVec("output.weight", N_EMBD, N_VOCAB, rmsW("output_norm.weight", N_EMBD, x2)));
        }

        private int mistral(int lastToken, float[] hidden) {
            float[] g = matVec("fc.weight", 2 * N_EMBD, N_EMBD, concat(embedRow(lastToken), hidden));
            float[] normed = rmsW("blk.0.attn_norm.weight", N_EMBD, g);
            float[] qa = rmsW("blk.0.attn_q_a_norm.weight", Q_LORA,
                    matVec("blk.0.attn_q_a.weight", N_EMBD, Q_LORA, normed));
            float[] q = matVec("blk.0.attn_q_b.weight", Q_LORA, N_HEAD * MLA_HEAD_DIM, qa);
            float[] kva = matVec("blk.0.attn_kv_a_mqa.weight", N_EMBD, KV_LORA + MLA_ROT, normed);
            float[] kvc = rmsW("blk.0.attn_kv_a_norm.weight", KV_LORA,
                    java.util.Arrays.copyOf(kva, KV_LORA));
            float[] kpe = java.util.Arrays.copyOfRange(kva, KV_LORA, KV_LORA + MLA_ROT);
            float[] kb = get("blk.0.attn_k_b.weight", MLA_NOPE * KV_LORA * N_HEAD);
            float[] vb = get("blk.0.attn_v_b.weight", KV_LORA * MLA_HEAD_DIM * N_HEAD);
            float[] heads = new float[N_HEAD * MLA_HEAD_DIM];
            for (int h = 0; h < N_HEAD; h++) {
                float score = 0f;
                for (int d = 0; d < MLA_NOPE; d++) {
                    float k = 0f;
                    for (int l = 0; l < KV_LORA; l++) {
                        k += kb[(h * KV_LORA + l) * MLA_NOPE + d] * kvc[l];
                    }
                    score += q[h * MLA_HEAD_DIM + d] * k;
                }
                for (int d = 0; d < MLA_ROT; d++) {
                    score += q[h * MLA_HEAD_DIM + MLA_NOPE + d] * kpe[d];
                }
                if (!Float.isFinite(score)) {
                    throw new AssertionError("the reference MLA score is not finite");
                }
                for (int d = 0; d < MLA_HEAD_DIM; d++) {
                    float v = 0f;
                    for (int l = 0; l < KV_LORA; l++) {
                        v += vb[(h * MLA_HEAD_DIM + d) * KV_LORA + l] * kvc[l];
                    }
                    heads[h * MLA_HEAD_DIM + d] = v;
                }
            }
            float[] attn = matVec("blk.0.attn_output.weight", N_HEAD * MLA_HEAD_DIM, N_EMBD, heads);
            float[] x1 = add(g, attn);
            float[] fn = rmsW("blk.0.ffn_norm.weight", N_EMBD, x1);
            float[] x2 = add(x1, ffn("blk.0", fn));
            return argmax(matVec("output.weight", N_EMBD, N_VOCAB, rmsW("output_norm.weight", N_EMBD, x2)));
        }

        /** {@code W_o . v} for the one-key attention, where the softmax weight is 1. */
        private float[] valueOnly(String layer, float[] normed, int nHeadKv) {
            float[] v = matVec(layer + ".attn_v.weight", N_EMBD, nHeadKv * HEAD_DIM, normed);
            float[] heads = new float[N_HEAD * HEAD_DIM];
            int rep = N_HEAD / nHeadKv;
            for (int h = 0; h < N_HEAD; h++) {
                int src = (h / rep) * HEAD_DIM;
                System.arraycopy(v, src, heads, h * HEAD_DIM, HEAD_DIM);
            }
            return matVec(layer + ".attn_output.weight", N_HEAD * HEAD_DIM, N_EMBD, heads);
        }

        private float[] ffn(String layer, float[] in) {
            float[] gate = matVec(layer + ".ffn_gate.weight", N_EMBD, N_FF, in);
            float[] up = matVec(layer + ".ffn_up.weight", N_EMBD, N_FF, in);
            float[] act = new float[N_FF];
            for (int i = 0; i < N_FF; i++) {
                act[i] = silu(gate[i]) * up[i];
            }
            return matVec(layer + ".ffn_down.weight", N_FF, N_EMBD, act);
        }

        private float[] matVec(String name, int ne0, int ne1, float[] x) {
            float[] w = get(name, ne0 * ne1);
            float[] out = new float[ne1];
            for (int o = 0; o < ne1; o++) {
                float acc = 0f;
                for (int i = 0; i < ne0; i++) {
                    acc += w[o * ne0 + i] * x[i];
                }
                out[o] = acc;
            }
            return out;
        }

        private float[] rmsW(String name, int n, float[] x) {
            return rmsRaw(x, EPS, get(name, n));
        }

        private float[] lnW(String name, int n, float[] x) {
            float[] w = get(name, n);
            double mean = 0;
            for (float v : x) {
                mean += v;
            }
            mean /= x.length;
            double var = 0;
            for (float v : x) {
                var += (v - mean) * (v - mean);
            }
            var /= x.length;
            float r = (float) (1.0 / Math.sqrt(var + EPS));
            float[] out = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                out[i] = (float) ((x[i] - mean) * r) * w[i];
            }
            return out;
        }

        private float[] embedRow(int token) {
            float[] table = get("token_embd.weight", N_EMBD * N_VOCAB);
            float[] row = new float[N_EMBD];
            System.arraycopy(table, token * N_EMBD, row, 0, N_EMBD);
            return row;
        }
    }

    // ------------------------------------------------------------------ kernels

    private static float[] rms(float[] x, float eps) {
        return rmsRaw(x, eps, null);
    }

    private static float[] rmsRaw(float[] x, float eps, float[] weight) {
        double sum = 0;
        for (float v : x) {
            sum += (double) v * v;
        }
        float r = (float) (1.0 / Math.sqrt(sum / x.length + eps));
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (weight == null ? x[i] : x[i] * weight[i]) * r;
        }
        return out;
    }

    private static float[] perHeadRms(float[] x, int nHead, float eps) {
        if (nHead <= 0 || x.length % nHead != 0) {
            throw new IllegalStateException(x.length + " values do not split into " + nHead + " heads");
        }
        int width = x.length / nHead;
        float[] out = new float[x.length];
        for (int h = 0; h < nHead; h++) {
            double sum = 0;
            for (int d = 0; d < width; d++) {
                float v = x[h * width + d];
                sum += (double) v * v;
            }
            float r = (float) (1.0 / Math.sqrt(sum / width + eps));
            for (int d = 0; d < width; d++) {
                out[h * width + d] = x[h * width + d] * r;
            }
        }
        return out;
    }

    private static float[] scale(float[] x, float s) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] * s;
        }
        return out;
    }

    private static float[] add(float[] a, float[] b) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] + b[i];
        }
        return out;
    }

    private static float[] concat(float[] a, float[] b) {
        float[] out = java.util.Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    /** Records the hidden row of the target's final stream. */
    private static final class RecordingTap implements HiddenTap {
        private float[] row;

        @Override
        public boolean wants(int layer) {
            return layer == TAP;
        }

        @Override
        public void onHidden(int layer, float[] rows, int nRows, int width) {
            assertEquals("one prefill token", 1, nRows);
            row = java.util.Arrays.copyOf(rows, width);
        }

        float[] row() {
            assertTrue("the target's final stream was tapped", row != null);
            return row;
        }
    }
}
