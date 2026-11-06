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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The DeepSeek EAGLE-3 draft's forward against an independent reference of vLLM's
 * {@code deepseek_eagle3.DeepseekV2Eagle3DecoderLayer} and {@code combine_hidden_states}: the reference recomputes
 * the draft from its known synthetic weights — the single decoder layer's multi-head latent attention (absorbed
 * {@code attn_k_b} / {@code attn_v_b}, the low-rank query path and the latent KV path) and the dense MLP — and the
 * test compares its argmax with the first token the engine's drafter proposes from the same target hidden taps.
 */
public class Eagle3MlaDraftReferenceTest {

    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int HEAD_DIM = 32;
    private static final int ROT = 16;
    private static final int NOPE = HEAD_DIM - ROT;
    private static final int KV_LORA = 32;
    private static final int Q_LORA = 32;
    private static final int N_FF = 128;
    private static final int N_VOCAB = 320;
    private static final int INPUT = 2 * N_EMBD;
    private static final int N_ENC = 3 * N_EMBD;
    private static final int[] TAPS = {0, 1, 2};
    private static final int TARGET_LAYERS = 3;
    private static final int PROMPT_TOKEN = 7;
    private static final float EPS = 1e-6f;

    private static final Map<String, float[]> TENSORS = new HashMap<>();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static SyntheticModels.Config targetDims() {
        return SyntheticModels.Config.mistralLarge4().toBuilder().nLayer(TARGET_LAYERS).build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    private static EngineConfig config() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory()).build();
    }

    @Test
    public void theFirstProposalMatchesTheReferenceForward() throws IOException {
        Path targetPath = SyntheticModels.deepseek2(targetDims()).writeTempFile();
        Path draftPath = draftFile();
        int[] proposal;
        float[] tapsAtLast;
        int lastToken;
        try (EngineModel target = LLMModelFactory.open(targetPath.toString(), config())) {
            ForwardBackend fb = target.forwardBackend();
            RecordingTap rec = new RecordingTap();
            fb.setHiddenTap(rec);
            lastToken = argmax(fb.forwardRows(new int[]{PROMPT_TOKEN}, 0, new int[]{0})[0]);
            tapsAtLast = rec.columns();
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
        assertEquals("the MLA draft's argmax matches the reference", referenceArgmax(lastToken, tapsAtLast),
                proposal[0]);
    }

    // ------------------------------------------------------------------ reference

    /**
     * The reference's draft token: vLLM's encoder fusion, then the DeepSeek EAGLE-3 decoder layer at position 0
     * (where RoPE is the identity) and the output head.
     */
    private static int referenceArgmax(int lastToken, float[] taps) {
        float[] g = matVec("fc.weight", N_ENC, N_EMBD, taps);
        float[] e = rms("blk.0.attn_norm.weight", N_EMBD, embedRow(lastToken));
        float[] gn = rms("blk.0.attn_norm_2.weight", N_EMBD, g);
        float[] cat = concat(e, gn);

        float[] qa = rms("blk.0.attn_q_a_norm.weight", Q_LORA,
                matVec("blk.0.attn_q_a.weight", INPUT, Q_LORA, cat));
        float[] q = matVec("blk.0.attn_q_b.weight", Q_LORA, N_HEAD * HEAD_DIM, qa);
        float[] kva = matVec("blk.0.attn_kv_a_mqa.weight", INPUT, KV_LORA + ROT, cat);
        float[] kvc = rms("blk.0.attn_kv_a_norm.weight", KV_LORA, Arrays.copyOf(kva, KV_LORA));
        float[] kpe = Arrays.copyOfRange(kva, KV_LORA, KV_LORA + ROT);
        float[] kb = tensor("blk.0.attn_k_b.weight", NOPE * KV_LORA * N_HEAD);
        float[] vb = tensor("blk.0.attn_v_b.weight", KV_LORA * HEAD_DIM * N_HEAD);

        float[] heads = new float[N_HEAD * HEAD_DIM];
        for (int h = 0; h < N_HEAD; h++) {
            // one key in the cache: the softmax is 1, so the score only has to be finite
            float score = 0f;
            for (int d = 0; d < NOPE; d++) {
                float k = 0f;
                for (int l = 0; l < KV_LORA; l++) {
                    k += kb[(h * KV_LORA + l) * NOPE + d] * kvc[l];
                }
                score += q[h * HEAD_DIM + d] * k;
            }
            for (int d = 0; d < ROT; d++) {
                score += q[h * HEAD_DIM + NOPE + d] * kpe[d];
            }
            if (!Float.isFinite(score)) {
                throw new AssertionError("reference score is not finite");
            }
            for (int d = 0; d < HEAD_DIM; d++) {
                float v = 0f;
                for (int l = 0; l < KV_LORA; l++) {
                    v += vb[(h * HEAD_DIM + d) * KV_LORA + l] * kvc[l];
                }
                heads[h * HEAD_DIM + d] = v;
            }
        }
        float[] attn = matVec("blk.0.attn_output.weight", N_HEAD * HEAD_DIM, N_EMBD, heads);
        float[] ffnInp = add(attn, g);
        float[] ffnIn = rms("blk.0.ffn_norm.weight", N_EMBD, ffnInp);
        float[] gate = matVec("blk.0.ffn_gate.weight", N_EMBD, N_FF, ffnIn);
        float[] up = matVec("blk.0.ffn_up.weight", N_EMBD, N_FF, ffnIn);
        float[] act = new float[N_FF];
        for (int i = 0; i < N_FF; i++) {
            act[i] = silu(gate[i]) * up[i];
        }
        float[] down = matVec("blk.0.ffn_down.weight", N_FF, N_EMBD, act);
        float[] hn = rms("output_norm.weight", N_EMBD, add(down, ffnInp));
        return argmax(matVec("output.weight", N_EMBD, N_VOCAB, hn));
    }

    private static float[] embedRow(int token) {
        float[] table = tensor("token_embd.weight", N_EMBD * N_VOCAB);
        float[] row = new float[N_EMBD];
        System.arraycopy(table, token * N_EMBD, row, 0, N_EMBD);
        return row;
    }

    // ------------------------------------------------------------------ the draft file

    /** The draft GGUF: the synthetic dimensions and the deterministic weights the reference recomputes. */
    private Path draftFile() throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("eagle3");
        w.str("general.name", "synthetic-eagle3-mla");
        w.u32(w.archKey("vocab_size"), N_VOCAB);
        w.u32(w.archKey("context_length"), 64);
        w.u32(w.archKey("embedding_length"), N_EMBD);
        w.u32(w.archKey("block_count"), 1);
        w.u32(w.archKey("feed_forward_length"), N_FF);
        w.u32(w.archKey("attention.head_count"), N_HEAD);
        w.u32(w.archKey("attention.head_count_kv"), 1);
        w.u32(w.archKey("attention.key_length"), KV_LORA + ROT);
        w.u32(w.archKey("attention.value_length"), KV_LORA);
        w.u32(w.archKey("attention.key_length_mla"), HEAD_DIM);
        w.u32(w.archKey("attention.value_length_mla"), HEAD_DIM);
        w.u32(w.archKey("attention.q_lora_rank"), Q_LORA);
        w.u32(w.archKey("attention.kv_lora_rank"), KV_LORA);
        w.u32(w.archKey("rope.dimension_count"), ROT);
        w.f32(w.archKey("rope.freq_base"), 10000f);
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), EPS);
        w.i32Array(w.archKey("target_layers"), TAPS);
        w.u32(w.archKey("target_hidden_size"), N_EMBD);

        f32(w, "token_embd.weight", N_EMBD, N_VOCAB);
        f32(w, "fc.weight", N_ENC, N_EMBD);
        ones(w, "output_norm.weight", N_EMBD);
        f32(w, "output.weight", N_EMBD, N_VOCAB);
        ones(w, "blk.0.attn_norm.weight", N_EMBD);
        ones(w, "blk.0.attn_norm_2.weight", N_EMBD);
        f32(w, "blk.0.attn_q_a.weight", INPUT, Q_LORA);
        ones(w, "blk.0.attn_q_a_norm.weight", Q_LORA);
        f32(w, "blk.0.attn_q_b.weight", Q_LORA, N_HEAD * HEAD_DIM);
        f32(w, "blk.0.attn_kv_a_mqa.weight", INPUT, KV_LORA + ROT);
        ones(w, "blk.0.attn_kv_a_norm.weight", KV_LORA);
        f32(w, "blk.0.attn_k_b.weight", NOPE, KV_LORA, N_HEAD);
        f32(w, "blk.0.attn_v_b.weight", KV_LORA, HEAD_DIM, N_HEAD);
        f32(w, "blk.0.attn_output.weight", N_HEAD * HEAD_DIM, N_EMBD);
        ones(w, "blk.0.ffn_norm.weight", N_EMBD);
        f32(w, "blk.0.ffn_gate.weight", N_EMBD, N_FF);
        f32(w, "blk.0.ffn_up.weight", N_EMBD, N_FF);
        f32(w, "blk.0.ffn_down.weight", N_FF, N_EMBD);
        return w.writeTempFile();
    }

    private static void f32(GgufTestWriter w, String name, int... ne) {
        long count = 1;
        long[] shape = new long[ne.length];
        for (int i = 0; i < ne.length; i++) {
            count *= ne[i];
            shape[i] = ne[i];
        }
        w.tensorF32(name, shape, tensor(name, Math.toIntExact(count)));
    }

    private static void ones(GgufTestWriter w, String name, int n) {
        w.tensorFilled(name, GGMLType.F32, 1.0f, n);
    }

    // ------------------------------------------------------------------ deterministic weights

    /** The synthetic weights, also the reference's: an LCG keyed by the tensor name. */
    private static synchronized float[] tensor(String name, int count) {
        float[] cached = TENSORS.get(name);
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
        TENSORS.put(name, out);
        return out;
    }

    // ------------------------------------------------------------------ kernels

    private static float[] matVec(String name, int ne0, int ne1, float[] x) {
        float[] w = tensor(name, ne0 * ne1);
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

    private static float[] rms(String name, int n, float[] x) {
        tensor(name, n);                       // the norm weights are ones; the width must match
        if (x.length != n) {
            throw new IllegalStateException(name + " normalizes " + n + " values, got " + x.length);
        }
        double sum = 0;
        for (float v : x) {
            sum += (double) v * v;
        }
        float r = (float) (1.0 / Math.sqrt(sum / x.length + EPS));
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] * r;
        }
        return out;
    }

    private static float[] concat(float[] a, float[] b) {
        float[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static float[] add(float[] a, float[] b) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] + b[i];
        }
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

    /** Records the hidden rows of the three target taps for the one-token prefill. */
    private static final class RecordingTap implements HiddenTap {
        private final Map<Integer, float[]> rows = new HashMap<>();

        @Override
        public boolean wants(int layer) {
            return layer >= TAPS[0] && layer <= TAPS[TAPS.length - 1];
        }

        @Override
        public void onHidden(int layer, float[] row, int nRows, int width) {
            assertEquals("one prefill token", 1, nRows);
            rows.put(layer, Arrays.copyOf(row, width));
        }

        /** The three tap rows concatenated in tap order: the encoder's fusion input. */
        float[] columns() {
            float[] out = new float[N_EMBD * TAPS.length];
            for (int t = 0; t < TAPS.length; t++) {
                float[] row = rows.get(TAPS[t]);
                assertTrue("tap " + TAPS[t] + " delivered", row != null);
                System.arraycopy(row, 0, out, t * N_EMBD, N_EMBD);
            }
            return out;
        }
    }
}
