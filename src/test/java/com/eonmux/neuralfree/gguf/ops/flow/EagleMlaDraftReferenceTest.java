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

/**
 * The DeepSeek-MTP-shaped EAGLE draft's forward against an independent reference of vLLM's {@code deepseek_eagle}
 * {@code DeepseekV2Model}: the reference recomputes the draft from its known synthetic weights — the fusion
 * {@code fc . [enorm(embed) | hnorm(target hidden)]}, the single decoder layer's multi-head latent attention
 * (absorbed {@code attn_k_b} / {@code attn_v_b}, the low-rank query path and the latent KV path), the dense MLP and
 * the final norm — and the test compares its argmax with the first token the engine's drafter proposes from the
 * same target hidden row.
 *
 * <p>Two decoder rows are cached when the draft starts from a one-token prefill: the prefill's own row at position
 * 0, which pairs the prompt token with the zero hidden row a first call has no predecessor for, and the draft
 * step's row at position 1, which pairs the target's last token with its hidden row. The query of the step attends
 * over both keys causally, each carrying the rotary rotation of its own position, so the reference computes the
 * softmax over the two scores rather than assuming a single key.</p>
 */
public class EagleMlaDraftReferenceTest {

    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int HEAD_DIM = 32;
    private static final int ROT = 16;
    private static final int NOPE = HEAD_DIM - ROT;
    private static final int KV_LORA = 32;
    private static final int Q_LORA = 32;
    private static final int N_FF = 128;
    private static final int N_VOCAB = 320;
    private static final int TAP = 0;
    private static final int PROMPT_TOKEN = 7;
    private static final float EPS = 1e-6f;

    private static final Map<String, float[]> TENSORS = new HashMap<>();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static SyntheticModels.Config targetDims() {
        return SyntheticModels.Config.mistralLarge4().toBuilder().nLayer(1).build();
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
        float[] tapAtLast;
        int lastToken;
        try (EngineModel target = LLMModelFactory.open(targetPath.toString(), config())) {
            ForwardBackend fb = target.forwardBackend();
            RecordingTap rec = new RecordingTap();
            fb.setHiddenTap(rec);
            lastToken = argmax(fb.forwardRows(new int[]{PROMPT_TOKEN}, 0, new int[]{0})[0]);
            tapAtLast = rec.row();
            fb.setHiddenTap(com.eonmux.neuralfree.gguf.engine.HiddenTap.NONE);
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
        assertEquals("the MLA draft's argmax matches the reference", referenceArgmax(lastToken, tapAtLast),
                proposal[0]);
    }

    // ------------------------------------------------------------------ reference

    /**
     * The reference's draft token: vLLM's encoder fusion over the normed embedding and target hidden row, then the
     * DeepSeek decoder layer over the two cached rows (the prefill row at position 0 and the step at position 1) and
     * the output head.
     */
    private static int referenceArgmax(int lastToken, float[] hidden) {
        float[] g0 = fusion(PROMPT_TOKEN, new float[N_EMBD]);
        float[] g = fusion(lastToken, hidden);
        float[] attn = attention(g, g0);
        float[] ffnInp = add(attn, g);
        float[] ffnIn = rms("blk.0.ffn_norm.weight", ffnInp);
        float[] gate = matVec("blk.0.ffn_gate.weight", N_EMBD, N_FF, ffnIn);
        float[] up = matVec("blk.0.ffn_up.weight", N_EMBD, N_FF, ffnIn);
        float[] act = new float[N_FF];
        for (int i = 0; i < N_FF; i++) {
            act[i] = silu(gate[i]) * up[i];
        }
        float[] down = matVec("blk.0.ffn_down.weight", N_FF, N_EMBD, act);
        float[] out = rms("output_norm.weight", add(down, ffnInp));
        return argmax(matVec("output.weight", N_EMBD, N_VOCAB, out));
    }

    /** {@code fc . [enorm(embed(token)) | hnorm(hidden)]}. */
    private static float[] fusion(int token, float[] hidden) {
        float[] e = rms("enc.enorm.weight", embedRow(token));
        float[] h = rms("enc.hnorm.weight", hidden);
        return matVec("fc.weight", 2 * N_EMBD, N_EMBD, concat(e, h));
    }

    /** The layer's multi-head latent attention with the query at position 1 over the keys at positions 0 ({@code
     * before}) and 1 ({@code g}). */
    private static float[] attention(float[] g, float[] before) {
        float[] q = query(g);
        float[][] first = latent(before);
        float[][] second = latent(g);
        float[][] kvc = {first[0], second[0]};
        float[][] kpe = {rope(first[1], 0), rope(second[1], 1)};
        float[] kb = tensor("blk.0.attn_k_b.weight", NOPE * KV_LORA * N_HEAD);
        float[] vb = tensor("blk.0.attn_v_b.weight", KV_LORA * HEAD_DIM * N_HEAD);
        float kqScale = (float) (1.0 / Math.sqrt(HEAD_DIM));

        float[] heads = new float[N_HEAD * HEAD_DIM];
        for (int hd = 0; hd < N_HEAD; hd++) {
            float[] qRope = rope(Arrays.copyOfRange(q, hd * HEAD_DIM + NOPE, hd * HEAD_DIM + HEAD_DIM), 1);
            float[] score = new float[2];
            float max = -Float.MAX_VALUE;
            for (int j = 0; j < 2; j++) {
                float dot = 0f;
                for (int d = 0; d < NOPE; d++) {
                    float k = 0f;
                    for (int l = 0; l < KV_LORA; l++) {
                        k += kb[(hd * KV_LORA + l) * NOPE + d] * kvc[j][l];
                    }
                    dot += q[hd * HEAD_DIM + d] * k;
                }
                for (int d = 0; d < ROT; d++) {
                    dot += qRope[d] * kpe[j][d];
                }
                score[j] = dot * kqScale;
                max = Math.max(max, score[j]);
            }
            float sum = 0f;
            for (int j = 0; j < 2; j++) {
                score[j] = (float) Math.exp(score[j] - max);
                sum += score[j];
            }
            for (int d = 0; d < HEAD_DIM; d++) {
                float v = 0f;
                for (int j = 0; j < 2; j++) {
                    float value = 0f;
                    for (int l = 0; l < KV_LORA; l++) {
                        value += vb[(hd * HEAD_DIM + d) * KV_LORA + l] * kvc[j][l];
                    }
                    v += score[j] / sum * value;
                }
                heads[hd * HEAD_DIM + d] = v;
            }
        }
        return matVec("blk.0.attn_output.weight", N_HEAD * HEAD_DIM, N_EMBD, heads);
    }

    /** The step's query: {@code attn_norm -> q_a -> q_a_norm -> q_b}. */
    private static float[] query(float[] g) {
        float[] a = rms("blk.0.attn_norm.weight", g);
        float[] qa = rms("blk.0.attn_q_a_norm.weight", matVec("blk.0.attn_q_a.weight", N_EMBD, Q_LORA, a));
        return matVec("blk.0.attn_q_b.weight", Q_LORA, N_HEAD * HEAD_DIM, qa);
    }

    /** A row's compressed KV latent and its rotary key: {@code attn_norm -> kv_a_mqa -> kv_a_norm | key}. */
    private static float[][] latent(float[] g) {
        float[] a = rms("blk.0.attn_norm.weight", g);
        float[] kva = matVec("blk.0.attn_kv_a_mqa.weight", N_EMBD, KV_LORA + ROT, a);
        return new float[][]{
                rms("blk.0.attn_kv_a_norm.weight", Arrays.copyOf(kva, KV_LORA)),
                Arrays.copyOfRange(kva, KV_LORA, KV_LORA + ROT)};
    }

    /** The NORM-style rotation of a rotary key at {@code position}: adjacent pairs, {@code base^(-2i/ROT)}. */
    private static float[] rope(float[] x, int position) {
        float[] out = x.clone();
        float thetaScale = (float) Math.pow(10_000f, -2f / ROT);
        for (int i = 0; i < ROT / 2; i++) {
            float theta = position * (float) Math.pow(thetaScale, i);
            float c = (float) Math.cos(theta);
            float s = (float) Math.sin(theta);
            float x0 = x[2 * i];
            float x1 = x[2 * i + 1];
            out[2 * i] = x0 * c - x1 * s;
            out[2 * i + 1] = x0 * s + x1 * c;
        }
        return out;
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
        GgufTestWriter w = new GgufTestWriter().architecture("eagle");
        w.str("general.name", "synthetic-eagle-mla");
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
        w.i32Array(w.archKey("target_layers"), TAP);
        w.u32(w.archKey("target_hidden_size"), N_EMBD);

        f32(w, "token_embd.weight", N_EMBD, N_VOCAB);
        f32(w, "fc.weight", 2 * N_EMBD, N_EMBD);
        f32(w, "enc.enorm.weight", N_EMBD);
        f32(w, "enc.hnorm.weight", N_EMBD);
        f32(w, "output_norm.weight", N_EMBD);
        f32(w, "output.weight", N_EMBD, N_VOCAB);
        f32(w, "blk.0.attn_norm.weight", N_EMBD);
        f32(w, "blk.0.attn_q_a.weight", N_EMBD, Q_LORA);
        f32(w, "blk.0.attn_q_a_norm.weight", Q_LORA);
        f32(w, "blk.0.attn_q_b.weight", Q_LORA, N_HEAD * HEAD_DIM);
        f32(w, "blk.0.attn_kv_a_mqa.weight", N_EMBD, KV_LORA + ROT);
        f32(w, "blk.0.attn_kv_a_norm.weight", KV_LORA);
        f32(w, "blk.0.attn_k_b.weight", NOPE, KV_LORA, N_HEAD);
        f32(w, "blk.0.attn_v_b.weight", KV_LORA, HEAD_DIM, N_HEAD);
        f32(w, "blk.0.attn_output.weight", N_HEAD * HEAD_DIM, N_EMBD);
        f32(w, "blk.0.ffn_norm.weight", N_EMBD);
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

    // ------------------------------------------------------------------ deterministic weights

    /** The synthetic weights, also the reference's: an LCG keyed by the tensor name (norm weights around 1). */
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
        boolean norm = name.endsWith("norm.weight");
        for (int i = 0; i < count; i++) {
            s = s * 6364136223846793005L + 1442695040888963407L;
            float v = ((s >>> 40) & 0xFFFF) / 65535f * 0.2f - 0.1f;
            out[i] = norm ? 1f + v : v;
        }
        TENSORS.put(name, out);
        return out;
    }

    // ------------------------------------------------------------------ kernels

    /** {@code out[o] = sum_i w[o * ne0 + i] * x[i]}. */
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

    /** The weighted RMS norm the draft layer applies: {@code x[i] * weight[i] / rms(x)}. */
    private static float[] rms(String name, float[] x) {
        float[] w = tensor(name, x.length);
        double sum = 0;
        for (float v : x) {
            sum += (double) v * v;
        }
        float r = (float) (1.0 / Math.sqrt(sum / x.length + EPS));
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] * r * w[i];
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

    /** Records the hidden row of the one requested target tap for the one-token prefill. */
    private static final class RecordingTap implements com.eonmux.neuralfree.gguf.engine.HiddenTap {
        private float[] row;

        @Override
        public boolean wants(int layer) {
            return layer == TAP;
        }

        @Override
        public void onHidden(int layer, float[] rows, int nRows, int width) {
            assertEquals("one prefill token", 1, nRows);
            row = Arrays.copyOf(rows, width);
        }

        float[] row() {
            if (row == null) {
                throw new AssertionError("tap " + TAP + " delivered no row");
            }
            return row;
        }
    }
}
