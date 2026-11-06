package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.util.Random;

/**
 * The synthetic lfm3a generator GGUF shared by the speech tests: the MaskGIT depthformer over twelve codebooks of a
 * four-code vocabulary and a two-layer liquid-codec decoder (one short convolution, one gated attention block),
 * laid out in the tensor names and ggml dimensions of tools/mtmd/clip-impl.h with the reference's hardcoded
 * geometry shrunk through the override keys. Two depthformer weight sets exist: seeded random weights for the
 * graph parity tests, and a hand-set constant path where the first frame samples code 0 in every codebook and the
 * second draws the end code, so the pipeline test runs one decoded frame and stops.
 */
public final class Lfm3aClipFixture {

    /** Backbone hidden width the depthformer projection takes. */
    static final int TALKER = Lfm3aBackboneFixture.EMB;
    /** Depthformer width. */
    static final int EMBD = 4;
    /** Depthformer blocks. */
    static final int LAYERS = 2;
    /** Depthformer heads and kv heads. */
    static final int HEADS = 2;
    static final int KV = 1;
    static final int DHEAD = EMBD / HEADS;
    /** Depthformer FFN width. */
    static final int FF = 8;
    /** Codebook pages and the vocabulary of one codebook. */
    static final int CODEBOOKS = 12;
    static final int VOCAB = 4;
    /** Codec decoder width and FFN width. */
    static final int DEC = 8;
    static final int DEC_FF = 8;
    /** Codec geometry overrides (the reference's 30/6/1280/320, shrunk for the tests). */
    static final int WINDOW = 6;
    static final int FRAME_STACK = 2;
    static final int N_FFT = 16;
    static final int HOP = 4;
    /** FSQ level rows per codebook. */
    static final int LVL = 2;
    /** Codec conv taps (the state slot holds taps - 1). */
    static final int TAPS = 3;
    /** Codec attention heads. */
    static final int DEC_HEADS = 2;
    static final int DEC_KV = 1;
    static final int DEC_DHEAD = 2;

    private Lfm3aClipFixture() {
    }

    /** The seeded-random writer: every weight drawn from a fixed-seed generator, all norms positive. */
    static GgufTestWriter randomWeights(long seed) {
        GgufTestWriter w = base();
        Random rnd = new Random(seed);
        depthformerRandom(w, rnd);
        codecRandom(w, rnd);
        return w;
    }

    /**
     * The constant-path writer: the projection maps the hidden state's second entry into the first row and adds
     * one, the mask embedding is zero and the embedding pages make code 0 the argmax while masked, so the first
     * frame samples all zeros; the feedback of code 0 negates the first row, which makes the end code
     * ({@code deterministicEos}) or code 1 ({@code deterministicNoEos}) the argmax of the second frame.
     */
    public static GgufTestWriter deterministicEos() {
        return constantPath(new float[]{-2f, 0f, 0f, 0f});
    }

    /**
     * The constant path with the end code never the first maximum: the row's first entry keeps the end code's
     * logit below every drawn code's unpenalized logit (the half logit against the full one), and the unit
     * repetition penalty of {@link #constantPath} keeps the drawn codes on top regardless, so the generation only
     * ends at the caller's frame bound, with every frame deterministic in the model's zero temperature.
     */
    static GgufTestWriter deterministicNoEos() {
        return constantPath(new float[]{0.5f, 0, 0, 0});
    }

    private static GgufTestWriter constantPath(float[] endRow) {
        GgufTestWriter w = base();
        depthformerConstant(w, endRow);
        codecRandom(w, new Random(7L));
        // temp 0 turns every draw into the argmax of the hand-set logits, and a unit repetition penalty keeps the
        // unpenalized end row from overtaking the penalized drawn codes (the default 1.2 would end every run)
        w.f32("clip.gen.audio.sampling.temp", 0f);
        w.f32("clip.gen.audio.sampling.penalty_repeat", 1f);
        return w;
    }

    private static GgufTestWriter base() {
        GgufTestWriter w = new GgufTestWriter().architecture("clip");
        w.bool("clip.has_gen_audio_encoder", true);
        w.str("clip.gen.audio.projector_type", Lfm3aDepthformer.PROJECTOR_TYPE);
        w.u32("clip.gen.audio.embedding_length", EMBD);
        w.u32("clip.gen.audio.projection_dim", TALKER);
        w.u32("clip.gen.audio.block_count", LAYERS);
        w.u32("clip.gen.audio.attention.head_count", HEADS);
        w.u32("clip.gen.audio.attention.head_count_kv", KV);
        w.u32("clip.gen.audio.feed_forward_length", FF);
        w.f32("clip.gen.audio.attention.layer_norm_epsilon", 1e-5f);
        w.u32("clip.gen.wav.window", WINDOW);
        w.u32("clip.gen.wav.frame_stack", FRAME_STACK);
        w.u32("clip.gen.wav.n_fft", N_FFT);
        w.u32("clip.gen.wav.hop_length", HOP);
        return w;
    }

    // ------------------------------------------------------------------ depthformer

    private static void depthformerRandom(GgufTestWriter w, Random rnd) {
        w.tensorF32("a.gen.code.proj_in.weight", new long[]{TALKER, EMBD * CODEBOOKS},
                random(rnd, TALKER * EMBD * CODEBOOKS, 0.5f));
        w.tensorF32("a.gen.code.proj_in.bias", new long[]{EMBD * CODEBOOKS},
                random(rnd, EMBD * CODEBOOKS, 0.3f));
        w.tensorF32("a.gen.code.embd.weight", new long[]{EMBD, VOCAB, CODEBOOKS},
                random(rnd, EMBD * VOCAB * CODEBOOKS, 0.8f));
        w.tensorF32("a.gen.code.out_embd.weight", new long[]{TALKER, (long) CODEBOOKS * VOCAB},
                random(rnd, TALKER * VOCAB * CODEBOOKS, 0.6f));
        w.tensorF32("a.gen.code.output_norm.weight", new long[]{EMBD}, random(rnd, EMBD, 0.2f, 0.9f));
        w.tensorF32("a.gen.code.head_norm.weight", new long[]{EMBD, CODEBOOKS},
                random(rnd, EMBD * CODEBOOKS, 0.2f, 1.2f));
        w.tensorF32("a.gen.code.mask_embd", new long[]{EMBD}, random(rnd, EMBD, 0.4f));
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.gen.code.blk." + il + ".";
            w.tensorF32(p + "ln1.weight", new long[]{EMBD}, random(rnd, EMBD, 0.2f, 1.1f));
            w.tensorF32(p + "attn_q.weight", new long[]{EMBD, HEADS * DHEAD}, random(rnd, EMBD * HEADS * DHEAD, 0.5f));
            w.tensorF32(p + "attn_q_norm.weight", new long[]{DHEAD}, random(rnd, DHEAD, 0.2f, 1.0f));
            w.tensorF32(p + "attn_k.weight", new long[]{EMBD, KV * DHEAD}, random(rnd, EMBD * KV * DHEAD, 0.5f));
            w.tensorF32(p + "attn_k_norm.weight", new long[]{DHEAD}, random(rnd, DHEAD, 0.2f, 1.0f));
            w.tensorF32(p + "attn_v.weight", new long[]{EMBD, KV * DHEAD}, random(rnd, EMBD * KV * DHEAD, 0.5f));
            w.tensorF32(p + "attn_out.weight", new long[]{HEADS * DHEAD, EMBD}, random(rnd, HEADS * DHEAD * EMBD, 0.5f));
            w.tensorF32(p + "attn_gate.weight", new long[]{EMBD, HEADS}, random(rnd, EMBD * HEADS, 0.5f));
            w.tensorF32(p + "ln2.weight", new long[]{EMBD}, random(rnd, EMBD, 0.2f, 1.1f));
            w.tensorF32(p + "ffn_gate.weight", new long[]{EMBD, FF}, random(rnd, EMBD * FF, 0.5f));
            w.tensorF32(p + "ffn_up.weight", new long[]{EMBD, FF}, random(rnd, EMBD * FF, 0.5f));
            w.tensorF32(p + "ffn_down.weight", new long[]{FF, EMBD}, random(rnd, FF * EMBD, 0.5f));
        }
    }

    private static void depthformerConstant(GgufTestWriter w, float[] endRow) {
        // the hidden state enters as (0, 5, 0, ...); every codebook's projection slice moves it to its row 0
        float[] projIn = new float[TALKER * EMBD * CODEBOOKS];
        float[] projInBias = new float[EMBD * CODEBOOKS];
        for (int p = 0; p < CODEBOOKS; p++) {
            projIn[(p * EMBD) * TALKER + 1] = 1f;
            projInBias[p * EMBD] = 1f;
        }
        w.tensorF32("a.gen.code.proj_in.weight", new long[]{TALKER, EMBD * CODEBOOKS}, projIn);
        w.tensorF32("a.gen.code.proj_in.bias", new long[]{EMBD * CODEBOOKS}, projInBias);
        float[] embd = new float[EMBD * VOCAB * CODEBOOKS];
        for (int p = 0; p < CODEBOOKS; p++) {
            int off = p * EMBD * VOCAB;
            System.arraycopy(new float[]{1, 5, 0, 0}, 0, embd, off, EMBD);              // code 0
            System.arraycopy(new float[]{-1, -5, 0, 0}, 0, embd, off + EMBD, EMBD);     // code 1
            System.arraycopy(new float[]{-1, -5, 0, 0}, 0, embd, off + 2 * EMBD, EMBD); // code 2
            System.arraycopy(endRow, 0, embd, off + 3 * EMBD, EMBD);                    // the end code
        }
        w.tensorF32("a.gen.code.embd.weight", new long[]{EMBD, VOCAB, CODEBOOKS}, embd);
        // every code's row negates the first state axis, so the feedback of the drawn code 0 does too
        float[] outRow = new float[TALKER * VOCAB];
        outRow[1] = -1f;
        float[] outEmbd = new float[TALKER * VOCAB * CODEBOOKS];
        for (int p = 0; p < CODEBOOKS; p++) {
            System.arraycopy(outRow, 0, outEmbd, p * TALKER * VOCAB, outRow.length);
        }
        w.tensorF32("a.gen.code.out_embd.weight", new long[]{TALKER, (long) CODEBOOKS * VOCAB}, outEmbd);
        w.tensorF32("a.gen.code.output_norm.weight", new long[]{EMBD}, new float[]{1, 1, 1, 1});
        float[] headNorm = new float[EMBD * CODEBOOKS];
        java.util.Arrays.fill(headNorm, 1f);
        w.tensorF32("a.gen.code.head_norm.weight", new long[]{EMBD, CODEBOOKS}, headNorm);
        w.tensorF32("a.gen.code.mask_embd", new long[]{EMBD}, new float[]{0, 0, 0, 0});
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.gen.code.blk." + il + ".";
            w.tensorFilled(p + "ln1.weight", GGMLType.F32, 1f, EMBD);
            w.tensorFilled(p + "attn_q.weight", GGMLType.F32, 0f, EMBD, HEADS * DHEAD);
            w.tensorFilled(p + "attn_q_norm.weight", GGMLType.F32, 1f, DHEAD);
            w.tensorFilled(p + "attn_k.weight", GGMLType.F32, 0f, EMBD, KV * DHEAD);
            w.tensorFilled(p + "attn_k_norm.weight", GGMLType.F32, 1f, DHEAD);
            w.tensorFilled(p + "attn_v.weight", GGMLType.F32, 0f, EMBD, KV * DHEAD);
            w.tensorFilled(p + "attn_out.weight", GGMLType.F32, 0f, HEADS * DHEAD, EMBD);
            w.tensorFilled(p + "attn_gate.weight", GGMLType.F32, 0f, EMBD, HEADS);
            w.tensorFilled(p + "ln2.weight", GGMLType.F32, 1f, EMBD);
            w.tensorFilled(p + "ffn_gate.weight", GGMLType.F32, 0f, EMBD, FF);
            w.tensorFilled(p + "ffn_up.weight", GGMLType.F32, 0f, EMBD, FF);
            w.tensorFilled(p + "ffn_down.weight", GGMLType.F32, 0f, FF, EMBD);
        }
    }

    // ------------------------------------------------------------------ codec

    private static void codecRandom(GgufTestWriter w, Random rnd) {
        // the FSQ table indexes the codebook only: the end code (the last vocabulary row) has no waveform, so
        // the table is one row short of the depthformer's vocabulary (lfm3.py:473-478)
        float[] table = new float[LVL * (VOCAB - 1)];
        for (int c = 0; c < VOCAB - 1; c++) {
            table[c * LVL] = (c % 2) - 0.5f;      // the low bit of the code
            table[c * LVL + 1] = (c / 2) - 0.5f;  // the high bit
        }
        w.tensorF32("a.gen.wav.quant.first.codebook.weight", new long[]{LVL, VOCAB - 1}, table);
        w.tensorF32("a.gen.wav.quant.first.out_proj.weight", new long[]{LVL * CODEBOOKS, DEC},
                random(rnd, LVL * CODEBOOKS * DEC, 0.3f));
        w.tensorF32("a.gen.wav.quant.first.out_proj.bias", new long[]{DEC}, random(rnd, DEC, 0.2f));
        w.tensorF32("a.gen.wav.tfm.in_proj.weight", new long[]{DEC, DEC}, random(rnd, DEC * DEC, 0.3f));
        w.tensorF32("a.gen.wav.tfm.in_proj.bias", new long[]{DEC}, random(rnd, DEC, 0.2f));
        w.tensorF32("a.gen.wav.tfm.out_proj.weight", new long[]{DEC, N_FFT + 2}, random(rnd, DEC * (N_FFT + 2), 0.2f));
        w.tensorF32("a.gen.wav.tfm.out_proj.bias", new long[]{N_FFT + 2}, random(rnd, N_FFT + 2, 0.1f));
        w.tensorF32("a.gen.wav.tfm.output_norm.weight", new long[]{DEC}, random(rnd, DEC, 0.2f, 1.0f));
        for (int il = 0; il < 2; il++) {
            String p = "a.gen.wav.tfm.blk." + il + ".";
            w.tensorF32(p + "ln1.weight", new long[]{DEC}, random(rnd, DEC, 0.2f, 1.1f));
            w.tensorF32(p + "ln2.weight", new long[]{DEC}, random(rnd, DEC, 0.2f, 1.1f));
            w.tensorF32(p + "attn_post_norm.weight", new long[]{DEC}, random(rnd, DEC, 0.2f, 1.0f));
            w.tensorF32(p + "ffn_post_norm.weight", new long[]{DEC}, random(rnd, DEC, 0.2f, 1.0f));
            w.tensorF32(p + "ffn_gate.weight", new long[]{DEC, DEC_FF}, random(rnd, DEC * DEC_FF, 0.4f));
            w.tensorF32(p + "ffn_up.weight", new long[]{DEC, DEC_FF}, random(rnd, DEC * DEC_FF, 0.4f));
            w.tensorF32(p + "ffn_down.weight", new long[]{DEC_FF, DEC}, random(rnd, DEC_FF * DEC, 0.4f));
            if (il == 0) {
                w.tensorF32(p + "conv_pw1.weight", new long[]{DEC, 3 * DEC}, random(rnd, 3 * DEC * DEC, 0.4f));
                w.tensorF32(p + "conv_dw.weight", new long[]{TAPS, DEC}, random(rnd, TAPS * DEC, 0.4f));
                w.tensorF32(p + "conv_pw2.weight", new long[]{DEC, DEC}, random(rnd, DEC * DEC, 0.4f));
            } else {
                w.tensorF32(p + "attn_q.weight", new long[]{DEC, DEC_HEADS * DEC_DHEAD},
                        random(rnd, DEC * DEC_HEADS * DEC_DHEAD, 0.4f));
                w.tensorF32(p + "attn_q_norm.weight", new long[]{DEC_DHEAD}, random(rnd, DEC_DHEAD, 0.2f, 1.0f));
                w.tensorF32(p + "attn_k.weight", new long[]{DEC, DEC_KV * DEC_DHEAD},
                        random(rnd, DEC * DEC_KV * DEC_DHEAD, 0.4f));
                w.tensorF32(p + "attn_k_norm.weight", new long[]{DEC_DHEAD}, random(rnd, DEC_DHEAD, 0.2f, 1.0f));
                w.tensorF32(p + "attn_v.weight", new long[]{DEC, DEC_KV * DEC_DHEAD},
                        random(rnd, DEC * DEC_KV * DEC_DHEAD, 0.4f));
                w.tensorF32(p + "attn_out.weight", new long[]{DEC_HEADS * DEC_DHEAD, DEC},
                        random(rnd, DEC_HEADS * DEC_DHEAD * DEC, 0.4f));
                w.tensorF32(p + "attn_gate.weight", new long[]{DEC, DEC_HEADS}, random(rnd, DEC * DEC_HEADS, 0.4f));
            }
        }
    }

    private static float[] random(Random rnd, int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rnd.nextFloat() * 2f - 1f) * scale;
        }
        return v;
    }

    private static float[] random(Random rnd, int n, float lo, float hi) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = lo + rnd.nextFloat() * (hi - lo);
        }
        return v;
    }
}
