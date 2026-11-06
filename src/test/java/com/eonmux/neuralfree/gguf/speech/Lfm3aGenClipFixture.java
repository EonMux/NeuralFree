package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.file.Path;

/**
 * A tiny synthetic lfm3a generator file whose graph math is hand-computable. The depthformer's projection, every
 * block weight and the mask embedding are zero (norms one), so each block is the identity and the codebook logits
 * reduce to {@code A[code] * embd[codebook][code][0]}, with {@code A} strictly decreasing except the end code. The
 * mask embedding decides the fresh pass: zero embeds read {@code A[0]} (every codebook draws code 0), a negative
 * one flips the argmax onto the end code (every codebook draws it). The feedback embedding maps codebook {@code cb}
 * to state axis {@code cb} scaled by {@code code + 1}, so the feedback of any frame is known by inspection. The
 * codec's quantizer and transformer weights are zero, so the decoded positions are constant and the spectrum is the
 * {@code out_proj} bias alone: magnitudes of -30 give samples below {@code 1e-11}.
 *
 * <p>Geometry: state width = the paired backbone's embedding width, depthformer 8 wide with 2 heads / 1 KV head of
 * dim 4, one block, 12 codebook pages of vocabulary 6; codec 2 FSQ levels of 5 codes, width 8, one convolution
 * block then one gated attention block with 2 heads of dim 4.
 */
public final class Lfm3aGenClipFixture {

    /** Depthformer width. */
    public static final int N_EMBD = 8;
    /** Depthformer heads and KV heads; head dim 4. */
    public static final int HEADS = 2;
    public static final int KV = 1;
    /** Depthformer FFN width. */
    public static final int FF = 8;
    /** Codebook vocabulary: five decodable codes plus the end code. */
    public static final int VOCAB = 6;
    /** Decodable codes; the end code {@code VOCAB - 1} has no waveform. */
    public static final int FSQ_VOCAB = 5;
    /** Codec FSQ levels and transformer width. */
    public static final int FSQ_LEVELS = 2;
    public static final int QUANT_OUT = 8;
    public static final int DEC = 8;
    /** Codec attention: 2 heads of dim 4, 1 KV head. */
    public static final int DEC_HEADS = 2;
    public static final int DEC_KV = 1;
    public static final int DEC_DHEAD = 4;
    /** Convolution kernel of the codec's conv block. */
    public static final int D_CONV = 3;
    /** The log-magnitude the constant spectrum carries. */
    public static final float BIAS_MAG = -30f;

    /** The codebook-embedding column values; the end code's is negative. */
    static final float[] A = {5f, 4f, 3f, 2f, 1f, -1f};

    private final int stateWidth;
    private final boolean endCodeDraws;

    /**
     * @param stateWidth   the backbone hidden width {@code proj_in} takes
     * @param endCodeDraws true makes every masked codebook draw the end code, false code 0
     */
    public Lfm3aGenClipFixture(int stateWidth, boolean endCodeDraws) {
        if (stateWidth <= 0) {
            throw new IllegalArgumentException("state width must be positive, got " + stateWidth);
        }
        this.stateWidth = stateWidth;
        this.endCodeDraws = endCodeDraws;
    }

    /** The backbone hidden width this fixture's projection takes. */
    public int stateWidth() {
        return stateWidth;
    }

    /** The fixture as a clip generator GGUF: the lfm3a metadata and every tensor. */
    public GgufTestWriter writer() {
        GgufTestWriter w = new GgufTestWriter().architecture("clip");
        w.bool("clip.has_gen_audio_encoder", true);
        w.str("clip.gen.audio.projector_type", Lfm3aDepthformer.PROJECTOR_TYPE);
        w.u32("clip.gen.audio.embedding_length", N_EMBD);
        w.u32("clip.gen.audio.block_count", 1);
        w.u32("clip.gen.audio.attention.head_count", HEADS);
        w.u32("clip.gen.audio.attention.head_count_kv", KV);
        w.u32("clip.gen.audio.feed_forward_length", FF);
        w.u32("clip.gen.audio.projection_dim", stateWidth);
        w.f32("clip.gen.audio.attention.layer_norm_epsilon", 1e-5f);
        w.f32("clip.gen.audio.sampling.temp", 0f);
        w.u32("clip.gen.audio.sampling.top_k", 0);
        depthformer(w);
        codec(w);
        return w;
    }

    public Path write() throws IOException {
        return writer().writeTempFile();
    }

    /**
     * The depthformer: zero projection over unit norms, so {@code cur} is the token and mask rows alone and the
     * head reads them through the identity blocks.
     */
    private void depthformer(GgufTestWriter w) {
        put(w, "a.gen.code.proj_in.weight", new float[stateWidth * N_EMBD * Lfm3aDepthformer.N_CODEBOOKS],
                stateWidth, N_EMBD * Lfm3aDepthformer.N_CODEBOOKS);
        put(w, "a.gen.code.proj_in.bias",
                new float[N_EMBD * Lfm3aDepthformer.N_CODEBOOKS], N_EMBD * Lfm3aDepthformer.N_CODEBOOKS);
        float[] embd = new float[N_EMBD * VOCAB * Lfm3aDepthformer.N_CODEBOOKS];
        for (int cb = 0; cb < Lfm3aDepthformer.N_CODEBOOKS; cb++) {
            for (int v = 0; v < VOCAB; v++) {
                embd[0 + N_EMBD * (v + VOCAB * cb)] = A[v];
            }
        }
        put(w, "a.gen.code.embd.weight", embd, N_EMBD, VOCAB, Lfm3aDepthformer.N_CODEBOOKS);
        float[] outEmbd = new float[stateWidth * VOCAB * Lfm3aDepthformer.N_CODEBOOKS];
        for (int cb = 0; cb < Lfm3aDepthformer.N_CODEBOOKS; cb++) {
            for (int v = 0; v < VOCAB; v++) {
                outEmbd[(cb % stateWidth) + stateWidth * (v + VOCAB * cb)] = v + 1;
            }
        }
        put(w, "a.gen.code.out_embd.weight", outEmbd, stateWidth, (long) VOCAB * Lfm3aDepthformer.N_CODEBOOKS);
        float[] mask = new float[N_EMBD];
        if (endCodeDraws) {
            mask[0] = -3f;
        }
        put(w, "a.gen.code.mask_embd", mask, N_EMBD, 1);
        put(w, "a.gen.code.output_norm.weight", ones(N_EMBD), N_EMBD);
        put(w, "a.gen.code.head_norm.weight", ones(N_EMBD * Lfm3aDepthformer.N_CODEBOOKS),
                N_EMBD, Lfm3aDepthformer.N_CODEBOOKS);

        int dHead = N_EMBD / HEADS;
        String b = "a.gen.code.blk.0.";
        put(w, b + "ln1.weight", ones(N_EMBD), N_EMBD);
        put(w, b + "attn_q.weight", new float[N_EMBD * HEADS * dHead], N_EMBD, HEADS * dHead);
        put(w, b + "attn_q_norm.weight", ones(dHead), dHead);
        put(w, b + "attn_k.weight", new float[N_EMBD * KV * dHead], N_EMBD, KV * dHead);
        put(w, b + "attn_k_norm.weight", ones(dHead), dHead);
        put(w, b + "attn_v.weight", new float[N_EMBD * KV * dHead], N_EMBD, KV * dHead);
        put(w, b + "attn_out.weight", new float[HEADS * dHead * N_EMBD], HEADS * dHead, N_EMBD);
        put(w, b + "attn_gate.weight", new float[N_EMBD * HEADS], N_EMBD, HEADS);
        put(w, b + "ln2.weight", ones(N_EMBD), N_EMBD);
        put(w, b + "ffn_gate.weight", new float[N_EMBD * FF], N_EMBD, FF);
        put(w, b + "ffn_up.weight", new float[N_EMBD * FF], N_EMBD, FF);
        put(w, b + "ffn_down.weight", new float[FF * N_EMBD], FF, N_EMBD);
    }

    /**
     * The codec: zero quantizer and transformer over unit norms, so every decoded position is the constant zero
     * vector and the spectrum is {@code out_proj.bias} alone: magnitudes of {@code -30} and zero phase.
     */
    private void codec(GgufTestWriter w) {
        int nBins = Lfm3aCodec.DEFAULT_N_FFT / 2 + 1;
        float[] fsq = new float[FSQ_LEVELS * FSQ_VOCAB];
        for (int lvl = 0; lvl < FSQ_LEVELS; lvl++) {
            for (int v = 0; v < FSQ_VOCAB; v++) {
                fsq[lvl + FSQ_LEVELS * v] = v + 1 + 10 * lvl;
            }
        }
        put(w, "a.gen.wav.quant.first.codebook.weight", fsq, FSQ_LEVELS, FSQ_VOCAB);
        put(w, "a.gen.wav.quant.first.out_proj.weight", new float[QUANT_OUT * FSQ_LEVELS
                * Lfm3aDepthformer.N_CODEBOOKS], FSQ_LEVELS * Lfm3aDepthformer.N_CODEBOOKS, QUANT_OUT);
        put(w, "a.gen.wav.quant.first.out_proj.bias", new float[QUANT_OUT], QUANT_OUT);
        put(w, "a.gen.wav.tfm.in_proj.weight", new float[DEC * QUANT_OUT], DEC, QUANT_OUT);
        put(w, "a.gen.wav.tfm.in_proj.bias", new float[DEC], DEC);
        float[] tfmOutBias = new float[2 * nBins];
        for (int i = 0; i < nBins; i++) {
            tfmOutBias[i] = BIAS_MAG;
        }
        put(w, "a.gen.wav.tfm.out_proj.weight", new float[2 * nBins * DEC], DEC, 2 * nBins);
        put(w, "a.gen.wav.tfm.out_proj.bias", tfmOutBias, 2 * nBins);
        put(w, "a.gen.wav.tfm.output_norm.weight", ones(DEC), DEC);

        String c = "a.gen.wav.tfm.blk.0.";
        put(w, c + "ln1.weight", ones(DEC), DEC);
        put(w, c + "attn_post_norm.weight", ones(DEC), DEC);
        put(w, c + "ln2.weight", ones(DEC), DEC);
        put(w, c + "ffn_post_norm.weight", ones(DEC), DEC);
        put(w, c + "ffn_gate.weight", new float[DEC * FF], DEC, FF);
        put(w, c + "ffn_up.weight", new float[DEC * FF], DEC, FF);
        put(w, c + "ffn_down.weight", new float[FF * DEC], FF, DEC);
        put(w, c + "conv_pw1.weight", new float[D_CONV * DEC * DEC], DEC, D_CONV * DEC);
        put(w, c + "conv_dw.weight", new float[D_CONV * DEC], D_CONV, DEC);
        put(w, c + "conv_pw2.weight", new float[DEC * DEC], DEC, DEC);

        String a = "a.gen.wav.tfm.blk.1.";
        put(w, a + "ln1.weight", ones(DEC), DEC);
        put(w, a + "attn_post_norm.weight", ones(DEC), DEC);
        put(w, a + "ln2.weight", ones(DEC), DEC);
        put(w, a + "ffn_post_norm.weight", ones(DEC), DEC);
        put(w, a + "ffn_gate.weight", new float[DEC * FF], DEC, FF);
        put(w, a + "ffn_up.weight", new float[DEC * FF], DEC, FF);
        put(w, a + "ffn_down.weight", new float[FF * DEC], FF, DEC);
        put(w, a + "attn_q.weight", new float[DEC * DEC_HEADS * DEC_DHEAD], DEC, DEC_HEADS * DEC_DHEAD);
        put(w, a + "attn_q_norm.weight", ones(DEC_DHEAD), DEC_DHEAD);
        put(w, a + "attn_k.weight", new float[DEC * DEC_KV * DEC_DHEAD], DEC, DEC_KV * DEC_DHEAD);
        put(w, a + "attn_k_norm.weight", ones(DEC_DHEAD), DEC_DHEAD);
        put(w, a + "attn_v.weight", new float[DEC * DEC_KV * DEC_DHEAD], DEC, DEC_KV * DEC_DHEAD);
        put(w, a + "attn_out.weight", new float[DEC_HEADS * DEC_DHEAD * DEC], DEC_HEADS * DEC_DHEAD, DEC);
        put(w, a + "attn_gate.weight", new float[DEC * DEC_HEADS], DEC, DEC_HEADS);
    }

    private static float[] ones(int n) {
        float[] v = new float[n];
        java.util.Arrays.fill(v, 1f);
        return v;
    }

    private static void put(GgufTestWriter w, String name, float[] values, long... ne) {
        w.tensorFromFloats(name, GGMLType.F32, ne, values);
    }
}
