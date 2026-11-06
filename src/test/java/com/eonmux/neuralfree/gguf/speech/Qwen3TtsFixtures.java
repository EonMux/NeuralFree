package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

/**
 * The synthetic qwen3tts generator GGUF shared by the speech tests: a two-wide code predictor over one block and a
 * code2wav decoder whose quantizer, pre-transformer, one 2x upsample block and one DAC block are laid out in the
 * tensor names and ggml dimensions of tools/mtmd/clip-impl.h. The weights make the predictor's code path exactly
 * predictable and the decoder's constant path analytically known.
 */
final class Qwen3TtsFixtures {

    /** Predictor and talker width. */
    static final int W = 2;
    /** Codebook vocabulary of both predictor pages. */
    static final int VOCAB = 4;
    /** Acoustic codebooks: the predictor pages and the code2wav acoustic RVQ pages. */
    static final int CODEBOOKS = 2;
    /** Codes per frame: codec 0 plus the acoustic codebooks. */
    static final int CODES = CODEBOOKS + 1;
    /** Decoder width written by the quantizer and read by the pre-convolution. */
    static final int DEC = 2;
    /** Pre-transformer width. */
    static final int C = 4;
    /** Pre-transformer heads and head width. */
    static final int HEADS = 2;
    static final int D_HEAD = C / HEADS;
    /** Pre-transformer FFN width. */
    static final int FF = 4;
    /** ConvNeXt expand width. */
    static final int CONVNEXT_FF = 4;
    /** Decoder window of the fixture (the default 72, kept small enough for the tests). */
    static final int WINDOW = 72;
    /** The golden output sample of the constant-path fixture. */
    static final float GOLDEN = 0.2f;

    private Qwen3TtsFixtures() {
    }

    /** The fixture writer. */
    static GgufTestWriter writer() {
        GgufTestWriter w = new GgufTestWriter().architecture("clip");
        w.str("clip.gen.audio.projector_type", Qwen3TtsCodePredictor.PROJECTOR_TYPE);
        w.u32("clip.gen.audio.embedding_length", W);
        w.u32("clip.gen.audio.feed_forward_length", W);
        w.u32("clip.gen.audio.block_count", 1);
        w.u32("clip.gen.audio.attention.head_count", 1);
        w.u32("clip.gen.audio.attention.head_count_kv", 1);
        w.f32("clip.gen.audio.attention.layer_norm_epsilon", 1e-5f);
        w.u32("clip.gen.wav.attention.head_count", HEADS);
        w.u32("clip.gen.wav.attention.head_count_kv", HEADS);
        w.f32("clip.gen.wav.attention.layer_norm_epsilon", 1e-5f);
        w.f32("clip.gen.wav.rope.freq_base", 10000f);
        w.u32("clip.gen.wav.window", WINDOW);

        predictor(w);
        code2wav(w);
        return w;
    }

    /** The code predictor: identity projection, page-wise tables and zeroed blocks (so the hidden is the residual). */
    private static void predictor(GgufTestWriter w) {
        put(w, "a.gen.code.proj_in.weight", new float[]{1f, 0f, 0f, 1f}, W, W);
        put(w, "a.gen.code.proj_in.bias", new float[W], W);
        float[] embd = new float[W * VOCAB * CODEBOOKS];
        embd[0 + W * (2 + VOCAB * 0)] = 1f;   // page 0, code 2 -> [1, 0]
        embd[1 + W * (3 + VOCAB * 1)] = 1f;   // page 1, code 3 -> [0, 1]
        put(w, "a.gen.code.embd.weight", embd, W, VOCAB, CODEBOOKS);
        float[] head = new float[W * VOCAB * CODEBOOKS];
        head[0 + W * (2 + VOCAB * 0)] = 10f;  // page 0 draws code 2 for any positive row
        head[1 + W * (2 + VOCAB * 0)] = 10f;
        head[0 + W * (3 + VOCAB * 1)] = 10f;  // page 1 draws code 3
        head[1 + W * (3 + VOCAB * 1)] = 10f;
        put(w, "a.gen.code.head.weight", head, W, VOCAB, CODEBOOKS);
        float[] outEmbd = new float[W * VOCAB];
        outEmbd[0 + W * 0] = 1f;              // codec 0 code 0 -> [1, 0]
        outEmbd[1 + W * 1] = 1f;              // codec 0 code 1 -> [0, 1]
        put(w, "a.gen.code.out_embd.weight", outEmbd, W, VOCAB);
        put(w, "a.gen.code.output_norm.weight", ones(W), W);

        String b = "a.gen.code.blk.0.";
        put(w, b + "ln1.weight", ones(W), W);
        put(w, b + "attn_q.weight", new float[W * W], W, W);
        put(w, b + "attn_q_norm.weight", ones(W), W);
        put(w, b + "attn_k.weight", new float[W * W], W, W);
        put(w, b + "attn_k_norm.weight", ones(W), W);
        put(w, b + "attn_v.weight", new float[W * W], W, W);
        put(w, b + "attn_out.weight", new float[W * W], W, W);
        put(w, b + "ln2.weight", ones(W), W);
        put(w, b + "ffn_gate.weight", new float[W * W], W, W);
        put(w, b + "ffn_up.weight", new float[W * W], W, W);
        put(w, b + "ffn_down.weight", new float[W * W], W, W);
    }

    /**
     * The code2wav decoder. The quantizer projects every code to zero, so the graph's constant path is
     * {@code pre_conv bias -> tfm out bias -> upsample conv bias -> ConvNeXt residual -> dac bias -> residual
     * biases -> post conv bias}, i.e. {@link #GOLDEN} on every sample.
     */
    private static void code2wav(GgufTestWriter w) {
        float[] cbFirst = new float[DEC * VOCAB];
        put(w, "a.gen.wav.quant.first.codebook.weight", cbFirst, DEC, VOCAB);
        put(w, "a.gen.wav.quant.first.in_proj.weight", new float[DEC * DEC], DEC, DEC);
        put(w, "a.gen.wav.quant.first.out_proj.weight", new float[DEC * DEC], DEC, DEC);
        put(w, "a.gen.wav.quant.rest.codebook.weight", new float[DEC * VOCAB * CODEBOOKS], DEC, VOCAB, CODEBOOKS);
        put(w, "a.gen.wav.quant.rest.in_proj.weight", new float[DEC * DEC], DEC, DEC);
        put(w, "a.gen.wav.quant.rest.out_proj.weight", new float[DEC * DEC], DEC, DEC);

        put(w, "a.gen.wav.pre_conv.weight", new float[DEC * C], 1, DEC, C);
        put(w, "a.gen.wav.pre_conv.bias", new float[C], C);
        put(w, "a.gen.wav.tfm.in_proj.weight", new float[C * C], C, C);
        put(w, "a.gen.wav.tfm.in_proj.bias", new float[C], C);
        put(w, "a.gen.wav.tfm.out_proj.weight", new float[C * C], C, C);
        put(w, "a.gen.wav.tfm.out_proj.bias", new float[]{0.5f, -0.25f, 0.75f, 0.0f}, C);
        put(w, "a.gen.wav.tfm.output_norm.weight", ones(C), C);
        String t = "a.gen.wav.tfm.blk.0.";
        put(w, t + "ln1.weight", ones(C), C);
        put(w, t + "attn_q.weight", new float[C * C], C, C);
        put(w, t + "attn_k.weight", new float[C * C], C, C);
        put(w, t + "attn_v.weight", new float[C * C], C, C);
        put(w, t + "attn_out.weight", new float[C * C], C, C);
        put(w, t + "ls1.weight", new float[C], C);
        put(w, t + "ln2.weight", ones(C), C);
        put(w, t + "ffn_gate.weight", new float[C * FF], C, FF);
        put(w, t + "ffn_up.weight", new float[C * FF], C, FF);
        put(w, t + "ffn_down.weight", new float[FF * C], FF, C);
        put(w, t + "ls2.weight", new float[C], C);

        String u = "a.gen.wav.up.blk.0.";
        put(w, u + "conv.weight", new float[2 * 2 * C], 2, 2, C);
        put(w, u + "conv.bias", new float[]{1f, -1f}, 2);
        put(w, u + "dwconv.weight", new float[7 * 1 * 2], 7, 1, 2);
        put(w, u + "dwconv.bias", new float[2], 2);
        put(w, u + "norm.weight", ones(2), 2);
        put(w, u + "norm.bias", new float[2], 2);
        put(w, u + "pw1.weight", new float[2 * CONVNEXT_FF], 2, CONVNEXT_FF);
        put(w, u + "pw1.bias", new float[CONVNEXT_FF], CONVNEXT_FF);
        put(w, u + "pw2.weight", new float[CONVNEXT_FF * 2], CONVNEXT_FF, 2);
        put(w, u + "pw2.bias", new float[2], 2);
        put(w, u + "gamma", new float[2], 2);

        put(w, "a.gen.wav.dac.entry.weight", new float[1 * 2 * 2], 1, 2, 2);
        put(w, "a.gen.wav.dac.entry.bias", new float[]{0.1f, -0.2f}, 2);
        String d = "a.gen.wav.dac.blk.0.";
        put(w, d + "snake.alpha", new float[2], 2);
        put(w, d + "snake.beta", new float[2], 2);
        put(w, d + "conv.weight", new float[2 * 2 * 2], 2, 2, 2);
        put(w, d + "conv.bias", new float[]{0.05f, 0.05f}, 2);
        String r = d + "res.0.";
        put(w, r + "act1.alpha", new float[2], 2);
        put(w, r + "act1.beta", new float[2], 2);
        put(w, r + "conv1.weight", new float[1 * 2 * 2], 1, 2, 2);
        put(w, r + "conv1.bias", new float[]{0.1f, -0.1f}, 2);
        put(w, r + "act2.alpha", new float[2], 2);
        put(w, r + "act2.beta", new float[2], 2);
        put(w, r + "conv2.weight", new float[1 * 2 * 2], 1, 2, 2);
        put(w, r + "conv2.bias", new float[]{0.01f, 0.02f}, 2);
        put(w, "a.gen.wav.dac.post_snake.alpha", new float[2], 2);
        put(w, "a.gen.wav.dac.post_snake.beta", new float[2], 2);
        put(w, "a.gen.wav.dac.post_conv.weight", new float[1 * 2 * 1], 1, 2, 1);
        put(w, "a.gen.wav.dac.post_conv.bias", new float[]{GOLDEN}, 1);
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
