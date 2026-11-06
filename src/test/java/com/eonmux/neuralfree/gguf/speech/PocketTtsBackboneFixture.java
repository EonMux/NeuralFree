package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.file.Path;

/**
 * A tiny pockettts backbone: a one-layer LayerNorm decoder whose layer weights are zero, so the residual stream is
 * the constant token embedding and the biased output norm (all ones and zeros) yields a zero hidden state. Every
 * token embedding is 1, so any prompt produces the same conditioning row and the flow network alone decides the
 * frames. The vocabulary carries a byte-fallback SPM tokenizer and {@code <|audio_bos|>}, so any ASCII text
 * tokenizes and the pipeline finds its start token.
 */
public final class PocketTtsBackboneFixture {

    /** The backbone width: the flow network's conditioning width. */
    public static final int EMB = PocketTtsFixtures.EMB;
    static final int LAYERS = 1;
    static final int HEADS = 2;
    static final int KV = 1;
    static final int DHEAD = 4;
    static final int FF = 16;
    /** Byte-fallback tokens plus the start token and fillers. */
    static final int VOCAB = 264;
    static final int AUDIO_BOS = 259;
    static final int BOS_BEFORE_VOICE = 260;

    private PocketTtsBackboneFixture() {
    }

    static GgufTestWriter writer() {
        return writer(false);
    }

    /**
     * @param bosBeforeVoice whether the vocabulary carries {@code <|bos_before_voice|>} and gives it a distinct
     *                       embedding row, so a test can tell the voice prompt's rows apart
     */
    static GgufTestWriter writer(boolean bosBeforeVoice) {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("pockettts");
        w.u32("pockettts.context_length", 128);
        w.u32("pockettts.embedding_length", EMB);
        w.u32("pockettts.block_count", LAYERS);
        w.u32("pockettts.feed_forward_length", FF);
        w.u32("pockettts.attention.head_count", HEADS);
        w.u32("pockettts.attention.head_count_kv", KV);
        w.u32("pockettts.attention.key_length", DHEAD);
        w.u32("pockettts.attention.value_length", DHEAD);
        w.u32("pockettts.rope.dimension_count", DHEAD);
        w.f32("pockettts.attention.layer_norm_epsilon", 1e-5f);
        w.f32("pockettts.rope.freq_base", 10000f);
        tokenizer(w, bosBeforeVoice);
        tensors(w, bosBeforeVoice);
        return w;
    }

    private static void tokenizer(GgufTestWriter w, boolean bosBeforeVoice) {
        String[] tokens = new String[VOCAB];
        int[] types = new int[VOCAB];
        float[] scores = new float[VOCAB];
        tokens[0] = "<unk>";
        types[0] = 2;
        tokens[1] = "<s>";
        types[1] = 3;
        tokens[2] = "</s>";
        types[2] = 3;
        for (int b = 0; b < 256; b++) {
            tokens[3 + b] = String.format("<0x%02X>", b);
            types[3 + b] = 6;
            scores[3 + b] = -1f;
        }
        tokens[AUDIO_BOS] = "<|audio_bos|>";
        types[AUDIO_BOS] = 3;
        scores[AUDIO_BOS] = -1f;
        if (bosBeforeVoice) {
            tokens[BOS_BEFORE_VOICE] = "<|bos_before_voice|>";
            types[BOS_BEFORE_VOICE] = 3;
            scores[BOS_BEFORE_VOICE] = -1f;
        }
        for (int i = AUDIO_BOS + 1; i < VOCAB; i++) {
            if (bosBeforeVoice && i == BOS_BEFORE_VOICE) {
                continue;
            }
            tokens[i] = "<|filler_" + i + "|>";
            types[i] = 3;
            scores[i] = -1f;
        }
        w.str("tokenizer.ggml.model", "llama");
        w.str("tokenizer.ggml.pre", "default");
        w.strArray("tokenizer.ggml.tokens", tokens);
        w.f32Array("tokenizer.ggml.scores", scores);
        w.i32Array("tokenizer.ggml.token_type", types);
        w.u32("tokenizer.ggml.unknown_token_id", 0);
        w.u32("tokenizer.ggml.bos_token_id", 1);
        w.u32("tokenizer.ggml.eos_token_id", 2);
        w.bool("tokenizer.ggml.add_bos_token", false);
        w.bool("tokenizer.ggml.add_eos_token", false);
    }

    private static void tensors(GgufTestWriter w, boolean bosBeforeVoice) {
        if (bosBeforeVoice) {
            float[] table = new float[EMB * VOCAB];
            java.util.Arrays.fill(table, 1f);
            for (int c = 0; c < EMB; c++) {
                table[BOS_BEFORE_VOICE * EMB + c] = 7f;
            }
            w.tensorF32("token_embd.weight", new long[]{EMB, VOCAB}, table);
        } else {
            w.tensorFilled("token_embd.weight", GGMLType.F32, 1f, EMB, VOCAB);
        }
        w.tensorFilled("output_norm.weight", GGMLType.F32, 1f, EMB);
        w.tensorFilled("output_norm.bias", GGMLType.F32, 0f, EMB);
        w.tensorFilled("blk.0.attn_norm.weight", GGMLType.F32, 1f, EMB);
        w.tensorFilled("blk.0.attn_norm.bias", GGMLType.F32, 0f, EMB);
        w.tensorFilled("blk.0.attn_q.weight", GGMLType.F32, 0f, EMB, HEADS * DHEAD);
        w.tensorFilled("blk.0.attn_k.weight", GGMLType.F32, 0f, EMB, KV * DHEAD);
        w.tensorFilled("blk.0.attn_v.weight", GGMLType.F32, 0f, EMB, KV * DHEAD);
        w.tensorFilled("blk.0.attn_output.weight", GGMLType.F32, 0f, HEADS * DHEAD, EMB);
        w.tensorFilled("blk.0.ffn_norm.weight", GGMLType.F32, 1f, EMB);
        w.tensorFilled("blk.0.ffn_norm.bias", GGMLType.F32, 0f, EMB);
        w.tensorFilled("blk.0.ffn_up.weight", GGMLType.F32, 0f, EMB, FF);
        w.tensorFilled("blk.0.ffn_down.weight", GGMLType.F32, 0f, FF, EMB);
    }

    public static Path write() throws IOException {
        return writer().writeTempFile();
    }

    /** The backbone whose vocabulary carries {@code <|bos_before_voice|>} with a distinct embedding row. */
    public static Path writeWithBosBeforeVoice() throws IOException {
        return writer(true).writeTempFile();
    }
}
