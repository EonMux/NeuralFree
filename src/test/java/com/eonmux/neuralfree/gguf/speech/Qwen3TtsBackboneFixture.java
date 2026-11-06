package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.file.Path;

/**
 * A tiny qwen3tts backbone: a one-layer qwen3vl-shaped decoder whose layer weights are zero, so the final hidden
 * state is the RMS norm of the prompt embedding row and the LM head alone decides the codec token. Every token
 * embedding is 1 and head row 0 (codec 0) is 1 while the other codec rows are 0, so top-k 1 always samples
 * {@code <|codec_0|>}. The vocabulary carries the pipeline's control tokens and a byte-fallback SPM tokenizer, so
 * any ASCII text tokenizes.
 */
public final class Qwen3TtsBackboneFixture {

    /** The backbone embedding width: the talker hidden width a paired generator file must take. */
    public static final int EMB = 8;
    static final int LAYERS = 1;
    static final int HEADS = 2;
    static final int KV = 1;
    static final int DHEAD = 4;
    static final int FF = 16;
    static final int TEXT = 272;
    static final int CODEC = 32;
    static final int VOCAB = TEXT + CODEC;

    static final String CODEC_0 = "<|codec_0|>";
    static final String CODEC_BOS = "<|codec_bos|>";
    static final String CODEC_EOS = "<|codec_eos_token|>";
    static final String CODEC_PAD = "<|codec_pad|>";
    static final String CODEC_THINK = "<|codec_think|>";
    static final String CODEC_THINK_BOS = "<|codec_think_bos|>";
    static final String CODEC_THINK_EOS = "<|codec_think_eos|>";
    static final String TTS_PAD = "<tts_pad>";
    static final String TTS_BOS = "<tts_text_bos>";
    static final String TTS_EOS = "<tts_text_eod>";
    static final String LANGUAGE = "<|codec_language_english|>";

    static final String[] TEXT_SPECIALS = {"<|im_start|>", "<|im_end|>", TTS_PAD, TTS_BOS, TTS_EOS};

    private Qwen3TtsBackboneFixture() {
    }

    static GgufTestWriter writer() {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("qwen3tts");
        w.u32("qwen3tts.context_length", 64);
        w.u32("qwen3tts.embedding_length", EMB);
        w.u32("qwen3tts.block_count", LAYERS);
        w.u32("qwen3tts.feed_forward_length", FF);
        w.u32("qwen3tts.attention.head_count", HEADS);
        w.u32("qwen3tts.attention.head_count_kv", KV);
        w.u32("qwen3tts.attention.key_length", DHEAD);
        w.u32("qwen3tts.attention.value_length", DHEAD);
        w.f32("qwen3tts.attention.layer_norm_rms_epsilon", 1e-5f);
        w.f32("qwen3tts.rope.freq_base", 10000f);
        w.i32Array("qwen3tts.rope.dimension_sections", 1, 1, 0, 0);
        tokenizer(w);
        tensors(w);
        return w;
    }

    private static void tokenizer(GgufTestWriter w) {
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
        int id = 259;
        for (String piece : TEXT_SPECIALS) {
            tokens[id] = piece;
            types[id] = 3;
            scores[id] = -1f;
            id++;
        }
        while (id < TEXT) {
            tokens[id] = "<|filler_" + (id - 259 - TEXT_SPECIALS.length) + "|>";
            types[id] = 3;
            scores[id] = -1f;
            id++;
        }
        tokens[TEXT] = CODEC_0;
        for (int i = 1; i < 16; i++) {
            tokens[TEXT + i] = "<|codec_" + i + "|>";
        }
        tokens[TEXT + 16] = CODEC_BOS;
        tokens[TEXT + 17] = CODEC_EOS;
        tokens[TEXT + 18] = CODEC_PAD;
        tokens[TEXT + 19] = CODEC_THINK;
        tokens[TEXT + 20] = CODEC_THINK_BOS;
        tokens[TEXT + 21] = CODEC_THINK_EOS;
        tokens[TEXT + 22] = LANGUAGE;
        for (int i = 23; i < CODEC; i++) {
            tokens[TEXT + i] = "<|codec_extra_" + i + "|>";
        }
        for (int i = TEXT; i < VOCAB; i++) {
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

    private static void tensors(GgufTestWriter w) {
        w.tensorFilled("token_embd.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 1f, EMB, VOCAB);
        w.tensorFilled("output_norm.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 1f, EMB);
        float[] head = new float[EMB * CODEC];
        for (int c = 0; c < EMB; c++) {
            head[c] = 1f;
        }
        w.tensorF32("output.weight", new long[]{EMB, CODEC}, head);
        w.tensorFilled("blk.0.attn_norm.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 1f, EMB);
        w.tensorFilled("blk.0.attn_q.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 0f, EMB, HEADS * DHEAD);
        w.tensorFilled("blk.0.attn_k.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 0f, EMB, KV * DHEAD);
        w.tensorFilled("blk.0.attn_v.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 0f, EMB, KV * DHEAD);
        w.tensorFilled("blk.0.attn_q_norm.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 1f, DHEAD);
        w.tensorFilled("blk.0.attn_k_norm.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 1f, DHEAD);
        w.tensorFilled("blk.0.attn_output.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 0f, HEADS * DHEAD, EMB);
        w.tensorFilled("blk.0.ffn_norm.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 1f, EMB);
        w.tensorFilled("blk.0.ffn_gate.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 0f, EMB, FF);
        w.tensorFilled("blk.0.ffn_up.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 0f, EMB, FF);
        w.tensorFilled("blk.0.ffn_down.weight", com.eonmux.neuralfree.gguf.GGMLType.F32, 0f, FF, EMB);
    }

    public static Path write() throws IOException {
        return writer().writeTempFile();
    }
}
