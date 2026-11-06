package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.file.Path;

/**
 * A tiny synthetic lfm3 backbone shared by the lfm3a speech tests. Both profiles keep four blocks in the layout of
 * {@code LLM_ARCH_LFM3} (gated sliding-window attention, short convolution, full attention, short convolution) and
 * unit norms; the differences are the vocabulary and the embedding table.
 *
 * <p>The text-to-speech profile ({@link #ttsWriter(boolean)}, {@link #ttsBackbone()}) has zero-weight blocks, so
 * every position's hidden state stays the raw token embedding row; the last prompt row is exactly
 * {@code token_embd[audio_start]}, which the fixture sets to {@code (0, 5, 0, 0)} — the value the paired
 * {@link Lfm3aClipFixture} constant path expects. Its vocabulary is a byte-fallback SPM tokenizer with the chat and
 * audio control tokens, and it carries the model's {@code tokenizer.chat_template.tts}.
 *
 * <p>The omni profile ({@link #Lfm3aBackboneFixture(boolean, boolean)}) fills the token embedding with ones and
 * gives the untied head disjoint control directions: the speech-path tokens along the first twelve axes,
 * {@code <|im_end|>} along the last. The hidden state of any position is the RMS norm of its token row, and a
 * feedback embedding of the frame codes lands on one of the two axes, so the head's largest row picks the next
 * step. Its vocabulary is that of the pipeline fixture: SPM specials, the 256 byte tokens, the four control pieces
 * and fillers.
 */
public final class Lfm3aBackboneFixture {

    /** The backbone embedding width: the talker width a paired generator file must take. */
    public static final int EMB = 16;
    static final int LAYERS = 4;
    static final int HEADS = 2;
    static final int KV = 1;
    static final int DHEAD = EMB / HEADS;
    static final int FF = 16;
    static final int N_ROT = DHEAD;
    static final int WINDOW = 32;
    static final int L_CACHE = 4;

    /** The text-to-speech profile's token count (specials, the 256 byte tokens and the control pieces). */
    static final int TEXT = 272;
    /** The omni profile's token count. */
    static final int VOCAB = 300;
    static final int AUDIO_START = 259;
    static final int MIXED_START = 260;
    static final int TEXT_END = 261;
    static final int IM_END = 262;
    static final int FILLER = 263;

    public static final String AUDIO_START_PIECE = "<|audio_start|>";
    public static final String MIXED_START_PIECE = "<|mixed_start|>";
    public static final String TEXT_END_PIECE = "<|text_end|>";
    public static final String IM_END_PIECE = "<|im_end|>";

    /** The tts prompt template of conversion/lfm3.py (the non-thinking variant). */
    static final String TTS_TEMPLATE = "{%- set names = {\"ar\": \"Arabic\", \"da\": \"Danish\", \"de\": \"German\","
            + " \"el\": \"Greek\", \"en\": \"American English\", \"en-gb\": \"British English\","
            + " \"en-us\": \"American English\", \"es\": \"Spanish\", \"fi\": \"Finnish\", \"fr\": \"French\","
            + " \"he\": \"Hebrew\", \"it\": \"Italian\", \"ja\": \"Japanese\", \"ko\": \"Korean\", \"nl\": \"Dutch\","
            + " \"pl\": \"Polish\", \"pt\": \"Portuguese\", \"sv\": \"Swedish\", \"tr\": \"Turkish\","
            + " \"zh\": \"Chinese\"} -%}"
            + "{%- set code = (language | lower) if language is defined and language else 'en' -%}"
            + "{%- set lang = names[code] if code in names else language -%}"
            + "{%- set v = (voice | lower) if voice is defined and voice and (voice | lower) in ['female', 'male']"
            + " else 'female' -%}"
            + "{{- bos_token -}}"
            + "{{- '<|im_start|>system\\nPerform TTS in ' + lang + ' using a ' + v + ' voice.<|im_end|>\\n' -}}"
            + "{{- '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}"
            + "{{- '<|im_start|>assistant\\n<|audio_start|>' -}}";

    /** Head row weights along the speech axis; the largest live row wins the argmax. */
    static final float FILLER_W = 10.5f;
    /** Above the fillers, so the continuing TEXT logit bias is what keeps audio_start unsampled. */
    static final float AUDIO_START_W = 10.8f;
    static final float MIXED_START_W = 9f;
    static final float TEXT_END_W = 7f;
    static final float IM_END_W = 6f;
    /** A filler weight above every control row, so the interleaved text slots sample visible text. */
    static final float FILLER_WINS_W = 12f;

    private final boolean textEndWins;
    private final boolean omniTokens;
    private final boolean fillerWins;

    /**
     * An omni-profile backbone.
     *
     * @param textEndWins true raises {@code <|text_end|>} above the fillers, so a text-only continuation stops
     *                    there; false lets the fillers win and the turn run to the caps
     * @param omniTokens  false omits the four control pieces, an omni-unready vocabulary
     */
    public Lfm3aBackboneFixture(boolean textEndWins, boolean omniTokens) {
        this(textEndWins, omniTokens, false);
    }

    /**
     * @param textEndWins true raises {@code <|text_end|>} above the fillers, so a text-only continuation stops
     *                    there; false lets the fillers win and the turn run to the caps
     * @param omniTokens  false omits the four control pieces, an omni-unready vocabulary
     * @param fillerWins  true raises the fillers above the switch tokens, so an interleaved text slot samples the
     *                    filler piece instead of a control token
     */
    public Lfm3aBackboneFixture(boolean textEndWins, boolean omniTokens, boolean fillerWins) {
        this.textEndWins = textEndWins;
        this.omniTokens = omniTokens;
        this.fillerWins = fillerWins;
    }

    /** The omni profile as an lfm3 backbone GGUF. */
    public GgufTestWriter writer() {
        GgufTestWriter w = base(VOCAB, "synthetic-lfm3-omni");
        w.u32("lfm3.attention.key_length", DHEAD);
        w.u32("lfm3.attention.value_length", DHEAD);
        w.u32("lfm3.rope.dimension_count", N_ROT);
        w.f32("lfm3.rope.freq_base", 10000f);
        w.f32("lfm3.rope.freq_base_swa", 10000f);
        w.f32("lfm3.attention.layer_norm_rms_epsilon", 1e-6f);
        w.u32("lfm3.shortconv.l_cache", L_CACHE);
        w.u32("lfm3.attention.sliding_window", WINDOW);
        w.boolArray("lfm3.attention.sliding_window_pattern", true, false, false, false);
        w.bool("lfm3.attention.causal", true);
        tokenizer(w, VOCAB, omniTokens);
        w.u32("tokenizer.ggml.eos_token_id", IM_END);
        omniTensors(w);
        return w;
    }

    /** The omni profile written to a temporary file. */
    public Path write() throws IOException {
        return writer().writeTempFile();
    }

    /** The text-to-speech profile's backbone writer; {@code ttsTemplate} leaves the template out when false. */
    public static GgufTestWriter ttsWriter(boolean ttsTemplate) {
        GgufTestWriter w = base(TEXT, "synthetic-lfm3-tts");
        w.u32("lfm3.attention.key_length", DHEAD);
        w.u32("lfm3.attention.value_length", DHEAD);
        w.u32("lfm3.rope.dimension_count", N_ROT);
        w.f32("lfm3.rope.freq_base", 10000f);
        w.f32("lfm3.rope.freq_base_swa", 10000f);
        w.f32("lfm3.attention.layer_norm_rms_epsilon", 1e-5f);
        w.u32("lfm3.shortconv.l_cache", L_CACHE);
        w.u32("lfm3.attention.sliding_window", WINDOW);
        w.boolArray("lfm3.attention.sliding_window_pattern", true, false, false, false);
        w.bool("lfm3.attention.causal", true);
        tokenizer(w, TEXT, true);
        if (ttsTemplate) {
            w.str("tokenizer.chat_template.tts", TTS_TEMPLATE);
        }
        ttsTensors(w);
        return w;
    }

    /** The text-to-speech profile written to a temporary file. */
    public static Path ttsBackbone() throws IOException {
        return ttsWriter(true).writeTempFile();
    }

    private static GgufTestWriter base(int vocab, String name) {
        GgufTestWriter w = new GgufTestWriter().architecture("lfm3");
        w.str("general.name", name);
        w.u32("lfm3.vocab_size", vocab);
        w.u32("lfm3.context_length", 1024);
        w.u32("lfm3.embedding_length", EMB);
        w.u32("lfm3.block_count", LAYERS);
        w.u32Array("lfm3.attention.head_count", HEADS, 0, HEADS, 0);
        w.u32Array("lfm3.attention.head_count_kv", KV, 0, KV, 0);
        w.u32("lfm3.feed_forward_length", FF);
        return w;
    }

    /** The text-to-speech profile: zero blocks over unit norms, the audio-start row the constant path reads. */
    private static void ttsTensors(GgufTestWriter w) {
        float[] embd = new float[EMB * TEXT];
        embd[(259 + 2) * EMB + 1] = 5f; // <|audio_start|> is the third SPECIALS entry
        w.tensorFromFloats("token_embd.weight", GGMLType.F32, new long[]{EMB, TEXT}, embd);
        w.tensorFilled("output_norm.weight", GGMLType.F32, 1f, EMB);
        w.tensorFilled("rope_freqs.weight", GGMLType.F32, 0f, N_ROT / 2);
        for (int il = 0; il < LAYERS; il++) {
            String p = "blk." + il + ".";
            w.tensorFilled(p + "attn_norm.weight", GGMLType.F32, 1f, EMB);
            w.tensorFilled(p + "ffn_norm.weight", GGMLType.F32, 1f, EMB);
            w.tensorFilled(p + "ffn_gate.weight", GGMLType.F32, 0f, EMB, FF);
            w.tensorFilled(p + "ffn_up.weight", GGMLType.F32, 0f, EMB, FF);
            w.tensorFilled(p + "ffn_down.weight", GGMLType.F32, 0f, FF, EMB);
            if (il % 2 == 0) {
                w.tensorFilled(p + "attn_q.weight", GGMLType.F32, 0f, EMB, HEADS * DHEAD);
                w.tensorFilled(p + "attn_q_norm.weight", GGMLType.F32, 1f, DHEAD);
                w.tensorFilled(p + "attn_k.weight", GGMLType.F32, 0f, EMB, KV * DHEAD);
                w.tensorFilled(p + "attn_k_norm.weight", GGMLType.F32, 1f, DHEAD);
                w.tensorFilled(p + "attn_v.weight", GGMLType.F32, 0f, EMB, KV * DHEAD);
                w.tensorFilled(p + "attn_output.weight", GGMLType.F32, 0f, HEADS * DHEAD, EMB);
                w.tensorFilled(p + "attn_gate.weight", GGMLType.F32, 0f, EMB, HEADS);
            } else {
                w.tensorFilled(p + "shortconv.conv.weight", GGMLType.F32, 0f, L_CACHE, EMB);
                w.tensorFilled(p + "shortconv.in_proj.weight", GGMLType.F32, 0f, EMB, 3L * EMB);
                w.tensorFilled(p + "shortconv.out_proj.weight", GGMLType.F32, 0f, EMB, EMB);
            }
        }
    }

    /** Zero blocks over unit norms; the token embedding and head carry the omni test's control directions. */
    private void omniTensors(GgufTestWriter w) {
        float[] embd = filled(EMB * VOCAB, 1f);
        float[] head = new float[EMB * VOCAB];
        float textEndWeight = textEndWins ? 11f : TEXT_END_W;
        speechRow(head, AUDIO_START, AUDIO_START_W);
        speechRow(head, MIXED_START, MIXED_START_W);
        speechRow(head, TEXT_END, textEndWeight);
        speechRow(head, FILLER, fillerWins ? FILLER_WINS_W : FILLER_W);
        head[(EMB - 1) + EMB * IM_END] = IM_END_W;
        w.tensorFromFloats("token_embd.weight", GGMLType.F32, new long[]{EMB, VOCAB}, embd);
        w.tensorFromFloats("output.weight", GGMLType.F32, new long[]{EMB, VOCAB}, head);
        w.tensorFilled("output_norm.weight", GGMLType.F32, 1f, EMB);
        w.tensorFilled("rope_freqs.weight", GGMLType.F32, 10000f, N_ROT / 2);
        for (int il = 0; il < LAYERS; il++) {
            String b = "blk." + il + ".";
            w.tensorFilled(b + "attn_norm.weight", GGMLType.F32, 1f, EMB);
            w.tensorFilled(b + "ffn_norm.weight", GGMLType.F32, 1f, EMB);
            w.tensorFilled(b + "ffn_gate.weight", GGMLType.F32, 0f, EMB, FF);
            w.tensorFilled(b + "ffn_up.weight", GGMLType.F32, 0f, EMB, FF);
            w.tensorFilled(b + "ffn_down.weight", GGMLType.F32, 0f, FF, EMB);
            if (il % 2 == 0) {
                w.tensorFilled(b + "attn_q.weight", GGMLType.F32, 0f, EMB, HEADS * DHEAD);
                w.tensorFilled(b + "attn_q_norm.weight", GGMLType.F32, 1f, DHEAD);
                w.tensorFilled(b + "attn_k.weight", GGMLType.F32, 0f, EMB, KV * DHEAD);
                w.tensorFilled(b + "attn_k_norm.weight", GGMLType.F32, 1f, DHEAD);
                w.tensorFilled(b + "attn_v.weight", GGMLType.F32, 0f, EMB, KV * DHEAD);
                w.tensorFilled(b + "attn_output.weight", GGMLType.F32, 0f, HEADS * DHEAD, EMB);
                w.tensorFilled(b + "attn_gate.weight", GGMLType.F32, 0f, EMB, HEADS);
            } else {
                w.tensorFilled(b + "shortconv.conv.weight", GGMLType.F32, 0f, L_CACHE, EMB);
                w.tensorFilled(b + "shortconv.in_proj.weight", GGMLType.F32, 0f, EMB, 3L * EMB);
                w.tensorFilled(b + "shortconv.out_proj.weight", GGMLType.F32, 0f, EMB, EMB);
            }
        }
    }

    /** Writes {@code value} onto the first twelve axes of one head row, the speech direction. */
    private static void speechRow(float[] head, int token, float value) {
        for (int r = 0; r < 12; r++) {
            head[r + EMB * token] = value;
        }
    }

    private static float[] filled(int n, float value) {
        float[] v = new float[n];
        java.util.Arrays.fill(v, value);
        return v;
    }

    /**
     * The SPM vocabulary of {@code vocab} tokens: unk, bos, eos, the 256 byte tokens, the four control pieces and
     * fillers. The omni profile always writes the control pieces; a non-omni vocabulary ({@code omniTokens} false)
     * writes fillers in their place.
     */
    private static void tokenizer(GgufTestWriter w, int vocab, boolean controls) {
        String[] tokens = new String[vocab];
        int[] types = new int[vocab];
        float[] scores = new float[vocab];
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
        String[] pieces = controls
                ? new String[]{AUDIO_START_PIECE, MIXED_START_PIECE, TEXT_END_PIECE, IM_END_PIECE}
                : new String[]{"<|filler_a|>", "<|filler_b|>", "<|filler_c|>", "<|filler_d|>"};
        for (String piece : pieces) {
            tokens[id] = piece;
            types[id] = 3;
            scores[id] = -1f;
            id++;
        }
        while (id < vocab) {
            tokens[id] = "<|filler_" + (id - FILLER) + "|>";
            types[id] = 3;
            scores[id] = -1f;
            id++;
        }
        w.str("tokenizer.ggml.model", "llama");
        w.str("tokenizer.ggml.pre", "default");
        w.strArray("tokenizer.ggml.tokens", tokens);
        w.f32Array("tokenizer.ggml.scores", scores);
        w.i32Array("tokenizer.ggml.token_type", types);
        w.u32("tokenizer.ggml.unknown_token_id", 0);
        w.u32("tokenizer.ggml.bos_token_id", 1);
        if (vocab != VOCAB) {
            w.u32("tokenizer.ggml.eos_token_id", 2);
        }
        w.bool("tokenizer.ggml.add_bos_token", false);
        w.bool("tokenizer.ggml.add_eos_token", false);
    }
}
