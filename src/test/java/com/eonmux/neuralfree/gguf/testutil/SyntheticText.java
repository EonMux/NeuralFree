package com.eonmux.neuralfree.gguf.testutil;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * The shared metadata, shape checks and tokenizer vocabularies of the {@link SyntheticModels} fixtures. Each writer
 * gets the keys the llama.cpp converters and loader read for its architecture, and a complete tokenizer: SPM with
 * byte fallback, byte-level BPE with merges, or WordPiece in phantom-space form.
 */
final class SyntheticText {

    private static final String SPM_SPACE = "▁";
    private static final String BPE_SPACE = "Ġ";

    private SyntheticText() {
    }

    static GgufTestWriter base(String arch, SyntheticModels.Config c) {
        GgufTestWriter w = new GgufTestWriter().architecture(arch);
        w.str("general.name", "synthetic-" + arch);
        w.u32(w.archKey("vocab_size"), c.nVocab());
        w.u32(w.archKey("context_length"), c.nCtxTrain());
        w.u32(w.archKey("embedding_length"), c.nEmbd());
        w.u32(w.archKey("block_count"), c.nLayer());
        return w;
    }

    /** Head counts and FFN length; key/value lengths when forced or when they differ from n_embd/n_head. */
    static void attentionKeys(GgufTestWriter w, SyntheticModels.Config c, boolean alwaysHeadDim) {
        w.u32(w.archKey("feed_forward_length"), c.nFf());
        w.u32(w.archKey("attention.head_count"), c.nHead());
        w.u32(w.archKey("attention.head_count_kv"), c.nHeadKv());
        if (alwaysHeadDim || (long) c.nHead() * c.headDim() != c.nEmbd()) {
            w.u32(w.archKey("attention.key_length"), c.headDim());
            w.u32(w.archKey("attention.value_length"), c.headDim());
        }
    }

    static String blk(int il, String suffix) {
        return "blk." + il + "." + suffix;
    }

    static void requireAttention(SyntheticModels.Config c, String arch) {
        if (c.nHead() <= 0 || c.nHeadKv() <= 0 || c.headDim() <= 0 || c.nFf() <= 0) {
            throw new IllegalArgumentException(arch + " needs positive head count, kv head count, head dim and"
                    + " feed-forward length");
        }
        if (c.nHead() % c.nHeadKv() != 0) {
            throw new IllegalArgumentException(arch + ": n_head (" + c.nHead() + ") must be a multiple of n_head_kv ("
                    + c.nHeadKv() + ")");
        }
    }

    static void requireSsm(SyntheticModels.Config c, String arch) {
        requirePositive(arch + " ssm conv kernel", c.ssmConv());
        requirePositive(arch + " ssm state size", c.ssmState());
        requirePositive(arch + " ssm group count", c.ssmGroups());
        requirePositive(arch + " ssm time step rank", c.ssmDtRank());
    }

    static void requirePositive(String what, long v) {
        if (v <= 0) {
            throw new IllegalArgumentException(what + " must be positive, got " + v);
        }
    }

    /**
     * The SPM vocab of the llama fixture ({@code <unk>}, {@code <s>}, {@code </s>}, then the byte tokens and scored
     * pieces), for fixtures that carry their own tokenizer beside a non-llama architecture.
     */
    static void spm(GgufTestWriter w, int nVocab) {
        spmVocab(w, nVocab, new String[] {"<unk>", "<s>", "</s>"},
                new int[] {SyntheticModels.TOKEN_TYPE_UNKNOWN, SyntheticModels.TOKEN_TYPE_CONTROL,
                        SyntheticModels.TOKEN_TYPE_CONTROL});
        w.u32("tokenizer.ggml.unknown_token_id", 0).u32("tokenizer.ggml.bos_token_id", 1)
                .u32("tokenizer.ggml.eos_token_id", 2).bool("tokenizer.ggml.add_bos_token", true)
                .bool("tokenizer.ggml.add_eos_token", false);
    }

    /** SPM vocab: specials, the 256 {@code <0xNN>} byte tokens, then scored pieces. */
    static void spmVocab(GgufTestWriter w, int nVocab, String[] specials, int[] specialTypes) {
        spmVocab(w, nVocab, specials, specialTypes, 0);
    }

    /**
     * SPM vocab with the scored pieces rotated by {@code pieceShift} ids: the vocabulary holds the same token texts,
     * but each piece sits at another id, so two fixtures describe token-incompatible vocabularies.
     */
    static void spmVocab(GgufTestWriter w, int nVocab, String[] specials, int[] specialTypes, int pieceShift) {
        int fixed = specials.length + 256;
        requireVocab("SPM", nVocab, fixed + 1);
        List<String> tokens = new ArrayList<>(nVocab);
        float[] scores = new float[nVocab];
        int[] types = new int[nVocab];
        for (int i = 0; i < specials.length; i++) {
            tokens.add(specials[i]);
            types[i] = specialTypes[i];
        }
        for (int b = 0; b < 256; b++) {
            types[tokens.size()] = SyntheticModels.TOKEN_TYPE_BYTE;
            tokens.add(String.format("<0x%02X>", b));
        }
        List<String> pieces = letterPieces(nVocab - fixed, SPM_SPACE, true);
        if (pieceShift != 0) {
            java.util.Collections.rotate(pieces, pieceShift);
        }
        for (int i = 0; i < pieces.size(); i++) {
            scores[tokens.size()] = -i;
            types[tokens.size()] = SyntheticModels.TOKEN_TYPE_NORMAL;
            tokens.add(pieces.get(i));
        }
        w.str("tokenizer.ggml.model", "llama").str("tokenizer.ggml.pre", "default")
                .strArray("tokenizer.ggml.tokens", tokens).f32Array("tokenizer.ggml.scores", scores)
                .i32Array("tokenizer.ggml.token_type", types);
    }

    /**
     * The Gemma SPM vocab: {@code <pad>} 0, {@code <eos>} 1, {@code <bos>} 2, {@code <unk>} 3 and the turn
     * markers, then the byte tokens and pieces; BOS added, no space prefix.
     */
    static void gemmaVocab(GgufTestWriter w, int nVocab) {
        spmVocab(w, nVocab,
                new String[] {"<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"},
                new int[] {SyntheticModels.TOKEN_TYPE_CONTROL, SyntheticModels.TOKEN_TYPE_CONTROL,
                        SyntheticModels.TOKEN_TYPE_CONTROL, SyntheticModels.TOKEN_TYPE_UNKNOWN,
                        SyntheticModels.TOKEN_TYPE_CONTROL, SyntheticModels.TOKEN_TYPE_CONTROL});
        w.u32("tokenizer.ggml.padding_token_id", 0).u32("tokenizer.ggml.eos_token_id", 1)
                .u32("tokenizer.ggml.bos_token_id", 2).u32("tokenizer.ggml.unknown_token_id", 3)
                .bool("tokenizer.ggml.add_bos_token", true).bool("tokenizer.ggml.add_eos_token", false)
                .bool("tokenizer.ggml.add_space_prefix", false);
    }

    /** Byte-level BPE vocab: specials, the 256 GPT-2 byte symbols, then two-symbol merges. */
    static void bpeVocab(GgufTestWriter w, int nVocab, String pre, String[] specials) {
        int fixed = specials.length + 256;
        requireVocab("BPE", nVocab, fixed + 1);
        List<String> tokens = new ArrayList<>(nVocab);
        int[] types = new int[nVocab];
        for (String s : specials) {
            types[tokens.size()] = SyntheticModels.TOKEN_TYPE_CONTROL;
            tokens.add(s);
        }
        for (int cp : gpt2ByteSymbols()) {
            types[tokens.size()] = SyntheticModels.TOKEN_TYPE_NORMAL;
            tokens.add(new String(Character.toChars(cp)));
        }
        List<String> merges = new ArrayList<>();
        for (String first : bpeFirstSymbols()) {
            for (char second = 'a'; second <= 'z' && tokens.size() < nVocab; second++) {
                types[tokens.size()] = SyntheticModels.TOKEN_TYPE_NORMAL;
                tokens.add(first + second);
                merges.add(first + " " + second);
            }
        }
        if (tokens.size() != nVocab) {
            throw new IllegalArgumentException("BPE fixture vocab supports at most " + tokens.size() + " tokens, asked "
                    + nVocab);
        }
        w.str("tokenizer.ggml.model", "gpt2").str("tokenizer.ggml.pre", pre)
                .strArray("tokenizer.ggml.tokens", tokens).i32Array("tokenizer.ggml.token_type", types)
                .strArray("tokenizer.ggml.merges", merges);
    }

    /**
     * Byte-level BPE vocab whose plain words are single tokens through their own merge chains: every prefix of a
     * GPT-2 byte-encoded word is a token joined by a merge, so a merge-rank tokenizer forms the whole word (a
     * checkpoint's {@code " yes"} answer label sits in its vocabulary this way).
     */
    static void bpeVocabWords(GgufTestWriter w, int nVocab, String pre, String[] specials, String[] words) {
        Set<String> generated = new LinkedHashSet<>();
        for (String first : bpeFirstSymbols()) {
            for (char second = 'a'; second <= 'z'; second++) {
                generated.add(first + second);
            }
        }
        List<String> tokens = new ArrayList<>(nVocab);
        int[] types = new int[nVocab];
        List<String> merges = new ArrayList<>();
        for (String s : specials) {
            types[tokens.size()] = SyntheticModels.TOKEN_TYPE_CONTROL;
            tokens.add(s);
        }
        for (int cp : gpt2ByteSymbols()) {
            types[tokens.size()] = SyntheticModels.TOKEN_TYPE_NORMAL;
            tokens.add(new String(Character.toChars(cp)));
        }
        for (String word : words) {
            String encoded = gpt2ByteEncode(word);
            for (int end = 2; end <= encoded.length(); end++) {
                String prefix = encoded.substring(0, end);
                // the word itself must be a token (an ignore-merges pre emits it whole, a label read needs it);
                // only proper prefixes may fall to the generated two-letter fillers
                boolean fullWord = end == encoded.length();
                if (tokens.contains(prefix) || !fullWord && generated.contains(prefix)) {
                    continue;
                }
                types[tokens.size()] = SyntheticModels.TOKEN_TYPE_NORMAL;
                tokens.add(prefix);
                merges.add(encoded.substring(0, end - 1) + " " + encoded.charAt(end - 1));
            }
        }
        for (String first : bpeFirstSymbols()) {
            for (char second = 'a'; second <= 'z' && tokens.size() < nVocab; second++) {
                String token = first + second;
                if (!tokens.contains(token)) {
                    types[tokens.size()] = SyntheticModels.TOKEN_TYPE_NORMAL;
                    tokens.add(token);
                    merges.add(first + " " + second);
                }
            }
        }
        if (tokens.size() != nVocab) {
            throw new IllegalArgumentException("BPE fixture vocab supports at most " + tokens.size() + " tokens,"
                    + " asked " + nVocab);
        }
        w.str("tokenizer.ggml.model", "gpt2").str("tokenizer.ggml.pre", pre)
                .strArray("tokenizer.ggml.tokens", tokens).i32Array("tokenizer.ggml.token_type", types)
                .strArray("tokenizer.ggml.merges", merges);
    }

    /** The text in GPT-2 byte encoding: printable bytes as themselves, the rest shifted by 256. */
    private static String gpt2ByteEncode(String text) {
        byte[] bytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        StringBuilder out = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            int v = b & 0xff;
            boolean printable = v >= '!' && v <= '~' || v >= 0xA1 && v <= 0xAC || v >= 0xAE && v <= 0xFF;
            out.appendCodePoint(printable ? v : v + 256);
        }
        return out.toString();
    }

    /** WordPiece vocab in llama.cpp phantom-space form: words carry U+2581, "##" continuations are bare. */
    static void wpmVocab(GgufTestWriter w, int nVocab) {
        String[] specials = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};
        requireVocab("WPM", nVocab, specials.length + 1);
        List<String> tokens = new ArrayList<>(nVocab);
        int[] types = new int[nVocab];
        for (String s : specials) {
            types[tokens.size()] = SyntheticModels.TOKEN_TYPE_CONTROL;
            tokens.add(s);
        }
        for (String piece : letterPieces(nVocab - specials.length, SPM_SPACE, false)) {
            types[tokens.size()] = SyntheticModels.TOKEN_TYPE_NORMAL;
            tokens.add(piece);
        }
        w.str("tokenizer.ggml.model", "bert").str("tokenizer.ggml.pre", "default")
                .strArray("tokenizer.ggml.tokens", tokens).i32Array("tokenizer.ggml.token_type", types);
    }

    private static void requireVocab(String kind, int nVocab, int min) {
        if (nVocab < min) {
            throw new IllegalArgumentException(kind + " fixture vocab needs at least " + min + " tokens, got " + nVocab);
        }
    }

    /**
     * {@code count} distinct pieces over a-z by increasing length, each as "{space}word" and as the bare
     * continuation "word"; the lone space marker comes first when requested.
     */
    private static List<String> letterPieces(int count, String space, boolean includeLoneSpace) {
        Set<String> out = new LinkedHashSet<>();
        if (includeLoneSpace && count > 0) {
            out.add(space);
        }
        for (int len = 1; out.size() < count; len++) {
            int combos = (int) Math.pow(26, len);
            for (int k = 0; k < combos && out.size() < count; k++) {
                StringBuilder word = new StringBuilder();
                for (int p = 0, v = k; p < len; p++, v /= 26) {
                    word.insert(0, (char) ('a' + v % 26));
                }
                out.add(space + word);
                if (out.size() < count) {
                    out.add(word.toString());
                }
            }
        }
        return new ArrayList<>(out);
    }

    /** First symbols of the BPE merges: the GPT-2 space symbol, then a-z. */
    private static List<String> bpeFirstSymbols() {
        List<String> out = new ArrayList<>();
        out.add(BPE_SPACE);
        for (char ch = 'a'; ch <= 'z'; ch++) {
            out.add(String.valueOf(ch));
        }
        return out;
    }

    /** Code points of GPT-2 {@code bytes_to_unicode} in its listing order (llama.cpp gguf-py/gguf/vocab.py). */
    private static int[] gpt2ByteSymbols() {
        List<Integer> bs = new ArrayList<>();
        for (int b = '!'; b <= '~'; b++) {
            bs.add(b);
        }
        for (int b = 0xA1; b <= 0xAC; b++) {
            bs.add(b);
        }
        for (int b = 0xAE; b <= 0xFF; b++) {
            bs.add(b);
        }
        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }
        int[] out = new int[cs.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = cs.get(i);
        }
        return out;
    }
}
