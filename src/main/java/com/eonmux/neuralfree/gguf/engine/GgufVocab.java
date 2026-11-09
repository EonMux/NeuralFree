package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;

import java.nio.charset.StandardCharsets;

/** A SamplerVocab on a GGUF model's tokenizer, following llama.cpp's piece semantics per id. */
final class GgufVocab implements SamplerVocab {

    private static final int SPECIAL_ATTRS = TokenizerVocab.ATTR_UNKNOWN | TokenizerVocab.ATTR_CONTROL;

    private final GGUFTokenizer tokenizer;
    private final TokenizerVocab vocab;
    private final SpecialTokens special;
    private final byte[][] pieces;
    private final byte[][] specialPieces;

    GgufVocab(GGUFTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.vocab = tokenizer.vocab();
        this.special = tokenizer.specialTokens();
        this.pieces = new byte[vocab.size()][];
        this.specialPieces = new byte[vocab.size()][];
    }

    @Override
    public int vocabSize() {
        return vocab.size();
    }

    @Override
    public boolean isEog(int id) {
        return special.isEog(id);
    }

    @Override
    public int eotToken() {
        return special.eot();
    }

    @Override
    public int eosToken() {
        return special.eos();
    }

    /** token_to_piece(id, lstrip = 0, special = false): UNKNOWN and CONTROL give no bytes, a byte token its byte. */
    @Override
    public byte[] piece(int id) {
        if ((vocab.attr(id) & SPECIAL_ATTRS) != 0) {
            return new byte[0];
        }
        if (vocab.isByte(id)) {
            return new byte[]{(byte) vocab.byteValue(id)};
        }
        byte[] cached = pieces[id];
        if (cached == null) {
            // the visible piece bytes (GGUFTokenizer.decode of one token, plain decoding)
            cached = tokenizer.decode(id).getBytes(StandardCharsets.UTF_8);
            pieces[id] = cached;
        }
        return cached;
    }

    /** The cached special = true piece: CONTROL and UNKNOWN tokens render their raw text bytes. */
    @Override
    public byte[] grammarPiece(int id) {
        if ((vocab.attr(id) & SPECIAL_ATTRS) != 0) {
            byte[] cached = specialPieces[id];
            if (cached == null) {
                cached = vocab.text(id).getBytes(StandardCharsets.UTF_8);
                specialPieces[id] = cached;
            }
            return cached;
        }
        return piece(id);
    }

    /** detokenize({id}, special = true): special tokens render their raw text, visible tokens their piece. */
    @Override
    public String pieceText(int id) {
        if ((vocab.attr(id) & SPECIAL_ATTRS) != 0) {
            return vocab.text(id);
        }
        if (vocab.isByte(id)) {
            return new String(piece(id), StandardCharsets.ISO_8859_1);
        }
        return tokenizer.decode(id);
    }

    @Override
    public int[] tokenize(String text) {
        return tokenizer.encode(text, false, false);
    }

    /** detokenize semantics of the tag passes: special tokens named in the text resolve to their ids. */
    @Override
    public int[] tokenizeSpecial(String text) {
        return tokenizer.encode(text, false, true);
    }
}
