package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.grammar.GbnfParser;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The sampler vocabulary of a GGUF tokenizer renders llama.cpp's {@code token_to_piece} bytes: a byte-level BPE
 * token that holds part of a UTF-8 character keeps its raw bytes (the grammar sees a valid prefix, as llama.cpp's
 * partial-UTF-8 tracking does, not a U+FFFD), a decoding overlay (Hugging Face clean-up) changes only the display
 * text, and control tokens render nothing as pieces but their text as grammar pieces.
 */
public class GgufVocabTest {

    private static final int END = 0;
    /** {@code 叫} (E5 8F AB) split after its second byte: E5 8F. */
    private static final int HEAD = 1;
    /** The last byte AB alone. */
    private static final int TAIL = 2;
    /** The whole character. */
    private static final int WHOLE = 3;
    private static final int LETTER = 4;
    /** {@code " ."}, which the clean-up overlay renders as {@code "."}. */
    private static final int SPACE_DOT = 5;

    /** A GPT-2 byte-level vocabulary: E5 is {@code å}, 8F {@code ı}, AB {@code «}, a space {@code Ġ}. */
    private static GgufVocab vocab(boolean cleanUp) {
        return new GgufVocab(tokenizer(cleanUp));
    }

    private static GGUFTokenizer tokenizer(boolean cleanUp) {
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", "gpt2");
        md.put("tokenizer.ggml.pre", "gpt-2");
        md.put("tokenizer.ggml.tokens", new ArrayList<Object>(List.of("<|endoftext|>", "åı", "«", "åı«", "a", "Ġ.")));
        md.put("tokenizer.ggml.token_type", new ArrayList<Object>(List.of(3, 1, 1, 1, 1, 1)));
        md.put("tokenizer.ggml.merges", new ArrayList<Object>(List.of("åı «")));
        md.put("tokenizer.ggml.bos_token_id", END);
        md.put("tokenizer.ggml.eos_token_id", END);
        if (cleanUp) {
            md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, true);
        }
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static byte[] bytes(int... v) {
        byte[] b = new byte[v.length];
        for (int i = 0; i < v.length; i++) {
            b[i] = (byte) v[i];
        }
        return b;
    }

    @Test
    public void aSplitCharacterKeepsItsRawBytes() {
        GgufVocab v = vocab(false);
        assertArrayEquals(bytes(0xE5, 0x8F), v.piece(HEAD));
        assertArrayEquals(bytes(0xE5, 0x8F), v.grammarPiece(HEAD));
        assertArrayEquals(bytes(0xAB), v.grammarPiece(TAIL));
        assertArrayEquals(bytes(0xE5, 0x8F, 0xAB), v.grammarPiece(WHOLE));
    }

    @Test
    public void theGrammarAdmitsACharacterWrittenInTwoTokens() {
        GgufVocab v = vocab(false);
        GrammarStage g = new GrammarStage(GbnfParser.parse("root ::= \"叫\""), v);
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], List.of(g), v);
        float[] logits = new float[v.vocabSize()];
        Arrays.fill(logits, -10f);
        logits[HEAD] = 10f;
        logits[WHOLE] = 5f;
        int first = sampler.sample(logits.clone());
        assertEquals("the split head is a valid prefix of the character", HEAD, first);
        sampler.accept(first);
        assertTrue(g.admits(TAIL));
        assertFalse(g.admits(LETTER));
        assertFalse(g.admits(WHOLE));
        float[] next = new float[v.vocabSize()];
        Arrays.fill(next, -10f);
        next[TAIL] = 1f;
        next[LETTER] = 10f;
        int second = sampler.sample(next);
        assertEquals("only the completing byte follows", TAIL, second);
        sampler.accept(second);
        assertTrue(g.admits(END));
    }

    @Test
    public void theCleanUpOverlayChangesTheTextButNotThePieces() {
        GgufVocab plain = vocab(false);
        GgufVocab cleaned = vocab(true);
        assertEquals(".", cleaned.pieceText(SPACE_DOT));
        assertEquals(" .", plain.pieceText(SPACE_DOT));
        assertArrayEquals(bytes(' ', '.'), cleaned.piece(SPACE_DOT));
        assertArrayEquals(bytes(' ', '.'), cleaned.grammarPiece(SPACE_DOT));
        assertArrayEquals(bytes(0xE5, 0x8F), cleaned.grammarPiece(HEAD));
    }

    @Test
    public void aControlTokenIsNoPieceButItsTextAsAGrammarPiece() {
        GgufVocab v = vocab(false);
        assertArrayEquals(new byte[0], v.piece(END));
        assertArrayEquals("<|endoftext|>".getBytes(java.nio.charset.StandardCharsets.UTF_8), v.grammarPiece(END));
    }

    /** Every engine of one tokenizer reads one sampler vocabulary, so the piece caches and grammar table build once. */
    @Test
    public void engineVocabulariesOfOneTokenizerAreOneHandle() {
        GGUFTokenizer one = tokenizer(false);
        SamplerVocab first = Sampler.vocab(one);
        assertSame(first, Sampler.vocab(one));
        assertNotSame("another tokenizer has its own handle", first, Sampler.vocab(tokenizer(false)));
    }
}
