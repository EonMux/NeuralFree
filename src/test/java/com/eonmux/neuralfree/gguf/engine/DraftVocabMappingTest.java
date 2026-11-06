package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link DraftVocabMapping}: the token-level intersection of two tokenizers, with the word-initial space marker
 * normalized ({@code Ġ} in byte-level BPE, {@code ▁} in SentencePiece), unknown-token fallbacks for tokens outside
 * it, and the logits mask that keeps a greedy draft lossless.
 */
public class DraftVocabMappingTest {

    /** GGUF {@code llama_token_type}: normal, unknown and control. */
    private static final int NORMAL = 1;
    private static final int UNKNOWN = 2;
    private static final int CONTROL = 3;

    /** A small vocab over {@code model}: its tokens and one token type per id. */
    private static GGUFTokenizer tokenizer(String model, String pre, String[] tokens, int[] types) {
        return tokenizer(model, pre, tokens, types, new HashMap<String, Object>());
    }

    /** A small vocab over {@code model} with extra GGUF scalar keys. */
    private static GGUFTokenizer tokenizer(String model, String pre, String[] tokens, int[] types,
                                           Map<String, Object> extra) {
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, model);
        md.put(TokenizerVocab.KEY_PRE, pre);
        md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(Arrays.asList(tokens)));
        List<Object> typeList = new ArrayList<>(types.length);
        for (int type : types) {
            typeList.add(type);
        }
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, typeList);
        for (int id = 0; id < types.length; id++) {
            if (types[id] == UNKNOWN) {
                md.put("tokenizer.ggml.unknown_token_id", id);
                break;
            }
        }
        if ("gpt2".equals(model)) {
            md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        }
        md.putAll(extra);
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    /** A SentencePiece target over {@code <unk> <s> </s> ▁hello hello ▁world ▁rare}. */
    private static GGUFTokenizer spmTarget() {
        return tokenizer("llama", "default",
                new String[]{"<unk>", "<s>", "</s>", "▁hello", "hello", "▁world", "▁rare"},
                new int[]{UNKNOWN, CONTROL, CONTROL, NORMAL, NORMAL, NORMAL, NORMAL});
    }

    /** A byte-level BPE draft over {@code <unk> Ġhello hello Ġworld rare}. */
    private static GGUFTokenizer bpeDraft() {
        return tokenizer("gpt2", "gpt-2",
                new String[]{"<unk>", "Ġhello", "hello", "Ġworld", "rare"},
                new int[]{UNKNOWN, NORMAL, NORMAL, NORMAL, NORMAL});
    }

    @Test
    public void tokenTextsIntersectAcrossTokenizerFamilies() {
        DraftVocabMapping m = DraftVocabMapping.of(spmTarget(), 7, bpeDraft(), 5);
        assertEquals("no unknown/special tokens match, but the pieces do", 4, m.intersectionSize());
        assertFalse(m.isIdentity());
        // ▁hello (3) is Ġhello (1); hello (4) is hello (2); ▁world (5) is Ġworld (3)
        assertEquals(1, m.toDraft(3));
        assertEquals(3, m.toTarget(1));
        assertEquals(2, m.toDraft(4));
        assertEquals(4, m.toTarget(2));
        assertEquals(3, m.toDraft(5));
        assertEquals(5, m.toTarget(3));
        // ▁rare is not in the draft vocabulary: the draft sees its unknown token (0)
        assertEquals(0, m.toDraft(6));
        // "rare" of the draft is not in the target vocabulary: it cannot be proposed
        assertEquals(-1, m.toTarget(4));
    }

    @Test
    public void theLogitsMaskKeepsOnlyIntersectedDraftTokens() {
        DraftVocabMapping m = DraftVocabMapping.of(spmTarget(), 7, bpeDraft(), 5);
        float[] logits = {1f, 2f, 3f, 4f, 5f};
        m.constrain(logits);
        assertEquals(1f, logits[0], 0f);
        assertEquals(2f, logits[1], 0f);
        assertEquals(3f, logits[2], 0f);
        assertEquals(4f, logits[3], 0f);
        assertEquals("-inf outside the intersection", Float.NEGATIVE_INFINITY, logits[4], 0f);
    }

    @Test
    public void tokensBeyondTheDraftVocabularyAreMaskedToo() {
        DraftVocabMapping m = DraftVocabMapping.of(spmTarget(), 7, bpeDraft(), 5);
        float[] logits = new float[7];
        Arrays.fill(logits, 1f);
        m.constrain(logits);
        assertEquals(1f, logits[3], 0f);
        assertEquals(Float.NEGATIVE_INFINITY, logits[5], 0f);
        assertEquals(Float.NEGATIVE_INFINITY, logits[6], 0f);
    }

    @Test
    public void anIdentityMappingPassesTokenIdsThrough() {
        DraftVocabMapping m = DraftVocabMapping.identity(5, 6);
        assertTrue(m.isIdentity());
        assertEquals("the smaller vocabulary bounds the ids both share", 5, m.intersectionSize());
        assertEquals(3, m.toDraft(3));
        assertEquals(3, m.toTarget(3));
        assertEquals("a draft token above the target's vocabulary cannot be proposed", -1, m.toTarget(5));
        assertTrue(m.canStartDraft(4));
        assertFalse(m.canStartDraft(5));
        float[] logits = {1f, 2f, 3f};
        m.constrain(logits);
        assertArrayEqualsLogits(new float[]{1f, 2f, 3f}, logits);
    }

    @Test
    public void aDraftVocabularyWithoutUnknownUsesItsEndOfSequenceToken() {
        Map<String, Object> eos = new HashMap<>();
        eos.put("tokenizer.ggml.eos_token_id", 2);
        GGUFTokenizer target = tokenizer("gpt2", "gpt-2",
                new String[]{"a", "b", "c"}, new int[]{NORMAL, NORMAL, NORMAL});
        GGUFTokenizer draft = tokenizer("gpt2", "gpt-2",
                new String[]{"a", "b", "</s>"}, new int[]{NORMAL, NORMAL, CONTROL}, eos);

        DraftVocabMapping m = DraftVocabMapping.of(target, 3, draft, 3);
        assertEquals(2, m.intersectionSize());
        assertEquals("a target token outside the intersection becomes the draft's end-of-sequence id", 2,
                m.toDraft(2));
        assertEquals("the draft's end-of-sequence token is not proposable", -1, m.toTarget(2));
    }

    @Test
    public void aVocabularyWithNeitherUnknownNorEndOfSequenceIsRefused() {
        GGUFTokenizer noFallback = tokenizer("gpt2", "gpt-2", new String[]{"a", "b"},
                new int[]{NORMAL, NORMAL});
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> DraftVocabMapping.of(spmTarget(), 7, noFallback, 2));
        assertTrue(e.getMessage(), e.getMessage().contains("draft tokenizer"));
        assertTrue(e.getMessage(), e.getMessage().contains("neither an unknown nor an end-of-sequence"));
    }

    @Test
    public void vocabulariesWithNoSharedTextAreRefused() {
        GGUFTokenizer other = tokenizer("gpt2", "gpt-2", new String[]{"<oov>", "x", "y"},
                new int[]{UNKNOWN, NORMAL, NORMAL});
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> DraftVocabMapping.of(spmTarget(), 7, other, 3));
        assertTrue(e.getMessage(), e.getMessage().contains("share no token text"));
    }

    private static void assertArrayEqualsLogits(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], 0f);
        }
    }
}
