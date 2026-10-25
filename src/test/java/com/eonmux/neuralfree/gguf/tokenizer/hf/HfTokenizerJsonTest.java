package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfTokenizerJson}: model types and fields, the two merges forms with llama.cpp's space re-encoding, added-token
 * attributes, component views, and the error contract of {@link HfTokenizerJson#parse}.
 */
public class HfTokenizerJsonTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static HfTokenizerJson of(String json) {
        return HfTokenizerJson.of(Json.parse(json), "tokenizer.json");
    }

    private static String bpe(String merges) {
        return "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"ab\":2,\"Ġ\":3},\"merges\":" + merges + "}}";
    }

    private static void assertMessageContains(Throwable e, String expected) {
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }

    @Test
    public void mergesInTheStringFormAreKeptAsWritten() {
        HfTokenizerJson tj = of(bpe("[\"a b\",\"ab Ġ\"]"));
        assertArrayEquals(new String[]{"a b", "ab Ġ"}, tj.model().merges());
        assertEquals(2, tj.model().mergeCount());
        assertEquals("ab Ġ", tj.model().merge(1));
    }

    @Test
    public void mergesInThePairFormAreJoinedWithSpacesReencoded() {
        HfTokenizerJson tj = of(bpe("[[\"a\",\"b\"],[\"a b\",\" c\"],[\"Ġ\",\"Ġ\"]]"));
        assertArrayEquals(new String[]{"a b", "aĠb Ġc", "Ġ Ġ"}, tj.model().merges());
    }

    @Test
    public void mergesMixingTheTwoFormsAreRejected() {
        JsonException e = assertThrows(JsonException.class, () -> of(bpe("[[\"a\",\"b\"],\"a b\"]")));
        assertMessageContains(e, "model.merges[1]");
        e = assertThrows(JsonException.class, () -> of(bpe("[\"a b\",[\"a\",\"b\"]]")));
        assertMessageContains(e, "model.merges[1]");
    }

    @Test
    public void malformedMergesAreRejected() {
        assertThrows(JsonException.class, () -> of(bpe("[\"ab\"]")));
        assertThrows(JsonException.class, () -> of(bpe("[\"a b c\"]")));
        assertThrows(JsonException.class, () -> of(bpe("[[\"a\",\"b\",\"c\"]]")));
        assertThrows(JsonException.class, () -> of(bpe("[[\"a\",1]]")));
    }

    @Test
    public void bpeModelFieldsAreRead() {
        HfTokenizerJson tj = of("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"<unk>\":0,\"a\":2},\"merges\":[],"
                + "\"unk_token\":\"<unk>\",\"fuse_unk\":true,\"byte_fallback\":true,\"continuing_subword_prefix\":\"\","
                + "\"end_of_word_suffix\":\"</w>\",\"dropout\":0.1,\"ignore_merges\":true}}");
        HfTokenizerJson.Model m = tj.model();
        assertEquals(HfTokenizerJson.ModelType.BPE, m.type());
        assertEquals(3, m.vocabSize());
        assertEquals("<unk>", m.token(0));
        assertNull(m.token(1));
        assertEquals("a", m.token(2));
        assertEquals("<unk>", m.unkToken());
        assertTrue(m.fuseUnk());
        assertTrue(m.byteFallback());
        assertEquals("", m.continuingSubwordPrefix());
        assertEquals("</w>", m.endOfWordSuffix());
        assertEquals(0.1, m.dropout(), 0.0);
        assertTrue(m.ignoreMerges());
        assertEquals(-1, m.unkId());
    }

    @Test
    public void absentBpeFieldsKeepTheirDefaults() {
        HfTokenizerJson.Model m = of("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[],"
                + "\"dropout\":null,\"unk_token\":null}}").model();
        assertNull(m.unkToken());
        assertNull(m.dropout());
        assertNull(m.continuingSubwordPrefix());
        assertFalse(m.fuseUnk());
        assertFalse(m.byteFallback());
        assertFalse(m.ignoreMerges());
    }

    @Test
    public void vocabCopiesDoNotChangeTheView() {
        HfTokenizerJson.Model m = of(bpe("[]")).model();
        String[] vocab = m.vocab();
        vocab[0] = "changed";
        assertEquals("a", m.token(0));
    }

    @Test
    public void sharedIdsAndIdsOutOfRangeAreRejected() {
        JsonException e = assertThrows(JsonException.class,
                () -> of("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":0},\"merges\":[]}}"));
        assertMessageContains(e, "share id 0");
        assertThrows(JsonException.class,
                () -> of("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":-1},\"merges\":[]}}"));
        assertThrows(JsonException.class,
                () -> of("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":16777216},\"merges\":[]}}"));
    }

    @Test
    public void aModelWithoutTypeIsBpeOrUnigramByItsVocabulary() {
        HfTokenizerJson bpe = of("{\"model\":{\"vocab\":{\"a\":0,\"b\":1},\"merges\":[\"a b\"]}}");
        assertEquals(HfTokenizerJson.ModelType.BPE, bpe.model().type());
        HfTokenizerJson unigram = of("{\"model\":{\"vocab\":[[\"<unk>\",0.0],[\"a\",-1.5]],\"unk_id\":0}}");
        assertEquals(HfTokenizerJson.ModelType.UNIGRAM, unigram.model().type());
    }

    @Test
    public void unigramPiecesAndScoresAreRead() {
        HfTokenizerJson.Model m = of("{\"model\":{\"type\":\"Unigram\",\"unk_id\":1,\"byte_fallback\":true,"
                + "\"vocab\":[[\"<pad>\",0],[\"<unk>\",0.0],[\"▁a\",-2.25]]}}").model();
        assertEquals(HfTokenizerJson.ModelType.UNIGRAM, m.type());
        assertArrayEquals(new String[]{"<pad>", "<unk>", "▁a"}, m.vocab());
        assertArrayEquals(new double[]{0.0, 0.0, -2.25}, m.scores(), 0.0);
        assertEquals(1, m.unkId());
        assertTrue(m.byteFallback());
        assertEquals(0, m.mergeCount());
        assertThrows(JsonException.class,
                () -> of("{\"model\":{\"type\":\"Unigram\",\"unk_id\":3,\"vocab\":[[\"a\",0.0]]}}"));
        assertThrows(JsonException.class,
                () -> of("{\"model\":{\"type\":\"Unigram\",\"vocab\":[[\"a\"]]}}"));
    }

    @Test
    public void wordPieceAndWordLevelModelsAreRead() {
        HfTokenizerJson.Model wp = of("{\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"[UNK]\","
                + "\"continuing_subword_prefix\":\"##\",\"max_input_chars_per_word\":64,"
                + "\"vocab\":{\"[UNK]\":0,\"##a\":1}}}").model();
        assertEquals(HfTokenizerJson.ModelType.WORD_PIECE, wp.type());
        assertEquals("##", wp.continuingSubwordPrefix());
        assertEquals(64, wp.maxInputCharsPerWord());
        HfTokenizerJson.Model wpDefaults = of("{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"[UNK]\":0}}}").model();
        assertEquals("[UNK]", wpDefaults.unkToken());
        assertEquals("##", wpDefaults.continuingSubwordPrefix());
        assertEquals(100, wpDefaults.maxInputCharsPerWord());
        HfTokenizerJson.Model wl = of("{\"model\":{\"type\":\"WordLevel\",\"unk_token\":\"[UNK]\","
                + "\"vocab\":{\"[UNK]\":0,\"hello\":1}}}").model();
        assertEquals(HfTokenizerJson.ModelType.WORD_LEVEL, wl.type());
        assertEquals("hello", wl.token(1));
        assertEquals("<unk>", of("{\"model\":{\"type\":\"WordLevel\",\"vocab\":{\"<unk>\":0}}}").model().unkToken());
        JsonException e = assertThrows(JsonException.class,
                () -> of("{\"model\":{\"type\":\"Wordy\",\"vocab\":{}}}"));
        assertMessageContains(e, "unknown model type 'Wordy'");
    }

    @Test
    public void addedTokenAttributesAreRead() {
        HfTokenizerJson tj = of("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]},\"added_tokens\":["
                + "{\"id\":5,\"content\":\"<|im_start|>\",\"single_word\":false,\"lstrip\":true,\"rstrip\":false,"
                + "\"normalized\":false,\"special\":true},"
                + "{\"id\":6,\"content\":\" word\",\"single_word\":true,\"lstrip\":false,\"rstrip\":true,"
                + "\"normalized\":true,\"special\":false},"
                + "{\"id\":7,\"content\":\"<x>\",\"special\":true},"
                + "{\"id\":8,\"content\":\"y\"}]}");
        List<HfTokenizerJson.AddedToken> added = tj.addedTokens();
        assertEquals(4, added.size());
        assertEquals(new HfTokenizerJson.AddedToken(5, "<|im_start|>", true, true, false, false, false), added.get(0));
        assertEquals(new HfTokenizerJson.AddedToken(6, " word", false, false, true, true, true), added.get(1));
        assertFalse("an absent 'normalized' is false for special tokens", added.get(2).normalized());
        assertTrue("an absent 'normalized' is true for other tokens", added.get(3).normalized());
        assertSame(added.get(1), tj.addedToken(" word"));
        assertNull(tj.addedToken("missing"));
        assertThrows(UnsupportedOperationException.class, () -> added.remove(0));
        assertThrows(JsonException.class, () -> of("{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
                + "\"added_tokens\":[{\"id\":-3,\"content\":\"x\"}]}"));
        assertThrows(JsonException.class, () -> of("{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
                + "\"added_tokens\":[{\"id\":3,\"content\":\"x\",\"lstrip\":\"yes\"}]}"));
    }

    @Test
    public void componentsAreViewsOrNull() {
        HfTokenizerJson tj = of("{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},\"normalizer\":null,"
                + "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false},"
                + "\"decoder\":{\"type\":\"ByteLevel\"}}");
        assertNull(tj.normalizer());
        assertNull(tj.postProcessor());
        assertEquals("ByteLevel", tj.preTokenizer().str("type"));
        assertFalse(tj.preTokenizer().bool("add_prefix_space"));
        assertEquals("pre_tokenizer", tj.preTokenizer().path());
        assertEquals("ByteLevel", tj.decoder().str("type"));
        assertThrows(JsonException.class,
                () -> of("{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},\"decoder\":\"ByteLevel\"}"));
        assertThrows(JsonException.class, () -> of("{\"normalizer\":null}"));
    }

    @Test
    public void parseReadsAFileAndWrapsErrorsAsInvalidCheckpoint() throws IOException {
        Path dir = tmp.newFolder().toPath();
        Path good = dir.resolve("tokenizer.json");
        Files.write(good, bpe("[[\"a\",\"b\"]]").getBytes(StandardCharsets.UTF_8));
        HfTokenizerJson tj = HfTokenizerJson.parse(good);
        assertEquals(good.toString(), tj.source());
        assertArrayEquals(new String[]{"a b"}, tj.model().merges());

        Files.write(good, "{\"model\":".getBytes(StandardCharsets.UTF_8));
        IOException syntax = assertThrows(IOException.class, () -> HfTokenizerJson.parse(good));
        assertTrue(syntax.getMessage(), syntax.getMessage().startsWith("Invalid Hugging Face checkpoint '" + dir
                + "': " + good));
        assertTrue(syntax.getCause() instanceof JsonException);

        Files.write(good, bpe("[\"ab\"]").getBytes(StandardCharsets.UTF_8));
        IOException shape = assertThrows(IOException.class, () -> HfTokenizerJson.parse(good));
        assertMessageContains(shape, "Invalid Hugging Face checkpoint '" + dir + "'");
        assertMessageContains(shape, "model.merges[0]");

        Files.write(good, new byte[]{'{', (byte) 0xC3, '}'});
        assertThrows(IOException.class, () -> HfTokenizerJson.parse(good));
    }
}
