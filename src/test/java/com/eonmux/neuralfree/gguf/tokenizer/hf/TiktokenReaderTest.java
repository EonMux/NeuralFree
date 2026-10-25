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
import java.util.Base64;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The tiktoken-rank readers: {@link TiktokenReader} ({@code base64 rank} lines) and {@link TekkenReader} (tekken.json,
 * tiktoken ranks plus special tokens), with mistral_common's loading rules.
 */
public class TiktokenReaderTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static byte[] ascii(String s) {
        return s.getBytes(StandardCharsets.ISO_8859_1);
    }

    private static String b64(String s) {
        return Base64.getEncoder().encodeToString(s.getBytes(StandardCharsets.UTF_8));
    }

    private static void assertMessageContains(Throwable e, String expected) {
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }

    // ------------------------------------------------------------------ tiktoken

    @Test
    public void linesDecodeToTokenBytesAndRanks() throws IOException {
        String text = b64("!") + " 0\n" + b64("\"") + " 1\r\n\r\n" + b64("#$") + "\t2\r" + b64("é") + " 3";
        TiktokenReader r = TiktokenReader.parse(ascii(text));
        assertEquals(4, r.size());
        assertArrayEquals("!".getBytes(StandardCharsets.UTF_8), r.token(0));
        assertEquals(0, r.rank(0));
        assertArrayEquals("#$".getBytes(StandardCharsets.UTF_8), r.token(2));
        assertEquals(2, r.rankOf("#$".getBytes(StandardCharsets.UTF_8)));
        assertEquals(3, r.rankOf(new byte[]{(byte) 0xC3, (byte) 0xA9}));
        assertEquals(-1, r.rankOf("zz".getBytes(StandardCharsets.UTF_8)));
        assertArrayEquals("\"".getBytes(StandardCharsets.UTF_8), r.tokenOfRank(1));
        assertNull(r.tokenOfRank(7));
        assertEquals(3, r.maxRank());
    }

    @Test
    public void tokenCopiesDoNotChangeTheReader() throws IOException {
        TiktokenReader r = TiktokenReader.parse(ascii(b64("ab") + " 0\n"));
        r.token(0)[0] = 'z';
        assertArrayEquals(ascii("ab"), r.token(0));
    }

    @Test
    public void malformedLinesAreRejectedWithTheirLineNumber() {
        IOException base64 = assertThrows(IOException.class,
                () -> TiktokenReader.parse(ascii(b64("a") + " 0\n%%% 1\n")));
        assertMessageContains(base64, "Invalid tiktoken file: line 2: ");
        assertMessageContains(base64, "not valid base64");
        assertMessageContains(assertThrows(IOException.class,
                () -> TiktokenReader.parse(ascii(b64("a") + " 0 extra\n"))), "line 1: expected");
        assertThrows(IOException.class, () -> TiktokenReader.parse(ascii(b64("a") + "\n")));
        assertThrows(IOException.class, () -> TiktokenReader.parse(ascii(b64("a") + " -1\n")));
        assertThrows(IOException.class, () -> TiktokenReader.parse(ascii(b64("a") + " 1e3\n")));
        assertThrows(IOException.class, () -> TiktokenReader.parse(ascii(b64("a") + " 99999999999\n")));
        assertThrows(IOException.class, () -> TiktokenReader.parse(ascii(b64("a") + " 0\n   \n")));
    }

    @Test
    public void repeatedTokensAndRanksAreRejected() {
        assertMessageContains(assertThrows(IOException.class,
                () -> TiktokenReader.parse(ascii(b64("a") + " 0\n" + b64("b") + " 0\n"))), "rank 0 is assigned twice");
        assertMessageContains(assertThrows(IOException.class,
                () -> TiktokenReader.parse(ascii(b64("a") + " 0\n" + b64("a") + " 1\n"))), "repeats the token");
    }

    @Test
    public void readNamesTheCheckpointAndFile() throws IOException {
        Path dir = tmp.newFolder().toPath();
        Path file = dir.resolve("qwen.tiktoken");
        Files.write(file, ascii(b64("a") + " 0\n" + b64("b") + " 1\n"));
        assertEquals(2, TiktokenReader.read(file).size());
        Files.write(file, ascii("a"));
        IOException e = assertThrows(IOException.class, () -> TiktokenReader.read(file));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid Hugging Face checkpoint '" + dir + "': "
                + file + ": line 1: "));
    }

    // ------------------------------------------------------------------ tekken

    /** A tekken.json with the 256 single bytes plus {@code extra} merged tokens. */
    private static String tekken(String version, String specialTokens, int numSpecial, String... extra) {
        StringBuilder vocab = new StringBuilder();
        for (int i = 0; i < 256 + extra.length; i++) {
            byte[] bytes = i < 256 ? new byte[]{(byte) i} : extra[i - 256].getBytes(StandardCharsets.UTF_8);
            if (i > 0) {
                vocab.append(',');
            }
            vocab.append("{\"rank\":").append(i).append(",\"token_bytes\":\"")
                    .append(Base64.getEncoder().encodeToString(bytes)).append("\",\"token_str\":null}");
        }
        int vocabSize = numSpecial + 256 + extra.length;
        return "{\"config\":{\"pattern\":\"\\\\p{L}+|\\\\s+\",\"num_vocab_tokens\":" + (256 + extra.length)
                + ",\"default_vocab_size\":" + vocabSize + ",\"default_num_special_tokens\":" + numSpecial
                + ",\"version\":\"" + version + "\"},\"vocab\":[" + vocab + "]"
                + (specialTokens == null ? "" : ",\"special_tokens\":" + specialTokens) + "}";
    }

    private static TekkenReader tekkenOf(String json) {
        return TekkenReader.of(Json.parse(json), "tekken.json");
    }

    private static final String SPECIALS = "[{\"rank\":0,\"token_str\":\"<unk>\",\"is_control\":true},"
            + "{\"rank\":1,\"token_str\":\"<s>\",\"is_control\":true},"
            + "{\"rank\":2,\"token_str\":\"</s>\",\"is_control\":false}]";

    @Test
    public void tekkenReadsVocabSpecialTokensPatternAndVersion() {
        TekkenReader t = tekkenOf(tekken("v13", SPECIALS, 5, "ab", "abc"));
        assertEquals("v13", t.version());
        assertEquals(13, t.versionNumber());
        assertEquals("\\p{L}+|\\s+", t.pattern());
        assertEquals(263, t.vocabSize());
        assertEquals(258, t.numVocabTokens());
        assertEquals(5, t.numSpecialTokens());
        assertEquals(258, t.innerVocabSize());
        assertEquals("<s>", t.specialToken(1));
        assertFalse(t.specialIsControl(2));
        assertEquals("<SPECIAL_3>", t.specialToken(3));
        assertEquals("<SPECIAL_4>", t.specialToken(4));
        assertTrue(t.specialIsControl(4));
        assertEquals(1, t.specialId("<s>"));
        assertEquals(-1, t.specialId("[INST]"));
        assertArrayEquals(new byte[]{65}, t.tokenBytes(65));
        assertArrayEquals(ascii("abc"), t.tokenBytes(257));
    }

    @Test
    public void tekkenTruncatesTheVocabularyToTheConfiguredSize() {
        String json = tekken("v13", SPECIALS, 4, "ab", "abc").replace("\"default_vocab_size\":262",
                "\"default_vocab_size\":261");
        TekkenReader t = tekkenOf(json);
        assertEquals(257, t.innerVocabSize());
        assertArrayEquals(ascii("ab"), t.tokenBytes(256));
    }

    @Test
    public void tekkenUpToV7DefaultsTheSpecialTokens() {
        TekkenReader t = tekkenOf(tekken("v7", null, 20));
        assertEquals(20, t.numSpecialTokens());
        assertEquals("[INST]", t.specialToken(3));
        assertEquals("[TOOL_CONTENT]", t.specialToken(19));
        JsonException e = assertThrows(JsonException.class, () -> tekkenOf(tekken("v11", null, 20)));
        assertMessageContains(e, "special_tokens");
    }

    @Test
    public void tekkenRejectsInconsistentFiles() {
        assertMessageContains(assertThrows(JsonException.class, () -> tekkenOf(tekken("13", SPECIALS, 5))),
                "is not a tokenizer version");
        assertMessageContains(assertThrows(JsonException.class, () -> tekkenOf(tekken("v13", SPECIALS, 2))),
                "exceed default_num_special_tokens");
        String duplicate = "[{\"rank\":0,\"token_str\":\"<s>\"},{\"rank\":1,\"token_str\":\"<s>\"}]";
        assertMessageContains(assertThrows(JsonException.class, () -> tekkenOf(tekken("v13", duplicate, 3))),
                "appears more than once");
        String misranked = "[{\"rank\":1,\"token_str\":\"<s>\"}]";
        assertThrows(JsonException.class, () -> tekkenOf(tekken("v13", misranked, 3)));
        String wrongByte = tekken("v13", SPECIALS, 3).replace("{\"rank\":65,\"token_bytes\":\"QQ==\"",
                "{\"rank\":65,\"token_bytes\":\"Qg==\"");
        assertMessageContains(assertThrows(JsonException.class, () -> tekkenOf(wrongByte)), "single byte 65");
        String repeated = tekken("v13", SPECIALS, 3, "ab", "ab");
        assertMessageContains(assertThrows(JsonException.class, () -> tekkenOf(repeated)), "repeat");
        String tooLarge = tekken("v13", SPECIALS, 3).replace("\"default_vocab_size\":259",
                "\"default_vocab_size\":300");
        assertThrows(JsonException.class, () -> tekkenOf(tooLarge));
    }

    @Test
    public void tekkenReadWrapsErrorsAsInvalidCheckpoint() throws IOException {
        Path dir = tmp.newFolder().toPath();
        Path file = dir.resolve("tekken.json");
        Files.write(file, tekken("v13", SPECIALS, 3, "ab").getBytes(StandardCharsets.UTF_8));
        assertEquals(257, TekkenReader.read(file).innerVocabSize());
        Files.write(file, tekken("v13", null, 3).getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> TekkenReader.read(file));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid Hugging Face checkpoint '" + dir + "': "));
        assertMessageContains(e, "special_tokens");
    }
}
