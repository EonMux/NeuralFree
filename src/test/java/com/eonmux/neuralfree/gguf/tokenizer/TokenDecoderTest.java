package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.PrependScheme;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

/** {@link TokenDecoder} against Hugging Face tokenizers {@code decode} results. */
public class TokenDecoderTest {

    private static final int METASPACE = 0x2581;

    private static List<String> pieces(String... p) {
        return Arrays.asList(p);
    }

    @Test
    public void byteLevel() {
        TokenDecoder d = TokenDecoder.byteLevel(true, true, true);
        assertEquals(" h�é一�", d.decode(pieces("Ġh", "é", "Ã©", "一", "Ã")));
        assertEquals("���", d.decode(pieces("íłĢ")));
    }

    @Test
    public void lossyUtf8ReplacesEachMaximalSubpart() {
        assertEquals("��", TokenDecoder.lossyUtf8(new byte[] {(byte) 0xE6, (byte) 0x97, (byte) 0xFF}));
        assertEquals("�", TokenDecoder.lossyUtf8(new byte[] {(byte) 0xF0, (byte) 0x9F, (byte) 0x98}));
        assertEquals("�A", TokenDecoder.lossyUtf8(new byte[] {(byte) 0xF0, (byte) 0x9F, 0x41}));
        assertEquals("����",
                TokenDecoder.lossyUtf8(new byte[] {(byte) 0xF4, (byte) 0x90, (byte) 0x80, (byte) 0x80}));
        assertEquals("��", TokenDecoder.lossyUtf8(new byte[] {(byte) 0xC0, (byte) 0xAF}));
        assertEquals("a日😀", TokenDecoder.lossyUtf8("a日😀".getBytes(java.nio.charset.StandardCharsets.UTF_8)));
    }

    @Test
    public void metaspace() {
        assertEquals("ab c ", TokenDecoder.metaspace(METASPACE, PrependScheme.ALWAYS, true)
                .decode(pieces("▁▁a▁b", "▁c▁")));
        assertEquals(" a b", TokenDecoder.metaspace(METASPACE, PrependScheme.NEVER, true).decode(pieces("▁a", "▁b")));
    }

    @Test
    public void wordPiece() {
        assertEquals("##a bc. don't", TokenDecoder.wordPiece("##", true)
                .decode(pieces("##a", "b", "##c", ".", "do", "n't")));
        assertEquals("hello world .", TokenDecoder.wordPiece("##", false).decode(pieces("hel", "##lo", "world", ".")));
    }

    @Test
    public void bpeDecoder() {
        assertEquals("hello world", TokenDecoder.bpe("</w>").decode(pieces("hel", "lo</w>", "world</w>")));
        assertEquals("a b c", TokenDecoder.bpe("</w>").decode(pieces("a</w>b</w>", "c</w>")));
    }

    @Test
    public void replace() {
        assertEquals(" ab ", TokenDecoder.replaceRegex("▁+", " ").decode(pieces("▁▁a", "b▁")));
        assertEquals(" a b", TokenDecoder.replace("▁", " ").decode(pieces("▁a", "▁b")));
        assertEquals(Arrays.asList("", "-a-"), TokenDecoder.replaceRegex("x*", "-").decodeChain(pieces("", "a")));
    }

    @Test
    public void strip() {
        assertEquals("a b", TokenDecoder.strip(' ', 1, 1).decode(pieces(" a ", "  b")));
        assertEquals("ab", TokenDecoder.strip('x', 3, 2).decode(pieces("xxaxx", "xb")));
        assertEquals("", TokenDecoder.strip('x', 2, 2).decode(pieces("x", "xx")));
    }

    @Test
    public void byteFallback() {
        TokenDecoder d = TokenDecoder.byteFallback();
        assertEquals("a日b", d.decode(pieces("a", "<0xE6>", "<0x97>", "<0xA5>", "b")));
        assertEquals("a����b��c",
                d.decode(pieces("a", "<0xE6>", "<0x97>", "<0xA5>", "<0xFF>", "b", "<0x+A>", "<0xe6>", "c")));
        assertEquals(10, TokenDecoder.byteToken("<0x+A>"));
        assertEquals(-1, TokenDecoder.byteToken("<0x-1>"));
        assertEquals(-1, TokenDecoder.byteToken("<0xGG>"));
        assertEquals(-1, TokenDecoder.byteToken("<0x0A0>"));
    }

    @Test
    public void fuseAndSequence() {
        assertEquals(Collections.singletonList("ab"), TokenDecoder.fuse().decodeChain(pieces("a", "b")));
        TokenDecoder seq = TokenDecoder.sequence(Arrays.asList(TokenDecoder.replace("▁", " "),
                TokenDecoder.byteFallback(), TokenDecoder.fuse(), TokenDecoder.strip(' ', 1, 0)));
        assertEquals("hello! world", seq.decode(pieces("▁hello", "<0x21>", "▁world")));
        assertEquals("ab", TokenDecoder.identity().decode(pieces("a", "b")));
    }

    @Test
    public void huggingFaceJsonRoundTrips() {
        String json = "{\"type\":\"Sequence\",\"decoders\":["
                + "{\"type\":\"Replace\",\"pattern\":{\"String\":\"▁\"},\"content\":\" \"},"
                + "{\"type\":\"ByteFallback\"},{\"type\":\"Fuse\"},"
                + "{\"type\":\"Strip\",\"content\":\" \",\"start\":1,\"stop\":0},"
                + "{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true},"
                + "{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"first\",\"split\":false},"
                + "{\"type\":\"WordPiece\",\"prefix\":\"##\",\"cleanup\":true},"
                + "{\"type\":\"BPEDecoder\",\"suffix\":\"</w>\"}]}";
        TokenDecoder d = TokenDecoder.fromCanonicalJson(json);
        assertEquals(TokenDecoder.Kind.SEQUENCE, d.kind());
        assertEquals(8, d.children().size());
        TokenDecoder back = TokenDecoder.fromCanonicalJson(d.toCanonicalJson());
        assertEquals(d, back);
        assertEquals(d.hashCode(), back.hashCode());
        assertEquals(d.toCanonicalJson(), back.toCanonicalJson());
    }

    @Test
    public void llamaStyleSentencePieceDecoderFromJson() {
        TokenDecoder d = TokenDecoder.fromCanonicalJson("{\"type\":\"Sequence\",\"decoders\":["
                + "{\"type\":\"Replace\",\"pattern\":{\"String\":\"▁\"},\"content\":\" \"},"
                + "{\"type\":\"ByteFallback\"},{\"type\":\"Fuse\"},"
                + "{\"type\":\"Strip\",\"content\":\" \",\"start\":1,\"stop\":0}]}");
        assertEquals("hello world!", d.decode(pieces("▁hello", "▁world", "<0x21>")));
    }

    @Test
    public void legacyMetaspaceAddPrefixSpace() {
        assertEquals(TokenDecoder.metaspace(METASPACE, PrependScheme.NEVER, true), TokenDecoder.fromCanonicalJson(
                "{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"add_prefix_space\":false}"));
        assertEquals(TokenDecoder.metaspace(METASPACE, PrependScheme.ALWAYS, true), TokenDecoder.fromCanonicalJson(
                "{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"add_prefix_space\":true}"));
        assertEquals(TokenDecoder.metaspace(METASPACE, PrependScheme.NEVER, true), TokenDecoder.fromCanonicalJson(
                "{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"add_prefix_space\":false,"
                        + "\"prepend_scheme\":\"first\"}"));
        assertEquals(TokenDecoder.metaspace(METASPACE, PrependScheme.FIRST, true), TokenDecoder.fromCanonicalJson(
                "{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"add_prefix_space\":true,"
                        + "\"prepend_scheme\":\"first\"}"));
    }

    @Test
    public void ctcIsRejected() {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> TokenDecoder.fromCanonicalJson(
                "{\"type\":\"CTC\",\"pad_token\":\"<pad>\",\"word_delimiter_token\":\"|\",\"cleanup\":true}"));
        assertEquals("tokenizer decoder CTC is not supported", e.getMessage());
    }

    @Test
    public void jsonRejectsUnknownOrMalformedInput() {
        assertSame(TokenDecoder.identity(), TokenDecoder.fromJson(null, "test"));
        assertThrows(GgufFormatException.class, () -> TokenDecoder.fromCanonicalJson("{\"type\":\"Nope\"}"));
        assertThrows(GgufFormatException.class, () -> TokenDecoder.fromCanonicalJson("{\"type\":\"Fuse\",\"x\":1}"));
        assertThrows(GgufFormatException.class, () -> TokenDecoder.fromCanonicalJson(
                "{\"type\":\"Strip\",\"content\":\"ab\",\"start\":1,\"stop\":0}"));
        assertThrows(GgufFormatException.class, () -> TokenDecoder.fromCanonicalJson(
                "{\"type\":\"Strip\",\"content\":\" \",\"start\":-1,\"stop\":0}"));
        assertThrows(GgufFormatException.class, () -> TokenDecoder.fromCanonicalJson(
                "{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"sometimes\"}"));
        assertThrows(IllegalArgumentException.class, () -> TokenDecoder.strip(0xD800, 0, 0));
    }
}
