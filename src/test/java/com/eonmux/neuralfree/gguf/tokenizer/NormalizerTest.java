package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Normalizer} against Hugging Face tokenizers {@code normalize_str} results, and the
 * {@link PrecompiledCharsmap} readers: llama.cpp's longest-match lookup used by
 * {@link UgmTokenizer} and Hugging Face's grapheme lookup.
 */
public class NormalizerTest {

    /** XCDA units: 512 nodes, keys "Ａ" (EF BC A1) to "A", "a" to "x" and "ab" to "y". */
    private static byte[] charsmap() {
        int[] units = new int[512];
        units[0] = 0x100 << 10;
        units[0x1EF] = (0x100 << 10) | 0xEF;
        units[0x053] = (0x100 << 10) | 0xBC;
        units[0x1F2] = (0x100 << 10) | (1 << 8) | 0xA1;
        units[0x0F2] = 0x80000000;
        units[0x161] = (0x100 << 10) | (1 << 8) | 0x61;
        units[0x061] = 0x80000000 | 2;
        units[0x003] = (0x100 << 10) | (1 << 8) | 0x62;
        units[0x103] = 0x80000000 | 4;
        byte[] replacements = "A\0x\0y\0".getBytes(StandardCharsets.US_ASCII);
        ByteBuffer bb = ByteBuffer.allocate(4 + units.length * 4 + replacements.length).order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(units.length * 4);
        for (int u : units) bb.putInt(u);
        bb.put(replacements);
        return bb.array();
    }

    private static byte[] utf8(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    @Test
    public void unicodeForms() {
        assertEquals("fi1", Normalizer.nfkc().normalize("ﬁ①"));
        assertEquals("é", Normalizer.nfd().normalize("é"));
        assertEquals("é", Normalizer.nfc().normalize("é"));
        assertEquals("fi", Normalizer.nfkd().normalize("ﬁ"));
    }

    @Test
    public void lowercaseIsFullAndContextFree() {
        assertEquals("àbç", Normalizer.lowercase().normalize("ÀBÇ"));
        assertEquals("i̇σ", Normalizer.lowercase().normalize("İΣ"));
    }

    @Test
    public void stripAccentsAfterNfd() {
        Normalizer n = Normalizer.sequence(Arrays.asList(Normalizer.nfd(), Normalizer.stripAccents()));
        assertEquals("cafe n", n.normalize("café ñ"));
    }

    @Test
    public void stripUsesUnicodeWhitespace() {
        assertEquals("x y", Normalizer.strip(true, true).normalize("  x y　\n"));
        assertEquals("a  ", Normalizer.strip(true, false).normalize("  a  "));
    }

    @Test
    public void replaceStringAndRegex() {
        assertEquals("a▁b▁▁c", Normalizer.replace(" ", "▁").normalize("a b  c"));
        assertEquals("a b", Normalizer.replaceRegex("\\s+", " ").normalize("a \t\n b"));
        assertEquals("-a-b-", Normalizer.replaceRegex("x*", "-").normalize("ab"));
        assertEquals("a.b", Normalizer.replace("..", ".").normalize("a..b"));
        assertEquals("", Normalizer.replaceRegex("x*", "-").normalize(""));
    }

    @Test
    public void prependSkipsEmptyText() {
        assertEquals("▁abc", Normalizer.prepend("▁").normalize("abc"));
        assertEquals("", Normalizer.prepend("▁").normalize(""));
    }

    @Test
    public void bertNormalizer() {
        assertEquals("hello   中  国  world",
                Normalizer.bert(true, true, null, true).normalize("Héllo\t\u0000 中国 WORLD�"));
        assertEquals("hello\tx", Normalizer.bert(false, false, null, true).normalize("Héllo\tX"));
        assertEquals("abc d e f", Normalizer.bert(true, false, false, false).normalize("a\u000Bb\u0085c\u2028d\te\u00A0f"));
        assertEquals("héllo", Normalizer.bert(true, true, false, true).normalize("Héllo"));
        assertTrue(Normalizer.isChineseChar(0x2B920));
        assertFalse(Normalizer.isChineseChar(0x2B820));
    }

    @Test
    public void nmtAndByteLevel() {
        assertEquals("a b c", Normalizer.nmt().normalize("a\u0001\tb​c"));
        assertEquals("aĠÃ©", Normalizer.byteLevel().normalize("a é"));
    }

    @Test
    public void identity() {
        assertTrue(Normalizer.identity().isIdentity());
        assertTrue(Normalizer.sequence(Collections.singletonList(Normalizer.identity())).isIdentity());
        assertFalse(Normalizer.nfc().isIdentity());
        assertEquals("Ab", Normalizer.identity().normalize("Ab"));
        assertSame(Normalizer.identity(), Normalizer.fromJson(null, "test"));
    }

    @Test
    public void huggingFaceJsonRoundTrips() {
        String json = "{\"type\":\"Sequence\",\"normalizers\":["
                + "{\"type\":\"Prepend\",\"prepend\":\"▁\"},"
                + "{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},\"content\":\"▁\"},"
                + "{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"\\\\s+\"},\"content\":\" \"},"
                + "{\"type\":\"BertNormalizer\",\"clean_text\":true,\"handle_chinese_chars\":true,"
                + "\"strip_accents\":null,\"lowercase\":true},"
                + "{\"type\":\"Strip\",\"strip_left\":true,\"strip_right\":false},"
                + "{\"type\":\"NFKC\"},{\"type\":\"Lowercase\"},{\"type\":\"StripAccents\"},{\"type\":\"Nmt\"},"
                + "{\"type\":\"ByteLevel\"},"
                + "{\"type\":\"Precompiled\",\"precompiled_charsmap\":\""
                + Base64.getEncoder().encodeToString(charsmap()) + "\"}]}";
        Normalizer n = Normalizer.fromCanonicalJson(json);
        assertEquals(Normalizer.Kind.SEQUENCE, n.kind());
        assertEquals(11, n.children().size());
        Normalizer back = Normalizer.fromCanonicalJson(n.toCanonicalJson());
        assertEquals(n, back);
        assertEquals(n.hashCode(), back.hashCode());
        assertEquals(n.toCanonicalJson(), back.toCanonicalJson());
    }

    @Test
    public void bertDefaultsApplyToAbsentMembers() {
        assertEquals(Normalizer.bert(true, true, null, true),
                Normalizer.fromCanonicalJson("{\"type\":\"BertNormalizer\"}"));
    }

    @Test
    public void jsonRejectsUnknownOrMalformedInput() {
        assertThrows(GgufFormatException.class, () -> Normalizer.fromCanonicalJson("{\"type\":\"Unknown\"}"));
        assertThrows(GgufFormatException.class, () -> Normalizer.fromCanonicalJson("{\"type\":\"NFC\",\"x\":1}"));
        assertThrows(GgufFormatException.class, () -> Normalizer.fromCanonicalJson(
                "{\"type\":\"Replace\",\"pattern\":{\"String\":\"a\",\"Regex\":\"b\"},\"content\":\"\"}"));
        assertThrows(GgufFormatException.class, () -> Normalizer.fromCanonicalJson(
                "{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"(\"},\"content\":\"\"}"));
        assertThrows(GgufFormatException.class, () -> Normalizer.fromCanonicalJson(
                "{\"type\":\"Precompiled\",\"precompiled_charsmap\":\"%%%\"}"));
        assertThrows(GgufFormatException.class, () -> Normalizer.fromCanonicalJson("not json"));
    }

    @Test
    public void charsmapLongestMatchForUgm() {
        PrecompiledCharsmap cm = new PrecompiledCharsmap(charsmap());
        int replacements = 4 + 512 * 4;
        int[] res = new int[3];
        byte[] blob = cm.replace(utf8("abc"), 0, res);
        assertArrayEquals(new int[] {replacements + 4, 1, 2}, res);
        assertEquals('y', blob[res[0]]);
        assertSame(blob, cm.replace(utf8("za"), 1, res));
        assertArrayEquals(new int[] {replacements + 2, 1, 1}, res);
        cm.replace(utf8("Ａ"), 0, res);
        assertArrayEquals(new int[] {replacements, 1, 3}, res);
        assertNull(cm.replace(utf8("b"), 0, res));
    }

    /**
     * The XCDA lookup {@link UgmTokenizer} used before the charsmap was extracted into
     * {@link PrecompiledCharsmap} (same algorithm, reporting a match as a boolean): the reference
     * the extracted lookup must reproduce.
     */
    private static final class ReferenceXcda {
        private final byte[] blob;
        private final int size;
        private final int replacementsOffset;
        private final int replacementsSize;

        ReferenceXcda(byte[] blob) {
            long bytes = (blob[0] & 0xFFL) | (blob[1] & 0xFFL) << 8 | (blob[2] & 0xFFL) << 16 | (blob[3] & 0xFFL) << 24;
            this.blob = blob;
            this.size = (int) (bytes / 4);
            this.replacementsOffset = 4 + (int) bytes;
            this.replacementsSize = blob.length - replacementsOffset;
        }

        private long node(int index) {
            if (index < 0 || index >= size) throw new IllegalStateException("Index out of array bounds in XCDA array");
            int o = 4 + index * 4;
            return (blob[o] & 0xFFL) | (blob[o + 1] & 0xFFL) << 8 | (blob[o + 2] & 0xFFL) << 16 | (blob[o + 3] & 0xFFL) << 24;
        }

        private int base(int index) {
            long p = node(index);
            return (int) ((p >>> 10) << ((p & (1L << 9)) >>> 6));
        }

        private long lcheck(int index) { return node(index) & ((1L << 31) | 0xFF); }
        private boolean leaf(int index) { return ((node(index) >>> 8) & 1) != 0; }
        private int value(int index) { return (int) (node(index) & ((1L << 31) - 1)); }

        boolean replace(byte[] input, int off, int[] res) {
            if (size == 0) return false;
            int longestLength = 0;
            int longestOffset = 0;
            int nodeIndex = base(0);
            for (int p = off; p < input.length; p++) {
                int c = input[p] & 0xFF;
                if (c == 0) break;
                nodeIndex ^= c;
                if (lcheck(nodeIndex) != c) break;
                boolean isLeaf = leaf(nodeIndex);
                nodeIndex ^= base(nodeIndex);
                if (isLeaf) {
                    longestLength = p - off + 1;
                    longestOffset = value(nodeIndex);
                }
            }
            if (longestLength == 0) return false;
            if (longestOffset >= replacementsSize) {
                throw new IllegalStateException("Index out of array bounds in precompiled charsmap");
            }
            int start = replacementsOffset + longestOffset;
            int maxLen = replacementsSize - longestOffset;
            int len = 0;
            while (len < maxLen && blob[start + len] != 0) len++;
            if (len == maxLen) throw new IllegalStateException("Unterminated string in precompiled charsmap");
            res[0] = start;
            res[1] = len;
            res[2] = longestLength;
            return true;
        }
    }

    @Test
    public void charsmapLookupIsIdenticalToThePreExtractionUgmLookup() {
        byte[] blob = charsmap();
        PrecompiledCharsmap cm = new PrecompiledCharsmap(blob);
        ReferenceXcda reference = new ReferenceXcda(blob);
        byte[] alphabet = {'a', 'b', 'c', 0, (byte) 0xEF, (byte) 0xBC, (byte) 0xA1, (byte) 0x80};
        int checked = 0;
        for (int len = 1; len <= 4; len++) {
            int[] digits = new int[len];
            while (true) {
                byte[] input = new byte[len];
                for (int i = 0; i < len; i++) input[i] = alphabet[digits[i]];
                for (int off = 0; off < len; off++) {
                    int[] expected = new int[3];
                    int[] actual = new int[3];
                    boolean found = reference.replace(input, off, expected);
                    byte[] out = cm.replace(input, off, actual);
                    assertEquals(found, out != null);
                    assertArrayEquals(expected, actual);
                    if (out != null) assertEquals(blob[expected[0]], out[actual[0]]);
                    checked++;
                }
                int k = len - 1;
                while (k >= 0 && ++digits[k] == alphabet.length) digits[k--] = 0;
                if (k < 0) break;
            }
        }
        assertEquals(8 + 2 * 64 + 3 * 512 + 4 * 4096, checked);
    }

    @Test
    public void charsmapShortestMatchAndGraphemesForHuggingFace() {
        PrecompiledCharsmap cm = new PrecompiledCharsmap(charsmap());
        assertEquals("x", cm.transform(utf8("ab")));
        assertEquals("A", cm.transform(utf8("Ａ")));
        assertNull(cm.transform(utf8("b")));
        assertEquals("xb", cm.normalize("ab"));
        assertEquals("zAy", cm.normalize("zＡ́y"));
        assertEquals("zÁ", Normalizer.precompiled(cm).normalize("zÁ"));
    }

    @Test
    public void precompiledNormalizerRequiresUtf8Replacements() {
        byte[] blob = charsmap();
        blob[4 + 512 * 4 + 2] = (byte) 0xFF;
        PrecompiledCharsmap cm = new PrecompiledCharsmap(blob);
        assertThrows(IllegalArgumentException.class, () -> Normalizer.precompiled(cm));
        assertThrows(GgufFormatException.class, () -> Normalizer.fromCanonicalJson(
                "{\"type\":\"Precompiled\",\"precompiled_charsmap\":\"" + Base64.getEncoder().encodeToString(blob) + "\"}"));
    }

    @Test
    public void charsmapValidatesItsHeader() {
        assertThrows(IllegalArgumentException.class, () -> new PrecompiledCharsmap(new byte[3]));
        byte[] tooShort = Arrays.copyOf(charsmap(), 100);
        assertThrows(IllegalArgumentException.class, () -> new PrecompiledCharsmap(tooShort));
        byte[] unterminated = Arrays.copyOf(charsmap(), 4 + 512 * 4 + 5);
        PrecompiledCharsmap cm = new PrecompiledCharsmap(unterminated);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> cm.replace(utf8("ab"), 0, new int[3]));
        assertEquals("Unterminated string in precompiled charsmap", e.getMessage());
        assertEquals("x", cm.transform(utf8("ab")));
    }
}
