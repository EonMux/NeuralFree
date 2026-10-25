package com.eonmux.neuralfree.utils.json;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** Parsing: value model, strict grammar, caps, duplicate keys, surrogates and file decoding. */
public class JsonTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @SuppressWarnings("unchecked")
    private static Map<String, Object> obj(String text, Json.Limits limits) {
        return (Map<String, Object>) Json.parse(text, limits);
    }

    private static JsonException fails(String text, Json.Limits limits) {
        return assertThrows(JsonException.class, () -> Json.parse(text, limits));
    }

    private static String nested(int depth) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < depth; i++) sb.append('[');
        for (int i = 0; i < depth; i++) sb.append(']');
        return sb.toString();
    }

    @Test
    public void parsesTheValueModel() {
        Map<String, Object> m = obj("{\"a\": 1, \"b\": 2.5, \"c\": \"x\", \"d\": true, \"e\": null, \"f\": [1, 2], \"g\": {}}",
                Json.Limits.STRICT);
        assertTrue(m instanceof LinkedHashMap);
        assertEquals(Arrays.asList("a", "b", "c", "d", "e", "f", "g"), new ArrayList<>(m.keySet()));
        assertEquals(1L, m.get("a"));
        assertEquals(2.5, m.get("b"));
        assertEquals("x", m.get("c"));
        assertEquals(Boolean.TRUE, m.get("d"));
        assertTrue(m.containsKey("e"));
        assertNull(m.get("e"));
        assertTrue(m.get("f") instanceof ArrayList);
        assertEquals(Arrays.asList(1L, 2L), m.get("f"));
        assertTrue(m.get("g") instanceof LinkedHashMap);
    }

    @Test
    public void integersFittingSixtyFourBitsAreLongsAndLongerOnesBigIntegers() {
        assertEquals(Long.MAX_VALUE, Json.parse("9223372036854775807"));
        assertEquals(Long.MIN_VALUE, Json.parse("-9223372036854775808"));
        assertEquals(new BigInteger("9223372036854775808"), Json.parse("9223372036854775808"));
        assertEquals(new BigInteger("1000000000000000019884624838656"), Json.parse("1000000000000000019884624838656"));
        assertEquals(0L, Json.parse("-0"));
    }

    @Test
    public void fractionsExponentsAndPythonConstantsAreDoubles() {
        assertEquals(1000.0, Json.parse("1e3"));
        assertEquals(1.0, Json.parse("1.0"));
        assertEquals(-0.25, Json.parse("-2.5E-1"));
        assertTrue(Double.isNaN((Double) Json.parse("NaN")));
        assertEquals(Double.POSITIVE_INFINITY, Json.parse("Infinity"));
        assertEquals(Double.NEGATIVE_INFINITY, Json.parse("-Infinity"));
        assertEquals(Arrays.asList(Double.NEGATIVE_INFINITY, 1L), Json.parse("[-Infinity, 1]"));
    }

    @Test
    public void numbersFollowTheStrictGrammar() {
        assertEquals("extra data at position 1", fails("01", Json.Limits.STRICT).getMessage());
        assertEquals("invalid number '1.' at position 0", fails("1.", Json.Limits.STRICT).getMessage());
        assertEquals("invalid number '-x' at position 0", fails("-x", Json.Limits.STRICT).getMessage());
        assertEquals("invalid number '1e+' at position 0", fails("1e+", Json.Limits.STRICT).getMessage());
        assertEquals("unexpected character '.' at position 0", fails(".5", Json.Limits.STRICT).getMessage());
        assertEquals("unexpected character '+' at position 0", fails("+1", Json.Limits.STRICT).getMessage());
    }

    @Test
    public void integerLiteralsAreBoundedLikePython() {
        StringBuilder ok = new StringBuilder();
        for (int i = 0; i < Json.MAX_INTEGER_DIGITS; i++) ok.append('7');
        assertTrue(Json.parse(ok.toString()) instanceof BigInteger);
        JsonException e = fails(ok.append('7').toString(), Json.Limits.STRICT);
        assertTrue(e.getMessage(), e.getMessage().contains(String.valueOf(Json.MAX_INTEGER_DIGITS)));
        assertEquals(0, e.position());
    }

    @Test
    public void nestingDepthIsCapped() {
        Json.parse(nested(Json.Limits.DEFAULT_MAX_DEPTH));
        JsonException e = fails(nested(Json.Limits.DEFAULT_MAX_DEPTH + 1), Json.Limits.STRICT);
        assertEquals("nesting depth exceeds maxDepth 512 at position 512", e.getMessage());
        assertEquals(512, e.position());

        Json.Limits three = Json.Limits.STRICT.withMaxDepth(3);
        Json.parse("[{\"a\": [1]}]", three);
        assertEquals("nesting depth exceeds maxDepth 3 at position 8", fails("[{\"a\": [[1]]}]", three).getMessage());
    }

    @Test
    public void valueCountIsCapped() {
        Json.Limits three = Json.Limits.STRICT.withMaxValues(3);
        Json.parse("[1, 2]", three);
        Json.parse("{\"a\": 1, \"b\": 2}", three);
        assertEquals("document exceeds maxValues 3 at position 7", fails("[1, 2, 3]", three).getMessage());
    }

    @Test
    public void stringLengthIsCappedForValuesAndKeys() {
        Json.Limits three = Json.Limits.STRICT.withMaxStringChars(3);
        assertEquals("abc", Json.parse("\"abc\"", three));
        assertEquals("ab\n", Json.parse("\"ab\\n\"", three));
        assertTrue(fails("\"abcd\"", three).getMessage().startsWith("string exceeds maxStringChars 3"));
        assertTrue(fails("\"ab\\nc\"", three).getMessage().startsWith("string exceeds maxStringChars 3"));
        assertTrue(fails("{\"abcd\": 1}", three).getMessage().startsWith("string exceeds maxStringChars 3"));
    }

    @Test
    public void duplicateKeysAreRejectedUnderStrict() {
        JsonException e = fails("{\"a\":1,\"a\":2}", Json.Limits.STRICT);
        assertEquals("duplicate key 'a' at a at position 7", e.getMessage());
        assertEquals(7, e.position());
        assertEquals("duplicate key 'k' at x[1].k at position 21",
                fails("{\"x\":[{\"k\":1},{\"k\":1,\"k\":2}]}", Json.Limits.STRICT).getMessage());
    }

    @Test
    public void duplicateKeysKeepTheLastValueAtTheFirstPositionUnderLenientAndPython() {
        for (Json.Limits limits : Arrays.asList(Json.Limits.LENIENT, Json.Limits.PYTHON)) {
            Map<String, Object> m = obj("{\"a\": 1, \"b\": 0, \"a\": 2}", limits);
            assertEquals(Arrays.asList("a", "b"), new ArrayList<>(m.keySet()));
            assertEquals(2L, m.get("a"));
        }
        assertEquals(Json.DuplicateKeys.REJECT, Json.Limits.STRICT.duplicateKeys());
        assertEquals(Json.DuplicateKeys.LAST_WINS_LOGGED, Json.Limits.LENIENT.duplicateKeys());
        assertEquals(Json.DuplicateKeys.LAST_WINS, Json.Limits.PYTHON.duplicateKeys());
    }

    @Test
    public void surrogatePairsDecodeAndLoneSurrogatesFail() {
        assertEquals("\uD83D\uDE00", Json.parse("\"\\ud83d\\ude00\""));
        assertEquals("\uD83D\uDE00", Json.parse("\"\\uD83D\\uDE00\""));
        assertEquals("x\uD83D\uDE00y", Json.parse("\"x\uD83D\uDE00y\""));
        assertEquals("\u00e9/", Json.parse("\"\\u00E9\\/\""));
        assertEquals("lone surrogate '\\ud83d' at position 1", fails("\"\\ud83d\"", Json.Limits.STRICT).getMessage());
        assertEquals("lone surrogate '\\ude00' at position 1", fails("\"\\ude00\"", Json.Limits.STRICT).getMessage());
        assertEquals("lone surrogate '\\ud83d' at position 1", fails("\"\\ud83d\\u0041\"", Json.Limits.STRICT).getMessage());
        assertEquals("lone surrogate in string at position 2", fails("\"a\uD83Dx\"", Json.Limits.STRICT).getMessage());
        assertEquals("lone surrogate in string at position 1", fails("\"\uDE00\"", Json.Limits.STRICT).getMessage());
    }

    @Test
    public void stringSyntaxErrorsCarryTheirPosition() {
        assertEquals("invalid control character in string at position 2", fails("\"a\nb\"", Json.Limits.STRICT).getMessage());
        assertEquals("invalid escape '\\x' at position 4", fails("\"a\\x\"", Json.Limits.STRICT).getMessage());
        assertEquals("invalid \\u escape at position 4", fails("\"a\\u12g4\"", Json.Limits.STRICT).getMessage());
        assertEquals("truncated \\u escape at position 4", fails("\"a\\u12", Json.Limits.STRICT).getMessage());
        assertEquals("unterminated string at position 4", fails("\"abc", Json.Limits.STRICT).getMessage());
    }

    @Test
    public void structuralErrorsAndTrailingDataCarryTheirPosition() {
        assertEquals("extra data at position 3", fails("{} x", Json.Limits.STRICT).getMessage());
        assertEquals("extra data at position 4", fails("[1] [2]", Json.Limits.STRICT).getMessage());
        assertEquals("unexpected character ']' at position 3", fails("[1,]", Json.Limits.STRICT).getMessage());
        assertEquals("expected property name at position 7", fails("{\"a\":1,}", Json.Limits.STRICT).getMessage());
        assertEquals("expected ':' at position 5", fails("{\"a\" 1}", Json.Limits.STRICT).getMessage());
        assertEquals("expected ',' or '}' at position 7", fails("{\"a\":1 \"b\":2}", Json.Limits.STRICT).getMessage());
        assertEquals("expected ',' or ']' at position 3", fails("[1 2]", Json.Limits.STRICT).getMessage());
        assertEquals("unexpected end of input at position 2", fails("  ", Json.Limits.STRICT).getMessage());
        assertEquals(Arrays.asList(1L, 2L), Json.parse(" \t\r\n[ 1 ,\n2 ] "));
    }

    @Test
    public void limitsRejectInvalidBounds() {
        assertThrows(IllegalArgumentException.class, () -> new Json.Limits(0, 1, 1, Json.DuplicateKeys.REJECT));
        assertThrows(IllegalArgumentException.class, () -> new Json.Limits(1, 0, 1, Json.DuplicateKeys.REJECT));
        assertThrows(IllegalArgumentException.class, () -> new Json.Limits(1, 1, -1, Json.DuplicateKeys.REJECT));
        assertThrows(NullPointerException.class, () -> new Json.Limits(1, 1, 1, null));
        assertEquals(512, Json.Limits.STRICT.maxDepth());
        assertEquals(50_000_000L, Json.Limits.STRICT.maxValues());
        assertEquals(268_435_456, Json.Limits.STRICT.maxStringChars());
    }

    // ------------------------------------------------------------------ files

    private Path write(byte[] bytes) throws IOException {
        Path p = tmp.newFile().toPath();
        Files.write(p, bytes);
        return p;
    }

    private static byte[] concat(byte[] a, byte[] b) {
        byte[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    @Test
    public void filesMayStartWithAByteOrderMark() throws IOException {
        byte[] bom = {(byte) 0xEF, (byte) 0xBB, (byte) 0xBF};
        Path p = write(concat(bom, "{\"a\": \"\u00e9\"}".getBytes(StandardCharsets.UTF_8)));
        Map<?, ?> m = (Map<?, ?>) Json.parse(p, 1024, Json.Limits.STRICT);
        assertEquals("\u00e9", m.get("a"));
    }

    @Test
    public void invalidUtf8IsRejectedWithItsByteOffset() throws IOException {
        Path p = write(new byte[]{'{', '"', 'a', '"', ':', '"', (byte) 0xC3, (byte) 0xA9, (byte) 0xFF, '"', '}'});
        JsonException e = assertThrows(JsonException.class, () -> Json.parse(p, 1024, Json.Limits.STRICT));
        assertEquals(p + ": invalid UTF-8 byte sequence at position 8", e.getMessage());
        assertEquals(8, e.position());

        Path surrogate = write(new byte[]{'"', (byte) 0xED, (byte) 0xA0, (byte) 0x80, '"'});
        assertEquals(1, assertThrows(JsonException.class, () -> Json.parse(surrogate, 1024, Json.Limits.STRICT)).position());
    }

    @Test
    public void fileSyntaxErrorsNameTheFile() throws IOException {
        Path p = write("{\"a\": 1} x".getBytes(StandardCharsets.UTF_8));
        JsonException e = assertThrows(JsonException.class, () -> Json.parse(p, 1024, Json.Limits.STRICT));
        assertEquals(p + ": extra data at position 9", e.getMessage());
        assertEquals(9, e.position());
    }

    @Test
    public void fileSizeIsCapped() throws IOException {
        Path p = write("[1, 2, 3]".getBytes(StandardCharsets.UTF_8));
        assertEquals(Arrays.asList(1L, 2L, 3L), Json.parse(p, 9, Json.Limits.STRICT));
        JsonException e = assertThrows(JsonException.class, () -> Json.parse(p, 8, Json.Limits.STRICT));
        assertTrue(e.getMessage(), e.getMessage().startsWith(p + ": size 9 bytes exceeds the limit of 8 bytes"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse(p, -1, Json.Limits.STRICT));
    }
}
