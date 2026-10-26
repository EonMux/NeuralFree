package com.eonmux.neuralfree.utils.json;

import org.junit.Test;

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** `Json.Limits.UNTRUSTED`, the new `Limits` fields and the `parse(CharSequence)` / `parse(byte[])` overloads:
 *  every refusal precedes the allocation or recursion it bounds. */
public class JsonLimitsTest {

    private static JsonException fails(String text, Json.Limits limits) {
        return assertThrows(JsonException.class, () -> Json.parse(text, limits));
    }

    @Test
    public void untrustedCapsAreAsSpecified() {
        Json.Limits u = Json.Limits.UNTRUSTED;
        assertEquals(128, u.maxDepth());
        assertEquals(16_777_216L, u.maxBytes());
        assertEquals(4_194_304L, u.maxValues());
        assertEquals(8_388_608, u.maxStringChars());
        assertEquals(1_048_576, u.maxMembers());
        assertEquals(Json.DuplicateKeys.REJECT, u.duplicateKeys());
        assertEquals(false, u.allowNonFinite());
    }

    @Test
    public void unboundedFieldsKeepTheConstantsBehaviourUnchanged() {
        for (Json.Limits limits : Arrays.asList(Json.Limits.STRICT, Json.Limits.LENIENT, Json.Limits.PYTHON)) {
            assertEquals(Long.MAX_VALUE, limits.maxBytes());
            assertEquals(Integer.MAX_VALUE, limits.maxMembers());
            assertEquals(true, limits.allowNonFinite());
        }
        assertEquals(512, Json.Limits.STRICT.maxDepth());
        assertEquals(50_000_000L, Json.Limits.STRICT.maxValues());
        assertEquals(268_435_456, Json.Limits.STRICT.maxStringChars());
        assertEquals(Json.DuplicateKeys.REJECT, Json.Limits.STRICT.duplicateKeys());
        assertEquals(Json.DuplicateKeys.LAST_WINS_LOGGED, Json.Limits.LENIENT.duplicateKeys());
        assertEquals(Json.DuplicateKeys.LAST_WINS, Json.Limits.PYTHON.duplicateKeys());
    }

    @Test
    public void untrustedAdmitsItsOwnCaps() {
        StringBuilder nested = new StringBuilder();
        for (int i = 0; i < 128; i++) nested.append('[');
        for (int i = 0; i < 128; i++) nested.append(']');
        assertTrue(Json.parse(nested.toString(), Json.Limits.UNTRUSTED) instanceof List);
        assertEquals(Arrays.asList(1L, 2L), Json.parse(" \t[ 1 ,\n2 ] ", Json.Limits.UNTRUSTED));
        assertEquals(2.5, Json.parse(new StringBuilder("2.5"), Json.Limits.UNTRUSTED));
        assertEquals(2.5, Json.parse("2.5", Json.Limits.STRICT)); // parse(String, Limits) delegates
    }

    @Test
    public void boundsAreValidated() {
        assertThrows(IllegalArgumentException.class,
                () -> new Json.Limits(1, 1, 1, Json.DuplicateKeys.REJECT, -1L, 1, true));
        assertThrows(IllegalArgumentException.class,
                () -> new Json.Limits(1, 1, 1, Json.DuplicateKeys.REJECT, 1L, -1, true));
        Json.Limits u = Json.Limits.UNTRUSTED;
        assertEquals(1024L, u.withMaxBytes(1024).maxBytes());
        assertEquals(16, u.withMaxMembers(16).maxMembers());
        assertEquals(128, u.withMaxMembers(16).maxDepth());
        assertEquals(true, u.withAllowNonFinite(true).allowNonFinite());
        assertEquals(false, Json.Limits.STRICT.withAllowNonFinite(false).allowNonFinite());
        assertEquals(Json.DuplicateKeys.REJECT, u.withMaxBytes(1).duplicateKeys());
    }

    @Test
    public void depth129IsRefusedBeforeRecursion() {
        StringBuilder nested = new StringBuilder();
        for (int i = 0; i < 129; i++) nested.append('[');
        for (int i = 0; i < 129; i++) nested.append(']');
        JsonException e = fails(nested.toString(), Json.Limits.UNTRUSTED);
        assertEquals("nesting depth exceeds maxDepth 128 at position 128", e.getMessage());
        assertEquals(128, e.position());
    }

    @Test
    public void byteInputIsRefusedBeforeDecoding() {
        byte[] tooBig = new byte[(int) Json.Limits.UNTRUSTED.maxBytes() + 1];
        Arrays.fill(tooBig, (byte) 0xFF); // invalid UTF-8, so a refusal after the decode would report it instead
        JsonException e = assertThrows(JsonException.class,
                () -> Json.parse(tooBig, 0, tooBig.length, Json.Limits.UNTRUSTED));
        assertEquals("input length 16777217 bytes exceeds maxBytes 16777216", e.getMessage());

        byte[] small = new byte[3];
        JsonException bounded = assertThrows(JsonException.class,
                () -> Json.parse(small, 0, small.length, Json.Limits.UNTRUSTED.withMaxBytes(2)));
        assertEquals("input length 3 bytes exceeds maxBytes 2", bounded.getMessage());
    }

    @Test
    public void charSequenceOverloadBoundsItsUtf8Length() {
        JsonException e = fails("\u00e9\u00e9\u00e9", Json.Limits.UNTRUSTED.withMaxBytes(1)); // 6 UTF-8 bytes, 3 chars
        assertEquals("input length 6 bytes exceeds maxBytes 1", e.getMessage());
        StringBuilder text = new StringBuilder("\u00e9\u00e9\u00e9");
        assertEquals("input length 6 bytes exceeds maxBytes 1",
                assertThrows(JsonException.class, () -> Json.parse(text, Json.Limits.UNTRUSTED.withMaxBytes(1)))
                        .getMessage());
    }

    @Test
    public void exactByteBoundAdmitsItsDocument() {
        assertEquals(Arrays.asList(1L, 2L), Json.parse("[1, 2]", Json.Limits.UNTRUSTED.withMaxBytes(6)));
        JsonException e = fails("[1, 2]", Json.Limits.UNTRUSTED.withMaxBytes(5));
        assertEquals("input length 6 bytes exceeds maxBytes 5", e.getMessage());
    }

    @Test
    public void memberCountIsCappedPerObject() {
        Json.Limits three = Json.Limits.UNTRUSTED.withMaxMembers(3);
        assertEquals(Arrays.asList(1L, 2L, 3L), Json.parse("[1, 2, 3]", three));
        JsonException e = fails("{\"a\":1,\"b\":2,\"c\":3,\"d\":4}", three);
        assertEquals("object member exceeds maxMembers 3 at position 19", e.getMessage());
        assertEquals(19, e.position());
    }

    @Test
    public void memberCountIsCappedPerArray() {
        Json.Limits three = Json.Limits.UNTRUSTED.withMaxMembers(3);
        assertEquals(Arrays.asList(1L, 2L, 3L), Json.parse("[1, 2, 3]", three));
        JsonException e = fails("[1, 2, 3, 4]", three);
        assertEquals("array member exceeds maxMembers 3 at position 10", e.getMessage());
        assertEquals(10, e.position());
        Json.Limits one = Json.Limits.UNTRUSTED.withMaxMembers(1); // each container has its own member budget
        JsonException nested = fails("{\"a\":[1, 2]}", one);
        assertEquals("array member exceeds maxMembers 1 at position 9", nested.getMessage());
        assertEquals(9, nested.position());
        Object root = Json.parse("{\"a\": [1]}", one);
        assertEquals("[1]", Json.toString(((java.util.Map<?, ?>) root).get("a"), Json.Style.CANONICAL));
    }

    @Test
    public void memberCountIsCappedAtTheUntrustedBound() {
        StringBuilder members = new StringBuilder("[");
        for (int i = 0; i < Json.Limits.UNTRUSTED.maxMembers(); i++) {
            if (i > 0) members.append(',');
            members.append('1');
        }
        members.append(']');
        Json.parse(members.toString(), Json.Limits.UNTRUSTED);
        JsonException e = fails(members.substring(0, members.length() - 1) + ",1]", Json.Limits.UNTRUSTED);
        assertTrue(e.getMessage(), e.getMessage().startsWith("array member exceeds maxMembers 1048576"));
    }

    @Test
    public void nonFiniteLiteralsAreRejectedUnderUntrusted() {
        assertEquals("non-finite literal 'NaN' is not allowed at position 0",
                fails("NaN", Json.Limits.UNTRUSTED).getMessage());
        assertEquals("non-finite literal '-Infinity' is not allowed at position 1",
                fails("[-Infinity, 1]", Json.Limits.UNTRUSTED).getMessage());
        assertEquals("non-finite literal 'Infinity' is not allowed at position 0",
                fails("Infinity ", Json.Limits.UNTRUSTED).getMessage());
        Json.Limits allowing = Json.Limits.UNTRUSTED.withAllowNonFinite(true);
        List<Object> mixed = new ArrayList<>();
        mixed.add(Double.NEGATIVE_INFINITY);
        mixed.add(1L);
        assertEquals(mixed, Json.parse("[-Infinity, 1]", allowing));
        assertTrue(Double.isNaN((Double) Json.parse("NaN", allowing)));
        assertTrue(Double.isNaN((Double) Json.parse("NaN", Json.Limits.STRICT)));
    }

    @Test
    public void duplicateKeysAreRejectedUnderUntrusted() {
        JsonException e = fails("{\"a\":1,\"a\":2}", Json.Limits.UNTRUSTED);
        assertEquals("duplicate key 'a' at a at position 7", e.getMessage());
        assertEquals(7, e.position());
    }

    @Test
    public void valueCountIsCappedThroughTheNewOverload() {
        Json.Limits three = Json.Limits.UNTRUSTED.withMaxValues(3);
        assertEquals(Arrays.asList(1L, 2L), Json.parse("[1, 2]", three));
        JsonException e = fails("[1, 2, 3]", three);
        assertEquals("document exceeds maxValues 3 at position 7", e.getMessage());
        assertEquals(7, e.position());
        assertEquals(new BigInteger("1000000000000000019884624838656"),
                Json.parse("1000000000000000019884624838656", Json.Limits.UNTRUSTED));
    }

    @Test
    public void stringCharsAreCappedAtTheUntrustedBound() {
        char[] chars = new char[Json.Limits.UNTRUSTED.maxStringChars() + 1];
        Arrays.fill(chars, 'a');
        JsonException e = fails("\"" + new String(chars) + "\"", Json.Limits.UNTRUSTED);
        assertEquals("string exceeds maxStringChars 8388608 at position 8388610", e.getMessage());
        assertEquals(8_388_610, e.position());
    }

    @Test
    public void stringCapIsRefusedBeforeTheUnescapedValueIsBuilt() {
        Json.Limits three = Json.Limits.UNTRUSTED.withMaxStringChars(3);
        JsonException e = fails("\"\\n\\n\\n\\na\"", three);
        assertTrue(e.getMessage(), e.getMessage().startsWith("string exceeds maxStringChars 3"));
        assertEquals("ab\n", Json.parse("\"ab\\n\"", three));
        assertTrue(fails("{\"abcd\": 1}", three).getMessage().startsWith("string exceeds maxStringChars 3"));
    }

    // ------------------------------------------------------------------ byte arrays

    @Test
    public void byteArraysDecodeStrictUtf8() {
        Json.Limits u = Json.Limits.UNTRUSTED;
        byte[] text = "[\"\u00e9\"]".getBytes(StandardCharsets.UTF_8);
        assertEquals(Arrays.asList("\u00e9"), Json.parse(text, 0, text.length, u));

        JsonException badUtf8 = assertThrows(JsonException.class,
                () -> Json.parse(new byte[]{'"', (byte) 0xFF, '"'}, 0, 3, u));
        assertEquals("invalid UTF-8 byte sequence at position 1", badUtf8.getMessage());
        assertEquals(1, badUtf8.position());

        byte[] surrogate = {'"', (byte) 0xED, (byte) 0xA0, (byte) 0x80, '"'};
        assertEquals(1, assertThrows(JsonException.class, () -> Json.parse(surrogate, 0, surrogate.length, u)).position());

        byte[] big = "1000000000000000019884624838656".getBytes(StandardCharsets.UTF_8);
        assertEquals(new BigInteger("1000000000000000019884624838656"), Json.parse(big, 0, big.length, u));
    }

    @Test
    public void byteArraysParseTheirWindowAndBoundsAreExplicit() {
        byte[] bytes = "prefix [1, 2] suffix".getBytes(StandardCharsets.UTF_8);
        assertEquals(Arrays.asList(1L, 2L), Json.parse(bytes, 7, 6, Json.Limits.UNTRUSTED));
        assertEquals("extra data at position 4",
                assertThrows(JsonException.class,
                        () -> Json.parse("[1] x".getBytes(StandardCharsets.UTF_8), 0, 5, Json.Limits.UNTRUSTED))
                        .getMessage());
        byte[] bom = {(byte) 0xEF, (byte) 0xBB, (byte) 0xBF, '[', '1', ']'};
        assertEquals(Arrays.asList(1L), Json.parse(bom, 0, bom.length, Json.Limits.UNTRUSTED));
        byte[] doc = "[1]".getBytes(StandardCharsets.UTF_8);
        assertThrows(IllegalArgumentException.class, () -> Json.parse(doc, -1, 3, Json.Limits.UNTRUSTED));
        assertThrows(IllegalArgumentException.class, () -> Json.parse(doc, 0, 4, Json.Limits.UNTRUSTED));
        assertThrows(IllegalArgumentException.class, () -> Json.parse(null, 0, 3, Json.Limits.UNTRUSTED));
        assertThrows(NullPointerException.class, () -> Json.parse(doc, 0, 3, null));
    }
}
