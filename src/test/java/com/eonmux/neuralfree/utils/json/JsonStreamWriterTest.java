package com.eonmux.neuralfree.utils.json;

import org.junit.Test;

import java.io.IOException;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** `JsonStreamWriter` writes the same bytes as `Json.write`; `Style.API` escapes U+2028/2029 and rejects non-finite. */
public class JsonStreamWriterTest {

    /** A parsed-document-shaped value: LinkedHashMap, ArrayList, Long, Double, String, Boolean, null. */
    private static Map<String, Object> map(Object... pairs) {
        LinkedHashMap<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) m.put((String) pairs[i], pairs[i + 1]);
        return m;
    }

    /** Writes {@code v} through a writer, using the typed overloads for scalars and begin/end for containers. */
    private static String viaWriter(Object v, Json.Style style) throws IOException {
        StringBuilder out = new StringBuilder();
        JsonStreamWriter w = new JsonStreamWriter(out, style);
        write(w, v);
        return out.toString();
    }

    @SuppressWarnings("unchecked")
    private static void write(JsonStreamWriter w, Object v) throws IOException {
        if (v == null) w.nullValue();
        else if (v instanceof String) w.value((String) v);
        else if (v instanceof Boolean) w.value((Boolean) v);
        else if (v instanceof Long) w.value((long) (Long) v);
        else if (v instanceof Double) w.value((double) (Double) v);
        else if (v instanceof BigInteger) w.value((BigInteger) v);
        else if (v instanceof List) {
            w.beginArray();
            for (Object item : (List<Object>) v) write(w, item);
            w.endArray();
        } else {
            w.beginObject();
            for (Map.Entry<String, Object> e : ((Map<String, Object>) v).entrySet()) {
                w.name(e.getKey());
                write(w, e.getValue());
            }
            w.endObject();
        }
    }

    @Test
    public void writesTheSameBytesAsWriteInEveryStyle() throws IOException {
        Map<String, Object> doc = map("a", 1L, "b", 2.5, "c", "\u00e9\"\\\t", "d", BigInteger.TEN,
                "e", Arrays.asList(true, false, null, -0.0, Long.MAX_VALUE, Long.MIN_VALUE),
                "f", map("g", Collections.emptyList(), "h", Collections.emptyMap()));
        for (Json.Style style : Arrays.asList(Json.Style.API, Json.Style.CANONICAL, Json.Style.PYTHON_DEFAULT)) {
            assertEquals(Json.toString(doc, style), viaWriter(doc, style));
        }
        assertEquals("{\"a\":1,\"b\":2.5,\"c\":\"\u00e9\\\"\\\\\\t\",\"d\":10,"
                + "\"e\":[true,false,null,-0.0,9223372036854775807,-9223372036854775808],"
                + "\"f\":{\"g\":[],\"h\":{}}}",
                Json.toString(doc, Json.Style.API));
    }

    @Test
    public void valueObjectDelegatesToWrite() throws IOException {
        Map<String, Object> nested = map("x", Arrays.asList(1L, "two", map("n", null), 2.5),
                "y", true, "z", new BigInteger("-123456789012345678901234567890"));
        StringBuilder out = new StringBuilder();
        JsonStreamWriter w = new JsonStreamWriter(out, Json.Style.API);
        w.beginObject();
        w.name("a");
        w.value(nested);
        w.name("b");
        w.value("string");
        w.name("c");
        w.value((Object) null);
        w.name("d");
        w.value(7L);
        w.endObject();
        assertEquals(map("a", nested, "b", "string", "c", null, "d", 7L),
                Json.parse(out.toString(), Json.Limits.UNTRUSTED));
        assertEquals(Json.toString(map("a", nested, "b", "string", "c", null, "d", 7L), Json.Style.API),
                out.toString());
    }

    @Test
    public void apiEscapesLineSeparators() {
        assertEquals("\"x\\u2028y\\u2029\"", Json.toString("x\u2028y\u2029", Json.Style.API));
        assertEquals("\"x\u2028y\u2029\"", Json.toString("x\u2028y\u2029", Json.Style.CANONICAL)); // unchanged: raw
        // PYTHON_DEFAULT escapes it, but through ensureAscii, not the new flag
        assertEquals("\"x\\u2028\"", Json.toString("x\u2028", Json.Style.PYTHON_DEFAULT));
        assertEquals("{\n  \"a\": \"x\\u2028y\\u2029\"\n}",
                Json.toString(map("a", "x\u2028y\u2029"), new Json.Style(false, "  ", ",", ": ", false, true, true)));
        assertTrue(Json.Style.API.escapeLineSeparators());
    }

    @Test
    public void apiRejectsNonFinite() {
        Map<String, Object> doc = map("a", 1L, "b", Double.NaN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Json.toString(doc, Json.Style.API));
        assertTrue(e.getMessage(), e.getMessage().contains("non-finite"));
        StringBuilder out = new StringBuilder();
        JsonStreamWriter w = new JsonStreamWriter(out, Json.Style.API);
        assertThrows(IllegalArgumentException.class, () -> w.value(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> w.value(Double.NEGATIVE_INFINITY));
        assertEquals("{\"a\":NaN}", Json.toString(map("a", Double.NaN), Json.Style.CANONICAL)); // constants keep it
        assertEquals("Infinity", Json.toString(Double.POSITIVE_INFINITY, Json.Style.PYTHON_DEFAULT));
    }

    @Test
    public void flagsOfTheStylesAreUnchangedAndExplicit() {
        assertEquals(true, Json.Style.API.rejectNonFinite());
        assertEquals(false, Json.Style.API.ensureAscii());
        assertEquals(",", Json.Style.API.itemSep());
        assertEquals(":", Json.Style.API.keySep());
        assertEquals(false, Json.Style.API.sortKeys());
        assertEquals(null, Json.Style.API.indent());
        for (Json.Style style : Arrays.asList(Json.Style.PYTHON_DEFAULT, Json.Style.CANONICAL)) {
            assertEquals(false, style.escapeLineSeparators());
            assertEquals(false, style.rejectNonFinite());
        }
    }

    // ------------------------------------------------------------------ nest misuse

    private static JsonStreamWriter writer() {
        return new JsonStreamWriter(new StringBuilder(), Json.Style.API);
    }

    @Test
    public void nestingMisuseThrowsIllegalStateException() throws IOException {
        assertThrows(IllegalStateException.class, () -> writer().endObject());
        assertThrows(IllegalStateException.class, () -> writer().endArray());
        assertThrows(IllegalStateException.class, () -> writer().name("k")); // name outside an object
        JsonStreamWriter noName = writer();
        noName.beginObject();
        assertThrows(IllegalStateException.class, () -> noName.value(1L)); // value where a name is due
        JsonStreamWriter noValue = writer();
        noValue.beginObject();
        noValue.name("k");
        assertThrows(IllegalStateException.class, () -> noValue.name("k2"));
        assertThrows(IllegalStateException.class, () -> noValue.endObject());
        noValue.beginArray(); // a container is the pending member's value, as writesTheSameBytesAsWrite nests them
        noValue.endArray();
        noValue.endObject();
        JsonStreamWriter containerNoName = writer();
        containerNoName.beginObject();
        assertThrows(IllegalStateException.class, () -> containerNoName.beginArray()); // name due
        JsonStreamWriter wrongClose = writer();
        wrongClose.beginObject();
        assertThrows(IllegalStateException.class, () -> wrongClose.endArray());
        wrongClose.endObject();
    }

    @Test
    public void emptyAndRepeatedTopLevelValuesAreLegal() throws IOException {
        StringBuilder out = new StringBuilder();
        JsonStreamWriter w = new JsonStreamWriter(out, Json.Style.API);
        w.beginObject();
        w.endObject();
        w.value(1L);
        w.value(2L);
        assertEquals("{}12", out.toString());
    }

    @Test
    public void chunksParsedBackReadTheSameValues() throws IOException {
        Map<String, Object> member = map("a", Arrays.asList(1L, 2.0, null), "b", true);
        Map<String, Object> big = map("k", Arrays.asList(member, member, member));
        StringBuilder out = new StringBuilder();
        JsonStreamWriter w = new JsonStreamWriter(out, Json.Style.API);
        write(w, big);
        assertEquals(Json.toString(big, Json.Style.API), out.toString());
        assertEquals(big, Json.parse(out.toString(), Json.Limits.UNTRUSTED));
    }
}
