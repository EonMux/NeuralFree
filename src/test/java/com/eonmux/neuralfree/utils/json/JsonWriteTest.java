package com.eonmux.neuralfree.utils.json;

import org.junit.Test;

import java.io.IOException;
import java.io.StringWriter;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** Writing: styles, escaping, key handling, float repr and failure cases. */
public class JsonWriteTest {

    private static Map<Object, Object> map(Object... kv) {
        Map<Object, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put(kv[i], kv[i + 1]);
        return m;
    }

    @Test
    public void canonicalSortsNestedKeysCompactlyAndKeepsNonAscii() {
        Object doc = map("b", map("d", 1L, "c", Arrays.asList(true, null)), "a", "\u00e9");
        assertEquals("{\"a\":\"\u00e9\",\"b\":{\"c\":[true,null],\"d\":1}}", Json.toString(doc, Json.Style.CANONICAL));
        assertEquals(Json.toString(doc, Json.Style.CANONICAL),
                Json.toString(Json.parse("{\"b\": {\"c\": [true, null], \"d\": 1}, \"a\": \"\\u00e9\"}"), Json.Style.CANONICAL));
    }

    @Test
    public void canonicalOrdersKeysByCodePoint() {
        Object doc = map("\uD83D\uDE00", 1L, "\uFFFD", 2L, "a", 3L);
        assertEquals("{\"a\":3,\"\uFFFD\":2,\"\uD83D\uDE00\":1}", Json.toString(doc, Json.Style.CANONICAL));
    }

    @Test
    public void pythonDefaultMatchesJsonDumps() {
        Object doc = map("a", Arrays.asList(1L, 2.5, "x\ny"), "b", "\u00e9\uD83D\uDE00", "c", map());
        assertEquals("{\"a\": [1, 2.5, \"x\\ny\"], \"b\": \"\\u00e9\\ud83d\\ude00\", \"c\": {}}",
                Json.toString(doc, Json.Style.PYTHON_DEFAULT));
        assertEquals("\"\\\"\\\\\\b\\f\\n\\r\\t\\u0001\\u007f~\"",
                Json.toString("\"\\\b\f\n\r\t\u0001\u007f~", Json.Style.PYTHON_DEFAULT));
        assertEquals("\"\u007f\"", Json.toString("\u007f", Json.Style.CANONICAL));
    }

    @Test
    public void indentWritesOneMemberPerLine() {
        Json.Style style = new Json.Style(false, "  ", ",", ": ", false);
        assertEquals("{\n  \"a\": [\n    1,\n    2\n  ],\n  \"b\": []\n}",
                Json.toString(map("a", Arrays.asList(1L, 2L), "b", new ArrayList<>()), style));
    }

    @Test
    public void numbersOfEveryAcceptedType() {
        List<Object> nums = Arrays.asList(1, (short) 2, (byte) 3, 4L, new BigInteger("1000000000000000019884624838656"),
                0.5f, 1e-05, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, -0.0);
        assertEquals("[1,2,3,4,1000000000000000019884624838656,0.5,1e-05,NaN,Infinity,-Infinity,-0.0]",
                Json.toString(nums, Json.Style.CANONICAL));
    }

    @Test
    public void floatReprMatchesPython() {
        assertEquals("1e-05", Json.floatRepr(1e-05));
        assertEquals("0.0001", Json.floatRepr(1e-04));
        assertEquals("1.0", Json.floatRepr(1.0));
        assertEquals("-2.5", Json.floatRepr(-2.5));
        assertEquals("0.1", Json.floatRepr(0.1));
        assertEquals("0.30000000000000004", Json.floatRepr(0.1 + 0.2));
        assertEquals("123.456", Json.floatRepr(123.456));
        assertEquals("1234567890123456.0", Json.floatRepr(1234567890123456.0));
        assertEquals("1e+16", Json.floatRepr(1e16));
        assertEquals("2e+23", Json.floatRepr(2e23));
        assertEquals("1.7976931348623157e+308", Json.floatRepr(Double.MAX_VALUE));
        assertEquals("5e-324", Json.floatRepr(Double.MIN_VALUE));
        assertEquals("7.120236347223045e-307", Json.floatRepr(Math.scalb(1.0, -1017)));
        assertEquals("7.291122019556398e-304", Json.floatRepr(Math.scalb(1.0, -1007)));
        assertEquals("inf", Json.floatRepr(Double.POSITIVE_INFINITY));
        assertEquals("-inf", Json.floatRepr(Double.NEGATIVE_INFINITY));
        assertEquals("nan", Json.floatRepr(Double.NaN));
        assertEquals("0.0", Json.floatRepr(0.0));
        assertEquals("-0.0", Json.floatRepr(-0.0));
    }

    @Test
    public void keysAreCoercedAsPythonDoes() {
        Map<Object, Object> m = map(1L, "a", true, "b", null, "c", 1.5, "d", 7, "e");
        assertEquals("{\"1\":\"a\",\"true\":\"b\",\"null\":\"c\",\"1.5\":\"d\",\"7\":\"e\"}",
                Json.toString(m, new Json.Style(false, null, ",", ":", false)));
        assertEquals("{\"1\":\"a\",\"2\":\"b\",\"10\":\"c\"}",
                Json.toString(map(10L, "c", 2L, "b", 1L, "a"), Json.Style.CANONICAL));
        JsonException mixed = assertThrows(JsonException.class,
                () -> Json.toString(map("a", 1L, 2L, 2L), Json.Style.CANONICAL));
        assertTrue(mixed.getMessage(), mixed.getMessage().equals("cannot order keys of types String and Long")
                || mixed.getMessage().equals("cannot order keys of types Long and String"));
        JsonException badKey = assertThrows(JsonException.class,
                () -> Json.toString(map(Arrays.asList(1L), 1L), Json.Style.CANONICAL));
        assertTrue(badKey.getMessage(), badKey.getMessage().startsWith("keys must be str, int, float, bool or None"));
    }

    @Test
    public void unsupportedValuesAndCyclesFail() {
        assertEquals("Object of type Object is not JSON serializable",
                assertThrows(JsonException.class, () -> Json.toString(new Object(), Json.Style.CANONICAL)).getMessage());
        List<Object> cyclic = new ArrayList<>();
        cyclic.add(cyclic);
        assertEquals("Circular reference detected",
                assertThrows(JsonException.class, () -> Json.toString(cyclic, Json.Style.CANONICAL)).getMessage());
        List<Object> shared = Collections.singletonList(1L);
        assertEquals("[[1], [1]]", Json.toString(Arrays.asList(shared, shared), Json.Style.PYTHON_DEFAULT));
    }

    @Test
    public void adapterMapsValuesKeysAndOrder() {
        Object marker = new Object();
        Json.Adapter adapter = new Json.Adapter() {
            @Override
            public Object value(Object v) {
                return v == marker ? null : v;
            }

            @Override
            public int compareKeys(Object a, Object b) {
                return -Json.compareKeys(a, b);
            }
        };
        assertEquals("{\"b\":null,\"a\":[null]}",
                Json.toString(map("a", Arrays.asList(marker), "b", marker), Json.Style.CANONICAL, adapter));
    }

    @Test
    public void writesToAnyAppendable() throws IOException {
        StringWriter out = new StringWriter();
        Json.write(map("k", Arrays.asList(1L, "v")), Json.Style.PYTHON_DEFAULT, out);
        assertEquals("{\"k\": [1, \"v\"]}", out.toString());
    }
}
