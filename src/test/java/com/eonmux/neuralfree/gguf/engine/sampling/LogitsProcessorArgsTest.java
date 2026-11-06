package com.eonmux.neuralfree.gguf.engine.sampling;

import org.junit.Test;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The custom logits processor arguments of a request's {@code vllm_xargs}: typing, normalization and refusals. */
public class LogitsProcessorArgsTest {

    @Test
    public void valuesAreNormalizedAndReadable() {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("target_token", 42);
        values.put("label", "hello");
        values.put("on", true);
        values.put("alpha", 0.5f);
        values.put("ratio", 2.5d);
        values.put("ids", Arrays.asList(1, 2L, "x", false));
        LogitsProcessorArgs args = LogitsProcessorArgs.of(values);

        assertFalse(args.isEmpty());
        assertTrue(args.has("target_token"));
        assertEquals(42, args.integer("target_token"));
        assertEquals("hello", args.string("label"));
        assertTrue(args.flag("on"));
        assertEquals(0.5d, args.number("alpha"), 0d);
        assertEquals(2.5d, args.number("ratio"), 0d);
        assertEquals(Arrays.asList(1L, 2L, "x", false), args.list("ids"));
        assertEquals("integers are held as Long", 42L, args.get("target_token"));
        assertNull(args.get("absent"));
        assertFalse(args.has("absent"));

        Map<String, Object> returned = args.values();
        assertEquals(6, returned.size());
        assertThrows(UnsupportedOperationException.class, () -> returned.put("x", 1));
        assertThrows(UnsupportedOperationException.class, () -> args.list("ids").add(9));
    }

    @Test
    public void emptyHoldsNoArgument() {
        assertTrue(LogitsProcessorArgs.empty().isEmpty());
        assertSame(LogitsProcessorArgs.empty(), LogitsProcessorArgs.of(new LinkedHashMap<>()));
    }

    @Test
    public void malformedValuesAreRefusedByName() {
        Map<String, Object> nested = new LinkedHashMap<>();
        nested.put("a", new LinkedHashMap<>());
        assertMessage(() -> LogitsProcessorArgs.of(nested), "vllm_xargs.a");

        Map<String, Object> nestedList = new LinkedHashMap<>();
        nestedList.put("a", Arrays.asList(1, Arrays.asList(2)));
        assertMessage(() -> LogitsProcessorArgs.of(nestedList), "vllm_xargs.a[1]");

        Map<String, Object> nullEntry = new LinkedHashMap<>();
        nullEntry.put("a", Arrays.asList("x", null));
        assertMessage(() -> LogitsProcessorArgs.of(nullEntry), "vllm_xargs.a[1]");

        Map<String, Object> nullValue = new LinkedHashMap<>();
        nullValue.put("a", null);
        assertMessage(() -> LogitsProcessorArgs.of(nullValue), "vllm_xargs.a");

        Map<String, Object> blank = new LinkedHashMap<>();
        blank.put(" ", 1);
        assertMessage(() -> LogitsProcessorArgs.of(blank), "blank");
    }

    @Test
    public void gettersRefuseMissingAndMistypedArguments() {
        LogitsProcessorArgs args = LogitsProcessorArgs.of(values());
        assertMessage(() -> args.string("missing"), "vllm_xargs.missing is missing");
        assertMessage(() -> args.integer("label"), "vllm_xargs.label must be an integer");
        assertMessage(() -> args.string("target_token"), "vllm_xargs.target_token must be a string");
        assertMessage(() -> args.number("label"), "vllm_xargs.label must be a number");
        assertMessage(() -> args.flag("label"), "vllm_xargs.label must be a boolean");
        assertMessage(() -> args.list("label"), "vllm_xargs.label must be an array");

        Map<String, Object> huge = new LinkedHashMap<>();
        huge.put("n", Long.MAX_VALUE);
        LogitsProcessorArgs large = LogitsProcessorArgs.of(huge);
        assertMessage(() -> large.integer("n"), "outside the integer range");
    }

    private static Map<String, Object> values() {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("target_token", 7);
        values.put("label", "x");
        return values;
    }

    private static void assertMessage(Runnable call, String expected) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, call::run);
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }

    @Test
    public void theListGetterReturnsTheSameReadOnlyView() {
        LogitsProcessorArgs args = LogitsProcessorArgs.of(Map.of("ids", List.of(1, 2)));
        assertEquals(List.of(1L, 2L), args.list("ids"));
    }
}
