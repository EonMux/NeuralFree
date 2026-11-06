package com.eonmux.neuralfree.gguf.inspect;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * {@link GgufDump} over a real GGUF fixture: the reference's line formats, unsigned integers, string and array
 * truncation, the tensor table, its omissions, and the JSON report's fields.
 */
public class GgufDumpTest {

    private static final String LONG_NAME = "a name that is deliberately longer than the sixty character dump "
            + "preview limit";

    @Test
    public void textReportHasTheReferenceShape() throws Exception {
        Path file = fixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            String text = text(src, GgufDump.Options.text());
            String host = ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN ? "LITTLE" : "BIG";
            assertTrue(text, text.startsWith("* File is LITTLE endian, script is running on a " + host
                    + " endian host.\n"));
            assertTrue(text, text.contains("* Dumping 8 key/value pair(s)\n"));
            assertTrue(text, text.contains("GGUF.version = 3"));
            assertTrue(text, text.contains("GGUF.tensor_count = 2"));
            assertTrue(text, text.contains("GGUF.kv_count = 5"));
            assertTrue(text, text.contains("general.architecture = 'synthetic'"));
            assertTrue(text, text.contains("= 4294967295"));
            assertTrue(text, text.contains("= [1, 2, 3, 4, 5, 6, ...]"));
            assertTrue(text, text.contains("'a name that is deliberately longer than the sixty charact...'"));
            assertTrue(text, text.contains("* Dumping 2 tensor(s)\n"));
            assertTrue(text, text.contains("w.a"));
            assertTrue(text, text.contains("F32"));
            assertTrue(text, text.contains("    4,     1,     1,     1"));
        }
    }

    @Test
    public void noTensorsStopsAfterTheKeyValueStore() throws Exception {
        Path file = fixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            String text = text(src, GgufDump.Options.text().noTensors(true));
            assertTrue(text, text.contains("* Dumping 8 key/value pair(s)"));
            assertFalse(text, text.contains("* Dumping 2 tensor(s)"));
            assertFalse(text, text.contains("w.a"));
        }
    }

    @Test
    public void jsonReportCarriesTheReferenceFields() throws Exception {
        Path file = fixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            @SuppressWarnings("unchecked")
            Map<String, Object> report = (Map<String, Object>) Json.parse(json(src, GgufDump.Options.text()));
            assertEquals(file.toString(), report.get("filename"));
            assertEquals("LITTLE", report.get("endian"));
            @SuppressWarnings("unchecked")
            Map<String, Object> metadata = (Map<String, Object>) report.get("metadata");
            @SuppressWarnings("unchecked")
            Map<String, Object> version = (Map<String, Object>) metadata.get("GGUF.version");
            assertEquals("UINT32", version.get("type"));
            assertEquals(Long.valueOf(4L), version.get("offset"));
            assertEquals(Long.valueOf(3L), version.get("value"));
            @SuppressWarnings("unchecked")
            Map<String, Object> arch = (Map<String, Object>) metadata.get("general.architecture");
            assertEquals("STRING", arch.get("type"));
            assertEquals(Long.valueOf(src.getKeyOffset("general.architecture")), arch.get("offset"));
            assertEquals("synthetic", arch.get("value"));

            @SuppressWarnings("unchecked")
            Map<String, Object> array = (Map<String, Object>) metadata.get("k.array");
            assertEquals("ARRAY", array.get("type"));
            assertEquals(java.util.Collections.singletonList("INT32"), array.get("array_types"));
            assertFalse("an array value is omitted without --json-array", array.containsKey("value"));

            @SuppressWarnings("unchecked")
            Map<String, Object> tensors = (Map<String, Object>) report.get("tensors");
            @SuppressWarnings("unchecked")
            Map<String, Object> tensor = (Map<String, Object>) tensors.get("w.a");
            assertEquals(java.util.Arrays.asList(4L), tensor.get("shape"));
            assertEquals("F32", tensor.get("type"));
            assertEquals(Long.valueOf(0L), tensor.get("offset"));
        }
    }

    @Test
    public void jsonArrayIncludesArrayValuesAndNoTensorsOmitsTheTable() throws Exception {
        Path file = fixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            @SuppressWarnings("unchecked")
            Map<String, Object> report = (Map<String, Object>) Json.parse(
                    json(src, GgufDump.Options.text().jsonArray(true)));
            @SuppressWarnings("unchecked")
            Map<String, Object> metadata = (Map<String, Object>) report.get("metadata");
            @SuppressWarnings("unchecked")
            Map<String, Object> array = (Map<String, Object>) metadata.get("k.array");
            assertEquals(java.util.Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L), array.get("value"));

            @SuppressWarnings("unchecked")
            Map<String, Object> noTensors = (Map<String, Object>) Json.parse(
                    json(src, GgufDump.Options.text().noTensors(true)));
            assertTrue(((Map<?, ?>) noTensors.get("tensors")).isEmpty());
        }
    }

    @Test
    public void jsonMakesUintValuesUnsigned() throws Exception {
        Path file = fixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            @SuppressWarnings("unchecked")
            Map<String, Object> report = (Map<String, Object>) Json.parse(json(src, GgufDump.Options.text()));
            @SuppressWarnings("unchecked")
            Map<String, Object> metadata = (Map<String, Object>) report.get("metadata");
            @SuppressWarnings("unchecked")
            Map<String, Object> max = (Map<String, Object>) metadata.get("k.max");
            assertEquals(Long.valueOf(4294967295L), max.get("value"));
        }
    }

    private static String text(GGUFFileParser src, GgufDump.Options options) throws Exception {
        StringBuilder out = new StringBuilder();
        GgufDump.text(src, options, out);
        return out.toString();
    }

    private static String json(GGUFFileParser src, GgufDump.Options options) throws Exception {
        StringBuilder out = new StringBuilder();
        GgufDump.write(src, options.json(true), out);
        return out.toString();
    }

    private static GgufTestWriter fixture() {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("synthetic");
        w.u32("general.alignment", 32);
        w.str("general.name", LONG_NAME);
        w.u32("k.max", 4294967295L);
        w.i32Array("k.array", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        w.tensorF32("w.a", new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        w.tensorF32("w.b", new long[]{2, 2}, 5.0f, 6.0f, 7.0f, 8.0f);
        return w;
    }
}
