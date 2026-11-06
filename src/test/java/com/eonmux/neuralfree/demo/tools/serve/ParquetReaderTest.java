package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The Parquet reader over small Arrow-written files: flat string and numeric columns under the
 * uncompressed, Snappy and GZip codecs, the delta and dictionary encodings, null values, and the
 * nested {@code LIST<STRUCT>} and {@code LIST<STRING>} shapes of the reference's conversation
 * datasets.
 */
public class ParquetReaderTest {

    private static Path fixture(String name) throws IOException {
        Path resource = Path.of("src", "test", "resources", "parquet", name);
        if (!Files.isRegularFile(resource)) {
            throw new IOException("missing parquet fixture " + resource.toAbsolutePath());
        }
        Path tmp = Files.createTempFile("parquet-" + name.replace('.', '-'), ".parquet");
        Files.copy(resource, tmp, StandardCopyOption.REPLACE_EXISTING);
        tmp.toFile().deleteOnExit();
        return tmp;
    }

    @Test
    public void flatColumnsReadUnderEverySupportedCodec() throws IOException {
        for (String name : new String[]{"flat_none.parquet", "flat_snappy.parquet", "flat_gzip.parquet"}) {
            List<Map<String, Object>> rows = ParquetReader.read(fixture(name));
            assertEquals(name, 3, rows.size());
            assertEquals("What is 1+1?", rows.get(0).get("question"));
            assertEquals("2", rows.get(0).get("answer"));
            assertEquals(2L, rows.get(0).get("output_tokens"));
            assertEquals(0.0, (Double) rows.get(0).get("norm_distance"), 0.0);
            assertEquals(50L, rows.get(2).get("output_tokens"));
            assertEquals(1.0, (Double) rows.get(2).get("norm_distance"), 0.0);
            assertEquals(300, ((String) rows.get(2).get("question")).length());
        }
    }

    @Test
    public void dictionaryAndDeltaEncodingsReadTheSameValues() throws IOException {
        List<Map<String, Object>> dictionary = ParquetReader.read(fixture("flat_snappy.parquet"));
        List<Map<String, Object>> delta = ParquetReader.read(fixture("flat_delta.parquet"));
        assertEquals(dictionary.size(), delta.size());
        for (int i = 0; i < dictionary.size(); i++) {
            assertEquals(dictionary.get(i).get("question"), delta.get(i).get("question"));
            assertEquals(dictionary.get(i).get("answer"), delta.get(i).get("answer"));
            assertEquals(dictionary.get(i).get("output_tokens"), delta.get(i).get("output_tokens"));
            assertEquals(dictionary.get(i).get("norm_distance"), delta.get(i).get("norm_distance"));
        }
    }

    @Test
    public void nullValuesStayNull() throws IOException {
        List<Map<String, Object>> rows = ParquetReader.read(fixture("flat_nulls.parquet"));
        assertEquals(4, rows.size());
        assertEquals("a", rows.get(0).get("question"));
        assertNull(rows.get(1).get("question"));
        assertNull(rows.get(1).get("output_tokens"));
        assertEquals("c", rows.get(2).get("question"));
        assertNull(rows.get(3).get("question"));
        assertEquals(4L, rows.get(3).get("output_tokens"));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void nestedListsAndStructsReassemble() throws IOException {
        List<Map<String, Object>> rows = ParquetReader.read(fixture("nested_snappy.parquet"));
        assertEquals(3, rows.size());
        assertEquals("hello", rows.get(0).get("prompt"));
        List<Object> turns = (List<Object>) rows.get(0).get("turns");
        assertEquals(List.of("a", "b"), turns);
        assertTrue(((List<Object>) rows.get(2).get("turns")).isEmpty());
        List<Object> conversations = (List<Object>) rows.get(0).get("conversations");
        assertEquals(2, conversations.size());
        Map<String, Object> first = (Map<String, Object>) conversations.get(0);
        assertEquals("human", first.get("from"));
        assertEquals("hi", first.get("value"));
        Map<String, Object> second = (Map<String, Object>) conversations.get(1);
        assertEquals("gpt", second.get("from"));
        assertEquals("hello", second.get("value"));
        List<Object> messages = (List<Object>) rows.get(1).get("messages");
        assertEquals("user", ((Map<String, Object>) messages.get(0)).get("role"));
        assertEquals("yo", ((Map<String, Object>) messages.get(0)).get("content"));
    }

    @Test
    public void aFileWithoutTheMagicIsRefused() throws IOException {
        Path bad = Files.createTempFile("parquet-bad", ".parquet");
        bad.toFile().deleteOnExit();
        Files.write(bad, "not a parquet file at all".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        try {
            ParquetReader.read(bad);
            fail("a file without PAR1 must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("PAR1"));
        }
    }
}
