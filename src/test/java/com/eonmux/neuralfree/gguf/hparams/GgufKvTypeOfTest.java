package com.eonmux.neuralfree.gguf.hparams;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/** {@link GgufKv#typeOf}: the stored value type of map-backed and parser-backed metadata. */
public class GgufKvTypeOfTest {

    @Test
    public void reportsInferredTypesOfMapValues() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "llama");
        m.put("llama.flag", Boolean.TRUE);
        m.put("llama.count", 7);
        m.put("llama.big", 5_000_000_000L);
        m.put("llama.scale", 0.5f);
        m.put("llama.ratio", 0.25);
        m.put("llama.list", Arrays.asList(1, 2, 3));
        GgufKv kv = GgufKv.ofMap(m);

        assertEquals(GGUFValueType.STRING, kv.typeOf(GgufKeys.GENERAL_ARCHITECTURE));
        assertEquals(GGUFValueType.BOOL, kv.typeOf("%s.flag"));
        assertEquals(GGUFValueType.INT32, kv.typeOf("%s.count"));
        assertEquals(GGUFValueType.INT64, kv.typeOf("%s.big"));
        assertEquals(GGUFValueType.FLOAT32, kv.typeOf("%s.scale"));
        assertEquals(GGUFValueType.FLOAT64, kv.typeOf("%s.ratio"));
        assertEquals(GGUFValueType.ARRAY, kv.typeOf("%s.list"));
        assertNull(kv.typeOf("%s.absent"));
        assertNull(kv.typeOf("plain.absent"));
    }

    @Test
    public void reportsStoredTypesOfParsedFile() throws IOException {
        GgufTestWriter w = new GgufTestWriter()
                .architecture("llama")
                .u32("llama.block_count", 4)
                .u8("llama.small", 3)
                .bool("llama.causal", false)
                .str("llama.name", "true")
                .f32Array("llama.values", 1f, 2f);
        Path path = w.writeTempFile();
        GGUFFileParser parser = null;
        try {
            parser = new GGUFFileParser(path.toString());
            GgufKv kv = GgufKv.of(parser);
            assertEquals(GGUFValueType.UINT32, kv.typeOf("%s.block_count"));
            assertEquals(GGUFValueType.UINT8, kv.typeOf("%s.small"));
            assertEquals(GGUFValueType.BOOL, kv.typeOf("%s.causal"));
            assertEquals(GGUFValueType.STRING, kv.typeOf("%s.name"));
            assertEquals(GGUFValueType.ARRAY, kv.typeOf("%s.values"));
            assertNull(kv.typeOf("%s.missing"));
        } finally {
            try {
                if (parser != null) {
                    parser.close();
                }
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }
}
