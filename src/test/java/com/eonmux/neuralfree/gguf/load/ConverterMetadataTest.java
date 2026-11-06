package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ConverterMetadata}: the converter's {@code --metadata} keys with the types {@code set_gguf_meta_model}
 * writes (strings, the INT32 and FLOAT32 sampling keys, the tag and language arrays, the expanded base model and
 * dataset lists), the generic typing of other keys, {@code --model-name} winning, the replaced list keys of the
 * source removed by {@link ConverterMetadata#apply}, and the refusals naming the flag and key.
 */
public class ConverterMetadataTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final String WHAT = "--metadata 'm.json'";

    private static ConverterMetadata of(String json) {
        return ConverterMetadata.fromJson((Map<?, ?>) Json.parse(json), WHAT);
    }

    private static MetadataOverrides.Override find(ConverterMetadata m, String key) {
        MetadataOverrides.Override found = null;
        for (MetadataOverrides.Override o : m.overrides()) {
            if (o.key().equals(key)) {
                found = o;
            }
        }
        if (found == null) {
            throw new AssertionError(key + " is not overridden in " + m.overrides());
        }
        return found;
    }

    @Test
    public void knownKeysTakeTheConverterWriterTypes() {
        ConverterMetadata m = of("{\"general.name\": \"Model\", \"general.author\": \"A\","
                + " \"general.sampling.top_k\": 40, \"general.sampling.temp\": 1,"
                + " \"general.sampling.top_p\": 0.9, \"general.sampling.sequence\": \"kpt\","
                + " \"general.sampling.mirostat\": 2, \"general.tags\": [\"chat\", \"llama\"],"
                + " \"general.languages\": [\"en\"]}");
        assertEquals(GGUFValueType.STRING, find(m, GgufKeys.GENERAL_NAME).type());
        assertEquals("A", find(m, GgufKeys.GENERAL_AUTHOR).value());
        assertEquals(GGUFValueType.INT32, find(m, GgufKeys.GENERAL_SAMPLING_TOP_K).type());
        assertEquals(40, find(m, GgufKeys.GENERAL_SAMPLING_TOP_K).value());
        // an integral temperature is still the FLOAT32 add_sampling_temp writes
        assertEquals(GGUFValueType.FLOAT32, find(m, GgufKeys.GENERAL_SAMPLING_TEMP).type());
        assertEquals(1.0f, find(m, GgufKeys.GENERAL_SAMPLING_TEMP).value());
        assertEquals(0.9f, find(m, GgufKeys.GENERAL_SAMPLING_TOP_P).value());
        assertEquals("kpt", find(m, GgufKeys.GENERAL_SAMPLING_SEQUENCE).value());
        assertEquals(GGUFValueType.INT32, find(m, GgufKeys.GENERAL_SAMPLING_MIROSTAT).type());
        MetadataOverrides.Override tags = find(m, "general.tags");
        assertEquals(GGUFValueType.ARRAY, tags.type());
        assertEquals(GGUFValueType.STRING, tags.elementType());
        assertEquals(List.of("chat", "llama"), tags.value());
        assertEquals(List.of("en"), find(m, "general.languages").value());
    }

    @Test
    public void baseModelAndDatasetListsExpandToTheirIndexedKeys() {
        ConverterMetadata m = of("{\"general.base_models\": [{\"name\": \"Base\", \"organization\": \"Org\","
                + " \"repo_url\": \"https://hf.co/org/base\"}, {\"name\": \"Other\"}],"
                + " \"general.datasets\": [{\"name\": \"D\", \"url\": \"https://d\"}]}");
        MetadataOverrides.Override count = find(m, GgufKeys.GENERAL_BASE_MODEL_COUNT);
        assertEquals(GGUFValueType.UINT32, count.type());
        assertEquals(2, count.value());
        assertEquals("Base", find(m, "general.base_model.0.name").value());
        assertEquals("Org", find(m, "general.base_model.0.organization").value());
        assertEquals("https://hf.co/org/base", find(m, "general.base_model.0.repo_url").value());
        assertEquals("Other", find(m, "general.base_model.1.name").value());
        assertEquals(1, find(m, GgufKeys.GENERAL_DATASET_COUNT).value());
        assertEquals("https://d", find(m, "general.dataset.0.url").value());
        assertEquals("count, three base model fields, one, dataset count and two fields", 8, m.overrides().size());
    }

    @Test
    public void otherKeysTakeTheTypeOfTheirJsonValue() {
        ConverterMetadata m = of("{\"x.s\": \"v\", \"x.b\": true, \"x.i\": 7, \"x.l\": 4294967296,"
                + " \"x.f\": 0.25, \"x.d\": 1e300, \"x.ints\": [1, 2], \"x.longs\": [1, 4294967296],"
                + " \"x.floats\": [1, 0.5], \"x.bools\": [true, false]}");
        assertEquals(GGUFValueType.STRING, find(m, "x.s").type());
        assertEquals(GGUFValueType.BOOL, find(m, "x.b").type());
        assertEquals(7, find(m, "x.i").value());
        assertEquals(4294967296L, find(m, "x.l").value());
        assertEquals(GGUFValueType.FLOAT32, find(m, "x.f").type());
        assertEquals(0.25f, find(m, "x.f").value());
        assertEquals(GGUFValueType.FLOAT64, find(m, "x.d").type());
        assertEquals(GGUFValueType.INT32, find(m, "x.ints").elementType());
        assertEquals(GGUFValueType.INT64, find(m, "x.longs").elementType());
        assertEquals(List.of(1L, 4294967296L), find(m, "x.longs").value());
        assertEquals(GGUFValueType.FLOAT32, find(m, "x.floats").elementType());
        assertEquals(List.of(1.0f, 0.5f), find(m, "x.floats").value());
        assertEquals(GGUFValueType.BOOL, find(m, "x.bools").elementType());
    }

    @Test
    public void theModelNameWinsOverTheFile() {
        ConverterMetadata m = of("{\"general.name\": \"From File\"}").withModelName("Flag");
        assertEquals("Flag", find(m, GgufKeys.GENERAL_NAME).value());
        assertEquals("Flag", find(ConverterMetadata.none().withModelName("Flag"), GgufKeys.GENERAL_NAME).value());
        assertThrows(IllegalArgumentException.class, () -> ConverterMetadata.none().withModelName(""));
    }

    @Test
    public void refusalsNameTheFlagAndTheKey() {
        assertRefused("{\"a\": null}", "'a'", "null");
        assertRefused("{\"a\": {\"b\": 1}}", "'a'", "JSON object");
        assertRefused("{\"a\": []}", "'a'", "empty array");
        assertRefused("{\"a\": [\"x\", 1]}", "'a'", "mix");
        assertRefused("{\"a\": [\"x\", null]}", "'a[1]'", "null");
        assertRefused("{\"general.name\": 3}", "'general.name'", "not a string");
        assertRefused("{\"general.sampling.top_k\": 0.5}", "'general.sampling.top_k'", "not an integer");
        assertRefused("{\"general.sampling.top_k\": 4294967296}", "'general.sampling.top_k'", "INT32");
        assertRefused("{\"general.sampling.temp\": \"hot\"}", "'general.sampling.temp'", "not a number");
        assertRefused("{\"general.tags\": [\"a\", 1]}", "'general.tags[1]'", "not a string");
        assertRefused("{\"general.tags\": []}", "'general.tags'", "empty array");
        assertRefused("{\"general.base_models\": {\"name\": \"x\"}}", "'general.base_models'", "array of objects");
        assertRefused("{\"general.base_models\": [\"x\"]}", "'general.base_models[0]'", "not an object");
        assertRefused("{\"general.base_models\": [{\"nick\": \"x\"}]}", "'general.base_models[0].nick'",
                "not one of");
        assertRefused("{\"general.datasets\": [{\"name\": 1}]}", "'general.datasets[0].name'", "not a string");
        assertRefused("{\"a\": 1e999999}", "'a'", "not finite");
    }

    private static void assertRefused(String json, String key, String detail) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> of(json));
        assertTrue(e.getMessage(), e.getMessage().startsWith(WHAT));
        assertTrue(e.getMessage(), e.getMessage().contains(key));
        assertTrue(e.getMessage(), e.getMessage().contains(detail));
    }

    @Test
    public void applyOverridesTheSourceAndReplacesItsListKeys() throws Exception {
        Path gguf = SyntheticModels.llama().writeTempFile();
        GGUFFileParser parser = new GGUFFileParser(gguf.toString());
        // a source that already carries three base models and a dataset
        ModelSource carrying = new OverriddenSource(parser, List.of(
                MetadataOverrides.Override.scalar(GgufKeys.GENERAL_BASE_MODEL_COUNT, GGUFValueType.UINT32, 3),
                MetadataOverrides.Override.scalar("general.base_model.2.name", GGUFValueType.STRING, "Old"),
                MetadataOverrides.Override.scalar("general.dataset.0.name", GGUFValueType.STRING, "Kept")));
        ConverterMetadata m = of("{\"general.base_models\": [{\"name\": \"New\"}], \"general.tags\": [\"t\"]}");
        try (ModelSource view = m.apply(carrying)) {
            Map<String, Object> md = view.getMetadata();
            assertEquals(1, md.get(GgufKeys.GENERAL_BASE_MODEL_COUNT));
            assertEquals("New", view.getString("general.base_model.0.name"));
            assertFalse("the replaced list leaves no stale entry", md.containsKey("general.base_model.2.name"));
            assertNull(view.getValueType("general.base_model.2.name"));
            assertNull(view.getString("general.base_model.2.name"));
            assertEquals("Kept", view.getString("general.dataset.0.name"));
            assertEquals(GGUFValueType.STRING, view.getArrayElementType("general.tags"));
            assertEquals(parser.getString(GgufKeys.GENERAL_ARCHITECTURE),
                    view.getString(GgufKeys.GENERAL_ARCHITECTURE));
        }
    }

    @Test
    public void nothingToApplyReturnsTheSourceItself() throws Exception {
        Path gguf = SyntheticModels.llama().writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(gguf.toString())) {
            assertSame(parser, ConverterMetadata.none().apply(parser));
            assertTrue(ConverterMetadata.none().isEmpty());
            assertSame(parser, of("{}").apply(parser));
        }
    }
}
