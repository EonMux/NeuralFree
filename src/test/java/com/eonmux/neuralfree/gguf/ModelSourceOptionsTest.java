package com.eonmux.neuralfree.gguf;

import org.junit.Test;

import java.math.BigInteger;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link ModelSourceOptions}: defaults, every property, strict parsing of bad values, and value semantics. */
public class ModelSourceOptionsTest {

    private static ModelSourceOptions parse(String key, String value) {
        Properties p = new Properties();
        p.setProperty(key, value);
        return ModelSourceOptions.fromProperties(p);
    }

    private static void assertRejected(String key, String value, String expected) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(key, value));
        assertTrue(e.getMessage(), e.getMessage().contains(key));
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }

    @Test
    public void defaults() {
        ModelSourceOptions o = ModelSourceOptions.defaults();
        assertEquals(ModelSourceOptions.Materialize.AUTO, o.materialize());
        assertEquals(2L << 30, o.memoryBudget());
        assertEquals(Paths.get(System.getProperty("user.home", ""), ".cache", "neuralfree", "hf"), o.cacheDir());
        assertEquals(ModelSourceOptions.OutType.SOURCE, o.outType());
        assertEquals(ModelSourceOptions.OnlineQuant.NONE, o.onlineQuant());
        assertFalse(o.exactQuant());
        assertTrue(o.includeMtp());
        assertTrue(o.stDenseModules());
        assertTrue(o.configOverrides().isEmpty());
        assertNull(o.tokenizerDir());
        assertNull(o.loraBaseConfig());
        assertNull(o.baseDir());
        assertEquals(o, ModelSourceOptions.fromProperties(new Properties()));
    }

    @Test
    public void parsesEveryProperty() {
        Properties p = new Properties();
        p.setProperty(ModelSourceOptions.PROP_MATERIALIZE, "disk");
        p.setProperty(ModelSourceOptions.PROP_MEMORY_MAX, "512M");
        p.setProperty(ModelSourceOptions.PROP_CACHE_DIR, "/var/cache/nf");
        p.setProperty(ModelSourceOptions.PROP_OUTTYPE, "bf16");
        p.setProperty(ModelSourceOptions.PROP_ONLINE_QUANT, "iq4-nl");
        p.setProperty(ModelSourceOptions.PROP_EXACT_QUANT, "TRUE");
        p.setProperty(ModelSourceOptions.PROP_MTP, "false");
        p.setProperty(ModelSourceOptions.PROP_ST_DENSE, " false ");
        p.setProperty(ModelSourceOptions.PROP_OVERRIDES, " {\"rope_theta\": 10000} ");
        p.setProperty(ModelSourceOptions.PROP_TOKENIZER, "/models/tok");
        p.setProperty(ModelSourceOptions.PROP_LORA_BASE_CONFIG, "/models/base/config.json");
        p.setProperty(ModelSourceOptions.PROP_BASE_DIR, "/models/base");
        ModelSourceOptions o = ModelSourceOptions.fromProperties(p);
        assertEquals(ModelSourceOptions.Materialize.DISK, o.materialize());
        assertEquals(512L << 20, o.memoryBudget());
        assertEquals(Paths.get("/var/cache/nf"), o.cacheDir());
        assertEquals(ModelSourceOptions.OutType.BF16, o.outType());
        assertEquals(ModelSourceOptions.OnlineQuant.IQ4_NL, o.onlineQuant());
        assertTrue(o.exactQuant());
        assertFalse(o.includeMtp());
        assertFalse(o.stDenseModules());
        assertEquals(Collections.singletonMap("rope_theta", 10000L), o.configOverrides());
        assertEquals(Paths.get("/models/tok"), o.tokenizerDir());
        assertEquals(Paths.get("/models/base/config.json"), o.loraBaseConfig());
        assertEquals(Paths.get("/models/base"), o.baseDir());
        assertEquals(o, o.toBuilder().build());
    }

    @Test
    public void parsesEveryEnumValue() {
        for (ModelSourceOptions.Materialize m : ModelSourceOptions.Materialize.values()) {
            assertEquals(m, parse(ModelSourceOptions.PROP_MATERIALIZE, m.name().toLowerCase(Locale.ROOT)).materialize());
        }
        for (ModelSourceOptions.OutType t : ModelSourceOptions.OutType.values()) {
            assertEquals(t, parse(ModelSourceOptions.PROP_OUTTYPE, t.name().toLowerCase(Locale.ROOT)).outType());
        }
        for (ModelSourceOptions.OnlineQuant q : ModelSourceOptions.OnlineQuant.values()) {
            assertEquals(q, parse(ModelSourceOptions.PROP_ONLINE_QUANT, q.name().toLowerCase(Locale.ROOT)).onlineQuant());
        }
    }

    @Test
    public void parsesSizes() {
        assertEquals(0L, ModelSourceOptions.parseSize("p", "0"));
        assertEquals(4096L, ModelSourceOptions.parseSize("p", "4096"));
        assertEquals(3L << 10, ModelSourceOptions.parseSize("p", "3k"));
        assertEquals(3L << 20, ModelSourceOptions.parseSize("p", "3M"));
        assertEquals(3L << 30, ModelSourceOptions.parseSize("p", " 3 G "));
    }

    @Test
    public void blankValuesKeepTheDefaults() {
        assertEquals(ModelSourceOptions.defaults(), parse(ModelSourceOptions.PROP_MATERIALIZE, "  "));
        assertEquals(ModelSourceOptions.defaults(), parse(ModelSourceOptions.PROP_OVERRIDES, ""));
    }

    @Test
    public void rejectsBadValues() {
        assertRejected(ModelSourceOptions.PROP_MATERIALIZE, "ram", "use auto, memory, disk, none");
        assertRejected(ModelSourceOptions.PROP_MEMORY_MAX, "lots", "is not a size; use <n>[K|M|G]");
        assertRejected(ModelSourceOptions.PROP_MEMORY_MAX, "2T", "is not a size");
        assertRejected(ModelSourceOptions.PROP_MEMORY_MAX, "G", "is not a size");
        assertRejected(ModelSourceOptions.PROP_MEMORY_MAX, "-1G", "must not be negative");
        assertRejected(ModelSourceOptions.PROP_MEMORY_MAX, "9000000000000G", "overflows a 64-bit byte count");
        assertRejected(ModelSourceOptions.PROP_OUTTYPE, "fp8", "use source, auto, f32, f16, bf16, q8_0");
        assertRejected(ModelSourceOptions.PROP_ONLINE_QUANT, "q2_k", "is invalid");
        assertRejected(ModelSourceOptions.PROP_EXACT_QUANT, "yes", "is not a boolean");
        assertRejected(ModelSourceOptions.PROP_MTP, "1", "is not a boolean");
        assertRejected(ModelSourceOptions.PROP_ST_DENSE, "on", "is not a boolean");
        assertRejected(ModelSourceOptions.PROP_OVERRIDES, "[1, 2]", "is not a JSON object");
        assertRejected(ModelSourceOptions.PROP_OVERRIDES, "{\"a\": }", "is not valid JSON");
        assertRejected(ModelSourceOptions.PROP_OVERRIDES, "{\"a\": 1} {}", "is not valid JSON");
        assertRejected(ModelSourceOptions.PROP_OVERRIDES, "{\"a\": 1, \"a\": 2}", "is not valid JSON");
        assertRejected(ModelSourceOptions.PROP_CACHE_DIR, "bad\u0000path", "is not a valid path");
        assertRejected(ModelSourceOptions.PROP_TOKENIZER, "bad\u0000path", "is not a valid path");
        assertRejected(ModelSourceOptions.PROP_LORA_BASE_CONFIG, "bad\u0000path", "is not a valid path");
        assertRejected(ModelSourceOptions.PROP_BASE_DIR, "bad\u0000path", "is not a valid path");
    }

    @Test
    public void overridesAreParsedIntoAnUnmodifiableTree() {
        ModelSourceOptions o = parse(ModelSourceOptions.PROP_OVERRIDES,
                "{\"text_config\": {\"rope_theta\": 1e6, \"layers\": [1, 2]}, \"max\": 1000000000000000019884624838656}");
        Map<String, Object> overrides = o.configOverrides();
        assertEquals(Arrays.asList("text_config", "max"), Arrays.asList(overrides.keySet().toArray()));
        assertEquals(new BigInteger("1000000000000000019884624838656"), overrides.get("max"));
        @SuppressWarnings("unchecked")
        Map<String, Object> text = (Map<String, Object>) overrides.get("text_config");
        assertEquals(1e6, text.get("rope_theta"));
        @SuppressWarnings("unchecked")
        List<Object> layers = (List<Object>) text.get("layers");
        assertEquals(Arrays.asList(1L, 2L), layers);
        assertThrows(UnsupportedOperationException.class, () -> overrides.put("x", 1L));
        assertThrows(UnsupportedOperationException.class, () -> text.remove("rope_theta"));
        assertThrows(UnsupportedOperationException.class, () -> layers.add(3L));
    }

    @Test
    public void builderCopiesOverrideMapsDeeply() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("n", 3);
        Map<String, Object> source = new LinkedHashMap<>();
        source.put("text_config", inner);
        ModelSourceOptions o = ModelSourceOptions.builder().configOverrides(source).build();
        inner.put("n", 4);
        source.put("extra", true);
        assertEquals(Collections.singletonMap("text_config", Collections.singletonMap("n", 3L)), o.configOverrides());
        assertEquals(o, ModelSourceOptions.builder().configOverridesJson("{\"text_config\": {\"n\": 3}}").build());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ModelSourceOptions.builder().configOverrides(Collections.singletonMap("p", Paths.get("/x"))));
        assertTrue(e.getMessage(), e.getMessage().contains(ModelSourceOptions.PROP_OVERRIDES));
        assertTrue(e.getMessage(), e.getMessage().contains("'p'"));
        assertTrue(ModelSourceOptions.builder().configOverrides(null).build().configOverrides().isEmpty());
    }

    @Test
    public void builderRejectsNegativeBudgetAndNulls() {
        assertThrows(IllegalArgumentException.class, () -> ModelSourceOptions.builder().memoryBudget(-1).build());
        assertThrows(NullPointerException.class, () -> ModelSourceOptions.builder().materialize(null));
        assertThrows(NullPointerException.class, () -> ModelSourceOptions.builder().cacheDir(null));
        assertThrows(NullPointerException.class, () -> ModelSourceOptions.builder().outType(null));
        assertThrows(NullPointerException.class, () -> ModelSourceOptions.builder().onlineQuant(null));
        assertThrows(NullPointerException.class, () -> ModelSourceOptions.fromProperties(null));
    }

    @Test
    public void valueSemantics() {
        ModelSourceOptions a = ModelSourceOptions.builder().outType(ModelSourceOptions.OutType.Q8_0)
                .configOverridesJson("{}").build();
        ModelSourceOptions b = ModelSourceOptions.defaults().toBuilder().outType(ModelSourceOptions.OutType.Q8_0)
                .configOverridesJson(" {} ").build();
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertEquals(a.toString(), b.toString());
        assertTrue(a.toString(), a.toString().contains("outType=Q8_0"));
        ModelSourceOptions c = a.toBuilder().configOverridesJson("{\"b\": 1, \"a\": [true, null]}").build();
        assertTrue(c.toString(), c.toString().contains("overrides={\"a\":[true,null],\"b\":1}"));
        assertEquals(c, c.toBuilder().build());
        assertNotEquals(a, ModelSourceOptions.defaults());
        assertNotEquals(a, a.toBuilder().baseDir(Paths.get("/b")).build());
        assertEquals(ModelSourceOptions.defaults(), a.toBuilder().outType(ModelSourceOptions.OutType.SOURCE)
                .configOverridesJson(null).build());
    }
}
