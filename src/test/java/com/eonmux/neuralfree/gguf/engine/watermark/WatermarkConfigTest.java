package com.eonmux.neuralfree.gguf.engine.watermark;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link WatermarkConfig}: the vLLM field names, the key sources, value ranges and the derived dual keys. */
public class WatermarkConfigTest {

    private static Properties properties(String json) {
        Properties p = new Properties();
        p.setProperty(WatermarkConfig.PROP_CONFIG, json);
        return p;
    }

    /** The configuration of {@code json} with no environment key source. */
    private static WatermarkConfig fromJson(String json) {
        return WatermarkConfig.fromProperties(properties(json), Collections.emptyMap());
    }

    /** The environment a test passes: the named variables with their values. */
    private static Map<String, String> env(String... namesAndValues) {
        Map<String, String> out = new HashMap<>();
        for (int i = 0; i < namesAndValues.length; i += 2) {
            out.put(namesAndValues[i], namesAndValues[i + 1]);
        }
        return out;
    }

    @Test
    public void theConfigReadsVllmsJsonFields() {
        WatermarkConfig config = fromJson("{\"key\": 18446744073709551615,"
                + " \"algorithm\": \"dual_key_gumbel\", \"alpha\": 0.25, \"context_width\": 6,"
                + " \"deduplicate_contexts\": \"all\", \"deduplicate_contexts_max_history\": 1024,"
                + " \"prf\": \"philox\", \"allow_target_only_watermarking\": true}");
        assertEquals(-1L, config.key());
        assertEquals(WatermarkAlgorithm.DUAL_KEY_GUMBEL, config.algorithm());
        assertEquals(0.25, config.alpha(), 0.0);
        assertEquals(6, config.contextWidth());
        assertEquals(WatermarkContextScope.ALL, config.deduplicateContexts());
        assertEquals(Integer.valueOf(1024), config.deduplicateContextsMaxHistory());
        assertEquals(WatermarkPrfName.PHILOX, config.prf());
        assertTrue(config.allowTargetOnlyWatermarking());
        assertTrue(config.supportsSpeculativeDecoding());
    }

    @Test
    public void absentFieldsTakeTheDefaults() {
        WatermarkConfig config = fromJson("{\"key\": \"42\"}");
        assertEquals(42L, config.key());
        assertEquals(WatermarkAlgorithm.GUMBEL, config.algorithm());
        assertEquals(WatermarkConfig.DEFAULT_ALPHA, config.alpha(), 0.0);
        assertEquals(WatermarkConfig.DEFAULT_CONTEXT_WIDTH, config.contextWidth());
        assertEquals(WatermarkContextScope.SINGLE_TURN, config.deduplicateContexts());
        assertEquals(WatermarkConfig.DEFAULT_DEDUPLICATE_MAX_HISTORY,
                config.deduplicateContextsMaxHistory());
        assertFalse(config.supportsSpeculativeDecoding());
    }

    @Test
    public void anAbsentOrBlankPropertyIsNoWatermark() {
        assertNull(WatermarkConfig.fromProperties(new Properties(), Collections.emptyMap()));
        assertNull(WatermarkConfig.fromProperties(properties("   "), Collections.emptyMap()));
    }

    @Test
    public void aNullHistoryBoundSearchesTheWholeScope() {
        WatermarkConfig config = fromJson(
                "{\"key\": 1, \"deduplicate_contexts_max_history\": null}");
        assertNull(config.deduplicateContextsMaxHistory());
    }

    @Test
    public void unknownFieldsAndBadValuesAreRefusedByName() {
        assertTrue(refusalOf("{\"key\": 1, \"fork\": true}").contains("fork"));
        assertTrue(refusalOf("{}").contains("key"));
        assertTrue(refusalOf("{\"key\": 1, \"alpha\": 1.5}").contains("alpha"));
        assertTrue(refusalOf("{\"key\": 1, \"context_width\": 0}").contains("context_width"));
        assertTrue(refusalOf("{\"key\": 1, \"algorithm\": \"markov\"}").contains("markov"));
        assertTrue(refusalOf("{\"key\": 1, \"deduplicate_contexts\": \"maybe\"}").contains("maybe"));
        assertTrue(refusalOf("{\"key\": 1, \"deduplicate_contexts_max_history\": 0}")
                .contains("deduplicate_contexts_max_history"));
        assertTrue(refusalOf("[1]").contains("JSON object"));
        assertTrue(refusalOf("{\"key\": 18446744073709551616}").contains("64-bit"));
    }

    /** The message of the refusal of {@code json} with no environment key source. */
    private static String refusalOf(String json) {
        return assertThrows(IllegalArgumentException.class, () -> fromJson(json)).getMessage();
    }

    @Test
    public void aKeyAboveTheSignedRangeKeepsItsUnsignedBits() {
        WatermarkConfig config = WatermarkConfig.builder(-1L).build();
        assertEquals("18446744073709551615", Long.toUnsignedString(config.key()));
    }

    @Test
    public void theDerivedKeysAreVllms() {
        assertEquals("15691511769010696906", Long.toUnsignedString(WatermarkConfig.keyA(0L)));
        assertEquals("9736202488081271922", Long.toUnsignedString(WatermarkConfig.keyB(0L)));
        assertEquals("37150756557148556", Long.toUnsignedString(WatermarkConfig.keyA(42L)));
    }

    @Test
    public void theDescriptionLeavesTheSecretKeyOut() {
        WatermarkConfig config = WatermarkConfig.builder(987654321L).build();
        assertFalse(config.toString().contains("987654321"));
    }

    @Test
    public void aBuildWithoutAKeyIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> WatermarkConfig.builder().build());
    }

    @Test
    public void theTargetRoleUsesKeyBOfADualKeyConfigAndTheSingleKeyOtherwise() {
        float[] logits = {0.1f, 0.7f, 0.2f, 0.4f};
        int[] context = {3, 1};
        WatermarkConfig dual = WatermarkConfig.builder(1L)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).contextWidth(2).build();
        Watermarker target = dual.targetWatermarker();
        Watermarker keyB = new GumbelWatermarker(WatermarkConfig.keyB(1L), 2);
        assertEquals(keyB.sample(logits, context, () -> 0.5),
                target.sample(logits, context, () -> 0.5));

        WatermarkConfig single = WatermarkConfig.builder(1L).contextWidth(2).build();
        GumbelWatermarker singleTarget = (GumbelWatermarker) single.targetWatermarker();
        GumbelWatermarker direct = new GumbelWatermarker(1L, 2);
        assertEquals(direct.sample(logits, context, () -> 0.5),
                singleTarget.sample(logits, context, () -> 0.5));
    }

    @Test
    public void theDraftRoleUsesKeyAAndRefusesASingleKeyWithoutTargetOnly() {
        WatermarkConfig dual = WatermarkConfig.builder(9L)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).contextWidth(3).build();
        Watermarker draft = dual.draftWatermarker();
        Watermarker keyA = new GumbelWatermarker(WatermarkConfig.keyA(9L), 3);
        float[] logits = {0.2f, 0.5f, 0.1f, 0.9f, 0.3f};
        assertEquals(keyA.sample(logits, new int[]{1, 2, 3}, () -> 0.5),
                draft.sample(logits, new int[]{1, 2, 3}, () -> 0.5));

        WatermarkConfig single = WatermarkConfig.builder(9L).build();
        assertTrue(assertThrows(IllegalArgumentException.class, single::draftWatermarker).getMessage()
                .contains("allow_target_only_watermarking"));
        WatermarkConfig targetOnly = WatermarkConfig.builder(9L).allowTargetOnlyWatermarking(true).build();
        assertNull(targetOnly.draftWatermarker());
    }

    @Test
    public void onlyTheDualKeyAlgorithmSeparatesTheDraftRole() {
        assertFalse(WatermarkConfig.builder(1L).build().supportsSpeculativeDecoding());
        assertTrue(WatermarkConfig.builder(1L).algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).build()
                .supportsSpeculativeDecoding());
    }

    @Test
    public void equalConfigsShareTheirValues() {
        assertEquals(WatermarkConfig.builder(7L).contextWidth(3).build(),
                WatermarkConfig.builder(7L).contextWidth(3).build());
    }

    @Test
    public void thePropertyReachesTheSamplerConfiguration() {
        Properties p = new Properties();
        p.setProperty(WatermarkConfig.PROP_CONFIG, "{\"key\": 5, \"context_width\": 2}");
        com.eonmux.neuralfree.gguf.engine.SamplerConfig config =
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.fromProperties(p);
        assertEquals(WatermarkConfig.builder(5L).contextWidth(2).build(), config.watermark());
        assertEquals("a copy keeps the watermark", config.watermark(), config.toBuilder().build().watermark());
        assertNull(com.eonmux.neuralfree.gguf.engine.SamplerConfig.defaults().watermark());
    }

    @Test
    public void theKeyComesFromTheEnvironmentVariable() {
        WatermarkConfig config = WatermarkConfig.fromProperties(new Properties(),
                env(WatermarkConfig.ENV_KEY, " 1234 "));
        assertEquals(1234L, config.key());
        assertEquals(WatermarkAlgorithm.GUMBEL, config.algorithm());
        assertEquals(WatermarkConfig.DEFAULT_CONTEXT_WIDTH, config.contextWidth());
    }

    @Test
    public void theEnvironmentVariableKeepsTheUnsignedRange() {
        WatermarkConfig config = WatermarkConfig.fromProperties(new Properties(),
                env(WatermarkConfig.ENV_KEY, "18446744073709551615"));
        assertEquals(-1L, config.key());
    }

    @Test
    public void theKeyComesFromTheFileTheEnvironmentNames() throws IOException {
        Path file = Files.createTempFile("watermark-key", ".txt");
        try {
            Files.write(file, " 987654321 \n".getBytes(StandardCharsets.UTF_8));
            WatermarkConfig config = WatermarkConfig.fromProperties(new Properties(),
                    env(WatermarkConfig.ENV_KEY_FILE, " " + file + " "));
            assertEquals(987654321L, config.key());
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void theEnvironmentVariableWinsOverTheFile() throws IOException {
        Path file = Files.createTempFile("watermark-key", ".txt");
        try {
            Files.write(file, "1\n".getBytes(StandardCharsets.UTF_8));
            Map<String, String> env = env(WatermarkConfig.ENV_KEY, "2",
                    WatermarkConfig.ENV_KEY_FILE, file.toString());
            assertEquals(2L, WatermarkConfig.fromProperties(new Properties(), env).key());
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void theEnvironmentKeyCarriesTheJsonSettings() {
        WatermarkConfig config = WatermarkConfig.fromProperties(
                properties("{\"algorithm\": \"dual_key_gumbel\", \"context_width\": 2}"),
                env(WatermarkConfig.ENV_KEY, "77"));
        assertEquals(77L, config.key());
        assertEquals(WatermarkAlgorithm.DUAL_KEY_GUMBEL, config.algorithm());
        assertEquals(2, config.contextWidth());
    }

    @Test
    public void anEnvironmentKeyBesideTheJsonKeyIsRefusedByName() throws IOException {
        String both = assertThrows(IllegalArgumentException.class, () -> WatermarkConfig.fromProperties(
                properties("{\"key\": 1}"), env(WatermarkConfig.ENV_KEY, "2"))).getMessage();
        assertTrue(both, both.contains(WatermarkConfig.ENV_KEY) && both.contains("key"));

        Path file = Files.createTempFile("watermark-key", ".txt");
        try {
            Files.write(file, "2\n".getBytes(StandardCharsets.UTF_8));
            String fileBoth = assertThrows(IllegalArgumentException.class, () -> WatermarkConfig.fromProperties(
                    properties("{\"key\": 1}"), env(WatermarkConfig.ENV_KEY_FILE, file.toString())))
                    .getMessage();
            assertTrue(fileBoth, fileBoth.contains(WatermarkConfig.ENV_KEY_FILE) && fileBoth.contains("key"));
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void aMalformedEnvironmentKeyIsRefusedWithoutRepeatingIt() {
        String message = assertThrows(IllegalArgumentException.class, () -> WatermarkConfig.fromProperties(
                new Properties(), env(WatermarkConfig.ENV_KEY, "not-a-key"))).getMessage();
        assertTrue(message, message.contains(WatermarkConfig.ENV_KEY));
        assertFalse(message, message.contains("not-a-key"));
        assertThrows(IllegalArgumentException.class, () -> WatermarkConfig.fromProperties(
                new Properties(), env(WatermarkConfig.ENV_KEY, "  ")));
        assertThrows(IllegalArgumentException.class, () -> WatermarkConfig.fromProperties(
                new Properties(), env(WatermarkConfig.ENV_KEY, "18446744073709551616")));
    }

    @Test
    public void anUnreadableOrEmptyKeyFileIsRefusedByName() throws IOException {
        Path missing = Files.createTempDirectory("watermark-key").resolve("absent");
        assertTrue(assertThrows(IllegalArgumentException.class, () -> WatermarkConfig.fromProperties(
                new Properties(), env(WatermarkConfig.ENV_KEY_FILE, missing.toString()))).getMessage()
                .contains(WatermarkConfig.ENV_KEY_FILE));
        Path empty = Files.createTempFile("watermark-key", ".txt");
        try {
            assertThrows(IllegalArgumentException.class, () -> WatermarkConfig.fromProperties(
                    new Properties(), env(WatermarkConfig.ENV_KEY_FILE, empty.toString())));
        } finally {
            Files.deleteIfExists(empty);
        }
    }

    @Test
    public void anEnvironmentKeyIsNotWrittenToASystemProperty() {
        Properties system = System.getProperties();
        String before = system.getProperty(WatermarkConfig.PROP_CONFIG);
        String key = "3141592653589793238";
        WatermarkConfig.fromProperties(new Properties(), env(WatermarkConfig.ENV_KEY, key));
        assertEquals(before, system.getProperty(WatermarkConfig.PROP_CONFIG));
        assertNull(system.getProperty("neuralfree.watermark.key"));
        assertFalse(system.toString().contains(key));
    }

    @Test
    public void theEnvironmentFormLeavesTheKeyOutOfTheDescriptionAndErrors() {
        String key = "3141592653589793238";
        WatermarkConfig config = WatermarkConfig.fromProperties(new Properties(),
                env(WatermarkConfig.ENV_KEY, key));
        assertFalse(config.toString(), config.toString().contains(key));
        String badJson = assertThrows(IllegalArgumentException.class, () -> WatermarkConfig.fromProperties(
                properties("{\"context_width\": 0}"),
                env(WatermarkConfig.ENV_KEY, "5"))).getMessage();
        assertTrue(badJson, badJson.contains("context_width"));
        assertFalse(badJson, badJson.contains("{\"context_width\": 0}"));
    }
}
