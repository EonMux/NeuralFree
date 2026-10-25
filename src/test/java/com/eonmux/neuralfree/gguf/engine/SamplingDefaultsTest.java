package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.lang.reflect.Proxy;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SamplingDefaults#read}: only present keys are read, with GGUF typing (integer settings need an integer
 * type, the others any number), finiteness and the ranges the sampler accepts, every error naming the key; the
 * llama.cpp keys without a NeuralFree stage are ignored. {@link SamplingDefaults#forModel} and
 * {@link EngineConfig#modelSamplingDefaults()}: the opt-out.
 */
public class SamplingDefaultsTest {

    private static Map<String, Object> metadata() {
        Map<String, Object> m = new HashMap<>();
        m.put(GgufKeys.GENERAL_ARCHITECTURE, "llama");
        return m;
    }

    private static SamplingDefaults read(Map<String, Object> m) {
        return SamplingDefaults.read(GgufKv.ofMap(m));
    }

    private static SamplingDefaults readOne(String key, Object value) {
        Map<String, Object> m = metadata();
        m.put(key, value);
        return read(m);
    }

    private static void assertRejected(String key, Object value) {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> readOne(key, value));
        assertTrue(e.getMessage(), e.getMessage().contains("'" + key + "'"));
    }

    @Test
    public void absentKeysGiveNone() {
        SamplingDefaults d = read(metadata());
        assertSame(SamplingDefaults.NONE, d);
        assertTrue(d.isEmpty());
        assertNull(d.temp());
        assertNull(d.maxNewTokens());
    }

    @Test
    public void everyKeyIsRead() {
        Map<String, Object> m = metadata();
        m.put(GgufKeys.GENERAL_SAMPLING_TEMP, 0.6f);
        m.put(GgufKeys.GENERAL_SAMPLING_TOP_K, 20);
        m.put(GgufKeys.GENERAL_SAMPLING_TOP_P, 0.95f);
        m.put(GgufKeys.GENERAL_SAMPLING_MIN_P, 0.0f);
        m.put(GgufKeys.GENERAL_SAMPLING_PENALTY_LAST_N, -1);
        m.put(GgufKeys.GENERAL_SAMPLING_PENALTY_REPEAT, 1.1f);
        m.put(GgufKeys.NF_SAMPLING_PRESENCE_PENALTY, 1.5f);
        m.put(GgufKeys.NF_SAMPLING_FREQUENCY_PENALTY, -0.5f);
        m.put(GgufKeys.NF_SAMPLING_MAX_NEW_TOKENS, 256);
        SamplingDefaults d = read(m);
        assertFalse(d.isEmpty());
        assertEquals(0.6f, d.temp(), 0f);
        assertEquals(20, d.topK().intValue());
        assertEquals(0.95f, d.topP(), 0f);
        assertEquals(0.0f, d.minP(), 0f);
        assertEquals(-1, d.penaltyLastN().intValue());
        assertEquals(1.1f, d.penaltyRepeat(), 0f);
        assertEquals(1.5f, d.presencePenalty(), 0f);
        assertEquals(-0.5f, d.frequencyPenalty(), 0f);
        assertEquals(256, d.maxNewTokens().intValue());
        assertNull(d.typicalP());
        assertEquals(SamplingDefaults.builder().temp(0.6f).topK(20).topP(0.95f).minP(0f).penaltyLastN(-1)
                .penaltyRepeat(1.1f).presencePenalty(1.5f).frequencyPenalty(-0.5f).maxNewTokens(256).build(), d);
    }

    @Test
    public void onlyPresentKeysAreSet() {
        SamplingDefaults d = readOne(GgufKeys.GENERAL_SAMPLING_TOP_P, 0.8f);
        assertEquals(0.8f, d.topP(), 0f);
        assertNull(d.temp());
        assertNull(d.topK());
        assertNull(d.minP());
        assertNull(d.penaltyRepeat());
        assertNull(d.penaltyLastN());
        assertNull(d.presencePenalty());
        assertNull(d.frequencyPenalty());
        assertNull(d.maxNewTokens());
    }

    @Test
    public void floatKeysAcceptEveryNumericType() {
        assertEquals(1.0f, readOne(GgufKeys.GENERAL_SAMPLING_TEMP, 1).temp(), 0f);
        assertEquals(0.7f, readOne(GgufKeys.GENERAL_SAMPLING_TEMP, 0.7).temp(), 0f);
        assertEquals(2.0f, readOne(GgufKeys.GENERAL_SAMPLING_PENALTY_REPEAT, 2L).penaltyRepeat(), 0f);
    }

    @Test
    public void integerKeysAcceptEveryIntegerTypeInRange() {
        assertEquals(40, readOne(GgufKeys.GENERAL_SAMPLING_TOP_K, 40L).topK().intValue());
        assertEquals(128, readOne(GgufKeys.GENERAL_SAMPLING_PENALTY_LAST_N, (short) 128).penaltyLastN().intValue());
        assertEquals(-1, readOne(GgufKeys.NF_SAMPLING_MAX_NEW_TOKENS, -1).maxNewTokens().intValue());
        assertRejected(GgufKeys.GENERAL_SAMPLING_TOP_K, 1L << 40);
    }

    @Test
    public void wrongTypesAreRejectedNamingTheKey() {
        assertRejected(GgufKeys.GENERAL_SAMPLING_TEMP, "0.7");
        assertRejected(GgufKeys.GENERAL_SAMPLING_TOP_K, 40.0f);
        assertRejected(GgufKeys.GENERAL_SAMPLING_PENALTY_LAST_N, 64.0);
        assertRejected(GgufKeys.GENERAL_SAMPLING_PENALTY_REPEAT, Boolean.TRUE);
        assertRejected(GgufKeys.GENERAL_SAMPLING_TOP_P, Arrays.asList(0.9f, 0.8f));
        assertRejected(GgufKeys.NF_SAMPLING_MAX_NEW_TOKENS, 256.0f);
        assertRejected(GgufKeys.NF_SAMPLING_PRESENCE_PENALTY, "high");
    }

    @Test
    public void nonFiniteValuesAreRejectedNamingTheKey() {
        assertRejected(GgufKeys.GENERAL_SAMPLING_TEMP, Float.NaN);
        assertRejected(GgufKeys.GENERAL_SAMPLING_TEMP, Double.POSITIVE_INFINITY);
        assertRejected(GgufKeys.NF_SAMPLING_FREQUENCY_PENALTY, Float.NEGATIVE_INFINITY);
        assertRejected(GgufKeys.NF_SAMPLING_PRESENCE_PENALTY, 1e300);
    }

    @Test
    public void valuesTheSamplerRejectsAreRejectedNamingTheKey() {
        assertRejected(GgufKeys.GENERAL_SAMPLING_TOP_P, 1.5f);
        assertRejected(GgufKeys.GENERAL_SAMPLING_MIN_P, -0.1f);
        assertRejected(GgufKeys.GENERAL_SAMPLING_PENALTY_REPEAT, 0f);
        assertRejected(GgufKeys.GENERAL_SAMPLING_PENALTY_LAST_N, -2);
        assertRejected(GgufKeys.NF_SAMPLING_MAX_NEW_TOKENS, 0);
        assertRejected(GgufKeys.NF_SAMPLING_MAX_NEW_TOKENS, -5);
    }

    @Test
    public void disablingValuesAreAccepted() {
        assertEquals(0f, readOne(GgufKeys.GENERAL_SAMPLING_TEMP, 0f).temp(), 0f);
        assertEquals(-3f, readOne(GgufKeys.GENERAL_SAMPLING_TEMP, -3f).temp(), 0f);
        assertEquals(0, readOne(GgufKeys.GENERAL_SAMPLING_TOP_K, 0).topK().intValue());
        assertEquals(-1, readOne(GgufKeys.GENERAL_SAMPLING_TOP_K, -1).topK().intValue());
        assertEquals(1f, readOne(GgufKeys.GENERAL_SAMPLING_TOP_P, 1f).topP(), 0f);
        assertEquals(0, readOne(GgufKeys.GENERAL_SAMPLING_PENALTY_LAST_N, 0).penaltyLastN().intValue());
    }

    @Test
    public void stagesWithoutANeuralFreeSamplerAreIgnored() {
        Map<String, Object> m = metadata();
        m.put(GgufKeys.GENERAL_SAMPLING_SEQUENCE, "edkypmxt");
        m.put(GgufKeys.GENERAL_SAMPLING_MIROSTAT, 2);
        m.put(GgufKeys.GENERAL_SAMPLING_MIROSTAT_TAU, 5.0f);
        m.put(GgufKeys.GENERAL_SAMPLING_MIROSTAT_ETA, 0.1f);
        m.put(GgufKeys.GENERAL_SAMPLING_XTC_PROBABILITY, 0.5f);
        m.put(GgufKeys.GENERAL_SAMPLING_XTC_THRESHOLD, 0.1f);
        assertSame(SamplingDefaults.NONE, read(m));
    }

    @Test
    public void builderValidatesLikeTheSampler() {
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().topP(1.5f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().typicalP(-0.1f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().temp(Float.NaN).build());
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().penaltyRepeat(-1f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().penaltyLastN(-2).build());
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().maxNewTokens(0).build());
        assertEquals(0.5f, SamplingDefaults.builder().typicalP(0.5f).build().typicalP(), 0f);
        assertTrue(SamplingDefaults.builder().build().isEmpty());
    }

    // ---- opt-out

    private static EngineModel modelWith(SamplingDefaults defaults, AtomicInteger calls) {
        return (EngineModel) Proxy.newProxyInstance(EngineModel.class.getClassLoader(),
                new Class<?>[]{EngineModel.class}, (proxy, method, args) -> {
                    if (method.getName().equals("modelSamplingDefaults")) {
                        calls.incrementAndGet();
                        return defaults;
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
    }

    @Test
    public void modelDefaultsApplyByDefault() {
        SamplingDefaults d = SamplingDefaults.builder().temp(0.3f).maxNewTokens(64).build();
        AtomicInteger calls = new AtomicInteger();
        assertTrue(EngineConfig.defaults().modelSamplingDefaults());
        assertSame(d, SamplingDefaults.forModel(modelWith(d, calls), EngineConfig.defaults()));
        assertEquals(1, calls.get());
    }

    @Test
    public void optOutIgnoresTheModelDefaults() {
        SamplingDefaults d = SamplingDefaults.builder().temp(0.3f).build();
        AtomicInteger calls = new AtomicInteger();
        EngineConfig off = EngineConfig.builder().modelSamplingDefaults(false).build();
        assertSame(SamplingDefaults.NONE, SamplingDefaults.forModel(modelWith(d, calls), off));
        assertEquals(0, calls.get());
        assertFalse(off.toBuilder().build().modelSamplingDefaults());
        assertTrue(off.toString().contains("modelSamplingDefaults=false"));
    }

    @Test
    public void optOutProperty() {
        Properties p = new Properties();
        assertTrue(EngineConfig.fromProperties(p).modelSamplingDefaults());
        p.setProperty(EngineConfig.PROP_MODEL_SAMPLING_DEFAULTS, "false");
        assertFalse(EngineConfig.fromProperties(p).modelSamplingDefaults());
        p.setProperty(EngineConfig.PROP_MODEL_SAMPLING_DEFAULTS, "TRUE");
        assertTrue(EngineConfig.fromProperties(p).modelSamplingDefaults());
        p.setProperty(EngineConfig.PROP_MODEL_SAMPLING_DEFAULTS, "off");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage().contains(EngineConfig.PROP_MODEL_SAMPLING_DEFAULTS));
    }
}
