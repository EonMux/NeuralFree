package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link RopeOverrides}: the builder and property forms of the rope and YaRN options, their
 *  {@code %s.rope.*} metadata override list, and how {@code ModelSourceOptions} carries them. */
public class RopeOverridesTest {

    // ---------------------------------------------------------------- defaults and the builder

    @Test
    public void defaultsSetNothing() {
        RopeOverrides defaults = RopeOverrides.defaults();
        assertTrue(defaults.isEmpty());
        assertTrue(defaults.toOverrides("llama").isEmpty());
        assertEquals(defaults, RopeOverrides.builder().build());
        assertEquals(defaults, defaults.toBuilder().build());
        assertTrue(RopeOverrides.fromProperties(new Properties()).isEmpty());
    }

    @Test
    public void builderSetsEveryValue() {
        RopeOverrides rope = RopeOverrides.builder()
                .scalingType(RopeOverrides.ScalingType.YARN)
                .freqScale(4.0f)
                .freqBase(10000.0f)
                .yarnExtFactor(1.13f)
                .yarnAttnFactor(1.4f)
                .yarnBetaFast(32.0f)
                .yarnBetaSlow(1.0f)
                .yarnOrigCtx(8192)
                .build();
        assertTrue(!rope.isEmpty());
        assertEquals(RopeOverrides.ScalingType.YARN, rope.scalingType());
        assertEquals(4.0f, rope.freqScale(), 0.0f);
        assertEquals(10000.0f, rope.freqBase(), 0.0f);
        assertEquals(1.13f, rope.yarnExtFactor(), 0.0f);
        assertEquals(1.4f, rope.yarnAttnFactor(), 0.0f);
        assertEquals(32.0f, rope.yarnBetaFast(), 0.0f);
        assertEquals(1.0f, rope.yarnBetaSlow(), 0.0f);
        assertEquals(8192, (int) rope.yarnOrigCtx());
    }

    /** {@code yarnOrigCtx(0)} keeps the model's training context and stores nothing. */
    @Test
    public void zeroOrigCtxStoresNothing() {
        assertTrue(RopeOverrides.builder().yarnOrigCtx(0).build().isEmpty());
        assertTrue(RopeOverrides.builder().yarnOrigCtx(0).yarnOrigCtx(4096).build().toOverrides("llama")
                .stream().anyMatch(o -> o.key().endsWith("original_context_length")
                        && ((Integer) o.value()).intValue() == 4096));
        assertTrue(RopeOverrides.builder().yarnOrigCtx(0).build().toOverrides("llama").stream()
                .noneMatch(o -> o.key().endsWith("original_context_length")));
    }

    @Test
    public void rejectsBadBuilderValues() {
        assertThrows(NullPointerException.class, () -> RopeOverrides.builder().scalingType(null));
        assertThrows(IllegalArgumentException.class, () -> RopeOverrides.builder().freqScale(0.0f / 0.0f));
        assertThrows(IllegalArgumentException.class, () -> RopeOverrides.builder().freqBase(1.0f / 0.0f));
        assertThrows(IllegalArgumentException.class, () -> RopeOverrides.builder().yarnOrigCtx(-1));
        assertThrows(IllegalArgumentException.class,
                () -> RopeOverrides.ScalingType.parse("logarithmic"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> RopeOverrides.ScalingType.parse("yarn "));
        assertTrue(e.getMessage().contains(RopeOverrides.PROP_ROPE_SCALING));
        assertEquals(RopeOverrides.ScalingType.LONG, RopeOverrides.ScalingType.parse("long"));
        assertEquals(RopeOverrides.ScalingType.NONE, RopeOverrides.ScalingType.parse("none"));
    }

    @Test
    public void buildersAreEqualAndRoundtrip() {
        RopeOverrides a = RopeOverrides.builder().scalingType(RopeOverrides.ScalingType.LINEAR).freqScale(2.0f).build();
        RopeOverrides b = a.toBuilder().build();
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertTrue(!a.equals(RopeOverrides.builder().freqScale(2.0f).freqBase(10000.0f).build()));
        assertTrue(a.toOverrides("llama").equals(b.toOverrides("llama")));
    }

    // ---------------------------------------------------------------- fromProperties

    @Test
    public void parsesProperties() {
        Properties p = new Properties();
        p.setProperty(RopeOverrides.PROP_ROPE_SCALING, "yarn");
        p.setProperty(RopeOverrides.PROP_ROPE_SCALE, "4.0");
        p.setProperty(RopeOverrides.PROP_ROPE_FREQ_BASE, "10000.0");
        p.setProperty(RopeOverrides.PROP_YARN_EXT_FACTOR, "1.13");
        p.setProperty(RopeOverrides.PROP_YARN_ATTN_FACTOR, "1.4");
        p.setProperty(RopeOverrides.PROP_YARN_BETA_FAST, "32.0");
        p.setProperty(RopeOverrides.PROP_YARN_BETA_SLOW, "1.0");
        p.setProperty(RopeOverrides.PROP_YARN_ORIG_CTX, "8192");
        RopeOverrides rope = RopeOverrides.fromProperties(p);
        assertEquals(RopeOverrides.ScalingType.YARN, rope.scalingType());
        assertEquals(4.0f, rope.freqScale(), 0.0f);
        assertEquals(10000.0f, rope.freqBase(), 0.0f);
        assertEquals(1.13f, rope.yarnExtFactor(), 0.0f);
        assertEquals(1.4f, rope.yarnAttnFactor(), 0.0f);
        assertEquals(32.0f, rope.yarnBetaFast(), 0.0f);
        assertEquals(1.0f, rope.yarnBetaSlow(), 0.0f);
        assertEquals(8192, (int) rope.yarnOrigCtx());
    }

    @Test
    public void ignoresBlankAndUnknownProperties() {
        Properties p = new Properties();
        p.setProperty(RopeOverrides.PROP_ROPE_SCALING, " ");
        assertTrue(RopeOverrides.fromProperties(p).isEmpty());
        p.setProperty("neuralfree.rope.unrelated", "v");
        assertTrue(RopeOverrides.fromProperties(p).isEmpty());
    }

    @Test
    public void rejectsBadPropertyValues() {
        Properties p = new Properties();
        p.setProperty(RopeOverrides.PROP_ROPE_SCALING, "quadratic");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> RopeOverrides.fromProperties(p));
        assertTrue(e.getMessage().contains(RopeOverrides.PROP_ROPE_SCALING));
        p.setProperty(RopeOverrides.PROP_ROPE_SCALING, "yarn");
        p.setProperty(RopeOverrides.PROP_ROPE_FREQ_SCALE, "much");
        e = assertThrows(IllegalArgumentException.class, () -> RopeOverrides.fromProperties(p));
        assertTrue(e.getMessage().contains(RopeOverrides.PROP_ROPE_FREQ_SCALE));
        p.remove(RopeOverrides.PROP_ROPE_FREQ_SCALE);
        p.setProperty(RopeOverrides.PROP_YARN_ORIG_CTX, "eight");
        e = assertThrows(IllegalArgumentException.class, () -> RopeOverrides.fromProperties(p));
        assertTrue(e.getMessage().contains(RopeOverrides.PROP_YARN_ORIG_CTX));
    }

    // ---------------------------------------------------------------- the %s.rope.* metadata overrides

    @Test
    public void toOverridesWritesTheRopeKeysOfTheArchitecture() {
        RopeOverrides rope = RopeOverrides.builder()
                .scalingType(RopeOverrides.ScalingType.YARN)
                .freqScale(4.0f)
                .freqBase(10000.0f)
                .yarnAttnFactor(1.4f)
                .yarnExtFactor(1.13f)
                .yarnBetaFast(32.0f)
                .yarnBetaSlow(1.0f)
                .yarnOrigCtx(8192)
                .build();
        List<MetadataOverrides.Override> ots = rope.toOverrides("qwen3");
        assertEquals(8, ots.size());
        assertEquals(String.format(GgufKeys.ROPE_SCALING_TYPE, "qwen3"), ots.get(0).key());
        assertEquals(GGUFValueType.STRING, ots.get(0).type());
        assertEquals("yarn", ots.get(0).value());
        assertEquals(String.format(GgufKeys.ROPE_SCALING_FACTOR, "qwen3"), ots.get(1).key());
        assertEquals(GGUFValueType.FLOAT32, ots.get(1).type());
        assertEquals(4.0f, ots.get(1).value());
        assertEquals(String.format(GgufKeys.ROPE_FREQ_BASE, "qwen3"), ots.get(2).key());
        assertEquals(GGUFValueType.FLOAT32, ots.get(2).type());
        assertEquals(String.format(GgufKeys.ROPE_SCALING_ATTN_FACTOR, "qwen3"), ots.get(3).key());
        assertEquals(String.format(GgufKeys.ROPE_SCALING_YARN_EXT_FACTOR, "qwen3"), ots.get(4).key());
        assertEquals(String.format(GgufKeys.ROPE_SCALING_YARN_BETA_FAST, "qwen3"), ots.get(5).key());
        assertEquals(String.format(GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, "qwen3"), ots.get(6).key());
        assertEquals(String.format(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, "qwen3"), ots.get(7).key());
        assertEquals(GGUFValueType.INT32, ots.get(7).type());
        assertEquals(8192, ots.get(7).value());

        // a null arch is an error; only the set options are written
        assertThrows(NullPointerException.class, () -> rope.toOverrides(null));
        assertTrue(RopeOverrides.builder().freqScale(2.0f).build().toOverrides("llama").size() == 1);
        assertEquals(String.format(GgufKeys.ROPE_SCALING_FACTOR, "llama"),
                RopeOverrides.builder().freqScale(2.0f).build().toOverrides("llama").get(0).key());
    }

    // ---------------------------------------------------------------- ModelSourceOptions carries them

    @Test
    public void flowsThroughModelSourceOptions() {
        ModelSourceOptions o = ModelSourceOptions.builder()
                .ropeScaling(RopeOverrides.ScalingType.YARN)
                .ropeFreqScale(4.0f)
                .ropeFreqBase(10000.0f)
                .yarnOrigCtx(8192)
                .build();
        assertEquals(RopeOverrides.ScalingType.YARN, o.ropeOverrides().scalingType());
        assertEquals(4.0f, o.ropeOverrides().freqScale(), 0.0f);
        assertEquals(10000.0f, o.ropeOverrides().freqBase(), 0.0f);
        assertEquals(8192, (int) o.ropeOverrides().yarnOrigCtx());

        ModelSourceOptions replacing = ModelSourceOptions.builder().ropeFreqBase(10000.0f).build()
                .toBuilder().ropeOverrides(RopeOverrides.builder().freqScale(2.0f)).build();
        assertTrue(replacing.ropeOverrides().freqBase() == null);
        assertEquals(2.0f, replacing.ropeOverrides().freqScale(), 0.0f);

        assertEquals(ModelSourceOptions.builder().build().ropeOverrides(), RopeOverrides.defaults());
    }

    @Test
    public void flowsThroughProperties() {
        Properties p = new Properties();
        p.setProperty(RopeOverrides.PROP_ROPE_SCALING, "linear");
        p.setProperty(RopeOverrides.PROP_ROPE_FREQ_SCALE, "4.0");
        p.setProperty(RopeOverrides.PROP_YARN_ORIG_CTX, "8192");
        ModelSourceOptions o = ModelSourceOptions.fromProperties(p);
        assertEquals(RopeOverrides.ScalingType.LINEAR, o.ropeOverrides().scalingType());
        assertEquals(4.0f, o.ropeOverrides().freqScale(), 0.0f);
        assertEquals(8192, (int) o.ropeOverrides().yarnOrigCtx());

        assertTrue(ModelSourceOptions.fromProperties(new Properties()).ropeOverrides().isEmpty());
    }
}
