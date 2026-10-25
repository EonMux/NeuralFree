package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.SamplerConfig.Setting;
import org.junit.Test;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Properties;
import java.util.function.ToDoubleFunction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Precedence between the settings a caller gives and a model's sampling defaults: a setting given by a property or
 * a {@link SamplerConfig.Builder} setter is explicit and kept; a default or a {@code fallback*} setter is a fallback
 * the model's value replaces ({@link SamplerConfig#withModelDefaults}). The token budget follows the same rule
 * ({@link GenerationRequest#withModelDefaults}).
 */
public class SamplerConfigModelDefaultsTest {

    /** A model default for every setting, each different from the llama.cpp default and the caller's value. */
    private static final SamplingDefaults MODEL = SamplingDefaults.builder().temp(0.3f).topK(7).topP(0.5f)
            .minP(0.2f).typicalP(0.6f).penaltyRepeat(1.3f).penaltyLastN(16).presencePenalty(0.7f)
            .frequencyPenalty(0.4f).maxNewTokens(99).build();

    /** Gives one setting a value through a builder. */
    private interface Setter {
        void set(SamplerConfig.Builder b, double v);
    }

    /** One setting: its property, the caller's and the model's values, how it is set and read back. */
    private static final class Case {
        final Setting setting;
        final String property;
        final double user;
        final double model;
        final Setter explicit;
        final Setter fallback;
        final ToDoubleFunction<SamplerConfig> get;

        Case(Setting setting, String property, double user, double model, Setter explicit, Setter fallback,
             ToDoubleFunction<SamplerConfig> get) {
            this.setting = setting;
            this.property = property;
            this.user = user;
            this.model = model;
            this.explicit = explicit;
            this.fallback = fallback;
            this.get = get;
        }

        String propertyValue() {
            return setting == Setting.TOP_K || setting == Setting.PENALTY_LAST_N
                    ? String.valueOf((int) user) : String.valueOf((float) user);
        }
    }

    private static List<Case> cases() {
        List<Case> c = new ArrayList<>();
        c.add(new Case(Setting.TEMPERATURE, SamplerConfig.PROP_TEMPERATURE, 1.25, 0.3,
                (b, v) -> b.temperature((float) v), (b, v) -> b.fallbackTemperature((float) v),
                SamplerConfig::temperature));
        c.add(new Case(Setting.TOP_K, SamplerConfig.PROP_TOP_K, 3, 7,
                (b, v) -> b.topK((int) v), (b, v) -> b.fallbackTopK((int) v), SamplerConfig::topK));
        c.add(new Case(Setting.TOP_P, SamplerConfig.PROP_TOP_P, 0.75, 0.5,
                (b, v) -> b.topP((float) v), (b, v) -> b.fallbackTopP((float) v), SamplerConfig::topP));
        c.add(new Case(Setting.MIN_P, SamplerConfig.PROP_MIN_P, 0.125, 0.2,
                (b, v) -> b.minP((float) v), (b, v) -> b.fallbackMinP((float) v), SamplerConfig::minP));
        c.add(new Case(Setting.TYPICAL_P, SamplerConfig.PROP_TYPICAL_P, 0.875, 0.6,
                (b, v) -> b.typicalP((float) v), (b, v) -> b.fallbackTypicalP((float) v), SamplerConfig::typicalP));
        c.add(new Case(Setting.REPEAT_PENALTY, SamplerConfig.PROP_REPEAT_PENALTY, 1.5, 1.3,
                (b, v) -> b.repeatPenalty((float) v), (b, v) -> b.fallbackRepeatPenalty((float) v),
                SamplerConfig::repeatPenalty));
        c.add(new Case(Setting.PENALTY_LAST_N, SamplerConfig.PROP_PENALTY_LAST_N, 32, 16,
                (b, v) -> b.penaltyLastN((int) v), (b, v) -> b.fallbackPenaltyLastN((int) v),
                SamplerConfig::penaltyLastN));
        c.add(new Case(Setting.PRESENCE_PENALTY, SamplerConfig.PROP_PRESENCE_PENALTY, 0.25, 0.7,
                (b, v) -> b.presencePenalty((float) v), (b, v) -> b.fallbackPresencePenalty((float) v),
                SamplerConfig::presencePenalty));
        c.add(new Case(Setting.FREQUENCY_PENALTY, SamplerConfig.PROP_FREQUENCY_PENALTY, -0.5, 0.4,
                (b, v) -> b.frequencyPenalty((float) v), (b, v) -> b.fallbackFrequencyPenalty((float) v),
                SamplerConfig::frequencyPenalty));
        return c;
    }

    private static void assertValue(String what, double expected, Case c, SamplerConfig s) {
        assertEquals(what + " " + c.setting, (float) expected, (float) c.get.applyAsDouble(s), 0f);
    }

    /** Every case other than {@code given} carries the model's value. */
    private static void assertOthersTakeTheModel(Case given, SamplerConfig s) {
        for (Case o : cases()) {
            if (o.setting != given.setting) {
                assertValue("model value", o.model, o, s);
            }
        }
    }

    @Test
    public void everySettingIsCovered() {
        EnumSet<Setting> covered = EnumSet.noneOf(Setting.class);
        for (Case c : cases()) {
            covered.add(c.setting);
        }
        assertEquals(EnumSet.allOf(Setting.class), covered);
    }

    @Test
    public void unsetSettingsTakeTheModelDefaults() {
        SamplerConfig s = SamplerConfig.defaults().withModelDefaults(MODEL);
        for (Case c : cases()) {
            assertValue("model value", c.model, c, s);
        }
        assertTrue(s.explicit().isEmpty());
    }

    @Test
    public void builderSettingsWinOverTheModel() {
        for (Case c : cases()) {
            SamplerConfig.Builder b = SamplerConfig.builder();
            c.explicit.set(b, c.user);
            SamplerConfig given = b.build();
            assertTrue(given.isExplicit(c.setting));
            SamplerConfig s = given.withModelDefaults(MODEL);
            assertValue("builder value", c.user, c, s);
            assertOthersTakeTheModel(c, s);
            assertEquals(EnumSet.of(c.setting), s.explicit());
        }
    }

    @Test
    public void propertySettingsWinOverTheModel() {
        for (Case c : cases()) {
            Properties p = new Properties();
            p.setProperty(c.property, c.propertyValue());
            SamplerConfig given = SamplerConfig.fromProperties(p);
            assertEquals(EnumSet.of(c.setting), given.explicit());
            SamplerConfig s = given.withModelDefaults(MODEL);
            assertValue("property value", c.user, c, s);
            assertOthersTakeTheModel(c, s);
        }
    }

    @Test
    public void blankPropertiesAreNotExplicit() {
        Properties p = new Properties();
        p.setProperty(SamplerConfig.PROP_TEMPERATURE, "  ");
        p.setProperty(SamplerConfig.PROP_SEED, "7");
        p.setProperty(SamplerConfig.PROP_GREEDY, "false");
        SamplerConfig s = SamplerConfig.fromProperties(p);
        assertTrue(s.explicit().isEmpty());
        assertEquals(7L, s.seed());
        assertEquals(0.3f, s.withModelDefaults(MODEL).temperature(), 0f);
    }

    @Test
    public void fallbackSettingsGiveWayToTheModel() {
        for (Case c : cases()) {
            SamplerConfig.Builder b = SamplerConfig.builder();
            c.fallback.set(b, c.user);
            SamplerConfig given = b.build();
            assertFalse(given.isExplicit(c.setting));
            assertValue("fallback value", c.user, c, given);
            assertValue("fallback without model defaults", c.user, c, given.withModelDefaults(SamplingDefaults.NONE));
            assertValue("model value over fallback", c.model, c, given.withModelDefaults(MODEL));
        }
    }

    @Test
    public void theLastSetterDecidesWhetherASettingIsExplicit() {
        SamplerConfig fallbackLast = SamplerConfig.builder().temperature(1.25f).fallbackTemperature(0.9f).build();
        assertFalse(fallbackLast.isExplicit(Setting.TEMPERATURE));
        assertEquals(0.3f, fallbackLast.withModelDefaults(MODEL).temperature(), 0f);

        SamplerConfig explicitLast = SamplerConfig.builder().fallbackTemperature(0.9f).temperature(1.25f).build();
        assertTrue(explicitLast.isExplicit(Setting.TEMPERATURE));
        assertEquals(1.25f, explicitLast.withModelDefaults(MODEL).temperature(), 0f);
    }

    @Test
    public void explicitSettingsSurviveCopies() {
        SamplerConfig s = SamplerConfig.builder().topK(3).fallbackTopP(0.75f).build();
        assertEquals(EnumSet.of(Setting.TOP_K), s.withSeed(11L).explicit());
        assertEquals(EnumSet.of(Setting.TOP_K), s.toBuilder().build().explicit());
        assertEquals(0.75f, s.toBuilder().build().topP(), 0f);
        SamplerConfig merged = s.withModelDefaults(MODEL);
        assertEquals(3, merged.topK());
        assertEquals(0.5f, merged.topP(), 0f);
        assertEquals(EnumSet.of(Setting.TOP_K), merged.explicit());
    }

    @Test
    public void explicitSetIsUnmodifiable() {
        SamplerConfig s = SamplerConfig.builder().topK(3).build();
        assertThrows(UnsupportedOperationException.class, () -> s.explicit().add(Setting.TOP_P));
    }

    @Test
    public void emptyDefaultsLeaveTheSamplerUnchanged() {
        SamplerConfig s = SamplerConfig.builder().fallbackTemperature(0.9f).seed(5L).build();
        assertSame(s, s.withModelDefaults(SamplingDefaults.NONE));
    }

    @Test
    public void seedAndGreedyAreNotModelSettings() {
        SamplerConfig s = SamplerConfig.greedyDecoding().withSeed(3L).withModelDefaults(MODEL);
        assertTrue(s.isGreedy());
        assertEquals(3L, s.seed());
        assertEquals(0.3f, s.temperature(), 0f);
    }

    @Test
    public void wholeContextWindowFromTheModelIsKept() {
        SamplingDefaults d = SamplingDefaults.builder().penaltyLastN(SamplerConfig.PENALTY_WHOLE_CONTEXT).build();
        assertEquals(SamplerConfig.PENALTY_WHOLE_CONTEXT, SamplerConfig.defaults().withModelDefaults(d).penaltyLastN());
    }

    @Test
    public void legacyParametersAreExplicit() {
        SamplerConfig s = EngineConfig.defaults().legacySampler(1.25f, 0.75f, 1.5f, 32, 0.125f, 3)
                .withModelDefaults(MODEL);
        assertEquals(1.25f, s.temperature(), 0f);
        assertEquals(0.75f, s.topP(), 0f);
        assertEquals(1.5f, s.repeatPenalty(), 0f);
        assertEquals(32, s.penaltyLastN());
        assertEquals(0.125f, s.minP(), 0f);
        assertEquals(3, s.topK());
        assertEquals(0.6f, s.typicalP(), 0f);
        assertEquals(0.7f, s.presencePenalty(), 0f);
        assertEquals(0.4f, s.frequencyPenalty(), 0f);
    }

    // ---- token budget

    @Test
    public void unsetTokenBudgetTakesTheModelDefault() {
        GenerationRequest r = GenerationRequest.defaults();
        assertFalse(r.maxNewTokensExplicit());
        GenerationRequest merged = r.withModelDefaults(MODEL);
        assertEquals(99, merged.maxNewTokens());
        assertFalse(merged.maxNewTokensExplicit());
        assertEquals(0.3f, merged.sampler().temperature(), 0f);
    }

    @Test
    public void unsetTokenBudgetWithoutAModelDefaultStays512() {
        GenerationRequest r = GenerationRequest.defaults();
        assertSame(r, r.withModelDefaults(SamplingDefaults.NONE));
        SamplingDefaults noBudget = SamplingDefaults.builder().temp(0.3f).build();
        assertEquals(GenerationRequest.DEFAULT_MAX_NEW_TOKENS, r.withModelDefaults(noBudget).maxNewTokens());
        assertEquals(512, GenerationRequest.DEFAULT_MAX_NEW_TOKENS);
    }

    @Test
    public void explicitTokenBudgetWinsOverTheModel() {
        GenerationRequest r = GenerationRequest.builder().maxNewTokens(64).build();
        assertTrue(r.maxNewTokensExplicit());
        assertEquals(64, r.withModelDefaults(MODEL).maxNewTokens());
        GenerationRequest unlimited = GenerationRequest.builder().maxNewTokens(GenerationRequest.UNLIMITED).build();
        assertTrue(unlimited.withModelDefaults(MODEL).unlimited());
    }

    @Test
    public void fallbackTokenBudgetGivesWayToTheModel() {
        GenerationRequest r = GenerationRequest.builder().fallbackMaxNewTokens(2560).build();
        assertFalse(r.maxNewTokensExplicit());
        assertEquals(99, r.withModelDefaults(MODEL).maxNewTokens());
        assertEquals(2560, r.withModelDefaults(SamplingDefaults.builder().topK(7).build()).maxNewTokens());
        SamplingDefaults unlimited = SamplingDefaults.builder().maxNewTokens(GenerationRequest.UNLIMITED).build();
        assertTrue(r.withModelDefaults(unlimited).unlimited());
    }

    @Test
    public void tokenBudgetStateSurvivesCopies() {
        assertTrue(GenerationRequest.builder().maxNewTokens(64).build().toBuilder().build().maxNewTokensExplicit());
        GenerationRequest fallback = GenerationRequest.builder().fallbackMaxNewTokens(64).build();
        assertFalse(fallback.toBuilder().build().maxNewTokensExplicit());
        assertEquals(64, fallback.toBuilder().build().maxNewTokens());
    }

    @Test
    public void requestKeepsItsOtherFieldsWhenMerged() {
        GenerationRequest r = GenerationRequest.builder().sampler(SamplerConfig.builder().topK(3).build())
                .stopStrings("User:").ignoreEos(true).cancellation(() -> false).build();
        GenerationRequest merged = r.withModelDefaults(MODEL);
        assertEquals(3, merged.sampler().topK());
        assertEquals(0.5f, merged.sampler().topP(), 0f);
        assertEquals(r.stopStrings(), merged.stopStrings());
        assertTrue(merged.ignoreEos());
        assertTrue(merged.cancellable());
        assertSame(r.cancellation(), merged.cancellation());
    }
}
