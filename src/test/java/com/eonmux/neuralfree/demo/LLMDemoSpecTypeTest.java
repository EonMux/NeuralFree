package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code --spec-type} (llama.cpp common/arg.cpp {@code --spec-type}): auto, draft-mtp and draft-simple reach the
 * engine configuration, draft-mtp without {@code --draft} drafts with the target's own NextN / MTP head, and
 * contradictory or unknown values are usage errors.
 */
public class LLMDemoSpecTypeTest {

    private static final String[] SPECULATIVE_PROPS = {
            EngineConfig.PROP_DRAFT, EngineConfig.PROP_DRAFT_TYPE, EngineConfig.PROP_MTP
    };

    private final Map<String, String> saved = new HashMap<>();

    @Before
    public void clearSpeculativeProperties() {
        for (String key : SPECULATIVE_PROPS) {
            saved.put(key, System.getProperty(key));
            System.clearProperty(key);
        }
    }

    @After
    public void restoreSpeculativeProperties() {
        for (Map.Entry<String, String> e : saved.entrySet()) {
            if (e.getValue() == null) {
                System.clearProperty(e.getKey());
            } else {
                System.setProperty(e.getKey(), e.getValue());
            }
        }
    }

    private static EngineConfig engine(String... args) throws UsageException {
        return LLMDemoOptions.parse(args).engine;
    }

    @Test
    public void detectionIsTheDefault() throws UsageException {
        assertEquals(DraftType.AUTO, engine("--draft", "draft.gguf").draftType());
    }

    @Test
    public void draftSimpleRunsTheDraftFileAsASeparateModel() throws UsageException {
        EngineConfig c = engine("--draft", "draft.gguf", "--spec-type", "draft-simple");
        assertEquals(DraftType.MODEL, c.draftType());
        assertEquals("draft.gguf", c.draftModelPath());
        assertFalse(c.mtp());
    }

    @Test
    public void draftMtpWithoutADraftFileUsesTheTargetsOwnHead() throws UsageException {
        EngineConfig own = engine("--spec-type", "draft-mtp");
        assertTrue(own.mtp());
        assertTrue(own.speculative());
        EngineConfig file = engine("--draft", "draft.gguf", "--spec-type", "draft-mtp");
        assertEquals(DraftType.MTP, file.draftType());
        assertFalse(file.mtp());
    }

    @Test
    public void unknownOrContradictoryTypesAreUsageErrors() {
        UsageException unknown = assertThrows(UsageException.class, () -> engine("--spec-type", "ngram-cache"));
        assertTrue(unknown.getMessage(), unknown.getMessage().startsWith("--spec-type"));
        assertThrows(UsageException.class, () -> engine("--spec-type"));
        assertThrows(UsageException.class, () -> engine("--mtp", "--spec-type", "draft-simple"));
    }

    @Test
    public void thePropertyIsTheDefaultTheFlagOverrides() throws UsageException {
        System.setProperty(EngineConfig.PROP_DRAFT_TYPE, "draft-simple");
        assertEquals(DraftType.MODEL, engine("--draft", "draft.gguf").draftType());
        assertEquals(DraftType.AUTO, engine("--draft", "draft.gguf", "--spec-type", "auto").draftType());
    }
}
