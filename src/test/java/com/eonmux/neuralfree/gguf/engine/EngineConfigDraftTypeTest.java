package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EngineConfig#draftType()}: llama.cpp's {@code --spec-type} names and aliases through
 * {@code neuralfree.draft.type}, automatic detection by default, and {@code draft-mtp} without a draft file drafting
 * with the model's own NextN / MTP blocks as llama.cpp does.
 */
public class EngineConfigDraftTypeTest {

    private static EngineConfig fromProperty(String value) {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_DRAFT_TYPE, value);
        return EngineConfig.fromProperties(p);
    }

    @Test
    public void detectionIsTheDefault() {
        assertEquals(DraftType.AUTO, EngineConfig.defaults().draftType());
        assertEquals(DraftType.AUTO, EngineConfig.fromProperties(new Properties()).draftType());
    }

    @Test
    public void llamaCppNamesAndAliasesParse() {
        assertEquals(DraftType.AUTO, fromProperty("auto").draftType());
        assertEquals(DraftType.MTP, DraftType.parse("draft-mtp"));
        assertEquals(DraftType.MTP, DraftType.parse("MTP"));
        assertEquals(DraftType.MODEL, DraftType.parse("draft-simple"));
        assertEquals(DraftType.MODEL, DraftType.parse(" simple "));
        assertEquals(DraftType.MODEL, DraftType.parse("model"));
        assertEquals("draft-simple", DraftType.MODEL.llamaName());
        assertEquals("draft-mtp", DraftType.MTP.llamaName());
    }

    @Test
    public void otherValuesAreRefusedNamingTheProperty() {
        IllegalArgumentException bad = assertThrows(IllegalArgumentException.class, () -> fromProperty("ngram"));
        assertTrue(bad.getMessage(), bad.getMessage().contains(EngineConfig.PROP_DRAFT_TYPE));
        assertTrue(bad.getMessage(), bad.getMessage().contains("draft-simple"));
        IllegalArgumentException eagle = assertThrows(IllegalArgumentException.class,
                () -> DraftType.parse("draft-eagle3"));
        assertTrue(eagle.getMessage(), eagle.getMessage().contains("auto"));
    }

    @Test
    public void draftMtpWithoutADraftFileDraftsWithTheModelsOwnHead() {
        EngineConfig c = EngineConfig.builder().draftType(DraftType.MTP).build();
        assertTrue(c.mtp());
        assertTrue(c.speculative());
        assertTrue(c.memory().nextnMemory());
        assertTrue(c.memory().stateSnapshots());
        EngineConfig withFile = c.toBuilder().draftModelPath("/models/draft.gguf").build();
        assertFalse("a draft file replaces the implied own head", withFile.mtp());
        assertFalse(c.toBuilder().draftType(DraftType.AUTO).build().mtp());
        assertTrue(EngineConfig.builder().mtp(true).build().toBuilder().draftType(DraftType.AUTO).build().mtp());
    }

    @Test
    public void draftMtpWithADraftFileRunsThatFile() {
        EngineConfig c = EngineConfig.builder().draftModelPath("/models/draft.gguf").draftType(DraftType.MTP).build();
        assertFalse(c.mtp());
        assertEquals(DraftType.MTP, c.draftType());
        assertFalse(c.memory().nextnMemory());
    }

    @Test
    public void draftSimpleNeedsADraftFileRatherThanTheOwnHead() {
        assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.builder().draftType(DraftType.MODEL).mtp(true).build());
        EngineConfig c = EngineConfig.builder().draftModelPath("/models/draft.gguf").draftType(DraftType.MODEL)
                .build();
        assertEquals(DraftType.MODEL, c.draftType());
        assertEquals(DraftType.MODEL, c.toBuilder().build().draftType());
        assertTrue(c.toString(), c.toString().contains("draftType=draft-simple"));
    }
}
