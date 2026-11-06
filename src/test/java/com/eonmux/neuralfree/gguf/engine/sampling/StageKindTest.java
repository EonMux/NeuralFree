package com.eonmux.neuralfree.gguf.engine.sampling;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/** llama.cpp's sampler names, aliases and characters ({@code common_sampler_types_from_names/_chars}). */
public class StageKindTest {

    @Test
    public void everyLabelAndItsSpellingsNameTheKind() {
        for (StageKind kind : StageKind.values()) {
            assertEquals(kind, StageKind.ofName(kind.label()));
            assertEquals(kind, StageKind.ofName(kind.label().replace('_', '-').toUpperCase()));
            assertEquals(kind, StageKind.ofName(kind.label().replace("_", "")));
        }
        assertEquals(StageKind.TOP_P, StageKind.ofName("nucleus"));
        assertEquals(StageKind.TEMPERATURE, StageKind.ofName("temp"));
        assertEquals(StageKind.TYPICAL_P, StageKind.ofName("typ"));
        assertThrows(IllegalArgumentException.class, () -> StageKind.ofName("top_q"));
    }

    @Test
    public void theCharactersFollowLlamaCpp() {
        assertEquals(Arrays.asList(StageKind.DRY, StageKind.TOP_K, StageKind.TYPICAL_P, StageKind.TOP_P,
                StageKind.TOP_N_SIGMA, StageKind.MIN_P, StageKind.TEMPERATURE, StageKind.XTC, StageKind.INFILL,
                StageKind.PENALTIES, StageKind.ADAPTIVE_P), StageKind.ofChars("dkypsmtxiea"));
        assertThrows(IllegalArgumentException.class, () -> StageKind.ofChars("kq"));
    }
}
