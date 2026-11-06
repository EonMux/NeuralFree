package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The per-side patch merge read the way clip.cpp load_hparams reads n_merge. */
public class PatchMergeRuleTest {

    private static GgufKv kv(Object... pairs) {
        Map<String, Object> m = new HashMap<>();
        for (int i = 0; i < pairs.length; i += 2) {
            m.put((String) pairs[i], pairs[i + 1]);
        }
        return GgufKv.ofMap(m);
    }

    @Test
    public void aFixedMergeReadsNoKey() {
        PatchMergeRule rule = PatchMergeRule.fixed(2);
        assertEquals(2, rule.read(kv("clip.vision.spatial_merge_size", 4)));
        assertNull(rule.key());
        assertTrue(rule.toString().contains("fixed"));
    }

    @Test
    public void anOptionalKeyFallsBackToTheFamilyDefault() {
        PatchMergeRule rule = PatchMergeRule.spatialMergeSize(2);
        assertEquals(2, rule.read(kv()));
        assertEquals(3, rule.read(kv("clip.vision.spatial_merge_size", 3)));
        assertEquals(2, rule.read(kv("clip.vision.projector.scale_factor", 4)));
        assertEquals(4, PatchMergeRule.projectorScaleFactor(1).read(kv("clip.vision.projector.scale_factor", 4)));
    }

    @Test
    public void aRequiredOrNonPositiveMergeIsRefusedByKey() {
        PatchMergeRule rule = PatchMergeRule.required("clip.vision.projector.scale_factor");
        assertTrue(assertThrows(GgufFormatException.class, () -> rule.read(kv())).getMessage()
                .contains("clip.vision.projector.scale_factor"));
        assertEquals(2, rule.read(kv("clip.vision.projector.scale_factor", 2)));
        assertTrue(assertThrows(GgufFormatException.class,
                () -> PatchMergeRule.spatialMergeSize(2).read(kv("clip.vision.spatial_merge_size", 0)))
                .getMessage().contains("clip.vision.spatial_merge_size"));
        assertThrows(IllegalArgumentException.class, () -> PatchMergeRule.fixed(0));
        assertThrows(IllegalArgumentException.class, () -> PatchMergeRule.optional(" ", 1));
    }
}
