package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Row addressing, sizing, windows and rotation of KV caches: ring and full layouts, Q8_0 row bytes, the MLA latent
 * alias, ring sizing, the SWA visibility rules of llama.cpp {@code is_masked_swa}, and the llama.cpp rotation-plan
 * rules.
 */
public class KvLayoutTest {

    private static final KvWindow SWA_1024 = KvWindow.of(SwaType.STANDARD, 1024, true);

    @Test
    public void ringRowOfWrapsModuloRows() {
        KvLayout ring = KvLayout.of(1280, true, 1024, 1024, KvDType.F16, KvDType.F16, SWA_1024);
        assertEquals(0, ring.rowOf(0));
        assertEquals(1279, ring.rowOf(1279));
        assertEquals(0, ring.rowOf(1280));
        assertEquals(1160, ring.rowOf(5000));
        assertThrows(IllegalArgumentException.class, () -> ring.rowOf(-1));
    }

    @Test
    public void fullCacheRowOfIsIdentityWithinContext() {
        KvLayout full = KvLayout.of(4096, false, 512, 512, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        assertEquals(0, full.rowOf(0));
        assertEquals(4095, full.rowOf(4095));
        assertThrows(IllegalArgumentException.class, () -> full.rowOf(4096));
    }

    @Test
    public void rowBytesFollowTheElementType() {
        KvLayout l = KvLayout.of(16, false, 1024, 256, KvDType.Q8_0, KvDType.F16, KvWindow.FULL_CAUSAL);
        assertEquals(1024 / 32 * 34, l.rowBytesK());
        assertEquals(256 * 2, l.rowBytesV());
        assertEquals(16L * (1024 / 32 * 34), l.kBytes());
        assertEquals(16L * 512, l.vBytes());
        assertEquals(l.kBytes() + l.vBytes(), l.totalBytes());
        assertThrows(IllegalArgumentException.class,
                () -> KvLayout.of(16, false, 100, 100, KvDType.Q8_0, KvDType.F16, KvWindow.FULL_CAUSAL));
    }

    @Test
    public void latentLayoutAliasesVIntoTheKRow() {
        KvLayout latent = KvLayout.latent(64, false, 512 + 64, 512, KvDType.F16, KvWindow.FULL_CAUSAL);
        assertTrue(latent.vAliasesK());
        assertEquals(0L, latent.vBytes());
        assertEquals(latent.rowBytesK(), latent.rowBytesV());
        assertThrows(IllegalArgumentException.class,
                () -> KvLayout.latent(64, false, 512, 576, KvDType.F16, KvWindow.FULL_CAUSAL));
    }

    @Test
    public void ringRowsPadSwaPlusUbatchTo256AndClampToContext() {
        assertEquals(1536, KvLayout.ringRows(8192, 1024, 512, false));
        assertEquals(4352, KvLayout.ringRows(32768, 4096, 1, false));
        assertEquals(1000, KvLayout.ringRows(1000, 1024, 512, false));
        assertEquals(8192, KvLayout.ringRows(8192, 1024, 512, true));
    }

    @Test
    public void ringNeedsACausalWindowItCanHold() {
        assertThrows(IllegalArgumentException.class,
                () -> KvLayout.of(1280, true, 64, 64, KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL));
        assertThrows(IllegalArgumentException.class,
                () -> KvLayout.of(512, true, 64, 64, KvDType.F16, KvDType.F16, SWA_1024));
    }

    @Test
    public void standardWindowStartsAtPosMinusSwaPlusOne() {
        assertEquals(0, SWA_1024.firstVisible(0));
        assertEquals(0, SWA_1024.firstVisible(1023));
        assertEquals(1, SWA_1024.firstVisible(1024));
        assertEquals(3977, SWA_1024.firstVisible(5000));
        assertTrue(SWA_1024.visible(5000, 3977));
        assertFalse(SWA_1024.visible(5000, 3976));
        assertFalse(SWA_1024.visible(5000, 5001));
    }

    @Test
    public void chunkedAndSymmetricWindowsMatchLlamaCpp() {
        KvWindow chunked = KvWindow.of(SwaType.CHUNKED, 8192, true);
        assertEquals(8192, chunked.firstVisible(10000));
        assertFalse(chunked.visible(10000, 8191));
        assertTrue(chunked.visible(10000, 8192));

        KvWindow symmetric = KvWindow.of(SwaType.SYMMETRIC, 128, false);
        assertTrue(symmetric.visible(100, 164));
        assertFalse(symmetric.visible(100, 165));
        assertTrue(symmetric.visible(100, 36));
        assertFalse(symmetric.visible(100, 35));
        assertEquals(164, symmetric.lastVisible(100));
    }

    @Test
    public void rotationPlanFollowsLlamaCppRules() {
        RotationPlan q8 = RotationPlan.of(KvDType.Q8_0, KvDType.Q8_0, 128, 128, 128, 128, true, false);
        assertEquals(128, q8.nRotK());
        assertEquals(64, q8.nRotV());

        RotationPlan mixedDims = RotationPlan.of(KvDType.Q8_0, KvDType.F16, 192, 128, 192, 128, true, false);
        assertEquals(64, mixedDims.nRotK());
        assertFalse(mixedDims.rotateV());

        assertSame(RotationPlan.NONE, RotationPlan.of(KvDType.F16, KvDType.F16, 128, 128, 128, 128, true, false));
        assertSame(RotationPlan.NONE, RotationPlan.of(KvDType.Q8_0, KvDType.Q8_0, 128, 128, 128, 128, false, false));
        assertSame(RotationPlan.NONE, RotationPlan.of(KvDType.Q8_0, KvDType.Q8_0, -1, -1, 128, 128, true, false));

        RotationPlan forced = RotationPlan.of(KvDType.F16, KvDType.F16, 128, 128, 128, 128, false, true);
        assertEquals(128, forced.nRotK());
        assertFalse(forced.rotateV());
    }

    @Test
    public void kvTypeParsingNamesTheProperty() {
        assertEquals(KvDType.Q8_0, KvDType.parse("q8_0", "neuralfree.kv.type"));
        assertEquals(KvDType.F16, KvDType.parse(" f16 ", "neuralfree.kv.type"));
        assertEquals(KvDType.Q4_0, KvDType.parse("q4_0", "neuralfree.kv.typeK"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> KvDType.parse("q4_k", "neuralfree.kv.typeK"));
        assertTrue(e.getMessage().contains("neuralfree.kv.typeK"));
        assertTrue(e.getMessage().contains("iq4_nl"));
    }
}
