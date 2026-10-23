package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The Q/K norm geometry both attention mixers share: rows and eps per kind, weight-row tiling, per-head scalars, and
 * which projections each graph normalizes (bert.cpp:115-129, mpt.cpp:106-118, k2-horizon.cpp:255-261).
 */
public class QkNormGeometryTest {

    private static final float NORM_EPS = 1e-5f;
    private static final float RMS_EPS = 1e-6f;

    private static QkNormGeometry of(QkNormKind kind, int headDim, int nHeads) {
        return QkNormGeometry.of(kind, headDim, nHeads, NORM_EPS, RMS_EPS, 3);
    }

    @Test
    public void perHeadKindsNormalizeEachHead() {
        QkNormGeometry rms = of(QkNormKind.RMS_HEAD, 64, 8);
        assertEquals(NormKind.RMS, rms.normKind());
        assertEquals(RMS_EPS, rms.eps(), 0f);
        assertEquals(8, rms.rowsPerToken());
        assertEquals(64, rms.rowLen());
        QkNormGeometry ln = of(QkNormKind.LN_HEAD, 64, 8);
        assertEquals(NormKind.LAYER, ln.normKind());
        assertEquals(NORM_EPS, ln.eps(), 0f);
        assertEquals(8, ln.rowsPerToken());
    }

    @Test
    public void splitHalfNormalizesHalfHeadsWithTwoWeightRows() {
        QkNormGeometry g = of(QkNormKind.RMS_SPLIT_HALF, 64, 4);
        assertEquals(8, g.rowsPerToken());
        assertEquals(32, g.rowLen());
        assertEquals(2, g.weightRows("attn_q_norm.weight", 64));
        assertThrows(IllegalArgumentException.class, () -> g.weightRows("attn_q_norm.weight", 32));
        assertThrows(IllegalArgumentException.class, () -> of(QkNormKind.RMS_SPLIT_HALF, 63, 4));
    }

    @Test
    public void fullKindsNormalizeTheWholeProjection() {
        QkNormGeometry rms = of(QkNormKind.RMS_FULL, 64, 8);
        assertEquals(NormKind.RMS, rms.normKind());
        assertEquals(1, rms.rowsPerToken());
        assertEquals(512, rms.rowLen());
        QkNormGeometry ln = of(QkNormKind.LN_FULL, 64, 2);
        assertEquals(NormKind.LAYER, ln.normKind());
        assertEquals(NORM_EPS, ln.eps(), 0f);
        assertEquals(1, ln.rowsPerToken());
        assertEquals(128, ln.rowLen());
        assertEquals(1, ln.weightRows("attn_k_norm.weight", 128));
        ln.checkBias("attn_k_norm.bias", 128, 1);
        assertThrows(IllegalArgumentException.class, () -> ln.checkBias("attn_k_norm.bias", 64, 1));
        assertThrows(IllegalArgumentException.class, () -> ln.weightRows("attn_k_norm.weight", 64));
    }

    @Test
    public void weightRowsTileTheRowsPerToken() {
        QkNormGeometry g = of(QkNormKind.RMS_HEAD, 16, 4);
        assertEquals(1, g.weightRows("w", 16));
        assertEquals(4, g.weightRows("w", 64));
        assertEquals(2, g.weightRows("w", 32));
        assertThrows(IllegalArgumentException.class, () -> g.weightRows("w", 48));
        assertThrows(IllegalArgumentException.class, () -> g.weightRows("w", 20));
    }

    @Test
    public void perHeadScalarsAreRecognizedOnlyForPerHeadKinds() {
        assertTrue(of(QkNormKind.RMS_HEAD, 16, 4).perHeadScalar(1, 4));
        assertFalse(of(QkNormKind.RMS_HEAD, 16, 4).perHeadScalar(16, 1));
        assertFalse(of(QkNormKind.RMS_FULL, 16, 4).perHeadScalar(1, 4));
    }

    @Test
    public void invalidGeometryIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> of(QkNormKind.NONE, 64, 4));
        assertThrows(IllegalArgumentException.class, () -> of(QkNormKind.RMS_HEAD, 0, 4));
        assertThrows(IllegalArgumentException.class, () -> of(QkNormKind.RMS_HEAD, 64, 0));
    }

    @Test
    public void bertNormalizesEachProjectionWhoseWeightExists() {
        assertTrue(QkNormGeometry.applies(QkNormKind.LN_FULL, Arch.JINA_BERT_V2, false, true, false));
        assertFalse(QkNormGeometry.applies(QkNormKind.LN_FULL, Arch.JINA_BERT_V2, true, true, false));
        assertFalse(QkNormGeometry.applies(QkNormKind.LN_FULL, Arch.JINA_BERT_V2, false, false, true));
        assertTrue(QkNormGeometry.applies(QkNormKind.LN_FULL, Arch.JINA_BERT_V2, true, false, true));
    }

    @Test
    public void mptNormalizesBothProjectionsWhenTheQueryNormExists() {
        assertTrue(QkNormGeometry.applies(QkNormKind.LN_FULL, Arch.MPT, false, true, false));
        assertTrue(QkNormGeometry.applies(QkNormKind.LN_FULL, Arch.MPT, true, true, false));
        assertFalse(QkNormGeometry.applies(QkNormKind.LN_FULL, Arch.MPT, true, false, true));
        assertFalse(QkNormGeometry.applies(QkNormKind.LN_FULL, Arch.MPT, false, false, true));
    }

    @Test
    public void k2HorizonChecksEachNormTensor() {
        assertTrue(QkNormGeometry.applies(QkNormKind.RMS_HEAD, Arch.K2_HORIZON, false, true, false));
        assertFalse(QkNormGeometry.applies(QkNormKind.RMS_HEAD, Arch.K2_HORIZON, true, true, false));
    }

    @Test
    public void graphsThatCheckEachTensorSkipTheMissingNorm() {
        for (Arch arch : new Arch[]{Arch.STABLELM, Arch.GLM4_MOE, Arch.STEP35}) {
            assertTrue(arch.toString(), QkNormGeometry.applies(QkNormKind.RMS_HEAD, arch, false, true, false));
            assertFalse(arch.toString(), QkNormGeometry.applies(QkNormKind.RMS_HEAD, arch, true, true, false));
        }
        assertFalse(QkNormGeometry.applies(QkNormKind.LN_HEAD, Arch.STABLELM, false, false, true));
        assertTrue(QkNormGeometry.applies(QkNormKind.LN_HEAD, Arch.STABLELM, true, false, true));
    }

    @Test
    public void talkieNormalizesItsKeysWithoutAWeight() {
        assertTrue(QkNormGeometry.applies(QkNormKind.RMS_HEAD, Arch.TALKIE, true, true, false));
    }

    @Test
    public void otherGraphsNormalizeBothProjections() {
        assertTrue(QkNormGeometry.applies(QkNormKind.RMS_HEAD, Arch.QWEN3, true, true, false));
        assertTrue(QkNormGeometry.applies(QkNormKind.RMS_HEAD, Arch.LLAMA, false, false, false));
        assertFalse(QkNormGeometry.applies(QkNormKind.NONE, Arch.QWEN3, false, true, true));
    }
}
