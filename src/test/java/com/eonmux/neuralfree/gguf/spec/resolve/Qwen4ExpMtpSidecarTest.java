package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.Qwen4ExpSidecars;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * qwen4exp MTP-only sidecar files: the MTP block with no compress ratio takes the last trunk QSA layer's
 * (qwen4exp.cpp:141-156, ik_llama.cpp #2369), and a shared sidecar without its own {@code token_embd} or LM
 * head passes the loader checks and drafts over the target's (qwen4exp.cpp:274-300,
 * {@code llama_model_share_mtp_io}).
 */
public class Qwen4ExpMtpSidecarTest {

    private static final String ARCH = "qwen4exp";

    private static List<TensorRequirement> tensors() {
        return ArchRegistry.get(Arch.QWEN4EXP).tensors();
    }

    @Test
    public void anMtpBlockWithoutRatioTakesTheLastTrunkQsaRatio() {
        ModelHparams hp = Qwen4ExpSidecars.loadHparams(4, 0);
        assertArrayEquals(new int[] {4, 4}, hp.compressRatioArray());
        assertEquals(4, hp.indexerKpool());
    }

    @Test
    public void anExplicitMtpRatioIsKept() {
        ModelHparams hp = Qwen4ExpSidecars.loadHparams(4, 4);
        assertArrayEquals(new int[] {4, 4}, hp.compressRatioArray());
    }

    @Test
    public void aFileWithNoQsaRatioAnywhereStaysDense() {
        ModelHparams hp = Qwen4ExpSidecars.loadHparams(0, 0);
        assertArrayEquals(new int[] {0, 0}, hp.compressRatioArray());
        assertEquals(0, hp.indexerKpool());
    }

    @Test
    public void theInheritedRatioGivesTheNextnBlockItsQsaIndexer() {
        ModelSpec target = Qwen4ExpSidecars.buildTarget();
        ModelSpec draft = Qwen4ExpSidecars.buildSidecar(target, 4, 0, Qwen4ExpSidecars.sharedSidecarTensors());
        LayerSpec block = draft.nextnLayers().get(0);
        DsaSpec dsa = block.dsa();
        assertNotNull(dsa);
        assertEquals(8, dsa.topK());
        assertEquals(4, dsa.kpool());
        assertTrue(dsa.full());
    }

    @Test
    public void aDenseMtpBlockHasNoQsaIndexer() {
        ModelSpec target = Qwen4ExpSidecars.buildTarget();
        ModelSpec draft = Qwen4ExpSidecars.buildSidecar(target, 0, 0, Qwen4ExpSidecars.sharedSidecarTensors());
        assertNull(draft.nextnLayers().get(0).dsa());
    }

    @Test
    public void theSharedSidecarPassesTheLoaderChecks() {
        ModelHparams hp = Qwen4ExpSidecars.loadHparams(4, 0);
        TensorCatalog tc = TensorCatalog.of(ARCH, Qwen4ExpSidecars.sharedSidecarTensors());
        assertTrue(ModelSpecValidator.isMtpOnly(hp, tc));
        ModelSpecValidator.requireTensors(hp, tc, tensors(), true);
        assertTrue(TensorClaims.unclaimed(hp, tc, tensors()).isEmpty());
    }

    @Test
    public void tokenEmbdStaysRequiredOutsideMtpOnlyFiles() {
        ModelHparams hp = Qwen4ExpSidecars.loadHparams(4, 0);
        TensorCatalog tc = TensorCatalog.of(ARCH, Qwen4ExpSidecars.sharedSidecarTensors());
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> ModelSpecValidator.requireTensors(hp, tc, tensors(), false));
        assertTrue(e.missing().toString(), e.missing().contains("token_embd.weight"));
    }

    @Test
    public void aReducedHeadMustCarryItsOutputTensor() {
        ModelHparams hp = Qwen4ExpSidecars.loadHparams(4, 0);
        List<GGUFTensorInfo> ts = Qwen4ExpSidecars.sharedSidecarTensors();
        ts.add(new GGUFTensorInfo("d2t", new long[] {64}, GGMLType.I32, 0L));
        TensorCatalog tc = TensorCatalog.of(ARCH, ts);
        MissingTensorException e = assertThrows(MissingTensorException.class,
                () -> ModelSpecValidator.requireTensors(hp, tc, tensors(), true));
        assertTrue(e.missing().toString(), e.missing().toString().contains("output.weight"));
    }

    @Test
    public void theSharedSidecarHasNoEmbeddingOrHeadOfItsOwn() {
        ModelSpec draft = Qwen4ExpSidecars.buildSidecar(Qwen4ExpSidecars.buildTarget(), 4, 0,
                Qwen4ExpSidecars.sharedSidecarTensors());
        assertNull(draft.embedding().tokenEmbd());
        assertNull(draft.head().output());
        assertNull(draft.head().outputNorm());
    }

    @Test
    public void aSidecarWithItsOwnEmbeddingDoesNotTieItsHead() {
        ModelSpec draft = Qwen4ExpSidecars.buildSidecar(Qwen4ExpSidecars.buildTarget(), 4, 0,
                Qwen4ExpSidecars.sidecarWithEmbeddingTensors());
        assertNotNull(draft.embedding().tokenEmbd());
        assertNull(draft.head().output());
    }

    @Test
    public void aSidecarWithBothIoTensorsUsesThemUntied() {
        ModelSpec draft = Qwen4ExpSidecars.buildSidecar(Qwen4ExpSidecars.buildTarget(), 4, 0,
                Qwen4ExpSidecars.fullSidecarTensors());
        assertNotNull(draft.embedding().tokenEmbd());
        assertEquals("output.weight", draft.head().output().name());
    }

    @Test
    public void aModelWithTrunkStillTiesItsHeadToTheEmbeddings() {
        List<GGUFTensorInfo> ts = Qwen4ExpSidecars.targetTensors();
        ts.removeIf(t -> t.name().equals("output.weight"));
        ModelSpec spec = ModelSpecBuilder.build(GgufKv.ofMap(Qwen4ExpSidecars.hparams(4, 4)),
                TensorCatalog.of(ARCH, ts));
        assertTrue(spec.head().tied());
        assertEquals("token_embd.weight", spec.head().output().name());
    }
}
