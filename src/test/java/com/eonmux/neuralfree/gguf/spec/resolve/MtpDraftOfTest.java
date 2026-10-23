package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.ARCH;
import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.N_EMBD;
import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.N_VOCAB;
import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.TRUNK;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ModelSpecBuilder#isMtpDraftOf} separates an MTP draft file of the target's model (the full file or its
 * MTP-only sidecar) from another model of the family that keeps its NextN blocks, as llama.cpp's converters do by
 * default: a different width, trunk depth or vocabulary makes it a separate draft model, and
 * {@link ModelSpecBuilder#buildMtpDraft} refuses it with the same reason.
 */
public class MtpDraftOfTest {

    private static ModelSpec target() {
        return ModelSpecBuilder.build(GgufKv.ofMap(NemotronNextnFixture.metadata()),
                TensorCatalog.of(ARCH, NemotronNextnFixture.tensors()), MemoryConfig.defaults());
    }

    @Test
    public void theTargetsOwnFullFileAndSidecarAreMtpDraftsOfIt() {
        ModelSpec target = target();
        GgufKv kv = GgufKv.ofMap(NemotronNextnFixture.metadata());
        TensorCatalog full = TensorCatalog.of(ARCH, NemotronNextnFixture.tensors());
        TensorCatalog sidecar = TensorCatalog.of(ARCH,
                NemotronNextnFixture.mtpOnly(NemotronNextnFixture.tensors(), TRUNK));
        assertNull(ModelSpecBuilder.mtpDraftMismatch(kv, full, target));
        assertTrue(ModelSpecBuilder.isMtpDraftOf(kv, full, target));
        assertTrue(ModelSpecBuilder.isMtpDraftOf(kv, sidecar, target));
    }

    @Test
    public void aWiderModelOfTheFamilyIsNotAnMtpDraftOfTheTarget() {
        assertSibling(2 * N_EMBD, TRUNK, N_VOCAB, "n_embd " + 2 * N_EMBD);
    }

    @Test
    public void aDeeperModelOfTheFamilyIsNotAnMtpDraftOfTheTarget() {
        assertSibling(N_EMBD, 2 * TRUNK, N_VOCAB, 2 * TRUNK + " trunk layers");
    }

    @Test
    public void aModelWithAnotherVocabularyIsNotAnMtpDraftOfTheTarget() {
        assertSibling(N_EMBD, TRUNK, N_VOCAB + 16, "vocabulary of " + (N_VOCAB + 16));
    }

    @Test
    public void aFileWithoutNextnBlocksIsNotAnMtpDraft() {
        ModelSpec target = target();
        GgufKv kv = GgufKv.ofMap(NemotronNextnFixture.metadata());
        TensorCatalog noNextn = TensorCatalog.of(ARCH,
                NemotronNextnFixture.withoutNextn(NemotronNextnFixture.tensors(), TRUNK));
        assertFalse(ModelSpecBuilder.isMtpDraftOf(kv, noNextn, target));
        String reason = ModelSpecBuilder.mtpDraftMismatch(kv, noNextn, target);
        assertTrue(reason, reason.contains("no NextN block"));
        assertTrue(reason, reason.contains("blk." + TRUNK + ".nextn.eh_proj.weight"));
    }

    private static void assertSibling(int nEmbd, int trunk, int nVocab, String expected) {
        ModelSpec target = target();
        Map<String, Object> metadata = NemotronNextnFixture.metadata(nEmbd, trunk, nVocab);
        GgufKv kv = GgufKv.ofMap(metadata);
        TensorCatalog full = TensorCatalog.of(ARCH, NemotronNextnFixture.tensors(nEmbd, trunk, nVocab));
        ModelSpec own = ModelSpecBuilder.build(kv, full, MemoryConfig.defaults());
        assertTrue(ModelSpecBuilder.isNextnDraft(kv, full));
        assertTrue("the sibling is an MTP draft of its own model", ModelSpecBuilder.isMtpDraftOf(kv, full, own));
        assertFalse(ModelSpecBuilder.isMtpDraftOf(kv, full, target));
        String reason = ModelSpecBuilder.mtpDraftMismatch(kv, full, target);
        assertTrue(reason, reason.contains(expected));
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> ModelSpecBuilder.buildMtpDraft(kv, full, MemoryConfig.defaults(), target));
        assertEquals(reason, refused.getMessage());
    }
}
