package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The token-type table joins the input embedding only on the BERT graph (models/bert.cpp:91-94); the decision heads
 * of modern-bert and lfm2 and the clef decoder read it themselves (models/modern-bert.cpp:260, lfm2.cpp:434,
 * clef.cpp:600).
 */
public class EmbeddingResolverTest {

    private static final int N_EMBD = 32;

    /** The architectures that run {@code llama_model_bert::graph} (models.h:341, 352, 363, 374). */
    private static final Arch[] BERT_GRAPH = {
            Arch.BERT, Arch.JINA_BERT_V2, Arch.JINA_BERT_V3, Arch.NOMIC_BERT, Arch.NOMIC_BERT_MOE};

    private static EmbeddingSpec resolve(Arch arch) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0).nEmbd(N_EMBD).build();
        TensorCatalog tc = TensorCatalog.of(arch.ggufName(), Arrays.asList(
                new GGUFTensorInfo("token_embd.weight", new long[] {N_EMBD, 16}, GGMLType.F32, 0L),
                new GGUFTensorInfo("token_types.weight", new long[] {N_EMBD, 2}, GGMLType.F32, 0L)));
        return new EmbeddingResolver(hp, tc).resolve();
    }

    @Test
    public void bertGraphsAddTheTokenTypeRow() {
        for (Arch arch : BERT_GRAPH) {
            assertTrue(arch.ggufName(), EmbeddingResolver.addsInputTokenTypes(ArchRegistry.traits(arch)));
            EmbeddingSpec e = resolve(arch);
            assertNotNull(arch.ggufName(), e.tokenTypes());
            assertEquals("token_types.weight", e.tokenTypes().name());
        }
    }

    @Test
    public void otherGraphsLeaveTheTableToTheirHeads() {
        for (Arch arch : new Arch[] {Arch.MODERN_BERT, Arch.LFM2, Arch.CLEF, Arch.LLAMA}) {
            assertFalse(arch.ggufName(), EmbeddingResolver.addsInputTokenTypes(ArchRegistry.traits(arch)));
            assertNull(arch.ggufName(), resolve(arch).tokenTypes());
        }
    }
}
