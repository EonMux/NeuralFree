package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;
import org.junit.function.ThrowingRunnable;

import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The shared encode / encodeTokens gate of the GPU and CPU orchestrators: embedding models are the only encoders,
 * every decision model (vllm-decision2, clef, modern-bert / lfm2 decision blocks) is sent to classify whatever its
 * pooling, generative models encode only with a pooling type, per-token output needs NONE pooling, and reranking
 * needs RANK pooling.
 */
public class EncodeGateTest {

    private static final int N_LAYER = 4;
    private static final int N_EMBD = 16;

    private static ModelSpec spec(Arch arch, PoolingKind pooling, int nLayerDecision) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(N_LAYER, 0)
                .nCtxTrain(512).nEmbd(N_EMBD).normRmsEps(1e-6f).nLayerDecision(nLayerDecision);
        return ModelSpec.builder(hb.build(), TensorCatalog.of(arch.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.empty())
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().pooling(pooling).build())
                .build();
    }

    private static ModelSpec spec(Arch arch, PoolingKind pooling) {
        return spec(arch, pooling, 0);
    }

    @Test
    public void onlyEmbeddingModelsAreEncoders() {
        assertEquals(SupportClass.EMBED, spec(Arch.BERT, PoolingKind.MEAN).supportClass());
        assertTrue(EncodeGate.isEncoder(spec(Arch.BERT, PoolingKind.MEAN)));
        assertEquals(SupportClass.CLS, spec(Arch.VLLM_DECISION2, PoolingKind.NONE).supportClass());
        assertFalse(EncodeGate.isEncoder(spec(Arch.VLLM_DECISION2, PoolingKind.NONE)));
        assertEquals(SupportClass.CLS, spec(Arch.CLEF, PoolingKind.NONE).supportClass());
        assertFalse(EncodeGate.isEncoder(spec(Arch.CLEF, PoolingKind.NONE)));
        assertFalse(EncodeGate.isEncoder(spec(Arch.LLAMA, PoolingKind.MEAN)));
    }

    @Test
    public void clsDecisionModelsAreSentToClassify() {
        for (Arch arch : new Arch[] {Arch.VLLM_DECISION2, Arch.CLEF}) {
            for (PoolingKind pooling : new PoolingKind[] {PoolingKind.NONE, PoolingKind.MEAN, PoolingKind.LAST}) {
                ModelSpec s = spec(arch, pooling);
                assertDecision(() -> EncodeGate.checkEncode(s), "encode", arch);
                assertDecision(() -> EncodeGate.checkEncodeTokens(s), "encodeTokens", arch);
            }
        }
    }

    @Test
    public void decisionBlockModelsAreSentToClassify() {
        for (Arch arch : new Arch[] {Arch.MODERN_BERT, Arch.LFM2}) {
            ModelSpec s = spec(arch, PoolingKind.NONE, 2);
            assertDecision(() -> EncodeGate.checkEncode(s), "encode", arch);
            assertDecision(() -> EncodeGate.checkEncodeTokens(s), "encodeTokens", arch);
        }
    }

    @Test
    public void modernBertWithoutDecisionBlocksEncodes() {
        ModelSpec s = spec(Arch.MODERN_BERT, PoolingKind.NONE);
        EncodeGate.checkEncode(s);
        EncodeGate.checkEncodeTokens(s);
    }

    @Test
    public void embeddingModelsEncodeAndPoolPerSequence() {
        EncodeGate.checkEncode(spec(Arch.BERT, PoolingKind.MEAN));
        EncodeGate.checkEncode(spec(Arch.BERT, PoolingKind.NONE));
        EncodeGate.checkEncodeTokens(spec(Arch.BERT, PoolingKind.NONE));
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> EncodeGate.checkEncodeTokens(spec(Arch.BERT, PoolingKind.CLS)));
        assertTrue(e.getMessage(), e.getMessage().contains("pooling CLS"));
    }

    @Test
    public void generativeModelsEncodeOnlyWithAPoolingType() {
        EncodeGate.checkEncode(spec(Arch.LLAMA, PoolingKind.LAST));
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> EncodeGate.checkEncode(spec(Arch.LLAMA, PoolingKind.NONE)));
        assertTrue(e.getMessage(), e.getMessage().contains("has neither"));
        assertThrows(UnsupportedOperationException.class,
                () -> EncodeGate.checkEncodeTokens(spec(Arch.LLAMA, PoolingKind.NONE)));
    }

    @Test
    public void pocketTtsExposesPerTokenRows() {
        EncodeGate.checkEncodeTokens(spec(Arch.POCKETTTS, PoolingKind.NONE));
    }

    @Test
    public void rerankNeedsRankPooling() {
        EncodeGate.checkRerank(spec(Arch.QWEN3, PoolingKind.RANK));
        EncodeGate.checkRerank(spec(Arch.BERT, PoolingKind.RANK));
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> EncodeGate.checkRerank(spec(Arch.QWEN3, PoolingKind.LAST)));
        assertEquals("reranking needs RANK pooling; qwen3 pools by LAST", e.getMessage());
    }

    private static void assertDecision(ThrowingRunnable call, String what, Arch arch) {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class, call);
        assertEquals(what + ": " + arch.ggufName() + " is a decision model; score it with classify", e.getMessage());
    }
}
