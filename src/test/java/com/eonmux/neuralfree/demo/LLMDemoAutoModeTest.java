package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.Mode;
import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;

/**
 * The demo's AUTO mode follows the model's head, as the llama.cpp tools do: decision heads classify, RANK pooling
 * reranks a given document (else classifies), any other GGUF pooling type embeds, and only a model without either
 * falls back to its architecture's support class.
 */
public class LLMDemoAutoModeTest {

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

    private static Mode auto(Arch arch, PoolingKind pooling, String... args) throws UsageException {
        return LLMDemo.autoMode(spec(arch, pooling, 0), LLMDemoOptions.parse(args));
    }

    @Test
    public void pooledGenerativeModelsEmbedOrRank() throws UsageException {
        assertEquals(Mode.EMBED, auto(Arch.QWEN3, PoolingKind.LAST));
        assertEquals(Mode.EMBED, auto(Arch.QWEN3, PoolingKind.MEAN));
        assertEquals(Mode.CLASSIFY, auto(Arch.QWEN3, PoolingKind.RANK));
        assertEquals(Mode.RERANK, auto(Arch.QWEN3, PoolingKind.RANK, "--document", "a document"));
        assertEquals(Mode.GENERATE, auto(Arch.QWEN3, PoolingKind.NONE));
    }

    @Test
    public void decisionModelsClassifyWhateverTheirClass() throws UsageException {
        LLMDemoOptions none = LLMDemoOptions.parse(new String[0]);
        assertEquals(Mode.CLASSIFY, LLMDemo.autoMode(spec(Arch.MODERN_BERT, PoolingKind.NONE, 2), none));
        assertEquals(Mode.CLASSIFY, LLMDemo.autoMode(spec(Arch.LFM2, PoolingKind.NONE, 2), none));
        assertEquals(Mode.CLASSIFY, LLMDemo.autoMode(spec(Arch.VLLM_DECISION2, PoolingKind.MEAN, 0), none));
    }

    @Test
    public void unpooledModelsFollowTheirSupportClass() throws UsageException {
        assertEquals(Mode.EMBED, auto(Arch.BERT, PoolingKind.NONE));
        assertEquals(Mode.EMBED, auto(Arch.MODERN_BERT, PoolingKind.NONE));
        assertEquals(Mode.GENERATE, auto(Arch.LFM2, PoolingKind.NONE));
    }
}
