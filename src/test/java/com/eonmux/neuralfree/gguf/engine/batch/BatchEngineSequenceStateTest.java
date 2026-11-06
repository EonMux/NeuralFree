package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's sequence operations go through the memory's sequence view ({@code ModelMemory.sequences()}),
 * which keeps the per-sequence state beside the pages in step: a request admitted into the slot a retired request
 * freed starts from a zeroed recurrent state, so over the synthetic Mamba2 every request, the one in the reused slot
 * included, generates what the same prompt generates alone through {@link GenerationLoop}.
 */
public class BatchEngineSequenceStateTest {

    private static final int N_NEW = 6;

    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static GenerationResult reference(Path model, int[] prompt) throws IOException {
        EngineConfig one = config(1);
        try (GGUFModel m = new GGUFModel(model.toString(), one)) {
            return GenerationLoop.forModel(m, one).generate(prompt, request(), (TokenCallback) null);
        }
    }

    @Test
    public void aRequestInAReusedSlotStartsFromAFreshRecurrentState() throws IOException {
        Path model = SyntheticModels.mamba2().writeTempFile();
        int[][] prompts = {{5, 9, 13, 40}, {7, 22, 31, 8, 3}, {11, 4, 60}};
        GenerationResult[] refs = new GenerationResult[prompts.length];
        for (int i = 0; i < prompts.length; i++) {
            refs[i] = reference(model, prompts[i]);
            assertEquals(N_NEW, refs[i].tokenCount());
        }
        EngineConfig two = config(2);
        try (GGUFModel m = new GGUFModel(model.toString(), two); BatchEngine engine = BatchEngine.forModel(m, two)) {
            assertEquals(2, engine.maxSequences());
            BatchSequence[] seqs = new BatchSequence[prompts.length];
            for (int i = 0; i < prompts.length; i++) {
                seqs[i] = engine.submit(prompts[i], request(), (TokenCallback) null);
            }
            engine.drain();
            assertTrue("the third request runs in a slot an earlier request freed",
                    seqs[2].seq() == seqs[0].seq() || seqs[2].seq() == seqs[1].seq());
            for (int i = 0; i < prompts.length; i++) {
                assertEquals("request " + i + " state", BatchSequence.State.DONE, seqs[i].state());
                assertArrayEquals("request " + i + " token ids", refs[i].tokens(), seqs[i].result().tokens());
            }
        }
    }
}
