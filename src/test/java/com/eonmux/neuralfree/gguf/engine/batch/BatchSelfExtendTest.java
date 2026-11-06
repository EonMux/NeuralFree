package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's self-extend ({@link SelfExtend}, llama.cpp's {@code grp_attn_n}/{@code grp_attn_w} over
 * {@code llama_memory_seq_add} and {@code llama_memory_seq_div}): a sequence whose request groups its positions
 * keeps decoding past the context it would otherwise fill, because every group window is divided back into the
 * position space; the plain request of the same prompt stops with {@code CONTEXT_FULL}. The request validation and
 * the engine's refusals name the fields.
 */
public class BatchSelfExtendTest {

    private static final int N_CTX = 24;
    private static final int N_TOKENS = 12;

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).kvPageSize(8).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    private static GenerationResult run(Path model, int[] prompt, GenerationRequest request) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            BatchSequence seq = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(String.valueOf(seq.error()), seq.state() == BatchSequence.State.DONE);
            assertFalse("self-extend never shifts the context", seq.truncated());
            assertTrue("the run compressed its positions: " + engine.stats().grpAttnEdits(),
                    engine.stats().grpAttnEdits() > 0);
            return seq.result();
        }
    }

    @Test
    public void groupAttentionCompressesTheStoredPositionsAndChangesTheGeneration() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(1);
        int budget = 8; // prompt and decodes fit the rows; only the self-extend moves the positions

        GenerationResult plain;
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            BatchSequence seq = engine.submit(prompt, request(budget),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(String.valueOf(seq.error()), seq.state() == BatchSequence.State.DONE);
            plain = seq.result();
            assertEquals(0, engine.stats().grpAttnEdits());
        }
        assertEquals(budget, plain.tokenCount());

        GenerationResult extended = run(model, prompt,
                request(budget).toBuilder().grpAttnN(2).grpAttnW(8).build());
        assertEquals(budget, extended.tokenCount());
        assertNotEquals("the compressed positions change what the model attends", plain.text(),
                extended.text());
    }

    @Test
    public void theSameRequestCompressesTheSameWayTwice() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        GenerationRequest grouped = request(8).toBuilder().grpAttnN(2).grpAttnW(8).build();
        GenerationResult first = run(model, tokens(2), grouped);
        GenerationResult second = run(model, tokens(2), grouped);
        assertArrayEquals(first.tokens(), second.tokens());
        assertEquals(first.text(), second.text());
    }

    @Test
    public void theBuilderRefusesAGroupPairItCannotRun() {
        assertThrows(IllegalArgumentException.class, () -> request(1).toBuilder().grpAttnN(0));
        assertThrows(IllegalArgumentException.class, () -> request(1).toBuilder().grpAttnW(0));
        // llama.cpp's ga_w % ga_n == 0
        assertThrows(IllegalArgumentException.class,
                () -> request(1).toBuilder().grpAttnN(3).grpAttnW(8).build());
        GenerationRequest ok = request(1).toBuilder().grpAttnN(4).grpAttnW(8).build();
        assertTrue(ok.selfExtend());
        assertEquals(4, ok.grpAttnN());
        assertEquals(8, ok.grpAttnW());
        assertFalse(request(1).selfExtend());
    }

    @Test
    public void aGroupPairBesideAContextShiftRefusesAtSubmit() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        GenerationRequest both = request(4).toBuilder().nKeep(4).grpAttnN(2).grpAttnW(8).build();
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> engine.submit(tokens(3), both, (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("grp_attn_n"));
            assertTrue(e.getMessage(), e.getMessage().contains("n_keep"));
        }
    }
}
