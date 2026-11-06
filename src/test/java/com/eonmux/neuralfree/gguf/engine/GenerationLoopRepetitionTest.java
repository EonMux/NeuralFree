package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * {@code repetition_detection} over the real single-sequence loop of a synthetic model: a single
 * {@code allowed_token_ids} id makes greedy decoding repeat one token, which the detection stops after
 * {@code min_count} tokens (or at {@code min_tokens}), while the same request without it runs to its budget.
 */
public class GenerationLoopRepetitionTest {

    private static final int REPEATED = 20;
    private static final int[] PROMPT = {4, 17, 30, 43};

    private static GGUFModel model;

    @BeforeClass
    public static void open() throws IOException {
        Path file = SyntheticModels.llama().writeTempFile();
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
        model = new GGUFModel(file.toString(), EngineConfig.builder().memory(mc).build());
    }

    @AfterClass
    public static void close() throws IOException {
        model.close();
    }

    private static GenerationRequest.Builder oneToken() {
        return GenerationRequest.builder().maxNewTokens(12).sampler(SamplerConfig.greedyDecoding())
                .allowedTokenIds(REPEATED);
    }

    private static int[] repeated(int n) {
        int[] out = new int[n];
        Arrays.fill(out, REPEATED);
        return out;
    }

    @Test
    public void theLoopStopsARepeatedTokenAtMinCount() {
        GenerationResult result = model.generate(PROMPT, oneToken()
                .repetitionDetection(RepetitionDetection.of(3, 1, 4)).build(), null);
        assertEquals(StopReason.REPETITION, result.stopReason());
        assertArrayEquals("the repeats stay in the output", repeated(4), result.tokens());
        assertEquals(model.tokenizer().decode(repeated(4)), result.text());
    }

    @Test
    public void minTokensDefersTheStop() {
        GenerationResult result = model.generate(PROMPT, oneToken().minTokens(7)
                .repetitionDetection(RepetitionDetection.of(3, 1, 4)).build(), null);
        assertEquals(StopReason.REPETITION, result.stopReason());
        assertEquals(7, result.tokenCount());
    }

    @Test
    public void withoutDetectionTheBudgetEndsIt() {
        GenerationResult result = model.generate(PROMPT, oneToken().build(), null);
        assertEquals(StopReason.MAX_TOKENS, result.stopReason());
        assertArrayEquals(repeated(12), result.tokens());
    }
}
