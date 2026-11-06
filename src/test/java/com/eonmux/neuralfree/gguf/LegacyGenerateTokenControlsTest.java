package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.RepetitionDetection;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import org.junit.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The legacy {@link LLMModel#generate(int[], GenerationRequest, TokenCallback)} adapter refuses the per-request
 * token controls its loop has no parameter for, naming each request field, before it runs anything.
 */
public class LegacyGenerateTokenControlsTest {

    /** A model offering only the legacy API; any generation call fails the test. */
    private static final class LegacyOnly implements LLMModel {
        @Override public int[] generate(int[] t, int n, float temp, float p) { throw new AssertionError("ran"); }
        @Override public int[] generate(int[] t, int n, float temp, float p, TokenCallback c) {
            throw new AssertionError("ran");
        }
        @Override public int[] generateWithPrompt(String s, int n, float temp, float p) { throw new AssertionError(); }
        @Override public int[] generateWithPrompt(String s, int n, float temp, float p, TokenCallback c) {
            throw new AssertionError();
        }
        @Override public void resetKVCache() { }
        @Override public GGUFTokenizer tokenizer() { throw new AssertionError("tokenizer read"); }
        @Override public void printInfo() { }
        @Override public String architecture() { return "legacy"; }
        @Override public String name() { return "legacy"; }
        @Override public int vocabSize() { return 16; }
        @Override public int contextLength() { return 64; }
        @Override public int kvCacheContext() { return 64; }
        @Override public int embeddingLength() { return 8; }
        @Override public int layerCount() { return 1; }
        @Override public int headCount() { return 1; }
        @Override public int headCountKv() { return 1; }
        @Override public int feedForwardLength() { return 8; }
        @Override public float rmsNormEps() { return 1e-5f; }
        @Override public float ropeFreqBase() { return 10000f; }
        @Override public int headDim() { return 8; }
        @Override public int kvDim() { return 8; }
        @Override public void close() { }
    }

    private static String refusal(GenerationRequest.Builder request) {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> new LegacyOnly().generate(new int[]{1, 2}, request.sampler(SamplerConfig.greedyDecoding())
                        .build(), null));
        return e.getMessage();
    }

    @Test
    public void eachTokenControlRefusesByName() {
        assertTrue(refusal(GenerationRequest.builder().stopTokenIds(3)).contains("stop_token_ids"));
        assertTrue(refusal(GenerationRequest.builder().allowedTokenIds(3)).contains("allowed_token_ids"));
        assertTrue(refusal(GenerationRequest.builder().skipSpecialTokens(false)).contains("skip_special_tokens"));
        assertTrue(refusal(GenerationRequest.builder().repetitionDetection(RepetitionDetection.of(2, 1, 2)))
                .contains("repetition_detection"));
    }

    @Test
    public void everyRefusedControlIsListedTogether() {
        String message = refusal(GenerationRequest.builder().stopTokenIds(3).allowedTokenIds(4)
                .repetitionDetection(RepetitionDetection.of(2, 1, 2)));
        assertTrue(message, message.contains("stop_token_ids") && message.contains("allowed_token_ids")
                && message.contains("repetition_detection"));
    }
}
