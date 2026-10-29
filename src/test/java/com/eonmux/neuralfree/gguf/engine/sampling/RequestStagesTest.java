package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The request-driven parity stages (PARITY P5): {@link MinTokensStage} bans the vocabulary's end-of-generation
 * ids until the minimum is generated and releases them after, and {@link BadWordsStage} bans the token that
 * would continue a banned phrase from the accepted history — including a phrase spanning three tokens and a
 * single-token phrase banned everywhere — while a request without the stages bans nothing.
 */
public class RequestStagesTest {

    /** ASCII byte tokens: id {@code c} renders the byte {@code c}; id 0 is EOG, ids 1..127 are their code point. */
    private static final class ByteVocab implements SamplerVocab {
        @Override public int vocabSize() { return 128; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) {
            return id <= 0 ? new byte[0] : new byte[]{(byte) id};
        }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
            int[] ids = new int[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                ids[i] = bytes[i] & 0xFF;
            }
            return ids;
        }
    }

    private static float[] logits(float... pairs) {
        float[] out = new float[128];
        java.util.Arrays.fill(out, -10f);
        for (int i = 0; i < pairs.length; i += 2) {
            out[(int) pairs[i]] = pairs[i + 1];
        }
        return out;
    }

    private static int sample(Sampler sampler, float[] logits) {
        return sampler.sample(logits);
    }

    @Test
    public void minTokensBansEogUntilTheBudgetIsGenerated() {
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                RequestStages.of(requestWithMinTokens(2), new ByteVocab()), new ByteVocab());
        // the EOG token (0) ranks first but must lose while the budget runs
        assertEquals('a', sample(sampler, logits(0, 9f, 'a', 8f, 'b', 7f)));
        sampler.accept('a');
        assertEquals('b', sample(sampler, logits(0, 9f, 'b', 7f)));
        sampler.accept('b');
        assertEquals("the budget is spent: the EOG token wins again", 0,
                sample(sampler, logits(0, 9f, 'b', 7f)));
    }

    @Test
    public void minTokensResetsWithTheChain() {
        SamplerVocab vocab = new ByteVocab();
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                RequestStages.of(requestWithMinTokens(1), vocab), vocab);
        assertEquals('a', sample(sampler, logits(0, 9f, 'a', 8f)));
        sampler.accept('a');
        sampler.reset();
        assertEquals("after the reset the budget runs from the start", 'a',
                sample(sampler, logits(0, 9f, 'a', 8f)));
    }

    @Test
    public void badWordsBanTheContinuationOfEveryBannedPhrase() {
        SamplerVocab vocab = new ByteVocab();
        // "no" spans two tokens; "x" is a single token banned everywhere
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                RequestStages.of(requestWithBadWords("no", "x"), vocab), vocab);
        // nothing banned at an empty history: 'n' survives and is sampled
        assertEquals('n', sample(sampler, logits('n', 9f, 'y', 8f, 'x', 7f)));
        sampler.accept('n');
        // the history ends with 'n': 'o' would spell "no" and is masked, 'y' wins
        assertEquals('y', sample(sampler, logits('o', 9f, 'y', 8f, 'x', 7f)));
        sampler.accept('y');
        // 'x' is banned everywhere, in any context
        assertEquals('e', sample(sampler, logits('x', 9f, 'e', 8f)));
        sampler.accept('e');
        assertEquals('s', sample(sampler, logits('s', 9f, 'o', 8f)));
        sampler.accept('s');
        // the history ends with "yes": no phrase continuation, 'n' is free again
        assertEquals('n', sample(sampler, logits('n', 9f, 'o', 8f)));
    }

    @Test
    public void aThreeTokenPhraseBansOnlyItsExactCompletion() {
        SamplerVocab vocab = new ByteVocab();
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                RequestStages.of(requestWithBadWords("abc"), vocab), vocab);
        assertEquals('a', sample(sampler, logits('a', 9f, 'z', 8f)));
        sampler.accept('a');
        // history [a]: 'b' starts "ab", which is not yet the phrase — free
        assertEquals('b', sample(sampler, logits('b', 9f, 'z', 8f)));
        sampler.accept('b');
        // history [a, b]: 'c' would complete "abc" and is masked
        assertEquals('z', sample(sampler, logits('c', 9f, 'z', 8f)));
        sampler.accept('z');
        // history [a, b, z]: the phrase is broken, 'c' is free again
        assertEquals('c', sample(sampler, logits('c', 9f, 'z', 8f)));
    }

    @Test
    public void emptyPhrasesRefuseAndAnEmptyRequestBuildsNothing() {
        SamplerVocab vocab = new ByteVocab();
        assertEquals(0, RequestStages.of(requestWithBadWords(), vocab).size());
        try {
            BadWordsStage.of(List.of(""), vocab);
            throw new AssertionError("expected a refusal");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("tokenizes to no tokens"));
        }
    }

    private static com.eonmux.neuralfree.gguf.engine.GenerationRequest requestWithMinTokens(int n) {
        return com.eonmux.neuralfree.gguf.engine.GenerationRequest.builder()
                .maxNewTokens(8)
                .sampler(com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding())
                .minTokens(n)
                .build();
    }

    private static com.eonmux.neuralfree.gguf.engine.GenerationRequest requestWithBadWords(String... words) {
        com.eonmux.neuralfree.gguf.engine.GenerationRequest.Builder builder =
                com.eonmux.neuralfree.gguf.engine.GenerationRequest.builder()
                        .maxNewTokens(8)
                        .sampler(com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding());
        for (String word : words) {
            builder.badWord(word);
        }
        return builder.build();
    }
}
