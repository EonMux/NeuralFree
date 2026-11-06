package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The request-driven sampling stages: {@link MinTokensStage} bans the vocabulary's end-of-generation
 * ids until the minimum is generated and releases them after, and {@link BadWordsStage} bans the token that
 * would continue a banned phrase from the accepted history — including a phrase spanning three tokens and a
 * single-token phrase banned everywhere — while a request without the stages bans nothing. The request's
 * {@code constraint} builds a grammar stage ({@link GrammarStage}, or {@link TriggeredGrammarStage} behind a
 * trigger) that masks every token the grammar rejects, carries through the request's builder round trip, and an
 * unusable grammar refuses by name.
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
    public void theMinTokensBanYieldsWhenOnlyEndOfGenerationRemains() {
        SamplerVocab vocab = new ByteVocab();
        SamplerStage grammar = RequestStages.constraintStage(ConstraintSpec.gbnf("root ::= \"a\""),
                vocab::tokenize, vocab);
        // the assembly order: the grammar masks first, the unspent min-tokens budget bans behind it
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(grammar, new MinTokensStage(3, vocab)), vocab);
        // while the word is open the grammar admits 'a', so the budget bans EOG (0) behind it
        assertEquals('a', sample(sampler, logits(0, 9f, 'a', 8f)));
        sampler.accept('a');
        // the word is closed and the grammar leaves EOG alone in the set: the ban must not empty it —
        // EOG stays selectable and outranks the grammar-rejected decoy
        assertEquals(0, sample(sampler, logits('b', 8f, 0, 9f)));
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

    @Test
    public void theRequestStagesBuildInTheConstraintsHeadOrder() {
        SamplerVocab vocab = new ByteVocab();
        com.eonmux.neuralfree.gguf.engine.GenerationRequest request =
                com.eonmux.neuralfree.gguf.engine.GenerationRequest.builder()
                        .maxNewTokens(8)
                        .sampler(com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding())
                        .minTokens(2)
                        .badWord("x")
                        .allowedTokenIds(5, 6)
                        .build();
        List<SamplerStage> stages = RequestStages.of(request, vocab);
        assertEquals(3, stages.size());
        // the head's masking order (vLLM applies the masks to the logits before sampling in this order): the
        // allowed ids, then the bad words, then the min-tokens ban, which yields behind them
        assertTrue(stages.get(0) instanceof AllowedTokensStage);
        assertTrue(stages.get(1) instanceof BadWordsStage);
        assertTrue(stages.get(2) instanceof MinTokensStage);
        for (SamplerStage stage : stages) {
            assertTrue("a request mask stage masks", stage.masks());
            assertFalse("a request mask stage is no grammar", stage.constrains());
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

    // ================================================================== grammar constraints

    @Test
    public void aConstraintStageMasksTheTokensTheGrammarRejects() {
        SamplerVocab vocab = new ByteVocab();
        SamplerStage stage = RequestStages.constraintStage(ConstraintSpec.gbnf("root ::= \"ok\""),
                vocab::tokenize, vocab);
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(stage), vocab);
        // only the grammar's first byte keeps its logit
        assertEquals('o', sample(sampler, logits('x', 9f, 'o', 8f)));
        sampler.accept('o');
        assertEquals('k', sample(sampler, logits('x', 9f, 'k', 8f)));
    }

    @Test
    public void aTriggerGatedConstraintRunsAsTheTriggeredStage() {
        SamplerVocab vocab = new ByteVocab();
        SamplerStage stage = RequestStages.constraintStage(
                ConstraintSpec.gbnf("root ::= \"ok\"").withTriggers(GrammarTriggers.lazyWords(List.of("<t>"))),
                vocab::tokenize, vocab);
        assertTrue(stage instanceof TriggeredGrammarStage);
        SamplerStage plain = RequestStages.constraintStage(ConstraintSpec.gbnf("root ::= \"ok\""),
                vocab::tokenize, vocab);
        assertTrue(plain instanceof GrammarStage);
        assertNotEquals(plain.getClass(), stage.getClass());
    }

    @Test
    public void noConstraintAndAnUnusableConstraint() {
        SamplerVocab vocab = new ByteVocab();
        assertNull(RequestStages.constraintStage(null, vocab::tokenize, vocab));
        assertThrows(IllegalArgumentException.class,
                () -> RequestStages.constraintStage(ConstraintSpec.gbnf("root ::= (unclosed"), vocab::tokenize,
                        vocab));
    }

    @Test
    public void aGrammarStageKeepsTheGreedyChainOffTheOnDeviceArgmax() {
        SamplerVocab vocab = new ByteVocab();
        SamplerStage grammar = RequestStages.constraintStage(ConstraintSpec.gbnf("root ::= \"ok\""),
                vocab::tokenize, vocab);
        Sampler constrained = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(grammar), vocab);
        assertFalse("the on-device argmax sees no grammar mask", constrained.argmaxOnly());
        Sampler plain = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0], vocab);
        assertTrue(plain.argmaxOnly());
    }

    @Test
    public void aRequestCarriesItsConstraintThroughTheBuilderRoundTrip() {
        com.eonmux.neuralfree.gguf.engine.GenerationRequest request =
                com.eonmux.neuralfree.gguf.engine.GenerationRequest.builder()
                        .maxNewTokens(8)
                        .constraint(ConstraintSpec.jsonSchema("{\"type\": \"object\"}", null))
                        .build();
        assertSame(request.constraint(), request.toBuilder().build().constraint());
        assertNull(com.eonmux.neuralfree.gguf.engine.GenerationRequest.defaults().constraint());
    }
}
