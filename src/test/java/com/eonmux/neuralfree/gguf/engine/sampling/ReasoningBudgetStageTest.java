package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The reasoning-budget stage (llama.cpp common/reasoning-budget.cpp): the start tag arms a countdown, a natural end
 * tag or the exhausted budget ends it, the budget forces the message plus the first end tag token by token over any
 * logits, a cut short mid character waits for a complete UTF-8 piece, and {@code force()} is the mid-stream
 * reasoning-end control. Over a byte vocabulary the tags ride their ASCII bytes.
 */
public class ReasoningBudgetStageTest {

    /** Byte vocabulary: id {@code c} renders the byte {@code c} (0 is EOG, pieces of 0 are empty). */
    private static final class ByteVocab implements SamplerVocab {
        @Override public int vocabSize() { return 256; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return id <= 0 ? new byte[0] : new byte[]{(byte) id}; }
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

    private static final String START = "<think>";
    private static final String END = "</think>";

    /** Logits favoring {@code best} over every other id, the forced token included. */
    private static float[] logits(int best) {
        float[] out = new float[256];
        java.util.Arrays.fill(out, -10f);
        out[best] = 9f;
        return out;
    }

    private static int[] bytes(String text) {
        byte[] raw = text.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[raw.length];
        for (int i = 0; i < raw.length; i++) {
            ids[i] = raw[i] & 0xFF;
        }
        return ids;
    }

    private static ReasoningBudgetStage stage(GenerationRequest request) {
        return ReasoningBudgetStage.of(ReasoningFilter.thinkTags(), request, new ByteVocab());
    }

    private static Sampler sampler(ReasoningBudgetStage stage) {
        return Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.<SamplerStage>singletonList(stage), new ByteVocab());
    }

    private static GenerationRequest budgetRequest(int budget) {
        return GenerationRequest.builder().maxNewTokens(64).reasoningBudget(budget).build();
    }

    private static void accept(Sampler sampler, int[] tokens) {
        for (int token : tokens) {
            sampler.accept(token);
        }
    }

    @Test
    public void theBudgetCountsInsideTheBlockAndForcesTheEndTag() {
        ReasoningBudgetStage stage = stage(budgetRequest(2));
        Sampler sampler = sampler(stage);
        accept(sampler, bytes("before"));                       // outside: no start tag seen
        assertEquals(ReasoningBudgetStage.State.IDLE, stage.state());
        accept(sampler, bytes(START));
        assertEquals(ReasoningBudgetStage.State.COUNTING, stage.state());
        assertEquals('x', sampler.sample(logits('x')));         // first reasoning token passes
        sampler.accept('x');
        assertEquals('y', sampler.sample(logits('y')));         // second passes, the budget is spent after it
        sampler.accept('y');
        assertEquals(ReasoningBudgetStage.State.FORCING, stage.state());
        int[] end = bytes(END);
        for (int i = 0; i < end.length; i++) {
            assertTrue(stage.forcing());
            assertEquals("every forced token wins over a stronger rival", end[i], sampler.sample(logits('z')));
            sampler.accept(end[i]);
        }
        assertEquals(ReasoningBudgetStage.State.DONE, stage.state());
        assertEquals('z', sampler.sample(logits('z')));         // after the end tag the argmax rules again
    }

    @Test
    public void reasoningTheParserOpensAtTheOutputsStartArmsTheBudgetThere() {
        String prompt = "<|im_start|>assistant\n";
        ReasoningFilter implicit = ReasoningFilter.thinkTags().opensImplicitly();
        ReasoningBudgetStage stage = ReasoningBudgetStage.of(implicit, budgetRequest(1), new ByteVocab(), prompt);
        assertTrue(stage.armsAtOutput());
        Sampler sampler = sampler(stage);
        accept(sampler, bytes(prompt));
        assertEquals("prompt tokens do not count down", ReasoningBudgetStage.State.IDLE, stage.state());
        assertEquals('x', sampler.sample(logits('x')));
        assertEquals(ReasoningBudgetStage.State.COUNTING, stage.state());
        sampler.accept('x');
        assertEquals(ReasoningBudgetStage.State.FORCING, stage.state());
        assertEquals('<', sampler.sample(logits('z')));

        ReasoningBudgetStage zero = ReasoningBudgetStage.of(implicit, budgetRequest(0), new ByteVocab(), prompt);
        Sampler forced = sampler(zero);
        accept(forced, bytes(prompt));
        assertEquals("a zero budget forces from the first output token", '<', forced.sample(logits('z')));
    }

    @Test
    public void aPromptThatClosedTheBlockOrAParserThatWaitsForAStartTagLeavesTheBudgetIdle() {
        ReasoningFilter implicit = ReasoningFilter.thinkTags().opensImplicitly();
        String closed = "<|im_start|>assistant\n" + START + END + "\n";
        ReasoningBudgetStage stage = ReasoningBudgetStage.of(implicit, budgetRequest(0), new ByteVocab(), closed);
        assertFalse(stage.armsAtOutput());
        Sampler sampler = sampler(stage);
        accept(sampler, bytes(closed));
        assertEquals('z', sampler.sample(logits('z')));
        assertEquals(ReasoningBudgetStage.State.DONE, stage.state());

        ReasoningBudgetStage tagged = ReasoningBudgetStage.of(ReasoningFilter.thinkTags(), budgetRequest(0),
                new ByteVocab(), "<|im_start|>assistant\n");
        assertFalse(tagged.armsAtOutput());
        Sampler plain = sampler(tagged);
        accept(plain, bytes("<|im_start|>assistant\n"));
        assertEquals('z', plain.sample(logits('z')));
        assertEquals(ReasoningBudgetStage.State.IDLE, tagged.state());
    }

    @Test
    public void aStartTagInThePromptKeepsItsCountWhenTheParserAlsoOpensImplicitly() {
        String prompt = "assistant " + START;
        ReasoningBudgetStage stage = ReasoningBudgetStage.of(ReasoningFilter.thinkTags().opensImplicitly(),
                budgetRequest(2), new ByteVocab(), prompt);
        Sampler sampler = sampler(stage);
        accept(sampler, bytes(prompt));
        assertEquals(ReasoningBudgetStage.State.COUNTING, stage.state());
        assertEquals('a', sampler.sample(logits('a')));
        sampler.accept('a');
        assertEquals('b', sampler.sample(logits('b')));
        sampler.accept('b');
        assertEquals("the prompt's start tag armed it once", ReasoningBudgetStage.State.FORCING, stage.state());
    }

    @Test
    public void aNaturalEndTagDeactivatesWithoutForcing() {
        ReasoningBudgetStage stage = stage(budgetRequest(20));
        Sampler sampler = sampler(stage);
        accept(sampler, bytes(START + "hm" + END));
        assertEquals(ReasoningBudgetStage.State.DONE, stage.state());
        assertEquals('z', sampler.sample(logits('z')));
    }

    @Test
    public void aZeroBudgetForcesImmediatelyAfterTheStart() {
        ReasoningBudgetStage stage = stage(budgetRequest(0));
        Sampler sampler = sampler(stage);
        accept(sampler, bytes("hi" + START));
        assertEquals(ReasoningBudgetStage.State.FORCING, stage.state());
        assertEquals(bytes(END)[0], sampler.sample(logits('z')));
    }

    @Test
    public void aNewStartTagRearmsAFreshBudget() {
        ReasoningBudgetStage stage = stage(budgetRequest(1));
        Sampler sampler = sampler(stage);
        accept(sampler, bytes(START + "a" + END + "tail" + START));
        assertEquals(ReasoningBudgetStage.State.COUNTING, stage.state());
        assertEquals('x', sampler.sample(logits('x')));
        sampler.accept('x');
        assertEquals("the second block's budget is spent", ReasoningBudgetStage.State.FORCING, stage.state());
    }

    @Test
    public void anIncompleteUtf8PieceDelaysTheForcing() {
        ReasoningBudgetStage stage = stage(budgetRequest(1));
        Sampler sampler = sampler(stage);
        accept(sampler, bytes(START));
        sampler.accept(0xC3);                                   // the first byte of a two-byte character spends the budget
        assertEquals(ReasoningBudgetStage.State.WAITING_UTF8, stage.state());
        sampler.accept(0xA9);                                   // a lone continuation byte is still no complete piece
        assertEquals(ReasoningBudgetStage.State.WAITING_UTF8, stage.state());
        assertEquals('x', sampler.sample(logits('x')));         // waiting passes through
        sampler.accept('x');
        assertEquals("a complete ASCII piece completes the wait", ReasoningBudgetStage.State.FORCING, stage.state());
        assertEquals(bytes(END)[0], sampler.sample(logits('x')));
    }

    @Test
    public void forcingSurvivesATopKThatTruncatedTheForcedToken() {
        ReasoningBudgetStage stage = stage(budgetRequest(0));
        ByteVocab vocab = new ByteVocab();
        // a sampling chain whose top_k = 1 truncates to the argmax before the budget stage applies
        Sampler sampler = Sampler.create(SamplerConfig.defaults().toBuilder().topK(1).seed(7).build(), new int[0],
                Collections.<SamplerStage>singletonList(stage), vocab);
        accept(sampler, bytes(START));
        assertEquals(bytes(END)[0], sampler.sample(logits('z')));
    }

    @Test
    public void forceMovesOnlyACountingStage() {
        ReasoningBudgetStage stage = stage(GenerationRequest.builder().maxNewTokens(64).reasoningControl(true).build());
        Sampler sampler = sampler(stage);
        assertFalse("idle: nothing to force", stage.force());
        accept(sampler, bytes(START));
        assertTrue("counting: the control ends the reasoning", stage.force());
        assertEquals(ReasoningBudgetStage.State.FORCING, stage.state());
        assertFalse("already forcing", stage.force());
        assertTrue(stage.forcing());
        assertEquals(bytes(END)[0], sampler.sample(logits('z')));
    }

    @Test
    public void resetRestartsTheMachine() {
        ReasoningBudgetStage stage = stage(budgetRequest(0));
        Sampler sampler = sampler(stage);
        accept(sampler, bytes(START));
        assertEquals(ReasoningBudgetStage.State.FORCING, stage.state());
        sampler.reset();
        assertEquals(ReasoningBudgetStage.State.IDLE, stage.state());
        assertEquals('x', sampler.sample(logits('x')));
    }

    @Test
    public void theStageKeepsTheChainOffTheOnDeviceArgmaxPath() {
        ReasoningBudgetStage stage = stage(budgetRequest(8));
        Sampler sampler = sampler(stage);
        assertFalse(sampler.argmaxOnly());
        Sampler plain = Sampler.create(SamplerConfig.greedyDecoding());
        assertTrue(plain.argmaxOnly());
    }

    @Test
    public void theFactoryTakesTheParserTagsAndTheExplicitOverrides() {
        ByteVocab vocab = new ByteVocab();
        ReasoningBudgetStage fromParser = ReasoningBudgetStage.of(ReasoningFilter.thinkTags(),
                budgetRequest(4), vocab);
        assertEquals(ReasoningBudgetStage.State.IDLE, fromParser.state());
        ReasoningBudgetStage overridden = ReasoningBudgetStage.of(ReasoningFilter.none(),
                GenerationRequest.builder().maxNewTokens(8).reasoningBudget(4)
                        .reasoningBudgetStartTags(List.of("[THINK]")).reasoningBudgetEndTags(List.of("[/THINK]"))
                        .build(),
                vocab);
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.<SamplerStage>singletonList(overridden), vocab);
        accept(sampler, bytes("x[THINK]"));
        assertEquals("the explicit tags arm it where the parser names none",
                ReasoningBudgetStage.State.COUNTING, overridden.state());
    }

    @Test
    public void unarmedRequestsBuildNoStageAndUnresolvableTagsRefuse() {
        assertNull(ReasoningBudgetStage.of(ReasoningFilter.thinkTags(), GenerationRequest.builder().build(),
                new ByteVocab()));
        try {
            ReasoningBudgetStage.of(ReasoningFilter.none(), budgetRequest(4), new ByteVocab());
            throw new AssertionError("expected a refusal");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("reasoning_budget_start_tag"));
        }
    }

    @Test
    public void outOfRangeBudgetsRefuse() {
        try {
            GenerationRequest.builder().reasoningBudget(-2);
            throw new AssertionError("expected a refusal");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("reasoningBudget"));
        }
    }

    @Test
    public void utf8CompletenessFollowsLlamaCpp() {
        assertTrue(ReasoningBudgetStage.isCompleteUtf8(new byte[0]));
        assertTrue(ReasoningBudgetStage.isCompleteUtf8(new byte[]{'a'}));
        assertTrue(ReasoningBudgetStage.isCompleteUtf8(new byte[]{'a', (byte) 0xC3, (byte) 0xA9}));
        assertFalse(ReasoningBudgetStage.isCompleteUtf8(new byte[]{(byte) 0xC3}));
        assertFalse(ReasoningBudgetStage.isCompleteUtf8(new byte[]{(byte) 0xA9}));
        assertFalse(ReasoningBudgetStage.isCompleteUtf8(new byte[]{(byte) 0xE2, (byte) 0x82}));
        assertTrue(ReasoningBudgetStage.isCompleteUtf8(new byte[]{(byte) 0xE2, (byte) 0x82, (byte) 0xAC}));
    }
}
