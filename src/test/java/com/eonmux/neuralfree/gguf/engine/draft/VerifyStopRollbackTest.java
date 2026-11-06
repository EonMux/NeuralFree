package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import org.junit.Test;

import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.logits;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.request;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.sampling;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The state handling of a SpeculativeDecoder verify that the budget stops early: KV memory rolls
 * the unconfirmed positions back, recurrent memory restores its snapshot and replays the kept tokens, and a block
 * confirmed up to the budget drops nothing on either.
 */
public class VerifyStopRollbackTest {

    /** A stop inside the verify rebuilds the KV memory: the rejected positions are rolled back. */
    @Test
    public void stopInsideVerifyRollsBackTheUnconfirmedPositionsOnKvMemory() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the third row is never read: the replacement at position 1 spends the budget
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).build();

        GenerationResult r = dec.generate(new int[]{0}, request(3, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 0}, r.tokens());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertEquals(Arrays.asList(1), target.rollbacks);
        assertEquals(3, target.memory().sequence().nPast());
    }

    /** A fully confirmed block that ends the budget drops nothing. */
    @Test
    public void fullConfirmationInsideTheBudgetLeavesNoRejectedPositions() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(0.0, 0.0, 0.0, 1.0), logits(1.0, 0.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).build();

        GenerationResult r = dec.generate(new int[]{0}, request(4, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 3, 0}, r.tokens());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertTrue(target.rollbacks.isEmpty());
        assertEquals(4, target.memory().sequence().nPast());
    }

    /** Recurrent memory restores its snapshot and replays the kept tokens, after a stop like after a rejection. */
    @Test
    public void stopInsideVerifyRestoresAndReplaysOnRecurrentMemory() {
        FakeTarget target = new FakeTarget(true)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the third row is never read: the replacement at position 1 spends the budget
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0))
                // the replay prefill of the kept tokens reads one row of its own
                .add(logits(0.0, 1.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).build();

        GenerationResult r = dec.generate(new int[]{0}, request(3, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 0}, r.tokens());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertEquals(1, target.snapshots);
        assertEquals(1, target.restores);
        assertTrue(target.rollbacks.isEmpty());
        assertEquals(2, target.prefills.size());
        assertArrayEquals(new int[]{0}, target.prefills.get(0));
        assertArrayEquals(new int[]{1, 2}, target.prefills.get(1));
        assertEquals(1, target.prefillPositions.get(1).intValue());
        assertEquals(3, target.memory().sequence().nPast());
    }

    /** A block confirmed up to the budget needs no restore either. */
    @Test
    public void fullConfirmationInsideTheBudgetNeedsNoRestoreOnRecurrentMemory() {
        FakeTarget target = new FakeTarget(true)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(0.0, 0.0, 0.0, 1.0), logits(1.0, 0.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).build();

        GenerationResult r = dec.generate(new int[]{0}, request(4, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 3, 0}, r.tokens());
        assertEquals(1, target.snapshots);
        assertEquals(0, target.restores);
        assertEquals(4, target.memory().sequence().nPast());
    }
}
