package com.eonmux.neuralfree.gguf.engine.watermark;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link WatermarkState}: the repeated-context search of the generation scopes, with the prompt boundary and the
 * history window vLLM's {@code repeated_context_mask} applies.
 */
public class WatermarkStateTest {

    private static WatermarkConfig config(int contextWidth, WatermarkContextScope scope, Integer maxHistory) {
        return WatermarkConfig.builder(5L).contextWidth(contextWidth).deduplicateContexts(scope)
                .deduplicateContextsMaxHistory(maxHistory).build();
    }

    @Test
    public void thePrfContextIsTheLastAcceptedTokensPaddedWithMinusOne() {
        WatermarkState state = new WatermarkState(config(3, WatermarkContextScope.NONE, 8192));
        assertArrayEquals(new int[]{-1, -1, -1}, state.context());
        state.accept(7);
        state.accept(8);
        state.accept(9);
        state.accept(10);
        assertArrayEquals(new int[]{8, 9, 10}, state.context());
        assertEquals(4, state.accepted());
    }

    @Test
    public void theNoneScopeNeverRepeats() {
        WatermarkState state = new WatermarkState(config(1, WatermarkContextScope.NONE, 8192));
        state.accept(3);
        state.accept(4);
        state.accept(3);
        assertFalse(state.repeatedContext());
    }

    @Test
    public void theSingleTurnScopeSearchesOnlyGeneratedTokens() {
        // a prompt that repeats a context does not make the first generated token's context repeated
        WatermarkState state = new WatermarkState(config(2, WatermarkContextScope.SINGLE_TURN, 8192));
        state.startPrompt();
        state.accept(5);
        state.accept(5);
        state.accept(5);
        state.accept(5);
        state.endPrompt();
        state.accept(9);
        assertFalse(state.repeatedContext());
        // the generated window [-1, 9] has not occurred before
        state.accept(9);
        assertFalse(state.repeatedContext());
        // the generated window [9, 9] occurred at the previous generated position
        state.accept(9);
        assertTrue(state.repeatedContext());
    }

    @Test
    public void theAllScopeSkipsPartialContextsAndSearchesThePrompt() {
        WatermarkState state = new WatermarkState(config(2, WatermarkContextScope.ALL, 8192));
        state.startPrompt();
        state.accept(5);
        state.accept(9);
        state.accept(9);
        state.endPrompt();
        // no generated token yet: the context is incomplete
        assertTrue(state.repeatedContext());
        state.accept(9);
        // one generated token: still incomplete
        assertTrue(state.repeatedContext());
        state.accept(9);
        // the complete generated context [9, 9] occurred in the prompt
        assertTrue(state.repeatedContext());
    }

    @Test
    public void theHistoryWindowBoundsTheSearch() {
        WatermarkState recent = new WatermarkState(config(1, WatermarkContextScope.SINGLE_TURN, 2));
        recent.accept(3);
        recent.accept(4);
        recent.accept(5);
        recent.accept(3);
        assertFalse("an occurrence outside the two-position window does not repeat", recent.repeatedContext());

        WatermarkState whole = new WatermarkState(config(1, WatermarkContextScope.SINGLE_TURN, null));
        whole.accept(3);
        whole.accept(4);
        whole.accept(5);
        whole.accept(3);
        assertTrue("the whole scope searches every position", whole.repeatedContext());
    }

    @Test
    public void aSavedStateRestoresItsContextAndHistory() {
        WatermarkState state = new WatermarkState(config(2, WatermarkContextScope.SINGLE_TURN, 8192));
        state.accept(1);
        state.accept(2);
        WatermarkState.Snapshot saved = state.save();
        state.accept(3);
        state.accept(3);
        state.accept(3);
        state.restore(saved);
        assertArrayEquals(new int[]{1, 2}, state.context());
        assertEquals(2, state.accepted());
        assertFalse(state.repeatedContext());
        state.accept(3);
        state.accept(3);
        assertFalse(state.repeatedContext());
        state.accept(3);
        assertTrue(state.repeatedContext());
    }

    @Test
    public void aResetForgetsEverything() {
        WatermarkState state = new WatermarkState(config(2, WatermarkContextScope.ALL, 8192));
        state.accept(1);
        state.accept(2);
        state.reset();
        assertArrayEquals(new int[]{-1, -1}, state.context());
        assertEquals(0, state.accepted());
    }

    @Test
    public void negativeTokensAreRefused() {
        WatermarkState state = new WatermarkState(config(2, WatermarkContextScope.NONE, 8192));
        org.junit.Assert.assertThrows(IllegalArgumentException.class, () -> state.accept(-1));
    }
}
