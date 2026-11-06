package com.eonmux.neuralfree.demo;

import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link Interrupts} has the llama.cpp CLI semantics: the first Ctrl-C sets the flag the session cancels with, a
 * second press before the flag is cleared exits with 130, and a press while the session waits for input exits.
 */
public class InterruptsTest {

    private final List<Integer> exits = new ArrayList<>();
    private final Interrupts interrupts = new Interrupts(exits::add,
            new PrintStream(new ByteArrayOutputStream(), true));

    @Test
    public void theFirstPressSetsTheFlagWithoutExiting() {
        assertFalse(interrupts.requested());
        interrupts.press();
        assertTrue(interrupts.requested());
        assertTrue(exits.isEmpty());
        interrupts.clear();
        assertFalse(interrupts.requested());
        interrupts.press();
        assertTrue("a cleared flag makes the next press a first one again", exits.isEmpty());
    }

    @Test
    public void aSecondPressExitsWith130() {
        interrupts.press();
        interrupts.press();
        assertEquals(List.of(Interrupts.EXIT_CODE), exits);
        assertEquals(130, Interrupts.EXIT_CODE);
    }

    @Test
    public void aPressAtThePromptExits() {
        interrupts.awaitingInput(true);
        interrupts.press();
        assertEquals(List.of(130), exits);
        assertTrue("the flag is set, so a session that keeps running ends its loop", interrupts.requested());
        interrupts.clear();
        interrupts.awaitingInput(false);
        interrupts.press();
        assertEquals("outside the prompt the first press only sets the flag", 1, exits.size());
    }
}
