package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The shell-style splitter of the sweep and startup command lines: whitespace, both quote kinds,
 * backslash escapes and the unterminated-quote refusal, matching {@code shlex.split}.
 */
public class ShellSplitTest {

    @Test
    public void wordsSplitOnWhitespace() {
        assertEquals(Arrays.asList("a", "b", "c"), ShellSplit.split("  a\tb\n c "));
    }

    @Test
    public void quotesGroupAndAreRemoved() {
        assertEquals(Arrays.asList("a b", "c"), ShellSplit.split("'a b' c"));
        assertEquals(Arrays.asList("a b", "c"), ShellSplit.split("\"a b\" c"));
    }

    @Test
    public void aBackslashEscapesTheNextCharacter() {
        assertEquals(Arrays.asList("a b"), ShellSplit.split("a\\ b"));
        assertEquals(Arrays.asList("a\"b"), ShellSplit.split("a\\\"b"));
        assertEquals(Arrays.asList("a\\b"), ShellSplit.split("\"a\\\\b\""));
    }

    @Test
    public void anEmptyQuotedWordIsAnArgument() {
        assertEquals(Arrays.asList("", "b"), ShellSplit.split("'' b"));
    }

    @Test
    public void anUnterminatedQuoteIsRefused() {
        try {
            ShellSplit.split("'open");
            fail("an unterminated quote must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("unterminated"));
        }
    }
}
