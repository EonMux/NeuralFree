package com.eonmux.neuralfree.demo;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * {@link MultilineInput} reads as llama.cpp common/console.cpp {@code readline_simple} with the caller's
 * {@code do ... while (another_line)} loop and the trailing-newline pop: {@code /} returns control, {@code \}
 * flips the line's default and keeps the newline, multiline mode joins until one of the two, and end of input
 * with nothing read is null.
 */
public class MultilineInputTest {

    private static MultilineInput reader(String text) {
        return new MultilineInput(new BufferedReader(new StringReader(text)));
    }

    @Test
    public void aPlainLineSubmitsWithoutItsNewline() throws IOException {
        MultilineInput in = reader("hello\nworld\n");
        assertEquals("hello", in.read(false));
        assertEquals("world", in.read(false));
        assertNull("end of input with nothing read", in.read(false));
    }

    @Test
    public void anEmptyLinePassesControlBack() throws IOException {
        assertEquals("", reader("\nnext\n").read(false));
    }

    @Test
    public void aBackslashJoinsTheNextLineKeepingTheNewline() throws IOException {
        MultilineInput in = reader("first\\\nsecond\nthird\n");
        assertEquals("first\nsecond", in.read(false));
        assertEquals("third", in.read(false));
    }

    @Test
    public void everyBackslashLineFlipsItsOwnDefaultNotTheNextOne() throws IOException {
        assertEquals("the reference passes the original mode to every readline call", "a\nb\nc",
                reader("a\\\nb\\\nc\n").read(false));
    }

    @Test
    public void aSlashReturnsControlWithoutTheSlash() throws IOException {
        assertEquals("done", reader("done/\nmore\n").read(false));
        assertEquals("a\nb", reader("a\\\nb/\n").read(false));
    }

    @Test
    public void multilineModeJoinsUntilASlashOrBackslash() throws IOException {
        MultilineInput in = reader("one\n\ntwo\nthree\\\nnext/\n");
        assertEquals("empty lines join in multiline mode", "one\n\ntwo\nthree", in.read(true));
        assertEquals("next", in.read(true));
    }

    @Test
    public void endOfInputSubmitsWhatWasJoined() throws IOException {
        assertEquals("partial", reader("partial\\\n").read(false));
        assertEquals("x\ny", reader("x\ny").read(true));
    }
}
