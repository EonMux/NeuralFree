package com.eonmux.neuralfree.server.tools;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The glob semantics the file and grep searches filter with (llama.cpp's {@code glob_match}). */
public class ToolGlobTest {

    @Test
    void aPatternWithoutSlashMatchesTheFileNameAtAnyDepth() {
        assertTrue(ToolGlob.match("*.cpp", "src/deep/A.cpp"));
        assertTrue(ToolGlob.match("*.cpp", "A.cpp"));
        assertFalse(ToolGlob.match("*.cpp", "src/A.h"));
        assertFalse(ToolGlob.match("a*.cpp", "src/bA.cpp"));
    }

    @Test
    void aPatternWithSeparatorsMatchesTheWholeRelativePathAndAutoAnchors() {
        assertTrue(ToolGlob.match("src/*.cpp", "src/A.cpp"));
        assertTrue(ToolGlob.match("src/*.cpp", "deep/src/A.cpp"), "auto-prefixed with the recursive star");
        assertFalse(ToolGlob.match("src/*.cpp", "other/A.cpp"));
        assertFalse(ToolGlob.match("src/*.cpp", "src/deep/A.cpp"), "one star stays inside one segment");
    }

    @Test
    void aPatternAlreadyAnchoredIsUsedAsGiven() {
        assertTrue(ToolGlob.match("**", "any/where.txt"));
        assertTrue(ToolGlob.match("**/A.cpp", "x/y/A.cpp"));
        assertFalse(ToolGlob.match("**/A.cpp", "x/y/A.h"));
    }

    @Test
    void theRecursiveStarCrossesSeparatorsAndTheStarDoesNot() {
        assertTrue(ToolGlob.match("a/**/b", "a/x/y/b"));
        assertTrue(ToolGlob.match("a/**/b", "a/b"));
        assertTrue(ToolGlob.match("a/**", "a/x/y"));
        assertFalse(ToolGlob.match("a/*", "a/b/c"));
        assertTrue(ToolGlob.match("a/*", "a/b"));
    }

    @Test
    void theQuestionMarkMatchesOneNonSeparator() {
        assertTrue(ToolGlob.match("a?c", "abc"));
        assertFalse(ToolGlob.match("a?c", "ac"));
        assertFalse(ToolGlob.match("a?c", "a/c"));
    }

    @Test
    void aCharacterClassRangesAndNegates() {
        assertTrue(ToolGlob.match("[a-c]at", "bat"));
        assertFalse(ToolGlob.match("[a-c]at", "dat"));
        assertTrue(ToolGlob.match("[!a]x", "bx"));
        assertFalse(ToolGlob.match("[!a]x", "ax"));
        assertTrue(ToolGlob.match("[]-]x", "]x"), "a leading ] and - are literals");
        assertFalse(ToolGlob.match("[a", "a"), "an unclosed class is a literal bracket");
        assertFalse(ToolGlob.match("[a", "["), "the bracket alone does not complete the pattern");
    }
}
