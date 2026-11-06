package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Tests of the code-point range set behind output-regex and key-pattern character classes
 * ({@link RegexCharSet}): union merges overlapping and adjacent ranges, complement covers exactly
 * the rest of Unicode, difference and membership are exact, the GBNF rendering escapes the
 * character-set metacharacters and the Java rendering matches exactly the set. Pure logic.
 */
public class RegexCharSetTest {

    private static final String MAX = new String(Character.toChars(RegexCharSet.MAX_CODE_POINT));

    @Test
    public void unionMergesOverlappingAndAdjacentRanges() {
        RegexCharSet set = RegexCharSet.range('a', 'c').union(RegexCharSet.range('b', 'f'))
                .union(RegexCharSet.single('g')).union(RegexCharSet.single('0'));
        assertEquals("[0a-g]", set.toGbnf(false));
        assertEquals("[^0a-g]", set.toGbnf(true));
    }

    @Test
    public void complementCoversTheRestOfUnicode() {
        assertEquals("[\\x00-/:-" + MAX + "]", RegexCharSet.DIGIT.complement().toGbnf(false));
        assertEquals("[0-9]", RegexCharSet.DIGIT.complement().complement().toGbnf(false));
        assertTrue(RegexCharSet.SPACE.union(RegexCharSet.SPACE.complement()).complement().isEmpty());
        assertEquals("[\\x00-" + MAX + "]", RegexCharSet.EMPTY.complement().toGbnf(false));
    }

    @Test
    public void theShorthandSetsAreEcmaScripts() {
        assertEquals("[0-9]", RegexCharSet.shorthand('d').toGbnf(false));
        assertEquals("[0-9A-Z_a-z]", RegexCharSet.shorthand('w').toGbnf(false));
        assertEquals("[\\t-\\r \u00A0\u1680\u2000-\u200A\u2028-\u2029\u202F\u205F\u3000\uFEFF]",
                RegexCharSet.shorthand('s').toGbnf(false));
        assertEquals(RegexCharSet.WORD.complement().toGbnf(false), RegexCharSet.shorthand('W').toGbnf(false));
    }

    @Test
    public void classMetacharactersAndControlsAreEscaped() {
        assertEquals("[\\\\]", RegexCharSet.single('\\').toGbnf(false));
        assertEquals("[\\]]", RegexCharSet.single(']').toGbnf(false));
        assertEquals("[\\[]", RegexCharSet.single('[').toGbnf(false));
        assertEquals("[\\-]", RegexCharSet.single('-').toGbnf(false));
        assertEquals("[\\x5E]", RegexCharSet.single('^').toGbnf(false));
        assertEquals("[\\[-\\x5E]", RegexCharSet.range('[', '^').toGbnf(false));
        assertEquals("[\\x7F\\uD800]", RegexCharSet.single(0x7F).union(RegexCharSet.single(0xD800))
                .toGbnf(false));
        assertEquals("\\\"", RegexCharSet.literalChar('"'));
        assertEquals("\\x0B", RegexCharSet.literalChar(0x0B));
        assertEquals("\u00E9", RegexCharSet.literalChar(0xE9));
    }

    @Test
    public void membershipAndDifference() {
        RegexCharSet letters = RegexCharSet.range('a', 'z');
        assertTrue(letters.contains('a') && letters.contains('z'));
        assertFalse(letters.contains('`') || letters.contains('{'));
        assertEquals("[a-cx-z]", letters.minus(RegexCharSet.range('d', 'w')).toGbnf(false));
        assertTrue(letters.minus(RegexCharSet.WORD).isEmpty());
        assertFalse(RegexCharSet.NON_LINE_TERMINATOR.contains('\n'));
        assertFalse(RegexCharSet.NON_LINE_TERMINATOR.contains(0x2029));
        assertTrue(RegexCharSet.NON_LINE_TERMINATOR.contains('\t'));
    }

    @Test
    public void javaRegexClassesMatchExactlyTheSet() {
        assertEquals("[\\x{2d}0-9a]",RegexCharSet.DIGIT.union(RegexCharSet.single('a'))
                .union(RegexCharSet.single('-')).toJavaRegex());
        Pattern space = Pattern.compile(RegexCharSet.SPACE.toJavaRegex());
        assertTrue(space.matcher("\u2028").matches());
        assertFalse(space.matcher("x").matches());
        Pattern none = Pattern.compile(RegexCharSet.EMPTY.toJavaRegex());
        assertFalse(none.matcher("a").find());
    }
}
