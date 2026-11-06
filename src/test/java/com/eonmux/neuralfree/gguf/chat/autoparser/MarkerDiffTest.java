package com.eonmux.neuralfree.gguf.chat.autoparser;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The diff split matches llama.cpp's {@code calculate_diff_split} (common/chat-auto-parser-helpers.cpp): the cases
 * of tests/test-chat-auto-parser.cpp, plus marker segmentation.
 */
public class MarkerDiffTest {

    @Test
    public void basicDifference() {
        MarkerDiff.Diff d = MarkerDiff.split("hello world", "hello test");
        assertEquals("hello ", d.prefix());
        assertEquals("world", d.left());
        assertEquals("test", d.right());
        assertEquals("", d.suffix());

        d = MarkerDiff.split("abc", "xyz");
        assertEquals("", d.prefix());
        assertEquals("abc", d.left());
        assertEquals("xyz", d.right());

        d = MarkerDiff.split("prefixA suffix", "prefixB suffix");
        assertEquals("prefix", d.prefix());
        assertEquals("A", d.left());
        assertEquals("B", d.right());
        assertEquals(" suffix", d.suffix());
    }

    @Test
    public void identicalStrings() {
        MarkerDiff.Diff d = MarkerDiff.split("hello", "hello");
        assertEquals("hello", d.prefix());
        assertEquals("", d.left());
        assertEquals("", d.right());
        assertEquals("", d.suffix());

        d = MarkerDiff.split("", "");
        assertEquals("", d.prefix());
        assertEquals("", d.left());
        assertEquals("", d.right());
    }

    @Test
    public void commonPrefix() {
        MarkerDiff.Diff d = MarkerDiff.split("abcdef", "abcxyz");
        assertEquals("abc", d.prefix());
        assertEquals("def", d.left());
        assertEquals("xyz", d.right());

        d = MarkerDiff.split("same", "sameagain");
        assertEquals("same", d.prefix());
        assertEquals("", d.left());
        assertEquals("again", d.right());
    }

    @Test
    public void commonSuffix() {
        MarkerDiff.Diff d = MarkerDiff.split("123end", "456end");
        assertEquals("", d.prefix());
        assertEquals("123", d.left());
        assertEquals("456", d.right());
        assertEquals("end", d.suffix());

        d = MarkerDiff.split("start", "end");
        assertEquals("", d.prefix());
        assertEquals("start", d.left());
        assertEquals("end", d.right());
    }

    @Test
    public void commonPrefixAndSuffix() {
        MarkerDiff.Diff d = MarkerDiff.split("helloXworld", "helloYworld");
        assertEquals("hello", d.prefix());
        assertEquals("X", d.left());
        assertEquals("Y", d.right());
        assertEquals("world", d.suffix());

        d = MarkerDiff.split("ABCmiddleXYZ", "ABCdifferentXYZ");
        assertEquals("ABC", d.prefix());
        assertEquals("middle", d.left());
        assertEquals("different", d.right());
        assertEquals("XYZ", d.suffix());
    }

    @Test
    public void emptyInputs() {
        MarkerDiff.Diff d = MarkerDiff.split("", "hello");
        assertEquals("", d.prefix());
        assertEquals("", d.left());
        assertEquals("hello", d.right());

        d = MarkerDiff.split("hello", "");
        assertEquals("", d.prefix());
        assertEquals("hello", d.left());
        assertEquals("", d.right());
    }

    @Test
    public void overlapKeepsOneCopyOfTheText() {
        MarkerDiff.Diff d = MarkerDiff.split("aa", "ab");
        assertEquals("a", d.prefix());
        assertEquals("a", d.left());
        assertEquals("b", d.right());
        assertEquals("", d.suffix());
    }

    @Test
    public void prefixInsertionRotatesTheDiff() {
        MarkerDiff.Diff d = MarkerDiff.split("same", "sameagain");
        assertEquals("same", d.prefix());
        assertEquals("", d.left());
        assertEquals("again", d.right());
        assertEquals("", d.suffix());
    }

    @Test
    public void generationPromptAppendedToAPrefixIsTheUniquePart() {
        String left = "<|im_start|>user\nHello<|im_end|>\n";
        String right = left + "<|im_start|>assistant\n<think>\n";
        MarkerDiff.Diff d = MarkerDiff.split(left, right);
        assertEquals(left, d.prefix());
        assertEquals("", d.left());
        assertEquals("<|im_start|>assistant\n<think>\n", d.right());
        assertEquals("", d.suffix());
    }

    @Test
    public void markersSegmentOnAngleAndSquareBrackets() {
        List<MarkerDiff.Segment> segments = MarkerDiff.segmentize("a<tag>b[call]c");
        assertEquals(5, segments.size());
        assertEquals(MarkerDiff.Type.TEXT, segments.get(0).type());
        assertEquals("a", segments.get(0).value());
        assertEquals(MarkerDiff.Type.MARKER, segments.get(1).type());
        assertEquals("<tag>", segments.get(1).value());
        assertEquals(MarkerDiff.Type.MARKER, segments.get(3).type());
        assertEquals("[call]", segments.get(3).value());
    }

    @Test
    public void anUnfinishedMarkerStaysText() {
        List<MarkerDiff.Segment> segments = MarkerDiff.segmentize("a<tag");
        StringBuilder joined = new StringBuilder();
        for (MarkerDiff.Segment segment : segments) {
            assertEquals(MarkerDiff.Type.TEXT, segment.type());
            joined.append(segment.value());
        }
        assertEquals("a<tag", joined.toString());
    }

    @Test
    public void markerBoundariesWinOverCharacterSplits() {
        // The opening tag is shared whole, so it is not split into a prefix '<' and a differing 't'.
        MarkerDiff.Diff d = MarkerDiff.split("prefix<tag>", "prefix</tag>suffix");
        assertEquals("prefix", d.prefix());
        assertEquals("<tag>", d.left());
        assertEquals("</tag>suffix", d.right());
    }

    @Test
    public void commonPrefixAndSuffixHelpers() {
        assertEquals("ab", MarkerDiff.commonPrefix("abcd", "abxy"));
        assertEquals("cd", MarkerDiff.commonSuffix("abcd", "xycd"));
        assertEquals("head ", MarkerDiff.untilCommonPrefix("head TARGETbody", "TARGET", "TARGETx"));
        assertEquals(" tail", MarkerDiff.afterCommonSuffix("xEND tail", "END", "END"));
        assertTrue(Arrays.asList("a").contains(MarkerDiff.commonPrefix("a", "a")));
    }
}
