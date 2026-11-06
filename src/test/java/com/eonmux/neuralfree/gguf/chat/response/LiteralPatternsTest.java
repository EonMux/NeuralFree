package com.eonmux.neuralfree.gguf.chat.response;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The literals a delimiter pattern spells ({@link LiteralPatterns}, vLLM {@code _literals}) and where a {@code ^} or a
 * lookahead counts as vLLM counts it: only the grammar reads a {@code ^} as the message start (a field's openers) and a
 * final lookahead as the next parameter's opener (an {@code xml-inline} value end); a closer's completion
 * ({@code _closer_rest}) and the message-start anchors take neither, so a pattern with one spells no literal there.
 */
public class LiteralPatternsTest {

    private static final String AHEAD = LiteralPatterns.AHEAD;

    @Test
    public void alternationsGroupsEscapesAndOptionalAtomsExpand() {
        assertEquals(Arrays.asList("<a>", "<b>"), LiteralPatterns.expand("<a>|<b>"));
        assertEquals(Arrays.asList("x", "xy", "xz"), LiteralPatterns.expand("x(?:y|z)?"));
        assertEquals(Collections.singletonList("<|end|>\n"), LiteralPatterns.expand("<\\|end\\|>\\n"));
        assertNull("a class spells more than literals", LiteralPatterns.expand("a[bc]"));
        assertNull("a dot spells more than literals", LiteralPatterns.expand("a."));
        assertNull("a shorthand spells more than literals", LiteralPatterns.expand("a\\w"));
        StringBuilder many = new StringBuilder("(?:a|b)");
        for (int i = 0; i < 4; i++) many.append("(?:a|b)");
        assertNull("more than " + LiteralPatterns.MAX_LITERALS + " strings", LiteralPatterns.expand(many.toString()));
    }

    @Test
    public void aStartAnchorSpellsLiteralsOnlyAtTheMessageStart() {
        assertNull(LiteralPatterns.expand("^<s>"));
        assertEquals(Collections.singletonList("<s>"), LiteralPatterns.expand("^<s>", true, false));
        assertEquals("a ^ inside a literal drops it", Collections.singletonList("<s>"),
                LiteralPatterns.expand("<s>|a^b", true, false));
        assertEquals(Collections.<String>emptyList(), LiteralPatterns.expand("^", true, false));
        assertEquals(Arrays.asList("<t>", "\n<t>"), LiteralPatterns.expand("(?:^|\\n)<t>", true, false));
    }

    @Test
    public void aFinalLookaheadSpellsItsTextAfterTheMarker() {
        String end = "\\n(?:</p>|(?=<p name=\"))";
        assertNull(LiteralPatterns.expand(end));
        assertEquals(Arrays.asList("\n</p>", "\n" + AHEAD + "<p name=\""), LiteralPatterns.expand(end, false, true));
        assertNull("a lookahead reads as a literal only at the end", LiteralPatterns.expand("(?=a)b", false, true));
        assertNull("one lookahead per literal", LiteralPatterns.expand("a(?=b)(?=c)", false, true));
    }

    @Test
    public void aCloserCompletesOnlyWithLiteralStartAnchors() {
        String output = "<call>{\"function\": {\"name\": \"f\", \"arguments\": {}}}</call>";
        assertEquals("<start>", closerRest("\"start_anchor\": \"<start>\"", "\"close\": \"</call><start>\"", output));
        assertEquals("<start>", closerRest("\"start_anchor_pattern\": \"<start>|<begin>\"",
                "\"close\": \"</call><start>\"", output));
        // vLLM's _closer_rest and message_start_anchors read no ^ and no lookahead
        assertEquals("", closerRest("\"start_anchor_pattern\": \"^<start>\"", "\"close\": \"</call><start>\"", output));
        assertEquals("", closerRest("\"start_anchor_pattern\": \"<start>(?=x)\"", "\"close\": \"</call><start>\"",
                output));
        assertEquals("a closer pattern has no literals to complete", "", closerRest("\"start_anchor\": \"<start>\"",
                "\"close_pattern\": \"</call><start>\"", output));
    }

    private static String closerRest(String anchor, String close, String output) {
        String json = "{" + anchor + ", \"fields\": {\"thinking\": {\"open\": \"<t>\", \"close\": \"</t>\"},"
                + " \"tool_calls\": {\"open\": \"<call>\", " + close + ", \"content\": \"json\"}, \"content\": {}}}";
        ResponseParser p = new ResponseParser(ResponseTemplate.parse(json).forServing(), "<start>", ToolArguments.NONE,
                null);
        p.feed(output, new ResponseParser.Listener() {
        });
        return p.closerRest();
    }

    @Test
    public void messageStartAnchorsAreTheLiteralsTheAnchorSpellsWithoutFlags() {
        assertEquals(Arrays.asList("<start>", "<begin>"), anchors("\"start_anchor_pattern\": \"<start>|<begin>\""));
        assertNull(anchors("\"start_anchor_pattern\": \"^<start>\""));
        assertEquals(Collections.<String>emptyList(), TemplateGrammar.messageStartAnchors(
                template("\"start_anchor_pattern\": \"^<start>\"", "\"open\": \"<t>\"")));
    }

    private static List<String> anchors(String anchor) {
        return template(anchor, "\"open\": \"<t>\"").messageStartAnchors();
    }

    private static ResponseTemplate template(String anchor, String thinkingOpen) {
        return ResponseTemplate.parse("{" + anchor + ", \"fields\": {\"thinking\": {" + thinkingOpen
                + ", \"close\": \"</t>\"}, \"content\": {}}}");
    }

    @Test
    public void aFieldOpenerWithAStartAnchorOpensOnlyAtTheMessageStart() {
        ResponseTemplate.Field thinking = template("\"start_anchor\": \"<s>\"", "\"open_pattern\": \"^<t>\"")
                .field(ResponseTemplate.THINKING);
        assertEquals(Collections.singletonList("<t>"),
                TemplateGrammar.openers(thinking, Collections.singletonList("<s>")));
        assertNull("elsewhere the ^ never matches", TemplateGrammar.openers(thinking, null));
        assertNull("so no reasoning format is derived there", TemplateGrammar.thinkingFormat(thinking, null, false));
        assertTrue(TemplateGrammar.thinkingFormat(thinking, Collections.singletonList("<s>"), false) != null);
    }
}
