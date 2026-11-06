package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link EscapedNewlines}, against vLLM's tests/parser/engine/test_minicpmv_reasoning.py {@code TestNewlineRecovery}:
 * escapes outside code, LaTeX and doubled backslashes become newlines, every split of a stream gives the whole-text
 * result, the reasoning and the content hold back separately, and configured verbatim spans keep their text.
 */
public class EscapedNewlinesTest {

    private static final List<String> NO_SPANS = Collections.emptyList();

    @Test
    public void recoversNewlinesOutsideCodeAndLatex() {
        String text = "first\\nsecond\\rthird\\r\\nfourth `code\\nvalue` ```python\\nvalue = '\\\\n'\\n``` "
                + "$x\\ny$ $$x\\ny$$ \\(x\\ny\\) \\[x\\ny\\] escaped \\\\n";
        assertEquals("first\nsecond\nthird\nfourth `code\\nvalue` ```python\\nvalue = '\\\\n'\\n``` "
                + "$x\\ny$ $$x\\ny$$ \\(x\\ny\\) \\[x\\ny\\] escaped \\\\n", whole(text, NO_SPANS));
    }

    @Test
    public void everySplitMatchesTheWholeText() {
        String text = "first\\nsecond `code\\nvalue` $x\\ny$ ```text\\nvalue\\n``` last\\r\\nline";
        String expected = whole(text, NO_SPANS);
        for (int split = 0; split <= text.length(); split++) {
            EscapedNewlines n = new EscapedNewlines(NO_SPANS);
            String actual = n.feed(text.substring(0, split), false) + n.feed(text.substring(split), true);
            assertEquals("split " + split, expected, actual);
        }
        EscapedNewlines n = new EscapedNewlines(NO_SPANS);
        StringBuilder chars = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            chars.append(n.feed(text.substring(i, i + 1), false));
        }
        chars.append(n.feed("", true));
        assertEquals(expected, chars.toString());
    }

    @Test
    public void backslashRunsStayConsistentAcrossSplits() {
        for (int count = 1; count <= 6; count++) {
            StringBuilder b = new StringBuilder();
            for (int i = 0; i < count; i++) b.append('\\');
            String text = b.append('n').toString();
            String expected = whole(text, NO_SPANS);
            assertEquals(count == 1 ? "\n" : text, expected);
            for (int split = 0; split <= text.length(); split++) {
                EscapedNewlines n = new EscapedNewlines(NO_SPANS);
                assertEquals(count + "/" + split, expected,
                        n.feed(text.substring(0, split), false) + n.feed(text.substring(split), true));
            }
        }
    }

    @Test
    public void verbatimSpansKeepTheirEscapes() {
        List<String> spans = Arrays.asList("<tool_call>", "</tool_call>");
        String text = "a\\nb<tool_call>{\"s\":\"x\\ny\"}</tool_call>c\\nd";
        String expected = "a\nb<tool_call>{\"s\":\"x\\ny\"}</tool_call>c\nd";
        for (int split = 0; split <= text.length(); split++) {
            EscapedNewlines n = new EscapedNewlines(spans);
            assertEquals("split " + split, expected,
                    n.feed(text.substring(0, split), false) + n.feed(text.substring(split), true));
        }
        assertEquals("a held-back partial opener is released at the end", "x<tool_",
                new EscapedNewlines(spans).feed("x<tool_", true));
    }

    @Test
    public void reasoningAndContentHoldBackSeparately() {
        List<String> routed = new ArrayList<>();
        EscapedNewlines.RecoveringSink sink = new EscapedNewlines.RecoveringSink(
                (channel, tokenId, text) -> routed.add(channel + ":" + text), NO_SPANS);
        sink.accept(ReasoningFilter.Channel.REASONING, 0, "reason\\");
        assertEquals(Collections.singletonList("REASONING:reason"), routed);
        sink.accept(ReasoningFilter.Channel.REASONING, 1, "nnext\\");
        sink.accept(ReasoningFilter.Channel.CONTENT, 1, "answer\\");
        sink.accept(ReasoningFilter.Channel.CONTENT, 2, "nend");
        sink.finish();
        assertEquals(Arrays.asList("REASONING:reason", "REASONING:\nnext", "REASONING:\\", "CONTENT:answer",
                "CONTENT:\nend"), routed);
    }

    @Test
    public void filterOptionsRefuseFiltersTheyDoNotApplyTo() {
        assertThrows(IllegalStateException.class, () -> ReasoningFilter.none().opensImplicitly());
        assertThrows(IllegalStateException.class, () -> ReasoningFilter.harmony().reopening());
        assertThrows(IllegalStateException.class, () -> ReasoningFilter.inkling().discardingReasoning());
        assertThrows(IllegalStateException.class, () -> ReasoningFilter.thinkTags().withContentPrefix("<think>"));
        assertThrows(IllegalArgumentException.class, () -> ReasoningFilter.none().withContentPrefix(""));
        assertThrows(IllegalArgumentException.class, () -> ReasoningFilter.thinkTags().dropping(
                Collections.<String>emptyList()));
        assertThrows(IllegalArgumentException.class, () -> ReasoningFilter.thinkTags().recoveringEscapedNewlines(
                Collections.singletonList("<a>")));
        assertThrows(NullPointerException.class, () -> ReasoningFilter.perRequest(vars -> null));
    }

    @Test
    public void droppedTagsRenderByIdAndLeaveTheOutput() {
        ReasoningFilter f = ReasoningFilter.thinkTags().dropping(Collections.singletonList("<|im_end|>"))
                .withTagTokens(t -> "<|im_end|>".equals(t) ? 7 : -1);
        assertEquals("<|im_end|>", f.tagTokens().get(7));
        StringBuilder content = new StringBuilder();
        ReasoningFilter.State st = f.newState((channel, tokenId, text) -> content.append(text));
        st.onToken(1, "done");
        st.onToken(7, "");
        st.finish();
        assertEquals("done", content.toString());
    }

    private static String whole(String text, List<String> spans) {
        return new EscapedNewlines(spans).feed(text, true);
    }
}
