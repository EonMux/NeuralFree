package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.Test;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The per-request {@code reasoning_format} of llama.cpp: {@code auto} and {@code deepseek} keep the
 * configured filter, {@code none} routes everything to content with the tags kept (on a harmony model it
 * keeps parsing messages and only inlines the reasoning), {@code deepseek-legacy} only on a non-stream, and
 * unknown names are a 400 naming the field.
 */
public class ReasoningFormatsTest {

    private static final ReasoningFilter THINK = ReasoningFilter.thinkTags();
    private static final ReasoningFilter HARMONY = ReasoningFilter.harmony();

    @Test
    void autoAndDeepseekKeepTheConfiguredFilter() {
        assertSame(THINK, ReasoningFormats.resolve("auto", THINK, false));
        assertSame(THINK, ReasoningFormats.resolve("deepseek", THINK, true));
        assertSame(THINK, ReasoningFormats.resolve(null, THINK, false));
        assertSame(THINK, ReasoningFormats.resolve("  ", THINK, false));
    }

    @Test
    void noneRoutesEverythingToContentKeepingTheTags() {
        ReasoningFilter raw = ReasoningFormats.resolve("none", THINK, false);
        assertEquals(ReasoningFilter.Mode.NONE, raw.mode());
        assertNotSame(ReasoningFilter.none(), raw);
        // the raw variant keeps the think tags as content: the start tag's text reaches the sink
        StringBuilder text = new StringBuilder();
        ReasoningFilter.State state = raw.newState((channel, tokenId, t) -> text.append(t));
        state.onToken(-2, "<think>");
        state.onToken(-3, "hmm");
        state.onToken(-2, "</think>");
        state.finish();
        assertEquals("<think>hmm</think>", text.toString(), "the raw text carries the tags");
    }

    @Test
    void noneOnAHarmonyModelKeepsParsingAndInlinesTheReasoning() {
        ReasoningFilter raw = ReasoningFormats.resolve("none", HARMONY, false);
        assertEquals(ReasoningFilter.Mode.HARMONY_CHANNELS, raw.mode(), "tool calls are still parsed");
        assertTrue(raw.reasoningInline());
        assertFalse(HARMONY.reasoningInline(), "the configured filter is untouched");
        assertTrue(ReasoningFormats.resolve("deepseek-legacy", HARMONY, true).reasoningInline());
        assertSame(HARMONY, ReasoningFormats.resolve("auto", HARMONY, true));
    }

    @Test
    void deepseekLegacyStaysInlineOnlyOnAStream() {
        assertEquals(ReasoningFilter.Mode.NONE, ReasoningFormats.resolve("deepseek-legacy", THINK, true).mode());
        assertSame(THINK, ReasoningFormats.resolve("deepseek-legacy", THINK, false));
    }

    @Test
    void anUnknownNameIsRefusedByName() {
        try {
            ReasoningFormats.resolve("sql", THINK, false);
        } catch (WebApplicationException e) {
            String message = view(e.getResponse().getEntity()).obj("error").str("message");
            assertTrue(message.contains("reasoning_format"), message);
            assertTrue(message.contains("sql"), message);
            assertTrue(message.contains("deepseek-legacy"), message);
            return;
        }
        throw new AssertionError("an unknown reasoning_format was accepted");
    }

    @Test
    void theBodyCarriesTheField() {
        assertSame(THINK, ReasoningFormats.fromBody(body("{}"), THINK, false));
        assertEquals(ReasoningFilter.Mode.NONE, ReasoningFormats.fromBody(
                body("{\"reasoning_format\": \"none\"}"), THINK, false).mode());
        assertSame(THINK, ReasoningFormats.fromBody(
                body("{\"reasoning_format\": \"deepseek\"}"), THINK, true));
    }

    private static JsonView body(String json) {
        return JsonView.of(Json.parse(json, Json.Limits.UNTRUSTED), "body");
    }
}
