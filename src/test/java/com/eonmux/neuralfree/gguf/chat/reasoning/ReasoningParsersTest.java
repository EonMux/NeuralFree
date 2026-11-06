package com.eonmux.neuralfree.gguf.chat.reasoning;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.plugin.ReasoningParserProvider;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The reasoning parser registry: vLLM's parser names select their family's delimiters, {@code auto} defers to the
 * chat template, plugin rows cannot shadow built-ins, and {@link EngineConfig} refuses names nothing registers.
 */
public class ReasoningParsersTest {

    @Test
    public void namesSelectTheirFamilysDelimiters() {
        assertEquals(Collections.singletonList("<think>"), filter("deepseek_r1").startTags());
        assertEquals(Collections.singletonList("<seed:think>"), filter("seed_oss").startTags());
        assertEquals(Collections.singletonList("[THINK]"), filter("mistral").startTags());
        assertEquals(Collections.singletonList("<mm:think>"), filter("minimax_m3").startTags());
        assertEquals(Collections.singletonList("<|START_THINKING|>"), filter("cohere_command4").startTags());
        assertEquals(Collections.singletonList("<tool_call>"), filter("qwen3").interruptTags());
        assertEquals(Arrays.asList(ReasoningFilter.DSML_FUNCTION_CALLS, ReasoningFilter.DSML_TOOL_CALLS),
                filter("deepseek_v4").interruptTags());
        assertEquals(ReasoningFilter.Mode.HARMONY_CHANNELS, filter("openai_gptoss").mode());
        assertEquals(ReasoningFilter.Mode.HARMONY_CHANNELS, filter("muse_glimmer").mode());
        assertSame(ReasoningFilter.none(), filter("identity"));
    }

    @Test
    public void graniteSplitsAtItsResponseMarker() {
        String out = "Here is my thought process: weigh it.Here is my response: Yes.";
        StringBuilder reasoning = new StringBuilder();
        StringBuilder content = new StringBuilder();
        ReasoningFilter.State st = filter("granite").newState((channel, tokenId, text) ->
                (channel == ReasoningFilter.Channel.REASONING ? reasoning : content).append(text));
        for (int i = 0; i < out.length(); i++) {
            st.onToken(i, out.substring(i, i + 1));
        }
        st.finish();
        assertEquals(" weigh it.", reasoning.toString());
        assertEquals("content after the end marker starts at its first non-space", "Yes.", content.toString());
    }

    @Test
    public void spellingsAndAliasesResolveAndAutoDefers() {
        assertSame(ReasoningParsers.forName("glm45"), ReasoningParsers.forName("GLM47"));
        assertSame(ReasoningParsers.forName("seed_oss"), ReasoningParsers.forName("Seed-OSS"));
        assertTrue(ReasoningParsers.isAuto(null));
        assertTrue(ReasoningParsers.isAuto(" Auto "));
        assertFalse(ReasoningParsers.isAuto("qwen3"));
        assertNull(ReasoningParsers.forName("no_such_parser"));
        assertTrue(ReasoningParsers.names().contains("deepseek_r1"));
    }

    @Test
    public void pluginRowsCannotShadowBuiltInsOrAuto() {
        ReasoningParsers.Factory builtIn = ReasoningParsers.forName("qwen3");
        IllegalStateException clash = assertThrows(IllegalStateException.class, () -> ReasoningParsers.build(
                Collections.singletonList(builtIn), Collections.singletonList(provider(row("qwen3")))));
        assertTrue(clash.getMessage(), clash.getMessage().contains("built-in"));
        IllegalStateException auto = assertThrows(IllegalStateException.class, () -> ReasoningParsers.build(
                Collections.singletonList(builtIn), Collections.singletonList(provider(row("auto")))));
        assertTrue(auto.getMessage(), auto.getMessage().contains("auto"));
        assertEquals("qwen3, its two aliases and the plugin row with its alias", 5, ReasoningParsers.build(Collections.singletonList(builtIn),
                Collections.singletonList(provider(row("mine")))).size());
    }

    @Test
    public void engineConfigRefusesUnregisteredParserNames() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_REASONING_PARSER, "no_such_parser");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_REASONING_PARSER));
        Properties q = new Properties();
        q.setProperty(EngineConfig.PROP_TOOL_PARSER, "no_such_parser");
        assertThrows(IllegalArgumentException.class, () -> EngineConfig.fromProperties(q));
        Properties ok = new Properties();
        ok.setProperty(EngineConfig.PROP_REASONING_PARSER, "seed_oss");
        ok.setProperty(EngineConfig.PROP_TOOL_PARSER, "none");
        EngineConfig c = EngineConfig.fromProperties(ok);
        assertEquals("seed_oss", c.reasoningParser());
        assertEquals("none", c.toBuilder().build().toolParser());
        assertEquals(EngineConfig.PARSER_AUTO, EngineConfig.defaults().reasoningParser());
    }

    private static ReasoningFilter filter(String name) {
        return ReasoningParsers.forName(name).create();
    }

    private static ReasoningParsers.Factory row(String name) {
        return new ReasoningParsers.Factory() {
            @Override public String name() { return name; }
            @Override public List<String> aliases() { return Collections.singletonList(name + "_alias"); }
            @Override public ReasoningFilter create() { return ReasoningFilter.thinkTags(); }
        };
    }

    private static ReasoningParserProvider<ReasoningParsers.Factory> provider(ReasoningParsers.Factory f) {
        return () -> Collections.singletonList(f);
    }
}
