package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.OutputGates;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.grammar.ConstraintGate;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * vLLM's {@code hf} and {@code beam} parsers selected by name on a loaded model: the model's
 * {@code tokenizer.response_template} metadata builds the reasoning filter and the tool parser, the reasoning filter
 * hands a tool region whole to the tool parser, which reads its calls typed by the tools' schemas; a model without
 * the metadata, a template without the field a parser needs, mixed parser families and the tool guarantees {@code hf}
 * cannot enforce are refused by name; an output constraint waits for the template's thinking closer; an {@code auto}
 * reasoning parser beside a template tool parser parses no reasoning; and each model's template is parsed once.
 */
public class ResponseTemplateToolsTest {

    private static final String PREFIX = "<|turn>model\n";
    private static final String TOOL_GENERATION =
            "<|tool_call>call:set_alarm{hour:7,label:<|\"|>morning<|\"|>}<tool_call|>";

    @Test
    public void theReasoningFilterHandsTheToolRegionToTheTemplatesToolParser() {
        GGUFTokenizer tok = tokenizer(gemma4());
        ReasoningFilter filter = ReasoningParsers.forModel("hf", tok);
        StringBuilder reasoning = new StringBuilder();
        StringBuilder content = new StringBuilder();
        ReasoningFilter.State st = filter.forRequest(Collections.<String, Object>emptyMap()).newState(
                (channel, id, text) -> (channel == ReasoningFilter.Channel.REASONING ? reasoning : content).append(text),
                PREFIX);
        String output = "<|channel>thought\nplan<channel|>Let me check." + TOOL_GENERATION;
        for (int i = 0; i < output.length(); i++) {
            st.onToken(i, output.substring(i, i + 1));
        }
        st.finish();
        assertEquals("plan", reasoning.toString());

        ToolRouting routing = ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "hf", tok, filter, PREFIX,
                ParallelToolCalls.PARALLEL, ToolStrict.AUTO);
        ToolParse parse = routing.parser().parse(content.toString(), false);
        assertEquals("Let me check.", parse.content());
        assertEquals(1, parse.calls().size());
        assertEquals("set_alarm", parse.calls().get(0).name());
        assertEquals("{\"hour\": 7, \"label\": \"morning\"}", parse.calls().get(0).arguments());
        assertNull(routing.constraint());

        ToolParse partial = routing.parser().parse("Let me check.<|tool_call>call:set_alarm{hour:7", true);
        assertEquals("an open region waits for its closer", "Let me check.", partial.content());
        assertTrue(partial.calls().isEmpty());
        assertEquals(Collections.<String>emptyList(), routing.parser().triggers());
    }

    @Test
    public void aModelWithoutTheMetadataOrTheNeededFieldIsRefusedByName() {
        GGUFTokenizer bare = tokenizer(null);
        refuses(() -> ReasoningParsers.forModel("hf", bare), "--reasoning-parser hf requires `response_template`");
        refuses(() -> ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "beam", bare, ReasoningFilter.none(), null,
                ParallelToolCalls.PARALLEL, ToolStrict.AUTO), "--tool-call-parser beam requires `response_template`");
        refuses(() -> ToolParserRegistry.forName("hf").create(Collections.<ChatTool>emptyList()),
                "it needs the model's tokenizer");
        String noTools = "{\"defaults\": {\"role\": \"assistant\"}, \"start_anchor\": \"<|turn>model\\n\","
                + " \"fields\": {\"content\": {\"close\": [\"<turn|>\"]},"
                + " \"thinking\": {\"open\": \"<|channel>thought\\n\", \"close\": \"<channel|>\"}}}";
        GGUFTokenizer reasoningOnly = tokenizer(noTools);
        assertNotNull(ReasoningParsers.forModel("hf", reasoningOnly));
        refuses(() -> ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "hf", reasoningOnly, ReasoningFilter.none(),
                null, ParallelToolCalls.PARALLEL, ToolStrict.AUTO), "requires a response_template with a tool_calls field");
        refuses(() -> ReasoningParsers.forModel("hf", tokenizer("{\"fields\": {}}")), "Invalid response_template");
    }

    @Test
    public void theLoadCheckRefusesANamedToolParserTheModelCannotServe() {
        // vLLM validate_tokenizer_response_template, run when the server loads the model
        GGUFTokenizer bare = tokenizer(null);
        ToolRouting.checkFormat("hf", tokenizer(gemma4()));
        ToolRouting.checkFormat("beam", tokenizer(gemma4()));
        ToolRouting.checkFormat("auto", bare);
        ToolRouting.checkFormat("none", bare);
        ToolRouting.checkFormat("hermes", bare);
        refuses(() -> ToolRouting.checkFormat("HF", bare), "--tool-call-parser hf requires `response_template`");
        String noTools = "{\"start_anchor\": \"<|turn>model\\n\", \"fields\": {\"content\": {\"close\": [\"<turn|>\"]},"
                + " \"thinking\": {\"open\": \"<|channel>thought\\n\", \"close\": \"<channel|>\"}}}";
        refuses(() -> ToolRouting.checkFormat("beam", tokenizer(noTools)),
                "requires a response_template with a tool_calls field");
        refuses(() -> ToolRouting.checkFormat("hf", tokenizer("{\"fields\": {}}")), "Invalid response_template");
        refuses(() -> ToolRouting.checkFormat("no_such_parser", bare), "is not registered");
    }

    @Test
    public void theToolGuaranteesAndMixedFamiliesAreRefusedAtRouting() {
        GGUFTokenizer tok = tokenizer(gemma4());
        ReasoningFilter hf = ReasoningParsers.forModel("hf", tok);
        refuses(() -> ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "hf", tok, hf, PREFIX,
                ParallelToolCalls.PARALLEL, ToolStrict.AUTO), "required tool choice");
        assertSame("beam's grammar forbids the calls", ToolRouting.none(), ToolRouting.plan(
                chat(ChatRequest.ToolChoice.none()), "beam", tok, ReasoningFilter.none(), PREFIX,
                ParallelToolCalls.PARALLEL, ToolStrict.AUTO));
        assertSame(ToolRouting.none(), ToolRouting.plan(chat(ChatRequest.ToolChoice.none()), "hf", tok, hf, PREFIX,
                ParallelToolCalls.PARALLEL, ToolStrict.AUTO));
        refuses(() -> ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "beam", tok, hf, PREFIX,
                ParallelToolCalls.PARALLEL, ToolStrict.AUTO),
                "--reasoning-parser hf and --tool-call-parser beam cannot be combined");
        refuses(() -> EngineConfig.builder().reasoningParser("hf").toolParser("hermes").build(), "cannot be combined");
        assertEquals("hf", EngineConfig.builder().reasoningParser("hf").toolParser("hf").build().toolParser());
        assertEquals("beam", EngineConfig.builder().reasoningParser("auto").toolParser("beam").build().toolParser());
    }

    @Test
    public void anOutputConstraintWaitsForTheThinkingCloserUnlessThinkingIsOff() {
        ReasoningFilter hf = ReasoningParsers.forModel("hf", tokenizer(gemma4()));
        ConstraintGate gate = OutputGates.forChat(hf.forRequest(Collections.<String, Object>emptyMap()), PREFIX, null);
        assertNotNull(gate);
        assertNull(OutputGates.forChat(hf.forRequest(Collections.<String, Object>singletonMap("enable_thinking", false)),
                PREFIX, null));
        assertNull("the prompt's content already started", OutputGates.forChat(
                hf.forRequest(Collections.<String, Object>emptyMap()), PREFIX + "answer", null));
        assertFalse(gate.toString().isEmpty());
    }

    @Test
    public void anAutoReasoningParserBesideATemplateToolParserParsesNoReasoning() {
        GGUFTokenizer tok = tokenizer(gemma4());
        ReasoningFilter filter = ReasoningParsers.forModel("auto", "hf", tok);
        assertNotNull(filter.responseTemplate());
        assertFalse(filter.parsesReasoning());
        assertTrue(ReasoningParsers.forModel("hf", "hf", tok).parsesReasoning());
        assertNull("another tool parser keeps the chat template's reasoning",
                ReasoningParsers.forModel("auto", "hermes", tok).responseTemplate());
        StringBuilder reasoning = new StringBuilder();
        StringBuilder content = new StringBuilder();
        ReasoningFilter.State st = filter.newState((channel, id, text) ->
                (channel == ReasoningFilter.Channel.REASONING ? reasoning : content).append(text), PREFIX);
        String output = "<|channel>thought\nplan<channel|>Let me check." + TOOL_GENERATION;
        for (int i = 0; i < output.length(); i += 5) {
            st.onToken(i, output.substring(i, Math.min(output.length(), i + 5)));
        }
        st.finish();
        assertEquals("", reasoning.toString());
        assertEquals("the thinking text is content, without its delimiters", "planLet me check." + TOOL_GENERATION,
                content.toString());
        assertFalse(filter.startsInReasoning(PREFIX + "<|channel>thought\n"));
        assertNull("no output constraint waits for a reasoning nobody parses",
                OutputGates.forChat(filter, PREFIX, null));
        ToolRouting routing = ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "hf", tok, filter, PREFIX,
                ParallelToolCalls.PARALLEL, ToolStrict.AUTO);
        assertEquals("set_alarm", routing.parser().parse(content.toString(), false).calls().get(0).name());
    }

    @Test
    public void eachModelsTemplateIsParsedOnce() {
        GGUFTokenizer tok = tokenizer(gemma4());
        assertSame(parsedTemplate(ReasoningParsers.forModel("hf", tok)),
                parsedTemplate(ReasoningParsers.forModel("beam", "beam", tok)));
        assertSame("a request's routing reuses the parse", parsedTemplate(
                ReasoningParsers.forModel("auto", "hf", tok)), parsedTemplate(
                ReasoningParsers.forModel("hf", tok)));
        GGUFTokenizer broken = tokenizer("{\"fields\": {}}");
        refuses(() -> ReasoningParsers.forModel("hf", broken), "Invalid response_template");
        refuses(() -> ReasoningParsers.forModel("auto", "beam", broken), "Invalid response_template");
    }

    /** The parsed template behind a response-template filter. */
    private static Object parsedTemplate(ReasoningFilter filter) {
        return filter.forRequest(Collections.<String, Object>emptyMap()).responseTemplate().template();
    }

    private static ChatRequest chat(ChatRequest.ToolChoice choice) {
        Map<String, Object> hour = new LinkedHashMap<>();
        hour.put("oneOf", java.util.Arrays.asList(Collections.singletonMap("type", "integer"),
                Collections.singletonMap("type", "null")));
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("hour", hour);
        props.put("label", Collections.singletonMap("type", "string"));
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("type", "object");
        params.put("properties", props);
        return ChatRequest.builder()
                .tools(Collections.singletonList(ChatTool.builder("set_alarm").parameters(params).build()))
                .toolChoice(choice).build();
    }

    private static void refuses(Runnable r, String message) {
        try {
            r.run();
            fail("expected a refusal naming " + message);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(message));
        }
    }

    private static String gemma4() {
        try (InputStream in = ResponseTemplateToolsTest.class.getResourceAsStream("/response-templates/gemma4.json")) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    /** A SentencePiece byte vocabulary carrying {@code responseTemplate} (none when null). */
    private static GGUFTokenizer tokenizer(String responseTemplate) {
        List<Object> tokens = new ArrayList<>();
        List<Object> scores = new ArrayList<>();
        List<Object> types = new ArrayList<>();
        add(tokens, scores, types, "<unk>", 2);
        add(tokens, scores, types, "<s>", 3);
        add(tokens, scores, types, "</s>", 3);
        for (int b = 0; b < 256; b++) {
            add(tokens, scores, types, String.format("<0x%02X>", b), 6);
        }
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", "llama");
        md.put("tokenizer.ggml.tokens", tokens);
        md.put("tokenizer.ggml.scores", scores);
        md.put("tokenizer.ggml.token_type", types);
        md.put("tokenizer.ggml.bos_token_id", 1);
        md.put("tokenizer.ggml.eos_token_id", 2);
        md.put("tokenizer.ggml.unknown_token_id", 0);
        if (responseTemplate != null) {
            md.put(GgufKeys.TOKENIZER_RESPONSE_TEMPLATE, responseTemplate);
        }
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static void add(List<Object> tokens, List<Object> scores, List<Object> types, String t, int type) {
        tokens.add(t);
        scores.add(-1.0f);
        types.add(type);
    }
}
