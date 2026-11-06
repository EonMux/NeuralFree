package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.chat.response.ResponseTemplateTest.PREFIX;
import static com.eonmux.neuralfree.gguf.chat.response.ResponseTemplateTest.THINKING;
import static com.eonmux.neuralfree.gguf.chat.response.ResponseTemplateTest.TOOL_GENERATION;
import static com.eonmux.neuralfree.gguf.chat.response.ResponseTemplateTest.gemma4;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@code hf} / {@code beam} reasoning filter streams a response template's regions as vLLM's
 * {@code _ResponseStream} does — thinking as reasoning, content as content, a tool region whole to the content — for
 * every chunking, from the state the prompt leaves; the reasoning end vLLM's structured outputs wait for; and the
 * tool guarantees {@code hf} refuses by name, which {@code beam} leaves to its grammar ({@link BeamGrammarTest}).
 */
public class ResponseRoutingTest {

    private static final TemplateParsing HF = TemplateParsing.of("hf", gemma4());
    private static final TemplateParsing BEAM = TemplateParsing.of("beam", gemma4());

    /** [reasoning, content] of {@code output} fed in {@code chunk}-character pieces after {@code prompt}. */
    private static List<String> stream(TemplateParsing parsing, String prompt, String output, int chunk) {
        StringBuilder reasoning = new StringBuilder();
        StringBuilder content = new StringBuilder();
        ReasoningFilter.State st = ReasoningFilter.responseTemplate(parsing).newState((channel, tokenId, text) ->
                (channel == ReasoningFilter.Channel.REASONING ? reasoning : content).append(text), prompt);
        for (int i = 0; i < output.length(); i += chunk) {
            st.onToken(i, output.substring(i, Math.min(output.length(), i + chunk)));
        }
        st.finish();
        return Arrays.asList(reasoning.toString(), content.toString());
    }

    private static void assertStreams(String prompt, String output, String reasoning, String content) {
        for (int chunk : new int[]{1, 7, 10_000}) {
            assertEquals("chunk " + chunk, Arrays.asList(reasoning, content), stream(HF, prompt, output, chunk));
        }
    }

    @Test
    public void regionsStreamAsReasoningAndContentForEveryChunking() {
        assertStreams(PREFIX, "Hello<turn|>", "", "Hello");
        assertStreams(PREFIX, THINKING + "The answer is 42.<turn|>", "plan", "The answer is 42.");
        assertStreams(PREFIX, "Let me check." + TOOL_GENERATION, "", "Let me check." + TOOL_GENERATION);
        assertStreams(PREFIX, THINKING + TOOL_GENERATION, "plan", TOOL_GENERATION);
    }

    @Test
    public void aPromptThatOpensTheThinkingRegionStartsInReasoning() {
        String prompt = PREFIX + "<|channel>thought\n";
        assertStreams(prompt, "continued reasoning<channel|>Final answer<turn|>", "continued reasoning", "Final answer");
        assertTrue(ReasoningFilter.responseTemplate(HF).startsInReasoning(prompt));
        assertFalse(ReasoningFilter.responseTemplate(HF).startsInReasoning(PREFIX));
    }

    @Test
    public void theReasoningEndsWhereVllmsGateSeesIt() {
        assertFalse("thinking on, no region open yet", HF.reasoningEnded(PREFIX));
        assertTrue("thinking off", HF.withThinking(false).reasoningEnded(PREFIX));
        assertFalse("the prompt opened the thinking region", HF.reasoningEnded(PREFIX + "<|channel>thought\n"));
        assertTrue("the prompt's content started", HF.reasoningEnded(PREFIX + "answer"));
        ResponseTemplate held = ResponseTemplate.parse(ResponseTemplateTest.gemma4Json()
                .replace("\"content\": {\"close\"", "\"content\": {\"open_pattern\": \"<\\\\|channel>final\\\\n\", \"close\""));
        assertTrue("a content opener the prompt's end holds back", TemplateParsing.of("hf", held)
                .reasoningEnded(PREFIX + "<|channel>final\n"));
        ResponseTemplate repeated = ResponseTemplate.parse(ResponseTemplateTest.gemma4Json()
                .replace("\"open\": \"<|channel>thought\\n\"", "\"open\": \"<|channel>thought\\n\", \"repeats\": true"));
        assertFalse("a repeating thinking region never ends", TemplateParsing.of("hf", repeated)
                .reasoningEnded(PREFIX + "answer"));
    }

    @Test
    public void aCloserCutOffBeforeTheNextMessagesAnchorIsCompleted() {
        String json = "{\"start_anchor\": \"<start>\", \"fields\": {\"thinking\": {\"open\": \"<t>\", \"close\": \"</t>\"},"
                + " \"tool_calls\": {\"open\": \"<call>\", \"close\": \"</call><start>\", \"content\": \"json\"},"
                + " \"content\": {}}}";
        ResponseParser p = new ResponseParser(ResponseTemplate.parse(json).forServing(), "<start>", ToolArguments.NONE,
                null);
        p.feed("<call>{\"function\": {\"name\": \"f\", \"arguments\": {}}}</call>", new ResponseParser.Listener() {
        });
        assertEquals("<start>", p.closerRest());
    }

    @Test
    public void theFilterMatchesTheTemplatesLiteralsAsSpecialTokens() {
        ReasoningFilter f = ReasoningFilter.responseTemplate(HF);
        assertEquals(Collections.singletonList("<|channel>thought\n"), f.startTags());
        assertEquals(Collections.singletonList("<channel|>"), f.endTags());
        Map<Integer, String> ids = f.withTagTokens(t -> t.equals("<turn|>") ? 7 : t.equals("<tool_call|>") ? 8 : -1)
                .tagTokens();
        assertEquals("<turn|>", ids.get(7));
        assertEquals("<tool_call|>", ids.get(8));
    }

    @Test
    public void hfRefusesTheToolGuaranteesItCannotEnforce() {
        ChatTool strict = ChatTool.builder("f").strict(true).build();
        ChatTool plain = ChatTool.builder("f").build();
        refuses(() -> HF.checkTools(chat(strict, ChatRequest.ToolChoice.auto()), ParallelToolCalls.PARALLEL),
                "strict tools", "tools");
        refuses(() -> HF.checkTools(chat(plain, ChatRequest.ToolChoice.required()), ParallelToolCalls.PARALLEL),
                "required tool choice", "tool_choice");
        refuses(() -> HF.checkTools(chat(plain, ChatRequest.ToolChoice.function("f")), ParallelToolCalls.PARALLEL),
                "named tool choice", "tool_choice");
        refuses(() -> HF.checkTools(chat(plain, ChatRequest.ToolChoice.auto()), ParallelToolCalls.SINGLE),
                "parallel_tool_calls=False", "parallel_tool_calls");
        HF.checkTools(chat(plain, ChatRequest.ToolChoice.auto()), ParallelToolCalls.PARALLEL);
        HF.checkTools(chat(plain, ChatRequest.ToolChoice.auto()), ParallelToolCalls.of(null, false));
        HF.checkTools(chat(plain, ChatRequest.ToolChoice.none()), ParallelToolCalls.SINGLE);
    }

    @Test
    public void beamLeavesTheToolGuaranteesToItsGrammar() {
        ChatTool strict = ChatTool.builder("f").strict(true).build();
        ChatTool plain = ChatTool.builder("f").build();
        BEAM.checkTools(chat(plain, ChatRequest.ToolChoice.required()), ParallelToolCalls.PARALLEL);
        BEAM.checkTools(chat(plain, ChatRequest.ToolChoice.function("f")), ParallelToolCalls.PARALLEL);
        BEAM.checkTools(chat(plain, ChatRequest.ToolChoice.none()), ParallelToolCalls.PARALLEL);
        BEAM.checkTools(chat(strict, ChatRequest.ToolChoice.auto()), ParallelToolCalls.SINGLE);
    }

    private static ChatRequest chat(ChatTool tool, ChatRequest.ToolChoice choice) {
        return ChatRequest.builder().tools(Collections.singletonList(tool)).toolChoice(choice).build();
    }

    private static void refuses(Runnable r, String guarantee, String parameter) {
        try {
            r.run();
            fail("expected a refusal of " + guarantee);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(guarantee));
            assertTrue(e.getMessage(), e.getMessage().contains("'" + parameter + "'"));
        }
    }
}
