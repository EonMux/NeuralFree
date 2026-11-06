package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Responses API end to end over the synthetic llama: a hermes-template model with the response store on (the
 * message, function-call, stream, store and background paths), and a thinking-template model with the store off
 * (reasoning items, and the refusals a storeless server gives).
 */
class ResponsesResourceTest {

    private static final String TEMPLATE = "{%- if tools %}<|im_start|>system\nTools:"
            + "{% for t in tools %} {{ t.function.name }}{% endfor %}. Call one inside <tool_call></tool_call>."
            + "<|im_end|>\n{% endif %}"
            + "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            + "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    private static final String THINKING = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
            + "<|im_end|>\n{% endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}";

    private static EngineService service;
    private static EngineService thinking;
    private static ResponsesResource responses;
    private static ResponsesResource thinker;
    private static OpenAiResource openAi;

    @BeforeAll
    static void open() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Path model = SyntheticModels.llama(config).str("tokenizer.chat_template", TEMPLATE).writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(1024, 2),
                ServerOptions.defaults().withResponseStore(8));
        responses = new ResponsesResource(service);
        openAi = new OpenAiResource(service);
        Path think = SyntheticModels.llama().str("tokenizer.chat_template", THINKING).writeTempFile();
        thinking = new EngineService(think.toString(), ServerTestSupport.pagedConfig(512, 2));
        thinker = new ResponsesResource(thinking);
    }

    @AfterAll
    static void close() {
        service.close();
        thinking.close();
    }

    private static JsonView create(String json) {
        return view(responses.create(json).getEntity());
    }

    private static String outputText(JsonView response) {
        StringBuilder b = new StringBuilder();
        for (Object o : (List<?>) response.raw("output")) {
            JsonView item = JsonView.of(o, "item");
            if ("message".equals(item.str("type"))) {
                b.append(at(item, "content", 0).str("text"));
            }
        }
        return b.toString();
    }

    @Test
    void aStringInputAnswersTheResponseObjectTheChatRouteWouldWrite() {
        JsonView r = create("{\"input\": \"hi\", \"max_output_tokens\": 4, \"temperature\": 0}");
        assertTrue(r.str("id").startsWith("resp_"));
        assertEquals("response", r.str("object"));
        assertEquals("incomplete", r.str("status"));
        assertEquals("max_output_tokens", r.obj("incomplete_details").str("reason"));
        JsonView item = at(r, "output", 0);
        assertEquals("message", item.str("type"));
        assertEquals("output_text", at(item, "content", 0).str("type"));
        JsonView chat = at(view(openAi.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0}""")
                .getEntity()), "choices", 0).obj("message");
        assertEquals(chat.str("content"), outputText(r));
        JsonView usage = r.obj("usage");
        assertEquals(4L, usage.i64("output_tokens"));
        long counted = view(responses.inputTokens("{\"input\": \"hi\"}").getEntity()).i64("input_tokens");
        assertEquals(counted, usage.i64("input_tokens"));
        assertEquals(counted + 4, usage.i64("total_tokens"));
    }

    @Test
    void aStructuralTagTextFormatHoldsTheOutputToItsFormat() {
        JsonView r = create("""
                {"input": "hi", "max_output_tokens": 32, "temperature": 0, "store": false,
                 "text": {"format": {"type": "structural_tag", "format": {"type": "tag", "begin": "<a>",
                   "content": {"type": "json_schema", "json_schema": {"enum": [1, 2]}}, "end": "</a>"}}}}""");
        assertTrue(outputText(r).matches("<a>[12]</a>"), "the format holds the output: " + outputText(r));
    }

    @Test
    void aNamedFunctionBecomesAFunctionCallItem() {
        JsonView r = create("""
                {"input": "Weather in Paris?", "max_output_tokens": 600, "temperature": 0,
                 "tools": [{"type": "function", "name": "get_weather",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                                  "required": ["city"], "additionalProperties": false}}],
                 "tool_choice": {"type": "function", "name": "get_weather"}}""");
        JsonView call = null;
        for (Object o : (List<?>) r.raw("output")) {
            JsonView item = JsonView.of(o, "item");
            if ("function_call".equals(item.str("type"))) {
                call = item;
            }
        }
        assertTrue(call != null, "a function_call item: " + r.raw("output"));
        assertEquals("get_weather", call.str("name"));
        assertFalse(call.str("call_id").isEmpty());
        JsonView args = JsonView.of(Json.parse(call.str("arguments"), Json.Limits.UNTRUSTED), "args");
        assertEquals("Paris", args.str("city"));
    }

    @Test
    void theStreamCarriesTheNumberedEventSequenceAndEndsWithTheResponse() throws IOException {
        String text = streamText(responses.create("""
                {"input": "hi", "max_output_tokens": 5, "temperature": 0, "stream": true}"""));
        List<JsonView> events = dataFrames(text);
        assertEquals("response.created", events.get(0).str("type"));
        assertEquals("response.in_progress", events.get(1).str("type"));
        assertTrue(text.contains("event: response.output_item.added\n"));
        StringBuilder deltas = new StringBuilder();
        List<String> types = new ArrayList<>();
        for (int i = 0; i < events.size(); i++) {
            JsonView e = events.get(i);
            assertEquals(i, e.i64("sequence_number"));
            types.add(e.str("type"));
            if ("response.output_text.delta".equals(e.str("type"))) {
                deltas.append(e.str("delta"));
            }
        }
        JsonView last = events.get(events.size() - 1);
        assertEquals("response.incomplete", last.str("type"));
        JsonView response = last.obj("response");
        assertEquals(deltas.toString(), outputText(response));
        assertTrue(types.indexOf("response.output_text.done") < types.indexOf("response.output_item.done"));
        assertEquals(5L, response.obj("usage").i64("output_tokens"));
    }

    @Test
    void storedResponsesRetrieveListContinueAndDelete() {
        JsonView first = create("{\"input\": \"hi\", \"max_output_tokens\": 3, \"temperature\": 0}");
        String id = first.str("id");
        assertTrue(first.bool("store"));
        assertEquals(first.raw("output"), view(responses.retrieve(id, false, null).getEntity()).raw("output"));
        JsonView items = view(responses.inputItems(id, null, null, null).getEntity());
        assertEquals("list", items.str("object"));
        assertTrue(at(items, "data", 0).str("id").startsWith("msg_"));

        JsonView next = create("{\"input\": \"more\", \"previous_response_id\": \"" + id
                + "\", \"max_output_tokens\": 2, \"temperature\": 0}");
        assertEquals(id, next.str("previous_response_id"));
        String said = ServerJson.write(outputText(first));
        long explicit = view(responses.inputTokens("""
                {"input": [{"role": "user", "content": "hi"},
                           {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": %s}]},
                           {"role": "user", "content": "more"}]}""".formatted(said)).getEntity()).i64("input_tokens");
        assertEquals(explicit, next.obj("usage").i64("input_tokens"), "the continuation renders the whole turn");

        assertTrue(view(responses.delete(id).getEntity()).bool("deleted"));
        assertEquals(404, statusOf(() -> responses.retrieve(id, false, null)));
        assertEquals(404, statusOf(() -> responses.create("{\"input\": \"x\", \"previous_response_id\": \""
                + id + "\"}")));
        assertEquals(400, statusOf(() -> responses.cancel(next.str("id"))), "a synchronous response");
    }

    @Test
    void aBackgroundResponseRunsDetachedReplaysItsEventsAndCancels() throws Exception {
        JsonView queued = create("""
                {"input": "hi", "background": true, "max_output_tokens": 900, "temperature": 0,
                 "extra_params": {"ignore_eos": true}}""");
        assertEquals("queued", queued.str("status"));
        JsonView cancelled = view(responses.cancel(queued.str("id")).getEntity());
        assertEquals("cancelled", cancelled.str("status"));
        assertEquals(400, statusOf(() -> responses.cancel(queued.str("id"))), "already cancelled");

        JsonView done = create("""
                {"input": "hi", "background": true, "max_output_tokens": 3, "temperature": 0}""");
        String all = streamText(responses.retrieve(done.str("id"), true, null));
        List<JsonView> events = dataFrames(all);
        assertEquals("response.incomplete", events.get(events.size() - 1).str("type"));
        List<JsonView> later = dataFrames(streamText(responses.retrieve(done.str("id"), true, 1)));
        assertEquals(2L, later.get(0).i64("sequence_number"));
        assertEquals(events.size() - 2, later.size());
        assertEquals("incomplete", view(responses.retrieve(done.str("id"), false, null).getEntity()).str("status"));
    }

    @Test
    void reasoningBecomesAReasoningItemCountedAsReasoningTokens() {
        JsonView r = view(thinker.create("{\"input\": \"hi\", \"max_output_tokens\": 6, \"temperature\": 0}")
                .getEntity());
        JsonView item = at(r, "output", 0);
        assertEquals("reasoning", item.str("type"));
        assertEquals("reasoning_text", at(item, "content", 0).str("type"));
        assertFalse(r.bool("store"), "a storeless server stores nothing");
        JsonView usage = r.obj("usage");
        assertEquals(usage.i64("output_tokens"), usage.obj("output_tokens_details").i64("reasoning_tokens"));
    }

    @Test
    void refusalsNameTheProblem() {
        assertEquals(400, statusOf(() -> responses.create("{\"max_output_tokens\": 2}")));
        assertEquals(404, statusOf(() -> responses.create("{\"input\": \"x\", \"previous_response_id\": \"resp_no\"}")));
        assertEquals(404, statusOf(() -> responses.retrieve("resp_no", false, null)));
        assertTrue(errorOf(() -> responses.create("{\"input\": \"x\", \"background\": true, \"store\": false}"))
                .contains("response store"));
        assertEquals(400, statusOf(() -> responses.create("{\"input\": \"x\", \"conversation\": \"c\"}")));
        assertTrue(errorOf(() -> thinker.create("{\"input\": \"x\", \"previous_response_id\": \"resp_a\"}"))
                .contains(ServerOptions.RESPONSE_STORE_PROPERTY));
        JsonView skipped = create("""
                {"input": "x", "max_output_tokens": 1, "tools": [{"type": "web_search_preview"}]}""");
        assertEquals("response", skipped.str("object"), "a built-in tool is skipped, not refused");
        assertEquals(400, statusOf(() -> responses.inputItems(skipped.str("id"), 0, null, null)));
    }

    @Test
    void chatInputTokensCountTheRenderedPrompt() {
        JsonView counted = view(openAi.chatInputTokens(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}").getEntity());
        assertEquals("response.input_tokens", counted.str("object"));
        assertEquals(view(responses.inputTokens("{\"input\": \"hi\"}").getEntity()).i64("input_tokens"),
                counted.i64("input_tokens"));
    }

    @Test
    void inputItemsPageInEitherOrder() {
        JsonView r = create("""
                {"input": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                           {"role": "user", "content": "c"}], "max_output_tokens": 1}""");
        JsonView page = view(responses.inputItems(r.str("id"), 2, "asc", null).getEntity());
        assertTrue(page.bool("has_more"));
        assertEquals(2, ((List<?>) page.raw("data")).size());
        JsonView rest = view(responses.inputItems(r.str("id"), 2, "asc", page.str("last_id")).getEntity());
        assertFalse(rest.bool("has_more"));
        assertEquals("c", at(rest, "data", 0).str("content"));
        JsonView desc = view(responses.inputItems(r.str("id"), 1, null, null).getEntity());
        assertEquals(at(rest, "data", 0).str("id"), at(desc, "data", 0).str("id"), "desc starts at the newest");
    }
}
