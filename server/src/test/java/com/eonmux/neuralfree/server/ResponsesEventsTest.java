package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Responses event mapping on its own: argument text reaching the call it continues, the final output following
 * the tool parser's final reading, and a failure keeping the response's sequence and items.
 */
class ResponsesEventsTest {

    private final List<Map<String, Object>> events = new ArrayList<>();

    private ResponsesEvents events() {
        Map<String, Object> head = ResponsesEvents.head(ServerJson.parse("{\"input\": \"x\"}"), "resp_1", "m", 1,
                false);
        return new ResponsesEvents(head, (type, data) -> events.add(data), null, false);
    }

    private static ChatChoice.Delta started(int index, String id, String name) {
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("index", index);
        call.put("id", id);
        call.put("type", "function");
        call.put("function", Map.of("name", name, "arguments", ""));
        return new ChatChoice.Delta(Map.of("tool_calls", List.of(call)), null);
    }

    private static ChatChoice.Delta arguments(int index, String text) {
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("index", index);
        call.put("function", Map.of("arguments", text));
        return new ChatChoice.Delta(Map.of("tool_calls", List.of(call)), null);
    }

    private static ChatChoice.Delta content(String text) {
        return new ChatChoice.Delta(Map.of("content", text), null);
    }

    private static GenerationResult result() {
        return new GenerationResult(new int[] {1}, "", StopReason.EOS, null, GenerationResult.Timings.of(3, 1, 1, 1));
    }

    private List<String> types() {
        List<String> out = new ArrayList<>();
        for (Map<String, Object> e : events) {
            out.add((String) e.get("type"));
        }
        return out;
    }

    @Test
    void argumentsReachTheirCallWhenTrailingContentComesInTheSameFeed() throws IOException {
        ResponsesEvents r = events();
        r.start();
        r.accept(List.of(started(0, "call_a", "f"), arguments(0, "{\"a\":")));
        // the parser reports the trailing content before the rest of the arguments
        r.accept(List.of(content("done"), arguments(0, "1}")));
        Map<String, Object> done = events.stream()
                .filter(e -> "response.function_call_arguments.done".equals(e.get("type"))).findFirst().orElseThrow();
        assertEquals("{\"a\":1}", done.get("arguments"));
        assertTrue(types().indexOf("response.function_call_arguments.done")
                < types().indexOf("response.output_text.delta"), "the call closes before the message opens");
    }

    @Test
    void theFinalOutputFollowsTheFinalParse() throws IOException {
        ResponsesEvents r = events();
        r.start();
        r.accept(List.of(started(0, "call_a", "f"), arguments(0, "{}")));
        // the final reading turned the streamed call into content
        Map<String, Object> response = r.finish(
                result(),
                3, 0, new ToolParse("<call>f</call>", List.of()));
        List<?> output = (List<?>) response.get("output");
        assertEquals(1, output.size());
        Map<?, ?> message = (Map<?, ?>) output.get(0);
        assertEquals("message", message.get("type"));
        assertEquals("<call>f</call>", ((Map<?, ?>) ((List<?>) message.get("content")).get(0)).get("text"));

        events.clear();
        ResponsesEvents s = events();
        s.start();
        s.accept(List.of(content("hi"), started(0, "call_b", "g"), arguments(0, "{\"x\":")));
        Map<String, Object> fixed = s.finish(
                result(),
                3, 0, new ToolParse("hi", List.of(new ToolCall("g", "{\"x\":2}", "call_b", true))));
        List<?> items = (List<?>) fixed.get("output");
        assertEquals(2, items.size());
        Map<?, ?> call = (Map<?, ?>) items.get(1);
        assertEquals("{\"x\":2}", call.get("arguments"), "the final arguments, not the streamed prefix");
        assertEquals("call_b", call.get("call_id"));
    }

    @Test
    void aFailureContinuesTheSequenceAndKeepsTheItems() throws IOException {
        ResponsesEvents r = events();
        r.start();
        r.accept(List.of(content("partial")));
        Map<String, Object> failed = r.fail("boom");
        assertEquals("failed", failed.get("status"));
        assertEquals(1, ((List<?>) failed.get("output")).size(), "the produced message stays");
        for (int i = 0; i < events.size(); i++) {
            assertEquals(i, events.get(i).get("sequence_number"), "sequence numbers run on through the failure");
        }
        assertEquals("response.failed", events.get(events.size() - 1).get("type"));
        assertTrue(types().contains("response.output_item.done"), "the open item is closed");
    }

    @Test
    void theMergedMessageKeepsTheLogprobsOfEveryStreamedMessage() {
        Map<String, Object> first = message("msg_1", List.of("a"));
        Map<String, Object> call = new LinkedHashMap<>(Map.of("id", "fc_1", "type", "function_call",
                "call_id", "call_a", "name", "f", "arguments", "{}"));
        Map<String, Object> second = message("msg_2", List.of("b"));
        List<Map<String, Object>> out = ResponsesOutput.reconciled(List.of(first, call, second),
                new ToolParse("Let me done", List.of(new ToolCall("f", "{}", "call_a", true))));
        assertEquals(2, out.size());
        Map<?, ?> part = (Map<?, ?>) ((List<?>) out.get(0).get("content")).get(0);
        assertEquals(List.of("a", "b"), part.get("logprobs"));
        assertEquals("function_call", out.get(1).get("type"));
    }

    private static Map<String, Object> message(String id, List<Object> logprobs) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "output_text");
        part.put("logprobs", logprobs);
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("type", "message");
        m.put("content", List.of(part));
        return m;
    }
}
