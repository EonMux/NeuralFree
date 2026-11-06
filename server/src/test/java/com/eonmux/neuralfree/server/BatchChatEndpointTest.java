package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The batched chat endpoint ({@code POST /v1/chat/completions/batch}, vLLM's {@code BatchChatCompletionRequest})
 * over the real service: {@code messages} lists conversations, every other field is shared, and one generation
 * batch answers one choice per conversation with usage summed over them. The vLLM extras apply per conversation
 * ({@code truncate_prompt_tokens}, {@code return_token_ids} on each choice, the cached-token usage). Streaming,
 * tools, {@code prompt_logprobs} and {@code n > 1} refuse by name; a broken conversation refuses naming its index.
 */
public class BatchChatEndpointTest {

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(64).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config);
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void twoConversationsAnswerAsTwoChoicesOverOneBatch() {
        Response response = resource.batchChatCompletions("""
                {"messages": [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "hello"}]],
                 "max_tokens": 4, "temperature": 0}""");
        assertEquals(200, response.getStatus());
        JsonView body = ServerTestSupport.view(response.getEntity());
        assertEquals("chat.completion", body.str("object"));
        assertEquals(2, ServerTestSupport.size(body, "choices"));
        for (int i = 0; i < 2; i++) {
            JsonView choice = ServerTestSupport.at(body, "choices", i);
            assertEquals(i, choice.i32("index"), "one choice per conversation, in order");
            assertEquals("assistant", JsonView.of(choice.raw("message"), "message").str("role"));
            assertNotNull(JsonView.of(choice.raw("message"), "message").raw("content"));
            assertTrue(choice.str("finish_reason").equals("stop") || choice.str("finish_reason").equals("length"));
        }
        JsonView usage = JsonView.of(body.raw("usage"), "usage");
        assertEquals(8L, usage.i64("completion_tokens"), "every conversation fills the token budget");
        assertEquals(usage.i64("prompt_tokens") + usage.i64("completion_tokens"), usage.i64("total_tokens"),
                "usage sums the conversations");
        assertTrue(usage.i64("prompt_tokens") > 4L, "prompt tokens count both conversations");
    }

    @Test
    void sharedSamplingAndLogprobsReachEveryConversation() {
        Response response = resource.batchChatCompletions("""
                {"messages": [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "hello"}]],
                 "max_tokens": 2, "temperature": 0, "logprobs": true, "top_logprobs": 2}""");
        assertEquals(200, response.getStatus());
        JsonView body = ServerTestSupport.view(response.getEntity());
        for (int i = 0; i < 2; i++) {
            JsonView choice = ServerTestSupport.at(body, "choices", i);
            JsonView logprobs = JsonView.of(choice.raw("logprobs"), "logprobs");
            List<?> content = (List<?>) logprobs.raw("content");
            assertEquals(2, content.size(), "one logprobs entry per generated token of the conversation");
            JsonView first = JsonView.of(content.get(0), "content[i]");
            assertEquals(2, ((List<?>) first.raw("top_logprobs")).size(), "the shared top_logprobs count");
        }
    }

    @Test
    void streamingToolsAndNAreRefusedByName() {
        String two = """
                {"messages": [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "hello"}]],
                 "max_tokens": 2, "temperature": 0,""";
        assertEquals(400, ServerTestSupport.statusOf(() -> resource.batchChatCompletions(two + " \"stream\": true}")),
                "batch chat completions do not stream");
        assertEquals(400, ServerTestSupport.statusOf(() -> resource.batchChatCompletions(two + " \"n\": 2}")),
                "n must be 1");
        assertEquals(400, ServerTestSupport.statusOf(() -> resource.batchChatCompletions("""
                {"messages": [[{"role": "user", "content": "hi"}]], "tools": [{"type": "function",
                 "function": {"name": "f", "parameters": {}}}], "max_tokens": 2}""")),
                "tools are not supported");
        assertEquals(400, ServerTestSupport.statusOf(() -> resource.batchChatCompletions(two
                + " \"tool_choice\": \"auto\"}")), "tool_choice must be none");
    }

    @Test
    void aBrokenConversationRefusesNamingItsIndex() {
        assertEquals(400, ServerTestSupport.statusOf(
                () -> resource.batchChatCompletions("""
                        {"messages": [], "max_tokens": 2}""")), "the conversation list may not be empty");
        assertEquals(400, ServerTestSupport.statusOf(
                () -> resource.batchChatCompletions("""
                        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}""")),
                "a conversation is an array of messages, not one message");
        assertEquals(400, ServerTestSupport.statusOf(
                () -> resource.batchChatCompletions("""
                        {"messages": [[{"role": "user", "content": "hi"}], []], "max_tokens": 2}""")),
                "an empty conversation refuses");
        String message = ServerTestSupport.errorOf(() -> resource.batchChatCompletions("""
                {"messages": [[{"role": "user", "content": "hi"}], [42]], "max_tokens": 2}"""));
        assertTrue(message.startsWith("messages[1]:"), "the refusal names the broken conversation: " + message);
    }

    @Test
    void tooManyConversationsRefuseByTheGenerationBudget() {
        StringBuilder conversations = new StringBuilder("[");
        for (int i = 0; i < 1025; i++) {
            if (i > 0) {
                conversations.append(',');
            }
            conversations.append("[{\"role\": \"user\", \"content\": \"hi\"}]");
        }
        conversations.append(']');
        String body = "{\"messages\": " + conversations + ", \"max_tokens\": 1}";
        assertEquals(400, ServerTestSupport.statusOf(() -> resource.batchChatCompletions(body)),
                "the conversations times n budget bounds one request");
    }

    @Test
    void vllmExtrasApplyToEveryConversation() {
        Response response = resource.batchChatCompletions("""
                {"messages": [[{"role": "user", "content": "hi there"}], [{"role": "user", "content": "hello"}]],
                 "max_tokens": 2, "temperature": 0, "return_token_ids": true, "truncate_prompt_tokens": 3}""");
        assertEquals(200, response.getStatus());
        JsonView body = ServerTestSupport.view(response.getEntity());
        JsonView usage = JsonView.of(body.raw("usage"), "usage");
        assertEquals(6L, usage.i64("prompt_tokens"), "each conversation's prompt is cut to three tokens");
        long generated = 0;
        for (int i = 0; i < 2; i++) {
            generated += ((List<?>) ServerTestSupport.at(body, "choices", i).raw("token_ids")).size();
        }
        assertEquals(usage.i64("completion_tokens"), generated, "every choice names its generated ids");
        assertTrue(usage.obj("prompt_tokens_details").i64("cached_tokens") >= 0L,
                "usage reports the cached prompt tokens");
    }

    @Test
    void promptLogprobsAreRefusedByName() {
        String message = ServerTestSupport.errorOf(() -> resource.batchChatCompletions("""
                {"messages": [[{"role": "user", "content": "hi"}]], "max_tokens": 2, "prompt_logprobs": 1}"""));
        assertTrue(message.contains("prompt_logprobs"), message);
    }
}
