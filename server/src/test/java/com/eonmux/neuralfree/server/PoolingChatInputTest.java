package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's chat-message embedding input over the real service: a {@code messages} conversation, batched
 * {@code messages}, or {@code input} holding chat messages, each rendered through the model's chat template
 * and embedded once per conversation — on the embeddings and the pooling endpoints alike. A request's
 * {@code chat_template} renders in place of the model's only on a server launched with
 * {@code --trust-request-chat-template}.
 */
public class PoolingChatInputTest {

    private static EngineService bert;
    private static EngineService rank;
    private static Path rankModel;
    private static PoolingResource pooled;
    private static PoolingResource classifier;

    @BeforeAll
    static void open() throws IOException {
        Path bertModel = SyntheticModels.bert().writeTempFile();
        bert = new EngineService(bertModel.toString(), ServerTestSupport.config().build());
        rankModel = SyntheticModels.rank("neg", "pos").writeTempFile();
        rank = new EngineService(rankModel.toString(), ServerTestSupport.config().build());
        pooled = new PoolingResource(bert);
        classifier = new PoolingResource(rank);
    }

    @AfterAll
    static void close() {
        bert.close();
        rank.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus());
        return view(response.getEntity());
    }

    private static String user(String content) {
        return "{\"role\": \"user\", \"content\": \"" + content + "\"}";
    }

    @Test
    void messagesEmbedOneConversation() {
        JsonView single = ok(classifier.embeddings("{\"messages\": [" + user("hello there") + "]}"));
        assertEquals(1, size(single, "data"));
        int[] rendered = rank.encode(rank.formatChat(ChatRequest.builder()
                .addMessage(Map.of("role", "user", "content", "hello there")).build()), false, true);
        assertEquals(rendered.length, single.obj("usage").i64("prompt_tokens"),
                "the conversation renders and embeds as one prompt");
        JsonView batched = ok(classifier.embeddings(
                "{\"messages\": [[" + user("a") + "], [" + user("b") + "]]}"));
        assertEquals(2, size(batched, "data"), "one embedding per conversation");
    }

    @Test
    void inputCarriesTheSameConversations() {
        JsonView messages = ok(pooled.embeddings("{\"messages\": [" + user("hello") + "]}"));
        JsonView input = ok(pooled.embeddings("{\"input\": [" + user("hello") + "]}"));
        assertEquals(at(messages, "data", 0).raw("embedding"), at(input, "data", 0).raw("embedding"),
                "the normalized form answers the same vector");
        JsonView batchInput = ok(pooled.embeddings(
                "{\"input\": [[" + user("a") + "], [" + user("b") + "]]}"));
        assertEquals(2, size(batchInput, "data"));
        JsonView batchMessages = ok(pooled.embeddings(
                "{\"messages\": [[" + user("a") + "], [" + user("b") + "]]}"));
        assertEquals(at(batchMessages, "data", 1).raw("embedding"), at(batchInput, "data", 1).raw("embedding"),
                "batched input matches batched messages");
    }

    @Test
    void poolingAnswersTheConversationPerTask() {
        JsonView embeddings = ok(pooled.embeddings("{\"messages\": [" + user("hello") + "]}"));
        JsonView pooling = ok(pooled.pooling("{\"messages\": [" + user("hello") + "]}"));
        assertEquals(at(embeddings, "data", 0).raw("embedding"), at(pooling, "data", 0).raw("data"),
                "the embed task answers the embeddings vector");
        JsonView classify = ok(classifier.pooling(
                "{\"task\": \"classify\", \"messages\": [" + user("hello") + "]}"));
        assertEquals(2, ((java.util.List<?>) at(classify, "data", 0).raw("data")).size());
    }

    @Test
    void malformedConversationsRefuseByName() {
        assertEquals(400, statusOf(() -> pooled.embeddings("{\"messages\": []}")));
        assertEquals(400, statusOf(() -> pooled.embeddings("{\"messages\": [{\"content\": \"no role\"}]}")));
        assertEquals(400, statusOf(() -> pooled.embeddings("{\"messages\": [\"a string\"]}")));
        assertEquals(400, statusOf(() -> pooled.embeddings("{\"input\": [[\"a string\"]]}")));
        assertEquals(400, statusOf(() -> pooled.embeddings("{\"messages\": [[]]}")));
    }

    @Test
    void completionFormsStillAnswer() {
        JsonView text = ok(pooled.embeddings("{\"input\": \"plain text\"}"));
        assertEquals(1, size(text, "data"));
        JsonView ids = ok(pooled.embeddings("{\"input\": [5, 6, 7]}"));
        assertEquals(3L, ids.obj("usage").i64("prompt_tokens"));
    }

    @Test
    void aRequestChatTemplateRendersOnlyOnATrustingServer() throws IOException {
        String template = "{% for m in messages %}[{{ m['content'] }}]{% endfor %}";
        String body = "{\"messages\": [" + user("hello there") + "], \"chat_template\": \"" + template + "\"}";
        String refusal = errorOf(() -> classifier.embeddings(body));
        assertTrue(refusal.contains("chat_template") && refusal.contains("--trust-request-chat-template"),
                "the refusal names the field and the flag: " + refusal);
        assertEquals(400, statusOf(() -> classifier.pooling(body)));
        EngineService trusting = new EngineService(rankModel.toString(), EngineConfig.builder().build(),
                ServerOptions.defaults().withTrustRequestChatTemplate(true));
        try {
            JsonView embedded = ok(new PoolingResource(trusting).embeddings(body));
            int[] rendered = trusting.encode(trusting.formatChat(ChatRequest.builder()
                    .addMessage(Map.of("role", "user", "content", "hello there")).chatTemplate(template).build()),
                    false, true);
            assertEquals(rendered.length, embedded.obj("usage").i64("prompt_tokens"),
                    "the request's template rendered the conversation");
        } finally {
            trusting.close();
        }
    }
}
