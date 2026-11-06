package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The chat completion's per-request output controls over the real service: {@code echo} prepends the
 * continued last message to the answer (vLLM's chat echo, in one delta on a stream), and it stays
 * silent when the last message is another role; with {@code return_token_ids} the ids ride beside the echo. The
 * context here is wider than the OpenAiResourceTest one, since a continued conversation renders longer.
 */
public class ChatControlsResourceTest {

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2));
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void echoPrependsTheLastAssistantMessage() {
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"},"
                + " {\"role\": \"assistant\", \"content\": \"OK.\"}],"
                + " \"continue_final_message\": true, \"max_tokens\": 3, \"temperature\": 0";
        JsonView echoed = view(resource.chatCompletions(body + ", \"echo\": true}").getEntity());
        assertTrue(at(echoed, "choices", 0).obj("message").str("content").startsWith("OK."),
                "the echo carries the continued message");
        JsonView plain = view(resource.chatCompletions(body + "}").getEntity());
        assertFalse(at(plain, "choices", 0).obj("message").str("content").startsWith("OK."),
                "without echo the message starts with the generation");
    }

    @Test
    void echoDoesNotApplyWhenTheLastMessageIsAnotherRole() {
        JsonView body = view(resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 2,"
                        + " \"temperature\": 0, \"echo\": true}").getEntity());
        assertFalse(at(body, "choices", 0).obj("message").str("content").startsWith("hi"),
                "a user message last is not the continued role");
    }

    @Test
    void aStreamedEchoCarriesTheContinuedMessageInItsOwnDelta() throws Exception {
        Response response = resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"},"
                        + " {\"role\": \"assistant\", \"content\": \"OK.\"}],"
                        + " \"continue_final_message\": true, \"echo\": true, \"max_tokens\": 2,"
                        + " \"temperature\": 0, \"stream\": true}");
        StreamingOutput stream = (StreamingOutput) response.getEntity();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        stream.write(out);
        boolean echoed = false;
        for (String frame : out.toString(StandardCharsets.UTF_8).split("\n\n")) {
            if (!frame.startsWith("data: ") || frame.equals("data: [DONE]")) {
                continue;
            }
            JsonView chunk = JsonView.of(Json.parse(frame.substring("data: ".length()), Json.Limits.UNTRUSTED),
                    "chunk");
            if (!(chunk.raw("choices") instanceof List<?> choices) || choices.isEmpty()) {
                continue;
            }
            JsonView delta = JsonView.of(choices.get(0), "choice").obj("delta");
            if (delta.has("content") && delta.str("content").equals("OK.")) {
                echoed = true;
            }
        }
        assertTrue(echoed, "a delta carries the echoed message");
    }

    @Test
    void echoAndReturnTokenIdsShapeOneAnswer() {
        JsonView body = view(resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"},"
                        + " {\"role\": \"assistant\", \"content\": \"OK.\"}],"
                        + " \"continue_final_message\": true, \"echo\": true, \"return_token_ids\": true,"
                        + " \"max_tokens\": 3, \"temperature\": 0}").getEntity());
        JsonView choice = at(body, "choices", 0);
        assertTrue(choice.obj("message").str("content").startsWith("OK."), "the echo leads the content");
        assertEquals(body.obj("usage").i64("completion_tokens"), ((List<?>) choice.raw("token_ids")).size(),
                "token_ids name the generated tokens only, not the echo");
        assertEquals(body.obj("usage").i64("prompt_tokens"), ((List<?>) body.raw("prompt_token_ids")).size(),
                "the response names the prompt ids");
    }

    @Test
    void aStreamedEchoFollowsTheRoleChunkThatCarriesThePromptIds() throws Exception {
        Response response = resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"},"
                        + " {\"role\": \"assistant\", \"content\": \"OK.\"}],"
                        + " \"continue_final_message\": true, \"echo\": true, \"return_token_ids\": true,"
                        + " \"max_tokens\": 2, \"temperature\": 0, \"stream\": true}");
        List<JsonView> frames = ServerTestSupport.dataFrames(ServerTestSupport.streamText(response));
        JsonView first = frames.get(0);
        assertTrue(first.raw("prompt_token_ids") instanceof List<?>, "the role chunk carries the prompt ids");
        assertEquals("assistant", at(first, "choices", 0).obj("delta").str("role"));
        JsonView second = frames.get(1);
        assertEquals("OK.", at(second, "choices", 0).obj("delta").str("content"), "the echo delta comes next");
        assertFalse(second.has("prompt_token_ids"), "only the first chunk names the prompt ids");
    }
}
