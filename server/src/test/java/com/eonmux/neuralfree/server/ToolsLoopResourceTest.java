package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.backend;
import static com.eonmux.neuralfree.server.ServerTestSupport.config;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import jakarta.ws.rs.WebApplicationException;
/**
 * The chat tool loop end to end over the synthetic llama carrying a hermes-style template: a request listing a
 * server tool ({@code type: "server"}) and forcing its call through {@code tool_choice} runs the tool on the
 * server, feeds the answer back as a tool message, and generates the next round — the response carries every
 * executed call with its real result, the usage sums the rounds, and the refusals name the field. The client
 * function tools keep their old path: no loop, the calls go back to the caller.
 */
public class ToolsLoopResourceTest {

    private static final String TEMPLATE = "{%- if tools %}<|im_start|>system\nTools:"
            + "{% for t in tools %} {{ t.function.name }}{% endfor %}. Call one inside <tool_call></tool_call>."
            + "<|im_end|>\n{% endif %}"
            + "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            + "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    private static final String SERVER_TOOL = "[{\"type\": \"server\", \"tool\": \"get_info\"}]";

    private static final String FORCED = "\"tool_choice\": {\"type\": \"function\","
            + " \"function\": {\"name\": \"get_info\"}}";

    private static EngineService service;
    private static OpenAiResource openAi;
    private static String roundsBackup;
    private static Path modelPath;
    private static com.eonmux.neuralfree.gguf.engine.EngineConfig engineConfig;
    private static ServerOptions serverOptions;

    @org.junit.jupiter.api.io.TempDir
    Path dir;

    @BeforeAll
    static void open() throws IOException {
        SyntheticModels.Config model = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Path path = SyntheticModels.llama(model).str("tokenizer.chat_template", TEMPLATE).writeTempFile();
        modelPath = path;
        engineConfig = config().memory(com.eonmux.neuralfree.gguf.memory.MemoryConfig.builder().nCtx(4096)
                        .ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build()).backend(backend())
                .build();
        serverOptions = new ServerOptions(null, Duration.ofMinutes(30), false, 0, null,
                ServerOptions.DEFAULT_SLEEP_IDLE_SECONDS, 1, null, false, null, null, null,
                List.of("get_info"), null, null, null, null);
        service = new EngineService(path.toString(), engineConfig, serverOptions);
        openAi = new OpenAiResource(service);
        roundsBackup = System.setProperty(ToolsLoop.MAX_ROUNDS_PROPERTY, "2");
    }

    @AfterAll
    static void close() {
        service.close();
        if (roundsBackup == null) {
            System.clearProperty(ToolsLoop.MAX_ROUNDS_PROPERTY);
        } else {
            System.setProperty(ToolsLoop.MAX_ROUNDS_PROPERTY, roundsBackup);
        }
    }

    private static String chat(String tools, String extra) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": \"What is the runtime?\"}],"
                + " \"max_tokens\": 400, \"temperature\": 0, \"tools\": " + tools + ", " + extra + "}";
    }

    @Test
    void theLoopExecutesTheToolAndFeedsTheAnswerBackEveryRound() {
        JsonView response = view(openAi.chatCompletions(chat(SERVER_TOOL, FORCED)).getEntity());
        JsonView choice = at(response, "choices", 0);
        JsonView executed = JsonView.of(((List<?>) choice.raw("server_tool_calls")).get(0),
                "server_tool_calls[0]");
        assertEquals("get_info", executed.str("name"));
        assertTrue(executed.str("id").startsWith("call_"), "the synthesized id rides the history too");
        JsonView result = executed.obj("result");
        assertFalse(result.str("os").isEmpty());
        assertTrue(result.has("cwd"));
        JsonView usage = response.obj("usage");
        assertTrue(usage.i64("completion_tokens") > 0);
        assertTrue(usage.i64("prompt_tokens") > 0);
    }

    @Test
    void aRouterModeRequestRunsTheLoopOnItsModel() throws IOException {
        java.nio.file.Files.copy(modelPath, dir.resolve("a.gguf"));
        ModelRouter router = new ModelRouter(dir, null, 0, 0, true, engineConfig, serverOptions,
                ModelRouter.systemDeviceMemory());
        EngineService unloaded = new EngineService(null, engineConfig);
        try {
            OpenAiResource routed = new OpenAiResource(unloaded, router);
            JsonView response = view(routed.chatCompletions(
                    chat(SERVER_TOOL, FORCED + ", \"model\": \"a\"")).getEntity());
            JsonView choice = at(response, "choices", 0);
            JsonView executed = at(choice, "server_tool_calls", 0);
            assertEquals("get_info", executed.str("name"));
            assertFalse(executed.obj("result").str("os").isEmpty());
            assertTrue(response.obj("usage").i64("completion_tokens") > 0);
        } finally {
            router.close();
            unloaded.close();
        }
    }

    @Test
    void theRoundBoundKeepsThePendingCallsInTheResponse() {
        System.setProperty(ToolsLoop.MAX_ROUNDS_PROPERTY, "1");
        try {
            JsonView response = view(openAi.chatCompletions(chat(SERVER_TOOL, "\"tool_choice\": \"required\""))
                    .getEntity());
            JsonView choice = at(response, "choices", 0);
            assertEquals(1, size(choice, "server_tool_calls"));
            assertEquals("get_info", at(choice, "server_tool_calls", 0).str("name"));
            assertEquals("tool_calls", choice.str("finish_reason"));
        } finally {
            if (roundsBackup == null) {
                System.clearProperty(ToolsLoop.MAX_ROUNDS_PROPERTY);
            } else {
                System.setProperty(ToolsLoop.MAX_ROUNDS_PROPERTY, "2");
            }
        }
    }

    /** The refusal message of a call that must be refused, naming it in the failure when it is not. */
    private static String refusal(Runnable call, String label) {
        try {
            call.run();
        } catch (WebApplicationException e) {
            String message = view(e.getResponse().getEntity()).obj("error").str("message");
            assertTrue(message.contains(label), label + " refusal: " + message);
            return message;
        }
        throw new AssertionError(label + " was not refused");
    }

    @Test
    void theLoopRefusalsNameTheField() {
        refusal(() -> openAi.chatCompletions(chat(SERVER_TOOL,
                "\"tool_choice\": \"required\", \"stream\": true")), "stream");
        refusal(() -> openAi.chatCompletions(chat(SERVER_TOOL,
                "\"tool_choice\": \"required\", \"n\": 2")), "one choice");
        refusal(() -> openAi.chatCompletions(chat(SERVER_TOOL,
                "\"tool_choice\": \"required\", \"best_of\": 2")), "best_of");
        refusal(() -> openAi.chatCompletions(chat(
                "[{\"type\": \"server\", \"tool\": \"no_such\"}]", "\"tool_choice\": \"required\"")), "no_such");
        refusal(() -> openAi.chatCompletions(chat(
                "[{\"type\": \"web_search_preview\"}]", "\"tool_choice\": \"required\"")), "web_search_preview");
        refusal(() -> openAi.chatCompletions(chat(
                "[{\"type\": \"server\"}]", "\"tool_choice\": \"required\"")), "tools[0].tool");
    }

    @Test
    void aRequestWithoutServerToolsKeepsThePlainPath() {
        String clientTools = "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                + " \"parameters\": {\"type\": \"object\", \"properties\": {}, \"additionalProperties\": false}}}]";
        JsonView response = view(openAi.chatCompletions(chat(clientTools, "\"tool_choice\": \"required\""))
                .getEntity());
        JsonView choice = at(response, "choices", 0);
        assertEquals(1, size(choice.obj("message"), "tool_calls"));
        assertEquals("get_weather",
                at(choice.obj("message"), "tool_calls", 0).obj("function").str("name"));
        assertFalse(choice.has("server_tool_calls"), "no server execution happened");
    }
}
