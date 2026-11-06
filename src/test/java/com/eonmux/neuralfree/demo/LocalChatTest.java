package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * {@link LocalChat} over a real synthetic llama model and the engine generation loop: the session's turns render
 * through the chat template and generate, {@code /regen} under greedy sampling reproduces the answer, Ctrl-C
 * cancels through the request hook, a prompt over the KV cache leaves the oldest exchanges out (or fails by name
 * when even the last message does not fit), media is refused naming the kind, and each prompt renders with and each
 * request carries the template variables of {@code --thinking}, {@code --reasoning-effort} and
 * {@code --chat-template-kwargs}.
 */
public class LocalChatTest {

    private static final int N_CTX = 64;

    private static GGUFModel model;
    private static EngineConfig config;

    @BeforeClass
    public static void open() throws IOException {
        Path file = SyntheticModels.llama().writeTempFile();
        config = EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        model = new GGUFModel(file.toString(), config);
    }

    @AfterClass
    public static void close() throws IOException {
        model.close();
    }

    private final ByteArrayOutputStream console = new ByteArrayOutputStream();
    private final PrintStream out = new PrintStream(console, true);
    private final List<String> stats = new ArrayList<>();
    private final Interrupts interrupts = new Interrupts(code -> {
        throw new AssertionError("unexpected exit " + code);
    }, out);

    private LocalChat backend(LLMDemoOptions opts) {
        return new LocalChat(model, TurnGenerator.loop(model, config), opts, out, stats::add);
    }

    private static LLMDemoOptions opts(String... args) throws LLMDemoOptions.UsageException {
        return LLMDemoOptions.parse(args);
    }

    private String output() {
        return new String(console.toByteArray(), StandardCharsets.UTF_8);
    }

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    @Test
    public void aSessionGeneratesAndRegenReproducesTheGreedyAnswer() throws Exception {
        LLMDemoOptions o = opts("-n", "3", "--temp", "0", "--system", "");
        ChatSession chat = new ChatSession(backend(o), o, new BufferedReader(new StringReader("hello\n/regen\n")),
                out, null, interrupts);
        chat.run();
        List<Map<String, Object>> messages = chat.messages();
        assertEquals(2, messages.size());
        assertEquals("assistant", messages.get(1).get("role"));
        assertEquals("two generations ran", 2, output().split("\\[Prompt: ", -1).length - 1);
        assertTrue(output().contains("model      : llama"));
        assertEquals("each turn hands four statistics lines to the sink", 8, stats.size());
        assertTrue(stats.get(1), stats.get(1).equals("  Input: hello"));
        String first = stats.get(2);
        String second = stats.get(6);
        assertEquals("greedy decoding over the same history answers the same", first, second);
    }

    @Test
    public void aCancelledTurnStopsAtOnce() throws Exception {
        LLMDemoOptions o = opts("-n", "16", "--temp", "0", "--system", "");
        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(message("user", "hello"));
        ChatAnswer answer = backend(o).answer(messages, () -> true);
        assertNull(answer.error());
        assertTrue(output(), output().contains("[interrupted]"));
        assertTrue(output(), output().contains("(CANCELLED)"));
    }

    @Test
    public void theOldestExchangesLeaveAPromptThatFillsTheCache() throws Exception {
        LLMDemoOptions o = opts("-n", "2", "--temp", "0", "--system", "");
        String words = "hello world hello world hello world hello world";
        List<Map<String, Object>> messages = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            messages.add(message("user", words));
            messages.add(message("assistant", words));
        }
        messages.add(message("user", "hi"));
        ChatAnswer answer = backend(o).answer(messages, () -> false);
        assertNull(answer.error());
        assertTrue(output(), output().contains("[context full: the oldest"));
    }

    @Test
    public void aLastMessageOverTheCacheFailsByName() throws Exception {
        LLMDemoOptions o = opts("-n", "2", "--temp", "0", "--system", "");
        StringBuilder words = new StringBuilder();
        for (int i = 0; i < N_CTX; i++) {
            words.append("hello world ");
        }
        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(message("user", words.toString()));
        ChatAnswer answer = backend(o).answer(messages, () -> false);
        assertNotNull(answer.error());
        assertTrue(answer.error(), answer.error().contains("fills the KV cache of " + N_CTX));
    }

    @Test
    public void mediaIsRefusedNamingTheKind() throws Exception {
        LocalChat chat = backend(opts());
        for (String kind : new String[]{"image", "audio", "video"}) {
            String refusal = chat.mediaRefusal(kind);
            assertNotNull(refusal);
            assertTrue(refusal, refusal.contains("no " + kind + " pipeline"));
        }
    }

    @Test
    public void theConsoleRequestCarriesTheThinkingSwitchAsATemplateVariable() throws Exception {
        GenerationRequest base = GenerationRequest.builder()
                .templateVariables(Collections.singletonMap("reasoning_effort", "high")).build();
        GenerationRequest off = LocalChat.templated(base, false, Collections.emptyMap());
        assertEquals(Boolean.FALSE, off.templateVariables().get("enable_thinking"));
        assertEquals("the request's own variables stay", "high", off.templateVariables().get("reasoning_effort"));
        assertEquals(Boolean.TRUE, LocalChat.templated(base, true, Collections.emptyMap()).templateVariables()
                .get("enable_thinking"));
    }

    @Test
    public void anAutoThinkingSwitchLeavesEnableThinkingToTheTemplate() {
        GenerationRequest base = GenerationRequest.builder().build();
        assertFalse(LocalChat.templated(base, null, Collections.emptyMap()).templateVariables()
                .containsKey("enable_thinking"));
        ChatRequest chat = LocalChat.chatRequest(Collections.singletonList(textMessage("user", "Hi")), null,
                Collections.singletonMap("reasoning_effort", "low"));
        assertNull(chat.enableThinking());
        assertEquals(Collections.singletonMap("reasoning_effort", "low"), chat.templateVariables());
    }

    @Test
    public void chatTemplateKwargsBindAfterTheThinkingSwitch() {
        Map<String, Object> kwargs = new LinkedHashMap<>();
        kwargs.put("enable_thinking", true);
        kwargs.put("style", "terse");
        GenerationRequest base = GenerationRequest.builder().build();
        Map<String, Object> variables = LocalChat.templated(base, false, kwargs).templateVariables();
        assertEquals("a kwarg replaces --thinking's value, as llama.cpp binds kwargs last", Boolean.TRUE,
                variables.get("enable_thinking"));
        assertEquals("terse", variables.get("style"));
        ChatRequest chat = LocalChat.chatRequest(Collections.singletonList(textMessage("user", "Hi")), false, kwargs);
        assertEquals(Boolean.FALSE, chat.enableThinking());
        assertEquals(variables, chat.templateVariables());
    }

    @Test
    public void theTemplateVariableFlagsReachTheRenderedPromptAndTheRequest() throws Exception {
        LLMDemoOptions flags = opts("--thinking", "--reasoning-effort", "low",
                "--chat-template-kwargs", "{\"enable_thinking\": false, \"persona\": \"terse\"}");
        GenerationRequest turn = LocalChat.templated(GenerationRequest.builder().build(), flags.thinking,
                flags.templateKwargs);
        assertEquals("low", turn.templateVariables().get("reasoning_effort"));
        assertEquals("the kwargs bind over --thinking", Boolean.FALSE, turn.templateVariables().get("enable_thinking"));
        assertEquals("terse", turn.templateVariables().get("persona"));
        ChatRequest chat = LocalChat.chatRequest(Collections.singletonList(textMessage("user", "hi")), flags.thinking,
                flags.templateKwargs);
        assertEquals("the effort is the request's own, bound with its aliases", "low", chat.reasoningEffort());
        assertEquals(turn.templateVariables(), chat.templateVariables());

        String template = "{% for m in messages %}{{ m.content }}{% endfor %}|{{ reasoning_effort }}"
                + "|{{ reasoning_strength }}|{{ persona }}|{{ enable_thinking }}";
        GGUFModel templated = new GGUFModel(SyntheticModels.llama()
                .str(ChatTemplateSource.KEY_DEFAULT, template).writeTempFile().toString(), config);
        try {
            assertEquals("hi|low|low|terse|False", templated.tokenizer().formatChat(chat));
            assertEquals("Be brief.hi|low|low|terse|False", LocalChat.singleTurnPrompt(templated.tokenizer(),
                    "Be brief.", "hi", flags.thinking, flags.templateKwargs));
        } finally {
            templated.close();
        }
    }

    @Test
    public void theSingleTurnPromptRendersLikeTheTokenizersOwnSingleTurn() {
        assertEquals(model.tokenizer().formatChat("Be brief.", "Hi", false),
                LocalChat.singleTurnPrompt(model.tokenizer(), "Be brief.", "Hi", false, Collections.emptyMap()));
        assertEquals(model.tokenizer().formatChat("", "Hi", true),
                LocalChat.singleTurnPrompt(model.tokenizer(), "", "Hi", true, Collections.emptyMap()));
    }

    private static Map<String, String> textMessage(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }
}
