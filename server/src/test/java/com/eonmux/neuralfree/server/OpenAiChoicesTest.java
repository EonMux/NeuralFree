package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Several choices per request: {@code n} choices decoded side by side (with {@code seed + i} per choice), every
 * OpenAI prompt form, {@code echo}, the aggregated usage and the streamed choice indices with the usage on the
 * last finish chunk or, under {@code stream_options.include_usage}, in its own chunk.
 */
public class OpenAiChoicesTest {

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(64, 4));
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void nChoicesAnswerWithTheirIndicesAndSummedUsage() {
        JsonView body = view(resource.completions("""
                {"prompt": "hello", "max_tokens": 3, "temperature": 0, "n": 3}""").getEntity());
        assertEquals(3, size(body, "choices"));
        for (int i = 0; i < 3; i++) {
            JsonView choice = at(body, "choices", i);
            assertEquals(i, choice.i64("index"));
            assertNotNull(choice.str("text"));
        }
        JsonView usage = body.obj("usage");
        long generated = usage.i64("completion_tokens");
        assertTrue(generated >= 3, "every choice's tokens count: " + generated);
        assertEquals(usage.i64("prompt_tokens") + generated, usage.i64("total_tokens"));
        String first = at(body, "choices", 0).str("text");
        for (int i = 1; i < 3; i++) {
            assertEquals(first, at(body, "choices", i).str("text"), "greedy choices agree");
        }
    }

    @Test
    void aSeededRequestRepeatsItsChoices() {
        String request = """
                {"prompt": "hello", "max_tokens": 4, "temperature": 1.0, "seed": 11, "n": 2}""";
        JsonView a = view(resource.completions(request).getEntity());
        JsonView b = view(resource.completions(request).getEntity());
        for (int i = 0; i < 2; i++) {
            assertEquals(at(a, "choices", i).str("text"), at(b, "choices", i).str("text"),
                    "choice " + i + " samples with the same per-choice seed");
        }
    }

    @Test
    void choiceSeedsFollowVllm() {
        GenerationRequest seeded = GenerationRequest.builder()
                .sampler(GenerationRequest.defaults().sampler().toBuilder().seed(40).build()).build();
        assertSame(seeded, OpenAiRequests.forChoice(seeded, 0));
        assertEquals(42L, OpenAiRequests.forChoice(seeded, 2).sampler().seed());
        GenerationRequest random = GenerationRequest.builder().build();
        assertEquals(SamplerConfig.RANDOM_SEED, random.sampler().seed());
        assertSame(random, OpenAiRequests.forChoice(random, 3), "unseeded choices draw their own seeds");
    }

    @Test
    void everyPromptFormAnswersChoicePTimesNPlusI() {
        JsonView strings = view(resource.completions("""
                {"prompt": ["hello", "world"], "max_tokens": 2, "temperature": 0, "n": 2}""").getEntity());
        assertEquals(4, size(strings, "choices"));
        for (int i = 0; i < 4; i++) {
            assertEquals(i, at(strings, "choices", i).i64("index"));
        }
        int[] hello = service.encode("hello", true, true);
        int[] world = service.encode("world", true, true);
        assertEquals(hello.length + world.length, strings.obj("usage").i64("prompt_tokens"),
                "each prompt counts once, whatever its n");
        assertEquals(at(strings, "choices", 0).str("text"), at(strings, "choices", 1).str("text"));

        StringBuilder ids = new StringBuilder();
        for (int i = 0; i < hello.length; i++) {
            ids.append(i == 0 ? "" : ", ").append(hello[i]);
        }
        JsonView tokens = view(resource.completions("{\"prompt\": [" + ids + "], \"max_tokens\": 2,"
                + " \"temperature\": 0}").getEntity());
        assertEquals(at(strings, "choices", 0).str("text"), at(tokens, "choices", 0).str("text"),
                "token ids prompt the same as their text");
        JsonView nested = view(resource.completions("{\"prompt\": [[" + ids + "], [" + ids + "]],"
                + " \"max_tokens\": 2, \"temperature\": 0}").getEntity());
        assertEquals(2, size(nested, "choices"));
    }

    @Test
    void echoPrependsThePrompt() {
        JsonView body = view(resource.completions("""
                {"prompt": "hello", "max_tokens": 2, "temperature": 0, "echo": true}""").getEntity());
        assertTrue(at(body, "choices", 0).str("text").startsWith("hello"));
    }

    @Test
    void aStreamFinishesEveryChoiceAndCarriesTheUsageOnTheLastFinish() throws IOException {
        List<JsonView> frames = dataFrames(streamText(resource.completions("""
                {"prompt": "hello", "max_tokens": 3, "temperature": 0, "n": 2, "stream": true}""")));
        Set<Long> finished = new HashSet<>();
        int withUsage = 0;
        for (JsonView frame : frames) {
            JsonView choice = at(frame, "choices", 0);
            if (choice.raw("finish_reason") != null) {
                finished.add(choice.i64("index"));
            }
            if (frame.raw("usage") != null) {
                withUsage++;
                assertEquals(2, finished.size(), "the usage rides the last finish chunk");
            }
        }
        assertEquals(Set.of(0L, 1L), finished);
        assertEquals(1, withUsage);
    }

    @Test
    void includeUsageAddsTheUsageChunk() throws IOException {
        List<JsonView> frames = dataFrames(streamText(resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0, "n": 2,
                 "stream": true, "stream_options": {"include_usage": true}}""")));
        JsonView last = frames.get(frames.size() - 1);
        assertEquals(0, size(last, "choices"));
        assertTrue(last.obj("usage").i64("completion_tokens") >= 2);
        Map<Long, Integer> roles = new HashMap<>();
        for (JsonView frame : frames.subList(0, frames.size() - 1)) {
            assertNull(frame.raw("usage"), "only the usage chunk carries the usage");
            JsonView choice = at(frame, "choices", 0);
            if (JsonView.of(choice.raw("delta"), "delta").has("role")) {
                roles.merge(choice.i64("index"), 1, Integer::sum);
            }
        }
        assertEquals(Map.of(0L, 1, 1L, 1), roles, "each choice opens with one role delta");
    }

    @Test
    void chatChoicesAnswerTogether() {
        JsonView body = view(resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0,
                 "n": 2}""").getEntity());
        assertEquals(2, size(body, "choices"));
        for (int i = 0; i < 2; i++) {
            JsonView message = at(body, "choices", i).obj("message");
            assertEquals("assistant", message.str("role"));
            assertNull(message.raw("tool_calls"), "no tools, no calls");
        }
    }
}
