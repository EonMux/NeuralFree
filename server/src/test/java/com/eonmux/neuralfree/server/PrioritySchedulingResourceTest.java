package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.batch.SchedulingPolicy;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static com.eonmux.neuralfree.server.ServerTestSupport.config;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's request {@code priority} over the real service under each scheduling policy: with
 * {@code neuralfree.scheduling.policy=priority} the completions, chat, Responses and llama.cpp {@code /completion}
 * routes accept any 32-bit value, streamed or not; under the default {@code fcfs} policy the llama.cpp route
 * refuses a nonzero value naming {@code priority}, as the OpenAI routes do.
 */
class PrioritySchedulingResourceTest {

    private static EngineService priority;
    private static EngineService fcfs;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config().memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).schedulingPolicy(SchedulingPolicy.PRIORITY).build()).build();
        priority = new EngineService(model.toString(), cfg);
        fcfs = new EngineService(model.toString(), pagedConfig(64, 2));
    }

    @AfterAll
    static void close() {
        priority.close();
        fcfs.close();
    }

    @Test
    void thePriorityPolicyServesAnyPriorityOnTheOpenAiRoutes() throws IOException {
        assertEquals(SchedulingPolicy.PRIORITY, priority.schedulingPolicy());
        OpenAiResource openAi = new OpenAiResource(priority);
        Response completion = openAi.completions("""
                {"prompt": "hi", "max_tokens": 2, "temperature": 0, "priority": -5}""");
        assertEquals(200, completion.getStatus());
        assertEquals("text_completion", view(completion.getEntity()).str("object"));
        assertEquals(200, statusOf(() -> openAi.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "temperature": 0,
                 "priority": 2147483647}""")));
        Response stream = openAi.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "temperature": 0,
                 "priority": 3, "stream": true}""");
        assertFalse(dataFrames(streamText(stream)).isEmpty(), "the stream runs");
        assertEquals(400, statusOf(() -> openAi.completions("""
                {"prompt": "hi", "max_tokens": 2, "priority": 2147483648}""")),
                "a value wider than 32 bits still refuses");
    }

    @Test
    void thePriorityPolicyServesAnyPriorityOnTheResponsesAndLlamaRoutes() {
        ResponsesResource responses = new ResponsesResource(priority);
        assertEquals(200, statusOf(() -> responses.create("""
                {"input": "hi", "max_output_tokens": 2, "temperature": 0, "priority": 7, "store": false}""")));
        LlamaCompletionResource llama = new LlamaCompletionResource(priority);
        assertEquals(200, statusOf(() -> llama.completion("""
                {"prompt": "hi", "n_predict": 2, "temperature": 0, "priority": -1}""")));
    }

    @Test
    void fcfsRefusesANonzeroPriorityOnTheResponsesAndLlamaRoutes() {
        assertEquals(SchedulingPolicy.FCFS, fcfs.schedulingPolicy());
        String refusal = errorOf(() -> new LlamaCompletionResource(fcfs).completion("""
                {"prompt": "hi", "n_predict": 2, "priority": 4}"""));
        assertTrue(refusal.startsWith("priority 4"), refusal);
        assertEquals(400, statusOf(() -> new ResponsesResource(fcfs).create("""
                {"input": "hi", "max_output_tokens": 2, "priority": 7, "store": false}""")));
        assertEquals(200, statusOf(() -> new LlamaCompletionResource(fcfs).completion("""
                {"prompt": "hi", "n_predict": 2, "temperature": 0, "priority": 0}""")));
    }
}
