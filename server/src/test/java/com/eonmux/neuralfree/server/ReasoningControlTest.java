package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The reasoning budget and the mid-stream control over the real engine: the synthetic llama runs a
 * {@code /completion} whose prompt opens a reasoning block, the zero budget forces the end-tag tokens from the
 * first sample, the control endpoint refuses malformed bodies and unknown ids, a live {@code reasoning_control}
 * completion ends its reasoning on demand, and an unarmed one refuses the control by name. The end tag rides the
 * byte-fallback tokens of {@code </think>}, so the forced sequence is exact.
 */
class ReasoningControlTest {

    private static EngineService service;
    private static OpenAiResource openAi;
    private static LlamaCompletionResource completions;
    private static int[] endTagTokens;

    @BeforeAll
    static void open() throws IOException {
        // no space prefix: the tags must tokenize to their exact byte sequence, the way they appear mid-stream
        GgufTestWriter writer = SyntheticModels.llama();
        writer.bool("tokenizer.ggml.add_space_prefix", false);
        Path model = writer.writeTempFile();
        EngineConfig config = EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(512).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config);
        openAi = new OpenAiResource(service);
        completions = new LlamaCompletionResource(service);
        endTagTokens = service.encode("</think>", false, true);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void anExhaustedBudgetForcesTheEndTagFromTheNextSample() {
        Response response = completions.completion("""
                {"prompt": "hi<think>", "n_predict": 12, "temperature": 0, "ignore_eos": true, "return_tokens": true,
                 "reasoning_budget_tokens": 0, "reasoning_budget_start_tag": "<think>",
                 "reasoning_budget_end_tag": "</think>"}""");
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        List<?> tokens = (List<?>) body.raw("tokens");
        assertEquals(12, tokens.size());
        for (int i = 0; i < endTagTokens.length; i++) {
            assertEquals(endTagTokens[i], ((Number) tokens.get(i)).intValue(),
                    "generated token " + i + " is the forced end-tag token");
        }
        assertTrue(body.str("content").contains("</think>"),
                "the forced end tag reaches the text: " + body.str("content"));
    }

    @Test
    void theControlEndpointRefusesMalformedBodiesAndUnknownIds() {
        assertEquals(400, statusOf(() -> openAi.control("{}")));
        assertTrue(errorOf(() -> openAi.control("{\"action\": \"reasoning_end\"}")).contains("id"));
        assertEquals(400, statusOf(() -> openAi.control(
                "{\"id\": \"chatcmpl-x\", \"action\": \"reasoning_begin\"}")));
        assertTrue(errorOf(() -> openAi.control("{\"id\": \"chatcmpl-x\", \"action\": \"reasoning_begin\"}"))
                .contains("unknown control action"));
        assertEquals(400, statusOf(() -> openAi.control(
                "{\"id\": \"chatcmpl-absent\", \"action\": \"reasoning_end\"}")));
        assertTrue(errorOf(() -> openAi.control("{\"id\": \"chatcmpl-absent\", \"action\": \"reasoning_end\"}"))
                .contains("no active completion for this id"));
    }

    @Test
    void aLiveReasoningControlCompletionEndsItsReasoningOnDemand() throws Exception {
        int[] prompt = service.encode("hi<think>", true, true);
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(256).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding())
                .reasoningControl(true)
                .reasoningBudgetStartTags(List.of("<think>")).reasoningBudgetEndTags(List.of("</think>"))
                .build();
        GenerationBatch batch = GenerationBatch.start(service,
                List.of(new GenerationBatch.Job(prompt, request, null, null, null)), null);
        service.completions().track("test-live", batch);
        awaitRunning(batch);
        assertTrue(forceUntilForced("test-live", 10_000),
                "the prompt's reasoning block arms the stage");
        batch.awaitAll();
        GenerationResult result = batch.result(0);
        assertTrue(contains(result.tokens(), endTagTokens),
                "the forced end tag runs in the generated tokens: " + java.util.Arrays.toString(result.tokens()));
        assertEquals(LiveCompletions.Kind.NO_COMPLETION, service.completions().endReasoning("test-live").kind());
    }

    @Test
    void anUnarmedCompletionRefusesTheControlByName() throws Exception {
        int[] prompt = service.encode("hello", true, true);
        // the context bounds the run: long enough to be seen running on the fast synthetic model
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(GenerationRequest.UNLIMITED)
                .sampler(SamplerConfig.greedyDecoding()).build();
        GenerationBatch batch = GenerationBatch.start(service,
                List.of(new GenerationBatch.Job(prompt, request, null, null, null)), null);
        service.completions().track("test-unarmed", batch);
        awaitRunning(batch);
        assertEquals(LiveCompletions.Kind.NOT_ARMED, service.completions().endReasoning("test-unarmed").kind());
        batch.cancel();
        batch.awaitAll();
        assertEquals(LiveCompletions.Kind.NO_COMPLETION, service.completions().endReasoning("test-unarmed").kind());
    }

    /** Waits until the batch's first choice is registered and running. */
    private static void awaitRunning(GenerationBatch batch) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 10_000;
        while (System.currentTimeMillis() < deadline) {
            BatchSequence sequence = batch.sequence(0);
            if (sequence != null && sequence.state() == BatchSequence.State.RUNNING) {
                return;
            }
            Thread.onSpinWait();
        }
        throw new AssertionError("the generation never started");
    }

    /** Forces the reasoning end, waiting until the armed stage counts (it arms at prefill). */
    private static boolean forceUntilForced(String id, long timeoutMillis) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (System.currentTimeMillis() < deadline) {
            // llama.cpp answers success while the stage is not counting yet; the count says it took effect
            if (service.completions().endReasoning(id).forced() > 0) {
                return true;
            }
            Thread.sleep(20);
        }
        return false;
    }

    /** Whether {@code seq} appears in {@code tokens} as a contiguous run. */
    private static boolean contains(int[] tokens, int[] seq) {
        if (seq.length == 0 || seq.length > tokens.length) {
            return false;
        }
        outer:
        for (int i = 0; i + seq.length <= tokens.length; i++) {
            for (int j = 0; j < seq.length; j++) {
                if (tokens[i + j] != seq[j]) {
                    continue outer;
                }
            }
            return true;
        }
        return false;
    }
}
