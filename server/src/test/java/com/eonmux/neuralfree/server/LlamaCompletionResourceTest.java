package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** llama.cpp's native {@code /completion}: its prompt forms, body fields, stream framing and refusals. */
class LlamaCompletionResourceTest {

    private static EngineService service;
    private static EngineService fim;
    private static LlamaCompletionResource llama;
    private static LlamaCompletionResource infill;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2));
        llama = new LlamaCompletionResource(service);
        Path fimModel = SyntheticModels.llama().u32("tokenizer.ggml.fim_pre_token_id", 10)
                .u32("tokenizer.ggml.fim_suf_token_id", 11).u32("tokenizer.ggml.fim_mid_token_id", 12)
                .writeTempFile();
        fim = new EngineService(fimModel.toString(), ServerTestSupport.pagedConfig(512, 2));
        infill = new LlamaCompletionResource(fim);
    }

    @AfterAll
    static void close() {
        service.close();
        fim.close();
    }

    @Test
    void infillAnswersOverTheFimPrompt() {
        JsonView body = view(infill.infill("""
                {"input_prefix": "def f(", "input_suffix": "):\\n", "prompt": "x",
                 "input_extra": [{"filename": "a.py", "text": "y = 1"}], "n_predict": 3, "ignore_eos": true}"""
        ).getEntity());
        assertEquals(3L, body.i64("tokens_predicted"));
        int[] expected = fim.infillPrompt(fim.encode("def f(", false, false), fim.encode("):\n", false, false),
                List.of(new com.eonmux.neuralfree.gguf.engine.InfillPrompt.Chunk("a.py", "y = 1")),
                fim.encode("x", false, true), 3);
        assertEquals(expected.length, body.i64("tokens_evaluated"));
        assertEquals(12, expected[expected.length - 1], "the prompt ends at the FIM middle token");
    }

    @Test
    void infillRefusesAModelWithoutFimTokensAndMalformedInput() {
        assertEquals(501, statusOf(() -> llama.infill("{\"input_prefix\": \"a\", \"input_suffix\": \"b\"}")));
        for (String body : new String[]{
                "{\"input_suffix\": \"b\"}",
                "{\"input_prefix\": \"a\"}",
                "{\"input_prefix\": \"a\", \"input_suffix\": \"b\", \"prompt\": [1]}",
                "{\"input_prefix\": \"a\", \"input_suffix\": \"b\", \"input_extra\": {}}",
                "{\"input_prefix\": \"a\", \"input_suffix\": \"b\", \"input_extra\": [{\"filename\": \"f\"}]}",
                "{\"input_prefix\": \"a\", \"input_suffix\": \"b\", \"input_extra\": [{\"text\": \"t\", \"filename\": 3}]}"}) {
            assertEquals(400, statusOf(() -> infill.infill(body)), body);
        }
    }

    @Test
    void aRepetitionStopReportsTheLimitStopType() {
        JsonView body = view(llama.completion("""
                {"prompt": "hello", "n_predict": 8, "temperature": 0, "allowed_token_ids": [20],
                 "repetition_detection": {"max_pattern_size": 1, "min_count": 4}}""").getEntity());
        assertTrue(body.bool("stop"));
        assertEquals("limit", body.str("stop_type"), "llama.cpp reports its heuristic early stops as limit");
        assertEquals(4L, body.i64("tokens_predicted"));
    }

    @Test
    void oneStringPromptAnswersTheNativeObject() {
        JsonView body = view(llama.completion("""
                {"prompt": "hello", "n_predict": 4, "ignore_eos": true, "temperature": 0,
                 "return_tokens": true}""").getEntity());
        assertTrue(body.bool("stop"));
        assertEquals("limit", body.str("stop_type"));
        assertEquals(4L, body.i64("tokens_predicted"));
        assertEquals(4, size(body, "tokens"));
        JsonView plain = view(llama.completion("""
                {"prompt": "hello", "n_predict": 4, "ignore_eos": true, "temperature": 0}""").getEntity());
        assertEquals(0, size(plain, "tokens"), "without return_tokens the final object lists no tokens");
        assertEquals(service.encode("hello", true, true).length, body.i64("tokens_evaluated"));
        assertEquals(4L, body.obj("generation_settings").i64("n_predict"));
        assertEquals(4L, body.obj("timings").i64("predicted_n"));
        assertEquals("", body.str("stopping_word"));
        assertTrue(body.i64("id_slot") >= 0);
    }

    @Test
    void anArrayWithoutTokenIdsIsSeveralPromptsAndAMixedArrayIsOne() {
        Object many = ServerTestSupport.parsed(llama.completion("""
                {"prompt": ["a", [5, 6, 7]], "n_predict": 2, "ignore_eos": true}""").getEntity());
        assertTrue(many instanceof List<?> list && list.size() == 2, "one object per prompt");
        int[] text = service.encode("hi", false, true);
        JsonView mixed = view(llama.completion("""
                {"prompt": [5, "hi", 6], "n_predict": 1, "ignore_eos": true}""").getEntity());
        assertEquals(2 + text.length, mixed.i64("tokens_evaluated"));
    }

    @Test
    void nProbsAndResponseFieldsShapeTheObject() {
        JsonView body = view(llama.completion("""
                {"prompt": "hello", "n_predict": 2, "ignore_eos": true, "n_probs": 3}""").getEntity());
        JsonView first = at(body, "completion_probabilities", 0);
        assertEquals(3, size(first, "top_logprobs"));
        assertTrue(first.has("id") && first.has("logprob") && first.has("bytes"));
        JsonView picked = view(llama.completion("""
                {"prompt": "hello", "n_predict": 2, "response_fields": ["content", "timings/predicted_n", "nope"]}"""
        ).getEntity());
        assertEquals(List.of("content", "timings/predicted_n"), List.copyOf(picked.keys()));
    }

    @Test
    void aStreamSendsPiecesThenTheFinalObjectWithoutDone() throws IOException {
        String text = streamText(llama.completion("""
                {"prompt": "hello", "n_predict": 3, "ignore_eos": true, "stream": true}"""));
        List<JsonView> frames = dataFrames(text);
        assertFalse(text.contains("[DONE]"));
        JsonView last = frames.get(frames.size() - 1);
        assertTrue(last.bool("stop"));
        assertEquals("", last.str("content"));
        long previous = 0;
        for (JsonView f : frames.subList(0, frames.size() - 1)) {
            assertFalse(f.bool("stop"));
            assertEquals(1, size(f, "tokens"), "one chunk per generated token");
            assertTrue(f.i64("tokens_predicted") > previous, "each chunk is a later token");
            previous = f.i64("tokens_predicted");
        }
        assertTrue(frames.size() >= 2 && previous <= 3);
        assertEquals(3L, last.i64("tokens_predicted"));
    }

    @Test
    void aDisaggregatedTransferRefusesOnTheLlamaCppRoutes() {
        String body = "{\"prompt\": \"x\", \"n_predict\": 1,"
                + " \"kv_transfer_params\": {\"do_remote_decode\": true}}";
        assertEquals(400, statusOf(() -> llama.completion(body)));
        assertTrue(errorOf(() -> llama.completion(body)).contains("kv_transfer_params"));
        assertEquals(400, statusOf(() -> infill.infill("{\"input_prefix\": \"a\", \"input_suffix\": \"b\","
                + " \"kv_transfer_params\": {\"do_remote_decode\": true}}")));
    }

    @Test
    void malformedRequestsRefuse() {
        for (String body : new String[]{
                "{\"n_predict\": 2}",
                "{\"prompt\": []}",
                "{\"prompt\": [1, 2.5]}",
                "{\"prompt\": [999999999]}",
                "{\"prompt\": {\"prompt_string\": \"x\", \"multimodal_data\": [\"AA==\"]}}",
                "{\"prompt\": \"x\", \"n_predict\": -2}",
                "{\"prompt\": \"x\", \"post_sampling_probs\": \"yes\"}",
                "{\"prompt\": \"x\", \"response_fields\": \"content\"}"}) {
            assertEquals(400, statusOf(() -> llama.completion(body)), body);
        }
    }
}
