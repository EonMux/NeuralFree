package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The offline batch runner ({@code neuralfree run-batch}, OpenAI's Batch JSONL format) over the real service: input records of
 * {@code {"custom_id", "method", "url", "body"}} answer in input order with output records of
 * {@code {"id", "custom_id", "response", "error"}} whose bodies are what the HTTP endpoints answer. Generation
 * records share the engine waves; pooling records (embeddings, rerank, score) run between the waves; a refused
 * record writes its error record and the batch continues; a malformed envelope aborts the run naming its line;
 * the CLI checks the paths before the model loads; the command runs a batch from an http(s) input URL to an
 * http(s) output URL through the staging directory with the metrics server up.
 */
public class BatchRunnerTest {

    private static EngineService service;
    private static OpenAiResource resource;
    private static EngineService rank;
    private static PoolingResource rankPooling;
    private static Path model;
    private static EngineConfig config;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        config = EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(64).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config);
        resource = new OpenAiResource(service);
        Path rankModel = SyntheticModels.rank("neg", "pos").writeTempFile();
        rank = new EngineService(rankModel.toString(), ServerTestSupport.config().build());
        rankPooling = new PoolingResource(rank);
    }

    @AfterAll
    static void close() {
        service.close();
        rank.close();
    }

    private static void write(Path path, String... lines) throws IOException {
        Files.writeString(path, String.join("\n", lines) + "\n");
    }

    /** A view of a raw JSON member value. */
    private static JsonView of(Object raw) {
        return JsonView.of(raw, "value");
    }

    private static List<JsonView> records(Path output) throws IOException {
        List<JsonView> out = new ArrayList<>();
        for (String line : Files.readAllLines(output)) {
            if (!line.isBlank()) {
                out.add(ServerTestSupport.view(line));
            }
        }
        return out;
    }

    private static void assertOk(JsonView record, String customId, String object) {
        assertEquals(customId, record.str("custom_id"));
        assertTrue(record.str("id").startsWith("batch_req-"), "the output record carries a batch id");
        assertNull(record.raw("error"), customId + " succeeded");
        JsonView response = of(record.raw("response"));
        assertEquals(200, response.i32("status_code"));
        assertTrue(response.str("request_id").startsWith("req-"));
        assertEquals(object, of(response.raw("body")).str("object"));
    }

    private static void assertError(JsonView record, String customId, int status) {
        assertEquals(customId, record.str("custom_id"));
        JsonView response = of(record.raw("response"));
        assertEquals(status, response.i32("status_code"));
        assertNull(response.raw("body"));
        JsonView error = of(record.raw("error"));
        assertNotNull(error.str("message"));
        assertNotNull(error.str("type"));
    }

    @TempDir
    Path dir;

    /** {@code best_of} records run their candidates in the shared wave and answer their best {@code n}; the
     *  records after them still read their own slots. */
    @Test
    void bestOfRecordsAnswerTheirBestNAndKeepLaterSlotsAligned() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        String plain = "{\"prompt\": \"hello\", \"max_tokens\": 3, \"temperature\": 0}";
        write(input,
                "{\"custom_id\": \"b0\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                        + " \"body\": {\"prompt\": \"pick\", \"n\": 1, \"best_of\": 3, \"max_tokens\": 2,"
                        + " \"temperature\": 0.9, \"seed\": 3}}",
                "{\"custom_id\": \"b1\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                        + " \"body\": {\"messages\": [{\"role\": \"user\", \"content\": \"pick\"}], \"n\": 1,"
                        + " \"best_of\": 2, \"max_tokens\": 2, \"temperature\": 0.9, \"seed\": 3}}",
                "{\"custom_id\": \"p\", \"method\": \"POST\", \"url\": \"/v1/completions\", \"body\": " + plain + "}");
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        assertEquals(3, summary.succeeded());
        List<JsonView> out = records(output);
        assertOk(out.get(0), "b0", "text_completion");
        assertOk(out.get(1), "b1", "chat.completion");
        for (int i = 0; i < 2; i++) {
            JsonView body = of(of(out.get(i).raw("response")).raw("body"));
            assertEquals(1, ((List<?>) body.raw("choices")).size(), "best_of answers its best n alone");
            assertNull(of(((List<?>) body.raw("choices")).get(0)).raw("logprobs"),
                    "the ranking capture stays out of the body");
        }
        JsonView later = of(of(out.get(2).raw("response")).raw("body"));
        JsonView direct = ServerTestSupport.view(resource.completions(plain).getEntity());
        assertEquals(of(((List<?>) direct.raw("choices")).get(0)).str("text"),
                of(((List<?>) later.raw("choices")).get(0)).str("text"),
                "the record after the candidates answers its own generation");
    }

    @Test
    void recordsAnswerInInputOrderWithEndpointBodies() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        write(input,
                "{\"custom_id\": \"c2\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                        + " \"body\": {\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                        + " \"max_tokens\": 4, \"temperature\": 0}}",
                "{\"custom_id\": \"c0\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                        + " \"body\": {\"prompt\": \"hello\", \"max_tokens\": 4, \"temperature\": 0}}",
                "{\"custom_id\": \"c1\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                        + " \"body\": {\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}],"
                        + " \"max_tokens\": 4, \"temperature\": 0, \"n\": 2}}");
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        assertEquals(3, summary.requests());
        assertEquals(3, summary.succeeded());
        assertEquals(0, summary.failed());
        List<JsonView> out = records(output);
        assertEquals(3, out.size(), "one output record per input record");
        assertEquals(List.of("c2", "c0", "c1"), out.stream().map(r -> r.str("custom_id")).toList(),
                "the outputs keep the input order");
        assertOk(out.get(0), "c2", "chat.completion");
        assertOk(out.get(1), "c0", "text_completion");
        assertOk(out.get(2), "c1", "chat.completion");
        assertEquals(2, ServerTestSupport.size(of(of(out.get(2).raw("response")).raw("body")), "choices"),
                "a record's n choices still apply");
    }

    @Test
    void aRecordBodyIsWhatTheEndpointAnswers() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 4,"
                + " \"temperature\": 0}";
        write(input, "{\"custom_id\": \"chat\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                + " \"body\": " + body + "}");
        BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        JsonView batchBody = of(of(records(output).get(0).raw("response")).raw("body"));
        JsonView endpointBody = ServerTestSupport.view(resource.chatCompletions(body).getEntity());
        assertEquals(ServerTestSupport.at(endpointBody, "choices", 0).raw("message"),
                ServerTestSupport.at(batchBody, "choices", 0).raw("message"),
                "the runner's body is the endpoint's body");
        JsonView batchUsage = of(batchBody.raw("usage"));
        JsonView endpointUsage = of(endpointBody.raw("usage"));
        assertEquals(endpointUsage.i64("prompt_tokens"), batchUsage.i64("prompt_tokens"));
        assertEquals(endpointUsage.i64("completion_tokens"), batchUsage.i64("completion_tokens"));
    }

    @Test
    void refusedRecordsWriteErrorsAndTheBatchContinues() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        write(input,
                "{\"custom_id\": \"ok\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                        + " \"body\": {\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                        + " \"max_tokens\": 2, \"temperature\": 0}}",
                "{\"custom_id\": \"noprompt\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                        + " \"body\": {\"max_tokens\": 2}}",
                "{\"custom_id\": \"stream\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                        + " \"body\": {\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                        + " \"max_tokens\": 2, \"stream\": true}}",
                "{\"custom_id\": \"url\", \"method\": \"POST\", \"url\": \"/v1/unknown\","
                        + " \"body\": {}}",
                "{\"custom_id\": \"get\", \"method\": \"GET\", \"url\": \"/v1/completions\","
                        + " \"body\": {\"prompt\": \"hello\"}}",
                "");
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        assertEquals(5, summary.requests(), "blank lines are skipped");
        assertEquals(1, summary.succeeded());
        assertEquals(4, summary.failed());
        List<JsonView> out = records(output);
        assertEquals(5, out.size());
        assertOk(out.get(0), "ok", "chat.completion");
        assertError(out.get(1), "noprompt", 400);
        assertError(out.get(2), "stream", 400);
        assertError(out.get(3), "url", 400);
        assertError(out.get(4), "get", 400);
        JsonView error = of(out.get(3).raw("error"));
        assertTrue(error.str("message").contains("not a supported endpoint"),
                "an unknown URL refuses naming the endpoints: " + error.str("message"));
        assertEquals("method must be POST, got GET",
                of(out.get(4).raw("error")).str("message"));
    }

    @Test
    void aPriorityTheFcfsPolicyRefusesFailsItsRecordAlone() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        write(input,
                "{\"custom_id\": \"urgent\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                        + " \"body\": {\"prompt\": \"hello\", \"max_tokens\": 2, \"priority\": -1}}",
                "{\"custom_id\": \"plain\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                        + " \"body\": {\"prompt\": \"hello\", \"max_tokens\": 2, \"priority\": 0}}");
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        assertEquals(1, summary.succeeded());
        assertEquals(1, summary.failed());
        List<JsonView> out = records(output);
        assertError(out.get(0), "urgent", 400);
        String message = of(out.get(0).raw("error")).str("message");
        assertTrue(message.startsWith("priority -1"), message);
        assertOk(out.get(1), "plain", "text_completion");
    }

    @Test
    void poolingRecordsRouteThroughTheirEndpoints() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        write(input,
                "{\"custom_id\": \"embed\", \"method\": \"POST\", \"url\": \"/v1/embeddings\","
                        + " \"body\": {\"input\": \"hello\"}}",
                "{\"custom_id\": \"rerank\", \"method\": \"POST\", \"url\": \"/v1/rerank\","
                        + " \"body\": {\"query\": \"q\", \"documents\": [\"d\"]}}");
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        assertEquals(2, summary.requests());
        assertEquals(0, summary.succeeded());
        assertEquals(2, summary.failed());
        List<JsonView> out = records(output);
        assertError(out.get(0), "embed", 400);
        assertError(out.get(1), "rerank", 400);
        assertTrue(of(out.get(0).raw("error")).str("message").contains("ooling")
                        || of(out.get(0).raw("error")).str("message").contains("embed"),
                "the generative model refuses embeddings by name");
    }

    @Test
    void scoreRecordsRouteThroughTheScoreEndpoint() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        String body = "{\"queries\": \"question\", \"documents\": [\"answer one\", \"answer two\"]}";
        write(input,
                "{\"custom_id\": \"score\", \"method\": \"POST\", \"url\": \"/score\", \"body\": " + body + "}",
                "{\"custom_id\": \"score-v1\", \"method\": \"POST\", \"url\": \"/v1/score\", \"body\": "
                        + body + "}");
        BatchRunner.Summary summary = BatchRunner.run(rank, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        assertEquals(2, summary.requests());
        assertEquals(2, summary.succeeded());
        assertEquals(0, summary.failed());
        List<JsonView> out = records(output);
        assertOk(out.get(0), "score", "list");
        assertOk(out.get(1), "score-v1", "list");
        JsonView endpoint = ServerTestSupport.view(rankPooling.score(body).getEntity());
        for (int i = 0; i < out.size(); i++) {
            JsonView batchBody = of(of(out.get(i).raw("response")).raw("body"));
            assertEquals(endpoint.raw("data"), batchBody.raw("data"),
                    "the record's body is the score endpoint's body");
            assertEquals(endpoint.obj("usage").i64("total_tokens"),
                    batchBody.obj("usage").i64("total_tokens"));
        }
        double scored = ServerTestSupport.at(endpoint, "data", 0).f64("score");
        double reranked = ServerTestSupport.at(ServerTestSupport.view(rankPooling.rerank(
                        "{\"query\": \"question\", \"documents\": [\"answer one\"]}").getEntity()),
                "results", 0).f64("relevance_score");
        assertEquals(reranked, scored, 1e-9, "the batch score is the cross-encoder's rerank score");
    }

    @Test
    void scoreRoutesBySuffixAndRunsInlineLikeThePoolingRecords() {
        assertEquals("score", BatchRunner.route("/score"));
        assertEquals("score", BatchRunner.route("/v1/score"));
        assertEquals("score", BatchRunner.route("/v2/score"), "an unversioned endpoint matches by suffix");
        assertFalse(BatchRunner.isGeneration("score"), "score runs between the waves, not on them");
        assertTrue(BatchRunner.isGeneration("chat"));
    }

    @Test
    void aScoreRecordOnAGenerativeModelWritesItsErrorRecord() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        write(input,
                "{\"custom_id\": \"score\", \"method\": \"POST\", \"url\": \"/v1/score\","
                        + " \"body\": {\"queries\": \"q\", \"documents\": [\"d\"]}}",
                "{\"custom_id\": \"chat\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                        + " \"body\": {\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                        + " \"max_tokens\": 2, \"temperature\": 0}}");
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        assertEquals(2, summary.requests());
        assertEquals(1, summary.succeeded());
        assertEquals(1, summary.failed());
        List<JsonView> out = records(output);
        assertError(out.get(0), "score", 400);
        assertOk(out.get(1), "chat", "chat.completion");
        JsonView error = of(out.get(0).raw("error"));
        assertTrue(error.str("message").contains("scoring"),
                "the refusal names scoring: " + error.str("message"));
    }

    @Test
    void audioRecordsWithoutAnAudioProjectorAreRefusedAndTheBatchContinues() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        write(input,
                "{\"custom_id\": \"transcribe\", \"method\": \"POST\", \"url\": \"/v1/audio/transcriptions\","
                        + " \"body\": {\"file_url\": \"data:audio/wav;base64,AAAA\"}}",
                "{\"custom_id\": \"translate\", \"method\": \"POST\", \"url\": \"/v1/audio/translations\","
                        + " \"body\": {\"file_url\": \"https://example.com/audio.wav\"}}",
                "{\"custom_id\": \"chat\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                        + " \"body\": {\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                        + " \"max_tokens\": 2, \"temperature\": 0}}");
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        assertEquals(3, summary.requests());
        assertEquals(1, summary.succeeded());
        assertEquals(2, summary.failed());
        List<JsonView> out = records(output);
        assertError(out.get(0), "transcribe", 400);
        assertError(out.get(1), "translate", 400);
        assertOk(out.get(2), "chat", "chat.completion");
        JsonView error = of(out.get(0).raw("error"));
        assertTrue(error.str("message").contains("audio input is not supported"), error.str("message"));
    }

    @Test
    void smallWavesKeepTheInputOrder() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        String record = "{\"custom_id\": \"c%d\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                + " \"body\": {\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"max_tokens\": 2, \"temperature\": 0}}";
        write(input, record.formatted(0), record.formatted(1), record.formatted(2));
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, 1);
        assertEquals(3, summary.succeeded());
        assertEquals(List.of("c0", "c1", "c2"), records(output).stream()
                .map(r -> r.str("custom_id")).toList(), "waves of one record still answer in input order");
    }

    @Test
    void echoTextSurvivesTheSharedWaves() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        write(input, "{\"custom_id\": \"echo\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                + " \"body\": {\"prompt\": \"hello\", \"max_tokens\": 2, \"temperature\": 0, \"echo\": true}}");
        BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        JsonView choice = ServerTestSupport.at(of(of(records(output).get(0).raw("response")).raw("body")),
                "choices", 0);
        assertTrue(choice.str("text").startsWith("hello"), "the echoed prompt leads the text");
    }

    @Test
    void vllmExtrasShapeTheRecordBodies() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        write(input,
                "{\"custom_id\": \"cmpl\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                        + " \"body\": {\"prompt\": \"hello there\", \"max_tokens\": 2, \"temperature\": 0,"
                        + " \"return_token_ids\": true, \"prompt_logprobs\": 0, \"logprobs\": 1,"
                        + " \"return_tokens_as_token_ids\": true}}",
                "{\"custom_id\": \"chat\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\","
                        + " \"body\": {\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                        + " \"max_tokens\": 2, \"temperature\": 0, \"return_token_ids\": true}}");
        BatchRunner.run(service, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        List<JsonView> out = records(output);
        JsonView completion = of(of(out.get(0).raw("response")).raw("body"));
        JsonView choice = ServerTestSupport.at(completion, "choices", 0);
        List<?> promptIds = (List<?>) choice.raw("prompt_token_ids");
        assertEquals(completion.obj("usage").i64("prompt_tokens"), promptIds.size());
        List<?> promptLogprobs = (List<?>) choice.raw("prompt_logprobs");
        assertEquals(promptIds.size(), promptLogprobs.size(), "one prompt_logprobs entry per prompt token");
        assertNull(promptLogprobs.get(0), "no row predicts the first prompt token");
        for (Object t : (List<?>) choice.obj("logprobs").raw("tokens")) {
            assertTrue(String.valueOf(t).startsWith("token_id:"), "the record honors return_tokens_as_token_ids");
        }
        assertNotNull(completion.obj("usage").obj("prompt_tokens_details").raw("cached_tokens"));
        JsonView chat = of(of(out.get(1).raw("response")).raw("body"));
        assertEquals(chat.obj("usage").i64("prompt_tokens"), ((List<?>) chat.raw("prompt_token_ids")).size());
        assertEquals(chat.obj("usage").i64("completion_tokens"),
                ((List<?>) ServerTestSupport.at(chat, "choices", 0).raw("token_ids")).size());
    }

    @Test
    void aMalformedEnvelopeAbortsTheRunNamingItsLine() throws IOException {
        Path input = dir.resolve("in.jsonl");
        write(input,
                "{\"custom_id\": \"ok\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                        + " \"body\": {\"prompt\": \"hello\"}}",
                "{\"method\": \"POST\", \"url\": \"/v1/completions\", \"body\": {\"prompt\": \"hello\"}}");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> BatchRunner.validate(input));
        assertTrue(e.getMessage().contains("line 2"), "the refusal names the line: " + e.getMessage());
        write(input, "[]");
        e = assertThrows(IllegalArgumentException.class, () -> BatchRunner.validate(input));
        assertTrue(e.getMessage().contains("line 1"), e.getMessage());
        write(input, "{\"custom_id\": \"c\", \"method\": \"POST\", \"url\": \"/v1/completions\"}");
        e = assertThrows(IllegalArgumentException.class, () -> BatchRunner.validate(input));
        assertTrue(e.getMessage().contains("body"), e.getMessage());
    }

    @Test
    void theCommandLineChecksItsArgumentsBeforeTheModelLoads() throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        write(input, "{\"custom_id\": \"c\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                + " \"body\": {\"prompt\": \"hello\"}}");
        assertNull(BatchRunner.parseArgs(new String[]{"--help"}), "--help prints and asks for nothing else");
        assertEquals(BatchRunner.DEFAULT_MAX_INFLIGHT,
                BatchRunner.parseArgs(new String[]{"-i", input.toString(), "-o", output.toString(),
                        "-m", "m.gguf"}).maxInflight(), "the wave size defaults to the reference floor");
        assertEquals(7, BatchRunner.parseArgs(new String[]{"-i", input.toString(), "-o", output.toString(),
                "-m", "m.gguf", "--max-inflight", "7"}).maxInflight());
        for (String[] args : new String[][]{
                {"-o", output.toString(), "-m", "m.gguf"},
                {"-i", input.toString(), "-m", "m.gguf"},
                {"-i", input.toString(), "-o", output.toString()},
                {"-i", input.toString(), "-o", output.toString(), "-m", "m.gguf", "--max-inflight", "0"}}) {
            BatchRunner.Args parsed = BatchRunner.parseArgs(args);
            assertThrows(Exception.class, () -> BatchRunner.checkArgs(parsed),
                    "the argument set " + String.join(" ", List.of(args)) + " refuses");
        }
        BatchRunner.Args same = BatchRunner.parseArgs(new String[]{"-i", input.toString(), "-o", input.toString(),
                "-m", "m.gguf"});
        IOException e = assertThrows(IOException.class, () -> BatchRunner.checkArgs(same));
        assertTrue(e.getMessage().contains("the same file"), e.getMessage());
    }

    /** Two generation records of the remote batch tests. */
    private static final String REMOTE_RECORDS = "{\"custom_id\": \"r0\", \"method\": \"POST\", \"url\":"
            + " \"/v1/completions\", \"body\": {\"prompt\": \"hello\", \"max_tokens\": 3, \"temperature\": 0}}\n"
            + "{\"custom_id\": \"r1\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\":"
            + " {\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 3, \"temperature\": 0}}\n";

    /** Transfers that retry at once. */
    private static BatchFiles quickFiles() {
        return new BatchFiles(java.net.http.HttpClient.newHttpClient(), 2, java.time.Duration.ZERO,
                java.time.Duration.ofSeconds(10), java.time.Duration.ofSeconds(10));
    }

    /** Opens a fresh engine over the test model, as the command opens one over its {@code -m}. */
    private static EngineService openEngine(String path) throws IOException {
        assertEquals(model.toString(), path, "the command opens the model it was given");
        return new EngineService(path, config);
    }

    @Test
    void theCommandRunsAUrlInputIntoAUrlOutputThroughTheStagingDirectory() throws IOException {
        Path staging = Files.createDirectory(dir.resolve("staging"));
        try (BatchHttpFixture http = new BatchHttpFixture()) {
            http.on("/in.jsonl", BatchHttpFixture.Reply.ok(REMOTE_RECORDS))
                    .on("/out.jsonl", BatchHttpFixture.Reply.ok(""));
            BatchRunner.Args args = BatchRunner.parseArgs(new String[]{"-i", http.url("/in.jsonl"),
                    "-o", http.url("/out.jsonl"), "-m", model.toString(), "--output-tmp-dir", staging.toString(),
                    "--enable-metrics", "--port", "0"});
            BatchRunner.Summary summary = BatchRunner.execute(args, quickFiles(), BatchRunnerTest::openEngine);
            assertEquals(2, summary.requests());
            assertEquals(2, summary.succeeded());
            List<BatchHttpFixture.Received> received = http.received();
            assertEquals(List.of("GET", "PUT"), received.stream().map(BatchHttpFixture.Received::method).toList());
            Path uploaded = dir.resolve("uploaded.jsonl");
            Files.writeString(uploaded, received.get(1).body());
            List<JsonView> out = records(uploaded);
            assertEquals(2, out.size(), "the uploaded file holds every record");
            assertOk(out.get(0), "r0", "text_completion");
            assertOk(out.get(1), "r1", "chat.completion");
        }
        try (java.util.stream.Stream<Path> left = Files.list(staging)) {
            assertEquals(0, left.count(), "the download and the staged output are removed");
        }
    }

    @Test
    void aFailedUploadKeepsTheFinishedOutputInTheStagingDirectory() throws IOException {
        Path staging = Files.createDirectory(dir.resolve("staging"));
        Path input = dir.resolve("in.jsonl");
        Files.writeString(input, REMOTE_RECORDS);
        try (BatchHttpFixture http = new BatchHttpFixture()) {
            http.on("/out.jsonl", BatchHttpFixture.Reply.status(502));
            BatchRunner.Args args = BatchRunner.parseArgs(new String[]{"-i", input.toString(),
                    "-o", http.url("/out.jsonl"), "-m", model.toString(), "--output-tmp-dir", staging.toString()});
            IOException e = assertThrows(IOException.class,
                    () -> BatchRunner.execute(args, quickFiles(), BatchRunnerTest::openEngine));
            assertTrue(e.getMessage().contains("502"), e.getMessage());
            assertEquals(2, http.received().size(), "the upload was retried");
        }
        List<Path> kept;
        try (java.util.stream.Stream<Path> left = Files.list(staging)) {
            kept = left.toList();
        }
        assertEquals(1, kept.size(), "the staged output stays");
        assertEquals(2, records(kept.get(0)).size());
    }

    @Test
    void anUnreachableInputUrlFailsBeforeTheModelLoads() throws IOException {
        int port;
        try (java.net.ServerSocket socket = new java.net.ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        BatchRunner.Args args = BatchRunner.parseArgs(new String[]{"-i", "http://127.0.0.1:" + port + "/in.jsonl",
                "-o", dir.resolve("out.jsonl").toString(), "-m", model.toString()});
        IOException e = assertThrows(IOException.class, () -> BatchRunner.execute(args, quickFiles(),
                path -> {
                    throw new AssertionError("the engine must not open when the input cannot be fetched");
                }));
        assertTrue(e.getMessage().contains("download of --input-file"), e.getMessage());
    }

    @Test
    void theRemoteAndMetricsOptionsParseAsTheReferenceSpellsThem() {
        BatchRunner.Args defaults = BatchRunner.parseArgs(new String[]{"-i", "in.jsonl", "-o", "out.jsonl"});
        assertFalse(defaults.input().isUrl());
        assertFalse(defaults.enableMetrics());
        assertEquals(BatchRunner.DEFAULT_METRICS_HOST, defaults.metricsHost());
        assertEquals(BatchRunner.DEFAULT_METRICS_PORT, defaults.metricsPort());
        assertNull(defaults.outputTmpDir());

        BatchRunner.Args remote = BatchRunner.parseArgs(new String[]{"--input-file=https://h/in.jsonl",
                "--output-file", "http://h:9/out.jsonl", "--output-tmp-dir=/tmp/stage", "--enable-metrics",
                "--port=9100", "--host", "0.0.0.0"});
        assertTrue(remote.input().isUrl());
        assertTrue(remote.output().isUrl());
        assertEquals(Path.of("/tmp/stage"), remote.outputTmpDir());
        assertTrue(remote.enableMetrics());
        assertEquals(9100, remote.metricsPort());
        assertEquals("0.0.0.0", remote.metricsHost());

        assertEquals("10.0.0.1", BatchRunner.parseArgs(new String[]{"-i", "a", "-o", "b", "--enable-metrics",
                "--url", "10.0.0.1"}).metricsHost(), "--url is the deprecated spelling of --host");
        assertEquals("10.0.0.2", BatchRunner.parseArgs(new String[]{"-i", "a", "-o", "b", "--enable-metrics",
                "--url", "10.0.0.1", "--host", "10.0.0.2"}).metricsHost(), "--host wins over --url");
    }

    @Test
    void unsupportedRemoteAndMetricsOptionsAreRefusedByName() {
        String[][] refused = {
                {"-i", "ftp://h/in.jsonl", "-o", "out.jsonl"},
                {"-i", "in.jsonl", "-o", "s3://bucket/out.jsonl"},
                {"-i", "in.jsonl", "-o", "out.jsonl", "--port", "9000"},
                {"-i", "in.jsonl", "-o", "out.jsonl", "--host", "0.0.0.0"},
                {"-i", "in.jsonl", "-o", "out.jsonl", "--enable-metrics", "--port", "70000"},
                {"-i", "in.jsonl", "-o", "out.jsonl", "--enable-metrics=yes"},
                {"-i", "in.jsonl", "-o", "out.jsonl", "--output-tmp-dir"},
                {"-i", "in.jsonl", "-o", "out.jsonl", "--no-such-option"}};
        String[] named = {"--input-file", "--output-file", "--enable-metrics", "--enable-metrics", "--port",
                "--enable-metrics", "--output-tmp-dir", "--no-such-option"};
        for (int i = 0; i < refused.length; i++) {
            String[] args = refused[i];
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> BatchRunner.parseArgs(args), String.join(" ", args));
            assertTrue(e.getMessage().contains(named[i]), e.getMessage());
        }
    }

    @Test
    void aUrlOutputSkipsTheLocalChecksAndTheStagingDirectoryMustBeADirectory() throws IOException {
        Path input = dir.resolve("in.jsonl");
        write(input, "{\"custom_id\": \"c\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                + " \"body\": {\"prompt\": \"hello\"}}");
        BatchRunner.checkArgs(BatchRunner.parseArgs(new String[]{"-i", input.toString(),
                "-o", "http://127.0.0.1:1/missing-dir/out.jsonl", "-m", "m.gguf"}));
        BatchRunner.checkArgs(BatchRunner.parseArgs(new String[]{"-i", "http://127.0.0.1:1/in.jsonl",
                "-o", dir.resolve("out.jsonl").toString(), "-m", "m.gguf"}));
        BatchRunner.Args notADirectory = BatchRunner.parseArgs(new String[]{"-i", input.toString(),
                "-o", dir.resolve("out.jsonl").toString(), "-m", "m.gguf", "--output-tmp-dir", input.toString()});
        IOException e = assertThrows(IOException.class, () -> BatchRunner.checkArgs(notADirectory));
        assertTrue(e.getMessage().contains("--output-tmp-dir"), e.getMessage());
    }

    @Test
    void theLauncherRunsThisClass() {
        assertEquals(BatchRunner.class.getName(), com.eonmux.neuralfree.cli.ServeCommand.BATCH_RUNNER_CLASS,
                "neuralfree run-batch launches the batch runner by this name");
    }
}
