package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferParams;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Disaggregated prefill/decode between two services: the prefill instance answers the transfer's parameters, the
 * decode instance forwards them under the same {@code transfer_id}, imports the chain into its prefix cache and
 * generates what an untransferred run generates — over the completions, scale-out, chat, Cohere and Anthropic
 * routes. The roles refuse each other's side by name, a missing transfer follows {@code kv_load_failure_policy},
 * the status route reports the connector, and a connector this build does not ship refuses the server's start.
 */
public class KvTransferServiceTest {

    private static final String PROMPT_TEXT = "The quick brown fox jumps over the lazy dog.";
    private static final int NEW_TOKENS = 6;

    private static Path model;
    private static EngineConfig config;
    private static EngineService prefill;
    private static EngineService decode;
    private static EngineService reference;
    private static OpenAiResource prefillApi;
    private static OpenAiResource decodeApi;
    private static OpenAiResource referenceApi;
    private static int[] prompt;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        config = pagedConfig(128, 2);
        prefill = service(producerJson());
        decode = service(consumerJson("fail"));
        reference = new EngineService(model.toString(), config);
        prefillApi = new OpenAiResource(prefill);
        decodeApi = new OpenAiResource(decode);
        referenceApi = new OpenAiResource(reference);
        prompt = prefill.tokenizer().encode(PROMPT_TEXT, true, true);
    }

    @AfterAll
    static void close() {
        prefill.close();
        decode.close();
        reference.close();
    }

    private static String producerJson() {
        return "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_producer\","
                + " \"engine_id\": \"prefill-1\", \"kv_ip\": \"127.0.0.1\", \"kv_port\": 0,"
                + " \"kv_buffer_size\": 268435456}";
    }

    private static String consumerJson(String loadFailurePolicy) {
        return "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_consumer\","
                + " \"engine_id\": \"decode-1\", \"kv_ip\": \"127.0.0.1\", \"kv_port\": 0,"
                + " \"kv_load_failure_policy\": \"" + loadFailurePolicy + "\"}";
    }

    private static EngineService service(String json) throws IOException {
        return new EngineService(model.toString(), config, ServerOptions.defaults()
                .withKvTransfer(KvTransferConfig.parse(json, "--kv-transfer-config")));
    }

    private static ReasoningFilter.Sink sink() {
        return ReasoningFilter.contentTo((tokenId, text) -> { });
    }

    private static GenerationRequest request(int newTokens, KvTransferParams transfer) {
        return GenerationRequest.builder().maxNewTokens(newTokens).sampler(SamplerConfig.greedyDecoding())
                .kvTransfer(transfer).build();
    }

    private static String prefillBody(String transferId) {
        return "{\"prompt\": \"" + PROMPT_TEXT + "\", \"max_tokens\": 4, \"temperature\": 0,"
                + " \"kv_transfer_params\": {\"do_remote_decode\": true, \"transfer_id\": \"" + transferId + "\"}}";
    }

    /** The prefill response's {@code kv_transfer_params}, as the router would forward them. */
    private static JsonView prefill(String transferId) {
        Response response = prefillApi.completions(prefillBody(transferId));
        return view(response.getEntity()).obj("kv_transfer_params");
    }

    private static String decodeBody(JsonView transfer, int newTokens, boolean stream) {
        return "{\"prompt\": \"" + PROMPT_TEXT + "\", \"max_tokens\": " + newTokens + ", \"temperature\": 0,"
                + " \"return_token_ids\": true" + (stream ? ", \"stream\": true" : "") + ","
                + " \"kv_transfer_params\": {\"do_remote_prefill\": true,"
                + " \"transfer_id\": \"" + transfer.str("transfer_id") + "\","
                + " \"remote_engine_id\": \"" + transfer.str("remote_engine_id") + "\","
                + " \"remote_host\": \"" + transfer.str("remote_host") + "\","
                + " \"remote_port\": " + transfer.i32("remote_port") + ","
                + " \"remote_num_tokens\": " + transfer.i32("remote_num_tokens") + "}}";
    }

    private static JsonView firstChoice(JsonView body) {
        return JsonView.of(((List<?>) body.raw("choices")).get(0), "choices[0]");
    }

    private static List<?> completionTokens(Response response) {
        return (List<?>) firstChoice(view(response.getEntity())).raw("token_ids");
    }

    private static int[] tokenIds(List<?> tokens) {
        int[] out = new int[tokens.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) tokens.get(i)).intValue();
        }
        return out;
    }

    @Test
    void aPrefillExportFeedsTheDecodeInstance() {
        JsonView transfer = prefill("http-1");
        assertTrue(transfer.bool("do_remote_prefill"));
        assertFalse(transfer.bool("do_remote_decode"));
        assertEquals("prefill-1", transfer.str("remote_engine_id"));
        assertEquals("127.0.0.1", transfer.str("remote_host"));
        assertTrue(transfer.i32("remote_port") > 0);
        assertEquals("http-1", transfer.str("transfer_id"));
        assertEquals(prompt.length, transfer.i32("remote_num_tokens"));

        List<?> expected = completionTokens(referenceApi.completions(
                "{\"prompt\": \"" + PROMPT_TEXT + "\", \"max_tokens\": " + NEW_TOKENS + ", \"temperature\": 0,"
                        + " \"return_token_ids\": true}"));
        Response decoded = decodeApi.completions(decodeBody(transfer, NEW_TOKENS, false));
        JsonView body = view(decoded.getEntity());
        assertArrayEquals(tokenIds(expected), tokenIds(completionTokens(decoded)),
                "the transferred rows decode exactly like an untransferred run");
        assertEquals(expected.size(), body.obj("usage").i32("completion_tokens"));
        assertEquals(prompt.length - 1, body.obj("usage").obj("prompt_tokens_details").i64("cached_tokens"),
                "the decode adopts the prefilled rows and computes the last prompt token");
        JsonView answer = body.obj("kv_transfer_params");
        assertFalse(answer.bool("do_remote_prefill"));
        assertFalse(answer.bool("do_remote_decode"));
        assertEquals("http-1", answer.str("transfer_id"));
    }

    @Test
    void theScaleOutRouteCarriesTheTransfer() {
        String promptIds = Arrays.toString(prompt);
        JsonView prefillBody = generate(prefill, "{\"token_ids\": " + promptIds + ","
                + " \"sampling_params\": {\"max_tokens\": 4, \"temperature\": 0},"
                + " \"output_mode\": \"tokens\", \"stream\": false,"
                + " \"kv_transfer_params\": {\"do_remote_decode\": true, \"transfer_id\": \"scale-1\"}}");
        JsonView transfer = prefillBody.obj("kv_transfer_params");
        assertEquals("scale-1", transfer.str("transfer_id"));
        assertEquals(prompt.length, transfer.i32("remote_num_tokens"));

        JsonView decoded = generate(decode, "{\"token_ids\": " + promptIds + ","
                + " \"sampling_params\": {\"max_tokens\": " + NEW_TOKENS + ", \"temperature\": 0},"
                + " \"output_mode\": \"tokens\", \"stream\": false, \"return_token_ids\": true,"
                + " \"kv_transfer_params\": {\"do_remote_prefill\": true, \"transfer_id\": \"scale-1\","
                + " \"remote_engine_id\": \"" + transfer.str("remote_engine_id") + "\","
                + " \"remote_host\": \"" + transfer.str("remote_host") + "\","
                + " \"remote_port\": " + transfer.i32("remote_port") + ","
                + " \"remote_num_tokens\": " + transfer.i32("remote_num_tokens") + "}}");
        JsonView expected = generate(reference, "{\"token_ids\": " + promptIds + ","
                + " \"sampling_params\": {\"max_tokens\": " + NEW_TOKENS + ", \"temperature\": 0},"
                + " \"output_mode\": \"tokens\", \"stream\": false, \"return_token_ids\": true}");
        assertEquals(((List<?>) firstChoice(expected).raw("token_ids")).size(),
                ((List<?>) firstChoice(decoded).raw("token_ids")).size());
        assertEquals(firstChoice(expected).raw("token_ids"), firstChoice(decoded).raw("token_ids"),
                "the scale-out route answers the transfer's tokens like an untransferred run");
        assertEquals(Boolean.FALSE, decoded.obj("kv_transfer_params").raw("do_remote_prefill"));
    }

    /** The scale-out generate response of {@code service} for {@code body}. */
    private static JsonView generate(EngineService service, String body) {
        return view(ScaleOutGenerate.generate(service, JsonView.of(Json.parse(body, Json.Limits.UNTRUSTED),
                "body"), null, new ConcurrentHashMap<>()).getEntity());
    }

    @Test
    void theChatRouteCarriesTheTransfer() {
        JsonView exported = view(prefillApi.chatCompletions("{\"model\": \"m\", \"messages\": [{\"role\":"
                + " \"user\", \"content\": \"" + PROMPT_TEXT + "\"}], \"max_tokens\": 4,"
                + " \"temperature\": 0, \"kv_transfer_params\": {\"do_remote_decode\": true,"
                + " \"transfer_id\": \"chat-1\"}}").getEntity()).obj("kv_transfer_params");
        assertEquals("chat-1", exported.str("transfer_id"));
        assertTrue(exported.i32("remote_num_tokens") > 0, "the rendered chat prompt's rows");

        String decodeRequest = "{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \""
                + PROMPT_TEXT + "\"}], \"max_tokens\": " + NEW_TOKENS + ", \"temperature\": 0,"
                + " \"kv_transfer_params\": {\"do_remote_prefill\": true, \"transfer_id\": \"chat-1\","
                + " \"remote_engine_id\": \"" + exported.str("remote_engine_id") + "\","
                + " \"remote_host\": \"" + exported.str("remote_host") + "\","
                + " \"remote_port\": " + exported.i32("remote_port") + ","
                + " \"remote_num_tokens\": " + exported.i32("remote_num_tokens") + "}}";
        JsonView decoded = view(decodeApi.chatCompletions(decodeRequest).getEntity());
        JsonView expected = view(referenceApi.chatCompletions("{\"model\": \"m\", \"messages\": [{\"role\":"
                + " \"user\", \"content\": \"" + PROMPT_TEXT + "\"}], \"max_tokens\": " + NEW_TOKENS
                + ", \"temperature\": 0}").getEntity());
        assertEquals(firstChoice(expected).obj("message").str("content"),
                firstChoice(decoded).obj("message").str("content"),
                "the chat route answers the transfer's text like an untransferred run");
        assertEquals(exported.i32("remote_num_tokens") - 1,
                decoded.obj("usage").obj("prompt_tokens_details").i64("cached_tokens"));
        assertEquals(Boolean.FALSE, decoded.obj("kv_transfer_params").raw("do_remote_prefill"));
    }

    @Test
    void theCohereRouteCarriesTheTransfer() {
        CohereChatResource prefillCohere = new CohereChatResource(prefill);
        CohereChatResource decodeCohere = new CohereChatResource(decode);
        CohereChatResource referenceCohere = new CohereChatResource(reference);
        JsonView exported = view(cohere(prefillCohere, "{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"" + PROMPT_TEXT + "\"}], \"max_tokens\": 4, \"temperature\": 0,"
                + " \"kv_transfer_params\": {\"do_remote_decode\": true, \"transfer_id\": \"cohere-1\"}}")
                .getEntity()).obj("kv_transfer_params");
        assertEquals("cohere-1", exported.str("transfer_id"));
        assertTrue(exported.i32("remote_num_tokens") > 0, "the rendered chat prompt's rows");

        String decodeRequest = "{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \""
                + PROMPT_TEXT + "\"}], \"max_tokens\": " + NEW_TOKENS + ", \"temperature\": 0,"
                + " \"kv_transfer_params\": {\"do_remote_prefill\": true, \"transfer_id\": \"cohere-1\","
                + " \"remote_engine_id\": \"" + exported.str("remote_engine_id") + "\","
                + " \"remote_host\": \"" + exported.str("remote_host") + "\","
                + " \"remote_port\": " + exported.i32("remote_port") + ","
                + " \"remote_num_tokens\": " + exported.i32("remote_num_tokens") + "}}";
        JsonView decoded = view(cohere(decodeCohere, decodeRequest).getEntity());
        JsonView expected = view(cohere(referenceCohere, "{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"" + PROMPT_TEXT + "\"}], \"max_tokens\": " + NEW_TOKENS
                + ", \"temperature\": 0}").getEntity());
        assertEquals(cohereText(expected), cohereText(decoded),
                "the Cohere route answers the transfer's text like an untransferred run");
        assertEquals(exported.i32("remote_num_tokens") - 1, decoded.obj("usage").i64("cached_tokens"));
        assertEquals(Boolean.FALSE, decoded.obj("kv_transfer_params").raw("do_remote_prefill"));
    }

    @Test
    void theAnthropicRouteCarriesTheTransfer() {
        AnthropicResource prefillAnthropic = new AnthropicResource(prefill);
        AnthropicResource decodeAnthropic = new AnthropicResource(decode);
        AnthropicResource referenceAnthropic = new AnthropicResource(reference);
        JsonView exported = view(anthropic(prefillAnthropic, "{\"model\": \"m\", \"max_tokens\": 4,"
                + " \"temperature\": 0, \"messages\": [{\"role\": \"user\", \"content\": \"" + PROMPT_TEXT
                + "\"}], \"kv_transfer_params\": {\"do_remote_decode\": true, \"transfer_id\": \"anthropic-1\"}}")
                .getEntity()).obj("kv_transfer_params");
        assertEquals("anthropic-1", exported.str("transfer_id"));
        assertTrue(exported.i32("remote_num_tokens") > 0, "the rendered chat prompt's rows");

        JsonView decoded = view(anthropic(decodeAnthropic, anthropicDecodeBody(exported, "anthropic-1",
                NEW_TOKENS, false)).getEntity());
        JsonView expected = view(anthropic(referenceAnthropic, "{\"model\": \"m\", \"max_tokens\": "
                + NEW_TOKENS + ", \"temperature\": 0, \"messages\": [{\"role\": \"user\", \"content\": \""
                + PROMPT_TEXT + "\"}]}").getEntity());
        assertEquals(anthropicText(expected), anthropicText(decoded),
                "the Anthropic route answers the transfer's text like an untransferred run");
        assertEquals(Boolean.FALSE, decoded.obj("kv_transfer_params").raw("do_remote_prefill"));
    }

    @Test
    void anAnthropicStreamAnswersTheTransferOnItsMessageDelta() throws IOException {
        JsonView transfer = view(anthropic(new AnthropicResource(prefill), "{\"model\": \"m\", \"max_tokens\": 4,"
                + " \"temperature\": 0, \"messages\": [{\"role\": \"user\", \"content\": \"" + PROMPT_TEXT
                + "\"}], \"kv_transfer_params\": {\"do_remote_decode\": true, \"transfer_id\":"
                + " \"anthropic-stream-1\"}}").getEntity()).obj("kv_transfer_params");
        Response streamed = anthropic(new AnthropicResource(decode),
                anthropicDecodeBody(transfer, "anthropic-stream-1", NEW_TOKENS, true));
        JsonView delta = null;
        for (String line : streamText(streamed).split("\n")) {
            if (!line.startsWith("data: ")) {
                continue;
            }
            JsonView event = view(line.substring("data: ".length()));
            if ("message_delta".equals(event.str("type")) && event.raw("kv_transfer_params") != null) {
                delta = event.obj("kv_transfer_params");
            }
        }
        assertNotNull(delta, "the message_delta event carries kv_transfer_params");
        assertEquals(Boolean.FALSE, delta.bool("do_remote_prefill"));
        assertEquals("anthropic-stream-1", delta.str("transfer_id"));
    }

    /** The Anthropic decode request of an exported transfer, at {@code stream}. */
    private static String anthropicDecodeBody(JsonView transfer, String transferId, int newTokens, boolean stream) {
        return "{\"model\": \"m\", \"max_tokens\": " + newTokens + ", \"temperature\": 0"
                + (stream ? ", \"stream\": true" : "") + ", \"messages\": [{\"role\": \"user\", \"content\": \""
                + PROMPT_TEXT + "\"}], \"kv_transfer_params\": {\"do_remote_prefill\": true,"
                + " \"transfer_id\": \"" + transferId + "\","
                + " \"remote_engine_id\": \"" + transfer.str("remote_engine_id") + "\","
                + " \"remote_host\": \"" + transfer.str("remote_host") + "\","
                + " \"remote_port\": " + transfer.i32("remote_port") + ","
                + " \"remote_num_tokens\": " + transfer.i32("remote_num_tokens") + "}}";
    }

    /** An Anthropic call whose 400 refusal names its reason. */
    private static Response anthropic(AnthropicResource resource, String body) {
        try {
            return resource.messages(body);
        } catch (jakarta.ws.rs.WebApplicationException e) {
            throw new AssertionError(String.valueOf(e.getResponse().getEntity()), e);
        }
    }

    /** The text of an Anthropic message's {@code text} content blocks, joined. */
    private static String anthropicText(JsonView response) {
        Object raw = response.raw("content");
        StringBuilder out = new StringBuilder();
        if (raw instanceof List) {
            for (Object block : (List<?>) raw) {
                if (block instanceof Map && "text".equals(((Map<?, ?>) block).get("type"))) {
                    out.append(((Map<?, ?>) block).get("text"));
                }
            }
        }
        return out.toString();
    }

    /** A Cohere call whose 400 refusal names its reason. */
    private static Response cohere(CohereChatResource resource, String body) {
        try {
            return resource.chat(body);
        } catch (jakarta.ws.rs.WebApplicationException e) {
            throw new AssertionError(String.valueOf(e.getResponse().getEntity()), e);
        }
    }

    /** The text of a Cohere message's {@code text} content blocks, joined. */
    private static String cohereText(JsonView response) {
        Object raw = response.obj("message").raw("content");
        StringBuilder out = new StringBuilder();
        if (raw instanceof List) {
            for (Object block : (List<?>) raw) {
                if (block instanceof Map && "text".equals(((Map<?, ?>) block).get("type"))) {
                    out.append(((Map<?, ?>) block).get("text"));
                }
            }
        }
        return out.toString();
    }

    @Test
    void aStreamedDecodeAnswersTheTransferOnItsFinishChunk() throws IOException {
        JsonView transfer = prefill("stream-1");
        Response streamed = decodeApi.completions(decodeBody(transfer, NEW_TOKENS, true));
        JsonView finish = null;
        for (JsonView frame : dataFrames(streamText(streamed))) {
            if (!(frame.raw("choices") instanceof List) || ((List<?>) frame.raw("choices")).isEmpty()) {
                continue;
            }
            Object reason = firstChoice(frame).raw("finish_reason");
            if (reason != null && frame.raw("kv_transfer_params") != null) {
                finish = frame;
            }
        }
        assertNotNull(finish, "the finish chunk carries kv_transfer_params");
        JsonView answer = finish.obj("kv_transfer_params");
        assertFalse(answer.bool("do_remote_prefill"));
        assertEquals("stream-1", answer.str("transfer_id"));
    }

    @Test
    void eachRoleRefusesTheOtherSideByName() {
        IllegalArgumentException prefillOnDecode = assertThrows(IllegalArgumentException.class,
                () -> decode.submit(prompt, request(2, KvTransferParams.prefillRequest()), (TokenCallback) null));
        assertTrue(prefillOnDecode.getMessage().contains("kv_consumer"), prefillOnDecode.getMessage());

        IllegalArgumentException decodeOnPrefill = assertThrows(IllegalArgumentException.class,
                () -> prefill.submit(prompt, request(2, KvTransferParams.decodeRequest("prefill-1", null,
                        "127.0.0.1", 1, "t-role", 0)), (TokenCallback) null));
        assertTrue(decodeOnPrefill.getMessage().contains("kv_producer"), decodeOnPrefill.getMessage());
    }

    @Test
    void aRequestWithoutAConfiguredConnectorRefusesByName() {
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> reference.submit(prompt, request(2, KvTransferParams.prefillRequest()), (TokenCallback) null));
        assertTrue(refused.getMessage().contains("--kv-transfer-config"), refused.getMessage());
    }

    @Test
    void aMissingTransferFollowsTheLoadFailurePolicy() throws IOException {
        UncheckedIOException failed = assertThrows(UncheckedIOException.class,
                () -> decode.submit(prompt, request(NEW_TOKENS, KvTransferParams.decodeRequest("prefill-1", null,
                        "127.0.0.1", 1, "no-such-transfer", 0)), (TokenCallback) null));
        assertTrue(failed.getMessage().contains("no-such-transfer"), failed.getMessage());

        try (EngineService recompute = service(consumerJson("recompute"))) {
            EngineService.Submission submission = recompute.submit(prompt, request(NEW_TOKENS,
                    KvTransferParams.decodeRequest("prefill-1", null, "127.0.0.1", 1, "no-such-transfer", 0)),
                    sink(), null, null, null);
            assertTrue(submission.await(60, TimeUnit.SECONDS));
            assertNotNull(submission.sequence(), String.valueOf(submission.refused()));
            Map<String, Object> answer = submission.kvTransferParams();
            assertEquals(Boolean.FALSE, answer.get("do_remote_prefill"));
            assertEquals("no-such-transfer", answer.get("transfer_id"));
            assertArrayEquals(tokenIds(completionTokens(referenceApi.completions(
                            "{\"prompt\": \"" + PROMPT_TEXT + "\", \"max_tokens\": " + NEW_TOKENS
                                    + ", \"temperature\": 0, \"return_token_ids\": true}"))),
                    submission.sequence().result().tokens(), "the recompute prefills the prompt locally");
        } catch (InterruptedException e) {
            throw new AssertionError("the recompute did not finish", e);
        }
    }

    @Test
    void aRemoteHostOutsideTheAllowlistRefusesByNameUnderBothPolicies() throws IOException {
        GenerationRequest outside = request(NEW_TOKENS, KvTransferParams.decodeRequest("prefill-1", null,
                "10.9.8.7", 14579, "no-such-transfer", 0));
        IllegalArgumentException onFail = assertThrows(IllegalArgumentException.class,
                () -> decode.submit(prompt, outside, (TokenCallback) null));
        assertTrue(onFail.getMessage().contains("allowed_remote_hosts"), onFail.getMessage());

        try (EngineService recompute = service(consumerJson("recompute"))) {
            IllegalArgumentException onRecompute = assertThrows(IllegalArgumentException.class,
                    () -> recompute.submit(prompt, outside, sink(), null, null, null));
            assertTrue(onRecompute.getMessage().contains("allowed_remote_hosts"), onRecompute.getMessage());
        }
    }

    @Test
    void theHttpRouteRefusesARemoteHostOutsideTheAllowlist() {
        WebApplicationException refused = assertThrows(WebApplicationException.class,
                () -> decodeApi.completions("{\"prompt\": \"" + PROMPT_TEXT + "\","
                        + " \"max_tokens\": " + NEW_TOKENS + ", \"temperature\": 0,"
                        + " \"kv_transfer_params\": {\"do_remote_prefill\": true, \"transfer_id\": \"outside-1\","
                        + " \"remote_engine_id\": \"prefill-1\", \"remote_host\": \"10.9.8.7\","
                        + " \"remote_port\": 14579, \"remote_num_tokens\": 1}}"));
        assertEquals(400, refused.getResponse().getStatus());
        String message = view(refused.getResponse().getEntity()).obj("error").str("message");
        assertTrue(message.contains("allowed_remote_hosts"), message);
    }

    @Test
    void aServerGeneratedTransferIdIsAnUnguessableUuid() {
        JsonView first = view(prefillApi.completions("{\"prompt\": \"" + PROMPT_TEXT + "\","
                + " \"max_tokens\": 1, \"temperature\": 0,"
                + " \"kv_transfer_params\": {\"do_remote_decode\": true}}").getEntity())
                .obj("kv_transfer_params");
        JsonView second = view(prefillApi.completions("{\"prompt\": \"" + PROMPT_TEXT + "\","
                + " \"max_tokens\": 1, \"temperature\": 0,"
                + " \"kv_transfer_params\": {\"do_remote_decode\": true}}").getEntity())
                .obj("kv_transfer_params");
        assertTrue(first.bool("do_remote_prefill"));
        assertEquals(4, UUID.fromString(first.str("transfer_id")).version());
        assertEquals(4, UUID.fromString(second.str("transfer_id")).version());
        assertNotEquals(first.str("transfer_id"), second.str("transfer_id"));
    }

    @Test
    void theStatusRouteReportsTheConnector() {
        JsonView body = view(new KvTransferResource(decode).status().getEntity());
        assertTrue(body.bool("enabled"));
        assertEquals("kv_consumer", body.str("kv_role"));
        assertEquals("decode-1", body.str("engine_id"));
        assertEquals("TcpKvConnector", body.str("kv_connector"));
        assertEquals("TcpKvConnector", body.obj("connector").str("connector"));
        assertFalse(view(new KvTransferResource(reference).status().getEntity()).bool("enabled"));
    }

    @Test
    void theRequestFieldRefusesVllmBlockIds() {
        String message = errorOf(() -> prefillApi.completions("{\"prompt\": \"hi\", \"max_tokens\": 1,"
                + " \"kv_transfer_params\": {\"do_remote_decode\": true, \"remote_block_ids\": [[1]]}}"));
        assertTrue(message.contains("remote_block_ids"), message);
    }

    @Test
    void anUnsupportedConnectorRefusesTheStart() {
        ServerOptions options = ServerOptions.defaults().withKvTransfer(KvTransferConfig.parse(
                "{\"kv_connector\": \"NixlConnector\", \"kv_role\": \"kv_both\"}", "--kv-transfer-config"));
        UnsupportedOperationException refused = assertThrows(UnsupportedOperationException.class,
                () -> new EngineService(model.toString(), config, options));
        assertTrue(refused.getMessage().contains("NixlConnector"), refused.getMessage());
    }
}
