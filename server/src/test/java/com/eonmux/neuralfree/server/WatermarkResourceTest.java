package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkState;
import com.eonmux.neuralfree.gguf.engine.watermark.Watermarker;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.SplittableRandom;

import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The watermark detection endpoint and the per-request {@code watermarking} switch: a watermarked sequence built
 * with the engine's key is detected, an independent sequence is not, and a request opts in and out of the engine's
 * watermark while an engine that serves none refuses an explicit opt-in by name.
 */
class WatermarkResourceTest {

    private static final WatermarkConfig WATERMARK = WatermarkConfig.builder(424242L).contextWidth(4).build();

    private static EngineService service;
    private static WatermarkResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = pagedConfig(256, 2);
        service = new EngineService(model.toString(), config, ServerOptions.defaults().withWatermark(WATERMARK));
        resource = new WatermarkResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** {@code count} watermarked token ids under the engine's key over random logits rows. */
    private static int[] watermarkedTokens(int count, long seed) {
        int vocabulary = service.vocabSize();
        Watermarker watermarker = Watermarker.create(WATERMARK);
        WatermarkState state = new WatermarkState(WATERMARK);
        SplittableRandom random = new SplittableRandom(seed);
        int[] tokens = new int[count];
        float[] logits = new float[vocabulary];
        for (int position = 0; position < count; position++) {
            for (int token = 0; token < vocabulary; token++) {
                logits[token] = (float) (random.nextDouble() * 2.0);
            }
            tokens[position] = watermarker.sample(logits, state.context(), () -> 0.5);
            state.accept(tokens[position]);
        }
        return tokens;
    }

    private static String tokenIdsBody(int[] tokens) {
        return tokenIdsBody(tokens, "");
    }

    private static String tokenIdsBody(int[] tokens, String extra) {
        StringBuilder json = new StringBuilder("{\"token_ids\": [");
        for (int i = 0; i < tokens.length; i++) {
            if (i > 0) {
                json.append(", ");
            }
            json.append(tokens[i]);
        }
        return json.append(']').append(extra).append('}').toString();
    }

    private static JsonView bodyOf(Object entity) {
        return view(entity);
    }

    @Test
    void aWatermarkedSequenceIsDetected() {
        int[] tokens = watermarkedTokens(192, 11L);
        Response response = resource.detectOf(view(tokenIdsBody(tokens)), service);
        JsonView body = bodyOf(response.getEntity());
        assertEquals(200, response.getStatus());
        assertTrue(body.bool("is_watermarked"));
        assertEquals(192, body.i32("num_scored_tokens"));
        assertTrue(body.f64("p_value") < 0.01);
        assertEquals("gumbel", body.str("algorithm"));
        assertEquals(4, body.i32("context_width"));
    }

    @Test
    void anIndependentSequenceIsNotDetected() {
        int vocabulary = service.vocabSize();
        SplittableRandom random = new SplittableRandom(99L);
        int[] tokens = new int[192];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = random.nextInt(vocabulary);
        }
        JsonView body = bodyOf(resource.detectOf(view(tokenIdsBody(tokens)), service).getEntity());
        assertFalse(body.bool("is_watermarked"));
        assertTrue(body.f64("p_value") > 0.01);
    }

    @Test
    void textIsTokenizedAndScored() {
        JsonView body = bodyOf(resource.detectOf(view("{\"text\": \"hello world\"}"), service).getEntity());
        assertEquals(service.encode("hello world", false, false).length, body.i32("num_scored_tokens"));
        assertFalse(body.bool("is_watermarked"));
    }

    @Test
    void theThresholdAndDeduplicationArePerRequest() {
        JsonView empty = bodyOf(resource.detectOf(view("{\"token_ids\": [], \"p_value_threshold\": 0.5,"
                + " \"deduplicate_contexts\": false}"), service).getEntity());
        assertEquals(0, empty.i32("num_scored_tokens"));
        assertFalse(empty.bool("is_watermarked"));

        int[] tokens = watermarkedTokens(64, 5L);
        JsonView deduplicated = bodyOf(resource.detectOf(
                view(tokenIdsBody(tokens, ", \"deduplicate_contexts\": true")), service).getEntity());
        assertTrue(deduplicated.bool("is_watermarked"));
    }

    @Test
    void exactlyOneSequenceAndValidValuesAreRequired() {
        assertEquals(400, refusal("{}"));
        assertEquals(400, refusal("{\"text\": \"hi\", \"token_ids\": [1]}"));
        assertEquals(400, refusal("{\"text\": 5}"));
        assertEquals(400, refusal("{\"token_ids\": 5}"));
        assertEquals(400, refusal("{\"token_ids\": [-1]}"));
        assertEquals(400, refusal("{\"token_ids\": [" + service.vocabSize() + "]}"));
        assertEquals(400, refusal("{\"text\": \"hi\", \"p_value_threshold\": 0}"));
        assertEquals(400, refusal("{\"text\": \"hi\", \"p_value_threshold\": 1}"));
        assertEquals(400, refusal("{\"text\": \"hi\", \"deduplicate_contexts\": \"yes\"}"));
    }

    private static int refusal(String json) {
        WebApplicationException e = assertThrows(WebApplicationException.class,
                () -> resource.detectOf(JsonView.of(Json.parse(json, Json.Limits.UNTRUSTED), "body"), service));
        return e.getResponse().getStatus();
    }

    @Test
    void aCompletionIsWatermarkedAndDetectedEndToEnd() {
        OpenAiResource openAi = new OpenAiResource(service);
        String plainBody = "{\"prompt\": \"hello\", \"max_tokens\": 64, \"temperature\": 1.0, \"seed\": 7,"
                + " \"return_token_ids\": true, \"watermarking\": false}";
        String watermarkedBody = plainBody.replace("\"watermarking\": false", "\"watermarking\": true");
        int[] plain = completionTokens(openAi, plainBody);
        int[] watermarked = completionTokens(openAi, watermarkedBody);
        assertFalse(java.util.Arrays.equals(plain, watermarked),
                "the keyed draw replaces the seeded categorical draw");
        assertTrue(com.eonmux.neuralfree.gguf.engine.watermark.WatermarkDetector.create(WATERMARK)
                .detect(watermarked).isWatermarked());
    }

    /** The first choice's generated token ids of a completions request. */
    private static int[] completionTokens(OpenAiResource openAi, String body) {
        Response response = openAi.completions(body);
        assertEquals(200, response.getStatus());
        JsonView choice = ServerTestSupport.at(bodyOf(response.getEntity()), "choices", 0);
        java.util.List<?> ids = (java.util.List<?>) choice.raw("token_ids");
        int[] tokens = new int[ids.size()];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = ((Number) ids.get(i)).intValue();
        }
        return tokens;
    }

    @Test
    void aServerWithoutAWatermarkRefusesDetection() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (EngineService plain = new EngineService(model.toString(), pagedConfig(64, 1))) {
            WatermarkResource plainResource = new WatermarkResource(plain);
            WebApplicationException e = assertThrows(WebApplicationException.class,
                    () -> plainResource.detectOf(view("{\"text\": \"hi\"}"), plain));
            assertEquals(400, e.getResponse().getStatus());
            assertTrue(ServerTestSupport.errorOf(() -> plainResource.detectOf(view("{\"text\": \"hi\"}"), plain))
                    .contains("watermark"));
        }
    }

    @Test
    void aRequestOptsOutOfTheEngineWatermark() {
        GenerationRequest base = OpenAiRequests.sampling(view("{}"), -1, service.vocabSize(), -1,
                LogprobsMode.RAW_LOGPROBS, LogitsProcessors.empty(), WATERMARK);
        assertEquals(WATERMARK, base.sampler().watermark());
        GenerationRequest disabled = OpenAiRequests.sampling(view("{\"watermarking\": false}"), -1,
                service.vocabSize(), -1, LogprobsMode.RAW_LOGPROBS, LogitsProcessors.empty(), WATERMARK);
        assertNull(disabled.sampler().watermark());
        GenerationRequest enabled = OpenAiRequests.sampling(view("{\"watermarking\": true}"), -1,
                service.vocabSize(), -1, LogprobsMode.RAW_LOGPROBS, LogitsProcessors.empty(), WATERMARK);
        assertEquals(WATERMARK, enabled.sampler().watermark());
        GenerationRequest unwatermarked = OpenAiRequests.sampling(view("{}"), -1, service.vocabSize(), -1,
                LogprobsMode.RAW_LOGPROBS, LogitsProcessors.empty(), null);
        assertNull(unwatermarked.sampler().watermark());
    }

    @Test
    void anExplicitOptInWithoutAnEngineWatermarkIsRefused() {
        WebApplicationException e = assertThrows(WebApplicationException.class,
                () -> OpenAiRequests.sampling(view("{\"watermarking\": true}"), -1, service.vocabSize(), -1,
                        LogprobsMode.RAW_LOGPROBS, LogitsProcessors.empty(), null));
        assertEquals(400, e.getResponse().getStatus());
        WebApplicationException notBoolean = assertThrows(WebApplicationException.class,
                () -> OpenAiRequests.sampling(view("{\"watermarking\": \"yes\"}"), -1, service.vocabSize(), -1,
                        LogprobsMode.RAW_LOGPROBS, LogitsProcessors.empty(), WATERMARK));
        assertEquals(400, notBoolean.getResponse().getStatus());
    }

    @Test
    void theAnthropicMessagesRouteCarriesTheEngineWatermark() {
        GenerationRequest watermarked = AnthropicRequests.request(view("{\"max_tokens\": 8}"),
                LogitsProcessors.empty(), WATERMARK);
        assertEquals(WATERMARK, watermarked.sampler().watermark());
        GenerationRequest plain = AnthropicRequests.request(view("{\"max_tokens\": 8}"),
                LogitsProcessors.empty(), null);
        assertNull(plain.sampler().watermark());
    }

    @Test
    void theLaunchFlagSetsTheWatermarkProperty() {
        Map<String, String> properties = ServerLaunchArgs
                .parse(Map.of(), "--watermark-config", "{\"key\": 7, \"algorithm\": \"dual_key_gumbel\"}")
                .properties();
        assertEquals("{\"key\": 7, \"algorithm\": \"dual_key_gumbel\"}",
                properties.get(WatermarkConfig.PROP_CONFIG));
        assertThrows(IllegalArgumentException.class,
                () -> ServerLaunchArgs.parse(Map.of(), "--watermark-config", "{\"key\": 1, \"alpha\": 2}"));
        assertThrows(IllegalArgumentException.class,
                () -> ServerLaunchArgs.parse(Map.of(), "--watermark-config", "not json"));
    }

    @Test
    void theEnvironmentMaySupplyTheKeyTheLaunchJsonOmits() {
        Map<String, String> properties = ServerLaunchArgs.parse(
                Map.of(WatermarkConfig.ENV_KEY, "11"),
                "--watermark-config", "{\"algorithm\": \"dual_key_gumbel\"}").properties();
        assertEquals("{\"algorithm\": \"dual_key_gumbel\"}", properties.get(WatermarkConfig.PROP_CONFIG));
        assertThrows(IllegalArgumentException.class, () -> ServerLaunchArgs.parse(
                Map.of(WatermarkConfig.ENV_KEY, "11"), "--watermark-config", "{\"key\": 7}"));
        assertThrows(IllegalArgumentException.class, () -> ServerLaunchArgs.parse(Map.of(),
                "--watermark-config", "{\"algorithm\": \"dual_key_gumbel\"}"));
    }
}
