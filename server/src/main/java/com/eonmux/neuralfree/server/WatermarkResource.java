package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkDetection;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkDetector;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The watermark detection endpoint ({@code POST /v1/watermark/detect}), the server face of the engine's text
 * watermark ({@code --watermark-config}, vLLM's {@code WatermarkConfig}): it scores a token sequence under the
 * engine's key and answers the {@link WatermarkDetection} fields.
 *
 * <p>The body names the sequence with {@code token_ids} (the exact ids generation produced) or {@code text}
 * (tokenized by the served model without a BOS); exactly one of the two is required. Optional:
 * {@code p_value_threshold} (in {@code (0, 1)}, vLLM's 0.01 by default) and {@code deduplicate_contexts} (a
 * boolean, true by default: a repeated context is scored once). The engine's key, algorithm, {@code alpha} and
 * context width come from its watermark configuration; a server that serves no watermark refuses with 400.
 */
@Path("/")
public class WatermarkResource {

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public WatermarkResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public WatermarkResource(EngineService service, ModelRouter router) {
        this.service = service;
        this.router = router;
    }

    public WatermarkResource() {
    }

    /** The detection of one request body. */
    @POST
    @Path("/v1/watermark/detect")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response detect(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        try {
            return detectOf(body, target);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** {@link #detect(String)} over a parsed body, for the HTTP layer's tests and manual wiring. */
    Response detectOf(JsonView body, EngineService target) {
        WatermarkConfig config = target.options().watermark();
        if (config == null) {
            throw ServerJson.badRequest("watermarking: the server serves no watermark; start it with"
                    + " --watermark-config (or " + WatermarkConfig.PROP_CONFIG + ")");
        }
        int[] tokens = tokens(body, target);
        float threshold = WatermarkDetector.DEFAULT_P_VALUE_THRESHOLD;
        Object rawThreshold = body.raw("p_value_threshold");
        if (rawThreshold != null) {
            if (!(rawThreshold instanceof Number)) {
                throw ServerJson.badRequest("p_value_threshold must be a number in (0, 1)");
            }
            threshold = ((Number) rawThreshold).floatValue();
        }
        boolean deduplicate = true;
        Object rawDeduplicate = body.raw("deduplicate_contexts");
        if (rawDeduplicate != null) {
            if (!(rawDeduplicate instanceof Boolean)) {
                throw ServerJson.badRequest("deduplicate_contexts must be a boolean");
            }
            deduplicate = (Boolean) rawDeduplicate;
        }
        WatermarkDetector detector;
        try {
            detector = WatermarkDetector.create(config, threshold, deduplicate);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        try {
            return Response.ok(ServerJson.write(detectionBody(detector.detect(tokens), config))).build();
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** The sequence the body names: its {@code token_ids}, or its {@code text} tokenized without a BOS. */
    private static int[] tokens(JsonView body, EngineService target) {
        boolean hasTokens = body.raw("token_ids") != null;
        boolean hasText = body.raw("text") != null;
        if (hasTokens == hasText) {
            throw ServerJson.badRequest("watermark detection needs exactly one of 'token_ids' and 'text'");
        }
        if (hasText) {
            Object raw = body.raw("text");
            if (!(raw instanceof String text)) {
                throw ServerJson.badRequest("text must be a string");
            }
            return target.encode(text, false, false);
        }
        if (!(body.raw("token_ids") instanceof List<?> list)) {
            throw ServerJson.badRequest("token_ids must be an array of token ids");
        }
        int vocabSize = target.vocabSize();
        int[] tokens = new int[list.size()];
        for (int i = 0; i < tokens.length; i++) {
            Object value = list.get(i);
            if (!(value instanceof Long) && !(value instanceof Integer)) {
                throw ServerJson.badRequest("token_ids[" + i + "] must be a token id");
            }
            long id = ((Number) value).longValue();
            if (id < 0 || id >= vocabSize) {
                throw ServerJson.badRequest("token_ids[" + i + "] is " + id + ", outside the vocabulary of "
                        + vocabSize + " tokens");
            }
            tokens[i] = (int) id;
        }
        return tokens;
    }

    private static Map<String, Object> detectionBody(WatermarkDetection detection, WatermarkConfig config) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("score", detection.score());
        out.put("p_value", detection.pValue());
        out.put("num_scored_tokens", detection.numScoredTokens());
        out.put("is_watermarked", detection.isWatermarked());
        out.put("algorithm", config.algorithm().label());
        out.put("context_width", config.contextWidth());
        return out;
    }
}
