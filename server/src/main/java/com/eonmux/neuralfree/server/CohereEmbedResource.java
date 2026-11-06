package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Cohere's embed API, {@code POST /v2/embed}, as vLLM serves it (vllm/entrypoints/pooling/embed): exactly one of
 * {@code texts} (strings), {@code images} (image URLs, data URIs included) or {@code inputs} (objects whose
 * {@code content} holds {@code text} and {@code image_url} parts) is given and non-empty. Texts pool as completion
 * inputs; images and mixed inputs render as one user message each through the model's chat template and pool
 * through the media path ({@link PoolingResource#embeddedOf}), so a CLIP dual encoder embeds an image and a
 * projector model pools its media prompt. {@code truncate} is {@code END} (default; keep the first
 * {@code max_tokens} tokens), {@code START} (keep the last) or {@code NONE} (refuse an input longer than
 * {@code max_tokens}); {@code output_dimension} keeps the leading values of the normalized embedding; {@code
 * embedding_types} (default {@code ["float"]}) picks {@code float}, {@code binary} (sign bits packed most
 * significant first, minus 128), {@code ubinary} (the packed bytes) and {@code base64} (little-endian float32).
 * {@code input_type} names a task instruction; NeuralFree models define none, so it refuses by name, as vLLM does
 * for such a model.
 */
@Path("/")
public class CohereEmbedResource {

    /** The {@code embedding_types} vLLM's Cohere protocol accepts. */
    static final Set<String> EMBEDDING_TYPES = Set.of("float", "binary", "ubinary", "base64");

    private static final Set<String> FIELDS = Set.of("model", "input_type", "texts", "images", "inputs",
            "output_dimension", "embedding_types", "truncate", "max_tokens", "priority");

    @Inject
    EngineService service;

    @Inject
    HttpServerRequest http;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public CohereEmbedResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public CohereEmbedResource(EngineService service, ModelRouter router) {
        this(service, router, null);
    }

    /** A resource over an explicit service, router and request, so a delegated request keeps its request id. */
    public CohereEmbedResource(EngineService service, ModelRouter router, HttpServerRequest http) {
        this.service = service;
        this.router = router;
        this.http = http;
    }

    public CohereEmbedResource() {
    }

    @POST
    @Path("/v2/embed")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response embed(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        return OpenAiResource.mapped(() -> embedOf(target, body));
    }

    private Response embedOf(EngineService service, JsonView body) {
        for (String key : body.keys()) {
            if (!FIELDS.contains(key)) {
                throw ServerJson.badRequest("unknown field '" + key + "' in a Cohere embed request");
            }
        }
        String model = service.loras().model(body);
        LoraSelection lora = service.loras().selection(body, model);
        if (service.perTokenEmbeddings() && service.sparsePooling() == null
                && service.poolingKind() == com.eonmux.neuralfree.gguf.arch.PoolingKind.NONE) {
            throw ServerJson.badRequest("the model pools by none; /v2/embed answers pooled embeddings");
        }
        if (body.raw("input_type") != null) {
            throw ServerJson.badRequest("Unsupported input_type '" + body.str("input_type") + "'. This model does"
                    + " not define any input_type task instructions.");
        }
        CohereRequest request = CohereRequest.of(body);
        JsonView pooling = JsonView.of(request.poolingBody(), "body");
        EmbeddingOptions options = EmbeddingOptions.of(pooling, service.maxModelLength(),
                service.sparsePooling() != null);
        List<PoolingInputs.Input> inputs = PoolingInputs.of(pooling, service, options);
        PoolingResource pooler = new PoolingResource(service);
        List<float[]> vectors = new ArrayList<>(inputs.size());
        long tokens = 0;
        for (PoolingInputs.Input input : inputs) {
            PoolingResource.Embedded embedded = pooler.embeddedOf(lora, input);
            request.checkLength(embedded.tokens(), input.field());
            vectors.add(options.shaped(embedded.vector()));
            tokens += embedded.tokens();
        }
        return Response.ok(ServerJson.write(response(this, request, model, vectors, tokens))).build();
    }

    private static Map<String, Object> response(CohereEmbedResource resource, CohereRequest request, String model,
                                                List<float[]> vectors, long tokens) {
        Map<String, Object> billed = new LinkedHashMap<>();
        long imageTokens = request.hasImage() ? tokens : 0;
        billed.put("input_tokens", tokens - imageTokens);
        billed.put("image_tokens", imageTokens);
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("api_version", Map.of("version", "2"));
        meta.put("billed_units", billed);
        String named = OpenAiRequests.requestId(resource.http, null);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", "embd-" + (named != null ? named : Long.toString(System.nanoTime())));
        out.put("embeddings", typed(vectors, request.embeddingTypes()));
        if (request.texts() != null) {
            out.put("texts", request.texts());
        }
        out.put("meta", meta);
        out.put("response_type", "embeddings_by_type");
        return out;
    }

    /** The {@code embeddings} object: one list per requested type, in the request's order. */
    static Map<String, Object> typed(List<float[]> vectors, List<String> types) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (String type : types) {
            List<Object> values = new ArrayList<>(vectors.size());
            for (float[] v : vectors) {
                switch (type) {
                    case "float" -> values.add(EmbeddingOptions.values(v));
                    case "binary" -> values.add(packed(v, true));
                    case "ubinary" -> values.add(packed(v, false));
                    case "base64" -> values.add(base64(v));
                    default -> throw new IllegalArgumentException("embedding type " + type);
                }
            }
            out.put(type, values);
        }
        return out;
    }

    /**
     * The sign bits of {@code v} ({@code x >= 0} is 1) packed eight per byte, most significant first, as numpy's
     * {@code packbits}; {@code signed} shifts each byte by -128 (Cohere's {@code binary}).
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 when the width is not a multiple of 8
     */
    static List<Object> packed(float[] v, boolean signed) {
        if (v.length % 8 != 0) {
            throw ServerJson.badRequest("Embedding dimension must be a multiple of 8 for binary embedding types,"
                    + " but got " + v.length + ".");
        }
        List<Object> out = new ArrayList<>(v.length / 8);
        for (int b = 0; b < v.length; b += 8) {
            int bits = 0;
            for (int i = 0; i < 8; i++) {
                bits = bits << 1 | (v[b + i] >= 0 ? 1 : 0);
            }
            out.add(signed ? bits - 128 : bits);
        }
        return out;
    }

    /** {@code v} as base64 little-endian float32 bytes. */
    static String base64(float[] v) {
        ByteBuffer bytes = ByteBuffer.allocate(Float.BYTES * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            bytes.putFloat(x);
        }
        return Base64.getEncoder().encodeToString(bytes.array());
    }
}
