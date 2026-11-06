package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.ws.rs.core.Response;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The inference endpoints the cloud protocol adapters dispatch to. A SageMaker {@code /invocations} body names its
 * request type by its fields (vLLM's invocation types), and a Vertex AI instance names its route by
 * {@code @requestFormat} (llama.cpp's {@code register_gcp_compat}); both then answer through the resource that
 * already serves the matching path, so the adapters add a dispatch shape rather than a second implementation of an
 * endpoint.
 *
 * <p>The invocation types follow vLLM's priority order: a {@code ChatCompletionRequest} before a
 * {@code CompletionRequest}, then the pooling types — {@code EmbeddingRequest} (or {@code ClassificationRequest}
 * on a sequence-classifier model), {@code RerankRequest}, {@code ScoreRequest} and the task-named
 * {@code PoolingRequest}. A body that names none of them refuses with vLLM's message listing the types.</p>
 */
final class CloudProtocols {

    /** The response id prefix of each route's answer, kept as the resource's own prefix. */
    private static final String INVOCATION_TYPES = "ChatCompletionRequest, CompletionRequest, EmbeddingRequest,"
            + " ClassificationRequest, RerankRequest, ScoreRequest, PoolingRequest";

    /**
     * A Vertex AI instance's {@code @requestFormat}: both the registered path and llama.cpp's camelCase alias
     * ({@code path_to_gcp_format}) resolve to the path the endpoint serves.
     */
    private static final Map<String, String> VERTEX_FORMATS = vertexFormats();

    private CloudProtocols() {
    }

    /** The request types a SageMaker {@code /invocations} body may name, in vLLM's priority order. */
    enum Invocation {
        CHAT("/v1/chat/completions"),
        COMPLETION("/v1/completions"),
        EMBEDDING("/v1/embeddings"),
        CLASSIFICATION("/classify"),
        RERANK("/v1/rerank"),
        SCORE("/score"),
        POOLING("/pooling");

        private final String route;

        Invocation(String route) {
            this.route = route;
        }

        /** The path whose resource answers this request type. */
        String route() {
            return route;
        }
    }

    /**
     * The invocation a SageMaker body names, or null when its fields match none: {@code messages} is a chat
     * completion, {@code prompt} a completion, {@code query} with {@code documents} a rerank, one of vLLM's score
     * side pairs a score, and {@code input} an embedding (a classification on a sequence-classifier model) unless a
     * {@code task} names the generic pooling request.
     */
    static Invocation invocationOf(JsonView body, EngineService target) {
        if (body.raw("messages") != null) {
            return Invocation.CHAT;
        }
        if (body.raw("prompt") != null) {
            return Invocation.COMPLETION;
        }
        if (body.raw("query") != null && body.raw("documents") != null) {
            return Invocation.RERANK;
        }
        if (hasScoreSides(body)) {
            return Invocation.SCORE;
        }
        if (body.raw("input") != null) {
            if (body.raw("task") != null) {
                return Invocation.POOLING;
            }
            return target != null && target.sequenceClassifier() ? Invocation.CLASSIFICATION : Invocation.EMBEDDING;
        }
        return null;
    }

    /** The invocation of {@code body}, or a 400 with vLLM's message listing the accepted request types. */
    static Invocation requireInvocation(JsonView body, EngineService target) {
        Invocation invocation = invocationOf(body, target);
        if (invocation == null) {
            throw ServerJson.badRequest("Cannot find suitable handler for request. Expected one of: ["
                    + INVOCATION_TYPES + "]");
        }
        return invocation;
    }

    /** Whether the body carries one of vLLM's score side pairs. */
    private static boolean hasScoreSides(JsonView body) {
        return (body.has("queries") && (body.has("documents") || body.has("items")))
                || (body.has("data_1") && body.has("data_2"))
                || (body.has("text_1") && body.has("text_2"));
    }

    /**
     * Answers {@code bodyText} on {@code route} through that path's resource. The request and router travel with
     * it, so streaming cancels on disconnect and a body {@code model} still selects its routed model.
     *
     * @throws jakarta.ws.rs.WebApplicationException when the endpoint refuses the request
     */
    static Response invoke(String route, EngineService service, ModelRouter router, HttpServerRequest http,
                           String bodyText) {
        switch (route) {
            case "/v1/chat/completions":
                return new OpenAiResource(service, router, http).chatCompletions(bodyText);
            case "/v1/completions":
                return new OpenAiResource(service, router, http).completions(bodyText);
            case "/v1/embeddings":
                return new PoolingResource(service, router, http).embeddings(bodyText);
            case "/classify":
                return new PoolingResource(service, router, http).classify(bodyText);
            case "/v1/rerank":
            case "/v2/rerank":
                return new PoolingResource(service, router, http).rerank(bodyText);
            case "/score":
            case "/v1/score":
                return new PoolingResource(service, router, http).score(bodyText);
            case "/pooling":
                return new PoolingResource(service, router, http).pooling(bodyText);
            case "/v2/embed":
                return new CohereEmbedResource(service, router, http).embed(bodyText);
            case "/v1/messages":
                return new AnthropicResource(service, router, http).messages(bodyText);
            case "/cohere/v2/chat":
                return new CohereChatResource(service, router, http).chat(bodyText);
            default:
                throw ServerJson.badRequest("no handler is registered for '" + route + "'");
        }
    }

    /**
     * The route a Vertex AI instance's {@code @requestFormat} names: a registered path, or llama.cpp's camelCase
     * form of one ({@code chatCompletions}), or null when neither is a route this server serves.
     */
    static String vertexRoute(String format) {
        return format == null ? null : VERTEX_FORMATS.get(format);
    }

    /** The paths and camelCase aliases a Vertex AI instance may name ({@code path_to_gcp_format} of each). */
    private static Map<String, String> vertexFormats() {
        Map<String, String> formats = new LinkedHashMap<>();
        for (String route : List.of("/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/rerank",
                "/v2/rerank", "/score", "/v1/score", "/classify", "/pooling", "/v2/embed", "/v1/messages",
                "/cohere/v2/chat")) {
            formats.putIfAbsent(route, route);
            formats.putIfAbsent(camelCase(route), route);
        }
        return Map.copyOf(formats);
    }

    /**
     * llama.cpp's {@code path_to_gcp_format}: drop a leading {@code /v1}, then capitalize the letter after each
     * {@code /}, {@code -} or {@code _} and stop before a path parameter — {@code /v1/chat/completions} is
     * {@code chatCompletions}.
     */
    static String camelCase(String path) {
        String s = path;
        if (s.length() > 3 && s.charAt(0) == '/' && s.charAt(1) == 'v' && s.charAt(2) == '1') {
            s = s.substring(3);
        }
        if (!s.isEmpty() && s.charAt(0) == '/') {
            s = s.substring(1);
        }
        StringBuilder out = new StringBuilder(s.length());
        boolean capitalize = false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == ':') {
                break;
            }
            if (c == '/' || c == '-' || c == '_') {
                capitalize = true;
            } else {
                out.append(capitalize ? Character.toUpperCase(c) : c);
                capitalize = false;
            }
        }
        return out.toString();
    }
}
