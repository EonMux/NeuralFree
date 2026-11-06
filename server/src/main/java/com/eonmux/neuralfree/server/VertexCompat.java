package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.quarkus.vertx.http.runtime.VertxHttpConfig;
import io.vertx.core.http.HttpHeaders;
import io.vertx.core.http.HttpServerRequest;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.RoutingContext;
import io.vertx.ext.web.handler.BodyHandler;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.regex.Pattern;

/**
 * Vertex AI's prediction protocol ({@code AIP_PREDICT_ROUTE}, llama.cpp's {@code register_gcp_compat}): when
 * {@code AIP_MODE=PREDICTION}, {@code POST AIP_PREDICT_ROUTE} (default {@code /predict}) takes
 * {@code {"instances": [...]}}, and each instance names the endpoint that answers it with {@code @requestFormat} —
 * either a served path or llama.cpp's camelCase alias ({@code chatCompletions}) — and carries that endpoint's body
 * without the format field. The answer is {@code {"predictions": [...]}}, one entry per instance, an instance that
 * fails contributing its error envelope as its prediction (llama.cpp's behaviour). Instances run in order; a
 * request is refused as a whole only when the envelope is malformed (a missing {@code instances} array, more than
 * {@value #MAX_INSTANCES} instances).
 *
 * <p>{@code AIP_HEALTH_ROUTE}, when set, answers the readiness probe as {@code /health} does, reusing
 * {@link OpsResource#ready()}. Streaming cannot be expressed in a prediction and is refused by name on any instance
 * that asks for it. Both routes are registered on the Vert.x router, ahead of Quarkus's CORS filter, so the
 * prediction route checks the API key itself like {@link MetricsGuard} does; the health route stays public like
 * {@code /health}.</p>
 */
@Singleton
public class VertexCompat {

    /** The environment variable that enables the protocol ({@code PREDICTION} on Vertex AI). */
    static final String MODE_ENV = "AIP_MODE";

    /** The environment variable naming the prediction path (llama.cpp's default is {@code /predict}). */
    static final String PREDICT_ENV = "AIP_PREDICT_ROUTE";

    /** The environment variable naming the health path; unset keeps the server's own {@code /health}. */
    static final String HEALTH_ENV = "AIP_HEALTH_ROUTE";

    /** A launch property carrying {@link #MODE_ENV} when the environment cannot (tests, a -D launch). */
    static final String MODE_PROPERTY = "neuralfree.server.aip.mode";

    /** A launch property carrying {@link #PREDICT_ENV}. */
    static final String PREDICT_PROPERTY = "neuralfree.server.aip.predict.route";

    /** A launch property carrying {@link #HEALTH_ENV}. */
    static final String HEALTH_PROPERTY = "neuralfree.server.aip.health.route";

    /** llama.cpp's {@code MAX_INSTANCES}. */
    static final int MAX_INSTANCES = 128;

    /** The prediction route's default path. */
    static final String DEFAULT_PREDICT = "/predict";

    /** One path segment: letters, digits, dash, dot, underscore, tilde; Vert.x parameters and wildcards excluded. */
    private static final Pattern SEGMENT = Pattern.compile("[A-Za-z0-9._~-]{1,128}");

    /** After the guards, ahead of the JAX-RS routes. */
    private static final int ORDER = Integer.MIN_VALUE + 500;

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring. */
    @Inject
    ModelRouter modelRouter;

    /** The readiness probe the health route answers with, so both report the same thing. */
    @Inject
    OpsResource ops;

    public VertexCompat() {
    }

    /** Test wiring over an explicit service, with no readiness resource. */
    VertexCompat(EngineService service, ModelRouter modelRouter) {
        this.service = service;
        this.modelRouter = modelRouter;
    }

    /** Installs the prediction and health routes when the container announces Vertex AI prediction. */
    void install(@Observes Router router, VertxHttpConfig httpConfig) {
        Map<String, String> env = System.getenv();
        Properties properties = System.getProperties();
        if (!enabled(env, properties)) {
            return;
        }
        String predict = predictPath(env, properties);
        String health = healthPath(env, properties);
        if (health != null && health.equals(predict)) {
            throw new IllegalArgumentException(HEALTH_ENV + " and " + PREDICT_ENV + " name the same path '"
                    + predict + "'");
        }
        ApiKeys keys = ApiKeys.from(env, properties);
        CorsPolicy cors = CorsPolicy.from(httpConfig.cors());
        router.post(predict).order(ORDER).handler(BodyHandler.create()).blockingHandler(context -> {
            if (!authorized(keys, cors, context)) {
                return;
            }
            Relay.write(context, answer(context.body() == null ? "" : context.body().asString(),
                    context.request()));
        });
        if (health != null) {
            router.get(health).order(ORDER).handler(context -> Relay.write(context, health()));
        }
    }

    /** One prediction request over the whole body: a refusal of the envelope answers as an HTTP error. */
    private Response answer(String bodyText, HttpServerRequest request) {
        try {
            return predict(bodyText, request);
        } catch (WebApplicationException e) {
            return e.getResponse();
        } catch (RuntimeException e) {
            return Response.status(400).type(MediaType.APPLICATION_JSON)
                    .entity(ServerJson.write(error(String.valueOf(e.getMessage())))).build();
        }
    }

    /**
     * The prediction envelope of {@code bodyText}: every instance dispatched to its route, its response (or error
     * envelope) as one prediction.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 when the envelope or an instance is malformed
     */
    Response predict(String bodyText) {
        return predict(bodyText, null);
    }

    /** {@link #predict(String)} answering on {@code request}, so a delegated generation cancels on disconnect. */
    Response predict(String bodyText, HttpServerRequest request) {
        JsonView body = ServerJson.parse(bodyText);
        Object raw = body.raw("instances");
        if (!(raw instanceof List<?> instances)) {
            throw ServerJson.badRequest("request body must include an array field named instances");
        }
        if (instances.size() > MAX_INSTANCES) {
            throw ServerJson.badRequest("instances array exceeds maximum size of " + MAX_INSTANCES);
        }
        List<Object> predictions = new ArrayList<>(instances.size());
        for (int i = 0; i < instances.size(); i++) {
            predictions.add(prediction(instances.get(i), "instances[" + i + "]", request));
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("predictions", predictions);
        return Response.ok(ServerJson.write(out)).type(MediaType.APPLICATION_JSON).build();
    }

    /** One prediction: the response body its route answers, or the error envelope it refuses with. */
    private Object prediction(Object raw, String where, HttpServerRequest request) {
        if (!(raw instanceof Map<?, ?> map)) {
            return error(where + " must be a JSON object");
        }
        Map<String, Object> instance = OpenAiRequests.stringKeys(map);
        Object format = instance.remove("@requestFormat");
        if (!(format instanceof String name)) {
            return error(where + " must include a string @requestFormat");
        }
        String route = CloudProtocols.vertexRoute(name);
        if (route == null) {
            return error("no handler registered for @requestFormat: " + name);
        }
        if (Boolean.TRUE.equals(instance.get("stream"))) {
            return error(where + ": streaming is not supported by the predict route; set stream to false or drop it");
        }
        try {
            Response response = CloudProtocols.invoke(route, service, modelRouter, request,
                    ServerJson.write(instance));
            return entity(response);
        } catch (WebApplicationException e) {
            return entity(e.getResponse());
        } catch (RuntimeException e) {
            return error(String.valueOf(e.getMessage()));
        }
    }

    /** A response's JSON entity parsed for the predictions array (raw text when it is not JSON). */
    private static Object entity(Response response) {
        Object entity = response.getEntity();
        if (entity == null) {
            return null;
        }
        String text = String.valueOf(entity);
        try {
            return Json.parse(text, Json.Limits.UNTRUSTED);
        } catch (RuntimeException notJson) {
            return text;
        }
    }

    /** The instance error llama.cpp embeds in {@code predictions}. */
    private static Map<String, Object> error(String message) {
        return ServerJson.errorBody(message, "invalid_request_error");
    }

    /** The readiness answer: {@link OpsResource#ready()} when the route has it, else this bean's own. */
    private Response health() {
        try {
            return ops == null ? ready() : ops.ready();
        } catch (WebApplicationException e) {
            return e.getResponse();
        }
    }

    /** 200 while a model is open (or the router runs), else the 503 envelope, as {@link OpsResource#ready()}. */
    private Response ready() {
        if (modelRouter != null && modelRouter.active()) {
            return Response.ok().build();
        }
        try {
            ServerJson.requireReady(service);
            return Response.ok().build();
        } catch (WebApplicationException e) {
            return e.getResponse();
        }
    }

    /** Whether the API key admits the request; a refusal is written with the CORS origin headers and ends it. */
    private static boolean authorized(ApiKeys keys, CorsPolicy cors, RoutingContext context) {
        if (!keys.enabled()) {
            return true;
        }
        String origin = context.request().getHeader("Origin");
        String presented = ApiKeys.presented(context.request().getHeader(HttpHeaders.AUTHORIZATION),
                context.request().getHeader("x-api-key"));
        if (keys.accepts(presented)) {
            return true;
        }
        var response = context.response()
                .setStatusCode(401)
                .putHeader("WWW-Authenticate", "Bearer")
                .putHeader(HttpHeaders.CONTENT_TYPE, "application/json");
        cors.stamp(response, origin);
        response.end(ApiKeys.refusalBody());
        return false;
    }

    /** Whether the container mode enables the protocol. */
    static boolean enabled(Map<String, String> env, Properties properties) {
        String mode = properties.getProperty(MODE_PROPERTY);
        if (mode == null) {
            mode = env.get(MODE_ENV);
        }
        return mode != null && mode.trim().equalsIgnoreCase("PREDICTION");
    }

    /** The prediction path: the property, else the environment variable, else {@value #DEFAULT_PREDICT}. */
    static String predictPath(Map<String, String> env, Properties properties) {
        String value = first(properties.getProperty(PREDICT_PROPERTY), env.get(PREDICT_ENV));
        return value == null || value.isBlank() ? DEFAULT_PREDICT : canonical(PREDICT_ENV, value);
    }

    /** The health path, or null when neither the property nor the environment variable names one. */
    static String healthPath(Map<String, String> env, Properties properties) {
        String value = first(properties.getProperty(HEALTH_PROPERTY), env.get(HEALTH_ENV));
        return value == null || value.isBlank() ? null : canonical(HEALTH_ENV, value);
    }

    private static String first(String property, String environment) {
        return property != null ? property : environment;
    }

    /**
     * The canonical path of {@code value}: a leading slash and slash-separated simple segments, so Vert.x matches it
     * literally (no parameter or wildcard syntax).
     *
     * @throws IllegalArgumentException naming {@code name} for an unusable path
     */
    static String canonical(String name, String value) {
        String path = value.trim();
        if (!path.startsWith("/")) {
            path = "/" + path;
        }
        StringBuilder out = new StringBuilder(path.length());
        for (String segment : path.split("/", -1)) {
            if (segment.isEmpty()) {
                continue;
            }
            if (!SEGMENT.matcher(segment).matches() || segment.equals(".") || segment.equals("..")) {
                throw new IllegalArgumentException(name + " must be a simple path such as /predict, got '" + value
                        + "': segments of letters, digits, dash, dot, underscore or tilde, no '..'");
            }
            out.append('/').append(segment);
        }
        if (out.length() == 0) {
            throw new IllegalArgumentException(name + " must be a path below the server root, got '" + value + "'");
        }
        return out.toString();
    }

    /**
     * The responses the Vertex routes write: a JAX-RS {@link Response} relayed to the Vert.x response. A streaming
     * body cannot be embedded in the prediction envelope (an instance refuses {@code stream}) or answered by the
     * readiness probe, so it is reported as a 500 rather than writing a half-envelope.
     */
    private static final class Relay {

        private Relay() {
        }

        static void write(RoutingContext context, Response response) {
            var out = context.response();
            Object entity = response.getEntity();
            if (entity instanceof StreamingOutput) {
                out.setStatusCode(500).putHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON)
                        .end(ServerJson.write(error("a streaming body cannot be relayed by these routes")));
                return;
            }
            out.setStatusCode(response.getStatus());
            MediaType type = response.getMediaType();
            out.putHeader(HttpHeaders.CONTENT_TYPE,
                    type == null ? MediaType.APPLICATION_JSON : type.toString());
            if (entity == null) {
                out.end();
            } else {
                out.end(String.valueOf(entity));
            }
        }
    }
}
