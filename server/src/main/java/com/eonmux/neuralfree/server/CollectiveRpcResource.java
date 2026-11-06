package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * vLLM's development collective RPC route ({@code POST /collective_rpc}): a trainer drives one engine method on
 * every replica through a single call. The body names the {@code method}, its positional {@code args} and keyword
 * {@code kwargs} (both strings, as vLLM's route carries them) and an optional {@code timeout} in seconds that
 * bounds each replica's wait. The answer is {@code {"results": [...]}} with one entry per replica, in pool order;
 * a method that returns nothing answers null entries, and an unknown method or argument refuses by name with the
 * supported set.
 *
 * <p>The dispatched methods are the weight-transfer lifecycle and the pool reads that verify it ({@link
 * CollectiveRpc}). In router mode every call takes the model from {@code ?model=} (with {@code ?autoload=} to
 * override the server's), so a call reaches one model's replicas; a missing name is a 400 and an unknown one a
 * 404. The route serves only with the {@link DevMode} switch on, and answers 404 without it, as vLLM registers it
 * only in its development mode.</p>
 */
@Path("/")
public class CollectiveRpcResource {

    /** The fields of the request body. */
    private static final Set<String> BODY_FIELDS = Set.of("method", "args", "kwargs", "timeout");

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where calls serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** The development-route switch; the route answers 404 while it is off. */
    @Inject
    DevMode devMode;

    /** Test and manual wiring: builds the resource over an explicit service and dev-mode switch. */
    public CollectiveRpcResource(EngineService service, DevMode devMode) {
        this(service, null, devMode);
    }

    /** Test and manual wiring over an explicit service, router and dev-mode switch. */
    public CollectiveRpcResource(EngineService service, ModelRouter router, DevMode devMode) {
        this.service = service;
        this.router = router;
        this.devMode = devMode;
    }

    public CollectiveRpcResource() {
    }

    @POST
    @Path("/collective_rpc")
    @Produces(MediaType.APPLICATION_JSON)
    public Response collectiveRpc(String body, @QueryParam("model") String model,
                                  @QueryParam("autoload") String autoload) {
        devMode.require();
        return collectiveRpcOf(body, ModelRouter.targetQuery(router, service, model, autoload));
    }

    /** Test and manual wiring: the call on the wired service, without a model query. */
    Response collectiveRpc(String body) {
        devMode.require();
        return collectiveRpcOf(body, service);
    }

    private static Response collectiveRpcOf(String bodyText, EngineService service) {
        JsonView body = ServerJson.parse(bodyText);
        WeightTransferBodies.requireKnownKeys(body, BODY_FIELDS, "the request body",
                "the fields are method, args, kwargs and timeout");
        String method = WeightTransferBodies.requiredStr(body, "method", "the request body");
        List<String> args = stringArgs(body);
        Map<String, String> kwargs = stringKwargs(body);
        long waitMillis = timeoutMillis(body);
        return ServerJson.operate(() -> {
            List<Object> results = new CollectiveRpc(service).call(method, args, kwargs, waitMillis);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("results", results);
            return out;
        });
    }

    /** The {@code args} array: absent gives none, and every entry must be a string. */
    private static List<String> stringArgs(JsonView body) {
        Object raw = body.raw("args");
        if (raw == null) {
            return List.of();
        }
        if (!(raw instanceof List<?> items)) {
            throw ServerJson.badRequest(body.where("args") + " must be a JSON array of strings");
        }
        List<String> out = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) {
            Object item = items.get(i);
            if (!(item instanceof String value)) {
                throw ServerJson.badRequest(body.where("args") + "[" + i + "] must be a string");
            }
            out.add(value);
        }
        return out;
    }

    /** The {@code kwargs} object: absent gives none, and every value must be a string. */
    private static Map<String, String> stringKwargs(JsonView body) {
        Object raw = body.raw("kwargs");
        if (raw == null) {
            return Map.of();
        }
        if (!(raw instanceof Map<?, ?> entries)) {
            throw ServerJson.badRequest(body.where("kwargs") + " must be a JSON object of strings");
        }
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : entries.entrySet()) {
            if (!(entry.getValue() instanceof String value)) {
                throw ServerJson.badRequest(body.where("kwargs") + "." + entry.getKey() + " must be a string");
            }
            out.put(String.valueOf(entry.getKey()), value);
        }
        return out;
    }

    /** The optional {@code timeout} in seconds as milliseconds; absent or null is no bound of its own. */
    private static long timeoutMillis(JsonView body) {
        if (body.raw("timeout") == null) {
            return -1;
        }
        double seconds;
        try {
            seconds = body.f64("timeout");
        } catch (RuntimeException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        if (!(seconds > 0) || Double.isInfinite(seconds)) {
            throw ServerJson.badRequest(body.where("timeout") + " must be a positive number of seconds");
        }
        double millis = seconds * 1000.0;
        if (millis > Long.MAX_VALUE) {
            throw ServerJson.badRequest(body.where("timeout") + " is too large");
        }
        return Math.max(1L, (long) millis);
    }
}
