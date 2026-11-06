package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The request bodies of the weight-transfer routes and of the collective RPC's weight methods, parsed once for both
 * surfaces. A tensor arrives as the object vLLM's in-process backend carries: its stored {@code name}, ggml element
 * type ({@code dtype}), ggml shape ({@code ne}, {@code ne[0]} first) and base64 bytes ({@code data_b64}). An update
 * is one tensor object or a list of them, either bare or under vLLM's {@code update_info} wrapper; an initialization
 * names its {@code backend} either bare or under {@code init_info}.
 *
 * <p>Every refusal is a 400 naming the field: an unknown member, a missing one, a malformed base64 payload or a
 * shape whose byte count disagrees with the data.</p>
 */
final class WeightTransferBodies {

    /** The fields of one update tensor's JSON object. */
    static final Set<String> TENSOR_FIELDS = Set.of("name", "dtype", "ne", "data_b64");

    private WeightTransferBodies() {
    }

    /**
     * The tensors of an HTTP update body: the tensor object or list itself, or vLLM's
     * {@code {"update_info": <payload>}} wrapper.
     *
     * @throws WebApplicationException 400 naming the first field that fails
     */
    static List<WeightUpdateTensor> updateTensors(String bodyText) {
        Object parsed = parse(bodyText, "the request body", "an update tensor or a list of them is required");
        if (parsed instanceof Map && ((Map<?, ?>) parsed).containsKey("update_info")) {
            JsonView wrapper = JsonView.of(parsed, "body");
            requireKnownKeys(wrapper, Set.of("update_info"), "the request body",
                    "the only field of the update wrapper is update_info");
            return tensorsOf(wrapper.raw("update_info"), "body", "update_info");
        }
        return tensorsOf(parsed, "body", "");
    }

    /**
     * The tensors of a collective RPC {@code update_info} payload: the serialized tensor object or list itself
     * (vLLM's RPC passes only strings, so the client serializes the payload).
     *
     * @throws WebApplicationException 400 naming the first field that fails
     */
    static List<WeightUpdateTensor> updatePayload(String payload) {
        Object parsed = parse(payload, "the update_info payload", "an update tensor or a list of them is required");
        return tensorsOf(parsed, "the update_info payload", "");
    }

    /**
     * The init info of an HTTP body: {@code {"backend": ...}}, or vLLM's {@code {"init_info": {...}}} wrapper; an
     * empty body names no backend.
     *
     * @return the named backend, or null when the body is empty or carries no backend
     * @throws WebApplicationException 400 naming the first field that fails
     */
    static String initBackend(String bodyText) {
        if (bodyText == null || bodyText.isBlank()) {
            return null;
        }
        JsonView body = ServerJson.parse(bodyText);
        JsonView info = body;
        if (body.has("init_info")) {
            Object inner = body.raw("init_info");
            if (!(inner instanceof Map)) {
                throw ServerJson.badRequest(body.where("init_info") + " must be a JSON object");
            }
            requireKnownKeys(body, Set.of("init_info"), "the request body", "the only field is init_info");
            info = JsonView.of(inner, body.source(), "init_info");
        }
        requireKnownKeys(info, Set.of("backend"), "the init info", "the init fields are backend");
        return optBackend(info);
    }

    /**
     * The backend a collective RPC init payload names: the serialized init-info object
     * ({@code {"backend": ...}}) or a bare backend name.
     *
     * @param payload the string the RPC carried, or null
     * @return the named backend, or null when the payload is null or blank
     * @throws WebApplicationException 400 naming the first field that fails
     */
    static String initPayload(String payload) {
        if (payload == null || payload.isBlank()) {
            return null;
        }
        String text = payload.trim();
        if (!text.startsWith("{")) {
            return text;
        }
        JsonView info = jsonObject(text, "the init_info payload");
        requireKnownKeys(info, Set.of("backend"), "the init info", "the init fields are backend");
        return optBackend(info);
    }

    /** The optional {@code backend} string of an init-info object, a wrongly typed one refusing as a 400. */
    private static String optBackend(JsonView info) {
        try {
            return info.optStr("backend", null);
        } catch (RuntimeException e) {
            throw ServerJson.badRequest(info.where("backend") + ": " + e.getMessage());
        }
    }

    /**
     * The optional weight version an HTTP body names under {@code field}; an empty body or a blank value gives
     * null, and any other member refuses by name.
     */
    static String optionalVersion(String bodyText, String field) {
        if (bodyText == null || bodyText.isBlank()) {
            return null;
        }
        JsonView view = ServerJson.parse(bodyText);
        requireKnownKeys(view, Set.of(field), "the body", "the field is " + field);
        try {
            String value = view.optStr(field, null);
            return value == null || value.isBlank() ? null : value.trim();
        } catch (RuntimeException e) {
            throw ServerJson.badRequest(view.where(field) + ": " + e.getMessage());
        }
    }

    /**
     * The version a collective RPC payload names: the serialized body object ({@code {"<field>": ...}}) or a bare
     * version string.
     *
     * @param required whether a missing or blank payload refuses
     * @throws WebApplicationException 400 naming the field when it is required and absent
     */
    static String versionPayload(String payload, String field, boolean required) {
        if (payload == null || payload.isBlank()) {
            if (required) {
                throw ServerJson.badRequest("the " + field + " payload is empty; the field is " + field);
            }
            return null;
        }
        String text = payload.trim();
        if (!text.startsWith("{")) {
            return text;
        }
        JsonView view = jsonObject(text, "the " + field + " payload");
        requireKnownKeys(view, Set.of(field), "the " + field + " payload", "the field is " + field);
        String value = requiredStr(view, field, "the " + field + " payload");
        return value.trim();
    }

    /** The required non-blank string member of {@code view}. */
    static String requiredStr(JsonView view, String key, String what) {
        String value;
        try {
            value = view.optStr(key, null);
        } catch (RuntimeException e) {
            throw ServerJson.badRequest(view.where(key) + ": " + e.getMessage());
        }
        if (value == null || value.isBlank()) {
            throw ServerJson.badRequest(what + " carries no '" + key + "'");
        }
        return value;
    }

    /** Refuses, naming it, every member of {@code view} outside {@code allowed}. */
    static void requireKnownKeys(JsonView view, Set<String> allowed, String what, String hint) {
        for (String key : view.keys()) {
            if (!allowed.contains(key)) {
                throw ServerJson.badRequest(what + " carries the unsupported field '" + key + "': " + hint);
            }
        }
    }

    /** Parses {@code text} as a JSON object, naming {@code what} in a 400. */
    private static JsonView jsonObject(String text, String what) {
        return objectView(parse(text, what, "a JSON object is required"), what, "");
    }

    /** Parses {@code text}, naming {@code what} in a 400 for a blank or invalid body. */
    private static Object parse(String text, String what, String emptyHint) {
        if (text == null || text.isBlank()) {
            throw ServerJson.badRequest(what + " is empty; " + emptyHint);
        }
        try {
            return Json.parse(text, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw ServerJson.badRequest("invalid JSON in " + what + ": " + e.getMessage());
        }
    }

    /** The tensors of a parsed payload: a list of tensor objects, or one tensor object. */
    private static List<WeightUpdateTensor> tensorsOf(Object parsed, String source, String path) {
        if (parsed instanceof List<?> items) {
            if (items.isEmpty()) {
                throw ServerJson.badRequest(where(source, path) + " carries no tensors");
            }
            List<WeightUpdateTensor> out = new ArrayList<>(items.size());
            for (int i = 0; i < items.size(); i++) {
                out.add(tensor(objectView(items.get(i), source, elementPath(path, i))));
            }
            return out;
        }
        return List.of(tensor(objectView(parsed, source, path)));
    }

    /** {@code value} as an object view, a non-object value refusing as a 400 naming its path. */
    private static JsonView objectView(Object value, String source, String path) {
        try {
            return JsonView.of(value, source, path);
        } catch (RuntimeException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** One update tensor of the body: its name, dtype, ggml shape and base64 data. */
    private static WeightUpdateTensor tensor(JsonView view) {
        requireKnownKeys(view, TENSOR_FIELDS, "the update tensor",
                "the fields of an update tensor are name, dtype, ne and data_b64");
        String name = requiredStr(view, "name", "the update tensor");
        String dtype = requiredStr(view, "dtype", "the update tensor " + name);
        long[] ne = shape(view, name);
        byte[] data = data(view, name);
        try {
            return WeightUpdateTensor.of(name, WeightUpdateTensor.parseDtype(dtype, view.where("dtype")), ne, data);
        } catch (ArithmeticException e) {
            throw ServerJson.badRequest("the update tensor " + name + " has a shape too large for its byte count");
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** The tensor's ggml shape: the {@code ne} array, every entry a positive whole number. */
    private static long[] shape(JsonView view, String name) {
        Object raw = view.raw("ne");
        if (!(raw instanceof List<?> dims)) {
            throw ServerJson.badRequest("the update tensor " + name + " carries no 'ne' array: the field is its"
                    + " ggml shape, ne[0] first");
        }
        if (dims.isEmpty()) {
            throw ServerJson.badRequest(view.where("ne") + " of the update tensor " + name + " is empty");
        }
        long[] ne = new long[dims.size()];
        for (int i = 0; i < ne.length; i++) {
            Object dim = dims.get(i);
            if (dim instanceof Number n && n.doubleValue() == Math.floor(n.doubleValue())) {
                ne[i] = n.longValue();
            } else {
                throw ServerJson.badRequest(view.where("ne") + "[" + i + "] of the update tensor " + name
                        + " must be a whole number, got " + dim);
            }
        }
        return ne;
    }

    /** The tensor's bytes, base64-decoded from {@code data_b64}. */
    private static byte[] data(JsonView view, String name) {
        String encoded = requiredStr(view, "data_b64", "the update tensor " + name);
        try {
            return Base64.getDecoder().decode(encoded);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(view.where("data_b64") + " of the update tensor " + name
                    + " is not valid base64: " + e.getMessage());
        }
    }

    private static String elementPath(String path, int index) {
        return path.isEmpty() ? "[" + index + "]" : path + "[" + index + "]";
    }

    private static String where(String source, String path) {
        return path.isEmpty() ? source : source + ":" + path;
    }
}
