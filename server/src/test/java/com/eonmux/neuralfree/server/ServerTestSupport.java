package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.mm.ClipDualEncoder;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** Shared helpers of the server tests: the paged test config, JSON views, statuses and SSE frames. */
final class ServerTestSupport {

    private ServerTestSupport() {
    }

    /**
     * The engine backend the tests run on: {@code neuralfree.backend} when the run sets it
     * ({@code ./gradlew :server:test -Pneuralfree.backend=cuda}), auto (the GPU when one is present) otherwise.
     */
    static EngineConfig.BackendMode backend() {
        return EngineConfig.BackendMode.parse(System.getProperty(EngineConfig.PROP_BACKEND, "auto"));
    }

    /** A config builder on the tests' backend ({@link #backend()}). */
    static EngineConfig.Builder config() {
        return EngineConfig.builder().backend(backend());
    }

    /**
     * The dual encoder at {@code path} opened as a service on {@code config} opens it: its load options and the
     * towers on the device the engine backend chooses, so its embeddings are the service's own.
     */
    static ClipDualEncoder dualReference(String path, EngineConfig config) throws IOException {
        return ClipDualEncoder.open(path, config.sourceOptions(),
                EngineService.onEngineBackend(MediaOptions.none(), config).encoder());
    }

    /** A paged config with {@code parallel} slots over an F32 KV cache of {@code nCtx} positions. */
    static EngineConfig pagedConfig(int nCtx, int parallel) {
        return config()
                .memory(MemoryConfig.builder()
                        .nCtx(nCtx).ubatch(8).parallel(parallel).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** A JSON response entity as a view. */
    static JsonView view(Object entity) {
        return JsonView.of(Json.parse(String.valueOf(entity), Json.Limits.UNTRUSTED), "body");
    }

    /** A JSON response entity as parsed, whatever its top-level shape. */
    static Object parsed(Object entity) {
        return Json.parse(String.valueOf(entity), Json.Limits.UNTRUSTED);
    }

    /** The {@code i}-th entry of a JSON array member. */
    static JsonView at(JsonView body, String key, int i) {
        return JsonView.of(((List<?>) body.raw(key)).get(i), key + "[" + i + "]");
    }

    /** The size of a JSON array member. */
    static int size(JsonView body, String key) {
        return ((List<?>) body.raw(key)).size();
    }

    /** The status a call answers: 200, or the status of the WebApplicationException it raised. */
    static int statusOf(Runnable call) {
        try {
            call.run();
            return 200;
        } catch (WebApplicationException e) {
            return e.getResponse().getStatus();
        }
    }

    /** The error message of a refused call. */
    static String errorOf(Runnable call) {
        try {
            call.run();
        } catch (WebApplicationException e) {
            return view(e.getResponse().getEntity()).obj("error").str("message");
        }
        throw new AssertionError("the call was not refused");
    }

    /** The full text of a streamed response. */
    static String streamText(Response response) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ((StreamingOutput) response.getEntity()).write(out);
        return out.toString(StandardCharsets.UTF_8);
    }

    /** The JSON payloads of an OpenAI stream's {@code data:} frames, the terminal {@code [DONE]} left out. */
    static List<JsonView> dataFrames(String stream) {
        List<JsonView> out = new ArrayList<>();
        for (String frame : stream.split("\n\n")) {
            for (String line : frame.split("\n")) {
                if (line.startsWith("data: ") && !line.equals("data: [DONE]")) {
                    out.add(view(line.substring("data: ".length())));
                }
            }
        }
        return out;
    }
}
