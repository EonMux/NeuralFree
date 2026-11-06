package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.demo.tools.CvectorGenerator;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Control vectors served over the real engine ({@code neuralfree.control-vector} on the server side): the service
 * opened with a control vector completes prompts through the REST surface, and its greedy completions differ from
 * the base engine's, because the loaded vector's adds reach every decode step.
 */
class EngineServiceControlVectorTest {

    private static EngineService base;
    private static EngineService controlled;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        float[][] rows = new float[2][];
        for (int il = 0; il < 2; il++) {
            Random r = new Random(700 + il);
            float[] row = new float[64];
            for (int i = 0; i < 64; i++) {
                row[i] = 0.2f * (r.nextFloat() - 0.5f);
            }
            rows[il] = row;
        }
        Path vector = CvectorGenerator.export(Arrays.asList(rows), 64, "llama",
                model.resolveSibling("cv-server.gguf"), true);

        base = new EngineService(model.toString(), config().build());
        controlled = new EngineService(model.toString(),
                config().addControlVector(vector.toString(), 2.0f).build());
        resource = new OpenAiResource(controlled);
    }

    @AfterAll
    static void close() {
        base.close();
        controlled.close();
    }

    private static EngineConfig.Builder config() {
        return EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(64).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build());
    }

    @Test
    void theControlledEngineAnswersCompletions() {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 8, "temperature": 0}""");
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertTrue(!firstChoice(body).str("text").isEmpty());
    }

    @Test
    void theVectorChangesTheGreedyCompletionOfTheBaseEngine() {
        assertNotEquals(completion(base), completion(controlled));
    }

    /** Greedy completion text of the OpenAI surface for one prompt. */
    private static String completion(EngineService service) {
        OpenAiResource r = new OpenAiResource(service);
        Response response = r.completions("""
                {"prompt": "The cat sat on the", "max_tokens": 16, "temperature": 0}""");
        assertEquals(200, response.getStatus());
        return firstChoice(view(response.getEntity())).str("text");
    }

    private static JsonView firstChoice(JsonView body) {
        return JsonView.of(((java.util.List<?>) body.raw("choices")).get(0), "choices[0]");
    }

    private static JsonView view(Object entity) {
        return JsonView.of(Json.parse(String.valueOf(entity), Json.Limits.UNTRUSTED), "body");
    }
}

