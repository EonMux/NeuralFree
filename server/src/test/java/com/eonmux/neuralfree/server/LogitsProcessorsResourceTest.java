package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorArgs;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorFactory;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Custom logits processors on the OpenAI endpoints: the engine loads the class {@code --logits-processors} names
 * (here through {@link EngineConfig#logitsProcessors()}), and a request activates it with its {@code vllm_xargs}
 * arguments. A processor that does not apply returns null and leaves the request alone; one refusing an argument
 * answers 400 naming the processor and the argument.
 */
class LogitsProcessorsResourceTest {

    /** Forces the token {@code vllm_xargs.target_token} when the request sets it, and applies to no other request. */
    public static final class ForceTokenFactory implements LogitsProcessorFactory {

        public ForceTokenFactory() {
        }

        @Override
        public String name() {
            return "force_token";
        }

        @Override
        public void validate(LogitsProcessorArgs args) {
            if (args.has("target_token")) {
                args.integer("target_token");
            }
        }

        @Override
        public SamplerStage create(LogitsProcessorArgs args) {
            if (!args.has("target_token")) {
                return null;
            }
            int target = args.integer("target_token");
            return new SamplerStage() {
                @Override
                public void apply(Sampler.Candidates candidates) {
                    for (int i = 0; i < candidates.size(); i++) {
                        if (candidates.id(i) != target) {
                            candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                        }
                    }
                    candidates.setSorted(false);
                }
            };
        }
    }

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = pagedConfig(256, 2).toBuilder()
                .logitsProcessors(Collections.singletonList(ForceTokenFactory.class.getName()))
                .build();
        service = new EngineService(model.toString(), config);
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void vllmXargsActivateTheProcessorForThatRequestOnly() {
        // the same request without arguments: its deterministic top token is what an ignored processor would emit
        JsonView plain = onlyChoice("{\"prompt\": \"value:\", \"max_tokens\": 4, \"temperature\": 0.5, "
                + "\"top_k\": 1, \"seed\": 7, \"return_token_ids\": true}");
        int natural = ((Number) ((List<?>) plain.raw("token_ids")).get(0)).intValue();
        int target = -1;
        for (int candidate : roundTrippingTokens(8)) {
            if (candidate != natural) {
                target = candidate;
                break;
            }
        }
        assertTrue(target >= 0, "a round-tripping token other than the model's first choice");

        // top_k 1 would prune the forced token if the processor ran after the truncating stages
        JsonView forced = onlyChoice("{\"prompt\": \"value:\", \"max_tokens\": 4, \"temperature\": 0.5, "
                + "\"top_k\": 1, \"seed\": 7, \"return_token_ids\": true, "
                + "\"vllm_xargs\": {\"target_token\": " + target + "}}");
        List<?> ids = (List<?>) forced.raw("token_ids");
        assertEquals(4, ids.size(), "the forced token fills every step");
        for (Object id : ids) {
            assertEquals(target, ((Number) id).intValue(), "every generated token is the forced one");
        }

        // without the argument the processor returns null and the request samples its own way
        assertTrue(plain.raw("token_ids") != null && ((List<?>) plain.raw("token_ids")).size() == 4);
    }

    /** The first choice of a completions response that answers 200. */
    private static JsonView onlyChoice(String body) {
        Response response = resource.completions(body);
        assertEquals(200, response.getStatus());
        return JsonView.of(((List<?>) view(response.getEntity()).raw("choices")).get(0), "choices[0]");
    }

    @Test
    void aRefusedArgumentAnswers400NamingIt() {
        WebApplicationException refused = assertThrows(WebApplicationException.class,
                () -> resource.completions("{\"prompt\": \"value:\", \"max_tokens\": 4, \"seed\": 7, "
                        + "\"vllm_xargs\": {\"target_token\": \"five\"}}"));
        assertEquals(400, refused.getResponse().getStatus());
        String message = view(refused.getResponse().getEntity()).obj("error").str("message");
        assertTrue(message.contains("vllm_xargs: force_token:"), message);
        assertTrue(message.contains("vllm_xargs.target_token must be an integer"), message);
    }

    /** The first {@code n} distinct vocabulary ids whose decoded text re-encodes to the id alone. */
    private static int[] roundTrippingTokens(int n) {
        GGUFTokenizer tokenizer = service.tokenizer();
        int[] out = new int[n];
        int found = 0;
        for (int id = 0; id < tokenizer.vocabSize() && found < n; id++) {
            String text = tokenizer.decode(new int[] {id});
            if (text.isEmpty()) {
                continue;
            }
            int[] encoded = tokenizer.encode(text, false, false);
            if (encoded.length == 1 && encoded[0] == id) {
                out[found++] = id;
            }
        }
        assertEquals(n, found, "found only " + found + " round-tripping tokens");
        return out;
    }
}
