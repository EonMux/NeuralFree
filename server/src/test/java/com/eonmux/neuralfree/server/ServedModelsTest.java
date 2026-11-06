package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Served model names: configured names are listed, echoed when a request asks for one and enforced (404 for any
 * other); without them the model path is the name and any requested name is accepted.
 */
class ServedModelsTest {

    private static EngineService service;
    private static OpenAiResource openAi;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2),
                new ServerOptions(" alpha, beta ,alpha ", ServerOptions.DEFAULT_TIMEOUT));
        openAi = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void theListTrimsAndDropsDuplicatesAndBlankListsFallBackToThePath() {
        assertEquals(List.of("alpha", "beta"), ServedModels.of("alpha,,beta,alpha", "/m.gguf").names());
        assertEquals(List.of("/m.gguf"), ServedModels.of("  ", "/m.gguf").names());
        assertEquals(List.of("/m.gguf"), ServedModels.of(null, "/m.gguf").names());
        assertThrows(IllegalArgumentException.class, () -> ServedModels.of(" , ", "/m.gguf"));
    }

    @Test
    void withoutConfiguredNamesAnyRequestedModelIsAccepted() {
        ServedModels path = ServedModels.of(null, "/m.gguf");
        assertEquals("/m.gguf", path.resolve(view("{\"model\": \"gpt-4o\"}")));
        assertEquals("/m.gguf", path.resolve(view("{}")));
    }

    @Test
    void everyNameIsListedAndTheRequestedOneIsEchoed() {
        JsonView models = view(openAi.models(null).getEntity());
        assertEquals("alpha", at(models, "data", 0).str("id"));
        assertEquals("beta", at(models, "data", 1).str("id"));
        assertEquals(service.modelPath(), at(models, "data", 1).str("root"));
        assertEquals("beta", view(openAi.completions("""
                {"model": "beta", "prompt": "hi", "max_tokens": 1}""").getEntity()).str("model"));
        assertEquals("alpha", view(openAi.completions("""
                {"prompt": "hi", "max_tokens": 1}""").getEntity()).str("model"));
    }

    @Test
    void anUnservedModelIsNotFound() {
        assertEquals(404, statusOf(() -> openAi.chatCompletions("""
                {"model": "gamma", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}""")));
        assertEquals(404, statusOf(() -> new PoolingResource(service).embeddings("""
                {"model": "gamma", "input": "hi"}""")));
    }
}
