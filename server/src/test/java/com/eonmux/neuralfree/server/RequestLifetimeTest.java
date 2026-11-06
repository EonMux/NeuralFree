package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import io.vertx.core.Handler;
import io.vertx.core.MultiMap;
import io.vertx.core.http.HttpServerRequest;
import io.vertx.core.http.HttpServerResponse;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * How long a request lives: the configurable budget refuses a request that outlives it with 503, and a client whose
 * connection closes cancels its generations ({@link ClientLink}).
 */
class RequestLifetimeTest {

    private static Path model;
    private static EngineService service;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(1024, 2));
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void theBudgetComesFromThePropertyInSeconds() {
        String old = System.getProperty(ServerOptions.TIMEOUT_PROPERTY);
        try {
            System.setProperty(ServerOptions.TIMEOUT_PROPERTY, " 90 ");
            assertEquals(Duration.ofSeconds(90), ServerOptions.fromSystemProperties().requestTimeout());
            System.setProperty(ServerOptions.TIMEOUT_PROPERTY, "soon");
            assertThrows(IllegalArgumentException.class, ServerOptions::fromSystemProperties);
            System.setProperty(ServerOptions.TIMEOUT_PROPERTY, "0");
            assertThrows(IllegalArgumentException.class, ServerOptions::fromSystemProperties);
            System.clearProperty(ServerOptions.TIMEOUT_PROPERTY);
            assertEquals(ServerOptions.DEFAULT_TIMEOUT, ServerOptions.fromSystemProperties().requestTimeout());
        } finally {
            if (old == null) {
                System.clearProperty(ServerOptions.TIMEOUT_PROPERTY);
            } else {
                System.setProperty(ServerOptions.TIMEOUT_PROPERTY, old);
            }
        }
    }

    @Test
    void aRequestOutlivingTheBudgetIsRefusedWith503() throws IOException {
        EngineService tight = new EngineService(model.toString(), ServerTestSupport.pagedConfig(1024, 2),
                new ServerOptions(null, Duration.ofMillis(1)));
        try {
            OpenAiResource openAi = new OpenAiResource(tight);
            String body = """
                    {"prompt": "hello", "max_tokens": 400, "ignore_eos": true}""";
            assertEquals(503, statusOf(() -> openAi.completions(body)));
            assertTrue(errorOf(() -> openAi.completions(body)).contains("budget"));
        } finally {
            tight.close();
        }
    }

    @Test
    void aClosedConnectionCancelsTheGenerations() {
        AtomicReference<Handler<Void>> onClose = new AtomicReference<>();
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(800).ignoreEos(true).build();
        GenerationBatch batch = GenerationBatch.start(service, List.of(
                new GenerationBatch.Job(service.encode("hello", true, false), request, null, null, null)),
                client(onClose));
        assertNotNull(onClose.get(), "the batch registered a close handler");
        onClose.get().handle(null);
        batch.awaitAll();
        assertEquals(StopReason.CANCELLED, batch.result(0).stopReason());
        assertTrue(batch.result(0).timings().generatedTokens() < 800);
    }

    /** An HTTP request whose response records the close handler it is given. */
    private static HttpServerRequest client(AtomicReference<Handler<Void>> onClose) {
        HttpServerResponse response = (HttpServerResponse) Proxy.newProxyInstance(
                RequestLifetimeTest.class.getClassLoader(), new Class<?>[]{HttpServerResponse.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("closeHandler")) {
                        @SuppressWarnings("unchecked")
                        Handler<Void> handler = (Handler<Void>) args[0];
                        onClose.set(handler);
                        return proxy;
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
        return (HttpServerRequest) Proxy.newProxyInstance(RequestLifetimeTest.class.getClassLoader(),
                new Class<?>[]{HttpServerRequest.class}, (proxy, method, args) -> {
                    if (method.getName().equals("response")) {
                        return response;
                    }
                    if (method.getName().equals("headers")) {
                        return MultiMap.caseInsensitiveMultiMap();
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
    }
}
