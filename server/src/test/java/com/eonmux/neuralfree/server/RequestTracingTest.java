package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.propagation.TextMapGetter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.trace.ReadWriteSpan;
import io.opentelemetry.sdk.trace.ReadableSpan;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.SpanProcessor;
import io.opentelemetry.sdk.trace.data.SpanData;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The per-request OpenTelemetry trace: a generation batch under {@code --otlp-traces-endpoint} emits one
 * {@code llm_request} server span joined to the caller's W3C trace context, carrying the {@code gen_ai.*} token
 * counts, latencies and sampler parameters; a failed batch marks the span ERROR; with tracing off no span is
 * emitted and a request with trace headers warns once.
 */
class RequestTracingTest {

    private static final AttributeKey<Long> PROMPT_TOKENS = AttributeKey.longKey("gen_ai.usage.prompt_tokens");
    private static final AttributeKey<Long> COMPLETION_TOKENS =
            AttributeKey.longKey("gen_ai.usage.completion_tokens");
    private static final AttributeKey<Long> MAX_TOKENS = AttributeKey.longKey("gen_ai.request.max_tokens");
    private static final AttributeKey<Long> N = AttributeKey.longKey("gen_ai.request.n");
    private static final AttributeKey<Double> TOP_P = AttributeKey.doubleKey("gen_ai.request.top_p");
    private static final AttributeKey<Double> TEMPERATURE = AttributeKey.doubleKey("gen_ai.request.temperature");
    private static final AttributeKey<Double> E2E = AttributeKey.doubleKey("gen_ai.latency.e2e");
    private static final AttributeKey<Double> TIME_IN_QUEUE = AttributeKey.doubleKey("gen_ai.latency.time_in_queue");
    private static final AttributeKey<Double> TIME_TO_FIRST_TOKEN =
            AttributeKey.doubleKey("gen_ai.latency.time_to_first_token");
    private static final AttributeKey<Double> MODEL_PREFILL =
            AttributeKey.doubleKey("gen_ai.latency.time_in_model_prefill");
    private static final AttributeKey<Double> MODEL_DECODE =
            AttributeKey.doubleKey("gen_ai.latency.time_in_model_decode");
    private static final AttributeKey<Double> MODEL_INFERENCE =
            AttributeKey.doubleKey("gen_ai.latency.time_in_model_inference");
    private static final AttributeKey<String> REQUEST_ID = AttributeKey.stringKey("gen_ai.request.id");
    private static final AttributeKey<String> RESPONSE_MODEL = AttributeKey.stringKey("gen_ai.response.model");

    /** traceparent of a caller trace the span must join. */
    private static final String TRACEPARENT =
            "00-47e35574e34a4b3d8e32f6f4b129cc56-3d92c8a70a1fb51a-01";
    private static final String TRACE_ID = "47e35574e34a4b3d8e32f6f4b129cc56";

    /** Every span the test's tracer provider saw, in end order. */
    private static final List<SpanData> SPANS = new CopyOnWriteArrayList<>();
    private static final SpanProcessor COLLECTOR = new SpanProcessor() {
        @Override
        public void onStart(Context context, ReadWriteSpan span) {
        }

        @Override
        public boolean isStartRequired() {
            return false;
        }

        @Override
        public void onEnd(ReadableSpan span) {
            SPANS.add(span.toSpanData());
        }

        @Override
        public boolean isEndRequired() {
            return true;
        }
    };

    private static Path model;
    private static EngineService traced;
    private static EngineService untraced;
    private static SdkTracerProvider provider;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        traced = new EngineService(model.toString(), pagedConfig(1024, 2),
                ServerOptions.defaults().withOtlpTracesEndpoint("http://127.0.0.1:4318"));
        untraced = new EngineService(model.toString(), pagedConfig(1024, 2));
        provider = SdkTracerProvider.builder().addSpanProcessor(COLLECTOR).build();
        GlobalOpenTelemetry.resetForTest();
        GlobalOpenTelemetry.set(OpenTelemetrySdk.builder().setTracerProvider(provider).build());
    }

    @AfterAll
    static void close() {
        GlobalOpenTelemetry.resetForTest();
        provider.close();
        traced.close();
        untraced.close();
    }

    @Test
    void aFinishedRequestEmitsTheServerSpanWithTheEngineAttributes() throws InterruptedException {
        int before = SPANS.size();
        int[] prompt = traced.encode("hello", true, false);
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(4).ignoreEos(true).build();
        MultiMap headers = MultiMap.caseInsensitiveMultiMap().add("traceparent", TRACEPARENT);
        GenerationBatch batch = GenerationBatch.start(traced, List.of(
                new GenerationBatch.Job(prompt, request, null, null, null)),
                client(headers), "chatcmpl-test-finished");
        batch.awaitAll();
        SpanData span = spanOf(before, "chatcmpl-test-finished");

        assertEquals("llm_request", span.getName());
        assertEquals(SpanKind.SERVER, span.getKind());
        assertTrue(TRACE_ID.equals(span.getTraceId()), "the span joins the caller's trace, got " + span.getTraceId());
        assertTrue(span.getParentSpanContext().isValid(), "the traceparent headers are the span's parent");
        assertTrue(span.getStartEpochNanos() > 0 && span.getEndEpochNanos() >= span.getStartEpochNanos());

        Map<AttributeKey<?>, Object> attrs = span.getAttributes().asMap();
        assertEquals(prompt.length, (long) attrs.get(PROMPT_TOKENS));
        assertEquals(batch.result(0).timings().generatedTokens(), (long) attrs.get(COMPLETION_TOKENS));
        assertEquals(4L, (long) attrs.get(MAX_TOKENS));
        assertEquals(1L, (long) attrs.get(N));
        assertEquals("chatcmpl-test-finished", attrs.get(REQUEST_ID));
        assertEquals(model.toString(), attrs.get(RESPONSE_MODEL));
        assertNull(attrs.get(TOP_P), "a parameter the request left to the defaults stays off the span");
        assertNull(attrs.get(TEMPERATURE));

        assertTrue((double) attrs.get(E2E) > 0);
        assertTrue((double) attrs.get(TIME_IN_QUEUE) >= 0);
        assertTrue((double) attrs.get(TIME_TO_FIRST_TOKEN) >= 0);
        assertTrue((double) attrs.get(MODEL_PREFILL) >= 0);
        assertTrue((double) attrs.get(MODEL_DECODE) >= 0);
        assertTrue((double) attrs.get(MODEL_INFERENCE) >= 0);
        assertTrue((double) attrs.get(TIME_IN_QUEUE) <= (double) attrs.get(E2E));
        assertTrue((double) attrs.get(TIME_TO_FIRST_TOKEN) <= (double) attrs.get(E2E));
        assertEquals(StatusCode.UNSET, span.getStatus().getStatusCode());
        assertTrue(span.getEvents().isEmpty(), "a served request records no exception");
    }

    @Test
    void theSpanOfTwoChoicesCountsBothAndReportsN() throws InterruptedException {
        int before = SPANS.size();
        int[] prompt = traced.encode("hello", true, false);
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(4).ignoreEos(true).build();
        GenerationBatch batch = GenerationBatch.start(traced, List.of(
                new GenerationBatch.Job(prompt, request, null, null, null),
                new GenerationBatch.Job(prompt, request, null, null, null)),
                client(null), "chatcmpl-test-two");
        batch.awaitAll();
        SpanData span = spanOf(before, "chatcmpl-test-two");

        assertEquals(prompt.length * 2, (long) span.getAttributes().asMap().get(PROMPT_TOKENS));
        assertEquals(batch.result(0).timings().generatedTokens() + batch.result(1).timings().generatedTokens(),
                (long) span.getAttributes().asMap().get(COMPLETION_TOKENS));
        assertEquals(2L, (long) span.getAttributes().asMap().get(N));
    }

    @Test
    void aRefusedChoiceMarksTheSpanError() throws InterruptedException {
        int before = SPANS.size();
        int[] oversize = new int[1100];
        Arrays.fill(oversize, traced.encode("a", true, false)[0]);
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(4).build();
        GenerationBatch batch = GenerationBatch.start(traced, List.of(
                new GenerationBatch.Job(oversize, request, null, null, null)),
                client(null), "chatcmpl-test-refused");
        batch.awaitAll();
        SpanData span = spanOf(before, "chatcmpl-test-refused");

        assertEquals(StatusCode.ERROR, span.getStatus().getStatusCode());
        assertTrue(span.getEvents().size() >= 1, "the refusal is recorded on the span");
        assertEquals(1100L, (long) span.getAttributes().asMap().get(PROMPT_TOKENS));
        assertEquals(0L, (long) span.getAttributes().asMap().get(COMPLETION_TOKENS));
        assertNull(span.getAttributes().asMap().get(TIME_TO_FIRST_TOKEN), "a refused choice samples no token");
    }

    @Test
    void theResourcePassesItsResponseIdToTheSpan() throws Exception {
        int before = SPANS.size();
        OpenAiResource openAi = new OpenAiResource(traced);
        openAi.completions("{\"prompt\": \"hello\", \"max_tokens\": 2}");
        SpanData span = spanOf(before, "cmpl-");
        assertEquals("llm_request", span.getName());
    }

    @Test
    void tracingOffEmitsNoSpanAndWarnsOnceAboutTraceContext() throws InterruptedException {
        RequestTracing.resetWarnings();
        int before = SPANS.size();
        Logger logger = Logger.getLogger(RequestTracing.class.getName());
        List<LogRecord> records = new ArrayList<>();
        java.util.logging.Handler handler = new java.util.logging.Handler() {
            @Override public void publish(LogRecord record) { records.add(record); }
            @Override public void flush() { }
            @Override public void close() { }
        };
        logger.addHandler(handler);
        try {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(2).ignoreEos(true).build();
            for (int round = 0; round < 2; round++) {
                GenerationBatch batch = GenerationBatch.start(untraced, List.of(
                        new GenerationBatch.Job(traced.encode("hello", true, false), request, null, null, null)),
                        client(MultiMap.caseInsensitiveMultiMap().add("traceparent", TRACEPARENT)), null);
                batch.awaitAll();
            }
            assertEquals(before, SPANS.size(), "with tracing off no span is emitted");
            assertEquals(1, records.size(), "the first trace-context request warns, later ones stay quiet");
            assertEquals(Level.WARNING, records.get(0).getLevel());
            assertTrue(records.get(0).getMessage().contains("tracing is disabled"));
            assertTrue(records.get(0).getMessage().contains(ServerOptions.OTLP_TRACES_ENDPOINT_FLAG));
        } finally {
            logger.removeHandler(handler);
            RequestTracing.resetWarnings();
        }
    }

    @Test
    void theEndpointPropertyParsesAndCopies() {
        String old = System.getProperty(ServerOptions.OTLP_TRACES_ENDPOINT_PROPERTY);
        try {
            assertNull(ServerOptions.fromSystemProperties().otlpTracesEndpoint());
            System.setProperty(ServerOptions.OTLP_TRACES_ENDPOINT_PROPERTY, " collector.example:4317 ");
            assertEquals("collector.example:4317", ServerOptions.fromSystemProperties().otlpTracesEndpoint());
            System.setProperty(ServerOptions.OTLP_TRACES_ENDPOINT_PROPERTY, " ");
            assertNull(ServerOptions.fromSystemProperties().otlpTracesEndpoint());
        } finally {
            if (old == null) {
                System.clearProperty(ServerOptions.OTLP_TRACES_ENDPOINT_PROPERTY);
            } else {
                System.setProperty(ServerOptions.OTLP_TRACES_ENDPOINT_PROPERTY, old);
            }
        }
        assertEquals("collector.example:4317",
                ServerOptions.defaults().withOtlpTracesEndpoint(" collector.example:4317 ").otlpTracesEndpoint());
        assertNull(ServerOptions.defaults().withOtlpTracesEndpoint(null).otlpTracesEndpoint());
    }

    @Test
    void theW3CGetterReadsTheHeadersCaseInsensitively() {
        MultiMap headers = MultiMap.caseInsensitiveMultiMap().add("Traceparent", TRACEPARENT);
        Context parent = W3CTraceContextPropagator.getInstance().extract(Context.root(), headers, getter());
        assertEquals(TRACE_ID, Span.fromContext(parent).getSpanContext().getTraceId());
        assertTrue(Span.fromContext(parent).getSpanContext().isValid());
        Context empty = W3CTraceContextPropagator.getInstance()
                .extract(Context.root(), MultiMap.caseInsensitiveMultiMap(), getter());
        assertTrue(!Span.fromContext(empty).getSpanContext().isValid(),
                "headers without trace context extract no parent");
    }

    /** The span ended after {@code before} whose {@code gen_ai.request.id} starts with {@code idPrefix}. */
    private static SpanData spanOf(int before, String idPrefix) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 10_000;
        while (System.currentTimeMillis() < deadline) {
            for (int i = before; i < SPANS.size(); i++) {
                SpanData span = SPANS.get(i);
                String id = span.getAttributes().get(REQUEST_ID);
                if (id != null && id.startsWith(idPrefix)) {
                    return span;
                }
            }
            Thread.sleep(10);
        }
        List<String> ids = new ArrayList<>();
        for (int i = before; i < SPANS.size(); i++) {
            ids.add(String.valueOf(SPANS.get(i).getAttributes().get(REQUEST_ID)));
        }
        throw new AssertionError("no span for " + idPrefix + " among " + ids);
    }

    /** The trace-header carrier the test exercises (the case-insensitivity comes from vert.x's MultiMap). */
    private static TextMapGetter<MultiMap> getter() {
        return new TextMapGetter<>() {
            @Override
            public Iterable<String> keys(MultiMap carrier) {
                return carrier == null ? Collections.emptyList() : carrier.names();
            }

            @Override
            public String get(MultiMap carrier, String key) {
                return carrier == null ? null : carrier.get(key);
            }
        };
    }

    /** An HTTP request with {@code headers} whose response records the close handler it is given. */
    private static HttpServerRequest client(MultiMap headers) {
        HttpServerResponse response = (HttpServerResponse) Proxy.newProxyInstance(
                RequestTracingTest.class.getClassLoader(), new Class<?>[]{HttpServerResponse.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("closeHandler")) {
                        return proxy;
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
        return (HttpServerRequest) Proxy.newProxyInstance(RequestTracingTest.class.getClassLoader(),
                new Class<?>[]{HttpServerRequest.class}, (proxy, method, args) -> switch (method.getName()) {
                    case "response" -> response;
                    case "headers" -> headers == null ? MultiMap.caseInsensitiveMultiMap() : headers;
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }
}
