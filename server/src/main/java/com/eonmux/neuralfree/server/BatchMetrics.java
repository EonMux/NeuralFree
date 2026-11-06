package com.eonmux.neuralfree.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.micrometer.core.instrument.binder.jvm.JvmGcMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmMemoryMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmThreadMetrics;
import io.micrometer.core.instrument.binder.system.ProcessorMetrics;
import io.micrometer.core.instrument.binder.system.UptimeMetrics;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.function.Supplier;

/**
 * The batch runner's Prometheus metrics server (vLLM's {@code run-batch --enable-metrics}): the server's
 * {@link EngineMetrics} meters and the JVM's, in a Prometheus registry of their own, served by the JDK's HTTP server
 * on every path, as {@code prometheus_client.start_http_server} serves them, without Quarkus. The meters read zero
 * until the engine opens; the series that follow the opened engine's shape (its replicas, the multimodal cache
 * counters tagged with its model name) register when the runner reports it open ({@link #opened}). When API keys
 * are configured ({@link ApiKeys}) a scrape must present one, as the server's
 * {@code /metrics} requires ({@link MetricsGuard}); without one it answers 401.
 */
final class BatchMetrics implements AutoCloseable {

    /** The Prometheus text exposition format the registry scrapes to. */
    static final String CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8";

    private final HttpServer server;
    private final PrometheusMeterRegistry registry;
    /** The engine meters' binder; held here, since micrometer reads it through a weak reference. */
    private final EngineMetrics engine;
    private final JvmGcMetrics gc;

    private BatchMetrics(HttpServer server, PrometheusMeterRegistry registry, EngineMetrics engine, JvmGcMetrics gc) {
        this.server = server;
        this.registry = registry;
        this.engine = engine;
        this.gc = gc;
    }

    /**
     * Starts serving on {@code host:port} ({@code port} 0 picks a free one) the meters over the service
     * {@code current} holds.
     *
     * @throws IOException when the address cannot be bound, naming it
     * @throws IllegalStateException when the API key configuration cannot be read
     */
    static BatchMetrics start(String host, int port, Supplier<EngineService> current) throws IOException {
        ApiKeys keys = ApiKeys.from(System.getenv(), System.getProperties());
        InetSocketAddress address = new InetSocketAddress(host, port);
        if (address.isUnresolved()) {
            throw new IOException("the metrics --host " + host + " does not resolve");
        }
        HttpServer server;
        try {
            server = HttpServer.create(address, 0);
        } catch (IOException e) {
            throw new IOException("cannot serve metrics on " + host + ":" + port + ": " + e.getMessage(), e);
        }
        try {
            PrometheusMeterRegistry registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
            EngineMetrics engine = new EngineMetrics(current);
            JvmGcMetrics gc = bind(registry, engine);
            server.createContext("/", exchange -> answer(exchange, keys, registry));
            server.start();
            return new BatchMetrics(server, registry, engine, gc);
        } catch (RuntimeException e) {
            server.stop(0);
            throw e;
        }
    }

    /** Binds the engine's and the JVM's meters; returns the GC binder, which holds listeners until closed. */
    private static JvmGcMetrics bind(PrometheusMeterRegistry registry, EngineMetrics engine) {
        engine.bindTo(registry);
        new JvmMemoryMetrics().bindTo(registry);
        new JvmThreadMetrics().bindTo(registry);
        new ProcessorMetrics().bindTo(registry);
        new UptimeMetrics().bindTo(registry);
        JvmGcMetrics gc = new JvmGcMetrics();
        gc.bindTo(registry);
        return gc;
    }

    /**
     * Registers the series of the opened {@code service}'s shape: its replicas' and vLLM's multimodal cache counters
     * ({@link EngineMetrics#bindOpened}). Called once, when the runner opens its engine.
     */
    void opened(EngineService service) {
        engine.bindOpened(registry, service);
    }

    /** The bound port. */
    int port() {
        return server.getAddress().getPort();
    }

    /** One scrape: GET or HEAD with an accepted key (when keys are configured) answers the registry's text. */
    private static void answer(HttpExchange exchange, ApiKeys keys, PrometheusMeterRegistry registry)
            throws IOException {
        try (exchange) {
            String method = exchange.getRequestMethod();
            if (!method.equals("GET") && !method.equals("HEAD")) {
                exchange.getResponseHeaders().set("Allow", "GET, HEAD");
                exchange.sendResponseHeaders(405, -1);
                return;
            }
            if (keys.enabled() && !keys.accepts(ApiKeys.presented(
                    exchange.getRequestHeaders().getFirst("Authorization"),
                    exchange.getRequestHeaders().getFirst("x-api-key")))) {
                exchange.getResponseHeaders().set("WWW-Authenticate", "Bearer");
                send(exchange, 401, "application/json", ApiKeys.refusalBody(), method);
                return;
            }
            send(exchange, 200, CONTENT_TYPE, registry.scrape(), method);
        }
    }

    private static void send(HttpExchange exchange, int status, String type, String body, String method)
            throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", type);
        if (method.equals("HEAD")) {
            exchange.sendResponseHeaders(status, -1);
            return;
        }
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    /** Stops serving and releases the meters. */
    @Override
    public void close() {
        server.stop(0);
        gc.close();
        registry.close();
    }
}
