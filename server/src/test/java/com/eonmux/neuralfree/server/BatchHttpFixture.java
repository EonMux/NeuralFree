package com.eonmux.neuralfree.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * A local HTTP server for the batch file tests: each path answers a queue of scripted replies (the last one
 * repeats), and every request is recorded with its method, its path with the query and its body.
 */
final class BatchHttpFixture implements AutoCloseable {

    /** One scripted answer: a status, a body, and a delay before the answer. */
    record Reply(int status, String body, long delayMillis) {
        static Reply ok(String body) {
            return new Reply(200, body, 0);
        }

        static Reply status(int status) {
            return new Reply(status, "scripted failure " + status, 0);
        }
    }

    /** One received request. */
    record Received(String method, String pathAndQuery, String body) { }

    private final HttpServer server;
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private final Map<String, Deque<Reply>> replies = new ConcurrentHashMap<>();
    private final List<Received> received = new CopyOnWriteArrayList<>();

    BatchHttpFixture() throws IOException {
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.setExecutor(executor);
        server.createContext("/", this::handle);
        server.start();
    }

    /** Scripts the replies of {@code path}, in order; the last one answers every further request. */
    BatchHttpFixture on(String path, Reply... scripted) {
        replies.put(path, new ArrayDeque<>(List.of(scripted)));
        return this;
    }

    /** The URL of {@code pathAndQuery} on this server. */
    String url(String pathAndQuery) {
        return "http://127.0.0.1:" + server.getAddress().getPort() + pathAndQuery;
    }

    /** The requests received so far. */
    List<Received> received() {
        return List.copyOf(received);
    }

    private void handle(HttpExchange exchange) throws IOException {
        try (exchange) {
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String query = exchange.getRequestURI().getRawQuery();
            String path = exchange.getRequestURI().getPath();
            received.add(new Received(exchange.getRequestMethod(), path + (query == null ? "" : "?" + query), body));
            Reply reply = next(path);
            if (reply.delayMillis() > 0) {
                Thread.sleep(reply.delayMillis());
            }
            byte[] bytes = reply.body().getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(reply.status(), bytes.length == 0 ? -1 : bytes.length);
            if (bytes.length > 0) {
                try (OutputStream out = exchange.getResponseBody()) {
                    out.write(bytes);
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (IOException e) {
            // the client gave up (a timeout test); nothing is left to answer
        }
    }

    private Reply next(String path) {
        Deque<Reply> queue = replies.get(path);
        if (queue == null) {
            return Reply.status(404);
        }
        synchronized (queue) {
            return queue.size() > 1 ? queue.poll() : queue.peek();
        }
    }

    @Override
    public void close() {
        server.stop(0);
        executor.shutdownNow();
    }
}
