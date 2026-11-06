package com.eonmux.neuralfree.gguf.remote;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A local HTTP server speaking the subset of the S3, GCS and Azure Blob APIs the object-store clients use: one bucket
 * or container of objects, a paginated listing and whole-object downloads. Requests are recorded so a test can count
 * the downloads a mirror repeats.
 */
final class FakeObjectStoreServer implements AutoCloseable {

    enum Kind { S3, GCS, AZURE }

    private final Kind kind;
    private final HttpServer server;
    private final Map<String, byte[]> objects = new ConcurrentHashMap<>();
    private final List<String> requests = Collections.synchronizedList(new ArrayList<>());
    private final List<String> authorizations = Collections.synchronizedList(new ArrayList<>());
    private final AtomicInteger objectGets = new AtomicInteger();
    private volatile int pageSize = 1000;
    private volatile String expectedSas;

    FakeObjectStoreServer(Kind kind) throws IOException {
        this.kind = kind;
        this.server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        this.server.createContext("/", this::handle);
        this.server.setExecutor(null);
        this.server.start();
    }

    String base() {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    void put(String key, byte[] bytes) {
        objects.put(key, bytes.clone());
    }

    void remove(String key) {
        objects.remove(key);
    }

    void pageSize(int value) {
        this.pageSize = value;
    }

    void requireSas(String sas) {
        this.expectedSas = sas;
    }

    int objectGets() {
        return objectGets.get();
    }

    List<String> requests() {
        synchronized (requests) {
            return new ArrayList<>(requests);
        }
    }

    List<String> authorizations() {
        synchronized (authorizations) {
            return new ArrayList<>(authorizations);
        }
    }

    @Override
    public void close() {
        server.stop(0);
    }

    private void handle(HttpExchange exchange) throws IOException {
        try {
            String rawPath = exchange.getRequestURI().getRawPath();
            String rawQuery = exchange.getRequestURI().getRawQuery();
            requests.add(exchange.getRequestMethod() + " " + rawPath + (rawQuery == null ? "" : "?" + rawQuery));
            String authorization = exchange.getRequestHeaders().getFirst("Authorization");
            if (authorization != null) {
                authorizations.add(authorization);
            }
            Map<String, String> query = query(rawQuery);
            switch (kind) {
                case S3:
                    handleS3(exchange, rawPath, query);
                    break;
                case GCS:
                    handleGcs(exchange, rawPath, query);
                    break;
                default:
                    handleAzure(exchange, rawPath, rawQuery, query);
                    break;
            }
        } catch (RuntimeException e) {
            respond(exchange, 500, ("error: " + e).getBytes(StandardCharsets.UTF_8), "text/plain");
        } finally {
            exchange.close();
        }
    }

    // ------------------------------------------------------------------ S3

    private void handleS3(HttpExchange exchange, String rawPath, Map<String, String> query) throws IOException {
        String path = rawPath.startsWith("/") ? rawPath.substring(1) : rawPath;
        int slash = path.indexOf('/');
        String bucket = slash < 0 ? path : path.substring(0, slash);
        String key = slash < 0 ? "" : decode(path.substring(slash + 1));
        if (key.isEmpty() && "2".equals(query.get("list-type"))) {
            listing(exchange, query.get("prefix"), query.get("continuation-token"), true);
            return;
        }
        if (!bucket.equals("bucket") && !bucket.equals("container")) {
            respond(exchange, 404, new byte[0], "text/plain");
            return;
        }
        getObject(exchange, key);
    }

    // ------------------------------------------------------------------ GCS

    private void handleGcs(HttpExchange exchange, String rawPath, Map<String, String> query) throws IOException {
        String prefix = "/storage/v1/b/";
        if (!rawPath.startsWith(prefix)) {
            respond(exchange, 404, new byte[0], "text/plain");
            return;
        }
        String rest = rawPath.substring(prefix.length());
        int slash = rest.indexOf('/');
        String bucket = slash < 0 ? rest : rest.substring(0, slash);
        String after = slash < 0 ? "" : rest.substring(slash + 1);
        if (!bucket.equals("bucket") && !bucket.equals("container")) {
            respond(exchange, 404, new byte[0], "text/plain");
            return;
        }
        if (after.equals("o") || after.equals("o/")) {
            listing(exchange, query.get("prefix"), query.get("pageToken"), false);
            return;
        }
        if (after.startsWith("o/")) {
            getObject(exchange, decode(after.substring(2)));
            return;
        }
        respond(exchange, 404, new byte[0], "text/plain");
    }

    // ------------------------------------------------------------------ Azure

    private void handleAzure(HttpExchange exchange, String rawPath, String rawQuery, Map<String, String> query)
            throws IOException {
        String path = rawPath.startsWith("/") ? rawPath.substring(1) : rawPath;
        int slash = path.indexOf('/');
        String container = slash < 0 ? path : path.substring(0, slash);
        String key = slash < 0 ? "" : decode(path.substring(slash + 1));
        if (!container.equals("container") && !container.equals("bucket")) {
            respond(exchange, 404, new byte[0], "text/plain");
            return;
        }
        String sas = expectedSas;
        if (sas != null && (rawQuery == null || !rawQuery.contains(sas))) {
            respond(exchange, 403, new byte[0], "text/plain");
            return;
        }
        if (key.isEmpty() && "container".equals(query.get("restype")) && "list".equals(query.get("comp"))) {
            listing(exchange, query.get("prefix"), query.get("marker"), true);
            return;
        }
        getObject(exchange, key);
    }

    // ------------------------------------------------------------------ shared handlers

    private void getObject(HttpExchange exchange, String key) throws IOException {
        objectGets.incrementAndGet();
        byte[] bytes = objects.get(key);
        if (bytes == null) {
            respond(exchange, 404, new byte[0], "text/plain");
            return;
        }
        respond(exchange, 200, bytes, "application/octet-stream");
    }

    private void listing(HttpExchange exchange, String prefix, String token, boolean xml) throws IOException {
        String from = prefix == null ? "" : prefix;
        List<String> keys = new ArrayList<>();
        for (String key : new TreeMap<>(objects).keySet()) {
            if (key.startsWith(from)) {
                keys.add(key);
            }
        }
        int offset = 0;
        if (token != null && !token.isEmpty()) {
            try {
                offset = Integer.parseInt(token);
            } catch (NumberFormatException e) {
                offset = 0;
            }
        }
        int end = Math.min(keys.size(), offset + Math.max(1, pageSize));
        List<String> page = keys.subList(Math.min(offset, keys.size()), end);
        String next = end < keys.size() ? Integer.toString(end) : null;
        StringBuilder body = new StringBuilder();
        if (xml && kind == Kind.AZURE) {
            body.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?><EnumerationResults ContainerName=\"container\">")
                    .append("<Blobs>");
            for (String key : page) {
                byte[] bytes = objects.get(key);
                body.append("<Blob><Name>").append(xmlEscape(key)).append("</Name><Properties><Content-Length>")
                        .append(bytes.length).append("</Content-Length><Etag>").append(etag(bytes))
                        .append("</Etag></Properties></Blob>");
            }
            body.append("</Blobs>");
            if (next != null) {
                body.append("<NextMarker>").append(next).append("</NextMarker>");
            } else {
                body.append("<NextMarker/>");
            }
            body.append("</EnumerationResults>");
            respond(exchange, 200, body.toString().getBytes(StandardCharsets.UTF_8), "application/xml");
            return;
        }
        if (xml) {
            body.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult>");
            for (String key : page) {
                byte[] bytes = objects.get(key);
                body.append("<Contents><Key>").append(xmlEscape(key)).append("</Key><LastModified>")
                        .append("2026-01-01T00:00:00.000Z</LastModified><ETag>\"")
                        .append(etag(bytes)).append("\"</ETag><Size>").append(bytes.length)
                        .append("</Size><StorageClass>STANDARD</StorageClass></Contents>");
            }
            if (next != null) {
                body.append("<NextContinuationToken>").append(next).append("</NextContinuationToken>");
            }
            body.append("</ListBucketResult>");
            respond(exchange, 200, body.toString().getBytes(StandardCharsets.UTF_8), "application/xml");
            return;
        }
        body.append("{\"kind\":\"storage#objects\"");
        if (!page.isEmpty()) {
            body.append(",\"items\":[");
            for (int i = 0; i < page.size(); i++) {
                String key = page.get(i);
                byte[] bytes = objects.get(key);
                if (i > 0) {
                    body.append(',');
                }
                body.append("{\"name\":\"").append(jsonEscape(key)).append("\",\"size\":\"")
                        .append(bytes.length).append("\",\"etag\":\"").append(etag(bytes)).append("\"}");
            }
            body.append(']');
        }
        if (next != null) {
            body.append(",\"nextPageToken\":\"").append(next).append('"');
        }
        body.append('}');
        respond(exchange, 200, body.toString().getBytes(StandardCharsets.UTF_8), "application/json");
    }

    private static void respond(HttpExchange exchange, int status, byte[] body, String contentType) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(status, body.length == 0 ? -1 : body.length);
        if (body.length > 0) {
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(body);
            }
        }
    }

    private static Map<String, String> query(String rawQuery) {
        Map<String, String> out = new LinkedHashMap<>();
        if (rawQuery == null || rawQuery.isEmpty()) {
            return out;
        }
        for (String pair : rawQuery.split("&")) {
            int eq = pair.indexOf('=');
            String name = eq < 0 ? pair : pair.substring(0, eq);
            String value = eq < 0 ? "" : pair.substring(eq + 1);
            out.put(decode(name), decode(value));
        }
        return out;
    }

    private static String decode(String value) {
        try {
            return URLDecoder.decode(value, StandardCharsets.UTF_8.name());
        } catch (java.io.UnsupportedEncodingException e) {
            throw new IllegalStateException("UTF-8 is not available on this JVM", e);
        }
    }

    private static String etag(byte[] bytes) {
        return Integer.toHexString(java.util.Arrays.hashCode(bytes));
    }

    private static String xmlEscape(String value) {
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\"", "&quot;").replace("'", "&apos;");
    }

    private static String jsonEscape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
