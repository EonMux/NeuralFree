package com.eonmux.neuralfree.gguf.hub;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HubClient#readRange} and {@link HubClient#readAll} against a local HTTP server that serves one
 * {@code Range}-capable file: the exact span comes back, a server that ignores {@code Range} is accepted only from
 * byte 0, a truncated body is retried and completed, a 404 is final after one request, a contradicted
 * {@code Content-Range} is final too, and a span past the listed size never reaches the transport. The client's
 * endpoint is the test server's, so nothing leaves the machine.
 */
public class HubClientRangeTest {

    private static final String REPO = "user/model";
    private static final String COMMIT = "0123456789abcdef0123456789abcdef01234567";
    private static final byte[] BODY = body(1000);

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private HttpServer server;
    private ExecutorService executor;
    private HubClient client;
    private HubRef ref;
    private String endpoint;
    private final AtomicInteger requests = new AtomicInteger();

    @Before
    public void startServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        executor = Executors.newFixedThreadPool(4);
        server.setExecutor(executor);
        server.createContext("/" + REPO + "/resolve/" + COMMIT + "/", this::handle);
        server.start();
        endpoint = "http://127.0.0.1:" + server.getAddress().getPort() + "/";
        client = new HubClient(new HubCache(tmp.getRoot().toPath()), null, endpoint);
        ref = HubRef.parse(REPO);
    }

    @After
    public void stopServer() {
        server.stop(0);
        executor.shutdownNow();
    }

    // ================================================================== fixtures

    private static byte[] body(int size) {
        byte[] body = new byte[size];
        for (int i = 0; i < size; i++) {
            body[i] = (byte) (i * 31 + 7);
        }
        return body;
    }

    private String url(String name) {
        return endpoint + REPO + "/resolve/" + COMMIT + "/" + name;
    }

    private HubClient.RepoFile file(String name) {
        return new HubClient.RepoFile(REPO, name, null, url(name), BODY.length);
    }

    private HubClient.RepoFile unsizedFile(String name) {
        return new HubClient.RepoFile(REPO, name, null, url(name), -1L);
    }

    private void handle(HttpExchange exchange) throws IOException {
        requests.incrementAndGet();
        String path = exchange.getRequestURI().getPath();
        String name = path.substring(path.lastIndexOf('/') + 1);
        try {
            if (name.equals("missing.bin")) {
                exchange.sendResponseHeaders(404, -1);
                exchange.close();
                return;
            }
            if ("HEAD".equals(exchange.getRequestMethod())) {
                exchange.getResponseHeaders().set("Content-Length", Integer.toString(BODY.length));
                exchange.sendResponseHeaders(200, -1);
                exchange.close();
                return;
            }
            String range = exchange.getRequestHeaders().getFirst("Range");
            if (name.equals("whole.bin")) {
                exchange.sendResponseHeaders(200, BODY.length);
                write(exchange, 0, BODY.length);
                return;
            }
            long start = 0L;
            long end = BODY.length - 1L;
            if (range != null) {
                String span = range.substring("bytes=".length());
                int dash = span.indexOf('-');
                start = Long.parseLong(span.substring(0, dash));
                String tail = span.substring(dash + 1);
                end = tail.isEmpty() ? BODY.length - 1L : Long.parseLong(tail);
            }
            long length = end - start + 1L;
            exchange.getResponseHeaders().set("Content-Range",
                    name.equals("bad-range.bin") ? "bytes 0-4/" + BODY.length
                            : "bytes " + start + "-" + end + "/" + BODY.length);
            exchange.sendResponseHeaders(206, length);
            if (name.equals("short.bin") && requests.get() == 1) {
                write(exchange, (int) start, (int) (length / 2));
                exchange.close();
                return;
            }
            write(exchange, (int) start, (int) length);
        } finally {
            exchange.close();
        }
    }

    private static void write(HttpExchange exchange, int start, int length) throws IOException {
        OutputStream out = exchange.getResponseBody();
        out.write(BODY, start, length);
        out.flush();
    }

    // ================================================================== ranged reads

    @Test
    public void readRangeReturnsExactlyTheRequestedSpan() throws Exception {
        byte[] got = client.readRange(ref, COMMIT, file("file.bin"), 10L, 19L);
        assertArrayEquals(java.util.Arrays.copyOfRange(BODY, 10, 20), got);
        assertEquals(1, requests.get());
    }

    @Test
    public void readRangeAcceptsAClientThatIgnoresTheRangeHeaderFromByteZero() throws Exception {
        byte[] got = client.readRange(ref, COMMIT, file("whole.bin"), 0L, 9L);
        assertArrayEquals(java.util.Arrays.copyOfRange(BODY, 0, 10), got);
    }

    @Test
    public void readRangeRetriesATruncatedBodyAndCompletesIt() throws Exception {
        byte[] got = client.readRange(ref, COMMIT, file("short.bin"), 100L, 299L);
        assertArrayEquals(java.util.Arrays.copyOfRange(BODY, 100, 300), got);
        assertEquals(2, requests.get());
    }

    @Test
    public void readRangeDeliversEveryChunkAtItsOffset() throws Exception {
        byte[] out = new byte[50];
        client.readRange(ref, COMMIT, file("file.bin"), 5L, 54L,
                (rangeOffset, bytes, offset, length) -> System.arraycopy(bytes, offset, out, (int) rangeOffset,
                        length));
        assertArrayEquals(java.util.Arrays.copyOfRange(BODY, 5, 55), out);
    }

    @Test
    public void readRangeRefusesAnHTTP404AfterOneRequest() {
        IOException e = assertThrows(IOException.class,
                () -> client.readRange(ref, COMMIT, file("missing.bin"), 0L, 9L));
        assertTrue(e.getMessage(), e.getMessage().contains("HTTP 404"));
        assertEquals(1, requests.get());
    }

    @Test
    public void readRangeRefusesAContradictedContentRangeAfterOneRequest() {
        IOException e = assertThrows(IOException.class,
                () -> client.readRange(ref, COMMIT, file("bad-range.bin"), 0L, 9L));
        assertTrue(e.getMessage(), e.getMessage().contains("Content-Range")
                || e.getMessage().contains("requested"));
        assertEquals(1, requests.get());
    }

    @Test
    public void readRangeRefusesASpanPastTheListedSizeWithoutARequest() {
        assertThrows(IllegalArgumentException.class,
                () -> client.readRange(ref, COMMIT, file("file.bin"), 999L, 1000L));
        assertEquals(0, requests.get());
    }

    @Test
    public void readRangeRefusesAUrlThatDoesNotNameTheFile() {
        HubClient.RepoFile wrong = new HubClient.RepoFile(REPO, "file.bin", null,
                endpoint + REPO + "/resolve/" + COMMIT + "/other.bin", BODY.length);
        assertThrows(IllegalArgumentException.class,
                () -> client.readRange(ref, COMMIT, wrong, 0L, 9L));
        assertEquals(0, requests.get());
    }

    // ================================================================== whole files

    @Test
    public void readAllReadsAFileOfKnownSize() throws Exception {
        assertArrayEquals(BODY, client.readAll(ref, COMMIT, file("file.bin"), BODY.length));
    }

    @Test
    public void readAllReadsAFileOfUnknownSizeToTheEnd() throws Exception {
        assertArrayEquals(BODY, client.readAll(ref, COMMIT, unsizedFile("file.bin"), BODY.length));
    }

    @Test
    public void readAllRefusesAFileOverItsLimit() {
        IOException e = assertThrows(IOException.class,
                () -> client.readAll(ref, COMMIT, file("file.bin"), 10L));
        assertTrue(e.getMessage(), e.getMessage().contains("over the 10-byte limit"));
    }

    @Test
    public void contentLengthReadsTheHeadLength() throws Exception {
        assertEquals(BODY.length, client.contentLength(ref, COMMIT, unsizedFile("file.bin")));
    }
}
