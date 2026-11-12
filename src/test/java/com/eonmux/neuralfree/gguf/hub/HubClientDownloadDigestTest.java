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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link HubClient#download} of an LFS file against a local HTTP server: the digest taken while the bytes arrive
 * accepts the file, also when a partial from an earlier attempt is resumed over {@code Range} (the kept bytes enter
 * the digest first), and refuses bytes that do not match the LFS pointer, dropping the partial. Nothing leaves the
 * machine.
 */
public class HubClientDownloadDigestTest {

    private static final String REPO = "user/model";
    private static final String COMMIT = "0123456789abcdef0123456789abcdef01234567";
    private static final String NAME = "weights.gguf";
    private static final String ETAG = "\"rev-1\"";
    private static final byte[] BODY = body(300_000);

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private HttpServer server;
    private ExecutorService executor;
    private HubClient client;
    private HubCache cache;
    private HubRef ref;
    private String endpoint;
    /** The first byte the last GET asked for, -1 before any. */
    private final AtomicLong lastRangeStart = new AtomicLong(-1L);

    @Before
    public void startServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        executor = Executors.newFixedThreadPool(2);
        server.setExecutor(executor);
        server.createContext("/" + REPO + "/resolve/" + COMMIT + "/", this::handle);
        server.start();
        endpoint = "http://127.0.0.1:" + server.getAddress().getPort() + "/";
        cache = new HubCache(tmp.getRoot().toPath());
        client = new HubClient(cache, null, endpoint);
        ref = HubRef.parse(REPO);
    }

    @After
    public void stopServer() {
        server.stop(0);
        executor.shutdownNow();
    }

    private static byte[] body(int size) {
        byte[] body = new byte[size];
        for (int i = 0; i < size; i++) {
            body[i] = (byte) (i * 131 + (i >>> 9));
        }
        return body;
    }

    private void handle(HttpExchange exchange) throws IOException {
        try {
            exchange.getResponseHeaders().set("ETag", ETAG);
            exchange.getResponseHeaders().set("Accept-Ranges", "bytes");
            if ("HEAD".equals(exchange.getRequestMethod())) {
                exchange.getResponseHeaders().set("Content-Length", Integer.toString(BODY.length));
                exchange.sendResponseHeaders(200, -1);
                return;
            }
            String range = exchange.getRequestHeaders().getFirst("Range");
            int start = range == null ? 0 : Integer.parseInt(range.substring("bytes=".length(), range.indexOf('-')));
            lastRangeStart.set(start);
            int length = BODY.length - start;
            if (range != null) {
                exchange.getResponseHeaders().set("Content-Range",
                        "bytes " + start + "-" + (BODY.length - 1) + "/" + BODY.length);
            }
            exchange.sendResponseHeaders(range == null ? 200 : 206, length);
            OutputStream out = exchange.getResponseBody();
            out.write(BODY, start, length);
            out.flush();
        } finally {
            exchange.close();
        }
    }

    private HubClient.RepoFile lfsFile(String oid) {
        return new HubClient.RepoFile(REPO, NAME, oid,
                endpoint + REPO + "/resolve/" + COMMIT + "/" + NAME, BODY.length);
    }

    private static String sha256Of(byte[] bytes) throws IOException {
        Path f = Files.createTempFile("digest", ".bin");
        try {
            Files.write(f, bytes);
            return HubClient.sha256(f);
        } finally {
            Files.deleteIfExists(f);
        }
    }

    @Test
    public void aFreshDownloadDigestsItsBytesAsTheyArrive() throws Exception {
        Path landed = client.download(ref, COMMIT, lfsFile(sha256Of(BODY)));
        assertArrayEquals(BODY, Files.readAllBytes(landed));
        assertEquals(0L, lastRangeStart.get());
    }

    @Test
    public void aResumedDownloadDigestsTheKeptBytesBeforeTheNewOnes() throws Exception {
        String oid = sha256Of(BODY);
        Path partial = Paths.get(cache.blobPath(ref, oid) + HubClient.PARTIAL_SUFFIX);
        Files.createDirectories(partial.getParent());
        int kept = 123_457;
        Files.write(partial, Arrays.copyOf(BODY, kept));
        Files.write(Paths.get(partial + ".etag"), ETAG.getBytes(java.nio.charset.StandardCharsets.UTF_8));

        Path landed = client.download(ref, COMMIT, lfsFile(oid));

        assertEquals("the download resumed after the kept bytes", kept, lastRangeStart.get());
        assertArrayEquals(BODY, Files.readAllBytes(landed));
        assertFalse(Files.exists(partial));
    }

    @Test
    public void bytesThatMissTheLfsPointerAreRefusedAndThePartialDropped() throws Exception {
        byte[] other = BODY.clone();
        other[other.length / 2] ^= 1;
        String wrongOid = sha256Of(other);
        try {
            client.download(ref, COMMIT, lfsFile(wrongOid));
            fail("bytes digesting to another pointer must be refused");
        } catch (IOException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("digest not to the LFS sha256 pointer"));
        }
        assertFalse(Files.exists(Paths.get(cache.blobPath(ref, wrongOid) + HubClient.PARTIAL_SUFFIX)));
    }

    @Test
    public void digestOfAPrefixMatchesTheDigestOfThosePrefixBytes() throws Exception {
        Path f = tmp.newFile("prefix.bin").toPath();
        Files.write(f, BODY);
        java.security.MessageDigest prefix = HubClient.digestOf(f, 1000);
        prefix.update(BODY, 1000, BODY.length - 1000);
        java.security.MessageDigest whole = HubClient.digestOf(f, BODY.length);
        assertArrayEquals(whole.digest(), prefix.digest());
        try {
            HubClient.digestOf(f, BODY.length + 1L);
            fail("a prefix longer than the file must be refused");
        } catch (IOException expected) {
            // the file holds fewer bytes
        }
    }
}
