package com.eonmux.neuralfree.gguf.media;

import com.sun.net.httpserver.HttpServer;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.function.ThrowingRunnable;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Base64;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MediaConnector}: RFC 2397 base64 data URLs and raw base64, http(s) fetches from a loopback server with
 * the allowed-domain check applied to every redirect hop, file URLs only inside the allowed local directory
 * (absolute and root-relative), the size limit on every source, and the named refusal of everything else.
 */
public class MediaConnectorTest {

    private static final byte[] PAYLOAD = "media-bytes".getBytes(StandardCharsets.US_ASCII);
    private static final String B64 = Base64.getEncoder().encodeToString(PAYLOAD);

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private HttpServer server;
    private String base;

    @Before
    public void startServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/media", ex -> respond(ex, 200, PAYLOAD));
        server.createContext("/missing", ex -> respond(ex, 404, "nope".getBytes(StandardCharsets.US_ASCII)));
        server.createContext("/empty", ex -> respond(ex, 200, new byte[0]));
        server.createContext("/big", ex -> respond(ex, 200, new byte[4096]));
        server.createContext("/hop", ex -> redirect(ex, "/media"));
        server.createContext("/away", ex -> redirect(ex, "http://elsewhere.invalid/media"));
        server.createContext("/loop", ex -> redirect(ex, "/loop"));
        server.start();
        base = "http://127.0.0.1:" + server.getAddress().getPort();
    }

    @After
    public void stopServer() {
        server.stop(0);
    }

    private static void respond(com.sun.net.httpserver.HttpExchange ex, int status, byte[] body)
            throws IOException {
        ex.sendResponseHeaders(status, body.length == 0 ? -1 : body.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(body);
        }
    }

    private static void redirect(com.sun.net.httpserver.HttpExchange ex, String location) throws IOException {
        ex.getResponseHeaders().add("Location", location);
        ex.sendResponseHeaders(302, -1);
        ex.close();
    }

    private static MediaConnector connector(MediaConnector.Options.Builder options) {
        return new MediaConnector(options.build());
    }

    private static MediaConnector defaults() {
        return new MediaConnector(MediaConnector.Options.defaults());
    }

    private static void assertRefused(ThrowingRunnable call, String fragment) {
        MediaInputException e = assertThrows(MediaInputException.class, call);
        assertTrue(e.getMessage(), e.getMessage().contains(fragment));
    }

    // ------------------------------------------------------------------ inline

    @Test
    public void base64DataUrlsDecodeWithTheirMediaType() {
        MediaConnector.Loaded loaded = defaults().load("data:image/png;base64," + B64, "image_url.url");
        assertArrayEquals(PAYLOAD, loaded.bytes());
        assertEquals(PAYLOAD.length, loaded.size());
        assertEquals("image/png", loaded.mediaType());
        loaded = defaults().load("data:image/jpeg;name=x.jpg;base64," + B64, "f");
        assertEquals("image/jpeg", loaded.mediaType());
        assertNull(defaults().load("data:;base64," + B64, "f").mediaType());
    }

    @Test
    public void rawBase64DecodesInBothAlphabetsIgnoringWhitespace() {
        assertArrayEquals(PAYLOAD, defaults().load(B64, "f").bytes());
        byte[] binary = {(byte) 0xFB, (byte) 0xFF, (byte) 0xBF};
        assertArrayEquals(binary, defaults().load(Base64.getUrlEncoder().encodeToString(binary), "f").bytes());
        String wrapped = B64.substring(0, 4) + "\n " + B64.substring(4);
        assertArrayEquals(PAYLOAD, defaults().inline(wrapped, "f").bytes());
    }

    @Test
    public void malformedInlineDataIsRefusedByField() {
        assertRefused(() -> defaults().load("data:image/png," + B64, "messages[0].content[1].image_url.url"),
                "messages[0].content[1].image_url.url is a data URL without ';base64'");
        assertRefused(() -> defaults().load("data:image/png;base64", "f"), "separator is missing");
        assertRefused(() -> defaults().load("not base64 at all!", "f"), "not valid base64");
        assertRefused(() -> defaults().load("  ", "f"), "f is empty");
        assertRefused(() -> defaults().inline(null, "input_audio.data"), "input_audio.data is empty");
        assertRefused(() -> defaults().load("ftp://host/x.png", "f"), "not valid base64");
    }

    @Test
    public void inlineDataNeverFetches() {
        assertRefused(() -> defaults().inline(base + "/media", "f"), "not valid base64");
    }

    @Test
    public void payloadsOverTheLimitAreRefused() {
        MediaConnector small = connector(MediaConnector.Options.builder().maxBytes(8));
        assertRefused(() -> small.load(B64, "f"), "exceeds the 8-byte limit");
        assertRefused(() -> small.load(base + "/big", "f"), "exceeds the 8-byte limit");
    }

    // ------------------------------------------------------------------ http

    @Test
    public void httpUrlsFetchTheirBodyFollowingRedirects() {
        assertArrayEquals(PAYLOAD, defaults().load(base + "/media", "f").bytes());
        assertArrayEquals(PAYLOAD, defaults().load(base + "/hop", "f").bytes());
    }

    @Test
    public void onlyAllowedDomainsAreFetchedOnEveryHop() {
        MediaConnector local = connector(MediaConnector.Options.builder()
                .allowedDomains(List.of("127.0.0.1")));
        assertArrayEquals(PAYLOAD, local.load(base + "/hop", "f").bytes());
        assertRefused(() -> local.load(base + "/away", "f"), "input URL domain: elsewhere.invalid");
        MediaConnector remote = connector(MediaConnector.Options.builder()
                .allowedDomains(List.of("Example.COM")));
        assertRefused(() -> remote.load(base + "/media", "image_url.url"),
                "image_url.url: the URL must be from one of the allowed domains [example.com]");
    }

    @Test
    public void disabledRedirectsAndRedirectLoopsAreRefused() {
        MediaConnector strict = connector(MediaConnector.Options.builder().allowRedirects(false));
        assertRefused(() -> strict.load(base + "/hop", "f"), "redirects are disabled");
        assertRefused(() -> defaults().load(base + "/loop", "f"), "redirects more than");
    }

    @Test
    public void failedFetchesAreRefused() {
        assertRefused(() -> defaults().load(base + "/missing", "f"), "failed with HTTP 404");
        assertRefused(() -> defaults().load(base + "/empty", "f"), "empty body");
        assertRefused(() -> defaults().load("http:///nohost", "f"), "no host");
    }

    // ------------------------------------------------------------------ file

    @Test
    public void fileUrlsAreRefusedWithoutALocalDirectory() throws IOException {
        Path file = tmp.newFile("a.png").toPath();
        Files.write(file, PAYLOAD);
        assertRefused(() -> defaults().load(file.toUri().toString(), "f"),
                "file URLs are refused; start the server with --allowed-local-media-path");
    }

    @Test
    public void fileUrlsLoadFromInsideTheLocalDirectory() throws IOException {
        Path root = tmp.newFolder("media").toPath().toRealPath();
        Path sub = Files.createDirectories(root.resolve("sub dir"));
        Files.write(sub.resolve("a+b.png"), PAYLOAD);
        MediaConnector local = connector(MediaConnector.Options.builder().localRoot(root));
        assertArrayEquals(PAYLOAD, local.load(sub.resolve("a+b.png").toUri().toString(), "f").bytes());
        assertArrayEquals(PAYLOAD, local.load("file://sub%20dir/a+b.png", "f").bytes());
        assertArrayEquals(PAYLOAD, local.load("file://localhost" + sub.resolve("a+b.png"), "f").bytes());
    }

    @Test
    public void fileUrlsOutsideTheLocalDirectoryAreRefused() throws IOException {
        Path root = tmp.newFolder("allowed").toPath().toRealPath();
        Path outside = tmp.newFile("secret.png").toPath();
        Files.write(outside, PAYLOAD);
        MediaConnector local = connector(MediaConnector.Options.builder().localRoot(root));
        assertRefused(() -> local.load(outside.toUri().toString(), "f"), "must be inside the allowed local");
        assertRefused(() -> local.load("file://../secret.png", "f"), "must be inside the allowed local");
        assertRefused(() -> local.load("file://" + root, "f"), "must be inside the allowed local");
        assertRefused(() -> local.load("file://nothing.png", "f"), "does not exist");
        Files.createSymbolicLink(root.resolve("link.png"), outside);
        assertRefused(() -> local.load("file://link.png", "f"), "must be inside the allowed local");
        Files.createDirectories(root.resolve("dir"));
        assertRefused(() -> local.load("file://dir", "f"), "is not a regular file");
    }

    // ------------------------------------------------------------------ options

    @Test
    public void optionsValidateAtTheBoundary() throws IOException {
        Path file = tmp.newFile("x").toPath();
        assertThrows(IllegalArgumentException.class, () -> MediaConnector.Options.builder().localRoot(file));
        assertThrows(IllegalArgumentException.class,
                () -> MediaConnector.Options.builder().timeout(Duration.ZERO));
        assertThrows(IllegalArgumentException.class, () -> MediaConnector.Options.builder().maxBytes(0));
        assertThrows(IllegalArgumentException.class,
                () -> MediaConnector.Options.builder().allowedDomains(List.of(" ")));
        MediaConnector.Options defaults = MediaConnector.Options.defaults();
        assertTrue(defaults.domainAllowed("anything.example"));
        assertTrue(defaults.allowRedirects());
        assertNull(defaults.localRoot());
        assertEquals(MediaConnector.Options.DEFAULT_MAX_BYTES, defaults.maxBytes());
        assertEquals(MediaConnector.Options.DEFAULT_TIMEOUT, defaults.timeout());
    }
}
