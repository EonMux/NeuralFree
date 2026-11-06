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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * {@link HubClient}'s single token resolution, huggingface_hub's {@code get_token} over a temporary
 * environment: the explicit token first, then {@code HF_TOKEN}, then the file {@code HF_TOKEN_PATH}
 * names, then {@code HF_HOME/token} (HF_HOME defaulting to {@code XDG_CACHE_HOME/huggingface}, then
 * {@code ~/.cache/huggingface}), where {@code HF_HUB_DISABLE_IMPLICIT_TOKEN} turns the implicit
 * sources off. The environment is injected, so no test reads the machine's real values.
 */
public class HubClientTokenTest {

    private static final String REPO = "user/model";
    private static final String COMMIT = "0123456789abcdef0123456789abcdef01234567";
    private static final String TOKEN_A = "hf_" + "a".repeat(34);
    private static final String TOKEN_B = "hf_" + "b".repeat(34);

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** The environment under test; empty per test, so no machine value leaks in. */
    private final Map<String, String> env = new LinkedHashMap<>();

    private HttpServer server;
    private ExecutorService executor;
    private final AtomicReference<String> authorization = new AtomicReference<>();

    @Before
    public void startServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        executor = Executors.newFixedThreadPool(2);
        server.setExecutor(executor);
        server.createContext("/api/models/" + REPO + "/refs", this::serveRefs);
        server.start();
    }

    @After
    public void stopServer() {
        server.stop(0);
        executor.shutdownNow();
    }

    // ================================================================== the resolution order

    @Test
    public void explicitTokenComesFirst() {
        env.put("HF_TOKEN", TOKEN_B);
        assertEquals(TOKEN_A, HubClient.resolveToken(TOKEN_A, env()));
    }

    @Test
    public void envTokenBeforeTheTokenFile() throws IOException {
        Path named = write("named-token", TOKEN_B);
        env.put("HF_TOKEN", TOKEN_A);
        env.put("HF_TOKEN_PATH", named.toString());
        assertEquals(TOKEN_A, resolve());
    }

    @Test
    public void blankEnvTokenFallsThroughToTheTokenFile() throws IOException {
        Path named = write("named-token", TOKEN_A);
        env.put("HF_TOKEN", "   ");
        env.put("HF_TOKEN_PATH", named.toString());
        assertEquals(TOKEN_A, resolve());
    }

    @Test
    public void tokenPathFileIsTrimmed() throws IOException {
        Path named = write("padded-token", "  " + TOKEN_A + "\r\n");
        env.put("HF_TOKEN_PATH", named.toString());
        assertEquals(TOKEN_A, resolve());
    }

    @Test
    public void tokenPathFileBeatsTheHfHomeToken() throws IOException {
        Path named = write("named-token", TOKEN_A);
        Path home = tmp.newFolder("home").toPath();
        Files.write(home.resolve("token"), TOKEN_B.getBytes(StandardCharsets.UTF_8));
        env.put("HF_TOKEN_PATH", named.toString());
        env.put("HF_HOME", home.toString());
        assertEquals(TOKEN_A, resolve());
    }

    @Test
    public void hfHomeTokenIsRead() throws IOException {
        Path home = tmp.newFolder("hfhome").toPath();
        Files.write(home.resolve("token"), TOKEN_A.getBytes(StandardCharsets.UTF_8));
        env.put("HF_HOME", home.toString());
        assertEquals(TOKEN_A, resolve());
    }

    @Test
    public void hfHomeBeatsTheXdgCacheDefault() throws IOException {
        Path hfHome = tmp.newFolder("hfhome").toPath();
        Path xdg = tmp.newFolder("xdg").toPath();
        Files.write(hfHome.resolve("token"), TOKEN_B.getBytes(StandardCharsets.UTF_8));
        Files.createDirectories(xdg.resolve("huggingface"));
        Files.write(xdg.resolve("huggingface").resolve("token"), TOKEN_A.getBytes(StandardCharsets.UTF_8));
        env.put("HF_HOME", hfHome.toString());
        env.put("XDG_CACHE_HOME", xdg.toString());
        assertEquals(TOKEN_B, resolve());
    }

    @Test
    public void xdgCacheDefaultIsRead() throws IOException {
        Path xdg = tmp.newFolder("xdg").toPath();
        Files.createDirectories(xdg.resolve("huggingface"));
        Files.write(xdg.resolve("huggingface").resolve("token"), TOKEN_A.getBytes(StandardCharsets.UTF_8));
        env.put("XDG_CACHE_HOME", xdg.toString());
        assertEquals(TOKEN_A, resolve());
    }

    @Test
    public void homeCacheDefaultIsRead() throws IOException {
        Path home = tmp.newFolder("home").toPath();
        Files.createDirectories(home.resolve(".cache").resolve("huggingface"));
        Files.write(home.resolve(".cache").resolve("huggingface").resolve("token"),
                TOKEN_A.getBytes(StandardCharsets.UTF_8));
        env.put("HOME", home.toString());
        assertEquals(TOKEN_A, resolve());
    }

    @Test
    public void emptySourcesAreAnonymous() {
        assertNull(resolve());
    }

    @Test
    public void missingTokenPathFileIsAnonymous() {
        env.put("HF_TOKEN_PATH", tmp.getRoot().toPath().resolve("absent-token").toString());
        assertNull(resolve());
    }

    @Test
    public void invalidEnvTokenIsAnonymousAndDoesNotFallThroughToTheFile() throws IOException {
        Path named = write("named-token", TOKEN_A);
        env.put("HF_TOKEN", "not a token");
        env.put("HF_TOKEN_PATH", named.toString());
        assertNull(resolve());
    }

    @Test
    public void invalidTokenPathFileIsAnonymous() throws IOException {
        Path named = write("junk-token", "tok");
        env.put("HF_TOKEN_PATH", named.toString());
        assertNull(resolve());
    }

    @Test
    public void emptyTokenPathFileReplacesTheHfHomeDefault() throws IOException {
        Path named = write("empty-token", "\n");
        Path home = tmp.newFolder("hfhome").toPath();
        Files.write(home.resolve("token"), TOKEN_A.getBytes(StandardCharsets.UTF_8));
        env.put("HF_TOKEN_PATH", named.toString());
        env.put("HF_HOME", home.toString());
        assertNull(resolve());
    }

    // ================================================================== the disable switch

    @Test
    public void disableSwitchStopsEveryImplicitSource() throws IOException {
        Path named = write("named-token", TOKEN_A);
        env.put("HF_TOKEN", TOKEN_B);
        env.put("HF_TOKEN_PATH", named.toString());
        env.put("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1");
        assertNull(resolve());
        env.remove("HF_TOKEN");
        assertNull(resolve());
    }

    @Test
    public void disableSwitchTakesHubForms() throws IOException {
        env.put("HF_TOKEN", TOKEN_A);
        for (String value : List.of("1", "on", "ON", "yes", "TRUE", "True")) {
            env.put("HF_HUB_DISABLE_IMPLICIT_TOKEN", value);
            assertNull("the switch value " + value + " must disable the implicit token", resolve());
        }
        for (String value : List.of("0", "false", "no", "off", "")) {
            env.put("HF_HUB_DISABLE_IMPLICIT_TOKEN", value);
            assertEquals("the switch value " + value + " must keep the implicit token",
                    TOKEN_A, resolve());
        }
    }

    @Test
    public void disableSwitchKeepsTheExplicitToken() {
        env.put("HF_TOKEN", TOKEN_B);
        env.put("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1");
        assertEquals(TOKEN_A, HubClient.resolveToken(TOKEN_A, env()));
    }

    // ================================================================== the client carries the token

    @Test
    public void clientCarriesTheTokenFileToken() throws IOException {
        Path named = write("named-token", "  " + TOKEN_A + "\n");
        env.put("HF_TOKEN_PATH", named.toString());
        HubClient client = new HubClient(new HubCache(tmp.getRoot().toPath()), null, endpoint(), env());
        assertEquals(TOKEN_A, client.tokenValue());
    }

    @Test
    public void clientCarriesTheEnvToken() {
        env.put("HF_TOKEN", TOKEN_B);
        HubClient client = new HubClient(new HubCache(tmp.getRoot().toPath()), null, endpoint(), env());
        assertEquals(TOKEN_B, client.tokenValue());
    }

    @Test
    public void clientUnderTheDisableSwitchIsAnonymous() {
        env.put("HF_TOKEN", TOKEN_A);
        env.put("HF_HUB_DISABLE_IMPLICIT_TOKEN", "true");
        HubClient client = new HubClient(new HubCache(tmp.getRoot().toPath()), null, endpoint(), env());
        assertNull(client.tokenValue());
    }

    @Test
    public void clientUnderTheDisableSwitchKeepsItsExplicitToken() {
        env.put("HF_TOKEN", TOKEN_B);
        env.put("HF_HUB_DISABLE_IMPLICIT_TOKEN", "true");
        HubClient client = new HubClient(new HubCache(tmp.getRoot().toPath()), TOKEN_A, endpoint(), env());
        assertEquals(TOKEN_A, client.tokenValue());
    }

    @Test
    public void requestCarriesTheFileTokenAsBearer() throws Exception {
        Path named = write("named-token", TOKEN_A + "\n");
        env.put("HF_TOKEN_PATH", named.toString());
        HubClient client = new HubClient(new HubCache(tmp.getRoot().toPath()), null, endpoint(), env());
        client.resolveCommit(HubRef.parse(REPO));
        assertEquals("Bearer " + TOKEN_A, authorization.get());
    }

    @Test
    public void disabledClientSendsNoAuthorization() throws Exception {
        env.put("HF_TOKEN", TOKEN_A);
        env.put("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1");
        HubClient client = new HubClient(new HubCache(tmp.getRoot().toPath()), null, endpoint(), env());
        client.resolveCommit(HubRef.parse(REPO));
        assertNull(authorization.get());
    }

    // ================================================================== fixtures

    /** The environment lookup {@link HubClient} reads: the map under test, never the machine's. */
    private Function<String, String> env() {
        return env::get;
    }

    /** The token resolved with no explicit value: the environment's own. */
    private String resolve() {
        return HubClient.resolveToken(null, env());
    }

    /** A file of {@code tmp} carrying {@code content}, UTF-8. */
    private Path write(String name, String content) throws IOException {
        Path file = tmp.getRoot().toPath().resolve(name);
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
        return file;
    }

    /** The test server's endpoint URL in the client's trailing-slash form. */
    private String endpoint() {
        return "http://127.0.0.1:" + server.getAddress().getPort() + "/";
    }

    /** One refs answer that records the request's Authorization header and names {@code main}. */
    private void serveRefs(HttpExchange exchange) throws IOException {
        authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
        byte[] body = ("[{\"name\":\"main\",\"targetCommit\":\"" + COMMIT + "\"}]")
                .getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, body.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(body);
        }
    }
}
