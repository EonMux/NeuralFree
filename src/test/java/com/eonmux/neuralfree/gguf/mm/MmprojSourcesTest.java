package com.eonmux.neuralfree.gguf.mm;

import com.sun.net.httpserver.HttpServer;
import org.junit.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MmprojSources}: the three projector sources of llama.cpp ({@code --mmproj}, {@code --mmproj-url},
 * {@code --mmproj-auto}), the sibling search being {@code find_best_sibling} of common/download.cpp (the deepest
 * shared directory, the exact quant tag, the closest quantization), and the URL download into a cache directory
 * served by an in-process http server.
 */
public class MmprojSourcesTest {

    private static Path dir(String name) throws IOException {
        return Files.createTempDirectory(name);
    }

    // ---------------------------------------------------------------- the sibling search

    @Test
    public void theDeepestSharedDirectoryWins() {
        List<String> candidates = List.of(
                "repo/mmproj-Q8_0.gguf",
                "repo/sub/mmproj-Q8_0.gguf",
                "repo/other/mmproj-Q4_K_M.gguf");
        assertEquals("repo/sub/mmproj-Q8_0.gguf",
                MmprojSources.bestSibling(candidates, "repo/sub/model-Q4_K_M.gguf", ""));
    }

    @Test
    public void theExactTagBreaksADepthTie() {
        List<String> candidates = List.of(
                "repo/mmproj-Q2_K.gguf",
                "repo/mmproj-IQ2_XS.gguf");
        assertEquals("repo/mmproj-Q2_K.gguf",
                MmprojSources.bestSibling(candidates, "repo/model-Q2_K.gguf", "Q2_K"));
    }

    @Test
    public void theClosestQuantizationWinsOtherwise() {
        List<String> candidates = List.of(
                "repo/mmproj-F16.gguf",
                "repo/mmproj-Q8_0.gguf");
        assertEquals("repo/mmproj-Q8_0.gguf",
                MmprojSources.bestSibling(candidates, "repo/model-Q4_K_M.gguf", ""));
    }

    @Test
    public void aCandidateOutsideTheModelDirectoryIsNoCandidate() {
        List<String> candidates = List.of(
                "repo/mmproj-Q4_K_M.gguf",
                "other/mmproj-Q4_K_M.gguf",
                "repo/another/deeper/mmproj-Q4_K_M.gguf");
        assertEquals("repo/mmproj-Q4_K_M.gguf",
                MmprojSources.bestSibling(candidates, "repo/model-Q4_K_M.gguf", ""));
    }

    @Test
    public void nonGgufFilesWithoutTheKeywordAreSkipped() {
        List<String> candidates = List.of(
                "repo/mmproj-Q8_0.json",
                "repo/tokenizer.gguf",
                "repo/model-Q4_K_M.gguf");
        assertNull(MmprojSources.bestSibling(candidates, "repo/model-Q4_K_M.gguf", ""));
    }

    @Test
    public void theRequestedTagPointsAtItsOwnSibling() {
        List<String> candidates = List.of(
                "repo/mmproj-Q8_0.gguf",
                "repo/mmproj-Q6_K.gguf");
        assertEquals("repo/mmproj-Q6_K.gguf",
                MmprojSources.bestSibling(candidates, "repo/model-Q6_K.gguf", "Q6_K"));
    }

    // ---------------------------------------------------------------- the sibling beside a model file

    @Test
    public void siblingBesidePicksTheClosestQuantization() throws IOException {
        Path dir = dir("mmproj-beside");
        try {
            Path model = dir.resolve("model-Q4_K_M.gguf");
            Files.write(model, new byte[] {1});
            Files.write(dir.resolve("mmproj-F16.gguf"), new byte[] {2});
            Files.write(dir.resolve("mmproj-Q8_0.gguf"), new byte[] {3});
            Files.write(dir.resolve("notes.txt"), new byte[] {4});
            assertEquals(dir.resolve("mmproj-Q8_0.gguf"), MmprojSources.siblingBeside(model));
        } finally {
            delete(dir);
        }
    }

    @Test
    public void siblingBesideAnswersNullWithoutASibling() throws IOException {
        Path dir = dir("mmproj-none");
        try {
            Path model = dir.resolve("model-Q4_K_M.gguf");
            Files.write(model, new byte[] {1});
            Files.write(dir.resolve("other-Q8_0.gguf"), new byte[] {2});
            assertNull(MmprojSources.siblingBeside(model));
        } finally {
            delete(dir);
        }
    }

    // ---------------------------------------------------------------- resolve

    @Test
    public void resolveRefusesAMissingExplicitProjectorNamingTheFlag() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MmprojSources.resolve("--mmproj", "gone.gguf", null, true, null, null));
        assertTrue(e.getMessage().contains("--mmproj"));
    }

    @Test
    public void resolveTakesTheExplicitProjector() throws IOException {
        Path dir = dir("mmproj-explicit");
        try {
            Path file = dir.resolve("p.gguf");
            Files.write(file, new byte[] {1});
            assertEquals(file, MmprojSources.resolve("--mmproj", file.toString(), "https://h/x", true, null, null));
        } finally {
            delete(dir);
        }
    }

    @Test
    public void resolveFindsTheAutoSiblingAndAnswersNullWithoutOne() throws IOException {
        Path dir = dir("mmproj-auto");
        try {
            Path model = dir.resolve("model-Q8_0.gguf");
            Files.write(model, new byte[] {1});
            Files.write(dir.resolve("mmproj-Q8_0.gguf"), new byte[] {2});
            assertEquals(dir.resolve("mmproj-Q8_0.gguf"),
                    MmprojSources.resolve("--mmproj", null, null, true, model, dir));
            assertNull(MmprojSources.resolve("--mmproj", null, null, false, model, dir));
        } finally {
            delete(dir);
        }
    }

    // ---------------------------------------------------------------- the URL download

    @Test
    public void theUrlDownloadCachesItsFileOnce() throws Exception {
        Path cache = dir("mmproj-url-cache");
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        AtomicInteger served = new AtomicInteger();
        byte[] body = "projector-bytes".getBytes(StandardCharsets.UTF_8);
        server.createContext("/models/mmproj-Q8_0.gguf", exchange -> {
            served.incrementAndGet();
            exchange.sendResponseHeaders(200, body.length);
            try (var out = exchange.getResponseBody()) {
                out.write(body);
            }
        });
        server.start();
        try {
            Path file = MmprojSources.downloadFromUrl(
                    "http://127.0.0.1:" + server.getAddress().getPort() + "/models/mmproj-Q8_0.gguf", cache);
            assertEquals(cache.resolve("mmproj-Q8_0.gguf"), file);
            assertArrayEquals(body, Files.readAllBytes(file));
            assertTrue(Files.isRegularFile(file));
            assertFalse(Files.exists(Paths.get(file + ".downloadInProgress")));
            MmprojSources.downloadFromUrl(
                    "http://127.0.0.1:" + server.getAddress().getPort() + "/models/mmproj-Q8_0.gguf", cache);
            assertEquals("the cached file is kept without a second fetch", 1, served.get());
        } finally {
            server.stop(0);
            delete(cache);
        }
    }

    @Test
    public void theUrlDownloadRefusesAServerErrorAndANonHttpUrl() throws Exception {
        Path cache = dir("mmproj-url-error");
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/gone.gguf", exchange -> exchange.sendResponseHeaders(404, -1));
        server.start();
        try {
            IOException e = assertThrows(IOException.class, () -> MmprojSources.downloadFromUrl(
                    "http://127.0.0.1:" + server.getAddress().getPort() + "/gone.gguf", cache));
            assertTrue(e.getMessage().contains("404"));
            assertFalse(Files.exists(cache.resolve("gone.gguf")));
            IllegalArgumentException bad = assertThrows(IllegalArgumentException.class,
                    () -> MmprojSources.downloadFromUrl("ftp://h/x.gguf", cache));
            assertTrue(bad.getMessage().contains("http or https"));
        } finally {
            server.stop(0);
            delete(cache);
        }
    }

    private static void delete(Path dir) throws IOException {
        try (var files = Files.list(dir)) {
            for (Path p : files.toArray(Path[]::new)) {
                Files.deleteIfExists(p);
            }
        }
        Files.deleteIfExists(dir);
    }
}
