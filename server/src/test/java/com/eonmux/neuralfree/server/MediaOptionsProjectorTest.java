package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The projector sources of {@link MediaOptions#fromSystemProperties()}, llama.cpp's {@code --mmproj} /
 * {@code --mmproj-url} / {@code --mmproj-auto}: the explicit {@code neuralfree.server.mmproj} file wins, its
 * blank value turns every source off, the {@code neuralfree.server.mmproj.url} download lands in the hub cache,
 * and the {@code neuralfree.server.mmproj.auto} search takes the {@code *mmproj*.gguf} sibling beside the
 * {@code neuralfree.server.model} file unless it is switched off.
 */
class MediaOptionsProjectorTest {

    private static final String MODEL_PROPERTY = "neuralfree.server.model";
    private static final String[] PROPERTIES = {MediaOptions.MMPROJ_PROPERTY, MediaOptions.MMPROJ_URL_PROPERTY,
            MediaOptions.MMPROJ_AUTO_PROPERTY, MODEL_PROPERTY};

    private final Map<String, String> saved = new HashMap<>();

    @BeforeEach
    void save() {
        for (String property : PROPERTIES) {
            saved.put(property, System.getProperty(property));
            System.clearProperty(property);
        }
    }

    @AfterEach
    void restore() {
        saved.forEach((property, value) -> {
            if (value == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, value);
            }
        });
    }

    private static String projector() {
        return MediaOptions.fromSystemProperties().mmproj();
    }

    @Test
    void theExplicitProjectorWinsAndBlankTurnsEverythingOff(@TempDir Path dir) throws IOException {
        Path mmproj = Files.write(dir.resolve("mmproj.gguf"), new byte[] {1});
        System.setProperty(MediaOptions.MMPROJ_PROPERTY, " " + mmproj + " ");
        System.setProperty(MediaOptions.MMPROJ_URL_PROPERTY, "http://127.0.0.1:1/nf-test-absent.gguf");
        assertEquals(mmproj.toString(), projector());
        System.setProperty(MediaOptions.MMPROJ_PROPERTY, "  ");
        assertNull(projector(), "a blank projector turns media input off over every other source");
    }

    @Test
    void theUrlPropertyDownloadsIntoTheHubCache() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        byte[] body = "projector-bytes".getBytes(StandardCharsets.UTF_8);
        server.createContext("/nf/nf-test-mmproj-Q8_0.gguf", exchange -> {
            exchange.sendResponseHeaders(200, body.length);
            try (var out = exchange.getResponseBody()) {
                out.write(body);
            }
        });
        server.start();
        try {
            System.setProperty(MediaOptions.MMPROJ_URL_PROPERTY,
                    "http://127.0.0.1:" + server.getAddress().getPort() + "/nf/nf-test-mmproj-Q8_0.gguf");
            Path file = HubCache.defaultRoot().resolve("nf-test-mmproj-Q8_0.gguf");
            try {
                assertEquals(file.toString(), projector());
                assertArrayEquals(body, Files.readAllBytes(file));
            } finally {
                Files.deleteIfExists(file);
            }
        } finally {
            server.stop(0);
        }
    }

    @Test
    void aFailingUrlDownloadRefusesNamingTheProperty() {
        System.setProperty(MediaOptions.MMPROJ_URL_PROPERTY, "http://127.0.0.1:1/nf-test-absent.gguf");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                MediaOptionsProjectorTest::projector);
        assertTrue(e.getMessage().contains(MediaOptions.MMPROJ_URL_PROPERTY), e.getMessage());
    }

    @Test
    void theAutoPropertySearchesBesideTheModelUnlessItIsOff(@TempDir Path dir) throws IOException {
        Path model = Files.write(dir.resolve("model-Q4_K_M.gguf"), new byte[] {1});
        Files.write(dir.resolve("mmproj-Q8_0.gguf"), new byte[] {2});
        System.setProperty(MODEL_PROPERTY, model.toString());
        assertEquals(dir.resolve("mmproj-Q8_0.gguf").toString(), projector());
        System.setProperty(MediaOptions.MMPROJ_AUTO_PROPERTY, " false ");
        assertNull(projector(), "--mmproj-auto=false turns the sibling search off");
    }

    @Test
    void withoutAModelPropertyTheSearchHasNothingToSitBeside() {
        assertNull(projector());
    }
}
