package com.eonmux.neuralfree.server;

import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static org.junit.jupiter.api.Assertions.*;

/** Static directory mounting, content revalidation and confinement of both ordinary paths and symlinks. */
public class UiDirectoryTest {

    @Test
    void directoryFilesServeAtTheirOwnPathsAndUseContentVersions(@TempDir Path root) throws IOException {
        Files.createDirectories(root.resolve("assets"));
        Files.writeString(root.resolve("index.html"), "<link href=\"/assets/site.css\"><script src='assets/site.js'></script>");
        Files.writeString(root.resolve("assets/site.css"), "body { color: red; }");
        Files.writeString(root.resolve("assets/site.js"), "'use strict';");
        UiResource ui = new UiResource(true, root, "/llm");
        String document = text(ui.root(null));
        assertTrue(document.contains("/llm/assets/site.css?v="), document);
        assertTrue(document.contains("/llm/assets/site.js?v="), document);
        Response plain = ui.staticFile("assets/site.css", null, null);
        assertEquals("no-cache", plain.getHeaderString("Cache-Control"));
        String version = plain.getHeaderString("ETag").substring(1, 13);
        Response versioned = ui.staticFile("assets/site.css", version, null);
        assertEquals("public, max-age=31536000, immutable", versioned.getHeaderString("Cache-Control"));
        assertEquals(304, ui.staticFile("assets/site.css", version, versioned.getHeaderString("ETag")).getStatus());
        Files.writeString(root.resolve("assets/site.css"), "body { color: blue; }");
        assertNotEquals(document, text(ui.root(null)));
        Response changed = ui.staticFile("assets/site.css", version, plain.getHeaderString("ETag"));
        assertEquals(200, changed.getStatus());
        assertEquals("no-cache", changed.getHeaderString("Cache-Control"));
        assertNotEquals(plain.getHeaderString("ETag"), changed.getHeaderString("ETag"));
    }

    @Test
    void theDirectoryReplacesRatherThanFallsBackToTheBundle(@TempDir Path root) throws IOException {
        UiResource ui = new UiResource(true, root, "");
        assertEquals(404, statusOf(() -> ui.root(null)), "a missing index must not throw a null-pointer error");
        assertEquals(404, statusOf(() -> ui.asset("app.js", null, null)));
        Files.createDirectory(root.resolve("ui"));
        Files.writeString(root.resolve("ui/app.js"), "custom application");
        assertEquals("custom application", text(ui.asset("app.js", null, null)));
        Files.createDirectories(root.resolve("docs"));
        Files.writeString(root.resolve("docs/index.html"), "<link href='style.css'>");
        Files.writeString(root.resolve("docs/style.css"), "h1 {}");
        assertTrue(text(ui.staticFile("docs", null, null)).contains("/docs/style.css?v="));
    }

    @Test
    void escapesAndSymlinksOutsideTheRootAreNeverRead(@TempDir Path dir) throws IOException {
        Path root = Files.createDirectory(dir.resolve("public"));
        Files.writeString(dir.resolve("secret.txt"), "not public");
        Files.createSymbolicLink(root.resolve("escape.txt"), dir.resolve("secret.txt"));
        Files.createSymbolicLink(root.resolve("escape-dir"), dir);
        Files.writeString(root.resolve("safe.txt"), "public");
        Files.createSymbolicLink(root.resolve("safe-link.txt"), root.resolve("safe.txt"));
        UiResource ui = new UiResource(true, root, "");
        for (String path : new String[]{"../secret.txt", "a/../../secret.txt", "/secret.txt", "..\\secret.txt",
                "%2e%2e/secret.txt", "escape.txt", "escape-dir/secret.txt", "escape-dir", "missing.txt"}) {
            assertEquals(404, statusOf(() -> ui.staticFile(path, null, null)), path);
        }
        assertEquals("public", text(ui.staticFile("safe-link.txt", null, null)));
    }

    @Test
    void aFileIsNotAStaticDirectory(@TempDir Path dir) throws IOException {
        Path file = Files.writeString(dir.resolve("file"), "x");
        assertTrue(assertThrows(IllegalArgumentException.class, () -> new UiResource(true, file, ""))
                .getMessage().contains(UiResource.PATH_PROPERTY));
    }

    private static String text(Response response) {
        return new String((byte[]) response.getEntity(), StandardCharsets.UTF_8);
    }
}
