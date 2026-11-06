package com.eonmux.neuralfree.server;

import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.*;

/** The bundled UI's document, content-derived URLs, cache policies, ETags, classpath loading and refusals. */
public class UiResourceTest {

    private static UiResource resource;

    @BeforeAll
    static void load() {
        resource = new UiResource();
    }

    @Test
    void theRootServesTheChatDocument() {
        Response document = resource.root(null);
        assertEquals(200, document.getStatus());
        assertTrue(String.valueOf(document.getMediaType()).startsWith("text/html"));
        assertTrue(bodyOf(document).toLowerCase().startsWith("<!doctype html"));
        assertTrue(bodyOf(document).contains("/ui/app.js?v="));
        assertEquals("no-cache", document.getHeaderString("Cache-Control"));
        assertEquals("require-corp", document.getHeaderString("Cross-Origin-Embedder-Policy"));
        assertEquals("same-origin", document.getHeaderString("Cross-Origin-Opener-Policy"));
        assertNotNull(document.getHeaderString("ETag"));
    }

    @Test
    void everyDocumentRouteServesTheSameResponse() {
        Response root = resource.root(null);
        assertEquals(bodyOf(root), bodyOf(resource.indexHtml(null)));
        assertEquals(bodyOf(root), bodyOf(resource.uiRoot(null)));
        assertEquals(root.getHeaderString("ETag"), resource.indexHtml(null).getHeaderString("ETag"));
    }

    @Test
    void aMatchingEtagAnswersNotModified() {
        String etag = resource.root(null).getHeaderString("ETag");
        Response fresh = resource.root(etag);
        assertEquals(304, fresh.getStatus());
        assertNull(fresh.getEntity());
        assertEquals(etag, fresh.getHeaderString("ETag"));
        assertEquals(304, resource.root("W/" + etag).getStatus());
        assertEquals(200, resource.root("\"stale\"").getStatus());
    }

    @Test
    void onlyCurrentVersionedAssetsCacheForever() {
        for (String name : List.of("app.js", "app.css", "markdown.js", "favicon.svg")) {
            Response unversioned = resource.asset(name, null, null);
            String version = versionOf(unversioned);
            Response versioned = resource.asset(name, version, null);
            assertEquals(200, versioned.getStatus());
            assertEquals("public, max-age=31536000, immutable", versioned.getHeaderString("Cache-Control"), name);
            assertEquals("no-cache", unversioned.getHeaderString("Cache-Control"), name);
            assertEquals("no-cache", resource.asset(name, "stale", null).getHeaderString("Cache-Control"), name);
            String etag = versioned.getHeaderString("ETag");
            assertEquals(304, resource.asset(name, version, etag).getStatus());
            Response stale = resource.asset(name, "stale", etag);
            assertEquals(304, stale.getStatus());
            assertEquals("no-cache", stale.getHeaderString("Cache-Control"));
            assertNull(versioned.getHeaderString("Cross-Origin-Embedder-Policy"));
        }
        assertTrue(String.valueOf(resource.asset("app.js", null, null).getMediaType()).startsWith("text/javascript"));
        assertTrue(String.valueOf(resource.asset("app.css", null, null).getMediaType()).startsWith("text/css"));
    }

    @Test
    void thePwaManifestRevalidatesEvenWithItsCurrentVersion() {
        Response manifest = resource.asset("manifest.webmanifest", null, null);
        assertEquals(200, manifest.getStatus());
        assertEquals("application/manifest+json", String.valueOf(manifest.getMediaType()));
        assertEquals("no-cache", resource.asset("manifest.webmanifest", versionOf(manifest), null)
                .getHeaderString("Cache-Control"));
    }

    @Test
    void theIconServesAsAnImage() {
        Response icon = resource.asset("favicon.svg", null, null);
        assertEquals(200, icon.getStatus());
        assertEquals("image/svg+xml", String.valueOf(icon.getMediaType()));
        assertTrue(bodyOf(icon).contains("<svg"));
    }

    @Test
    void anUnknownAssetIsRefusedByName() {
        WebApplicationException missing = assertThrows(WebApplicationException.class,
                () -> resource.asset("absent.js", null, null));
        assertEquals(404, missing.getResponse().getStatus());
        assertTrue(view(missing.getResponse().getEntity()).obj("error").str("message").contains("absent.js"));
        assertEquals("invalid_request_error", view(missing.getResponse().getEntity()).obj("error").str("type"));
    }

    @Test
    void malformedAssetNamesAreRefusedByName() {
        for (String name : List.of("../app.js", "a/b", "a\\b", "..", ".hidden", "a b", "<script>",
                "app.js%00", "a".repeat(200))) {
            WebApplicationException refused = assertThrows(WebApplicationException.class,
                    () -> resource.asset(name, null, null), name);
            assertEquals(400, refused.getResponse().getStatus(), name);
            assertTrue(view(refused.getResponse().getEntity()).obj("error").str("message")
                    .startsWith("Invalid UI asset"));
        }
    }

    @Test
    void everyDocumentReferenceNamesTheCurrentAssetHash() {
        String document = bodyOf(resource.root(null));
        Matcher references = Pattern.compile("(?:src|href)=\"/ui/([^\"?]+)\\?v=([0-9a-f]{12})\"").matcher(document);
        Set<String> referenced = new java.util.HashSet<>();
        while (references.find()) {
            String name = references.group(1);
            Response asset = resource.asset(name, null, null);
            assertEquals(versionOf(asset), references.group(2), name);
            referenced.add(name);
        }
        assertEquals(Set.of("app.js", "app.css", "markdown.js", "favicon.svg", "manifest.webmanifest"), referenced);
        assertFalse(Pattern.compile("(?:src|href)=\"/ui/[^\"?]+\"").matcher(document).find(),
                "no document reference may keep an unversioned asset URL");
    }

    @Test
    void thePanelsFollowTheSelectedModelInRouterMode() {
        String script = bodyOf(resource.asset("app.js", null, null));
        assertTrue(script.contains("/props?model="), "the server panel names the selected model");
        assertTrue(script.contains("/slots?model="), "the slots panel names the selected model");
        assertTrue(script.contains("'/models/load'"), "the router's load action is bundled");
        assertTrue(script.contains("encodeURIComponent"), "the model name travels URL-encoded");
        assertTrue(bodyOf(resource.root(null)).contains("id=\"model-load\""),
                "the document carries the load action");
    }

    @Test
    void theBundleCarriesTheWholeApp() {
        for (String name : List.of("index.html", "app.css", "app.js", "markdown.js",
                "favicon.svg", "manifest.webmanifest")) {
            assertTrue(resource.names().contains(name), name);
            assertEquals(200, resource.asset(name, null, null).getStatus(), name);
        }
    }

    @Test
    void etagsMatchTheAssetBytes() {
        String document = resource.root(null).getHeaderString("ETag");
        assertEquals(document, resource.asset("index.html", null, null).getHeaderString("ETag"));
        assertNotEquals(document, resource.asset("app.js", null, null).getHeaderString("ETag"));
        assertTrue(document.startsWith("\"") && document.endsWith("\""));
    }

    @Test
    void anAssetlessClasspathRefusesEveryRouteByName() {
        UiResource empty = new UiResource(true, Map.of(), null);
        WebApplicationException root = assertThrows(WebApplicationException.class, () -> empty.root(null));
        assertEquals(404, root.getResponse().getStatus());
        assertEquals("the bundled UI is not on the classpath",
                view(root.getResponse().getEntity()).obj("error").str("message"));
        assertEquals(404, statusOf(() -> empty.asset("app.js", null, null)));
        assertEquals(404, statusOf(() -> empty.indexHtml(null)));
        assertEquals(404, statusOf(() -> empty.uiRoot(null)));
        assertFalse(resource.names().isEmpty());
    }

    @Test
    void theDocumentAlsoServesUnderItsOwnAssetName() {
        Response alias = resource.asset("index.html", null, null);
        assertEquals(200, alias.getStatus());
        assertEquals("no-cache", alias.getHeaderString("Cache-Control"));
        assertEquals("require-corp", alias.getHeaderString("Cross-Origin-Embedder-Policy"));
    }

    @Test
    void assetsLoadFromAJarLikeTheUberJar(@TempDir Path dir) throws IOException {
        Path jar = dir.resolve("ui.jar");
        try (JarOutputStream out = new JarOutputStream(Files.newOutputStream(jar))) {
            entry(out, "ui/index.html", "<!doctype html><script src=\"/ui/app.js\"></script>");
            entry(out, "ui/app.js", "'use strict';");
            entry(out, "elsewhere/x.html", "x");
        }
        try (URLClassLoader loader = new URLClassLoader(new URL[]{jar.toUri().toURL()}, null)) {
            UiResource fromJar = new UiResource(true, UiResource.load(loader, ""), "");
            assertTrue(fromJar.names().contains("app.js"));
            assertFalse(fromJar.names().contains("x.html"));
            assertTrue(bodyOf(fromJar.root(null)).contains("/ui/app.js?v="));
            assertEquals("no-cache", fromJar.asset("app.js", null, null).getHeaderString("Cache-Control"));
        }
    }

    @Test
    void aChangedAssetChangesItsUrlAndTheDocumentEtag(@TempDir Path dir) throws IOException {
        Path ui = Files.createDirectory(dir.resolve("ui"));
        Files.writeString(ui.resolve("index.html"), "<script src=\"/ui/app.js\"></script>");
        Files.writeString(ui.resolve("app.js"), "first version");
        try (URLClassLoader loader = new URLClassLoader(new URL[]{dir.toUri().toURL()}, null)) {
            UiResource first = new UiResource(true, UiResource.load(loader, "/llm"), "/llm");
            Files.writeString(ui.resolve("app.js"), "second version");
            UiResource second = new UiResource(true, UiResource.load(loader, "/llm"), "/llm");
            assertNotEquals(bodyOf(first.root(null)), bodyOf(second.root(null)));
            assertNotEquals(first.root(null).getHeaderString("ETag"), second.root(null).getHeaderString("ETag"));
            assertTrue(bodyOf(second.root(null)).contains("/llm/ui/app.js?v="));
            assertEquals("first version", bodyOf(first.asset("app.js", null, null)),
                    "a loaded bundle is a stable snapshot, not re-read for each request");
        }
    }

    @Test
    void aDisabledUiRefusesEveryUiRoute() {
        UiResource disabled = new UiResource(false, Map.of(), null);
        assertEquals(404, statusOf(() -> disabled.root(null)));
        assertEquals(404, statusOf(() -> disabled.indexHtml(null)));
        assertEquals(404, statusOf(() -> disabled.uiRoot(null)));
        assertEquals(404, statusOf(() -> disabled.asset("app.js", null, null)));
    }

    @Test
    void theWebuiPropertyIsTrueFalseOrAbsent() {
        assertTrue(UiResource.enabled(Map.of()), "the UI serves unless turned off");
        assertTrue(UiResource.enabled(Map.of(UiResource.WEBUI_PROPERTY, "true")));
        assertFalse(UiResource.enabled(Map.of(UiResource.WEBUI_PROPERTY, "false")));
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> UiResource.enabled(Map.of(UiResource.WEBUI_PROPERTY, "maybe")));
        assertTrue(refused.getMessage().startsWith(UiResource.WEBUI_PROPERTY), refused.getMessage());
    }

    @Test
    void theUiConfigPropertyParsesAtStartupAndRefusesBadJsonByName() {
        assertEquals(Map.of(), UiResource.uiSettings(Map.of()), "without a config the defaults are empty");
        assertEquals("light", ((Map<?, ?>) UiResource.uiSettings(Map.of(UiResource.CONFIG_PROPERTY,
                "{\"theme\": \"light\"}"))).get("theme"));
        assertEquals(7, ((Number) UiResource.uiSettings(Map.of(UiResource.CONFIG_PROPERTY, "7"))).intValue(),
                "llama.cpp's ui_config takes any JSON value");
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> UiResource.uiSettings(Map.of(UiResource.CONFIG_PROPERTY, "{not json")));
        assertTrue(refused.getMessage().startsWith(UiResource.CONFIG_PROPERTY), refused.getMessage());
    }

    private static void entry(JarOutputStream out, String name, String content) throws IOException {
        out.putNextEntry(new JarEntry(name));
        out.write(content.getBytes(StandardCharsets.UTF_8));
        out.closeEntry();
    }

    private static String versionOf(Response response) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest((byte[]) response.getEntity())).substring(0, 12);
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    private static String bodyOf(Response response) {
        return new String((byte[]) response.getEntity(), StandardCharsets.UTF_8);
    }
}
