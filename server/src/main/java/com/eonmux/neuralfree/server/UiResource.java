package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.Json;
import io.quarkus.runtime.Startup;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.HeaderParam;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.HttpHeaders;
import jakarta.ws.rs.core.Response;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.JarURLConnection;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Enumeration;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;
import java.util.jar.JarEntry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * The bundled chat UI or the static directory selected by {@code --path}. Documents reference assets by a short
 * content hash; only requests naming the current hash are immutable. Documents, PWA metadata and unversioned or
 * stale assets revalidate with ETags. A disabled UI answers 404, and directory files must resolve inside its real
 * root, including through symlinks. {@link ApiPrefixRouter} mounts all these routes under the API prefix.
 */
@Startup
@Path("/")
public class UiResource {

    static final String ASSET_DIR = "ui";
    static final String WEBUI_PROPERTY = "neuralfree.server.webui";
    static final String PATH_PROPERTY = "neuralfree.server.ui.path";
    static final String CONFIG_PROPERTY = "neuralfree.server.ui.config";
    private static final String DOCUMENT = "index.html";
    private static final Set<String> NO_CACHE = Set.of("manifest.webmanifest", "sw.js", "version.json", "build.json");
    private static final String CACHE_IMMUTABLE = "public, max-age=31536000, immutable";
    private static final String CACHE_REVALIDATE = "no-cache";
    static final Pattern NAME = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._-]{0,127}");
    private static final Pattern REFERENCE = Pattern.compile("(?i)(\\b(?:src|href)\\s*=\\s*)([\"'])([^\"']*)\\2");

    /** An immutable asset snapshot with one digest computed from its bytes. */
    record Asset(String name, byte[] data, String type, String etag, String version) {
        Asset {
            data = data.clone();
        }

        @Override
        public byte[] data() {
            return data.clone();
        }

        static Asset of(String name, byte[] data) {
            try {
                String hash = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(data));
                return new Asset(name, data, mediaType(name), '"' + hash.substring(0, 32) + '"', hash.substring(0, 12));
            } catch (NoSuchAlgorithmException e) {
                throw new IllegalStateException("SHA-256 is not available", e);
            }
        }
    }

    private final Map<String, Asset> assets;
    private final java.nio.file.Path directory;
    private final boolean enabled;
    private final String prefix;

    /** Launch properties are checked at startup, including launches using only {@code -D} properties. */
    public UiResource() {
        this(enabled(System.getProperties()), directory(System.getProperties()),
                ApiPrefixRouter.fromProperties(System.getProperties()));
        uiSettings(System.getProperties());
    }

    UiResource(boolean enabled, Map<String, Asset> assets, String prefix) {
        this.enabled = enabled;
        this.assets = Map.copyOf(assets);
        this.directory = null;
        this.prefix = prefix == null ? "" : prefix;
    }

    UiResource(boolean enabled, java.nio.file.Path directory, String prefix) {
        this.enabled = enabled;
        this.directory = directory == null ? null : realDirectory(directory);
        this.prefix = prefix == null ? "" : prefix;
        this.assets = enabled && directory == null
                ? load(Thread.currentThread().getContextClassLoader(), this.prefix) : Map.of();
    }

    Set<String> names() {
        return assets.keySet();
    }

    @GET
    @Path("/")
    public Response root(@HeaderParam("If-None-Match") String ifNoneMatch) {
        return document(ifNoneMatch);
    }

    @GET
    @Path("/index.html")
    public Response indexHtml(@HeaderParam("If-None-Match") String ifNoneMatch) {
        return document(ifNoneMatch);
    }

    @GET
    @Path("/ui")
    public Response uiRoot(@HeaderParam("If-None-Match") String ifNoneMatch) {
        return directory == null ? document(ifNoneMatch) : staticFile("ui/index.html", null, ifNoneMatch);
    }

    /** A bundled asset, or the same path under the custom static root. */
    @GET
    @Path("/ui/{asset}")
    public Response asset(@PathParam("asset") String name, @QueryParam("v") String version,
                          @HeaderParam("If-None-Match") String ifNoneMatch) {
        requireEnabled();
        if (name == null || !NAME.matcher(name).matches() || name.contains("..")) {
            throw ServerJson.badRequest("Invalid UI asset '" + name + "'");
        }
        if (directory != null) {
            return staticFile("ui/" + name, version, ifNoneMatch);
        }
        if (name.equals(DOCUMENT)) {
            return document(ifNoneMatch);
        }
        Asset asset = assets.get(name);
        if (asset == null) {
            throw missing(name);
        }
        return serve(asset, cacheOf(asset, version), false, ifNoneMatch);
    }

    /** Static files at their directory-relative URLs; concrete API routes take precedence over this fallback. */
    @GET
    @Path("/{file: .+}")
    public Response staticFile(@PathParam("file") String name, @QueryParam("v") String version,
                               @HeaderParam("If-None-Match") String ifNoneMatch) {
        requireEnabled();
        if (directory == null) {
            throw missing(name);
        }
        java.nio.file.Path file = directoryFile(name);
        if (file == null) {
            throw missing(name);
        }
        String relative = directory.relativize(file).toString().replace('\\', '/');
        Asset asset = Asset.of(relative, read(file));
        boolean document = extension(relative).equals("html");
        if (document) {
            int slash = relative.lastIndexOf('/');
            String base = slash < 0 ? "" : relative.substring(0, slash + 1);
            asset = rewritten(asset, base, path -> {
                java.nio.file.Path target = directoryFile(path);
                return target == null ? null : Asset.of(path, read(target)).version();
            });
        }
        return serve(asset, document ? CACHE_REVALIDATE : cacheOf(asset, version), document, ifNoneMatch);
    }

    private Response document(String ifNoneMatch) {
        requireEnabled();
        if (directory != null) {
            return staticFile(DOCUMENT, null, ifNoneMatch);
        }
        Asset asset = assets.get(DOCUMENT);
        if (asset == null) {
            throw ServerJson.error(404, "the bundled UI is not on the classpath", "unavailable_error");
        }
        return serve(asset, CACHE_REVALIDATE, true, ifNoneMatch);
    }

    private static String cacheOf(Asset asset, String version) {
        String basename = asset.name().substring(asset.name().lastIndexOf('/') + 1);
        return !NO_CACHE.contains(basename) && asset.version().equals(version) ? CACHE_IMMUTABLE : CACHE_REVALIDATE;
    }

    private static Response serve(Asset asset, String cacheControl, boolean isolation, String ifNoneMatch) {
        boolean matches = ifNoneMatch != null
                && (ifNoneMatch.equals(asset.etag()) || ifNoneMatch.equals("W/" + asset.etag()));
        Response.ResponseBuilder out = matches ? Response.status(304) : Response.ok(asset.data()).type(asset.type());
        out.header(HttpHeaders.ETAG, asset.etag()).header(HttpHeaders.CACHE_CONTROL, cacheControl);
        if (isolation) {
            out.header("Cross-Origin-Embedder-Policy", "require-corp")
                    .header("Cross-Origin-Opener-Policy", "same-origin");
        }
        return out.build();
    }

    /** Resolves only regular files inside the real root; directories use their own index.html, never a listing. */
    private java.nio.file.Path directoryFile(String name) {
        if (name == null || name.startsWith("/") || name.contains("\\") || name.contains("\0")
                || name.contains("%") || Stream.of(name.split("/")).anyMatch(segment -> segment.equals(".."))) {
            return null;
        }
        try {
            java.nio.file.Path candidate = directory.resolve(name).normalize();
            if (!candidate.startsWith(directory)) {
                return null;
            }
            java.nio.file.Path real = candidate.toRealPath();
            if (!real.startsWith(directory)) {
                return null;
            }
            if (Files.isDirectory(real)) {
                real = real.resolve(DOCUMENT).toRealPath();
            }
            return real.startsWith(directory) && Files.isRegularFile(real) ? real : null;
        } catch (IOException e) {
            return null;
        }
    }

    private void requireEnabled() {
        if (!enabled) {
            throw ServerJson.error(404, "the web UI is disabled", "unavailable_error");
        }
    }

    private static WebApplicationException missing(String name) {
        return ServerJson.error(404, "Unknown UI asset '" + name + "'", "invalid_request_error");
    }

    private static String mediaType(String name) {
        String type = switch (extension(name)) {
            case "html" -> "text/html";
            case "js", "mjs" -> "text/javascript";
            case "css" -> "text/css";
            case "svg" -> "image/svg+xml";
            case "webmanifest" -> "application/manifest+json";
            case "json" -> "application/json";
            case "png" -> "image/png";
            case "jpg", "jpeg" -> "image/jpeg";
            case "gif" -> "image/gif";
            case "webp" -> "image/webp";
            case "ico" -> "image/x-icon";
            case "woff2" -> "font/woff2";
            case "txt" -> "text/plain";
            default -> "application/octet-stream";
        };
        return type.startsWith("text/") ? type + "; charset=utf-8" : type;
    }

    private static String extension(String name) {
        int dot = name.lastIndexOf('.');
        return dot < 0 ? "" : name.substring(dot + 1).toLowerCase(Locale.ROOT);
    }

    private Asset rewritten(Asset document, String base, UnaryOperator<String> versionOf) {
        String text = new String(document.data(), StandardCharsets.UTF_8);
        return Asset.of(document.name(), rewrite(text, prefix, base, versionOf).getBytes(StandardCharsets.UTF_8));
    }

    /** Versions local document references without changing external URLs or fragment-only links. */
    static String rewrite(String document, String prefix, String base, UnaryOperator<String> versionOf) {
        Matcher reference = REFERENCE.matcher(document);
        StringBuilder out = new StringBuilder();
        while (reference.find()) {
            String url = reference.group(3);
            String rewritten = versionedUrl(url, prefix, base, versionOf);
            reference.appendReplacement(out, Matcher.quoteReplacement(reference.group(1) + reference.group(2)
                    + rewritten + reference.group(2)));
        }
        reference.appendTail(out);
        return out.toString();
    }

    private static String versionedUrl(String url, String prefix, String base, UnaryOperator<String> versionOf) {
        if (url.isEmpty() || url.startsWith("#") || url.startsWith("//") || url.contains(":") || url.contains("\\")) {
            return url;
        }
        int fragment = url.indexOf('#');
        String anchor = fragment < 0 ? "" : url.substring(fragment);
        String path = fragment < 0 ? url : url.substring(0, fragment);
        int query = path.indexOf('?');
        String params = query < 0 ? "" : path.substring(query + 1);
        path = query < 0 ? path : path.substring(0, query);
        String name = path.startsWith("/") ? path.substring(1) : base + path;
        String version = versionOf.apply(name);
        if (version == null) {
            return url;
        }
        String remaining = Stream.of(params.split("&")).filter(p -> !p.isEmpty() && !p.startsWith("v="))
                .collect(java.util.stream.Collectors.joining("&"));
        return prefix + "/" + name + "?" + (remaining.isEmpty() ? "" : remaining + "&") + "v=" + version + anchor;
    }

    /** Loads and versions the bundle once, from exploded resources or jars without directory entries. */
    static Map<String, Asset> load(ClassLoader loader, String prefix) {
        Map<String, Asset> out = new LinkedHashMap<>();
        try {
            Enumeration<URL> roots = loader.getResources(ASSET_DIR);
            while (roots.hasMoreElements()) {
                URL root = roots.nextElement();
                if ("file".equals(root.getProtocol())) {
                    readDirectory(java.nio.file.Path.of(root.toURI()), out);
                } else if ("jar".equals(root.getProtocol())) {
                    readJar((JarURLConnection) root.openConnection(), out);
                }
            }
            if (out.isEmpty()) {
                URL entry = loader.getResource(ASSET_DIR + "/" + DOCUMENT);
                if (entry != null && "jar".equals(entry.getProtocol())) {
                    readJar((JarURLConnection) entry.openConnection(), out);
                }
            }
        } catch (IOException | URISyntaxException e) {
            throw new IllegalStateException("the bundled UI assets do not read from the classpath", e);
        }
        Asset document = out.get(DOCUMENT);
        if (document != null) {
            String text = rewrite(new String(document.data(), StandardCharsets.UTF_8), prefix, "", path -> {
                Asset target = path.startsWith("ui/") ? out.get(path.substring(3)) : null;
                return target == null || target == document ? null : target.version();
            });
            out.put(DOCUMENT, Asset.of(DOCUMENT, text.getBytes(StandardCharsets.UTF_8)));
        }
        return Map.copyOf(out);
    }

    private static void readDirectory(java.nio.file.Path root, Map<String, Asset> out) throws IOException {
        try (Stream<java.nio.file.Path> files = Files.walk(root)) {
            files.filter(Files::isRegularFile).forEach(file -> {
                String name = root.relativize(file).toString().replace('\\', '/');
                out.put(name, Asset.of(name, read(file)));
            });
        }
    }

    private static byte[] read(java.nio.file.Path file) {
        try {
            return Files.readAllBytes(file);
        } catch (IOException e) {
            throw new UncheckedIOException("the UI asset " + file + " does not read", e);
        }
    }

    private static void readJar(JarURLConnection connection, Map<String, Asset> out) throws IOException {
        Enumeration<JarEntry> entries = connection.getJarFile().entries();
        while (entries.hasMoreElements()) {
            JarEntry entry = entries.nextElement();
            String path = ASSET_DIR + "/";
            if (!entry.isDirectory() && entry.getName().startsWith(path)) {
                String name = entry.getName().substring(path.length());
                try (InputStream in = connection.getJarFile().getInputStream(entry)) {
                    out.put(name, Asset.of(name, in.readAllBytes()));
                }
            }
        }
    }

    static boolean enabled(Map<?, ?> properties) {
        Object value = properties.get(WEBUI_PROPERTY);
        if (value == null) {
            return true;
        }
        String text = String.valueOf(value).trim();
        if (!text.equalsIgnoreCase("true") && !text.equalsIgnoreCase("false")) {
            throw new IllegalArgumentException(WEBUI_PROPERTY + " must be true or false, got '" + text + "'");
        }
        return Boolean.parseBoolean(text);
    }

    /** llama.cpp's /props ui_settings, preserving any JSON value supplied by --ui-config. */
    static Object uiSettings(Map<?, ?> properties) {
        Object raw = properties.get(CONFIG_PROPERTY);
        if (raw == null) {
            return Map.of();
        }
        try {
            return Json.parse(String.valueOf(raw), Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(CONFIG_PROPERTY + " is not valid JSON: " + e.getMessage(), e);
        }
    }

    private static java.nio.file.Path directory(Map<?, ?> properties) {
        Object raw = properties.get(PATH_PROPERTY);
        return raw == null ? null : realDirectory(java.nio.file.Path.of(String.valueOf(raw)));
    }

    private static java.nio.file.Path realDirectory(java.nio.file.Path path) {
        try {
            java.nio.file.Path real = path.toRealPath();
            if (Files.isDirectory(real)) {
                return real;
            }
        } catch (IOException e) {
            throw new IllegalArgumentException(PATH_PROPERTY + " does not name a readable directory: " + path, e);
        }
        throw new IllegalArgumentException(PATH_PROPERTY + " does not name a directory: " + path);
    }
}
