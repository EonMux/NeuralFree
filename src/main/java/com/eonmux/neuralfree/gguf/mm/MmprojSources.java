package com.eonmux.neuralfree.gguf.mm;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * Where the projector of a multimodal model comes from, llama.cpp's {@code --mmproj} / {@code --mmproj-url} /
 * {@code --mmproj-auto} sources: an explicit path (a regular file, refused by name when it is not), a URL
 * downloaded once into the cache (http and https only), or the automatic sibling search that picks the
 * {@code *mmproj*.gguf} beside the model file. The sibling search is llama.cpp's {@code find_best_sibling}
 * (common/download.cpp): only files that share the model's directory prefix count, an exact quant tag match wins,
 * and otherwise the file whose quantization bits sit closest to the model's (or the requested tag's) does.
 * Immutable.
 */
public final class MmprojSources {

    /** The filename part the sibling search looks for. */
    public static final String SIBLING_KEYWORD = "mmproj";

    /** One connection and request timeout of a URL download. */
    private static final Duration TIMEOUT = Duration.ofSeconds(15);

    private MmprojSources() {
    }

    /**
     * The projector path of the three sources, in the priority the flags take: the explicit {@code path} when it is
     * given (a regular file, refused naming {@code flag} when it is not), else the {@code url} downloaded into
     * {@code cacheDir}, else the sibling of {@code modelPath} when {@code auto} asks for one; null when only the
     * automatic search ran and found no sibling, which is a text-only run and no error.
     *
     * @param flag      the flag {@code path} came from, named in the refusals
     * @param cacheDir  the directory a URL download lands in
     * @throws IOException          when the URL download fails
     * @throws IllegalArgumentException when the explicit path is not a regular file, the URL is not http or https,
     *                                  or the download target is unusable
     */
    public static Path resolve(String flag, String path, String url, boolean auto, Path modelPath, Path cacheDir)
            throws IOException {
        if (path != null && !path.trim().isEmpty()) {
            Path file = Paths.get(path.trim());
            if (!Files.isRegularFile(file)) {
                throw new IllegalArgumentException(flag + " is not a regular file: " + path.trim());
            }
            return file;
        }
        if (url != null && !url.trim().isEmpty()) {
            return downloadFromUrl(url.trim(), cacheDir);
        }
        return auto ? siblingBeside(modelPath) : null;
    }

    /**
     * The best {@code *mmproj*.gguf} of {@code candidates} for the model file {@code model}, llama.cpp's
     * {@code find_best_sibling}: a candidate counts when it ends in {@code .gguf}, carries the keyword, and its
     * directory prefix matches the model's path segments; the deepest shared prefix wins, then an exact
     * {@code tag} match ({@code -<TAG>.} in the name, as the download grammar writes it), then the smallest
     * distance of the candidate's quantization bits to the model's own (or the tag's). Null when none counts.
     *
     * @param tag the quant tag of the model request, or empty for none
     */
    public static String bestSibling(List<String> candidates, String model, String tag) {
        Objects.requireNonNull(candidates, "candidates");
        Objects.requireNonNull(model, "model");
        String tagUpper = tag == null ? "" : tag.toUpperCase(Locale.ROOT);
        int modelBits = !tagUpper.isEmpty() ? quantBits(tagUpper) : quantBits(model);
        List<String> modelParts = List.of(model.split("/"));
        String best = null;
        int bestDepth = 0;
        int bestDiff = 0;
        boolean bestExact = false;
        for (String candidate : candidates) {
            if (!candidate.toLowerCase(Locale.ROOT).endsWith(".gguf")
                    || !candidate.contains(SIBLING_KEYWORD)) {
                continue;
            }
            String[] parts = candidate.split("/");
            int modelDirs = modelParts.size() - 1;
            int depth = 0;
            while (depth < modelDirs && depth < parts.length - 1
                    && modelParts.get(depth).equals(parts[depth])) {
                depth++;
            }
            if (depth != parts.length - 1) {
                continue;
            }
            int diff = Math.abs(quantBits(candidate) - modelBits);
            boolean exact = !tagUpper.isEmpty() && candidate.toUpperCase(Locale.ROOT)
                    .contains("-" + tagUpper + ".");
            boolean better = best == null || depth > bestDepth
                    || (depth == bestDepth && exact && !bestExact)
                    || (depth == bestDepth && exact == bestExact && diff < bestDiff);
            if (better) {
                best = candidate;
                bestDepth = depth;
                bestDiff = diff;
                bestExact = exact;
            }
        }
        return best;
    }

    /**
     * A {@code *mmproj*.gguf} beside the model file {@code modelPath}: the best sibling of the {@code .gguf} files
     * of the model's directory, closest to the model's own quantization. Null when the directory holds none, and
     * for a model path with no directory.
     */
    public static Path siblingBeside(Path modelPath) {
        Path dir = modelPath.toAbsolutePath().normalize().getParent();
        if (dir == null || !Files.isDirectory(dir)) {
            return null;
        }
        try (Stream<Path> files = Files.list(dir)) {
            String best = null;
            int bestDiff = 0;
            String model = modelPath.getFileName().toString();
            for (Path file : files.sorted().toArray(Path[]::new)) {
                String name = file.getFileName().toString();
                if (!name.toLowerCase(Locale.ROOT).endsWith(".gguf") || !name.contains(SIBLING_KEYWORD)) {
                    continue;
                }
                int diff = Math.abs(quantBits(name) - quantBits(model));
                if (best == null || diff < bestDiff) {
                    best = name;
                    bestDiff = diff;
                }
            }
            return best == null ? null : dir.resolve(best);
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Downloads the projector at {@code url} into {@code cacheDir} and returns the file it landed at: the URL's
     * last path segment (its separators replaced, llama.cpp's {@code clean_file_name}) names the file, an already
     * cached file is kept, and the bytes land through a partial file moved into place when complete. Only http and
     * https are accepted; anything else is refused naming the URL.
     *
     * @throws IOException              when the download fails
     * @throws IllegalArgumentException when the URL is not an absolute http or https address
     */
    public static Path downloadFromUrl(String url, Path cacheDir) throws IOException {
        URI uri;
        try {
            uri = URI.create(url);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("--mmproj-url is not a valid URL: " + url, e);
        }
        String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
        if (!scheme.equals("http") && !scheme.equals("https")) {
            throw new IllegalArgumentException("--mmproj-url must be an http or https URL, got '" + url + "'");
        }
        String name = cleanName(uri.getPath());
        if (name.isEmpty()) {
            throw new IllegalArgumentException("--mmproj-url " + url + " names no file to cache the download as");
        }
        Path local = cacheDir.resolve(name);
        if (Files.isRegularFile(local)) {
            return local;
        }
        Path partial = Paths.get(local + ".downloadInProgress");
        Files.createDirectories(cacheDir);
        HttpClient client = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(TIMEOUT)
                .build();
        HttpRequest request = HttpRequest.newBuilder(uri).timeout(TIMEOUT).GET().build();
        HttpResponse<Path> response;
        try {
            response = client.send(request, HttpResponse.BodyHandlers.ofFile(partial));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("--mmproj-url download interrupted: " + url, e);
        } catch (IOException e) {
            throw new IOException("--mmproj-url cannot download '" + url + "': " + e.getMessage(), e);
        }
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            Files.deleteIfExists(partial);
            throw new IOException("--mmproj-url download answered HTTP " + response.statusCode() + " for '"
                    + url + "'");
        }
        try {
            Files.move(partial, local, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(partial, local, StandardCopyOption.REPLACE_EXISTING);
        }
        return local;
    }

    /** The first digit run of a GGUF quant tag ({@code Q4_K_M} → 4, {@code F16} → 16), 0 without one. */
    private static int quantBits(String name) {
        String tag = HubFilename.tagOf(name);
        for (int i = 0; i < tag.length(); i++) {
            if (tag.charAt(i) >= '0' && tag.charAt(i) <= '9') {
                StringBuilder digits = new StringBuilder();
                while (i < tag.length() && digits.length() < 9 && tag.charAt(i) >= '0'
                        && tag.charAt(i) <= '9') {
                    digits.append(tag.charAt(i));
                    i++;
                }
                return Integer.parseInt(digits.toString());
            }
        }
        return 0;
    }

    /** The URL's last path segment with its separators replaced ({@code /} and {@code \} to {@code _}). */
    private static String cleanName(String path) {
        String p = path == null ? "" : path;
        int slash = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
        String name = slash >= 0 ? p.substring(slash + 1) : p;
        return name.replace('/', '_').replace('\\', '_');
    }

    /**
     * The trailing quant tag of a GGUF filename ({@code [-.]}-separated alphanumeric segment, uppercased), the
     * rule {@code HubCache.GgufFilename} parses the repository names with; re-declared here so the sibling search
     * runs without the hub package.
     */
    private static final class HubFilename {

        private HubFilename() {
        }

        static String tagOf(String name) {
            String without = name.toLowerCase(Locale.ROOT).endsWith(".gguf")
                    ? name.substring(0, name.length() - ".gguf".length()) : name;
            java.util.regex.Matcher m = TAG_PATTERN.matcher(without);
            return m.find() ? m.group(1).toUpperCase(Locale.ROOT) : "";
        }
    }

    private static final java.util.regex.Pattern TAG_PATTERN =
            java.util.regex.Pattern.compile("[-.]([A-Za-z0-9_]+)$");
}
