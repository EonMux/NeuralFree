package com.eonmux.neuralfree.gguf.media;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLDecoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Loads the bytes a media reference names, the counterpart of vLLM's {@code MediaConnector}
 * (vllm/multimodal/media/connector.py) and llama.cpp's {@code handle_media} (tools/server/server-common.cpp):
 *
 * <ul>
 *   <li>{@code data:} URLs of RFC 2397 with base64 data, any media type;</li>
 *   <li>{@code http://} and {@code https://} URLs, fetched with a timeout and a size limit, only from the allowed
 *       domains when a domain list is set ({@code --allowed-media-domains}); every redirect target passes the same
 *       scheme and domain check before it is followed, so a redirect cannot leave the allowed set;</li>
 *   <li>{@code file://} URLs, only for files inside the allowed local directory ({@code --allowed-local-media-path},
 *       llama.cpp's {@code --media-path}), symbolic links resolved before the check, and refused without one;</li>
 *   <li>any other string is read as raw base64, llama.cpp's fallback.</li>
 * </ul>
 *
 * Every payload, fetched or decoded, is bounded by {@link Options#maxBytes()}. Failures raise
 * {@link MediaInputException} naming the field and the reason. Immutable and thread-safe.
 */
public final class MediaConnector {

    /** Redirect hops one fetch follows at most. */
    static final int MAX_REDIRECTS = 5;

    private final Options options;
    private final HttpClient http;

    /** A connector over {@code options}. */
    public MediaConnector(Options options) {
        this.options = Objects.requireNonNull(options, "options");
        this.http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NEVER)
                .connectTimeout(options.timeout())
                .build();
    }

    /** The options the connector loads under. */
    public Options options() {
        return options;
    }

    /** Loaded media bytes and the media type the reference declared ({@code data:} URLs), or null. */
    public static final class Loaded {
        private final byte[] bytes;
        private final String mediaType;

        Loaded(byte[] bytes, String mediaType) {
            this.bytes = bytes;
            this.mediaType = mediaType;
        }

        /** The payload (a copy). */
        public byte[] bytes() {
            return bytes.clone();
        }

        /** The payload size in bytes. */
        public int size() {
            return bytes.length;
        }

        /** The declared media type such as {@code image/png}, or null when the reference declares none. */
        public String mediaType() {
            return mediaType;
        }
    }

    /**
     * The bytes {@code reference} names.
     *
     * @param reference a data, http(s) or file URL, or raw base64
     * @param field     the request field the reference came from, named in errors
     * @throws MediaInputException when the reference is malformed, refused, unreachable or too large
     */
    public Loaded load(String reference, String field) {
        if (reference == null || reference.trim().isEmpty()) {
            throw new MediaInputException(field + " is empty: a media URL or base64 data is required");
        }
        String ref = reference.trim();
        if (startsWithIgnoreAsciiCase(ref, "data:")) {
            return dataUrl(ref, field);
        }
        if (startsWithIgnoreAsciiCase(ref, "http://") || startsWithIgnoreAsciiCase(ref, "https://")) {
            return new Loaded(fetch(ref, field), null);
        }
        if (startsWithIgnoreAsciiCase(ref, "file:")) {
            return new Loaded(localFile(ref, field), null);
        }
        return new Loaded(base64(ref, 0, ref.length(), field), null);
    }

    /**
     * Whether {@code s} starts with the lower-case ASCII {@code prefix}, upper-case ASCII letters folded: the test
     * {@code s.toLowerCase(Locale.ROOT).startsWith(prefix)} makes for these scheme prefixes, without lower-casing a
     * reference that may be tens of megabytes of base64.
     */
    private static boolean startsWithIgnoreAsciiCase(String s, String prefix) {
        if (s.length() < prefix.length()) {
            return false;
        }
        for (int i = 0; i < prefix.length(); i++) {
            char c = s.charAt(i);
            if (c >= 'A' && c <= 'Z') {
                c = (char) (c + ('a' - 'A'));
            }
            if (c != prefix.charAt(i)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Decodes raw base64 (or a base64 {@code data:} URL) without fetching anything: the form of fields that carry
     * inline data only, such as llama.cpp's {@code multimodal_data} and OpenAI's {@code input_audio.data}.
     *
     * @throws MediaInputException when the data is not base64 or is too large
     */
    public Loaded inline(String data, String field) {
        if (data == null || data.trim().isEmpty()) {
            throw new MediaInputException(field + " is empty: base64 data is required");
        }
        String ref = data.trim();
        if (ref.regionMatches(true, 0, "data:", 0, 5)) {
            return dataUrl(ref, field);
        }
        return new Loaded(base64(ref, 0, ref.length(), field), null);
    }

    // ------------------------------------------------------------------ data and base64

    /** An RFC 2397 data URL, {@code data:[<mediatype>][;<param>=<value>]*;base64,<data>} (vLLM's parsing). */
    private Loaded dataUrl(String url, String field) {
        int comma = url.indexOf(',');
        if (comma < 0) {
            throw new MediaInputException(field + " is not a valid data URL: the ',' separator is missing");
        }
        String spec = url.substring(5, comma);
        int lastSemi = spec.lastIndexOf(';');
        if (lastSemi < 0 || !"base64".equalsIgnoreCase(spec.substring(lastSemi + 1))) {
            throw new MediaInputException(field + " is a data URL without ';base64': only base64 data URLs are"
                    + " supported");
        }
        String head = spec.substring(0, lastSemi);
        int firstSemi = head.indexOf(';');
        String mediaType = (firstSemi < 0 ? head : head.substring(0, firstSemi)).trim();
        return new Loaded(base64(url, comma + 1, url.length(), field), mediaType.isEmpty() ? null : mediaType);
    }

    /**
     * The bytes of the base64 text {@code data[from, to)}, whitespace skipped, in the standard or the URL-safe
     * alphabet. The region is counted first, so the size refusal fires before anything is allocated, then copied
     * once into a byte array the decoder reads; a character outside Latin-1 becomes {@code '?'}, as the string
     * decoder's ISO-8859-1 encoding makes it, so an invalid input fails with the decoder's own message.
     */
    private byte[] base64(String data, int from, int to, String field) {
        int length = 0;
        for (int i = from; i < to; i++) {
            if (!Character.isWhitespace(data.charAt(i))) {
                length++;
            }
        }
        long decodedBound = length / 4L * 3L;
        if (decodedBound > options.maxBytes() + 3L) {
            throw tooLarge(field, decodedBound);
        }
        byte[] text = new byte[length];
        boolean urlSafe = false;
        int at = 0;
        for (int i = from; i < to; i++) {
            char c = data.charAt(i);
            if (!Character.isWhitespace(c)) {
                urlSafe |= c == '-' || c == '_';
                text[at++] = c <= 0xFF ? (byte) c : (byte) '?';
            }
        }
        byte[] bytes;
        try {
            bytes = urlSafe ? Base64.getUrlDecoder().decode(text) : Base64.getDecoder().decode(text);
        } catch (IllegalArgumentException e) {
            throw new MediaInputException(field + " is not an http(s), data or file URL, and not valid base64: "
                    + e.getMessage());
        }
        if (bytes.length == 0) {
            throw new MediaInputException(field + " decodes to no bytes");
        }
        requireSize(field, bytes.length);
        return bytes;
    }

    // ------------------------------------------------------------------ http

    /** The body of an http(s) URL, redirects followed manually so each hop passes the scheme and domain check. */
    private byte[] fetch(String url, String field) {
        URI uri = checkedHttp(url, field);
        for (int hop = 0; ; hop++) {
            HttpResponse<InputStream> response = send(uri, field);
            int status = response.statusCode();
            if (status >= 300 && status < 400 && response.headers().firstValue("location").isPresent()) {
                closeQuietly(response.body());
                if (!options.allowRedirects()) {
                    throw new MediaInputException(field + ": " + uri + " redirects (HTTP " + status
                            + "), and media URL redirects are disabled");
                }
                if (hop >= MAX_REDIRECTS) {
                    throw new MediaInputException(field + ": " + url + " redirects more than " + MAX_REDIRECTS
                            + " times");
                }
                String location = response.headers().firstValue("location").get();
                uri = checkedHttp(uri.resolve(location).toString(), field);
                continue;
            }
            if (status < 200 || status >= 300) {
                closeQuietly(response.body());
                throw new MediaInputException(field + ": fetching " + uri + " failed with HTTP " + status);
            }
            return readBounded(response, field, uri);
        }
    }

    /** The parsed URL when it is http(s) with a host inside the allowed domains. */
    private URI checkedHttp(String url, String field) {
        URI uri;
        try {
            uri = new URI(url);
        } catch (URISyntaxException e) {
            throw new MediaInputException(field + " is not a valid URL: " + e.getMessage());
        }
        String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
        if (!scheme.equals("http") && !scheme.equals("https")) {
            throw new MediaInputException(field + ": a media fetch only follows http and https URLs, got '"
                    + uri + "'");
        }
        String host = uri.getHost();
        if (host == null || host.isEmpty()) {
            throw new MediaInputException(field + " has no host: " + url);
        }
        if (!options.domainAllowed(host)) {
            throw new MediaInputException(field + ": the URL must be from one of the allowed domains "
                    + options.allowedDomains() + "; input URL domain: " + host);
        }
        return uri;
    }

    private HttpResponse<InputStream> send(URI uri, String field) {
        HttpRequest request = HttpRequest.newBuilder(uri).timeout(options.timeout()).GET().build();
        try {
            return http.send(request, HttpResponse.BodyHandlers.ofInputStream());
        } catch (IOException e) {
            throw new MediaInputException(field + ": fetching " + uri + " failed: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new MediaInputException(field + ": fetching " + uri + " was interrupted", e);
        }
    }

    private byte[] readBounded(HttpResponse<InputStream> response, String field, URI uri) {
        long declared = response.headers().firstValueAsLong("content-length").orElse(-1L);
        try (InputStream in = response.body()) {
            if (declared > options.maxBytes()) {
                throw tooLarge(field, declared);
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            long total = 0;
            int n;
            while ((n = in.read(buffer)) > 0) {
                total += n;
                if (total > options.maxBytes()) {
                    throw tooLarge(field, total);
                }
                out.write(buffer, 0, n);
            }
            if (total == 0) {
                throw new MediaInputException(field + ": " + uri + " answered an empty body");
            }
            return out.toByteArray();
        } catch (IOException e) {
            throw new MediaInputException(field + ": reading " + uri + " failed: " + e.getMessage(), e);
        }
    }

    private static void closeQuietly(InputStream in) {
        try {
            in.close();
        } catch (IOException ignored) {
            // the body of a refused response is discarded
        }
    }

    // ------------------------------------------------------------------ file

    /** A file URL's bytes, only from inside the allowed local directory. */
    private byte[] localFile(String url, String field) {
        Path root = options.localRoot();
        if (root == null) {
            throw new MediaInputException(field + ": file URLs are refused; start the server with"
                    + " --allowed-local-media-path to load local media");
        }
        Path file = filePath(url, field, root);
        Path real;
        try {
            real = file.toRealPath();
        } catch (IOException e) {
            throw new MediaInputException(field + ": the file " + file + " does not exist or cannot be opened");
        }
        if (!real.startsWith(root) || real.equals(root)) {
            throw new MediaInputException(field + ": the file path " + file + " must be inside the allowed local"
                    + " media path " + root);
        }
        if (!Files.isRegularFile(real, LinkOption.NOFOLLOW_LINKS)) {
            throw new MediaInputException(field + ": " + file + " is not a regular file");
        }
        try {
            long size = Files.size(real);
            requireSize(field, size);
            byte[] bytes = Files.readAllBytes(real);
            if (bytes.length == 0) {
                throw new MediaInputException(field + ": the file " + file + " is empty");
            }
            return bytes;
        } catch (IOException e) {
            throw new MediaInputException(field + ": reading " + file + " failed: " + e.getMessage(), e);
        }
    }

    /**
     * The path a file URL names: {@code file:///abs/path} (and {@code file://localhost/abs/path}) is absolute, as
     * vLLM reads it; {@code file://name/in/root} is relative to the allowed directory, as llama.cpp's
     * {@code --media-path} reads it. Percent escapes are decoded.
     */
    private static Path filePath(String url, String field, Path root) {
        String rest = url.substring(url.indexOf(':') + 1);
        if (rest.startsWith("//localhost/")) {
            rest = rest.substring("//localhost".length());
        } else if (rest.startsWith("///")) {
            rest = rest.substring(2);
        } else if (rest.startsWith("//")) {
            rest = rest.substring(2);
        }
        String decoded;
        try {
            // '+' is a literal in a path, not the form encoding of a space
            decoded = URLDecoder.decode(rest.replace("+", "%2B"), StandardCharsets.UTF_8);
        } catch (IllegalArgumentException e) {
            throw new MediaInputException(field + " is not a valid file URL: " + e.getMessage());
        }
        if (decoded.isEmpty()) {
            throw new MediaInputException(field + " names no file path: " + url);
        }
        try {
            return decoded.startsWith("/") ? Paths.get(decoded) : root.resolve(decoded);
        } catch (InvalidPathException e) {
            throw new MediaInputException(field + " is not a valid file URL: " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------ limits

    private void requireSize(String field, long size) {
        if (size > options.maxBytes()) {
            throw tooLarge(field, size);
        }
    }

    private MediaInputException tooLarge(String field, long size) {
        return new MediaInputException(field + ": the media payload of " + size + " bytes exceeds the "
                + options.maxBytes() + "-byte limit");
    }

    /**
     * Where and how media may be loaded from. Immutable; built with {@link #builder()}.
     */
    public static final class Options {

        /** The default fetch timeout, llama.cpp's and vLLM's audio fetch timeout. */
        public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(10);
        /** The default payload limit, vLLM's {@code VLLM_MAX_AUDIO_CLIP_FILESIZE_MB} of 25 MB. */
        public static final long DEFAULT_MAX_BYTES = 25L * 1024 * 1024;

        private final List<String> allowedDomains;
        private final Path localRoot;
        private final Duration timeout;
        private final long maxBytes;
        private final boolean allowRedirects;

        private Options(Builder b) {
            this.allowedDomains = Collections.unmodifiableList(new ArrayList<>(b.allowedDomains));
            this.localRoot = b.localRoot;
            this.timeout = b.timeout;
            this.maxBytes = b.maxBytes;
            this.allowRedirects = b.allowRedirects;
        }

        /** Any domain, no local files, the default timeout and size limit, redirects followed. */
        public static Options defaults() {
            return builder().build();
        }

        /** A builder at the defaults. */
        public static Builder builder() {
            return new Builder();
        }

        /** The hosts http(s) URLs may name, lower case; empty allows any host. */
        public List<String> allowedDomains() {
            return allowedDomains;
        }

        /** The real path of the directory file URLs may read from, or null when file URLs are refused. */
        public Path localRoot() {
            return localRoot;
        }

        /** The connect and request timeout of one fetch. */
        public Duration timeout() {
            return timeout;
        }

        /** The largest payload one reference may load. */
        public long maxBytes() {
            return maxBytes;
        }

        /** Whether http(s) redirects are followed (each target re-checked). */
        public boolean allowRedirects() {
            return allowRedirects;
        }

        /** Whether {@code host} passes the domain list (vLLM matches the URL hostname exactly). */
        public boolean domainAllowed(String host) {
            return allowedDomains.isEmpty() || allowedDomains.contains(host.toLowerCase(Locale.ROOT));
        }

        /** Builder of {@link Options}. */
        public static final class Builder {
            private final List<String> allowedDomains = new ArrayList<>();
            private Path localRoot;
            private Duration timeout = DEFAULT_TIMEOUT;
            private long maxBytes = DEFAULT_MAX_BYTES;
            private boolean allowRedirects = true;

            private Builder() {
            }

            /**
             * The hosts http(s) URLs may name; empty allows any.
             *
             * @throws IllegalArgumentException for a blank entry
             */
            public Builder allowedDomains(List<String> domains) {
                allowedDomains.clear();
                for (String d : Objects.requireNonNull(domains, "domains")) {
                    if (d == null || d.trim().isEmpty()) {
                        throw new IllegalArgumentException("an allowed media domain is blank");
                    }
                    allowedDomains.add(d.trim().toLowerCase(Locale.ROOT));
                }
                return this;
            }

            /**
             * The directory file URLs may read from, or null to refuse file URLs.
             *
             * @throws IllegalArgumentException when the path is not an existing directory
             */
            public Builder localRoot(Path root) {
                if (root == null) {
                    this.localRoot = null;
                    return this;
                }
                if (!Files.isDirectory(root)) {
                    throw new IllegalArgumentException("the allowed local media path " + root
                            + " must be an existing directory");
                }
                try {
                    this.localRoot = root.toRealPath();
                } catch (IOException e) {
                    throw new IllegalArgumentException("the allowed local media path " + root
                            + " cannot be resolved: " + e.getMessage(), e);
                }
                return this;
            }

            /**
             * The fetch timeout.
             *
             * @throws IllegalArgumentException when it is not positive
             */
            public Builder timeout(Duration timeout) {
                Objects.requireNonNull(timeout, "timeout");
                if (timeout.isZero() || timeout.isNegative()) {
                    throw new IllegalArgumentException("the media fetch timeout must be positive, got " + timeout);
                }
                this.timeout = timeout;
                return this;
            }

            /**
             * The payload limit in bytes.
             *
             * @throws IllegalArgumentException when it is not positive or exceeds the largest array
             */
            public Builder maxBytes(long maxBytes) {
                if (maxBytes <= 0 || maxBytes > Integer.MAX_VALUE - 8) {
                    throw new IllegalArgumentException("the media size limit must be in [1, " + (Integer.MAX_VALUE - 8)
                            + "] bytes, got " + maxBytes);
                }
                this.maxBytes = maxBytes;
                return this;
            }

            /** Whether http(s) redirects are followed. */
            public Builder allowRedirects(boolean allow) {
                this.allowRedirects = allow;
                return this;
            }

            /** The options. */
            public Options build() {
                return new Options(this);
            }
        }
    }
}
