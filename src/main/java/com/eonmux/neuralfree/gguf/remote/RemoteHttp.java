package com.eonmux.neuralfree.gguf.remote;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.Base64;
import java.util.Locale;
import java.util.Map;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/**
 * The machinery the three object-store clients share: the HTTP client, RFC 3986 encoding, request signing helpers,
 * streaming downloads with retries, and a hardened XML reader for the S3 and Azure listings. All values that reach a
 * URL or a signature are encoded here, never by the callers.
 */
final class RemoteHttp {

    /** One connection timeout of every store request. */
    static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(15);
    /** One complete request timeout; large model files need minutes, not seconds. */
    static final Duration REQUEST_TIMEOUT = Duration.ofMinutes(30);
    /** Largest listing response accepted. */
    static final long MAX_LISTING_BYTES = 64L << 20;
    /** Most listing pages and objects accepted: a store cannot make the loader page forever. */
    static final int MAX_LISTING_PAGES = 10_000;
    static final int MAX_LISTING_OBJECTS = 10_000_000;
    /** Attempts of one object download (the count HubClient uses). */
    private static final int MAX_ATTEMPTS = 3;
    private static final char[] HEX = "0123456789abcdef".toCharArray();

    private RemoteHttp() {
    }

    /** A client with a connect timeout, normal redirects and no automatic retries. */
    static HttpClient client() {
        return HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(CONNECT_TIMEOUT)
                .build();
    }

    // ------------------------------------------------------------------ encoding

    /**
     * Percent-encodes one path segment (RFC 3986 unreserved characters stay, everything else becomes {@code %XX}).
     * A space is {@code %20}, never {@code +}.
     */
    static String encodeSegment(String value) {
        StringBuilder out = new StringBuilder(value.length() + 16);
        for (byte raw : value.getBytes(StandardCharsets.UTF_8)) {
            int b = raw & 0xFF;
            if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') || b == '-'
                    || b == '.' || b == '_' || b == '~') {
                out.append((char) b);
            } else {
                out.append('%').append(HEX[b >>> 4]).append(HEX[b & 0xF]);
            }
        }
        return out.toString();
    }

    /** Percent-encodes a key, keeping its {@code /} separators. */
    static String encodeKey(String key) {
        StringBuilder out = new StringBuilder(key.length() + 16);
        String[] parts = key.split("/", -1);
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) {
                out.append('/');
            }
            out.append(encodeSegment(parts[i]));
        }
        return out.toString();
    }

    /** Percent-encodes a whole value, its {@code /} included (a GCS object name in one path segment). */
    static String encodeWhole(String value) {
        return encodeSegment(value);
    }

    /** Percent-encodes a query value (RFC 3986; a space is {@code %20}). */
    static String encodeQuery(String value) {
        return encodeSegment(value);
    }

    // ------------------------------------------------------------------ digests and signing

    /** SHA-256 of {@code bytes}. */
    static byte[] sha256(byte[] bytes) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(bytes);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available on this JVM", e);
        }
    }

    /** SHA-256 of {@code text} (UTF-8), lowercase hex. */
    static String sha256Hex(String text) {
        return hex(sha256(text.getBytes(StandardCharsets.UTF_8)));
    }

    /** HMAC-SHA256 of {@code data} under {@code key}. */
    static byte[] hmacSha256(byte[] key, String data) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(key, "HmacSHA256"));
            return mac.doFinal(data.getBytes(StandardCharsets.UTF_8));
        } catch (NoSuchAlgorithmException | java.security.InvalidKeyException e) {
            throw new IllegalStateException("HmacSHA256 is not available on this JVM", e);
        }
    }

    /** HMAC-SHA256 of {@code data} under the raw bytes of a base64 account key. */
    static byte[] hmacSha256Base64(String base64Key, String data) {
        return hmacSha256(Base64.getDecoder().decode(base64Key), data);
    }

    /** Lowercase hex of {@code bytes}. */
    static String hex(byte[] bytes) {
        char[] out = new char[bytes.length * 2];
        for (int i = 0; i < bytes.length; i++) {
            out[2 * i] = HEX[(bytes[i] >>> 4) & 0xF];
            out[2 * i + 1] = HEX[bytes[i] & 0xF];
        }
        return new String(out);
    }

    /** Standard Base64 of {@code bytes}. */
    static String base64(byte[] bytes) {
        return Base64.getEncoder().encodeToString(bytes);
    }

    /** The SHA-256 of the empty body, SigV4's payload hash of every GET. */
    static String emptyPayloadHash() {
        return sha256Hex("");
    }

    static String lower(String value) {
        return value.toLowerCase(Locale.ROOT);
    }

    // ------------------------------------------------------------------ HTTP

    /** Builds a fresh request; downloads re-sign on every attempt. */
    interface RequestSupplier {
        HttpRequest get() throws IOException;
    }

    /**
     * Runs one GET and writes the body to {@code destination}. A 404 is a {@link java.io.FileNotFoundException}; a
     * 4xx other than 429 is not retried, a 5xx or a transport failure is tried again after a growing pause.
     */
    static void download(HttpClient http, RequestSupplier supplier, Path destination, String what) throws IOException {
        IOException failure = null;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            if (attempt > 0) {
                pause(attempt);
            }
            HttpRequest request = supplier.get();
            HttpResponse<InputStream> response;
            try {
                response = http.send(request, HttpResponse.BodyHandlers.ofInputStream());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("the download of " + what + " was interrupted", e);
            } catch (IOException e) {
                failure = new IOException("the download of " + what + " failed: " + e.getMessage(), e);
                continue;
            }
            int status = response.statusCode();
            if (status >= 200 && status < 300) {
                try (InputStream in = response.body()) {
                    write(in, destination);
                }
                return;
            }
            response.body().close();
            if (status == 404) {
                throw new java.io.FileNotFoundException("the object store has no " + what);
            }
            IOException e = new IOException("the download of " + what + " answered HTTP " + status);
            if (status == 429 || status >= 500) {
                failure = e;
                continue;
            }
            throw e;
        }
        throw failure != null ? failure : new IOException("the download of " + what + " failed");
    }

    /** A GET whose body is read fully into memory, bounded by {@code maxBytes}. */
    static byte[] getBytes(HttpClient http, HttpRequest request, long maxBytes, String what) throws IOException {
        HttpResponse<InputStream> response;
        try {
            response = http.send(request, HttpResponse.BodyHandlers.ofInputStream());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("the request for " + what + " was interrupted", e);
        }
        int status = response.statusCode();
        try (InputStream in = response.body()) {
            if (status < 200 || status >= 300) {
                throw new IOException("the request for " + what + " answered HTTP " + status);
            }
            return readBounded(in, maxBytes, what);
        }
    }

    /** Reads at most {@code maxBytes} from {@code in}, refusing a longer body. */
    static byte[] readBounded(InputStream in, long maxBytes, String what) throws IOException {
        byte[] buffer = new byte[64 * 1024];
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        long total = 0;
        int n;
        while ((n = in.read(buffer)) >= 0) {
            total += n;
            if (total > maxBytes) {
                throw new IOException("the response for " + what + " is over the limit of " + maxBytes + " bytes");
            }
            out.write(buffer, 0, n);
        }
        return out.toByteArray();
    }

    /** Streams {@code in} into {@code destination}, creating it. */
    static void write(InputStream in, Path destination) throws IOException {
        Files.createDirectories(destination.toAbsolutePath().getParent());
        try (OutputStream out = Files.newOutputStream(destination, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            byte[] buffer = new byte[128 * 1024];
            int n;
            while ((n = in.read(buffer)) >= 0) {
                out.write(buffer, 0, n);
            }
        }
    }

    private static void pause(int attempt) throws IOException {
        try {
            Thread.sleep(500L << attempt);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("the download was interrupted", e);
        }
    }

    // ------------------------------------------------------------------ XML

    /**
     * A reader of one listing document, with DTDs and external entities off: a listing is untrusted input.
     *
     * @throws IOException when the document is not well-formed XML
     */
    static XMLStreamReader xml(byte[] document, String what) throws IOException {
        XMLInputFactory factory = XMLInputFactory.newFactory();
        setIfSupported(factory, XMLInputFactory.SUPPORT_DTD, Boolean.FALSE);
        setIfSupported(factory, XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, Boolean.FALSE);
        setIfSupported(factory, XMLInputFactory.IS_COALESCING, Boolean.TRUE);
        try {
            return factory.createXMLStreamReader(new ByteArrayInputStream(document), StandardCharsets.UTF_8.name());
        } catch (XMLStreamException e) {
            throw new IOException("the listing of " + what + " is not well-formed XML: " + e.getMessage(), e);
        }
    }

    private static void setIfSupported(XMLInputFactory factory, String property, Object value) {
        try {
            factory.setProperty(property, value);
        } catch (IllegalArgumentException ignored) {
            // a provider without the property keeps its default, which is safe here
        }
    }

    /** A listing's text content with the surrounding whitespace removed, or the empty string. */
    static String text(XMLStreamReader reader) throws XMLStreamException {
        String value = reader.getElementText();
        return value == null ? "" : value.trim();
    }

    /** The query string of {@code params}, keys sorted, both sides encoded. */
    static String canonicalQuery(Map<String, String> params) {
        java.util.List<String> keys = new java.util.ArrayList<>(params.keySet());
        keys.sort(String::compareTo);
        StringBuilder out = new StringBuilder();
        for (String key : keys) {
            if (out.length() > 0) {
                out.append('&');
            }
            out.append(encodeQuery(key)).append('=').append(encodeQuery(params.get(key)));
        }
        return out.toString();
    }
}
