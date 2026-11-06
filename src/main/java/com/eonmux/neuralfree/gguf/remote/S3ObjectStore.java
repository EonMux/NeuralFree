package com.eonmux.neuralfree.gguf.remote;

import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * S3 and S3-compatible object storage over the REST API: ListObjectsV2 for the listing and GetObject for the bytes,
 * path-style ({@code <endpoint>/<bucket>/<key>}), signed with AWS Signature Version 4 when access keys are present
 * and anonymous otherwise. The endpoint is {@code https://s3.<region>.amazonaws.com} unless configured, so MinIO,
 * Ceph and other S3-compatible stores work through the same client. Immutable and thread-safe.
 */
public final class S3ObjectStore implements ObjectStore {

    private static final String SERVICE = "s3";
    private static final String ALGORITHM = "AWS4-HMAC-SHA256";
    private static final DateTimeFormatter AMZ_DATE =
            DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'").withZone(ZoneOffset.UTC);
    private static final DateTimeFormatter DATE =
            DateTimeFormatter.ofPattern("yyyyMMdd").withZone(ZoneOffset.UTC);

    private final HttpClient http;
    private final String endpoint;
    private final String bucket;
    private final String region;
    private final String accessKey;
    private final String secretKey;
    private final String sessionToken;
    private final Clock clock;

    /**
     * A client of one bucket. Access keys may be null (anonymous); providing one without the other is refused.
     *
     * @param endpoint     the S3 endpoint, scheme included, without its trailing slash
     * @param region       the signing region
     * @param accessKey    the access key id, or null
     * @param secretKey    the secret access key, or null
     * @param sessionToken the session token of temporary credentials, or null
     */
    public S3ObjectStore(String endpoint, String bucket, String region, String accessKey, String secretKey,
                         String sessionToken, Clock clock) {
        this.endpoint = validEndpoint(endpoint);
        this.bucket = Objects.requireNonNull(bucket, "bucket");
        if (bucket.isEmpty()) {
            throw new IllegalArgumentException("the S3 bucket is empty");
        }
        this.region = region == null || region.trim().isEmpty() ? "us-east-1" : region.trim();
        this.accessKey = blankToNull(accessKey);
        this.secretKey = blankToNull(secretKey);
        this.sessionToken = blankToNull(sessionToken);
        if (this.accessKey == null != (this.secretKey == null)) {
            throw new IllegalArgumentException("S3 access keys need both the access key id and the secret access key");
        }
        this.clock = clock == null ? Clock.systemUTC() : clock;
        this.http = RemoteHttp.client();
    }

    /** {@link #S3ObjectStore(String, String, String, String, String, String, Clock)} on the system UTC clock. */
    public S3ObjectStore(String endpoint, String bucket, String region, String accessKey, String secretKey,
                         String sessionToken) {
        this(endpoint, bucket, region, accessKey, secretKey, sessionToken, Clock.systemUTC());
    }

    private static String validEndpoint(String endpoint) {
        Objects.requireNonNull(endpoint, "endpoint");
        String trimmed = endpoint.trim();
        while (trimmed.endsWith("/")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        URI uri;
        try {
            uri = URI.create(trimmed);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("the S3 endpoint '" + endpoint + "' is not a URL", e);
        }
        String scheme = uri.getScheme();
        if (scheme == null || !(scheme.equalsIgnoreCase("http") || scheme.equalsIgnoreCase("https"))
                || uri.getHost() == null) {
            throw new IllegalArgumentException("the S3 endpoint '" + endpoint + "' is not an http or https URL");
        }
        return trimmed;
    }

    @Override
    public String description() {
        return "S3 bucket '" + bucket + "' at " + endpoint;
    }

    @Override
    public List<RemoteObject> list(String prefix) throws IOException {
        String token = null;
        List<RemoteObject> out = new ArrayList<>();
        int pages = 0;
        do {
            if (++pages > RemoteHttp.MAX_LISTING_PAGES) {
                throw new IOException("the listing of " + description() + " has more than "
                        + RemoteHttp.MAX_LISTING_PAGES + " pages");
            }
            Map<String, String> query = new LinkedHashMap<>();
            query.put("list-type", "2");
            if (prefix != null && !prefix.isEmpty()) {
                query.put("prefix", prefix);
            }
            if (token != null) {
                query.put("continuation-token", token);
            }
            String queryText = RemoteHttp.canonicalQuery(query);
            HttpRequest request = signed("GET", bucketPath(null), queryText, RemoteHttp.emptyPayloadHash());
            String what = "listing of " + description() + " under '" + prefix + "'";
            byte[] body = RemoteHttp.getBytes(http, request, RemoteHttp.MAX_LISTING_BYTES, what);
            Paged page = parse(body, what);
            out.addAll(page.objects);
            if (out.size() > RemoteHttp.MAX_LISTING_OBJECTS) {
                throw new IOException("the listing of " + description() + " holds more than "
                        + RemoteHttp.MAX_LISTING_OBJECTS + " objects");
            }
            token = page.nextToken;
        } while (token != null && !token.isEmpty());
        out.sort((a, b) -> a.key().compareTo(b.key()));
        return out;
    }

    @Override
    public void download(String key, Path destination) throws IOException {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(destination, "destination");
        String path = bucketPath(key);
        String what = "object '" + key + "' of " + description();
        RemoteHttp.download(http, () -> signed("GET", path, "", RemoteHttp.emptyPayloadHash()),
                destination, what);
    }

    // ------------------------------------------------------------------ requests

    /** The path-style URI path of the bucket or one of its objects. */
    private String bucketPath(String key) {
        URI base = URI.create(endpoint);
        String basePath = base.getRawPath() == null ? "" : base.getRawPath();
        while (basePath.endsWith("/")) {
            basePath = basePath.substring(0, basePath.length() - 1);
        }
        String path = basePath + "/" + RemoteHttp.encodeSegment(bucket);
        return key == null ? path : path + "/" + RemoteHttp.encodeKey(key);
    }

    private HttpRequest signed(String method, String path, String queryText, String payloadHash)
            throws IOException {
        Instant now = clock.instant();
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("host", hostHeader());
        headers.put("x-amz-content-sha256", payloadHash);
        headers.put("x-amz-date", AMZ_DATE.format(now));
        if (sessionToken != null) {
            headers.put("x-amz-security-token", sessionToken);
        }
        HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(endpoint + path
                        + (queryText.isEmpty() ? "" : "?" + queryText)))
                .timeout(RemoteHttp.REQUEST_TIMEOUT)
                .method(method, HttpRequest.BodyPublishers.noBody());
        b.header("x-amz-content-sha256", payloadHash);
        b.header("x-amz-date", headers.get("x-amz-date"));
        if (sessionToken != null) {
            b.header("x-amz-security-token", sessionToken);
        }
        if (accessKey != null) {
            b.header("Authorization", authorization(method, path, queryText, headers, payloadHash,
                    accessKey, secretKey, region, SERVICE, now));
        }
        return b.build();
    }

    /** The authority Java sends as the Host header. */
    private String hostHeader() {
        URI uri = URI.create(endpoint);
        String authority = uri.getRawAuthority();
        if (authority == null || authority.isEmpty()) {
            throw new IllegalArgumentException("the S3 endpoint '" + endpoint + "' has no authority");
        }
        return authority;
    }

    /**
     * The SigV4 Authorization header (AWS's "Signature Calculations for the Authorization Header"): the canonical
     * request over the sorted signed headers, the credential scope, and the signature of the request.
     */
    static String authorization(String method, String canonicalUri, String canonicalQuery,
                                Map<String, String> headers, String payloadHash,
                                String accessKey, String secretKey, String region, String service, Instant when) {
        List<String> names = new ArrayList<>(headers.keySet());
        for (int i = 0; i < names.size(); i++) {
            names.set(i, RemoteHttp.lower(names.get(i)));
        }
        names.sort(String::compareTo);
        StringBuilder canonicalHeaders = new StringBuilder();
        StringBuilder signedHeaders = new StringBuilder();
        for (String name : names) {
            String value = header(headers, name);
            canonicalHeaders.append(name).append(':').append(value.trim()).append('\n');
            if (signedHeaders.length() > 0) {
                signedHeaders.append(';');
            }
            signedHeaders.append(name);
        }
        String canonicalRequest = method + "\n" + canonicalUri + "\n" + canonicalQuery + "\n"
                + canonicalHeaders + "\n" + signedHeaders + "\n" + payloadHash;
        String date = DATE.format(when);
        String amzDate = AMZ_DATE.format(when);
        String scope = date + "/" + region + "/" + service + "/aws4_request";
        String stringToSign = ALGORITHM + "\n" + amzDate + "\n" + scope + "\n"
                + RemoteHttp.sha256Hex(canonicalRequest);
        byte[] key = signingKey(secretKey, date, region, service);
        String signature = RemoteHttp.hex(RemoteHttp.hmacSha256(key, stringToSign));
        return ALGORITHM + " Credential=" + accessKey + "/" + scope + ",SignedHeaders=" + signedHeaders
                + ",Signature=" + signature;
    }

    private static String header(Map<String, String> headers, String lowerName) {
        for (Map.Entry<String, String> e : headers.entrySet()) {
            if (e.getKey().equalsIgnoreCase(lowerName)) {
                return e.getValue();
            }
        }
        return "";
    }

    private static byte[] signingKey(String secretKey, String date, String region, String service) {
        byte[] kDate = RemoteHttp.hmacSha256(("AWS4" + secretKey).getBytes(StandardCharsets.UTF_8), date);
        byte[] kRegion = RemoteHttp.hmacSha256(kDate, region);
        byte[] kService = RemoteHttp.hmacSha256(kRegion, service);
        return RemoteHttp.hmacSha256(kService, "aws4_request");
    }

    // ------------------------------------------------------------------ listing

    /** One page of ListObjectsV2. */
    private static final class Paged {
        final List<RemoteObject> objects = new ArrayList<>();
        String nextToken;
    }

    private static Paged parse(byte[] body, String what) throws IOException {
        Paged page = new Paged();
        XMLStreamReader reader = RemoteHttp.xml(body, what);
        try {
            String key = null;
            long size = -1;
            String etag = null;
            boolean inContents = false;
            while (reader.hasNext()) {
                int event = reader.next();
                if (event == XMLStreamReader.START_ELEMENT) {
                    String name = reader.getLocalName();
                    if (name.equals("Contents")) {
                        inContents = true;
                        key = null;
                        size = -1;
                        etag = null;
                    } else if (inContents && name.equals("Key")) {
                        key = RemoteHttp.text(reader);
                    } else if (inContents && name.equals("Size")) {
                        size = parseSize(RemoteHttp.text(reader));
                    } else if (inContents && name.equals("ETag")) {
                        etag = stripQuotes(RemoteHttp.text(reader));
                    } else if (name.equals("NextContinuationToken")) {
                        page.nextToken = RemoteHttp.text(reader);
                    }
                } else if (event == XMLStreamReader.END_ELEMENT && reader.getLocalName().equals("Contents")) {
                    if (key != null) {
                        page.objects.add(new RemoteObject(key, size, etag));
                    }
                    inContents = false;
                }
            }
        } catch (XMLStreamException e) {
            throw new IOException("the listing of " + what + " is not well-formed XML: " + e.getMessage(), e);
        } finally {
            try {
                reader.close();
            } catch (XMLStreamException ignored) {
                // closing a reader over a byte array cannot fail meaningfully
            }
        }
        return page;
    }

    private static long parseSize(String text) throws IOException {
        if (text == null || text.isEmpty()) {
            return -1L;
        }
        try {
            long value = Long.parseLong(text);
            return value < 0 ? -1L : value;
        } catch (NumberFormatException e) {
            throw new IOException("the listing carries the invalid object size '" + text + "'", e);
        }
    }

    private static String stripQuotes(String value) {
        if (value.length() >= 2 && value.startsWith("\"") && value.endsWith("\"")) {
            return value.substring(1, value.length() - 1);
        }
        return value;
    }

    private static String blankToNull(String value) {
        return value == null || value.trim().isEmpty() ? null : value.trim();
    }

    @Override
    public String toString() {
        return "S3ObjectStore{" + bucket + " at " + endpoint + "}";
    }
}
