package com.eonmux.neuralfree.gguf.remote;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Google Cloud Storage over the JSON API: {@code /storage/v1/b/<bucket>/o} for the listing (pages of {@code items}
 * with {@code name}, {@code size} and {@code etag}) and {@code ?alt=media} for the bytes. A bearer token authenticates
 * when configured; without one the request is anonymous, which public buckets accept. Immutable and thread-safe.
 */
public final class GcsObjectStore implements ObjectStore {

    private static final Logger LOG = LogManager.getLogger(GcsObjectStore.class);

    private final HttpClient http;
    private final String endpoint;
    private final String bucket;
    private final String accessToken;

    /**
     * A client of one bucket.
     *
     * @param endpoint    the JSON API root, without its trailing slash ({@code https://storage.googleapis.com})
     * @param bucket      the bucket name
     * @param accessToken an OAuth bearer token, or null for anonymous access
     */
    public GcsObjectStore(String endpoint, String bucket, String accessToken) {
        this.endpoint = validEndpoint(endpoint);
        this.bucket = Objects.requireNonNull(bucket, "bucket");
        if (bucket.isEmpty()) {
            throw new IllegalArgumentException("the GCS bucket is empty");
        }
        this.accessToken = accessToken == null || accessToken.trim().isEmpty() ? null : accessToken.trim();
        this.http = RemoteHttp.client();
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
            throw new IllegalArgumentException("the GCS endpoint '" + endpoint + "' is not a URL", e);
        }
        String scheme = uri.getScheme();
        if (scheme == null || !(scheme.equalsIgnoreCase("http") || scheme.equalsIgnoreCase("https"))
                || uri.getHost() == null) {
            throw new IllegalArgumentException("the GCS endpoint '" + endpoint + "' is not an http or https URL");
        }
        return trimmed;
    }

    @Override
    public String description() {
        return "GCS bucket '" + bucket + "' at " + endpoint;
    }

    @Override
    public List<RemoteObject> list(String prefix) throws IOException {
        List<RemoteObject> out = new ArrayList<>();
        String pageToken = null;
        int pages = 0;
        do {
            if (++pages > RemoteHttp.MAX_LISTING_PAGES) {
                throw new IOException("the listing of " + description() + " has more than "
                        + RemoteHttp.MAX_LISTING_PAGES + " pages");
            }
            StringBuilder url = new StringBuilder(endpoint).append("/storage/v1/b/")
                    .append(RemoteHttp.encodeSegment(bucket)).append("/o?");
            if (prefix != null && !prefix.isEmpty()) {
                url.append("prefix=").append(RemoteHttp.encodeQuery(prefix)).append('&');
            }
            if (pageToken != null) {
                url.append("pageToken=").append(RemoteHttp.encodeQuery(pageToken)).append('&');
            }
            url.append("maxResults=1000");
            String what = "listing of " + description() + " under '" + prefix + "'";
            HttpRequest request = request(url.toString());
            byte[] body = RemoteHttp.getBytes(http, request, RemoteHttp.MAX_LISTING_BYTES, what);
            Object parsed;
            try {
                parsed = Json.parse(new String(body, java.nio.charset.StandardCharsets.UTF_8),
                        Json.Limits.UNTRUSTED);
            } catch (JsonException e) {
                throw new IOException("the listing of " + what + " is not valid JSON: " + e.getMessage(), e);
            }
            if (!(parsed instanceof Map)) {
                throw new IOException("the listing of " + what + " is not a JSON object");
            }
            Map<?, ?> root = (Map<?, ?>) parsed;
            Object items = root.get("items");
            if (items instanceof List) {
                for (Object item : (List<?>) items) {
                    if (item instanceof Map) {
                        RemoteObject object = object((Map<?, ?>) item, what);
                        if (object != null) {
                            out.add(object);
                        }
                    }
                }
            } else if (items != null) {
                throw new IOException("the listing of " + what + " carries 'items' of type "
                        + items.getClass().getSimpleName());
            }
            Object next = root.get("nextPageToken");
            pageToken = next instanceof String && !((String) next).isEmpty() ? (String) next : null;
            if (out.size() > RemoteHttp.MAX_LISTING_OBJECTS) {
                throw new IOException("the listing of " + description() + " holds more than "
                        + RemoteHttp.MAX_LISTING_OBJECTS + " objects");
            }
        } while (pageToken != null);
        out.sort((a, b) -> a.key().compareTo(b.key()));
        return out;
    }

    private static RemoteObject object(Map<?, ?> item, String what) throws IOException {
        Object name = item.get("name");
        if (!(name instanceof String) || ((String) name).isEmpty()) {
            return null;
        }
        long size = -1L;
        Object rawSize = item.get("size");
        if (rawSize instanceof String) {
            try {
                size = Long.parseLong((String) rawSize);
            } catch (NumberFormatException e) {
                throw new IOException("the listing of " + what + " carries the invalid object size '" + rawSize
                        + "'", e);
            }
        } else if (rawSize instanceof Long) {
            size = (Long) rawSize;
        }
        Object etag = item.get("etag");
        return new RemoteObject((String) name, size < 0 ? -1L : size,
                etag instanceof String ? (String) etag : null);
    }

    @Override
    public void download(String key, Path destination) throws IOException {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(destination, "destination");
        String url = endpoint + "/storage/v1/b/" + RemoteHttp.encodeSegment(bucket) + "/o/"
                + RemoteHttp.encodeWhole(key) + "?alt=media";
        String what = "object '" + key + "' of " + description();
        RemoteHttp.download(http, () -> request(url), destination, what);
    }

    private HttpRequest request(String url) {
        HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(url))
                .timeout(RemoteHttp.REQUEST_TIMEOUT)
                .GET();
        if (accessToken != null) {
            b.header("Authorization", "Bearer " + accessToken);
        }
        LOG.debug("GCS request {}", url);
        return b.build();
    }

    @Override
    public String toString() {
        return "GcsObjectStore{" + bucket + " at " + endpoint + "}";
    }
}
