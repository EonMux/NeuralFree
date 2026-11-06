package com.eonmux.neuralfree.gguf.remote;

import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.nio.file.Path;
import java.time.Clock;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Azure Blob Storage over its REST API: {@code ?restype=container&comp=list} for the listing and the blob URL for
 * the bytes. A shared access signature is appended when configured, an account key signs with SharedKey when that is
 * configured instead, and without either the request is anonymous (a public container). Immutable and thread-safe.
 */
public final class AzureBlobObjectStore implements ObjectStore {

    /** The REST API version every request announces. */
    static final String API_VERSION = "2021-08-06";
    private static final DateTimeFormatter RFC1123 =
            DateTimeFormatter.RFC_1123_DATE_TIME.withZone(ZoneOffset.UTC);

    private final HttpClient http;
    private final String endpoint;
    private final String container;
    private final String account;
    private final String accountKey;
    private final String sasToken;
    private final Clock clock;

    /**
     * A client of one container.
     *
     * @param endpoint   the blob service endpoint, without its trailing slash
     * @param container  the container name
     * @param account    the storage account name (needed for SharedKey signing), or null
     * @param accountKey the base64 account key for SharedKey signing, or null
     * @param sasToken   a shared access signature, with or without its leading {@code ?}, or null
     */
    public AzureBlobObjectStore(String endpoint, String container, String account, String accountKey,
                                String sasToken, Clock clock) {
        this.endpoint = validEndpoint(endpoint);
        this.container = Objects.requireNonNull(container, "container");
        if (container.isEmpty()) {
            throw new IllegalArgumentException("the Azure container is empty");
        }
        this.account = blankToNull(account);
        this.accountKey = blankToNull(accountKey);
        this.sasToken = normalizeSas(sasToken);
        if (this.accountKey != null && this.account == null) {
            throw new IllegalArgumentException("an Azure account key needs the storage account name");
        }
        this.clock = clock == null ? Clock.systemUTC() : clock;
        this.http = RemoteHttp.client();
    }

    /** {@link #AzureBlobObjectStore(String, String, String, String, String, Clock)} on the system UTC clock. */
    public AzureBlobObjectStore(String endpoint, String container, String account, String accountKey,
                                String sasToken) {
        this(endpoint, container, account, accountKey, sasToken, Clock.systemUTC());
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
            throw new IllegalArgumentException("the Azure endpoint '" + endpoint + "' is not a URL", e);
        }
        String scheme = uri.getScheme();
        if (scheme == null || !(scheme.equalsIgnoreCase("http") || scheme.equalsIgnoreCase("https"))
                || uri.getHost() == null) {
            throw new IllegalArgumentException("the Azure endpoint '" + endpoint + "' is not an http or https URL");
        }
        return trimmed;
    }

    private static String normalizeSas(String sas) {
        String value = blankToNull(sas);
        if (value == null) {
            return null;
        }
        int start = value.startsWith("?") ? 1 : 0;
        return value.substring(start);
    }

    @Override
    public String description() {
        return "Azure container '" + container + "' at " + endpoint;
    }

    @Override
    public List<RemoteObject> list(String prefix) throws IOException {
        List<RemoteObject> out = new ArrayList<>();
        String marker = null;
        int pages = 0;
        do {
            if (++pages > RemoteHttp.MAX_LISTING_PAGES) {
                throw new IOException("the listing of " + description() + " has more than "
                        + RemoteHttp.MAX_LISTING_PAGES + " pages");
            }
            StringBuilder query = new StringBuilder("restype=container&comp=list");
            if (prefix != null && !prefix.isEmpty()) {
                query.append("&prefix=").append(RemoteHttp.encodeQuery(prefix));
            }
            if (marker != null && !marker.isEmpty()) {
                query.append("&marker=").append(RemoteHttp.encodeQuery(marker));
            }
            String url = containerUrl(query.toString());
            String what = "listing of " + description() + " under '" + prefix + "'";
            byte[] body = RemoteHttp.getBytes(http, request("GET", url), RemoteHttp.MAX_LISTING_BYTES, what);
            Page page = parse(body, what);
            out.addAll(page.objects);
            if (out.size() > RemoteHttp.MAX_LISTING_OBJECTS) {
                throw new IOException("the listing of " + description() + " holds more than "
                        + RemoteHttp.MAX_LISTING_OBJECTS + " objects");
            }
            marker = page.nextMarker;
        } while (marker != null && !marker.isEmpty());
        out.sort((a, b) -> a.key().compareTo(b.key()));
        return out;
    }

    @Override
    public void download(String key, Path destination) throws IOException {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(destination, "destination");
        String url = appendSas(endpoint + "/" + RemoteHttp.encodeSegment(container) + "/"
                + RemoteHttp.encodeKey(key));
        String what = "object '" + key + "' of " + description();
        RemoteHttp.download(http, () -> request("GET", url), destination, what);
    }

    /** The container URL with the SAS token and, for a listing, its query. */
    private String containerUrl(String query) {
        StringBuilder url = new StringBuilder(endpoint).append('/').append(RemoteHttp.encodeSegment(container));
        StringBuilder params = new StringBuilder();
        if (query != null && !query.isEmpty()) {
            params.append(query);
        }
        if (sasToken != null) {
            if (params.length() > 0) {
                params.append('&');
            }
            params.append(sasToken);
        }
        if (params.length() > 0) {
            url.append('?').append(params);
        }
        return url.toString();
    }

    /** {@code url} with the SAS token appended as its query. */
    private String appendSas(String url) {
        return sasToken == null ? url : url + "?" + sasToken;
    }

    private HttpRequest request(String method, String url) {
        HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(url))
                .timeout(RemoteHttp.REQUEST_TIMEOUT)
                .method(method, HttpRequest.BodyPublishers.noBody());
        String date = RFC1123.format(clock.instant());
        b.header("x-ms-date", date);
        b.header("x-ms-version", API_VERSION);
        if (sasToken == null && accountKey != null) {
            URI uri = URI.create(url);
            String resource = canonicalResource(uri);
            b.header("Authorization", "SharedKey " + account + ":"
                    + RemoteHttp.base64(RemoteHttp.hmacSha256Base64(accountKey,
                    stringToSign(method, date, resource))));
        }
        return b.build();
    }

    /** The canonicalized resource, {@code /<account>/<container>[/<blob>]} plus its sorted query parameters. */
    private String canonicalResource(URI uri) {
        StringBuilder out = new StringBuilder("/").append(account).append('/').append(container);
        String path = uri.getRawPath();
        int at = path.indexOf('/', 1);
        if (at >= 0 && at + 1 < path.length()) {
            out.append('/').append(decode(path.substring(at + 1)));
        }
        String rawQuery = uri.getRawQuery();
        if (rawQuery != null && !rawQuery.isEmpty()) {
            Map<String, String> params = new java.util.TreeMap<>();
            for (String pair : rawQuery.split("&")) {
                int eq = pair.indexOf('=');
                String name = eq < 0 ? pair : pair.substring(0, eq);
                String value = eq < 0 ? "" : pair.substring(eq + 1);
                params.put(decode(name).toLowerCase(Locale.ROOT), decode(value));
            }
            for (Map.Entry<String, String> e : params.entrySet()) {
                out.append('\n').append(e.getKey()).append(':').append(e.getValue());
            }
        }
        return out.toString();
    }

    /**
     * The SharedKey string to sign of a request with no entity headers: the verb, the empty date and range lines,
     * then the {@code x-ms-*} headers and the canonicalized resource.
     */
    static String stringToSign(String method, String xMsDate, String canonicalResource) {
        return method + "\n"
                + "\n" // Content-Encoding
                + "\n" // Content-Language
                + "\n" // Content-Length
                + "\n" // Content-MD5
                + "\n" // Content-Type
                + "\n" // Date
                + "\n" // If-Modified-Since
                + "\n" // If-Match
                + "\n" // If-None-Match
                + "\n" // If-Unmodified-Since
                + "\n" // Range
                + "x-ms-date:" + xMsDate + "\n"
                + "x-ms-version:" + API_VERSION + "\n"
                + canonicalResource;
    }

    private static String decode(String value) {
        try {
            return java.net.URLDecoder.decode(value, java.nio.charset.StandardCharsets.UTF_8.name());
        } catch (java.io.UnsupportedEncodingException e) {
            throw new IllegalStateException("UTF-8 is not available on this JVM", e);
        }
    }

    // ------------------------------------------------------------------ listing

    private static final class Page {
        final List<RemoteObject> objects = new ArrayList<>();
        String nextMarker;
    }

    private static Page parse(byte[] body, String what) throws IOException {
        Page page = new Page();
        XMLStreamReader reader = RemoteHttp.xml(body, what);
        try {
            String name = null;
            long size = -1;
            String etag = null;
            boolean inBlob = false;
            while (reader.hasNext()) {
                int event = reader.next();
                if (event == XMLStreamReader.START_ELEMENT) {
                    String element = reader.getLocalName();
                    if (element.equals("Blob")) {
                        inBlob = true;
                        name = null;
                        size = -1;
                        etag = null;
                    } else if (inBlob && element.equals("Name")) {
                        name = RemoteHttp.text(reader);
                    } else if (inBlob && element.equals("Content-Length")) {
                        size = parseSize(RemoteHttp.text(reader));
                    } else if (inBlob && element.equals("Etag")) {
                        etag = RemoteHttp.text(reader);
                    } else if (element.equals("NextMarker")) {
                        page.nextMarker = RemoteHttp.text(reader);
                    }
                } else if (event == XMLStreamReader.END_ELEMENT && reader.getLocalName().equals("Blob")) {
                    if (name != null) {
                        page.objects.add(new RemoteObject(name, size, etag));
                    }
                    inBlob = false;
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
            throw new IOException("the listing carries the invalid blob size '" + text + "'", e);
        }
    }

    private static String blankToNull(String value) {
        return value == null || value.trim().isEmpty() ? null : value.trim();
    }

    @Override
    public String toString() {
        return "AzureBlobObjectStore{" + container + " at " + endpoint + "}";
    }
}
