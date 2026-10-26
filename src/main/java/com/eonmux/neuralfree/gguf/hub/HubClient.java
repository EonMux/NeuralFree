package com.eonmux.neuralfree.gguf.hub;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The Hugging Face Hub over {@link HttpClient} with TLS verification on (PARITY.md WP TL-7), the
 * mirror of llama.cpp's {@code common/hf-cache.cpp} and {@code common/download.cpp}: the refs and
 * tree APIs give the repository's files, and the resolve endpoint downloads them into
 * {@link HubCache} with the Hub's ETag check and {@code Range} resume, where every LFS download is
 * verified against its sha256 pointer before it enters the snapshot tree. Access is the
 * {@code HF_TOKEN} environment value in its valid form, carried in the {@code Authorization} header
 * only: never logged, never printed, never in an error message. {@code HF_HUB_OFFLINE} is the
 * offline mode whose callers read {@link HubCache} instead. The responses, the paths they name and
 * the oids they carry are untrusted input (§3.6): JSON is parsed under {@link Json.Limits#UNTRUSTED},
 * repository paths are confined by {@link HubCache#snapshotPath}, and an oid is hex before anything
 * else in the cache reads it. Immutable; safe over concurrent threads.
 */
public final class HubClient {

    /** One file of a repository listing: its path, its git or LFS oid, its resolve URL and its size. */
    public static final class RepoFile {

        private final String repoId;
        private final String path;
        private final String oid;
        private final String url;
        private final long size;

        RepoFile(String repoId, String path, String oid, String url, long size) {
            this.repoId = repoId;
            this.path = path;
            this.oid = oid;
            this.url = url;
            this.size = size;
        }

        /** The repository id ({@code user/model}). */
        public String repoId() {
            return repoId;
        }

        /** The file's path inside the repository, relative and forward-slash separated. */
        public String path() {
            return path;
        }

        /** The git blob oid (40-hex) or the LFS sha256 pointer (64-hex), or null when none is listed. */
        public String oid() {
            return oid;
        }

        /** Whether this file carries a 64-hex LFS sha256 pointer the download is verified against. */
        public boolean isLfs() {
            return HubCache.isSha256Oid(oid);
        }

        /** The {@code resolve/<commit>/<path>} URL the file is served from. */
        public String url() {
            return url;
        }

        /** The listed size in bytes; -1 when the tree entry carried none. */
        public long size() {
            return size;
        }

        /** The filename parts llama.cpp's downloader parses from {@link #path}, or null for non-GGUF files. */
        public HubCache.GgufFilename filename() {
            return HubCache.GgufFilename.parse(path);
        }

        @Override
        public String toString() {
            return repoId + " " + path + (size >= 0 ? " (" + size + " bytes)" : "");
        }
    }

    /** The suffix the in-progress download is written under until it completes and verifies (llama.cpp). */
    static final String PARTIAL_SUFFIX = ".downloadInProgress";

    /** llama.cpp's downloader retries a failed attempt this many times, back-doubling between each. */
    private static final int MAX_ATTEMPTS = 3;

    /** One connection and request timeout for every hub call: 15 seconds, no silent hang. */
    private static final java.time.Duration TIMEOUT = java.time.Duration.ofSeconds(15);

    private final HubCache cache;
    private final HttpClient http;
    private final String endpoint;
    private final String token;

    /** A client over the cache {@code cache}, the default endpoint and the {@code HF_TOKEN} token. */
    public HubClient(HubCache cache) {
        this.cache = cache;
        // TLS verification stays on: the default SSLContext verifies every certificate and host name,
        // as llama.cpp's http backend and the plan's TLS-on rule both fix
        this.http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(TIMEOUT)
                .build();
        this.endpoint = endpoint();
        this.token = token();
    }

    /** The {@code HF_TOKEN} environment value in llama.cpp's valid form, or null. */
    static String token() {
        // llama.cpp is_valid_hf_token: "hf_" and then 34..253 alphanumerics; an invalid token means
        // anonymous access and the value itself is never echoed back, logged or thrown
        String t = System.getenv("HF_TOKEN");
        if (t == null) {
            return null;
        }
        String trimmed = t.trim();
        if (trimmed.length() < 37 || trimmed.length() > 256 || !trimmed.startsWith("hf_")) {
            return null;
        }
        for (int i = 3; i < trimmed.length(); i++) {
            char c = trimmed.charAt(i);
            boolean ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
            if (!ok) {
                return null;
            }
        }
        return trimmed;
    }

    /** llama.cpp's {@code get_model_endpoint}: {@code MODEL_ENDPOINT}, then {@code HF_ENDPOINT}, then the Hub. */
    static String endpoint() {
        String v = trimmedEnv("MODEL_ENDPOINT");
        if (v.isEmpty()) {
            v = trimmedEnv("HF_ENDPOINT");
        }
        if (v.isEmpty()) {
            return "https://huggingface.co/";
        }
        return v.endsWith("/") ? v : v + "/";
    }

    /** The offline mode llama.cpp's downloader would run under: {@code HF_HUB_OFFLINE} is 1 or true. */
    public static boolean offline() {
        String v = trimmedEnv("HF_HUB_OFFLINE").toLowerCase(Locale.ROOT);
        return v.equals("1") || v.equals("true");
    }

    /** An environment value, or the empty string instead of null. */
    private static String trimmedEnv(String name) {
        String v = System.getenv(name);
        return v == null ? "" : v.trim();
    }

    /** The cache the client downloads into. */
    public HubCache cache() {
        return cache;
    }

    // ================================================================== hub API

    /**
     * The commit {@code ref} resolves to online: llama.cpp's get_repo_commit over the refs API, which
     * writes the branch's commit into the cache's refs as it resolves. Main wins; entries whose name
     * is no valid ref name or whose commit is no 40-hex hash are skipped, and a repository with no
     * valid branch is a refusal because the snapshot layout is keyed on the commit.
     */
    public String resolveCommit(HubRef ref) throws IOException, InterruptedException {
        Object parsed = apiGet(endpoint + "api/models/" + ref.repository() + "/refs");
        if (!(parsed instanceof List)) {
            throw new IOException("the refs response of '" + ref + "' on " + endpoint
                    + " is not a list");
        }
        String commit = null;
        String branch = null;
        for (Object o : (List<?>) parsed) {
            String name = stringOf(o, "name");
            String target = stringOf(o, "targetCommit");
            if (name == null || target == null || !HubCache.isValidOid(target, 40)
                    || !isValidRefName(name)) {
                continue;
            }
            if (name.equals("main")) {
                commit = target;
                branch = "main";
                break;
            }
            if (branch == null) {
                commit = target;
                branch = name;
            }
        }
        if (commit == null) {
            throw new IOException("no valid branch for '" + ref + "' on " + endpoint);
        }
        cache.writeRef(ref, branch, commit);
        return commit.toLowerCase(Locale.ROOT);
    }

    /**
     * The repository's files at {@code commit}, llama.cpp's get_repo_files over the tree API: one
     * {@link RepoFile} per regular file whose path stays inside the snapshot tree
     * ({@link HubCache#snapshotPath}) and whose oid, when listed, is 40- or 64-hex; an entry of any
     * other form is skipped, never followed.
     */
    public List<RepoFile> listFiles(HubRef ref, String commit)
            throws IOException, InterruptedException {
        String hex = HubCache.validOid(commit, "commit");
        Object parsed = apiGet(endpoint + "api/models/" + ref.repository() + "/tree/" + hex
                + "?recursive=true");
        if (!(parsed instanceof List)) {
            throw new IOException("the tree response of '" + ref + "' on " + endpoint
                    + " is not a list");
        }
        List<RepoFile> out = new ArrayList<>();
        for (Object o : (List<?>) parsed) {
            String type = stringOf(o, "type");
            String path = stringOf(o, "path");
            if (!"file".equals(type) || path == null || path.isEmpty()) {
                continue;
            }
            try {
                cache.snapshotPath(ref, hex, path);
            } catch (IllegalArgumentException e) {
                continue;
            }
            String oid = oidOf(o);
            if (oid != null && !HubCache.isSha256Oid(oid) && !HubCache.isValidOid(oid, 40)) {
                continue;
            }
            out.add(new RepoFile(ref.repository(), path, oid, resolveUrl(endpoint, ref.repository(),
                    hex, path), sizeOf(o)));
        }
        return out;
    }

    /** The resolve URL of one repository file at {@code commit}, its path percent-encoded per segment. */
    static String resolveUrl(String endpoint, String repoId, String commit, String path) {
        return endpoint + encodePath(repoId) + "/resolve/" + commit + "/" + encodePath(path);
    }

    // ================================================================== downloads

    /**
     * Downloads {@code file} of the revision {@code commit} into {@link #cache} and returns the
     * snapshot path it lands at, as llama.cpp's common_download_file_single_online runs it: a HEAD
     * gives the current ETag, a cached file with an unchanged ETag is kept, a changed cache entry is
     * redownloaded from the top, an in-progress partial is resumed over {@code Range}, and an LFS
     * file's bytes must digest to their sha256 pointer before the blob is linked into the snapshot
     * tree. RepoFile URLs with other file names ({@link #expectFile}) are never accepted.
     */
    public Path download(HubRef ref, String commit, RepoFile file)
            throws IOException, InterruptedException {
        String hex = HubCache.validOid(commit, "commit");
        expectFile(ref, hex, file);
        Path snapshot = cache.snapshotPath(ref, hex, file.path());
        String url = file.url().isEmpty()
                ? resolveUrl(endpoint, ref.repository(), hex, file.path()) : file.url();
        // llama.cpp: an LFS object's bytes land in blobs/<oid>, and the snapshot path links to it
        Path local = file.oid() != null ? cache.blobPath(ref, file.oid()) : snapshot;
        if (!isInside(cache.root(), local)) {
            throw new IOException("the download path of '" + file.path() + "' leaves the cache root");
        }
        Head head;
        try {
            head = head(url);
        } catch (IOException | InterruptedException e) {
            // llama.cpp: a failed HEAD keeps an already-complete file where it stands
            if (Files.isRegularFile(local)) {
                return cache.finalizeFile(local, snapshot);
            }
            throw e;
        }
        if (head.status < 200 || head.status >= 300) {
            if (Files.isRegularFile(local)) {
                return cache.finalizeFile(local, snapshot);
            }
            throw new IOException("HEAD over the Hub answered HTTP " + head.status + " for '"
                    + file.path() + "'");
        }
        if (Files.isRegularFile(local) && (head.etag == null
                || head.etag.equals(etag(local)))) {
            // no server etag, or the same one the last download was stamped with: cached
            return cache.finalizeFile(local, snapshot);
        }
        if (Files.isRegularFile(local)) {
            Files.delete(local);
            Files.deleteIfExists(Paths.get(local + ".etag"));
        }
        Path partial = Paths.get(local + PARTIAL_SUFFIX);
        Files.createDirectories(partial.getParent());
        if (Files.isRegularFile(partial) && head.etag != null
                && !head.etag.equals(etag(partial))) {
            // a partial of another revision resumes into nothing: drop it (llama.cpp common_pull_file)
            Files.deleteIfExists(partial);
            Files.deleteIfExists(Paths.get(partial + ".etag"));
        }
        writeEtag(partial, head.etag);
        IOException failure = null;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            if (attempt > 0) {
                // llama.cpp: 2s, then doubled per further attempt
                Thread.sleep(1000L << attempt);
            }
            if (Files.isRegularFile(partial) && !head.ranges) {
                // the server serves no ranges: an incomplete partial cannot continue either
                Files.deleteIfExists(partial);
            }
            long offset = head.ranges && Files.isRegularFile(partial) ? Files.size(partial) : 0;
            try {
                pull(url, partial, offset);
            } catch (IOException e) {
                failure = e;
                continue;
            }
            if (file.isLfs() && !sha256(partial).toLowerCase(Locale.ROOT).equals(file.oid())) {
                Files.deleteIfExists(partial);
                Files.deleteIfExists(Paths.get(partial + ".etag"));
                throw new IOException("the downloaded bytes of '" + file.path()
                        + "' digest not to the LFS sha256 pointer recorded in the repository");
            }
            try {
                Files.move(partial, local, StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(partial, local, StandardCopyOption.REPLACE_EXISTING);
            }
            writeEtag(local, head.etag);
            Files.deleteIfExists(Paths.get(partial + ".etag"));
            return cache.finalizeFile(local, snapshot);
        }
        // a stopped download keeps its partial, so the next attempt resumes from it
        throw failure != null ? failure : new IOException("download of '" + file.path() + "' failed");
    }

    /** The output of one repository HEAD request: status, ETag and whether ranges are served. */
    private static final class Head {
        int status;
        String etag;
        boolean ranges;
    }

    /** One HEAD carrying the token and the Hub user agent, without retries (llama.cpp carries none). */
    private Head head(String url) throws IOException, InterruptedException {
        HttpResponse<Void> r = http.send(requestBuilder(url)
                        .method("HEAD", HttpRequest.BodyPublishers.noBody())
                        .build(),
                HttpResponse.BodyHandlers.discarding());
        Head head = new Head();
        head.status = r.statusCode();
        head.etag = r.headers().firstValue("etag").orElse(null);
        head.ranges = !"none".equals(r.headers().firstValue("accept-ranges").orElse(""));
        return head;
    }

    /** One GET over the resolve endpoint, appended to the partial body from {@code offset}. */
    private void pull(String url, Path partial, long offset) throws IOException, InterruptedException {
        HttpRequest.Builder b = requestBuilder(url).GET();
        if (offset > 0) {
            b.header("Range", "bytes=" + offset + "-");
        }
        HttpResponse<InputStream> r = http.send(b.build(), HttpResponse.BodyHandlers.ofInputStream());
        int status = r.statusCode();
        if (offset > 0 && status != 206) {
            // llama.cpp: a resume answered with anything else than 206 Partial Content cannot go on
            r.body().close();
            Files.deleteIfExists(partial);
            throw new IOException("the resume of '" + partial.getFileName() + "' answered HTTP "
                    + status);
        }
        if (offset == 0 && status != 200) {
            r.body().close();
            throw new IOException("the download of '" + partial.getFileName() + "' answered HTTP "
                    + status);
        }
        try (InputStream in = r.body(); OutputStream out = Files.newOutputStream(partial,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            byte[] buf = new byte[64 * 1024];
            int n;
            while ((n = in.read(buf)) >= 0) {
                out.write(buf, 0, n);
            }
        }
    }

    /**
     * One GET returning a parsed JSON document: UTF-8 bytes, at most
     * {@link Json.Limits.UNTRUSTED} {@code maxBytes()}, parsed under exactly those limits.
     */
    private Object apiGet(String url) throws IOException, InterruptedException {
        HttpRequest request = requestBuilder(url)
                .header("Accept", "application/json")
                .GET()
                .build();
        HttpResponse<byte[]> r = http.send(request, HttpResponse.BodyHandlers.ofByteArray());
        if (r.statusCode() < 200 || r.statusCode() >= 300) {
            throw new IOException("GET over the Hub answered HTTP " + r.statusCode() + " for '"
                    + endpointUrl(url) + "'");
        }
        if (r.body().length > Json.Limits.UNTRUSTED.maxBytes()) {
            throw new IOException("the response of '" + endpointUrl(url) + "' is "
                    + r.body().length + " bytes, over the untrusted limit of "
                    + Json.Limits.UNTRUSTED.maxBytes());
        }
        String body = new String(r.body(), StandardCharsets.UTF_8);
        try {
            return Json.parse(body, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IOException("the response of '" + endpointUrl(url)
                    + "' is no valid JSON within the untrusted limits", e);
        }
    }

    /** The URL without its query, for error messages that keep none of the body. */
    private static String endpointUrl(String url) {
        int q = url.indexOf('?');
        return q >= 0 ? url.substring(0, q) : url;
    }

    /** A request builder with the Hub user agent and, when a token applies, the bearer header set for
     * the call that uses it; the query string never carries the token.
     */
    private HttpRequest.Builder requestBuilder(String url) {
        HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(url))
                .header("User-Agent", "neuralfree")
                .timeout(TIMEOUT);
        if (token != null) {
            b.header("Authorization", "Bearer " + token);
        }
        return b;
    }

    // ================================================================== helpers

    /**
     * llama.cpp's downloader gives every file it downloads the URL of its own path only; a
     * {@link RepoFile} whose URL does not name it is refused before it can fetch another one.
     */
    private void expectFile(HubRef ref, String commit, RepoFile file) {
        String expected = resolveUrl(endpoint, ref.repository(), commit, file.path());
        if (!expected.equals(file.url())) {
            throw new IllegalArgumentException("the registered URL of '" + file.path()
                    + "' does not name it");
        }
    }

    /** Whether {@code file} resolves inside {@code root}; a download never lands outside it. */
    private static boolean isInside(Path root, Path file) {
        Path normalized = file.toAbsolutePath().normalize();
        return normalized.startsWith(root) && !normalized.equals(root);
    }

    /** One "field is a string" read of a JSON response object; null when it is missing or another type. */
    private static String stringOf(Object o, String key) {
        if (!(o instanceof Map)) {
            return null;
        }
        Object v = ((Map<?, ?>) o).get(key);
        return v instanceof String ? (String) v : null;
    }

    /** The oid of one tree entry, from the LFS pointer when listed, else the git oid; null when neither. */
    private static String oidOf(Object o) {
        if (!(o instanceof Map)) {
            return null;
        }
        Map<?, ?> entry = (Map<?, ?>) o;
        Object lfs = entry.get("lfs");
        if (lfs instanceof Map) {
            Object oid = ((Map<?, ?>) lfs).get("oid");
            if (oid instanceof String) {
                return (String) oid;
            }
        }
        Object oid = entry.get("oid");
        return oid instanceof String ? (String) oid : null;
    }

    /** One "field is a non-negative number" read of a tree entry; -1 when anything else is there. */
    private static long sizeOf(Object o) {
        if (!(o instanceof Map)) {
            return -1L;
        }
        Object v = ((Map<?, ?>) o).get("size");
        if (v instanceof Long && ((Long) v).longValue() >= 0L) {
            return ((Long) v).longValue();
        }
        return -1L;
    }

    /** The ETag a file is compared against, from {@code <file>.etag}, or the empty string. */
    private static String etag(Path file) {
        Path etagFile = Paths.get(file + ".etag");
        if (!Files.isRegularFile(etagFile)) {
            return "";
        }
        try {
            return new String(Files.readAllBytes(etagFile), StandardCharsets.UTF_8).trim();
        } catch (IOException e) {
            return "";
        }
    }

    /** Writes one ETag next to its file, when the server supplied one. */
    private static void writeEtag(Path file, String etag) throws IOException {
        if (etag == null) {
            return;
        }
        Files.write(Paths.get(file + ".etag"), etag.getBytes(StandardCharsets.UTF_8));
    }

    /** The SHA-256 digest of {@code file}'s bytes in lowercase hex, the LFS pointer's form. */
    static String sha256(Path file) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available on this JVM", e);
        }
        try (InputStream in = Files.newInputStream(file)) {
            byte[] buf = new byte[128 * 1024];
            int n;
            while ((n = in.read(buf)) >= 0) {
                digest.update(buf, 0, n);
            }
        }
        StringBuilder hex = new StringBuilder(digest.getDigestLength() * 2);
        for (byte b : digest.digest()) {
            hex.append(Character.forDigit((b >> 4) & 0xF, 16));
            hex.append(Character.forDigit(b & 0xF, 16));
        }
        return hex.toString();
    }

    /** A ref name of the refs API, the rule {@link HubCache#refPath} applies to its files. */
    private static boolean isValidRefName(String name) {
        if (name.isEmpty() || name.length() > 256) {
            return false;
        }
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            boolean ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                    || c == '.' || c == '-' || c == '_';
            if (!ok) {
                return false;
            }
        }
        return !name.equals(".") && !name.equals("..");
    }

    /** Percent-encodes each {@code /}-separated segment of a path (UTF-8), keeping the separators. */
    static String encodePath(String path) {
        StringBuilder out = new StringBuilder();
        for (String segment : path.replace('\\', '/').split("/", -1)) {
            if (out.length() > 0) {
                out.append('/');
            }
            out.append(URLEncoder.encode(segment, StandardCharsets.UTF_8).replace("+", "%20"));
        }
        return out.toString();
    }
}
