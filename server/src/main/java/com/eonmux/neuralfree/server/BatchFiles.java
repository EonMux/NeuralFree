package com.eonmux.neuralfree.server;

import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * The batch runner's input and output files, local or remote (vLLM's {@code run_batch}): an {@code -i} URL is
 * downloaded with GET into a staging directory before the batch is validated (a local input that is no regular
 * file, such as a pipe, is copied there, since the batch is read twice), and an {@code -o} URL receives the
 * finished output file with one PUT once every record is written. Staging goes to {@code --output-tmp-dir} when
 * given, else to the system temporary directory. Only {@code http} and {@code https} URLs are accepted; any other
 * {@code scheme://} is refused naming the option. A connection failure, a timeout or a {@code 408}, {@code 429}
 * or {@code 5xx} answer is retried; any other non-{@code 2xx} answer fails at once. Messages name a URL without
 * its query, so a pre-signed URL's signature never reaches the log. Immutable.
 */
final class BatchFiles {

    /** The transfer attempts of one file (vLLM's upload retries). */
    static final int DEFAULT_ATTEMPTS = 5;

    /** The pause between two attempts. */
    static final Duration DEFAULT_RETRY_DELAY = Duration.ofSeconds(5);

    /** The wait for a download's response headers (vLLM's input download timeout). */
    static final Duration DEFAULT_DOWNLOAD_TIMEOUT = Duration.ofSeconds(300);

    /** The wait for an upload's response (vLLM's upload timeout). */
    static final Duration DEFAULT_UPLOAD_TIMEOUT = Duration.ofSeconds(1000);

    private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(30);

    /** The most characters of an error answer a failure message quotes. */
    private static final int MAX_ERROR_TEXT = 1000;

    /** A {@code scheme://} prefix; a Windows drive path such as {@code C:\x} does not match. */
    private static final Pattern SCHEME = Pattern.compile("^[A-Za-z][A-Za-z0-9+.-]*://.*", Pattern.DOTALL);

    /** A batch file's place: a local path, or an {@code http}/{@code https} URL. Immutable. */
    record Location(String option, String raw, Path path, URI url) {

        /** Whether the file is remote. */
        boolean isUrl() {
            return url != null;
        }

        /** The place as messages name it: the path, or the URL without its query and fragment. */
        String display() {
            if (url == null) {
                return path.toString();
            }
            return url.getScheme() + "://" + url.getRawAuthority() + (url.getRawPath() == null ? "" : url.getRawPath());
        }
    }

    private final HttpClient client;
    private final int attempts;
    private final Duration retryDelay;
    private final Duration downloadTimeout;
    private final Duration uploadTimeout;

    BatchFiles(HttpClient client, int attempts, Duration retryDelay, Duration downloadTimeout,
               Duration uploadTimeout) {
        if (attempts < 1) {
            throw new IllegalArgumentException("attempts must be at least 1, got " + attempts);
        }
        this.client = client;
        this.attempts = attempts;
        this.retryDelay = retryDelay;
        this.downloadTimeout = downloadTimeout;
        this.uploadTimeout = uploadTimeout;
    }

    /** The runner's transfers: the reference's attempts, pause and timeouts, redirects followed within a scheme. */
    static BatchFiles standard() {
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(CONNECT_TIMEOUT)
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
        return new BatchFiles(client, DEFAULT_ATTEMPTS, DEFAULT_RETRY_DELAY, DEFAULT_DOWNLOAD_TIMEOUT,
                DEFAULT_UPLOAD_TIMEOUT);
    }

    /**
     * The place {@code raw} names for {@code option}: an {@code http}/{@code https} URL with a host, or a local path.
     *
     * @throws IllegalArgumentException for another scheme, a malformed URL or a URL without a host
     */
    static Location location(String option, String raw) {
        if (raw == null || raw.isBlank()) {
            throw new IllegalArgumentException(option + " must name a file or an http(s) URL");
        }
        if (!SCHEME.matcher(raw).matches()) {
            return new Location(option, raw, Path.of(raw), null);
        }
        String scheme = raw.substring(0, raw.indexOf(':')).toLowerCase(Locale.ROOT);
        if (!scheme.equals("http") && !scheme.equals("https")) {
            throw new IllegalArgumentException(option + " URL scheme '" + scheme
                    + "' is not supported; a batch file URL must be http or https");
        }
        URI url;
        try {
            url = new URI(raw);
        } catch (URISyntaxException e) {
            throw new IllegalArgumentException(option + " is not a valid URL: " + e.getMessage(), e);
        }
        if (url.getHost() == null || url.getHost().isEmpty()) {
            throw new IllegalArgumentException(option + " URL has no host");
        }
        return new Location(option, raw, null, url);
    }

    // ------------------------------------------------------------------ input

    /** The batch input as a local file; a downloaded copy is deleted on close. */
    static final class Input implements AutoCloseable {
        private final Path path;
        private final Path stagingDir;

        private Input(Path path, Path stagingDir) {
            this.path = path;
            this.stagingDir = stagingDir;
        }

        /** The local file to read. */
        Path path() {
            return path;
        }

        @Override
        public void close() throws IOException {
            if (stagingDir != null) {
                Files.deleteIfExists(path);
                Files.deleteIfExists(stagingDir);
            }
        }
    }

    /**
     * The input of {@code location}: a local regular file as it is; a URL downloaded with GET, or a pipe copied,
     * into a fresh {@code tmp_batch_input_*} directory under {@code tmpDir} (the system temporary directory when
     * null).
     */
    Input openInput(Location location, Path tmpDir) throws IOException {
        if (!location.isUrl() && Files.isRegularFile(location.path())) {
            return new Input(location.path(), null);
        }
        Path dir = tmpDir == null ? Files.createTempDirectory("tmp_batch_input_")
                : Files.createTempDirectory(tmpDir, "tmp_batch_input_");
        Path file = dir.resolve("input.jsonl");
        Input input = new Input(file, dir);
        try {
            if (!location.isUrl()) {
                // a pipe reads once; the batch is read twice (validation, then the run)
                System.err.println("Staging " + location.display() + " so it can be read twice");
                Files.copy(location.path(), file);
                return input;
            }
            System.err.println("Downloading " + location.display() + " to " + file);
            withRetries("download of " + location.option() + " " + location.display(),
                    () -> download(location.url(), file));
            return input;
        } catch (IOException | RuntimeException e) {
            input.close();
            throw e;
        }
    }

    /** One GET of {@code url} into {@code file}. */
    private void download(URI url, Path file) throws IOException, TransferException {
        HttpRequest request = HttpRequest.newBuilder(url).timeout(downloadTimeout).GET().build();
        HttpResponse<InputStream> response = send(request, HttpResponse.BodyHandlers.ofInputStream());
        try (InputStream body = response.body()) {
            checkStatus(response.statusCode(), body);
            Files.copy(body, file, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw failureOf(e);
        }
    }

    // ------------------------------------------------------------------ output

    /**
     * The batch output: a local path written in place, or a staged {@code tmp_batch_output_*.jsonl} file that
     * {@link #finish} uploads. Closing deletes the staged file, except after a failed upload, whose message names
     * the kept file so the finished batch is not lost.
     */
    final class Output implements AutoCloseable {
        private final Location location;
        private final Path path;
        private boolean keep;

        private Output(Location location, Path path) {
            this.location = location;
            this.path = path;
        }

        /** The local file the records go to. */
        Path path() {
            return path;
        }

        /** Publishes the finished file: uploads a URL output with PUT; a local output is already in place. */
        void finish() throws IOException {
            if (!location.isUrl()) {
                return;
            }
            System.err.println("Uploading outputs to " + location.display());
            try {
                withRetries("upload of " + location.option() + " " + location.display(),
                        () -> upload(location.url(), path));
            } catch (IOException e) {
                keep = true;
                throw new IOException(e.getMessage() + " The output stays at " + path + ".", e);
            }
        }

        @Override
        public void close() throws IOException {
            if (location.isUrl() && !keep) {
                Files.deleteIfExists(path);
            }
        }
    }

    /** The output of {@code location}; a URL output stages under {@code tmpDir} (the system one when null). */
    Output openOutput(Location location, Path tmpDir) throws IOException {
        if (!location.isUrl()) {
            return new Output(location, location.path());
        }
        Path staged = tmpDir == null ? Files.createTempFile("tmp_batch_output_", ".jsonl")
                : Files.createTempFile(tmpDir, "tmp_batch_output_", ".jsonl");
        return new Output(location, staged);
    }

    /** One PUT of {@code file} to {@code url}. */
    private void upload(URI url, Path file) throws IOException, TransferException {
        HttpRequest request = HttpRequest.newBuilder(url).timeout(uploadTimeout)
                .PUT(HttpRequest.BodyPublishers.ofFile(file)).build();
        HttpResponse<InputStream> response = send(request, HttpResponse.BodyHandlers.ofInputStream());
        try (InputStream body = response.body()) {
            checkStatus(response.statusCode(), body);
            body.transferTo(OutputStream.nullOutputStream());
        } catch (IOException e) {
            throw failureOf(e);
        }
    }

    // ------------------------------------------------------------------ transfers

    /** One transfer attempt. */
    @FunctionalInterface
    private interface Attempt {
        void run() throws IOException, TransferException;
    }

    /** A failed attempt: its message and whether another attempt may succeed. */
    private static final class TransferException extends Exception {
        private final boolean retryable;

        TransferException(String message, boolean retryable, Throwable cause) {
            super(message, cause);
            this.retryable = retryable;
        }

        /** The failure of an answer: retryable for 408, 429 and 5xx. */
        static TransferException ofStatus(int status, String text) {
            boolean retryable = status == 408 || status == 429 || status >= 500;
            return new TransferException("HTTP status " + status + (text.isEmpty() ? "" : ": " + text),
                    retryable, null);
        }
    }

    /** Runs {@code attempt} up to {@link #attempts} times while its failure is retryable. */
    private void withRetries(String what, Attempt attempt) throws IOException {
        for (int i = 1; ; i++) {
            try {
                attempt.run();
                return;
            } catch (TransferException e) {
                if (!e.retryable || i >= attempts) {
                    throw new IOException("The " + what + " failed (attempt " + i + " of " + attempts + "): "
                            + e.getMessage() + ".", e);
                }
                System.err.println("The " + what + " failed (attempt " + i + " of " + attempts + "): "
                        + e.getMessage() + ". Retrying in " + retryDelay.toSeconds() + " seconds...");
                pause();
            }
        }
    }

    /** Sends one request; a connection failure or a timeout is a retryable failure. */
    private <T> HttpResponse<T> send(HttpRequest request, HttpResponse.BodyHandler<T> handler)
            throws IOException, TransferException {
        try {
            return client.send(request, handler);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new InterruptedIOException("interrupted while transferring a batch file");
        } catch (IOException e) {
            throw failureOf(e);
        }
    }

    /**
     * The retryable failure of a connection, a timeout or a body that broke off; an interruption stays an
     * {@link InterruptedIOException} and ends the transfer.
     */
    private static TransferException failureOf(IOException e) throws InterruptedIOException {
        if (e instanceof InterruptedIOException interrupted && !(e instanceof java.net.SocketTimeoutException)
                && !(e instanceof java.net.http.HttpTimeoutException)) {
            throw interrupted;
        }
        String reason = e.getMessage() == null ? e.getClass().getSimpleName()
                : e.getClass().getSimpleName() + ": " + e.getMessage();
        return new TransferException(reason, true, e);
    }

    /** Refuses a non-{@code 2xx} answer, quoting the start of its body. */
    private static void checkStatus(int status, InputStream body) throws TransferException {
        if (status >= 200 && status < 300) {
            return;
        }
        String text;
        try {
            text = new String(body.readNBytes(MAX_ERROR_TEXT), StandardCharsets.UTF_8).strip();
        } catch (IOException e) {
            text = "";
        }
        throw TransferException.ofStatus(status, text);
    }

    private void pause() throws InterruptedIOException {
        try {
            Thread.sleep(retryDelay.toMillis());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new InterruptedIOException("interrupted while waiting to retry a batch file transfer");
        }
    }
}
