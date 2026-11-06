package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.server.BatchHttpFixture.Reply;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.http.HttpClient;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The batch runner's remote files over a local HTTP server, as vLLM's {@code run_batch} moves them: an input URL
 * is fetched with GET into the staging directory and removed afterwards, an output URL receives the finished file
 * with PUT, only http(s) URLs pass, transient failures (connection refused, timeouts, 5xx) are retried, other
 * answers fail at once, a failed upload keeps the staged file, and no message carries a URL's query.
 */
class BatchFilesTest {

    private static final String RECORDS = "{\"custom_id\": \"a\"}\n{\"custom_id\": \"b\"}\n";

    @TempDir
    Path dir;

    private BatchHttpFixture http;

    @BeforeEach
    void startServer() throws IOException {
        http = new BatchHttpFixture();
    }

    @AfterEach
    void stopServer() {
        http.close();
    }

    /** Transfers that retry at once, {@code attempts} times, with short timeouts. */
    private static BatchFiles files(int attempts) {
        return new BatchFiles(HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build(), attempts,
                Duration.ZERO, Duration.ofMillis(500), Duration.ofSeconds(5));
    }

    private static long entries(Path directory) throws IOException {
        try (Stream<Path> listed = Files.list(directory)) {
            return listed.count();
        }
    }

    @Test
    void onlyHttpAndHttpsUrlsAreRemoteAndOtherSchemesAreRefusedByName() {
        assertTrue(BatchFiles.location("--input-file", "http://host:8000/in.jsonl").isUrl());
        assertTrue(BatchFiles.location("--input-file", "HTTPS://host/in.jsonl?sig=x").isUrl());
        BatchFiles.Location local = BatchFiles.location("--input-file", "data/in.jsonl");
        assertFalse(local.isUrl());
        assertEquals(Path.of("data/in.jsonl"), local.path());
        assertFalse(BatchFiles.location("--input-file", "C:\\batches\\in.jsonl").isUrl(), "a drive path is local");
        for (String refused : new String[]{"ftp://host/in.jsonl", "file:///tmp/in.jsonl", "s3://bucket/in.jsonl"}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> BatchFiles.location("--output-file", refused));
            assertTrue(e.getMessage().contains("--output-file"), e.getMessage());
            assertTrue(e.getMessage().contains("http or https"), e.getMessage());
        }
        IllegalArgumentException noHost = assertThrows(IllegalArgumentException.class,
                () -> BatchFiles.location("--input-file", "http:///in.jsonl"));
        assertTrue(noHost.getMessage().contains("no host"), noHost.getMessage());
        IllegalArgumentException malformed = assertThrows(IllegalArgumentException.class,
                () -> BatchFiles.location("--input-file", "http://host/in put.jsonl"));
        assertTrue(malformed.getMessage().contains("not a valid URL"), malformed.getMessage());
        assertThrows(IllegalArgumentException.class, () -> BatchFiles.location("--input-file", " "));
    }

    @Test
    void anInputUrlIsDownloadedWithGetIntoTheStagingDirectoryAndRemovedOnClose() throws IOException {
        http.on("/in.jsonl", Reply.ok(RECORDS));
        Path staging = Files.createDirectory(dir.resolve("staging"));
        BatchFiles.Location location = BatchFiles.location("--input-file", http.url("/in.jsonl?sig=secret"));
        Path downloaded;
        try (BatchFiles.Input input = files(1).openInput(location, staging)) {
            downloaded = input.path();
            assertTrue(downloaded.startsWith(staging), "the download stages under --output-tmp-dir: " + downloaded);
            assertEquals(RECORDS, Files.readString(downloaded));
        }
        assertFalse(Files.exists(downloaded));
        assertEquals(0, entries(staging), "the staging directory is removed with the download");
        assertEquals("GET", http.received().get(0).method());
        assertEquals("/in.jsonl?sig=secret", http.received().get(0).pathAndQuery());
    }

    @Test
    void aLocalInputIsReadInPlace() throws IOException {
        Path local = Files.writeString(dir.resolve("in.jsonl"), RECORDS);
        try (BatchFiles.Input input = files(1).openInput(BatchFiles.location("-i", local.toString()), null)) {
            assertEquals(local, input.path());
        }
        assertTrue(Files.exists(local), "closing never deletes the caller's file");
    }

    @Test
    void aClientErrorFailsAtOnceAndTheMessageOmitsTheQuery() {
        http.on("/in.jsonl", Reply.status(404));
        BatchFiles.Location location = BatchFiles.location("--input-file", http.url("/in.jsonl?sig=secret"));
        IOException e = assertThrows(IOException.class, () -> files(3).openInput(location, dir));
        assertTrue(e.getMessage().contains("404"), e.getMessage());
        assertTrue(e.getMessage().contains("--input-file"), e.getMessage());
        assertTrue(e.getMessage().contains("attempt 1 of 3"), e.getMessage());
        assertFalse(e.getMessage().contains("secret"), "a pre-signed URL's query stays out: " + e.getMessage());
        assertEquals(1, http.received().size(), "a 404 is not retried");
    }

    @Test
    void aServerErrorIsRetriedUntilItSucceeds() throws IOException {
        http.on("/in.jsonl", Reply.status(503), Reply.status(500), Reply.ok(RECORDS));
        BatchFiles.Location location = BatchFiles.location("--input-file", http.url("/in.jsonl"));
        try (BatchFiles.Input input = files(3).openInput(location, dir)) {
            assertEquals(RECORDS, Files.readString(input.path()));
        }
        assertEquals(3, http.received().size());
    }

    @Test
    void aRefusedConnectionFailsAfterEveryAttemptAndLeavesNoStaging() throws IOException {
        int port;
        try (ServerSocket socket = new ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        Path staging = Files.createDirectory(dir.resolve("staging"));
        BatchFiles.Location location = BatchFiles.location("--input-file", "http://127.0.0.1:" + port + "/in.jsonl");
        IOException e = assertThrows(IOException.class, () -> files(2).openInput(location, staging));
        assertTrue(e.getMessage().contains("attempt 2 of 2"), e.getMessage());
        assertTrue(e.getMessage().contains("ConnectException"), e.getMessage());
        assertEquals(0, entries(staging), "a failed download removes its staging directory");
    }

    @Test
    void aTimeoutIsANetworkFailure() {
        http.on("/slow.jsonl", new Reply(200, RECORDS, 3000));
        BatchFiles.Location location = BatchFiles.location("--input-file", http.url("/slow.jsonl"));
        IOException e = assertThrows(IOException.class, () -> files(1).openInput(location, dir));
        assertTrue(e.getMessage().contains("HttpTimeoutException"), e.getMessage());
    }

    @Test
    void anOutputUrlReceivesTheFinishedFileWithPutAndTheStagedCopyIsRemoved() throws IOException {
        http.on("/out.jsonl", new Reply(201, "", 0));
        Path staging = Files.createDirectory(dir.resolve("staging"));
        BatchFiles.Location location = BatchFiles.location("--output-file", http.url("/out.jsonl?sig=secret"));
        Path staged;
        try (BatchFiles.Output output = files(1).openOutput(location, staging)) {
            staged = output.path();
            assertTrue(staged.startsWith(staging), "the output stages under --output-tmp-dir: " + staged);
            Files.writeString(staged, RECORDS);
            assertTrue(http.received().isEmpty(), "nothing is uploaded before the batch finishes");
            output.finish();
        }
        assertFalse(Files.exists(staged));
        assertEquals(1, http.received().size());
        assertEquals("PUT", http.received().get(0).method());
        assertEquals("/out.jsonl?sig=secret", http.received().get(0).pathAndQuery());
        assertEquals(RECORDS, http.received().get(0).body());
    }

    @Test
    void anOutputIsNotUploadedWithoutFinishAndItsStagedCopyIsRemoved() throws IOException {
        BatchFiles.Location location = BatchFiles.location("--output-file", http.url("/out.jsonl"));
        Path staged;
        try (BatchFiles.Output output = files(1).openOutput(location, dir)) {
            staged = output.path();
            Files.writeString(staged, RECORDS);
        }
        assertFalse(Files.exists(staged));
        assertTrue(http.received().isEmpty(), "a run that did not finish uploads nothing");
    }

    @Test
    void aFailedUploadKeepsTheStagedFileAndNamesIt() throws IOException {
        http.on("/out.jsonl", Reply.status(500));
        BatchFiles.Location location = BatchFiles.location("--output-file", http.url("/out.jsonl"));
        Path staged;
        IOException e;
        try (BatchFiles.Output output = files(2).openOutput(location, dir)) {
            staged = output.path();
            Files.writeString(staged, RECORDS);
            e = assertThrows(IOException.class, output::finish);
        }
        assertTrue(e.getMessage().contains("upload of --output-file"), e.getMessage());
        assertTrue(e.getMessage().contains("500"), e.getMessage());
        assertTrue(e.getMessage().contains(staged.toString()), "the kept file is named: " + e.getMessage());
        assertTrue(Files.exists(staged), "the finished batch is not lost");
        assertEquals(RECORDS, Files.readString(staged));
        assertEquals(2, http.received().size(), "a 500 is retried");
    }

    @Test
    void aForbiddenUploadIsNotRetried() throws IOException {
        http.on("/out.jsonl", Reply.status(403));
        BatchFiles.Location location = BatchFiles.location("--output-file", http.url("/out.jsonl"));
        try (BatchFiles.Output output = files(3).openOutput(location, dir)) {
            Files.writeString(output.path(), RECORDS);
            IOException e = assertThrows(IOException.class, output::finish);
            assertTrue(e.getMessage().contains("403"), e.getMessage());
        }
        assertEquals(1, http.received().size());
    }

    @Test
    void aLocalOutputIsWrittenInPlace() throws IOException {
        Path local = dir.resolve("out.jsonl");
        try (BatchFiles.Output output = files(1).openOutput(BatchFiles.location("-o", local.toString()), dir)) {
            assertEquals(local, output.path());
            Files.writeString(output.path(), RECORDS);
            output.finish();
        }
        assertEquals(RECORDS, Files.readString(local));
    }

    @Test
    void aPipedInputIsStagedSoItCanBeReadTwice() throws Exception {
        Path fifo = dir.resolve("in.fifo");
        Process mkfifo = new ProcessBuilder("mkfifo", fifo.toString()).start();
        org.junit.jupiter.api.Assumptions.assumeTrue(mkfifo.waitFor() == 0, "mkfifo is available");
        Thread writer = new Thread(() -> {
            try {
                Files.writeString(fifo, RECORDS);
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        });
        writer.start();
        Path staging = Files.createDirectory(dir.resolve("staging"));
        Path staged;
        try (BatchFiles.Input input = files(1).openInput(BatchFiles.location("-i", fifo.toString()), staging)) {
            staged = input.path();
            assertTrue(staged.startsWith(staging), "the pipe stages under --output-tmp-dir: " + staged);
            assertEquals(RECORDS, Files.readString(staged));
            assertEquals(RECORDS, Files.readString(staged), "the staged copy reads again");
        }
        writer.join(5000);
        assertFalse(Files.exists(staged));
        assertEquals(0, entries(staging));
    }
}
