package com.eonmux.neuralfree.gguf.remote;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link GcsObjectStore}: JSON listing pages, anonymous and bearer requests, and downloads. */
public class GcsObjectStoreTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private FakeObjectStoreServer gcs;
    private GcsObjectStore store;

    @Before
    public void start() throws IOException {
        gcs = new FakeObjectStoreServer(FakeObjectStoreServer.Kind.GCS);
        store = new GcsObjectStore(gcs.base(), "bucket", null);
    }

    @After
    public void stop() {
        gcs.close();
    }

    @Test
    public void listsPagesInKeyOrder() throws IOException {
        gcs.pageSize(2);
        gcs.put("models/a/1.bin", "one".getBytes(StandardCharsets.UTF_8));
        gcs.put("models/a/2.bin", "two".getBytes(StandardCharsets.UTF_8));
        gcs.put("models/a/3.bin", "three".getBytes(StandardCharsets.UTF_8));
        List<RemoteObject> listed = store.list("models/a/");
        assertEquals(3, listed.size());
        assertEquals("models/a/1.bin", listed.get(0).key());
        assertEquals(3L, listed.get(0).size());
    }

    @Test
    public void downloadsObjectNamesWithSlashes() throws IOException {
        gcs.put("models/a shard/1.bin", "payload".getBytes(StandardCharsets.UTF_8));
        Path out = tmp.newFolder().toPath().resolve("out.bin");
        store.download("models/a shard/1.bin", out);
        assertEquals("payload", new String(Files.readAllBytes(out), StandardCharsets.UTF_8));
    }

    @Test
    public void aMissingObjectIsAFileNotFound() {
        assertThrows(FileNotFoundException.class,
                () -> store.download("missing.bin", tmp.newFolder().toPath().resolve("out")));
    }

    @Test
    public void aBearerTokenAuthenticates() throws IOException {
        GcsObjectStore token = new GcsObjectStore(gcs.base(), "bucket", "ya29.token");
        gcs.put("models/one.bin", new byte[]{1});
        assertEquals(1, token.list("models/").size());
        assertEquals("Bearer ya29.token", gcs.authorizations().get(0));
    }

    @Test
    public void anonymousAccessCarriesNoAuthorization() throws IOException {
        gcs.put("models/one.bin", new byte[]{1});
        assertEquals(1, store.list("models/").size());
        assertTrue(gcs.authorizations().isEmpty());
    }
}
