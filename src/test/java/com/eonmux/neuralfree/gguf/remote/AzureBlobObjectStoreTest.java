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
import java.util.Base64;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link AzureBlobObjectStore}: XML listing pages, SAS and SharedKey requests, and downloads. */
public class AzureBlobObjectStoreTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private FakeObjectStoreServer azure;

    @Before
    public void start() throws IOException {
        azure = new FakeObjectStoreServer(FakeObjectStoreServer.Kind.AZURE);
    }

    @After
    public void stop() {
        azure.close();
    }

    @Test
    public void listsPagesInKeyOrder() throws IOException {
        azure.pageSize(2);
        azure.put("models/a/1.bin", "one".getBytes(StandardCharsets.UTF_8));
        azure.put("models/a/2.bin", "two".getBytes(StandardCharsets.UTF_8));
        azure.put("models/a/3.bin", "three".getBytes(StandardCharsets.UTF_8));
        AzureBlobObjectStore store = new AzureBlobObjectStore(azure.base(), "container", null, null, null);
        List<RemoteObject> listed = store.list("models/a/");
        assertEquals(3, listed.size());
        assertEquals("models/a/1.bin", listed.get(0).key());
        assertEquals(3L, listed.get(0).size());
    }

    @Test
    public void aSasTokenIsAppendedToEveryRequest() throws IOException {
        azure.requireSas("sv=1&sig=abc");
        azure.put("models/one.bin", "payload".getBytes(StandardCharsets.UTF_8));
        AzureBlobObjectStore store = new AzureBlobObjectStore(azure.base(), "container", null, null,
                "?sv=1&sig=abc");
        assertEquals(1, store.list("models/").size());
        Path out = tmp.newFolder().toPath().resolve("out.bin");
        store.download("models/one.bin", out);
        assertEquals("payload", new String(Files.readAllBytes(out), StandardCharsets.UTF_8));
    }

    @Test
    public void aMissingSasTokenIsRefusedByTheStore() {
        azure.requireSas("sig=abc");
        AzureBlobObjectStore store = new AzureBlobObjectStore(azure.base(), "container", null, null, null);
        assertThrows(IOException.class, () -> store.list("models/"));
    }

    @Test
    public void anAccountKeySignsWithSharedKey() throws IOException {
        azure.put("models/one.bin", "payload".getBytes(StandardCharsets.UTF_8));
        String key = Base64.getEncoder().encodeToString("secret".getBytes(StandardCharsets.UTF_8));
        AzureBlobObjectStore store = new AzureBlobObjectStore(azure.base(), "container", "account", key, null);
        assertEquals(1, store.list("models/").size());
        assertTrue(azure.authorizations().get(0).startsWith("SharedKey account:"));
    }

    @Test
    public void aMissingObjectIsAFileNotFound() {
        AzureBlobObjectStore store = new AzureBlobObjectStore(azure.base(), "container", null, null, null);
        assertThrows(FileNotFoundException.class,
                () -> store.download("missing.bin", tmp.newFolder().toPath().resolve("out")));
    }
}
