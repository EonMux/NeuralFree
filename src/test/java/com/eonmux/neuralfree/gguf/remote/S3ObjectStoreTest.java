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
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link S3ObjectStore}: listing pages, key encoding, downloads and the SigV4 signature. */
public class S3ObjectStoreTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private FakeObjectStoreServer s3;
    private S3ObjectStore store;

    @Before
    public void start() throws IOException {
        s3 = new FakeObjectStoreServer(FakeObjectStoreServer.Kind.S3);
        store = new S3ObjectStore(s3.base(), "bucket", "us-east-1", null, null, null);
    }

    @After
    public void stop() {
        s3.close();
    }

    @Test
    public void listsPagesInKeyOrder() throws IOException {
        s3.pageSize(2);
        s3.put("models/a/1.bin", bytes("one"));
        s3.put("models/a/2.bin", bytes("two"));
        s3.put("models/a/3.bin", bytes("three"));
        s3.put("models/b/4.bin", bytes("four"));
        List<RemoteObject> listed = store.list("models/a/");
        assertEquals(3, listed.size());
        assertEquals("models/a/1.bin", listed.get(0).key());
        assertEquals(3L, listed.get(0).size());
        assertEquals("models/a/3.bin", listed.get(2).key());
    }

    @Test
    public void downloadsKeysWithSpacesAndPlusSigns() throws IOException {
        s3.put("models/my file +1.bin", bytes("payload"));
        Path out = tmp.newFolder().toPath().resolve("out.bin");
        store.download("models/my file +1.bin", out);
        assertEquals("payload", new String(Files.readAllBytes(out), StandardCharsets.UTF_8));
    }

    @Test
    public void aMissingObjectIsAFileNotFound() {
        assertThrows(FileNotFoundException.class,
                () -> store.download("missing.bin", tmp.newFolder().toPath().resolve("out")));
    }

    @Test
    public void signsWithSharedCredentials() throws IOException {
        S3ObjectStore signed = new S3ObjectStore(s3.base(), "bucket", "us-east-1", "AKID", "SECRET", null);
        s3.put("models/one.bin", bytes("x"));
        assertEquals(1, signed.list("models/").size());
        assertTrue(s3.authorizations().get(0).startsWith("AWS4-HMAC-SHA256 Credential=AKID/"));
        signed.download("models/one.bin", tmp.newFolder().toPath().resolve("out"));
    }

    @Test
    public void anonymousAccessCarriesNoAuthorization() throws IOException {
        s3.put("models/one.bin", bytes("x"));
        assertEquals(1, store.list("models/").size());
        assertTrue(s3.authorizations().isEmpty());
    }

    @Test
    public void accessKeyWithoutSecretIsRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> new S3ObjectStore(s3.base(), "bucket", "us-east-1", "AKID", null, null));
    }

    @Test
    public void authorizationMatchesTheAwsSigV4Example() {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("host", "examplebucket.s3.amazonaws.com");
        headers.put("range", "bytes=0-9");
        headers.put("x-amz-content-sha256", RemoteHttp.emptyPayloadHash());
        headers.put("x-amz-date", "20130524T000000Z");
        String authorization = S3ObjectStore.authorization("GET", "/test.txt", "", headers,
                RemoteHttp.emptyPayloadHash(), "AKIAIOSFODNN7EXAMPLE",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "us-east-1", "s3",
                Instant.parse("2013-05-24T00:00:00Z"));
        assertEquals("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
                + "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,"
                + "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41", authorization);
    }

    private static byte[] bytes(String text) {
        return text.getBytes(StandardCharsets.UTF_8);
    }
}
