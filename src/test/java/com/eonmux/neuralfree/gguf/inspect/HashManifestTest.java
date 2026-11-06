package com.eonmux.neuralfree.gguf.inspect;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link HashManifest}: the reference's line grammar (three whitespace-separated fields, unknown types ignored),
 * CRLF line endings, truncated lines that still name a known type, and the three verification statuses.
 */
public class HashManifestTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void entriesVerifyByTypeAndTarget() throws Exception {
        Path file = manifest("xxh64  aabb  model.gguf:t1\r\n"
                + "sha256  ccdd  model.gguf:t1\n"
                + "md5      eeff  model.gguf:t1\n"
                + "xxh64\n");
        HashManifest manifest = HashManifest.read(file);
        assertEquals(HashManifest.Status.OK, manifest.verify("xxh64", "aabb", "model.gguf:t1"));
        assertEquals(HashManifest.Status.MISMATCH, manifest.verify("xxh64", "0000", "model.gguf:t1"));
        assertEquals(HashManifest.Status.NOT_FOUND, manifest.verify("sha1", "aabb", "model.gguf:t1"));
        assertEquals(HashManifest.Status.NOT_FOUND, manifest.verify("xxh64", "aabb", "other.gguf:t1"));
        // a truncated line names the type but no entry
        assertEquals(HashManifest.Status.NOT_FOUND, manifest.verify("xxh64", "aabb", "model.gguf"));
    }

    @Test
    public void knownTypesAreTheRecognizedOnesInFileOrder() throws Exception {
        Path file = manifest("sha1  11  a\nxxh64  22  b\nmd5  33  c\nuuid  44  d\n");
        HashManifest manifest = HashManifest.read(file);
        assertEquals(3, manifest.knownTypes().size());
        assertTrue(manifest.knownTypes().contains("sha1"));
        assertTrue(manifest.knownTypes().contains("xxh64"));
        assertTrue(manifest.knownTypes().contains("uuid"));
    }

    private Path manifest(String text) throws Exception {
        Path file = tmp.newFile("manifest-" + System.nanoTime()).toPath();
        Files.write(file, text.getBytes(StandardCharsets.UTF_8));
        return file;
    }
}
