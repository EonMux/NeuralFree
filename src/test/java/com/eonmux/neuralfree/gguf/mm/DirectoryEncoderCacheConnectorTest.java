package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits.Modality;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link DirectoryEncoderCacheConnector}, the shared-filesystem encoder cache of a disaggregated deployment: what a
 * producer saves a consumer loads under the item's key, a consumer saves nothing, an entry another writer corrupted
 * is refused naming its file, and the directory is created on demand with no temporary file left behind.
 */
public class DirectoryEncoderCacheConnectorTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static MediaKey key(String uuid) {
        return MediaKey.ofUuid(Modality.IMAGE, uuid, null);
    }

    private static EncoderCacheEntry entry(int seed) {
        return EncoderCacheEntry.of(List.of(EncoderCacheEntry.Chunk.text("<image>"),
                EncoderCacheEntry.Chunk.rows(new float[][]{{seed, seed + 1}, {seed + 2, seed + 3}})));
    }

    @Test
    public void aProducerSavesWhatAConsumerLoads() throws IOException {
        Path dir = tmp.newFolder("shared").toPath();
        DirectoryEncoderCacheConnector producer =
                new DirectoryEncoderCacheConnector(dir, EncoderCacheConnector.Role.PRODUCER);
        DirectoryEncoderCacheConnector consumer =
                new DirectoryEncoderCacheConnector(dir, EncoderCacheConnector.Role.CONSUMER);
        MediaKey cat = key("cat");
        assertFalse(consumer.has(cat));
        assertNull(consumer.get(cat));
        producer.put(cat, entry(1));
        assertTrue(consumer.has(cat));
        EncoderCacheEntry got = consumer.get(cat);
        assertEquals(2, got.rows());
        assertEquals(2, got.width());
        assertArrayEquals(new float[]{1, 2}, got.chunks().get(1).rows()[0], 0.0f);
        assertEquals("<image>", got.chunks().get(0).text());
        try (Stream<Path> files = Files.list(dir)) {
            assertEquals("one entry file, no temporary file", 1, files.count());
        }
    }

    @Test
    public void aConsumerSavesNothing() throws IOException {
        DirectoryEncoderCacheConnector consumer = new DirectoryEncoderCacheConnector(
                tmp.newFolder("shared").toPath(), EncoderCacheConnector.Role.CONSUMER);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> consumer.put(key("cat"), entry(1)));
        assertTrue(e.getMessage(), e.getMessage().contains("ec_consumer"));
        assertTrue(consumer.isConsumer());
        assertFalse(consumer.isProducer());
    }

    @Test
    public void bothRolesSaveAndLoad() throws IOException {
        Path dir = tmp.newFolder("shared").toPath();
        DirectoryEncoderCacheConnector both =
                new DirectoryEncoderCacheConnector(dir, EncoderCacheConnector.Role.BOTH);
        both.put(key("cat"), entry(2));
        assertEquals(2, both.get(key("cat")).rows());
        assertTrue(both.isProducer());
        assertTrue(both.isConsumer());
    }

    @Test
    public void aCorruptEntryIsRefusedNamingItsFile() throws IOException {
        Path dir = tmp.newFolder("shared").toPath();
        DirectoryEncoderCacheConnector both =
                new DirectoryEncoderCacheConnector(dir, EncoderCacheConnector.Role.BOTH);
        both.put(key("cat"), entry(3));
        Path file;
        try (Stream<Path> files = Files.list(dir)) {
            file = files.findFirst().orElseThrow(AssertionError::new);
        }
        Files.write(file, "not an encoder cache entry".getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> both.get(key("cat")));
        assertTrue(e.getMessage(), e.getMessage().contains(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("cat"));
    }

    @Test
    public void theDirectoryIsCreatedOnDemandAndKeysArePortable() throws IOException {
        Path dir = tmp.newFolder("shared").toPath().resolve("a").resolve("b");
        DirectoryEncoderCacheConnector both =
                new DirectoryEncoderCacheConnector(dir, EncoderCacheConnector.Role.BOTH);
        assertTrue(Files.isDirectory(dir));
        both.put(MediaKey.ofUuid(Modality.IMAGE, "a/b:c d", null), entry(4));
        try (Stream<Path> files = Files.list(dir)) {
            Path file = files.findFirst().orElseThrow(AssertionError::new);
            assertTrue(file.getFileName().toString(), file.getFileName().toString()
                    .matches("[0-9a-f]{64}\\.nfec"));
        }
        assertEquals(2, both.get(MediaKey.ofUuid(Modality.IMAGE, "a/b:c d", null)).rows());
    }
}
