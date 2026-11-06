package com.eonmux.neuralfree.gguf.media;

import com.eonmux.neuralfree.gguf.media.MediaLimits.Modality;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * vLLM's {@code mm_hash} as {@link MediaKey}: a client's uuid is the item's id unless the modality's loader options
 * are hash factors, an item without a uuid is the SHA-256 digest of its bytes, embedding rows digest their shape and
 * float bytes, and keys of different sources or modalities never collide.
 */
public class MediaKeyTest {

    private static final byte[] PNG = "not really a png".getBytes(StandardCharsets.UTF_8);

    @Test
    public void aUuidWithoutHashFactorsIsTheItemsId() throws Exception {
        MediaKey key = MediaKey.ofUuid(Modality.IMAGE, "cat-1", null);
        assertEquals("cat-1", key.id());
        assertEquals(MediaKey.Source.UUID, key.source());
        assertEquals("image", key.modality());
        assertEquals("an empty factor is no factor", key, MediaKey.ofUuid(Modality.IMAGE, "cat-1", ""));
        assertEquals("image:uuid:cat-1", key.toString());
    }

    @Test
    public void hashFactorsHashTheUuidAsVllmDoes() {
        MediaKey plain = MediaKey.ofUuid(Modality.IMAGE, "cat-1", null);
        MediaKey gray = MediaKey.ofUuid(Modality.IMAGE, "cat-1", "media_io_kwargs=image{image_mode=\"L\"}");
        assertEquals(64, gray.id().length());
        assertTrue(gray.id().matches("[0-9a-f]{64}"));
        assertNotEquals(plain, gray);
        assertEquals(gray, MediaKey.ofUuid(Modality.IMAGE, "cat-1", "media_io_kwargs=image{image_mode=\"L\"}"));
        assertNotEquals(gray, MediaKey.ofUuid(Modality.IMAGE, "cat-2", "media_io_kwargs=image{image_mode=\"L\"}"));
    }

    @Test
    public void anItemWithoutAUuidIsTheDigestOfItsBytes() throws Exception {
        String expected = MediaKey.hex(MessageDigest.getInstance("SHA-256").digest(PNG));
        MediaKey key = MediaKey.ofContent(Modality.IMAGE, PNG, null);
        assertEquals(expected, key.id());
        assertEquals(MediaKey.Source.CONTENT, key.source());
        assertNotEquals("factors fold into the digest", key, MediaKey.ofContent(Modality.IMAGE, PNG, "f"));
    }

    @Test
    public void sourcesAndModalitiesKeepApart() {
        MediaKey content = MediaKey.ofContent(Modality.IMAGE, PNG, null);
        assertNotEquals("a uuid spelled like a digest names another item", content,
                MediaKey.ofUuid(Modality.IMAGE, content.id(), null));
        assertNotEquals(MediaKey.ofUuid(Modality.IMAGE, "x", null), MediaKey.ofUuid(Modality.AUDIO, "x", null));
        assertNotEquals(MediaKey.ofUuid(Modality.IMAGE, "x", null).bits(),
                MediaKey.ofUuid(Modality.AUDIO, "x", null).bits());
    }

    @Test
    public void embeddingRowsDigestTheirShapeAndFloatBytes() {
        float[][] rows = {{1f, 2f, 3f, 4f}, {5f, 6f, 7f, 8f}};
        MediaKey key = MediaKey.ofRows(rows);
        assertEquals(MediaKey.Source.ROWS, key.source());
        assertEquals(key, MediaKey.ofRows(new float[][] {{1f, 2f, 3f, 4f}, {5f, 6f, 7f, 8f}}));
        assertNotEquals(key, MediaKey.ofRows(new float[][] {{1f, 2f, 3f, 4f}, {5f, 6f, 7f, 8.0001f}}));
        assertNotEquals("the same floats in another shape", key,
                MediaKey.ofRows(new float[][] {{1f, 2f}, {3f, 4f}, {5f, 6f}, {7f, 8f}}));
        assertThrows(IllegalArgumentException.class, () -> MediaKey.ofRows(new float[][] {{1f}, {1f, 2f}}));
        assertThrows(IllegalArgumentException.class, () -> MediaKey.ofRows(new float[0][]));
    }

    @Test
    public void aBlankUuidIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> MediaKey.ofUuid(Modality.IMAGE, "  ", null));
    }

    @Test
    public void aHandleRoundTripsThroughParse() {
        MediaKey uuid = MediaKey.ofUuid(Modality.IMAGE, "cat:1", null);
        assertEquals("a uuid keeps its colons", uuid, MediaKey.parse(uuid.toString()));
        MediaKey content = MediaKey.ofContent(Modality.AUDIO, PNG, "f");
        assertEquals(content, MediaKey.parse(content.toString()));
        MediaKey rows = MediaKey.ofRows(new float[][] {{1f, 2f}});
        assertEquals(rows, MediaKey.parse(rows.toString()));
        assertEquals("the rows key carries the embeddings modality", "embeddings:rows:",
                rows.toString().substring(0, "embeddings:rows:".length()));
    }

    @Test
    public void aMalformedHandleIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> MediaKey.parse("image"));
        assertThrows(IllegalArgumentException.class, () -> MediaKey.parse("image:other:id"));
        assertThrows(IllegalArgumentException.class, () -> MediaKey.parse("image:uuid:"));
    }
}
