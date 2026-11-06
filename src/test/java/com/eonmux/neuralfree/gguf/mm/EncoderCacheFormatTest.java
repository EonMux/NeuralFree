package com.eonmux.neuralfree.gguf.mm;

import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EncoderCacheFormat}: every chunk kind of an {@link EncoderCacheEntry} survives the file form with its rows,
 * grids, lead pads, canvas tiles and retention masks, and a foreign, truncated, corrupt or misnamed file is refused
 * rather than fed to a prompt.
 */
public class EncoderCacheFormatTest {

    private static final String KEY = "image:content:0123456789abcdef";

    private static float[][] rows(int from) {
        return new float[][]{{from, from + 1, from + 2}, {from + 3, from + 4, from + 5}};
    }

    private static EncoderCacheEntry entry() {
        List<EncoderCacheEntry.Chunk> chunks = new ArrayList<>();
        chunks.add(EncoderCacheEntry.Chunk.text("<image>"));
        chunks.add(EncoderCacheEntry.Chunk.imageId());
        chunks.add(EncoderCacheEntry.Chunk.rows(rows(1)));
        chunks.add(EncoderCacheEntry.Chunk.grid(new float[][]{{7, 8, 9}, {10, 11, 12}, {13, 14, 15}}, 2, 2,
                new boolean[]{true, false, true, true}));
        chunks.add(EncoderCacheEntry.Chunk.leadPadded(rows(20), 1));
        chunks.add(EncoderCacheEntry.Chunk.canvas(rows(30), 1, 1,
                Arrays.asList(new EncoderCacheEntry.CanvasTile(1, 1, 0), new EncoderCacheEntry.CanvasTile(1, 1, 0))));
        return EncoderCacheEntry.of(chunks);
    }

    @Test
    public void everyChunkKindRoundTrips() throws IOException {
        EncoderCacheEntry entry = entry();
        EncoderCacheEntry got = EncoderCacheFormat.decode(KEY, EncoderCacheFormat.encode(KEY, entry));
        assertEquals(entry.rows(), got.rows());
        assertEquals(3, got.width());
        assertEquals(9, got.rows());
        List<EncoderCacheEntry.Chunk> chunks = got.chunks();
        assertEquals(6, chunks.size());
        assertEquals(EncoderCacheEntry.Kind.TEXT, chunks.get(0).kind());
        assertEquals("<image>", chunks.get(0).text());
        assertEquals(EncoderCacheEntry.Kind.IMAGE_ID, chunks.get(1).kind());
        assertEquals(EncoderCacheEntry.Kind.ROWS, chunks.get(2).kind());
        assertArrayEquals(rows(1)[0], chunks.get(2).rows()[0], 0.0f);
        assertArrayEquals(rows(1)[1], chunks.get(2).rows()[1], 0.0f);
        EncoderCacheEntry.Chunk grid = chunks.get(3);
        assertEquals(2, grid.nx());
        assertEquals(2, grid.ny());
        assertEquals(3, grid.rowCount());
        assertArrayEquals(new boolean[]{true, false, true, true}, grid.mask());
        EncoderCacheEntry.Chunk lead = chunks.get(4);
        assertEquals(1, lead.lead());
        assertArrayEquals(rows(20)[1], lead.rows()[1], 0.0f);
        EncoderCacheEntry.Chunk canvas = chunks.get(5);
        assertEquals(1, canvas.nx());
        assertEquals(1, canvas.ny());
        assertEquals(2, canvas.canvas().size());
        assertEquals(new EncoderCacheEntry.CanvasTile(1, 1, 0), canvas.canvas().get(0));
        assertArrayEquals(rows(30)[0], canvas.rows()[0], 0.0f);
    }

    @Test
    public void aForeignFileIsRefused() {
        byte[] foreign = new byte[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        IOException e = assertThrows(IOException.class, () -> EncoderCacheFormat.decode(KEY, foreign));
        assertTrue(e.getMessage(), e.getMessage().contains("magic"));
    }

    @Test
    public void aTruncatedOrCorruptFileIsRefused() {
        byte[] good = EncoderCacheFormat.encode(KEY, entry());
        byte[] shortFile = Arrays.copyOf(good, good.length - 1);
        IOException e = assertThrows(IOException.class, () -> EncoderCacheFormat.decode(KEY, shortFile));
        assertTrue(e.getMessage(), e.getMessage().contains("corrupt"));
        byte[] corrupt = good.clone();
        corrupt[good.length / 2] ^= 0x40;
        e = assertThrows(IOException.class, () -> EncoderCacheFormat.decode(KEY, corrupt));
        assertTrue(e.getMessage(), e.getMessage().contains("corrupt"));
        assertThrows(IOException.class, () -> EncoderCacheFormat.decode(KEY, new byte[10]));
    }

    @Test
    public void anEntryStoredUnderAnotherKeyIsRefused() {
        byte[] good = EncoderCacheFormat.encode(KEY, entry());
        IOException e = assertThrows(IOException.class,
                () -> EncoderCacheFormat.decode("image:content:other", good));
        assertTrue(e.getMessage(), e.getMessage().contains(KEY));
        assertTrue(e.getMessage(), e.getMessage().contains("other"));
    }

    @Test
    public void aFileWithTrailingBytesIsRefused() {
        byte[] good = EncoderCacheFormat.encode(KEY, entry());
        byte[] longer = Arrays.copyOf(good, good.length + 4);
        assertThrows(IOException.class, () -> EncoderCacheFormat.decode(KEY, longer));
    }

    @Test
    public void anEntryWithoutRowsKeepsItsMarkerChunks() throws IOException {
        EncoderCacheEntry markers = EncoderCacheEntry.of(Arrays.asList(EncoderCacheEntry.Chunk.text("x"),
                EncoderCacheEntry.Chunk.imageId()));
        EncoderCacheEntry got = EncoderCacheFormat.decode(KEY, EncoderCacheFormat.encode(KEY, markers));
        assertEquals(0, got.rows());
        assertEquals(0, got.width());
        assertEquals(2, got.chunks().size());
        assertNull(got.chunks().get(0).rows());
    }
}
