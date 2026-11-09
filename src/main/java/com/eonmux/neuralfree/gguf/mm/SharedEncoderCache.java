package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The shared encoder cache exchange of one media context (vLLM's encoder cache connector): a consumer loads an
 * item's encoding from the connector before it reads the item's data, decodes or encodes it, and a producer saves
 * the encoding of every item it encodes. The sharing key is the item's {@link MediaKey}, and an entry loaded from
 * the connector is checked against the projector's row width before it is used and cached in the context's
 * encoded layer, so the item's data is never read and its encoder never runs again.
 */
final class SharedEncoderCache {

    /** The connector this context loads from and saves to, or null. */
    private final EncoderCacheConnector connector;
    /** The context's encoded layer, which a connector hit is cached into and a producer's save comes from. */
    private final EncoderOutputCache<MediaKey, List<MediaPiece>> encoded;
    /** The embedding width of every media row, which a loaded entry is checked against. */
    private final int width;
    private final AtomicLong loads = new AtomicLong();
    private final AtomicLong saves = new AtomicLong();

    SharedEncoderCache(EncoderCacheConnector connector, EncoderOutputCache<MediaKey, List<MediaPiece>> encoded,
                       int width) {
        this.connector = connector;
        this.encoded = encoded;
        this.width = width;
    }

    /** Whether a consumer connector holds {@code key}; false without a connector or on a producer. */
    boolean holds(MediaKey key) {
        return connector != null && connector.isConsumer() && connector.has(key);
    }

    /** Closes the connector; a context without one closes nothing. */
    void close() throws IOException {
        if (connector != null) {
            connector.close();
        }
    }

    /** The items this consumer loaded from the shared encoder cache, since the context opened. */
    long loads() {
        return loads.get();
    }

    /** The items this producer saved to the shared encoder cache, since the context opened. */
    long saves() {
        return saves.get();
    }

    /**
     * The encoding the encoded layer, or for a consumer the shared encoder cache, holds for {@code source}, or null.
     * A shared entry is checked against this projector's row width and cached locally, so the item's data is never
     * read and its encoder never runs.
     */
    List<MediaPiece> shared(MediaItem source) {
        MediaKey key = source.key();
        List<MediaPiece> hit = encoded.get(key);
        if (hit != null) {
            return hit;
        }
        if (connector == null || !connector.isConsumer()) {
            return null;
        }
        EncoderCacheEntry entry;
        try {
            entry = connector.get(key);
        } catch (IOException e) {
            throw new MediaInputException("the shared encoder cache cannot load item " + key + ": "
                    + e.getMessage(), e);
        }
        if (entry == null) {
            return null;
        }
        List<MediaPiece> pieces = piecesOf(entry, key);
        encoded.put(key, pieces);
        loads.incrementAndGet();
        return pieces;
    }

    /** {@code pieces} cached locally and, for a producer, saved to the shared encoder cache. */
    List<MediaPiece> store(MediaItem source, List<MediaPiece> pieces) {
        MediaKey key = source.key();
        encoded.put(key, pieces);
        if (connector == null || !connector.isProducer()) {
            return pieces;
        }
        try {
            connector.put(key, entryOf(pieces));
        } catch (IOException e) {
            throw new MediaInputException("the shared encoder cache cannot save item " + key + ": "
                    + e.getMessage(), e);
        }
        saves.incrementAndGet();
        return pieces;
    }

    /** The shared-cache entry of one item's pieces. */
    private static EncoderCacheEntry entryOf(List<MediaPiece> pieces) {
        List<EncoderCacheEntry.Chunk> chunks = new ArrayList<>(pieces.size());
        for (MediaPiece p : pieces) {
            switch (p.kind) {
                case TEXT:
                    chunks.add(EncoderCacheEntry.Chunk.text(p.text));
                    break;
                case IMAGE_ID:
                    chunks.add(EncoderCacheEntry.Chunk.imageId());
                    break;
                case ROWS:
                    chunks.add(EncoderCacheEntry.Chunk.rows(p.rows));
                    break;
                case GRID:
                    chunks.add(EncoderCacheEntry.Chunk.grid(p.rows, p.nx, p.ny, p.mask));
                    break;
                case LEAD_PADDED:
                    chunks.add(EncoderCacheEntry.Chunk.leadPadded(p.rows, p.lead));
                    break;
                case CANVAS:
                    chunks.add(EncoderCacheEntry.Chunk.canvas(p.rows, p.nx, p.ny, canvasTilesOf(p.canvas)));
                    break;
                default:
                    throw new IllegalStateException("unhandled media piece " + p.kind);
            }
        }
        return EncoderCacheEntry.of(chunks);
    }

    /** The pieces of a shared-cache entry, checked against this projector's embedding width. */
    private List<MediaPiece> piecesOf(EncoderCacheEntry entry, MediaKey key) {
        if (entry.width() != 0 && entry.width() != width) {
            throw new MediaInputException("the shared encoder cache holds item " + key + " at " + entry.width()
                    + " values per row while this projector writes " + width + "; the producer runs another"
                    + " projector");
        }
        List<MediaPiece> out = new ArrayList<>(entry.chunks().size());
        for (EncoderCacheEntry.Chunk chunk : entry.chunks()) {
            switch (chunk.kind()) {
                case TEXT:
                    out.add(MediaPiece.text(chunk.text()));
                    break;
                case IMAGE_ID:
                    out.add(MediaPiece.imageId());
                    break;
                case ROWS:
                    out.add(MediaPiece.rows(chunk.rows()));
                    break;
                case GRID:
                    out.add(chunk.mask() == null ? MediaPiece.grid(chunk.rows(), chunk.nx(), chunk.ny())
                            : MediaPiece.grid(chunk.rows(), chunk.nx(), chunk.ny(), chunk.mask()));
                    break;
                case LEAD_PADDED:
                    out.add(MediaPiece.leadPadded(chunk.rows(), chunk.lead()));
                    break;
                case CANVAS:
                    out.add(MediaPiece.canvas(chunk.rows(), chunk.nx(), chunk.ny(), canvasTiles(chunk.canvas())));
                    break;
                default:
                    throw new IllegalStateException("unhandled shared encoder cache chunk " + chunk.kind());
            }
        }
        return Collections.unmodifiableList(out);
    }

    private static List<EncoderCacheEntry.CanvasTile> canvasTilesOf(List<MediaRopePositions.CanvasTile> tiles) {
        List<EncoderCacheEntry.CanvasTile> out = new ArrayList<>(tiles.size());
        for (MediaRopePositions.CanvasTile tile : tiles) {
            out.add(new EncoderCacheEntry.CanvasTile(tile.tokensX(), tile.tokensY(), tile.separators()));
        }
        return out;
    }

    private static List<MediaRopePositions.CanvasTile> canvasTiles(List<EncoderCacheEntry.CanvasTile> tiles) {
        List<MediaRopePositions.CanvasTile> out = new ArrayList<>(tiles.size());
        for (EncoderCacheEntry.CanvasTile tile : tiles) {
            out.add(new MediaRopePositions.CanvasTile(tile.tokensX(), tile.tokensY(), tile.separators()));
        }
        return out;
    }
}
