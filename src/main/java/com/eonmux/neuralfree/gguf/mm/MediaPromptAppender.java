package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;

import java.util.List;

/**
 * Appends one encoded item's pieces to a prompt under its key (mtmd's tokenizer over the encoded item): marker
 * text through the text encoder, an image-id tag through it too, and every embedding chunk as media rows. Each
 * embedding chunk is piece {@code n} of the item (a sliced image's overview and slices, an audio clip's windows),
 * so every chunk's identity names exactly its own rows and one key names the whole item in the encoded-media
 * cache and the prefix cache alike. A block whose lead pads depend on the prompt before it
 * ({@link MediaMarkers#imageLeadPadAlign()}) takes the pads its place needs and is identified as
 * {@code key#n@leadK}, {@code K} its lead pad count; the prefix cache keys its rows by their content, since they
 * differ per pad count.
 */
final class MediaPromptAppender {

    private final MediaMarkers markers;
    private final boolean mrope;

    /** The appender of one projector family's markers over a text model that is M-RoPE when {@code mrope}. */
    MediaPromptAppender(MediaMarkers markers, boolean mrope) {
        this.markers = markers;
        this.mrope = mrope;
    }

    /**
     * Appends one item's pieces under its key.
     *
     * @param nonCausal whether the item's rows decode without the causal mask
     *                  ({@link MediaMarkers#nonCausalImage()})
     */
    void append(MediaPrompt.Builder prompt, MediaKey key, List<MediaPiece> pieces, int imageIndex,
                MediaContext.TextEncoder text, boolean nonCausal) {
        int run = 0;
        for (MediaPiece p : pieces) {
            if (p.kind == MediaPiece.Kind.TEXT) {
                if (!p.text.isEmpty()) {
                    prompt.appendText(text.encode(p.text));
                }
            } else if (p.kind == MediaPiece.Kind.IMAGE_ID) {
                prompt.appendText(text.encode("<image_id>" + imageIndex + "</image_id>"));
            } else if (p.kind == MediaPiece.Kind.LEAD_PADDED) {
                MediaPiece block = p.leadPadded(prompt.tokens(), markers.imageLeadPadAlign());
                prompt.addMedia(key + "#" + run++ + "@lead" + block.lead,
                        positions(block, prompt.position()), block.rows, nonCausal);
            } else {
                prompt.addMedia(key, run++, positions(p, prompt.position()), p.rows, nonCausal);
            }
        }
    }

    /**
     * The positions of a row piece at {@code pos0}: the 2D grid (zero-based for SenseNova-U1,
     * {@link MediaMarkers#zeroBasedImageGrid()}) or the canvas on M-RoPE, else one position per row. A pruned
     * grid takes its kept cells' positions.
     */
    private MediaRopePositions positions(MediaPiece p, int pos0) {
        if (p.kind == MediaPiece.Kind.CANVAS) {
            return MediaRopePositions.canvas(pos0, p.nx, p.ny, p.canvas);
        }
        if (!mrope) {
            return MediaRopePositions.normal(pos0, p.rows.length);
        }
        if (p.kind == MediaPiece.Kind.GRID) {
            if (p.mask != null) {
                return MediaRopePositions.image2dMasked(pos0, p.nx, p.ny, p.mask);
            }
            return markers.zeroBasedImageGrid() ? MediaRopePositions.image2dZeroBased(pos0, p.nx, p.ny)
                    : MediaRopePositions.image2d(pos0, p.nx, p.ny);
        }
        return MediaRopePositions.audio1d(pos0, p.rows.length);
    }
}
