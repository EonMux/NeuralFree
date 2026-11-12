package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.video.VideoPruner;

import java.util.ArrayList;
import java.util.List;

/**
 * The temporal groups of a video layout ({@link MediaContext}): the token pruning over their chunks and the copy of
 * a group's pieces into the layout. Stateless.
 */
final class VideoGroups {

    private VideoGroups() {
    }

    /**
     * The video token pruning over every group's chunk: a configured rate keeps {@code 1 - rate} of the video's
     * embedding rows (vLLM's EVS, which always keeps the first frame's rows, or VidCom2, which keeps at least one
     * row per frame). The groups must be single uniform chunks, the layout the temporal-patch and fixed-size
     * families produce; the others refuse pruning by name.
     */
    static void prune(List<List<MediaPiece>> groups, double rate, VideoPruner.Method method, String field) {
        if (rate <= 0 || groups.isEmpty()) {
            return;
        }
        List<MediaPiece> chunks = new ArrayList<>(groups.size());
        for (List<MediaPiece> group : groups) {
            MediaPiece chunk = singleGridChunk(group);
            if (chunk == null) {
                throw new MediaInputException(field + ": the video token pruning runs over single-chunk frames,"
                        + " and the projector's frame layout splits a frame over " + group.size() + " pieces");
            }
            chunks.add(chunk);
        }
        int rowsPerFrame = chunks.get(0).rows.length;
        for (MediaPiece chunk : chunks) {
            if (chunk.rows.length != rowsPerFrame) {
                throw new MediaInputException(field + ": the video token pruning needs uniform frames, and the"
                        + " frames encode to " + rowsPerFrame + " and " + chunk.rows.length + " rows");
            }
        }
        float[][] rows = new float[chunks.size() * rowsPerFrame][];
        for (int g = 0; g < chunks.size(); g++) {
            System.arraycopy(chunks.get(g).rows, 0, rows, g * rowsPerFrame, rowsPerFrame);
        }
        boolean[] mask = method == VideoPruner.Method.EVS
                ? VideoPruner.evsMask(rows, chunks.size(), rowsPerFrame, rate)
                : VideoPruner.vidcom2Mask(rows, chunks.size(), rowsPerFrame, rate);
        for (int g = 0; g < groups.size(); g++) {
            replaceChunk(groups.get(g), chunks.get(g).withMask(mask, g * rowsPerFrame, rowsPerFrame));
        }
    }

    /** The one grid chunk of a group, or null when the group is not a single chunk. */
    private static MediaPiece singleGridChunk(List<MediaPiece> group) {
        MediaPiece chunk = null;
        for (MediaPiece p : group) {
            if (p.kind == MediaPiece.Kind.GRID) {
                if (chunk != null) {
                    return null;
                }
                chunk = p;
            } else if (p.kind != MediaPiece.Kind.TEXT) {
                return null;
            }
        }
        return chunk;
    }

    /** Swaps the grid chunk of a group for the pruned one. */
    private static void replaceChunk(List<MediaPiece> group, MediaPiece pruned) {
        for (int i = 0; i < group.size(); i++) {
            if (group.get(i).kind == MediaPiece.Kind.GRID) {
                group.set(i, pruned);
                return;
            }
        }
    }

    /** A group's pieces, the chunk pruned to no rows dropped (its stamp text stays, as vLLM's does). */
    static void add(List<MediaPiece> out, List<MediaPiece> group) {
        for (MediaPiece p : group) {
            if (p.kind == MediaPiece.Kind.GRID && p.rows.length == 0) {
                continue;
            }
            out.add(p);
        }
    }
}
