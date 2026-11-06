package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * A prompt of text and media chunks (llama.cpp server's {@code server_tokens}): the immutable chunk sequence a
 * media-aware submission runs. Its {@link #nTokens()} rows are the KV cells the prompt occupies, while its
 * {@link #nPositions()} positions are what the sequence's position cursor advances by — the two differ when a
 * media chunk takes fewer positions than rows (an M-RoPE image grid).
 *
 * <p>Built with a builder that appends text, media and prompt-embedding chunks; a prompt holds at least one chunk
 * and one row, and its media rows share one width. {@link #isTextOnly()} is true when no chunk carries media: the
 * prompt is then a plain token run. The prefix cache serves both kinds, keying a media prompt's rows by the
 * identities of its items ({@link MediaChunk#item()}).</p>
 */
public final class MediaPrompt {

    private final List<MediaChunk> chunks;
    private final int nTokens;
    private final int nPositions;
    private final int nMediaChunks;
    private final int mediaWidth; // width of the media rows, -1 when the prompt is text only
    private final int[] checkpointRows;

    private MediaPrompt(List<MediaChunk> chunks, int nTokens, int nPositions, int nMediaChunks, int mediaWidth,
                        int[] checkpointRows) {
        this.chunks = chunks;
        this.nTokens = nTokens;
        this.nPositions = nPositions;
        this.nMediaChunks = nMediaChunks;
        this.mediaWidth = mediaWidth;
        this.checkpointRows = checkpointRows;
    }

    /** A prompt of one text chunk. */
    public static MediaPrompt ofText(int[] tokens) {
        return builder().addText(tokens).build();
    }

    /** A fresh builder. */
    public static Builder builder() {
        return new Builder();
    }

    /** The chunks, in prompt order. */
    public List<MediaChunk> chunks() {
        return chunks;
    }

    /** KV rows the prompt occupies ({@code mtmd_helper_get_n_tokens}). */
    public int nTokens() {
        return nTokens;
    }

    /** Positions the prompt advances the sequence by ({@code mtmd_helper_get_n_pos}). */
    public int nPositions() {
        return nPositions;
    }

    /** Media chunks the prompt holds. */
    public int nMediaChunks() {
        return nMediaChunks;
    }

    /** True when no chunk carries media: the prompt is a plain token run. */
    public boolean isTextOnly() {
        return nMediaChunks == 0;
    }

    /**
     * Width of the media rows, or -1 for a text-only prompt.
     */
    public int mediaWidth() {
        return mediaWidth;
    }

    /**
     * Prompt rows the prefill takes a context checkpoint at (chat-message boundaries the renderer named, llama.cpp's
     * message spans), ascending; empty when the prompt names none. The batch engine's checkpointed prefix cache
     * captures a checkpoint at each row on the prefill's way, so a later prompt that branches the conversation
     * before this prompt's end restores there.
     */
    public int[] checkpointRows() {
        return checkpointRows.clone();
    }

    /**
     * The prompt's real token ids in order, media rows skipped (the repetition-penalty and prompt-render view;
     * media rows hold no token, as llama.cpp's {@code LLAMA_TOKEN_NULL} cells do).
     */
    public int[] samplerTokens() {
        int[] out = new int[nTokens - countMediaRows()];
        int w = 0;
        for (MediaChunk c : chunks) {
            if (c.isText()) {
                int[] tokens = c.tokens();
                for (int t : tokens) {
                    out[w++] = t;
                }
            }
        }
        return out;
    }

    private int countMediaRows() {
        int n = 0;
        for (MediaChunk c : chunks) {
            if (!c.isText()) {
                n += c.nTokens();
            }
        }
        return n;
    }

    /**
     * The prompt row of every entry of {@link #samplerTokens()}: row {@code i} of the returned array is the KV row
     * that holds sampler token {@code i}, media rows counted in the numbering (vLLM counts a media placeholder as
     * one token position of {@code prompt_token_ids}). The rows between two entries hold media or prompt-embedding
     * rows, which carry no token id.
     */
    public int[] samplerTokenRows() {
        int[] out = new int[nTokens - countMediaRows()];
        int w = 0;
        int row = 0;
        for (MediaChunk c : chunks) {
            if (c.isText()) {
                for (int i = 0; i < c.nTokens(); i++) {
                    out[w++] = row++;
                }
            } else {
                row += c.nTokens();
            }
        }
        return out;
    }

    /**
     * A prompt whose identified media chunks hold the rows {@code shared} returns for their id — the canonical
     * encoder output an engine's encoder cache keeps — with the chunks' kinds, positions and mask rules and every
     * unidentified or text chunk unchanged. The builder's width rule re-checks against the shared rows.
     */
    public MediaPrompt withSharedMedia(BiFunction<String, float[][], float[][]> shared) {
        Objects.requireNonNull(shared, "shared");
        Builder b = null;
        for (int i = 0; i < chunks.size(); i++) {
            MediaChunk c = chunks.get(i);
            float[][] canonical = c.isText() || c.id() == null ? null : shared.apply(c.id(), c.sharedRows());
            if (canonical == null || canonical == c.sharedRows()) {
                if (b != null) {
                    b.add(c);
                }
                continue;
            }
            if (b == null) {
                b = builder();
                for (int j = 0; j < i; j++) {
                    b.add(chunks.get(j));
                }
            }
            b.add(c.item() == null ? MediaChunk.media(c.id(), c.positions(), canonical, c.nonCausal())
                    : MediaChunk.media(c.item(), c.piece(), c.positions(), canonical, c.nonCausal()));
        }
        if (b == null) {
            return this;
        }
        for (int row : checkpointRows) {
            b.checkpointRow(row);
        }
        return b.build();
    }

    @Override
    public String toString() {
        return "MediaPrompt{" + chunks.size() + " chunks, " + nMediaChunks + " media, rows=" + nTokens
                + ", pos=" + nPositions + "}";
    }

    /** Accumulates chunks and builds the validated prompt. */
    public static final class Builder {

        private final List<MediaChunk> chunks = new ArrayList<>();
        private final List<Integer> checkpointRows = new ArrayList<>();
        private int mediaWidth = -1;

        /**
         * Names prompt row {@code row} a checkpoint row: the prefill takes a context checkpoint there (a chat
         * message's start the renderer named). Every row must lie inside the prompt and be named once.
         *
         * @throws IllegalArgumentException for a row outside {@code (0, rows)} when the prompt builds, or a row
         *                                  named twice
         */
        public Builder checkpointRow(int row) {
            if (row <= 0 || checkpointRows.contains(row)) {
                throw new IllegalArgumentException("a checkpoint row must be a positive interior prompt row named"
                        + " once, got " + row);
            }
            checkpointRows.add(row);
            return this;
        }

        /**
         * Appends the text tokens as one chunk.
         *
         * @throws IllegalArgumentException when the array is empty or holds a negative token id
         */
        public Builder addText(int[] tokens) {
            chunks.add(MediaChunk.text(tokens));
            return this;
        }

        /**
         * Appends text tokens the way llama.cpp's {@code mtmd_tokenizer::add_text} does: an empty array adds
         * nothing, and tokens after a text chunk extend that chunk instead of starting a new one.
         *
         * @throws IllegalArgumentException when a token id is negative
         */
        public Builder appendText(int[] tokens) {
            if (tokens == null || tokens.length == 0) {
                return this;
            }
            int last = chunks.size() - 1;
            if (last >= 0 && chunks.get(last).isText()) {
                int[] head = chunks.get(last).tokens();
                int[] merged = new int[head.length + tokens.length];
                System.arraycopy(head, 0, merged, 0, head.length);
                System.arraycopy(tokens, 0, merged, head.length, tokens.length);
                chunks.set(last, MediaChunk.text(merged));
            } else {
                chunks.add(MediaChunk.text(tokens));
            }
            return this;
        }

        /** The position the next chunk starts at: the positions of every chunk added so far. */
        public int position() {
            int pos = 0;
            for (MediaChunk c : chunks) {
                pos += c.nPositions();
            }
            return pos;
        }

        /**
         * The KV rows of every chunk added so far: text tokens and embedding rows ({@code n_past} of the chunks
         * before the next one, as {@code mtmd_tokenize} counts it).
         */
        public int tokens() {
            int n = 0;
            for (MediaChunk c : chunks) {
                n += c.nTokens();
            }
            return n;
        }

        /** Whether no chunk was added yet. */
        public boolean isEmpty() {
            return chunks.isEmpty();
        }

        /**
         * Appends a media chunk of embedding rows with their positions, the rows their own item
         * ({@link MediaChunk#media(MediaRopePositions, float[][], boolean)}).
         *
         * @throws IllegalArgumentException when the row width differs from the media rows added before
         */
        public Builder addMedia(MediaRopePositions positions, float[][] rows, boolean nonCausal) {
            return addRows(MediaChunk.media(positions, rows, nonCausal));
        }

        /**
         * Appends piece {@code piece} of the media item {@code item}: embedding rows with their positions.
         *
         * @throws IllegalArgumentException when the row width differs from the media rows added before
         */
        public Builder addMedia(MediaKey item, int piece, MediaRopePositions positions, float[][] rows,
                                boolean nonCausal) {
            return addRows(MediaChunk.media(item, piece, positions, rows, nonCausal));
        }

        /**
         * Appends an identified media chunk of embedding rows ({@link MediaChunk#media(String,
         * MediaRopePositions, float[][], boolean)}): the rows are held as given and {@code id} is what an
         * engine's encoder cache keys the shared encoder output by.
         *
         * @throws IllegalArgumentException when the id is blank or the row width differs from the media rows
         *                                  added before
         */
        public Builder addMedia(String id, MediaRopePositions positions, float[][] rows, boolean nonCausal) {
            return addRows(MediaChunk.media(id, positions, rows, nonCausal));
        }

        /**
         * Appends a run of prompt-embedding rows at the consecutive positions {@code pos0 ..}
         * ({@link MediaChunk#embeds}).
         *
         * @throws IllegalArgumentException when the row width differs from the media rows added before
         */
        public Builder addEmbeds(int pos0, float[][] rows) {
            return addRows(MediaChunk.embeds(pos0, rows));
        }

        /**
         * Appends a built chunk as-is, keeping its kind, rows and identity; an embedding chunk's row width must
         * match the media rows added before.
         *
         * @throws IllegalArgumentException when an embedding chunk's row width differs from the media rows added
         *                                  before
         */
        public Builder add(MediaChunk chunk) {
            Objects.requireNonNull(chunk, "chunk");
            if (chunk.isText()) {
                chunks.add(chunk);
                return this;
            }
            return addRows(chunk);
        }

        private Builder addRows(MediaChunk chunk) {
            int width = chunk.width();
            if (mediaWidth >= 0 && width != mediaWidth) {
                throw new IllegalArgumentException("media rows share one width: a chunk of " + width
                        + " after chunks of " + mediaWidth);
            }
            mediaWidth = width;
            chunks.add(chunk);
            return this;
        }

        /**
         * Builds the prompt.
         *
         * @throws IllegalStateException  when no chunk was added
         * @throws IllegalArgumentException when a named checkpoint row does not lie inside the prompt
         */
        public MediaPrompt build() {
            if (chunks.isEmpty()) {
                throw new IllegalStateException("a prompt holds at least one chunk");
            }
            int rows = 0;
            int pos = 0;
            int media = 0;
            for (MediaChunk c : chunks) {
                rows += c.nTokens();
                pos += c.nPositions();
                media += c.isText() ? 0 : 1;
            }
            int[] boundaries = sortedRows(checkpointRows, rows);
            return new MediaPrompt(Collections.unmodifiableList(new ArrayList<>(chunks)), rows, pos, media,
                    mediaWidth, boundaries);
        }

        /** The named checkpoint rows, deduplicated, ascending, each an interior prompt row. */
        private static int[] sortedRows(List<Integer> named, int rows) {
            if (named.isEmpty()) {
                return new int[0];
            }
            int[] out = named.stream().mapToInt(Integer::intValue).distinct().sorted().toArray();
            for (int row : out) {
                if (row >= rows) {
                    throw new IllegalArgumentException("checkpoint row " + row + " lies outside the prompt's "
                            + rows + " rows");
                }
            }
            return out;
        }
    }
}
