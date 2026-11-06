package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;

import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Objects;

/**
 * One input chunk of a media prompt (llama.cpp {@code mtmd_input_chunk}): a run of text tokens or a run of media
 * embedding rows with their M-RoPE positions. A text chunk takes one KV row and one position per token; a media
 * chunk takes one KV row per embedding but can take fewer positions than rows (an image grid of
 * {@link MediaRopePositions#nTokens()} rows occupies {@link MediaRopePositions#nPositions()} positions), which is
 * what decouples the position cursor from the stored rows.
 *
 * <p>A media chunk also says whether its rows attend each other beyond the position rule
 * ({@code mtmd_decode_use_non_causal}): a non-causal chunk runs its rows without the causal mask, as llama.cpp
 * decodes a Gemma-style image chunk with {@code llama_set_causal_attn(false)}.</p>
 *
 * <p>A run of precomputed prompt-embedding rows ({@link #embeds}, vLLM's {@code prompt_embeds}) is a causal
 * embedding chunk of one position per row whose rows are independent prompt rows rather than one media item: each
 * row's KV depends only on the rows before it, so the prefix cache keys such rows one by one and a prefill may
 * resume inside the run, while a media item ({@link #media}) is cached and prefilled as a whole.</p>
 *
 * <p>A media chunk belongs to one media item, named by the item's {@link MediaKey} (vLLM's {@code mm_hash}: the
 * client's uuid, or the digest of the item's bytes or embedding rows), and is the item's {@link #piece()}-th run of
 * rows (a tiled image's overview and slices are separate pieces of one item). The key is the one identity the
 * encoded-media cache and the prefix cache share, so an item has one name everywhere. Media built without an item
 * key ({@link #media(MediaRopePositions, float[][], boolean)}, an {@code image_embeds} part) are their own item,
 * keyed by a digest of their rows.</p>
 *
 * <p>A media chunk may also carry an identifier ({@link #id()}): the item key and the piece, or the identity an
 * engine's encoder cache keys its shared encoder output by, wherever the medium sits in the prompt. An identified
 * chunk holds its rows without copying (see {@link #media(String, MediaRopePositions, float[][], boolean)}); its
 * {@link #embeddings()} still returns copies.</p>
 *
 * <p>Immutable; arrays are copied at construction (except an identified chunk's rows) and accessors return
 * copies.</p>
 */
public final class MediaChunk {

    private final int[] tokens;        // text chunk; null for a media chunk
    private final float[][] embd;      // media chunk rows; null for a text chunk
    private final MediaRopePositions positions; // media chunk positions; null for a text chunk
    private final boolean nonCausal;
    private final boolean promptEmbeds;
    private final String id;           // media identity, or null for text, prompt embeddings or unidentified media
    private final MediaKey item;       // the media item; null for text and prompt embeddings
    private final int piece;           // the chunk's index among its item's row runs

    private MediaChunk(int[] tokens, float[][] embd, MediaRopePositions positions, boolean nonCausal,
                       boolean promptEmbeds, String id, MediaKey item, int piece) {
        this.tokens = tokens;
        this.embd = embd;
        this.positions = positions;
        this.nonCausal = nonCausal;
        this.promptEmbeds = promptEmbeds;
        this.id = id;
        this.item = item;
        this.piece = piece;
    }

    /**
     * A chunk of text tokens.
     *
     * @throws IllegalArgumentException when the array is empty or holds a negative token id
     */
    public static MediaChunk text(int[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("a text chunk holds at least one token");
        }
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] < 0) {
                throw new IllegalArgumentException("token " + i + " has negative id " + tokens[i]);
            }
        }
        return new MediaChunk(tokens.clone(), null, null, false, false, null, null, 0);
    }

    /**
     * A chunk of media embedding rows that are their own media item ({@link MediaKey#ofRows}): one row per
     * {@link MediaRopePositions#nTokens()}, every row the same width, attending each other without the causal mask
     * when {@code nonCausal} is set.
     *
     * @throws IllegalArgumentException when a row is empty, a row is null or the row count differs from the
     *                                  positions' row count
     */
    public static MediaChunk media(MediaRopePositions positions, float[][] rows, boolean nonCausal) {
        float[][] copy = checkedRows(positions, rows);
        return new MediaChunk(null, copy, positions, nonCausal, false, null, MediaKey.ofRows(copy), 0);
    }

    /**
     * {@link #media(MediaRopePositions, float[][], boolean)} carrying the identity {@code id}: the digest the
     * projector computed for the medium's bytes, which an encoder cache keys its shared encoder output by. The
     * rows are held as given, not copied: the caller hands their ownership over (an encoder cache stores the
     * arrays as its canonical encoder output and other sequences' chunks reference them), so the arrays must not
     * change after the chunk is built. The rows are the chunk's item ({@link MediaKey#ofRows}), so the prefix
     * cache keys them by their content whatever identity carries them.
     *
     * @throws IllegalArgumentException when {@code id} is blank, a row is empty, a row is null or the row count
     *                                  differs from the positions' row count
     */
    public static MediaChunk media(String id, MediaRopePositions positions, float[][] rows, boolean nonCausal) {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("an identified media chunk needs a non-blank id");
        }
        requireRows(positions, rows);
        return new MediaChunk(null, rows, positions, nonCausal, false, id, MediaKey.ofRows(rows), 0);
    }

    /**
     * Piece {@code piece} of the media item {@code item}: embedding rows as
     * {@link #media(String, MediaRopePositions, float[][], boolean)} takes them, held as given, with the item key
     * and piece set ({@link #item()}, {@link #piece()}) and the identity {@code item + "#" + piece}.
     *
     * @throws IllegalArgumentException when the piece index is negative, a row is empty, a row is null or the row
     *                                  count differs from the positions' row count
     */
    public static MediaChunk media(MediaKey item, int piece, MediaRopePositions positions, float[][] rows,
                                   boolean nonCausal) {
        Objects.requireNonNull(item, "item");
        requirePiece(piece);
        requireRows(positions, rows);
        return new MediaChunk(null, rows, positions, nonCausal, false, item + "#" + piece, item, piece);
    }

    /**
     * A run of prompt-embedding rows at the consecutive positions {@code pos0 ..}, attending causally (vLLM's
     * {@code prompt_embeds}, llama.cpp's {@code set_position_normal}).
     *
     * @throws IllegalArgumentException when the run is empty, a row is null or empty, the rows differ in width,
     *                                  or {@code pos0} is negative
     */
    public static MediaChunk embeds(int pos0, float[][] rows) {
        Objects.requireNonNull(rows, "rows");
        MediaRopePositions positions = MediaRopePositions.normal(pos0, rows.length);
        requireRows(positions, rows);
        return new MediaChunk(null, copyRows(rows), positions, false, true, null, null, 0);
    }

    /** A copy of every row of {@code rows}. */
    private static float[][] copyRows(float[][] rows) {
        float[][] copy = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            copy[i] = rows[i].clone();
        }
        return copy;
    }

    private static void requirePiece(int piece) {
        if (piece < 0) {
            throw new IllegalArgumentException("a media item's piece index is not negative, got " + piece);
        }
    }

    /**
     * A validated copy of an embedding chunk's rows: one per position row, non-empty, of one width.
     */
    private static float[][] checkedRows(MediaRopePositions positions, float[][] rows) {
        requireRows(positions, rows);
        return copyRows(rows);
    }

    /**
     * The checks every embedding-chunk factory runs before the rows are stored: one row per position row, every
     * row non-empty and of one width.
     */
    private static void requireRows(MediaRopePositions positions, float[][] rows) {
        Objects.requireNonNull(positions, "positions");
        Objects.requireNonNull(rows, "rows");
        if (rows.length == 0) {
            throw new IllegalArgumentException("a media chunk holds at least one row");
        }
        if (rows.length != positions.nTokens()) {
            throw new IllegalArgumentException("the chunk holds " + rows.length + " rows, its positions cover "
                    + positions.nTokens());
        }
        int width = rows[0].length;
        if (width <= 0) {
            throw new IllegalArgumentException("media rows hold " + width + " values");
        }
        for (int i = 0; i < rows.length; i++) {
            Objects.requireNonNull(rows[i], "row " + i);
            if (rows[i].length != width) {
                throw new IllegalArgumentException("media rows share one width: row " + i + " holds "
                        + rows[i].length + " values after " + width);
            }
        }
    }

    /** True when the chunk is text, false when it carries media embeddings. */
    public boolean isText() {
        return tokens != null;
    }

    /**
     * The text tokens (a copy).
     *
     * @throws IllegalStateException when the chunk is a media chunk
     */
    public int[] tokens() {
        if (tokens == null) {
            throw new IllegalStateException("a media chunk holds no tokens");
        }
        return tokens.clone();
    }

    /**
     * The media rows (row copies).
     *
     * @throws IllegalStateException when the chunk is a text chunk
     */
    public float[][] embeddings() {
        if (embd == null) {
            throw new IllegalStateException("a text chunk holds no media rows");
        }
        float[][] out = new float[embd.length][];
        for (int i = 0; i < embd.length; i++) {
            out[i] = embd[i].clone();
        }
        return out;
    }

    /**
     * The media positions.
     *
     * @throws IllegalStateException when the chunk is a text chunk
     */
    public MediaRopePositions positions() {
        if (positions == null) {
            throw new IllegalStateException("a text chunk holds no media positions");
        }
        return positions;
    }

    /** KV rows the chunk occupies: one per token or per media embedding. */
    public int nTokens() {
        return tokens != null ? tokens.length : embd.length;
    }

    /** Positions the chunk advances the sequence by ({@code mtmd_input_chunk_get_n_pos}). */
    public int nPositions() {
        return tokens != null ? tokens.length : positions.nPositions();
    }

    /** The width of the embedding rows; 0 for a text chunk. */
    public int width() {
        return tokens != null ? 0 : embd[0].length;
    }

    /** Whether the chunk's rows attend each other without the causal mask. */
    public boolean nonCausal() {
        return nonCausal;
    }

    /**
     * Whether the chunk is one media item ({@link #media}): its rows attend each other (a shared base position or
     * the non-causal mask), so it prefills as one forward batch, a reused prefix never ends inside it and it takes
     * the encoder budget as a whole. Text and prompt-embedding rows are independent and split anywhere.
     */
    public boolean isMediaItem() {
        return embd != null && !promptEmbeds;
    }

    /** Whether the chunk is a run of prompt-embedding rows ({@link #embeds}) rather than text or a media item. */
    public boolean isPromptEmbeds() {
        return promptEmbeds;
    }

    /**
     * Feeds rows {@code [from, to)} to {@code digest} as little-endian IEEE-754 binary32 bytes, row after row (the
     * byte image vLLM hashes a block's {@code prompt_embeds} rows over, {@link MediaKey#feedRows}).
     *
     * @throws IllegalStateException     when the chunk is a text chunk
     * @throws IndexOutOfBoundsException when the range is outside the rows
     */
    public void digestRows(MessageDigest digest, int from, int to) {
        Objects.requireNonNull(digest, "digest");
        if (embd == null) {
            throw new IllegalStateException("a text chunk holds no media rows");
        }
        MediaKey.feedRows(digest, embd, from, to);
    }

    /**
     * The chunk's media identity (the item key and piece, or the projector's content digest), or null when the
     * chunk is text or unidentified.
     */
    public String id() {
        return id;
    }

    /**
     * The media item the chunk belongs to (vLLM's {@code mm_hash}), which keys its rows in the prefix cache.
     *
     * @throws IllegalStateException when the chunk is text or a prompt-embedding run
     */
    public MediaKey item() {
        if (item == null) {
            throw new IllegalStateException("only a media item's chunk carries the item key");
        }
        return item;
    }

    /** The chunk's index among its media item's row runs (0 for an item of one run); 0 for other chunks. */
    public int piece() {
        return piece;
    }

    /**
     * The media rows as held, without copying: the caller may read them but must not change them (an identified
     * chunk's arrays are the encoder cache's canonical output, shared with other sequences).
     *
     * @throws IllegalStateException when the chunk is a text chunk
     */
    public float[][] sharedRows() {
        if (embd == null) {
            throw new IllegalStateException("a text chunk holds no media rows");
        }
        return embd;
    }

    @Override
    public String toString() {
        if (isText()) {
            return "MediaChunk{text " + tokens.length + "}";
        }
        String head = promptEmbeds ? "embeds " : item != null ? "media " + item + "#" + piece + ", " : "media ";
        return "MediaChunk{" + head + embd.length + " rows, " + positions + (nonCausal ? ", non-causal" : "")
                + (id != null && item == null ? ", id " + id : "") + "}";
    }

    /** Two chunks hold the same kind, contents and identity. */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof MediaChunk)) {
            return false;
        }
        MediaChunk other = (MediaChunk) o;
        return nonCausal == other.nonCausal && promptEmbeds == other.promptEmbeds
                && Arrays.equals(tokens, other.tokens)
                && Objects.equals(positions, other.positions)
                && Objects.equals(id, other.id)
                && Arrays.deepEquals(embd, other.embd);
    }

    @Override
    public int hashCode() {
        int h = Objects.hash(nonCausal, promptEmbeds, positions, id);
        h = 31 * h + Arrays.hashCode(tokens);
        h = 31 * h + Arrays.deepHashCode(embd);
        return h;
    }
}
