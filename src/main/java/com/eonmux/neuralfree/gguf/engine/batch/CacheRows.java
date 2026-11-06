package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;

import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The keyed rows of a sequence as the prefix cache hashes, compares and stores them (vLLM's block-hash inputs, the
 * token ids and the extra keys of {@code generate_block_hash_extra_keys}): each row's token, the position its KV
 * cells store and, for a row without a token, its key — for a media row the identity of its media item
 * ({@link MediaChunk#item()}, vLLM's {@code mm_hash}: the client's uuid or a content digest, the key the
 * encoded-media cache uses too) with the chunk's piece of the item, its position layout and mask rule, and the
 * row's index in the chunk; for a prompt-embedding row the SHA-256 digest of the row itself. Two rows are equal
 * when token, key and index agree; the rows before them fix where they sit, so equal row prefixes write equal KV
 * rows. A media chunk's rows attend each other (a shared base position or the non-causal mask) and prefill as one
 * forward batch ({@link MediaChunk#isMediaItem()}, the rule the prefill's chunking follows too), so a reused
 * prefix never ends inside a media chunk ({@link #cut}); text and prompt-embedding rows split anywhere.
 *
 * <p>Immutable. A text-only view holds tokens alone: its rows store their own index as position and carry no
 * key.</p>
 */
final class CacheRows {

    /** The token of a row that holds none (a media or prompt-embedding row; llama.cpp's {@code LLAMA_TOKEN_NULL}). */
    static final int NO_TOKEN = -1;
    /** The item index of a row outside any media item (a text or prompt-embedding row). */
    static final int NO_ITEM = -1;

    private final int[] tokens;
    /** The stored position of every row; null when row {@code r} stores position {@code r}. */
    private final int[] positions;
    /** The key of every row without a token, null for token rows; null when every row holds a token. */
    private final String[] keys;
    /** The media item of every media row, null for the other rows; null with {@code keys}. */
    private final MediaKey[] media;
    /** The index of a media row in its item, {@link #NO_ITEM} for the other rows; null with {@code keys}. */
    private final int[] items;
    /** The position the row after the view stores (where fed decode rows continue). */
    private final int endPosition;
    /** The prompt chunks the rows came from and the first row of each, for the prompt-embedding digests. */
    private final List<MediaChunk> chunks;
    private final int[] chunkRows;

    private CacheRows(int[] tokens, int[] positions, String[] keys, MediaKey[] media, int[] items,
                      int endPosition, List<MediaChunk> chunks, int[] chunkRows) {
        this.tokens = tokens;
        this.positions = positions;
        this.keys = keys;
        this.media = media;
        this.items = items;
        this.endPosition = endPosition;
        this.chunks = chunks;
        this.chunkRows = chunkRows;
    }

    /** The view of a plain token run. */
    static CacheRows ofTokens(int[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        return new CacheRows(tokens.clone(), null, null, null, null, tokens.length, Collections.emptyList(),
                new int[0]);
    }

    /**
     * The view of a prompt's rows: a text chunk's tokens at consecutive positions, a media chunk's rows under its
     * item's identity at the positions its section 0 stores, a prompt-embedding run's rows under their own digests.
     */
    static CacheRows of(MediaPrompt prompt) {
        Objects.requireNonNull(prompt, "prompt");
        if (prompt.isTextOnly()) {
            return ofTokens(prompt.samplerTokens());
        }
        int n = prompt.nTokens();
        List<MediaChunk> chunks = prompt.chunks();
        Builder b = new Builder(n, chunks.size());
        for (MediaChunk c : chunks) {
            b.add(c);
        }
        return new CacheRows(b.tokens, b.positions, b.keys, b.media, b.items, prompt.nPositions(),
                chunks, b.chunkRows);
    }

    /** Fills the row arrays of a media prompt chunk by chunk. */
    private static final class Builder {
        final int[] tokens;
        final int[] positions;
        final String[] keys;
        final MediaKey[] media;
        final int[] items;
        final int[] chunkRows;
        int row;
        int pos;
        int chunk;

        Builder(int rows, int nChunks) {
            tokens = new int[rows];
            positions = new int[rows];
            keys = new String[rows];
            media = new MediaKey[rows];
            items = new int[rows];
            chunkRows = new int[nChunks];
        }

        void add(MediaChunk c) {
            chunkRows[chunk++] = row;
            int n = c.nTokens();
            if (c.isText()) {
                int[] t = c.tokens();
                for (int i = 0; i < n; i++) {
                    put(t[i], pos + i, null, null, NO_ITEM);
                }
            } else if (c.isPromptEmbeds()) {
                MessageDigest digest = MediaKey.sha256();
                for (int i = 0; i < n; i++) {
                    c.digestRows(digest, i, i + 1);
                    String key = MediaKey.hex(digest.digest());
                    put(NO_TOKEN, c.positions().position(0, i), key, null, NO_ITEM);
                }
            } else {
                String key = chunkKey(c);
                for (int i = 0; i < n; i++) {
                    put(NO_TOKEN, c.positions().position(0, i), key, c.item(), i);
                }
            }
            pos += c.nPositions();
        }

        private void put(int token, int position, String key, MediaKey item, int index) {
            tokens[row] = token;
            positions[row] = position;
            keys[row] = key;
            media[row] = item;
            items[row] = index;
            row++;
        }
    }

    /**
     * The key of a media chunk's rows: its item's identity, its piece of the item, its position layout (kind, grid,
     * rows) and its mask rule. The rows before the chunk fix its base position.
     */
    private static String chunkKey(MediaChunk c) {
        MediaRopePositions p = c.positions();
        return c.item() + "#" + c.piece() + "@" + p.kind() + ":" + p.nx() + "x" + p.ny() + ":" + p.nTokens()
                + (c.nonCausal() ? ":non-causal" : "");
    }

    /**
     * The view extended by the decode tokens {@code fed}, fed after the rows: token rows at the positions that
     * follow the view's last.
     */
    CacheRows withFed(int[] fed) {
        Objects.requireNonNull(fed, "fed");
        int n = tokens.length;
        int[] t = Arrays.copyOf(tokens, n + fed.length);
        System.arraycopy(fed, 0, t, n, fed.length);
        if (keys == null && positions == null) {
            return new CacheRows(t, null, null, null, null, t.length, chunks, chunkRows);
        }
        int[] p = Arrays.copyOf(positions, t.length);
        int[] it = Arrays.copyOf(items, t.length);
        for (int k = 0; k < fed.length; k++) {
            p[n + k] = endPosition + k;
            it[n + k] = NO_ITEM;
        }
        return new CacheRows(t, p, Arrays.copyOf(keys, t.length), Arrays.copyOf(media, t.length),
                it, endPosition + fed.length, chunks, chunkRows);
    }

    /**
     * Rows {@code [from, to)} as a view of their own (a cached block's keys): tokens, keys and item indexes, without
     * positions or source chunks.
     */
    CacheRows slice(int from, int to) {
        Objects.checkFromToIndex(from, to, tokens.length);
        int[] t = Arrays.copyOfRange(tokens, from, to);
        if (keys == null) {
            return new CacheRows(t, null, null, null, null, t.length, Collections.emptyList(), new int[0]);
        }
        return new CacheRows(t, null, Arrays.copyOfRange(keys, from, to), Arrays.copyOfRange(media, from, to),
                Arrays.copyOfRange(items, from, to), t.length, Collections.emptyList(), new int[0]);
    }

    /** Rows in the view. */
    int length() {
        return tokens.length;
    }

    /** Whether every row holds a token (a plain token run). */
    boolean tokensOnly() {
        return keys == null;
    }

    /** The tokens of rows {@code [from, to)}, {@link #NO_TOKEN} for a row without one (a copy). */
    int[] tokens(int from, int to) {
        return Arrays.copyOfRange(tokens, from, to);
    }

    /**
     * The position row {@code r} stores, or for {@code r == length()} the position the row after the view would
     * store: where a prefix of {@code r} rows ends in positions.
     */
    int positionAt(int r) {
        Objects.checkIndex(r, tokens.length + 1);
        if (r == tokens.length) {
            return endPosition;
        }
        return positions == null ? r : positions[r];
    }

    /** The stored positions of rows {@code [from, to)}, or null when every row stores its own index. */
    int[] positions(int from, int to) {
        return positions == null ? null : Arrays.copyOfRange(positions, from, to);
    }

    /**
     * The longest reuse at most {@code reuse} rows long that ends outside a media chunk: a prefix whose next row
     * continues a chunk begun before it backs up to the chunk's first row, as the chunk prefills whole.
     */
    int cut(int reuse) {
        if (items == null || reuse <= 0 || reuse >= tokens.length || items[reuse] <= 0) {
            return reuse;
        }
        return reuse - items[reuse];
    }

    /**
     * The common row prefix of {@code a[aFrom, ..)} and {@code b[bFrom, ..)}, at most {@code max} rows: rows agree
     * when their tokens, keys and item indexes do.
     */
    static int commonLength(CacheRows a, int aFrom, CacheRows b, int bFrom, int max) {
        int n = Math.min(max, Math.min(a.tokens.length - aFrom, b.tokens.length - bFrom));
        for (int i = 0; i < n; i++) {
            if (!sameRow(a, aFrom + i, b, bFrom + i)) {
                return i;
            }
        }
        return Math.max(n, 0);
    }

    private static boolean sameRow(CacheRows a, int i, CacheRows b, int j) {
        if (a.tokens[i] != b.tokens[j]) {
            return false;
        }
        String ka = a.keys == null ? null : a.keys[i];
        String kb = b.keys == null ? null : b.keys[j];
        if (ka == null || kb == null) {
            return ka == kb;
        }
        return a.items[i] == b.items[j] && ka.equals(kb);
    }

    /**
     * The extra keys of the block over rows {@code [from, to)} as KV events publish them, in vLLM's order: the
     * per-request adapters' content identity when the block starts a chain under adapters ({@code chainKey}, null
     * for a later block), every media chunk the block covers as its item's identity ({@link MediaKey#id()}, vLLM's
     * {@code mm_hash}) with the chunk's first row relative to {@code from}, the
     * cache salt when the block starts a salted chain, and the SHA-256 digest of the block's prompt-embedding rows'
     * bytes. Empty for a plain token block that starts no keyed chain.
     */
    List<BlockExtraKey> extraKeys(int from, int to, PrefixKey chainKey) {
        List<BlockExtraKey> out = new ArrayList<>(2);
        if (chainKey != null && chainKey.adapters() != null) {
            out.add(BlockExtraKey.lora(chainKey.adapters()));
        }
        if (keys != null) {
            for (int r = from; r < to; r++) {
                if (items[r] != NO_ITEM && (r == from || items[r] == 0)) {
                    out.add(BlockExtraKey.media(media[r].id(), r - items[r] - from));
                }
            }
        }
        if (chainKey != null && chainKey.salt() != null) {
            out.add(BlockExtraKey.salt(chainKey.salt()));
        }
        String embeds = embedsDigest(from, to);
        if (embeds != null) {
            out.add(BlockExtraKey.promptEmbeds(embeds));
        }
        return out;
    }

    /** The SHA-256 digest of the prompt-embedding rows among rows {@code [from, to)}, or null when there are none. */
    private String embedsDigest(int from, int to) {
        MessageDigest digest = null;
        for (int c = 0; c < chunks.size(); c++) {
            MediaChunk chunk = chunks.get(c);
            int start = chunkRows[c];
            int lo = Math.max(from, start);
            int hi = Math.min(to, start + chunk.nTokens());
            if (chunk.isPromptEmbeds() && lo < hi) {
                if (digest == null) {
                    digest = MediaKey.sha256();
                }
                chunk.digestRows(digest, lo - start, hi - start);
            }
        }
        return digest == null ? null : MediaKey.hex(digest.digest());
    }

    @Override
    public String toString() {
        return "CacheRows[" + tokens.length + " rows" + (keys == null ? "" : ", keyed") + "]";
    }
}
