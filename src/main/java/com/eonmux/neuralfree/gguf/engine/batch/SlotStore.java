package com.eonmux.neuralfree.gguf.engine.batch;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The llama.cpp slot actions ({@code /slots/{id}?action=save|restore|erase}) over the batch engine's paged memory. A
 * slot's state is the history its last sequence left in the prefix cache when it retired ({@link History}, recorded
 * by {@link BatchState#release}): the cached blocks that prefix walks are the slot's KV rows, wherever they live.
 *
 * <ul>
 *   <li>Save writes the chain the history still finds cached — whole blocks, then the shared rows of the block it
 *       ends inside, from the memory or the host-RAM tier — as a {@link SlotFile}, reading every row through the
 *       memory's host transfer.</li>
 *   <li>Restore installs the file's blocks into the prefix cache (fresh pages, evicting free cached blocks for room)
 *       and makes them the slot's history: the next request whose prompt starts with those tokens adopts them, as
 *       llama.cpp's restored slot reuses its prompt.</li>
 *   <li>Erase forgets the history and evicts the blocks only it used, in the memory and in the host-RAM tier.</li>
 * </ul>
 *
 * A slot of a running sequence refuses (the server waits for an idle engine first, as llama.cpp defers the task).
 * Without a prefix cache, or over a memory whose blocks do not carry a sequence's whole state
 * ({@link BlockLayout#refusal}), every action refuses by name. Confined to the engine's thread.
 */
final class SlotStore {

    /**
     * The history a slot's last sequence left: its committed rows' keys, how many of them, and its chain key.
     * Immutable.
     */
    static final class History {
        final CacheRows rows;
        final int count;
        final PrefixKey key;

        History(CacheRows rows, int count, PrefixKey key) {
            this.rows = Objects.requireNonNull(rows, "rows");
            this.key = Objects.requireNonNull(key, "key");
            if (count < 1 || count > rows.length()) {
                throw new IllegalArgumentException("a history of " + count + " of " + rows.length() + " rows");
            }
            this.count = count;
        }
    }

    private final BatchState state;
    private final BlockLayout layout;
    /** Why slot actions refuse on this engine, or null when they run. */
    private final String refusal;

    /**
     * The slot actions over {@code state}'s prefix cache, moving blocks through {@code layout}; {@code refusal} names
     * why every action refuses (no prefix cache, or blocks that cannot leave the memory), null when they run.
     *
     * @throws IllegalArgumentException when neither a refusal nor a layout is given
     */
    SlotStore(BatchState state, String refusal, BlockLayout layout) {
        this.state = Objects.requireNonNull(state, "state");
        if (refusal == null && (layout == null || state.cache == null)) {
            throw new IllegalArgumentException("slot actions run over a prefix cache and a block layout");
        }
        this.refusal = refusal;
        this.layout = refusal == null ? layout : null;
    }

    /** The layout cached blocks move through, or null when slot actions refuse. */
    BlockLayout layout() {
        return layout;
    }

    /**
     * Writes slot {@code seq}'s cached history to {@code out}.
     *
     * @throws UnsupportedOperationException when slot actions refuse on this engine, or the history holds media rows
     * @throws IllegalStateException         when the slot runs a sequence
     */
    SlotResult save(int seq, Path out) throws IOException {
        check(seq, "save");
        Chain chain = chain(seq);
        SlotFile.write(out, layout, chain.key, chain.images);
        return new SlotResult(chain.tokens, Files.size(out));
    }

    /**
     * The slot's cached history as the bytes of a slot file, for a KV transfer connector
     * ({@link com.eonmux.neuralfree.gguf.engine.kvtransfer.KvConnector}): the same format {@link #save} writes.
     *
     * @throws UnsupportedOperationException when slot actions refuse on this engine, or the history holds media rows
     * @throws IllegalStateException         when the slot runs a sequence
     */
    byte[] saveBytes(int seq) throws IOException {
        check(seq, "save");
        Chain chain = chain(seq);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        SlotFile.write(out, layout, chain.key, chain.images);
        return out.toByteArray();
    }

    /** The chain a slot's history still finds cached, and its row count. */
    private Chain chain(int seq) {
        History h = state.histories[seq];
        List<BlockImage> images = new ArrayList<>();
        PrefixKey key = PrefixKey.NONE;
        int tokens = 0;
        if (h != null) {
            key = h.key;
            images.addAll(state.cache.images(h.rows, h.key, h.count));
            for (BlockImage image : images) {
                tokens += image.rows();
            }
        }
        return new Chain(key, images, tokens);
    }

    /** A slot's cached chain: its extra keys, block images and rows. */
    private static final class Chain {
        final PrefixKey key;
        final List<BlockImage> images;
        final int tokens;

        Chain(PrefixKey key, List<BlockImage> images, int tokens) {
            this.key = key;
            this.images = images;
            this.tokens = tokens;
        }
    }

    /**
     * Installs the slot file at {@code in} into the prefix cache as slot {@code seq}'s history.
     *
     * @throws UnsupportedOperationException when slot actions refuse on this engine
     * @throws IllegalStateException         when the slot runs a sequence, or no cached block can be evicted to
     *                                       make room
     * @throws IOException                   for an unreadable file, another layout's file or a malformed chain
     */
    SlotResult restore(int seq, Path in) throws IOException {
        check(seq, "restore");
        SlotFile.Contents contents = SlotFile.read(in, layout);
        int tokens = install(seq, contents, in.toString());
        return new SlotResult(tokens, Files.size(in));
    }

    /**
     * Installs a transferred chain (the bytes of a slot file, from a
     * {@link com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferPayload}) into the prefix cache as slot
     * {@code seq}'s history: its blocks take fresh pages, evicting free cached blocks for room, so the next request
     * whose prompt starts with the chain's tokens adopts its rows.
     *
     * @return the tokens the chain restored
     * @throws UnsupportedOperationException when slot actions refuse on this engine
     * @throws IllegalStateException         when the slot runs a sequence, or no cached block can be evicted to
     *                                       make room
     * @throws IOException                   for a malformed or foreign chain
     */
    int restoreBytes(int seq, byte[] chain) throws IOException {
        check(seq, "restore");
        SlotFile.Contents contents = SlotFile.read(new ByteArrayInputStream(chain),
                "the transferred chain of slot " + seq, layout);
        return install(seq, contents, "the transferred chain of slot " + seq);
    }

    /** Installs {@code contents} into slot {@code seq}'s prefix cache and history; returns the rows. */
    private int install(int seq, SlotFile.Contents contents, String source) throws IOException {
        checkOffsets(contents, source);
        if (contents.tokens() > state.nCtx) {
            throw new IOException(source + " holds " + contents.tokens() + " tokens, more than the slot context of "
                    + state.nCtx);
        }
        Set<PrefixCache.Block> keep = Collections.newSetFromMap(new IdentityHashMap<>());
        PrefixCache.Block parent = null;
        int[] tokens = new int[contents.tokens()];
        int at = 0;
        for (BlockImage image : contents.blocks) {
            parent = state.cache.install(parent, image, contents.key, layout, keep);
            keep.add(parent);
            System.arraycopy(image.tokens(), 0, tokens, at, image.rows());
            at += image.rows();
        }
        state.histories[seq] = tokens.length == 0 ? null
                : new History(CacheRows.ofTokens(tokens), tokens.length, contents.key);
        state.cache.trimToBudget();
        return tokens.length;
    }

    /**
     * Forgets slot {@code seq}'s history and evicts the cached blocks no other slot's history walks, answering the
     * tokens the history held (llama.cpp's {@code n_erased}).
     *
     * @throws UnsupportedOperationException when slot actions refuse on this engine
     * @throws IllegalStateException         when the slot runs a sequence
     */
    int erase(int seq) {
        check(seq, "erase");
        History h = state.histories[seq];
        state.histories[seq] = null;
        if (h == null) {
            return 0;
        }
        PrefixCache.Match m = state.cache.match(h.rows, h.key, h.count);
        Set<PrefixCache.Block> spare = Collections.newSetFromMap(new IdentityHashMap<>());
        for (History other : state.histories) {
            if (other != null) {
                spare.addAll(state.cache.match(other.rows, other.key, other.count).held());
            }
        }
        PrefixCache.Block last = null;
        if (m.partial != null) {
            // a longer block the history only shares the start of belongs to another history
            last = m.partial.rows.length() == m.partialRows ? m.partial : null;
        } else if (!m.blocks.isEmpty()) {
            last = m.blocks.get(m.blocks.size() - 1);
        }
        if (last != null) {
            state.cache.evictChain(last, spare);
        }
        state.cache.forgetHost(h.rows, h.key, h.count, e -> walkedByOther(e, h));
        return h.count;
    }

    /** Whether a history other than {@code erased} walks the host block {@code e}'s whole chain. */
    private boolean walkedByOther(HostBlockCache.Entry e, History erased) {
        for (History other : state.histories) {
            if (other == null || other == erased || !e.key.equals(other.key)) {
                continue;
            }
            int[] want = other.rows.tokens(0, other.count);
            if (e.continues(want, want.length) == e.image.rows()) {
                return true;
            }
        }
        return false;
    }

    /** Refuses an action on this engine, a slot id outside the slots, or a slot that runs a sequence. */
    private void check(int seq, String action) {
        if (refusal != null) {
            throw new UnsupportedOperationException("slot " + action + " refused: " + refusal);
        }
        if (seq < 0 || seq >= state.slots.length) {
            throw new IllegalArgumentException("slot " + seq + " outside the " + state.slots.length + " slots");
        }
        if (state.slots[seq] != null) {
            throw new IllegalStateException("slot " + seq + " is processing a request; " + action
                    + " waits for it to finish");
        }
    }

    /** A cached token block stores its rows at their offsets in the chain: the positions adoption expects. */
    private void checkOffsets(SlotFile.Contents contents, String source) throws IOException {
        int base = 0;
        for (int i = 0; i < contents.blocks.size(); i++) {
            int[] positions = contents.blocks.get(i).positions();
            for (int r = 0; r < positions.length; r++) {
                if (positions[r] != base + r) {
                    throw new IOException(source + " stores position " + positions[r] + " at row " + r + " of block "
                            + i + "; a cached token block stores its chain offset " + (base + r));
                }
            }
            base += positions.length;
        }
    }
}
