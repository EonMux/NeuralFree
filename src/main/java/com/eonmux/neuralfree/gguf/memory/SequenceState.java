package com.eonmux.neuralfree.gguf.memory;

import java.util.Arrays;
import java.util.Objects;

/**
 * Host-side record of one sequence held in memory: one entry per stored row, in the order the rows were written
 * ({@code n_past} = {@link #nPast()}), each with the row's token and stored position (llama.cpp's per-cell
 * {@code ext.tok} with the cell position). Positions never descend; a contiguous record stores entry {@code i} at
 * position {@code i}, while a paged record can store several media rows at one shared M-RoPE position. An embedding
 * row records the token its model gives it ({@link #appendEmbeddings(int, int)}): {@code -1} for no token, or the
 * stand-in id of an n-gram per-layer embedding; either way the entry is marked as an embedding row
 * ({@link #embeddingRow}), so {@link #textTokens()} gives the prompt's text view (llama.cpp server
 * {@code get_text_tokens}). Used for prefix reuse (longest common prefix between generations), rollback, the n-gram
 * predecessor lookup ({@link #tokenAtOrBefore}) and prompt-lookup drafting. This is the one mutable runtime object
 * of the memory API; it is owned by a {@link ModelMemory} and is not thread-safe.
 *
 * <p>{@link #reusable()} is the contiguous-reuse frontier: the length of the stored
 * prefix whose rows were computed with exactly the tokens at positions {@code [0, reusable)}. It never exceeds
 * {@code nPast}, is raised only by appends that extend the exact same prefix, and is lowered by every rewrite:
 * {@link #truncate}, {@link #capReusable}, and a context shift ({@link #shift}) lowers it to the kept count.
 * {@link #commonPrefix} never reports past it, so a generation can never reuse rows that were computed with a
 * different prefix.
 *
 * <p>{@link #adapters()} names the per-request LoRA adapters the stored rows were computed under
 * ({@code LoraPlan.identity()}, null for slot 0, the startup adapters): {@link #commonPrefix(int[], String)} reuses
 * nothing computed under other adapters.</p>
 */
public final class SequenceState {

    private int[] tokens;
    private int[] positions;
    private boolean[] embedding;
    private int nPast;
    private int reusable;
    private boolean wrapped;
    private String adapters;
    private final int contextLength;
    private final int capacity;

    /** @param nCtx context length; it also bounds the stored rows ({@link #capacity()}) */
    public SequenceState(int nCtx) {
        this(nCtx, nCtx);
    }

    /**
     * @param nCtx     context length, the positions a sequence spans ({@link #nCtx()})
     * @param capacity stored rows the record holds at most; at least {@code nCtx}, larger for a paged sequence whose
     *                 media rows can share positions
     * @throws IllegalArgumentException when {@code nCtx} is not positive or {@code capacity} is below it
     */
    public SequenceState(int nCtx, int capacity) {
        if (nCtx <= 0) {
            throw new IllegalArgumentException("sequence context must be positive, got " + nCtx);
        }
        if (capacity < nCtx) {
            throw new IllegalArgumentException("record capacity " + capacity + " is below the context " + nCtx);
        }
        this.contextLength = nCtx;
        this.capacity = capacity;
        int initial = Math.min(capacity, 1024);
        this.tokens = new int[initial];
        this.positions = new int[initial];
        this.embedding = new boolean[initial];
    }

    /** Number of stored entries (on a contiguous record, the next position to write). */
    public int nPast() { return nPast; }

    /**
     * Length of the stored prefix whose rows were computed with exactly the stored tokens: the reuse frontier
     * of the contiguous path ({@code ≤ nPast}). Raised only by contiguous appends, lowered by every rewrite
     * ({@link #truncate}, {@link #capReusable}, {@link #shift}); {@link #clear} resets it to 0.
     */
    public int reusable() { return reusable; }

    /** Lowers the reuse frontier to {@code at} (a KV edit at or above it invalidates the rows after it). */
    public void capReusable(int at) {
        if (reusable > Math.max(0, at)) {
            this.reusable = Math.max(0, at);
        }
    }

    /** Context length. */
    public int nCtx() { return contextLength; }

    /** Stored rows the record holds at most ({@code nCtx} for a contiguous sequence). */
    public int capacity() { return capacity; }

    /** True once any ring cache has overwritten a stored row since the last {@link #clear()}. */
    public boolean wrapped() { return wrapped; }

    /**
     * Records that a ring cache overwrote a stored row. Informational: whether positions can be dropped is decided
     * from the stored rows ({@link ModelMemory#canRollback(int)}), not from this flag.
     */
    public void markWrapped() { this.wrapped = true; }

    /** Token of entry {@code i} (the position {@code i} of a contiguous record). */
    public int token(int i) {
        checkEntry(i);
        return tokens[i];
    }

    /** Stored position of entry {@code i}. */
    public int position(int i) {
        checkEntry(i);
        return positions[i];
    }

    /** Whether entry {@code i} is an embedding row (its token, if any, is the stand-in its model hashes). */
    public boolean embeddingRow(int i) {
        checkEntry(i);
        return embedding[i];
    }

    /**
     * Records {@code token} as the id of the token entry {@code i}, which was stored before its id reached the host
     * (a chained greedy decode reads its input id from the device's previous selection): the entry's row was
     * computed with exactly that id, so the reuse frontier stays where it is.
     *
     * @throws IndexOutOfBoundsException when {@code i} is not a stored entry
     * @throws IllegalArgumentException  when the entry is an embedding row
     */
    public void settleToken(int i, int token) {
        checkEntry(i);
        if (embedding[i]) {
            throw new IllegalArgumentException("entry " + i + " is an embedding row; only a token entry takes an id");
        }
        tokens[i] = token;
    }

    private void checkEntry(int i) {
        if (i < 0 || i >= nPast) {
            throw new IndexOutOfBoundsException("entry " + i + " outside the stored range [0, " + nPast + ")");
        }
    }

    /** Copy of the stored tokens. */
    public int[] tokens() {
        return Arrays.copyOf(tokens, nPast);
    }

    /** The tokens of the entries that are not embedding rows, in order: the text the sequence holds. */
    public int[] textTokens() {
        int[] out = new int[nPast];
        int n = 0;
        for (int i = 0; i < nPast; i++) {
            if (!embedding[i]) {
                out[n++] = tokens[i];
            }
        }
        return n == nPast ? out : Arrays.copyOf(out, n);
    }

    /**
     * The token of the latest entry stored at a position at or before {@code pos}, or {@code -1} when no entry is
     * (llama.cpp {@code seq_pos_tok_le}: the nearest cell at or before a position, the last written among cells
     * sharing it, {@code LLAMA_TOKEN_NULL} when none).
     */
    public int tokenAtOrBefore(int pos) {
        int lo = 0;
        int hi = nPast;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (positions[mid] <= pos) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo == 0 ? -1 : tokens[lo - 1];
    }

    /**
     * Appends tokens written at the {@code n} positions after the last stored one ({@code [nPast, nPast + n)} on a
     * contiguous record).
     *
     * @throws IllegalStateException when the context would overflow
     */
    public void append(int[] newTokens, int off, int n) {
        if (off < 0 || n < 0 || off + n > newTokens.length) {
            throw new IndexOutOfBoundsException("token range [" + off + ", " + (off + n) + ") outside " + newTokens.length);
        }
        int first = nextPosition();
        reserve(n);
        // The frontier only rises by a contiguous append, an extension of the exact prefix.
        boolean contiguous = reusable == nPast && first == nPast;
        System.arraycopy(newTokens, off, tokens, nPast, n);
        Arrays.fill(embedding, nPast, nPast + n, false);
        for (int i = 0; i < n; i++) {
            positions[nPast + i] = first + i;
        }
        nPast += n;
        if (contiguous) {
            reusable = nPast;
        }
    }

    /**
     * Appends {@code n} rows of a paged step stored at {@code rowPositions[off + i]}: a token row
     * ({@code rowTokens[off + i] >= 0}) records its id, an embedding row (negative) records {@code embeddingToken}
     * and is marked as one. Positions do not descend, from the last stored one on. The reuse frontier stays: a paged
     * record is not the contiguous reuse path.
     *
     * @param embeddingToken the id an embedding row records: {@code -1} for none, or its model's stand-in
     * @throws IllegalArgumentException when a range is outside its array or a position descends
     * @throws IllegalStateException    when the record would overflow its capacity
     */
    public void appendRows(int[] rowTokens, int[] rowPositions, int off, int n, int embeddingToken) {
        if (off < 0 || n < 0 || off + n > rowTokens.length || off + n > rowPositions.length) {
            throw new IllegalArgumentException("row range [" + off + ", " + (off + n) + ") outside "
                    + rowTokens.length + " tokens / " + rowPositions.length + " positions");
        }
        int last = nPast == 0 ? 0 : positions[nPast - 1];
        for (int i = 0; i < n; i++) {
            int p = rowPositions[off + i];
            if (p < last) {
                throw new IllegalArgumentException("row " + i + " stores position " + p + " below " + last);
            }
            last = p;
        }
        reserve(n);
        for (int i = 0; i < n; i++) {
            boolean row = rowTokens[off + i] < 0;
            tokens[nPast + i] = row ? embeddingToken : rowTokens[off + i];
            embedding[nPast + i] = row;
        }
        System.arraycopy(rowPositions, off, positions, nPast, n);
        nPast += n;
    }

    /**
     * Appends {@code n} embedding rows that carry no token id: {@link #appendEmbeddings(int, int)} recording
     * {@code -1}, the {@code ForwardBatch} embedding-row marker.
     */
    public void appendEmbeddings(int n) {
        appendEmbeddings(n, -1);
    }

    /**
     * Appends {@code n} embedding rows written at the positions after the last stored one, each recording
     * {@code token}: {@code -1} for no token id, or the stand-in id the model hashes for an embedding row (the n-gram
     * per-layer embedding, llama.cpp {@code ple_embd_token}). The reuse frontier stays: no prompt token identifies a
     * written embedding row, so nothing after it is reusable.
     *
     * @throws IllegalArgumentException when {@code n} is negative
     * @throws IllegalStateException    when the context would overflow
     */
    public void appendEmbeddings(int n, int token) {
        if (n < 0) {
            throw new IllegalArgumentException("embedding row count " + n + " is negative");
        }
        int first = nextPosition();
        reserve(n);
        Arrays.fill(tokens, nPast, nPast + n, token);
        Arrays.fill(embedding, nPast, nPast + n, true);
        for (int i = 0; i < n; i++) {
            positions[nPast + i] = first + i;
        }
        nPast += n;
    }

    /** The position after the last stored one: 0 for an empty record. */
    private int nextPosition() {
        return nPast == 0 ? 0 : positions[nPast - 1] + 1;
    }

    /** Grows the arrays for {@code n} more entries. @throws IllegalStateException when the capacity overflows */
    private void reserve(int n) {
        if ((long) nPast + n > capacity) {
            throw new IllegalStateException("context full: " + nPast + " + " + n + " tokens exceed " + capacity);
        }
        if (nPast + n > tokens.length) {
            int cap = (int) Math.min(capacity, Math.max((long) tokens.length * 2, (long) nPast + n));
            tokens = Arrays.copyOf(tokens, cap);
            positions = Arrays.copyOf(positions, cap);
            embedding = Arrays.copyOf(embedding, cap);
        }
    }

    /** Drops the last {@code n} entries. */
    public void truncate(int n) {
        if (n < 0 || n > nPast) {
            throw new IllegalArgumentException("cannot drop " + n + " of " + nPast + " stored positions");
        }
        nPast -= n;
        capReusable(nPast);
    }

    /**
     * Drops every entry stored at a position in {@code [lo, hi)}, keeping the others in order (a removal from the
     * sequence, llama.cpp {@code seq_rm}). The reuse frontier falls to the first dropped entry.
     *
     * @throws IllegalArgumentException when {@code lo} is negative
     */
    public void removePositions(int lo, int hi) {
        if (lo < 0) {
            throw new IllegalArgumentException("removal starts before the sequence: position " + lo);
        }
        int w = 0;
        int firstDropped = -1;
        for (int i = 0; i < nPast; i++) {
            if (positions[i] >= lo && positions[i] < hi) {
                if (firstDropped < 0) {
                    firstDropped = i;
                }
                continue;
            }
            tokens[w] = tokens[i];
            positions[w] = positions[i];
            embedding[w] = embedding[i];
            w++;
        }
        nPast = w;
        if (firstDropped >= 0) {
            capReusable(firstDropped);
        }
    }

    /**
     * Rewrites the token of every entry stored at {@code positions[i]} to {@code tokens[i]}, the entries staying
     * where they are: the commit of a compaction, whose KV rows already sit at their final positions and only the
     * record's ids follow them ({@code SequenceMemory#moveRows}). The positions must ascend strictly and each must
     * name exactly one stored entry; the entries keep their order and their embedding marks. The reuse frontier
     * falls to the first rewritten entry, whose row was computed with another token.
     *
     * @throws IllegalArgumentException when a position is not stored, or the positions descend or repeat
     */
    public void retokenize(int[] positions, int[] tokens) {
        Objects.requireNonNull(positions, "positions");
        Objects.requireNonNull(tokens, "tokens");
        if (positions.length != tokens.length) {
            throw new IllegalArgumentException("retokenize names " + positions.length + " positions and "
                    + tokens.length + " tokens");
        }
        int first = -1;
        for (int i = 0; i < positions.length; i++) {
            int p = positions[i];
            if (i > 0 && p <= positions[i - 1]) {
                throw new IllegalArgumentException("retokenize positions must ascend strictly, got " + p
                        + " after " + positions[i - 1]);
            }
            int at = entryAt(p);
            if (at < 0) {
                throw new IllegalArgumentException("retokenize names position " + p + ", which no stored entry"
                        + " holds");
            }
            this.tokens[at] = tokens[i];
            embedding[at] = false;
            if (first < 0) {
                first = at;
            }
        }
        if (first >= 0) {
            capReusable(first);
        }
    }

    /** Index of the first entry stored at {@code p}, or -1 when none is (the stored positions never descend). */
    private int entryAt(int p) {
        int lo = 0;
        int hi = nPast;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (positions[mid] < p) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo < nPast && positions[lo] == p ? lo : -1;
    }

    /**
     * Moves the entries stored at a position in {@code [lo, hi)} by {@code delta} (llama.cpp {@code seq_add}): an
     * entry moved below position 0 leaves, as its cell does, and the entries are reordered by position, the written
     * order kept among entries sharing one. The reuse frontier falls to the first moved entry.
     *
     * @throws IllegalArgumentException when {@code lo} is negative
     */
    public void shiftPositions(int lo, int hi, int delta) {
        if (lo < 0) {
            throw new IllegalArgumentException("shift starts before the sequence: position " + lo);
        }
        int firstMoved = -1;
        for (int i = 0; i < nPast; i++) {
            if (positions[i] >= lo && positions[i] < hi) {
                firstMoved = firstMoved < 0 ? i : firstMoved;
                positions[i] += delta;
            }
        }
        if (firstMoved < 0 || delta == 0) {
            return;
        }
        capReusable(firstMoved);
        reorderByPosition();
    }

    /**
     * Divides the positions of the entries stored at a position in {@code [lo, hi)} by {@code d} (llama.cpp
     * {@code seq_div}): the entries keep their tokens, several can end up sharing the one position, and they are
     * reordered by position as {@link #shiftPositions} reorders them, the written order kept among entries sharing
     * one. The reuse frontier falls to the first moved entry.
     *
     * @throws IllegalArgumentException when {@code lo} is negative or {@code d} is below 2
     */
    public void divPositions(int lo, int hi, int d) {
        if (lo < 0) {
            throw new IllegalArgumentException("division starts before the sequence: position " + lo);
        }
        if (d < 2) {
            throw new IllegalArgumentException("position division needs a divisor of at least 2, got " + d);
        }
        int firstMoved = -1;
        for (int i = 0; i < nPast; i++) {
            if (positions[i] >= lo && positions[i] < hi) {
                firstMoved = firstMoved < 0 ? i : firstMoved;
                positions[i] /= d;
            }
        }
        if (firstMoved < 0) {
            return;
        }
        capReusable(firstMoved);
        reorderByPosition();
    }

    /** Drops entries at negative positions and stably sorts the rest by position. */
    private void reorderByPosition() {
        long[] keys = new long[nPast];
        int n = 0;
        for (int i = 0; i < nPast; i++) {
            if (positions[i] >= 0) {
                keys[n++] = ((long) positions[i] << 32) | i;
            }
        }
        Arrays.sort(keys, 0, n);
        int[] sortedTokens = new int[tokens.length];
        int[] sortedPositions = new int[positions.length];
        boolean[] sortedEmbedding = new boolean[embedding.length];
        for (int k = 0; k < n; k++) {
            int i = (int) (keys[k] & 0xFFFFFFFFL);
            sortedTokens[k] = tokens[i];
            sortedPositions[k] = positions[i];
            sortedEmbedding[k] = embedding[i];
        }
        tokens = sortedTokens;
        positions = sortedPositions;
        embedding = sortedEmbedding;
        if (n < nPast) {
            capReusable(0);
        }
        nPast = n;
    }

    /**
     * Replaces this record by the entries of {@code src} stored at a position in {@code [lo, hi)} (llama.cpp
     * {@code seq_cp}: the copied cells keep their tokens). The reuse frontier covers the copied part of the source's
     * when the copy starts at position 0, else nothing.
     *
     * @throws IllegalArgumentException when {@code src} is this record or {@code lo} is negative
     * @throws IllegalStateException    when the copied entries overflow this record's capacity
     */
    public void copyFrom(SequenceState src, int lo, int hi) {
        if (src == this) {
            throw new IllegalArgumentException("a record cannot copy itself");
        }
        if (lo < 0) {
            throw new IllegalArgumentException("copy starts before the sequence: position " + lo);
        }
        clear();
        int sourcePrefix = 0;
        for (int i = 0; i < src.nPast; i++) {
            int p = src.positions[i];
            if (p < lo || p >= hi) {
                continue;
            }
            reserve(1);
            tokens[nPast] = src.tokens[i];
            positions[nPast] = p;
            embedding[nPast] = src.embedding[i];
            nPast++;
            if (i == sourcePrefix) {
                sourcePrefix++;
            }
        }
        reusable = lo == 0 ? Math.min(src.reusable, sourcePrefix) : 0;
        wrapped = src.wrapped;
    }

    /** Forgets every position. */
    public void clear() {
        nPast = 0;
        reusable = 0;
        wrapped = false;
        adapters = null;
    }

    /** The identity of the per-request adapters the stored rows were computed under, null for slot 0. */
    public String adapters() {
        return adapters;
    }

    /**
     * Records that the rows from now on are computed under the adapters of {@code identity} (null for slot 0);
     * a change while rows are stored lowers the reuse frontier to 0, since those rows ran under other adapters.
     */
    public void adapters(String identity) {
        if (!Objects.equals(identity, adapters)) {
            capReusable(0);
            this.adapters = identity;
        }
    }

    /**
     * Context shift on a contiguous token record: drops entries {@code [nKeep, nKeep + nDiscard)} and moves the tail
     * down by {@code nDiscard} positions, keeping the wrap flag. The reuse frontier falls to {@code nKeep}: the kept rows were computed with the
     * discarded context between them, so only the kept prefix is reusable ({@link #capReusable}).
     *
     * @see com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs
     */
    public void shift(int nKeep, int nDiscard) {
        if (nKeep < 0 || nDiscard <= 0 || (long) nKeep + nDiscard > nPast) {
            throw new IllegalArgumentException("context shift of keep=" + nKeep + " discard=" + nDiscard
                    + " is invalid for " + nPast + " stored positions");
        }
        int tail = nPast - nKeep - nDiscard;
        System.arraycopy(tokens, nKeep + nDiscard, tokens, nKeep, tail);
        System.arraycopy(positions, nKeep + nDiscard, positions, nKeep, tail);
        System.arraycopy(embedding, nKeep + nDiscard, embedding, nKeep, tail);
        for (int i = nKeep; i < nKeep + tail; i++) {
            positions[i] -= nDiscard;
        }
        nPast -= nDiscard;
        capReusable(nKeep);
    }

    /**
     * Length of the common prefix of the stored tokens and {@code prompt}, never past {@link #reusable()}, for a
     * generation in slot 0: nothing when the stored rows ran under per-request adapters.
     */
    public int commonPrefix(int[] prompt) {
        return commonPrefix(prompt, null);
    }

    /**
     * Length of the common prefix of the stored tokens and {@code prompt}, never past {@link #reusable()}, for a
     * generation under the adapters of {@code identity}: nothing when the stored rows ran under others.
     */
    public int commonPrefix(int[] prompt, String identity) {
        if (!Objects.equals(identity, adapters)) {
            return 0;
        }
        int n = Math.min(Math.min(nPast, reusable), prompt.length);
        int i = 0;
        while (i < n && tokens[i] == prompt[i]) {
            i++;
        }
        return i;
    }

    @Override
    public String toString() {
        return "SequenceState{nPast=" + nPast + ", reuse=" + reusable + ", nCtx=" + contextLength
                + (wrapped ? ", wrapped" : "") + '}';
    }
}
