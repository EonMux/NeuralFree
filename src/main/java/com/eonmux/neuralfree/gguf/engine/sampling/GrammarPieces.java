package com.eonmux.neuralfree.gguf.engine.sampling;

import java.util.Map;
import java.util.WeakHashMap;

/**
 * The grammar's view of a vocabulary, decoded once: for every token id its end-of-generation flag, whether the mask
 * refuses its piece outright (empty, or starting on the terminating 0), and its piece's code points
 * ({@code decode_utf8} from a clean state, a trailing 0 included) with the partial UTF-8 tail it leaves. A grammar
 * state without a pending partial character decodes every piece the same way, so the rejection walk reads these
 * instead of decoding the vocabulary per step. Immutable once built and shared by every grammar stage over the same
 * vocabulary object ({@link #of}); the pieces are read through {@link SamplerVocab#grammarPiece}, whatever bytes it
 * renders.
 */
final class GrammarPieces {

    /** Flag: the token is an end-of-generation id. */
    static final byte EOG = 1;
    /** Flag: the mask refuses the token's piece (empty, or starting on the terminating 0). */
    static final byte REFUSED = 2;

    /** The UTF-8 sequence length minus one by lead-byte high nibble; -1 marks a continuation byte. */
    private static final int[] LOOKUP = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};

    private static final Map<SamplerVocab, Slot> CACHE = new WeakHashMap<>();

    private final int size;
    private final byte[] flags;
    private final int[] base;
    private final int[] cps;
    private final int[] endValue;
    private final int[] endRemain;

    private GrammarPieces(SamplerVocab vocab) {
        this.size = vocab.vocabSize();
        this.flags = new byte[size];
        this.base = new int[size];
        this.endValue = new int[size];
        this.endRemain = new int[size];
        Decoded out = new Decoded(Math.max(16, size * 4));
        for (int id = 0; id < size; id++) {
            base[id] = out.length;
            if (vocab.isEog(id)) {
                flags[id] = EOG;
                out.put(0);
                continue;
            }
            byte[] piece = vocab.grammarPiece(id);
            if (piece.length == 0 || piece[0] == 0) {
                flags[id] = REFUSED;
                out.put(0);
                continue;
            }
            decode(piece, 0, -1, out);
            endValue[id] = out.endValue;
            endRemain[id] = out.endRemain;
        }
        this.cps = out.trimmed();
    }

    /**
     * The table of {@code vocab}, built on first use and kept while the vocabulary object is reachable. The build runs
     * under its vocabulary's own slot, so a first constrained request on one model does not wait for another's.
     */
    static GrammarPieces of(SamplerVocab vocab) {
        Slot slot;
        synchronized (CACHE) {
            slot = CACHE.computeIfAbsent(vocab, v -> new Slot());
        }
        return slot.get(vocab);
    }

    /** One vocabulary's table, built once by the first caller while later callers of the vocabulary wait. */
    private static final class Slot {
        private GrammarPieces pieces;

        synchronized GrammarPieces get(SamplerVocab vocab) {
            if (pieces == null) {
                pieces = new GrammarPieces(vocab);
            }
            return pieces;
        }
    }

    /** The token ids the table covers, {@code [0, size)}. */
    int size() {
        return size;
    }

    byte flags(int id) {
        return flags[id];
    }

    /** The decoded code points of every covered token, each run ending on 0. */
    int[] codePoints() {
        return cps;
    }

    /** Where token {@code id}'s code points start in {@link #codePoints()}. */
    int base(int id) {
        return base[id];
    }

    int endValue(int id) {
        return endValue[id];
    }

    int endRemain(int id) {
        return endRemain[id];
    }

    /** A growable code point buffer with the partial tail of the last decode. */
    static final class Decoded {
        int[] data;
        int length;
        int endValue;
        int endRemain;

        Decoded(int capacity) {
            this.data = new int[capacity];
        }

        void put(int cp) {
            if (length == data.length) {
                data = java.util.Arrays.copyOf(data, Math.max(16, data.length * 2));
            }
            data[length++] = cp;
        }

        void clear() {
            length = 0;
        }

        int[] trimmed() {
            return length == data.length ? data : java.util.Arrays.copyOf(data, length);
        }
    }

    /**
     * {@code decode_utf8}: appends {@code src}'s code points and a trailing 0 to {@code out}, starting from the
     * partial character {@code (startValue, startRemain)}, and leaves the partial tail in {@code out.endValue} and
     * {@code out.endRemain}. An invalid sequence leaves the single 0 and the tail {@code (0, -1)} (or the negative
     * remainder of an invalid lead byte), as llama.cpp clears the code points it decoded.
     */
    static void decode(byte[] src, int startValue, int startRemain, Decoded out) {
        int from = out.length;
        int value = startValue;
        int nRemain = startRemain;
        int pos = 0;
        while (pos < src.length && nRemain > 0) {
            int nextByte = src[pos] & 0xFF;
            if ((nextByte >> 6) != 2) {
                out.length = from;
                out.put(0);
                out.endValue = 0;
                out.endRemain = -1;
                return;
            }
            value = (value << 6) + (nextByte & 0x3F);
            pos++;
            nRemain--;
        }
        if (startRemain > 0 && nRemain == 0) {
            out.put(value);
        }
        while (pos < src.length) {
            int firstByte = src[pos] & 0xFF;
            nRemain = LOOKUP[firstByte >> 4] - 1;
            if (nRemain < 0) {
                out.length = from;
                out.put(0);
                out.endValue = 0;
                out.endRemain = nRemain;
                return;
            }
            int mask = (1 << (7 - nRemain)) - 1;
            value = firstByte & mask;
            pos++;
            while (pos < src.length && nRemain > 0) {
                value = (value << 6) + (src[pos] & 0x3F);
                pos++;
                nRemain--;
            }
            if (nRemain == 0) {
                out.put(value);
            }
        }
        out.put(0);
        out.endValue = value;
        out.endRemain = nRemain;
    }
}
