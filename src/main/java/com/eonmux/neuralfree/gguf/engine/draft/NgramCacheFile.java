package com.eonmux.neuralfree.gguf.engine.draft;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Reading and writing an {@link NgramCache} in llama.cpp's n-gram cache file layout
 * ({@code common_ngram_cache_save}, common/ngram-cache.cpp), little-endian: a sequence of records, each
 * <ol>
 *   <li>the key of the n-gram, 4 fixed {@code int32} token ids (padding{@code NgramCache#NULL}, that is -1),</li>
 *   <li>{@code int32} count of part's tokens ({@code > 0}),</li>
 *   <li>that many pairs of {@code int32} token id ({@code >= 0}) and {@code int32} count ({@code > 0}).</li>
 * </ol>
 * Loading validates all lengths (llama.cpp asserts each read reached its full length): a record must be whole, no
 * count may be non-positive, padding tokens must form a contiguous suffix of the key, and the file must end exactly
 * after the last record; every failure names the file offset it happens at. Loading keeps only the first record of a
 * repeated key, as llama.cpp's hash-map insert does. Saving writes the part keys and each part's rows sorted
 * (llama.cpp writes hash-map iteration order; a stable order keeps files reproducible and keeps one record per key).
 */
public final class NgramCacheFile {

    /** Bytes of one {@code int32} field. */
    private static final int FIELD_BYTES = 4;
    /** Bytes of one fixed-width key. */
    private static final int KEY_BYTES = 16;

    private NgramCacheFile() {
    }

    /** Writes {@code cache} to {@code file}, replacing any existing content. */
    public static void save(NgramCache cache, Path file) throws IOException {
        try (OutputStream out = Files.newOutputStream(file)) {
            save(cache, out);
        }
    }

    /** Writes {@code cache} to {@code out}; does not close {@code out}. */
    public static void save(NgramCache cache, OutputStream out) throws IOException {
        Objects.requireNonNull(cache, "cache");
        Objects.requireNonNull(out, "out");
        Map<NgramCache.Key, Map<Integer, Integer>> entries = cache.entries();
        if (entries.isEmpty()) {
            return;
        }
        List<NgramCache.Key> keys = entries.keySet().stream().sorted(keyOrder())
                .collect(Collectors.toList());
        for (NgramCache.Key key : keys) {
            writeInt32(out, key.t0());
            writeInt32(out, key.t1());
            writeInt32(out, key.t2());
            writeInt32(out, key.t3());
            Map<Integer, Integer> part = entries.get(key);
            writeInt32(out, part.size());
            List<Map.Entry<Integer, Integer>> sorted = part.entrySet().stream()
                    .sorted(Comparator.comparingInt(Map.Entry::getKey)).collect(Collectors.toList());
            for (Map.Entry<Integer, Integer> e : sorted) {
                writeInt32(out, e.getKey());
                writeInt32(out, e.getValue());
            }
        }
    }

    /** Loads the cache stored in {@code file}. */
    public static NgramCache load(Path file) throws IOException {
        try (InputStream in = Files.newInputStream(file)) {
            return load(in, file.toString());
        }
    }

    /**
     * Loads from {@code in} to its end; does not close {@code in}.
     *
     * @param name file name used in the error messages
     */
    public static NgramCache load(InputStream in, String name) throws IOException {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(name, "name");
        NgramCache cache = new NgramCache();
        byte[] field = new byte[FIELD_BYTES];
        byte[] pair = new byte[2 * FIELD_BYTES];
        long offset = 0;
        while (true) {
            byte[] keyBytes = new byte[KEY_BYTES];
            int n = in.readNBytes(keyBytes, 0, KEY_BYTES);
            if (n == 0) {
                return cache;
            }
            if (n < KEY_BYTES) {
                throw io(name, offset, "a key needs " + KEY_BYTES + " bytes, found " + n);
            }
            int[] tokens = new int[NgramCache.MAX_KEY];
            for (int i = 0; i < NgramCache.MAX_KEY; i++) {
                tokens[i] = int32LE(keyBytes, i * FIELD_BYTES);
            }
            NgramCache.Key key = paddedKey(name, offset, tokens);
            offset += KEY_BYTES;
            int ntokens = int32LE(safeRead(in, field, name, offset, "the token-count field of n-gram "
                    + key + " is truncated"), 0);
            offset += FIELD_BYTES;
            if (ntokens <= 0) {
                throw io(name, offset - FIELD_BYTES, "n-gram " + key + " stores a non-positive token count " + ntokens);
            }
            if (cache.hasPart(key)) {
                // llama.cpp's hash map insert keeps only the first record of a key.
                skipRows(in, name, offset, key, ntokens, pair);
                continue;
            }
            for (int i = 0; i < ntokens; i++) {
                byte[] row = safeRead(in, pair, name, offset, "n-gram " + key + " counts " + ntokens + " tokens but"
                        + " only " + i + " were complete");
                int token = int32LE(row, 0);
                int count = int32LE(row, FIELD_BYTES);
                if (token < 0) {
                    throw io(name, offset, "n-gram " + key + " counts negative token id " + token);
                }
                if (count <= 0) {
                    throw io(name, offset + FIELD_BYTES, "n-gram " + key + " counts token " + token + " with"
                            + " non-positive count " + count);
                }
                cache.addCount(key, token, count);
                offset += row.length;
            }
        }
    }

    /** Padding-free tokens must form a contiguous prefix with ids {@code >= 0} and length at least 1. */
    private static NgramCache.Key paddedKey(String name, long offset, int[] tokens) throws IOException {
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] == NgramCache.NULL) {
                if (i == 0) {
                    throw io(name, offset, "key is entirely padding, no token ids");
                }
                for (int j = i; j < tokens.length; j++) {
                    if (tokens[j] != NgramCache.NULL) {
                        throw io(name, offset, "key has token id " + tokens[j] + " behind its padding");
                    }
                }
                break;
            }
            if (tokens[i] < 0) {
                throw io(name, offset, "key token " + i + " has invalid id " + tokens[i]);
            }
        }
        return new NgramCache.Key(tokens[0], tokens[1], tokens[2], tokens[3]);
    }

    private static void skipRows(InputStream in, String name, long offset, NgramCache.Key key, int ntokens,
                                 byte[] pair) throws IOException {
        for (int i = 0; i < ntokens; i++) {
            int n = in.readNBytes(pair, 0, pair.length);
            if (n < pair.length) {
                throw io(name, offset, "n-gram " + key + " counts " + ntokens + " tokens but only " + i
                        + " were complete");
            }
        }
    }

    /** Reads {@code out.length} bytes exactly, or reports the truncation at {@code offset}. */
    private static byte[] safeRead(InputStream in, byte[] out, String name, long offset, String message)
            throws IOException {
        if (in.readNBytes(out, 0, out.length) < out.length) {
            throw io(name, offset, message);
        }
        return out;
    }

    private static void writeInt32(OutputStream out, int v) throws IOException {
        out.write(v & 0xFF);
        out.write((v >>> 8) & 0xFF);
        out.write((v >>> 16) & 0xFF);
        out.write((v >>> 24) & 0xFF);
    }

    private static int int32LE(byte[] b, int off) {
        return (b[off] & 0xFF) | ((b[off + 1] & 0xFF) << 8) | ((b[off + 2] & 0xFF) << 16)
                | ((b[off + 3] & 0xFF) << 24);
    }

    private static UncheckedIOException io(String name, long offset, String message) {
        return new UncheckedIOException(name + " at offset " + offset + ": " + message, new IOException(message));
    }

    private static Comparator<NgramCache.Key> keyOrder() {
        return Comparator.comparingInt(NgramCache.Key::t0)
                .thenComparingInt(NgramCache.Key::t1)
                .thenComparingInt(NgramCache.Key::t2)
                .thenComparingInt(NgramCache.Key::t3);
    }
}
