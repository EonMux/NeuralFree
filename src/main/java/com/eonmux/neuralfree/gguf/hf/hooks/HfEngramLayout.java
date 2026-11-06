package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * The Engram hash layout of the DeepSeek-V4.1 converter ({@code EngramLayout.from_args},
 * deepseek41.py:17-43): for every Engram layer and n-gram order 2..{@code max_ngram_size}, one prime bucket modulus
 * per head, never reused across layers. The flattened moduli are the rows of the concatenated Engram table
 * ({@code ple.head_vocab_sizes}) and their prefix sums its row offsets ({@code ple.head_offsets}).
 */
public final class HfEngramLayout {

    private final List<Integer> layers;
    private final int maxNgram;
    private final int heads;
    private final int[][] sizes;
    private final long[] rows;
    private final long[] headSizes;
    private final long[] headOffsets;

    private HfEngramLayout(List<Integer> layers, int maxNgram, int heads, int[][] sizes) {
        this.layers = Collections.unmodifiableList(new ArrayList<>(layers));
        this.maxNgram = maxNgram;
        this.heads = heads;
        this.sizes = sizes;
        this.rows = new long[sizes.length];
        this.headSizes = new long[sizes.length * (maxNgram - 1) * heads];
        int at = 0;
        for (int layer = 0; layer < sizes.length; layer++) {
            for (int size : sizes[layer]) {
                headSizes[at++] = size;
                rows[layer] += size;
            }
        }
        this.headOffsets = new long[headSizes.length];
        long offset = 0;
        for (int i = 0; i < headSizes.length; i++) {
            headOffsets[i] = offset;
            offset += headSizes[i];
        }
    }

    /**
     * The layout of {@code cfg}.
     *
     * @throws GgufFormatException when a required key is missing or not a non-negative integer, or a layer's moduli
     *                             do not sum to its {@code engram_num_embeddings} entry
     */
    public static HfEngramLayout of(HfConfig cfg) {
        List<Integer> layers = requireList(cfg, "engram_layer_ids");
        int maxNgram = cfg.requireInt("engram_max_ngram_size");
        int heads = cfg.requireInt("engram_n_heads");
        int vocab = cfg.requireInt("engram_vocab_size");
        if (layers.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": 'engram_layer_ids' is empty");
        }
        if (maxNgram < 1 || heads < 0 || vocab < 1) {
            throw new GgufFormatException(cfg.source() + ": Engram needs engram_max_ngram_size >= 1,"
                    + " engram_n_heads >= 0 and engram_vocab_size >= 1");
        }
        int[][] sizes = primes(layers.size(), maxNgram, heads, vocab);
        List<Integer> embeddings = requireList(cfg, "engram_num_embeddings");
        if (embeddings.size() < layers.size()) {
            throw new GgufFormatException(cfg.source() + ": 'engram_num_embeddings' has " + embeddings.size()
                    + " entries for " + layers.size() + " Engram layers");
        }
        for (int layer = 0; layer < sizes.length; layer++) {
            long sum = 0;
            for (int size : sizes[layer]) {
                sum += size;
            }
            long expected = embeddings.get(layer);
            if (sum != expected) {
                throw new GgufFormatException(cfg.source() + ": Engram layer " + layers.get(layer)
                        + ": the bucket moduli sum to " + sum + ", engram_num_embeddings says " + expected);
            }
        }
        return new HfEngramLayout(layers, maxNgram, heads, sizes);
    }

    /** {@code EngramLayout.from_args}'s primes: {@code [layer][order * heads + head]}. */
    static int[][] primes(int layerCount, int maxNgram, int headCount, int vocab) {
        java.util.Set<Integer> seen = new java.util.HashSet<>();
        int[][] out = new int[layerCount][];
        for (int layer = 0; layer < layerCount; layer++) {
            int[] layerSizes = new int[(maxNgram - 1) * headCount];
            int at = 0;
            for (int order = 0; order < maxNgram - 1; order++) {
                int current = vocab - 1;
                for (int head = 0; head < headCount; head++) {
                    int candidate = current + 1;
                    while (!isPrime(candidate) || seen.contains(candidate)) {
                        candidate++;
                    }
                    current = candidate;
                    seen.add(candidate);
                    layerSizes[at++] = candidate;
                }
            }
            out[layer] = layerSizes;
        }
        return out;
    }

    /** The converter's {@code _is_prime}. */
    static boolean isPrime(int n) {
        if (n < 2 || n % 2 == 0) {
            return n == 2;
        }
        for (int d = 3; (long) d * d <= n; d += 2) {
            if (n % d == 0) {
                return false;
            }
        }
        return true;
    }

    private static List<Integer> requireList(HfConfig cfg, String key) {
        Optional<List<Integer>> value = cfg.findIntList(key);
        if (!value.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required Engram key '" + key + "' is missing");
        }
        return value.get();
    }

    /** The Engram layer ids, in configuration order. */
    public List<Integer> layers() { return layers; }

    /** {@code engram_max_ngram_size}. */
    public int maxNgram() { return maxNgram; }

    /** {@code engram_n_heads}. */
    public int heads() { return heads; }

    /** The table rows of each layer: the sum of its bucket moduli. */
    public long[] rows() {
        return rows.clone();
    }

    /** The bucket modulus of every head, in row order. */
    public long[] headSizes() { return headSizes.clone(); }

    /** The row offset of every head into the concatenated table. */
    public long[] headOffsets() { return headOffsets.clone(); }

    /** The hash multipliers, {@code maxNgram} per layer in layer order. */
    public long[] multipliers(long compressedVocab) {
        int[] ids = new int[layers.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = layers.get(i);
        }
        return HfEngramHash.multipliers(ids, maxNgram, compressedVocab);
    }
}
