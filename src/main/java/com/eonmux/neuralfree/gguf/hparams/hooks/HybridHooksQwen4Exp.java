package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import java.util.Arrays;

/**
 * The qwen4exp-specific parts of {@link HybridHooksGdn#qwen4exp}: low-rank hyper-connections, the QSA
 * (pooled-key) indexer of the full-attention layers and the PLE n-gram hash embedding module of one linear
 * layer (qwen4exp.cpp:45-164). Malformed metadata is rejected at load, as the reference loader does with
 * catchable errors (qwen4exp.cpp:9-24).
 */
final class HybridHooksQwen4Exp {

    /** {@code LLAMA_MAX_PLE_NGRAM} (llama-hparams.h:13). */
    static final int MAX_PLE_NGRAM = 8;
    /** {@code LLAMA_MAX_PLE_HEADS} (llama-hparams.h:14). */
    static final int MAX_PLE_HEADS = 64;
    /** Indexer key row layout: raw key then pooled key (qwen4exp.cpp:81). */
    private static final int KPOOL_ROW_RAW_AND_POOLED = 2;

    private HybridHooksQwen4Exp() {
    }

    /**
     * qwen4exp.cpp:45-54: {@code hyper_connection.count > 1} residual streams mixed through a rank
     * {@code hyper_connection.low_rank > 0} projection; the trunk output carries all streams.
     */
    static void readHyperConnections(ModelHparamsBuilder b, GgufKv kv) {
        b.dsv4HcMult(kv.u32Required(GgufKeys.HYPER_CONNECTION_COUNT));       // qwen4exp.cpp:45
        b.hcLowRank(kv.u32Required(GgufKeys.HYPER_CONNECTION_LOW_RANK));     // qwen4exp.cpp:46
        if (b.dsv4HcMult() <= 1) {                                           // qwen4exp.cpp:49-52
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.HYPER_CONNECTION_COUNT)
                    + "' must be greater than one, got " + b.dsv4HcMult());
        }
        HybridHooks.requireNonZero(kv, GgufKeys.HYPER_CONNECTION_LOW_RANK, b.hcLowRank());   // qwen4exp.cpp:53
        long out = (long) b.dsv4HcMult() * b.nEmbd();                        // qwen4exp.cpp:54
        if (out > Integer.MAX_VALUE) {
            throw HybridHooks.invalid(b, "hyper-connection output width " + out + " overflows");
        }
        b.nEmbdOutImpl((int) out);
    }

    /**
     * qwen4exp.cpp:56-83: the indexer sizes (all required and nonzero) and the per-layer
     * {@code attention.compress_ratios}; every QSA layer pools keys over one shared ratio above 1 that
     * divides {@code attention.indexer.top_k}.
     */
    static void readIndexer(ModelHparamsBuilder b, GgufKv kv) {
        b.indexerNHead(kv.u32Required(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT));        // qwen4exp.cpp:56
        b.indexerHeadSize(kv.u32Required(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH));     // qwen4exp.cpp:57
        b.indexerTopK(kv.u32Required(GgufKeys.ATTENTION_INDEXER_TOP_K));              // qwen4exp.cpp:58
        HybridHooks.requireNonZero(kv, GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, b.indexerNHead());   // qwen4exp.cpp:59
        HybridHooks.requireNonZero(kv, GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, b.indexerHeadSize()); // qwen4exp.cpp:60
        HybridHooks.requireNonZero(kv, GgufKeys.ATTENTION_INDEXER_TOP_K, b.indexerTopK());          // qwen4exp.cpp:61
        int[] ratios = kv.keyOrArr(GgufKeys.ATTENTION_COMPRESS_RATIOS, b.nLayerAll(), 0);   // qwen4exp.cpp:62
        b.compressRatioArray(ratios);
        int kpool = 0;                                                        // qwen4exp.cpp:65-75
        for (int r : ratios) {
            if (r == 0) {
                continue;
            }
            if (kpool != 0 && r != kpool) {
                throw HybridHooks.invalid(b, "QSA layers must share one compress ratio, got " + kpool + " and " + r);
            }
            kpool = r;
        }
        if (kpool == 1 || (kpool > 0 && b.indexerTopK() % kpool != 0)) {     // qwen4exp.cpp:76-79
            throw HybridHooks.invalid(b, "QSA needs a compress ratio above 1 that divides the budget, got " + kpool
                    + " and " + b.indexerTopK());
        }
        b.indexerKpool(kpool);
        b.indexerKpoolRow(KPOOL_ROW_RAW_AND_POOLED);                          // qwen4exp.cpp:81
        b.indexerKpoolByOrder(true);                                          // qwen4exp.cpp:82
        b.indexerKpoolSelectTail(true);                                       // qwen4exp.cpp:83
    }

    /**
     * qwen4exp.cpp:86-146: the optional PLE module. {@code ple.layers} names exactly one layer; the n-gram
     * size is in {@code [2, MAX_PLE_NGRAM]}, the head count {@code (ngram - 1) * heads_per_ngram} in
     * {@code [1, MAX_PLE_HEADS]}, and the hash arrays cover the n-gram size / head count. Head token ranges
     * are stored as uint64 and must fit the int32 row index. Absent, every PLE field stays zero.
     */
    static void readPle(ModelHparamsBuilder b, GgufKv kv) {
        b.isPleLayerArray(new boolean[b.nLayerAll()]);                        // qwen4exp.cpp:86
        b.pleNHeads(0);                                                       // qwen4exp.cpp:87
        int nPle = Math.max(0, kv.arrayLength(GgufKeys.PLE_LAYERS));          // qwen4exp.cpp:89-90
        if (nPle == 0) {
            return;
        }
        int[] layers = kv.u32Array(GgufKeys.PLE_LAYERS);                      // qwen4exp.cpp:92-93
        if (nPle != 1) {                                                      // qwen4exp.cpp:94-98
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.PLE_LAYERS) + "' lists " + nPle
                    + " layers, but only one PLE layer is supported");
        }
        for (int il : layers) {                                               // qwen4exp.cpp:99-104
            if (il < 0 || il >= b.nLayerAll()) {
                throw HybridHooks.invalid(b, "PLE layer " + il + " is out of range");
            }
            b.isPleLayer(il, true);
        }
        b.pleNgramSize(kv.u32Required(GgufKeys.PLE_NGRAM_SIZE));              // qwen4exp.cpp:106
        b.pleHeadsPerNgram(kv.u32Required(GgufKeys.PLE_HEADS_PER_NGRAM));     // qwen4exp.cpp:107
        b.pleConvKernel(kv.u32Required(GgufKeys.PLE_CONV_KERNEL));            // qwen4exp.cpp:108
        b.pleEosTokenId(kv.u32Required(GgufKeys.PLE_EOS_TOKEN_ID));           // qwen4exp.cpp:109
        b.pleImageTokenId(kv.u32(GgufKeys.PLE_IMAGE_TOKEN_ID, b.pleImageTokenId()));      // qwen4exp.cpp:111
        b.nEmbdPerLayer(kv.u32Required(GgufKeys.EMBEDDING_LENGTH_PER_LAYER)); // qwen4exp.cpp:112
        HybridHooks.requireNonZero(kv, GgufKeys.PLE_CONV_KERNEL, b.pleConvKernel());          // qwen4exp.cpp:113
        HybridHooks.requireNonZero(kv, GgufKeys.EMBEDDING_LENGTH_PER_LAYER, b.nEmbdPerLayer());   // qwen4exp.cpp:114

        int ngram = b.pleNgramSize();
        if (ngram < 2 || ngram > MAX_PLE_NGRAM) {                             // qwen4exp.cpp:118-120
            throw HybridHooks.invalid(b, "PLE n-gram size " + ngram + " is out of range");
        }
        long nHeads = (long) (ngram - 1) * b.pleHeadsPerNgram();              // qwen4exp.cpp:116
        if (nHeads == 0 || nHeads > MAX_PLE_HEADS) {                          // qwen4exp.cpp:121-123
            throw HybridHooks.invalid(b, "PLE head count " + nHeads + " is out of range");
        }
        b.pleNHeads((int) nHeads);
        b.pleHeadDim(b.nEmbdPerLayer());                                      // qwen4exp.cpp:117

        b.pleLayerMultipliers(readU64(kv, GgufKeys.PLE_LAYER_MULTIPLIERS, ngram, MAX_PLE_NGRAM));   // qwen4exp.cpp:125, 129
        long[] offsets = readU64(kv, GgufKeys.PLE_HEAD_OFFSETS, (int) nHeads, MAX_PLE_HEADS);       // qwen4exp.cpp:126, 134
        long[] vocab = readU64(kv, GgufKeys.PLE_HEAD_VOCAB_SIZES, (int) nHeads, MAX_PLE_HEADS);     // qwen4exp.cpp:127, 135
        for (int h = 0; h < nHeads; h++) {                                    // qwen4exp.cpp:136-145
            if (vocab[h] == 0 || Long.compareUnsigned(offsets[h], Integer.MAX_VALUE) > 0
                    || Long.compareUnsigned(vocab[h], Integer.MAX_VALUE) > 0
                    || offsets[h] + vocab[h] > Integer.MAX_VALUE) {
                throw HybridHooks.invalid(b, "PLE head " + h + " range does not fit the int32 row index");
            }
        }
        b.pleHeadOffsets(Arrays.copyOf(offsets, (int) nHeads));
        b.pleHeadVocabSizes(Arrays.copyOf(vocab, (int) nHeads));
    }

    /**
     * The PLE convolution history is a row of the recurrent cache, so the PLE layer must be a linear
     * attention layer (qwen4exp.cpp:159-164).
     */
    static void requirePleRecurrent(ModelHparamsBuilder b) {
        for (int il = 0; il < b.nLayerAll(); il++) {
            if (b.isPleLayer(il) && !b.isRecurrent(il)) {
                throw HybridHooks.invalid(b, "PLE layer " + il + " is not a linear attention layer");
            }
        }
    }

    /**
     * A uint64 array with at least {@code nMin} entries (qwen4exp.cpp:17-24) and at most {@code nMax}
     * ({@code get_arr} into a fixed array, llama-model-loader.cpp:382-384).
     */
    private static long[] readU64(GgufKv kv, String keyFmt, int nMin, int nMax) {
        long[] v = kv.u64Array(keyFmt);
        if (v == null) {
            throw new GgufFormatException("required metadata key '" + kv.key(keyFmt) + "' is missing");
        }
        if (v.length < nMin) {
            throw new GgufFormatException("metadata key '" + kv.key(keyFmt) + "' has " + v.length
                    + " entries, but at least " + nMin + " are required");
        }
        if (v.length > nMax) {
            throw new GgufFormatException("metadata key '" + kv.key(keyFmt) + "' has " + v.length
                    + " entries, exceeding the maximum " + nMax);
        }
        return v;
    }
}
