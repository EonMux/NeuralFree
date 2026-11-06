package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The {@code random-rerank} dataset, vLLM's {@code RandomDatasetForReranking}: synthetic rerank
 * batches whose query and documents walk the allowed vocabulary exactly as the random dataset's
 * prompts do. A reranker request carries one query and {@code --random-batch-size} documents
 * ({@code --no-reranker} instead carries the query once and the documents of the reduced batch);
 * the sampled ids are decoded to text through the tokenizer's {@code detokenize}.
 */
final class RandomRerankDataset implements ServeBenchDataset {

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;
    private final int[] allowed;

    RandomRerankDataset(ServeBenchOptions opts, BenchTokenizer tokenizer, int vocabSize, int[] skipIds) {
        this.opts = opts;
        this.tokenizer = tokenizer;
        this.allowed = SyntheticSampler.allowed(vocabSize, skipIds);
    }

    @Override
    public List<BenchSample> sample() {
        boolean reranker = !opts.noReranker();
        int batchSize = Math.max(1, opts.randomBatchSize());
        int numRequests = opts.numPrompts();
        if (!reranker) {
            if (numRequests <= 1 || batchSize <= 1) {
                throw new IllegalArgumentException("--no-reranker needs more than one prompt and a"
                        + " --random-batch-size greater than one, got " + numRequests + " and "
                        + batchSize);
            }
            numRequests--;
            batchSize--;
        }
        int separator = reranker ? 1 : 0;
        int inputLen = opts.inputLen() != null ? opts.inputLen() : opts.randomInputLen();
        int queryLenParam = reranker ? inputLen / 2 - separator : inputLen;
        Random random = new Random(opts.seed());

        SyntheticSampler.Draw[] query = SyntheticSampler.draws(opts, random, 1, queryLenParam, 0);
        SyntheticSampler.drawOffsets(query, random, allowed.length);
        int queryLen = query[0].inputLen;
        int documentLenParam = reranker ? inputLen - queryLen - separator : inputLen;
        if (documentLenParam < 1) {
            throw new IllegalArgumentException("--random-input-len " + inputLen + " leaves no room for a"
                    + " document beside the " + queryLen + "-token query; raise it");
        }
        SyntheticSampler.Draw[] documents = SyntheticSampler.draws(opts, random, numRequests,
                documentLenParam, 0);
        SyntheticSampler.drawOffsets(documents, random, allowed.length);

        String queryText = SyntheticSampler.text(tokenizer,
                SyntheticSampler.walk(allowed, query[0].offset, 0, queryLen));
        List<String> documentTexts = new ArrayList<>(documents.length);
        int[] documentLens = new int[documents.length];
        for (int i = 0; i < documents.length; i++) {
            int[] ids = SyntheticSampler.walk(allowed, documents[i].offset, i + 1,
                    documents[i].inputLen);
            documentLens[i] = ids.length;
            documentTexts.add(SyntheticSampler.text(tokenizer, ids));
        }

        List<BenchSample> out = new ArrayList<>();
        for (int i = 0; i < numRequests; i += batchSize) {
            List<String> texts = new ArrayList<>();
            texts.add(queryText);
            int promptLen = reranker ? (queryLen + separator) : queryLen;
            for (int d = i; d < Math.min(i + batchSize, numRequests); d++) {
                texts.add(documentTexts.get(d));
                promptLen += documentLens[d] + (reranker ? separator : 0);
            }
            out.add(BenchSample.ofRerank(texts, promptLen, opts.requestIdPrefix() + (i / batchSize)));
        }
        return out;
    }
}
