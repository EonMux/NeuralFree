package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The {@code timed_trace} dataset, vLLM's {@code TimedTrace}: a JSONL replay of a timed request
 * trace. Each record names its arrival timestamp and input and output lengths, and its
 * {@code hash_ids} expand into synthetic prompt tokens, one deterministic chunk of
 * {@code --timed-trace-chunk-hash-size} tokens per hash. With {@code --self-timed} the timestamps
 * schedule the requests; without it the traffic flags do. The label flags select the record's field
 * names, and {@code --timed-trace-sec-multiplier} converts timestamps to seconds.
 */
final class TimedTraceDataset implements ServeBenchDataset {

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;
    private final int[] vocabulary;
    private final Map<String, int[]> expanded = new LinkedHashMap<>();

    TimedTraceDataset(ServeBenchOptions opts, BenchTokenizer tokenizer, int vocabSize, int[] skipIds) {
        this.opts = opts;
        this.tokenizer = tokenizer;
        List<Integer> allowed = new ArrayList<>(vocabSize);
        for (int id = 0; id < vocabSize; id++) {
            if (!ServeBenchDataset.skipped(id, skipIds)) {
                allowed.add(id);
            }
        }
        if (allowed.isEmpty()) {
            throw new IllegalArgumentException("the vocabulary of " + vocabSize
                    + " tokens has no allowed token left: the tokenizer reports only special ids");
        }
        this.vocabulary = new int[allowed.size()];
        for (int i = 0; i < vocabulary.length; i++) {
            vocabulary[i] = allowed.get(i);
        }
    }

    @Override
    public List<BenchSample> sample() {
        Path path = BenchDatasets.requiredPath(opts, "timed_trace");
        List<Map<String, Object>> rows;
        try {
            rows = BenchDatasets.readFile(path);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the timed_trace dataset " + path + ": "
                    + e.getMessage(), e);
        }
        List<BenchSample> out = new ArrayList<>();
        for (int ind = 0; ind < rows.size() && out.size() < opts.numPrompts(); ind++) {
            Map<String, Object> row = rows.get(ind);
            Object inputLength = row.get(opts.timedTraceLabelInputLength());
            if (!(inputLength instanceof Number)) {
                throw new IllegalArgumentException("the input length field '"
                        + opts.timedTraceLabelInputLength() + "' is missing from timed_trace record "
                        + (ind + 1) + " of " + path + "; available fields: " + row.keySet());
            }
            Object outputLength = row.get(opts.timedTraceLabelOutputLength());
            if (!(outputLength instanceof Number)) {
                throw new IllegalArgumentException("the output length field '"
                        + opts.timedTraceLabelOutputLength() + "' is missing from timed_trace record "
                        + (ind + 1) + " of " + path + "; available fields: " + row.keySet());
            }
            Object timestamp = row.get(opts.timedTraceLabelTimestamp());
            if (!(timestamp instanceof Number)) {
                throw new IllegalArgumentException("the timestamp field '"
                        + opts.timedTraceLabelTimestamp() + "' is missing from timed_trace record "
                        + (ind + 1) + " of " + path + "; available fields: " + row.keySet());
            }
            int[] prompt = expand(row.get(opts.timedTraceLabelHashIds()),
                    ((Number) inputLength).intValue());
            BenchSample sample = BenchSample.ofTokens(prompt, ((Number) outputLength).intValue(),
                    opts.requestIdPrefix() + ind);
            out.add(sample.withTimestamp(((Number) timestamp).doubleValue() * opts.timedTraceSecMultiplier()));
        }
        return out;
    }

    /** The prompt tokens of one record's hashes, each chunk cached by its hash and size. */
    private int[] expand(Object hashes, int targetLength) {
        if (hashes == null) {
            return new int[0];
        }
        if (!(hashes instanceof List)) {
            throw new IllegalArgumentException("the '" + opts.timedTraceLabelHashIds()
                    + "' field of a timed_trace record is an array, not " + hashes.getClass().getSimpleName());
        }
        List<Integer> prompt = new ArrayList<>();
        int remaining = targetLength;
        for (Object hash : (List<?>) hashes) {
            if (remaining <= 0) {
                break;
            }
            int size = Math.min(opts.timedTraceChunkHashSize(), remaining);
            String key = String.valueOf(hash) + ":" + size;
            int[] chunk = expanded.get(key);
            if (chunk == null) {
                long keySeed = String.valueOf(key).hashCode() & 0xFFFFFFFFL;
                List<Integer> population = new ArrayList<>(vocabulary.length);
                for (int id : vocabulary) {
                    population.add(id);
                }
                List<Integer> drawn = new PythonRandom(keySeed).choices(population, size);
                chunk = new int[size];
                for (int i = 0; i < size; i++) {
                    chunk[i] = drawn.get(i);
                }
                expanded.put(key, chunk);
            }
            for (int id : chunk) {
                prompt.add(id);
            }
            remaining -= size;
        }
        int[] out = new int[prompt.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = prompt.get(i);
        }
        return out;
    }
}
