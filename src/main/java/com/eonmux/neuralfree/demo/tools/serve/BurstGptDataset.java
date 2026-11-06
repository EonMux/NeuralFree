package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * The {@code burstgpt} dataset, vLLM's {@code BurstGptDataset}: a BurstGPT CSV log (the header
 * {@code Timestamp,Model,Request tokens,Response tokens}), filtered to the {@code GPT-4} rows with a
 * positive response length, sampled with replacement under the seed to {@code numPrompts}. Each row
 * becomes a synthetic token-id prompt of the row's request length — {@code (i + j) % vocabSize} —
 * and asks for the row's response length, so the workload replays the trace's length distribution.
 */
final class BurstGptDataset implements ServeBenchDataset {

    /** The CSV columns the trace reads, in order. */
    static final String HEADER = "Timestamp,Model,Request tokens,Response tokens";

    private final ServeBenchOptions opts;
    private final int vocabSize;

    BurstGptDataset(ServeBenchOptions opts, int vocabSize) {
        this.opts = opts;
        this.vocabSize = vocabSize;
    }

    @Override
    public List<BenchSample> sample() {
        Path path = datasetPath();
        List<long[]> rows;
        try {
            rows = load(path);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the burstgpt dataset " + path + ": " + e.getMessage(), e);
        }
        if (rows.isEmpty()) {
            throw new IllegalStateException("the burstgpt dataset " + path
                    + " carries no GPT-4 row with a positive response length");
        }
        Collections.shuffle(rows, new Random(opts.seed()));
        List<BenchSample> out = new ArrayList<>(opts.numPrompts());
        for (int i = 0; i < opts.numPrompts(); i++) {
            long[] row = rows.get(i % rows.size());
            int inputLen = (int) row[0];
            int outputLen = (int) row[1];
            int[] tokens = new int[inputLen];
            for (int j = 0; j < inputLen; j++) {
                tokens[j] = (i + j) % vocabSize;
            }
            out.add(BenchSample.ofTokens(tokens, outputLen, opts.requestIdPrefix() + i));
        }
        return out;
    }

    /**
     * The trace: the header row, then every {@code GPT-4} row with a positive response length as a
     * {@code [request tokens, response tokens]} pair. A short or non-numeric row names its line.
     */
    private List<long[]> load(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        List<long[]> out = new ArrayList<>();
        boolean headerSeen = false;
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            String[] fields = line.split(",");
            if (!headerSeen) {
                if (fields.length < 2 || !fields[1].trim().equalsIgnoreCase("Model")) {
                    throw new IOException("the burstgpt dataset needs the header " + HEADER + ", line 1 of "
                            + path + " reads '" + line + "'");
                }
                headerSeen = true;
                continue;
            }
            if (fields.length < 4) {
                throw new IOException("line " + (i + 1) + " of " + path + " carries " + fields.length
                        + " fields, the trace needs 4");
            }
            if (!fields[1].trim().equals("GPT-4")) {
                continue;
            }
            long requestTokens;
            long responseTokens;
            try {
                requestTokens = Long.parseLong(fields[2].trim());
                responseTokens = Long.parseLong(fields[3].trim());
            } catch (NumberFormatException e) {
                throw new IOException("line " + (i + 1) + " of " + path + " carries non-integer token counts: '"
                        + line + "'");
            }
            if (responseTokens > 0) {
                out.add(new long[]{requestTokens, responseTokens});
            }
        }
        return out;
    }

    /** The required {@code --dataset-path}, refused by name when absent or missing. */
    private Path datasetPath() {
        if (opts.datasetPath() == null) {
            throw new IllegalArgumentException("--dataset-path must be provided for the burstgpt dataset");
        }
        Path path = Paths.get(opts.datasetPath());
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("the burstgpt dataset file does not exist: " + path);
        }
        return path;
    }
}
