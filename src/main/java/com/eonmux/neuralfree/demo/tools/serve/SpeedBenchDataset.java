package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The {@code speed_bench} dataset, vLLM's {@code SpeedBench}: a local SPEED-Bench directory holding
 * {@code <subset>.jsonl}; every record's first message carries the prompt and its {@code category}
 * filters the sample. The records shuffle under the seed and every request asks for
 * {@code --speed-bench-output-len} tokens (4096 by default).
 */
final class SpeedBenchDataset implements ServeBenchDataset {

    /** The reference's default SPEED-Bench output budget. */
    static final int DEFAULT_OUTPUT_LEN = 4096;

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;

    SpeedBenchDataset(ServeBenchOptions opts, BenchTokenizer tokenizer) {
        this.opts = opts;
        this.tokenizer = tokenizer;
    }

    @Override
    public List<BenchSample> sample() {
        Path dir = BenchDatasets.requiredPath(opts, "speed_bench");
        if (!Files.isDirectory(dir)) {
            throw new IllegalArgumentException("the speed_bench dataset path is a directory holding "
                    + opts.speedBenchDatasetSubset() + ".jsonl, not " + dir);
        }
        Path file = dir.resolve(opts.speedBenchDatasetSubset() + ".jsonl");
        if (!Files.isRegularFile(file)) {
            throw new IllegalArgumentException("the speed_bench dataset " + dir + " has no "
                    + opts.speedBenchDatasetSubset() + ".jsonl");
        }
        List<Map<String, Object>> rows;
        try {
            rows = BenchDatasets.readFile(file);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the speed_bench dataset " + file + ": "
                    + e.getMessage(), e);
        }
        List<String> prompts = new ArrayList<>();
        for (int i = 0; i < rows.size(); i++) {
            Map<String, Object> row = rows.get(i);
            if (!(row.get("messages") instanceof List)) {
                throw new IllegalArgumentException("every speed_bench record needs a 'messages' array,"
                        + " record " + (i + 1) + " of " + file + " has none");
            }
            List<?> messages = (List<?>) row.get("messages");
            if (messages.isEmpty() || !(messages.get(0) instanceof Map)) {
                throw new IllegalArgumentException("record " + (i + 1) + " of " + file
                        + " needs a first message object");
            }
            Object content = ((Map<?, ?>) messages.get(0)).get("content");
            if (!(content instanceof String)) {
                throw new IllegalArgumentException("record " + (i + 1) + " of " + file
                        + " needs a string 'content' in its first message");
            }
            Object category = row.get("category");
            if (opts.speedBenchCategory() == null || opts.speedBenchCategory().equals(category)) {
                prompts.add((String) content);
            }
        }
        BenchDatasets.shuffle(prompts, opts.seed(), opts.disableShuffle());
        int target = BenchDatasets.targetNum(opts, prompts.size());
        List<BenchSample> out = new ArrayList<>();
        for (int i = 0; i < prompts.size() && out.size() < target; i++) {
            String prompt = BenchDatasets.renderPrompt(opts, tokenizer, prompts.get(i));
            out.add(BenchSample.ofText(prompt, BenchDatasets.promptLen(opts, tokenizer, prompt),
                    opts.speedBenchOutputLen(), opts.requestIdPrefix() + i));
        }
        return ServeBenchDataset.oversample(out, target, opts.requestIdPrefix(),
                opts.noOversample(), opts.seed());
    }
}
