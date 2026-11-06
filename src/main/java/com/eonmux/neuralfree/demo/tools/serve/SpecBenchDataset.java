package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The {@code spec_bench} dataset, vLLM's {@code SpecBench}: a JSONL file of Spec-Bench questions
 * whose {@code turns} array carries the prompt and whose {@code category} filters the sample. The
 * records shuffle under the seed and the workload oversamples to {@code num-prompts}; every request
 * asks for {@code --spec-bench-output-len} tokens (256 by default).
 */
final class SpecBenchDataset implements ServeBenchDataset {

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;

    SpecBenchDataset(ServeBenchOptions opts, BenchTokenizer tokenizer) {
        this.opts = opts;
        this.tokenizer = tokenizer;
    }

    @Override
    public List<BenchSample> sample() {
        Path path = BenchDatasets.requiredPath(opts, "spec_bench");
        List<Map<String, Object>> rows;
        try {
            rows = BenchDatasets.readFile(path);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the spec_bench dataset " + path + ": "
                    + e.getMessage(), e);
        }
        List<String> prompts = new ArrayList<>();
        for (int i = 0; i < rows.size(); i++) {
            Map<String, Object> row = rows.get(i);
            if (!(row.get("turns") instanceof List)) {
                throw new IllegalArgumentException("every spec_bench record needs a 'turns' array, record "
                        + (i + 1) + " of " + path + " has none");
            }
            List<?> turns = (List<?>) row.get("turns");
            if (turns.isEmpty() || !(turns.get(0) instanceof String)) {
                throw new IllegalArgumentException("record " + (i + 1) + " of " + path
                        + " needs a first turn string");
            }
            Object category = row.get("category");
            if (opts.specBenchCategory() == null || opts.specBenchCategory().equals(category)) {
                prompts.add((String) turns.get(0));
            }
        }
        BenchDatasets.shuffle(prompts, opts.seed(), opts.disableShuffle());
        int target = BenchDatasets.targetNum(opts, prompts.size());
        List<BenchSample> out = new ArrayList<>();
        for (int i = 0; i < prompts.size() && out.size() < target; i++) {
            String prompt = BenchDatasets.renderPrompt(opts, tokenizer, prompts.get(i));
            out.add(BenchSample.ofText(prompt, BenchDatasets.promptLen(opts, tokenizer, prompt),
                    opts.specBenchOutputLen(), opts.requestIdPrefix() + i));
        }
        return ServeBenchDataset.oversample(out, target, opts.requestIdPrefix(),
                opts.noOversample(), opts.seed());
    }
}
