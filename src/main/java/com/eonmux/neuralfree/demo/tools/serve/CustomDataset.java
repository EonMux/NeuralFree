package com.eonmux.neuralfree.demo.tools.serve;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The {@code custom} dataset, vLLM's {@code CustomDataset}: a JSONL file whose records carry a
 * {@code prompt} and, when no {@code --custom-output-len} fixes the length, an {@code output_tokens}
 * count. The records shuffle under the seed, the prompts count through the server's tokenizer, and
 * the workload oversamples to {@code numPrompts}. A record without its {@code output_tokens} where
 * one is needed is refused with its record number.
 */
final class CustomDataset implements ServeBenchDataset {

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;

    CustomDataset(ServeBenchOptions opts, BenchTokenizer tokenizer) {
        this.opts = opts;
        this.tokenizer = tokenizer;
    }

    @Override
    public List<BenchSample> sample() {
        Path path = datasetPath();
        List<Map<?, ?>> data;
        try {
            data = load(path);
        } catch (IOException | JsonException e) {
            throw new IllegalStateException("cannot read the custom dataset " + path + ": " + e.getMessage(), e);
        }
        if (!opts.disableShuffle()) {
            BenchDatasets.shuffle(data, opts.seed(), false);
        }
        int fixedOutput = opts.customOutputLen();
        boolean outputFromRecord = fixedOutput < 0;
        boolean noTokenizer = opts.skipTokenizerInit();
        int target = BenchDatasets.targetNum(opts, data.size());
        List<BenchSample> out = new ArrayList<>();
        int index = 0;
        for (Map<?, ?> record : data) {
            if (out.size() >= target) {
                break;
            }
            Object prompt = record.get("prompt");
            if (!(prompt instanceof String)) {
                throw new IllegalArgumentException("every record of a custom dataset needs a 'prompt' string"
                        + ", record " + index + " of " + path + " has none");
            }
            int outputLen;
            if (noTokenizer) {
                outputLen = 1;
            } else if (outputFromRecord) {
                Object fromRecord = record.get("output_tokens");
                if (!(fromRecord instanceof Number)) {
                    throw new IllegalArgumentException("if no output length is provided the custom dataset"
                            + " must contain an 'output_tokens' field; record " + index + " of " + path
                            + " carries '" + fromRecord + "'");
                }
                outputLen = ((Number) fromRecord).intValue();
            } else {
                outputLen = fixedOutput;
            }
            String text = noTokenizer ? (String) prompt
                    : BenchDatasets.renderPrompt(opts, tokenizer, (String) prompt);
            int promptLen = noTokenizer ? 1 : tokenizer.tokenize(text).length;
            out.add(BenchSample.ofText(text, promptLen, outputLen, opts.requestIdPrefix() + index));
            index++;
        }
        return ServeBenchDataset.oversample(out, target, opts.requestIdPrefix(),
                opts.noOversample(), opts.seed());
    }

    /** The JSONL file: one JSON object per line; every other line names its number. */
    private List<Map<?, ?>> load(Path path) throws IOException, JsonException {
        List<Map<?, ?>> out = new ArrayList<>();
        List<String> lines = Files.readAllLines(path);
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            Object parsed = Json.parse(line, Json.Limits.UNTRUSTED);
            if (!(parsed instanceof Map)) {
                throw new IOException("line " + (i + 1) + " of " + path + " is not a JSON object");
            }
            out.add((Map<?, ?>) parsed);
        }
        if (out.isEmpty()) {
            throw new IOException("the custom dataset " + path + " carries no record");
        }
        return out;
    }

    /** The required {@code --dataset-path}, refused by name when absent or missing. */
    private Path datasetPath() {
        if (opts.datasetPath() == null) {
            throw new IllegalArgumentException("--dataset-path must be provided for the custom dataset");
        }
        String lower = opts.datasetPath().toLowerCase(java.util.Locale.ROOT);
        if (!lower.endsWith(".jsonl")) {
            throw new IllegalArgumentException("only the JSONL format is supported for the custom dataset,"
                    + " not '" + opts.datasetPath() + "'");
        }
        Path path = Paths.get(opts.datasetPath());
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("the custom dataset file does not exist: " + path);
        }
        return path;
    }
}
