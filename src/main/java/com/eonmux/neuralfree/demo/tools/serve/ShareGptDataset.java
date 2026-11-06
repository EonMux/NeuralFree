package com.eonmux.neuralfree.demo.tools.serve;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The {@code sharegpt} dataset, vLLM's {@code ShareGptDataset}: it loads a ShareGPT JSON file (an
 * array of conversations, each with a {@code conversations} array of {@code from}/{@code value}
 * turns), keeps the conversations of at least two turns, shuffles them under the seed, and turns
 * the first turn into the prompt and the second turn's token count into the output length (the
 * {@code --sharegpt-output-len} flag fixes it instead). Sequences the reference's bounds prune are
 * dropped, and the workload oversamples to {@code numPrompts}.
 */
final class ShareGptDataset implements ServeBenchDataset {

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;

    ShareGptDataset(ServeBenchOptions opts, BenchTokenizer tokenizer) {
        this.opts = opts;
        this.tokenizer = tokenizer;
    }

    @Override
    public List<BenchSample> sample() {
        Path path = datasetPath();
        List<Map<?, ?>> entries;
        try {
            entries = load(path);
        } catch (IOException | JsonException e) {
            throw new IllegalStateException("cannot read the sharegpt dataset " + path + ": " + e.getMessage(), e);
        }
        List<Map<?, ?>> data = new ArrayList<>(entries);
        if (!opts.disableShuffle()) {
            BenchDatasets.shuffle(data, opts.seed(), false);
        }
        boolean fixedOutput = opts.sharegptOutputLen() != null;
        List<BenchSample> out = new ArrayList<>();
        int index = 0;
        for (Map<?, ?> entry : data) {
            if (out.size() >= opts.numPrompts()) {
                break;
            }
            Object conversations = entry.get("conversations");
            if (!(conversations instanceof List) || ((List<?>) conversations).size() < 2) {
                continue;
            }
            List<?> turns = (List<?>) conversations;
            String prompt = turnValue(turns.get(0));
            String completion = turnValue(turns.get(1));
            if (prompt == null || completion == null) {
                continue;
            }
            int outputLen = fixedOutput ? opts.sharegptOutputLen() : tokenizer.tokenize(completion).length;
            int promptLen = tokenizer.tokenize(prompt).length;
            if (!ServeBenchDataset.isValidSequence(promptLen, outputLen, fixedOutput)) {
                continue;
            }
            out.add(BenchSample.ofText(prompt, promptLen, outputLen, opts.requestIdPrefix() + index));
            index++;
        }
        return ServeBenchDataset.oversample(out, opts.numPrompts(), opts.requestIdPrefix(),
                opts.noOversample(), opts.seed());
    }

    /** The ShareGPT file: a JSON array of objects; anything else names the file and the shape. */
    private List<Map<?, ?>> load(Path path) throws IOException, JsonException {
        Object parsed = Json.parse(Files.readString(path), Json.Limits.UNTRUSTED);
        if (!(parsed instanceof List)) {
            throw new IOException("the sharegpt dataset must be a JSON array of conversations, not "
                    + parsed.getClass().getSimpleName());
        }
        List<Map<?, ?>> out = new ArrayList<>();
        for (Object entry : (List<?>) parsed) {
            if (!(entry instanceof Map)) {
                throw new IOException("every sharegpt entry must be an object");
            }
            out.add((Map<?, ?>) entry);
        }
        return out;
    }

    /** A turn's {@code value}, or null when the turn is not an object carrying one. */
    private static String turnValue(Object turn) {
        if (!(turn instanceof Map)) {
            return null;
        }
        Object value = ((Map<?, ?>) turn).get("value");
        return value instanceof String ? (String) value : null;
    }

    /** The required {@code --dataset-path}, refused by name when absent. */
    private Path datasetPath() {
        if (opts.datasetPath() == null) {
            throw new IllegalArgumentException("--dataset-path must be provided for the sharegpt dataset");
        }
        Path path = Paths.get(opts.datasetPath());
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("the sharegpt dataset file does not exist: " + path);
        }
        return path;
    }
}
