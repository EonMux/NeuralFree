package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The {@code blazedit} dataset, vLLM's {@code BlazeditDataset} (the {@code vdaita/edit_5k_char} and
 * {@code edit_10k_char} datasets): every record pairs a Python file with a change request and the
 * normalized Levenshtein distance between them, and the prompt is the reference's edit template.
 * Records outside {@code --blazedit-min-distance} and {@code --blazedit-max-distance} are skipped;
 * the default output length is 4000 tokens.
 */
final class BlazeditDataset implements ServeBenchDataset {

    /** The reference's default output length: a 10k character edit at three characters per token. */
    static final int DEFAULT_OUTPUT_LEN = 4000;

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;

    BlazeditDataset(ServeBenchOptions opts, BenchTokenizer tokenizer) {
        this.opts = opts;
        this.tokenizer = tokenizer;
    }

    @Override
    public List<BenchSample> sample() {
        Path path = BenchDatasets.requiredPath(opts, "blazedit");
        List<Map<String, Object>> rows;
        try {
            rows = BenchDatasets.readRows(path);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the blazedit dataset " + path + ": "
                    + e.getMessage(), e);
        }
        BenchDatasets.shuffle(rows, opts.seed(), opts.disableShuffle());
        int outputLen = opts.hfOutputLen() != null ? opts.hfOutputLen() : DEFAULT_OUTPUT_LEN;
        List<BenchSample> out = new ArrayList<>();
        int index = 0;
        for (Map<String, Object> row : rows) {
            if (out.size() >= opts.numPrompts()) {
                break;
            }
            String code = BenchDatasets.string(row, "code", "a blazedit record");
            String change = BenchDatasets.string(row, "change_request", "a blazedit record");
            double distance = BenchDatasets.number(row, "norm_distance", "a blazedit record");
            if (distance < opts.blazeditMinDistance() || distance > opts.blazeditMaxDistance()) {
                continue;
            }
            String prompt = "Given a code file, please apply the change requests and generate the new"
                    + " file.\n\nOriginal file:\n```python\n" + code + "\n```\n\nChange request:\n"
                    + change + "\n\nPlease generate the new code file in the \"New file\" section below.";
            out.add(BenchSample.ofText(prompt, BenchDatasets.promptLen(opts, tokenizer, prompt),
                    outputLen, opts.requestIdPrefix() + index));
            index++;
        }
        return ServeBenchDataset.oversample(out, opts.numPrompts(), opts.requestIdPrefix(),
                opts.noOversample(), opts.seed());
    }

    /** The distance range flag's check: both bounds inside {@code [0, 1]} and ordered. */
    static void validateDistance(double min, double max) {
        if (!(min >= 0.0 && min <= 1.0) || !(max >= 0.0 && max <= 1.0)) {
            throw new IllegalArgumentException("the blazedit distances must be in [0, 1], got "
                    + String.format(Locale.ROOT, "[%s, %s]", min, max));
        }
        if (min > max) {
            throw new IllegalArgumentException("the blazedit minimum distance cannot exceed the maximum,"
                    + " got [" + min + ", " + max + "]");
        }
    }
}
