package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The {@code custom_audio} dataset, vLLM's {@code CustomAudioDataset}: a JSONL file whose records
 * carry an audio path (and an optional prompt and {@code output_tokens} count). The records shuffle
 * under the seed, every request uploads its file to {@code /v1/audio/transcriptions}, and the audio
 * duration feeds the run's RTFx. Without {@code --custom-output-len} (-1) each record's
 * {@code output_tokens} supplies the output budget.
 */
final class CustomAudioDataset implements ServeBenchDataset {

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;

    CustomAudioDataset(ServeBenchOptions opts, BenchTokenizer tokenizer) {
        this.opts = opts;
        this.tokenizer = tokenizer;
    }

    @Override
    public List<BenchSample> sample() {
        Path path = BenchDatasets.requiredPath(opts, "custom_audio");
        if (!path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".jsonl")) {
            throw new IllegalArgumentException("only the JSONL format is supported for the custom_audio"
                    + " dataset, not '" + path + "'");
        }
        List<Map<String, Object>> rows;
        try {
            rows = BenchDatasets.readFile(path);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the custom_audio dataset " + path + ": "
                    + e.getMessage(), e);
        }
        BenchDatasets.shuffle(rows, opts.seed(), opts.disableShuffle());
        Path baseDir = BenchDatasets.baseDir(path);
        boolean outputFromRecord = opts.customOutputLen() < 0;
        int target = BenchDatasets.targetNum(opts, rows.size());
        List<BenchSample> out = new ArrayList<>();
        for (int i = 0; i < rows.size() && out.size() < target; i++) {
            Map<String, Object> row = rows.get(i);
            String prompt = BenchDatasets.optionalString(row, "prompt");
            if (prompt == null) {
                prompt = "";
            }
            int outputLen;
            if (opts.skipTokenizerInit()) {
                outputLen = opts.customOutputLen() >= 0 ? opts.customOutputLen() : 256;
            } else if (outputFromRecord) {
                if (!(row.get("output_tokens") instanceof Number)) {
                    throw new IllegalArgumentException("if no output length is provided the"
                            + " custom_audio dataset must contain an 'output_tokens' field; record "
                            + (i + 1) + " of " + path + " carries '" + row.get("output_tokens") + "'");
                }
                outputLen = ((Number) row.get("output_tokens")).intValue();
            } else {
                outputLen = opts.customOutputLen();
            }
            BenchDatasets.AudioFile audio = BenchDatasets.audio(row.get("audio"), baseDir);
            out.add(BenchSample.ofAudio(audio.path, audio.durationSec,
                    BenchDatasets.promptLen(opts, tokenizer, prompt), outputLen,
                    opts.requestIdPrefix() + i));
        }
        return ServeBenchDataset.oversample(out, target, opts.requestIdPrefix(),
                opts.noOversample(), opts.seed());
    }
}
