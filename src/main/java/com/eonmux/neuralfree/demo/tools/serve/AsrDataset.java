package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The {@code asr} dataset, vLLM's {@code ASRDataset} over a local dataset directory: records carry
 * an audio path (and a transcript, which the benchmark does not send) and every request uploads its
 * file to {@code /v1/audio/transcriptions}. The records shuffle under the seed; audio shorter than
 * {@code --asr-min-audio-len-sec} or longer than {@code --asr-max-audio-len-sec} is skipped, and the
 * durations feed the run's RTFx. The default output budget is 1024 tokens.
 */
final class AsrDataset implements ServeBenchDataset {

    /** The reference's default transcription budget. */
    static final int DEFAULT_OUTPUT_LEN = 1024;

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;

    AsrDataset(ServeBenchOptions opts, BenchTokenizer tokenizer) {
        this.opts = opts;
        this.tokenizer = tokenizer;
    }

    @Override
    public List<BenchSample> sample() {
        Path path = BenchDatasets.requiredPath(opts, "asr");
        List<Map<String, Object>> rows;
        try {
            rows = BenchDatasets.readRows(path);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the asr dataset " + path + ": " + e.getMessage(), e);
        }
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("the asr dataset " + path + " carries no record");
        }
        BenchDatasets.shuffle(rows, opts.seed(), opts.disableShuffle());
        Path baseDir = BenchDatasets.baseDir(path);
        int outputLen = opts.hfOutputLen() != null ? opts.hfOutputLen() : DEFAULT_OUTPUT_LEN;
        int promptLen = opts.skipTokenizerInit() ? 1 : tokenizer.tokenize("").length;
        List<BenchSample> out = new ArrayList<>();
        int index = 0;
        for (Map<String, Object> row : rows) {
            if (out.size() >= opts.numPrompts()) {
                break;
            }
            if (!row.containsKey("audio")) {
                throw new IllegalArgumentException("every asr record needs an 'audio' field, one of "
                        + path + " has none");
            }
            BenchDatasets.AudioFile audio = BenchDatasets.audio(row.get("audio"), baseDir);
            if (audio.durationSec < opts.asrMinAudioLenSec()
                    || audio.durationSec > opts.asrMaxAudioLenSec()) {
                continue;
            }
            out.add(BenchSample.ofAudio(audio.path, audio.durationSec, promptLen, outputLen,
                    opts.requestIdPrefix() + index));
            index++;
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("every asr record of " + path + " lies outside the"
                    + " configured audio length range [" + opts.asrMinAudioLenSec() + ", "
                    + opts.asrMaxAudioLenSec() + "] seconds");
        }
        return ServeBenchDataset.oversample(out, opts.numPrompts(), opts.requestIdPrefix(),
                opts.noOversample(), opts.seed());
    }
}
