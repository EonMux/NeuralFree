package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * The workload of a benchmark run: a list of {@link BenchSample}s, as vLLM's datasets produce a
 * list of {@code SampleRequest}s. The name selects the dataset ({@link #NAMES}): {@code random},
 * {@code random-mm}, {@code random-rerank} and {@code prefix_repetition} synthesize their prompts,
 * the file and directory datasets load {@code --dataset-path} and count their text prompts with the
 * tokenizer, and the multimodal and audio datasets build chat content parts or transcription
 * uploads. Every dataset is seeded, so the same flags and seed sample the same workload.
 */
public interface ServeBenchDataset {

    /** The dataset names the commands accept ({@code --dataset-name}). */
    List<String> NAMES = List.of("random", "random-mm", "random-rerank", "prefix_repetition",
            "sharegpt", "custom", "custom_image", "custom_audio", "burstgpt", "spec_bench",
            "speed_bench", "timed_trace", "blazedit", "asr", "hf");

    /** The names whose prompts are synthesized, which refuse {@code --dataset-path} as vLLM refuses it. */
    List<String> SYNTHETIC = List.of("random", "random-mm", "random-rerank", "prefix_repetition");

    /** The datasets whose requests target a chat endpoint. */
    List<String> CHAT = List.of("random-mm", "custom_image");

    /** The datasets whose requests target {@code /rerank}. */
    List<String> RERANK = List.of("random-rerank");

    /** The datasets whose requests target {@code /v1/audio/transcriptions}. */
    List<String> AUDIO = List.of("custom_audio", "asr");

    /** The datasets the reference samples without a tokenizer ({@code --skip-tokenizer-init}). */
    List<String> TOKENIZER_FREE = List.of("custom", "custom_image", "custom_audio");

    /** The datasets whose prompts are token ids, which only the completions endpoint carries. */
    List<String> TOKEN_PROMPTS = List.of("random", "prefix_repetition", "timed_trace", "burstgpt");

    /** The default pruning of the file datasets: vLLM's {@code is_valid_sequence} bounds. */
    int MIN_LEN = 4;
    int MAX_PROMPT_LEN = 1024;
    int MAX_TOTAL_LEN = 2048;

    /**
     * The workload: the file datasets oversample to {@code numPrompts} unless {@code --no-oversample},
     * the random dataset yields exactly {@code numPrompts}, and {@code prefix_repetition} yields whole
     * prefix groups ({@code numPrompts / numPrefixes} requests per prefix).
     */
    List<BenchSample> sample();

    /**
     * Builds the dataset named {@code name} over the options' dataset flags.
     *
     * @param tokenizer the server-side prompt counter, used by the file datasets
     * @param vocabSize the served vocabulary size ({@code /tokenizer_info})
     * @param skipIds   the token ids a synthetic prompt never carries (the special and stop ids)
     * @throws IllegalArgumentException for a name outside {@link #NAMES} or a missing
     *                                  {@code --dataset-path} where one is required
     */
    static ServeBenchDataset create(String name, ServeBenchOptions opts, BenchTokenizer tokenizer,
                                    int vocabSize, int[] skipIds) {
        switch (name) {
            case "random":
                return new RandomDataset(opts, vocabSize, skipIds);
            case "random-mm":
                return new RandomMultiModalDataset(opts, tokenizer, vocabSize, skipIds);
            case "random-rerank":
                return new RandomRerankDataset(opts, tokenizer, vocabSize, skipIds);
            case "prefix_repetition":
                return new PrefixRepetitionDataset(opts, vocabSize, skipIds);
            case "sharegpt":
                return new ShareGptDataset(opts, tokenizer);
            case "custom":
                return new CustomDataset(opts, tokenizer);
            case "custom_image":
                return new CustomImageDataset(opts, tokenizer);
            case "custom_audio":
                return new CustomAudioDataset(opts, tokenizer);
            case "burstgpt":
                return new BurstGptDataset(opts, vocabSize);
            case "spec_bench":
                return new SpecBenchDataset(opts, tokenizer);
            case "speed_bench":
                return new SpeedBenchDataset(opts, tokenizer);
            case "timed_trace":
                return new TimedTraceDataset(opts, tokenizer, vocabSize, skipIds);
            case "blazedit":
                return new BlazeditDataset(opts, tokenizer);
            case "asr":
                return new AsrDataset(opts, tokenizer);
            case "hf":
                return new HfDataset(opts, tokenizer);
            default:
                throw new IllegalArgumentException("unknown dataset name '" + name + "': " + NAMES);
        }
    }

    /**
     * Whether a file dataset keeps a sequence: none of vLLM's pruning bounds trips — a prompt of at
     * least {@value MIN_LEN} tokens, an output of at least {@value MIN_LEN} tokens (skipped when the
     * caller fixed the output length), a prompt of at most {@value MAX_PROMPT_LEN} and a total of at
     * most {@value MAX_TOTAL_LEN}.
     */
    static boolean isValidSequence(int promptLen, int outputLen, boolean skipMinOutputLenCheck) {
        boolean promptTooShort = promptLen < MIN_LEN;
        boolean outputTooShort = !skipMinOutputLenCheck && outputLen < MIN_LEN;
        boolean promptTooLong = promptLen > MAX_PROMPT_LEN;
        boolean combinedTooLong = promptLen + outputLen > MAX_TOTAL_LEN;
        return !(promptTooShort || outputTooShort || promptTooLong || combinedTooLong);
    }

    /** {@link #isValidSequence(int, int, boolean)} with the reference's bounds supplied by the caller. */
    static boolean isValidSequence(int promptLen, int outputLen, int minLen, int maxPromptLen,
                                   int maxTotalLen) {
        boolean promptTooShort = promptLen < minLen;
        boolean outputTooShort = outputLen < minLen;
        boolean promptTooLong = promptLen > maxPromptLen;
        boolean combinedTooLong = promptLen + outputLen > maxTotalLen;
        return !(promptTooShort || outputTooShort || promptTooLong || combinedTooLong);
    }

    /**
     * Repeats sampled requests until {@code numRequests} of them exist (vLLM's
     * {@code maybe_oversample_requests}): each missing one is a copy of a uniformly chosen sample
     * under a fresh id, drawn from the dataset's seed. {@code --no-oversample} keeps the sample as is.
     */
    static List<BenchSample> oversample(List<BenchSample> sampled, int numRequests, String requestPrefix,
                                        boolean noOversample, long seed) {
        if (noOversample) {
            return sampled;
        }
        List<BenchSample> out = new ArrayList<>(sampled);
        if (!out.isEmpty() && out.size() < numRequests) {
            PythonRandom random = new PythonRandom(seed);
            int original = out.size();
            int needed = numRequests - original;
            for (int i = 0; i < needed; i++) {
                BenchSample picked = random.choice(sampled);
                out.add(picked.withRequestId(requestPrefix + (original + i)));
            }
        }
        Set<String> ids = new HashSet<>();
        for (BenchSample s : out) {
            if (!ids.add(s.requestId())) {
                throw new IllegalStateException("duplicate request id '" + s.requestId()
                        + "' in the sampled workload");
            }
        }
        return out;
    }

    /** The ids a synthetic prompt never carries: every entry of {@code skipIds} that is non-negative. */
    static boolean skipped(int id, int[] skipIds) {
        for (int s : skipIds) {
            if (s == id) {
                return true;
            }
        }
        return false;
    }
}
