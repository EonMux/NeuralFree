package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The {@code hf} dataset: vLLM's Hugging Face datasets, loaded either from a local file or dataset
 * directory or, over {@link HfDatasetSource}, from a Hub repository. The rows' schema selects the
 * reference dataset they stand for — conversations, turns, instruction/input pairs, messages,
 * audio, code edits, interleaved images, MLPerf prompts, reasoning problems, questions, or a plain
 * prompt — and the datasets the reference renders client-side are rendered through the model's chat
 * template ({@link BenchDatasets#renderPrompt}) before they are counted. {@code --hf-subset}
 * selects a subdirectory and {@code --hf-split} the split-suffixed files.
 */
final class HfDataset implements ServeBenchDataset {

    /** MT-Bench's default output budget, vLLM's {@code MTBenchDataset.DEFAULT_OUTPUT_LEN}. */
    private static final int MT_BENCH_OUTPUT_LEN = 256;

    /** InstructCoder's default output budget, vLLM's {@code InstructCoderDataset.DEFAULT_OUTPUT_LEN}. */
    private static final int INSTRUCT_CODER_OUTPUT_LEN = 200;

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;

    HfDataset(ServeBenchOptions opts, BenchTokenizer tokenizer) {
        this.opts = opts;
        this.tokenizer = tokenizer;
    }

    @Override
    public List<BenchSample> sample() {
        HfDatasetSource.Source source = HfDatasetSource.resolve(opts);
        List<Map<String, Object>> rows;
        try {
            rows = readRows(source.path, source.subset, source.split);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the hf dataset " + source.path + ": "
                    + e.getMessage(), e);
        }
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("the hf dataset " + source.path + " carries no record");
        }
        BenchDatasets.shuffle(rows, opts.seed(), opts.disableShuffle());
        Path baseDir = BenchDatasets.baseDir(source.path);
        Kind kind = Kind.of(rows.get(0), source.path);
        List<BenchSample> out = new ArrayList<>();
        int index = 0;
        for (Map<String, Object> row : rows) {
            if (out.size() >= opts.numPrompts()) {
                break;
            }
            BenchSample sample = sample(kind, row, baseDir, index);
            if (sample != null) {
                out.add(sample);
                index++;
            }
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("the hf dataset " + source.path + " sampled no request: its"
                    + " sequences fall outside the reference's length bounds");
        }
        return ServeBenchDataset.oversample(out, opts.numPrompts(), opts.requestIdPrefix(),
                opts.noOversample(), opts.seed());
    }

    /** The rows of the selected subset and split files. */
    private List<Map<String, Object>> readRows(Path path, String subset, String split) throws IOException {
        return HfDatasetSource.readRows(new HfDatasetSource.Source(path, subset, split));
    }

    /** One row as the request its schema describes; null when the reference's pruning drops it. */
    private BenchSample sample(Kind kind, Map<String, Object> row, Path baseDir, int index) {
        String requestId = opts.requestIdPrefix() + index;
        switch (kind) {
            case CONVERSATION:
                return conversation(row, requestId);
            case TURNS:
                return turns(row, requestId);
            case INSTRUCT:
                return instruct(row, requestId);
            case MESSAGES:
                return messages(row, requestId);
            case MLPERF:
                return mlperf(row, requestId);
            case AUDIO:
                return audioRow(row, baseDir, requestId);
            case BLAZEDIT:
                return blazedit(row, requestId);
            case IMAGE:
                return image(row, baseDir, index, requestId);
            case AIMO:
                return aimo(row, requestId);
            case QUESTION:
                return question(row, requestId);
            case PROMPT:
                return prompt(row, requestId);
            default:
                throw new IllegalStateException("unhandled hf row kind " + kind);
        }
    }

    /** A conversation's first turn, the completion's length as the output budget (ShareGPT). */
    private BenchSample conversation(Map<String, Object> row, String requestId) {
        if (!(row.get("conversations") instanceof List)) {
            return null;
        }
        List<?> turns = (List<?>) row.get("conversations");
        if (turns.size() < 2) {
            return null;
        }
        String prompt = turnValue(turns.get(0));
        String completion = turnValue(turns.get(1));
        if (prompt == null || completion == null) {
            return null;
        }
        boolean fixed = opts.hfOutputLen() != null;
        int promptLen = BenchDatasets.promptLen(opts, tokenizer, prompt);
        int outputLen = fixed ? opts.hfOutputLen() : tokenizer.tokenize(completion).length;
        if (!fixed && !ServeBenchDataset.isValidSequence(promptLen, outputLen, false)) {
            return null;
        }
        return BenchSample.ofText(prompt, promptLen, outputLen, requestId);
    }

    private static String turnValue(Object turn) {
        if (!(turn instanceof Map)) {
            return null;
        }
        Object value = ((Map<?, ?>) turn).get("value");
        return value instanceof String ? (String) value : null;
    }

    /** MT-Bench's first turn, rendered through the chat template. */
    private BenchSample turns(Map<String, Object> row, String requestId) {
        Object turns = row.get("turns");
        if (!(turns instanceof List) || ((List<?>) turns).isEmpty()) {
            return null;
        }
        Object first = ((List<?>) turns).get(0);
        String prompt = first instanceof String ? (String) first : turnValue(first);
        if (prompt == null) {
            return null;
        }
        return rendered(prompt, outputLen(MT_BENCH_OUTPUT_LEN), requestId);
    }

    /** InstructCoder's input and instruction, rendered through the chat template. */
    private BenchSample instruct(Map<String, Object> row, String requestId) {
        String input = BenchDatasets.optionalString(row, "input");
        String instruction = BenchDatasets.optionalString(row, "instruction");
        if (input == null || instruction == null) {
            return null;
        }
        String prompt = input + "\n\n" + instruction
                + " Just output the code, do not include any explanation.";
        return rendered(prompt, outputLen(INSTRUCT_CODER_OUTPUT_LEN), requestId);
    }

    private BenchSample messages(Map<String, Object> row, String requestId) {
        if (!(row.get("messages") instanceof List) || ((List<?>) row.get("messages")).isEmpty()) {
            return null;
        }
        Object first = ((List<?>) row.get("messages")).get(0);
        if (!(first instanceof Map)) {
            return null;
        }
        Object content = ((Map<?, ?>) first).get("content");
        if (!(content instanceof String)) {
            return null;
        }
        return rendered((String) content, outputLen(SpeedBenchDataset.DEFAULT_OUTPUT_LEN), requestId);
    }

    /** MLPerf's system prompt and question, rendered as a two-message conversation. */
    private BenchSample mlperf(Map<String, Object> row, String requestId) {
        String system = BenchDatasets.optionalString(row, "system_prompt");
        String question = BenchDatasets.optionalString(row, "question");
        String answer = BenchDatasets.optionalString(row, "output");
        if (system == null || question == null) {
            return null;
        }
        String prompt = BenchDatasets.renderPrompt(opts, tokenizer, system, question);
        int promptLen = BenchDatasets.promptLen(opts, tokenizer, prompt);
        int outputLen = opts.hfOutputLen() != null ? opts.hfOutputLen()
                : answer == null ? 256 : tokenizer.tokenize(answer).length;
        if (opts.hfOutputLen() == null
                && !ServeBenchDataset.isValidSequence(promptLen, outputLen, false)) {
            return null;
        }
        return BenchSample.ofText(prompt, promptLen, outputLen, requestId);
    }

    private BenchSample audioRow(Map<String, Object> row, Path baseDir, String requestId) {
        BenchDatasets.AudioFile audio = BenchDatasets.audio(row.get("audio"), baseDir);
        if (audio.durationSec < opts.asrMinAudioLenSec() || audio.durationSec > opts.asrMaxAudioLenSec()) {
            return null;
        }
        int outputLen = opts.hfOutputLen() != null ? opts.hfOutputLen() : AsrDataset.DEFAULT_OUTPUT_LEN;
        int promptLen = opts.skipTokenizerInit() ? 1 : tokenizer.tokenize("").length;
        return BenchSample.ofAudio(audio.path, audio.durationSec, promptLen, outputLen, requestId);
    }

    private BenchSample blazedit(Map<String, Object> row, String requestId) {
        String code = BenchDatasets.string(row, "code", "an hf blazedit record");
        String change = BenchDatasets.string(row, "change_request", "an hf blazedit record");
        double distance = BenchDatasets.number(row, "norm_distance", "an hf blazedit record");
        if (distance < opts.blazeditMinDistance() || distance > opts.blazeditMaxDistance()) {
            return null;
        }
        String prompt = "Given a code file, please apply the change requests and generate the new file."
                + "\n\nOriginal file:\n```python\n" + code + "\n```\n\nChange request:\n" + change
                + "\n\nPlease generate the new code file in the \"New file\" section below.";
        return rendered(prompt, outputLen(BlazeditDataset.DEFAULT_OUTPUT_LEN), requestId);
    }

    private BenchSample image(Map<String, Object> row, Path baseDir, int index, String requestId) {
        List<Object> content = CustomImageDataset.contentOf(row, baseDir, index + 1);
        String text = CustomImageDataset.textOf(content);
        int promptLen = BenchDatasets.promptLen(opts, tokenizer, text);
        int outputLen = Math.max(0, opts.customOutputLen());
        return BenchSample.ofChat(content, promptLen, outputLen, requestId);
    }

    /** AIMO's problem and its solution, the solution's length as the output budget. */
    private BenchSample aimo(Map<String, Object> row, String requestId) {
        String prompt = BenchDatasets.optionalString(row, "problem");
        String completion = BenchDatasets.optionalString(row, "solution");
        if (prompt == null) {
            return null;
        }
        int promptLen = BenchDatasets.promptLen(opts, tokenizer, prompt);
        int outputLen;
        if (opts.hfOutputLen() != null) {
            outputLen = opts.hfOutputLen();
        } else if (completion != null) {
            outputLen = tokenizer.tokenize(completion).length;
            if (!ServeBenchDataset.isValidSequence(promptLen, outputLen, 4, 2048, 32000)) {
                return null;
            }
        } else {
            outputLen = 256;
        }
        return BenchSample.ofText(prompt, promptLen, outputLen, requestId);
    }

    /** GSM8K's question, rendered through the chat template with the reference's fixed budget. */
    private BenchSample question(Map<String, Object> row, String requestId) {
        String prompt = BenchDatasets.optionalString(row, "question");
        if (prompt == null) {
            return null;
        }
        return rendered(prompt, outputLen(256), requestId);
    }

    /** A plain prompt (HumanEval and the custom-like Hugging Face rows). */
    private BenchSample prompt(Map<String, Object> row, String requestId) {
        String prompt = BenchDatasets.string(row, "prompt", "an hf record");
        int outputLen;
        if (opts.skipTokenizerInit()) {
            outputLen = 1;
        } else if (opts.hfOutputLen() != null) {
            outputLen = opts.hfOutputLen();
        } else if (row.get("output_tokens") instanceof Number) {
            outputLen = ((Number) row.get("output_tokens")).intValue();
        } else if (row.get("output_tokens") instanceof String
                && numeric((String) row.get("output_tokens")) != null) {
            outputLen = numeric((String) row.get("output_tokens"));
        } else {
            outputLen = 256;
        }
        return rendered(prompt, outputLen, requestId);
    }

    /** A text prompt rendered through the model's chat template, then counted. */
    private BenchSample rendered(String prompt, int outputLen, String requestId) {
        String text = BenchDatasets.renderPrompt(opts, tokenizer, prompt);
        return BenchSample.ofText(text, BenchDatasets.promptLen(opts, tokenizer, text), outputLen, requestId);
    }

    /** The fixed {@code --hf-output-len}, else the dataset's own default. */
    private int outputLen(int fallback) {
        return opts.hfOutputLen() != null ? opts.hfOutputLen() : fallback;
    }

    /** A delimited field's integer value, null when the text is not one. */
    private static Integer numeric(String text) {
        try {
            return Integer.parseInt(text.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** The row schemas a Hugging Face dataset may carry, in the reference's dispatch order. */
    private enum Kind {
        MLPERF("system_prompt"),
        CONVERSATION("conversations"),
        TURNS("turns"),
        INSTRUCT("instruction"),
        AUDIO("audio"),
        MESSAGES("messages"),
        BLAZEDIT("code"),
        IMAGE("content"),
        AIMO("problem"),
        QUESTION("question"),
        PROMPT("prompt");

        private final String column;

        Kind(String column) {
            this.column = column;
        }

        static Kind of(Map<String, Object> row, Path path) {
            if (row.containsKey("system_prompt") && row.containsKey("question")
                    && row.containsKey("output")) {
                return MLPERF;
            }
            if (row.containsKey("code") && row.containsKey("change_request")) {
                return BLAZEDIT;
            }
            if (row.containsKey("content") || row.containsKey("image_files")) {
                return IMAGE;
            }
            if (row.containsKey("conversations")) {
                return CONVERSATION;
            }
            if (row.containsKey("turns") && !row.containsKey("messages")) {
                return TURNS;
            }
            if (row.containsKey("instruction") && row.containsKey("input")) {
                return INSTRUCT;
            }
            for (Kind kind : values()) {
                if (row.containsKey(kind.column)) {
                    return kind;
                }
            }
            throw new IllegalArgumentException("the hf dataset " + path + " carries none of the"
                    + " supported columns: conversations, turns, instruction/input, messages, audio,"
                    + " code, content, system_prompt, problem, question or prompt");
        }
    }
}
