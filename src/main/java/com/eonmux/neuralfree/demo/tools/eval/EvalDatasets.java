package com.eonmux.neuralfree.demo.tools.eval;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.eonmux.neuralfree.demo.tools.serve.HfDatasetSource;

/**
 * The math task-accuracy suites of the reference's {@code llama-eval.py}, loaded as Hugging Face
 * rows: GSM8K ({@code openai/gsm8k}, {@code main}/{@code test}) and AIME (the 2024 validation split,
 * AIME 2025's two parts and AIME 2026). Each row becomes a question, the suite's prompt template and
 * the expected answer the regex grader matches; the gold answer is normalized the way the reference
 * normalizes it. A dataset directory or file given by {@code --dataset-source} replaces the Hub
 * repository.
 */
public final class EvalDatasets {

    /** The suite names the command accepts, in the reference's order. */
    public static final List<String> NAMES = List.of("aime", "aime2025", "aime2026", "gsm8k");

    /** The prompt of the AIME suites; the reference's template registry entry. */
    static final String AIME_TEMPLATE = "Solve the following math problem step by step. Put your"
            + " answer inside \\boxed{}.\n\n{question}\n\nRemember to put your answer inside"
            + " \\boxed{}.\n";

    /** The prompt of GSM8K; the reference's template registry entry. */
    static final String GSM8K_TEMPLATE = "{question}\nPlease reason step by step, and put your final"
            + " numeric answer within \\boxed{} without any extra characters.\n";

    /** The leading digit run of a gold answer, Python's {@code re.match(r"\d+", s)}. */
    private static final Pattern LEADING_DIGITS = Pattern.compile("\\d+");

    private EvalDatasets() {
    }

    /**
     * Loads a suite, downloading its canonical Hub repository unless {@code source} names a local
     * path or another repository.
     *
     * @param type   one of {@link #NAMES}
     * @param source the override, or null for the suite's canonical repository
     * @throws IllegalArgumentException naming an unknown suite, a missing source, or a row without
     *                                  the field the suite requires
     * @throws IllegalStateException    when the repository cannot be read
     */
    public static EvalDataset load(String type, String source) {
        Suite suite = Suite.of(type);
        HfDatasetSource.Source resolved = resolve(suite, source);
        List<Map<String, Object>> rows;
        try {
            rows = HfDatasetSource.readRows(resolved);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the " + type + " dataset " + resolved.path()
                    + ": " + e.getMessage(), e);
        }
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("the " + type + " dataset " + resolved.path()
                    + " carries no record");
        }
        List<EvalQuestion> questions = new ArrayList<>(rows.size());
        for (int i = 0; i < rows.size(); i++) {
            questions.add(question(suite, rows.get(i), i + 1));
        }
        return new RowsDataset(type, questions);
    }

    /** The override with the suite's defaults, or the suite's canonical repository. */
    private static HfDatasetSource.Source resolve(Suite suite, String source) {
        if (source != null) {
            boolean local = isLocal(source);
            // A local file or directory is read as it lies; the canonical subset and split only
            // name the layout of the Hub repository.
            return HfDatasetSource.resolve(source, local ? null : suite.subset,
                    local ? null : suite.split);
        }
        return HfDatasetSource.resolve(suite.repository, suite.subset, suite.split);
    }

    private static boolean isLocal(String source) {
        try {
            return Files.exists(Paths.get(source));
        } catch (InvalidPathException e) {
            return false;
        }
    }

    /** One row as the suite's question. */
    private static EvalQuestion question(Suite suite, Map<String, Object> row, int line) {
        switch (suite) {
            case GSM8K:
                return gsm8k(row, line);
            case AIME:
                return aime(row, line, suite);
            case AIME2025:
                return aime(row, line, suite);
            case AIME2026:
                return aime2026(row, line);
            default:
                throw new IllegalStateException("unhandled dataset suite " + suite);
        }
    }

    private static EvalQuestion gsm8k(Map<String, Object> row, int line) {
        String text = optionalString(row, "problem");
        if (text == null) {
            text = requiredString(row, "question", "gsm8k", line);
        }
        String expected;
        if (row.containsKey("gold")) {
            expected = String.valueOf(row.get("gold"));
        } else {
            String gold = requiredString(row, "answer", "gsm8k", line);
            int marker = gold.lastIndexOf("####");
            if (marker >= 0) {
                gold = gold.substring(marker + 4).trim();
            }
            String normalized = normalizeNumber(gold);
            expected = normalized != null ? normalized : gold;
        }
        return new EvalQuestion(text, render(GSM8K_TEMPLATE, text), expected);
    }

    private static EvalQuestion aime(Map<String, Object> row, int line, Suite suite) {
        String text = suite == Suite.AIME ? optionalString(row, "problem") : null;
        if (text == null) {
            text = requiredString(row, "question", suite.name, line);
        }
        Object answer = row.get("answer");
        if (answer == null) {
            throw new IllegalArgumentException("row " + line + " of the " + suite.name
                    + " dataset has no 'answer' field");
        }
        String expected = answer instanceof String
                ? normalizeOrSame((String) answer) : String.valueOf(answer);
        return new EvalQuestion(text, render(AIME_TEMPLATE, text), expected);
    }

    private static EvalQuestion aime2026(Map<String, Object> row, int line) {
        String text = requiredString(row, "problem", "aime2026", line);
        Object answer = row.get("answer");
        if (answer == null) {
            throw new IllegalArgumentException("row " + line + " of the aime2026 dataset has no"
                    + " 'answer' field");
        }
        return new EvalQuestion(text, render(AIME_TEMPLATE, text), String.valueOf(answer));
    }

    /** The template with the question substituted; no other brace is special. */
    private static String render(String template, String question) {
        return template.replace("{question}", question);
    }

    /** A gold answer's leading integer as the reference prints it, or the answer unchanged. */
    static String normalizeOrSame(String answer) {
        String normalized = normalizeNumber(answer);
        return normalized != null ? normalized : answer;
    }

    /** The leading digit run parsed as an integer and printed back, null when it has none. */
    static String normalizeNumber(String value) {
        Matcher matcher = LEADING_DIGITS.matcher(value);
        if (!matcher.lookingAt()) {
            return null;
        }
        return new BigInteger(matcher.group()).toString();
    }

    private static String optionalString(Map<String, Object> row, String key) {
        Object value = row.get(key);
        return value instanceof String ? (String) value : null;
    }

    private static String requiredString(Map<String, Object> row, String key, String suite, int line) {
        Object value = row.get(key);
        if (!(value instanceof String)) {
            throw new IllegalArgumentException("row " + line + " of the " + suite
                    + " dataset has no '" + key + "' field");
        }
        return (String) value;
    }

    /** A suite's canonical source and the prompt style its rows use. */
    private enum Suite {
        AIME("aime", "AI-MO/aimo-validation-aime", null, "train"),
        AIME2025("aime2025", "opencompass/AIME2025", null, "test"),
        AIME2026("aime2026", "MathArena/aime_2026", "default", "train"),
        GSM8K("gsm8k", "openai/gsm8k", "main", "test");

        private final String name;
        private final String repository;
        private final String subset;
        private final String split;

        Suite(String name, String repository, String subset, String split) {
            this.name = name;
            this.repository = repository;
            this.subset = subset;
            this.split = split;
        }

        static Suite of(String name) {
            for (Suite suite : values()) {
                if (suite.name.equals(name)) {
                    return suite;
                }
            }
            throw new IllegalArgumentException("--dataset is one of " + NAMES + ", not '" + name
                    + "'");
        }
    }

    /** The in-memory rows of one suite. */
    private static final class RowsDataset implements EvalDataset {

        private final String type;
        private final List<EvalQuestion> questions;

        RowsDataset(String type, List<EvalQuestion> questions) {
            this.type = type;
            this.questions = new ArrayList<>(questions);
        }

        @Override
        public String type() {
            return type;
        }

        @Override
        public int size() {
            return questions.size();
        }

        @Override
        public EvalQuestion question(int index) {
            return questions.get(index);
        }
    }
}
