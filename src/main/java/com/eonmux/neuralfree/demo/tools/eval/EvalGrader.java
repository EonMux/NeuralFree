package com.eonmux.neuralfree.demo.tools.eval;

import java.io.IOException;
import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * How a reply is scored: the regex grader extracts the answer with the suite's pattern (the last
 * number, or the last {@code \boxed{}} for AIME) and compares it with the gold answer; the LLM
 * grader asks a chat model to extract the answer and compares that; the CLI grader runs an external
 * script, whose exit status is the verdict. The reply is graded on its last ten lines, as the
 * reference truncates it.
 */
public final class EvalGrader {

    /** The grader kinds the command accepts. */
    public static final List<String> TYPES = List.of("regex", "cli", "llm");

    /** The answer patterns the reference keeps per suite. */
    private static final Map<String, String> PATTERNS = new LinkedHashMap<>();
    /** The sample answers the reference shows the LLM grader. */
    private static final Map<String, List<String>> SAMPLE_ANSWERS = new LinkedHashMap<>();
    /** The last boxed answer, which AIME replies are read with first. */
    private static final Pattern BOXED = Pattern.compile("\\\\boxed\\{([^}]+)\\}", Pattern.CASE_INSENSITIVE);
    /** The number of trailing reply lines the reference grades. */
    static final int GRADED_LINES = 10;
    private static final long CLI_TIMEOUT_SECONDS = 30;

    static {
        String aime = "\\boxed{(\\d+)}|\\b(\\d+)\\b";
        PATTERNS.put("aime", aime);
        PATTERNS.put("aime2025", aime);
        PATTERNS.put("aime2026", aime);
        PATTERNS.put("gsm8k", "\\b(\\d+)\\b");
        List<String> numbers = new ArrayList<>();
        numbers.add("42");
        numbers.add("-123");
        numbers.add("999");
        for (String suite : PATTERNS.keySet()) {
            SAMPLE_ANSWERS.put(suite, numbers);
        }
    }

    /** A grading verdict: whether the reply is correct, and the answer that was extracted. */
    public static final class Grade {

        private final boolean correct;
        private final String answer;

        Grade(boolean correct, String answer) {
            this.correct = correct;
            this.answer = answer;
        }

        public boolean correct() {
            return correct;
        }

        /** The extracted answer, null when none was found. */
        public String answer() {
            return answer;
        }
    }

    private final String type;
    private final String datasetType;
    private final Pattern pattern;
    private final String script;
    private final EvalClient llmClient;
    private final String llmModel;

    private EvalGrader(String type, String datasetType, String script, EvalClient llmClient,
                       String llmModel) {
        this.type = type;
        this.datasetType = datasetType;
        this.pattern = "regex".equals(type) ? compile(datasetType) : null;
        this.script = script;
        this.llmClient = llmClient;
        this.llmModel = llmModel;
    }

    /** The regex grader of a suite. */
    public static EvalGrader regex(String datasetType) {
        requireSuite(datasetType);
        return new EvalGrader("regex", datasetType, null, null, null);
    }

    /** The LLM grader over a chat endpoint. */
    public static EvalGrader llm(String datasetType, EvalClient client, String model) {
        requireSuite(datasetType);
        if (client == null || model == null || model.isEmpty()) {
            throw new IllegalArgumentException("--grader-type llm requires --grader-model or --model");
        }
        return new EvalGrader("llm", datasetType, null, client, model);
    }

    /** The CLI grader over an executable script. */
    public static EvalGrader cli(String datasetType, String script) {
        requireSuite(datasetType);
        if (script == null || script.isEmpty()) {
            throw new IllegalArgumentException("--grader-type cli requires --grader-script");
        }
        File file = new File(script);
        if (!file.isFile()) {
            throw new IllegalArgumentException("--grader-script does not name a file: " + script);
        }
        return new EvalGrader("cli", datasetType, script, null, null);
    }

    private static void requireSuite(String datasetType) {
        if (!PATTERNS.containsKey(datasetType)) {
            throw new IllegalArgumentException("the grader supports the math suites " + PATTERNS.keySet()
                    + ", not '" + datasetType + "'");
        }
    }

    /**
     * The suite's answer pattern as Java reads it. The reference's AIME text opens with
     * {@code \boxed{...}}, which Python reads as a word boundary plus literal text (its boxed
     * answers are found by {@link #BOXED} first); here the alternative matches the literal
     * {@code \boxed{123}} and the number alternative finds the same last number either way.
     */
    private static Pattern compile(String datasetType) {
        if ("gsm8k".equals(datasetType)) {
            return Pattern.compile("\\b(\\d+)\\b", Pattern.CASE_INSENSITIVE);
        }
        return Pattern.compile("\\\\boxed\\{(\\d+)\\}|\\b(\\d+)\\b", Pattern.CASE_INSENSITIVE);
    }

    /** The grader's kind ({@code regex}, {@code llm} or {@code cli}). */
    public String type() {
        return type;
    }

    /** The reference's pattern text of the suite, as its grader log records it; null for the other kinds. */
    public String patternText() {
        return "regex".equals(type) ? PATTERNS.get(datasetType) : null;
    }

    /**
     * Grades one reply.
     *
     * @param gold    the expected answer
     * @param reply   the model's reply, already truncated to its last lines
     * @param problem the prompt the reply answers
     */
    public Grade grade(String gold, String reply, String problem) {
        if ("regex".equals(type)) {
            return gradeRegex(gold, reply);
        }
        if ("cli".equals(type)) {
            return gradeCli(gold, reply);
        }
        return gradeLlm(gold, reply);
    }

    /** The regex verdict: the last extracted answer compared with the gold answer. */
    private Grade gradeRegex(String gold, String reply) {
        String answer = extract(reply);
        if (answer == null) {
            return new Grade(false, null);
        }
        return new Grade(answer.strip().equals(gold.strip()), answer);
    }

    /** The answer a regex grader extracts, null when the reply carries none. */
    String extract(String reply) {
        if (pattern == null) {
            return null;
        }
        if ("aime".equals(datasetType) || "aime2025".equals(datasetType)) {
            Matcher boxed = BOXED.matcher(reply);
            String lastBoxed = null;
            while (boxed.find()) {
                lastBoxed = boxed.group(1).trim();
            }
            if (lastBoxed != null) {
                return lastBoxed;
            }
        }
        Matcher matcher = pattern.matcher(reply);
        List<String> matches = new ArrayList<>();
        while (matcher.find()) {
            String value = matcher.group(1);
            if (value == null || value.isEmpty()) {
                value = matcher.groupCount() >= 2 ? matcher.group(2) : null;
            }
            matches.add(value == null ? "" : value.strip());
        }
        for (int i = matches.size() - 1; i >= 0; i--) {
            if (!matches.get(i).isEmpty()) {
                return matches.get(i);
            }
        }
        return null;
    }

    /** The LLM verdict: the model's extracted answer compared case-insensitively with the gold one. */
    private Grade gradeLlm(String gold, String reply) {
        List<String> samples = SAMPLE_ANSWERS.get(datasetType);
        StringBuilder examples = new StringBuilder();
        for (int i = 0; i < samples.size(); i++) {
            if (i > 0) {
                examples.append('\n');
            }
            examples.append("Example ").append(i + 1).append(": ").append(samples.get(i));
        }
        String system = "You are an answer extraction system. Your task is to extract the answer"
                + " from the model's response.\n\nHere are some examples of extracted answers to"
                + " demonstrate what you are supposed to output:\n\n" + examples + "\n\nWhen"
                + " extracting the answer, provide only the extracted answer itself, nothing else."
                + " If there is no clear answer that can be extracted from the response, reply with"
                + " 'no answer'.";
        String user = "Extract the answer from the following response:\n\n\"" + reply + "\"\n\n"
                + "Please provide only the extracted answer, nothing else. If there is no clear"
                + " answer that can be extracted from the response, reply with 'no answer'.";
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", llmModel);
        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(message("system", system));
        messages.add(message("user", user));
        body.put("messages", messages);
        body.put("temperature", 0);
        try {
            Map<String, Object> response = llmClient.chatOnce(body, null);
            String answer = content(response);
            if (answer == null) {
                return new Grade(false, null);
            }
            answer = answer.strip();
            boolean correct = answer.toLowerCase(Locale.ROOT)
                    .equals(gold.strip().toLowerCase(Locale.ROOT));
            return new Grade(correct, answer);
        } catch (EvalServerException e) {
            return new Grade(false, null);
        }
    }

    /** The CLI verdict: the script's exit status, with the reply as its answer on success. */
    private Grade gradeCli(String gold, String reply) {
        Process process;
        try {
            process = new ProcessBuilder(script, "--answer", reply, "--expected", gold)
                    .redirectErrorStream(true)
                    .start();
            if (!process.waitFor(CLI_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return new Grade(false, null);
            }
        } catch (IOException e) {
            return new Grade(false, null);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return new Grade(false, null);
        }
        boolean correct = process.exitValue() == 0;
        return new Grade(correct, correct ? reply : null);
    }

    /** The first choice's message content, null when the reply carries none. */
    private static String content(Map<String, Object> reply) {
        Object choices = reply.get("choices");
        if (!(choices instanceof List) || ((List<?>) choices).isEmpty()) {
            return null;
        }
        Object first = ((List<?>) choices).get(0);
        if (!(first instanceof Map)) {
            return null;
        }
        Object message = ((Map<?, ?>) first).get("message");
        if (!(message instanceof Map)) {
            return null;
        }
        Object value = ((Map<?, ?>) message).get("content");
        return value instanceof String ? (String) value : null;
    }

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", role);
        message.put("content", content);
        return message;
    }

    /**
     * The reply as the reference grades it: its last {@link #GRADED_LINES} lines, unchanged when it
     * has no more than that.
     */
    public static String truncate(String response) {
        if (response == null) {
            return null;
        }
        String[] lines = response.split("\n", -1);
        if (lines.length <= GRADED_LINES) {
            return response;
        }
        StringBuilder out = new StringBuilder();
        for (int i = lines.length - GRADED_LINES; i < lines.length; i++) {
            if (i > lines.length - GRADED_LINES) {
                out.append('\n');
            }
            out.append(lines[i]);
        }
        return out.toString();
    }
}
