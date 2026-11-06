package com.eonmux.neuralfree.demo.tools.eval;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One case of a run and its verdict: the task's prompt and expected answer, the server's reply, the
 * answer the grader extracted, the grader's log, and the usage the server reported. The JSON form
 * uses the reference's field names, so a state file saved here can be resumed and inspected the way
 * llama-eval's is.
 */
public final class EvalCase {

    /** The case finished and its grader reached a verdict. */
    public static final String OK = "ok";
    /** Every request of the case failed after its retries; the case counts as not answered. */
    public static final String SERVER_ERROR = "server_error";
    /** The case has not run yet. */
    public static final String PENDING = "pending";

    private final String taskId;
    private final String prompt;
    private final String expected;
    private final String questionText;
    private final String response;
    private final String answer;
    private final Map<String, Object> graderLog;
    private final boolean correct;
    private final String status;
    private final Long tokens;
    private final Double tpsGen;
    private final Double tGenMs;
    private final String reasoningContent;
    private final String serverName;
    private final int chunkIdx;
    private final int problemIdx;

    public EvalCase(String taskId, String prompt, String expected, String questionText,
                    String response, String answer, Map<String, Object> graderLog, boolean correct,
                    String status, Long tokens, Double tpsGen, Double tGenMs,
                    String reasoningContent, String serverName, int chunkIdx, int problemIdx) {
        this.taskId = require(taskId, "taskId");
        this.prompt = require(prompt, "prompt");
        this.expected = require(expected, "expected");
        this.questionText = questionText == null ? "" : questionText;
        this.response = response;
        this.answer = answer;
        this.graderLog = graderLog == null ? new LinkedHashMap<>() : new LinkedHashMap<>(graderLog);
        this.correct = correct;
        this.status = require(status, "status");
        this.tokens = tokens;
        this.tpsGen = tpsGen;
        this.tGenMs = tGenMs;
        this.reasoningContent = reasoningContent;
        this.serverName = serverName;
        this.chunkIdx = chunkIdx;
        this.problemIdx = problemIdx;
    }

    private static String require(String value, String name) {
        if (value == null) {
            throw new IllegalArgumentException("an evaluation case needs a " + name);
        }
        return value;
    }

    /** A not-yet-run case of the question at {@code problemIdx}. */
    static EvalCase pending(String taskId, EvalQuestion question, int chunkIdx, int problemIdx) {
        return new EvalCase(taskId, question.prompt(), question.expected(), question.text(), null,
                null, new LinkedHashMap<>(), false, PENDING, null, null, null, null, null,
                chunkIdx, problemIdx);
    }

    public String taskId() {
        return taskId;
    }

    public String prompt() {
        return prompt;
    }

    public String expected() {
        return expected;
    }

    public String questionText() {
        return questionText;
    }

    public String response() {
        return response;
    }

    public String answer() {
        return answer;
    }

    public Map<String, Object> graderLog() {
        return new LinkedHashMap<>(graderLog);
    }

    public boolean correct() {
        return correct;
    }

    public String status() {
        return status;
    }

    /** The completion tokens the server reported, null when it reported none. */
    public Long tokens() {
        return tokens;
    }

    /** The server's generation speed in tokens per second, null when it reported none. */
    public Double tpsGen() {
        return tpsGen;
    }

    /** The server's generation time in milliseconds, null when it reported none. */
    public Double tGenMs() {
        return tGenMs;
    }

    public String reasoningContent() {
        return reasoningContent;
    }

    public String serverName() {
        return serverName;
    }

    public int chunkIdx() {
        return chunkIdx;
    }

    public int problemIdx() {
        return problemIdx;
    }

    /** Whether the status is the reference's server-error marker. */
    public boolean serverError() {
        return SERVER_ERROR.equals(status);
    }

    /** The state file's object of this case. */
    public Map<String, Object> toJson() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("task_id", taskId);
        out.put("prompt", prompt);
        out.put("expected", expected);
        out.put("question_text", questionText);
        out.put("response", response);
        out.put("answer", answer);
        out.put("grader_log", graderLog);
        out.put("correct", correct);
        out.put("status", status);
        out.put("tokens", tokens);
        out.put("tps_gen", tpsGen);
        out.put("t_gen_ms", tGenMs);
        out.put("reasoning_content", reasoningContent);
        out.put("server_name", serverName);
        out.put("chunk_idx", chunkIdx);
        out.put("problem_idx", problemIdx);
        return out;
    }

    /** A case read from a state file; missing optional fields keep their empty value. */
    static EvalCase fromJson(Map<?, ?> map) {
        Map<String, Object> log = new LinkedHashMap<>();
        Object rawLog = map.get("grader_log");
        if (rawLog instanceof Map) {
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) rawLog).entrySet()) {
                log.put(String.valueOf(entry.getKey()), entry.getValue());
            }
        }
        return new EvalCase(String.valueOf(map.get("task_id")), string(map.get("prompt"), ""),
                string(map.get("expected"), ""), string(map.get("question_text"), ""),
                nullableString(map.get("response")), nullableString(map.get("answer")), log,
                Boolean.TRUE.equals(map.get("correct")), string(map.get("status"), PENDING),
                longValue(map.get("tokens")), doubleValue(map.get("tps_gen")),
                doubleValue(map.get("t_gen_ms")), nullableString(map.get("reasoning_content")),
                nullableString(map.get("server_name")), intValue(map.get("chunk_idx")),
                intValue(map.get("problem_idx")));
    }

    private static String string(Object value, String fallback) {
        return value == null ? fallback : String.valueOf(value);
    }

    private static String nullableString(Object value) {
        return value == null ? null : String.valueOf(value);
    }

    private static Long longValue(Object value) {
        return value instanceof Number ? ((Number) value).longValue() : null;
    }

    private static Double doubleValue(Object value) {
        return value instanceof Number ? ((Number) value).doubleValue() : null;
    }

    private static int intValue(Object value) {
        return value instanceof Number ? ((Number) value).intValue() : 0;
    }
}
