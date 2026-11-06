package com.eonmux.neuralfree.demo.tools.eval;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import com.eonmux.neuralfree.demo.tools.serve.PythonRandom;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

/**
 * A run's tasks and verdicts: the suite, the model, the sampling parameters, the shuffled task
 * list, and every case's state. It saves itself after each case as the reference's state JSON (the
 * same field names, so a run can be resumed), and writes the readable HTML report beside it. The
 * confidence interval is the Wilson score interval of the completed cases.
 */
public final class EvalState {

    /** One task: the dataset index it asks and the id the state file keys it by. */
    public static final class TaskRef {

        private final int index;
        private final String taskId;

        TaskRef(int index, String taskId) {
            this.index = index;
            this.taskId = taskId;
        }

        public int index() {
            return index;
        }

        public String taskId() {
            return taskId;
        }
    }

    private final String datasetType;
    private final EvalDataset dataset;
    private final String modelName;
    private final EvalSampling sampling;
    private final Path outputFile;
    private final Map<String, EvalCase> cases = new LinkedHashMap<>();
    private final Object lock = new Object();
    private List<TaskRef> tasks = new ArrayList<>();
    private List<TaskRef> allTasks = new ArrayList<>();
    private long total;
    private int correct;
    private double totalTime;

    public EvalState(EvalDataset dataset, String modelName, EvalSampling sampling, Path outputFile) {
        this.datasetType = dataset.type();
        this.dataset = dataset;
        this.modelName = modelName;
        this.sampling = sampling;
        this.outputFile = outputFile;
        this.total = 0;
    }

    /** The suite's name. */
    public String datasetType() {
        return datasetType;
    }

    /** The loaded suite. */
    public EvalDataset dataset() {
        return dataset;
    }

    /** The model name the run records, possibly null. */
    public String modelName() {
        return modelName;
    }

    /** The sampling parameters the run sends. */
    public EvalSampling sampling() {
        return sampling;
    }

    /** The state JSON file. */
    public Path outputFile() {
        return outputFile;
    }

    /** The readable report written beside the state JSON. */
    public Path htmlFile() {
        return outputFile.resolveSibling(outputFile.getFileName().toString() + ".html");
    }

    /** The tasks this invocation runs. */
    public List<TaskRef> tasks() {
        synchronized (lock) {
            return new ArrayList<>(tasks);
        }
    }

    /** Every task of the run, the ones already decided included. */
    public List<TaskRef> allTasks() {
        synchronized (lock) {
            return new ArrayList<>(allTasks);
        }
    }

    /** The question at a dataset index. */
    public EvalQuestion question(int index) {
        return dataset.question(index);
    }

    /** The recorded case, null when the task has not run. */
    public EvalCase caseOf(String taskId) {
        synchronized (lock) {
            return cases.get(taskId);
        }
    }

    /**
     * Lays out the shuffled task list: {@code nCases} tasks over passes of the whole suite, each
     * pass reshuffled with the reference's Python generator from the same seed.
     */
    public void setupTasks(Integer nCases, long seed) {
        int size = dataset.size();
        if (size == 0) {
            throw new IllegalArgumentException("the " + datasetType + " dataset carries no question");
        }
        int wanted = nCases == null ? size : nCases;
        if (wanted < 1) {
            throw new IllegalArgumentException("--n_cases must be at least 1, got " + wanted);
        }
        PythonRandom random = new PythonRandom(seed);
        List<TaskRef> laidOut = new ArrayList<>(wanted);
        for (int chunk = 0; (long) chunk * size < wanted; chunk++) {
            int chunkSize = Math.min(size, wanted - chunk * size);
            List<Integer> indices = new ArrayList<>(size);
            for (int i = 0; i < size; i++) {
                indices.add(i);
            }
            random.shuffle(indices);
            for (int k = 0; k < chunkSize; k++) {
                int index = indices.get(k);
                laidOut.add(new TaskRef(index,
                        String.format(Locale.ROOT, "%s_%03d_%03d", datasetType, chunk, index)));
            }
        }
        synchronized (lock) {
            this.tasks = new ArrayList<>(laidOut);
            this.allTasks = new ArrayList<>(laidOut);
            this.total = laidOut.size();
        }
    }

    /** Records a case and recomputes the completed verdicts. */
    public void addResult(EvalCase result) {
        synchronized (lock) {
            cases.put(result.taskId(), result);
            this.correct = countCorrect(cases.values());
            if (total == 0) {
                total = cases.size();
            }
        }
    }

    /** Adds the time one case took to the run's accumulated time. */
    public void addTime(double seconds) {
        synchronized (lock) {
            totalTime += seconds;
        }
    }

    /** The time the run has accumulated, as the state file records it. */
    public double accumulatedSeconds() {
        synchronized (lock) {
            return totalTime;
        }
    }

    /** The completed cases. */
    public int completed() {
        synchronized (lock) {
            int count = 0;
            for (EvalCase c : cases.values()) {
                if (EvalCase.OK.equals(c.status())) {
                    count++;
                }
            }
            return count;
        }
    }

    /** The cases whose every request failed. */
    public int nServerErrors() {
        synchronized (lock) {
            int count = 0;
            for (EvalCase c : cases.values()) {
                if (c.serverError()) {
                    count++;
                }
            }
            return count;
        }
    }

    /** The correct cases so far. */
    public int correct() {
        synchronized (lock) {
            return correct;
        }
    }

    /** Every task of the run, as the state file lists them. */
    public List<String> taskIds() {
        synchronized (lock) {
            List<String> ids = new ArrayList<>(tasksToSave().size());
            for (TaskRef ref : tasksToSave()) {
                ids.add(ref.taskId());
            }
            return ids;
        }
    }

    /** Whether every task has an {@code ok} verdict. */
    public boolean isComplete() {
        synchronized (lock) {
            List<TaskRef> all = tasksToSave();
            if (all.isEmpty()) {
                return false;
            }
            for (TaskRef ref : all) {
                EvalCase c = cases.get(ref.taskId());
                if (c == null || !EvalCase.OK.equals(c.status())) {
                    return false;
                }
            }
            return true;
        }
    }

    /** The tasks without an {@code ok} verdict, server errors and errors included. */
    public List<TaskRef> pendingTasks() {
        synchronized (lock) {
            List<TaskRef> pending = new ArrayList<>();
            for (TaskRef ref : tasksToSave()) {
                EvalCase c = cases.get(ref.taskId());
                if (c == null || !EvalCase.OK.equals(c.status())) {
                    pending.add(ref);
                }
            }
            return pending;
        }
    }

    /** Replaces the tasks of this invocation; the completed cases stay. */
    public void setTasks(List<TaskRef> newTasks) {
        synchronized (lock) {
            this.tasks = new ArrayList<>(newTasks);
        }
    }

    /** The Wilson score interval of the completed cases: lower, upper. */
    public double[] accuracyCi() {
        int completed;
        int right;
        synchronized (lock) {
            completed = completed();
            right = correct;
        }
        return wilsonInterval(right, completed);
    }

    /** The Wilson score interval of {@code correct} of {@code total}, 95% by default. */
    static double[] wilsonInterval(int correct, int total) {
        if (total == 0) {
            return new double[]{0.0, 1.0};
        }
        double z = 1.96;
        double p = (double) correct / total;
        double z2 = z * z / total;
        double center = (p + z2 / 2) / (1 + z2);
        double margin = z * Math.sqrt((p * (1 - p) + z2 / 4) / total) / (1 + z2);
        return new double[]{center - margin, center + margin};
    }

    /** The unbiased pass@k estimator of the Codex paper. */
    static double passAtK(int n, int c, int k) {
        if (n - c < k) {
            return 1.0;
        }
        double product = 1.0;
        for (int i = n - c + 1; i <= n; i++) {
            product *= 1.0 - (double) k / i;
        }
        return 1.0 - product;
    }

    /** Whether any problem ran more than once, so a pass@k report adds to the accuracy. */
    public boolean hasMultipleSamples() {
        synchronized (lock) {
            Map<Integer, Integer> runs = new LinkedHashMap<>();
            for (EvalCase c : cases.values()) {
                if (!EvalCase.OK.equals(c.status())) {
                    continue;
                }
                Integer count = runs.get(c.problemIdx());
                runs.put(c.problemIdx(), count == null ? 1 : count + 1);
                if (runs.get(c.problemIdx()) > 1) {
                    return true;
                }
            }
            return false;
        }
    }

    /** The pass@1, pass@5, pass@10 and pass@n of a run that sampled each problem more than once. */
    public void printPassAtK(Appendable out) {
        Map<Integer, List<Boolean>> byProblem = new LinkedHashMap<>();
        synchronized (lock) {
            for (EvalCase c : cases.values()) {
                if (!EvalCase.OK.equals(c.status())) {
                    continue;
                }
                List<Boolean> verdicts = byProblem.get(c.problemIdx());
                if (verdicts == null) {
                    verdicts = new ArrayList<>();
                    byProblem.put(c.problemIdx(), verdicts);
                }
                verdicts.add(c.correct());
            }
        }
        if (byProblem.isEmpty()) {
            return;
        }
        int samples = Integer.MAX_VALUE;
        for (List<Boolean> verdicts : byProblem.values()) {
            samples = Math.min(samples, verdicts.size());
        }
        int problems = byProblem.size();
        List<Integer> ks = new ArrayList<>();
        for (int k : new int[]{1, 5, 10, 100, samples}) {
            if (k <= samples && !ks.contains(k)) {
                ks.add(k);
            }
        }
        try {
            out.append("\n============================================================\n");
            out.append("  " + datasetType + ": " + problems + " problems, " + samples
                    + " sample(s) each\n");
            out.append("============================================================\n");
            for (int k : ks) {
                double score = 0.0;
                for (List<Boolean> verdicts : byProblem.values()) {
                    int right = 0;
                    for (boolean correctCase : verdicts) {
                        if (correctCase) {
                            right++;
                        }
                    }
                    score += passAtK(verdicts.size(), right, k);
                }
                score /= problems;
                out.append(String.format(Locale.ROOT, "  pass@%-4d %.4f%n", k, score));
            }
            if (samples == 1) {
                int solved = 0;
                for (List<Boolean> verdicts : byProblem.values()) {
                    if (verdicts.get(0)) {
                        solved++;
                    }
                }
                double[] ci = wilsonInterval(solved, problems);
                out.append(String.format(Locale.ROOT, "  %d/%d solved   95%% CI [%.4f, %.4f]%n",
                        solved, problems, ci[0], ci[1]));
            }
        } catch (IOException e) {
            throw new IllegalStateException("cannot print the pass@k report", e);
        }
    }

    /** Writes the state JSON and the HTML report. */
    public void dump() {
        Map<String, Object> root;
        List<TaskRef> references;
        Map<String, Object> caseObjects = new LinkedHashMap<>();
        long totalSnapshot;
        int correctSnapshot;
        double timeSnapshot;
        double[] ci;
        synchronized (lock) {
            references = tasksToSave();
            for (TaskRef ref : references) {
                EvalCase c = cases.get(ref.taskId());
                if (c == null) {
                    EvalQuestion question = dataset.question(ref.index());
                    c = EvalCase.pending(ref.taskId(), question, chunkIndex(ref.taskId()),
                            ref.index());
                }
                caseObjects.put(ref.taskId(), c.toJson());
            }
            totalSnapshot = total;
            correctSnapshot = correct;
            timeSnapshot = totalTime;
            ci = wilsonInterval(correct, completedUnlocked());
            root = new LinkedHashMap<>();
            root.put("id", datasetType);
            root.put("model_name", modelName);
            List<String> ids = new ArrayList<>();
            for (TaskRef ref : references) {
                ids.add(ref.taskId());
            }
            root.put("tasks", ids);
            Map<String, Object> states = new LinkedHashMap<>();
            states.put("total", totalSnapshot);
            states.put("correct", correctSnapshot);
            states.put("total_time", timeSnapshot);
            states.put("ci_lower", ci[0]);
            states.put("ci_upper", ci[1]);
            states.put("cases", caseObjects);
            root.put("task_states", states);
            root.put("sampling_config", sampling.asMap());
        }
        String json = Json.toString(root, new Json.Style(true, "  ", ",", ": ", false));
        write(outputFile, json);
        write(htmlFile(), html(root));
    }

    /** The cases map's completed count; the caller holds the lock. */
    private int completedUnlocked() {
        int count = 0;
        for (EvalCase c : cases.values()) {
            if (EvalCase.OK.equals(c.status())) {
                count++;
            }
        }
        return count;
    }

    private List<TaskRef> tasksToSave() {
        return allTasks.isEmpty() ? new ArrayList<>(tasks) : new ArrayList<>(allTasks);
    }

    private static int countCorrect(Iterable<EvalCase> values) {
        int count = 0;
        for (EvalCase c : values) {
            if (c.correct()) {
                count++;
            }
        }
        return count;
    }

    private static void write(Path file, String text) {
        try {
            Path parent = file.toAbsolutePath().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path temp = file.resolveSibling(file.getFileName().toString() + ".tmp");
            Files.write(temp, text.getBytes(StandardCharsets.UTF_8));
            try {
                Files.move(temp, file, StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(temp, file, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException e) {
            throw new IllegalStateException("cannot write " + file + ": " + e.getMessage(), e);
        }
    }

    /**
     * Loads a saved state for a resume.
     *
     * @throws IllegalArgumentException when the file belongs to another suite
     * @throws IllegalStateException    when the file cannot be read or parsed
     */
    public static EvalState load(Path file, EvalDataset dataset, int nPredict) {
        Object parsed;
        try {
            parsed = Json.parse(Files.readString(file, StandardCharsets.UTF_8), Json.Limits.UNTRUSTED);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read the eval state " + file + ": "
                    + e.getMessage(), e);
        } catch (JsonException e) {
            throw new IllegalStateException("the eval state " + file + " is not JSON: "
                    + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalStateException("the eval state " + file + " is not a JSON object");
        }
        Map<?, ?> root = (Map<?, ?>) parsed;
        String id = root.get("id") == null ? null : String.valueOf(root.get("id"));
        if (!dataset.type().equals(id)) {
            throw new IllegalArgumentException("the eval state " + file + " is for dataset '"
                    + id + "', not '" + dataset.type() + "'");
        }
        Object configValue = root.get("sampling_config");
        Map<String, Object> config = new LinkedHashMap<>();
        if (configValue instanceof Map) {
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) configValue).entrySet()) {
                config.put(String.valueOf(entry.getKey()), entry.getValue());
            }
        }
        EvalState state = new EvalState(dataset,
                root.get("model_name") == null ? null : String.valueOf(root.get("model_name")),
                EvalSampling.fromMap(config, nPredict), file);
        state.totalTime = 0.0;
        Object tasksValue = root.get("tasks");
        if (tasksValue instanceof List) {
            for (Object entry : (List<?>) tasksValue) {
                String taskId = String.valueOf(entry);
                int index = problemIndex(taskId);
                if (index < 0 || index >= dataset.size()) {
                    throw new IllegalArgumentException("the eval state " + file + " names task '"
                            + taskId + "', outside the " + dataset.type() + " dataset ("
                            + dataset.size() + " questions)");
                }
                state.tasks.add(new TaskRef(index, taskId));
                state.allTasks.add(new TaskRef(index, taskId));
            }
        }
        Object statesValue = root.get("task_states");
        if (statesValue instanceof Map) {
            Map<?, ?> states = (Map<?, ?>) statesValue;
            Object casesValue = states.get("cases");
            if (casesValue instanceof Map) {
                for (Map.Entry<?, ?> entry : ((Map<?, ?>) casesValue).entrySet()) {
                    if (entry.getValue() instanceof Map) {
                        EvalCase c = EvalCase.fromJson((Map<?, ?>) entry.getValue());
                        state.cases.put(String.valueOf(entry.getKey()), c);
                    }
                }
            }
            state.total = longOf(states.get("total"), state.cases.size());
            state.correct = (int) longOf(states.get("correct"), countCorrect(state.cases.values()));
            state.totalTime = states.get("total_time") instanceof Number
                    ? ((Number) states.get("total_time")).doubleValue() : 0.0;
        }
        if (state.total == 0) {
            state.total = state.cases.size();
            state.correct = countCorrect(state.cases.values());
        }
        return state;
    }

    private static long longOf(Object value, long fallback) {
        return value instanceof Number ? ((Number) value).longValue() : fallback;
    }

    /** The dataset index encoded in a task id's last segment, 0 when it has none. */
    static int problemIndex(String taskId) {
        int last = taskId.lastIndexOf('_');
        if (last < 0) {
            return 0;
        }
        try {
            return Integer.parseInt(taskId.substring(last + 1));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /** The pass number encoded in a task id's second-to-last segment, 0 when it has none. */
    static int chunkIndex(String taskId) {
        int last = taskId.lastIndexOf('_');
        if (last < 0) {
            return 0;
        }
        int second = taskId.lastIndexOf('_', last - 1);
        if (second < 0) {
            return 0;
        }
        try {
            return Integer.parseInt(taskId.substring(second + 1, last));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /** The "Results: C/T correct" block. */
    public void printExistingSummary(Appendable out) {
        int completed;
        int right;
        double[] ci;
        synchronized (lock) {
            completed = completedUnlocked();
            right = countCorrect(cases.values());
            ci = wilsonInterval(right, completed);
        }
        try {
            out.append("============================================================\n");
            if (completed == 0) {
                out.append("Results: 0/0 correct (0.0%)\n");
            } else {
                out.append(String.format(Locale.ROOT, "Results: %d/%d correct (%.1f%%) [%.1f%%, %.1f%%]%n",
                        right, completed, right * 100.0 / completed, ci[0] * 100, ci[1] * 100));
            }
            out.append("============================================================\n");
            printServerErrors(out);
        } catch (IOException e) {
            throw new IllegalStateException("cannot print the summary", e);
        }
    }

    /** The server-error count line, when there is one. */
    public void printServerErrors(Appendable out) {
        int errors = nServerErrors();
        if (errors == 0) {
            return;
        }
        try {
            out.append("Server errors: " + errors + " (not counted in accuracy; run with --resume"
                    + " to retry them)\n");
        } catch (IOException e) {
            throw new IllegalStateException("cannot print the server errors", e);
        }
    }

    /** The task table the reference prints before a run. */
    public void printAllTasks(Appendable out) {
        try {
            out.append("\nTasks:\n");
            out.append("  Task ID             Dataset  Prompt (first 40 chars)"
                    + "                        Expected    Answer       Tokens  T/s     Gen s  Status\n");
            synchronized (lock) {
                for (TaskRef ref : tasksToSave()) {
                    EvalQuestion question = dataset.question(ref.index());
                    EvalCase c = cases.get(ref.taskId());
                    String status = c == null ? EvalCase.PENDING : c.status();
                    String answer = c != null && EvalCase.OK.equals(status) && c.answer() != null
                            ? c.answer() : "N/A";
                    String tokens = c != null && c.tokens() != null ? String.valueOf(c.tokens()) : "N/A";
                    String tps = c != null && c.tpsGen() != null
                            ? String.format(Locale.ROOT, "%.1f", c.tpsGen()) : "N/A";
                    String gen = c != null && c.tGenMs() != null
                            ? String.format(Locale.ROOT, "%.1f", c.tGenMs() / 1000.0) : "N/A";
                    String server = c != null && c.serverName() != null ? c.serverName() : "";
                    String symbol = c != null && c.correct() ? "✓ " : EvalCase.OK.equals(status) ? "✗ " : "";
                    out.append(String.format(Locale.ROOT, "  %-20s %s   %-40s    %-10s %-10s %-6s %-6s %-8s %s%s  %s%n",
                            ref.taskId(), datasetType.toUpperCase(Locale.ROOT),
                            truncate(firstLine(question.text()), 40), question.expected(), answer,
                            tokens, tps, gen, symbol, status, server));
                }
            }
            out.append('\n');
        } catch (IOException e) {
            throw new IllegalStateException("cannot print the tasks", e);
        }
    }

    /** One progress line for a finished case. */
    public void printProgress(Appendable out, EvalCase c, int totalTasks, int nCorrect,
                              int processed) {
        String answer = c.answer() == null ? "N/A" : c.answer();
        String tokens = c.tokens() == null ? "N/A" : String.valueOf(c.tokens());
        String tps = c.tpsGen() == null ? "N/A" : String.format(Locale.ROOT, "%.1f", c.tpsGen());
        String gen = c.tGenMs() == null ? "N/A" : String.format(Locale.ROOT, "%.1f", c.tGenMs() / 1000.0);
        String server = c.serverName() == null ? "N/A" : c.serverName();
        String question = truncate(firstLine(c.questionText()), 43);
        double ratio = processed > 0 ? (double) nCorrect / processed : 0.0;
        String symbol = c.correct() ? "✓" : c.serverError() ? "⚠" : "✗";
        try {
            out.append(String.format(Locale.ROOT,
                    "%3d/%3d  %-20s %s   %-40s    %-16s %-28s %-6s %-6s %-8s %s  [%3d/%3d, %.3f]  %s%n",
                    processed, totalTasks, c.taskId(), datasetType.toUpperCase(Locale.ROOT), question,
                    c.expected(), answer, tokens, tps, gen, symbol, nCorrect, processed, ratio,
                    server));
        } catch (IOException e) {
            throw new IllegalStateException("cannot print the progress", e);
        }
    }

    private static String firstLine(String text) {
        int newline = text.indexOf('\n');
        return newline < 0 ? text : text.substring(0, newline);
    }

    /** The first line cut to at most {@code max} characters. */
    private static String truncate(String text, int max) {
        String line = text;
        if (line.length() > max) {
            return line.substring(0, max - 1) + "...";
        }
        return line + "...";
    }

    /** The readable report: a summary bar, the task table with details, and the failure causes. */
    private String html(Map<String, Object> root) {
        Map<?, ?> states = (Map<?, ?>) root.get("task_states");
        Map<?, ?> caseObjects = (Map<?, ?>) states.get("cases");
        Stats stats = stats(caseObjects);
        StringBuilder html = new StringBuilder();
        appendHead(html, root);
        appendSummary(html, root, states, stats);
        appendRows(html, caseObjects);
        appendCauses(html, stats);
        html.append("</body>\n</html>\n");
        return html.toString();
    }

    /** The verdict counts and failure causes the report's bar and footer show. */
    private static final class Stats {

        private int completed;
        private int right;
        private int pending;
        private int serverErrors;
        private final Map<String, Integer> causes = new LinkedHashMap<>();
    }

    private static Stats stats(Map<?, ?> caseObjects) {
        Stats stats = new Stats();
        for (Object value : caseObjects.values()) {
            Map<?, ?> c = (Map<?, ?>) value;
            String status = String.valueOf(c.get("status"));
            boolean correctCase = Boolean.TRUE.equals(c.get("correct"));
            if (EvalCase.OK.equals(status)) {
                stats.completed++;
                if (correctCase) {
                    stats.right++;
                } else {
                    String cause = "incorrect";
                    Object log = c.get("grader_log");
                    if (log instanceof Map && ((Map<?, ?>) log).get("grader_type") != null) {
                        cause = "incorrect (" + ((Map<?, ?>) log).get("grader_type") + ")";
                    }
                    stats.causes.put(cause, stats.causes.getOrDefault(cause, 0) + 1);
                }
            } else if (EvalCase.PENDING.equals(status)) {
                stats.pending++;
            } else {
                if (EvalCase.SERVER_ERROR.equals(status)) {
                    stats.serverErrors++;
                }
                stats.causes.put(status, stats.causes.getOrDefault(status, 0) + 1);
            }
        }
        return stats;
    }

    private static void appendHead(StringBuilder html, Map<String, Object> root) {
        html.append("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n<title>")
                .append(escape(String.valueOf(root.get("id")).toUpperCase(Locale.ROOT)))
                .append(" Eval</title>\n<style>\n")
                .append("body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#fff;color:#222}\n")
                .append(".bar{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:2px 12px;font-size:13px;padding:8px 0}\n")
                .append(".label{color:#888}.value{color:#222}\n")
                .append("table{width:100%;border-collapse:collapse;font-size:13px;font-family:ui-monospace,monospace}\n")
                .append("th{text-align:left;padding:6px 8px;border-bottom:2px solid #ccc}\n")
                .append("td{padding:4px 8px;border-bottom:1px solid #eee;vertical-align:top}\n")
                .append(".task-row{cursor:pointer}.task-row:hover{background:#f5f5f5}\n")
                .append(".correct{color:#1a7f37}.incorrect{color:#cf222e}.pending{color:#888}\n")
                .append(".server-error{color:#8250df}.error{color:#9a6700}\n")
                .append(".details-row{display:none}.details-row.open{display:table-row}\n")
                .append(".details-content{padding:8px 16px;background:#f6f8fa;font-size:12px}\n")
                .append(".details-content pre{background:#fff;border:1px solid #e1e4e8;padding:8px;")
                .append("overflow-x:auto;white-space:pre-wrap;word-wrap:break-word;margin:4px 0 8px}\n")
                .append("</style>\n<script>function toggleDetails(id){")
                .append("var r=document.getElementById('details-'+id);r.classList.toggle('open');}</script>\n")
                .append("</head>\n<body>\n<div class=\"bar\">\n");
    }

    private void appendSummary(StringBuilder html, Map<String, Object> root, Map<?, ?> states,
                               Stats stats) {
        double[] ci = wilsonInterval(stats.right, stats.completed);
        double accuracy = stats.completed == 0 ? 0.0 : stats.right * 100.0 / stats.completed;
        appendBar(html, "Dataset", String.valueOf(root.get("id")).toUpperCase(Locale.ROOT));
        appendBar(html, "Model", root.get("model_name") == null ? "N/A"
                : String.valueOf(root.get("model_name")));
        appendBar(html, "Accuracy", String.format(Locale.ROOT, "%.1f%% [%.1f%%, %.1f%%]",
                accuracy, ci[0] * 100, ci[1] * 100));
        appendBar(html, "Correct", stats.right + " / " + stats.completed);
        appendBar(html, "Pending", String.valueOf(stats.pending));
        appendBar(html, "Server errors", String.valueOf(stats.serverErrors));
        appendBar(html, "Time", String.format(Locale.ROOT, "%.1fs", states.get("total_time")));
        appendBar(html, "Sampling", samplingText());
        html.append("</div>\n<table>\n<thead><tr>")
                .append("<th>Task ID</th><th>Status</th><th>Expected</th><th>Answer</th>")
                .append("<th>Tokens</th><th>T/s</th><th>Gen s</th><th>Server</th></tr></thead>\n<tbody>\n");
    }

    private void appendRows(StringBuilder html, Map<?, ?> caseObjects) {
        for (TaskRef ref : tasksToSave()) {
            Map<?, ?> c = (Map<?, ?>) caseObjects.get(ref.taskId());
            if (c == null) {
                continue;
            }
            String status = String.valueOf(c.get("status"));
            String css = statusClass(status, Boolean.TRUE.equals(c.get("correct")));
            html.append("<tr class=\"task-row\" onclick=\"toggleDetails('").append(escape(ref.taskId()))
                    .append("')\"><td>").append(escape(ref.taskId())).append("</td><td class=\"")
                    .append(css).append("\">").append(statusSymbol(status, c)).append("</td><td>")
                    .append(escape(text(c.get("expected")))).append("</td><td class=\"")
                    .append(css).append("\">")
                    .append(escape(EvalCase.OK.equals(status) ? text(c.get("answer")) : ""))
                    .append("</td><td>").append(number(c.get("tokens"))).append("</td><td>")
                    .append(decimal(c.get("tps_gen"))).append("</td><td>")
                    .append(seconds(c.get("t_gen_ms"))).append("</td><td>")
                    .append(escape(text(c.get("server_name")))).append("</td></tr>\n");
            appendDetails(html, ref, c);
        }
        html.append("</tbody>\n</table>\n");
    }

    private static void appendDetails(StringBuilder html, TaskRef ref, Map<?, ?> c) {
        html.append("<tr id=\"details-").append(escape(ref.taskId()))
                .append("\" class=\"details-row\"><td colspan=\"8\"><div class=\"details-content\">")
                .append("<b>Prompt</b><pre>").append(escape(text(c.get("prompt"))))
                .append("</pre><b>Response</b><pre>").append(escape(text(c.get("response"))))
                .append("</pre>");
        if (c.get("reasoning_content") != null) {
            html.append("<b>Reasoning</b><pre>")
                    .append(escape(text(c.get("reasoning_content")))).append("</pre>");
        }
        html.append("<b>Grader</b><pre>")
                .append(escape(Json.toString(c.get("grader_log"),
                        new Json.Style(true, "  ", ", ", ": ", false))))
                .append("</pre></div></td></tr>\n");
    }

    private static String statusSymbol(String status, Map<?, ?> c) {
        if (EvalCase.OK.equals(status)) {
            return Boolean.TRUE.equals(c.get("correct")) ? "✓" : "✗";
        }
        return EvalCase.PENDING.equals(status) ? "–" : EvalCase.SERVER_ERROR.equals(status) ? "⚠" : "!";
    }

    private static void appendCauses(StringBuilder html, Stats stats) {
        if (stats.causes.isEmpty()) {
            return;
        }
        html.append("<h3>Failures by cause</h3>\n<ul>\n");
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(stats.causes.entrySet());
        entries.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        for (Map.Entry<String, Integer> entry : entries) {
            html.append("<li>").append(escape(entry.getKey())).append(": ")
                    .append(entry.getValue()).append("</li>\n");
        }
        html.append("</ul>\n");
    }

    private static String statusClass(String status, boolean correct) {
        if (EvalCase.OK.equals(status)) {
            return correct ? "correct" : "incorrect";
        }
        if (EvalCase.PENDING.equals(status)) {
            return "pending";
        }
        return EvalCase.SERVER_ERROR.equals(status) ? "server-error" : "error";
    }

    private void appendBar(StringBuilder html, String label, String value) {
        html.append("<div class=\"label\">").append(escape(label)).append("</div><div class=\"value\">")
                .append(escape(value)).append("</div>\n");
    }

    private String samplingText() {
        Map<String, Object> config = sampling.asMap();
        if (config.isEmpty()) {
            return "default";
        }
        StringBuilder out = new StringBuilder();
        for (Map.Entry<String, Object> entry : config.entrySet()) {
            if (out.length() > 0) {
                out.append(", ");
            }
            out.append(entry.getKey()).append('=').append(entry.getValue());
        }
        return out.toString();
    }

    private static String text(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static String number(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static String decimal(Object value) {
        return value instanceof Number ? String.format(Locale.ROOT, "%.1f",
                ((Number) value).doubleValue()) : "";
    }

    private static String seconds(Object millis) {
        return millis instanceof Number ? String.format(Locale.ROOT, "%.1f",
                ((Number) millis).doubleValue() / 1000.0) : "";
    }

    /** HTML-escapes a value. */
    static String escape(String value) {
        if (value == null) {
            return "";
        }
        StringBuilder out = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '&':
                    out.append("&amp;");
                    break;
                case '<':
                    out.append("&lt;");
                    break;
                case '>':
                    out.append("&gt;");
                    break;
                case '"':
                    out.append("&quot;");
                    break;
                case '\'':
                    out.append("&#39;");
                    break;
                default:
                    out.append(c);
            }
        }
        return out.toString();
    }
}
