package com.eonmux.neuralfree.demo.tools.eval;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Drives a run: one worker pool per server, the pools competing for the tasks of one shared queue,
 * so a run spreads over every configured endpoint. Each worker sends its task's prompt, records the
 * verdict, saves the state, and reports the case; the driver drains the results in completion order
 * and prints the reference's progress lines and summary.
 */
public final class EvalRunner {

    private final List<EvalServer> servers;
    private final EvalGrader grader;
    private final String model;
    private final int retries;
    private final double retryDelay;
    private final Appendable out;
    private final Appendable err;

    /**
     * @param servers    the endpoints the run spreads over, at least one
     * @param retries    resends per task before it records a server error
     * @param retryDelay seconds before the first resend, doubled on each further one
     */
    public EvalRunner(List<EvalServer> servers, EvalGrader grader, String model, int retries,
                      double retryDelay, Appendable out, Appendable err) {
        if (servers == null || servers.isEmpty()) {
            throw new IllegalArgumentException("--server must name at least one server");
        }
        this.servers = new ArrayList<>(servers);
        this.grader = grader;
        this.model = model;
        this.retries = retries;
        this.retryDelay = retryDelay;
        this.out = out;
        this.err = err;
    }

    /**
     * Runs every task of {@code state}, saving the state after each case.
     *
     * @throws IllegalStateException when a server cannot be reached or a worker fails
     */
    public void evaluate(EvalState state, boolean verbose) {
        List<EvalClient> clients = new ArrayList<>();
        for (EvalServer server : servers) {
            clients.add(new EvalClient(server, model, retries, retryDelay,
                    line -> append(err, line + "\n")));
        }
        printHeader(state, clients);

        List<EvalState.TaskRef> tasks = state.tasks();
        BlockingQueue<EvalState.TaskRef> taskQueue = new LinkedBlockingQueue<>(tasks);
        int totalThreads = 0;
        for (EvalServer server : servers) {
            totalThreads += server.threads();
        }
        for (int i = 0; i < totalThreads; i++) {
            taskQueue.add(SENTINEL);
        }
        BlockingQueue<EvalCase> results = new LinkedBlockingQueue<>();

        List<ExecutorService> executors = new ArrayList<>();
        List<Future<?>> workers = new ArrayList<>();
        for (int s = 0; s < servers.size(); s++) {
            EvalServer server = servers.get(s);
            EvalClient client = clients.get(s);
            ExecutorService executor = Executors.newFixedThreadPool(server.threads(), r -> {
                Thread t = new Thread(r, "eval-" + server.name());
                t.setDaemon(true);
                return t;
            });
            executors.add(executor);
            for (int i = 0; i < server.threads(); i++) {
                workers.add(executor.submit(() -> runWorker(client, state, taskQueue, results)));
            }
        }

        int processed = 0;
        int nCorrect = 0;
        double sessionTime = 0.0;
        long last = System.nanoTime();
        try {
            while (processed < tasks.size()) {
                EvalCase result;
                try {
                    result = results.take();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("the evaluation was interrupted", e);
                }
                processed++;
                if (result.correct()) {
                    nCorrect++;
                }
                state.addResult(result);
                double elapsed = (System.nanoTime() - last) / 1e9;
                last = System.nanoTime();
                sessionTime += elapsed;
                state.addTime(elapsed);
                state.dump();
                state.printProgress(out, result, tasks.size(), nCorrect, processed);
                if (verbose) {
                    printVerbose(result);
                }
            }
        } finally {
            for (ExecutorService executor : executors) {
                executor.shutdownNow();
            }
            for (Future<?> worker : workers) {
                worker.cancel(true);
            }
        }
        for (Future<?> worker : workers) {
            try {
                worker.get();
            } catch (ExecutionException e) {
                throw new IllegalStateException("an evaluation worker failed: "
                        + e.getCause().getMessage(), e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("the evaluation was interrupted", e);
            } catch (java.util.concurrent.CancellationException e) {
                // a cancelled worker has already delivered its results or was never needed
            }
        }
        shutdown(executors);
        append(out, String.format(java.util.Locale.ROOT,
                "%nSession time: %.1fs | Total accumulated time: %.1fs%n", sessionTime,
                accumulated(state)));
        state.printExistingSummary(out);
        state.dump();
    }

    /** A worker's loop: take a task, run it, report it, until the sentinel. */
    private void runWorker(EvalClient client, EvalState state, BlockingQueue<EvalState.TaskRef> taskQueue,
                           BlockingQueue<EvalCase> results) {
        while (true) {
            EvalState.TaskRef task;
            try {
                task = taskQueue.take();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            if (task == SENTINEL || task == null) {
                return;
            }
            EvalCase result;
            try {
                result = process(client, state, task);
            } catch (RuntimeException e) {
                result = errorCase(state, task, e.getMessage());
            }
            results.add(result);
        }
    }

    /** One task: the request, the truncation, the verdict, and the case the state records. */
    private EvalCase process(EvalClient client, EvalState state, EvalState.TaskRef task) {
        EvalQuestion question = state.question(task.index());
        String sessionId = task.taskId() + "-"
                + UUID.randomUUID().toString().replace("-", "");
        try {
            EvalChat chat = client.generate(question.prompt(), state.sampling(), sessionId);
            if (!"stop".equals(chat.finishReason())) {
                Map<String, Object> log = new LinkedHashMap<>();
                log.put("finish_reason", chat.finishReason());
                return new EvalCase(task.taskId(), question.prompt(), question.expected(),
                        question.text(), chat.content(), null, log, false,
                        "error: finish_reason=" + chat.finishReason(), chat.tokens(),
                        chat.tpsGen(), chat.tGenMs(), chat.reasoningContent(),
                        client.server().name(), EvalState.chunkIndex(task.taskId()),
                        task.index());
            }
            String graded = EvalGrader.truncate(chat.content());
            EvalGrader.Grade grade = grader.grade(question.expected(), graded, question.prompt());
            Map<String, Object> log = new LinkedHashMap<>();
            log.put("pred", graded);
            log.put("grader_type", grader.type());
            if (grader.patternText() != null) {
                log.put("pattern", grader.patternText());
            }
            return new EvalCase(task.taskId(), question.prompt(), question.expected(),
                    question.text(), chat.content(), grade.answer(), log, grade.correct(),
                    EvalCase.OK, chat.tokens(), chat.tpsGen(), chat.tGenMs(),
                    chat.reasoningContent(), client.server().name(),
                    EvalState.chunkIndex(task.taskId()), task.index());
        } catch (EvalServerException e) {
            Map<String, Object> log = new LinkedHashMap<>();
            log.put("error", e.getMessage());
            return new EvalCase(task.taskId(), question.prompt(), question.expected(),
                    question.text(), null, null, log, false, EvalCase.SERVER_ERROR, null, null,
                    null, null, client.server().name(), EvalState.chunkIndex(task.taskId()),
                    task.index());
        }
    }

    /** A case a worker could not grade, recorded so a resume retries it. */
    private static EvalCase errorCase(EvalState state, EvalState.TaskRef task, String reason) {
        EvalQuestion question;
        try {
            question = state.question(task.index());
        } catch (RuntimeException e) {
            question = new EvalQuestion("", "", "");
        }
        Map<String, Object> log = new LinkedHashMap<>();
        log.put("error", reason == null ? "unexpected grading error" : reason);
        return new EvalCase(task.taskId(), question.prompt(), question.expected(), question.text(),
                null, null, log, false, "error: " + log.get("error"), null, null, null, null,
                null, EvalState.chunkIndex(task.taskId()), task.index());
    }

    private void printHeader(EvalState state, List<EvalClient> clients) {
        List<List<String>> models = new ArrayList<>();
        for (EvalClient client : clients) {
            try {
                models.add(client.models());
            } catch (EvalServerException e) {
                throw new IllegalStateException("cannot reach server " + client.server().name()
                        + " (" + client.server().url() + "): " + e.getMessage(), e);
            }
        }
        append(out, "\nProcessing " + state.tasks().size() + " "
                + state.datasetType().toUpperCase(java.util.Locale.ROOT) + " tasks ...\n");
        append(out, "Servers (" + servers.size() + "):\n");
        for (int i = 0; i < servers.size(); i++) {
            List<String> ids = models.get(i);
            String names = ids.isEmpty() ? "(none)" : String.join(", ", ids);
            append(out, "  " + (i + 1) + ". " + servers.get(i).name() + " — "
                    + servers.get(i).url() + " (" + servers.get(i).threads() + " threads) ["
                    + names + "]\n");
        }
        append(out, "Model: " + modelName() + "\n");
        append(out, "Grader: " + grader.type() + "\n");
        Map<String, Object> sampling = state.sampling().asMap();
        append(out, String.format(java.util.Locale.ROOT,
                "Sampling: temp=%s, top-k=%s, top-p=%s, min-p=%s, reasoning-effort=%s%n",
                samplingValue(sampling, "temperature"), samplingValue(sampling, "top_k"),
                samplingValue(sampling, "top_p"), samplingValue(sampling, "min_p"),
                samplingValue(sampling, "reasoning_effort")));
        append(out, "\n");
    }

    private String modelName() {
        return model == null ? "None" : model;
    }

    private static String samplingValue(Map<String, Object> sampling, String key) {
        Object value = sampling.get(key);
        return value == null ? "skip" : String.valueOf(value);
    }

    private void printVerbose(EvalCase c) {
        append(out, "\nCase " + c.taskId() + ": " + c.correct() + "\n");
        append(out, "  Expected: " + c.expected() + "\n");
        if (c.response() != null) {
            append(out, "  Response: " + c.response() + "\n");
        }
        if (c.answer() != null) {
            append(out, "  Answer: " + c.answer() + "\n");
        }
        append(out, "  Status: " + c.status() + "\n");
    }

    private static double accumulated(EvalState state) {
        // the runner adds each case's time to the state, so re-read it through a dump-independent path
        return state.accumulatedSeconds();
    }

    private static void shutdown(List<ExecutorService> executors) {
        for (ExecutorService executor : executors) {
            executor.shutdown();
        }
        for (ExecutorService executor : executors) {
            try {
                executor.awaitTermination(30, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    private static void append(Appendable sink, String text) {
        try {
            sink.append(text);
        } catch (IOException e) {
            throw new IllegalStateException("cannot write to the output", e);
        }
    }

    /** The sentinel that tells a worker there is no more work. */
    private static final EvalState.TaskRef SENTINEL = new EvalState.TaskRef(-1, "");
}
