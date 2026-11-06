package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The metrics of a rerank (pooling) run, computed as vLLM's {@code calculate_metrics_for_embeddings}
 * computes them: the completed and failed counts, the input tokens and input sequences, the request
 * and input-sequence throughputs over the run's duration, and the end-to-end latency statistics.
 * The report and the result JSON carry the reference's pooling keys.
 */
final class BenchPoolingMetrics {

    final int completed;
    final int failed;
    final long totalInput;
    final long totalInputSequences;
    final double requestThroughput;
    final double inputSequenceThroughput;
    final double totalTokenThroughput;
    final ServeBenchMetrics.Series e2el;

    private BenchPoolingMetrics(int completed, int failed, long totalInput, long totalInputSequences,
                                double requestThroughput, double inputSequenceThroughput,
                                double totalTokenThroughput, ServeBenchMetrics.Series e2el) {
        this.completed = completed;
        this.failed = failed;
        this.totalInput = totalInput;
        this.totalInputSequences = totalInputSequences;
        this.requestThroughput = requestThroughput;
        this.inputSequenceThroughput = inputSequenceThroughput;
        this.totalTokenThroughput = totalTokenThroughput;
        this.e2el = e2el;
    }

    /** The metrics of one rerank run over its outputs and duration. */
    static BenchPoolingMetrics compute(List<BenchRequestOutput> outputs, double durationSec,
                                       List<Double> percentiles) {
        int completed = 0;
        int failed = 0;
        long totalInput = 0;
        long totalSequences = 0;
        List<Double> e2els = new ArrayList<>();
        for (BenchRequestOutput out : outputs) {
            if (!out.success) {
                failed++;
                continue;
            }
            completed++;
            totalInput += out.promptLen;
            totalSequences += out.numInputSequences;
            e2els.add(out.latency);
        }
        return new BenchPoolingMetrics(completed, failed, totalInput, totalSequences,
                completed / durationSec, totalSequences / durationSec, totalInput / durationSec,
                new ServeBenchMetrics.Series(e2els, percentiles));
    }

    /** The report in the reference's layout. */
    void print(Appendable out, double durationSec, double requestRate, Integer maxConcurrency)
            throws IOException {
        banner(out, " Serving Benchmark Result ", '=');
        line(out, "Successful requests:", String.valueOf(completed));
        line(out, "Failed requests:", String.valueOf(failed));
        if (maxConcurrency != null) {
            line(out, "Maximum request concurrency:", String.valueOf(maxConcurrency));
        }
        if (!Double.isInfinite(requestRate)) {
            line(out, "Request rate configured (RPS):", f2(requestRate));
        }
        line(out, "Benchmark duration (s):", f2(durationSec));
        line(out, "Total input tokens:", String.valueOf(totalInput));
        line(out, "Request throughput (req/s):", f2(requestThroughput));
        line(out, "Input throughput (inputs/s):", f2(inputSequenceThroughput));
        line(out, "Total token throughput (tok/s):", f2(totalTokenThroughput));
        banner(out, "End-to-end Latency", '-');
        line(out, "Mean E2EL (ms):", f2(e2el.mean));
        line(out, "Median E2EL (ms):", f2(e2el.median));
        for (Map.Entry<Double, Double> p : e2el.percentiles.entrySet()) {
            line(out, "P" + ServeBenchMetrics.percentileName(p.getKey()) + " E2EL (ms):",
                    f2(p.getValue()));
        }
        banner(out, "", '=');
    }

    /** The result JSON of one rerank run. */
    Map<String, Object> resultJson(double durationSec, List<BenchRequestOutput> outputs,
                                   List<Double> percentiles, boolean detailed) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("duration", durationSec);
        result.put("completed", completed);
        result.put("total_input_tokens", totalInput);
        result.put("total_input_sequences", totalInputSequences);
        result.put("request_throughput", requestThroughput);
        result.put("input_sequence_throughput", inputSequenceThroughput);
        result.put("total_token_throughput", totalTokenThroughput);
        List<Integer> inputLens = new ArrayList<>();
        List<Double> latencies = new ArrayList<>();
        List<Double> queueTimes = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        for (BenchRequestOutput out : outputs) {
            inputLens.add(out.promptLen);
            latencies.add(out.success ? out.latency : null);
            queueTimes.add(out.success ? out.clientQueueTime : null);
            errors.add(out.error);
        }
        result.put("input_lens", inputLens);
        result.put("latencies", latencies);
        result.put("queue_times", queueTimes);
        result.put("errors", errors);
        result.put("mean_e2el_ms", e2el.mean);
        result.put("median_e2el_ms", e2el.median);
        result.put("std_e2el_ms", e2el.std);
        for (Map.Entry<Double, Double> p : e2el.percentiles.entrySet()) {
            result.put("p" + ServeBenchMetrics.percentileName(p.getKey()) + "_e2el_ms", p.getValue());
        }
        return result;
    }

    private static void banner(Appendable out, String title, char c) throws IOException {
        int pad = Math.max(0, 50 - title.length());
        int left = pad / 2;
        StringBuilder line = new StringBuilder();
        for (int i = 0; i < left; i++) {
            line.append(c);
        }
        line.append(title);
        while (line.length() < 50) {
            line.append(c);
        }
        out.append(line).append('\n');
    }

    private static void line(Appendable out, String label, String value) throws IOException {
        out.append(String.format(Locale.ROOT, "%-40s %-10s", label, value)).append('\n');
    }

    private static String f2(double value) {
        return String.format(Locale.ROOT, "%.2f", value);
    }
}
