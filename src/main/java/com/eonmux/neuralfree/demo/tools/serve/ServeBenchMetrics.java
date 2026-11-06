package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * The serving benchmark's metrics, computed as vLLM's {@code calculate_metrics} computes them: the
 * throughputs over the run's duration, the mean, median, standard deviation and the selected
 * percentiles of the TTFT, the TPOT (mean time per output token, first token excluded), the ITL and
 * the end-to-end latency, the goodput against the service level objectives, and the peak per-second
 * output token count with its concurrent request count. The report prints in the reference's
 * layout and the result JSON carries its key names.
 */
public final class ServeBenchMetrics {

    /** One latency series' statistics, in milliseconds. */
    public static final class Series {

        public final double mean;
        public final double std;
        public final double median;
        public final Map<Double, Double> percentiles = new TreeMap<>();

        Series(List<Double> seconds, List<Double> percentiles) {
            this.mean = BenchStats.mean(seconds) * 1000.0;
            this.std = BenchStats.std(seconds) * 1000.0;
            this.median = BenchStats.median(seconds) * 1000.0;
            for (double p : percentiles) {
                this.percentiles.put(p, BenchStats.percentile(seconds, p) * 1000.0);
            }
        }
    }

    /** The run's aggregates; the latency series are null when no request succeeded. */
    public final int completed;
    public final int failed;
    public final long totalInput;
    public final long totalOutput;
    public final double requestThroughput;
    public final double requestGoodput;
    public final double outputThroughput;
    public final double totalTokenThroughput;
    public final Series ttft;
    public final Series tpot;
    public final Series itl;
    public final Series e2el;
    public final Series clientQueueTime;
    public final Series e2elIncludingClientQueue;
    public final double maxOutputTokensPerS;
    public final int maxConcurrentRequests;
    public final double rtfx;
    public final List<Integer> actualOutputLens;

    private ServeBenchMetrics(int completed, int failed, long totalInput, long totalOutput,
                              double requestThroughput, double requestGoodput, double outputThroughput,
                              double totalTokenThroughput, Series ttft, Series tpot, Series itl, Series e2el,
                              Series clientQueueTime, Series e2elIncludingClientQueue,
                              double maxOutputTokensPerS, int maxConcurrentRequests, double rtfx,
                              List<Integer> actualOutputLens) {
        this.completed = completed;
        this.failed = failed;
        this.totalInput = totalInput;
        this.totalOutput = totalOutput;
        this.requestThroughput = requestThroughput;
        this.requestGoodput = requestGoodput;
        this.outputThroughput = outputThroughput;
        this.totalTokenThroughput = totalTokenThroughput;
        this.ttft = ttft;
        this.tpot = tpot;
        this.itl = itl;
        this.e2el = e2el;
        this.clientQueueTime = clientQueueTime;
        this.e2elIncludingClientQueue = e2elIncludingClientQueue;
        this.maxOutputTokensPerS = maxOutputTokensPerS;
        this.maxConcurrentRequests = maxConcurrentRequests;
        this.rtfx = rtfx;
        this.actualOutputLens = actualOutputLens;
    }

    /** The metrics of a run without a concurrency cap or a configured rate. */
    public static ServeBenchMetrics compute(List<BenchRequestOutput> outputs, double durationSec,
                                            List<Double> percentiles, Map<String, Double> goodput) {
        return compute(outputs, durationSec, percentiles, goodput, null, Double.POSITIVE_INFINITY);
    }

    /**
     * The metrics of one run: the outputs in workload order, the run's duration in seconds, the
     * percentiles to report and the goodput objectives in milliseconds (empty for none).
     *
     * @param maxConcurrency the client-side concurrency cap, or null for none; when present the
     *                       queue-wait series report, as the reference reports them
     * @param requestRate    the configured rate; a finite one adds the queue-inclusive end-to-end
     *                       series, an infinite one leaves it absent
     */
    public static ServeBenchMetrics compute(List<BenchRequestOutput> outputs, double durationSec,
                                            List<Double> percentiles, Map<String, Double> goodput,
                                            Integer maxConcurrency, double requestRate) {
        List<Double> ttfts = new ArrayList<>();
        List<Double> tpots = new ArrayList<>();
        List<Double> itls = new ArrayList<>();
        List<Double> e2els = new ArrayList<>();
        List<Double> queueTimes = new ArrayList<>();
        List<Double> e2elsIncludingQueue = new ArrayList<>();
        List<Integer> outputLens = new ArrayList<>();
        long totalInput = 0;
        long totalOutput = 0;
        double audioSeconds = 0.0;
        int completed = 0;
        int goodCompleted = 0;
        for (BenchRequestOutput out : outputs) {
            if (!out.success) {
                outputLens.add(0);
                continue;
            }
            int outputLen = out.outputTokens != null && out.outputTokens > 0
                    ? out.outputTokens : out.itls.size() + 1;
            outputLens.add(outputLen);
            totalInput += out.promptLen;
            totalOutput += outputLen;
            audioSeconds += out.inputAudioDuration;
            double tpot = 0.0;
            if (outputLen > 1) {
                tpot = (out.latency - out.ttft) / (outputLen - 1);
                tpots.add(tpot);
            }
            itls.addAll(out.itls);
            ttfts.add(out.ttft);
            e2els.add(out.latency);
            if (maxConcurrency != null) {
                queueTimes.add(out.clientQueueTime);
                if (!Double.isInfinite(requestRate)) {
                    e2elsIncludingQueue.add(out.latency + out.clientQueueTime);
                }
            }
            completed++;
            if (isGood(out, tpot, goodput)) {
                goodCompleted++;
            }
        }
        Series ttftSeries = ttfts.isEmpty() ? null : new Series(ttfts, percentiles);
        Series tpotSeries = ttfts.isEmpty() ? null : new Series(tpots, percentiles);
        Series itlSeries = ttfts.isEmpty() ? null : new Series(itls, percentiles);
        Series e2elSeries = new Series(e2els, percentiles);
        Series queueSeries = maxConcurrency == null ? null : new Series(queueTimes, percentiles);
        Series queueInclusiveSeries = maxConcurrency == null || Double.isInfinite(requestRate)
                ? null : new Series(e2elsIncludingQueue, percentiles);
        Peak peak = completed == 0 ? new Peak(0.0, 0) : peakPerSecond(outputs);
        return new ServeBenchMetrics(completed, outputs.size() - completed, totalInput, totalOutput,
                completed / durationSec, goodCompleted / durationSec,
                totalOutput / durationSec, (totalInput + totalOutput) / durationSec,
                ttftSeries, tpotSeries, itlSeries, e2elSeries, queueSeries, queueInclusiveSeries,
                peak.tokensPerSecond, peak.concurrentRequests, audioSeconds / durationSec, outputLens);
    }

    /** Whether one request meets every service level objective ({@code slo >= request metric}). */
    private static boolean isGood(BenchRequestOutput out, double tpot, Map<String, Double> goodput) {
        if (goodput.isEmpty()) {
            return false;
        }
        Double ttftSlo = goodput.get("ttft");
        if (ttftSlo != null && ttftSlo / 1000.0 < out.ttft) {
            return false;
        }
        Double tpotSlo = goodput.get("tpot");
        if (tpotSlo != null && tpotSlo / 1000.0 < tpot) {
            return false;
        }
        Double e2elSlo = goodput.get("e2el");
        if (e2elSlo != null && e2elSlo / 1000.0 < out.latency) {
            return false;
        }
        return true;
    }

    /** The per-second peak: the most output tokens any second carries and the requests active in it. */
    private static Peak peakPerSecond(List<BenchRequestOutput> outputs) {
        double minStart = Double.POSITIVE_INFINITY;
        double maxEnd = Double.NEGATIVE_INFINITY;
        List<BenchRequestOutput> successful = new ArrayList<>();
        for (BenchRequestOutput out : outputs) {
            if (!out.success) {
                continue;
            }
            successful.add(out);
            minStart = Math.min(minStart, out.startTime);
            maxEnd = Math.max(maxEnd, out.startTime + out.latency);
        }
        int durationSeconds = (int) Math.ceil(maxEnd - minStart) + 1;
        int[] tokens = new int[durationSeconds];
        int[] concurrent = new int[durationSeconds];
        for (BenchRequestOutput out : successful) {
            double tokenTime = out.startTime + out.ttft;
            int tokenSecond = (int) (tokenTime - minStart);
            if (tokenSecond >= 0 && tokenSecond < durationSeconds) {
                tokens[tokenSecond]++;
            }
            for (double itl : out.itls) {
                tokenTime += itl;
                tokenSecond = (int) (tokenTime - minStart);
                if (tokenSecond >= 0 && tokenSecond < durationSeconds) {
                    tokens[tokenSecond]++;
                }
            }
            int startSecond = Math.max(0, (int) (out.startTime - minStart));
            int endSecond = Math.min(durationSeconds - 1, (int) (out.startTime + out.latency - minStart));
            for (int s = startSecond; s <= endSecond; s++) {
                concurrent[s]++;
            }
        }
        double maxTokens = 0.0;
        int maxConcurrent = 0;
        for (int s = 0; s < durationSeconds; s++) {
            maxTokens = Math.max(maxTokens, tokens[s]);
            maxConcurrent = Math.max(maxConcurrent, concurrent[s]);
        }
        return new Peak(maxTokens, maxConcurrent);
    }

    /** The peak of one run. */
    private static final class Peak {

        final double tokensPerSecond;
        final int concurrentRequests;

        Peak(double tokensPerSecond, int concurrentRequests) {
            this.tokensPerSecond = tokensPerSecond;
            this.concurrentRequests = concurrentRequests;
        }
    }

    /**
     * The report in the reference's layout: the run's facts, then a block per selected metric.
     *
     * @param durationSec     the run's duration in seconds, the report's context line
     * @param selectedMetrics the metrics whose blocks print ({@code ttft, tpot, itl, e2el})
     * @param requestRate     the configured rate, or infinity to skip the line
     * @param maxConcurrency  the configured cap, or null to skip the line
     * @param hasGoodput      whether the goodput line prints
     */
    public void print(Appendable out, double durationSec, List<String> selectedMetrics, double requestRate,
                      Integer maxConcurrency, boolean hasGoodput) throws IOException {
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
        line(out, "Total generated tokens:", String.valueOf(totalOutput));
        line(out, "Request throughput (req/s):", f2(requestThroughput));
        if (hasGoodput) {
            line(out, "Request goodput (req/s):", f2(requestGoodput));
        }
        line(out, "Output token throughput (tok/s):", f2(outputThroughput));
        line(out, "Peak output token throughput (tok/s):", f2(maxOutputTokensPerS));
        line(out, "Peak concurrent requests:", String.valueOf(maxConcurrentRequests));
        if (rtfx > 0.0) {
            line(out, "RTFx (Inverse Real-Time Factor):", f2(rtfx));
        }
        line(out, "Total token throughput (tok/s):", f2(totalTokenThroughput));
        metricBlock(out, selectedMetrics, "ttft", "TTFT", "Time to First Token");
        metricBlock(out, selectedMetrics, "tpot", "TPOT", "Time per Output Token (excl. 1st token)");
        metricBlock(out, selectedMetrics, "itl", "ITL", "Inter-token Latency");
        metricBlock(out, selectedMetrics, "e2el", "E2EL", "End-to-end Latency");
        metricBlock(out, selectedMetrics, "client_queue_time", "Client Queue Time",
                "Client-side Queueing");
        metricBlock(out, selectedMetrics, "e2el_including_client_queue", "E2EL incl. Client Queue",
                "Queue-inclusive End-to-end Latency");
        banner(out, "", '=');
    }

    /** One metric's statistics block, skipped when the run has no sample for it. */
    private void metricBlock(Appendable out, List<String> selectedMetrics, String name, String label,
                             String header) throws IOException {
        if (!selectedMetrics.contains(name)) {
            return;
        }
        Series series = series(name);
        if (series == null) {
            return;
        }
        banner(out, header, '-');
        line(out, "Mean " + label + " (ms):", f2(series.mean));
        line(out, "Median " + label + " (ms):", f2(series.median));
        for (Map.Entry<Double, Double> p : series.percentiles.entrySet()) {
            line(out, "P" + percentileName(p.getKey()) + " " + label + " (ms):", f2(p.getValue()));
        }
    }

    /** The series of {@code name}, or null. */
    private Series series(String name) {
        switch (name) {
            case "ttft":
                return ttft;
            case "tpot":
                return tpot;
            case "itl":
                return itl;
            case "e2el":
                return e2el;
            case "client_queue_time":
                return clientQueueTime;
            case "e2el_including_client_queue":
                return e2elIncludingClientQueue;
            default:
                return null;
        }
    }

    /** A centered banner: {@code title} centered within a 50-character rule of {@code c}. */
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

    /** The result JSON of one run: the reference's keys, the per-request arrays only when detailed. */
    public Map<String, Object> resultJson(double durationSec, List<BenchRequestOutput> outputs,
                                          List<String> selectedMetrics, boolean hasGoodput, boolean detailed) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("duration", durationSec);
        result.put("completed", completed);
        result.put("failed", failed);
        result.put("total_input_tokens", totalInput);
        result.put("total_output_tokens", totalOutput);
        result.put("request_throughput", requestThroughput);
        result.put("request_goodput", hasGoodput ? requestGoodput : null);
        result.put("output_throughput", outputThroughput);
        result.put("total_token_throughput", totalTokenThroughput);
        result.put("max_output_tokens_per_s", maxOutputTokensPerS);
        result.put("max_concurrent_requests", maxConcurrentRequests);
        result.put("rtfx", rtfx);
        List<Integer> inputLens = new ArrayList<>();
        List<Double> latencies = new ArrayList<>();
        List<Double> queueTimes = new ArrayList<>();
        for (BenchRequestOutput o : outputs) {
            inputLens.add(o.promptLen);
            latencies.add(o.success ? o.latency : null);
            queueTimes.add(o.success ? o.clientQueueTime : null);
        }
        result.put("input_lens", inputLens);
        result.put("output_lens", new ArrayList<>(actualOutputLens));
        result.put("latencies", latencies);
        result.put("queue_times", queueTimes);
        for (String name : List.of("ttft", "tpot", "itl", "e2el", "client_queue_time",
                "e2el_including_client_queue")) {
            if (!selectedMetrics.contains(name)) {
                continue;
            }
            Series series = series(name);
            if (series == null) {
                continue;
            }
            result.put("mean_" + name + "_ms", series.mean);
            result.put("median_" + name + "_ms", series.median);
            result.put("std_" + name + "_ms", series.std);
            for (Map.Entry<Double, Double> p : series.percentiles.entrySet()) {
                result.put("p" + percentileName(p.getKey()) + "_" + name + "_ms", p.getValue());
            }
        }
        if (detailed) {
            List<Double> ttfts = new ArrayList<>();
            List<List<Double>> itls = new ArrayList<>();
            List<Double> startTimes = new ArrayList<>();
            List<String> texts = new ArrayList<>();
            List<String> errors = new ArrayList<>();
            for (BenchRequestOutput o : outputs) {
                ttfts.add(o.success ? o.ttft : null);
                itls.add(o.success ? new ArrayList<>(o.itls) : List.of());
                startTimes.add(o.success ? o.startTime : null);
                texts.add(o.generatedText.toString());
                errors.add(o.error);
            }
            result.put("ttfts", ttfts);
            result.put("itls", itls);
            result.put("start_times", startTimes);
            result.put("generated_texts", texts);
            result.put("errors", errors);
        }
        return result;
    }

    /** A percentile's label: the integer when it is one, the value itself otherwise. */
    static String percentileName(double p) {
        return (int) p == p ? String.valueOf((int) p) : String.valueOf(p);
    }

    /** One aligned report line, the reference's 40/10 column layout. */
    private static void line(Appendable out, String label, String value) throws IOException {
        out.append(String.format(Locale.ROOT, "%-40s %-10s", label, value)).append('\n');
    }

    private static String f2(double v) {
        return String.format(Locale.ROOT, "%.2f", v);
    }
}
