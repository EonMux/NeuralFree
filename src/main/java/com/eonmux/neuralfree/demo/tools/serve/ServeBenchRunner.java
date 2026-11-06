package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

import com.eonmux.neuralfree.utils.json.Json;

/**
 * The orchestration one serving benchmark run shares: the server discovery (the model when none was
 * named, the tokenizer facts the datasets sample with), the workload sampling, the ready check, the
 * warmups, the requested server metrics' before-and-after snapshots, the probe requests, the
 * measured run over the arrival schedule (or the trace's own timestamps), the metrics report and the
 * result JSON. The {@code bench-serve}, {@code bench-throughput} and {@code bench-sweep} commands all
 * run through {@link #execute}; the steps print the reference's lines.
 */
public final class ServeBenchRunner {

    private ServeBenchRunner() {
    }

    /** The endpoint a dataset drives when {@code --endpoint} is absent. */
    static String resolvedEndpoint(ServeBenchOptions opts) {
        if (opts.endpointExplicit()) {
            return ServeBenchClient.canonicalEndpoint(opts.endpoint());
        }
        if (ServeBenchDataset.RERANK.contains(opts.datasetName())) {
            return ServeBenchClient.RERANK;
        }
        if (ServeBenchDataset.AUDIO.contains(opts.datasetName())) {
            return ServeBenchClient.AUDIO;
        }
        if (ServeBenchDataset.CHAT.contains(opts.datasetName())) {
            return ServeBenchClient.CHAT;
        }
        return ServeBenchClient.COMPLETIONS;
    }

    /**
     * One run: everything from discovery to the saved result file, returning the merged result JSON
     * (the run's facts plus the metrics). Every failure is an {@link IOException} naming its cause,
     * or an {@link IllegalArgumentException} naming the option.
     */
    public static Map<String, Object> execute(ServeBenchOptions opts, Appendable out, Appendable err)
            throws IOException {
        try {
            ServeBenchCommand.validate(opts, false);
        } catch (ServeBenchCommand.UsageException e) {
            throw new IOException(e.getMessage());
        }
        String baseUrl = opts.resolvedBaseUrl();
        String endpoint = resolvedEndpoint(opts);
        printTraffic(opts, endpoint, out);
        String requestModel;
        String resultModelId;
        if (opts.model() != null) {
            requestModel = opts.model();
            resultModelId = opts.model();
        } else {
            out.append("Model not specified, fetching first model from server...\n");
            BenchEndpoints.ServedModel model = BenchEndpoints.firstModel(baseUrl, opts.headers(),
                    opts.insecure());
            out.append("First model name: ").append(model.id)
                    .append(", first model id: ").append(model.root).append('\n');
            requestModel = model.id;
            resultModelId = model.root;
        }
        BenchTokenizer tokenizer = resolveTokenizer(opts, baseUrl, requestModel);
        BenchEndpoints.TokenizerInfo tokenizerInfo = tokenizerInfo(opts, tokenizer, baseUrl);
        ServeBenchDataset dataset = ServeBenchDataset.create(opts.datasetName(), opts, tokenizer,
                tokenizerInfo.vocabSize, tokenizerInfo.skipIds);
        List<BenchSample> samples = dataset.sample();
        if (samples.isEmpty()) {
            throw new IOException("the dataset sampled no request: raise --num-prompts or drop"
                    + " --no-oversample");
        }
        // a random dataset runs to its requested lengths, as the reference defaults ignore_eos for it
        boolean ignoreEos = opts.ignoreEos() || "random".equals(opts.datasetName())
                || "random-mm".equals(opts.datasetName()) || "random-rerank".equals(opts.datasetName());
        List<String> selectedMetrics = selectedMetrics(opts, endpoint);
        List<Double> selectedPercentiles = opts.percentiles();
        ServeBenchClient client = new ServeBenchClient(baseUrl, endpoint, opts.headers(), requestModel,
                opts.servedModelName(), ignoreEos, opts.samplingParams(), opts.logprobs(), opts.insecure(),
                opts.useBeamSearch());
        readyCheck(opts, client, samples, out, err);
        warmups(opts, client, samples, err);
        String[] requestModels = loraAssignment(opts, samples.size());
        if (opts.profile()) {
            startProfile(opts, baseUrl, out, err);
        }
        out.append("Starting main benchmark run...\n");
        printTrafficRun(opts, out);
        ArrivalSchedule schedule = schedule(opts, samples);
        PrometheusMetrics before = PrometheusMetrics.fetch(baseUrl, opts.headers(), opts.insecure());
        Probe probe = Probe.start(opts, client);
        long begin = System.nanoTime();
        List<BenchRequestOutput> outputs = client.runAll(samples, schedule, opts.maxConcurrency(),
                requestModels, progressSink(opts, samples.size(), err));
        double durationSec = (System.nanoTime() - begin) / 1e9;
        List<BenchRequestOutput> probes = probe == null ? null : probe.stop();
        PrometheusMetrics after = PrometheusMetrics.fetch(baseUrl, opts.headers(), opts.insecure());
        reportFailures(outputs, err);
        Optional<PrometheusMetrics.SpecDecode> specDecode = PrometheusMetrics.specDecode(before, after);
        Map<String, Object> result = ServeBenchClient.RERANK.equals(endpoint)
                ? poolingResult(opts, out, resultModelId, samples, outputs, durationSec, selectedPercentiles)
                : generationResult(opts, out, resultModelId, samples, outputs, durationSec, selectedMetrics);
        specDecode.ifPresent(stats -> stats.putInto(result));
        if (probes != null) {
            putProbeStats(result, probes, out);
        }
        if (opts.profile()) {
            stopProfile(opts, baseUrl, out, err);
        }
        finish(opts, out, resultModelId, result, outputs);
        return result;
    }

    /** Opens the server's profile window, reporting a refusal without aborting the benchmark. */
    private static void startProfile(ServeBenchOptions opts, String baseUrl, Appendable out,
                                     Appendable err) throws IOException {
        out.append("Starting profiler...\n");
        String error = BenchEndpoints.profile(baseUrl, "/start_profile", opts.headers(), opts.insecure());
        if (error == null) {
            out.append("Profiler started\n");
        } else {
            err.append("Profiler start failed: ").append(error).append('\n');
        }
    }

    /** Closes the server's profile window, reporting a refusal without aborting the benchmark. */
    private static void stopProfile(ServeBenchOptions opts, String baseUrl, Appendable out,
                                    Appendable err) throws IOException {
        out.append("Stopping profiler...\n");
        String error = BenchEndpoints.profile(baseUrl, "/stop_profile", opts.headers(), opts.insecure());
        if (error == null) {
            out.append("Profiler stopped\n");
        } else {
            err.append("Profiler stop failed: ").append(error).append('\n');
        }
    }

    /** The tokenizer the datasets count with: the local one, the server one, or none. */
    private static BenchTokenizer resolveTokenizer(ServeBenchOptions opts, String baseUrl, String modelId) {
        if (opts.skipTokenizerInit()) {
            return new BenchTokenizer() {
                @Override
                public int[] tokenize(String text) {
                    throw new IllegalStateException("--skip-tokenizer-init leaves no tokenizer to count '"
                            + text + "'");
                }
            };
        }
        if (opts.tokenizerPath() != null) {
            return LocalBenchTokenizer.open(opts.tokenizerPath());
        }
        return BenchEndpoints.tokenizer(baseUrl, opts.headers(), opts.insecure(), modelId);
    }

    /** The vocabulary facts the synthetic datasets sample from. */
    private static BenchEndpoints.TokenizerInfo tokenizerInfo(ServeBenchOptions opts,
                                                              BenchTokenizer tokenizer, String baseUrl)
            throws IOException {
        boolean needsVocabulary = ServeBenchDataset.SYNTHETIC.contains(opts.datasetName())
                || "burstgpt".equals(opts.datasetName()) || "timed_trace".equals(opts.datasetName());
        if (!needsVocabulary) {
            return new BenchEndpoints.TokenizerInfo(0, new int[0]);
        }
        if (tokenizer instanceof LocalBenchTokenizer) {
            return new BenchEndpoints.TokenizerInfo(tokenizer.vocabSize(), tokenizer.specialIds());
        }
        return BenchEndpoints.tokenizerInfo(baseUrl, opts.headers(), opts.insecure());
    }

    /** The metrics a run reports: the pooling endpoint's end-to-end latency alone, else the selection. */
    private static List<String> selectedMetrics(ServeBenchOptions opts, String endpoint) {
        if (ServeBenchClient.RERANK.equals(endpoint) && opts.percentileMetrics() == null) {
            return List.of("e2el");
        }
        return opts.percentileMetricNames();
    }

    /** The arrivals: the trace's own timestamps under {@code --self-timed}, else the traffic flags. */
    private static ArrivalSchedule schedule(ServeBenchOptions opts, List<BenchSample> samples) {
        if (opts.selfTimed()) {
            return ArrivalSchedule.ofTimestamps(samples);
        }
        return ArrivalSchedule.of(samples.size(), opts.requestRate(), opts.burstiness(),
                opts.rampUpStrategy(), opts.rampUpStartRps(), opts.rampUpEndRps(), new Random(opts.seed()));
    }

    /** The option line the run opens with, the reference's args echo. */
    private static void printTraffic(ServeBenchOptions opts, String endpoint, Appendable out)
            throws IOException {
        out.append("Serving benchmark: base_url=").append(opts.resolvedBaseUrl())
                .append(" endpoint=").append(endpoint)
                .append(" model=").append(opts.model() != null ? opts.model() : "<first from server>")
                .append(" dataset=").append(opts.datasetName())
                .append(" num_prompts=").append(String.valueOf(opts.numPrompts())).append('\n');
    }

    /** The traffic lines the measured run opens with, the reference's wording. */
    private static void printTrafficRun(ServeBenchOptions opts, Appendable out) throws IOException {
        if (opts.selfTimed()) {
            out.append("Self timing is set, using the timestamps from the trace file.\n");
            return;
        }
        if (opts.rampUpStrategy() != null) {
            out.append("Traffic ramp-up strategy: ").append(opts.rampUpStrategy()).append('\n');
            out.append("Will increase RPS from ").append(String.valueOf(opts.rampUpStartRps())).append(" to ")
                    .append(String.valueOf(opts.rampUpEndRps())).append(" RPS over the duration of the run\n");
        } else if (Double.isInfinite(opts.requestRate())) {
            out.append("Traffic request rate: inf\n");
        } else {
            out.append("Traffic request rate: ")
                    .append(String.format(Locale.ROOT, "%.2f", opts.requestRate())).append('\n');
        }
        String distribution = opts.burstiness() == 1.0 ? "Poisson process" : "Gamma distribution";
        out.append("Burstiness factor: ")
                .append(Double.isInfinite(opts.burstiness()) ? "inf" : String.valueOf(opts.burstiness()))
                .append(" (").append(distribution).append(")\n");
        out.append("Maximum request concurrency: ")
                .append(opts.maxConcurrency() != null ? String.valueOf(opts.maxConcurrency()) : "unlimited")
                .append('\n');
    }

    /** The ready check: the first request answers inside the timeout, or the run refuses to start. */
    private static void readyCheck(ServeBenchOptions opts, ServeBenchClient client, List<BenchSample> samples,
                                   Appendable out, Appendable err) throws IOException {
        if (opts.readyCheckTimeoutSec() <= 0) {
            out.append("Skipping endpoint ready check.\n");
            return;
        }
        err.append("Starting initial single prompt test run...\n");
        err.append("Waiting for endpoint to become up in ").append(String.valueOf(opts.readyCheckTimeoutSec()))
                .append(" seconds\n");
        long deadline = System.nanoTime() + opts.readyCheckTimeoutSec() * 1_000_000_000L;
        while (true) {
            BenchRequestOutput test = client.send(samples.get(0));
            if (test.success) {
                err.append("Initial test run completed.\n");
                return;
            }
            if (System.nanoTime() >= deadline) {
                throw new IOException("Initial test run failed - please make sure the benchmark arguments"
                        + " are correctly specified. Error: " + test.error);
            }
            try {
                Thread.sleep(1000L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("the ready check was interrupted", e);
            }
        }
    }

    /** The warmups: unmeasured requests over the same concurrency cap. */
    private static void warmups(ServeBenchOptions opts, ServeBenchClient client, List<BenchSample> samples,
                                Appendable err) throws IOException {
        if (opts.numWarmups() <= 0) {
            return;
        }
        err.append("Warming up with ").append(String.valueOf(opts.numWarmups())).append(" requests...\n");
        ArrivalSchedule allAtOnce = ArrivalSchedule.of(opts.numWarmups(), Double.POSITIVE_INFINITY, 1.0,
                null, null, null, new Random(opts.seed()));
        client.runAll(replicate(samples.get(0), opts.numWarmups()), allAtOnce, opts.maxConcurrency(), null);
        err.append("Warmup run completed.\n");
    }

    /** {@code n} copies of one sample, the ids numbered. */
    private static List<BenchSample> replicate(BenchSample sample, int n) {
        List<BenchSample> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(sample.withRequestId(sample.requestId() + "-warmup-" + i));
        }
        return out;
    }

    /**
     * The per-request model id of a LoRA run: the adapters round-robin or draw at random under the
     * seed; null when the run names no adapter.
     */
    private static String[] loraAssignment(ServeBenchOptions opts, int count) {
        List<String> loras = opts.loraModules();
        if (loras == null || loras.isEmpty()) {
            return null;
        }
        String[] out = new String[count];
        if ("round-robin".equals(opts.loraAssignment())) {
            for (int i = 0; i < count; i++) {
                out[i] = loras.get(i % loras.size());
            }
            return out;
        }
        Random random = new Random(opts.seed());
        for (int i = 0; i < count; i++) {
            out[i] = loras.get(random.nextInt(loras.size()));
        }
        return out;
    }

    /** The progress sink: a line on stderr every twentieth of the workload, unless disabled. */
    private static ServeBenchClient.Progress progressSink(ServeBenchOptions opts, int total, Appendable err) {
        if (opts.disableTqdm()) {
            return null;
        }
        int step = Math.max(1, total / 20);
        return new ServeBenchClient.Progress() {
            private int reported;

            @Override
            public boolean done() {
                return false;
            }

            @Override
            public void started(int count, int of) {
                if (count == 1 || count == of || count - reported >= step) {
                    reported = count;
                    try {
                        err.append("bench-serve: sent ").append(String.valueOf(count)).append('/')
                                .append(String.valueOf(of)).append('\n');
                    } catch (IOException e) {
                        throw new IllegalStateException("cannot write the benchmark progress", e);
                    }
                }
            }
        };
    }

    /** The first ten failures of a run, the reference's report. */
    private static void reportFailures(List<BenchRequestOutput> outputs, Appendable err) throws IOException {
        int shown = 0;
        for (BenchRequestOutput out : outputs) {
            if (out.success || shown >= 10) {
                continue;
            }
            if (shown == 0) {
                err.append("Failed requests during the benchmark run (capping the errors at 10):\n");
            }
            err.append("Error ").append(String.valueOf(shown)).append(": ").append(out.error).append('\n');
            shown++;
        }
    }

    /** The generation metrics, their report and the result keys. */
    private static Map<String, Object> generationResult(ServeBenchOptions opts, Appendable out, String modelId,
                                                        List<BenchSample> samples,
                                                        List<BenchRequestOutput> outputs, double durationSec,
                                                        List<String> selectedMetrics) throws IOException {
        ServeBenchMetrics metrics = ServeBenchMetrics.compute(outputs, durationSec, opts.percentiles(),
                opts.goodput(), opts.maxConcurrency(), opts.requestRate());
        metrics.print(out, durationSec, selectedMetrics, opts.requestRate(), opts.maxConcurrency(),
                !opts.goodput().isEmpty());
        Map<String, Object> result = frontMatter(opts, modelId);
        result.putAll(metrics.resultJson(durationSec, outputs, selectedMetrics, !opts.goodput().isEmpty(),
                opts.saveDetailed()));
        return result;
    }

    /** The rerank metrics, their report and the result keys. */
    private static Map<String, Object> poolingResult(ServeBenchOptions opts, Appendable out, String modelId,
                                                     List<BenchSample> samples,
                                                     List<BenchRequestOutput> outputs, double durationSec,
                                                     List<Double> selectedPercentiles) throws IOException {
        BenchPoolingMetrics metrics = BenchPoolingMetrics.compute(outputs, durationSec, selectedPercentiles);
        metrics.print(out, durationSec, opts.requestRate(), opts.maxConcurrency());
        Map<String, Object> result = frontMatter(opts, modelId);
        result.putAll(metrics.resultJson(durationSec, outputs, selectedPercentiles, opts.saveDetailed()));
        return result;
    }

    /** The probe requests' report and result keys. */
    private static void putProbeStats(Map<String, Object> result, List<BenchRequestOutput> probes,
                                      Appendable out) throws IOException {
        List<Double> latencies = new ArrayList<>();
        for (BenchRequestOutput probe : probes) {
            if (probe.success) {
                latencies.add(probe.latency);
            }
        }
        if (latencies.isEmpty()) {
            return;
        }
        double median = BenchStats.median(latencies) * 1000.0;
        double p99 = BenchStats.percentile(latencies, 99.0) * 1000.0;
        double max = 0.0;
        for (double latency : latencies) {
            max = Math.max(max, latency);
        }
        result.put("probe_completed", latencies.size());
        result.put("probe_failed", probes.size() - latencies.size());
        result.put("probe_median_e2el_ms", median);
        result.put("probe_p99_e2el_ms", p99);
        result.put("probe_max_e2el_ms", max * 1000.0);
        out.append(" Probe Requests ").append('\n');
        out.append(String.format(Locale.ROOT, "%-40s %-10s", "Probe requests completed:",
                String.valueOf(latencies.size()))).append('\n');
        out.append(String.format(Locale.ROOT, "%-40s %-10s", "Probe requests failed:",
                String.valueOf(probes.size() - latencies.size()))).append('\n');
        out.append(String.format(Locale.ROOT, "%-40s %-10.2f", "Median probe E2EL (ms):", median))
                .append('\n');
        out.append(String.format(Locale.ROOT, "%-40s %-10.2f", "P99 probe E2EL (ms):", p99)).append('\n');
        out.append(String.format(Locale.ROOT, "%-40s %-10.2f", "Max probe E2EL (ms):", max * 1000.0))
                .append('\n');
    }

    /** The result JSON of a finished run: the save, the plots and the printed file names. */
    private static void finish(ServeBenchOptions opts, Appendable out, String modelId,
                               Map<String, Object> result, List<BenchRequestOutput> outputs)
            throws IOException {
        boolean plots = opts.plotTimeline() || opts.plotDatasetStats();
        if (!opts.saveResult() && !opts.appendResult() && !plots) {
            return;
        }
        String when = String.valueOf(result.get("date"));
        Path file = resultPath(opts, modelId, when);
        if (opts.saveResult() || opts.appendResult()) {
            writeResult(file, result, opts.appendResult());
            out.append("Benchmark result saved to ").append(file.toString()).append('\n');
        }
        if (opts.plotTimeline()) {
            Path plot = withSuffix(file, ".timeline.html");
            BenchPlots.timeline(outputs, plot, opts.timelineItlThresholds());
            out.append("Timeline plot saved to: ").append(plot.toString()).append('\n');
        }
        if (opts.plotDatasetStats()) {
            Path plot = withSuffix(file, ".dataset_stats.svg");
            BenchPlots.datasetStats(outputs, plot);
            out.append("Dataset statistics plot saved to: ").append(plot.toString()).append('\n');
        }
    }

    /** {@code file} with its extension replaced by {@code suffix}. */
    static Path withSuffix(Path file, String suffix) {
        String name = file.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String stem = dot > 0 ? name.substring(0, dot) : name;
        return file.resolveSibling(stem + suffix);
    }

    /** The run's facts ahead of the metrics: the reference's front matter keys. */
    private static Map<String, Object> frontMatter(ServeBenchOptions opts, String modelId) {
        String backend = backendName(opts);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("date", LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")));
        result.put("endpoint_type", backend);
        result.put("backend", backend);
        result.put("label", opts.label() != null ? opts.label() : backend);
        result.put("model_id", modelId);
        result.put("tokenizer_id", opts.tokenizerPath() != null ? opts.tokenizerPath() : modelId);
        result.put("num_prompts", opts.numPrompts());
        result.putAll(opts.metadata());
        result.put("request_rate", Double.isInfinite(opts.requestRate()) ? "inf" : opts.requestRate());
        result.put("burstiness", opts.burstiness());
        result.put("max_concurrency", opts.maxConcurrency());
        if (opts.rampUpStrategy() != null) {
            result.put("ramp_up_strategy", opts.rampUpStrategy());
            result.put("ramp_up_start_rps", opts.rampUpStartRps());
            result.put("ramp_up_end_rps", opts.rampUpEndRps());
        }
        return result;
    }

    /** The reference's backend name of this run's endpoint. */
    static String backendName(ServeBenchOptions opts) {
        String endpoint = resolvedEndpoint(opts);
        if (ServeBenchClient.RERANK.equals(endpoint)) {
            return "vllm-rerank";
        }
        if (ServeBenchClient.AUDIO.equals(endpoint)) {
            return "openai-audio";
        }
        if (ServeBenchClient.CHAT.equals(endpoint)) {
            return "openai-chat";
        }
        return "openai";
    }

    /**
     * The result file: {@code {label}-{rate}qps{-concurrencyN}-{base model}-{date}.json} in
     * {@code --result-dir}, or exactly {@code --result-filename}.
     */
    static Path resultPath(ServeBenchOptions opts, String modelId, String when) {
        if (opts.resultFilename() != null) {
            return Paths.get(opts.resultDir() != null ? opts.resultDir() : ".", opts.resultFilename());
        }
        String baseModel = modelId;
        int slash = baseModel.lastIndexOf('/');
        if (slash >= 0) {
            baseModel = baseModel.substring(slash + 1);
        }
        String label = opts.label() != null ? opts.label() : backendName(opts);
        String rate = Double.isInfinite(opts.requestRate()) ? "inf"
                : String.format(Locale.ROOT, "%s", trimRate(opts.requestRate()));
        String concurrency = opts.maxConcurrency() != null ? "-concurrency" + opts.maxConcurrency() : "";
        String stem = opts.rampUpStrategy() != null
                ? label + "-ramp-up-" + opts.rampUpStrategy() + "-" + opts.rampUpStartRps() + "qps-"
                        + opts.rampUpEndRps() + "qps" + concurrency
                : label + "-" + rate + "qps" + concurrency;
        String name = stem + "-" + baseModel + "-" + when + ".json";
        return Paths.get(opts.resultDir() != null ? opts.resultDir() : ".", name);
    }

    /** A rate without a trailing {@code .0}, the reference's float formatting. */
    private static String trimRate(double rate) {
        if (rate == Math.rint(rate) && !Double.isInfinite(rate)) {
            return String.valueOf((long) rate);
        }
        return String.valueOf(rate);
    }

    /** The result JSON as text, Python's {@code json.dumps} layout. */
    static String jsonText(Object value) {
        StringBuilder out = new StringBuilder();
        try {
            Json.write(value, Json.Style.PYTHON_DEFAULT, out);
        } catch (IOException e) {
            throw new IllegalStateException("a benchmark result does not write as JSON", e);
        }
        return out.toString();
    }

    /** The result JSON on the file, appended after a newline when {@code append} and the file is not empty. */
    private static void writeResult(Path file, Map<String, Object> result, boolean append) throws IOException {
        Path parent = file.getParent();
        if (parent != null && !Files.exists(parent)) {
            Files.createDirectories(parent);
        }
        String json = jsonText(result);
        if (append && Files.exists(file) && Files.size(file) > 0) {
            byte[] existing = Files.readAllBytes(file);
            byte[] addition = ("\n" + json).getBytes(StandardCharsets.UTF_8);
            byte[] merged = new byte[existing.length + addition.length];
            System.arraycopy(existing, 0, merged, 0, existing.length);
            System.arraycopy(addition, 0, merged, existing.length, addition.length);
            Files.write(file, merged);
            return;
        }
        Files.write(file, json.getBytes(StandardCharsets.UTF_8));
    }

    /** The single-token probe requests at {@code --probe-request-rate}, outside the concurrency cap. */
    private static final class Probe {

        private final Thread thread;
        private final List<BenchRequestOutput> outputs = java.util.Collections.synchronizedList(
                new ArrayList<>());
        private volatile boolean stopped;

        private Probe(ServeBenchClient client, double rate, String prefix) {
            this.thread = new Thread(() -> {
                long intervalNanos = (long) (1_000_000_000.0 / rate);
                int index = 0;
                while (!stopped) {
                    BenchSample probe = BenchSample.ofText("Hi", 1, 1, prefix + "probe-" + index++);
                    outputs.add(client.send(probe));
                    long next = System.nanoTime() + intervalNanos;
                    while (!stopped && System.nanoTime() < next) {
                        try {
                            Thread.sleep(1L);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                    }
                }
            }, "bench-serve-probe");
            thread.setDaemon(true);
            thread.start();
        }

        /** The probes' outputs once the loop has stopped. */
        static Probe start(ServeBenchOptions opts, ServeBenchClient client) {
            if (!(opts.probeRequestRate() > 0.0)) {
                return null;
            }
            return new Probe(client, opts.probeRequestRate(), opts.requestIdPrefix());
        }

        List<BenchRequestOutput> stop() {
            stopped = true;
            try {
                thread.join(10_000L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            synchronized (outputs) {
                return new ArrayList<>(outputs);
            }
        }
    }
}
