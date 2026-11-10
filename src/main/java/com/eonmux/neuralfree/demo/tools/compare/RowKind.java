package com.eonmux.neuralfree.demo.tools.compare;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * The result rows {@code nftool bench-compare} reads, one kind per producing tool: the fields that identify a row by
 * default, the configuration fields the rows of one key must agree on, the metric compared by default, and which
 * metrics improve upward. Each kind is recognised by the fields its tool always writes.
 *
 * <p>A bench row's default key is llama.cpp's {@code LLAMA_BENCH_KEY_PROPERTIES} ({@code compare-llama-bench.py})
 * without its machine fields {@code cpu_info} and {@code gpu_info}, so results of two machines can still be paired,
 * plus the settings {@code nftool bench} writes beyond them: {@code devices}, {@code load_mode} and
 * {@code tg_readout}. Its configuration fields add the llama-bench settings its compare script does not key on.</p>
 */
public enum RowKind {

    /**
     * A test of {@code nftool bench} or llama.cpp's {@code llama-bench} (json, jsonl): one object per test with the
     * run's raw {@code samples_ts} and {@code samples_ns}.
     */
    BENCH("bench", "avg_ts",
            Arrays.asList("model_filename", "model_type", "backends", "n_gpu_layers", "n_cpu_moe",
                    "tensor_buft_overrides", "n_batch", "n_ubatch", "embeddings", "cpu_mask", "cpu_strict", "poll",
                    "n_threads", "type_k", "type_v", "use_mmap", "no_kv_offload", "split_mode", "main_gpu",
                    "tensor_split", "flash_attn", "devices", "load_mode", "tg_readout", "n_prompt", "n_gen",
                    "n_depth"),
            Arrays.asList("lazy_mode", "no_op_offload", "no_host", "repack", "fit_target", "fit_min_ctx")),
    /** A cell of {@code nftool batched-bench} or llama.cpp's {@code llama-batched-bench} (jsonl). */
    BATCHED("batched-bench", "speed",
            Arrays.asList("n_ubatch", "is_pp_shared", "is_tg_separate", "pp", "tg", "pl"),
            Arrays.asList("n_kv_max", "n_batch", "flash_attn", "n_gpu_layers", "n_threads", "n_threads_batch")),
    /**
     * A result of {@code nftool bench-serve} or vLLM's {@code vllm bench serve} (the saved result JSON, or a
     * {@code bench-sweep} summary holding one per run).
     */
    SERVE("bench-serve", "output_throughput",
            Arrays.asList("model_id", "endpoint_type", "num_prompts", "request_rate", "max_concurrency",
                    "burstiness"),
            Collections.emptyList());

    /** The readout a bench row without {@code tg_readout} measured: llama-bench's, and nftool's default. */
    static final String DEFAULT_TG_READOUT = "logits";

    private final String label;
    private final String defaultMetric;
    private final List<String> defaultKey;
    private final List<String> configFields;

    RowKind(String label, String defaultMetric, List<String> defaultKey, List<String> extraConfig) {
        this.label = label;
        this.defaultMetric = defaultMetric;
        this.defaultKey = Collections.unmodifiableList(defaultKey);
        List<String> config = new ArrayList<>(defaultKey);
        config.addAll(extraConfig);
        this.configFields = Collections.unmodifiableList(config);
    }

    /** The producing tool's name. */
    public String label() {
        return label;
    }

    /** The metric compared when {@code --metric} is absent. */
    public String defaultMetric() {
        return defaultMetric;
    }

    /** The fields identifying a row when {@code --key} is absent. */
    public List<String> defaultKey() {
        return defaultKey;
    }

    /**
     * The configuration fields of a row: the rows that pool under one key must agree on each one the key leaves
     * out, or they would mix configurations.
     */
    public List<String> configFields() {
        return configFields;
    }

    /**
     * The key value of {@code field} for a row that does not carry it: {@value #DEFAULT_TG_READOUT} for a bench row's
     * {@code tg_readout} (llama-bench reads the logits row of every generated token), else empty.
     */
    String absentKeyValue(String field) {
        return this == BENCH && field.equals("tg_readout") ? DEFAULT_TG_READOUT : "";
    }

    /**
     * Whether a larger {@code metric} is better: throughputs and speeds are, times and latencies are not.
     */
    public boolean higherIsBetter(String metric) {
        switch (this) {
            case BENCH:
                return !metric.endsWith("_ns");
            case BATCHED:
                return metric.startsWith("speed");
            default:
                return !metric.endsWith("_ms") && !metric.equals("duration");
        }
    }

    /** The kind of a result object by the fields its tool always writes, or null when it is none of them. */
    static RowKind of(Map<String, Object> row) {
        if (row.containsKey("n_prompt") && row.containsKey("n_gen") && row.containsKey("avg_ts")) {
            return BENCH;
        }
        if (row.containsKey("pp") && row.containsKey("tg") && row.containsKey("pl") && row.containsKey("speed")) {
            return BATCHED;
        }
        if (row.containsKey("completed") && (row.containsKey("output_throughput")
                || row.containsKey("request_throughput"))) {
            return SERVE;
        }
        return null;
    }
}
