package com.eonmux.neuralfree.demo.tools.serve;

import java.util.HashMap;
import java.util.Map;

/**
 * The default subset and split of the Hugging Face datasets vLLM's {@code hf} dataset knows
 * ({@code vllm/benchmarks/datasets.py}'s {@code SUPPORTED_DATASET_PATHS}): the values its
 * {@code get_samples} sets when the flags are absent. An unknown repository keeps the flags as
 * given, and the loader infers a single top-level subset from the repository's layout.
 */
final class HfRepoCatalog {

    /** A repository's default configuration selection. */
    static final class Entry {

        final String subset;
        final String split;

        Entry(String subset, String split) {
            this.subset = subset;
            this.split = split;
        }
    }

    private static final Map<String, Entry> ENTRIES = new HashMap<>();

    static {
        ENTRIES.put("Aeala/ShareGPT_Vicuna_unfiltered", new Entry(null, "train"));
        ENTRIES.put("likaixin/InstructCoder", new Entry(null, "train"));
        ENTRIES.put("philschmid/mt-bench", new Entry(null, "train"));
        ENTRIES.put("openai/openai_humaneval", new Entry(null, "test"));
        ENTRIES.put("openai/gsm8k", new Entry("main", "test"));
        ENTRIES.put("vdaita/edit_5k_char", new Entry(null, "train"));
        ENTRIES.put("vdaita/edit_10k_char", new Entry(null, "train"));
        ENTRIES.put("mgoin/mlperf-inference-llama2-data", new Entry(null, "train"));
        ENTRIES.put("mgoin/mlperf-inference-llama3.1-data", new Entry(null, "train"));
        ENTRIES.put("AI-MO/aimo-validation-aime", new Entry(null, "train"));
        ENTRIES.put("AI-MO/NuminaMath-1.5", new Entry(null, "train"));
        ENTRIES.put("AI-MO/NuminaMath-CoT", new Entry(null, "train"));
        ENTRIES.put("lmms-lab/LLaVA-OneVision-Data", new Entry(null, null));
    }

    private HfRepoCatalog() {
    }

    /** The repository's defaults; an unknown repository has none. */
    static Entry of(String repository) {
        return ENTRIES.get(repository);
    }
}
