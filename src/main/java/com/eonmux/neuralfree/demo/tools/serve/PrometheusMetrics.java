package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A server's Prometheus exposition, parsed into its samples, and the speculative-decoding
 * acceptance a benchmark run derives from two snapshots of it. vLLM's series
 * ({@code vllm:spec_decode_*}) are read as the reference reads them; NeuralFree's equivalent
 * counters ({@code neuralfree_spec_*}) stand in for a server that publishes its own names.
 */
final class PrometheusMetrics {

    /** One exposition line: its metric name, its labels and its value. */
    static final class Sample {

        final String name;
        final Map<String, String> labels;
        final double value;

        Sample(String name, Map<String, String> labels, double value) {
            this.name = name;
            this.labels = labels;
            this.value = value;
        }
    }

    /** The speculative-decoding counters two snapshots differ by. */
    static final class SpecDecode {

        final long drafts;
        final long draftTokens;
        final long acceptedTokens;
        final List<Double> perPositionRates;

        SpecDecode(long drafts, long draftTokens, long acceptedTokens, List<Double> perPositionRates) {
            this.drafts = drafts;
            this.draftTokens = draftTokens;
            this.acceptedTokens = acceptedTokens;
            this.perPositionRates = perPositionRates;
        }

        /** The share of drafted tokens the target accepted, in percent. */
        double acceptanceRate() {
            return (double) acceptedTokens / draftTokens * 100.0;
        }

        /** The mean number of accepted tokens per verify round, the first token included. */
        double acceptanceLength() {
            return drafts > 0 ? 1.0 + (double) acceptedTokens / drafts : 0.0;
        }

        /** The result keys vLLM writes for a run with speculative decoding. */
        void putInto(Map<String, Object> result) {
            result.put("spec_decode_acceptance_rate", acceptanceRate());
            result.put("spec_decode_acceptance_length", acceptanceLength());
            result.put("spec_decode_num_drafts", drafts);
            result.put("spec_decode_draft_tokens", draftTokens);
            result.put("spec_decode_accepted_tokens", acceptedTokens);
            result.put("spec_decode_per_position_acceptance_rates", new ArrayList<>(perPositionRates));
        }
    }

    private final List<Sample> samples;

    private PrometheusMetrics(List<Sample> samples) {
        this.samples = samples;
    }

    /** The samples of one exposition; malformed lines are skipped as Prometheus clients skip them. */
    static PrometheusMetrics parse(String text) {        List<Sample> samples = new ArrayList<>();
        for (String raw : text.split("\n")) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            int brace = line.indexOf('{');
            int space;
            String name;
            Map<String, String> labels = new LinkedHashMap<>();
            if (brace > 0) {
                space = line.lastIndexOf(' ');
                if (space <= brace) {
                    continue;
                }
                name = line.substring(0, brace).trim();
                String labelText = line.substring(brace + 1, line.lastIndexOf('}', space));
                labels.putAll(parseLabels(labelText));
            } else {
                space = line.indexOf(' ');
                if (space <= 0) {
                    continue;
                }
                name = line.substring(0, space).trim();
            }
            String valueText = line.substring(space + 1).trim();
            int valueEnd = 0;
            while (valueEnd < valueText.length() && !Character.isWhitespace(valueText.charAt(valueEnd))) {
                valueEnd++;
            }
            try {
                samples.add(new Sample(name, labels, Double.parseDouble(valueText.substring(0, valueEnd))));
            } catch (NumberFormatException ignored) {
                // an unparsable sample carries no measurement
            }
        }
        return new PrometheusMetrics(samples);
    }

    /** The server's exposition, or null when it does not serve one. */
    static PrometheusMetrics fetch(String baseUrl, Map<String, String> headers, boolean insecure) {
        return BenchEndpoints.metricsText(baseUrl, headers, insecure)
                .map(PrometheusMetrics::parse)
                .orElse(null);
    }

    /** A label list {@code k="v",k2="v2"}, quotes and backslashes unescaped. */
    private static Map<String, String> parseLabels(String text) {
        Map<String, String> labels = new LinkedHashMap<>();
        int i = 0;
        while (i < text.length()) {
            int eq = text.indexOf('=', i);
            if (eq < 0) {
                break;
            }
            String key = text.substring(i, eq).trim();
            int quote = text.indexOf('"', eq);
            if (quote < 0) {
                break;
            }
            StringBuilder value = new StringBuilder();
            i = quote + 1;
            while (i < text.length()) {
                char c = text.charAt(i);
                if (c == '\\' && i + 1 < text.length()) {
                    value.append(text.charAt(i + 1));
                    i += 2;
                } else if (c == '"') {
                    i++;
                    break;
                } else {
                    value.append(c);
                    i++;
                }
            }
            labels.put(key, value.toString());
            while (i < text.length() && (text.charAt(i) == ',' || Character.isWhitespace(text.charAt(i)))) {
                i++;
            }
        }
        return labels;
    }

    /** The sum of every sample of {@code name}, zero when absent. */
    double sum(String name) {
        double total = 0.0;
        for (Sample sample : samples) {
            if (sample.name.equals(name)) {
                total += sample.value;
            }
        }
        return total;
    }

    /** Whether the exposition carries any sample of {@code name}. */
    boolean has(String name) {
        for (Sample sample : samples) {
            if (sample.name.equals(name)) {
                return true;
            }
        }
        return false;
    }

    /** The per-position accepted-token counters of {@code name}, keyed by their {@code position} label. */
    Map<Integer, Double> byPosition(String name) {
        Map<Integer, Double> out = new java.util.TreeMap<>();
        for (Sample sample : samples) {
            if (!sample.name.equals(name)) {
                continue;
            }
            String position = sample.labels.get("position");
            if (position == null) {
                continue;
            }
            try {
                out.merge(Integer.parseInt(position.trim()), sample.value, Double::sum);
            } catch (NumberFormatException ignored) {
                // not a positional counter
            }
        }
        return out;
    }

    /**
     * The speculative-decoding counters {@code before} and {@code after} differ by, or empty when
     * the server publishes none or the run drafted nothing.
     */
    static Optional<SpecDecode> specDecode(PrometheusMetrics before, PrometheusMetrics after) {
        if (before == null || after == null) {
            return Optional.empty();
        }
        String draftsName = null;
        String draftTokensName = null;
        String acceptedName = null;
        String perPositionName = null;
        if (after.has("vllm:spec_decode_num_drafts_total")) {
            draftsName = "vllm:spec_decode_num_drafts_total";
            draftTokensName = "vllm:spec_decode_num_draft_tokens_total";
            acceptedName = "vllm:spec_decode_num_accepted_tokens_total";
            perPositionName = "vllm:spec_decode_num_accepted_tokens_per_pos_total";
        } else if (after.has("neuralfree_spec_rounds_total")) {
            draftsName = "neuralfree_spec_rounds_total";
            draftTokensName = "neuralfree_spec_drafted_tokens_total";
            acceptedName = "neuralfree_spec_accepted_tokens_total";
        } else {
            return Optional.empty();
        }
        long drafts = delta(before, after, draftsName);
        long draftTokens = delta(before, after, draftTokensName);
        long accepted = delta(before, after, acceptedName);
        if (draftTokens <= 0) {
            return Optional.empty();
        }
        List<Double> perPosition = new ArrayList<>();
        if (perPositionName != null && drafts > 0) {
            Map<Integer, Double> beforePos = before.byPosition(perPositionName);
            Map<Integer, Double> afterPos = after.byPosition(perPositionName);
            java.util.TreeSet<Integer> positions = new java.util.TreeSet<>(afterPos.keySet());
            positions.addAll(beforePos.keySet());
            for (int position : positions) {
                double deltaPos = afterPos.getOrDefault(position, 0.0) - beforePos.getOrDefault(position, 0.0);
                perPosition.add(deltaPos / drafts);
            }
        }
        return Optional.of(new SpecDecode(drafts, draftTokens, accepted, perPosition));
    }

    private static long delta(PrometheusMetrics before, PrometheusMetrics after, String name) {
        return Math.round(after.sum(name) - before.sum(name));
    }
}
