package com.eonmux.neuralfree.gguf.gpu.weights;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Per-tensor placement overrides: ordered rules that keep matched tensors in RAM on the host instead of placing
 * them on the device, so their products run on the host while the rest of the model runs on the GPU. They are the
 * llama.cpp {@code tensor_buft_overrides} (common/arg.cpp:2705-2747, applied per tensor name at
 * src/llama-model-loader.cpp:1248-1250): {@code -ot / --override-tensor} patterns, {@code --cpu-moe} and
 * {@code --n-cpu-moe} for the MoE expert tensors, {@code --n-cpu-ffn} for dense FFN weights.
 *
 * <p>A rule is a regular expression matched with {@link java.util.regex.Matcher#find} against the stored GGUF
 * tensor name (llama.cpp matches with {@code regex_search}, common/arg.cpp:228-248), so the reference patterns
 * match the llama.cpp names and NeuralFree's {@code .weight}-suffixed names alike, and a llama.cpp pattern string
 * works verbatim. A rule carries a target: {@code CPU} keeps the tensor on the host, {@code GPU} pins it to the
 * device. The first matching rule wins; a tensor no rule matches is placed as usual ({@link WeightPlacement}).
 * Rules built by the factories below keep the reference patterns of llama.cpp (common/common.h:1257-1259): the
 * expert pattern matches the routed {@code ffn_(up|down|gate|gate_up)_exps} and the chunk {@code ffn_*_chexps}
 * tensors, the dense pattern the plain {@code ffn_(up|down|gate)} weights. Immutable.</p>
 */
public final class TensorPlacementOverrides {

    /** llama.cpp {@code LLM_FFN_EXPS_REGEX}: the expert tensors of the routed and chunk expert sets. */
    public static final String EXPERT_REGEX = "\\.(ffn_(up|down|gate|gate_up)_(ch|)exps)";

    /** llama.cpp {@code LLM_FFN_DENSE_REGEX}: the dense FFN weights of a decoder layer. */
    public static final String DENSE_FFN_REGEX = "\\.(ffn_(up|down|gate)\\.)";

    /** Target of a rule: host RAM or the device. */
    public enum Target {
        /** Keep the tensor in RAM; its products run on the host. */
        HOST,
        /** Place the tensor on the device; shields it from later rules. */
        DEVICE;

        /** The llama.cpp buffer-type name of the target. */
        public String bufferTypeName() {
            return this == HOST ? "CPU" : "GPU";
        }

        /**
         * The target of a llama.cpp buffer-type name.
         *
         * @throws IllegalArgumentException for any name but CPU or GPU, ignoring case
         */
        public static Target parse(String name) {
            String s = Objects.requireNonNull(name, "buffer type").trim().toUpperCase(Locale.ROOT);
            if (s.equals("CPU")) {
                return HOST;
            }
            if (s.equals("GPU")) {
                return DEVICE;
            }
            throw new IllegalArgumentException("buffer type " + name + " is invalid; use CPU or GPU");
        }
    }

    /** One rule: a full-name pattern and the target its matches take. Immutable. */
    public static final class Rule {
        private final String pattern;
        private final Pattern compiled;
        private final Target target;

        private Rule(String pattern, Pattern compiled, Target target) {
            this.pattern = pattern;
            this.compiled = compiled;
            this.target = target;
        }

        /** The pattern source. */
        public String pattern() {
            return pattern;
        }

        /** Where a matching tensor goes. */
        public Target target() {
            return target;
        }

        /** True when the stored tensor name matches the pattern. */
        public boolean matches(String storedName) {
            return compiled.matcher(Objects.requireNonNull(storedName, "storedName")).find();
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Rule)) {
                return false;
            }
            Rule other = (Rule) o;
            return pattern.equals(other.pattern) && target == other.target;
        }

        @Override
        public int hashCode() {
            return 31 * pattern.hashCode() + target.hashCode();
        }

        @Override
        public String toString() {
            return pattern + "=" + target.bufferTypeName();
        }
    }

    private static final TensorPlacementOverrides NONE = new TensorPlacementOverrides(Collections.emptyList());

    private final List<Rule> rules;

    private TensorPlacementOverrides(List<Rule> rules) {
        this.rules = rules;
    }

    /** No overrides: every tensor is placed as usual. */
    public static TensorPlacementOverrides none() {
        return NONE;
    }

    /**
     * The overrides of one {@code -ot / --override-tensor} value: comma-separated {@code pattern=CPU|GPU} rules
     * (llama.cpp {@code parse_tensor_buffer_overrides}, common/arg.cpp:228-248).
     *
     * @throws IllegalArgumentException for an empty entry, a missing {@code =}, a buffer type that is not CPU or
     *                                  GPU, or a pattern that does not compile; the message names the entry
     */
    public static TensorPlacementOverrides parse(String spec) {
        Objects.requireNonNull(spec, "spec");
        Builder b = builder();
        for (String entry : spec.split(",")) {
            String e = entry.trim();
            if (e.isEmpty()) {
                continue;
            }
            int eq = e.indexOf('=');
            if (eq <= 0 || eq == e.length() - 1) {
                throw new IllegalArgumentException("override-tensor entry \"" + e + "\" needs <pattern>=<CPU|GPU>");
            }
            b.add(e.substring(0, eq), Target.parse(e.substring(eq + 1)));
        }
        return b.build();
    }

    /** {@code --cpu-moe}: every MoE expert tensor on the host (llama.cpp {@code llm_ffn_exps_cpu_override}). */
    public static TensorPlacementOverrides cpuMoe() {
        return builder().add(EXPERT_REGEX, Target.HOST).build();
    }

    /**
     * {@code --n-cpu-moe N}: the MoE expert tensors of the first N decoder layers on the host
     * (llama.cpp {@code llm_add_n_cpu_ffn_overrides} with {@code LLM_FFN_EXPS_REGEX}).
     *
     * @throws IllegalArgumentException when N is not positive
     */
    public static TensorPlacementOverrides nCpuMoe(int n) {
        return perLayer(n, EXPERT_REGEX);
    }

    /**
     * {@code --n-cpu-ffn N}: the dense FFN weights of the first N decoder layers on the host (llama.cpp
     * {@code llm_add_n_cpu_ffn_overrides} with {@code LLM_FFN_DENSE_REGEX}).
     *
     * @throws IllegalArgumentException when N is not positive
     */
    public static TensorPlacementOverrides nCpuFfn(int n) {
        return perLayer(n, DENSE_FFN_REGEX);
    }

    private static TensorPlacementOverrides perLayer(int n, String regex) {
        if (n <= 0) {
            throw new IllegalArgumentException("layer count must be positive, got " + n);
        }
        Builder b = builder();
        for (int i = 0; i < n; i++) {
            b.add("blk\\." + i + regex, Target.HOST);
        }
        return b.build();
    }

    /** A builder of ordered rules. */
    public static Builder builder() {
        return new Builder();
    }

    /** True when no rule is set. */
    public boolean isEmpty() {
        return rules.isEmpty();
    }

    /** The rules in application order (unmodifiable). */
    public List<Rule> rules() {
        return rules;
    }

    /** True when a rule keeps {@code storedName} in RAM (the first matching rule targets the host). */
    public boolean host(String storedName) {
        for (Rule r : rules) {
            if (r.matches(storedName)) {
                return r.target == Target.HOST;
            }
        }
        return false;
    }

    /** These rules followed by {@code more}'s rules; the earlier rule wins where both match. */
    public TensorPlacementOverrides with(TensorPlacementOverrides more) {
        Objects.requireNonNull(more, "more");
        if (more.isEmpty()) {
            return this;
        }
        if (isEmpty()) {
            return more;
        }
        List<Rule> merged = new ArrayList<>(rules.size() + more.rules.size());
        merged.addAll(rules);
        merged.addAll(more.rules);
        return new TensorPlacementOverrides(Collections.unmodifiableList(merged));
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof TensorPlacementOverrides)) {
            return false;
        }
        return rules.equals(((TensorPlacementOverrides) o).rules);
    }

    @Override
    public int hashCode() {
        return rules.hashCode();
    }

    @Override
    public String toString() {
        return "TensorPlacementOverrides" + rules;
    }

    /** Builder of ordered {@link Rule}s. */
    public static final class Builder {
        private final List<Rule> rules = new ArrayList<>();

        private Builder() {
        }

        /**
         * Appends one rule.
         *
         * @throws IllegalArgumentException when the pattern does not compile; the message names it
         */
        public Builder add(String pattern, Target target) {
            Objects.requireNonNull(pattern, "pattern");
            Objects.requireNonNull(target, "target");
            Pattern compiled;
            try {
                compiled = Pattern.compile(pattern);
            } catch (PatternSyntaxException e) {
                throw new IllegalArgumentException("override-tensor pattern \"" + pattern + "\" does not compile: "
                        + e.getMessage(), e);
            }
            rules.add(new Rule(pattern, compiled, target));
            return this;
        }

        public TensorPlacementOverrides build() {
            return new TensorPlacementOverrides(Collections.unmodifiableList(new ArrayList<>(rules)));
        }
    }
}
