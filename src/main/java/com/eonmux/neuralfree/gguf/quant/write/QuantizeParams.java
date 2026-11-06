package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.load.MetadataOverrides;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * The parameters of one quantize run, the port of {@code llama_model_quantize_params} (llama.cpp
 * src/llama-quant.cpp:1445-1465) plus the flag-side extras the quantize tool holds beside them: the
 * imatrix file and its include/exclude name filters ({@code --imatrix}, {@code --include-weights},
 * {@code --exclude-weights}), the per-tensor type overrides ({@code --tensor-type},
 * {@code --tensor-type-file}, compiled as regexes like the reference compiles them), the metadata
 * overrides ({@code --override-kv}) and the pruned layers ({@code --prune-layers}). Immutable; the
 * {@code with*} accessors derive modified copies.
 */
public final class QuantizeParams {

    /** The recipe. */
    public final QuantizeFtype ftype;
    /** Tensor type of {@code output.weight} ({@code --output-tensor-type}); null follows the mixture. */
    public final GGMLType outputTensorType;
    /** Tensor type of {@code token_embd.weight} ({@code --token-embedding-type}); null follows the mixture. */
    public final GGMLType tokenEmbeddingType;
    /** Allow quantizing tensors that are already quantized ({@code --allow-requantize}). */
    public final boolean allowRequantize;
    /** Quantize {@code output.weight} too; false is {@code --leave-output-tensor}. */
    public final boolean quantizeOutputTensor;
    /** Disable the mixture and quantize every tensor to the default type ({@code --pure}). */
    public final boolean pure;
    /** Keep the input's shard layout ({@code --keep-split}). */
    public final boolean keepSplit;
    /** Compute and print the sizes without writing ({@code --dry-run}). */
    public final boolean dryRun;
    /** The imatrix file ({@code --imatrix}); null runs without one. */
    public final Path imatrixFile;
    /** Tensor-name substrings that keep their imatrix weights ({@code --include-weights}). */
    public final List<String> includeWeights;
    /** Tensor-name substrings that lose their imatrix weights ({@code --exclude-weights}). */
    public final List<String> excludeWeights;
    /** Per-tensor type overrides: a regex pattern and its type, in flag order ({@code --tensor-type}). */
    public final List<TensorType> tensorTypes;
    /** The metadata overrides in flag order ({@code --override-kv}). */
    public final List<MetadataOverrides.Override> kvOverrides;
    /** The sorted, de-duplicated layer numbers to prune ({@code --prune-layers}). */
    public final List<Integer> pruneLayers;
    /** Worker threads for the row quantization; {@code 0} picks the default. */
    public final int nthread;

    /** One {@code --tensor-type} override: the tensor-name pattern and the type it selects. */
    public static final class TensorType {
        /** The regex pattern a tensor name must match ({@code std::regex_search}). */
        public final Pattern pattern;
        /** The pattern text as given. */
        public final String text;
        /** The tensor type the pattern selects. */
        public final GGMLType type;

        public TensorType(Pattern pattern, String text, GGMLType type) {
            this.pattern = Objects.requireNonNull(pattern, "pattern");
            this.text = Objects.requireNonNull(text, "text");
            this.type = Objects.requireNonNull(type, "type");
        }
    }

    private QuantizeParams(Builder b) {
        this.ftype = b.ftype;
        this.outputTensorType = b.outputTensorType;
        this.tokenEmbeddingType = b.tokenEmbeddingType;
        this.allowRequantize = b.allowRequantize;
        this.quantizeOutputTensor = b.quantizeOutputTensor;
        this.pure = b.pure;
        this.keepSplit = b.keepSplit;
        this.dryRun = b.dryRun;
        this.imatrixFile = b.imatrixFile;
        this.includeWeights = List.copyOf(b.includeWeights);
        this.excludeWeights = List.copyOf(b.excludeWeights);
        this.tensorTypes = List.copyOf(b.tensorTypes);
        this.kvOverrides = List.copyOf(b.kvOverrides);
        this.pruneLayers = List.copyOf(b.pruneLayers);
        this.nthread = b.nthread;
    }

    /** The defaults of {@code llama_model_quantize_default_params}: Q8_0, no imatrix, no overrides. */
    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.ftype = ftype;
        b.outputTensorType = outputTensorType;
        b.tokenEmbeddingType = tokenEmbeddingType;
        b.allowRequantize = allowRequantize;
        b.quantizeOutputTensor = quantizeOutputTensor;
        b.pure = pure;
        b.keepSplit = keepSplit;
        b.dryRun = dryRun;
        b.imatrixFile = imatrixFile;
        b.includeWeights = new ArrayList<>(includeWeights);
        b.excludeWeights = new ArrayList<>(excludeWeights);
        b.tensorTypes = new ArrayList<>(tensorTypes);
        b.kvOverrides = new ArrayList<>(kvOverrides);
        b.pruneLayers = new ArrayList<>(pruneLayers);
        b.nthread = nthread;
        return b;
    }

    /** The mutable builder of {@link QuantizeParams}. */
    public static final class Builder {
        private QuantizeFtype ftype = QuantizeFtype.Q8_0;
        private GGMLType outputTensorType;
        private GGMLType tokenEmbeddingType;
        private boolean allowRequantize;
        private boolean quantizeOutputTensor = true;
        private boolean pure;
        private boolean keepSplit;
        private boolean dryRun;
        private Path imatrixFile;
        private List<String> includeWeights = Collections.emptyList();
        private List<String> excludeWeights = Collections.emptyList();
        private List<TensorType> tensorTypes = Collections.emptyList();
        private List<MetadataOverrides.Override> kvOverrides = Collections.emptyList();
        private List<Integer> pruneLayers = Collections.emptyList();
        private int nthread;

        private Builder() {
        }

        public Builder ftype(QuantizeFtype v) {
            this.ftype = Objects.requireNonNull(v, "ftype");
            return this;
        }

        public Builder outputTensorType(GGMLType v) {
            this.outputTensorType = v;
            return this;
        }

        public Builder tokenEmbeddingType(GGMLType v) {
            this.tokenEmbeddingType = v;
            return this;
        }

        public Builder allowRequantize(boolean v) {
            this.allowRequantize = v;
            return this;
        }

        public Builder quantizeOutputTensor(boolean v) {
            this.quantizeOutputTensor = v;
            return this;
        }

        public Builder pure(boolean v) {
            this.pure = v;
            return this;
        }

        public Builder keepSplit(boolean v) {
            this.keepSplit = v;
            return this;
        }

        public Builder dryRun(boolean v) {
            this.dryRun = v;
            return this;
        }

        public Builder imatrixFile(Path v) {
            this.imatrixFile = v;
            return this;
        }

        public Builder includeWeights(List<String> v) {
            this.includeWeights = v == null ? Collections.emptyList() : v;
            return this;
        }

        public Builder excludeWeights(List<String> v) {
            this.excludeWeights = v == null ? Collections.emptyList() : v;
            return this;
        }

        public Builder tensorTypes(List<TensorType> v) {
            this.tensorTypes = v == null ? Collections.emptyList() : v;
            return this;
        }

        public Builder kvOverrides(List<MetadataOverrides.Override> v) {
            this.kvOverrides = v == null ? Collections.emptyList() : v;
            return this;
        }

        public Builder pruneLayers(List<Integer> v) {
            this.pruneLayers = v == null ? Collections.emptyList() : v;
            return this;
        }

        public Builder nthread(int v) {
            this.nthread = v;
            return this;
        }

        /**
         * Adds one {@code --tensor-type name=type} override; the name part compiles as a regex, like the
         * reference compiles its patterns.
         *
         * @throws IllegalArgumentException naming the entry when it has no {@code =}, no name, no type or
         *                                  a pattern that does not compile
         */
        public Builder tensorType(String entry) {
            int eq = entry.indexOf('=');
            if (eq <= 0) {
                throw new IllegalArgumentException("malformed tensor type '" + entry
                        + "': use <tensor name>=<ggml type>");
            }
            if (eq == entry.length() - 1) {
                throw new IllegalArgumentException("malformed tensor type '" + entry + "': no ggml type after '='");
            }
            String name = entry.substring(0, eq).toLowerCase(Locale.ROOT);
            GGMLType type = parseGgmlType(entry.substring(eq + 1));
            Pattern pattern;
            try {
                pattern = Pattern.compile(name);
            } catch (PatternSyntaxException e) {
                throw new IllegalArgumentException("tensor type pattern '" + name + "' does not compile: "
                        + e.getMessage(), e);
            }
            List<TensorType> l = new ArrayList<>(tensorTypes.size() + 1);
            l.addAll(tensorTypes);
            l.add(new TensorType(pattern, name, type));
            this.tensorTypes = l;
            return this;
        }

        /**
         * Adds one {@code key=type:value} metadata override ({@code --override-kv}), parsed like the
         * engine property parses them.
         *
         * @throws IllegalArgumentException naming the entry when it cannot be parsed
         */
        public Builder kvOverride(String entry) {
            List<MetadataOverrides.Override> l = new ArrayList<>(kvOverrides.size() + 1);
            l.addAll(kvOverrides);
            l.add(MetadataOverrides.parse(entry, "--override-kv"));
            this.kvOverrides = l;
            return this;
        }

        /**
         * Sets the {@code --prune-layers L0,L1,...} list: layer numbers sorted and de-duplicated.
         *
         * @throws IllegalArgumentException naming the list when an entry is not a non-negative integer
         */
        public Builder pruneLayers(String csv) {
            List<Integer> l = new ArrayList<>();
            for (String part : csv.split(",")) {
                String s = part.trim();
                int id;
                try {
                    id = Integer.parseInt(s);
                } catch (NumberFormatException e) {
                    id = -1;
                }
                if (id < 0) {
                    throw new IllegalArgumentException("invalid layer id '" + s + "' in --prune-layers");
                }
                l.add(id);
            }
            Collections.sort(l);
            for (int i = l.size() - 2; i >= 0; i--) {
                if (l.get(i).intValue() == l.get(i + 1).intValue()) {
                    l.remove(i + 1);
                }
            }
            this.pruneLayers = l;
            return this;
        }

        /** @throws IllegalArgumentException when both weight filters are set, as the reference refuses */
        public QuantizeParams build() {
            if (!includeWeights.isEmpty() && !excludeWeights.isEmpty()) {
                throw new IllegalArgumentException("--include-weights and --exclude-weights cannot be used "
                        + "together");
            }
            return new QuantizeParams(this);
        }
    }

    /**
     * A ggml type by its storage name, case-insensitive ({@code parse_ggml_type} over
     * {@code ggml_type_name}).
     *
     * @throws IllegalArgumentException naming the value when it is not a ggml type name
     */
    public static GGMLType parseGgmlType(String value) {
        String s = value.trim().toUpperCase(Locale.ROOT);
        try {
            return GGMLType.valueOf(s);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("invalid ggml type '" + value + "'; use a ggml type name"
                    + " such as q4_k, q8_0 or iq4_xs", e);
        }
    }
}
