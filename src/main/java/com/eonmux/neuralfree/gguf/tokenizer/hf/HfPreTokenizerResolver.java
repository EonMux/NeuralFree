package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerRegistry;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerRegistry.PreTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerType;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreTokenizerCanonicalizer.CanonicalPreTokenizer;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/**
 * Resolves {@code tokenizer.ggml.pre} from Hugging Face tokenizer files without Python: a
 * structural match of the canonical {@code pre_tokenizer} against NeuralFree's llama.cpp pre-tokenizers, checked
 * against llama.cpp's {@code chkhsh} table ({@link ChkhshTable}, the port of {@code get_vocab_base_pre},
 * LC/conversion/base.py:1803-2151).
 *
 * <p>Selection:</p>
 * <ol>
 *   <li>The structural candidates are the registry names whose {@link PreTokenizerType} splits as the canonical
 *       pre-tokenizer: its {@link CanonicalPreTokenizer#regexSequence()} equals the type's
 *       {@link PreTokenizerType#hfRegexes()} or {@link PreTokenizerType#regexes()} (both in
 *       {@link PreTokenizerCanonicalizer#canonicalRegex canonical text}) and the byte encoding agrees. They are
 *       ordered by type, then by name.</li>
 *   <li>The fingerprint is {@code sha256("[" + join(", ", ids) + "]")} of the ids the encoder gives for
 *       {@link ChkhshTable#checkText()}, special tokens included.</li>
 *   <li>A fingerprint name is chosen when it is a candidate (the last such row, as the converter's sequence of
 *       {@code if} statements selects) or when there are no candidates (the converter's own choice, which never
 *       reads the {@code pre_tokenizer}: a known fingerprint also decides when NeuralFree cannot canonicalize
 *       it).</li>
 *   <li>Otherwise the candidates are narrowed to those whose llama.cpp flags agree with the flags tokenizer.json
 *       states ({@code ignore_merges}, whitespace escaping, and {@code add_sep} when a {@code RobertaProcessing}
 *       post-processor sets it), when that leaves any; a single candidate is chosen, and of several the first is
 *       chosen and the ambiguity logged.</li>
 *   <li>With neither a candidate nor a known fingerprint the result is null.</li>
 * </ol>
 *
 * <p>Immutable and thread-safe.</p>
 */
public final class HfPreTokenizerResolver implements PreNameResolver {

    private static final Logger LOGGER = LogManager.getLogger(HfPreTokenizerResolver.class);

    /** How a {@link Resolution} was decided. */
    public enum Basis {
        /** The fingerprint names one of the structural candidates. */
        HASH_CONFIRMED,
        /**
         * No structural candidate (no tokenizer.json, a pre-tokenizer without a llama.cpp regex form, or one NeuralFree
         * cannot canonicalize); the fingerprint names the pre-tokenizer.
         */
        HASH_ONLY,
        /** One candidate remained; the fingerprint is unknown or names another pre-tokenizer. */
        STRUCTURAL,
        /** Several candidates remained; the fingerprint does not decide, and the first was taken. */
        STRUCTURAL_AMBIGUOUS,
        /** Nothing matched. */
        NONE
    }

    /** The outcome of one resolution, with what it was decided from. Immutable. */
    public static final class Resolution {
        private final String name;
        private final Basis basis;
        private final String chkhsh;
        private final List<String> hashNames;
        private final List<String> candidates;
        private final List<String> flagMatches;
        private final String canonicalizationError;

        Resolution(String name, Basis basis, String chkhsh, List<String> hashNames, List<String> candidates,
                   List<String> flagMatches, String canonicalizationError) {
            this.name = name;
            this.basis = basis;
            this.chkhsh = chkhsh;
            this.hashNames = hashNames;
            this.candidates = candidates;
            this.flagMatches = flagMatches;
            this.canonicalizationError = canonicalizationError;
        }

        /** The {@code tokenizer.ggml.pre} name, or null. */
        public String name() { return name; }

        public Basis basis() { return basis; }

        /** The fingerprint of the encoded {@code chktxt}. */
        public String chkhsh() { return chkhsh; }

        /** The names llama.cpp's table holds for {@link #chkhsh()}, in row order (empty when unknown). */
        public List<String> hashNames() { return hashNames; }

        /** The structural candidates in selection order (empty when none). */
        public List<String> candidates() { return candidates; }

        /**
         * The candidates whose llama.cpp flags agree with tokenizer.json, in selection order; empty when the flags
         * were not consulted or no candidate agrees.
         */
        public List<String> flagMatches() { return flagMatches; }

        /**
         * Why tokenizer.json's pre-tokenizer could not be canonicalized when the fingerprint decided instead
         * ({@link Basis#HASH_ONLY}); null otherwise.
         */
        public String canonicalizationError() { return canonicalizationError; }

        @Override
        public String toString() {
            return "Resolution{" + name + ", " + basis + ", chkhsh=" + chkhsh + ", hashNames=" + hashNames
                    + ", candidates=" + candidates + ", flagMatches=" + flagMatches
                    + (canonicalizationError != null ? ", canonicalizationError=" + canonicalizationError : "") + "}";
        }
    }

    private final ChkhshTable table;

    /** A resolver over the bundled llama.cpp table ({@link ChkhshTable#builtIn()}). */
    public HfPreTokenizerResolver() {
        this(ChkhshTable.builtIn());
    }

    /** A resolver over {@code table}. */
    public HfPreTokenizerResolver(ChkhshTable table) {
        this.table = Objects.requireNonNull(table, "table");
    }

    /**
     * {@inheritDoc}
     *
     * <p>Logs how the name was decided ({@link #explain}).</p>
     *
     * @throws GgufFormatException   as {@link #explain} does
     * @throws IllegalStateException when the encoder returns null
     */
    @Override
    public String resolve(HfTokenizerJson tj, Function<String, int[]> encodeWithSpecials) {
        Resolution r = explain(tj, encodeWithSpecials);
        log(tj, r);
        return r.name();
    }

    /**
     * Resolves as {@link #resolve} does and reports how, without logging.
     *
     * @throws GgufFormatException when tokenizer.json's pre-tokenizer, normalizer or post-processor is malformed or
     *         unsupported and the fingerprint is not in llama.cpp's table
     * @throws IllegalStateException when the encoder returns null
     */
    public Resolution explain(HfTokenizerJson tj, Function<String, int[]> encodeWithSpecials) {
        Objects.requireNonNull(encodeWithSpecials, "encodeWithSpecials");
        int[] ids = encodeWithSpecials.apply(table.checkText());
        if (ids == null) throw new IllegalStateException("the pre-tokenizer fingerprint encoder returned null");
        String chkhsh = ChkhshTable.hash(ids);
        List<String> hashNames = table.names(chkhsh);

        CanonicalPreTokenizer canonical = null;
        if (tj != null) {
            try {
                canonical = PreTokenizerCanonicalizer.canonicalize(tj);
            } catch (GgufFormatException e) {
                if (hashNames.isEmpty()) throw e;
                return new Resolution(table.name(chkhsh), Basis.HASH_ONLY, chkhsh, hashNames,
                        Collections.emptyList(), Collections.emptyList(), e.getMessage());
            }
        }
        List<String> candidates = canonical == null ? Collections.emptyList() : structuralCandidates(canonical);

        if (!hashNames.isEmpty()) {
            if (candidates.isEmpty()) {
                return new Resolution(table.name(chkhsh), Basis.HASH_ONLY, chkhsh, hashNames, candidates,
                        Collections.emptyList(), null);
            }
            for (int i = hashNames.size() - 1; i >= 0; i--) {
                if (candidates.contains(hashNames.get(i))) {
                    return new Resolution(hashNames.get(i), Basis.HASH_CONFIRMED, chkhsh, hashNames, candidates,
                            Collections.emptyList(), null);
                }
            }
        }
        if (candidates.isEmpty()) {
            return new Resolution(null, Basis.NONE, chkhsh, hashNames, candidates, Collections.emptyList(), null);
        }
        List<String> flagMatches = flagMatches(candidates, canonical);
        List<String> pool = flagMatches.isEmpty() ? candidates : flagMatches;
        Basis basis = pool.size() == 1 ? Basis.STRUCTURAL : Basis.STRUCTURAL_AMBIGUOUS;
        return new Resolution(pool.get(0), basis, chkhsh, hashNames, candidates, flagMatches, null);
    }

    /** The registry names whose type splits as {@code canonical}, ordered by type, then by name. */
    static List<String> structuralCandidates(CanonicalPreTokenizer canonical) {
        List<String> sequence = canonical.regexSequence();
        if (sequence == null) return Collections.emptyList();
        List<String> out = new ArrayList<>();
        for (PreTokenizerType type : PreTokenizerType.values()) {
            if (type.byteEncode() != canonical.byteEncode()) continue;
            if (!Registry.SEQUENCES.get(type).contains(sequence)) continue;
            out.addAll(Registry.NAMES.getOrDefault(type, Collections.emptyList()));
        }
        return Collections.unmodifiableList(out);
    }

    /** The candidates whose llama.cpp vocabulary flags agree with the flags tokenizer.json states, in order. */
    private static List<String> flagMatches(List<String> candidates, CanonicalPreTokenizer canonical) {
        List<String> out = new ArrayList<>();
        for (String name : candidates) {
            PreTokenizer p = PreTokenizerRegistry.forName(name);
            if (p.ignoreMerges() != canonical.ignoreMerges()) continue;
            if (p.escapeWhitespaces() != canonical.escapeWhitespaces()) continue;
            if (canonical.addSep() != null && p.addSep() != canonical.addSep()) continue;
            out.add(name);
        }
        return Collections.unmodifiableList(out);
    }

    private static void log(HfTokenizerJson tj, Resolution r) {
        String source = tj == null ? "the tokenizer" : tj.source();
        switch (r.basis()) {
            case HASH_CONFIRMED:
                LOGGER.debug("pre-tokenizer of {}: '{}', confirmed by chkhsh {}", source, r.name(), r.chkhsh());
                break;
            case HASH_ONLY:
                if (r.canonicalizationError() != null) {
                    LOGGER.warn("pre-tokenizer of {}: '{}' by chkhsh {}; its pre_tokenizer cannot be compared"
                            + " structurally: {}", source, r.name(), r.chkhsh(), r.canonicalizationError());
                } else if (tj == null) {
                    LOGGER.info("pre-tokenizer of {}: '{}' by chkhsh {} (no tokenizer.json)", source, r.name(),
                            r.chkhsh());
                } else {
                    LOGGER.info("pre-tokenizer of {}: '{}' by chkhsh {} (no llama.cpp pre-tokenizer splits like its"
                            + " pre_tokenizer)", source, r.name(), r.chkhsh());
                }
                break;
            case STRUCTURAL:
                if (r.hashNames().isEmpty()) {
                    LOGGER.warn("pre-tokenizer of {}: chkhsh {} is not in llama.cpp's table; using the structural"
                            + " match '{}'", source, r.chkhsh(), r.name());
                } else {
                    LOGGER.warn("pre-tokenizer of {}: chkhsh {} names {}, but the pre_tokenizer matches only"
                            + " '{}', which is used", source, r.chkhsh(), r.hashNames(), r.name());
                }
                break;
            case STRUCTURAL_AMBIGUOUS:
                LOGGER.warn("pre-tokenizer of {}: {} llama.cpp names split alike {} and chkhsh {} ({}) does not"
                        + " decide; using '{}', the first in PreTokenizerType order", source,
                        r.flagMatches().isEmpty() ? r.candidates().size() : r.flagMatches().size(),
                        r.flagMatches().isEmpty() ? r.candidates() : r.flagMatches(), r.chkhsh(),
                        r.hashNames().isEmpty() ? "not in llama.cpp's table" : "names " + r.hashNames(), r.name());
                break;
            case NONE:
                LOGGER.warn("pre-tokenizer of {} was not recognized: no llama.cpp pre-tokenizer splits like it and"
                        + " chkhsh {} is not in llama.cpp's table", source, r.chkhsh());
                break;
            default:
                throw new IllegalStateException("unhandled resolution basis " + r.basis());
        }
    }

    /** Registry names and comparable regex sequences per {@link PreTokenizerType}, built once. */
    private static final class Registry {
        static final Map<PreTokenizerType, List<String>> NAMES = names();
        static final Map<PreTokenizerType, Set<List<String>>> SEQUENCES = sequences();

        private static Map<PreTokenizerType, List<String>> names() {
            Map<PreTokenizerType, List<String>> byType = new EnumMap<>(PreTokenizerType.class);
            for (String name : PreTokenizerRegistry.names()) {
                PreTokenizerType type = PreTokenizerRegistry.forName(name).type();
                byType.computeIfAbsent(type, t -> new ArrayList<>()).add(name);
            }
            Map<PreTokenizerType, List<String>> frozen = new EnumMap<>(PreTokenizerType.class);
            for (Map.Entry<PreTokenizerType, List<String>> e : byType.entrySet()) {
                frozen.put(e.getKey(), Collections.unmodifiableList(e.getValue()));
            }
            return Collections.unmodifiableMap(frozen);
        }

        private static Map<PreTokenizerType, Set<List<String>>> sequences() {
            Map<PreTokenizerType, Set<List<String>>> out = new EnumMap<>(PreTokenizerType.class);
            for (PreTokenizerType type : PreTokenizerType.values()) {
                Set<List<String>> forms = new HashSet<>();
                forms.add(canonical(type.regexes()));
                if (!type.hfRegexes().isEmpty()) forms.add(canonical(type.hfRegexes()));
                out.put(type, Collections.unmodifiableSet(forms));
            }
            return Collections.unmodifiableMap(out);
        }

        private static List<String> canonical(List<String> regexes) {
            List<String> out = new ArrayList<>(regexes.size());
            for (String r : regexes) out.add(PreTokenizerCanonicalizer.canonicalRegex(r));
            return Collections.unmodifiableList(out);
        }
    }
}
