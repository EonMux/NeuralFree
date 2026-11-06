package com.eonmux.neuralfree.gguf.engine.sampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * The kinds of the sampler stages llama.cpp's configurable chain order names (the
 * {@code common_params_sampling::samplers} list, llama.cpp common/common.h:265-273). {@link #DEFAULT_ORDER} is that
 * list, which {@code SamplerConfig.order} defaults to; a configured order lists the kind labels in the order the
 * stages run, between the fixed bias stage (before the chain) and the dist stage (after it).
 *
 * <p>A configured chain entry is any llama.cpp sampler name:
 * {@code penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature;infill;adaptive_p}
 * ({@code common/sampler_type}). Mirostat is not a chain entry: llama.cpp replaces the whole chain by
 * {@code temp + mirostat} (or {@code mirostat_v2}, src/llama-sampler.cpp) when {@code mirostat} is 1 or 2, which
 * {@code SamplerConfig.mirostat} selects instead.
 */
public enum StageKind {

    /** llama.cpp {@code penalties}: the repetition, presence and frequency penalties over the accepted tokens. */
    PENALTIES("penalties"),
    /** llama.cpp {@code dry}: the DRY repetition penalty (DryStage). */
    DRY("dry"),
    /** llama.cpp {@code top_n_sigma}: the n-std-deviation logit mask (TopNSigmaStage). */
    TOP_N_SIGMA("top_n_sigma"),
    /** llama.cpp {@code top_k}: the {@code k} highest logits. */
    TOP_K("top_k"),
    /** llama.cpp {@code typ_p}: locally typical sampling. */
    TYPICAL_P("typ_p"),
    /** llama.cpp {@code top_p}: nucleus sampling. */
    TOP_P("top_p"),
    /** llama.cpp {@code min_p}: the {@code p * p_max} logit threshold. */
    MIN_P("min_p"),
    /** llama.cpp {@code xtc}: the cross-token crossover trim (XtcStage). */
    XTC("xtc"),
    /** llama.cpp {@code temperature}: logit scaling (argmax at a temperature of 0, llama_sampler_temp_impl). */
    TEMPERATURE("temperature"),
    /** llama.cpp {@code infill}: llama_sampler_infill's EOG-aware filter (InfillStage). */
    INFILL("infill"),
    /**
     * llama.cpp {@code adaptive_p}: trailing terminal selector in place of the dist stage (AdaptivePStage); llama.cpp
     * appends it at the very end of the chain and never builds a dist stage behind it (common/sampling.cpp:387-407).
     */
    ADAPTIVE_P("adaptive_p"),
    /**
     * llama.cpp {@code mirostat}: not a chain entry (see the class doc); a build chain that sets
     * {@code SamplerConfig.mirostat} to 1 or 2 replaces the order by {@code temperature + mirostat} as llama.cpp's
     * chain construction does.
     */
    MIROSTAT("mirostat");

    /** llama.cpp's default chain order, the first nine kinds (common/common.h:265-273). */
    public static final List<StageKind> DEFAULT_ORDER;

    static {
        List<StageKind> order = new ArrayList<>();
        Collections.addAll(order, PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE);
        DEFAULT_ORDER = Collections.unmodifiableList(order);
    }

    private final String label;

    StageKind(String label) {
        this.label = label;
    }

    /**
     * llama.cpp's stage name, as a {@code neuralfree.samplers} entry and a {@code general.sampling.sequence} entry.
     */
    public String label() {
        return label;
    }

    /**
     * The kind of a llama.cpp label.
     *
     * @throws IllegalArgumentException when the label names no kind
     */
    public static StageKind ofLabel(String label) {
        for (StageKind kind : values()) {
            if (kind.label.equals(label)) {
                return kind;
            }
        }
        throw new IllegalArgumentException("unknown sampler stage '" + label + "'; the stage names are "
                + joinedLabels());
    }

    /**
     * The kind a llama.cpp request names ({@code common_sampler_types_from_names}): a label, case-insensitively, in
     * kebab case ({@code top-k}) or without underscores ({@code topk}), or one of the aliases {@code nucleus},
     * {@code temp} and {@code typ}.
     *
     * @throws IllegalArgumentException when the name matches no kind
     */
    public static StageKind ofName(String name) {
        String key = Objects.requireNonNull(name, "sampler name").trim().toLowerCase(Locale.ROOT);
        switch (key) {
            case "nucleus":
                return TOP_P;
            case "temp":
                return TEMPERATURE;
            case "typ":
                return TYPICAL_P;
            default:
                break;
        }
        for (StageKind kind : values()) {
            if (kind.label.equals(key) || kind.label.replace('_', '-').equals(key)
                    || kind.label.replace("_", "").equals(key)) {
                return kind;
            }
        }
        throw new IllegalArgumentException("unknown sampler '" + name + "'; the stage names are " + joinedLabels());
    }

    /**
     * The kinds of a llama.cpp sampler-character string ({@code common_sampler_types_from_chars}): {@code d} dry,
     * {@code k} top_k, {@code y} typ_p, {@code p} top_p, {@code s} top_n_sigma, {@code m} min_p, {@code t}
     * temperature, {@code x} xtc, {@code i} infill, {@code e} penalties, {@code a} adaptive_p.
     *
     * @throws IllegalArgumentException on a character that names no kind
     */
    public static List<StageKind> ofChars(String chars) {
        List<StageKind> out = new ArrayList<>(chars.length());
        for (int i = 0; i < chars.length(); i++) {
            out.add(ofChar(chars.charAt(i)));
        }
        return out;
    }

    private static StageKind ofChar(char c) {
        switch (c) {
            case 'd': return DRY;
            case 'k': return TOP_K;
            case 'y': return TYPICAL_P;
            case 'p': return TOP_P;
            case 's': return TOP_N_SIGMA;
            case 'm': return MIN_P;
            case 't': return TEMPERATURE;
            case 'x': return XTC;
            case 'i': return INFILL;
            case 'e': return PENALTIES;
            case 'a': return ADAPTIVE_P;
            default:
                throw new IllegalArgumentException("unknown sampler character '" + c + "'; the characters are "
                        + "dkypsmtxiea");
        }
    }

    /**
     * The order of a '{@code ;}'-separated label list (llama.cpp's configurable sampler order).
     *
     * @throws IllegalArgumentException on an empty name, an unknown name or a duplicate name
     */
    public static List<StageKind> parseOrder(String joined) {
        Objects.requireNonNull(joined, "sampler order");
        List<StageKind> order = new ArrayList<>();
        for (String part : joined.split(";", -1)) {
            String label = part.trim();
            if (label.isEmpty()) {
                throw new IllegalArgumentException("sampler order has an empty stage name in '" + joined + "'");
            }
            StageKind kind = ofLabel(label);
            if (order.contains(kind)) {
                throw new IllegalArgumentException("sampler order repeats '" + label + "' in '" + joined + "'");
            }
            order.add(kind);
        }
        return Collections.unmodifiableList(order);
    }

    static String joinedLabels() {
        StringBuilder s = new StringBuilder();
        for (StageKind kind : values()) {
            if (s.length() > 0) {
                s.append(", ");
            }
            s.append(kind.label);
        }
        return s.toString();
    }
}
