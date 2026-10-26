package com.eonmux.neuralfree.gguf.engine.sampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The kinds of the sampler stages llama.cpp's configurable chain order names (the
 * {@code common_params_sampling::samplers} list, llama.cpp common/common.h:265-273). {@link #DEFAULT_ORDER} is that
 * list, which {@code SamplerConfig.order} defaults to; a configured order lists the kind labels in the order the
 * stages run, between the fixed bias stage (before the chain) and the dist stage (after it).
 *
 * <p>Until SA-2 adds the llama.cpp stages NeuralFree does not carry yet, a configured {@link #DRY}, {@link #TOP_N_SIGMA}
 * or {@link #XTC} kind builds no stage and applies nothing, which is the chain with those stages off.
 */
public enum StageKind {

    /** llama.cpp {@code penalties}: the repetition, presence and frequency penalties over the accepted tokens. */
    PENALTIES("penalties"),
    /** llama.cpp {@code dry}: no NeuralFree stage until SA-2, so building one applies nothing. */
    DRY("dry"),
    /** llama.cpp {@code top_n_sigma}: no NeuralFree stage until SA-2, so building one applies nothing. */
    TOP_N_SIGMA("top_n_sigma"),
    /** llama.cpp {@code top_k}: the {@code k} highest logits. */
    TOP_K("top_k"),
    /** llama.cpp {@code typ_p}: locally typical sampling. */
    TYPICAL_P("typ_p"),
    /** llama.cpp {@code top_p}: nucleus sampling. */
    TOP_P("top_p"),
    /** llama.cpp {@code min_p}: the {@code p * p_max} logit threshold. */
    MIN_P("min_p"),
    /** llama.cpp {@code xtc}: no NeuralFree stage until SA-2, so building one applies nothing. */
    XTC("xtc"),
    /** llama.cpp {@code temperature}: logit scaling (argmax at a temperature of 0, llama_sampler_temp_impl). */
    TEMPERATURE("temperature");

    /** llama.cpp's default chain order, the label order of the constants above. */
    public static final List<StageKind> DEFAULT_ORDER;

    static {
        List<StageKind> order = new ArrayList<>();
        Collections.addAll(order, values());
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
