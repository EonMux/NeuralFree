package com.eonmux.neuralfree.server;

import java.util.List;

/**
 * One decision-model family: how the head reads a question, the flags llama.cpp's {@code
 * server_decision_context::init} sets for it (tools/server/server-decision.cpp), the label namespace of its options,
 * the temperature buckets and confidence formula of its answers, and whether a request may attach media or read
 * jointly. Immutable.
 *
 * <p>{@link Read} names the readout: {@code ORDER} and {@code JOINT} are the per-option decision head the vllm-decision2
 * and clef models carry; {@code LABELS} reads the logits of one answer-code token per option at the last prompt token
 * (openjev, lev, nimble, pplx-decider, vllm-d3, granite-switch, lfm2-d1, lfm3-d1, lfm3-d1-omni); {@code MARKERS} reads
 * one question-type score
 * at each option's marker token from the embeddings output (laya); {@code KEV} reads the dot product of the last
 * prompt token's hidden state with every option's {@code <|box_end|>} end token (kev); {@code D1OMNI} reads the
 * d1-omni prompt, its pieces
 * cut to the token budgets the model was trained on, through the marker columns of the decision head (lfm2-d1-omni);
 * {@code POOLED} reads the pooled classifier scores of the sequence (minicpm-ocr-jev).</p>
 */
final class DecisionFamily {

    /** The readout of a family. */
    enum Read {
        /** The vllm-decision2 head: one option score per option span of a decision order. */
        ORDER,
        /** The clef head: every question of a request in one prompt, one score per option span. */
        JOINT,
        /** The logits of an answer code at the last prompt token, one per option. */
        LABELS,
        /** The embeddings output at one marker token per option, one question-type column. */
        MARKERS,
        /** The hidden states: the last token dotted with one {@code <|box_end|>} end token per option (kev). */
        KEV,
        /** The d1-omni prompt, its pieces cut to their token budgets, read at the option markers. */
        D1OMNI,
        /** The pooled classifier scores of the sequence, one per class. */
        POOLED
    }

    /** Where a family's option labels come from. */
    enum Labels {
        /** No label text; the template reads the option keys. */
        NONE,
        /** One token per letter of A-Z and a-z (openjev). */
        LETTERS,
        /** One token per code of A-Z then AA-ZZ (lev, nimble, pplx-decider, vllm-d3). */
        CODES,
        /** Per question type: yes/no forms, level digits, and choice codes (lfm2-d1). */
        D1,
        /** The {@code decision.labels} metadata, one label per option (h2o-lightning). */
        METADATA
    }

    /** The confidence formula of an answer. */
    enum Confidence {
        /** TypeSafe's per-type choice and score formulas (llama.cpp's default). */
        TYPESAFE,
        /** One minus the normalized entropy (vllm-decision2, vllm-d3). */
        ENTROPY,
        /** The probability of the answer, its top option (lfm3-d1, lfm3-d1-omni, d1 api.py). */
        MAX
    }

    /** The temperature bucket names of a family. */
    enum Buckets {
        /** {@code 2}, {@code 3_5}, {@code 6_10}, {@code 11} by option count. */
        OPTIONS,
        /** {@code small} (at most 8), {@code mid} (at most 26), {@code large} (lev). */
        LEV
    }

    /** The family's metadata type name. */
    final String type;
    final Read read;
    final Labels labels;
    final Confidence confidence;
    final Buckets buckets;
    final int nOptionsMax;
    final boolean strictPayload;
    final boolean choiceSorted;
    final boolean noulTrueFirst;
    final boolean noulCriteriaOrder;
    final boolean canonicalLegend;
    final boolean nullState;
    final boolean canUseImages;
    final boolean jointReads;
    /** A noul question reads a nine-point rating scale (lev). */
    final boolean ratingNoul;
    /** A question may leave out its instructions, the template has a default (vllm-d3). */
    final boolean optionalInstructions;
    /** The option cap of the {@code /v1/decisions} route: vLLM's strategy limit for the family. */
    final int vllmMaxOptions;
    /** A noul question also reads its descriptions under "yes" and "no" (h2o-lightning, lfm2-d1-omni). */
    final boolean noulAliases;
    /** Media answers read uncalibrated: temperature one when the metadata names no image temperature (lfm2-d1-omni). */
    final boolean uncalibratedMedia;
    /** A spoken question (input_audio instructions) refuses: the family has no audio input (lfm3-d1). */
    final boolean refusesSpoken;

    private DecisionFamily(String type, Read read, Labels labels, Confidence confidence, Buckets buckets,
                           int nOptionsMax, boolean strictPayload, boolean choiceSorted, boolean noulTrueFirst,
                           boolean noulCriteriaOrder, boolean canonicalLegend, boolean nullState,
                           boolean canUseImages, boolean jointReads, boolean ratingNoul,
                           boolean optionalInstructions, int vllmMaxOptions, boolean noulAliases,
                           boolean uncalibratedMedia, boolean refusesSpoken) {
        this.type = type;
        this.read = read;
        this.labels = labels;
        this.confidence = confidence;
        this.buckets = buckets;
        this.nOptionsMax = nOptionsMax;
        this.strictPayload = strictPayload;
        this.choiceSorted = choiceSorted;
        this.noulTrueFirst = noulTrueFirst;
        this.noulCriteriaOrder = noulCriteriaOrder;
        this.canonicalLegend = canonicalLegend;
        this.nullState = nullState;
        this.canUseImages = canUseImages;
        this.jointReads = jointReads;
        this.ratingNoul = ratingNoul;
        this.optionalInstructions = optionalInstructions;
        this.vllmMaxOptions = vllmMaxOptions;
        this.noulAliases = noulAliases;
        this.uncalibratedMedia = uncalibratedMedia;
        this.refusesSpoken = refusesSpoken;
    }

    /** The families of llama.cpp's decision types that this server reads, by metadata name. */
    private static final List<DecisionFamily> FAMILIES = List.of(
            new DecisionFamily("vllm-decision2", Read.ORDER, Labels.NONE, Confidence.ENTROPY,
                    Buckets.OPTIONS, 255, true, false, false, true, true, false, false, false, false, false, 26,
                    false, false, false),
            new DecisionFamily("clef", Read.JOINT, Labels.NONE, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 255, false, true, true, false, false, false, true, true, false, false, 26,
                    false, false, false),
            new DecisionFamily("openjev", Read.LABELS, Labels.LETTERS, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 52, false, false, true, false, false, false, true, false, false, false, 26,
                    false, false, false),
            new DecisionFamily("lev", Read.LABELS, Labels.CODES, Confidence.TYPESAFE,
                    Buckets.LEV, 255, false, false, false, false, false, false, false, false, true, false, 26,
                    false, false, false),
            new DecisionFamily("nimble", Read.LABELS, Labels.CODES, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 255, false, false, false, false, false, false, false, false, false, false, 26,
                    false, false, false),
            new DecisionFamily("pplx-decider", Read.LABELS, Labels.CODES, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 255, false, false, false, false, true, false, true, false, false, false, 26,
                    false, false, false),
            new DecisionFamily("vllm-d3", Read.LABELS, Labels.CODES, Confidence.ENTROPY,
                    Buckets.OPTIONS, 255, false, false, false, false, true, true, true, false, false, true, 255,
                    false, false, false),
            new DecisionFamily("laya", Read.MARKERS, Labels.NONE, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 255, false, false, false, false, false, false, false, false, false, false, 255,
                    false, false, false),
            new DecisionFamily("lfm2-d1", Read.LABELS, Labels.D1, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 255, false, false, true, false, false, true, true, false, false, false, 26,
                    false, false, false),
            new DecisionFamily("h2o-lightning", Read.LABELS, Labels.METADATA, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 0, false, false, true, false, false, false, true, false, false, false, 26,
                    true, false, false),
            new DecisionFamily("openbmb-minicpm-ocr-jev", Read.POOLED, Labels.NONE, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 2, false, true, false, false, false, false, true, false, false, false, 2,
                    false, false, false),
            new DecisionFamily("lfm2-d1-omni", Read.D1OMNI, Labels.NONE, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 255, false, false, false, false, false, true, true, false, false, false, 26,
                    true, true, false),
            new DecisionFamily("lfm3-d1", Read.LABELS, Labels.D1, Confidence.MAX,
                    Buckets.OPTIONS, 255, false, false, true, false, false, true, true, false, false, false, 26,
                    false, false, true),
            new DecisionFamily("lfm3-d1-omni", Read.LABELS, Labels.D1, Confidence.MAX,
                    Buckets.OPTIONS, 255, false, false, true, false, false, true, true, false, false, false, 26,
                    false, false, false),
            new DecisionFamily("granite-switch", Read.LABELS, Labels.CODES, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 255, false, false, true, false, false, false, false, false, false, false, 26,
                    false, false, false),
            new DecisionFamily("kev", Read.KEV, Labels.NONE, Confidence.TYPESAFE,
                    Buckets.OPTIONS, 255, false, false, false, false, false, false, false, false, false, false, 255,
                    false, false, false));

    /**
     * The family named {@code type}, or null when the type is not served.
     */
    static DecisionFamily of(String type) {
        if (type == null) {
            return null;
        }
        for (DecisionFamily family : FAMILIES) {
            if (family.type.equals(type)) {
                return family;
            }
        }
        return null;
    }

    /** True when the family reads a decision order (vllm-decision2, clef). */
    boolean readsOrder() {
        return read == Read.ORDER || read == Read.JOINT;
    }

    /** True when the model answers one fixed question, the prompt is entirely in its template (minicpm-ocr-jev). */
    boolean fixedQuestion() {
        return read == Read.POOLED;
    }

    /** The option labels and the question type default come from the family, not the request (h2o-lightning). */
    boolean defaultChoice() {
        return labels == Labels.METADATA;
    }

    /** A score question reads one to n_options_max levels, not the default two to ten (h2o-lightning). */
    boolean openScoreLevels() {
        return labels == Labels.METADATA;
    }

    /** A score question's answer is the median level, the first whose mass reaches one half (lfm3-d1, lfm3-d1-omni). */
    boolean medianScore() {
        return confidence == Confidence.MAX;
    }

    /** The temperature bucket name of a question with {@code nOptions} options (llama.cpp's get_temperature). */
    String bucket(int nOptions) {
        if (buckets == Buckets.LEV) {
            return nOptions <= 8 ? "small" : nOptions <= 26 ? "mid" : "large";
        }
        return nOptions <= 2 ? "2" : nOptions <= 5 ? "3_5" : nOptions <= 10 ? "6_10" : "11";
    }

    /** The parsing flags llama.cpp's {@code init} sets for this family, capped at {@code nOptionsMax} options. */
    DecisionQuestions.Flags flags(int nOptionsMax) {
        return new DecisionQuestions.Flags(strictPayload, choiceSorted, noulTrueFirst, noulCriteriaOrder,
                nOptionsMax, nullState, optionalInstructions, fixedQuestion(), defaultChoice(), noulAliases,
                openScoreLevels(), refusesSpoken);
    }
}
