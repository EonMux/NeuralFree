package com.eonmux.neuralfree.gguf.engine;

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

/**
 * The derived candidate head of a {@code dflash} drafter, built once at bind time and used only by the drafter:
 * the head the drafter would otherwise use — its own {@code output.weight}, else the target's, else the tied token
 * embeddings — re-stated as a smaller, lower-precision head over a candidate vocabulary. It keeps
 * {@link #vocabSize()} rows: the draft's special token ids (unknown, BOS, EOS, EOT, EOM, separator, padding, CLS
 * and the mask token the draft names), the ids of {@link #frequencyFile()} in rank order and an ascending fill of
 * the remaining ids (the BPE merge-rank prior), stored at {@link #headPrec()} or in the head's own precision and
 * mapped back to full-vocabulary ids by a synthesized {@code d2t}. The target verifies with its own head, so
 * committed tokens are unchanged; only the proposals move. {@link #matricesPrec()} stores the drafter's own
 * projection matrices (its encoder {@code fc}, per-block attention and FFN matrices, DFlash2 convolutions and the
 * selector's hidden projection) at a lower precision the same way, leaving tensors that are already quantized as
 * they are. Immutable; built with {@link #builder()}.
 *
 * <p>Properties: {@code neuralfree.draft.head.vocab}, {@code neuralfree.draft.head.freq},
 * {@code neuralfree.draft.head.keep}, {@code neuralfree.draft.head.prec} (vLLM's
 * {@code draft_lm_head_quantization} of a {@code --speculative-config}) and {@code neuralfree.draft.prec}. No
 * llama.cpp or vLLM command-line flag names a reduced draft vocabulary, so the vocabulary is properties-only.
 */
public final class DraftHeadOptions {

    /** The storage of a derived tensor: {@link #Q8_0} (32-value blocks, {@code amax/127} per block),
     * {@link #FP8} (E4M3FN codes of one scale per row, F8_E4M3_B128 for rows of whole 128-value blocks,
     * F8_E4M3_B32 otherwise — the per-row FP8 of vLLM's {@code fp8_per_channel} and SpecPilot's draft head) or
     * {@link #NVFP4} (E2M1 codes with an E4M3 scale per 16 values and a per-tensor FP32 global scale in the
     * {@code <base>.scale} sibling — vLLM's {@code draft_lm_head_quantization=nvfp4}). */
    public enum Prec {
        Q8_0, FP8, NVFP4;

        /**
         * The precision named {@code value}, case-insensitive.
         *
         * @throws IllegalArgumentException naming {@code key} for any other value
         */
        public static Prec of(String key, String value) {
            String v = Objects.requireNonNull(value, "value").trim().toLowerCase(Locale.ROOT);
            if (v.equals("q8_0")) {
                return Q8_0;
            }
            if (v.equals("fp8")) {
                return FP8;
            }
            if (v.equals("nvfp4")) {
                return NVFP4;
            }
            throw new IllegalArgumentException(key + "=" + value
                    + " is not a draft precision; use fp8, nvfp4 or q8_0");
        }
    }

    private static final DraftHeadOptions EMPTY = new DraftHeadOptions(0, null, new int[0], null, null);

    private final int vocabSize;
    private final String frequencyFile;
    private final int[] keepIds;
    private final Prec headPrec;
    private final Prec matricesPrec;

    private DraftHeadOptions(int vocabSize, String frequencyFile, int[] keepIds, Prec headPrec, Prec matricesPrec) {
        this.vocabSize = vocabSize;
        this.frequencyFile = frequencyFile;
        this.keepIds = keepIds;
        this.headPrec = headPrec;
        this.matricesPrec = matricesPrec;
    }

    /** The defaults: the drafter's own head and matrices, unchanged. */
    public static DraftHeadOptions defaults() {
        return EMPTY;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Rows the derived head keeps; {@code 0} keeps all of them. */
    public int vocabSize() {
        return vocabSize;
    }

    /** The token-frequency ranking file, or null for the ascending-id prior. */
    public String frequencyFile() {
        return frequencyFile;
    }

    /** Token ids the derived head always keeps, beside the draft's special ids. */
    public int[] keepIds() {
        return keepIds.clone();
    }

    /** Storage of the derived head, or null for the head's own. */
    public Prec headPrec() {
        return headPrec;
    }

    /** Storage of the drafter's own projection matrices, or null for the file's. */
    public Prec matricesPrec() {
        return matricesPrec;
    }

    /** True when any setting derives a tensor. */
    public boolean enabled() {
        return vocabSize > 0 || headPrec != null || matricesPrec != null;
    }

    /** The settings as {@code {k=v, ...}}, for the option summaries of {@code SpeculativeOptions}. */
    public String describe() {
        return "{vocab=" + (vocabSize > 0 ? vocabSize : "all")
                + (frequencyFile != null ? ", freq=" + frequencyFile : "")
                + (keepIds.length > 0 ? ", keep=" + Arrays.toString(keepIds) : "")
                + (headPrec != null ? ", headPrec=" + headPrec.name().toLowerCase(Locale.ROOT) : "")
                + (matricesPrec != null ? ", matricesPrec=" + matricesPrec.name().toLowerCase(Locale.ROOT) : "")
                + '}';
    }

    @Override
    public String toString() {
        return "DraftHeadOptions" + describe();
    }

    /** Mutable builder for {@link DraftHeadOptions}. */
    public static final class Builder {
        private int vocabSize;
        private String frequencyFile;
        private int[] keepIds = new int[0];
        private Prec headPrec;
        private Prec matricesPrec;

        private Builder() {
        }

        /**
         * Rows the derived head keeps ({@code neuralfree.draft.head.vocab}); {@code 0} keeps all.
         *
         * @throws IllegalArgumentException for a negative size
         */
        public Builder vocabSize(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("the draft head vocabulary size must not be negative, got " + v);
            }
            this.vocabSize = v;
            return this;
        }

        /** The token-frequency ranking file ({@code neuralfree.draft.head.freq}); null or blank clears it. */
        public Builder frequencyFile(String path) {
            this.frequencyFile = path == null || path.isBlank() ? null : path.trim();
            return this;
        }

        /**
         * Token ids the derived head always keeps ({@code neuralfree.draft.head.keep}).
         *
         * @throws IllegalArgumentException for a negative id
         */
        public Builder keepIds(int... ids) {
            for (int id : Objects.requireNonNull(ids, "ids")) {
                if (id < 0) {
                    throw new IllegalArgumentException("a kept draft-head token id must not be negative, got " + id);
                }
            }
            this.keepIds = ids.clone();
            return this;
        }

        /** Storage of the derived head ({@code neuralfree.draft.head.prec}); null keeps the head's own. */
        public Builder headPrec(Prec prec) {
            this.headPrec = prec;
            return this;
        }

        /** Storage of the drafter's own projection matrices ({@code neuralfree.draft.prec}); null keeps the file's. */
        public Builder matricesPrec(Prec prec) {
            this.matricesPrec = prec;
            return this;
        }

        /** Builds the options. */
        public DraftHeadOptions build() {
            return new DraftHeadOptions(vocabSize, frequencyFile, keepIds, headPrec, matricesPrec);
        }
    }
}
