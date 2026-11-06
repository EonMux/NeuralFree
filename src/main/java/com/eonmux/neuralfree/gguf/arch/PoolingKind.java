package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.LLMModel;

import java.util.Locale;
import java.util.Objects;

/**
 * Embedding pooling (llama.cpp {@code llama_pooling_type}); maps one-to-one onto
 * {@link LLMModel.PoolingType}.
 */
public enum PoolingKind {
    NONE(0),
    MEAN(1),
    CLS(2),
    LAST(3),
    RANK(4);

    /** GGUF value meaning "not specified" ({@code LLAMA_POOLING_TYPE_UNSPECIFIED}). */
    public static final int GGUF_CODE_UNSPECIFIED = -1;

    private final int ggufCode;

    PoolingKind(int ggufCode) {
        this.ggufCode = ggufCode;
    }

    /** The value stored in {@code %s.pooling_type}. */
    public int ggufCode() {
        return ggufCode;
    }

    /** The matching public API pooling type. */
    public LLMModel.PoolingType toPoolingType() {
        return LLMModel.PoolingType.valueOf(name());
    }

    /**
     * Pooling for a GGUF code.
     *
     * @return the pooling, or null for {@link #GGUF_CODE_UNSPECIFIED}
     * @throws IllegalArgumentException for an unknown code
     */
    public static PoolingKind fromGgufCode(long code) {
        if (code == GGUF_CODE_UNSPECIFIED) {
            return null;
        }
        for (PoolingKind p : values()) {
            if (p.ggufCode == code) {
                return p;
            }
        }
        throw new IllegalArgumentException("unknown pooling type " + code);
    }

    /**
     * The pooling of a llama.cpp {@code --pooling} name ({@code none}, {@code mean}, {@code cls}, {@code last},
     * {@code rank}) or of a vLLM pooler {@code pooling_type} name, ignoring case; vLLM's {@code ALL} names per-token
     * pooling and maps to {@link #NONE}.
     *
     * @throws IllegalArgumentException for any other name, including vLLM's per-step {@code STEP}
     */
    public static PoolingKind parse(String value) {
        String s = Objects.requireNonNull(value, "pooling").trim().toLowerCase(Locale.ROOT);
        switch (s) {
            case "none":
                return NONE;
            case "mean":
                return MEAN;
            case "cls":
                return CLS;
            case "last":
                return LAST;
            case "rank":
                return RANK;
            case "all":
                return NONE;
            case "step":
                throw new IllegalArgumentException("pooling " + value + " is not supported; vLLM's STEP pools the"
                        + " token embeddings of one generation step, which nothing here reads; use none, mean, cls,"
                        + " last or rank");
            default:
                throw new IllegalArgumentException("pooling " + value + " is invalid; use none, mean, cls, last or"
                        + " rank");
        }
    }
}
