package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.LLMModel;

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
}
