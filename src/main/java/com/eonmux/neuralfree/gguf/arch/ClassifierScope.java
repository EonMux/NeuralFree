package com.eonmux.neuralfree.gguf.arch;

import java.util.Locale;
import java.util.Objects;

/**
 * Where a model's classifier chain runs ({@code {arch}.classifier.scope}): over the pooled row of a sequence
 * classifier ({@code sequence}, vLLM's {@code classify}) or over every token of a token classifier ({@code token},
 * vLLM's {@code token_classify}). A file without the key keeps llama.cpp's RANK-only classifier, which runs only
 * when the pooling type is {@link PoolingKind#RANK}.
 */
public enum ClassifierScope {
    SEQUENCE("sequence"),
    TOKEN("token");

    private final String ggufName;

    ClassifierScope(String ggufName) {
        this.ggufName = ggufName;
    }

    /** The value stored in {@code {arch}.classifier.scope}. */
    public String ggufName() {
        return ggufName;
    }

    /**
     * The scope of a GGUF value, ignoring case.
     *
     * @throws IllegalArgumentException for any other value
     */
    public static ClassifierScope parse(String value) {
        Objects.requireNonNull(value, "classifier scope");
        String v = value.trim().toLowerCase(Locale.ROOT);
        for (ClassifierScope scope : values()) {
            if (scope.ggufName.equals(v)) {
                return scope;
            }
        }
        throw new IllegalArgumentException("classifier scope " + value + " is invalid; use sequence or token");
    }
}
