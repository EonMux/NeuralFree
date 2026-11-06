package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import java.util.Locale;
import java.util.Objects;

/**
 * The model family of a CLIP two-tower file, {@value #KEY}: which of vLLM's dual-encoder embedding models the towers
 * follow. Absent, the file is {@link #CLIP} (llama.cpp's legacy converter writes no such key).
 */
public enum ClipFamily {

    /**
     * vLLM's {@code CLIPEmbeddingModel} (vllm/model_executor/models/clip.py): causal text attention, a class token
     * and a pre-norm on the image side, the shortest-edge resize with a center crop, both towers projected, and
     * the cosine similarity as the score.
     */
    CLIP("clip"),
    /**
     * vLLM's {@code SiglipEmbeddingModel} (vllm/model_executor/models/siglip.py): bidirectional text attention
     * with the text head ({@code t.head}) over the last row, no class token, the squash resize, the attention
     * pooling head ({@code v.head}) on the image side, the text padded to the context as the model was trained,
     * and the sigmoid of the scaled cosine plus bias ({@code logit_scale}, {@code logit_bias}) as the score.
     */
    SIGLIP("siglip");

    /** The metadata key of the family, a string. */
    public static final String KEY = "neuralfree.clip.family";

    private final String keyName;

    ClipFamily(String keyName) {
        this.keyName = keyName;
    }

    /** The {@value #KEY} value of this family. */
    public String keyName() {
        return keyName;
    }

    /**
     * The family {@code kv} declares; {@link #CLIP} without {@value #KEY}.
     *
     * @throws GgufFormatException when the key names no family
     */
    public static ClipFamily of(GgufKv kv) {
        Objects.requireNonNull(kv, "kv");
        String name = kv.str(KEY, CLIP.keyName);
        for (ClipFamily f : values()) {
            if (f.keyName.equals(name)) {
                return f;
            }
        }
        throw new GgufFormatException("metadata key '" + KEY + "' is '" + name + "'; a two-tower file is 'clip' or"
                + " 'siglip'");
    }

    @Override
    public String toString() {
        return keyName.toLowerCase(Locale.ROOT);
    }
}
