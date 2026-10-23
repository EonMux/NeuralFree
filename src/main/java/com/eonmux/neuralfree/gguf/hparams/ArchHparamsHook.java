package com.eonmux.neuralfree.gguf.hparams;

/**
 * Architecture-specific hparams loading: the transcription of one llama.cpp
 * {@code llama_model_<arch>::load_arch_hparams} (src/models/&lt;arch&gt;.cpp). It runs after the generic keys
 * are loaded and may read further keys, override defaults, change per-layer arrays and layer counts.
 * Hooks are stateless.
 */
@FunctionalInterface
public interface ArchHparamsHook {

    /** Hook for architectures with no arch-specific keys. */
    ArchHparamsHook NONE = (b, kv) -> { };

    /**
     * Applies the architecture's keys to the builder.
     *
     * @throws GgufFormatException when a required key is missing or malformed
     */
    void apply(ModelHparamsBuilder b, GgufKv kv);
}
