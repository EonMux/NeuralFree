package com.eonmux.neuralfree.gguf.arch;

/** Kind of sequence memory an architecture needs (llama.cpp {@code llama_model::create_memory}). */
public enum MemoryKind {
    /** No memory: encoders, classifiers and diffusion models run full forwards. */
    NONE,
    /** Attention KV cache (full or sliding-window rings, decided per layer). */
    KV,
    /** Attention KV cache plus recurrent states on the recurrent layers. */
    HYBRID,
    /** Recurrent states only. */
    RECURRENT,
    /** MLA KV cache plus the DSA indexer cache. */
    DSA,
    /** DSA with interleaved sliding-window layers. */
    DSA_ISWA,
    /** Hybrid recurrent model whose attention layers use a DSA indexer. */
    HYBRID_IDX,
    /** MiniMax sparse attention caches. */
    MSA,
    /** DeepSeek-V4 compressed KV caches. */
    DSV4,
    /** Encoder output plus decoder self-attention and cross-attention caches. */
    ENC_DEC
}
