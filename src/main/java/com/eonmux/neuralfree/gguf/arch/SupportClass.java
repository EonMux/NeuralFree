package com.eonmux.neuralfree.gguf.arch;

/** What an architecture can be used for. */
public enum SupportClass {
    /** Autoregressive generation. */
    GEN,
    /** Pooled or per-token embeddings. */
    EMBED,
    /** Classification or ranking scores. */
    CLS,
    /** Masked-diffusion generation. */
    DIFF,
    /** Speculative-decoding drafter bound to a target model. */
    DRAFT,
    /** Audio codec or speech token model. */
    AUDIO,
    /** Encoder-decoder generation. */
    ENC_DEC,
    /** Not supported; the architecture entry carries the reason. */
    NA
}
