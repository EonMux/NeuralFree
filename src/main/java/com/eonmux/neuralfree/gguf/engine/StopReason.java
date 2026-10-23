package com.eonmux.neuralfree.gguf.engine;

/** Why a generation ended. */
public enum StopReason {
    /** The model produced an end-of-generation token, or ended generation on its own. */
    EOS,
    /** The requested number of new tokens was generated. */
    MAX_TOKENS,
    /** The decoded output contained one of the request's stop strings. */
    STOP_STRING,
    /** The context was full and could not be shifted. */
    CONTEXT_FULL,
    /** The request's cancellation hook asked generation to stop. */
    CANCELLED
}
