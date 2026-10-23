package com.eonmux.neuralfree.gguf.hparams;

/** GGUF metadata is missing, has the wrong type, or holds values the model cannot be built from. */
public class GgufFormatException extends IllegalArgumentException {

    private static final long serialVersionUID = 1L;

    public GgufFormatException(String message) {
        super(message);
    }

    public GgufFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
