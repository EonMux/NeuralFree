package com.eonmux.neuralfree.gguf.image;

/**
 * Thrown when an image buffer cannot be decoded. The message names the format and the reason, so a caller can
 * report the failing input to its user.
 */
public class ImageDecodeException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public ImageDecodeException(String message) {
        super(message);
    }

    public ImageDecodeException(String message, Throwable cause) {
        super(message, cause);
    }
}
