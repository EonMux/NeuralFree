package com.eonmux.neuralfree.gguf;

/**
 * Callback for streaming token generation.
 * Called once per generated token with the token ID and decoded text.
 */
@FunctionalInterface
public interface TokenCallback {
    void onToken(int tokenId, String text);
}
