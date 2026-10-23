package com.eonmux.neuralfree.gguf.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.function.IntFunction;

/**
 * Stateful per-generation decoder that emits text only at UTF-8 boundaries, so a multi-byte
 * character split across tokens is never rendered as U+FFFD.
 *
 * <p>Create one per generation, call {@link #decodeToken(int)} per sampled token and
 * {@link #flush()} at the end. Not thread-safe: it holds the pending partial sequence.</p>
 */
public class StreamingDecoder {

    private final IntFunction<byte[]> visibleBytes;
    private byte[] pending = new byte[4];
    private int pendingLen;

    /**
     * @param visibleBytes the bytes a token contributes to output; empty for hidden tokens
     */
    public StreamingDecoder(IntFunction<byte[]> visibleBytes) {
        if (visibleBytes == null) throw new IllegalArgumentException("visibleBytes is null");
        this.visibleBytes = visibleBytes;
    }

    /** Decodes one token, returning every complete character now available. */
    public String decodeToken(int tokenId) {
        byte[] decoded = visibleBytes.apply(tokenId);
        if (decoded == null || decoded.length == 0) return "";
        int totalLen = pendingLen + decoded.length;
        byte[] combined;
        if (pendingLen > 0) {
            combined = new byte[totalLen];
            System.arraycopy(pending, 0, combined, 0, pendingLen);
            System.arraycopy(decoded, 0, combined, pendingLen, decoded.length);
            pendingLen = 0;
        } else {
            combined = decoded;
        }
        int completeEnd = completeUtf8End(combined, totalLen);
        if (completeEnd < totalLen) {
            int leftover = totalLen - completeEnd;
            if (leftover > pending.length) pending = new byte[leftover];
            System.arraycopy(combined, completeEnd, pending, 0, leftover);
            pendingLen = leftover;
        }
        return completeEnd == 0 ? "" : new String(combined, 0, completeEnd, StandardCharsets.UTF_8);
    }

    /** Returns the pending partial sequence (rendered with U+FFFD) and clears it. */
    public String flush() {
        if (pendingLen == 0) return "";
        String result = new String(pending, 0, pendingLen, StandardCharsets.UTF_8);
        pendingLen = 0;
        return result;
    }

    /** Drops any pending partial sequence. */
    public void reset() {
        pendingLen = 0;
    }

    /** End index of the last complete UTF-8 sequence; a trailing incomplete sequence is excluded. */
    private static int completeUtf8End(byte[] buf, int len) {
        if (len == 0) return 0;
        int i = len - 1;
        while (i >= 0 && (buf[i] & 0xC0) == 0x80) i--;
        if (i < 0) return 0;
        int lead = buf[i] & 0xFF;
        int expected = lead < 0xC0 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
        return len - i >= expected ? len : i;
    }
}
